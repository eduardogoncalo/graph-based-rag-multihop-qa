from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmark.core.config_loader import DatasetConfig
from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import Document, GoldEvidence, Question
from benchmark.ingestion.canonical_export import CanonicalDataset
from benchmark.ingestion.chunking import chunk_document

RAW_FILE_CANDIDATES = (
    "dev.json",
    "test.json",
    "train.json",
)
ALIASES_FILE = "id_aliases.json"
HIPPORAG_QUESTIONS_FILE = "2wikimultihopqa.json"
HIPPORAG_CORPUS_FILE = "2wikimultihopqa_corpus.json"
DEV_FILE = "dev.json"

TWOWIKI_QUESTION_TYPE = "multi_hop_reasoning"


def load_twowiki_hipporag(
    config: DatasetConfig,
    *,
    sample_size: int | None = None,
    raw_file: str | Path | None = None,
    corpus_file: str | Path | None = None,
    dev_file: str | Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> CanonicalDataset:
    """Canonical 2Wiki a partir dos ficheiros de avaliação PUBLICADOS pelo HippoRAG
    (repo OSU-NLP-Group/HippoRAG, reproduce/dataset/) — decisão do autor 2026-07-09
    para comparabilidade item-idêntica com o HippoRAG 2 (ICML 2025).

    Fontes:
    - ``2wikimultihopqa.json``: as EXATAS 1000 perguntas usadas nos papers
      (auditoria 2026-07-09: todas existem no dev.json com contextos idênticos);
    - ``2wikimultihopqa_corpus.json``: o corpus de 6.119 passagens {title, text}
      (texto == " ".join(sentenças) — mesma convenção deste loader; os Documents
      vêm DAQUI, não da união dos contextos: 1 passagem de contexto, distrator
      "Tōku Made", fica de fora por não estar no corpus deles);
    - ``dev.json``: recupera supporting_facts e type por _id (o ficheiro deles
      não traz esses campos);
    - ``id_aliases.json``: answer_aliases via answer_id (Wikidata).
    """
    root = Path(config.raw_path)
    questions_path = Path(raw_file) if raw_file is not None else root / HIPPORAG_QUESTIONS_FILE
    corpus_path = Path(corpus_file) if corpus_file is not None else root / HIPPORAG_CORPUS_FILE
    dev_path = Path(dev_file) if dev_file is not None else root / DEV_FILE
    for path, label in ((questions_path, "questions"), (corpus_path, "corpus"), (dev_path, "dev")):
        if not path.exists():
            raise TwowikiRawDataMissingError(
                f"HippoRAG 2Wiki {label} file not found at {path}. Download is "
                "intentionally not automatic (repo OSU-NLP-Group/HippoRAG, reproduce/dataset/)."
            )

    raw_questions = _read_json_array(questions_path)
    raw_corpus = _read_json_array(corpus_path)
    dev_by_id = {row.get("_id"): row for row in _read_json_array(dev_path)}
    aliases_by_qid = _load_aliases(root / ALIASES_FILE)

    # AMOSTRAGEM, E PORQUE O CORPUS TEM DE ENCOLHER COM ELA.
    #
    # O 2Wiki não se parece com o MuSiQue aqui, e a diferença custa caro se
    # passar despercebida. No MuSiQue os documentos são recolhidos DAS perguntas
    # seleccionadas, portanto 20 perguntas dão 399 documentos e o corpus encolhe
    # sozinho. Aqui os documentos vêm do ficheiro de corpus publicado — 6119
    # passagens — e **não dependem das perguntas**. Amostrar 20 perguntas sem
    # mexer no corpus dava uma «amostra de smoke» com 6119 documentos para
    # indexar: cerca de treze horas só de LightRAG, contra os ~50 minutos das
    # 399 do MuSiQue. Deixava de ser um smoke.
    #
    # Quando há amostragem, o corpus passa a ser o das passagens que aparecem no
    # contexto das perguntas escolhidas — que é exactamente a regra do MuSiQue,
    # e mantém os distractores de cada pergunta.
    #
    # Sem amostragem nada disto corre: o caminho do `twowiki` eval1k, que é o
    # que produziu os resultados da dissertação, não tem bloco `sample:` nem
    # recebe `sample_size`, e continua a carregar as 1000 perguntas contra o
    # corpus inteiro, byte a byte como antes.
    num_questions, seed, veio_do_config = _resolver_amostra_hipporag(config, sample_size)
    chaves_do_corpus: set[tuple[str, str]] | None = None
    if num_questions is not None and num_questions < len(raw_questions):
        if veio_do_config:
            # Amostra aleatória com semente, o mesmo protocolo do MuSiQue.
            raw_questions = random.Random(seed).sample(raw_questions, num_questions)
        else:
            # `--sample-size` explícito mantém o corte por prefixo que sempre
            # teve. A CLI já declara essa amostra como divergente e salta a
            # verificação de impressão digital.
            raw_questions = raw_questions[: max(num_questions, 0)]
        chaves_do_corpus = _chaves_de_contexto(raw_questions)

    # Documents = o corpus deles (identidade por title+text), restringido às
    # passagens das perguntas quando houve amostragem.
    documents_by_id: dict[str, Document] = {}
    doc_id_by_key: dict[tuple[str, str], str] = {}
    for row in raw_corpus:
        title = str(row.get("title") or "").strip()
        paragraph_text = str(row.get("text") or "")
        if not title and not paragraph_text:
            continue
        if chaves_do_corpus is not None and (title, paragraph_text) not in chaves_do_corpus:
            continue
        document_id = deterministic_id(
            "doc", [config.dataset_id, config.dataset_version, title, paragraph_text]
        )
        if document_id not in documents_by_id:
            passage = f"{title}\n\n{paragraph_text}" if title else paragraph_text
            documents_by_id[document_id] = Document(
                document_id=document_id,
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                title=title or None,
                text=passage,
                source_path=str(corpus_path),
                metadata={"paragraph_text": paragraph_text},
            )
        doc_id_by_key[(title, paragraph_text)] = document_id

    questions: list[Question] = []
    evidence: list[GoldEvidence] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        question_text = str(raw_question.get("question") or "").strip()
        source_question_id = raw_question.get("_id")
        dev_row = dev_by_id.get(source_question_id)
        if not question_text or dev_row is None:
            continue

        question_id = deterministic_id(
            "q",
            [config.dataset_id, config.dataset_version, source_question_id, question_text],
        )

        # texto exato de cada título DENTRO do contexto da própria pergunta
        ctx_text_by_title: dict[str, str] = {}
        for raw_paragraph in raw_question.get("context") or []:
            if not isinstance(raw_paragraph, (list, tuple)) or len(raw_paragraph) != 2:
                continue
            title = str(raw_paragraph[0] or "").strip()
            sentences = raw_paragraph[1] if isinstance(raw_paragraph[1], list) else []
            ctx_text_by_title[title] = " ".join(
                str(s).strip() for s in sentences if str(s).strip()
            )

        supporting_titles: dict[str, list] = {}
        for fact in dev_row.get("supporting_facts") or []:
            if isinstance(fact, (list, tuple)) and len(fact) == 2:
                title = str(fact[0] or "").strip()
                if title:
                    supporting_titles.setdefault(title, []).append(fact[1])

        question_evidence: list[GoldEvidence] = []
        seen_docs: set[str] = set()
        for title, sent_ids in supporting_titles.items():
            paragraph_text = ctx_text_by_title.get(title)
            if paragraph_text is None:
                continue
            document_id = doc_id_by_key.get((title, paragraph_text))
            if document_id is None or document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            question_evidence.append(
                GoldEvidence(
                    evidence_id=deterministic_id("ev", [question_id, document_id]),
                    question_id=question_id,
                    document_id=document_id,
                    text=paragraph_text,
                    start_char=None,
                    end_char=None,
                    metadata={"supporting_sent_ids": sent_ids},
                )
            )

        answer = raw_question.get("answer")
        answer_id = raw_question.get("answer_id")
        answer_aliases: list[str] = []
        if isinstance(answer_id, str):
            answer_str = str(answer or "").strip().lower()
            answer_aliases = [
                alias
                for alias in aliases_by_qid.get(answer_id, [])
                if alias.strip().lower() != answer_str
            ]
        raw_evidences = raw_question.get("evidences")

        questions.append(
            Question(
                question_id=question_id,
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                document_id=None,
                question=question_text,
                question_type=TWOWIKI_QUESTION_TYPE,
                clause_type=None,
                gold_answer=_normalize_gold_answer(answer),
                gold_evidence=[e.evidence_id for e in question_evidence] or None,
                metadata={
                    "source_question_id": source_question_id,
                    "answer_aliases": answer_aliases,
                    "answer_id": answer_id,
                    "type": dev_row.get("type"),
                    "evidences": list(raw_evidences) if isinstance(raw_evidences, list) else [],
                    "provenance": "hipporag_reproduce_dataset",
                },
            )
        )
        evidence.extend(question_evidence)

    documents = list(documents_by_id.values())
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]
    return CanonicalDataset(
        documents=documents,
        chunks=chunks,
        questions=questions,
        gold_evidence=evidence,
    )


def _resolver_amostra_hipporag(
    config: DatasetConfig, sample_size: int | None
) -> tuple[int | None, int, bool]:
    """Quantas perguntas, com que semente, e se veio do config.

    Devolve `(num_questions, seed, veio_do_config)`. O terceiro elemento decide
    o método: do config, amostra aleatória com semente; do argumento, corte por
    prefixo, que é o que o `--sample-size` sempre fez.

    O argumento ganha ao config quando ambos existem — quem escreve
    `--sample-size` na linha de comandos está a pedir outra coisa de propósito.
    """
    if sample_size is not None:
        return sample_size, 42, False

    bruto = getattr(config, "sample", None)
    if not isinstance(bruto, dict):
        return None, 42, False

    num_questions = bruto.get("num_questions")
    if not isinstance(num_questions, int) or num_questions <= 0:
        return None, 42, False

    seed = bruto.get("seed", 42)
    if not isinstance(seed, int):
        seed = 42
    return num_questions, seed, True


def _chaves_de_contexto(raw_questions: list) -> set[tuple[str, str]]:
    """As passagens que aparecem no contexto destas perguntas.

    A chave é `(título, texto)` com o texto montado pela mesma convenção do
    ficheiro de corpus — `" ".join(sentenças)` — que é o que torna as duas
    fontes comparáveis. Uma passagem de contexto que não exista no corpus
    publicado (há uma, o distractor «Tōku Made») simplesmente não encontra par,
    e o resultado é o mesmo de antes: fica de fora.
    """
    chaves: set[tuple[str, str]] = set()
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        for raw_paragraph in raw_question.get("context") or []:
            if not isinstance(raw_paragraph, (list, tuple)) or len(raw_paragraph) != 2:
                continue
            title = str(raw_paragraph[0] or "").strip()
            sentences = raw_paragraph[1] if isinstance(raw_paragraph[1], list) else []
            texto = " ".join(str(s).strip() for s in sentences if str(s).strip())
            chaves.add((title, texto))
    return chaves


class TwowikiRawDataMissingError(FileNotFoundError):
    pass


def load_twowiki(
    config: DatasetConfig,
    *,
    sample_size: int | None = None,
    raw_file: str | Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> CanonicalDataset:
    """Load 2WikiMultiHopQA raw JSON into the canonical dataset format.

    Espelha o musique_loader: passagens deduplicadas por (title, paragraph_text)
    via deterministic_id — duas perguntas que citam a mesma passagem produzem UM
    Document; cada Question aponta para as passagens-suporte via gold_evidence
    (nível de documento, derivado dos supporting_facts). Amostragem = random
    simples com seed do config (mesmo protocolo do MuSiQue eval1k; a distribuição
    por tipo segue a do dev por expectativa).

    Diferenças de formato tratadas aqui (e só aqui):
    - raw é um JSON array (não JSONL); context = [title, [sentences]] e o texto
      do parágrafo é " ".join(sentences) (sentenças do 2wiki não trazem espaço
      inicial, ao contrário do HotpotQA);
    - answer_aliases vêm de id_aliases.json (JSONL Q_id -> aliases) via answer_id
      (Wikidata), cobrindo ~84% do dev;
    - metadata da pergunta guarda type (comparison/inference/compositional/
      bridge_comparison) e evidences (triplas Wikidata do caminho de raciocínio).
    """
    resolved_raw_file = (
        Path(raw_file) if raw_file is not None else find_twowiki_raw_file(config.raw_path)
    )
    raw_questions = _read_json_array(resolved_raw_file)
    aliases_by_qid = _load_aliases(Path(config.raw_path) / ALIASES_FILE)

    selected_questions = _apply_sampling(
        raw_questions,
        sample_size=sample_size,
        sample_config=_resolve_sample_config(config),
    )

    documents_by_id: dict[str, Document] = {}
    questions: list[Question] = []
    evidence: list[GoldEvidence] = []

    for raw_question in selected_questions:
        if not isinstance(raw_question, dict):
            continue

        question_text = str(raw_question.get("question") or "").strip()
        if not question_text:
            continue

        source_question_id = raw_question.get("_id")
        question_id = deterministic_id(
            "q",
            [
                config.dataset_id,
                config.dataset_version,
                source_question_id,
                question_text,
            ],
        )

        question_paragraphs = _collect_paragraphs(
            raw_question.get("context"),
            config=config,
            source_path=resolved_raw_file,
            documents_by_id=documents_by_id,
        )

        question_evidence = _build_question_evidence(
            question_id=question_id,
            paragraphs=question_paragraphs,
            supporting_facts=raw_question.get("supporting_facts"),
        )

        question = _build_question(
            config=config,
            question_id=question_id,
            question_text=question_text,
            source_question_id=source_question_id,
            raw_question=raw_question,
            aliases_by_qid=aliases_by_qid,
            evidence_ids=[item.evidence_id for item in question_evidence],
        )

        questions.append(question)
        evidence.extend(question_evidence)

    documents = list(documents_by_id.values())
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]

    return CanonicalDataset(
        documents=documents,
        chunks=chunks,
        questions=questions,
        gold_evidence=evidence,
    )


def find_twowiki_raw_file(raw_path: str | Path) -> Path:
    root = Path(raw_path)
    for candidate in RAW_FILE_CANDIDATES:
        path = root / candidate
        if path.exists():
            return path

    expected = ", ".join(str(root / candidate) for candidate in RAW_FILE_CANDIDATES)
    raise TwowikiRawDataMissingError(
        "2WikiMultiHopQA raw data was not found. Download is intentionally not "
        f"automatic. Place the raw JSON at one of: {expected}"
    )


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return [row for row in data if isinstance(row, dict)]


def _load_aliases(path: Path) -> dict[str, list[str]]:
    """id_aliases.json é JSONL: {\"Q_id\": ..., \"aliases\": [...]} por linha."""
    aliases: dict[str, list[str]] = {}
    if not path.exists():
        return aliases
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            qid = row.get("Q_id")
            raw_aliases = row.get("aliases")
            if isinstance(qid, str) and isinstance(raw_aliases, list):
                aliases[qid] = [str(a) for a in raw_aliases]
    return aliases


def _resolve_sample_config(config: DatasetConfig) -> dict[str, Any]:
    raw = getattr(config, "sample", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _apply_sampling(
    raw_questions: list[dict[str, Any]],
    *,
    sample_size: int | None,
    sample_config: dict[str, Any],
) -> list[dict[str, Any]]:
    num_questions = sample_size
    if num_questions is None:
        candidate = sample_config.get("num_questions")
        if isinstance(candidate, int):
            num_questions = candidate

    if num_questions is None or num_questions >= len(raw_questions):
        return list(raw_questions)
    if num_questions <= 0:
        return []

    seed_value = sample_config.get("seed", 42)
    if not isinstance(seed_value, int):
        seed_value = 42

    rng = random.Random(seed_value)
    return rng.sample(raw_questions, num_questions)


def _collect_paragraphs(
    raw_context: Any,
    *,
    config: DatasetConfig,
    source_path: Path,
    documents_by_id: dict[str, Document],
) -> list[tuple[Document, str]]:
    """context = [[title, [sentences]], ...] -> [(Document, title), ...]."""
    if not isinstance(raw_context, list):
        return []

    collected: list[tuple[Document, str]] = []
    for raw_paragraph in raw_context:
        if not isinstance(raw_paragraph, (list, tuple)) or len(raw_paragraph) != 2:
            continue

        title = str(raw_paragraph[0] or "").strip()
        sentences = raw_paragraph[1]
        if not isinstance(sentences, list):
            continue
        paragraph_text = " ".join(str(s).strip() for s in sentences if str(s).strip())
        if not title and not paragraph_text:
            continue

        document_id = deterministic_id(
            "doc",
            [config.dataset_id, config.dataset_version, title, paragraph_text],
        )

        document = documents_by_id.get(document_id)
        if document is None:
            passage = f"{title}\n\n{paragraph_text}" if title else paragraph_text
            document = Document(
                document_id=document_id,
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                title=title or None,
                text=passage,
                source_path=str(source_path),
                metadata={"paragraph_text": paragraph_text},
            )
            documents_by_id[document_id] = document

        collected.append((document, title))

    return collected


def _build_question_evidence(
    *,
    question_id: str,
    paragraphs: list[tuple[Document, str]],
    supporting_facts: Any,
) -> list[GoldEvidence]:
    """supporting_facts = [[title, sent_idx], ...] -> gold a nível de documento.

    O título identifica o parágrafo DENTRO do contexto da própria pergunta
    (auditoria de 2026-07-08: 0 supporting_facts órfãos no dev inteiro).
    """
    supporting_titles: dict[str, list[int]] = {}
    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and len(fact) == 2:
                title = str(fact[0] or "").strip()
                if title:
                    supporting_titles.setdefault(title, []).append(fact[1])

    evidence: list[GoldEvidence] = []
    seen: set[str] = set()
    for document, title in paragraphs:
        if title not in supporting_titles:
            continue
        if document.document_id in seen:
            continue
        seen.add(document.document_id)

        evidence_id = deterministic_id("ev", [question_id, document.document_id])
        evidence.append(
            GoldEvidence(
                evidence_id=evidence_id,
                question_id=question_id,
                document_id=document.document_id,
                text=str(document.metadata.get("paragraph_text") or ""),
                start_char=None,
                end_char=None,
                metadata={"supporting_sent_ids": supporting_titles[title]},
            )
        )

    return evidence


def _build_question(
    *,
    config: DatasetConfig,
    question_id: str,
    question_text: str,
    source_question_id: Any,
    raw_question: dict[str, Any],
    aliases_by_qid: dict[str, list[str]],
    evidence_ids: list[str],
) -> Question:
    answer = raw_question.get("answer")
    answer_id = raw_question.get("answer_id")
    answer_aliases: list[str] = []
    if isinstance(answer_id, str):
        answer_str = str(answer or "").strip().lower()
        answer_aliases = [
            alias
            for alias in aliases_by_qid.get(answer_id, [])
            if alias.strip().lower() != answer_str
        ]
    raw_evidences = raw_question.get("evidences")
    evidences = list(raw_evidences) if isinstance(raw_evidences, list) else []

    return Question(
        question_id=question_id,
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        document_id=None,
        question=question_text,
        question_type=TWOWIKI_QUESTION_TYPE,
        clause_type=None,
        gold_answer=_normalize_gold_answer(answer),
        gold_evidence=evidence_ids or None,
        metadata={
            "source_question_id": source_question_id,
            "answer_aliases": answer_aliases,
            "answer_id": answer_id,
            "type": raw_question.get("type"),
            "evidences": evidences,
        },
    )


def _normalize_gold_answer(answer: Any) -> str | list[str] | bool | None:
    if isinstance(answer, (str, bool)) or answer is None:
        return answer
    if isinstance(answer, list):
        return [str(value) for value in answer]
    return str(answer)
