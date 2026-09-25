from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"
NOT_ENOUGH_INFORMATION = "I do not have enough information to answer."


@dataclass(frozen=True)
class DocumentScopeResult:
    raw_data: dict[str, Any]
    final_context: str
    scope_metadata: dict[str, Any]

    @property
    def data(self) -> dict[str, list[dict[str, Any]]]:
        data = self.raw_data.get("data")
        if not isinstance(data, dict):
            return {"entities": [], "relationships": [], "chunks": [], "references": []}
        return {
            "entities": _list_of_dicts(data.get("entities")),
            "relationships": _list_of_dicts(data.get("relationships")),
            "chunks": _list_of_dicts(data.get("chunks")),
            "references": _list_of_dicts(data.get("references")),
        }


def apply_document_scope(
    raw_data: dict[str, Any],
    target_document_id: str,
    chunk_document_map: dict[str, str] | None = None,
) -> DocumentScopeResult:
    chunk_document_map = chunk_document_map or {}
    data = raw_data.get("data")
    data = data if isinstance(data, dict) else {}

    entities = _list_of_dicts(data.get("entities"))
    relationships = _list_of_dicts(data.get("relationships"))
    chunks = _list_of_dicts(data.get("chunks"))

    scoped_entities = [
        item
        for item in (_scope_source_item(item, target_document_id, chunk_document_map) for item in entities)
        if item is not None
    ]
    scoped_relationships = [
        item
        for item in (
            _scope_source_item(item, target_document_id, chunk_document_map)
            for item in relationships
        )
        if item is not None
    ]
    scoped_chunks = [
        item
        for item in (_scope_chunk(item, target_document_id, chunk_document_map) for item in chunks)
        if item is not None
    ]
    references = _references_from_chunks(scoped_chunks)
    scoped_data = {
        "entities": scoped_entities,
        "relationships": scoped_relationships,
        "chunks": scoped_chunks,
        "references": references,
    }
    scoped_raw_data = {
        **raw_data,
        "data": scoped_data,
        "metadata": {
            **(raw_data.get("metadata") if isinstance(raw_data.get("metadata"), dict) else {}),
            "document_scope": {
                "target_document_id": target_document_id,
                "document_scope_applied": True,
                "scope_mode": "strict",
                "scope_fallback": "none",
            },
        },
    }
    scope_metadata = {
        "target_document_id": target_document_id,
        "document_scope_applied": True,
        "scope_mode": "strict",
        "scope_fallback": "none",
        "nodes_before_scope": len(entities),
        "nodes_after_scope": len(scoped_entities),
        "relationships_before_scope": len(relationships),
        "relationships_after_scope": len(scoped_relationships),
        "chunks_before_scope": len(chunks),
        "chunks_after_scope": len(scoped_chunks),
    }
    return DocumentScopeResult(
        raw_data=scoped_raw_data,
        final_context=build_scoped_final_context(scoped_data),
        scope_metadata=scope_metadata,
    )


def build_scoped_final_context(data: dict[str, Any]) -> str:
    entities = _list_of_dicts(data.get("entities"))
    relationships = _list_of_dicts(data.get("relationships"))
    chunks = _list_of_dicts(data.get("chunks"))
    references = _list_of_dicts(data.get("references")) or _references_from_chunks(chunks)
    entities_str = "\n".join(json.dumps(entity, ensure_ascii=False) for entity in entities)
    relationships_str = "\n".join(
        json.dumps(relationship, ensure_ascii=False) for relationship in relationships
    )
    chunks_context = [
        {
            "reference_id": chunk.get("reference_id") or str(index),
            "content": chunk.get("content") or chunk.get("text") or "",
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    chunks_str = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks_context)
    references_str = "\n".join(
        f"[{reference.get('reference_id')}] {reference.get('file_path') or ''}".rstrip()
        for reference in references
        if reference.get("reference_id")
    )
    return f"""
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relationships_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{references_str}
```
""".strip()


def build_scoped_system_prompt(final_context: str) -> str:
    """Prompt do sistema da variante `lightrag_neo4j_doc_scoped_strict`.

    **Esta variante é inalcançável nos datasets que o pacote entrega**, e a
    razão é estrutural: exige `target_document_id` e levanta `ValueError` sem
    ele, mas as perguntas do MuSiQue e do 2Wiki são multi-documento por
    construção — os dois carregadores põem `document_id=None`. O único
    carregador que preenchia esse campo era o do CUAD, morto desde Junho de
    2026. Nenhum config de método a declara, e nem a CLI nem o `run_single`
    lhe despacham.

    O texto abaixo era da era CUAD: dizia «CUAD benchmark question» e pedia
    «contract-span answers». Foi generalizado a 2026-08-09. Como a variante não
    corre, a alteração **não muda resultado nenhum** do que o pacote executa —
    o que muda é que deixa de ensinar a quem lê que este pacote avalia
    contratos.

    A variante não foi removida: arrancá-la mexia no adaptador do LightRAG, que
    é um método vivo da dissertação, para não ganhar nada. Fica em pé e
    documentada.
    """
    return f"""---Role---

You are an expert AI assistant answering a document-scoped benchmark question.
Use only the provided Context. Do not use outside knowledge.

---Instructions---

1. Answer only if the Context contains direct evidence.
2. Prefer concise span answers when the question asks for a name, date, entity, or value.
3. If the answer cannot be found in the Context, state exactly: {NOT_ENOUGH_INFORMATION}
4. Do not invent citations.

---Context---

{final_context}
"""


def _scope_chunk(
    chunk: dict[str, Any],
    target_document_id: str,
    chunk_document_map: dict[str, str],
) -> dict[str, Any] | None:
    if _chunk_belongs_to_document(chunk, target_document_id, chunk_document_map):
        return dict(chunk)
    return None


def _scope_source_item(
    item: dict[str, Any],
    target_document_id: str,
    chunk_document_map: dict[str, str],
) -> dict[str, Any] | None:
    source_ids = _split_source_ids(item.get("source_id"))
    target_sources = [
        source_id
        for source_id in source_ids
        if _source_belongs_to_document(source_id, target_document_id, chunk_document_map)
    ]
    if not target_sources:
        return None
    scoped = dict(item)
    scoped["source_id"] = GRAPH_FIELD_SEP.join(target_sources)
    return scoped


def _chunk_belongs_to_document(
    chunk: dict[str, Any],
    target_document_id: str,
    chunk_document_map: dict[str, str],
) -> bool:
    for key in ("source_document_id", "document_id"):
        value = chunk.get(key)
        if _normalize_id(value) == target_document_id:
            return True
    file_path = str(chunk.get("file_path") or "")
    if target_document_id in file_path:
        return True
    for key in ("chunk_id", "source_id"):
        value = chunk.get(key)
        if any(
            _source_belongs_to_document(source_id, target_document_id, chunk_document_map)
            for source_id in _split_source_ids(value)
        ):
            return True
    return False


def _source_belongs_to_document(
    source_id: str,
    target_document_id: str,
    chunk_document_map: dict[str, str],
) -> bool:
    source_id = _normalize_id(source_id)
    if not source_id:
        return False
    if source_id == target_document_id:
        return True
    mapped_document = chunk_document_map.get(source_id)
    if mapped_document == target_document_id:
        return True
    return _document_id_from_chunk_id(source_id) == target_document_id


def _document_id_from_chunk_id(source_id: str) -> str | None:
    if "-chunk" in source_id:
        return source_id.split("-chunk", 1)[0]
    match = re.match(r"^(doc_[A-Za-z0-9]+)", source_id)
    if match:
        return match.group(1)
    return None


def _split_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(GRAPH_FIELD_SEP)
    return [_normalize_id(item) for item in raw_values if _normalize_id(item)]


def _normalize_id(value: Any) -> str:
    return str(value or "").strip()


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _references_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        reference_id = str(chunk.get("reference_id") or index)
        if reference_id in seen:
            continue
        seen.add(reference_id)
        references.append(
            {
                "reference_id": reference_id,
                "file_path": chunk.get("file_path") or chunk.get("source_document_id") or "",
            }
        )
    return references
