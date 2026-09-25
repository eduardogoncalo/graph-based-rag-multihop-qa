"""reproduce.sh has to run the cells the thesis ran, under the names the rest of
the protocol reads.

The retrieval parameters come from the scripts that produced the thesis cells
(run_lightrag_v1free.sh, run_ms_graphrag_v1free.sh, run_vector_v1free.sh,
run_cognee_v1free_k5.sh, hipporag2_official_supervisor.sh and
twowiki_arm_a_orchestrator.sh in the experimental repository). The cell names
have to match what the statistics, the consolidation and the audit look for,
or those steps find nothing.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "scripts" / "reproduce.sh"

THESIS_TOP_K = {
    "vector_rag": 5,
    "lightrag_neo4j": 40,
    "ms_graphrag": 20,
    "hipporag2": 5,
    "cognee": 5,
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(name: str) -> str:
    """The text of one shell function of reproduce.sh."""
    lines = REPRODUCE.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _call(function: str, *args: str, env: dict[str, str] | None = None) -> str:
    script = _function(function) + f'\n{function} "$@"\n'
    return subprocess.run(
        ["bash", "-c", script, "bash", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _suffix(method: str, reader: str = "v1") -> str:
    return _call("sufixo_de", method, env={"LEITOR": reader, "PATH": "/usr/bin:/bin"})


@pytest.mark.parametrize(("method", "top_k"), THESIS_TOP_K.items())
def test_top_k_is_the_thesis_value(method: str, top_k: int) -> None:
    assert _call("topk_de", method) == str(top_k)


def test_chunk_top_k_is_pinned_to_twenty() -> None:
    assert 'export CHUNK_TOP_K="${CHUNK_TOP_K:-20}"' in REPRODUCE.read_text(encoding="utf-8")


def test_controlled_cells_are_the_ones_the_statistics_read() -> None:
    mcnemar = _load("mcnemar_v1_arm")
    ours = {_suffix(method) for method in THESIS_TOP_K}
    theirs = {cell for pair in mcnemar.PRIMARY_SUFFIX for cell in pair}
    assert theirs <= ours


def test_native_against_controlled_pairs_find_both_cells() -> None:
    native = _load("mcnemar_native_arm")
    controlled = {_suffix(method) for method in THESIS_TOP_K}
    for _native_cell, controlled_cell in native.FAMILY_AB_SUFFIX:
        assert controlled_cell in controlled


def test_audit_cells_match_the_experiments_reproduce_creates() -> None:
    audit = _load("retrieval_audit_canonic")
    methods = [*THESIS_TOP_K, "single_document_context"]
    ours = {f"p_{_suffix(method)}" for method in methods}
    assert {experiment for experiment, *_ in audit.cells("p_")} == ours


def test_consolidation_reads_the_summaries_the_judge_writes() -> None:
    text = (ROOT / "scripts" / "consolidate_p5.py").read_text(encoding="utf-8")
    methods = [*THESIS_TOP_K, "single_document_context", "zero_shot_no_context"]
    for method in methods:
        assert f'"llm_judge_full_{_suffix(method)}_summary.json"' in text
    for native in ("lightrag", "ms_graphrag", "hipporag2", "cognee"):
        assert f'"llm_judge_full_{native}_native_summary.json"' in text


def test_a_grounded_run_never_gets_a_thesis_name() -> None:
    for method in THESIS_TOP_K:
        suffix = _suffix(method, reader="v2")
        assert "v1free" not in suffix
        assert "v2grounded" in suffix


def test_the_canonical_directory_comes_from_the_dataset_config() -> None:
    text = REPRODUCE.read_text(encoding="utf-8")
    assert 'CANONICO="data/canonical/${SLUG}"' not in text
    assert '["canonical_path"]' in text
    for config in (ROOT / "configs" / "datasets").glob("*.yaml"):
        assert "canonical_path:" in config.read_text(encoding="utf-8"), config.name
