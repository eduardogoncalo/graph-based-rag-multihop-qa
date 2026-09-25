from __future__ import annotations

import ast
import importlib.metadata as metadata
import inspect
from pathlib import Path

import lightrag
from lightrag import LightRAG, QueryParam


def main() -> None:
    package_root = Path(lightrag.__file__).resolve().parent
    kg_init = package_root / "kg" / "__init__.py"
    neo4j_impl = package_root / "kg" / "neo4j_impl.py"

    print("LightRAG import OK")
    print(f"package_version={metadata.version('lightrag-hku')}")
    print(f"package_location={package_root}")
    print(f"LightRAG_signature={inspect.signature(LightRAG)}")
    print(f"QueryParam_signature={inspect.signature(QueryParam)}")
    print("storage_registry=")
    for line in _storage_registry_lines(kg_init):
        print(line)
    print("neo4j_env_names=")
    for name in _neo4j_env_names(neo4j_impl):
        print(name)


def _storage_registry_lines(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "STORAGE_IMPLEMENTATIONS" in target_names or "STORAGES" in target_names:
            lines.append(f"{target_names[0]}={ast.unparse(node.value)}")
        if "STORAGE_ENV_REQUIREMENTS" in target_names:
            lines.append(f"{target_names[0]}={ast.unparse(node.value)}")
    return lines


def _neo4j_env_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if "NEO4J" in first.value.upper():
                names.add(first.value)
    return sorted(names)


if __name__ == "__main__":
    main()
