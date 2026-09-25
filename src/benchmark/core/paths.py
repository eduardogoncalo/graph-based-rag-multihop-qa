from __future__ import annotations

from pathlib import Path


def dataset_artifact_dir(
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str,
) -> Path:
    return Path(artifacts_dir) / f"{dataset_id}_{dataset_version}" / method_id
