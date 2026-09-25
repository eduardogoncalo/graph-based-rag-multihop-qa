from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str
    dataset_version: str
    name: str
    raw_path: str
    canonical_path: str
    splits_path: str


class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method_id: str
    enabled: bool = True
    description: str | None = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    dataset_id: str
    dataset_version: str
    methods: list[str] = Field(default_factory=list)
    agent_modes: list[str] = Field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml(path))


def load_method_config(path: str | Path) -> MethodConfig:
    return MethodConfig.model_validate(load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))
