"""Experiment configuration (YAML) with validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    name: str   # short label used in reports, e.g. "qwen2.5-0.5b"
    hf_id: str  # Hugging Face model id or local path


@dataclass
class DatasetSpec:
    name: str
    path: str


@dataclass
class GenerationSpec:
    max_new_tokens: int = 32
    temperature: float = 0.0  # 0 => greedy decoding


@dataclass
class ExperimentConfig:
    experiment_name: str
    models: list[ModelSpec]
    prompts: list[str]
    datasets: list[DatasetSpec]
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    semantic_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    max_examples: int | None = None
    output_dir: str = "results"
    seed: int = 42
    warmup: bool = True
    device: str | None = None

    def validate(self) -> None:
        if not self.models:
            raise ValueError("Config must list at least one model")
        if not self.prompts:
            raise ValueError("Config must list at least one prompt template")
        if not self.datasets:
            raise ValueError("Config must list at least one dataset")
        for label, names in (("model", [m.name for m in self.models]), ("dataset", [d.name for d in self.datasets])):
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {label} names in config: {names}")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be a positive integer or null")
        if self.generation.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive")


def config_from_dict(raw: dict) -> ExperimentConfig:
    try:
        cfg = ExperimentConfig(
            experiment_name=raw["experiment_name"],
            models=[ModelSpec(**m) for m in raw["models"]],
            prompts=list(raw["prompts"]),
            datasets=[DatasetSpec(**d) for d in raw["datasets"]],
            generation=GenerationSpec(**raw.get("generation", {})),
            semantic_model=raw.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2"),
            max_examples=raw.get("max_examples"),
            output_dir=raw.get("output_dir", "results"),
            seed=raw.get("seed", 42),
            warmup=raw.get("warmup", True),
            device=raw.get("device"),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a valid experiment config")
    return config_from_dict(raw)
