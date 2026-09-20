"""Experiment pipeline: dataset loading -> prompt construction -> LLM inference -> metrics -> tracking."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from .analysis import summarize
from .config import ExperimentConfig, GenerationSpec, ModelSpec
from .datasets import load_dataset
from .metrics import SemanticScorer, evaluate_prediction
from .models import LLM, load_model
from .prompts import get_template
from .utils import environment_info, set_seed

log = logging.getLogger(__name__)

ModelLoader = Callable[[ModelSpec, GenerationSpec, "str | None"], LLM]


def postprocess(text: str) -> str:
    """Keep the first non-empty line: models often keep talking after the answer."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass
class RunResult:
    run_dir: Path
    results: pd.DataFrame
    summary: pd.DataFrame


class ExperimentRunner:
    """Runs every (model x prompt x dataset) combination and records one row per example."""

    def __init__(self, config: ExperimentConfig, model_loader: ModelLoader = load_model,
                 scorer: SemanticScorer | None = None):
        self.config = config
        self.model_loader = model_loader
        self.scorer = scorer

    # ---- run directory ---------------------------------------------------------------
    def _create_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(self.config.output_dir) / f"{self.config.experiment_name}_{stamp}"
        run_dir, counter = base, 1
        while run_dir.exists():
            counter += 1
            run_dir = base.with_name(f"{base.name}-{counter}")
        run_dir.mkdir(parents=True)
        return run_dir

    def _save_metadata(self, run_dir: Path) -> None:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(self.config), sort_keys=False), encoding="utf-8")
        (run_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")

    # ---- main loop -------------------------------------------------------------------
    def run(self) -> RunResult:
        cfg = self.config
        set_seed(cfg.seed)
        run_dir = self._create_run_dir()
        self._save_metadata(run_dir)
        log.info("Run directory: %s", run_dir)

        scorer = self.scorer
        if scorer is None and cfg.semantic_model:
            scorer = SemanticScorer(cfg.semantic_model, cfg.device)

        datasets = {d.name: load_dataset(d.path, limit=cfg.max_examples) for d in cfg.datasets}
        templates = {name: get_template(name) for name in cfg.prompts}  # fail fast on typos

        rows: list[dict] = []
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as sink:
            for spec in cfg.models:
                log.info("Loading model %s (%s)", spec.name, spec.hf_id)
                model = self.model_loader(spec, cfg.generation, cfg.device)
                try:
                    if cfg.warmup:
                        model.generate([{"role": "user", "content": "Hello"}])
                    for prompt_name, template in templates.items():
                        for ds_name, examples in datasets.items():
                            log.info("  %s | %s | %s (%d examples)", spec.name, prompt_name, ds_name, len(examples))
                            for ex in examples:
                                generation = model.generate(template.to_messages(ex))
                                prediction = postprocess(generation.text)
                                metrics = evaluate_prediction(prediction, ex.answers, scorer)
                                row = {
                                    "model": spec.name,
                                    "prompt": prompt_name,
                                    "dataset": ds_name,
                                    "example_id": ex.id,
                                    "question": ex.question,
                                    "gold_answers": " | ".join(ex.answers),
                                    "raw_output": generation.text,
                                    "prediction": prediction,
                                    **metrics,
                                    "latency_s": generation.latency_s,
                                    "new_tokens": generation.new_tokens,
                                    "tokens_per_s": (generation.new_tokens / generation.latency_s)
                                    if generation.latency_s > 0 else float("nan"),
                                }
                                rows.append(row)
                                # Written immediately so a crash never loses finished work.
                                sink.write(json.dumps({k: _json_safe(v) for k, v in row.items()}) + "\n")
                                sink.flush()
                finally:
                    model.close()

        results = pd.DataFrame(rows)
        results.to_csv(run_dir / "results.csv", index=False)
        summary = summarize(results)
        summary.to_csv(run_dir / "summary.csv", index=False)
        log.info("Finished %d generations", len(results))
        return RunResult(run_dir=run_dir, results=results, summary=summary)
