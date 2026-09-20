import json

import pandas as pd
import pytest

from conftest import ScriptedLLM
from llm_eval_lab.runner import ExperimentRunner, postprocess


def _loader(registry):
    def load(spec, generation, device):
        llm = ScriptedLLM(spec.name, mode="perfect" if spec.name == "good" else "wrong")
        registry[spec.name] = llm
        return llm

    return load


def test_postprocess_keeps_first_non_empty_line():
    assert postprocess("\n  Paris  \nsecond line") == "Paris"
    assert postprocess("   ") == ""


def test_full_matrix_is_executed_and_tracked(make_config, scorer):
    registry = {}
    result = ExperimentRunner(make_config(), _loader(registry), scorer=scorer).run()

    # 2 models x 2 prompts x 3 examples
    assert len(result.results) == 12
    assert set(result.results["model"]) == {"good", "bad"}
    assert set(result.results["prompt"]) == {"zero_shot", "concise"}

    files = {p.name for p in result.run_dir.iterdir()}
    assert {"config.yaml", "environment.json", "predictions.jsonl", "results.csv", "summary.csv"} <= files

    lines = (result.run_dir / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 12 and "question" in json.loads(lines[0])
    assert all(llm.closed for llm in registry.values())  # memory released after each model


def test_metrics_reflect_model_quality(make_config, scorer):
    result = ExperimentRunner(make_config(), _loader({}), scorer=scorer).run()
    by_model = result.results.groupby("model")[["exact_match", "f1"]].mean()
    assert by_model.loc["good", "exact_match"] == 1.0
    assert by_model.loc["bad", "f1"] == 0.0
    assert result.results["latency_s"].eq(0.05).all()
    assert result.results["tokens_per_s"].gt(0).all()


def test_chatty_output_is_scored_by_first_line_and_contains_match(make_config, scorer):
    def loader(spec, generation, device):
        return ScriptedLLM(spec.name, mode="chatty")

    result = ExperimentRunner(make_config(models=("m",), prompts=("zero_shot",)), loader, scorer=scorer).run()
    row = result.results.iloc[0]
    assert row["prediction"] == "The answer is Paris."
    assert row["exact_match"] == 0.0 and row["contains_match"] == 1.0
    assert "Let me know" in row["raw_output"]  # raw text preserved for auditing


def test_warmup_call_is_made_but_not_recorded(make_config, scorer):
    registry = {}
    result = ExperimentRunner(make_config(models=("good",), prompts=("zero_shot",), warmup=True),
                              _loader(registry), scorer=scorer).run()
    assert len(registry["good"].calls) == 3 + 1  # 3 examples + 1 warm-up
    assert len(result.results) == 3


def test_max_examples_limits_each_dataset(make_config, scorer):
    result = ExperimentRunner(make_config(models=("good",), prompts=("zero_shot",), max_examples=2),
                              _loader({}), scorer=scorer).run()
    assert len(result.results) == 2


def test_semantic_scores_are_nan_without_scorer(make_config):
    result = ExperimentRunner(make_config(models=("good",), prompts=("zero_shot",)), _loader({})).run()
    assert result.results["semantic_similarity"].isna().all()


def test_unknown_prompt_fails_before_loading_any_model(make_config):
    loaded = []
    runner = ExperimentRunner(make_config(prompts=("does_not_exist",)),
                              lambda s, g, d: loaded.append(s) or ScriptedLLM(s.name))
    with pytest.raises(KeyError):
        runner.run()
    assert loaded == []


def test_model_is_closed_even_if_generation_crashes(make_config):
    llm = ScriptedLLM("boom")

    def explode(messages):
        raise RuntimeError("out of memory")

    llm.generate = explode
    runner = ExperimentRunner(make_config(models=("boom",), prompts=("zero_shot",)), lambda s, g, d: llm)
    with pytest.raises(RuntimeError):
        runner.run()
    assert llm.closed
