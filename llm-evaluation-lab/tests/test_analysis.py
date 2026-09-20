import pandas as pd
import pytest

from conftest import ScriptedLLM
from llm_eval_lab.analysis import (
    analyze_run,
    failure_cases,
    load_results,
    pareto_front,
    prompt_comparison,
    summarize,
)
from llm_eval_lab.runner import ExperimentRunner


def _frame():
    rows = []
    for model, prompt, f1, latency in [
        ("a", "p1", 1.0, 2.0),   # best quality, slowest
        ("a", "p2", 0.5, 2.5),   # dominated by (a, p1)
        ("b", "p1", 0.4, 0.5),   # fastest
        ("b", "p2", 0.3, 0.6),   # dominated by (b, p1)
    ]:
        for i in range(4):
            rows.append({"model": model, "prompt": prompt, "dataset": "d", "example_id": f"e{i}",
                         "exact_match": f1, "contains_match": f1, "f1": f1, "semantic_similarity": f1,
                         "latency_s": latency + i * 0.01, "tokens_per_s": 10.0,
                         "question": "q", "gold_answers": "g", "prediction": "p"})
    return pd.DataFrame(rows)


def test_summarize_aggregates_per_group():
    summary = summarize(_frame())
    assert len(summary) == 4
    row = summary[(summary.model == "a") & (summary.prompt == "p1")].iloc[0]
    assert row["n"] == 4 and row["f1"] == 1.0
    assert row["latency_p50_s"] == pytest.approx(2.015)
    assert row["latency_p95_s"] >= row["latency_p50_s"]


def test_pareto_front_marks_dominated_configurations():
    front = pareto_front(summarize(_frame(), by=["model", "prompt"]))
    flags = {(r.model, r.prompt): r.pareto_optimal for r in front.itertuples()}
    assert flags == {("a", "p1"): True, ("b", "p1"): True, ("a", "p2"): False, ("b", "p2"): False}


def test_prompt_comparison_pivots_models_by_prompt():
    table = prompt_comparison(_frame())
    assert table.loc["a", "p1"] == 1.0 and table.loc["b", "p2"] == pytest.approx(0.3)


def test_failure_cases_filters_misses():
    df = _frame()
    df.loc[df.model == "b", "contains_match"] = 0
    assert set(failure_cases(df)["model"]) == {"b"}


def test_analyze_run_writes_tables_plots_and_report(make_config, scorer):
    def loader(spec, generation, device):
        return ScriptedLLM(spec.name, mode="perfect" if spec.name == "good" else "wrong",
                           latency_s=0.05 if spec.name == "good" else 0.01)

    run = ExperimentRunner(make_config(), loader, scorer=scorer).run()
    outputs = analyze_run(run.run_dir)

    for name in ("summary_by_dataset.csv", "summary_overall.csv", "prompt_comparison.csv",
                 "tradeoff.png", "prompt_comparison.png", "report.md"):
        assert outputs[name].exists() and outputs[name].stat().st_size > 0

    report = outputs["report.md"].read_text()
    assert "Best configuration by token F1: **good +" in report
    assert "pareto_optimal" in report
    assert len(load_results(run.run_dir)) == 12


def test_analyze_run_without_plots(make_config, scorer, tmp_path):
    run = ExperimentRunner(make_config(models=("good",)), lambda s, g, d: ScriptedLLM("good"), scorer=scorer).run()
    outputs = analyze_run(run.run_dir, plots=False)
    assert "tradeoff.png" not in outputs


def test_load_results_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)
