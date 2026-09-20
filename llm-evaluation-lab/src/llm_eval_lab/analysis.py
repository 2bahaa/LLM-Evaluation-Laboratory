"""Result analysis with pandas: aggregation, prompt comparison, quality/latency trade-offs, plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ["exact_match", "contains_match", "f1", "semantic_similarity"]


def load_results(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"No results.csv in {run_dir}")
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean quality metrics and latency statistics per group (default: model x prompt x dataset)."""
    by = by or ["model", "prompt", "dataset"]
    return (
        df.groupby(by, sort=True)
        .agg(
            n=("example_id", "count"),
            exact_match=("exact_match", "mean"),
            contains_match=("contains_match", "mean"),
            f1=("f1", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            latency_mean_s=("latency_s", "mean"),
            latency_p50_s=("latency_s", "median"),
            latency_p95_s=("latency_s", lambda s: s.quantile(0.95)),
            tokens_per_s=("tokens_per_s", "mean"),
        )
        .reset_index()
    )


def prompt_comparison(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Model x prompt table of a chosen metric: which prompt style helps which model?"""
    return df.pivot_table(index="model", columns="prompt", values=metric, aggfunc="mean")


def pareto_front(summary: pd.DataFrame, quality: str = "f1", cost: str = "latency_mean_s") -> pd.DataFrame:
    """Flag configurations that no other configuration beats on both quality and latency.

    A configuration is *dominated* if another one has quality >= and latency <= with at least
    one strict inequality. Non-dominated rows form the quality/latency trade-off frontier.
    """
    data = summary.dropna(subset=[quality, cost]).copy()
    flags = []
    for _, row in data.iterrows():
        dominated = (
            (data[quality] >= row[quality])
            & (data[cost] <= row[cost])
            & ((data[quality] > row[quality]) | (data[cost] < row[cost]))
        ).any()
        flags.append(not bool(dominated))
    data["pareto_optimal"] = flags
    return data


def failure_cases(df: pd.DataFrame, metric: str = "contains_match", limit: int = 20) -> pd.DataFrame:
    """Rows where the model missed the answer - the fastest way to see *why* a prompt underperforms."""
    cols = ["model", "prompt", "dataset", "question", "gold_answers", "prediction"]
    return df.loc[df[metric] == 0, cols].head(limit)


# ---- plots ---------------------------------------------------------------------------------
def plot_tradeoff(overall: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = pareto_front(overall)
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in front.iterrows():
        ax.scatter(row["latency_mean_s"], row["f1"], s=90 if row["pareto_optimal"] else 50,
                   marker="*" if row["pareto_optimal"] else "o", zorder=3)
        ax.annotate(f'{row["model"]}\n{row["prompt"]}', (row["latency_mean_s"], row["f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean latency per answer (s)")
    ax.set_ylabel("Token F1")
    ax.set_title("Quality vs latency (stars = Pareto-optimal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_prompt_comparison(df: pd.DataFrame, path: str | Path, metric: str = "f1") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = prompt_comparison(df, metric)
    ax = table.plot(kind="bar", figsize=(7, 4.5), rot=0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model and prompt template")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---- report --------------------------------------------------------------------------------
def _markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    frame = df.round(digits)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def analyze_run(run_dir: str | Path, plots: bool = True) -> dict[str, Path]:
    """Create summary tables, plots and a Markdown report inside ``run_dir``."""
    run = Path(run_dir)
    df = load_results(run)
    outputs: dict[str, Path] = {}

    per_dataset = summarize(df)
    overall = summarize(df, by=["model", "prompt"])
    front = pareto_front(overall)
    comparison = prompt_comparison(df).reset_index()

    for name, frame in (("summary_by_dataset.csv", per_dataset), ("summary_overall.csv", front),
                        ("prompt_comparison.csv", comparison)):
        outputs[name] = run / name
        frame.to_csv(outputs[name], index=False)

    if plots:
        outputs["tradeoff.png"] = plot_tradeoff(overall, run / "tradeoff.png")
        outputs["prompt_comparison.png"] = plot_prompt_comparison(df, run / "prompt_comparison.png")

    best = front.sort_values(["f1", "latency_mean_s"], ascending=[False, True]).iloc[0]
    lines = [
        f"# Experiment report: {run.name}",
        "",
        f"* Generations evaluated: **{len(df)}**",
        f"* Best configuration by token F1: **{best['model']} + {best['prompt']}** "
        f"(F1 {best['f1']:.3f}, mean latency {best['latency_mean_s']:.3f}s)",
        "",
        "## Overall (model x prompt)",
        _markdown_table(front),
        "",
        "## Token F1 by model and prompt",
        _markdown_table(comparison),
        "",
        "## Per dataset",
        _markdown_table(per_dataset),
    ]
    if plots:
        lines += ["", "![Quality vs latency](tradeoff.png)", "", "![Prompt comparison](prompt_comparison.png)"]
    outputs["report.md"] = run / "report.md"
    outputs["report.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
