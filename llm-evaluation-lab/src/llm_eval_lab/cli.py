"""Command-line interface:  python -m llm_eval_lab.cli {run,analyze,prompts}"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .analysis import analyze_run
from .config import load_config
from .prompts import available_templates


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description="LLM Evaluation Laboratory")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an experiment defined in a YAML config")
    run.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    run.add_argument("--max-examples", type=int, help="Override max_examples for a quick smoke test")
    run.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    analyze = sub.add_parser("analyze", help="(Re)generate tables, plots and report for a finished run")
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--no-plots", action="store_true")

    sub.add_parser("prompts", help="List available prompt templates")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "prompts":
        print("\n".join(available_templates()))
        return 0

    if args.command == "analyze":
        outputs = analyze_run(args.run_dir, plots=not args.no_plots)
        print("Wrote:", *(str(p) for p in outputs.values()), sep="\n  ")
        return 0

    from .runner import ExperimentRunner  # heavy imports only when actually running

    config = load_config(args.config)
    if args.max_examples:
        config.max_examples = args.max_examples
    result = ExperimentRunner(config).run()
    outputs = analyze_run(result.run_dir, plots=not args.no_plots)
    print(f"\nRun complete: {result.run_dir}")
    print((result.run_dir / "report.md").read_text(encoding="utf-8").split("## Per dataset")[0])
    print(f"All outputs: {', '.join(sorted(p.name for p in outputs.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
