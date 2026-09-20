"""Question-answer dataset loading (JSONL or CSV)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    id: str
    question: str
    answers: list[str]  # one or more acceptable gold answers


def _to_answers(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [a.strip() for a in value.split("|") if a.strip()]
    return [str(a).strip() for a in value if str(a).strip()]


def _make_example(record: dict, fallback_id: str) -> Example:
    question = str(record.get("question", "")).strip()
    answers = _to_answers(record.get("answers", record.get("answer")))
    if not question:
        raise ValueError(f"{fallback_id}: missing 'question'")
    if not answers:
        raise ValueError(f"{fallback_id}: missing 'answer' / 'answers'")
    return Example(id=str(record.get("id") or fallback_id), question=question, answers=answers)


def load_dataset(path: str | Path, limit: int | None = None) -> list[Example]:
    """Load examples from ``.jsonl`` (one JSON object per line) or ``.csv`` (question, answer columns).

    ``answers`` may be a list, or a single string where alternatives are separated by ``|``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    examples: list[Example] = []
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        with p.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    examples.append(_make_example(json.loads(line), f"{p.stem}-{line_no}"))
    elif suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as handle:
            for row_no, row in enumerate(csv.DictReader(handle), start=1):
                examples.append(_make_example(row, f"{p.stem}-{row_no}"))
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}' (use .jsonl or .csv)")

    if not examples:
        raise ValueError(f"Dataset is empty: {p}")
    return examples[:limit] if limit else examples
