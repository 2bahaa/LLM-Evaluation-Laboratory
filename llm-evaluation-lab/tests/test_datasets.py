import json

import pytest

from llm_eval_lab.datasets import load_dataset


def test_load_jsonl_with_answers_list(dataset_path):
    examples = load_dataset(dataset_path)
    assert len(examples) == 3
    assert examples[1].answers == ["William Shakespeare", "Shakespeare"]


def test_limit(dataset_path):
    assert [e.id for e in load_dataset(dataset_path, limit=2)] == ["q1", "q2"]


def test_single_answer_field_and_generated_ids(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"question": "q?", "answer": "a | b"}) + "\n\n", encoding="utf-8")
    ex = load_dataset(path)[0]
    assert ex.id == "d-1" and ex.answers == ["a", "b"]


def test_load_csv(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("question,answer\nCapital of Italy?,Rome\n", encoding="utf-8")
    ex = load_dataset(path)[0]
    assert ex.question == "Capital of Italy?" and ex.answers == ["Rome"]


def test_invalid_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "no answer"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)
    txt = tmp_path / "x.txt"
    txt.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(txt)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(empty)
