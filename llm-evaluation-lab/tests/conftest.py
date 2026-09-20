"""Offline test doubles: a scripted LLM and a toy encoder, so nothing is downloaded."""
import json
from pathlib import Path

import numpy as np
import pytest

from llm_eval_lab.config import config_from_dict
from llm_eval_lab.metrics import SemanticScorer
from llm_eval_lab.models import Generation

QA = [
    {"id": "q1", "question": "What is the capital of France?", "answers": ["Paris"]},
    {"id": "q2", "question": "Who wrote Hamlet?", "answers": ["William Shakespeare", "Shakespeare"]},
    {"id": "q3", "question": "What is the chemical symbol for gold?", "answers": ["Au"]},
]


class ScriptedLLM:
    """Answers from a lookup table; ``mode`` controls how well it does."""

    def __init__(self, name: str, mode: str = "perfect", latency_s: float = 0.05):
        self.name, self.mode, self.latency_s = name, mode, latency_s
        self.calls: list[list[dict]] = []
        self.closed = False

    def generate(self, messages):
        self.calls.append(messages)
        question = messages[-1]["content"]
        gold = next((qa["answers"][0] for qa in QA if qa["question"] in question), "")
        if self.mode == "perfect":
            text = gold
        elif self.mode == "chatty":
            text = f"The answer is {gold}.\nLet me know if you need more."
        else:  # "wrong"
            text = "I have no idea"
        return Generation(text=text, latency_s=self.latency_s, new_tokens=max(1, len(text.split())))

    def close(self):
        self.closed = True


def char_encoder(texts):
    """Toy encoder: normalised letter histogram. Similar strings -> similar vectors."""
    out = np.zeros((len(texts), 26), dtype="float32")
    for i, text in enumerate(texts):
        for ch in text.lower():
            if "a" <= ch <= "z":
                out[i, ord(ch) - 97] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


@pytest.fixture
def scorer():
    return SemanticScorer(encode_fn=char_encoder)


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in QA), encoding="utf-8")
    return path


@pytest.fixture
def make_config(tmp_path, dataset_path):
    def _make(models=("good", "bad"), prompts=("zero_shot", "concise"), **overrides):
        raw = {
            "experiment_name": "unit",
            "output_dir": str(tmp_path / "results"),
            "models": [{"name": m, "hf_id": f"fake/{m}"} for m in models],
            "prompts": list(prompts),
            "datasets": [{"name": "qa", "path": str(dataset_path)}],
            "semantic_model": None,
            "warmup": False,
            **overrides,
        }
        return config_from_dict(raw)

    return _make
