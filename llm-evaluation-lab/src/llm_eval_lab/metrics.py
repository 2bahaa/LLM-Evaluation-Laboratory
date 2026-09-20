"""Evaluation metrics: exact match, contains-match, token F1, and embedding-based semantic similarity."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Sequence

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower().replace("°", " ")
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in golds))


def contains_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if any gold answer appears as a contiguous token sequence inside the prediction.

    Useful for chatty models that answer correctly inside a sentence ("It is Paris.").
    """
    pred_tokens = normalize_answer(prediction).split()
    for gold in golds:
        g = normalize_answer(gold).split()
        if not g:
            continue
        for i in range(len(pred_tokens) - len(g) + 1):
            if pred_tokens[i : i + len(g)] == g:
                return 1.0
    return 0.0


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction).split()
    return max((_f1(pred, normalize_answer(g).split()) for g in golds), default=0.0)


class SemanticScorer:
    """Cosine similarity between prediction and gold answers using a transformer encoder.

    Uses mean pooling over token embeddings (PyTorch) - the standard recipe for sentence-embedding
    models such as ``sentence-transformers/all-MiniLM-L6-v2``. Pass ``encode_fn`` to plug in any
    other encoder (used by the unit tests so they need no model download).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 encode_fn: Callable[[list[str]], np.ndarray] | None = None):
        if encode_fn is None and model_name is None:
            raise ValueError("Provide either model_name or encode_fn")
        self._encode_fn = encode_fn
        self._tokenizer = None
        self._model = None
        self._device = None
        if encode_fn is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            from .utils import select_device

            self._device = select_device(device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self._device)
            self._model.eval()
            self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._encode_fn is not None:
            return np.asarray(self._encode_fn(texts), dtype="float32")

        torch = self._torch
        batch = self._tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            hidden = self._model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

    def similarity(self, prediction: str, golds: Sequence[str]) -> float:
        if not prediction.strip() or not golds:
            return 0.0
        vectors = self.encode([prediction, *golds])
        vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return float(max(vectors[1:] @ vectors[0]))


def evaluate_prediction(prediction: str, golds: Sequence[str], scorer: SemanticScorer | None = None) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, golds),
        "contains_match": contains_match(prediction, golds),
        "f1": token_f1(prediction, golds),
        "semantic_similarity": scorer.similarity(prediction, golds) if scorer else float("nan"),
    }
