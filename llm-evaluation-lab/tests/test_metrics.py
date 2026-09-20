import math

import pytest

from llm_eval_lab.metrics import (
    SemanticScorer,
    contains_match,
    evaluate_prediction,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_answer():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("100°C") == "100 c"


def test_exact_match_ignores_case_articles_and_punctuation():
    assert exact_match("The Nile.", ["Nile"]) == 1.0
    assert exact_match("Nile river", ["Nile"]) == 0.0


def test_exact_match_uses_best_of_multiple_golds():
    assert exact_match("Shakespeare", ["William Shakespeare", "Shakespeare"]) == 1.0


def test_contains_match_finds_answer_inside_a_sentence():
    assert contains_match("It is Paris, of course.", ["Paris"]) == 1.0
    assert contains_match("It is Lyon.", ["Paris"]) == 0.0


def test_contains_match_requires_whole_tokens():
    assert contains_match("Parisian cafes", ["Paris"]) == 0.0
    assert contains_match("The Milky Way galaxy", ["Milky Way"]) == 1.0
    assert contains_match("about 100°C at sea level", ["100"]) == 1.0


def test_token_f1_partial_credit():
    assert token_f1("Leonardo da Vinci", ["Leonardo da Vinci"]) == 1.0
    assert token_f1("da Vinci", ["Leonardo da Vinci"]) == pytest.approx(0.8)  # p=1, r=2/3
    assert token_f1("Michelangelo", ["Leonardo da Vinci"]) == 0.0


def test_token_f1_empty_inputs():
    assert token_f1("", ["Paris"]) == 0.0
    assert token_f1("Paris", []) == 0.0


def test_semantic_scorer_prefers_similar_text(scorer):
    close = scorer.similarity("Paris", ["Paris"])
    far = scorer.similarity("zzzz qqqq", ["Paris"])
    assert close == pytest.approx(1.0, abs=1e-5)
    assert far < close


def test_semantic_scorer_handles_empty_prediction(scorer):
    assert scorer.similarity("   ", ["Paris"]) == 0.0


def test_semantic_scorer_needs_a_model_or_encoder():
    with pytest.raises(ValueError):
        SemanticScorer()


def test_evaluate_prediction_without_scorer_returns_nan_similarity():
    result = evaluate_prediction("Paris", ["Paris"])
    assert result["exact_match"] == 1.0 and result["f1"] == 1.0
    assert math.isnan(result["semantic_similarity"])


def test_evaluate_prediction_with_scorer(scorer):
    assert evaluate_prediction("Paris", ["Paris"], scorer)["semantic_similarity"] == pytest.approx(1.0, abs=1e-5)
