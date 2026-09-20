from pathlib import Path

import pytest

from llm_eval_lab.config import config_from_dict, load_config

REPO = Path(__file__).resolve().parents[1]


def test_shipped_config_is_valid():
    cfg = load_config(REPO / "configs" / "experiment.yaml")
    assert len(cfg.models) >= 2 and len(cfg.prompts) >= 2 and len(cfg.datasets) >= 2
    for dataset in cfg.datasets:
        assert (REPO / dataset.path).exists()


def test_missing_required_key():
    with pytest.raises(ValueError, match="Missing required config key"):
        config_from_dict({"experiment_name": "x"})


@pytest.mark.parametrize("override", [{"models": []}, {"prompts": []}, {"datasets": []}, {"max_examples": 0},
                                      {"generation": {"max_new_tokens": 0}}])
def test_validation_errors(make_config, override):
    with pytest.raises(ValueError):
        make_config(**override)


def test_duplicate_model_names_rejected(make_config):
    with pytest.raises(ValueError, match="Duplicate model"):
        make_config(models=("a", "a"))
