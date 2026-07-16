"""Unit tests for server-side round derivation and strategy selection (§4.2)."""
import pytest
from flwr.serverapp.strategy import FedXgbBagging

from fed_stroke.server_app import build_strategy, derive_num_rounds
from fed_stroke.strategies import OrderedFedXgbCyclic


def test_derive_num_rounds_bagging():
    # 40 trees / (2 sites * 1 epoch) = 20 rounds
    assert derive_num_rounds("bagging", 40, 2, 1) == 20


def test_derive_num_rounds_cyclic():
    # 40 trees / (1 epoch) = 40 rounds
    assert derive_num_rounds("cyclic", 40, 2, 1) == 40


def test_derive_num_rounds_indivisible():
    # 39 trees is not divisible by 2 trees/round (bagging)
    with pytest.raises(ValueError):
        derive_num_rounds("bagging", 39, 2, 1)


def test_derive_num_rounds_unknown_method():
    with pytest.raises(ValueError):
        derive_num_rounds("stacking", 40, 2, 1)


def _run_config(overrides=None):
    cfg = {
        "train-method": "bagging",
        "cyclic-order": "forward",
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "num-sites": 2,
    }
    cfg.update(overrides or {})
    return cfg


def test_build_strategy_bagging():
    strategy = build_strategy(_run_config({"train-method": "bagging"}))
    assert isinstance(strategy, FedXgbBagging)


def test_build_strategy_cyclic_uses_configured_order():
    strategy = build_strategy(
        _run_config({"train-method": "cyclic", "cyclic-order": "reverse"})
    )
    assert isinstance(strategy, OrderedFedXgbCyclic)
    assert strategy.order == "reverse"


def test_build_strategy_unknown_method():
    with pytest.raises(ValueError):
        build_strategy(_run_config({"train-method": "stacking"}))
