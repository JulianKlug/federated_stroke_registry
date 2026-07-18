"""Unit tests for server-side round derivation and strategy selection (§4.2)."""
import json

import numpy as np
import pytest
import xgboost as xgb
from flwr.serverapp.strategy import FedXgbBagging

from fed_stroke.server_app import build_strategy, derive_num_rounds, save_final_model
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


# --------------------------------------------------------------------------- #
# save_final_model stamps fed_run_config (§4.3 / A1) — verified at the producer
# side so a broken stamp can't slip through to the 1.d check as a silent fallback.
# --------------------------------------------------------------------------- #
def _tiny_booster():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 1, (40, 2))
    y = (x[:, 0] + rng.normal(0, 0.1, 40) > 0.5).astype(int)
    dm = xgb.DMatrix(x, label=y)
    return xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 0},
                     dm, num_boost_round=3)


def test_save_final_model_stamps_fed_run_config(tmp_path):
    params = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 4, "seed": 0}
    out_path = save_final_model(_tiny_booster(), tmp_path / "models", params, 40)

    reloaded = xgb.Booster()
    reloaded.load_model(str(out_path))
    embedded = json.loads(reloaded.attr("fed_run_config"))
    assert embedded["params"] == params
    assert embedded["total_trees"] == 40
