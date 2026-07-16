"""CRITICAL regression tests for _local_boost (§4.8).

This change touches working bagging code, so pin both branches:
- bagging must still return ONLY the newly boosted trees (the last N),
- cyclic must return the SAME booster grown by N trees.
"""
import numpy as np
import xgboost as xgb

from fed_stroke.client_app import _local_boost

PARAMS = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3}


def _synthetic_dmatrix(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = (X[:, 0] + rng.rand(n) * 0.1 > 0.5).astype(int)
    return xgb.DMatrix(X, label=y)


def test_bagging_returns_only_new_trees():
    dm = _synthetic_dmatrix()
    # Start from a booster that already has some trees (like a mid-run global model).
    bst = xgb.train(PARAMS, dm, num_boost_round=3)
    num_local_round = 2

    out = _local_boost(bst, num_local_round, dm, "bagging")

    # Bagging slices out exactly the newly added trees, regardless of prior size.
    assert out.num_boosted_rounds() == num_local_round


def test_cyclic_returns_full_grown_booster():
    dm = _synthetic_dmatrix()
    bst = xgb.train(PARAMS, dm, num_boost_round=3)
    n_before = bst.num_boosted_rounds()
    num_local_round = 2

    out = _local_boost(bst, num_local_round, dm, "cyclic")

    # Cyclic returns the same object, with the full ensemble grown by N.
    assert out is bst
    assert out.num_boosted_rounds() == n_before + num_local_round
