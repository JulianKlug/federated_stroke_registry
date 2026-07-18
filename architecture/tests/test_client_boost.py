"""CRITICAL regression tests for _local_boost (§4.8).

This change touches working bagging code, so pin both branches:
- bagging must still return ONLY the newly boosted trees (the last N),
- cyclic must return the SAME booster grown by N trees.
"""
import numpy as np
import xgboost as xgb

from fed_stroke.client_app import _local_boost, _train_round, round_seed

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


# --- Per-round-seed regression (the 1.d federated-vs-pooled gap; see
# out/1d_solution.md). Rebuilding a fresh Booster + load_model each round resets
# XGBoost's subsampling RNG to params["seed"]; with a FIXED seed and
# colsample_bytree < 1 over few features, every round draws the SAME single
# column, so the ensemble collapses to one feature. round_seed advances the seed
# per round to restore the diversity a single xgb.train gets from its advancing
# RNG. These tests fail on the pre-fix code (fixed seed) and pass on the fix. ---

# 2 features, both genuinely predictive; colsample_bytree=0.5 selects 1 of 2 per
# tree, so a fixed seed picks the same feature every round and drops the other.
SAMPLING_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "seed": 0,
    "tree_method": "hist",
    "nthread": 1,  # deterministic
}
_NUM_ROUNDS = 30


def _two_feature_data(n, seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    logit = 3.0 * (X[:, 0] - 0.5) + 3.0 * (X[:, 1] - 0.5)  # BOTH features matter
    y = (logit + rng.randn(n) * 0.3 > 0).astype(int)
    return xgb.DMatrix(X, label=y), y


def _distinct_split_features(bst) -> set:
    import json

    trees = json.loads(bytes(bst.save_raw("json")))[
        "learner"
    ]["gradient_booster"]["model"]["trees"]
    feats = set()
    for t in trees:
        for si, lc in zip(t["split_indices"], t["left_children"]):
            if lc != -1:  # -1 left child == leaf, so `si` is not a real split
                feats.add(int(si))
    return feats


def _cyclic_via_train_round(dm):
    """Replay the client's per-round sequence through the REAL _train_round: round
    1 gets global_model=None, later rounds get the serialized global — exactly
    what client_app.train does. This exercises the actual fix, not a copy."""
    raw = None
    for r in range(1, _NUM_ROUNDS + 1):
        global_model = None if raw is None else bytearray(raw)
        bst = _train_round(SAMPLING_PARAMS, r, 1, dm, "cyclic", global_model)
        raw = bst.save_raw("json")
    out = xgb.Booster()
    out.load_model(bytearray(raw))
    return out


def _cyclic_fixed_seed(dm):
    """Pre-fix control arm: same reload loop but the seed NEVER advances. Proves
    the failure mode is real (the test could otherwise pass vacuously)."""
    fixed = {**SAMPLING_PARAMS, "seed": int(SAMPLING_PARAMS["seed"]) + 1}
    raw = None
    for r in range(1, _NUM_ROUNDS + 1):
        if raw is None:
            bst = xgb.train(fixed, dm, num_boost_round=1)
        else:
            b = xgb.Booster(params=fixed)
            b.load_model(bytearray(raw))
            bst = _local_boost(b, 1, dm, "cyclic")
        raw = bst.save_raw("json")
    out = xgb.Booster()
    out.load_model(bytearray(raw))
    return out


def test_round_seed_advances_and_is_pure():
    params = {"seed": 7}
    assert round_seed(params, 1)["seed"] == 8
    assert round_seed(params, 5)["seed"] == 12
    # Deterministic (pure function of the round) and non-mutating.
    assert round_seed(params, 5)["seed"] == 12
    assert params == {"seed": 7}
    # Defaults to base seed 0 when unset.
    assert round_seed({}, 3)["seed"] == 3


def test_fixed_seed_reload_collapses_to_one_feature():
    # The pre-fix bug: with a fixed seed the whole ensemble splits on ONE feature.
    dm, _ = _two_feature_data(600, 0)
    bug = _cyclic_fixed_seed(dm)
    assert len(_distinct_split_features(bug)) == 1


def test_train_round_uses_all_features():
    # The fix: advancing the seed per round lets every feature be sampled, so the
    # ensemble uses BOTH features like a single-process xgb.train would.
    dm, _ = _two_feature_data(600, 0)
    fixed_features = _distinct_split_features(_cyclic_fixed_seed(dm))
    fixed_via_fix = _distinct_split_features(_cyclic_via_train_round(dm))
    reference = _distinct_split_features(xgb.train(SAMPLING_PARAMS, dm, num_boost_round=_NUM_ROUNDS))
    assert fixed_via_fix == reference == {0, 1}
    assert fixed_via_fix > fixed_features  # strictly more coverage than the bug


def test_train_round_matches_single_xgb_train_auc():
    # End-to-end: the per-round-reload federation, WITH the fix, tracks a single
    # in-process xgb.train and dominates the fixed-seed (pre-fix) run by a wide
    # margin — the ~0.1 AUC gap 1.d caught, in miniature.
    from sklearn.metrics import roc_auc_score

    dtr, _ = _two_feature_data(600, 0)
    dev, y_dev = _two_feature_data(400, 1)

    auc = lambda b: roc_auc_score(y_dev, b.predict(dev))
    fixed_seed_auc = auc(_cyclic_fixed_seed(dtr))
    fixed_auc = auc(_cyclic_via_train_round(dtr))
    reference_auc = auc(xgb.train(SAMPLING_PARAMS, dtr, num_boost_round=_NUM_ROUNDS))

    assert fixed_auc >= reference_auc - 0.03      # tracks the single-process fit
    assert fixed_auc >= fixed_seed_auc + 0.10     # recovers the collapsed AUC
