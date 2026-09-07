"""Pure-logic tests for fed_stroke.baseline (roadmap 1.d, §4.6).

Sub-second, synthetic-fixture tests matching the plain-pytest idiom of
test_metrics.py. No live federation.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from flwr.common.config import flatten_dict, unflatten_dict

from fed_stroke.baseline import (
    assert_matched_across_models,
    assert_prediction_roundtrip,
    compare_auc,
    load_matched_config,
    read_model_config,
    score_booster_on_half,
    split_half,
    train_pooled_booster,
)
from fed_stroke.dp.synthetic import assemble_site_frame
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.task import replace_keys

PARAMS = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "seed": 0}


def _make_half(path, prefix, seed, n=60, positive_frac=0.35, single_class=False):
    """Write a synthetic half parquet with unique, half-disjoint patient IDs.

    patient_id = case_admission_id.split('_')[0], so f"{prefix}{i}_1" gives a
    unique patient per row and disjoint patient sets across prefixes.
    """
    rng = np.random.RandomState(seed)
    age = rng.uniform(40, 90, n)
    nih = rng.uniform(0, 30, n)
    if single_class:
        y = np.zeros(n, dtype=int)
    else:
        logits = 0.05 * (age - 65) + 0.1 * (nih - 15) + rng.normal(0, 1, n)
        y = (logits > np.quantile(logits, 1 - positive_frac)).astype(int)
    # full frozen schema: the signal in (age, NIHSS), the other 39 columns in-range background
    df = assemble_site_frame([f"{prefix}{i}_1" for i in range(n)], age, nih, y, seed=seed)
    df.to_parquet(path)
    return df


def _two_halves(tmp_path, **kw):
    a = tmp_path / "geneva_half_A.parquet"
    b = tmp_path / "geneva_half_B.parquet"
    _make_half(a, "A", seed=1, **kw)
    _make_half(b, "B", seed=2, **{k: v for k, v in kw.items() if k != "single_class"})
    return [a, b]


# --------------------------------------------------------------------------- #
# compare_auc
# --------------------------------------------------------------------------- #
def test_compare_auc_within_tol_passes():
    pooled = {"A": {"auc_roc": 0.70}, "B": {"auc_roc": 0.65}}
    fed = {"A": {"auc_roc": 0.69}, "B": {"auc_roc": 0.63}}
    res = compare_auc(pooled, fed, tol=0.03)
    assert res["overall"] is True
    assert res["sites"]["A"]["passed"] is True
    assert res["sites"]["A"]["delta"] == pytest.approx(0.01)


def test_compare_auc_over_tol_fails():
    pooled = {"A": {"auc_roc": 0.70}}
    fed = {"A": {"auc_roc": 0.60}}
    res = compare_auc(pooled, fed, tol=0.03)
    assert res["overall"] is False
    assert res["sites"]["A"]["passed"] is False


@pytest.mark.parametrize("side", ["pooled", "fed"])
def test_compare_auc_nan_is_hard_fail(side):
    pooled = {"A": {"auc_roc": float("nan") if side == "pooled" else 0.70}}
    fed = {"A": {"auc_roc": float("nan") if side == "fed" else 0.70}}
    res = compare_auc(pooled, fed, tol=0.5)
    assert res["overall"] is False
    assert res["sites"]["A"]["passed"] is False
    assert math.isnan(res["sites"]["A"]["delta"])


def test_compare_auc_overall_is_and_of_sites():
    pooled = {"A": {"auc_roc": 0.70}, "B": {"auc_roc": 0.70}}
    fed = {"A": {"auc_roc": 0.70}, "B": {"auc_roc": 0.50}}  # B over tol
    res = compare_auc(pooled, fed, tol=0.03)
    assert res["sites"]["A"]["passed"] is True
    assert res["sites"]["B"]["passed"] is False
    assert res["overall"] is False


# --------------------------------------------------------------------------- #
# train_pooled_booster
# --------------------------------------------------------------------------- #
def test_train_pooled_booster_tree_count(tmp_path):
    halves = _two_halves(tmp_path)
    bst = train_pooled_booster(halves, PARAMS, num_boost_round=7)
    assert bst.num_boosted_rounds() == 7


def test_train_pooled_booster_is_deterministic(tmp_path):
    halves = _two_halves(tmp_path)
    b1 = train_pooled_booster(halves, PARAMS, num_boost_round=6)
    b2 = train_pooled_booster(halves, PARAMS, num_boost_round=6)
    _, valid = split_half(halves[0])
    dm = xgb.DMatrix(valid[FEATURE_COLS], label=valid[TARGET_COL])
    assert np.array_equal(b1.predict(dm), b2.predict(dm))


def test_train_pooled_booster_requires_seed(tmp_path):
    halves = _two_halves(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        train_pooled_booster(halves, {"objective": "binary:logistic"}, 5)


def test_train_pooled_booster_does_not_mutate_params(tmp_path):
    halves = _two_halves(tmp_path)
    params = dict(PARAMS)
    train_pooled_booster(halves, params, num_boost_round=3)
    assert params == PARAMS  # nthread override applied to a copy only


def test_train_pooled_leakage_guard(tmp_path, monkeypatch):
    """A validation patient that also appears in train must raise, not train."""
    halves = _two_halves(tmp_path)
    shared = assemble_site_frame(["shared_1"], [70.0], [10.0], [1])
    shared.insert(0, "patient_id", ["shared"])
    train_df = shared.copy()
    valid_df = shared.copy()  # same patient in train AND valid -> leak
    import fed_stroke.baseline as baseline
    monkeypatch.setattr(baseline, "split_half", lambda p: (train_df, valid_df))
    with pytest.raises(ValueError, match="leakage"):
        train_pooled_booster(halves, PARAMS, num_boost_round=3)


def test_train_pooled_no_leakage_on_clean_halves(tmp_path):
    """Real disjoint halves train without tripping the leakage guard."""
    halves = _two_halves(tmp_path)
    train_pooled_booster(halves, PARAMS, num_boost_round=3)  # must not raise


# --------------------------------------------------------------------------- #
# load_matched_config  (C1: pinned to the FL-path derivation, not a hand dict)
# --------------------------------------------------------------------------- #
def _write_pyproject(path, total_trees=8):
    path.write_text(
        "[tool.flwr.app]\n"
        "[tool.flwr.app.config]\n"
        f"total-trees = {total_trees}\n"
        'params.objective = "binary:logistic"\n'
        "params.seed = 0\n"
        "params.eta = 0.1\n"
        "params.max-depth = 4\n"
        "params.min-child-weight = 5\n"
        'params.tree-method = "hist"\n'
    )


def test_load_matched_config_equals_fl_path_derivation(tmp_path):
    import tomllib
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, total_trees=8)
    cfg_table = tomllib.load(open(pyproject, "rb"))["tool"]["flwr"]["app"]["config"]

    # FL-path derivation: flat run_config -> unflatten -> replace_keys. Pinning to
    # this (not a hand-copied dict) guards the tomllib-vs-run_config equivalence.
    expected = replace_keys(unflatten_dict(flatten_dict(cfg_table)))["params"]

    params, total_trees = load_matched_config(pyproject)
    assert params == expected
    assert total_trees == 8
    assert params["max_depth"] == 4  # dashed key was normalised to underscore


# --------------------------------------------------------------------------- #
# read_model_config
# --------------------------------------------------------------------------- #
def _tiny_booster(tmp_path):
    df = _make_half(tmp_path / "tiny.parquet", "T", seed=5, n=40)
    dm = xgb.DMatrix(df[FEATURE_COLS], label=df[TARGET_COL])
    return xgb.train(PARAMS, dm, num_boost_round=3)


def test_read_model_config_from_embedded_attr(tmp_path):
    bst = _tiny_booster(tmp_path)
    bst.set_attr(fed_run_config=json.dumps({"params": PARAMS, "total_trees": 40}))
    params, total_trees, prov = read_model_config(bst, tmp_path / "nope.toml")
    assert prov == "model"
    assert params == PARAMS
    assert total_trees == 40


def test_read_model_config_falls_back_to_pyproject(tmp_path):
    bst = _tiny_booster(tmp_path)  # no attribute set
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, total_trees=8)
    params, total_trees, prov = read_model_config(bst, pyproject)
    assert prov == "pyproject-fallback"
    assert total_trees == 8
    assert params["objective"] == "binary:logistic"


# --------------------------------------------------------------------------- #
# assert_matched_across_models
# --------------------------------------------------------------------------- #
def test_assert_matched_identical_returns_single():
    cfg = ({"eta": 0.1, "seed": 0}, 40)
    params, trees = assert_matched_across_models([cfg, cfg, cfg])
    assert params == {"eta": 0.1, "seed": 0}
    assert trees == 40


def test_assert_matched_tree_mismatch_raises():
    with pytest.raises(ValueError, match="matched-budget"):
        assert_matched_across_models([({"eta": 0.1}, 40), ({"eta": 0.1}, 41)])


def test_assert_matched_params_mismatch_raises():
    with pytest.raises(ValueError, match="matched-params"):
        assert_matched_across_models([({"eta": 0.1}, 40), ({"eta": 0.2}, 40)])


# --------------------------------------------------------------------------- #
# score_booster_on_half  [T2]
# --------------------------------------------------------------------------- #
def test_score_booster_on_half_returns_full_dict(tmp_path):
    halves = _two_halves(tmp_path)
    bst = train_pooled_booster(halves, PARAMS, num_boost_round=5)
    m = score_booster_on_half(bst, halves[0], operating_point=0.5, n_boot=50, boot_seed=0)
    for key in ("auc_roc", "auc_pr", "brier", "tn", "fp", "fn", "tp", "n", "n_pos"):
        assert key in m
    # scored on the exact test_size=0.2, seed=42 split
    _, valid = split_half(halves[0])
    assert m["n"] == len(valid)


# --------------------------------------------------------------------------- #
# assert_prediction_roundtrip
# --------------------------------------------------------------------------- #
def test_roundtrip_intact_booster_passes(tmp_path):
    halves = _two_halves(tmp_path)
    bst = train_pooled_booster(halves, PARAMS, num_boost_round=5)
    _, valid = split_half(halves[0])
    dm = xgb.DMatrix(valid[FEATURE_COLS], label=valid[TARGET_COL])
    assert_prediction_roundtrip(bst, dm)  # must not raise


def test_roundtrip_non_fixed_point_raises(tmp_path):
    halves = _two_halves(tmp_path)
    real = train_pooled_booster(halves, PARAMS, num_boost_round=5)
    _, valid = split_half(halves[0])
    dm = xgb.DMatrix(valid[FEATURE_COLS], label=valid[TARGET_COL])

    class TamperedBst:
        """Predicts all-zeros, but serializes to a real (non-zero) booster, so the
        reloaded predictions differ from the first predict -> not a fixed point."""
        def __init__(self, real):
            self._real = real

        def predict(self, d):
            return np.zeros(d.num_row())

        def save_raw(self, fmt):
            return self._real.save_raw(fmt)

    with pytest.raises(ValueError, match="fixed point"):
        assert_prediction_roundtrip(TamperedBst(real), dm)
