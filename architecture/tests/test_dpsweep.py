"""Pure-logic + monkeypatched-driver tests for the 1.1.b DP sweep (spec §4.9).

Sub-second, synthetic-fixture tests matching the plain-pytest idiom of
test_hpo.py / test_baseline.py. No live federation, no `flwr run`: the driver
integration test monkeypatches `run_dp_sweep._run_flwr` with a fake that drops
pre-built tiny models into the arm dirs and appends fake ledger lines at the
`dp.ledger-path` parsed out of the run-config string it was handed (proving the
driver reads the same absolute path it threaded, spec §4.2).
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fed_stroke import dpsweep
from fed_stroke.baseline import load_saved_booster
from fed_stroke.dp import DP_MODEL_FORMAT, DPBooster, per_site_tree_budget
from fed_stroke.dp import ledger as dp_ledger
from fed_stroke.dp.boost import FEATURE_RANGES, fixed_bin_edges
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.server_app import derive_num_rounds

TEMPLATE_PATH = (Path(__file__).resolve().parents[2]
                 / "docs" / "templates" / "1_1_b_report_skeleton.md")

HALVES = ["geneva_half_A.parquet", "geneva_half_B.parquet"]

TOTAL_TREES = 4
MAX_DEPTH = 2
NUM_SITES = 2
LOCAL_EPOCHS = 1


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_half(path, prefix, seed, n=60, positive_frac=0.35, n_missing=0):
    """Synthetic half parquet (test_baseline idiom): patient_id =
    case_admission_id.split('_')[0], unique per row, disjoint across prefixes."""
    rng = np.random.RandomState(seed)
    age = rng.uniform(40, 90, n)
    nih = rng.uniform(0, 30, n)
    if n_missing:
        nih[:n_missing] = np.nan
    logits = 0.05 * (age - 65) + 0.1 * (np.nan_to_num(nih, nan=15.0) - 15) \
        + rng.normal(0, 1, n)
    y = (logits > np.quantile(logits, 1 - positive_frac)).astype(int)
    df = pd.DataFrame({
        "case_admission_id": [f"{prefix}{i}_1" for i in range(n)],
        FEATURE_COLS[0]: age,
        FEATURE_COLS[1]: nih,
        TARGET_COL: y,
    })
    df.to_parquet(path)
    return df


def _two_halves(tmp_path, **kw):
    a = tmp_path / HALVES[0]
    b = tmp_path / HALVES[1]
    _make_half(a, "A", seed=1, **kw)
    _make_half(b, "B", seed=2, **kw)
    return [a, b]


def _params(max_depth=MAX_DEPTH):
    """fed_run_config['params'] as save_final_model stamps it: underscored keys
    (replace_keys output), subsample/colsample at the forced 1.0."""
    return {"objective": "binary:logistic", "base_score": 0.5, "seed": 0,
            "eta": 0.1, "max_depth": max_depth, "min_child_weight": 5,
            "subsample": 1.0, "colsample_bytree": 1.0}


def _train_xgb(total_trees=TOTAL_TREES, max_depth=MAX_DEPTH):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({FEATURE_COLS[0]: rng.uniform(40, 90, 200),
                      FEATURE_COLS[1]: rng.uniform(0, 30, 200)})
    y = (rng.rand(200) < 0.35).astype(int)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train({"objective": "binary:logistic", "max_depth": max_depth,
                      "seed": 0}, dtrain, num_boost_round=total_trees)


def _write_xgb_model(path, total_trees=TOTAL_TREES, max_depth=MAX_DEPTH):
    bst = _train_xgb(total_trees, max_depth)
    bst.set_attr(fed_run_config=json.dumps(
        {"params": _params(max_depth), "total_trees": total_trees}))
    path.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(path))
    return path


def _dp_meta(strategy, mechanism, eps, total_trees=TOTAL_TREES,
             max_depth=MAX_DEPTH):
    """Plausible dp/boost._mechanism_meta shape. Identity carries inf σ/ε in
    memory (scrubbed to null by to_json_bytes on the way to disk)."""
    num_rounds = derive_num_rounds(strategy, total_trees, NUM_SITES, LOCAL_EPOCHS)
    per_site = per_site_tree_budget(strategy, num_rounds, NUM_SITES, LOCAL_EPOCHS)
    if mechanism == "identity":
        acct = {"per_site_trees": None, "num_releases": 0,
                "noise_multiplier": float("inf"),
                "reported_epsilon": float("inf")}
    else:
        acct = {"per_site_trees": per_site,
                "num_releases": 2 * max_depth * per_site,
                "noise_multiplier": 30.0 / eps, "reported_epsilon": eps}
    return {
        "train_method": strategy,
        **acct,
        "dp": {"enabled": True, "mechanism": mechanism, "target_epsilon": eps,
               "delta": 1e-5, "clip_bound": 1.0, "max_bins": 32,
               "bin_strategy": "fixed_range"},
        "fed_run_config": {"params": _params(max_depth),
                           "total_trees": total_trees},
    }


def _make_dp_booster(strategy, mechanism, eps=None, total_trees=TOTAL_TREES,
                     max_depth=MAX_DEPTH):
    edges = fixed_bin_edges(FEATURE_RANGES, 32)
    trees = [{"feature": 0, "bin": 10 + i,
              "left": {"leaf": -0.05}, "right": {"leaf": 0.08}}
             for i in range(total_trees)]
    return DPBooster(trees, 0.0, edges, 32, feature_ranges=FEATURE_RANGES,
                     meta=_dp_meta(strategy, mechanism, eps, total_trees,
                                   max_depth))


def _write_dp_model(path, strategy, mechanism, eps=None,
                    total_trees=TOTAL_TREES, max_depth=MAX_DEPTH):
    booster = _make_dp_booster(strategy, mechanism, eps, total_trees, max_depth)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(booster.to_json_bytes())
    return path


# --------------------------------------------------------------------------- #
# §4.9.1  arm plan
# --------------------------------------------------------------------------- #
def test_arm_plan_order_labels_overrides():
    plan = dpsweep.arm_plan([1, 3, 5, 10], delta=1e-5)
    assert [a["label"] for a in plan] == \
        ["armA", "armB", "armC_eps10", "armC_eps5", "armC_eps3", "armC_eps1"]
    assert [a["arm"] for a in plan] == ["A", "B", "C", "C", "C", "C"]
    assert plan[0]["dp_overrides"] == {"dp.enabled": False}
    assert plan[1]["dp_overrides"] == {"dp.enabled": True,
                                       "dp.mechanism": "identity"}
    assert "dp.target-epsilon" not in plan[1]["dp_overrides"]  # B carries no ε
    assert plan[2]["dp_overrides"] == {
        "dp.enabled": True, "dp.mechanism": "gaussian",
        "dp.target-epsilon": 10.0, "dp.delta": 1e-5}
    assert plan[2]["epsilon"] == 10.0 and plan[-1]["epsilon"] == 1.0


def test_arm_plan_descending_regardless_of_input_order():
    plan = dpsweep.arm_plan([5, 1, 10, 3], delta=1e-5)
    assert [a["epsilon"] for a in plan if a["arm"] == "C"] == [10.0, 5.0, 3.0, 1.0]


def test_arm_plan_fractional_epsilon_label():
    plan = dpsweep.arm_plan([0.5], delta=1e-5)
    assert plan[-1]["label"] == "armC_eps0.5"


# --------------------------------------------------------------------------- #
# §4.9.2  run-config strings
# --------------------------------------------------------------------------- #
def _shared():
    return {"total-trees": TOTAL_TREES, "params.max-depth": MAX_DEPTH,
            "params.eta": 0.1, "params.min-child-weight": 5,
            "params.subsample": 1.0, "params.colsample-bytree": 1.0}


def test_run_config_strings_per_arm(tmp_path):
    plan = dpsweep.arm_plan([3], delta=1e-5)
    ledger = tmp_path / "led.jsonl"
    cfgs = {a["label"]: dpsweep.build_arm_run_config(
        a, _shared(), "bagging", 42, tmp_path, "example-halves", ledger)
        for a in plan}
    for label, cfg in cfgs.items():
        assert "params.subsample=1.0" in cfg
        assert "params.colsample-bytree=1.0" in cfg
        assert "holdout-frac=0.0" in cfg
        assert "holdout-eval=false" in cfg
        assert "n-boot=0" in cfg
        assert "save-model=true" in cfg
        assert "data-provenance='example-halves'" in cfg
        assert f"dp.ledger-path='{ledger}'" in cfg
        assert f"model-dir='{tmp_path / label}'" in cfg
        assert Path(str(ledger)).is_absolute()
    assert "dp.enabled=false" in cfgs["armA"]
    assert "dp.mechanism" not in cfgs["armA"]
    assert "dp.enabled=true" in cfgs["armB"]
    assert "dp.mechanism='identity'" in cfgs["armB"]
    assert "dp.target-epsilon" not in cfgs["armB"]
    assert "dp.delta" not in cfgs["armB"]
    assert "dp.mechanism='gaussian'" in cfgs["armC_eps3"]
    assert "dp.target-epsilon=3.0" in cfgs["armC_eps3"]
    assert "dp.delta=1e-05" in cfgs["armC_eps3"]


# --------------------------------------------------------------------------- #
# §4.9.3  shared-config resolution
# --------------------------------------------------------------------------- #
_APP_CFG = {"total-trees": 40, "operating-point": 0.5,
            "params": {"max-depth": 4, "eta": 0.1, "min-child-weight": 5,
                       "subsample": 0.8, "colsample-bytree": 0.8}}


def test_resolve_shared_config_base_forces_sampling():
    shared, notes = dpsweep.resolve_shared_config(_APP_CFG, None)
    assert shared["total-trees"] == 40
    assert shared["params.max-depth"] == 4
    assert shared["params.subsample"] == 1.0
    assert shared["params.colsample-bytree"] == 1.0
    assert notes == []


def test_resolve_shared_config_tuned_adoption_and_discard():
    tuned_text = """
[tool.flwr.app.config]
total-trees = 20
params.max-depth = 3
params.eta = 0.3
params.min-child-weight = 1
params.subsample = 0.8
params.colsample-bytree = 0.9
"""
    tuned = dpsweep.parse_tuned_table(tuned_text)
    shared, notes = dpsweep.resolve_shared_config(_APP_CFG, tuned)
    assert shared["total-trees"] == 20
    assert shared["params.max-depth"] == 3
    assert shared["params.eta"] == 0.3
    assert shared["params.min-child-weight"] == 1
    # tuned sampling knobs read but DISCARDED, with a divergence note each
    assert shared["params.subsample"] == 1.0
    assert shared["params.colsample-bytree"] == 1.0
    assert len(notes) == 2 and all("forced to 1.0" in n for n in notes)


def test_parse_tuned_table_missing_table_raises():
    with pytest.raises(ValueError, match="tool.flwr.app.config"):
        dpsweep.parse_tuned_table("[tool.other]\nx = 1\n")


def test_divisibility_failure_raises_before_any_run():
    with pytest.raises(ValueError):
        dpsweep.validate_divisibility("bagging", 5, NUM_SITES, LOCAL_EPOCHS)


# --------------------------------------------------------------------------- #
# §4.9.4  deltas
# --------------------------------------------------------------------------- #
def _arm(arm, label, eps, metrics_by_half):
    return {"arm": arm, "label": label, "epsilon": eps,
            "metrics": metrics_by_half}


def test_compute_deltas_values_and_signs():
    m = lambda auc, pr, brier: {"auc_roc": auc, "auc_pr": pr, "brier": brier}
    arms = [
        _arm("A", "armA", None, {"hA": m(0.80, 0.60, 0.15),
                                 "hB": m(0.78, 0.58, 0.16)}),
        _arm("B", "armB", None, {"hA": m(0.76, 0.55, 0.17),
                                 "hB": m(0.75, 0.54, 0.18)}),
        _arm("C", "armC_eps3", 3.0, {"hA": m(0.70, 0.50, 0.20),
                                     "hB": m(0.71, 0.49, 0.21)}),
    ]
    d = dpsweep.compute_deltas(arms, ["hA", "hB"])
    # Δ_priv = C − B (negative AUC = utility lost; positive Brier = worse)
    assert d["privacy_cost"]["armC_eps3"]["hA"]["auc_roc"] == pytest.approx(-0.06)
    assert d["privacy_cost"]["armC_eps3"]["hA"]["brier"] == pytest.approx(0.03)
    assert d["privacy_cost"]["armC_eps3"]["mean"]["auc_roc"] == \
        pytest.approx((-0.06 - 0.04) / 2)
    # Δ_learn = B − A
    assert d["learner_cost"]["hA"]["auc_roc"] == pytest.approx(-0.04)
    assert d["learner_cost"]["mean"]["brier"] == pytest.approx(0.02)
    assert d["sign_convention"] == dpsweep.SIGN_CONVENTION


def test_compute_deltas_none_propagation():
    m = lambda auc: {"auc_roc": auc, "auc_pr": 0.5, "brier": 0.2}
    arms = [
        _arm("A", "armA", None, {"hA": m(0.8), "hB": m(0.8)}),
        _arm("B", "armB", None, {"hA": m(float("nan")), "hB": m(0.75)}),
        _arm("C", "armC_eps1", 1.0, {"hA": m(0.7), "hB": None}),  # degenerate half
    ]
    d = dpsweep.compute_deltas(arms, ["hA", "hB"])
    assert d["privacy_cost"]["armC_eps1"]["hA"]["auc_roc"] is None  # NaN in B
    assert d["privacy_cost"]["armC_eps1"]["hB"]["auc_roc"] is None  # C missing
    assert d["privacy_cost"]["armC_eps1"]["mean"]["auc_roc"] is None
    assert d["learner_cost"]["hA"]["auc_roc"] is None
    assert d["learner_cost"]["hB"]["auc_roc"] == pytest.approx(-0.05)


# --------------------------------------------------------------------------- #
# §4.9.5  extract_dp_meta (+ the inf-crash guard, both defensive layers)
# --------------------------------------------------------------------------- #
def test_extract_dp_meta_gaussian_roundtrip(tmp_path):
    path = _write_dp_model(tmp_path / "final_model.json", "bagging", "gaussian",
                           eps=3.0)
    loaded = load_saved_booster(path)
    meta = dpsweep.extract_dp_meta(loaded.booster)
    assert meta["mechanism"] == "gaussian"       # nested at meta["dp"]["mechanism"]
    assert meta["sigma"] == pytest.approx(10.0)
    assert meta["epsilon"] == pytest.approx(3.0)
    assert meta["num_releases"] == 2 * MAX_DEPTH * 2
    assert meta["per_site_trees"] == 2


def test_extract_dp_meta_identity_from_saved_model_is_none(tmp_path):
    # to_json_bytes scrubs identity's inf σ/ε to null on serialize — the loaded
    # meta arrives as None and must pass through without TypeError.
    path = _write_dp_model(tmp_path / "final_model.json", "bagging", "identity")
    loaded = load_saved_booster(path)
    meta = dpsweep.extract_dp_meta(loaded.booster)
    assert meta["mechanism"] == "identity"
    assert meta["sigma"] is None
    assert meta["epsilon"] is None
    assert meta["num_releases"] == 0
    assert meta["per_site_trees"] is None


def test_extract_dp_meta_inf_fixture_mapped_to_none():
    # A hand-built (non-scrubbed) meta injecting inf — the residual-inf map.
    bst = _make_dp_booster("bagging", "identity")   # in-memory: inf σ/ε
    meta = dpsweep.extract_dp_meta(bst)
    assert meta["sigma"] is None and meta["epsilon"] is None


def test_arm_b_results_block_survives_strict_json():
    # The inf-crash guard: an arm-B block through json.dumps(allow_nan=False).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from flwr_proc import _json_safe
    block = {"label": "armB",
             "dp_meta": {"mechanism": "identity", "sigma": float("inf"),
                         "epsilon": float("-inf"), "num_releases": 0,
                         "per_site_trees": None},
             "metrics": {"hA": {"auc_roc": float("nan")}}}
    text = json.dumps(_json_safe(block), allow_nan=False)
    parsed = json.loads(text)
    assert parsed["dp_meta"]["sigma"] is None
    assert parsed["dp_meta"]["epsilon"] is None
    assert parsed["metrics"]["hA"]["auc_roc"] is None


# --------------------------------------------------------------------------- #
# §4.4  per-arm assertions
# --------------------------------------------------------------------------- #
def _plan_arm(label):
    plan = dpsweep.arm_plan([3.0], delta=1e-5)
    return next(a for a in plan if a["label"] == label)


def test_check_arm_model_passes_all_arms(tmp_path):
    per_site = per_site_tree_budget(
        "bagging", derive_num_rounds("bagging", TOTAL_TREES, NUM_SITES,
                                     LOCAL_EPOCHS), NUM_SITES, LOCAL_EPOCHS)
    xgb_path = _write_xgb_model(tmp_path / "a" / "final_model.json")
    assert dpsweep.check_arm_model(_plan_arm("armA"),
                                   load_saved_booster(xgb_path),
                                   TOTAL_TREES, per_site) == []
    b_path = _write_dp_model(tmp_path / "b" / "final_model.json", "bagging",
                             "identity")
    assert dpsweep.check_arm_model(_plan_arm("armB"),
                                   load_saved_booster(b_path),
                                   TOTAL_TREES, per_site) == []
    c_path = _write_dp_model(tmp_path / "c" / "final_model.json", "bagging",
                             "gaussian", eps=3.0)
    assert dpsweep.check_arm_model(_plan_arm("armC_eps3"),
                                   load_saved_booster(c_path),
                                   TOTAL_TREES, per_site) == []


def test_check_arm_model_catches_violations(tmp_path):
    per_site = 2
    # wrong format for arm A (a DP model where stock XGB is expected)
    dp_path = _write_dp_model(tmp_path / "x" / "final_model.json", "bagging",
                              "gaussian", eps=3.0)
    problems = dpsweep.check_arm_model(_plan_arm("armA"),
                                       load_saved_booster(dp_path),
                                       TOTAL_TREES, per_site)
    assert any("format" in p for p in problems)
    # ε off target by more than 1e-3 relative
    off = _write_dp_model(tmp_path / "y" / "final_model.json", "bagging",
                          "gaussian", eps=3.2)
    problems = dpsweep.check_arm_model(_plan_arm("armC_eps3"),
                                       load_saved_booster(off),
                                       TOTAL_TREES, per_site)
    assert any("reported_epsilon" in p for p in problems)
    # release-count invariant: wrong per_site (as if computed total//num_sites
    # on a cyclic run) must fail
    good = load_saved_booster(dp_path)
    problems = dpsweep.check_arm_model(_plan_arm("armC_eps3"), good,
                                       TOTAL_TREES, per_site + 1)
    assert any("per_site" in p for p in problems)


# --------------------------------------------------------------------------- #
# §4.9.6  ledger delta + validation (strategy-aware)
# --------------------------------------------------------------------------- #
def _entry(site, eps, sigma, config_hash="h1", mechanism="gaussian"):
    return {"date": "2026-07-23T00:00:00+00:00", "site": site,
            "config_hash": config_hash, "mechanism": mechanism,
            "num_releases": 8, "noise_multiplier": sigma, "epsilon": eps,
            "delta": 1e-5}


def _c_run(label="armC_eps3", eps=3.0, sigma=10.0):
    return {"label": label, "epsilon": eps, "sigma": sigma}


def test_ledger_delta_suffix_and_append_only():
    before = [_entry(HALVES[0], 3.0, 10.0)]
    after = before + [_entry(HALVES[1], 3.0, 10.0)]
    assert dpsweep.ledger_delta(before, after) == after[1:]
    with pytest.raises(ValueError, match="append-only"):
        dpsweep.ledger_delta(before, [_entry(HALVES[1], 1.0, 30.0)])


def test_ledger_bagging_two_entries_pass():
    delta = [_entry(HALVES[0], 3.0, 10.0), _entry(HALVES[1], 3.0, 10.0)]
    assert dpsweep.validate_sweep_ledger(delta, [_c_run()],
                                         "real-frozen-schema", "bagging") == []


def test_ledger_bagging_config_hash_mismatch_fails():
    delta = [_entry(HALVES[0], 3.0, 10.0, config_hash="h1"),
             _entry(HALVES[1], 3.0, 10.0, config_hash="h2")]
    issues = dpsweep.validate_sweep_ledger(delta, [_c_run()],
                                           "real-frozen-schema", "bagging")
    assert any("config_hash" in i for i in issues)


def test_ledger_extra_entry_on_a_or_b_run_fails():
    delta = [_entry(HALVES[0], 3.0, 10.0)]
    issues = dpsweep.validate_sweep_ledger(delta, [], "real-frozen-schema",
                                           "bagging")
    assert issues  # identity/non-DP must never ledger


def test_ledger_eps_sigma_mismatch_vs_meta_fails():
    delta = [_entry(HALVES[0], 3.5, 10.0), _entry(HALVES[1], 3.0, 12.0)]
    issues = dpsweep.validate_sweep_ledger(delta, [_c_run()],
                                           "real-frozen-schema", "bagging")
    assert any("epsilon" in i for i in issues)
    assert any("noise_multiplier" in i for i in issues)


def test_ledger_cyclic_one_entry_passes_two_fail():
    one = [_entry(HALVES[0], 3.0, 10.0)]
    assert dpsweep.validate_sweep_ledger(one, [_c_run()],
                                         "real-frozen-schema", "cyclic") == []
    two = one + [_entry(HALVES[1], 3.0, 10.0)]
    issues = dpsweep.validate_sweep_ledger(two, [_c_run()],
                                           "real-frozen-schema", "cyclic")
    assert any("expected 1" in i for i in issues)


def test_ledger_any_entry_on_example_halves_fails():
    delta = [_entry(HALVES[0], 3.0, 10.0)]
    for strategy in ("bagging", "cyclic"):
        issues = dpsweep.validate_sweep_ledger(delta, [_c_run()],
                                               "example-halves", strategy)
        assert any("provenance" in i for i in issues)


def test_unledgered_sites_cyclic_gap():
    totals = {HALVES[0]: 4.2}    # site B trained on round ≥ 2, never appended
    assert dpsweep.unledgered_sites(HALVES, totals) == [HALVES[1]]
    assert dpsweep.unledgered_sites(HALVES, {h: 1.0 for h in HALVES}) == []


# --------------------------------------------------------------------------- #
# §4.9.7  render_report
# --------------------------------------------------------------------------- #
def _metrics_fixture(auc=0.7):
    return {"auc_roc": auc, "auc_roc_lo": auc - 0.05, "auc_roc_hi": auc + 0.05,
            "auc_pr": 0.5, "auc_pr_lo": 0.45, "auc_pr_hi": 0.55,
            "brier": 0.2, "n": 12, "n_pos": 4}


def _results_fixture(provenance="real-frozen-schema", strategy="bagging",
                     unledgered=False, epsilons=(3.0, 1.0), degenerate=()):
    real = provenance == "real-frozen-schema"
    arms = [
        {"arm": "A", "label": "armA", "epsilon": None, "run_config": "cfgA",
         "model_path": "mA", "degenerate": False, "fmt": "xgboost-json",
         "n_trees": TOTAL_TREES, "dp_meta": None, "config_hash": None,
         "metrics": {h: _metrics_fixture(0.75) for h in HALVES}},
        {"arm": "B", "label": "armB", "epsilon": None, "run_config": "cfgB",
         "model_path": "mB", "degenerate": False, "fmt": DP_MODEL_FORMAT,
         "n_trees": TOTAL_TREES, "config_hash": None,
         "dp_meta": {"mechanism": "identity", "sigma": None, "epsilon": None,
                     "num_releases": 0, "per_site_trees": None},
         "metrics": {h: _metrics_fixture(0.72) for h in HALVES}},
    ]
    per_run_eps, totals = {}, {}
    for eps in sorted(epsilons, reverse=True):
        label = f"armC_eps{eps:g}"
        deg = label in degenerate
        arms.append({
            "arm": "C", "label": label, "epsilon": eps, "run_config": f"cfg{label}",
            "model_path": f"m{label}", "degenerate": deg,
            "fmt": None if deg else DP_MODEL_FORMAT,
            "n_trees": None if deg else TOTAL_TREES,
            "config_hash": f"hash{eps:g}" if real else None,
            "dp_meta": None if deg else {
                "mechanism": "gaussian", "sigma": 30.0 / eps, "epsilon": eps,
                "num_releases": 8, "per_site_trees": 2},
            "metrics": {h: (None if deg else _metrics_fixture(0.70 - 0.02 / eps))
                        for h in HALVES}})
    ledgered_sites = HALVES[:1] if unledgered else HALVES
    if real:
        for site in ledgered_sites:
            totals[site] = sum(epsilons) + 0.5
            per_run_eps[site] = {f"armC_eps{e:g}": e
                                 for e in sorted(epsilons, reverse=True)}
    deltas = dpsweep.compute_deltas(arms, HALVES)
    meta = {"date": "2026-07-23T00:00:00+00:00", "operator": "tester",
            "gate_ack": "R8 sign-off recorded in docs/logbook.md" if real else None,
            "data_provenance": provenance, "federation": "local-deployment",
            "strategy": strategy, "split_seed": 42, "delta": 1e-5,
            "epsilons": sorted(epsilons, reverse=True), "n_boot": 1000,
            "shared_config": _shared(), "tuned_config_path": None,
            "config_notes": [], "num_rounds": 2, "per_site_tree_budget": 2}
    cohort = {h: {"raw_rows": 60, "rows_after_dedup": 60,
                  "patients_after_dedup": 60,
                  "missingness": {FEATURE_COLS[0]: 0.0, FEATURE_COLS[1]: 0.05}}
              for h in HALVES}
    ledger = {"path": "/abs/dp_ledger.jsonl", "entries_added": [],
              "ledger_total": totals,
              "unledgered_sites": [HALVES[1]] if (real and unledgered) else [],
              "site_labels": {HALVES[0]: "node_A", HALVES[1]: "node_B"},
              "expected_sites": HALVES, "per_run_epsilon": per_run_eps}
    return dpsweep.build_results(meta, cohort, arms, deltas, ledger)


def test_render_report_template_headings_tripwire():
    template = TEMPLATE_PATH.read_text()
    headings = [ln for ln in template.splitlines() if ln.startswith("## ")]
    assert headings, "skeleton template lost its section headings?"
    report = dpsweep.render_report(_results_fixture())
    for heading in headings:
        assert heading in report, f"renderer drifted from template: {heading!r}"
    assert report.startswith("# 1.1.b DP sweep report")


def test_render_report_r6_both_epsilons_and_r9_arms():
    report = dpsweep.render_report(_results_fixture(epsilons=(3.0, 1.0)))
    # R9: all three arms, B→C flagged as headline, A→B as learner cost
    for token in ("| A |", "| B |", "C @ ε=3", "C @ ε=1"):
        assert token in report
    assert "B→C = cost of privacy (HEADLINE)" in report
    assert "A→B = learner cost" in report
    # R6: per-run ε AND composed ledger-total for every C arm/site
    for site in HALVES:
        assert f"armC_eps3: 3.0000" in report and f"armC_eps1: 1.0000" in report
    assert "4.5000" in report                       # composed total (3+1+0.5)
    assert "`ledger_total()` output pasted verbatim" in report
    # boundary boxes: 2 auto-checked, 1 operator-only; go/no-go blank
    assert "- [x] DP train replies carried `dp-site-weight`" in report
    assert "- [ ] Nothing inside-boundary" in report
    assert "______" in report.split("## Go/no-go")[1]
    assert dpsweep.REHEARSAL_BANNER not in report


def test_render_report_rehearsal_banner_iff_example_halves():
    rehearsal = dpsweep.render_report(_results_fixture(provenance="example-halves"))
    assert dpsweep.REHEARSAL_BANNER in rehearsal
    assert "No ledger entries were written" in rehearsal
    real = dpsweep.render_report(_results_fixture())
    assert dpsweep.REHEARSAL_BANNER not in real


def test_render_report_unledgered_site_never_blank_or_zero():
    report = dpsweep.render_report(
        _results_fixture(strategy="cyclic", unledgered=True))
    row = next(ln for ln in report.splitlines()
               if ln.startswith(f"| {HALVES[1]}"))
    assert dpsweep.UNLEDGERED_MARK in row
    assert "| 0 |" not in row and "|  |" not in row


def test_render_report_degenerate_arm_marked():
    report = dpsweep.render_report(
        _results_fixture(degenerate=("armC_eps1",)))
    assert "DEGENERATE" in report


def test_render_report_config_hashes_c_only():
    report = dpsweep.render_report(_results_fixture())
    assert "armC_eps3: `hash3`" in report
    assert "A/B: n/a — no DP, not ledgered" in report


# --------------------------------------------------------------------------- #
# §4.9.8  cohort stats
# --------------------------------------------------------------------------- #
def test_cohort_stats_dedup_and_missingness(tmp_path):
    halves = _two_halves(tmp_path, n=60, n_missing=6)
    stats = dpsweep.cohort_stats(halves, split_seed=42)
    for h in HALVES:
        st = stats[h]
        assert st["raw_rows"] == 60
        assert st["rows_after_dedup"] == 60          # unique patients already
        assert st["patients_after_dedup"] == 60
        assert st["missingness"][FEATURE_COLS[0]] == 0.0
        assert st["missingness"][FEATURE_COLS[1]] == pytest.approx(0.1)


def test_cohort_stats_counts_dedup(tmp_path):
    # two admissions for the same patient collapse to one row (R3 dedup)
    path = tmp_path / HALVES[0]
    df = _make_half(path, "A", seed=1, n=40)
    dup = df.iloc[[0]].assign(case_admission_id="A0_2")
    pd.concat([df, dup], ignore_index=True).to_parquet(path)
    stats = dpsweep.cohort_stats([path], split_seed=42)
    assert stats[HALVES[0]]["raw_rows"] == 41
    assert stats[HALVES[0]]["rows_after_dedup"] == 40
    assert stats[HALVES[0]]["patients_after_dedup"] == 40


# --------------------------------------------------------------------------- #
# §4.9.9  load_saved_booster contract (fed_stroke.baseline)
# --------------------------------------------------------------------------- #
def test_load_saved_booster_xgb_contract(tmp_path):
    path = tmp_path / "final_model.json"
    bst = _train_xgb(total_trees=2)
    bst.save_model(str(path))
    loaded = load_saved_booster(path)
    assert loaded.fmt == "xgboost-json"
    assert loaded.n_trees == 2
    X = pd.DataFrame({FEATURE_COLS[0]: [70.0], FEATURE_COLS[1]: [10.0]})
    probs = loaded.booster.predict(xgb.DMatrix(X))
    assert 0.0 <= float(probs[0]) <= 1.0


def test_load_saved_booster_dp_contract(tmp_path):
    path = _write_dp_model(tmp_path / "final_model.json", "bagging", "gaussian",
                           eps=3.0)
    loaded = load_saved_booster(path)
    assert loaded.fmt == DP_MODEL_FORMAT
    assert loaded.n_trees == TOTAL_TREES == len(loaded.booster.trees)
    probs = loaded.booster.predict(np.array([[70.0, 10.0]]))
    assert 0.0 <= float(probs[0]) <= 1.0


# --------------------------------------------------------------------------- #
# driver integration (monkeypatched runner, test_hpo.py pattern)
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_dp_sweep", SCRIPTS_DIR / "run_dp_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_run_config(run_config):
    out = {}
    for token in run_config.split():
        key, val = token.split("=", 1)
        out[key] = val.strip("'")
    return out


def _fake_cfg(half_paths):
    return {"tool": {
        "flwr": {"app": {"config": {
            "operating-point": 0.5, "num-sites": NUM_SITES,
            "local-epochs": LOCAL_EPOCHS, "total-trees": TOTAL_TREES,
            "params": {"max-depth": MAX_DEPTH, "eta": 0.1,
                       "min-child-weight": 5, "subsample": 0.8,
                       "colsample-bytree": 0.8},
        }}},
        "fed_stroke": {
            "superlink": {"address": "127.0.0.1:9093"},
            "nodes": {
                "node_A": {"data-path": str(half_paths[0])},
                "node_B": {"data-path": str(half_paths[1])},
            },
        },
    }}


def _make_fake_runner(strategy, sites, fail_labels=()):
    """Fake `_run_flwr`: drops a pre-built model into the model-dir parsed from
    the run-config string and, for real-provenance gaussian runs, appends ledger
    entries AT THE dp.ledger-path IT WAS HANDED (2/C-run bagging, 1 cyclic)."""
    calls = []

    def fake(federation, run_config, timeout):
        cfg = _parse_run_config(run_config)
        model_dir = Path(cfg["model-dir"])
        label = model_dir.name
        calls.append(label)
        if label in fail_labels:
            return False, False
        total_trees = int(cfg["total-trees"])
        max_depth = int(cfg["params.max-depth"])
        model_path = model_dir / "final_model.json"
        if cfg["dp.enabled"] == "false":
            _write_xgb_model(model_path, total_trees, max_depth)
        else:
            mechanism = cfg["dp.mechanism"]
            eps = (float(cfg["dp.target-epsilon"])
                   if "dp.target-epsilon" in cfg else None)
            _write_dp_model(model_path, strategy, mechanism, eps, total_trees,
                            max_depth)
            if (mechanism == "gaussian"
                    and cfg["data-provenance"] == "real-frozen-schema"):
                meta = _dp_meta(strategy, "gaussian", eps, total_trees,
                                max_depth)
                run_hash = dp_ledger.config_hash({"label": label, "eps": eps})
                n_appends = 2 if strategy == "bagging" else 1
                for site in sites[:n_appends]:
                    dp_ledger.append_entry(
                        cfg["dp.ledger-path"], site=site, mechanism="gaussian",
                        num_releases=meta["num_releases"],
                        noise_multiplier=meta["noise_multiplier"],
                        epsilon=meta["reported_epsilon"], delta=1e-5,
                        run_config_hash=run_hash)
        return True, False

    return fake, calls


def _drive(tmp_path, monkeypatch, strategy, provenance="real-frozen-schema",
           fail_labels=(), extra_argv=(), out_name="sweep"):
    halves = _two_halves(tmp_path, n=60)
    mod = _load_driver()
    fake, calls = _make_fake_runner(strategy, HALVES, fail_labels)
    monkeypatch.setattr(mod, "_load_pyproject", lambda: _fake_cfg(halves))
    monkeypatch.setattr(mod, "preflight_superlink", lambda cfg: None)
    monkeypatch.setattr(mod, "_run_flwr", fake)
    out_dir = tmp_path / out_name
    ledger = tmp_path / "dp_ledger.jsonl"
    argv = ["run_dp_sweep.py", "--strategy", strategy,
            "--epsilons", "3", "1", "--n-boot", "0",
            "--out-dir", str(out_dir), "--ledger-path", str(ledger),
            "--data-provenance", provenance, *extra_argv]
    if provenance == "real-frozen-schema":
        argv += ["--operator", "tester", "--gate-ack", "test acknowledgement"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return mod, calls, out_dir, ledger


@pytest.mark.parametrize("strategy", ["bagging", "cyclic"])
def test_driver_end_to_end(tmp_path, monkeypatch, strategy):
    mod, calls, out_dir, ledger = _drive(tmp_path, monkeypatch, strategy)
    # plan order honored: A first, B, then C at descending ε
    assert calls == ["armA", "armB", "armC_eps3", "armC_eps1"]
    results = json.loads((out_dir / "results.json").read_text())
    report = (out_dir / "report_1_1_b.md").read_text()
    # arm-B accounting arrives as null, not a raised ValueError
    arm_b = next(a for a in results["arms"] if a["label"] == "armB")
    assert arm_b["dp_meta"]["epsilon"] is None
    assert arm_b["dp_meta"]["sigma"] is None
    # every arm scored on BOTH halves
    for arm in results["arms"]:
        assert set(arm["metrics"]) == set(HALVES)
        assert all(m is not None for m in arm["metrics"].values())
    # strategy-aware ledger accounting
    entries = dp_ledger.read_entries(ledger)
    per_run = 2 if strategy == "bagging" else 1
    assert len(entries) == per_run * 2                    # 2 C runs fired
    if strategy == "bagging":
        assert results["ledger"]["unledgered_sites"] == []
        assert dpsweep.UNLEDGERED_MARK not in report
    else:
        assert results["ledger"]["unledgered_sites"] == [HALVES[1]]
        assert dpsweep.UNLEDGERED_MARK in report
    # composed totals only for ledgered sites, via ledger_total
    totals = results["ledger"]["ledger_total"]
    assert set(totals) == set(HALVES[: 2 if strategy == "bagging" else 1])
    assert dpsweep.REHEARSAL_BANNER not in report
    assert "test acknowledgement" in report


def test_driver_aborts_on_arm_a_failure(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="aborting the sweep"):
        _drive(tmp_path, monkeypatch, "bagging", fail_labels=("armA",))
    assert not (tmp_path / "sweep" / "results.json").exists()


def test_driver_c_failure_recorded_degenerate_without_abort(tmp_path,
                                                            monkeypatch):
    mod, calls, out_dir, ledger = _drive(tmp_path, monkeypatch, "bagging",
                                         fail_labels=("armC_eps3",))
    assert calls == ["armA", "armB", "armC_eps3", "armC_eps1"]  # continued
    results = json.loads((out_dir / "results.json").read_text())
    deg = next(a for a in results["arms"] if a["label"] == "armC_eps3")
    assert deg["degenerate"] is True
    assert all(m is None for m in deg["metrics"].values())
    report = (out_dir / "report_1_1_b.md").read_text()
    assert "DEGENERATE" in report


def test_driver_resume_skips_completed_arms(tmp_path, monkeypatch):
    _drive(tmp_path, monkeypatch, "bagging")
    # delete one C arm's model; --resume must re-fire ONLY that arm
    (tmp_path / "sweep" / "armC_eps1" / "final_model.json").unlink()
    mod, calls, out_dir, ledger = _drive(tmp_path, monkeypatch, "bagging",
                                         extra_argv=("--resume",))
    assert calls == ["armC_eps1"]
    # the re-fired C run double-appends — intent-to-spend, both compose
    assert len(dp_ledger.read_entries(ledger)) == 6


def test_driver_rehearsal_writes_no_ledger(tmp_path, monkeypatch):
    mod, calls, out_dir, ledger = _drive(tmp_path, monkeypatch, "bagging",
                                         provenance="example-halves")
    assert not ledger.exists()
    results = json.loads((out_dir / "results.json").read_text())
    assert results["ledger"]["entries_added"] == []
    assert results["ledger"]["ledger_total"] == {}
    report = (out_dir / "report_1_1_b.md").read_text()
    assert dpsweep.REHEARSAL_BANNER in report


def test_driver_real_provenance_requires_operator_and_gate_ack(tmp_path,
                                                               monkeypatch):
    halves = _two_halves(tmp_path, n=60)
    mod = _load_driver()
    monkeypatch.setattr(mod, "_load_pyproject", lambda: _fake_cfg(halves))
    monkeypatch.setattr(sys, "argv",
                        ["run_dp_sweep.py", "--data-provenance",
                         "real-frozen-schema", "--operator", "tester"])
    with pytest.raises(SystemExit, match="gate-ack"):
        mod.main()
