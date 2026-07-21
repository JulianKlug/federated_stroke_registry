"""Pure-logic + orchestration-guarantee tests for the 1.1.a HPO harness.

All fast, no `flwr run` (the E2E path is the spec §6 run matrix). Plain-pytest idiom
(no fixtures/conftest), mirroring tests/test_strategies.py; parquet-in-tmp_path halves
mirror tests/test_eval_final_model.py.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fed_stroke import hpo
from fed_stroke.baseline import score_booster_on_half, split_half
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.task import HOLDOUT_PARTITION_SEED, generate_splits, resolve_run_split


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _synthetic_half(n=300, seed=0, n_multi=20):
    """A Geneva-half-shaped frame: case_admission_id (some patients w/ 2 admissions),
    two features, binary rare-ish outcome. Enough rows that a stratified 20% split is
    two-class."""
    rng = np.random.RandomState(seed)
    pids = [f"P{i:04d}" for i in range(n)]
    caids = [f"{p}_1" for p in pids]
    # Give the first n_multi patients a second admission (patient-level split guard).
    for i in range(n_multi):
        caids.append(f"P{i:04d}_2")
    m = len(caids)
    return pd.DataFrame({
        "case_admission_id": caids,
        FEATURE_COLS[0]: rng.rand(m) * 80,
        FEATURE_COLS[1]: rng.randint(0, 30, m),
        TARGET_COL: rng.randint(0, 2, m),
    })


def _pids(df):
    return set(df["case_admission_id"].str.split("_").str[0])


# --------------------------------------------------------------------------- #
# §4.1  expand_grid
# --------------------------------------------------------------------------- #
def test_expand_grid_count_is_product_of_lengths():
    grid = {"a": [1, 2, 3], "b": [10, 20], "c": [7]}
    trials = hpo.expand_grid(grid)
    assert len(trials) == 3 * 2 * 1


def test_expand_grid_default_has_324_cells():
    assert len(hpo.expand_grid(hpo.DEFAULT_GRID)) == 324


def test_expand_grid_is_deterministic_and_sorted_keys():
    grid = {"b": [1, 2], "a": [3, 4]}
    t1 = hpo.expand_grid(grid)
    t2 = hpo.expand_grid(grid)
    assert t1 == t2
    # keys sorted -> "a" before "b" in every dict
    assert all(list(d.keys()) == ["a", "b"] for d in t1)


def test_expand_grid_single_value_grid_one_trial():
    assert hpo.expand_grid({"x": [1], "y": [2]}) == [{"x": 1, "y": 2}]


# --------------------------------------------------------------------------- #
# §4.1  valid_trials  (reuses server_app.derive_num_rounds)
# --------------------------------------------------------------------------- #
def test_valid_trials_bagging_rejects_odd_total_trees():
    trials = [{"total-trees": 39}, {"total-trees": 40}, {"total-trees": 41}]
    runnable, rejected = hpo.valid_trials(trials, "bagging", num_sites=2, local_epochs=1)
    assert [t["total-trees"] for t in runnable] == [40]
    assert {r["trial"]["total-trees"] for r in rejected} == {39, 41}
    assert all(r["reason"] for r in rejected)  # every rejection carries a reason


def test_valid_trials_cyclic_keeps_all():
    trials = [{"total-trees": 39}, {"total-trees": 40}]
    runnable, rejected = hpo.valid_trials(trials, "cyclic", num_sites=2, local_epochs=1)
    assert len(runnable) == 2 and rejected == []


# --------------------------------------------------------------------------- #
# §4.1  trial_id  (cross-process stable — NOT builtin hash())
# --------------------------------------------------------------------------- #
def test_trial_id_matches_pinned_hashlib_digest():
    import hashlib
    override = {"total-trees": 40, "params.max-depth": 4}
    canon = json.dumps(override, sort_keys=True).encode()
    assert hpo.trial_id(override) == hashlib.sha1(canon).hexdigest()[:10]


def test_trial_id_invariant_to_insertion_order():
    a = {"total-trees": 40, "params.max-depth": 4}
    b = {"params.max-depth": 4, "total-trees": 40}
    assert hpo.trial_id(a) == hpo.trial_id(b)


def test_trial_id_stable_across_pythonhashseed():
    # Builtin hash() is PYTHONHASHSEED-salted; a hashlib digest is not. Compute the
    # id in two subprocesses with different seeds and assert equality.
    snippet = (
        "from fed_stroke.hpo import trial_id;"
        "print(trial_id({'total-trees': 80, 'params.eta': 0.1}))"
    )
    arch = str(Path(__file__).resolve().parent.parent)
    outs = []
    for hashseed in ("0", "1"):
        env = {"PYTHONHASHSEED": hashseed, "PYTHONPATH": arch,
               "PATH": __import__("os").environ.get("PATH", "")}
        r = subprocess.run([sys.executable, "-c", snippet], env=env,
                           capture_output=True, text=True, check=True)
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- #
# §4.4  build_run_config  (exact string, absolute paths, no shell metachars)
# --------------------------------------------------------------------------- #
def test_build_run_config_exact_string():
    override = {"total-trees": 40, "params.max-depth": 4}
    cfg = hpo.build_run_config(
        override, "bagging", split_seed=1,
        model_dir="/abs/m", metrics_dir="/abs/mt",
        holdout_frac=0.0, holdout_eval=False, n_boot=0)
    assert cfg == (
        "train-method='bagging' params.max-depth=4 total-trees=40 "
        "split-seed=1 holdout-frac=0.0 holdout-eval=false save-model=true "
        "n-boot=0 model-dir='/abs/m' metrics-dir='/abs/mt'"
    )


def test_build_run_config_paths_absolute_and_no_shell_metacharacters():
    cfg = hpo.build_run_config(
        {"total-trees": 40}, "cyclic", 2,
        Path("/tmp/x/y"), Path("/tmp/x/y"), holdout_frac=0.2, holdout_eval=True)
    assert "model-dir='/tmp/x/y'" in cfg and "metrics-dir='/tmp/x/y'" in cfg
    assert "holdout-eval=true" in cfg and "n-boot=0" in cfg
    # guards the shell=False contract: none of these may appear
    assert not any(ch in cfg for ch in ";|&$<>`\n\\")


# --------------------------------------------------------------------------- #
# §4.4  subsample + resume predicate
# --------------------------------------------------------------------------- #
def test_subsample_trials_reproducible_and_full_when_over_len():
    trials = hpo.expand_grid(hpo.DEFAULT_GRID)
    s1 = hpo.subsample_trials(trials, 10, seed=7)
    s2 = hpo.subsample_trials(trials, 10, seed=7)
    assert s1 == s2 and len(s1) == 10
    s3 = hpo.subsample_trials(trials, 10, seed=8)
    assert s3 != s1  # a different seed generally differs
    assert hpo.subsample_trials(trials, 10_000, seed=0) == trials  # >= len -> full
    assert hpo.subsample_trials(trials, 0, seed=0) == trials       # 0 -> full


def test_resume_predicate(tmp_path):
    tid, seed = "abc123", 2
    assert not hpo.is_completed(tmp_path, tid, seed)
    mp = hpo.final_model_path(tmp_path, tid, seed)
    mp.parent.mkdir(parents=True)
    mp.write_text("{}")
    assert hpo.is_completed(tmp_path, tid, seed)
    assert not hpo.is_completed(tmp_path, tid, 3)  # a different seed still runnable


# --------------------------------------------------------------------------- #
# §4.2  trial_objective
# --------------------------------------------------------------------------- #
def test_trial_objective_mean_and_variance():
    per_repeat = [
        {"split_seed": 1, "site_aucs": {"A": 0.8, "B": 0.7}},
        {"split_seed": 2, "site_aucs": {"A": 0.9, "B": 0.8}},
    ]
    out = hpo.trial_objective(per_repeat, k_search=2)
    # repeat means 0.75, 0.85 -> objective 0.80; pop-vars 0.0025, 0.0025 -> mean 0.0025
    assert out["objective"] == pytest.approx(0.80)
    assert out["variance"] == pytest.approx(0.0025)
    assert out["n_valid_repeats"] == 2 and out["valid"]


def test_trial_objective_drops_none_repeat():
    per_repeat = [
        {"split_seed": 1, "site_aucs": {"A": 0.8, "B": 0.7}},
        {"split_seed": 2, "site_aucs": {"A": 0.9, "B": None}},  # dropped
        {"split_seed": 3, "site_aucs": {"A": 0.85, "B": 0.75}},
    ]
    out = hpo.trial_objective(per_repeat, k_search=3)
    assert out["n_valid_repeats"] == 2
    assert out["objective"] == pytest.approx((0.75 + 0.80) / 2)
    assert out["valid"]  # 2 >= ceil(3/2)=2
    assert out["per_repeat"][1]["dropped"] is True


def test_trial_objective_invalid_below_quorum():
    per_repeat = [
        {"split_seed": 1, "site_aucs": {"A": None, "B": None}},
        {"split_seed": 2, "site_aucs": {"A": None, "B": None}},
        {"split_seed": 3, "site_aucs": {"A": 0.8, "B": 0.7}},
    ]
    out = hpo.trial_objective(per_repeat, k_search=3)
    assert out["n_valid_repeats"] == 1  # < ceil(3/2)=2
    assert out["valid"] is False
    assert math.isnan(out["objective"])


# --------------------------------------------------------------------------- #
# §4.2  rank_trials / select_winner
# --------------------------------------------------------------------------- #
def _trial(tid, obj, var, valid=True):
    return {"trial_id": tid, "params": {}, "objective": obj, "variance": var,
            "n_valid_repeats": 3, "valid": valid, "per_repeat": []}


def test_rank_trials_orders_by_objective_then_variance_then_id():
    a = _trial("aaa", 0.80, 0.01)
    b = _trial("bbb", 0.80, 0.005)   # same obj, lower variance -> ahead of a
    c = _trial("ccc", 0.90, 0.02)    # highest obj -> first
    d = _trial("ddd", float("nan"), float("nan"), valid=False)  # sinks
    ranked = hpo.rank_trials([a, d, b, c])
    assert [t["trial_id"] for t in ranked] == ["ccc", "bbb", "aaa", "ddd"]


def test_select_winner_and_all_invalid_returns_none():
    ranked = hpo.rank_trials([_trial("x", 0.7, 0.01), _trial("y", 0.8, 0.01)])
    assert hpo.select_winner(ranked)["trial_id"] == "y"
    invalid = [_trial("p", float("nan"), float("nan"), valid=False)]
    assert hpo.select_winner(hpo.rank_trials(invalid)) is None


# --------------------------------------------------------------------------- #
# §4.3  summarize_top_ranges + build_signal
# --------------------------------------------------------------------------- #
def _valid_trial_with_params(tid, obj, params, repeat_objs):
    per_repeat = [{"split_seed": i, "site_aucs": {"A": o, "B": o},
                   "objective": o, "variance": 0.0, "dropped": False}
                  for i, o in enumerate(repeat_objs)]
    return {"trial_id": tid, "params": params, "objective": obj, "variance": 0.0,
            "n_valid_repeats": len(repeat_objs), "valid": True, "per_repeat": per_repeat}


def _params(tt, md, eta, mcw, ss, cs):
    return {"total-trees": tt, "params.max-depth": md, "params.eta": eta,
            "params.min-child-weight": mcw, "params.subsample": ss,
            "params.colsample-bytree": cs}


def test_summarize_top_ranges_picks_min_max_mode_over_top_decile():
    # 10 valid trials -> top decile = 1 trial (the ranked winner).
    ranked = [
        _valid_trial_with_params(f"t{i}", 0.9 - i * 0.05,
                                 _params(40 if i == 0 else 160, 3, 0.1, 5, 1.0, 1.0),
                                 [0.9 - i * 0.05])
        for i in range(10)
    ]
    ranges = hpo.summarize_top_ranges(ranked, top_frac=0.1)
    assert ranges["n_top"] == 1
    assert ranges["total-trees"] == {"min": 40, "max": 40, "mode": 40, "values": [40]}


def test_build_signal_separated_vs_flat():
    # well separated: winner ~0.9 (tight), runner-up ~0.7 (tight) -> no overlap
    sep = [
        _valid_trial_with_params("w", 0.90, _params(40, 3, 0.1, 5, 1.0, 1.0),
                                 [0.90, 0.905, 0.895]),
        _valid_trial_with_params("r", 0.70, _params(80, 4, 0.1, 5, 1.0, 1.0),
                                 [0.70, 0.705, 0.695]),
    ]
    sig = hpo.build_signal(sep)
    assert sig["top_objective"] == pytest.approx(0.90)
    assert sig["top_minus_median"] == pytest.approx(0.10)  # winner well above median
    assert sig["winner_vs_runnerup_ci_overlap"] is False

    # flat: two near-identical trials with wide per-repeat spread -> overlap
    flat = [
        _valid_trial_with_params("w", 0.71, _params(40, 3, 0.1, 5, 1.0, 1.0),
                                 [0.60, 0.71, 0.82]),
        _valid_trial_with_params("r", 0.70, _params(80, 4, 0.1, 5, 1.0, 1.0),
                                 [0.59, 0.70, 0.81]),
    ]
    assert hpo.build_signal(flat)["winner_vs_runnerup_ci_overlap"] is True


# --------------------------------------------------------------------------- #
# §4.3  build_results / validate_results / render_leaderboard
# --------------------------------------------------------------------------- #
def _good_results():
    ranked = [
        _valid_trial_with_params("w", 0.80, _params(40, 4, 0.1, 5, 1.0, 1.0),
                                 [0.79, 0.81]),
        {"trial_id": "bad", "params": _params(80, 3, 0.3, 20, 0.8, 0.8),
         "objective": float("nan"), "variance": float("nan"),
         "n_valid_repeats": 0, "valid": False, "per_repeat": []},
    ]
    meta = {"strategy": "bagging", "federation": "local-deployment",
            "search_seeds": [1, 2], "holdout_frac": 0.2,
            "holdout_partition_seed": HOLDOUT_PARTITION_SEED,
            "operating_point": 0.5, "data_provenance": "example-halves", "grid": {}}
    holdout = {"params": ranked[0]["params"], "holdout_frac": 0.2,
               "holdout_partition_seed": HOLDOUT_PARTITION_SEED, "site_metrics": {}}
    return hpo.build_results(meta, ranked, holdout)


def test_validate_results_accepts_well_formed():
    hpo.validate_results(_good_results())


def test_render_leaderboard_shows_valid_and_flagged_rows():
    md = hpo.render_leaderboard(_good_results())
    assert "`w`" in md and "`bad`" in md
    assert "⚠" in md                # the invalid trial is flagged, not dropped
    assert md.count("⚠") == 1


def test_validate_results_missing_top_level_key_raises():
    r = _good_results()
    del r["holdout"]
    with pytest.raises(ValueError):
        hpo.validate_results(r)


def test_validate_results_bad_or_missing_data_provenance_raises():
    r = _good_results()
    r["meta"]["data_provenance"] = "made-up"
    with pytest.raises(ValueError):
        hpo.validate_results(r)
    r["meta"].pop("data_provenance")
    with pytest.raises(ValueError):
        hpo.validate_results(r)


def test_validate_results_missing_holdout_frac_raises():
    r = _good_results()
    r["meta"].pop("holdout_frac")
    with pytest.raises(ValueError):
        hpo.validate_results(r)


def test_validate_results_missing_signal_key_raises():
    r = _good_results()
    r["signal"].pop("winner_vs_runnerup_ci_overlap")
    with pytest.raises(ValueError):
        hpo.validate_results(r)


# --------------------------------------------------------------------------- #
# §4.5  resolve_run_split  (the split-contract core)
# --------------------------------------------------------------------------- #
def test_resolve_run_split_byte_identical_to_legacy_seed42():
    df = _synthetic_half()
    a_tr, a_va = resolve_run_split(df.copy(), outcome=TARGET_COL,
                                   split_seed=42, holdout_frac=0.0)
    b_tr, b_va, _, _ = generate_splits(df.copy(), outcome=TARGET_COL,
                                       test_size=0.2, seed=42)
    assert set(a_tr["case_admission_id"]) == set(b_tr["case_admission_id"])
    assert set(a_va["case_admission_id"]) == set(b_va["case_admission_id"])


def test_resolve_run_split_distinct_seeds_distinct_dev_splits():
    df = _synthetic_half()
    tr1, va1 = resolve_run_split(df.copy(), TARGET_COL, split_seed=1, holdout_frac=0.2)
    tr2, va2 = resolve_run_split(df.copy(), TARGET_COL, split_seed=2, holdout_frac=0.2)
    assert set(va1["case_admission_id"]) != set(va2["case_admission_id"])


def test_resolve_run_split_holdout_is_patient_disjoint_from_every_search_split():
    df = _synthetic_half()
    _, held = resolve_run_split(df.copy(), TARGET_COL, holdout_frac=0.2,
                                holdout_eval=True)
    held_pids = _pids(held)
    for seed in (1, 2, 3):
        tr, va = resolve_run_split(df.copy(), TARGET_COL, split_seed=seed,
                                   holdout_frac=0.2, holdout_eval=False)
        assert held_pids.isdisjoint(_pids(tr))
        assert held_pids.isdisjoint(_pids(va))


def test_resolve_run_split_dev_held_partition_covers_all_and_is_disjoint():
    df = _synthetic_half()
    dev, held = resolve_run_split(df.copy(), TARGET_COL, holdout_frac=0.2,
                                  holdout_eval=True)
    dev_pids, held_pids = _pids(dev), _pids(held)
    assert dev_pids.isdisjoint(held_pids)
    assert dev_pids | held_pids == _pids(df)


def test_resolve_run_split_holdout_seed_is_fixed_partition_not_search_seed():
    # The DEV/HELD partition must NOT move with split_seed (else the hold-out leaks).
    df = _synthetic_half()
    _, held_a = resolve_run_split(df.copy(), TARGET_COL, split_seed=1,
                                  holdout_frac=0.2, holdout_eval=True)
    _, held_b = resolve_run_split(df.copy(), TARGET_COL, split_seed=999,
                                  holdout_frac=0.2, holdout_eval=True)
    assert _pids(held_a) == _pids(held_b)
    assert HOLDOUT_PARTITION_SEED == 42


def test_split_half_reconstructs_the_run_split(tmp_path):
    df = _synthetic_half()
    path = tmp_path / "geneva_half_A.parquet"
    df.to_parquet(path)
    # split_half at HPO params must match resolve_run_split on the same data.
    _, sh_valid = split_half(path, split_seed=2, holdout_frac=0.2, holdout_eval=False)
    _, rr_valid = resolve_run_split(pd.read_parquet(path), TARGET_COL,
                                    split_seed=2, holdout_frac=0.2, holdout_eval=False)
    assert set(sh_valid["case_admission_id"]) == set(rr_valid["case_admission_id"])


def test_score_booster_on_half_reconstructs_holdout_split(tmp_path):
    # The scorer with matching params scores the SAME HELD rows the run trained against.
    df = _synthetic_half()
    path = tmp_path / "geneva_half_A.parquet"
    df.to_parquet(path)
    dev, held = resolve_run_split(df.copy(), TARGET_COL, holdout_frac=0.2,
                                  holdout_eval=True)
    dtrain = xgb.DMatrix(dev[FEATURE_COLS], label=dev[TARGET_COL])
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 0},
                    dtrain, num_boost_round=5)
    m = score_booster_on_half(bst, path, 0.5, n_boot=0, boot_seed=0,
                              split_seed=42, holdout_frac=0.2, holdout_eval=True)
    assert m["n"] == len(held)  # scored exactly the HELD rows
