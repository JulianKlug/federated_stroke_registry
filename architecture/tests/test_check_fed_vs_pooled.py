"""[T3] End-to-end exit-code contract for scripts/check_fed_vs_pooled.py
(acceptance criteria 1 & 6).

Most cases are driven by `subprocess` on synthetic fixtures (two tiny halves + a
couple of stamped boosters) so the real process exit code is exercised. The
round-trip-failure case is driven in-process with a monkeypatch, because a
genuinely non-round-tripping model file cannot be produced with stock XGBoost —
its unit-level coverage lives in test_baseline.test_roundtrip_non_fixed_point_raises.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ARCH_DIR = Path(__file__).resolve().parent.parent
CLI = ARCH_DIR / "scripts" / "check_fed_vs_pooled.py"

BASE_PARAMS = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3,
               "seed": 0, "tree_method": "hist"}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _half_df(prefix, seed, n=80, single_class=False):
    rng = np.random.RandomState(seed)
    age = rng.uniform(40, 90, n)
    nih = rng.uniform(0, 30, n)
    if single_class:
        y = np.zeros(n, dtype=int)
    else:
        logits = 0.08 * (age - 65) + 0.15 * (nih - 15) + rng.normal(0, 1, n)
        y = (logits > np.quantile(logits, 0.6)).astype(int)
    return pd.DataFrame({
        "case_admission_id": [f"{prefix}{i}_1" for i in range(n)],
        "Age (calc.)": age, "NIH on admission": nih, "3M Death": y,
    })


def _write_halves(tmp_path, b_single_class=False):
    a = tmp_path / "geneva_half_A.parquet"
    b = tmp_path / "geneva_half_B.parquet"
    _half_df("A", 1).to_parquet(a)
    _half_df("B", 2, single_class=b_single_class).to_parquet(b)
    return a, b


def _dmatrix(df):
    return xgb.DMatrix(df[["Age (calc.)", "NIH on admission"]], label=df["3M Death"])


def _save_model(path, df, rounds, stamp_params=None, stamp_trees=None,
                shuffle_labels=False, seed=0):
    """Train a booster on `df` and save it, optionally stamping fed_run_config."""
    if shuffle_labels:
        df = df.copy()
        df["3M Death"] = np.random.RandomState(99).permutation(df["3M Death"].values)
    bst = xgb.train({**BASE_PARAMS, "seed": seed}, _dmatrix(df), num_boost_round=rounds)
    if stamp_params is not None:
        bst.set_attr(fed_run_config=json.dumps(
            {"params": stamp_params, "total_trees": stamp_trees}))
    bst.save_model(str(path))
    return path


def _run(tmp_path, fed, *extra, data=None, delta="1.0"):
    """Invoke the CLI as a subprocess; return the CompletedProcess."""
    data = data or [str(tmp_path / "geneva_half_A.parquet"),
                    str(tmp_path / "geneva_half_B.parquet")]
    cmd = [sys.executable, str(CLI), "--data", *data, "--fed", *fed,
           "--max-auc-delta", delta, "--out-dir", str(tmp_path / "metrics"),
           "--n-boot", "50", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ARCH_DIR))


def _artifact(tmp_path):
    return json.loads((tmp_path / "metrics" / "fed_vs_pooled.json").read_text())


# --------------------------------------------------------------------------- #
# e2e cases
# --------------------------------------------------------------------------- #
def test_all_within_tol_exits_zero(tmp_path):
    a, b = _write_halves(tmp_path)
    both = pd.concat([_half_df("A", 1), _half_df("B", 2)], ignore_index=True)
    _save_model(tmp_path / "bagging.json", both, rounds=6,
                stamp_params=BASE_PARAMS, stamp_trees=6)
    proc = _run(tmp_path, [f"bagging={tmp_path / 'bagging.json'}"], delta="1.0")
    assert proc.returncode == 0, proc.stderr
    art = _artifact(tmp_path)
    assert art["passed"] is True
    assert art["config_provenance"] == "model"
    assert art["roundtrip_ok"] is True


def test_over_tol_bagging_exits_nonzero(tmp_path):
    a, b = _write_halves(tmp_path)
    # A model trained on shuffled labels scores ~0.5 -> large delta vs pooled.
    _save_model(tmp_path / "bagging.json", _half_df("A", 1), rounds=6,
                stamp_params=BASE_PARAMS, stamp_trees=6, shuffle_labels=True)
    proc = _run(tmp_path, [f"bagging={tmp_path / 'bagging.json'}"], delta="0.0")
    assert proc.returncode != 0
    art = _artifact(tmp_path)
    assert art["passed"] is False
    assert art["gate"]["bagging"]["gating"] is True


def test_over_tol_cyclic_is_informational_exits_zero(tmp_path):
    a, b = _write_halves(tmp_path)
    _save_model(tmp_path / "cyclic_forward.json", _half_df("A", 1), rounds=6,
                stamp_params=BASE_PARAMS, stamp_trees=6, shuffle_labels=True)
    proc = _run(tmp_path, [f"cyclic_forward={tmp_path / 'cyclic_forward.json'}"],
                delta="0.0")
    assert proc.returncode == 0, proc.stderr
    art = _artifact(tmp_path)
    assert art["passed"] is True                      # cyclic does not gate
    assert art["gate"]["cyclic_forward"]["gating"] is False
    assert art["gate"]["cyclic_forward"]["overall"] is False  # recorded, not gated


def test_nan_auc_is_hard_fail(tmp_path):
    a, b = _write_halves(tmp_path, b_single_class=True)  # half_B valid is single-class
    both = pd.concat([_half_df("A", 1), _half_df("B", 2, single_class=True)],
                     ignore_index=True)
    _save_model(tmp_path / "bagging.json", both, rounds=6,
                stamp_params=BASE_PARAMS, stamp_trees=6)
    proc = _run(tmp_path, [f"bagging={tmp_path / 'bagging.json'}"], delta="1.0")
    assert proc.returncode != 0
    assert _artifact(tmp_path)["passed"] is False


def test_divergent_configs_exit_nonzero(tmp_path):
    a, b = _write_halves(tmp_path)
    _save_model(tmp_path / "m1.json", _half_df("A", 1), rounds=5,
                stamp_params=BASE_PARAMS, stamp_trees=5)
    _save_model(tmp_path / "m2.json", _half_df("A", 1), rounds=5,
                stamp_params=BASE_PARAMS, stamp_trees=7)  # divergent budget
    proc = _run(tmp_path, [f"bagging={tmp_path / 'm1.json'}",
                           f"cyclic_forward={tmp_path / 'm2.json'}"], delta="1.0")
    assert proc.returncode != 0
    assert "matched-budget" in proc.stderr


def _write_fixture_pyproject(tmp_path, total_trees=5):
    p = tmp_path / "pyproject.toml"
    p.write_text(
        "[tool.flwr.app]\n[tool.flwr.app.config]\n"
        f"total-trees = {total_trees}\n"
        'params.objective = "binary:logistic"\n'
        "params.seed = 0\nparams.max-depth = 2\nparams.eta = 0.3\n"
        'params.tree-method = "hist"\n'
    )
    return p


def test_pyproject_fallback_hard_fails_by_default(tmp_path):
    a, b = _write_halves(tmp_path)
    _save_model(tmp_path / "legacy.json", _half_df("A", 1), rounds=5)  # no stamp
    pyproj = _write_fixture_pyproject(tmp_path, total_trees=5)
    proc = _run(tmp_path, [f"bagging={tmp_path / 'legacy.json'}"],
                "--pyproject", str(pyproj), delta="1.0")
    assert proc.returncode != 0
    assert "provenance unverified" in proc.stderr


def test_pyproject_fallback_allowed_with_flag(tmp_path):
    a, b = _write_halves(tmp_path)
    _save_model(tmp_path / "legacy.json", _half_df("A", 1), rounds=5)  # no stamp
    pyproj = _write_fixture_pyproject(tmp_path, total_trees=5)
    proc = _run(tmp_path, [f"bagging={tmp_path / 'legacy.json'}"],
                "--pyproject", str(pyproj), "--allow-pyproject-fallback", delta="1.0")
    assert proc.returncode == 0, proc.stderr
    assert _artifact(tmp_path)["config_provenance"] == "pyproject-fallback"


# --------------------------------------------------------------------------- #
# round-trip failure — in-process, monkeypatched (see module docstring)
# --------------------------------------------------------------------------- #
def _load_cli_module():
    spec = importlib.util.spec_from_file_location("check_fed_vs_pooled", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_roundtrip_failure_exits_nonzero(tmp_path, monkeypatch):
    a, b = _write_halves(tmp_path)
    both = pd.concat([_half_df("A", 1), _half_df("B", 2)], ignore_index=True)
    _save_model(tmp_path / "bagging.json", both, rounds=6,
                stamp_params=BASE_PARAMS, stamp_trees=6)
    cli = _load_cli_module()

    def _boom(bst, dmatrix):
        raise ValueError("serialization round-trip not a fixed point: injected")
    monkeypatch.setattr(cli, "assert_prediction_roundtrip", _boom)
    monkeypatch.setattr(sys, "argv", [
        "check_fed_vs_pooled.py",
        "--data", str(a), str(b),
        "--fed", f"bagging={tmp_path / 'bagging.json'}",
        "--max-auc-delta", "1.0",
        "--out-dir", str(tmp_path / "metrics"), "--n-boot", "50",
    ])
    rc = cli.main()
    assert rc != 0
    assert _artifact(tmp_path)["roundtrip_ok"] is False
