"""fed_stroke: pooled-vs-federated baseline core (roadmap 1.d).

Reusable, testable functions behind `scripts/check_fed_vs_pooled.py`. No CLI, no
printing — the script and the unit tests both call these.

The one thing this module must get right is *provenance*: the pooled reference is
only a valid baseline if it is trained with the exact params + tree budget the
federated models were actually trained with. That provenance flows through the
saved model itself (the `fed_run_config` attribute stamped by
`server_app.save_final_model`), not through a re-read of `pyproject.toml`
(fallback only). See docs/specs/1d_federated_vs_pooled.md §4.
"""
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common.config import unflatten_dict

from fed_stroke.metrics import compute_binary_metrics
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.task import generate_splits, replace_keys


def load_matched_config(pyproject_path) -> tuple[dict, int]:
    """FALLBACK reader: [tool.flwr.app.config] -> (params, total_trees).

    Uses tomllib + unflatten_dict + task.replace_keys, the exact machinery the FL
    path uses to derive params from `context.run_config`. tomllib parses the TOML
    dotted keys (`params.eta`) into an *already-nested* dict, whereas the FL
    `context.run_config` is flat; the two converge to byte-identical params because
    `unflatten_dict` preserves an already-nested value unchanged. The
    test_baseline equivalence test pins that idempotency rather than hand-copying a
    dict. Used only when a model carries no embedded `fed_run_config`.
    """
    with open(pyproject_path, "rb") as fh:
        data = tomllib.load(fh)
    cfg_table = data["tool"]["flwr"]["app"]["config"]
    cfg = replace_keys(unflatten_dict(cfg_table))
    return cfg["params"], int(cfg_table["total-trees"])


def read_model_config(bst, pyproject_path) -> tuple[dict, int, str]:
    """Return (params, total_trees, provenance) for a saved federated model.

    Primary source: the model's own embedded `fed_run_config` attribute, written
    by `server_app.save_final_model` at save time, so the pooled reference matches
    the params/budget THIS model was actually trained with (including any
    `--run-config` override). provenance="model".

    Fallback: if the attribute is absent (legacy model), read `load_matched_config`
    (pyproject) and return provenance="pyproject-fallback" so the caller can warn
    (or hard-fail).
    """
    raw = bst.attr("fed_run_config")
    if raw is not None:
        cfg = json.loads(raw)
        return cfg["params"], int(cfg["total_trees"]), "model"
    params, total_trees = load_matched_config(pyproject_path)
    return params, total_trees, "pyproject-fallback"


def assert_matched_across_models(configs) -> tuple[dict, int]:
    """Given the (params, total_trees) read from every federated model, assert they
    are identical across all models and return the single agreed (params,
    total_trees). Divergence is a HARD FAIL: it means the strategies were not run
    at a matched budget, so no single pooled reference is valid.
    """
    if not configs:
        raise ValueError("no federated model configs to match against")
    ref_params, ref_trees = configs[0]
    for i, (params, total_trees) in enumerate(configs[1:], start=1):
        if total_trees != ref_trees:
            raise ValueError(
                "matched-budget violation: model %d has total_trees=%s, "
                "model 0 has total_trees=%s" % (i, total_trees, ref_trees)
            )
        if params != ref_params:
            raise ValueError(
                "matched-params violation: model %d params differ from model 0 "
                "(%s vs %s)" % (i, params, ref_params)
            )
    return ref_params, ref_trees


def split_half(data_path):
    """(train_df, valid_df) for one half via generate_splits(test_size=0.2, seed=42).

    The exact split contract `load_data_gva` uses, so `valid_df` here IS the split
    `client_app.evaluate` scored on. The returned frames carry the full column set
    (incl. `patient_id`/`case_admission_id`/target) — callers subset as needed.
    """
    data_df = pd.read_parquet(data_path)
    train_df, valid_df, _, _ = generate_splits(
        data_df, outcome=TARGET_COL, test_size=0.2, seed=42
    )
    return train_df, valid_df


def train_pooled_booster(half_paths, params, num_boost_round) -> xgb.Booster:
    """Concatenate every half's train_df, build ONE DMatrix, xgb.train once.

    Selects DMatrix columns as FEATURE_COLS (features) + TARGET_COL (label),
    exactly as `load_data_gva` does — `generate_splits` returns the FULL frame, so
    feeding it raw would leak ID columns in as features.

    Asserts pooled-train patient IDs are disjoint from every half's valid_df (no
    validation row leaks into the reference). Forces a DETERMINISTIC fit so the
    pooled reference is reproducible run-to-run (a ±0.01 wobble can flip PASS/FAIL
    at the calibrated tolerance): params must carry a pinned `seed`, AND the pooled
    fit overrides `nthread=1` (XGBoost `hist` can carry tiny float nondeterminism
    across threads). The config's `nthread` is a federation-throughput knob, not a
    correctness one, so overriding it for the reference is safe. Neither override
    mutates the caller's params dict.
    """
    if "seed" not in params:
        raise ValueError(
            "train_pooled_booster requires a pinned `seed` in params for a "
            "reproducible pooled reference; none present"
        )

    train_frames = []
    for path in half_paths:
        train_df, valid_df = split_half(path)
        train_pids = set(train_df["patient_id"])
        valid_pids = set(valid_df["patient_id"])
        leak = train_pids & valid_pids
        if leak:
            raise ValueError(
                f"leakage: {len(leak)} pooled-train patient IDs also appear in "
                f"{Path(path).name}'s validation split"
            )
        train_frames.append(train_df)

    pooled_train = pd.concat(train_frames, ignore_index=True)
    # Cross-half leakage guard: no pooled-train patient may appear in ANY half's
    # validation split (not just its own).
    pooled_train_pids = set(pooled_train["patient_id"])
    for path in half_paths:
        _, valid_df = split_half(path)
        cross_leak = pooled_train_pids & set(valid_df["patient_id"])
        if cross_leak:
            raise ValueError(
                f"leakage: {len(cross_leak)} pooled-train patient IDs appear in "
                f"{Path(path).name}'s validation split"
            )

    dtrain = xgb.DMatrix(pooled_train[FEATURE_COLS], label=pooled_train[TARGET_COL])
    fit_params = {**params, "nthread": 1}  # deterministic: single-thread + pinned seed
    return xgb.train(fit_params, dtrain, num_boost_round=num_boost_round)


def score_booster_on_half(bst, data_path, operating_point, n_boot, boot_seed) -> dict:
    """Rebuild the half's valid split and return compute_binary_metrics(...).

    The single shared scorer: the federated model, the pooled model, and 1.c's
    `eval_final_model.py` all score a booster on a half through this one function,
    so their AUCs match by construction. Mirrors `client_app.evaluate`: same split
    (`generate_splits`, test_size=0.2, seed=42), same DMatrix columns, same
    `compute_binary_metrics` call.
    """
    _, valid_df = split_half(data_path)
    valid_dmatrix = xgb.DMatrix(valid_df[FEATURE_COLS], label=valid_df[TARGET_COL])
    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    return compute_binary_metrics(
        y_true, y_prob, operating_point, n_boot=n_boot, boot_seed=boot_seed
    )


def assert_prediction_roundtrip(bst, dmatrix) -> None:
    """Serialization tripwire (independent of the AUC gate). Predict; re-serialize
    the booster (save_raw("json")); reload into a fresh Booster; predict again;
    assert bitwise-identical per-row output (np.array_equal).

    A save/load that is not a fixed point — a dropped or reordered tree, a
    truncated node, a wrong base_score — surfaces here even when it leaves AUC-ROC
    unchanged, because AUC is rank-only and blind to monotonic score distortion.
    Raises on any mismatch. Run on every federated model and the pooled model.
    """
    y_before = bst.predict(dmatrix)
    reloaded = xgb.Booster()
    reloaded.load_model(bytearray(bst.save_raw("json")))
    y_after = reloaded.predict(dmatrix)
    if not np.array_equal(y_before, y_after):
        n_diff = int(np.sum(y_before != y_after))
        raise ValueError(
            f"serialization round-trip not a fixed point: {n_diff}/{len(y_before)} "
            "rows changed prediction after save_raw->load_model"
        )


def compare_auc(pooled_by_site, federated_by_site, tol) -> dict:
    """Pure gate: per-site delta = |auc_fed - auc_pooled|; pass if delta <= tol.

    NaN on either side (single-class split) is a HARD FAIL, not a skip: `passed` is
    False and `delta` is NaN. Returns {"sites": {site: {pooled, federated, delta,
    passed}}, "overall": bool}. No I/O — directly unit-testable. Whether a
    strategy's result *gates the exit code* is the CLI's call (bagging gates;
    cyclic is informational), not this function's — it scores every strategy
    identically.
    """
    sites = {}
    overall = True
    for site, pooled_metrics in pooled_by_site.items():
        pooled_auc = float(pooled_metrics["auc_roc"])
        fed_auc = float(federated_by_site[site]["auc_roc"])
        if math.isnan(pooled_auc) or math.isnan(fed_auc):
            delta = float("nan")
            passed = False
        else:
            delta = abs(fed_auc - pooled_auc)
            passed = delta <= tol
        sites[site] = {
            "pooled": pooled_auc,
            "federated": fed_auc,
            "delta": delta,
            "passed": passed,
        }
        overall = overall and passed
    return {"sites": sites, "overall": overall}
