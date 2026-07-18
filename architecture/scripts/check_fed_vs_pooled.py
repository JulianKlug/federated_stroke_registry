"""Federated-vs-pooled correctness check on the Geneva 50/50 partition (roadmap 1.d).

Tripwire for the bug class the federated path can silently introduce
(patient-ID-stratified per-node splits, per-node DMatrix construction, per-round
tree aggregation, repeated save_raw/load_model round-trips). It trains ONE pooled
XGBoost on the union of the two halves' train splits — with the exact params +
tree budget the federated models were actually trained with, read from each
model's embedded `fed_run_config` — and gates each federated model's per-site AUC
against that reference.

Bagging hard-gates (it is the ensemble most directly comparable to pooling);
cyclic is reported as INFO because its last-site bias is a known non-bug (§8.2). A
serialization round-trip tripwire runs on every model independently of the AUC
deltas (AUC is rank-only and blind to a monotonic-distorting serialize bug).

Usage:
    python scripts/check_fed_vs_pooled.py \
        --data ../out/geneva_half_A.parquet ../out/geneva_half_B.parquet \
        --fed bagging=../out/models/bagging.json \
              cyclic_forward=../out/models/cyclic_forward.json \
              cyclic_reverse=../out/models/cyclic_reverse.json \
        --max-auc-delta 0.03

There is no --expected-trees flag: the budget is read from each model's embedded
config, so there is a single source of truth. See
docs/specs/1d_federated_vs_pooled.md §4.2.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Make `fed_stroke` importable regardless of CWD (script lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fed_stroke.baseline import (  # noqa: E402
    assert_matched_across_models,
    assert_prediction_roundtrip,
    compare_auc,
    read_model_config,
    score_booster_on_half,
    split_half,
    train_pooled_booster,
)
from fed_stroke.metrics import compute_binary_metrics  # noqa: E402
from fed_stroke.schema import FEATURE_COLS, TARGET_COL  # noqa: E402

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
METRICS_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "metrics"


def _parse_fed(items):
    """['bagging=path', ...] -> {'bagging': Path(path), ...}, order preserved."""
    out = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--fed entry must be LABEL=PATH, got: {item!r}")
        if label in out:
            raise SystemExit(f"duplicate --fed label: {label!r}")
        out[label] = Path(path)
    return out


def _load_model(path):
    bst = xgb.Booster()
    bst.load_model(str(path))
    bst.set_param({"eval_metric": "auc"})
    return bst


def _gates(label):
    """Bagging hard-gates; cyclic is informational (§8.2)."""
    return "cyclic" not in label.lower()


def _json_safe(obj):
    """Recursively replace NaN/inf floats with None so json.dumps(allow_nan=False)
    never emits the bare `NaN` token that strict parsers reject (§4.5 discipline)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _fmt(x):
    return "  NaN  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _valid_dmatrix(data_path):
    _, valid_df = split_half(data_path)
    return xgb.DMatrix(valid_df[FEATURE_COLS], label=valid_df[TARGET_COL])


def _score_combined(bst, data_paths, operating_point, n_boot, boot_seed):
    """Informational pooled-combined AUC: score on concat of every half's valid
    split (the gate itself is per-site; this is a diagnostic only)."""
    valid_frames = [split_half(p)[1] for p in data_paths]
    combined = pd.concat(valid_frames, ignore_index=True)
    dm = xgb.DMatrix(combined[FEATURE_COLS], label=combined[TARGET_COL])
    y_prob = bst.predict(dm)
    y_true = dm.get_label()
    return compute_binary_metrics(
        y_true, y_prob, operating_point, n_boot=n_boot, boot_seed=boot_seed
    )


def _run_calibrate(args, params, total_trees, fed_models):
    """Train the pooled reference across N seeds, print the no-bug
    |bagging_auc - pooled_auc| distribution per site, exit 0 without gating.

    Measures POOLED-side variance only (the bagging model is a single fixed run),
    so it understates the true no-bug spread — add margin when setting the
    tolerance (§6 note)."""
    gating = [lbl for lbl in fed_models if _gates(lbl)]
    if not gating:
        raise SystemExit("--calibrate needs a gating (bagging) model in --fed")
    ref_label = gating[0]
    ref_bst = fed_models[ref_label]
    ref_by_site = {
        Path(p).name: score_booster_on_half(
            ref_bst, p, args.operating_point, args.n_boot, args.boot_seed
        )
        for p in args.data
    }

    print(f"\nCalibration: pooled reference over {args.calibrate} seeds "
          f"vs fed '{ref_label}' (pooled-side variance only)\n")
    deltas_by_site = {Path(p).name: [] for p in args.data}
    for seed in range(args.calibrate):
        pooled = train_pooled_booster(
            args.data, {**params, "seed": seed}, total_trees
        )
        for p in args.data:
            site = Path(p).name
            m = score_booster_on_half(
                pooled, p, args.operating_point, args.n_boot, args.boot_seed
            )
            delta = abs(float(m["auc_roc"]) - float(ref_by_site[site]["auc_roc"]))
            deltas_by_site[site].append(delta)
            print(f"  seed={seed}  {site:<26} pooled_auc={m['auc_roc']:.4f}  "
                  f"|delta|={delta:.4f}")

    print(f"\n{'site':<26} {'min':>8} {'mean':>8} {'max':>8}")
    print("-" * 54)
    overall_max = 0.0
    for site, deltas in deltas_by_site.items():
        arr = np.array(deltas)
        overall_max = max(overall_max, float(arr.max()))
        print(f"{site:<26} {arr.min():>8.4f} {arr.mean():>8.4f} {arr.max():>8.4f}")
    print(f"\nMax observed no-bug |delta| = {overall_max:.4f}. Set --max-auc-delta "
          f"above this (add margin; pooled-side variance only). Record in "
          f"docs/logbook.md.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="The parquet halves (both are scored per site)")
    parser.add_argument("--fed", nargs="+", required=True, metavar="LABEL=PATH",
                        help="Federated models, e.g. bagging=../out/models/bagging.json")
    parser.add_argument("--max-auc-delta", type=float, default=0.03,
                        help="Per-site bagging tolerance (default 0.03 PROVISIONAL "
                             "until calibrated; see --calibrate)")
    parser.add_argument("--allow-pyproject-fallback", action="store_true",
                        help="Permit models with no embedded fed_run_config (params "
                             "provenance unverified). Default: hard fail.")
    parser.add_argument("--calibrate", type=int, default=None, metavar="N",
                        help="Train pooled over N seeds, print no-bug delta spread, "
                             "exit 0 without gating")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR,
                        help="Where fed_vs_pooled.{json,md} is written "
                             "(default out/metrics)")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH,
                        help="pyproject.toml for the fallback config reader "
                             "(only read when a model lacks fed_run_config)")
    # Scoring-only knobs (do not affect the trained model) mirror pyproject.
    parser.add_argument("--operating-point", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()

    fed_paths = _parse_fed(args.fed)

    # 1. Load every model; read its embedded config.
    fed_models = {}
    configs = []
    provenances = {}
    for label, path in fed_paths.items():
        bst = _load_model(path)
        params, total_trees, prov = read_model_config(bst, args.pyproject)
        fed_models[label] = bst
        configs.append((params, total_trees))
        provenances[label] = prov

    fallbacks = [lbl for lbl, p in provenances.items() if p == "pyproject-fallback"]
    if fallbacks and not args.allow_pyproject_fallback:
        print(f"FAIL: params provenance unverified for {fallbacks} (no "
              f"fed_run_config attribute). Re-run the federation after the "
              f"server_app embed, or pass --allow-pyproject-fallback.",
              file=sys.stderr)
        return 1
    provenance = "pyproject-fallback" if fallbacks else "model"

    # 2. Matched-budget invariant across all models.
    try:
        params, total_trees = assert_matched_across_models(configs)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    # --calibrate short-circuits: no gating, exit 0.
    if args.calibrate is not None:
        return _run_calibrate(args, params, total_trees, fed_models)

    # 3. Train the pooled reference (deterministic: pinned seed + nthread=1).
    if "seed" not in params:
        print("FAIL: matched params carry no pinned `seed`; pooled reference "
              "would be nondeterministic. Add params.seed to the config and "
              "re-run the federation.", file=sys.stderr)
        return 1
    pooled = train_pooled_booster(args.data, params, total_trees)
    pooled_trees = pooled.num_boosted_rounds()
    if pooled_trees != total_trees:
        print(f"FAIL: pooled model has {pooled_trees} trees, expected "
              f"{total_trees}.", file=sys.stderr)
        return 1

    # 4. Serialization round-trip tripwire (independent of AUC deltas).
    valid_dms = {Path(p).name: _valid_dmatrix(p) for p in args.data}
    roundtrip_ok = True
    roundtrip_errors = []
    for label, bst in [("pooled", pooled), *fed_models.items()]:
        for site, dm in valid_dms.items():
            try:
                assert_prediction_roundtrip(bst, dm)
            except ValueError as exc:
                roundtrip_ok = False
                roundtrip_errors.append(f"{label}@{site}: {exc}")

    # 5. Score the pooled reference per site (+ informational combined).
    pooled_by_site = {
        Path(p).name: score_booster_on_half(
            pooled, p, args.operating_point, args.n_boot, args.boot_seed
        )
        for p in args.data
    }
    pooled_combined = _score_combined(
        pooled, args.data, args.operating_point, args.n_boot, args.boot_seed
    )

    # 6. Score each federated model per site and compute the gate.
    federated_by_site = {}
    gate = {}
    tree_mismatch = []
    for label, bst in fed_models.items():
        n = bst.num_boosted_rounds()
        if n != total_trees:
            tree_mismatch.append(f"{label}: {n} trees, expected {total_trees}")
        fed_by_site = {
            Path(p).name: score_booster_on_half(
                bst, p, args.operating_point, args.n_boot, args.boot_seed
            )
            for p in args.data
        }
        federated_by_site[label] = fed_by_site
        result = compare_auc(pooled_by_site, fed_by_site, args.max_auc_delta)
        gate[label] = {"gating": _gates(label), "overall": result["overall"],
                       "sites": result["sites"]}

    # 7. Print the per-site table.
    print(f"\nFederated-vs-pooled check (max_auc_delta={args.max_auc_delta}, "
          f"total_trees={total_trees}, provenance={provenance})")
    print(f"pooled-combined AUC-ROC (informational) = {_fmt(pooled_combined['auc_roc'])}")
    nan_sites = []
    for label, g in gate.items():
        kind = "gating" if g["gating"] else "informational"
        print(f"\n=== {label} ({kind}) ===")
        print(f"{'site':<26} {'pooled':>8} {'fed':>8} {'delta':>8}  status")
        for site, s in g["sites"].items():
            nan = _is_nan(s["pooled"]) or _is_nan(s["federated"])
            if nan:
                nan_sites.append(f"{label}@{site}")
                status = "NaN"
            elif g["gating"]:
                status = "PASS" if s["passed"] else "FAIL"
            else:
                status = "INFO" if s["passed"] else "INFO(>tol)"
            print(f"{site:<26} {_fmt(s['pooled']):>8} {_fmt(s['federated']):>8} "
                  f"{_fmt(s['delta']):>8}  {status}")

    # 8. Decide the exit code.
    bagging_fail = any(
        (not s["passed"]) and not (_is_nan(s["pooled"]) or _is_nan(s["federated"]))
        for label, g in gate.items() if g["gating"]
        for s in g["sites"].values()
    )
    nan_fail = bool(nan_sites)
    hard_fail = bagging_fail or nan_fail or (not roundtrip_ok) or bool(tree_mismatch)
    passed = not hard_fail

    # Artifact.
    artifact = {
        "max_auc_delta": args.max_auc_delta,
        "total_trees": total_trees,
        "params": params,
        "config_provenance": provenance,
        "pooled": pooled_by_site,
        "pooled_combined": pooled_combined,
        "federated": federated_by_site,
        "gate": {
            label: {"gating": g["gating"], "overall": g["overall"], **g["sites"]}
            for label, g in gate.items()
        },
        "roundtrip_ok": roundtrip_ok,
        "passed": passed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "fed_vs_pooled.json"
    json_path.write_text(json.dumps(_json_safe(artifact), indent=2, allow_nan=False))
    (args.out_dir / "fed_vs_pooled.md").write_text(_render_md(artifact))
    print(f"\nWrote {json_path}")

    # Diagnostics + verdict.
    if tree_mismatch:
        print(f"FAIL (tree budget): {tree_mismatch}", file=sys.stderr)
    if not roundtrip_ok:
        print(f"FAIL (round-trip): {roundtrip_errors}", file=sys.stderr)
    if nan_fail:
        print(f"FAIL (NaN AUC, hard fail even for cyclic): {nan_sites}",
              file=sys.stderr)
    if bagging_fail:
        print("FAIL (bagging gate): a bagging per-site delta exceeds "
              f"{args.max_auc_delta}", file=sys.stderr)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _render_md(artifact) -> str:
    """Human-readable table mirroring the JSON (JSON stays source of truth)."""
    lines = [
        "# Federated-vs-pooled check (roadmap 1.d)",
        "",
        f"- max_auc_delta: `{artifact['max_auc_delta']}`",
        f"- total_trees: `{artifact['total_trees']}`",
        f"- config_provenance: `{artifact['config_provenance']}`",
        f"- roundtrip_ok: `{artifact['roundtrip_ok']}`",
        f"- **passed: `{artifact['passed']}`**",
        "",
    ]
    for label, g in artifact["gate"].items():
        kind = "gating" if g["gating"] else "informational (not gated)"
        lines.append(f"## {label} ({kind})")
        lines.append("")
        lines.append("| site | pooled AUC | fed AUC | delta | status |")
        lines.append("|------|-----------|---------|-------|--------|")
        for site in [k for k in g if k not in ("gating", "overall")]:
            s = g[site]
            nan = _is_nan(s["pooled"]) or _is_nan(s["federated"])
            if nan:
                status = "NaN"
            elif g["gating"]:
                status = "PASS" if s["passed"] else "FAIL"
            else:
                status = "INFO" if s["passed"] else "INFO(>tol)"
            pooled = "NaN" if _is_nan(s["pooled"]) else f"{s['pooled']:.4f}"
            fed = "NaN" if _is_nan(s["federated"]) else f"{s['federated']:.4f}"
            delta = "NaN" if _is_nan(s["delta"]) else f"{s['delta']:.4f}"
            lines.append(f"| {site} | {pooled} | {fed} | {delta} | {status} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
