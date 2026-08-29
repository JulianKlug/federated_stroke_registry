"""Systematic HPO harness driver (roadmap 1.1.a).

One of the two components that touch the process boundary (the other is
`run_dp_sweep.py`, roadmap 1.1.b — both share `scripts/flwr_proc.py`): it drives
the real federated pipeline by repeatedly firing `flwr run . <federation>` with
per-trial `--run-config` overrides, scores each trial's saved model offline on both
Geneva halves, selects a winner by mean site AUC-ROC (tie-break lower cross-site
variance), re-runs the winner once against a patient-disjoint hold-out, and emits
four artifacts (leaderboard, results JSON, narrowed ranges, tuned-config TOML).

All pure logic (grid, objective, ranking, artifact shaping) lives in
`fed_stroke.hpo`; this file only orchestrates. See docs/specs/1_1_a_hpo_harness.md §4.4.

Runs against a STANDING federation (bring it up with
`scripts/run_local_federation.sh start`); this driver never manages its lifecycle.

Examples (run from architecture/):
    python scripts/run_hpo.py --strategy bagging --dry-run
    python scripts/run_hpo.py --strategy bagging --max-trials 4 --search-seeds 1 2 3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import xgboost as xgb

# Make `fed_stroke` (and the scripts-dir commons) importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flwr_proc import (  # noqa: E402
    REPO_ROOT,
    _half_paths,
    _json_safe,
    _load_pyproject,
    _operating_point,
    _run_flwr,
    preflight_superlink,
)

from fed_stroke import hpo  # noqa: E402
from fed_stroke.baseline import score_booster_on_half  # noqa: E402
from fed_stroke.task import HOLDOUT_PARTITION_SEED  # noqa: E402


# scoring: load a saved model, score both halves offline (§3.4)
# --------------------------------------------------------------------------- #
def _load_booster(model_path):
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    bst.set_param({"eval_metric": "auc"})
    return bst


def _site_aucs(model_path, half_paths, operating_point, split_seed, holdout_frac,
               n_boot, holdout_eval=False):
    """Score a saved model on every half; return {half_name: auc_or_None}. A NaN AUC
    (single-class split) is recorded as None (the §4.3 convention).
    """
    bst = _load_booster(model_path)
    aucs = {}
    for half in half_paths:
        m = score_booster_on_half(
            bst, half, operating_point, n_boot=n_boot, boot_seed=0,
            split_seed=split_seed, holdout_frac=holdout_frac, holdout_eval=holdout_eval,
        )
        auc = m["auc_roc"]
        aucs[half.name] = None if (isinstance(auc, float) and math.isnan(auc)) else float(auc)
    return aucs


def _full_site_metrics(model_path, half_paths, operating_point, holdout_frac):
    """Full compute_binary_metrics dict per half on the HELD set, with bootstrap CIs
    (n_boot=1000) — the honest hold-out numbers (§4.4 step 4)."""
    bst = _load_booster(model_path)
    out = {}
    for half in half_paths:
        out[half.name] = score_booster_on_half(
            bst, half, operating_point, n_boot=1000, boot_seed=0,
            split_seed=42, holdout_frac=holdout_frac, holdout_eval=True,
        )
    return out


# --------------------------------------------------------------------------- #
# artifact rendering (narrowed ranges + tuned TOML; leaderboard is in hpo.py)
# --------------------------------------------------------------------------- #
def _render_narrowed_ranges(results, ranked):
    meta = results["meta"]
    sig = results["signal"]
    lines = [
        f"# Narrowed hyperparameter ranges — {meta['strategy']} "
        f"({meta['federation']})",
        "",
        f"Provenance: `{meta['data_provenance']}` · search seeds "
        f"{meta['search_seeds']} · holdout-frac {meta['holdout_frac']}",
        "",
    ]
    noise = (sig["winner_vs_runnerup_ci_overlap"]
             or (sig["top_minus_median"] is not None
                 and sig["top_minus_median"] < 0.01))
    if noise:
        lines += [
            "> **⚠ WARNING — NOISE-TIER RANKING.** The winner's objective spread "
            "overlaps the runner-up's and/or top≈median "
            f"(top {hpo._fmt(sig['top_objective'])}, median "
            f"{hpo._fmt(sig['median_objective'])}, gap "
            f"{hpo._fmt(sig['top_minus_median'])}). Do NOT read the 'winner' as a "
            "real optimum — over 2 features and ~1k rows the AUC gaps can be smaller "
            "than MC-CV split noise (§8). These ranges are provisional regardless "
            "(re-tuned cross-site in v1.3).",
            "",
        ]
    ranges = hpo.summarize_top_ranges(ranked)
    if not ranges:
        lines.append("_No valid trials — no ranges to narrow._\n")
        return "\n".join(lines)
    lines += [
        f"Top-decile valid trials (n={ranges['n_top']}):",
        "",
        "| knob | min | max | mode | values |",
        "|---|---|---|---|---|",
    ]
    for knob in hpo.KNOBS:
        r = ranges[knob]
        vals = ", ".join(str(v) for v in r["values"])
        lines.append(f"| `{knob}` | {r['min']} | {r['max']} | {r['mode']} | {vals} |")
    return "\n".join(lines) + "\n"


def _render_tuned_toml(results):
    meta = results["meta"]
    winner = results["winner"]
    hold = results["holdout"]
    p = winner["params"]
    lines = [
        f"# Tuned config — {meta['strategy']} (PROVISIONAL, Geneva first-pass)",
        f"# data_provenance = {meta['data_provenance']}",
    ]
    if meta["data_provenance"] != "real-frozen-schema":
        lines.append("#   ^ THROWAWAY plumbing proof — NOT the deliverable ranges "
                     "(spec §4.7, Decision 13). Only a real-frozen-schema run "
                     "feeds 1.1.b / the v1.3 re-run.")
    lines += [
        f"# search_seeds = {meta['search_seeds']}",
        f"# holdout_frac = {meta['holdout_frac']}, "
        f"holdout_partition_seed = {meta['holdout_partition_seed']}",
        "# hold-out (patient-disjoint HELD) per-site AUC-ROC (95% CI):",
    ]
    for site, m in hold["site_metrics"].items():
        lines.append(
            f"#   {site}: {hpo._fmt(m.get('auc_roc'))} "
            f"[{hpo._fmt(m.get('auc_roc_lo'))}, {hpo._fmt(m.get('auc_roc_hi'))}]"
        )
    lines += [
        "",
        "[tool.flwr.app.config]",
        f"total-trees = {p['total-trees']}",
        f"params.max-depth = {p['params.max-depth']}",
        f"params.eta = {p['params.eta']}",
        f"params.min-child-weight = {p['params.min-child-weight']}",
        f"params.subsample = {p['params.subsample']}",
        f"params.colsample-bytree = {p['params.colsample-bytree']}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--federation", default="local-deployment",
                   help="flwr federation to drive (swap for the cross-site link in v1.3)")
    p.add_argument("--strategy", choices=["bagging", "cyclic"], default="bagging",
                   help="aggregation strategy (bagging is the 1.1.a headline)")
    p.add_argument("--search-seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="repeated-CV seeds (train/valid sub-split of DEV)")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="patient-disjoint hold-out fraction; must be in (0, 1)")
    p.add_argument("--grid", type=Path, default=None,
                   help="JSON grid file; defaults to hpo.DEFAULT_GRID")
    p.add_argument("--max-trials", type=int, default=0,
                   help="0 = full grid; >0 = random subsample (escape hatch)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the --max-trials subsample (reproducible)")
    p.add_argument("--n-boot", type=int, default=0,
                   help="OFFLINE bootstrap resamples during search (0 = skip, fast)")
    p.add_argument("--run-timeout", type=float, default=900,
                   help="per-run wall-clock cap (s); an exceeding run is killed")
    p.add_argument("--out-dir", default="out/hpo",
                   help="artifact dir; relative resolves against the repo root")
    p.add_argument("--data-provenance", choices=["example-halves", "real-frozen-schema"],
                   default="example-halves",
                   help="stamped into every artifact; only real-frozen-schema is the "
                        "deliverable hand-off (§4.7)")
    p.add_argument("--resume", action="store_true",
                   help="skip any (trial, seed) whose final_model.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="print the trial plan + run count, launch nothing")
    return p.parse_args()


def _resolve_out_dir(raw):
    out = Path(raw)
    if not out.is_absolute():
        out = REPO_ROOT / out
    return out.resolve()


def _load_grid(args):
    if args.grid is None:
        return dict(hpo.DEFAULT_GRID)
    with open(args.grid) as fh:
        return json.load(fh)


def main() -> None:
    args = _parse_args()
    cfg = _load_pyproject()

    # Step 0 — every path the driver hands a run is ABSOLUTE (Decision 7).
    out_dir = _resolve_out_dir(args.out_dir)
    half_paths = _half_paths(cfg)
    operating_point = _operating_point(cfg)
    seeds = list(args.search_seeds)
    strategy = args.strategy

    # Step 1 — preflight.
    app_cfg = cfg["tool"]["flwr"]["app"]["config"]
    num_sites = int(app_cfg["num-sites"])
    local_epochs = int(app_cfg["local-epochs"])
    grid = _load_grid(args)
    trials = hpo.expand_grid(grid)
    runnable, rejected = hpo.valid_trials(trials, strategy, num_sites, local_epochs)
    runnable = hpo.subsample_trials(runnable, args.max_trials, args.seed)

    for r in rejected:
        print(f"[reject] {r['trial']}: {r['reason']}")
    n_runs = len(runnable) * len(seeds)
    print(f"[plan] strategy={strategy} federation={args.federation} "
          f"grid_cells={len(trials)} runnable={len(runnable)} "
          f"rejected={len(rejected)} seeds={seeds} search_runs={n_runs} "
          f"out_dir={out_dir}")

    if args.dry_run:
        print(f"[dry-run] would fire {n_runs} search runs "
              f"(+1 winner hold-out); launching nothing.")
        return

    if not 0.0 < args.holdout_frac < 1.0:
        raise SystemExit(f"--holdout-frac must be in (0, 1); got {args.holdout_frac} "
                         "(a search needs a non-empty DEV and a non-empty HELD)")
    if not runnable:
        raise SystemExit("no runnable trials after divisibility check + subsample")

    out_dir.mkdir(parents=True, exist_ok=True)

    # TCP probe proves the SuperLink listener is up, but NOT that both SuperNodes are
    # alive/pinned — so a canary run follows.
    preflight_superlink(cfg)

    canary = min(runnable, key=lambda t: t["total-trees"])
    canary_dir = out_dir / "_canary"
    print(f"[canary] firing smallest trial {hpo.trial_id(canary)} at seed {seeds[0]}")
    canary_cfg = hpo.build_run_config(
        canary, strategy, seeds[0], canary_dir, canary_dir,
        holdout_frac=args.holdout_frac, holdout_eval=False, n_boot=0)
    ok, _ = _run_flwr(args.federation, canary_cfg, args.run_timeout)
    if not ok or not (canary_dir / "final_model.json").exists():
        raise SystemExit(
            "canary run failed/timed out — the node topology looks dead or "
            "mis-pinned. Check scripts/run_local_federation.sh status/start "
            "before launching the sweep.")
    print("[canary] OK — topology live; starting sweep")

    # Step 2 — search.
    results = []
    for ti, override in enumerate(runnable, start=1):
        tid = hpo.trial_id(override)
        per_repeat = []
        for seed in seeds:
            seed_dir = hpo.trial_seed_dir(out_dir, tid, seed)
            model_path = hpo.final_model_path(out_dir, tid, seed)
            if args.resume and hpo.is_completed(out_dir, tid, seed):
                print(f"[{ti}/{len(runnable)}] {tid} seed{seed}: resume (skip run)")
            else:
                run_cfg = hpo.build_run_config(
                    override, strategy, seed, seed_dir, seed_dir,
                    holdout_frac=args.holdout_frac, holdout_eval=False, n_boot=0)
                ok, _ = _run_flwr(args.federation, run_cfg, args.run_timeout)
                if not ok or not model_path.exists():
                    print(f"[{ti}/{len(runnable)}] {tid} seed{seed}: DEGENERATE "
                          "(run failed/timed out/missing model)")
                    per_repeat.append({"split_seed": seed,
                                       "site_aucs": {h.name: None for h in half_paths}})
                    continue
            aucs = _site_aucs(model_path, half_paths, operating_point,
                              split_seed=seed, holdout_frac=args.holdout_frac,
                              n_boot=args.n_boot)
            per_repeat.append({"split_seed": seed, "site_aucs": aucs})
            print(f"[{ti}/{len(runnable)}] {tid} seed{seed}: {aucs}")

        trial = {"trial_id": tid, "params": override, "per_repeat": per_repeat}
        trial.update(hpo.trial_objective(per_repeat, len(seeds)))
        results.append(trial)

    # Step 3 — select.
    ranked = hpo.rank_trials(results)
    winner = hpo.select_winner(ranked)
    if winner is None:
        raise SystemExit("every trial is invalid (all-degenerate sweep) — no winner. "
                         "See the leaderboard for the ⚠-flagged trials.")
    print(f"[winner] {winner['trial_id']} objective={winner['objective']:.4f} "
          f"variance={winner['variance']:.6f} params={winner['params']}")

    # Step 4 — hold-out (train on DEV, evaluate on the disjoint HELD set).
    hold_dir = out_dir / "holdout"
    hold_cfg = hpo.build_run_config(
        winner["params"], strategy, seeds[0], hold_dir, hold_dir,
        holdout_frac=args.holdout_frac, holdout_eval=True, n_boot=0)
    ok, _ = _run_flwr(args.federation, hold_cfg, args.run_timeout)
    hold_model = hold_dir / "final_model.json"
    if not ok or not hold_model.exists():
        raise SystemExit("winner hold-out run failed — cannot report the "
                         "patient-disjoint baseline.")
    site_metrics = _full_site_metrics(hold_model, half_paths, operating_point,
                                      args.holdout_frac)
    print("[holdout] per-site AUC-ROC (HELD):")
    for site, m in site_metrics.items():
        print(f"    {site}: {hpo._fmt(m.get('auc_roc'))} "
              f"[{hpo._fmt(m.get('auc_roc_lo'))}, {hpo._fmt(m.get('auc_roc_hi'))}]")

    # Step 5 — write artifacts.
    meta = {
        "strategy": strategy,
        "federation": args.federation,
        "search_seeds": seeds,
        "holdout_frac": args.holdout_frac,
        "holdout_partition_seed": HOLDOUT_PARTITION_SEED,
        "operating_point": operating_point,
        "data_provenance": args.data_provenance,
        "grid": grid,
    }
    holdout = {
        "params": winner["params"],
        "holdout_frac": args.holdout_frac,
        "holdout_partition_seed": HOLDOUT_PARTITION_SEED,
        "site_metrics": site_metrics,
    }
    results_artifact = hpo.build_results(meta, ranked, holdout)
    hpo.validate_results(results_artifact)

    (out_dir / "results.json").write_text(
        json.dumps(_json_safe(results_artifact), indent=2, allow_nan=False) + "\n")
    (out_dir / "leaderboard.md").write_text(hpo.render_leaderboard(results_artifact))
    (out_dir / "narrowed_ranges.md").write_text(
        _render_narrowed_ranges(results_artifact, ranked))
    (out_dir / f"tuned_{strategy}.toml").write_text(
        _render_tuned_toml(results_artifact))
    print(f"[done] wrote results.json, leaderboard.md, narrowed_ranges.md, "
          f"tuned_{strategy}.toml under {out_dir}")


if __name__ == "__main__":
    main()
