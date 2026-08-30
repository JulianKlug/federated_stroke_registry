"""DP sweep driver (roadmap 1.1.b): the A→B→C comparator sweep at ε ∈ {1,3,5,10}.

One of the two components that touch the process boundary (the other is
`run_hpo.py`; both share `scripts/flwr_proc.py`): it fires the six matched
federated runs — arm A (stock XGBoost), arm B (DP learner, identity), arm C (DP
learner, gaussian) at each ε — on ONE shared config against a standing
federation, re-scores every saved model on both halves with bootstrap CIs,
extracts the accounting from the models and the ledger, and emits the filled
1.1.b report (`report_1_1_b.md`) plus `results.json`.

All pure logic (arm plan, run-config construction, assertions, deltas, report
rendering) lives in `fed_stroke.dpsweep`; this file only orchestrates.
Spec: docs/specs/1_1_b_dp_sweep_driver.md.

Runs against a STANDING federation (bring it up with
`scripts/run_local_federation.sh start`); this driver never manages its
lifecycle. Real-patient-data runs stay blocked until the R8 packet re-review is
signed off in docs/logbook.md — the `--gate-ack` flag stamps the operator's
affirmation of that sign-off into the artifacts (the driver cannot verify it).

Examples (run from anywhere):
    python scripts/run_dp_sweep.py --dry-run
    python scripts/run_dp_sweep.py --out-dir out/dp_sweep_rehearsal
    python scripts/run_dp_sweep.py --data-provenance real-frozen-schema \\
        --config out/hpo/tuned_bagging.toml --operator <name> \\
        --gate-ack "R8 sign-off recorded in docs/logbook.md YYYY-MM-DD"
"""
import argparse
import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `fed_stroke` (and the scripts-dir commons) importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flwr_proc import (  # noqa: E402
    REPO_ROOT,
    _half_paths,
    _json_safe,
    _load_pyproject,
    _node_ledger_path,
    _node_provenance,
    _operating_point,
    _run_flwr,
    preflight_superlink,
)

from fed_stroke import dpsweep  # noqa: E402
from fed_stroke.baseline import load_saved_booster, score_booster_on_half  # noqa: E402
from fed_stroke.dp import per_site_tree_budget  # noqa: E402
from fed_stroke.dp import ledger as dp_ledger  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--federation", default="local-deployment",
                   help="flwr federation to drive (swap for the cross-site link in v1.3)")
    p.add_argument("--strategy", choices=["bagging", "cyclic"], default="bagging",
                   help="bagging is the 1.1.b deliverable; cyclic is REFUSED on "
                        "real-frozen-schema data (only the round-1 site would ledger its "
                        "spend — reviewer A C2) and stays runnable as a rehearsal")
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0],
                   help="pilot grid (roadmap contract: 1 3 5 10); run DESCENDING "
                        "regardless of input order")
    p.add_argument("--delta", type=float, default=1e-5,
                   help="δ threaded into both the C run configs and ledger_total()")
    p.add_argument("--config", type=Path, default=None,
                   help="optional tuned TOML (e.g. tuned_bagging.toml from 1.1.a); "
                        "adopts total-trees/max-depth/eta/min-child-weight; its "
                        "subsample/colsample are discarded (forced 1.0)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="one seed for the runs AND the offline scoring (byte-identical "
                        "row sets across all six arms)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="offline bootstrap resamples for the 95%% CIs (terminal "
                        "scoring pass, not a search loop)")
    p.add_argument("--out-dir", default="out/dp_sweep",
                   help="artifact dir; relative resolves against the repo root; "
                        "per-arm subdirs inside")
    p.add_argument("--data-provenance", choices=["example-halves", "real-frozen-schema"],
                   default="example-halves",
                   help="must AGREE with the nodes' own declaration in pyproject "
                        "[tool.fed_stroke.nodes] (node-owned; the client refuses a "
                        "mismatch); real-frozen-schema requires --operator + --gate-ack")
    p.add_argument("--operator", default=None,
                   help="required for real provenance; defaults to the login user "
                        "on example-halves")
    p.add_argument("--gate-ack", default=None,
                   help="required for real provenance: an affirmative statement that "
                        "the R8 sign-off is recorded in docs/logbook.md; stamped "
                        "verbatim into the report and results.json")
    p.add_argument("--run-timeout", type=float, default=900,
                   help="per-run wall-clock cap (s); an exceeding run is killed")
    p.add_argument("--resume", action="store_true",
                   help="skip arms whose final_model.json exists (re-score only); a "
                        "MISSING model re-fires the run — for C arms this appends "
                        "the ledger again (intent-to-spend, by design)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the run plan with the exact run-config strings, "
                        "launch nothing")
    return p.parse_args()


def _resolve(raw):
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def main() -> None:
    args = _parse_args()
    cfg = _load_pyproject()

    # Step 0 — absolute paths + pyproject facts + the real-provenance guard.
    out_dir = _resolve(args.out_dir)
    # Node-owned DP-rail facts (reviewer A C1 / B F8): the SuperNodes write the ledger at
    # THEIR configured path and gate on THEIR provenance; the driver reads the same file
    # and must agree on provenance, or the client refuses every DP run.
    ledger_path = _node_ledger_path(cfg)
    node_prov = _node_provenance(cfg)
    if node_prov != args.data_provenance:
        raise SystemExit(
            f"--data-provenance {args.data_provenance!r} disagrees with the nodes' own "
            f"declaration {node_prov!r} (pyproject [tool.fed_stroke.nodes]). The node "
            "value is authoritative; the client would refuse every DP run.")
    half_paths = _half_paths(cfg)
    operating_point = _operating_point(cfg)
    app_cfg = cfg["tool"]["flwr"]["app"]["config"]
    num_sites = int(app_cfg["num-sites"])
    local_epochs = int(app_cfg["local-epochs"])
    node_labels = {Path(node["data-path"]).name: key
                   for key, node in cfg["tool"]["fed_stroke"]["nodes"].items()}
    expected_sites = sorted(node_labels)

    real = args.data_provenance == "real-frozen-schema"
    if real and (not args.operator or not args.gate_ack):
        raise SystemExit(
            "--data-provenance real-frozen-schema requires --operator AND "
            "--gate-ack (an affirmative statement that the R8 sign-off is "
            "recorded in docs/logbook.md). The driver cannot verify the "
            "sign-off; refusing to start without the acknowledgement.")
    operator = args.operator or getpass.getuser()
    if real and args.strategy == "cyclic":
        # A C2: cyclic ledgers only the round-1 site; the client refuses it on real data,
        # so refuse here before firing anything.
        raise SystemExit("--strategy cyclic is refused on real-frozen-schema data: only "
                         "the round-1 site would ledger its spend (cyclic round-1 append "
                         "gap). Run the go/no-go sweep on bagging.")

    # Step 1 — shared config, arm plan, divisibility; --dry-run exits here.
    tuned_table = None
    if args.config is not None:
        tuned_table = dpsweep.parse_tuned_table(args.config.read_text())
    shared, notes = dpsweep.resolve_shared_config(app_cfg, tuned_table)
    for note in notes:
        print(f"[config] {note}")
    plan = dpsweep.arm_plan(args.epsilons, args.delta)
    num_rounds = dpsweep.validate_divisibility(
        args.strategy, shared["total-trees"], num_sites, local_epochs)
    per_site = per_site_tree_budget(args.strategy, num_rounds, num_sites,
                                    local_epochs)
    run_cfgs = {
        arm["label"]: dpsweep.build_arm_run_config(
            arm, shared, args.strategy, args.split_seed, out_dir,
            args.data_provenance)
        for arm in plan
    }
    print(f"[plan] strategy={args.strategy} federation={args.federation} "
          f"provenance={args.data_provenance} arms={len(plan)} "
          f"epsilons={[a['epsilon'] for a in plan if a['arm'] == dpsweep.ARM_C]} "
          f"delta={args.delta:g} out_dir={out_dir}")

    if args.dry_run:
        for arm in plan:
            print(f"[dry-run] {arm['label']}: {run_cfgs[arm['label']]}")
        print(f"[dry-run] would fire {len(plan)} runs; launching nothing.")
        return

    # Step 2 — preflight + ledger snapshot.
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_superlink(cfg)
    entries_before = dp_ledger.read_entries(ledger_path)

    # Step 3 — the run matrix, sequential, in plan order (A first: every
    # infrastructure failure mode surfaces before any privacy budget is spent).
    arms_results = []
    entries_seen = list(entries_before)
    per_run_epsilon = {}  # {site: {label: per-run ε}} from this sweep's entries
    for arm in plan:
        label = arm["label"]
        model_path = out_dir / label / "final_model.json"
        fired = False
        if args.resume and model_path.exists():
            print(f"[{label}] resume (skip run; re-scoring existing model)")
        else:
            if args.resume and arm["arm"] == dpsweep.ARM_C and real:
                print(f"[{label}] resume: model missing — re-firing; this appends "
                      "the ledger AGAIN (intent-to-spend, both entries compose)")
            print(f"[{label}] flwr run ({args.federation})")
            ok, _ = _run_flwr(args.federation, run_cfgs[label], args.run_timeout)
            fired = True
        degenerate = fired and (not ok or not model_path.exists())
        if degenerate and arm["arm"] != dpsweep.ARM_C:
            # A is the built-in canary; B still spends nothing — abort before any
            # ε is spent rather than run C arms against a missing comparator.
            raise SystemExit(f"[{label}] run failed/timed out/missing model — "
                             "aborting the sweep (nothing spent yet). Check "
                             "scripts/run_local_federation.sh status.")

        entries_now = dp_ledger.read_entries(ledger_path)
        run_delta = dpsweep.ledger_delta(entries_seen, entries_now)
        entries_seen = entries_now

        arm_result = {
            "arm": arm["arm"], "label": label, "epsilon": arm["epsilon"],
            "run_config": run_cfgs[label], "model_path": str(model_path),
            "degenerate": degenerate, "fmt": None, "n_trees": None,
            "dp_meta": None, "config_hash": None,
            "metrics": {h.name: None for h in half_paths},
        }

        if degenerate:
            print(f"[{label}] DEGENERATE (run failed/timed out/missing model); "
                  "continuing")
            if run_delta:
                # Intent-to-spend: the entry lands on round 1 even when the model
                # never does. It composes; the report shows it.
                print(f"[{label}] NOTE: {len(run_delta)} ledger entries landed "
                      "for the failed run — they still compose (R6)")
                arm_result["config_hash"] = run_delta[0]["config_hash"]
                for e in run_delta:
                    per_run_epsilon.setdefault(e["site"], {})[label] = float(e["epsilon"])
            arms_results.append(arm_result)
            continue

        loaded = load_saved_booster(model_path)
        arm_result["fmt"] = loaded.fmt
        arm_result["n_trees"] = loaded.n_trees
        problems = dpsweep.check_arm_model(arm, loaded, shared["total-trees"],
                                           per_site)
        if problems:
            raise SystemExit(f"[{label}] post-run assertions failed:\n  "
                             + "\n  ".join(problems))
        if arm["arm"] != dpsweep.ARM_A:
            arm_result["dp_meta"] = dpsweep.extract_dp_meta(loaded.booster)

        # Ledger assertions for THIS run (strategy-aware, §4.4). A resumed arm
        # fired nothing, so it must have appended nothing.
        c_runs = []
        if fired and arm["arm"] == dpsweep.ARM_C:
            c_runs = [{"label": label,
                       "epsilon": arm_result["dp_meta"]["epsilon"],
                       "sigma": arm_result["dp_meta"]["sigma"]}]
        issues = dpsweep.validate_sweep_ledger(run_delta, c_runs,
                                               args.data_provenance, args.strategy)
        if issues:
            raise SystemExit(f"[{label}] ledger assertions failed:\n  "
                             + "\n  ".join(issues))
        if run_delta:
            arm_result["config_hash"] = run_delta[0]["config_hash"]
            for e in run_delta:
                per_run_epsilon.setdefault(e["site"], {})[label] = float(e["epsilon"])

        # Offline re-scoring on BOTH halves through the single shared scorer.
        for half in half_paths:
            m = score_booster_on_half(
                loaded.booster, half, operating_point,
                n_boot=args.n_boot, boot_seed=0,
                split_seed=args.split_seed, holdout_frac=0.0, holdout_eval=False)
            arm_result["metrics"][half.name] = m
            print(f"[{label}] {half.name}: auc_roc={m['auc_roc']:.4f} "
                  f"auc_pr={m['auc_pr']:.4f} brier={m['brier']:.4f}")
        arms_results.append(arm_result)

    # Step 4 — accounting: sweep-wide ledger delta + composition (the ONLY
    # composer is ledger_total; R6).
    entries_after = dp_ledger.read_entries(ledger_path)
    sweep_delta = dpsweep.ledger_delta(entries_before, entries_after)
    totals = dp_ledger.ledger_total(ledger_path, delta=args.delta)
    unledgered = dpsweep.unledgered_sites(expected_sites, totals) if real else []

    # Step 5 — deltas + artifacts.
    half_names = [h.name for h in half_paths]
    deltas = dpsweep.compute_deltas(arms_results, half_names)
    cohort = dpsweep.cohort_stats(half_paths, args.split_seed)
    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operator": operator,
        "gate_ack": args.gate_ack,
        "data_provenance": args.data_provenance,
        "federation": args.federation,
        "strategy": args.strategy,
        "split_seed": args.split_seed,
        "delta": args.delta,
        "epsilons": sorted((float(e) for e in args.epsilons), reverse=True),
        "n_boot": args.n_boot,
        "shared_config": shared,
        "tuned_config_path": str(args.config) if args.config else None,
        "config_notes": notes,
        "num_rounds": num_rounds,
        "per_site_tree_budget": per_site,
    }
    ledger_block = {
        "path": str(ledger_path),
        "entries_added": sweep_delta,
        "ledger_total": totals,
        "unledgered_sites": unledgered,
        "site_labels": node_labels,
        "expected_sites": expected_sites,
        "per_run_epsilon": per_run_epsilon,
    }
    results = dpsweep.build_results(meta, cohort, arms_results, deltas,
                                    ledger_block)
    (out_dir / "results.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False) + "\n")
    (out_dir / "report_1_1_b.md").write_text(dpsweep.render_report(results))
    print(f"[done] wrote results.json, report_1_1_b.md under {out_dir}")
    if unledgered:
        print(f"[warn] UNLEDGERED sites (spent ε, never appended — cyclic "
              f"round-1 gap): {', '.join(unledgered)}")


if __name__ == "__main__":
    main()
