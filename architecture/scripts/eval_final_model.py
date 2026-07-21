"""Score a saved federated XGBoost model on each Geneva half's validation split.

Debugging-grade sanity tool for roadmap 1.b (local Geneva data access is allowed
for debugging). In-run cyclic AUC is a self-evaluation on alternating sites, so
the only way to compare R1/R2/R3 for last-site bias is to score each *saved*
final model on the *same two* validation sets — which this script does.

Usage:
    python scripts/eval_final_model.py final_model.json \
        --data ../out/geneva_half_A.parquet ../out/geneva_half_B.parquet \
        --expected-trees 40
"""
import argparse
import sys
from pathlib import Path

import xgboost as xgb

# Make `fed_stroke` importable regardless of CWD (script lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fed_stroke.baseline import score_booster_on_half  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to saved final_model.json")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="One or more parquet halves to score the model on",
    )
    parser.add_argument(
        "--expected-trees",
        type=int,
        required=True,
        help="Assert the model has exactly this many boosted rounds (budget check)",
    )
    # Defaults mirror pyproject config (§4.5) so numbers match the FL run exactly.
    parser.add_argument("--operating-point", type=float, default=0.5,
                        help="Shared fixed confusion-matrix threshold (config default 0.5)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for the 95%% CIs (config default 1000)")
    parser.add_argument("--boot-seed", type=int, default=0,
                        help="RNG seed for reproducible bootstrap CIs (config default 0)")
    # Split-reconstruction flags (spec 1.1.a §4.5/§5): reproduce the exact split a
    # run used. Defaults reproduce today's flat seed-42 valid split; the HPO
    # hold-out report is reconstructed with --holdout-frac 0.2 --holdout-eval.
    parser.add_argument("--split-seed", type=int, default=42,
                        help="train/valid split seed (config default 42)")
    parser.add_argument("--holdout-frac", type=float, default=0.0,
                        help="patient-disjoint hold-out fraction; 0.0 = flat split (default)")
    parser.add_argument("--holdout-eval", action="store_true",
                        help="score on the HELD set (train on DEV) — the hold-out report split")
    args = parser.parse_args()

    bst = xgb.Booster()
    bst.load_model(str(args.model))
    bst.set_param({"eval_metric": "auc"})

    n_trees = bst.num_boosted_rounds()
    assert n_trees == args.expected_trees, (
        f"tree-budget mismatch: {args.model.name} has {n_trees} trees, "
        f"expected {args.expected_trees}"
    )
    print(f"{args.model.name}: {n_trees} trees (expected {args.expected_trees}) OK\n")

    header = (
        f"{'site':<26} {'auc_roc':>8} {'auc_roc_ci':>19} {'auc_pr':>8} "
        f"{'brier':>8} {'fixed(tn,fp,fn,tp)':>20} {'youden(tn,fp,fn,tp)':>21} "
        f"{'n_pos':>6} {'n':>6}"
    )
    print(header)
    print("-" * len(header))
    for data_path in args.data:
        m = score_booster_on_half(
            bst, data_path, args.operating_point, args.n_boot, args.boot_seed,
            split_seed=args.split_seed, holdout_frac=args.holdout_frac,
            holdout_eval=args.holdout_eval,
        )
        ci = f"[{m['auc_roc_lo']:.4f},{m['auc_roc_hi']:.4f}]"
        fixed = f"({m['tn']},{m['fp']},{m['fn']},{m['tp']})"
        youden = f"({m['tn_j']},{m['fp_j']},{m['fn_j']},{m['tp_j']})"
        print(
            f"{data_path.name:<26} {m['auc_roc']:>8.4f} {ci:>19} "
            f"{m['auc_pr']:>8.4f} {m['brier']:>8.4f} {fixed:>20} {youden:>21} "
            f"{m['n_pos']:>6} {m['n']:>6}"
        )


if __name__ == "__main__":
    main()
