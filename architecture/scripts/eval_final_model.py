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

import pandas as pd
import xgboost as xgb

# Make `fed_stroke` importable regardless of CWD (script lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fed_stroke.schema import FEATURE_COLS, TARGET_COL  # noqa: E402
from fed_stroke.task import generate_splits  # noqa: E402


def _site_auc(bst: xgb.Booster, data_path: Path) -> float:
    """Rebuild the identical in-run validation split and return its AUC.

    Mirrors client_app.evaluate: same split (`generate_splits`, test_size=0.2,
    seed=42) and the same `eval_set` AUC, so numbers are comparable to the FL run.
    """
    data_df = pd.read_parquet(data_path)
    _, valid_df, _, _ = generate_splits(
        data_df, outcome=TARGET_COL, test_size=0.2, seed=42
    )
    valid_dmatrix = xgb.DMatrix(valid_df[FEATURE_COLS], label=valid_df[TARGET_COL])

    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    return float(eval_results.split("\t")[1].split(":")[1])


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

    print(f"{'model':<28} {'site':<26} {'AUC':>8}")
    print("-" * 64)
    for data_path in args.data:
        auc = _site_auc(bst, data_path)
        print(f"{args.model.name:<28} {data_path.name:<26} {auc:>8.4f}")


if __name__ == "__main__":
    main()
