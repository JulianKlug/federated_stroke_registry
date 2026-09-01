"""SIDE JOB (Geneva-only phase): split the preprocessed GVA table into two node halves.

The 1.1.b go/no-go runs on a federated 2-node topology that is Geneva-only:
both SuperNodes hold GVA patients. This thin driver layers that TRANSITIONAL
partition on top of the real preprocessing (preprocess_gva.py) — it reads the
single frozen-schema parquet and writes the two half parquets the loopback
nodes point at. At v1.3 (Shenzhen joins) it is retired: the GVA node consumes
preprocess_gva.py's output directly.

Usage:
    python architecture/preprocessing/preprocess_gva.py --registry ... --ehr-dir ...
    python architecture/preprocessing/split_gva_halves.py \
        --frozen out/gva_frozen.parquet --out-dir out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from preprocess_gva import REPO_ROOT, write_node_parquet  # noqa: E402

SPLIT_SEED = 42   # half-partition seed (same as the dev halves)


def split_patient_disjoint_halves(
    df: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen table → (half_A, half_B), patient-level disjoint.

    Reuse preprocessing.prepare_geneva_halves.stratified_patient_split
    (50/50 at the patient level, stratified on per-patient max outcome,
    deterministic under `seed` — re-runs must not move patients between
    nodes, or cross-run ledger composition per site breaks).

    Input:
        df: preprocess_gva.build_frozen_gva_table output (read from its parquet).
        seed: partition seed.
    Output:
        (half_A, half_B): same columns as df; patient_id sets DISJOINT —
        assert the intersection is empty before returning (Claim 9 /
        reviewer B F9; one line, non-negotiable even though waived as a
        standalone check).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=REPO_ROOT / "out" / "gva_frozen.parquet",
                        help="preprocess_gva.py output parquet")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out",
                        help="halves destination — must match pyproject "
                             "[tool.fed_stroke.nodes] data-paths "
                             "(geneva_half_A.parquet / geneva_half_B.parquet)")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    df = pd.read_parquet(args.frozen)
    half_a, half_b = split_patient_disjoint_halves(df, args.seed)
    for half, name in ((half_a, "geneva_half_A.parquet"),
                       (half_b, "geneva_half_B.parquet")):
        summary = write_node_parquet(half, args.out_dir / name,
                                     source_files={})  # TODO: sha256 of --frozen
        print(summary)


if __name__ == "__main__":
    main()
