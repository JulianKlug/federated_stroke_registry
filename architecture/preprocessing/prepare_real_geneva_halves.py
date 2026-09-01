"""Produce the two REAL frozen-schema Geneva halves the SuperNodes train on.

Preprocessing-track deliverable (roadmap "Geneva preprocessing: EHR → tabular
in the frozen schema"): registry .xlsx + EHR extraction dir → two
patient-disjoint parquet halves at the paths `[tool.fed_stroke.nodes]` points
to, plus the aggregate-only smoke-test report compared against Shenzhen's.

Contract with the architecture layer (do NOT do the loader's job here):
- one row per case_admission_id; columns EXACTLY
  ['case_admission_id', *FEATURE_COLS, TARGET_COL] — no PII, no extras;
- missing values stay NaN (the loader applies MISSING_SENTINEL);
- NO one-row-per-patient dedup, NO train/valid split, NO sentinel encoding —
  R3/R4 and the sentinel are loader-side (task.py) and accountant-reviewed there;
- target is {0, 1} int (the R7 gate requires exactly binary labels);
- nothing data-derived becomes public config (feature ranges for the DP bins
  live in dp/boost.FEATURE_RANGES and must be clinically fixed, never fitted).

Builds on registry_alignement/ (cohort selection + OPSUM outcome rules in
build_gva_summary_table, EHR first-values in build_gva_first_values_table,
unit conversions in mappings/) and preprocessing/prepare_geneva_halves.py
(patient-disjoint stratified split).

Usage:
    python architecture/preprocessing/prepare_real_geneva_halves.py \
        --registry /mnt/hdd1/datasets/GVA_stroke_registry/stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir  /mnt/hdd1/datasets/GVA_stroke_registry/Extraction_YYYYMMDD \
        --out-dir  out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
ARCH_DIR = _HERE.parent
REPO_ROOT = ARCH_DIR.parent
# fed_stroke (frozen schema) + repo root (registry_alignement, preprocessing).
sys.path.insert(0, str(ARCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fed_stroke.schema import FEATURE_COLS, FEATURE_UNITS, TARGET_COL  # noqa: E402

SPLIT_SEED = 42          # half-partition seed (same as the dev halves)
SCHEMA_VERSION = "frozen-v1"   # bump on any FEATURE_COLS/TARGET_COL/unit change
PROVENANCE = "real-frozen-schema"


def build_frozen_geneva_table(registry_xlsx: Path, ehr_dir: Path) -> pd.DataFrame:
    """Registry + EHR → ONE tidy per-admission table in the frozen schema.

    Steps to populate (reuse, don't re-derive):
    - cohort: registry_alignement.build_gva_summary_table.preprocess
      (exact-duplicate rows dropped, 'Type of event' == 'Ischemic stroke');
    - outcome: the OPSUM 3M Death / 3M mRS reconciliation already coded there;
    - case_admission_id: preprocessing.prepare_geneva_halves.build_case_admission_id
      (== the loader's patient_id + '_' + EDS-last-4 derivation, task.py);
    - EHR features: registry_alignement.build_gva_first_values_table extraction,
      joined on case_admission_id;
    - unit-convert into the schema's units of record (FEATURE_UNITS; via
      registry_alignement.mappings.UNIT_CONVERSIONS), then rename to the frozen
      names (mappings.GVA_TO_FROZEN), then mappings.validate_frozen_columns.

    Input:
        registry_xlsx: the Geneva stroke-registry export (.xlsx).
        ehr_dir: the EHR extraction directory (patientvalue + lab CSVs).
    Output:
        DataFrame, one row per case_admission_id, columns EXACTLY
        ['case_admission_id', *FEATURE_COLS, TARGET_COL]:
        - case_admission_id: str, '<patient_id>_<eds_last_4>';
        - features: float, IN FEATURE_UNITS, missing = NaN (no sentinel),
          out-of-range values resolved HERE (data-quality error, not a
          privacy question);
        - target: int in {0, 1}; rows with underivable outcome dropped
          (count reported by the caller via the summary).
    """
    raise NotImplementedError


def split_patient_disjoint_halves(
    df: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen table → (half_A, half_B), patient-level disjoint.

    Reuse preprocessing.prepare_geneva_halves.stratified_patient_split
    (50/50 at the patient level, stratified on per-patient max outcome,
    deterministic under `seed` — re-runs must not move patients between
    nodes, or cross-run ledger composition per site breaks).

    Input:
        df: output of build_frozen_geneva_table.
        seed: partition seed.
    Output:
        (half_A, half_B): same columns as df; patient_id sets DISJOINT —
        assert the intersection is empty before returning (Claim 9 /
        reviewer B F9; one line, non-negotiable even though waived as a
        standalone check).
    """
    raise NotImplementedError


def write_halves(
    half_a: pd.DataFrame, half_b: pd.DataFrame, out_dir: Path,
    source_files: dict[str, str],
) -> dict:
    """Write the node parquets + provenance metadata; return the smoke summary.

    Input:
        half_a, half_b: outputs of split_patient_disjoint_halves.
        out_dir: parquet destination — the files MUST land at the paths
            pyproject [tool.fed_stroke.nodes] declares
            (out/geneva_half_A.parquet, out/geneva_half_B.parquet).
        source_files: {filename: sha256} of the inputs, stamped for audit.
    Output:
        dict, the roadmap smoke-test artifact — AGGREGATE-ONLY (this is the
        one report designed to cross sites for the Geneva/Shenzhen
        comparison): per half — row count, label balance, and per feature
        median / min / max / missing rate + unit-check pass/fail.
        Also written as <out_dir>/geneva_smoke_report.json.

    Side effects:
        - parquet key-value metadata stamped on both files: schema version
          (SCHEMA_VERSION), data-provenance (PROVENANCE), source hashes,
          creation date — the parquet-side provenance marker (B F8);
        - after these files exist, flip both nodes' data-provenance to
          'real-frozen-schema' in pyproject [tool.fed_stroke.nodes] (manual,
          deliberate — it arms the ledger).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True,
                        help="Geneva stroke-registry .xlsx")
    parser.add_argument("--ehr-dir", type=Path, required=True,
                        help="EHR extraction dir (patientvalue + lab CSVs)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out",
                        help="parquet destination (must match pyproject nodes)")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED,
                        help="patient-partition seed")
    args = parser.parse_args()

    df = build_frozen_geneva_table(args.registry, args.ehr_dir)
    half_a, half_b = split_patient_disjoint_halves(df, args.seed)
    summary = write_halves(half_a, half_b, args.out_dir,
                           source_files={})  # TODO: sha256 of the two inputs
    print(summary)


if __name__ == "__main__":
    main()
