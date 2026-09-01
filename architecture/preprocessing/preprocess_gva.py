"""GVA preprocessing: registry + EHR → the frozen-schema table the GVA node trains on.

THE production preprocessing for the Geneva site (roadmap "Geneva preprocessing:
EHR → tabular in the frozen schema"). Once Shenzhen joins (v1.3), the GVA
SuperNode consumes this pipeline's single output parquet directly, exactly as
Shenzhen's partner-run preprocessing produces theirs — one site, one table, one
smoke report. The Geneva-only-phase split into two halves is a SIDE JOB layered
on top (split_gva_halves.py), not part of this pipeline.

Contract with the architecture layer (do NOT do the loader's job here):
- one row per case_admission_id; columns EXACTLY
  ['case_admission_id', *FEATURE_COLS, TARGET_COL] — no PII, no extras;
- values IN the schema's units of record (FEATURE_UNITS);
- missing values stay NaN (the loader applies MISSING_SENTINEL);
- NO one-row-per-patient dedup, NO train/valid split, NO sentinel encoding —
  R3/R4 and the sentinel are loader-side (task.py) and accountant-reviewed there;
- target is {0, 1} int (the R7 gate requires exactly binary labels);
- nothing data-derived becomes public config (feature ranges for the DP bins
  live in dp/boost.FEATURE_RANGES and must be clinically fixed, never fitted).

Builds on registry_alignement/ (cohort selection + OPSUM outcome rules in
build_gva_summary_table, EHR first-values in build_gva_first_values_table,
unit conversions + frozen-name mapping + column validation in mappings/).

Usage:
    python architecture/preprocessing/preprocess_gva.py \
        --registry /mnt/hdd1/datasets/GVA_stroke_registry/stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir  /mnt/hdd1/datasets/GVA_stroke_registry/Extraction_YYYYMMDD \
        --out      out/gva_frozen.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
ARCH_DIR = _HERE.parent
REPO_ROOT = ARCH_DIR.parent
# fed_stroke (frozen schema) + repo root (registry_alignement).
sys.path.insert(0, str(ARCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fed_stroke.schema import FEATURE_COLS, FEATURE_UNITS, TARGET_COL  # noqa: E402

SCHEMA_VERSION = "frozen-v1"   # bump on any FEATURE_COLS/FEATURE_UNITS/TARGET_COL change
PROVENANCE = "real-frozen-schema"


def build_frozen_gva_table(registry_xlsx: Path, ehr_dir: Path) -> pd.DataFrame:
    """Registry + EHR → ONE tidy per-admission table in the frozen schema.

    Steps to populate (reuse, don't re-derive):
    - cohort: registry_alignement.build_gva_summary_table.preprocess
      (exact-duplicate rows dropped, 'Type of event' == 'Ischemic stroke');
    - outcome: the OPSUM 3M Death / 3M mRS reconciliation already coded there;
    - case_admission_id: preprocessing.prepare_geneva_halves.build_case_admission_id
      (== the loader's patient_id + '_' + EDS-last-4 derivation, task.py);
    - EHR features: registry_alignement.build_gva_first_values_table extraction,
      joined on case_admission_id;
    - unit-convert into FEATURE_UNITS (mappings.UNIT_CONVERSIONS), rename to the
      frozen names (mappings.GVA_TO_FROZEN), then mappings.validate_frozen_columns.

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
          (count reported via the summary).
    """
    raise NotImplementedError


def write_node_parquet(df: pd.DataFrame, out_path: Path,
                       source_files: dict[str, str]) -> dict:
    """Write ONE site's node parquet + provenance metadata; return the smoke summary.

    This is the artifact a SuperNode's node_config data-path points at — for the
    GVA node in the v1.3 cross-site topology, and (via split_gva_halves.py) for
    each loopback half in the Geneva-only phase.

    Input:
        df: a frozen-schema table (build_frozen_gva_table output or a half of it).
        out_path: parquet destination (must match the node's data-path).
        source_files: {filename: sha256} of the inputs, stamped for audit.
    Output:
        dict, the roadmap smoke-test artifact — AGGREGATE-ONLY (the one report
        designed to cross sites for the Geneva/Shenzhen comparison): row count,
        label balance, per feature median / min / max / missing rate + unit-check
        pass/fail. Also written next to the parquet as <stem>_smoke_report.json.

    Side effects:
        - parquet key-value metadata stamped: SCHEMA_VERSION, PROVENANCE,
          source hashes, creation date — the parquet-side provenance marker (B F8);
        - after the node file exists, flip that node's data-provenance to
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
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "gva_frozen.parquet",
                        help="the GVA node's parquet (node_config data-path)")
    args = parser.parse_args()

    df = build_frozen_gva_table(args.registry, args.ehr_dir)
    summary = write_node_parquet(df, args.out,
                                 source_files={})  # TODO: sha256 of the two inputs
    print(summary)


if __name__ == "__main__":
    main()
