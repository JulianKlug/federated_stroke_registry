"""GVA preprocessing: registry + EHR → the frozen-schema table the GVA node trains on.


Contract with the architecture layer:
- one row per case_admission_id; columns EXACTLY
  ['case_admission_id', *FEATURE_COLS, TARGET_COL] — no PII, no extras;
- values IN the schema's units of record (FEATURE_UNITS);
- missing values stay NaN (the loader applies MISSING_SENTINEL);
- NO one-row-per-patient dedup, NO train/valid split, NO sentinel encoding —
  R3/R4 and the sentinel are loader-side (task.py) and accountant-reviewed there;
- target is {0, 1} int (the R7 gate requires exactly binary labels);

Usage:
    python architecture/preprocessing/preprocess_gva.py \
        --registry /.../stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir  /.../Extraction_YYYYMMDD \
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
sys.path.insert(0, str(REPO_ROOT/ "registry_alignement"))

from fed_stroke.schema import FEATURE_COLS, FEATURE_UNITS, TARGET_COL  # noqa: E402
from registry_alignement import build_gva_first_values_table, build_gva_summary_table
from registry_alignement.build_gva_first_values_table import (
    load_concat_csvs,
    extract_pv_vital_first_values,
    extract_pv_lab_first_values,
    extract_lab_dosage_first_values,
    assemble_wide
)
from registry_alignement.geneva_preprocessing import utils


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
    # preprocess registry
    df = pd.read_excel(registry_xlsx)
    df, n_raw, n_filtered = build_gva_summary_table.build_cohort(df)
    # derive case_admission_id
    df['case_admission_id'] = utils.create_registry_case_identification_column(df)
    df = build_gva_summary_table.preprocess_outcome(df)
    df = build_gva_summary_table.preprocess_features(df)

    df["admission_date"] = build_gva_first_values_table.parse_yyyymmdd(df["Arrival at hospital"])

    vitals_prefix = "patientvalue"
    lab_prefix = "lab"
    print(f"[df] n_patients={len(df)}")

    print(f"[load]   PV files ({vitals_prefix}*.csv) from {ehr_dir}")
    vitals_df = load_concat_csvs(ehr_dir, vitals_prefix)
    vitals_df["case_admission_id"] = utils.create_ehr_case_identification_column(vitals_df)
    print(f"[load]   PV rows={len(vitals_df)}")

    print(f"[load]   lab files ({lab_prefix}*.csv) from {ehr_dir}")
    lab_df = load_concat_csvs(ehr_dir, lab_prefix)
    lab_df["case_admission_id"] = utils.create_ehr_case_identification_column(lab_df)
    print(f"[load]   lab rows={len(lab_df)}")

    per_var: dict[str, pd.DataFrame] = {}
    per_var.update(extract_pv_vital_first_values(vitals_df, df))
    per_var.update(extract_pv_lab_first_values(vitals_df, df))
    per_var.update(extract_lab_dosage_first_values(lab_df, df))

    wide_df = assemble_wide(df, per_var)

    return wide_df


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
