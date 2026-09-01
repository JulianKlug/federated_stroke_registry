"""
Build a per-patient CSV of the first lab/vital value recorded after admission
for the Geneva (GVA) ischemic-stroke cohort.

Cohort selection follows ``preprocessing.registry_cohort.preprocess`` (drop
dupes, filter to ``Type of event == 'Ischemic stroke'``).

The extraction itself (variable selection, value coercion, first-after-
admission logic) lives in the shared library ``preprocessing.first_values``;
this script is the CLI that wires registry + EHR dir to a CSV.

Usage:
    python build_gva_first_values_table.py \
        --registry data/gva_stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir   /path/to/Extraction_YYYYMMDD \
        --output-dir out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.case_ids import (  # noqa: E402
    create_ehr_case_identification_column,
    create_registry_case_identification_column,
)
from preprocessing.first_values import (  # noqa: E402
    assemble_wide,
    extract_lab_dosage_first_values,
    extract_pv_lab_first_values,
    extract_pv_vital_first_values,
    load_concat_csvs,
)
from preprocessing.registry_cohort import parse_yyyymmdd, preprocess  # noqa: E402


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def build_cohort(registry_xlsx: Path) -> pd.DataFrame:
    """Load the registry, apply ischemic-stroke cohort filter, return cohort
    dataframe with ``case_admission_id`` and ``admission_date`` columns."""
    df = pd.read_excel(registry_xlsx)
    df, n_raw, n_filtered = preprocess(df)
    print(f"[cohort] raw={n_raw}  ischemic_stroke={n_filtered}")

    df = df.copy()
    df["case_admission_id"] = create_registry_case_identification_column(df)
    df["admission_date"] = parse_yyyymmdd(df["Arrival at hospital"])

    # Same admission can appear on multiple registry rows when fields differ
    # in non-bookkeeping ways (e.g., transfer events) and so survive the
    # earlier drop_duplicates() in preprocess(). Collapse them here — for
    # first-value extraction one row per admission is what we want.
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["case_admission_id"], keep="first")
    n_dropped = n_before_dedup - len(df)
    if n_dropped:
        print(f"[cohort] deduped case_admission_id: dropped {n_dropped} extra rows")

    n_missing_admission = int(df["admission_date"].isna().sum())
    if n_missing_admission:
        print(
            f"[cohort] WARNING: {n_missing_admission} cohort patients have no "
            f"parseable 'Arrival at hospital' — first-value search will skip them."
        )

    return df[["case_admission_id", "admission_date"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="First lab/vital values after admission for the GVA ischemic-stroke cohort."
    )
    parser.add_argument(
        "--registry", required=True,
        help="Path to Geneva stroke registry .xlsx file.",
    )
    parser.add_argument(
        "--ehr-dir", required=True,
        help="Directory containing patientvalue*.csv and labo*.csv extractions.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-name", default="gva_first_values_after_admission.csv")
    parser.add_argument("--vitals-prefix", default="patientvalue")
    parser.add_argument("--lab-prefix", default="labo")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    ehr_dir = Path(args.ehr_dir)
    if not registry_path.exists():
        raise SystemExit(f"Registry not found: {registry_path}")
    if not ehr_dir.is_dir():
        raise SystemExit(f"EHR dir not found: {ehr_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort(registry_path)
    print(f"[cohort] n_patients={len(cohort)}")

    print(f"[load]   PV files ({args.vitals_prefix}*.csv) from {ehr_dir}")
    vitals_df = load_concat_csvs(ehr_dir, args.vitals_prefix)
    vitals_df["case_admission_id"] = create_ehr_case_identification_column(vitals_df)
    print(f"[load]   PV rows={len(vitals_df)}")

    print(f"[load]   lab files ({args.lab_prefix}*.csv) from {ehr_dir}")
    lab_df = load_concat_csvs(ehr_dir, args.lab_prefix)
    lab_df["case_admission_id"] = create_ehr_case_identification_column(lab_df)
    print(f"[load]   lab rows={len(lab_df)}")

    per_var: dict[str, pd.DataFrame] = {}
    per_var.update(extract_pv_vital_first_values(vitals_df, cohort))
    per_var.update(extract_pv_lab_first_values(vitals_df, cohort))
    per_var.update(extract_lab_dosage_first_values(lab_df, cohort))

    for var, frame in per_var.items():
        n_with = int(frame["case_admission_id"].nunique()) if not frame.empty else 0
        print(f"[stat]   {var:<22s} patients_with_first_value={n_with}")

    wide = assemble_wide(cohort, per_var)
    out_path = out_dir / args.output_name
    wide.to_csv(out_path, index=False)
    print(f"[write]  {out_path}  rows={len(wide)}  cols={len(wide.columns)}")


if __name__ == "__main__":
    main()
