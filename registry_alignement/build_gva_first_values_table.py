"""
Build a per-patient CSV of the first lab/vital value recorded after admission
for the Geneva (GVA) ischemic-stroke cohort.

Cohort selection follows ``build_gva_summary_table.preprocess`` (drop dupes,
filter to ``Type of event == 'Ischemic stroke'``).

Registry patients are matched to EHR rows via ``case_admission_id``
(``patient_id_EDS_last_4_digits``) — see ``geneva_preprocessing.utils``.

Two EHR sources are scanned per admission:
  - patientvalue (PV) CSVs: vital signs (pv.pulse, pv.fr, pv.temperature) and
    a subset of labs encoded as ``lab.result.sang.*`` with the value row at
    ``subkey == 'Valeur'``.
  - lab CSVs: dedicated labs keyed by ``dosage_label`` (lymphocytes,
    fibrinogène, HbA1c, ALAT, LDL, cystatine C, urates, urée, homocystéine).

"First after admission" = earliest record whose timestamp is >= the
``Arrival at hospital`` date in the registry (date-only granularity, so the
admission day itself is included).

Usage:
    python build_gva_first_values_table.py \
        --registry data/gva_stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir   /path/to/Extraction_YYYYMMDD \
        --output-dir out/
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Put both the package dir (so ``from mappings import ...`` inside
# build_gva_summary_table resolves) and the repo root (so we can import the
# sibling module with its fully-qualified name) on sys.path.
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from registry_alignement.build_gva_summary_table import parse_yyyymmdd, preprocess
from registry_alignement.geneva_preprocessing.utils import (
    create_ehr_case_identification_column,
    create_registry_case_identification_column,
)


# ---------------------------------------------------------------------------
# Variable selection
# ---------------------------------------------------------------------------
# (output_name, patient_value, subkey_filter) — subkey_filter=None means "any"
PV_VITALS: list[tuple[str, str, str | None]] = [
    ("pulse",       "pv.pulse",       "pulse"),
    ("fr",          "pv.fr",          None),
    ("temperature", "pv.temperature", "temperature"),
]

# Labs that live inside the patientvalue file as ``lab.result.sang.*``; the
# value row has ``subkey == 'Valeur'``.
PV_LABS: list[tuple[str, list[str]]] = [
    ("globules_blancs",     ["lab.result.sang.globules_blancs"]),
    ("neutrophiles_nb_abs", ["lab.result.sang.neutrophiles_nb.abs"]),
    ("crp",                 ["lab.result.sang.p_crp"]),
    ("inr",                 ["lab.result.sang.inr"]),
    ("d_dimeres",           ["lab.result.sang.p_d_dimeres"]),
    # Two equivalent keys for serum creatinine — merge into one variable.
    ("creatinine",          ["lab.result.sang.p_creatinine",
                             "lab.result.sang.creatinine"]),
]

# Labs from the dedicated lab file, keyed by ``dosage_label``.
LAB_DOSAGES: list[tuple[str, list[str]]] = [
    ("lymphocytes_nb_abs", ["lymphocytes-nb.abs"]),
    ("fibrinogene",        ["fibrinogène"]),
    ("hba1c",              ["hémoglobine glyquée"]),
    ("alat",               ["ALAT"]),
    ("ldl_calc",           ["LDL cholestérol calculé"]),
    ("cystatine_c",        ["cystatine C"]),
    ("urates",             ["urates"]),
    ("uree",               ["urée"]),
    ("homocysteine",       ["homocystéine"]),
]

# ``DD.MM.YYYY HH:MM`` — used by both PV ``datetime`` and lab ``sample_date``.
EHR_DATETIME_FORMAT = "%d.%m.%Y %H:%M"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_concat_csvs(ehr_dir: Path, prefix: str) -> pd.DataFrame:
    """Load every CSV under ``ehr_dir`` whose filename starts with ``prefix``."""
    files = sorted(
        ehr_dir / f for f in os.listdir(ehr_dir)
        if f.startswith(prefix) and f.endswith(".csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching '{prefix}*.csv' found under {ehr_dir}"
        )
    frames = [
        pd.read_csv(f, delimiter=";", encoding="utf-8", dtype=str)
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)


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
# First-value extraction
# ---------------------------------------------------------------------------
_CENSORED_RE = re.compile(r"^\s*([<>])\s*(.+?)\s*$")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert lab/vital strings to floats.

    Handles, in order:
      - censored values: ``>X -> 1.05 * X``, ``<X -> 0.95 * X``
        (OPSUM convention, see ``lab_preprocessing.correct_non_numerical_values``).
      - French thousand-separator apostrophes (``2'133 -> 2133``).
      - French decimal commas (``1,23 -> 1.23``).
      - Trailing dots and ``''``/``'-'``/``'nan'`` sentinels.
    Anything that still can't parse becomes NaN.
    """
    s = series.astype(str).str.strip()

    censored = s.str.extract(_CENSORED_RE)
    is_gt = censored[0].eq(">")
    is_lt = censored[0].eq("<")
    base = censored[1].where(is_gt | is_lt, s)

    base = (
        base.str.replace("'", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.rstrip(".")
    )
    base = base.replace({"": np.nan, "-": np.nan, "nan": np.nan, "NaN": np.nan})
    num = pd.to_numeric(base, errors="coerce")
    num = num.where(~is_gt, num * 1.05)
    num = num.where(~is_lt, num * 0.95)
    return num


def _earliest_after_admission(
    measurements: pd.DataFrame,
    cohort: pd.DataFrame,
    value_col: str,
    time_col: str,
    unit_col: str | None,
) -> pd.DataFrame:
    """For each cohort case, return the earliest measurement with
    ``time_col >= admission_date``.

    ``measurements`` is expected to already be filtered to a single variable
    and contain (at least) ``case_admission_id``, ``time_col``, ``value_col``.
    """
    if measurements.empty:
        return pd.DataFrame(
            columns=["case_admission_id", "first_value", "first_datetime", "first_unit"]
        )

    m = measurements.merge(
        cohort[["case_admission_id", "admission_date"]],
        on="case_admission_id",
        how="inner",
    )
    m = m.dropna(subset=[time_col, "admission_date", value_col])
    m = m[m[time_col] >= m["admission_date"]]
    if m.empty:
        return pd.DataFrame(
            columns=["case_admission_id", "first_value", "first_datetime", "first_unit"]
        )

    m = m.sort_values([time_col]).drop_duplicates(
        subset=["case_admission_id"], keep="first"
    )
    out_cols = {
        "case_admission_id": "case_admission_id",
        value_col: "first_value",
        time_col: "first_datetime",
    }
    keep = ["case_admission_id", value_col, time_col]
    if unit_col and unit_col in m.columns:
        keep.append(unit_col)
        out_cols[unit_col] = "first_unit"
    else:
        m["__unit"] = np.nan
        keep.append("__unit")
        out_cols["__unit"] = "first_unit"
    return m[keep].rename(columns=out_cols)


def extract_pv_vital_first_values(
    vitals_df: pd.DataFrame, cohort: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """For each PV vital (pulse / fr / temperature), return first-value frame."""
    results: dict[str, pd.DataFrame] = {}
    for out_name, pv_key, subkey in PV_VITALS:
        sub = vitals_df[vitals_df["patient_value"] == pv_key]
        if subkey is not None and "subkey" in sub.columns:
            sub = sub[sub["subkey"] == subkey]
        if sub.empty:
            results[out_name] = _earliest_after_admission(
                pd.DataFrame(), cohort, "value_num", "datetime_parsed", None
            )
            continue
        sub = sub.copy()
        sub["value_num"] = _coerce_numeric(sub["value"])
        sub["datetime_parsed"] = pd.to_datetime(
            sub["datetime"], format=EHR_DATETIME_FORMAT, errors="coerce"
        )
        unit_col = "unit" if "unit" in sub.columns else None
        results[out_name] = _earliest_after_admission(
            sub, cohort, "value_num", "datetime_parsed", unit_col
        )
    return results


def extract_pv_lab_first_values(
    vitals_df: pd.DataFrame, cohort: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """For each ``lab.result.sang.*`` lab inside the PV file, return first-value frame.

    The value row is identified by ``subkey == 'Valeur'``; unit is read from
    the matching ``Unite`` row at the same (case, datetime, patient_value).
    """
    results: dict[str, pd.DataFrame] = {}
    if "subkey" not in vitals_df.columns:
        for out_name, _ in PV_LABS:
            results[out_name] = pd.DataFrame(
                columns=["case_admission_id", "first_value", "first_datetime", "first_unit"]
            )
        return results

    for out_name, pv_keys in PV_LABS:
        sub = vitals_df[vitals_df["patient_value"].isin(pv_keys)].copy()
        if sub.empty:
            results[out_name] = pd.DataFrame(
                columns=["case_admission_id", "first_value", "first_datetime", "first_unit"]
            )
            continue
        sub["datetime_parsed"] = pd.to_datetime(
            sub["datetime"], format=EHR_DATETIME_FORMAT, errors="coerce"
        )
        # Pull the value (subkey == 'Valeur') and the unit (subkey == 'Unite')
        # for the same (case, datetime, patient_value) group.
        values = sub[sub["subkey"] == "Valeur"][
            ["case_admission_id", "datetime_parsed", "patient_value", "value"]
        ].copy()
        values["value_num"] = _coerce_numeric(values["value"])
        units = sub[sub["subkey"] == "Unite"][
            ["case_admission_id", "datetime_parsed", "patient_value", "value"]
        ].rename(columns={"value": "unit_str"})
        merged = values.merge(
            units,
            on=["case_admission_id", "datetime_parsed", "patient_value"],
            how="left",
        )
        results[out_name] = _earliest_after_admission(
            merged, cohort, "value_num", "datetime_parsed", "unit_str"
        )
    return results


def extract_lab_dosage_first_values(
    lab_df: pd.DataFrame, cohort: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """For each dosage label in the lab file, return first-value frame."""
    results: dict[str, pd.DataFrame] = {}
    for out_name, labels in LAB_DOSAGES:
        sub = lab_df[lab_df["dosage_label"].isin(labels)].copy()
        if sub.empty:
            results[out_name] = pd.DataFrame(
                columns=["case_admission_id", "first_value", "first_datetime", "first_unit"]
            )
            continue
        sub["value_num"] = _coerce_numeric(sub["value"])
        sub["sample_date_parsed"] = pd.to_datetime(
            sub["sample_date"], format=EHR_DATETIME_FORMAT, errors="coerce"
        )
        unit_col = "unit_of_measure" if "unit_of_measure" in sub.columns else None
        results[out_name] = _earliest_after_admission(
            sub, cohort, "value_num", "sample_date_parsed", unit_col
        )
    return results


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_wide(
    cohort: pd.DataFrame,
    per_var_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-variable first-value frames into a wide per-patient table."""
    out = cohort.copy()
    for var_name, frame in per_var_frames.items():
        renamed = frame.rename(
            columns={
                "first_value":    f"{var_name}_first_value",
                "first_datetime": f"{var_name}_first_datetime",
                "first_unit":     f"{var_name}_first_unit",
            }
        )
        out = out.merge(renamed, on="case_admission_id", how="left")
    return out


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
