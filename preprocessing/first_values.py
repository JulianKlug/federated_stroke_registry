"""First lab/vital value recorded after admission, extracted from the GVA EHR.

Registry patients are matched to EHR rows via ``case_admission_id``
(``patient_id_EDS_last_4_digits``) — see ``preprocessing.case_ids``.

Two EHR sources are scanned per admission:
  - patientvalue (PV) CSVs: vital signs (pv.pulse, pv.fr, pv.temperature) and
    a subset of labs encoded as ``lab.result.sang.*`` with the value row at
    ``subkey == 'Valeur'``.
  - lab CSVs: dedicated labs keyed by ``dosage_label`` (lymphocytes,
    fibrinogène, HbA1c, ALAT, LDL, cystatine C, urates, urée, homocystéine).

"First after admission" = earliest record whose timestamp is >= the
``Arrival at hospital`` date in the registry (date-only granularity, so the
admission day itself is included).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
