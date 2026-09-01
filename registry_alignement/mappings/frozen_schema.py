"""Per-site raw-name → frozen-schema mappings (the executable half of FROZEN_FEATURES.md).

The frozen names — `fed_stroke.schema.FEATURE_COLS` / `TARGET_COL` — are the pivot every
site maps INTO; they never change per site. One dict per site, mapping-as-data (same
pattern as GVA_TO_SHENZEN / UNIT_CONVERSIONS: reviewable, diffable, no logic).

The frozen schema dictates the UNITS too: `fed_stroke.schema.FEATURE_UNITS` is the unit
of record per frozen column, and dp/boost.FEATURE_RANGES is expressed in those units. Every
site's unit conversion must land THERE — UNIT_CONVERSIONS entries are means to that target,
never a target themselves.

Apply order in the site preprocessing (architecture/preprocessing/):
    unit-convert into FEATURE_UNITS (UNIT_CONVERSIONS, keyed by RAW name)
    → rename (this module)
    → validate_frozen_columns(...) — fail loudly on BOTH missing and stray columns.

This module deliberately does NOT import fed_stroke (registry_alignement stays
independent of the architecture layer); the caller passes the frozen names in.

Freeze discipline: editing these dicts after the schema freeze is a cross-site
contract change — bump fed_stroke SCHEMA_VERSION and re-obtain partner sign-off.
"""
from __future__ import annotations

# --- Geneva ------------------------------------------------------------------------
# raw column (registry .xlsx or EHR-derived *_first_value, post unit-conversion)
# -> frozen name. Source of truth: frozen_feature_names.xlsx (this dict is its
# geneva_variable_name / frozen_feature_name columns, verbatim). The trailing
# comment on each entry is the frozen unit of record (frozen_feature_unit) the
# value must arrive in BEFORE the rename.
GVA_TO_FROZEN: dict[str, str] = {
    "Age (calc.)": "age",  # years
    "Sex": "sex",  # binary
    # NOTE: semantic INVERSION — the Geneva field records "onset time known";
    # preprocessing must negate it before this rename lands it as wake_up_stroke.
    "Time of symptom onset known": "wake_up_stroke",  # binary
    "Prestroke disability (Rankin)": "pre_stroke_mrs",  # no unit
    "temperature_first_value": "temperature",  # °C
    "pulse_first_value": "heart_rate",  # bpm
    "fr_first_value": "respiratory_rate",  # bpm
    "1st syst. bp": "systolic_blood_pressure",  # mmHg
    "1st diast. bp": "diastolic_blood_pressure",  # mmHg
    "NIH on admission": "NIHSS",  # no unit
    "1st glucose": "glucose",  # mmol/L
    "GCS on admission": "GCS",  # no unit
    "globules_blancs_first_value": "white_blood_cell_count",  # G/l
    "neutrophiles_nb_abs_first_value": "neutrophil_count",  # G/l
    "lymphocytes_nb_abs_first_value": "lymphocyte_count",  # G/l
    "crp_first_value": "CRP",  # mg/l
    "inr_first_value": "INR",  # no unit
    "fibrinogene_first_value": "fibrinogen",  # g/l
    "d_dimeres_first_value": "d_dimer",  # ng/ml
    "hba1c_first_value": "hba1c",  # %
    "alat_first_value": "alt",  # U/l
    "ldl_calc_first_value": "LDL",  # mmol/l
    "creatinine_first_value": "creatinine",  # µmol/l
    "uree_first_value": "urea",  # mmol/l
    "MedHist Stroke": "med_hist_stroke",  # binary
    "MedHist TIA": "med_hist_tia",  # binary
    "MedHist ICH": "med_hist_ich",  # binary
    "MedHist Hypertension": "med_hist_hta",  # binary
    "MedHist Diabetes": "med_hist_diabetes",  # binary
    "MedHist Hyperlipidemia": "med_hist_hyperlipidemia",  # binary
    "MedHist Atrial Fibr.": "med_hist_af",  # binary
    "MedHist CHD": "med_hist_coronary_heart_disease",  # binary
    "MedHist Prost. heart valves": "med_hist_valv_heart_disease",  # binary
    "MedHist PAD": "med_hist_peripheral_artery_disease",  # binary
    "MedHist Smoking": "med_hist_smoking",  # binary
    "ODT": "ODT",  # min
    "ONT": "ONT",  # min
    "DNT": "DNT",  # min
    "OPT": "OPT",  # min
    "IVT with rtPA": "IVT",  # binary
    "IAT": "EVT",  # binary
    # Target — not a row in frozen_feature_names.xlsx (features only); stays the
    # OPSUM-reconciled registry outcome, identity-named.
    "3M Death": "3M Death",  # binary {0,1}
}

# --- Shenzhen ----------------------------------------------------------------------
# raw Shenzhen export column -> frozen name. Source of truth:
# frozen_feature_names.xlsx (shenzhen_variable_name / frozen_feature_name columns,
# verbatim); run remotely by the partner's preprocessing, never on Geneva
# infrastructure. Trailing comments are the frozen units of record.
SHENZHEN_TO_FROZEN: dict[str, str] = {
    "Age": "age",  # years
    "Sex": "sex",  # binary
    "Wake-up Stroke": "wake_up_stroke",  # binary
    "Pre-morbid mRS": "pre_stroke_mrs",  # no unit
    "First Vital Signs - Temperature": "temperature",  # °C
    "First Vital Signs - Pulse": "heart_rate",  # bpm
    "First Vital Signs - Respiration": "respiratory_rate",  # bpm
    "First Vital Signs - SBP": "systolic_blood_pressure",  # mmHg
    "First Vital Signs - DBP": "diastolic_blood_pressure",  # mmHg
    "First Post-onset NIHSS": "NIHSS",  # no unit
    "Point-of-care Glucose": "glucose",  # mmol/L
    "First GCS Score": "GCS",  # no unit
    "WBC": "white_blood_cell_count",  # G/l
    "Neutrophil Count": "neutrophil_count",  # G/l
    "Lymphocyte Count": "lymphocyte_count",  # G/l
    "hs-CRP": "CRP",  # mg/l
    "INR": "INR",  # no unit
    "Fibrinogen": "fibrinogen",  # g/l
    "D-dimer": "d_dimer",  # ng/ml
    "HbA1c": "hba1c",  # %
    "ALT": "alt",  # U/l
    "LDL-C": "LDL",  # mmol/l
    "Serum Creatinine": "creatinine",  # µmol/l
    "BUN": "urea",  # mmol/l
    "Prior cerebral infarction": "med_hist_stroke",  # binary
    "Prior transient ischemic attack (TIA)": "med_hist_tia",  # binary
    "Prior subarachnoid hemorrhage / intracerebral hemorrhage": "med_hist_ich",  # binary
    "Prior hypertension": "med_hist_hta",  # binary
    "Prior diabetes mellitus": "med_hist_diabetes",  # binary
    "Prior dyslipidemia": "med_hist_hyperlipidemia",  # binary
    "Prior atrial fibrillation": "med_hist_af",  # binary
    "Prior coronary artery disease": "med_hist_coronary_heart_disease",  # binary
    "Prior valvular heart disease": "med_hist_valv_heart_disease",  # binary
    "Prior peripheral artery disease": "med_hist_peripheral_artery_disease",  # binary
    "Smoking History": "med_hist_smoking",  # binary
    "ODT": "ODT",  # min
    "ONT": "ONT",  # min
    "DNT": "DNT",  # min
    "OPT": "OPT",  # min
    # TODO(shenzhen sign-off): no source column yet for "IVT" (binary)
    # TODO(shenzhen sign-off): no source column yet for "EVT" (binary)
    # TODO(shenzhen sign-off): outcome source for "3M Death",
}


def validate_frozen_columns(columns, frozen_features, target_col,
                            id_col: str = "case_admission_id") -> None:
    """Assert a produced table matches the frozen contract EXACTLY.

    columns: the produced DataFrame's columns (post-rename).
    frozen_features / target_col: pass fed_stroke.schema.FEATURE_COLS / TARGET_COL.
    Raises ValueError naming both the missing and the unexpected columns —
    a stray raw column is as much a schema bug as a missing feature (it would
    ride through the loader's split frames into out/).
    """
    produced = set(columns) - {id_col, target_col}
    expected = set(frozen_features)
    if produced == expected and target_col in columns and id_col in columns:
        return
    raise ValueError(
        "frozen-schema mismatch: "
        f"missing={sorted(expected - produced) + [c for c in (id_col, target_col) if c not in columns]} "
        f"unexpected={sorted(produced - expected)}"
    )
