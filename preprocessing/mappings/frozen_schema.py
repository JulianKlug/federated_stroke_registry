"""The frozen cross-site schema + per-site raw-name → frozen-name mappings.

The frozen names — `FROZEN_FEATURES` (a verbatim mirror of frozen_feature_names.xlsx, sheet
`with_overlap`) and `FROZEN_OUTCOMES` (the standardized raw outcome variables) — are the pivot
every site maps INTO; they never change per site. One dict per site, mapping-as-data (same
pattern as GVA_TO_SHENZEN / UNITS: reviewable, diffable, no logic).

The training LABEL is deliberately NOT part of this contract. The site table carries every
cohort admission with all FROZEN_OUTCOMES (missing outcomes stay NaN — no row is dropped for
an outcome); fed_stroke.schema picks TARGET_COL among the frozen outcomes (plus an optional
binarization rule, e.g. mRS ≤ 2) and the loader drops the unlabelled rows at load time. A
label change is therefore a training-config change, never a re-run of either site's
preprocessing.

`fed_stroke.schema.FEATURE_COLS` / `OUTCOME_COLS` / `FEATURE_UNITS` are a literal mirror of
these (the fed_stroke wheel ships without this package); architecture/tests/
test_preprocess_gva.py asserts the two copies are identical and architecture/preprocessing/
preprocess_gva.py warns loudly at build time if they ever drift.

The frozen schema dictates the UNITS too: `FROZEN_UNITS` is the unit of record per frozen
column (mirrors fed_stroke.schema.FEATURE_UNITS), and `FROZEN_RANGES` is the public,
data-independent plausibility range per feature in those units — a literal mirror of
fed_stroke.dp.boost.FEATURE_RANGES, the DP bin grid (asserted identical by
architecture/tests/test_preprocess_gva.py). preprocessing.frozen_table.nullify_out_of_range
applies FROZEN_RANGES after the rename: a value outside its range becomes NaN and is counted
per feature for the smoke report; a feature out of range wholesale fails the build.
Every site's unit handling must land in FROZEN_UNITS: preprocessing.frozen_table.
convert_to_frozen_units resolves each row's observed unit label through
mappings.unit_aliases.UNIT_ALIASES (label → factor, keyed by frozen unit), falling back to the
site's declared units for columns without a per-row label (e.g. mappings.gva_encodings.
GVA_DECLARED_UNITS). mappings.unit_conversions.UNIT_CONVERSIONS is NOT on this path — it
targets Shenzhen units for the comparison plots.

Apply order in the site preprocessing (preprocessing.frozen_table, driven by
architecture/preprocessing/preprocess_gva.wide_to_frozen):
    convert_to_frozen_units (RAW names; per-row unit check INTO FROZEN_UNITS)
    → encode_binaries (RAW names; features AND binary outcomes, per-site vocabularies,
      e.g. GVA_BINARY_ENCODINGS; NaN stays NaN)
    → rename_to_frozen (this module's per-site dict; fails on absent raw columns)
    → select_mapped_columns (keep exactly what the mapping produced)
    → validate_frozen_columns(...) — fail loudly on BOTH missing and stray columns
    → finalize_frozen_table (column order, dtypes, outcome ranges, id uniqueness)
    → nullify_out_of_range (FROZEN names; features outside FROZEN_RANGES → NaN, counted).

This module deliberately does NOT import fed_stroke (the preprocessing layer
stays independent of the architecture layer); the caller passes the frozen
names in.

Freeze discipline: editing these dicts after the schema freeze is a cross-site
contract change — bump fed_stroke SCHEMA_VERSION and re-obtain partner sign-off.
"""
from __future__ import annotations

# --- The frozen contract -----------------------------------------------------------
ID_COL = "case_admission_id"
BINARY_UNIT_PREFIX = "binary"   # 'binary' or 'binary (<convention>)': encoded, never unit-converted
UNITLESS = "no unit"            # scores / ratios: a missing unit label is expected, not a warning

# Standardized raw outcome variables every site delivers — NOT labels. The label (which column,
# which cut-off) is fed_stroke.schema's decision at training time; here they are just columns,
# NaN when not recorded, no row dropped. Names are frozen-style like the features.
FROZEN_OUTCOMES: list[str] = [
    "mrs_3m",             # modified Rankin Scale at 3 months, ordinal 0-6 (6 = dead)
    "death_3m",           # dead by 3 months (OPSUM-reconciled with mrs_3m / in-hospital death)
    "death_in_hospital",  # died during the index admission
]
# Allowed value range per outcome (an encoding contract, enforced hard by finalize_frozen_table —
# an outcome outside it is an encoding bug, not a plausibility question).
FROZEN_OUTCOME_RANGES: dict[str, tuple[float, float]] = {
    "mrs_3m": (0.0, 6.0),
    "death_3m": (0.0, 1.0),
    "death_in_hospital": (0.0, 1.0),
}

# Verbatim frozen_feature_names.xlsx[with_overlap].frozen_feature_name, row order preserved.
FROZEN_FEATURES: list[str] = [
    "age",
    "sex",
    "wake_up_stroke",
    "pre_stroke_mrs",
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "NIHSS",
    "glucose",
    "GCS",
    "white_blood_cell_count",
    "neutrophil_count",
    "lymphocyte_count",
    "CRP",
    "INR",
    "fibrinogen",
    "d_dimer",
    "hba1c",
    "alt",
    "LDL",
    "creatinine",
    "urea",
    "med_hist_stroke",
    "med_hist_tia",
    "med_hist_ich",
    "med_hist_hta",
    "med_hist_diabetes",
    "med_hist_hyperlipidemia",
    "med_hist_af",
    "med_hist_coronary_heart_disease",
    "med_hist_valv_heart_disease",
    "med_hist_peripheral_artery_disease",
    "med_hist_smoking",
    "ODT",
    "ONT",
    "DNT",
    "OPT",
    "IVT",
    "EVT",
]

# Unit of record per frozen column — frozen_feature_unit verbatim (incl. the xlsx's L/l
# case; unit_aliases.normalize_unit_label folds it). A parenthesised suffix documents an
# encoding convention the partner site must reproduce (base_unit() strips it for lookups).
# Covers the outcomes too, mirroring fed_stroke.schema.FEATURE_UNITS.
FROZEN_UNITS: dict[str, str] = {
    "age": "years",
    "sex": "binary (1 = female)",
    "wake_up_stroke": "binary",
    "pre_stroke_mrs": "no unit",
    "temperature": "°C",
    "heart_rate": "bpm",
    "respiratory_rate": "bpm",
    "systolic_blood_pressure": "mmHg",
    "diastolic_blood_pressure": "mmHg",
    "NIHSS": "no unit",
    "glucose": "mmol/L",
    "GCS": "no unit",
    "white_blood_cell_count": "G/l",
    "neutrophil_count": "G/l",
    "lymphocyte_count": "G/l",
    "CRP": "mg/l",
    "INR": "no unit",
    "fibrinogen": "g/l",
    "d_dimer": "ng/ml",
    "hba1c": "%",
    "alt": "U/l",
    "LDL": "mmol/l",
    "creatinine": "µmol/l",
    "urea": "mmol/l",
    "med_hist_stroke": "binary",
    "med_hist_tia": "binary",
    "med_hist_ich": "binary",
    "med_hist_hta": "binary",
    "med_hist_diabetes": "binary",
    "med_hist_hyperlipidemia": "binary",
    "med_hist_af": "binary",
    "med_hist_coronary_heart_disease": "binary",
    "med_hist_valv_heart_disease": "binary",
    "med_hist_peripheral_artery_disease": "binary",
    "med_hist_smoking": "binary",
    "ODT": "min",
    "ONT": "min",
    "DNT": "min",
    "OPT": "min",
    "IVT": "binary",
    "EVT": "binary",
    # outcomes
    "mrs_3m": "no unit (mRS 0-6; 6 = dead)",
    "death_3m": "binary (1 = death by 3 months)",
    "death_in_hospital": "binary (1 = died during the index admission)",
}


def base_unit(unit: str) -> str:
    """The unit proper, without a documenting '(...)' suffix: 'binary (1 = female)' -> 'binary',
    'no unit (mRS 0-6; 6 = dead)' -> 'no unit'. Use for every unit lookup."""
    return unit.split(" (", 1)[0].strip()


def is_binary_unit(unit: str) -> bool:
    """True for 'binary' and 'binary (...)' — such columns are encoded, never unit-converted."""
    return unit.startswith(BINARY_UNIT_PREFIX)


# Public, data-independent plausibility range per frozen feature, in FROZEN_UNITS — a LITERAL
# MIRROR of fed_stroke.dp.boost.FEATURE_RANGES (the DP bin grid; same keys, same ORDER). Kept
# in both layers because the fed_stroke wheel ships without this package; the equality is
# asserted in architecture/tests/test_preprocess_gva.py. A value outside its range is not
# representable in the DP grid (below lo it falls into the missing bin, above hi into the top
# bin), so these are also the bounds preprocessing.frozen_table.nullify_out_of_range applies.
# Editing a range is a cross-site contract change: change BOTH copies, re-obtain sign-off.
FROZEN_RANGES: dict[str, tuple[float, float]] = {
    "age": (0.0, 120.0),                        # years
    "sex": (0.0, 1.0),                          # binary (1 = female)
    "wake_up_stroke": (0.0, 1.0),               # binary
    "pre_stroke_mrs": (0.0, 5.0),               # mRS 0-5 (6 = dead is impossible pre-stroke)
    "temperature": (25.0, 45.0),                # °C
    "heart_rate": (0.0, 300.0),                 # bpm
    "respiratory_rate": (0.0, 100.0),           # /min
    "systolic_blood_pressure": (0.0, 300.0),    # mmHg
    "diastolic_blood_pressure": (0.0, 200.0),   # mmHg
    "NIHSS": (0.0, 42.0),                       # points
    "glucose": (0.0, 60.0),                     # mmol/L
    "GCS": (3.0, 15.0),                         # points
    "white_blood_cell_count": (0.0, 100.0),     # G/l
    "neutrophil_count": (0.0, 100.0),           # G/l
    "lymphocyte_count": (0.0, 50.0),            # G/l
    "CRP": (0.0, 600.0),                        # mg/l
    "INR": (0.0, 20.0),                         # ratio
    "fibrinogen": (0.0, 20.0),                  # g/l
    "d_dimer": (0.0, 100000.0),                 # ng/ml
    "hba1c": (0.0, 20.0),                       # %
    "alt": (0.0, 50000.0),                      # U/l
    "LDL": (0.0, 15.0),                         # mmol/l
    "creatinine": (0.0, 3000.0),                # µmol/l
    "urea": (0.0, 200.0),                       # mmol/l
    "med_hist_stroke": (0.0, 1.0),              # binary
    "med_hist_tia": (0.0, 1.0),                 # binary
    "med_hist_ich": (0.0, 1.0),                 # binary
    "med_hist_hta": (0.0, 1.0),                 # binary
    "med_hist_diabetes": (0.0, 1.0),            # binary
    "med_hist_hyperlipidemia": (0.0, 1.0),      # binary
    "med_hist_af": (0.0, 1.0),                  # binary
    "med_hist_coronary_heart_disease": (0.0, 1.0),     # binary
    "med_hist_valv_heart_disease": (0.0, 1.0),         # binary
    "med_hist_peripheral_artery_disease": (0.0, 1.0),  # binary
    "med_hist_smoking": (0.0, 1.0),             # binary
    "ODT": (0.0, 10080.0),                      # min — onset-to-door, capped at 7 d
    "ONT": (0.0, 2880.0),                       # min — onset-to-needle, capped at 48 h
    "DNT": (0.0, 10080.0),                      # min — door-to-needle, capped at 7 d
    "OPT": (0.0, 2880.0),                       # min — onset-to-puncture, capped at 48 h
    "IVT": (0.0, 1.0),                          # binary
    "EVT": (0.0, 1.0),                          # binary
}
assert list(FROZEN_RANGES) == FROZEN_FEATURES, "FROZEN_RANGES must list FROZEN_FEATURES in order"
for _name, (_lo, _hi) in FROZEN_RANGES.items():
    assert _lo < _hi, f"FROZEN_RANGES[{_name!r}]: need lo < hi"
    assert not is_binary_unit(FROZEN_UNITS[_name]) or (_lo, _hi) == (0.0, 1.0), \
        f"FROZEN_RANGES[{_name!r}]: binary features are (0, 1)"
del _name, _lo, _hi


# --- Geneva ------------------------------------------------------------------------
# raw column (registry .xlsx or EHR-derived *_first_value, post unit-conversion)
# -> frozen name. Source of truth: frozen_feature_names.xlsx (this dict is its
# geneva_variable_name / frozen_feature_name columns, verbatim). Trailing comments
# are informative only — FROZEN_UNITS is the authoritative unit of record.
GVA_TO_FROZEN: dict[str, str] = {
    "Age (calc.)": "age",  # years
    "Sex": "sex",  # binary
    "wake_up_stroke": "wake_up_stroke",  # binary
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
    # Outcomes — not rows in frozen_feature_names.xlsx (features only). The registry's
    # OPSUM-reconciled outcome fields (registry_cohort.preprocess_outcome), carried as
    # columns; NEVER used to drop rows here (the label is chosen at training time).
    "3M mRS": "mrs_3m",  # no unit (mRS 0-6)
    "3M Death": "death_3m",  # binary
    "Death in hospital": "death_in_hospital",  # binary
}

# Import-time contract checks — a typo here is a cross-site bug, so fail at import.
assert len(FROZEN_FEATURES) == 41 == len(set(FROZEN_FEATURES)), "FROZEN_FEATURES: 41 unique names"
assert not set(FROZEN_FEATURES) & set(FROZEN_OUTCOMES), "an outcome is never also a feature"
assert set(FROZEN_UNITS) == {*FROZEN_FEATURES, *FROZEN_OUTCOMES}, "FROZEN_UNITS must cover the schema"
assert set(FROZEN_OUTCOME_RANGES) == set(FROZEN_OUTCOMES), "every outcome declares its value range"
assert set(GVA_TO_FROZEN.values()) == {*FROZEN_FEATURES, *FROZEN_OUTCOMES}, \
    "GVA_TO_FROZEN must produce exactly FROZEN_FEATURES + FROZEN_OUTCOMES"

# --- Shenzhen ----------------------------------------------------------------------
# raw Shenzhen export column -> frozen name. Source of truth:
# frozen_feature_names.xlsx (shenzhen_variable_name / frozen_feature_name columns,
# verbatim); run remotely by the partner's preprocessing, never on Geneva
# infrastructure. Trailing comments are informative only — FROZEN_UNITS is authoritative.
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
    # TODO(shenzhen sign-off): outcome sources for "mrs_3m" (mRS 0-6 at 3 months),
    # "death_3m" (binary) and "death_in_hospital" (binary) — carried as columns, never
    # used to drop rows; the label is chosen in fed_stroke.schema at training time.
}


def validate_frozen_columns(columns, frozen_features, outcome_cols,
                            id_col: str = "case_admission_id") -> None:
    """Assert a produced table matches the frozen contract EXACTLY.

    columns: the produced DataFrame's columns (post-rename).
    frozen_features / outcome_cols: pass FROZEN_FEATURES / FROZEN_OUTCOMES (identical to
    fed_stroke.schema.FEATURE_COLS / OUTCOME_COLS — asserted by the architecture tests).
    Raises ValueError naming both the missing and the unexpected columns —
    a stray raw column is as much a schema bug as a missing feature (it would
    ride through the loader's split frames into out/).
    """
    produced = set(columns) - {id_col}
    expected = {*frozen_features, *outcome_cols}
    if produced == expected and id_col in columns:
        return
    missing = sorted(expected - produced) + ([id_col] if id_col not in columns else [])
    raise ValueError(
        f"frozen-schema mismatch: missing={missing} unexpected={sorted(produced - expected)}"
    )
