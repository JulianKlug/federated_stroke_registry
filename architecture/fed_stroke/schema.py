"""fed_stroke: the frozen cross-site data contract, and the training label chosen on top of it.

FEATURE_COLS / OUTCOME_COLS / FEATURE_UNITS are a LITERAL MIRROR of the shared preprocessing
layer's frozen schema — `preprocessing/mappings/frozen_schema.py` (FROZEN_FEATURES /
FROZEN_OUTCOMES / FROZEN_UNITS), itself the verbatim copy of frozen_feature_names.xlsx. They
are duplicated here on purpose: the fed_stroke wheel is built and shipped alone (hatch packages
= ["fed_stroke"]) and runs in the Shenzhen container without the root `preprocessing` package.
tests/test_preprocess_gva.py asserts the two copies are identical, and
architecture/preprocessing/preprocess_gva.py warns loudly at build time if they drift.

The node parquet carries every cohort admission with ALL outcome columns (NaN when not
recorded). The LABEL is decided HERE and nowhere else: TARGET_COL names one frozen outcome
column, TARGET_RULE optionally binarizes it (an mRS cut-off), and task.select_labelled_rows
derives the label and drops the unlabelled rows once at load time. Changing the label is a
change to this file, never a re-run of a site's preprocessing.

Column ORDER is part of the contract: task.load_data_arrays emits X in FEATURE_COLS order and
dp/boost.FEATURE_RANGES (asserted at import to carry the same keys in the same order) defines
the DP bin grid per column index. Reordering silently remaps every persisted DP model.

Freeze discipline: any change to the mirrored lists is a cross-site contract change — bump
SCHEMA_VERSION here and DP_MODEL_FORMAT in dp/boost.py, and re-obtain partner sign-off.
"""

# Stamped into the node parquet metadata by preprocess_gva.write_node_parquet and compared by
# the loader-side tooling. "frozen-v1" is the FIRST real frozen schema (41 features + 3 outcome
# columns, 2026-09); the earlier 2-feature placeholder (Age (calc.), NIH on admission) was
# never stamped anywhere.
SCHEMA_VERSION = "frozen-v1"

FEATURE_COLS = [
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

# Standardized raw outcome variables every site delivers as columns (mirror of FROZEN_OUTCOMES).
# Not labels: NaN when not recorded, no row dropped by the preprocessing.
OUTCOME_COLS = [
    "mrs_3m",             # modified Rankin Scale at 3 months, ordinal 0-6 (6 = dead)
    "death_3m",           # dead by 3 months (OPSUM-reconciled with mrs_3m / in-hospital death)
    "death_in_hospital",  # died during the index admission
]

# --- The training label (the ONLY place it is defined) -------------------------------------
# TARGET_COL: which frozen outcome column. TARGET_RULE: None when the column is already binary
# {0, 1}; otherwise a callable Series -> boolean/0-1 Series applied to the RECORDED values
# (NaN stays NaN and the row is dropped at load). task.derive_label validates the result.
# Examples:   TARGET_COL, TARGET_RULE = "death_3m", None            # death by 3 months (current)
#             TARGET_COL, TARGET_RULE = "mrs_3m", mrs_at_most(2)    # good functional outcome
#             TARGET_COL, TARGET_RULE = "death_in_hospital", None
# The roadmap's "pick the primary label" decision lands here; every arm (A/B/C) reads it.
TARGET_COL = "death_3m"
TARGET_RULE = None


def mrs_at_most(k: int):
    """Label rule for an mRS outcome: 1 if mRS <= k, else 0 (recorded values only)."""
    def rule(mrs):
        return mrs <= k
    rule.__name__ = f"mrs_at_most_{k}"
    return rule


def label_id() -> str:
    """Short identifier of the label in force ('death_3m', 'mrs_3m|mrs_at_most_2'), for logs
    and artifact stamps."""
    return TARGET_COL if TARGET_RULE is None else f"{TARGET_COL}|{TARGET_RULE.__name__}"


# Unit of record per frozen column — part of the contract, not documentation: every site's
# preprocessing must deliver values IN these units (unit conversions aim here), and
# dp/boost.FEATURE_RANGES is expressed in them — a feature arriving in other units silently
# lands in the wrong DP bins. 'binary (...)' / 'no unit (...)' document an encoding convention
# both sites must reproduce (sex: 1 = female). Mirrors preprocessing.mappings.frozen_schema.
# FROZEN_UNITS, outcomes included.
FEATURE_UNITS = {
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
assert len(FEATURE_COLS) == 41 == len(set(FEATURE_COLS)), "FEATURE_COLS: 41 unique frozen names"
assert not set(FEATURE_COLS) & set(OUTCOME_COLS), "an outcome is never also a feature"
assert set(FEATURE_UNITS) == {*FEATURE_COLS, *OUTCOME_COLS}, "FEATURE_UNITS must cover the schema"
assert TARGET_COL in OUTCOME_COLS, "the label must be one of the frozen outcome columns"
assert TARGET_RULE is None or callable(TARGET_RULE), "TARGET_RULE is None or a Series -> {0,1} callable"


def is_binary_feature(name: str) -> bool:
    """True for columns whose unit of record is 'binary' / 'binary (...)' (encoded {0, 1})."""
    return FEATURE_UNITS[name].startswith("binary")


# --- Missingness policy: variant 1, sentinel / missing bin (decided 2026-07-23) --------------
# REMOVE-IF-NO-DP: this constant and every use of it (grep for REMOVE-IF-NO-DP) exist ONLY
# because the DP learner's fixed public bins cannot represent NaN and the R7 precondition gate
# refuses non-finite features. Missing feature values are encoded at the loader as this PUBLIC
# out-of-range sentinel, and fixed_bin_edges reserves bin 0 for it, so "not recorded" is an
# explicit ordinary value that ALL comparator arms (stock XGBoost included) see identically —
# a DP-arm-only policy would confound the A→B→C decomposition with a missingness-encoding
# difference. If the project later moves forward WITHOUT DP, remove the sentinel encoding and
# the reserved bin, and let XGBoost's native NaN handling (learned per-split default
# direction) take over.
# Must stay strictly below every FEATURE_RANGES lower bound (fixed_bin_edges asserts this).
MISSING_SENTINEL = -1.0
