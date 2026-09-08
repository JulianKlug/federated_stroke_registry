"""Frozen-table tail of the GVA preprocessing: unit-convert → encode → rename → validate.

Synthetic frames only — no real registry / EHR data. Import mechanics: pytest's
pythonpath=["."] puts architecture/ on sys.path (fed_stroke); the repo root gives the shared
`preprocessing` package (a regular package, so it wins over the architecture/preprocessing
namespace dir whatever the path order); architecture/preprocessing/ gives preprocess_gva.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCH_DIR.parent
for _p in (REPO_ROOT, ARCH_DIR / "preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocessing.anonymise import ANONYMISATION_SPEC, anonymise_frozen  # noqa: E402
from preprocessing.build_log import format_build_log, record_exclusion, write_build_log  # noqa: E402
from preprocessing.case_ids import RAW_ID_COL  # noqa: E402
from preprocessing.registry_cohort import build_cohort  # noqa: E402
from preprocessing.frozen_table import (  # noqa: E402
    convert_to_frozen_units,
    encode_binaries,
    finalize_frozen_table,
    nullify_out_of_range,
    rename_to_frozen,
    select_mapped_columns,
)
from preprocessing.mappings import (  # noqa: E402
    FROZEN_FEATURES,
    FROZEN_OUTCOME_RANGES,
    FROZEN_OUTCOMES,
    FROZEN_RANGES,
    FROZEN_UNITS,
    GVA_BINARY_ENCODINGS,
    GVA_BINARY_FILLNA,
    GVA_DECLARED_UNITS,
    GVA_PRE_ENCODED_BINARIES,
    GVA_TO_FROZEN,
    ID_COL,
    SHENZHEN_TO_FROZEN,
    UNIT_ALIASES,
    UNITS,
    base_unit,
    is_binary_unit,
    normalize_unit_label,
    validate_frozen_columns,
)
import preprocess_gva  # noqa: E402  (architecture/preprocessing/preprocess_gva.py)

XLSX = REPO_ROOT / "preprocessing" / "mappings" / "frozen_feature_names.xlsx"
RAW_ID = RAW_ID_COL          # the raw frame (wide_to_frozen output)
ID = ID_COL                  # the de-identified node table (anonymise_frozen output)
TEST_KEY = bytes(range(32))
VALUE, UNIT = "_first_value", "_first_unit"

# Canonical GVA EHR unit label per *_first_value stem, as the EHR writes them.
EHR_LABELS = {
    "temperature": "°C", "pulse": "puls./min.", "fr": "cycles/min.",
    "globules_blancs": "G/l", "neutrophiles_nb_abs": "G/l", "lymphocytes_nb_abs": "G/l",
    "crp": "mg/l", "inr": None, "fibrinogene": "g/l", "d_dimeres": "ng/ml", "hba1c": "%",
    "alat": "U/l", "ldl_calc": "mmol/l", "creatinine": "µmol/l", "uree": "mmol/l",
}


def _ehr_frame(stem: str, values, labels) -> pd.DataFrame:
    return pd.DataFrame({f"{stem}{VALUE}": values, f"{stem}{UNIT}": labels})


def _convert(df, mapping):
    return convert_to_frozen_units(df, mapping, FROZEN_UNITS, GVA_DECLARED_UNITS, UNIT_ALIASES)


ALL_RANGES = {**FROZEN_RANGES, **FROZEN_OUTCOME_RANGES}


def _in_range(rng, frozen: str, n: int) -> np.ndarray:
    """Values in the central half of the column's declared range (feature or outcome)."""
    lo, hi = ALL_RANGES[frozen]
    return rng.uniform(lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo), n)


def _in_range_frozen(n: int = 4) -> pd.DataFrame:
    """A frozen-shaped frame with every feature inside its range (binaries 1.0, others midpoint)
    and the three outcome columns recorded."""
    df = pd.DataFrame({RAW_ID: [f"r_{i}" for i in range(n)]})
    for f in FROZEN_FEATURES:
        lo, hi = FROZEN_RANGES[f]
        df[f] = 1.0 if is_binary_unit(FROZEN_UNITS[f]) else (lo + hi) / 2
    df["mrs_3m"], df["death_3m"], df["death_in_hospital"] = 2.0, 0.0, 0.0
    return df


def _fake_wide_df() -> pd.DataFrame:
    """5-row stand-in for assemble_wide's output: every GVA_TO_FROZEN raw column, its EHR
    siblings (values inside the frozen plausibility ranges), hand-set rows the tests inspect,
    and strays that must vanish."""
    n = 5
    rng = np.random.default_rng(0)
    data = {RAW_ID: [f"p{i}_000{i}" for i in range(n)]}
    for raw, frozen in GVA_TO_FROZEN.items():
        if raw in GVA_BINARY_ENCODINGS:
            keys = list(GVA_BINARY_ENCODINGS[raw])
            data[raw] = [keys[i % len(keys)] for i in range(n)]
        elif raw in GVA_PRE_ENCODED_BINARIES:
            data[raw] = [i % 2 for i in range(n)]
        elif raw.endswith(VALUE):
            stem = raw[: -len(VALUE)]
            vals = _in_range(rng, frozen, n).round(2)
            vals[-1] = np.nan
            data[raw] = vals
            data[f"{stem}{UNIT}"] = [EHR_LABELS[stem]] * n
            data[f"{stem}_first_datetime"] = [pd.Timestamp("2020-01-01")] * n
        else:
            data[raw] = _in_range(rng, frozen, n).round(1)
    df = pd.DataFrame(data)
    df["Sex"] = ["Female", "Male", None, "Female", "Male"]
    df["IVT with rtPA"] = ["started before admission", "no", "yes", "no", "no"]
    df["MedHist Prost. heart valves"] = [None, "Biological", None, "Mechanical", None]
    df.loc[1, f"d_dimeres{VALUE}"] = 0.5          # mg/L row -> 500 ng/ml
    df.loc[1, f"d_dimeres{UNIT}"] = "mg/L"
    df.loc[2, f"pulse{UNIT}"] = None               # value without a label -> assumed bpm
    # outcomes: carried, never used to drop a row (row 3 has no 3-month outcome)
    df["3M Death"] = ["yes", "no", "yes", None, "no"]
    df["Death in hospital"] = ["no", "no", "yes", "no", "no"]
    df["3M mRS"] = [1.0, 6.0, 3.0, np.nan, 0.0]
    # strays: PII, intermediates, non-frozen labs
    df["Last name"] = "Doe"
    df["DOB"] = "1950-01-01"
    df["ZIP"] = "1200"
    df["admission_date"] = pd.Timestamp("2020-01-01")
    df["DPT"] = 30.0
    df["cystatine_c_first_value"] = 1.0
    df["cystatine_c_first_unit"] = "mg/l"
    return df


def _small_frozen(n=3) -> pd.DataFrame:
    df = pd.DataFrame({"death_3m": [1] * n, RAW_ID: [f"a_{i}" for i in range(n)],
                       "mrs_3m": [6.0] * n, "death_in_hospital": [0] * n})
    for f in FROZEN_FEATURES:
        df[f] = 1.0
    df["wake_up_stroke"] = 1   # int, must become float64
    return df


# ---------------------------------------------------------------- data-table consistency

def test_xlsx_matches_frozen_features_order():
    xlsx = pd.read_excel(XLSX, sheet_name="with_overlap")
    assert xlsx["frozen_feature_name"].tolist() == FROZEN_FEATURES


def test_xlsx_matches_frozen_units():
    xlsx = pd.read_excel(XLSX, sheet_name="with_overlap")
    for name, unit in zip(xlsx["frozen_feature_name"], xlsx["frozen_feature_unit"]):
        ours = FROZEN_UNITS[name]
        assert ours == unit or ours.startswith(unit + " ("), (name, unit, ours)
    assert FROZEN_UNITS["sex"] == "binary (1 = female)"
    assert "µ" in FROZEN_UNITS["creatinine"]   # MICRO SIGN, not Greek mu


def test_xlsx_matches_gva_to_frozen():
    xlsx = pd.read_excel(XLSX, sheet_name="with_overlap")
    pairs = list(zip(xlsx["geneva_variable_name"], xlsx["frozen_feature_name"]))
    # wake_up_stroke is DERIVED (registry_cohort.preprocess_features) from the xlsx's raw
    # 'Time of symptom onset known' source, so GVA_TO_FROZEN keys the derived column.
    expected = {str(g).strip(): f for g, f in pairs if f != "wake_up_stroke"}
    ours = {k: v for k, v in GVA_TO_FROZEN.items()
            if v not in FROZEN_OUTCOMES and v != "wake_up_stroke"}
    assert ours == expected
    assert GVA_TO_FROZEN["wake_up_stroke"] == "wake_up_stroke"
    (wake_src,) = [g for g, f in pairs if f == "wake_up_stroke"]
    assert str(wake_src).strip() in ("wake_up_stroke", "Time of symptom onset known")
    # the outcomes are not xlsx rows; they map the OPSUM-reconciled registry fields
    assert GVA_TO_FROZEN["3M Death"] == "death_3m"
    assert GVA_TO_FROZEN["3M mRS"] == "mrs_3m"
    assert GVA_TO_FROZEN["Death in hospital"] == "death_in_hospital"


def test_frozen_outcomes_are_columns_not_labels():
    assert FROZEN_OUTCOMES == ["mrs_3m", "death_3m", "death_in_hospital"]
    assert not set(FROZEN_OUTCOMES) & set(FROZEN_FEATURES)
    assert set(FROZEN_OUTCOME_RANGES) == set(FROZEN_OUTCOMES)
    assert base_unit(FROZEN_UNITS["mrs_3m"]) == "no unit"
    assert is_binary_unit(FROZEN_UNITS["death_3m"]) and is_binary_unit(FROZEN_UNITS["death_in_hospital"])
    assert base_unit("binary (1 = female)") == "binary" and base_unit("years") == "years"


def test_shenzhen_mapping_targets_frozen_names():
    assert set(SHENZHEN_TO_FROZEN.values()) <= set(FROZEN_FEATURES)


def test_unit_alias_tables_cover_frozen_units():
    for name in (*FROZEN_FEATURES, *FROZEN_OUTCOMES):
        unit = FROZEN_UNITS[name]
        if is_binary_unit(unit):
            continue
        key = normalize_unit_label(base_unit(unit))
        table = UNIT_ALIASES[key]
        assert table[key] == 1.0
        assert all(np.isfinite(f) and f > 0 for f in table.values())


def test_declared_units_resolve():
    for raw, unit in GVA_DECLARED_UNITS.items():
        assert unit == UNITS[raw]
        frozen_unit = FROZEN_UNITS[GVA_TO_FROZEN[raw]]
        table = UNIT_ALIASES[normalize_unit_label(base_unit(frozen_unit))]
        assert normalize_unit_label(unit) in table, (raw, unit)


def test_binary_encodings_cover_all_binary_columns():
    binary_raw = {raw for raw, fz in GVA_TO_FROZEN.items() if is_binary_unit(FROZEN_UNITS[fz])}
    assert binary_raw == set(GVA_BINARY_ENCODINGS) | set(GVA_PRE_ENCODED_BINARIES)
    assert {"3M Death", "Death in hospital"} <= set(GVA_BINARY_ENCODINGS)   # binary outcomes too
    assert set(GVA_BINARY_FILLNA) <= set(GVA_BINARY_ENCODINGS)
    for raw in GVA_BINARY_ENCODINGS:          # no numeric column carries an encoding
        assert is_binary_unit(FROZEN_UNITS[GVA_TO_FROZEN[raw]])


@pytest.mark.parametrize("raw, norm", [
    ("puls./min.", "puls./min"),
    ("cycles/min.", "cycles/min"),
    ("mmol/L", "mmol/l"),
    ("ng/mL", "ng/ml"),
    ("G/L", "G/l"),
    ("g/l", "g/l"),
    ("μmol/l", "µmol/l"),
    ("  °C ", "°C"),
    ("NIHSS  points", "NIHSS points"),
])
def test_normalize_unit_label(raw, norm):
    assert normalize_unit_label(raw) == norm


def test_normalize_keeps_giga_vs_gram_distinct():
    assert normalize_unit_label("G/l") != normalize_unit_label("g/l")


# ---------------------------------------------------------------- convert_to_frozen_units

def test_convert_per_row_factor_and_assumed():
    df = _ehr_frame("d_dimeres", [500.0, 0.5, 700.0, np.nan], ["ng/ml", "mg/L", None, "ng/ml"])
    out, rep = _convert(df, {"d_dimeres_first_value": "d_dimer"})
    assert out["d_dimeres_first_value"].tolist()[:3] == [500.0, 500.0, 700.0]
    assert np.isnan(out["d_dimeres_first_value"].iloc[3])
    c = rep["columns"]["d_dimer"]
    assert c["unit_source"] == "per_row"
    assert (c["n_values"], c["n_assumed_unit"], c["n_converted_by_factor"]) == (3, 1, 1)
    assert c["labels_seen"] == {"ng/ml": 1, "mg/L": 1}
    assert c["factors_applied"] == {"mg/L": 1000.0}
    assert rep["pass"] is True and rep["n_columns_checked"] == 1
    json.dumps(rep)


def test_convert_accepts_pulse_synonyms():
    df = _ehr_frame("pulse", [60.0, 70.0, 80.0], ["puls./min.", "/min", "bpm"])
    out, rep = _convert(df, {"pulse_first_value": "heart_rate"})
    assert out["pulse_first_value"].tolist() == [60.0, 70.0, 80.0]
    assert rep["columns"]["heart_rate"]["n_converted_by_factor"] == 0
    assert rep["warnings"] == []


def test_convert_unknown_label_raises():
    df = _ehr_frame("d_dimeres", [1.0], ["FEU µg/ml"])
    with pytest.raises(ValueError, match=r"d_dimeres_first_value.*FEU µg/ml"):
        _convert(df, {"d_dimeres_first_value": "d_dimer"})
    # an unknown label on a row WITHOUT a value is irrelevant
    df = _ehr_frame("d_dimeres", [np.nan], ["bogus"])
    _, rep = _convert(df, {"d_dimeres_first_value": "d_dimer"})
    assert rep["columns"]["d_dimer"]["n_values"] == 0


def test_convert_declared_unit_path():
    df = pd.DataFrame({"NIH on admission": [12.0, np.nan]})
    out, rep = _convert(df, {"NIH on admission": "NIHSS"})
    assert out["NIH on admission"].tolist()[0] == 12.0
    c = rep["columns"]["NIHSS"]
    assert c["unit_source"] == "declared"
    assert c["labels_seen"] == {"NIHSS points": 1}
    assert rep["warnings"] == []


def test_convert_no_unit_source_raises():
    df = pd.DataFrame({"foo_first_value": [1.0]})
    with pytest.raises(ValueError, match="no unit source"):
        _convert(df, {"foo_first_value": "CRP"})


def test_convert_skips_binaries_and_checks_unitless_outcome():
    df = pd.DataFrame({"Sex": ["Female"], "3M Death": ["yes"]})
    out, rep = _convert(df, {"Sex": "sex", "3M Death": "death_3m"})
    assert rep["columns"] == {}
    assert out["Sex"].tolist() == ["Female"]
    # the ordinal outcome is unitless: declared 'mRS' resolves via the base unit 'no unit'
    _, rep = _convert(pd.DataFrame({"3M mRS": [3.0, np.nan]}), {"3M mRS": "mrs_3m"})
    assert rep["columns"]["mrs_3m"]["labels_seen"] == {"mRS": 1} and rep["warnings"] == []


def test_convert_warning_for_dimensional_assumed_only():
    df = _ehr_frame("pulse", [70.0, 80.0], ["puls./min.", None])
    _, rep = _convert(df, {"pulse_first_value": "heart_rate"})
    assert len(rep["warnings"]) == 1 and "heart_rate" in rep["warnings"][0]
    df = _ehr_frame("inr", [1.0, 1.1], [None, None])
    _, rep = _convert(df, {"inr_first_value": "INR"})
    assert rep["warnings"] == []
    assert rep["columns"]["INR"]["n_assumed_unit"] == rep["columns"]["INR"]["n_values"] == 2


def test_convert_reports_all_errors_at_once():
    df = pd.concat([_ehr_frame("a", [1.0], ["bogus"]), _ehr_frame("b", [1.0], ["bogus2"])], axis=1)
    with pytest.raises(ValueError) as exc:
        _convert(df, {"a_first_value": "CRP", "b_first_value": "urea"})
    assert "a_first_value" in str(exc.value) and "b_first_value" in str(exc.value)


# ---------------------------------------------------------------- binaries + target

def test_encode_binaries_sex_ivt_valve():
    df = pd.DataFrame({
        "Sex": ["Female", "Male", None],
        "IVT with rtPA": ["started before admission", "yes", "no"],
        "MedHist Prost. heart valves": [None, "Biological", "Mechanical"],
        "MedHist Stroke": ["yes", None, "no"],
    })
    out = encode_binaries(df, {k: GVA_BINARY_ENCODINGS[k] for k in df.columns}, fillna=GVA_BINARY_FILLNA)
    assert out["Sex"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["Sex"].iloc[2])
    assert out["IVT with rtPA"].tolist() == [1.0, 1.0, 0.0]
    assert out["MedHist Prost. heart valves"].tolist() == [0.0, 1.0, 1.0]
    assert out["MedHist Stroke"].tolist()[0] == 1.0 and np.isnan(out["MedHist Stroke"].iloc[1])
    assert all(out[c].dtype == "float64" for c in df.columns)


def test_encode_binaries_unknown_value_raises():
    with pytest.raises(ValueError, match=r"MedHist Stroke.*'unknown'"):
        encode_binaries(pd.DataFrame({"MedHist Stroke": ["yes", "unknown"]}),
                        {"MedHist Stroke": GVA_BINARY_ENCODINGS["MedHist Stroke"]})
    with pytest.raises(ValueError, match="Sex"):   # already numeric -> encode once, loudly
        encode_binaries(pd.DataFrame({"Sex": [1.0, 0.0]}), {"Sex": GVA_BINARY_ENCODINGS["Sex"]})


def test_encode_binaries_missing_column_raises():
    with pytest.raises(ValueError, match="absent"):
        encode_binaries(pd.DataFrame({"x": [1]}), {"Sex": GVA_BINARY_ENCODINGS["Sex"]})


def test_binary_outcomes_encode_like_features_and_keep_nan():
    df = pd.DataFrame({"3M Death": ["yes", "no", None], "Death in hospital": ["no", None, "yes"]})
    out = encode_binaries(df, {c: GVA_BINARY_ENCODINGS[c] for c in df.columns})
    assert out["3M Death"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["3M Death"].iloc[2])
    assert np.isnan(out["Death in hospital"].iloc[1]) and out["Death in hospital"].iloc[2] == 1.0
    assert len(out) == 3                                   # nothing dropped
    with pytest.raises(ValueError, match="unknown"):
        encode_binaries(pd.DataFrame({"3M Death": ["unknown"]}), {"3M Death": GVA_BINARY_ENCODINGS["3M Death"]})


# ---------------------------------------------------------------- rename / select / validate / finalize

def test_rename_to_frozen_fails_on_missing_raw():
    with pytest.raises(ValueError, match=r"absent.*\['b'\]"):
        rename_to_frozen(pd.DataFrame({"a": [1]}), {"a": "x", "b": "y"})


def test_rename_to_frozen_fails_on_duplicate_result():
    with pytest.raises(ValueError, match="duplicate"):
        rename_to_frozen(pd.DataFrame({"a": [1], "b": [2]}), {"a": "x", "b": "x"})


def test_select_and_validate_catch_stray_and_missing():
    df = _fake_wide_df()
    # (a) a mapping that outruns the frozen list -> unexpected
    df["Height"] = 170.0
    m = {**GVA_TO_FROZEN, "Height": "height"}
    out = select_mapped_columns(rename_to_frozen(df, m), m)
    with pytest.raises(ValueError, match=r"unexpected=\['height'\]"):
        validate_frozen_columns(out.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID)
    # (b) a frozen name the mapping never produces -> missing (a feature, or an outcome)
    m = {k: v for k, v in GVA_TO_FROZEN.items() if k != "IAT"}
    out = select_mapped_columns(rename_to_frozen(df, m), m)
    with pytest.raises(ValueError, match=r"missing=\['EVT'\]"):
        validate_frozen_columns(out.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID)
    m = {k: v for k, v in GVA_TO_FROZEN.items() if k != "3M mRS"}
    out = select_mapped_columns(rename_to_frozen(df, m), m)
    with pytest.raises(ValueError, match=r"missing=\['mrs_3m'\]"):
        validate_frozen_columns(out.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID)
    # (c) strays (PII, intermediates, non-frozen labs) are gone after select
    out = select_mapped_columns(rename_to_frozen(df, GVA_TO_FROZEN), GVA_TO_FROZEN)
    assert not ({"Last name", "DOB", "ZIP", "admission_date", "DPT", "cystatine_c_first_value",
                 "pulse_first_unit", "pulse_first_datetime"} & set(out.columns))
    validate_frozen_columns(out.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID)


def _finalize(df):
    return finalize_frozen_table(df, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID,
                                 outcome_ranges=FROZEN_OUTCOME_RANGES)


def test_finalize_dtypes_order_and_uniqueness():
    out = _finalize(_small_frozen())
    assert out.columns.tolist() == [RAW_ID, *FROZEN_FEATURES, *FROZEN_OUTCOMES]
    assert all(out[c].dtype == "float64" for c in (*FROZEN_FEATURES, *FROZEN_OUTCOMES))
    assert pd.api.types.is_string_dtype(out[RAW_ID])

    # a missing outcome is a value, not a reason to drop the row
    with_nan = _small_frozen()
    with_nan.loc[0, "death_3m"] = np.nan
    out = _finalize(with_nan)
    assert len(out) == 3 and np.isnan(out.loc[0, "death_3m"])

    dup = _small_frozen()
    dup.loc[1, RAW_ID] = dup.loc[0, RAW_ID]
    with pytest.raises(ValueError, match="not unique"):
        _finalize(dup)

    bad_outcome = _small_frozen()
    bad_outcome.loc[0, "mrs_3m"] = 7.0                     # outside the declared 0-6
    with pytest.raises(ValueError, match=r"'mrs_3m'.*outside its declared range"):
        _finalize(bad_outcome)
    bad_outcome = _small_frozen()
    bad_outcome.loc[0, "death_3m"] = 2
    with pytest.raises(ValueError, match=r"'death_3m'.*outside its declared range"):
        _finalize(bad_outcome)

    bad_feature = _small_frozen().astype({"age": object})
    bad_feature.loc[0, "age"] = "old"
    with pytest.raises(ValueError, match="'age' is not numeric"):
        _finalize(bad_feature)


# ---------------------------------------------------------------- plausibility ranges

def test_nullify_out_of_range_nulls_and_counts():
    df = _in_range_frozen(4)
    df.loc[0, "diastolic_blood_pressure"] = 889.0     # entry error, above hi
    df.loc[1, "temperature"] = 5.2                    # below lo
    df.loc[2, "ODT"] = -30.0                          # negative timing
    df.loc[3, "GCS"] = np.nan                         # missing stays missing, never counted
    out, rep = nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES, max_out_of_range_frac=0.5)
    assert np.isnan(out.loc[0, "diastolic_blood_pressure"])
    assert np.isnan(out.loc[1, "temperature"]) and np.isnan(out.loc[2, "ODT"])
    assert out.loc[1, "diastolic_blood_pressure"] == df.loc[1, "diastolic_blood_pressure"]  # in-range untouched
    assert rep["pass"] is True and rep["n_nulled_total"] == 3
    c = rep["columns"]
    assert (c["diastolic_blood_pressure"]["n_above"], c["diastolic_blood_pressure"]["n_below"]) == (1, 0)
    assert (c["temperature"]["n_below"], c["ODT"]["n_below"]) == (1, 1)
    assert c["diastolic_blood_pressure"]["max_seen"] == 889.0
    assert c["GCS"]["n_values"] == 3 and c["GCS"]["n_below"] == c["GCS"]["n_above"] == 0
    assert all(c[f]["n_below"] == c[f]["n_above"] == 0 for f in FROZEN_FEATURES
               if f not in ("diastolic_blood_pressure", "temperature", "ODT"))
    assert set(c) == set(FROZEN_FEATURES)
    json.dumps(rep)
    # the outcomes and the id are not features and are untouched
    assert out[FROZEN_OUTCOMES].equals(df[FROZEN_OUTCOMES]) and out[RAW_ID].equals(df[RAW_ID])


def test_nullify_out_of_range_fails_on_mass_out_of_range():
    df = _in_range_frozen(100)
    df.loc[:29, "CRP"] = 1e6                           # 30 % of CRP off by orders of magnitude
    with pytest.raises(ValueError, match=r"'CRP'.*30/100.*suspected unit or mapping bug"):
        nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES)     # default gate 25 %
    out, rep = nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES, max_out_of_range_frac=0.5)
    assert out["CRP"].isna().sum() == 30 and rep["columns"]["CRP"]["n_above"] == 30
    # below the gate the values are nulled and counted, never raised
    df = _in_range_frozen(100)
    df.loc[:9, "CRP"] = 1e6                            # 10 %: entry-error territory
    out, rep = nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES)
    assert out["CRP"].isna().sum() == 10 and rep["pass"] is True


def test_nullify_out_of_range_reports_all_errors_and_missing_range():
    df = _in_range_frozen(10)
    df["CRP"] = 1e6
    df["urea"] = -5.0
    with pytest.raises(ValueError) as exc:
        nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES)
    assert "'CRP'" in str(exc.value) and "'urea'" in str(exc.value)
    ranges = {k: v for k, v in FROZEN_RANGES.items() if k != "LDL"}
    with pytest.raises(ValueError, match="'LDL': no plausibility range"):
        nullify_out_of_range(_in_range_frozen(3), ranges, FROZEN_FEATURES)


def test_wide_to_frozen_fails_on_feature_out_of_range_wholesale():
    df = _fake_wide_df()
    df["1st diast. bp"] = 889.0                       # every row implausible -> unit/mapping bug
    with pytest.raises(ValueError, match="diastolic_blood_pressure"):
        preprocess_gva.wide_to_frozen(df)


# ---------------------------------------------------------------- build log

def test_record_exclusion_counts_rows_and_patients():
    before = pd.DataFrame({ID: ["p1_1", "p1_2", "p2_1", "p3_1"], "x": [1, 2, 3, 4]})
    after = before.iloc[[0, 3]]                       # p1 keeps one admission, p2 loses its only one
    log: list[dict] = []
    e = record_exclusion(log, "stage A", "because", before, after)
    assert log == [e]
    assert (e["rows_before"], e["rows_excluded"], e["rows_after"]) == (4, 2, 2)
    assert (e["patients_before"], e["patients_excluded"], e["patients_after"]) == (3, 1, 2)
    # a frame without a patient key still counts rows
    e2 = record_exclusion(log, "stage B", "why", pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1]}))
    assert e2["rows_excluded"] == 1 and e2["patients_before"] is None and e2["patients_excluded"] is None


def test_build_cohort_records_exclusion_stages():
    # Case ID = 8-char prefix + patient id + 4-digit EDS (case_ids.build_case_admission_id)
    def cid(patient, eds):
        return f"20180101{patient}{eds}"
    raw = pd.DataFrame({
        "Case ID": [cid("00001", "0001"), cid("00001", "0001"),   # exact duplicate row
                    cid("00002", "0002"),                          # flagged duplicate
                    cid("00003", "0003"),                          # TIA (patient 00003 also below)
                    cid("00004", "0004"), cid("00004", "0004"),    # same Case ID, conflicting NIHSS
                    cid("00003", "0005")],                         # ischemic stroke, kept
        "Type of event": ["Ischemic stroke", "Ischemic stroke", "duplicate", "TIA",
                          "Ischemic stroke", "Ischemic stroke", "Ischemic stroke"],
        "NIH on admission": [3, 3, 1, 0, 5, 7, 2],
    })
    log: list[dict] = []
    df, n_raw, n_filtered = build_cohort(raw, exclusions=log)
    assert (n_raw, n_filtered, len(df)) == (7, 3, 3)
    assert [e["stage"] for e in log] == ["exact duplicate registry rows",
                                         "rows flagged 'duplicate' in Type of event",
                                         "not an ischemic stroke", "same Case ID entered twice"]
    rows = [(e["rows_before"], e["rows_excluded"], e["rows_after"]) for e in log]
    assert rows == [(7, 1, 6), (6, 1, 5), (5, 1, 4), (4, 1, 3)]
    pts = [(e["patients_before"], e["patients_excluded"], e["patients_after"]) for e in log]
    assert pts == [(4, 0, 4), (4, 1, 3), (3, 0, 3), (3, 0, 3)]   # the TIA drop excludes a row, not a patient
    assert all(e["reason"] for e in log)
    # the plain call path (registry_alignement) is unchanged
    df2, n_raw2, n_filtered2 = build_cohort(raw)
    assert (n_raw2, n_filtered2) == (7, 3) and df2.equals(df)


def test_format_build_log_lists_stages_and_nulled_features():
    log: list[dict] = []
    record_exclusion(log, "stage A", "reason A", pd.DataFrame({ID: ["p1_1", "p2_1"]}),
                     pd.DataFrame({ID: ["p1_1"]}))
    df = _in_range_frozen(4)
    df.loc[0, "diastolic_blood_pressure"] = 889.0
    df.loc[1, "ODT"] = -30.0
    _, oor = nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES, max_out_of_range_frac=0.5)
    outcomes = {"mrs_3m": {"n_recorded": 3, "n_missing": 1, "median": 2.0},
                "death_3m": {"n_recorded": 3, "n_missing": 1, "n_positive": 1}}
    text = format_build_log("T", {"date": "d", "output": "o"}, log, oor,
                            {"pass": True, "n_columns_checked": 27, "n_assumed_unit_total": 3,
                             "warnings": ["heart_rate: 3/4 values carry no unit label"]},
                            outcomes=outcomes)
    assert text.startswith("T\n")
    assert " 1 stage A" in text and "reason: reason A (50.0% of rows)" in text
    assert "total excluded: 1 of 2 rows (50.0%); 1 of 2 patients (50.0%) -> 1 rows, 1 patients kept" in text
    assert "Outcome availability" in text and "chosen in fed_stroke.schema" in text
    assert "mrs_3m" in text and "median 2" in text
    assert "death_3m" in text and "1 = 1 (33.3% of recorded)" in text and "25.0%" in text
    assert "diastolic_blood_pressure" in text and "889" in text
    assert "ODT" in text and "-30" in text
    assert "total nulled: 2 values across 2 features; 39 features untouched" in text
    assert "Unit check: pass=True" in text and "heart_rate: 3/4" in text
    assert "age " not in text.split("Nulled feature values")[1]     # untouched features not listed


def test_assemble_and_write_build_log(tmp_path):
    out = anonymise_frozen(preprocess_gva.wide_to_frozen(_fake_wide_df()), TEST_KEY)
    assert "exclusions" not in out.attrs                    # wide_to_frozen drops no row
    out.attrs["exclusions"] = []
    record_exclusion(out.attrs["exclusions"], "stage A", "reason A",
                     pd.DataFrame({ID: ["p1_1", "p2_1"]}), pd.DataFrame({ID: ["p1_1"]}))
    text = preprocess_gva.assemble_build_log(out, Path("registry.xlsx"), Path("ehr"),
                                             ehr_rows={"patientvalue": 10, "lab": 5})
    assert "GVA frozen-table build log" in text
    assert "schema_version" in text and preprocess_gva.SCHEMA_VERSION in text
    assert "anonymisation" in text and ANONYMISATION_SPEC in text
    assert "patientvalue=10, lab=5" in text
    assert "5 rows x 45 cols (5 patients; 41 features + 3 outcomes)" in text
    assert "label" in text and "chosen in fed_stroke.schema" in text
    assert " 1 stage A" in text
    assert "death_3m" in text and "1 = 2 (50.0% of recorded)" in text   # 4 recorded, 2 positive
    assert "mrs_3m" in text and "median 2" in text                     # [1, 6, 3, NaN, 0] -> 2
    assert "total nulled: 0 values across 0 features; 41 features untouched" in text
    path = write_build_log(tmp_path / "nested" / "gva_frozen.build.log", text)
    assert path.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------- node parquet + smoke report

def _frozen_with_build_attrs():
    out = anonymise_frozen(preprocess_gva.wide_to_frozen(_fake_wide_df()), TEST_KEY)
    out.attrs["exclusions"] = []
    record_exclusion(out.attrs["exclusions"], "stage A", "reason A",
                     pd.DataFrame({ID: ["p1_1", "p2_1"]}), pd.DataFrame({ID: ["p1_1"]}))
    out.attrs["build_log_path"] = "out/x.build.log"
    return out


def test_write_node_parquet_roundtrip_stamp_and_smoke_report(tmp_path):
    out = _frozen_with_build_attrs()
    path = tmp_path / "gva_frozen.parquet"
    hashes = {"registry/r.xlsx": "ab" * 32, "ehr/patientvalue_1.csv": "cd" * 32}
    report = preprocess_gva.write_node_parquet(out, path, source_files=hashes)

    # the parquet IS the frame (values, dtypes, order) — WITHOUT the attrs (raw extremes, paths)
    back = pd.read_parquet(path)
    assert back.columns.tolist() == [ID, *FROZEN_FEATURES, *FROZEN_OUTCOMES]
    pd.testing.assert_frame_equal(back, out)
    assert back.attrs == {} and out.attrs["unit_check"]["pass"] is True   # caller's frame untouched
    assert back[ID].str.fullmatch(r"[0-9a-f]{16}_\d{2}").all()
    raw_bytes = path.read_bytes()
    assert b"p0_0000" not in raw_bytes and b"max_seen" not in raw_bytes   # no raw id, no attrs

    # provenance stamp in the key-value metadata
    meta = preprocess_gva.read_node_metadata(path)
    assert meta["schema_version"] == preprocess_gva.SCHEMA_VERSION == report["schema_version"]
    assert meta["provenance"] == "real-frozen-schema" == report["provenance"]
    assert meta["source_files"] == hashes == report["source_files"]
    assert meta["feature_cols"] == FROZEN_FEATURES and meta["outcome_cols"] == FROZEN_OUTCOMES
    assert meta["n_rows"] == 5 and meta["created"] == report["created"]
    assert meta["anonymisation"] == out.attrs["anonymisation"] == report["anonymisation"]
    assert meta["anonymisation"]["spec"] == ANONYMISATION_SPEC

    # smoke report: written next to the parquet, identical to the returned dict, aggregate-only
    rp = tmp_path / "gva_frozen_smoke_report.json"
    assert report["smoke_report_path"] == str(rp) and report["node_file"] == "gva_frozen.parquet"
    assert json.loads(rp.read_text(encoding="utf-8")) == report
    assert (report["n_rows"], report["n_patients"], report["n_features"], report["n_outcomes"]) == (5, 5, 41, 3)
    assert report["outcomes"]["death_3m"] == {"n_recorded": 4, "n_missing": 1, "missing_rate": 0.2,
                                              "n_positive": 2, "positive_rate": 0.5}
    assert report["outcomes"]["mrs_3m"]["median"] == 2.0
    f = report["features"]["d_dimer"]
    assert set(f) == {"n_recorded", "missing_rate", "median", "p05", "p95"}   # percentiles, never min/max
    assert f["n_recorded"] == 4 and f["missing_rate"] == 0.2 and f["p95"] >= 500.0
    age = report["features"]["age"]
    assert age["n_recorded"] == 5 and age["p95"] <= 90.0               # registry numerics: no NaN in the fixture
    assert report["unit_check"]["pass"] is True and report["unit_check"]["n_columns_checked"] == 27
    assert report["out_of_range"] == {"pass": True, "n_nulled_total": 0, "max_out_of_range_frac": 0.25,
                                      "nulled": {}}
    assert report["exclusions"][0]["stage"] == "stage A" and report["build_log"] == "out/x.build.log"
    text = json.dumps(report) + json.dumps(meta)
    assert "p0_0000" not in text and "Doe" not in text                 # never an id, never a raw value


def test_write_node_parquet_rejects_contract_violations_before_writing(tmp_path):
    raw = preprocess_gva.wide_to_frozen(_fake_wide_df())
    out = anonymise_frozen(raw, TEST_KEY)
    # a raw (not de-identified) frame fails on its id column name — the structural guard
    with pytest.raises(ValueError, match=rf"missing=\['{ID}'\] unexpected=\['{RAW_ID}'\]"):
        preprocess_gva.write_node_parquet(raw, tmp_path / "raw.parquet", {})
    with pytest.raises(ValueError, match=r"missing=\['d_dimer'\]"):
        preprocess_gva.write_node_parquet(out.drop(columns=["d_dimer"]), tmp_path / "a.parquet", {})
    with pytest.raises(ValueError, match="ORDER"):
        preprocess_gva.write_node_parquet(out[[ID, *FROZEN_OUTCOMES, *FROZEN_FEATURES]],
                                          tmp_path / "b.parquet", {})
    dup = out.copy()
    dup.loc[1, ID] = dup.loc[0, ID]
    with pytest.raises(ValueError, match="unique"):
        preprocess_gva.write_node_parquet(dup, tmp_path / "c.parquet", {})
    ints = out.copy()
    ints["IVT"] = ints["IVT"].astype("int64")
    with pytest.raises(ValueError, match=r"float64.*IVT"):
        preprocess_gva.write_node_parquet(ints, tmp_path / "d.parquet", {})
    unstamped = out.copy()
    unstamped.attrs = {}
    with pytest.raises(ValueError, match="anonymisation stamp"):
        preprocess_gva.write_node_parquet(unstamped, tmp_path / "e.parquet", {})
    assert not list(tmp_path.iterdir())                                # nothing written on failure


def test_write_node_parquet_without_build_attrs(tmp_path):
    stamped = anonymise_frozen(preprocess_gva.wide_to_frozen(_fake_wide_df()), TEST_KEY)
    bare = stamped.copy()
    bare.attrs = {"anonymisation": stamped.attrs["anonymisation"]}    # e.g. a half read back from parquet
    report = preprocess_gva.write_node_parquet(bare, tmp_path / "half.parquet", {})
    assert report["unit_check"]["pass"] is None and "not available" in report["unit_check"]["note"]
    assert report["out_of_range"]["pass"] is None
    assert report["exclusions"] is None and report["build_log"] is None
    assert (tmp_path / "half_smoke_report.json").exists()
    meta = preprocess_gva.read_node_metadata(tmp_path / "half.parquet")
    assert meta["source_files"] == {} and meta["anonymisation"] == bare.attrs["anonymisation"]


def test_hash_inputs_selects_exactly_the_files_the_build_reads(tmp_path):
    reg = tmp_path / "registry.xlsx"
    reg.write_bytes(b"registry")
    ehr = tmp_path / "ehr"
    ehr.mkdir()
    (ehr / "patientvalue_1.csv").write_bytes(b"pv")
    (ehr / "lab_1.csv").write_bytes(b"lab")
    (ehr / "notes.txt").write_bytes(b"x")                              # not an input
    (ehr / "patientvalue_old.csv.bak").write_bytes(b"y")               # not an input
    hashes = preprocess_gva.hash_inputs(reg, ehr)
    assert set(hashes) == {"registry/registry.xlsx", "ehr/patientvalue_1.csv", "ehr/lab_1.csv"}
    assert hashes["registry/registry.xlsx"] == hashlib.sha256(b"registry").hexdigest()
    assert hashes["ehr/lab_1.csv"] == hashlib.sha256(b"lab").hexdigest()


# ---------------------------------------------------------------- end to end

def test_wide_to_frozen_end_to_end(capsys):
    out = preprocess_gva.wide_to_frozen(_fake_wide_df())
    assert out.columns.tolist() == [RAW_ID, *FROZEN_FEATURES, *FROZEN_OUTCOMES]   # raw id: not yet de-identified
    assert len(out) == 5                                   # EVERY admission kept, outcome or not
    assert out["d_dimer"].iloc[1] == 500.0                 # 0.5 mg/L -> ng/ml
    assert out["sex"].tolist()[:2] == [1.0, 0.0]
    assert out["IVT"].iloc[0] == 1.0                       # 'started before admission'
    assert not out["med_hist_valv_heart_disease"].isna().any()
    assert out["EVT"].isin([0.0, 1.0]).all()
    assert all(out[c].dtype == "float64" for c in (*FROZEN_FEATURES, *FROZEN_OUTCOMES))
    assert out[RAW_ID].is_unique
    # outcomes carried as recorded: row 3 has neither a 3-month death nor an mRS
    assert out["death_3m"].tolist()[:3] == [1.0, 0.0, 1.0] and np.isnan(out["death_3m"].iloc[3])
    assert out["mrs_3m"].tolist()[:3] == [1.0, 6.0, 3.0] and np.isnan(out["mrs_3m"].iloc[3])
    assert out["death_in_hospital"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out.attrs["outcomes"]["death_3m"] == {"n_recorded": 4, "n_missing": 1, "n_positive": 2}
    assert out.attrs["outcomes"]["mrs_3m"] == {"n_recorded": 4, "n_missing": 1, "median": 2.0}

    rep = out.attrs["unit_check"]
    assert rep["pass"] is True
    assert rep["n_columns_checked"] == 27                  # 41 features - 15 binaries + mrs_3m
    assert rep["columns"]["heart_rate"]["n_assumed_unit"] == 1
    assert rep["columns"]["d_dimer"]["factors_applied"] == {"mg/L": 1000.0}
    assert rep["columns"]["GCS"]["unit_source"] == "declared"
    assert rep["columns"]["mrs_3m"]["labels_seen"] == {"mRS": 4}
    json.dumps(rep)
    oor = out.attrs["out_of_range"]
    assert oor["pass"] is True and oor["n_nulled_total"] == 0     # fixture values are in range
    assert set(oor["columns"]) == set(FROZEN_FEATURES)
    json.dumps(oor)

    captured = capsys.readouterr()
    assert "[outcomes]" in captured.out and "[units] pass=True" in captured.out
    assert "[schema] OK" in captured.out and "no row dropped" in captured.out
    assert "[ranges] pass=True" in captured.out
    assert "architecture schema not yet frozen" not in captured.err   # fed_stroke.schema is frozen
    assert out.attrs["architecture_schema_frozen"] is True


def test_warn_if_architecture_schema_not_frozen(capsys, monkeypatch):
    assert preprocess_gva.warn_if_architecture_schema_not_frozen() is True
    assert capsys.readouterr().err == ""
    # a drift between the two copies of the contract is what the banner is for
    monkeypatch.setattr(preprocess_gva.arch_schema, "FEATURE_COLS", list(FROZEN_FEATURES[:2]))
    assert preprocess_gva.warn_if_architecture_schema_not_frozen() is False
    assert "not yet frozen" in capsys.readouterr().err


# ---------------------------------------------------------------- architecture mirror

def test_architecture_schema_mirrors_frozen_contract():
    """fed_stroke.schema is a literal copy of the preprocessing-side contract (the wheel ships
    without the root package); this is the test that keeps the two copies identical."""
    from fed_stroke.dp.boost import FEATURE_RANGES
    from fed_stroke.schema import (FEATURE_COLS, FEATURE_UNITS, MISSING_SENTINEL, OUTCOME_COLS,
                                   SCHEMA_VERSION, TARGET_COL)
    from fed_stroke.schema import ID_COL as ARCH_ID_COL
    assert list(FEATURE_COLS) == FROZEN_FEATURES
    assert list(OUTCOME_COLS) == FROZEN_OUTCOMES
    assert ARCH_ID_COL == ID_COL == ID                # the de-identified id column, both copies
    assert TARGET_COL in FROZEN_OUTCOMES              # the label is one of the carried outcomes
    assert FEATURE_UNITS == FROZEN_UNITS
    assert preprocess_gva.SCHEMA_VERSION == SCHEMA_VERSION
    # DP bin grid: same keys in the same ORDER (column index == feature), every lower bound
    # above the missing sentinel, binaries exactly (0, 1) — and the preprocessing-side mirror
    # FROZEN_RANGES is identical, pair for pair and in order (edit both copies together)
    assert list(FEATURE_RANGES) == FROZEN_FEATURES
    assert list(FROZEN_RANGES.items()) == list(FEATURE_RANGES.items())
    for name, (lo, hi) in FEATURE_RANGES.items():
        assert MISSING_SENTINEL < lo < hi, name
        if is_binary_unit(FROZEN_UNITS[name]):
            assert (lo, hi) == (0.0, 1.0), name
