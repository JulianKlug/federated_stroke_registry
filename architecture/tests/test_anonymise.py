"""preprocessing.anonymise: de-identification of the frozen table (anon-v1).

Synthetic frames only. Import mechanics as in test_preprocess_gva.py: the repo root gives the
shared `preprocessing` package, architecture/ (pytest pythonpath) gives fed_stroke.
"""
import hashlib
import hmac
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fed_stroke.task import dedup_one_row_per_patient  # noqa: E402
from preprocessing.anonymise import (  # noqa: E402
    ADMISSION_ORDINAL_WIDTH,
    AGE_JITTER_DOMAIN,
    AGE_JITTER_YEARS,
    AGE_TOP_CODE_YEARS,
    ANONYMISATION_SPEC,
    MIN_PSEUDONYM_KEY_BYTES,
    PSEUDONYM_HEX_LEN,
    anonymise_age,
    anonymise_frozen,
    key_fingerprint,
    keyed_age_jitter,
    pseudonymise_case_ids,
)
from preprocessing.case_ids import RAW_ID_COL  # noqa: E402
from preprocessing.mappings import (  # noqa: E402
    FROZEN_FEATURES,
    FROZEN_OUTCOMES,
    FROZEN_RANGES,
    FROZEN_UNITS,
    ID_COL,
    is_binary_unit,
)

TEST_KEY = bytes(range(32))
OTHER_KEY = b"\x01" * 32
ID_PATTERN = rf"[0-9a-f]{{{PSEUDONYM_HEX_LEN}}}_\d{{{ADMISSION_ORDINAL_WIDTH}}}"


def _raw_frozen(ids, ages) -> pd.DataFrame:
    """A wide_to_frozen-shaped raw frame: RAW_ID_COL ids, the given ages, every other feature at
    the midpoint of its range (binaries 1.0), the three outcomes recorded."""
    df = pd.DataFrame({RAW_ID_COL: ids})
    for f in FROZEN_FEATURES:
        lo, hi = FROZEN_RANGES[f]
        df[f] = 1.0 if is_binary_unit(FROZEN_UNITS[f]) else (lo + hi) / 2
    df["age"] = np.asarray(ages, dtype=float)
    df["mrs_3m"], df["death_3m"], df["death_in_hospital"] = 2.0, 0.0, 0.0
    return df


# ---------------------------------------------------------------- age

def test_anonymise_age_floors_jitters_and_topcodes():
    age = pd.Series([70.4, 88.9, 89.2, 104.3, np.nan, 0.5])
    assert anonymise_age(age, pd.Series([0] * 6)).tolist()[:4] == [70.0, 88.0, 89.0, 90.0]
    out = anonymise_age(age, pd.Series([1] * 6))
    assert out.tolist()[:4] == [71.0, 89.0, 90.0, 90.0]
    assert np.isnan(out.iloc[4]) and out.iloc[5] == 1.0
    assert anonymise_age(age, pd.Series([-2] * 6)).iloc[5] == 0.0        # clipped at 0


def test_anonymise_age_keeps_nan_dtype_and_range():
    rng = np.random.default_rng(0)
    age = pd.Series(rng.uniform(0, 120, 2000))
    age.iloc[::50] = np.nan
    jitter = pd.Series(rng.integers(-AGE_JITTER_YEARS, AGE_JITTER_YEARS + 1, 2000))
    out = anonymise_age(age, jitter)
    assert out.dtype == "float64" and out.isna().equals(age.isna())
    recorded = out.dropna()
    assert (recorded == np.floor(recorded)).all()                        # integer-valued
    lo, hi = FROZEN_RANGES["age"]
    assert recorded.min() >= lo and recorded.max() <= AGE_TOP_CODE_YEARS < hi
    with pytest.raises(ValueError, match="same index"):
        anonymise_age(age, jitter.iloc[::-1])


def test_keyed_age_jitter_deterministic_bounded_and_per_patient():
    pids = pd.Series([f"P{i}" for i in range(1000)] + ["P0", "P1"])
    j = keyed_age_jitter(pids, TEST_KEY)
    assert j.equals(keyed_age_jitter(pids, TEST_KEY)) and j.dtype == "int64"
    assert j.between(-AGE_JITTER_YEARS, AGE_JITTER_YEARS).all()
    assert j.iloc[1000] == j.iloc[0] and j.iloc[1001] == j.iloc[1]        # same patient, same offset
    assert not keyed_age_jitter(pids, OTHER_KEY).equals(j)


def test_keyed_age_jitter_roughly_uniform():
    pids = pd.Series([f"P{i}" for i in range(10_000)])
    share = keyed_age_jitter(pids, TEST_KEY).value_counts(normalize=True)
    assert set(share.index) == set(range(-AGE_JITTER_YEARS, AGE_JITTER_YEARS + 1))
    assert share.between(0.15, 0.25).all()


def test_jitter_and_pseudonym_pin_their_keyed_formulas():
    # the two HMACs are domain-separated: the pseudonym says nothing about the offset
    pid = b"P42"
    digest = hmac.new(TEST_KEY, AGE_JITTER_DOMAIN + pid, hashlib.sha256).digest()
    expected = int.from_bytes(digest[:4], "big") % (2 * AGE_JITTER_YEARS + 1) - AGE_JITTER_YEARS
    assert keyed_age_jitter(pd.Series(["P42"]), TEST_KEY).iloc[0] == expected
    pseud = hmac.new(TEST_KEY, pid, hashlib.sha256).hexdigest()[:PSEUDONYM_HEX_LEN]
    assert pseudonymise_case_ids(pd.Series(["P42_0007"]), TEST_KEY).iloc[0] == f"{pseud}_00"


# ---------------------------------------------------------------- ids

def test_pseudonymise_deterministic_format_ordinal_and_patient_count():
    ids = pd.Series(["P1_0003", "P1_0001", "P2_0009", "P3_0002"])
    out = pseudonymise_case_ids(ids, TEST_KEY)
    assert out.equals(pseudonymise_case_ids(ids, TEST_KEY))
    assert out.str.fullmatch(ID_PATTERN).all() and out.is_unique
    prefixes = out.str.split("_").str[0]
    assert prefixes.nunique() == 3 and prefixes.iloc[0] == prefixes.iloc[1]
    # ordinal = rank of the original id within the patient: P1_0001 -> _00, P1_0003 -> _01
    assert [s[-2:] for s in out] == ["01", "00", "00", "00"]
    other = pseudonymise_case_ids(ids, OTHER_KEY)
    assert not (other.str.split("_").str[0] == prefixes).any()


def test_pseudonymise_preserves_dedup_tiebreak():
    # tied labels within a patient: the loader keeps the lexicographically smallest id — the
    # SAME admission before and after pseudonymisation
    ids = ["P1_0003", "P1_0001", "P2_0009", "P2_0004", "P3_0002"]
    raw = pd.DataFrame({ID_COL: ids, "y": [1, 1, 0, 0, 1], "row": range(5)})
    raw["patient_id"] = raw[ID_COL].str.split("_").str[0]
    pseud = raw.assign(**{ID_COL: pseudonymise_case_ids(raw[ID_COL], TEST_KEY)})
    pseud["patient_id"] = pseud[ID_COL].str.split("_").str[0]
    kept_raw = dedup_one_row_per_patient(raw, "y")["row"].tolist()
    kept_pseud = dedup_one_row_per_patient(pseud, "y")["row"].tolist()
    assert kept_raw == kept_pseud == [1, 3, 4]


def test_pseudonymise_rejects_short_key_duplicates_and_overflow():
    ids = pd.Series(["P1_0001", "P2_0001"])
    with pytest.raises(ValueError, match=f"at least {MIN_PSEUDONYM_KEY_BYTES}"):
        pseudonymise_case_ids(ids, b"short")
    with pytest.raises(ValueError, match=f"at least {MIN_PSEUDONYM_KEY_BYTES}"):
        keyed_age_jitter(ids, b"short")
    with pytest.raises(ValueError, match="not unique"):
        pseudonymise_case_ids(pd.Series(["P1_0001", "P1_0001"]), TEST_KEY)
    too_many = pd.Series([f"P1_{i:04d}" for i in range(10 ** ADMISSION_ORDINAL_WIDTH + 1)])
    with pytest.raises(ValueError, match="admissions"):
        pseudonymise_case_ids(too_many, TEST_KEY)


def test_key_fingerprint_is_short_and_key_specific():
    fp = key_fingerprint(TEST_KEY)
    assert re.fullmatch(r"[0-9a-f]{8}", fp) and fp != key_fingerprint(OTHER_KEY)
    with pytest.raises(ValueError, match="at least"):
        key_fingerprint(b"short")


# ---------------------------------------------------------------- the frozen table

def test_anonymise_frozen_renames_stamps_and_drops_original_ids():
    ids = ["P1_0003", "P1_0001", "P2_0009", "P3_0002"]
    raw = _raw_frozen(ids, [70.4, 88.9, 95.0, np.nan])
    raw.attrs["unit_check"] = {"pass": True}
    out = anonymise_frozen(raw, TEST_KEY)

    assert out.columns.tolist() == [ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES]
    assert len(out) == 4 and RAW_ID_COL not in out.columns
    assert out[ID_COL].str.fullmatch(ID_PATTERN).all() and not set(ids) & set(out[ID_COL])
    others = [c for c in FROZEN_FEATURES if c != "age"] + FROZEN_OUTCOMES
    assert out[others].equals(raw[others])

    # age: completed years + the patient's offset (P1's two admissions share it), top-coded, NaN kept
    jitter = keyed_age_jitter(pd.Series(["P1", "P1", "P2", "P3"]), TEST_KEY)
    expected = (np.floor(raw["age"]) + jitter).clip(0.0, AGE_TOP_CODE_YEARS)
    assert out["age"].iloc[:2].tolist() == expected.iloc[:2].tolist()
    assert out["age"].iloc[2] == AGE_TOP_CODE_YEARS and np.isnan(out["age"].iloc[3])

    stamp = out.attrs["anonymisation"]
    assert stamp["spec"] == ANONYMISATION_SPEC and stamp["age_jitter_years"] == AGE_JITTER_YEARS
    assert stamp["key_fingerprint"] == key_fingerprint(TEST_KEY)
    assert stamp["age_top_coded_n"] == int((out["age"] == AGE_TOP_CODE_YEARS).sum()) >= 1
    assert out.attrs["unit_check"] == {"pass": True}                    # existing attrs carried
    assert raw.attrs == {"unit_check": {"pass": True}} and RAW_ID_COL in raw.columns   # input untouched
    assert anonymise_frozen(raw, TEST_KEY).equals(out)                  # deterministic
    with pytest.raises(ValueError, match=RAW_ID_COL):
        anonymise_frozen(out, TEST_KEY)                                 # already de-identified


def test_spec_names_the_transform_and_the_columns():
    assert ANONYMISATION_SPEC.startswith("anon-v1")
    assert RAW_ID_COL in ANONYMISATION_SPEC and ID_COL in ANONYMISATION_SPEC
    assert f"{AGE_TOP_CODE_YEARS:g}" in ANONYMISATION_SPEC and str(AGE_JITTER_YEARS) in ANONYMISATION_SPEC
