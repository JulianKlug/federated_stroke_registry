"""De-identification of a frozen-schema table — the LAST step before a node parquet is written.

Site-agnostic: every site driver (architecture/preprocessing/preprocess_gva.py today, the
Shenzhen driver later) calls anonymise_frozen on its wide_to_frozen output.

    raw frozen table (site-internal)            de-identified node table
    ┌───────────────────┬───────┐               ┌──────────────────────┬──────┐
    │ case_admission_id │ age   │               │ pseudo_admission_id  │ age  │
    │ 12345678_6202     │ 76.4  │   ────────►   │ 3fa9c1e2b7d4a018_00  │ 77.0 │
    │ 12345678_7115     │ 78.1  │               │ 3fa9c1e2b7d4a018_01  │ 79.0 │
    └───────────────────┴───────┘               └──────────────────────┴──────┘
      patient id + EDS suffix                     HMAC(key, patient id) + admission ordinal
      fractional years (DOB proxy)                floor + per-patient keyed jitter + top-code

- id: '<hmac-sha256(key, patient_id)[:16]>_<ordinal>'. The patient key stays derivable by
  split('_')[0] (loader R3 dedup / R4 split). The ordinal is the rank of the ORIGINAL id within
  the patient, so the loader's lexicographic tie-break keeps the same admission as before. The
  EDS suffix — a hospital administrative case identifier (SPHN §5.1.4) — disappears.
- age: completed years, plus a per-PATIENT offset uniform in ±AGE_JITTER_YEARS derived from the
  same key (deterministic across rebuilds, identical for all admissions of one patient so it
  cannot be averaged out, unpredictable without the key), then clipped to [0, 90]
  (SPHN §5.1.9 / HIPAA §164.514(b)(2): ages > 89 collapse into "90+").

The key belongs to the data provider (HRO Art. 26 coded data): 32+ random bytes, stored outside
the repo and away from the artefacts, immutable for the project lifetime — the R4 split hashes
the pseudonym, so a rotated key moves every patient's DEV/HELD assignment. Only its fingerprint
is ever recorded.
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pandas as pd

from .case_ids import RAW_ID_COL
from .mappings.frozen_schema import ID_COL

AGE_COL = "age"
AGE_TOP_CODE_YEARS = 90.0          # SPHN §5.1.9 / HIPAA §164.514(b)(2): > 89 -> "90+"
AGE_JITTER_YEARS = 2               # per-patient keyed offset, uniform integer in [-2, +2]
AGE_JITTER_DOMAIN = b"age-jitter|"  # HMAC domain separation: the pseudonym never reveals the offset
PSEUDONYM_HEX_LEN = 16             # 64 bits: collision probability ~ n^2 / 2^65
ADMISSION_ORDINAL_WIDTH = 2
MIN_PSEUDONYM_KEY_BYTES = 32
KEY_FINGERPRINT_HEX_LEN = 8
_JITTER_DIGEST_BYTES = 4

# Stamped into the parquet metadata and the smoke report: names the exact transform a file went
# through. Built from the constants so the two cannot drift.
ANONYMISATION_SPEC = (
    f"anon-v1: age floor-years + keyed per-patient jitter U{{-{AGE_JITTER_YEARS}..{AGE_JITTER_YEARS}}}"
    f" + clip[0,{AGE_TOP_CODE_YEARS:g}]; id hmac-sha256(patient_id)[:{PSEUDONYM_HEX_LEN}]"
    f"_admission-ordinal; column {RAW_ID_COL}->{ID_COL}"
)


def _check_key(key: bytes) -> None:
    if len(key) < MIN_PSEUDONYM_KEY_BYTES:
        raise ValueError(
            f"pseudonym key must be at least {MIN_PSEUDONYM_KEY_BYTES} random bytes; got {len(key)}"
        )


def key_fingerprint(key: bytes) -> str:
    """Short sha256 prefix of the key: says WHICH key built a file without revealing it."""
    _check_key(key)
    return hashlib.sha256(key).hexdigest()[:KEY_FINGERPRINT_HEX_LEN]


def _patient_ids(ids: pd.Series) -> pd.Series:
    return ids.astype(str).str.split("_").str[0]


def keyed_age_jitter(patient_ids: pd.Series, key: bytes) -> pd.Series:
    """Integer offset in [-AGE_JITTER_YEARS, +AGE_JITTER_YEARS] per patient, uniform.

    offset(pid) = HMAC(key, AGE_JITTER_DOMAIN ‖ pid)[:4] mod (2k+1) − k. Same patient -> same
    offset (also across rebuilds with the same key); 2^32 mod 5 leaves a negligible bias.
    """
    _check_key(key)
    n_offsets = 2 * AGE_JITTER_YEARS + 1

    def offset(pid: str) -> int:
        digest = hmac.new(key, AGE_JITTER_DOMAIN + pid.encode(), hashlib.sha256).digest()
        return int.from_bytes(digest[:_JITTER_DIGEST_BYTES], "big") % n_offsets - AGE_JITTER_YEARS

    return patient_ids.astype(str).map(offset).astype("int64")


def anonymise_age(age: pd.Series, jitter: pd.Series) -> pd.Series:
    """floor(age) + jitter, clipped to [0, AGE_TOP_CODE_YEARS]. NaN stays NaN; float64 out.

    e.g. offset +1: [70.4, 88.9, 89.2, 104.3, nan, 0.5] -> [71, 89, 90, 90, nan, 1]
    """
    if not age.index.equals(jitter.index):
        raise ValueError("anonymise_age: age and jitter must share the same index")
    out = np.floor(age.astype("float64")) + jitter.astype("float64")
    return out.clip(lower=0.0, upper=AGE_TOP_CODE_YEARS)


def pseudonymise_case_ids(ids: pd.Series, key: bytes) -> pd.Series:
    """'<patient_id>_<suffix>' -> '<hmac-sha256(key, patient_id)[:16]>_<ordinal>'.

    The ordinal is the 0-based rank of the original id within the patient (zero-padded), so
    '<pid>_0003' < '<pid>_0011' becomes '<pseud>_00' < '<pseud>_01': the loader's lexicographic
    tie-break (task.dedup_one_row_per_patient) keeps the same admission. Fails on a short key,
    non-unique input ids, a pseudonym collision, or too many admissions for the ordinal width.
    """
    _check_key(key)
    ids = ids.astype(str)
    if not ids.is_unique:
        raise ValueError("pseudonymise_case_ids: input ids not unique")
    pids = _patient_ids(ids)

    def pseudonym(pid: str) -> str:
        return hmac.new(key, pid.encode(), hashlib.sha256).hexdigest()[:PSEUDONYM_HEX_LEN]

    pseud = pids.map(pseudonym)
    if pseud.nunique() != pids.nunique():
        raise ValueError("pseudonymise_case_ids: pseudonym collision — two patients share one")

    # ordinal = position of the original id in its patient's sorted admissions
    order = ids.sort_values(kind="stable").index
    ordinal = ids.loc[order].groupby(pids.loc[order]).cumcount().reindex(ids.index)
    max_admissions = 10 ** ADMISSION_ORDINAL_WIDTH
    if (ordinal >= max_admissions).any():
        raise ValueError(
            f"pseudonymise_case_ids: a patient has >= {max_admissions} admissions; "
            "widen ADMISSION_ORDINAL_WIDTH"
        )

    return pseud + "_" + ordinal.astype(str).str.zfill(ADMISSION_ORDINAL_WIDTH)


def anonymise_frozen(df: pd.DataFrame, key: bytes) -> pd.DataFrame:
    """De-identify a wide_to_frozen output: age (floor + keyed jitter + top-code), ids
    (pseudonym + ordinal), column RAW_ID_COL -> ID_COL.

    Rows, row order and every other column are untouched. attrs gain 'anonymisation'
    (spec, jitter width, rows top-coded at 90, key fingerprint) — the stamp write_node_parquet
    requires; the key itself is never recorded.
    """
    if RAW_ID_COL not in df.columns:
        raise ValueError(f"anonymise_frozen: {RAW_ID_COL!r} absent — expects a wide_to_frozen output")

    out = df.copy()
    pids = _patient_ids(out[RAW_ID_COL])
    out[AGE_COL] = anonymise_age(out[AGE_COL], keyed_age_jitter(pids, key))
    out[RAW_ID_COL] = pseudonymise_case_ids(out[RAW_ID_COL], key)
    out = out.rename(columns={RAW_ID_COL: ID_COL})
    if len(out) != len(df):
        raise AssertionError("anonymise_frozen: row count changed")

    out.attrs = {
        **df.attrs,
        "anonymisation": {
            "spec": ANONYMISATION_SPEC,
            "age_jitter_years": AGE_JITTER_YEARS,
            "age_top_coded_n": int((out[AGE_COL] == AGE_TOP_CODE_YEARS).sum()),
            "key_fingerprint": key_fingerprint(key),
        },
    }
    return out
