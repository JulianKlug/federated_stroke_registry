"""Geneva stroke-registry cohort preprocessing.

Cohort filter (dedup + ischemic stroke), OPSUM outcome reconciliation, feature
cleanup, and timing derivation — shared by the registry summary tables
(registry_alignement/) and the frozen-schema node preprocessing
(architecture/preprocessing/).

Outcome preprocessing follows OPSUM's `outcome_preprocessing`:
  https://github.com/JulianKlug/OPSUM/blob/main/meta_data/geneva_stroke_unit_patient_characteristics.py
  - If `Death in hospital == 'yes'` -> `3M mRS = 6` and `3M Death = 'yes'`.
  - If `3M Death == 'yes'` and `3M mRS` is NaN -> `3M mRS = 6`.
  - If `3M mRS == 6` -> `3M Death = 'yes'`.
  - If `3M mRS` is known and != 6 and `3M Death` is NaN -> `3M Death = 'no'`.
"""
from __future__ import annotations

import pandas as pd

from .build_log import record_exclusion


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """Parse a numeric YYYYMMDD column into a datetime series (NaT for invalid)."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    valid = s.notna() & (s >= 19000101) & (s <= 21001231)
    if valid.any():
        parsed = pd.to_datetime(
            s[valid].astype(int).astype(str), format="%Y%m%d", errors="coerce"
        )
        out.loc[valid] = parsed
    return out


# ---------------------------------------------------------------------------
# Date+time parsing (lenient) used by the timing variables. Mirrors
# registry_alignement/geneva_preprocessing/gva_timings.ipynb so values like
# '0742' (HHMM as digits) or pandas Timestamps both round-trip correctly.
# ---------------------------------------------------------------------------
def _normalize_time(t) -> str:
    if pd.isna(t):
        return "00:00"
    s = str(t).strip()
    if ":" in s:
        parts = s.split(":")
        hh = parts[0].zfill(2)
        mm = parts[1].zfill(2) if len(parts) > 1 and parts[1].isdigit() else "00"
        return f"{hh}:{mm}"
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 3:
        digits = "0" + digits
    if len(digits) == 4:
        return digits[:2] + ":" + digits[2:]
    return "00:00"


def _normalize_date(d) -> str | None:
    if pd.isna(d):
        return None
    s = str(int(d)) if isinstance(d, (int, float)) and not pd.isna(d) else str(d).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:
        return digits  # YYYYMMDD
    try:
        return pd.to_datetime(s, dayfirst=False).strftime("%Y%m%d")
    except Exception:
        try:
            return pd.to_datetime(s, dayfirst=True).strftime("%Y%m%d")
        except Exception:
            return None


def _parse_date_time(date_val, time_val):
    date_norm = _normalize_date(date_val)
    if not date_norm:
        return pd.NaT
    time_norm = _normalize_time(time_val)
    try:
        return pd.to_datetime(date_norm + " " + time_norm, format="%Y%m%d %H:%M")
    except Exception:
        try:
            return pd.to_datetime(
                date_norm + time_norm.replace(":", ""), format="%Y%m%d%H%M"
            )
        except Exception:
            try:
                return pd.to_datetime(date_norm, format="%Y%m%d")
            except Exception:
                return pd.NaT


def compute_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Add ODT/ONT/DNT/DPT/OPT (minutes) columns derived from date+time pairs.

    ODT: Onset to Door (admission - onset)
    ONT: Onset to Needle (IVT - onset), falls back to 'Onset to treatment (min.)'
    DNT: Door to Needle (IVT - admission), falls back to 'Door to treatment (min.)'
    DPT: Door to Puncture (IAT - admission), falls back to 'Door to groin puncture (min.)'
    OPT: Onset to Puncture (IAT - onset), falls back to 'Onset to groin puncture (min.)'

    Mirrors registry_alignement/geneva_preprocessing/gva_timings.ipynb.
    """
    needed = {
        "Onset date", "Onset time",
        "Arrival at hospital", "Arrival time",
        "IVT start date", "IVT start time",
        "Date of groin puncture", "Time of groin puncture",
    }
    missing = needed - set(df.columns)
    if missing:
        print(f"[prep]  timings skipped, missing columns: {sorted(missing)}")
        return df

    onset_dt = df.apply(
        lambda r: _parse_date_time(r.get("Onset date"), r.get("Onset time")), axis=1
    )
    adm_dt = df.apply(
        lambda r: _parse_date_time(r.get("Arrival at hospital"), r.get("Arrival time")),
        axis=1,
    )
    ivt_dt = df.apply(
        lambda r: _parse_date_time(r.get("IVT start date"), r.get("IVT start time")),
        axis=1,
    )
    iat_dt = df.apply(
        lambda r: _parse_date_time(
            r.get("Date of groin puncture"), r.get("Time of groin puncture")
        ),
        axis=1,
    )

    df["ODT"] = (adm_dt - onset_dt).dt.total_seconds() / 60
    df["ONT"] = (ivt_dt - onset_dt).dt.total_seconds() / 60
    if "Onset to treatment (min.)" in df.columns:
        df["ONT"] = df["ONT"].fillna(df["Onset to treatment (min.)"])
    df["DNT"] = (ivt_dt - adm_dt).dt.total_seconds() / 60
    if "Door to treatment (min.)" in df.columns:
        df["DNT"] = df["DNT"].fillna(df["Door to treatment (min.)"])
    df["DPT"] = (iat_dt - adm_dt).dt.total_seconds() / 60
    if "Door to groin puncture (min.)" in df.columns:
        df["DPT"] = df["DPT"].fillna(df["Door to groin puncture (min.)"])
    df["OPT"] = (iat_dt - onset_dt).dt.total_seconds() / 60
    if "Onset to groin puncture (min.)" in df.columns:
        df["OPT"] = df["OPT"].fillna(df["Onset to groin puncture (min.)"])
    return df


def build_cohort(df: pd.DataFrame,
                 exclusions: list[dict] | None = None) -> tuple[pd.DataFrame, int, int]:
    """Drop duplicates, filter to ischemic stroke, derive outcome variables.

    `exclusions`: optional list that receives one build_log.record_exclusion entry per
    filtering stage (rows and distinct patients before/after, with the reason) — the
    frozen-table build log reads it. Return value unchanged: (df, n_raw, n_filtered).
    """
    if exclusions is None:
        exclusions = []
    n_raw = len(df)

    # 1. Drop exact duplicate rows
    before = df
    df = df.drop_duplicates().copy()
    record_exclusion(exclusions, "exact duplicate registry rows",
                     "identical rows (every column) repeated in the registry export",
                     before, df)
    # Drop rows explicitly labelled 'duplicate' in Type of event
    if "Type of event" in df.columns:
        before = df
        mask_dup = df["Type of event"].astype("string").str.lower().eq("duplicate")
        df = df.loc[~mask_dup].copy()
        record_exclusion(exclusions, "rows flagged 'duplicate' in Type of event",
                         "registrar marked the entry as a duplicate", before, df)

    # 2. Filter to ischemic stroke
    if "Type of event" not in df.columns:
        raise SystemExit("Column 'Type of event' not found; cannot filter to ischemic stroke.")
    before = df
    df = df.loc[df["Type of event"] == "Ischemic stroke"].copy()
    record_exclusion(exclusions, "not an ischemic stroke",
                     "Type of event != 'Ischemic stroke' (TIA, haemorrhage, other) — outside "
                     "the cohort definition; a patient with another ischemic-stroke admission "
                     "is kept", before, df)

    # 3. Collapse same-admission duplicates that the manual 'duplicate' flag
    # missed. Two registrars sometimes enter the same Case ID with conflicting
    # answers (different NIHSS, glucose, prior-stroke history, etc.); without
    # this pass the patient gets half a vote toward each answer in summaries.
    if "Case ID" in df.columns:
        before = df
        df = df.drop_duplicates(subset=["Case ID"], keep="first").copy()
        n_case_dedup_dropped = len(before) - len(df)
        record_exclusion(exclusions, "same Case ID entered twice",
                         "two registrars entered the same admission with conflicting answers; "
                         "the first row is kept", before, df)
        if n_case_dedup_dropped:
            print(f"[prep]  Case ID dedup: dropped {n_case_dedup_dropped} extra rows")
    n_filtered = len(df)

    return df, n_raw, n_filtered


def preprocess_outcome(df: pd.DataFrame) -> pd.DataFrame:
    # 3. Outcome preprocessing (OPSUM outcome_preprocessing, mutated in place).
    # Source: OPSUM/meta_data/geneva_stroke_unit_patient_characteristics.py
    if {"3M mRS", "3M Death", "Death in hospital"}.issubset(df.columns):
        in_hosp_death = df["Death in hospital"].eq("yes")
        three_m_death_yes = df["3M Death"].eq("yes")

        # If death in hospital, set 3M mRS to 6
        df.loc[in_hosp_death, "3M mRS"] = 6
        # If 3M Death is 'yes' and 3M mRS is NaN, set 3M mRS to 6
        df.loc[three_m_death_yes & df["3M mRS"].isna(), "3M mRS"] = 6

        # If death in hospital, set 3M Death to 'yes'
        df.loc[in_hosp_death, "3M Death"] = "yes"
        # If 3M mRS == 6, set 3M Death to 'yes'
        df.loc[df["3M mRS"] == 6, "3M Death"] = "yes"
        # If 3M mRS is not NaN and not 6 and 3M Death is NaN, set 3M Death to 'no'
        df.loc[
            (df["3M mRS"] != 6) & df["3M mRS"].notna() & df["3M Death"].isna(),
            "3M Death",
        ] = "no"
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # 4. Collapse TOAST "Unknown etiology" subtypes (with/despite evaluation)
    # into a single "Unknown etiology" bucket for the summary.
    if "Etiology TOAST" in df.columns:
        mask_unknown = df["Etiology TOAST"].astype("string").str.startswith(
            "Unknown etiology", na=False
        )
        df.loc[mask_unknown, "Etiology TOAST"] = "Unknown etiology"

    # 5. Derive timing variables (ODT, ONT, DNT, DPT) in minutes.
    df = compute_timings(df)

    # add wake-up stroke column
    df['wake_up_stroke'] = (df['Time of symptom onset known'] == 'wake up').astype(int)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Preprocess the Geneva stroke registry DataFrame for summary table.

    Returns:
        df: preprocessed DataFrame
        n_raw: number of raw rows before preprocessing
        n_filtered: number of rows after deduplication and filtering
    """
    df, n_raw, n_filtered = build_cohort(df)
    df = preprocess_outcome(df)
    df = preprocess_features(df)
    return df, n_raw, n_filtered
