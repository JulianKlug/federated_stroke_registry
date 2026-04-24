"""
Build a summary table of the Geneva (GVA) stroke registry database.

Input  : one Geneva stroke-registry .xlsx (e.g. data/gva_stroke_registry_post_hoc_modified.xlsx)
Outputs:
  - <output-dir>/gva_summary_table.csv  : per-variable summary row
  - <output-dir>/gva_metadata.csv       : cohort-level metadata

Spec: ./summary_table_specs.md
Overlap mapping: ./variable_comparison_summary.md (§1)
Outcome preprocessing follows OPSUM's `outcome_preprocessing`:
  https://github.com/JulianKlug/OPSUM/blob/main/meta_data/geneva_stroke_unit_patient_characteristics.py
  - If `Death in hospital == 'yes'` -> `3M mRS = 6` and `3M Death = 'yes'`.
  - If `3M Death == 'yes'` and `3M mRS` is NaN -> `3M mRS = 6`.
  - If `3M mRS == 6` -> `3M Death = 'yes'`.
  - If `3M mRS` is known and != 6 and `3M Death` is NaN -> `3M Death = 'no'`.

Usage:
    python build_gva_summary_table.py data/gva_stroke_registry_post_hoc_modified.xlsx
    python build_gva_summary_table.py <input.xlsx> --output-dir out/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mappings import GVA_TO_SHENZEN, UNITS


# Columns to exclude from the per-variable summary (PII or pure bookkeeping).
SKIP_COLS: set[str] = {
    "Case ID",
    "Last name",
    "First name",
    "Entry person",
    "ZIP",
}

# Column-name patterns to skip (likely bookkeeping / spurious export artefacts).
# Matches e.g. "Unnamed: 0", "Unnamed: 174" — any column pandas couldn't name.
SKIP_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^Unnamed:\s*\d+$"),
)


# Regex for HH:MM or HH:MM:SS timestamps (00:00-23:59[:59]).
_TIME_RE = re.compile(r"^\s*([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\s*$")


# Per-type template describing the shape of `summary_statistic`.
STAT_FORMAT: dict[str, str] = {
    "binary": "n (%)",
    "ordinal": "median (Q1-Q3)",
    "continuous": "median (Q1-Q3)",
    "date": "median date (Q1-Q3)",
    "time": "median time (Q1-Q3)",
    "categorical": "category: n (%); ...",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def is_yesno(series: pd.Series) -> bool:
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return False
    vals = series.dropna().astype(str).str.lower().str.strip()
    if vals.empty:
        return False
    return set(vals.unique()).issubset({"yes", "no"})


def _looks_like_time(values: pd.Series) -> bool:
    """True if every non-null string value matches HH:MM or HH:MM:SS."""
    s = values.dropna().astype(str)
    if s.empty:
        return False
    return s.map(lambda v: bool(_TIME_RE.match(v))).all()


def _time_to_minutes(values: pd.Series) -> pd.Series:
    """Convert HH:MM[:SS] strings to float minutes-since-midnight; NaN for unparseable."""
    def _parse(v):
        if pd.isna(v):
            return np.nan
        m = _TIME_RE.match(str(v))
        if not m:
            return np.nan
        parts = str(v).strip().split(":")
        h = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
        return h * 60 + mm + ss / 60.0

    return values.map(_parse)


def _minutes_to_hhmm(mins: float) -> str:
    total = int(round(mins))
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"


def _looks_like_yyyymmdd(values: pd.Series) -> bool:
    """True if every non-null numeric value parses as a plausible YYYYMMDD date."""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return False
    if not ((s >= 19000101) & (s <= 21001231)).all():
        return False
    as_int = s.astype(int)
    month = (as_int // 100) % 100
    day = as_int % 100
    return bool(((month >= 1) & (month <= 12)).all() and ((day >= 1) & (day <= 31)).all())


def detect_type(series: pd.Series, col_name: str) -> str:
    """Return one of: 'date', 'binary', 'ordinal', 'continuous', 'categorical'."""
    lname = col_name.lower()

    # Date: name-hinted numeric columns always parse as date (typos -> NaT);
    # otherwise, infer from value pattern.
    if pd.api.types.is_numeric_dtype(series):
        name_hint = "date" in lname or col_name == "DOB" or "arrival at" in lname
        if name_hint:
            return "date"
        if _looks_like_yyyymmdd(series):
            return "date"

    # Time: HH:MM[:SS] strings (e.g. "07:42")
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        if _looks_like_time(series):
            return "time"

    non_null = series.dropna()
    if non_null.empty:
        return "categorical"

    # Binary: yes/no strings, or {0,1} numeric
    if is_yesno(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        uniq = set(non_null.unique())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return "binary"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = non_null.nunique()
        # Ordinal: few distinct small non-negative integers (mRS, Rankin, etc.)
        if n_unique <= 10:
            try:
                as_int = non_null.astype(int)
                if (as_int == non_null).all() and as_int.min() >= 0 and as_int.max() <= 10:
                    return "ordinal"
            except (ValueError, TypeError):
                pass
        return "continuous"

    return "categorical"


def fmt_missing(n_missing: int, n_total: int) -> str:
    pct = (n_missing / n_total * 100) if n_total else 0.0
    return f"{n_missing}/{n_total} ({pct:.1f}%)"


def summarize(series: pd.Series, vtype: str, n_total: int) -> tuple[str, str]:
    """Return (summary_statistic, missing_string) for one variable."""
    n_missing = int(series.isna().sum())
    miss_str = fmt_missing(n_missing, n_total)
    non_null = series.dropna()
    if non_null.empty:
        return "n/a", miss_str

    if vtype == "binary":
        if is_yesno(series):
            pos = (series.astype(str).str.lower().str.strip() == "yes").sum()
        else:
            pos = int((series == 1).sum())
        n_obs = len(non_null)
        pct = pos / n_obs * 100
        return f"{pos} ({pct:.1f}%)", miss_str

    if vtype == "ordinal":
        q1, med, q3 = non_null.quantile([0.25, 0.5, 0.75])
        return f"{med:.0f} ({q1:.0f}-{q3:.0f})", miss_str

    if vtype == "continuous":
        q1, med, q3 = non_null.quantile([0.25, 0.5, 0.75])
        return f"{med:.1f} ({q1:.1f}-{q3:.1f})", miss_str

    if vtype == "date":
        parsed = parse_yyyymmdd(series).dropna()
        if parsed.empty:
            return "n/a", miss_str
        q1 = parsed.quantile(0.25).strftime("%Y-%m-%d")
        med = parsed.quantile(0.5).strftime("%Y-%m-%d")
        q3 = parsed.quantile(0.75).strftime("%Y-%m-%d")
        return f"{med} ({q1} - {q3})", miss_str

    if vtype == "time":
        mins = _time_to_minutes(series).dropna()
        if mins.empty:
            return "n/a", miss_str
        q1, med, q3 = mins.quantile([0.25, 0.5, 0.75])
        return (
            f"{_minutes_to_hhmm(med)} ({_minutes_to_hhmm(q1)}-{_minutes_to_hhmm(q3)})",
            miss_str,
        )

    # categorical / fallback
    vc = non_null.value_counts()
    n_obs = len(non_null)
    if len(vc) <= 5:
        parts = [f"{v}: {c} ({c / n_obs * 100:.1f}%)" for v, c in vc.items()]
        return "; ".join(parts), miss_str
    top = vc.head(3)
    parts = [f"{v}: {c} ({c / n_obs * 100:.1f}%)" for v, c in top.items()]
    return f"{len(vc)} categories; top: " + "; ".join(parts), miss_str


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop duplicates, filter to ischemic stroke, derive outcome variables."""
    n_raw = len(df)

    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().copy()
    # Drop rows explicitly labelled 'duplicate' in Type of event
    if "Type of event" in df.columns:
        mask_dup = df["Type of event"].astype("string").str.lower().eq("duplicate")
        df = df.loc[~mask_dup].copy()

    # 2. Filter to ischemic stroke
    if "Type of event" not in df.columns:
        raise SystemExit("Column 'Type of event' not found; cannot filter to ischemic stroke.")
    df = df.loc[df["Type of event"] == "Ischemic stroke"].copy()
    n_filtered = len(df)

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

    return df, n_raw, n_filtered


def build_metadata(df: pd.DataFrame, n_raw: int, n_filtered: int) -> dict:
    start_year = end_year = None
    if "Arrival at hospital" in df.columns:
        years = pd.to_numeric(df["Arrival at hospital"], errors="coerce").dropna()
        if not years.empty:
            years = (years.astype(int) // 10000)
            start_year = int(years.min())
            end_year = int(years.max())
    return {
        "n_patients": n_filtered,
        "n_raw_rows": n_raw,
        "n_dropped_preproc": n_raw - n_filtered,
        "start_year": start_year,
        "end_year": end_year,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _should_skip(col: str) -> bool:
    if col in SKIP_COLS:
        return True
    return any(p.match(str(col)) for p in SKIP_PATTERNS)


def build_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    rows = []
    for col in df.columns:
        if _should_skip(col):
            continue
        vtype = detect_type(df[col], col)
        stat, miss = summarize(df[col], vtype, n_total)
        rows.append(
            {
                "original_variable_name": col,
                "type": vtype,
                "summary_statistic": stat,
                "unit": UNITS.get(col, ""),
                "missing": miss,
                "variable_overlap": col in GVA_TO_SHENZEN,
                "shenzhen_variable_name": GVA_TO_SHENZEN.get(col, ""),
                "summary_statistic_format": STAT_FORMAT.get(vtype, ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summary table for GVA stroke registry")
    parser.add_argument("input", help="Path to Geneva stroke registry .xlsx file")
    parser.add_argument("--output-dir", default=".", help="Directory to write CSV outputs (default: .)")
    parser.add_argument("--summary-name", default="gva_summary_table.csv")
    parser.add_argument("--meta-name", default="gva_metadata.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load]  {in_path}")
    df = pd.read_excel(in_path)
    print(f"[load]  shape={df.shape}")

    df, n_raw, n_filtered = preprocess(df)
    print(f"[prep]  n_raw={n_raw}, n_after_dedup_and_filter={n_filtered}")

    summary_df = build_summary_rows(df)
    summary_path = out_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}  ({len(summary_df)} variables)")

    meta = build_metadata(df, n_raw, n_filtered)
    meta_path = out_dir / args.meta_name
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"[write] {meta_path}  -> {meta}")


if __name__ == "__main__":
    main()
