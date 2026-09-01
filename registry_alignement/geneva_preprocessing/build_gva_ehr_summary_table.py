"""
Build a summary table of the Geneva EHR first-value extraction.

Input  : the CSV produced by ``build_gva_first_values_table.py``
         (one row per ischemic-stroke admission, three columns per variable:
          ``<var>_first_value``, ``<var>_first_datetime``, ``<var>_first_unit``).
Outputs:
  - <output-dir>/gva_ehr_summary_table.csv  : per-variable summary row
  - <output-dir>/gva_ehr_metadata.csv       : cohort-level metadata

Format mirrors ``build_gva_summary_table.py``: the same column set, the same
type-detection / summary helpers, and the same metadata shape, so the EHR
table can sit beside the registry summary unchanged downstream.

The summarized variables are the ``<var>_first_value`` columns. Their unit is
read from the matching ``<var>_first_unit`` column (modal non-null value).
``variable_overlap`` is determined by mapping the EHR variable to its
registry-equivalent column (when one exists) and looking that up in
``GVA_TO_SHENZEN``.

Usage:
    python build_gva_ehr_summary_table.py out/gva_first_values_after_admission.csv
    python build_gva_ehr_summary_table.py <input.csv> --output-dir out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.mappings import GVA_TO_SHENZEN, UNITS  # noqa: E402
from registry_alignement.build_gva_summary_table import (  # noqa: E402
    STAT_FORMAT,
    detect_type,
    fmt_missing,
    summarize,
)


# Map each EHR first-value variable to the registry column it corresponds to,
# so we can reuse GVA_TO_SHENZEN / UNITS for overlap and unit annotation.
# Variables with no direct registry equivalent are absent here.
EHR_TO_REGISTRY: dict[str, str] = {
    "creatinine":  "1st creatinine",
    "ldl_calc":    "1st cholesterol LDL",
}

VALUE_SUFFIX = "_first_value"
UNIT_SUFFIX = "_first_unit"


def _modal_unit(series: pd.Series) -> str:
    """Return the most common non-null unit string in ``series`` (empty if none)."""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    return s.mode().iloc[0]


def _registry_name(var_base: str) -> str:
    return EHR_TO_REGISTRY.get(var_base, "")


def build_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    rows: list[dict] = []

    # Optionally summarize admission_date if present, so the EHR table
    # exposes the cohort time-window the same way the registry does.
    if "admission_date" in df.columns:
        series = pd.to_datetime(df["admission_date"], errors="coerce")
        n_missing = int(series.isna().sum())
        miss_str = fmt_missing(n_missing, n_total)
        non_null = series.dropna()
        if non_null.empty:
            stat = "n/a"
        else:
            q1 = non_null.quantile(0.25).strftime("%Y-%m-%d")
            med = non_null.quantile(0.5).strftime("%Y-%m-%d")
            q3 = non_null.quantile(0.75).strftime("%Y-%m-%d")
            stat = f"{med} ({q1} - {q3})"
        rows.append(
            {
                "original_variable_name": "admission_date",
                "type": "date",
                "summary_statistic": stat,
                "unit": "",
                "missing": miss_str,
                "variable_overlap": "Arrival at hospital" in GVA_TO_SHENZEN,
                "shenzhen_variable_name": GVA_TO_SHENZEN.get("Arrival at hospital", ""),
                "summary_statistic_format": STAT_FORMAT["date"],
            }
        )

    value_cols = [c for c in df.columns if c.endswith(VALUE_SUFFIX)]
    for col in value_cols:
        var_base = col[: -len(VALUE_SUFFIX)]
        unit_col = f"{var_base}{UNIT_SUFFIX}"

        series = pd.to_numeric(df[col], errors="coerce")
        vtype = detect_type(series, col)
        stat, miss = summarize(series, vtype, n_total)

        # Prefer the modal unit observed in the extraction; fall back to the
        # canonical registry unit if the EHR didn't carry one (e.g. INR).
        unit = _modal_unit(df[unit_col]) if unit_col in df.columns else ""
        if not unit:
            unit = UNITS.get(_registry_name(var_base), "")

        reg_name = _registry_name(var_base)
        rows.append(
            {
                "original_variable_name": col,
                "type": vtype,
                "summary_statistic": stat,
                "unit": unit,
                "missing": miss,
                "variable_overlap": bool(reg_name) and reg_name in GVA_TO_SHENZEN,
                "shenzhen_variable_name": GVA_TO_SHENZEN.get(reg_name, ""),
                "summary_statistic_format": STAT_FORMAT.get(vtype, ""),
            }
        )
    return pd.DataFrame(rows)


def build_metadata(df: pd.DataFrame) -> dict:
    start_year = end_year = None
    if "admission_date" in df.columns:
        parsed = pd.to_datetime(df["admission_date"], errors="coerce").dropna()
        if not parsed.empty:
            start_year = int(parsed.dt.year.min())
            end_year = int(parsed.dt.year.max())
    return {
        "n_patients": len(df),
        "start_year": start_year,
        "end_year": end_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summary table for the GVA EHR first-value extraction."
    )
    parser.add_argument(
        "input",
        help="CSV from build_gva_first_values_table.py "
             "(e.g. out/gva_first_values_after_admission.csv).",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--summary-name", default="gva_ehr_summary_table.csv")
    parser.add_argument("--meta-name", default="gva_ehr_metadata.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load]  {in_path}")
    df = pd.read_csv(in_path)
    print(f"[load]  shape={df.shape}")

    summary_df = build_summary_rows(df)
    summary_path = out_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}  ({len(summary_df)} variables)")

    meta = build_metadata(df)
    meta_path = out_dir / args.meta_name
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"[write] {meta_path}  -> {meta}")


if __name__ == "__main__":
    main()
