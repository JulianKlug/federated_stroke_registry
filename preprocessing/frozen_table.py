"""Raw wide table → frozen-schema table (site-agnostic logic).

Pure functions parameterised by the mapping tables in `preprocessing.mappings`; no site
names inside. Geneva binds its tables in architecture/preprocessing/preprocess_gva.
wide_to_frozen; the Shenzhen preprocessing binds its own tables to the same functions.

Every mismatch is a ValueError naming the column and the offending values — nothing is ever
silently NaN'd, and no row is ever dropped: the site table keeps every cohort admission; the
training label (and the rows that have one) is fed_stroke.schema's / the loader's decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mappings.frozen_schema import UNITLESS, base_unit, is_binary_unit
from .mappings.unit_aliases import normalize_unit_label


# ---------------------------------------------------------------------------
# 1. units
# ---------------------------------------------------------------------------
def convert_to_frozen_units(
    df: pd.DataFrame,
    raw_to_frozen: dict[str, str],
    frozen_units: dict[str, str],
    declared_units: dict[str, str],
    unit_aliases: dict[str, dict[str, float]],
    *,
    value_col_suffix: str = "_first_value",
    unit_col_suffix: str = "_first_unit",
) -> tuple[pd.DataFrame, dict]:
    """Per-row unit check + factor for every numeric (non-binary) mapped column, on RAW names.

    For each raw column in `raw_to_frozen` whose frozen unit is not binary, the observed unit
    of each row comes from the `{stem}{unit_col_suffix}` sibling column (EHR-derived
    `{stem}{value_col_suffix}` columns) or, for columns without a per-row label, from
    `declared_units[raw]`. The label is normalised (mappings.unit_aliases.normalize_unit_label)
    and looked up in `unit_aliases[normalised frozen unit]` → factor; the value is multiplied.

    Policy:
      - unknown label on a row WITH a value → error (never a silent NaN or pass-through);
      - value WITHOUT a label → assumed already in the frozen unit and counted per column as
        `n_assumed_unit`; on a dimensional unit (anything but UNITLESS) this adds a warning;
      - no unit source at all (no sibling column, not declared) → error;
      - errors are collected across all columns and raised as ONE ValueError after the loop.

    Returns (converted copy, report). The report is aggregate-only (no per-row data) and
    JSON-serialisable — it feeds the smoke report's "unit-check pass/fail":
        {pass, n_rows, n_columns_checked, n_assumed_unit_total, errors, warnings,
         columns: {frozen: {raw_col, frozen_unit, unit_source, n_values, n_assumed_unit,
                            n_converted_by_factor, labels_seen, factors_applied}}}
    """
    df = df.copy()
    errors: list[str] = []
    warnings: list[str] = []
    columns: dict[str, dict] = {}

    for raw, frozen in raw_to_frozen.items():
        if frozen not in frozen_units:
            errors.append(f"{raw!r} -> {frozen!r}: no frozen unit declared")
            continue
        unit = frozen_units[frozen]
        if is_binary_unit(unit):
            continue
        if raw not in df.columns:
            errors.append(f"{raw!r} -> {frozen!r}: column absent from the table")
            continue

        # Unit source: the per-row sibling label column, else the site's declared unit.
        unit_col = None
        if raw.endswith(value_col_suffix):
            candidate = raw[: -len(value_col_suffix)] + unit_col_suffix
            if candidate in df.columns:
                unit_col = candidate
        if unit_col is not None:
            labels = df[unit_col]
            unit_source = "per_row"
        elif raw in declared_units:
            labels = pd.Series(declared_units[raw], index=df.index)
            unit_source = "declared"
        else:
            errors.append(
                f"{raw!r} -> {frozen!r}: no unit source (no '*{unit_col_suffix}' sibling column "
                f"and not in declared_units)"
            )
            continue

        norm_unit = normalize_unit_label(base_unit(unit))   # 'no unit (mRS 0-6)' -> 'no unit'
        table = unit_aliases.get(norm_unit)
        if table is None:
            errors.append(f"{raw!r} -> {frozen!r}: no UNIT_ALIASES table for frozen unit {unit!r}")
            continue

        try:
            values = pd.to_numeric(df[raw], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            errors.append(f"{raw!r} -> {frozen!r}: non-numeric values ({exc})")
            continue
        has_value = values.notna()

        label_norm = labels.astype(object).map(normalize_unit_label, na_action="ignore").astype(object)
        blank = label_norm.isna() | (label_norm == "")
        has_label = has_value & ~blank

        factor = pd.to_numeric(label_norm.map(table), errors="coerce").astype("float64")
        unknown_mask = has_label & factor.isna()
        if unknown_mask.any():
            unknown = sorted(set(labels[unknown_mask].astype(str).str.strip()))
            errors.append(
                f"{raw!r} -> {frozen!r}: unknown unit label(s) {unknown} for frozen unit {unit!r}; "
                f"add to UNIT_ALIASES[{norm_unit!r}] with an explicit factor, or fix upstream"
            )
            continue
        factor = factor.where(has_label, 1.0)
        df[raw] = values * factor

        seen = labels[has_value].dropna().astype(str).str.strip()
        seen = seen[seen != ""]
        labels_seen = {str(k): int(v) for k, v in seen.value_counts().items()}
        factors_applied = {
            lbl: float(table[normalize_unit_label(lbl)])
            for lbl in labels_seen
            if table[normalize_unit_label(lbl)] != 1.0
        }
        n_values = int(has_value.sum())
        n_assumed = int((has_value & blank).sum())
        columns[frozen] = {
            "raw_col": raw,
            "frozen_unit": unit,
            "unit_source": unit_source,
            "n_values": n_values,
            "n_assumed_unit": n_assumed,
            "n_converted_by_factor": int((has_label & (factor != 1.0)).sum()),
            "labels_seen": labels_seen,
            "factors_applied": factors_applied,
        }
        if n_assumed and norm_unit != normalize_unit_label(UNITLESS):
            warnings.append(
                f"{frozen}: {n_assumed}/{n_values} values carry no unit label; assumed {unit!r}"
            )

    report = {
        "pass": not errors,
        "n_rows": int(len(df)),
        "n_columns_checked": len(columns),
        "n_assumed_unit_total": int(sum(c["n_assumed_unit"] for c in columns.values())),
        "errors": errors,
        "warnings": warnings,
        "columns": columns,
    }
    if errors:
        raise ValueError("unit check FAILED:\n  " + "\n  ".join(errors))
    return df, report


def format_unit_check(report: dict) -> str:
    """Printable per-column unit-check table (one line per frozen feature + warnings)."""
    lines = [
        f"[units] pass={report['pass']} columns={report['n_columns_checked']} "
        f"n_assumed_unit_total={report['n_assumed_unit_total']}"
    ]
    for frozen, c in report["columns"].items():
        lines.append(
            f"[units] {frozen:<34} {c['raw_col']:<34} {c['unit_source']:<8} "
            f"n={c['n_values']:>5} assumed={c['n_assumed_unit']:>5} "
            f"by_factor={c['n_converted_by_factor']:>4} labels={c['labels_seen']} "
            f"factors={c['factors_applied']}"
        )
    for w in report["warnings"]:
        lines.append(f"[units] WARNING {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. binaries (features and binary outcomes alike)
# ---------------------------------------------------------------------------
def encode_binaries(
    df: pd.DataFrame,
    encodings: dict[str, dict[str, int]],
    fillna: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Map raw vocabularies to {0, 1} float64 on RAW names; NaN stays NaN unless `fillna[col]`.

    Any observed non-NaN value outside the column's mapping raises (that includes an already
    numeric or boolean column — encode once, loudly).
    """
    df = df.copy()
    fillna = fillna or {}
    for col, mapping in encodings.items():
        if col not in df.columns:
            raise ValueError(f"encode_binaries: column {col!r} absent from the table")
        observed = set(df[col].dropna().unique())
        unexpected = observed - set(mapping)
        if unexpected:
            raise ValueError(
                f"encode_binaries: {col!r} has values not covered by its encoding: "
                f"{sorted(map(repr, unexpected))}; allowed: {sorted(mapping)}"
            )
        out = df[col].map(mapping)
        if col in fillna:
            out = out.fillna(fillna[col])
        df[col] = pd.to_numeric(out).astype("float64")
    return df


# ---------------------------------------------------------------------------
# 3. rename / select / finalize
# ---------------------------------------------------------------------------
def rename_to_frozen(df: pd.DataFrame, raw_to_frozen: dict[str, str]) -> pd.DataFrame:
    """Rename raw → frozen. FAILS on a raw key absent from the table (DataFrame.rename would
    silently ignore it) and on duplicate column names after the rename."""
    missing = [c for c in raw_to_frozen if c not in df.columns]
    if missing:
        raise ValueError(f"rename_to_frozen: raw columns absent from the table: {missing}")
    out = df.rename(columns=raw_to_frozen)
    dups = out.columns[out.columns.duplicated()].tolist()
    if dups:
        raise ValueError(f"rename_to_frozen: duplicate columns after rename: {sorted(set(dups))}")
    return out


def select_mapped_columns(
    df: pd.DataFrame, raw_to_frozen: dict[str, str], id_col: str = "case_admission_id"
) -> pd.DataFrame:
    """Keep exactly `[id_col, *raw_to_frozen.values()]` — what the MAPPING produced.

    Deliberately NOT selected by the frozen list: selecting by FROZEN_FEATURES would silently
    drop a mapping entry that outruns the frozen list and KeyError on a frozen name the mapping
    never produced. Selecting by the mapping keeps both defects visible for
    validate_frozen_columns to report as `unexpected=` / `missing=`. Everything unmapped —
    raw registry columns (incl. PII), `_first_datetime` / `_first_unit` siblings, non-frozen
    labs, derived intermediates — is dropped here.
    """
    if id_col not in df.columns:
        raise ValueError(f"select_mapped_columns: id column {id_col!r} absent from the table")
    wanted = [id_col, *raw_to_frozen.values()]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"select_mapped_columns: mapped columns absent after rename: {missing}")
    return df[wanted].copy()


def finalize_frozen_table(
    df: pd.DataFrame,
    frozen_features: list[str],
    outcome_cols: list[str],
    id_col: str = "case_admission_id",
    outcome_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Column order `[id_col, *frozen_features, *outcome_cols]`; features AND outcomes float64
    (NaN = not recorded — outcomes are never dropped here), each recorded outcome value inside
    its declared `outcome_ranges` entry (an encoding contract, hard error), id str and unique.
    Assumes validate_frozen_columns already passed."""
    ordered = [id_col, *frozen_features, *outcome_cols]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise ValueError(
            f"finalize_frozen_table: columns absent: {missing} (run validate_frozen_columns first)"
        )
    out = df[ordered].copy()
    for c in (*frozen_features, *outcome_cols):
        try:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"finalize_frozen_table: column {c!r} is not numeric ({exc})") from exc

    for c in outcome_cols:
        if outcome_ranges is None or c not in outcome_ranges:
            continue
        lo, hi = outcome_ranges[c]
        values = out[c]
        bad = values[(values < lo) | (values > hi)]
        if len(bad):
            raise ValueError(
                f"finalize_frozen_table: outcome {c!r} has {len(bad)} recorded values outside "
                f"its declared range [{lo:g}, {hi:g}] (min={values.min():g}, max={values.max():g}) "
                f"— an encoding bug, fix upstream"
            )

    if out[id_col].isna().any():
        raise ValueError(f"finalize_frozen_table: {id_col!r} has {int(out[id_col].isna().sum())} NaN ids")
    out[id_col] = out[id_col].astype(str)
    if not out[id_col].is_unique:
        dup = out[id_col][out[id_col].duplicated()]
        raise ValueError(
            f"finalize_frozen_table: {id_col!r} not unique: {dup.nunique()} duplicated ids, "
            f"e.g. {dup.unique()[:5].tolist()}"
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. public plausibility ranges
# ---------------------------------------------------------------------------
# Above this share of a feature's RECORDED values out of range, the cause is a unit or mapping
# bug (every value off by a factor -> essentially the whole feature out of range), not entry
# errors — fail, do not null away. Calibration (GVA registry, 2026-09-06): the timing variables
# carry 10-17 % out-of-range values from date/time entry errors (negative door-to-needle,
# onset-to-door beyond 7 d) and must pass; a unit bug that INFLATES values sits at ~100 %. A unit
# bug that SHRINKS values (e.g. creatinine in mg/dl) stays inside the range and is NOT caught
# here — that is the cross-site median divergence gate's job (roadmap smoke-test artifact).
MAX_OUT_OF_RANGE_FRAC = 0.25


def nullify_out_of_range(
    df: pd.DataFrame,
    frozen_ranges: dict[str, tuple[float, float]],
    frozen_features: list[str],
    *,
    max_out_of_range_frac: float = MAX_OUT_OF_RANGE_FRAC,
) -> tuple[pd.DataFrame, dict]:
    """Set feature values outside their public plausibility range to NaN, on FROZEN names.

    Runs after finalize_frozen_table (frozen names, frozen units, float64). For each feature,
    values < lo or > hi become NaN: an implausible value is an unknown, not a measurement with
    a slightly wrong magnitude, so it is nulled rather than clipped, and the loader then gives
    it the same missing sentinel every arm sees. The ranges are the DP bin grid (mappings.
    frozen_schema.FROZEN_RANGES == fed_stroke.dp.boost.FEATURE_RANGES): a value outside them is
    not representable there anyway (below lo it would fall into the missing bin silently).

    Missing values stay missing and are not counted. Binaries ({0, 1} inside (0, 1)) and every
    in-range value are untouched — a 0 inside a range that starts at 0 (age, glucose) is NOT
    caught here; that is the lower bounds' job, or a site-specific "0 means not recorded" rule.

    Fails loudly when any feature has more than `max_out_of_range_frac` of its recorded values
    out of range: that pattern is a unit or mapping bug and must be fixed upstream. Errors are
    collected across all features and raised as ONE ValueError.

    Returns (copy, report). The report is aggregate-only and JSON-serialisable — it feeds the
    smoke report next to the unit check:
        {pass, n_rows, n_nulled_total, max_out_of_range_frac, errors,
         columns: {feature: {lo, hi, n_values, n_below, n_above, frac_out_of_range,
                             min_seen, max_seen}}}
    """
    df = df.copy()
    errors: list[str] = []
    columns: dict[str, dict] = {}

    for feature in frozen_features:
        if feature not in frozen_ranges:
            errors.append(f"{feature!r}: no plausibility range declared")
            continue
        if feature not in df.columns:
            errors.append(f"{feature!r}: column absent from the table")
            continue
        lo, hi = frozen_ranges[feature]
        try:
            values = pd.to_numeric(df[feature], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            errors.append(f"{feature!r}: non-numeric values ({exc})")
            continue
        recorded = values.notna()
        below = values < lo            # NaN compares False -> missing is never counted
        above = values > hi
        n_values = int(recorded.sum())
        n_below, n_above = int(below.sum()), int(above.sum())
        frac = (n_below + n_above) / n_values if n_values else 0.0
        min_seen = float(values.min()) if n_values else None
        max_seen = float(values.max()) if n_values else None
        columns[feature] = {
            "lo": float(lo),
            "hi": float(hi),
            "n_values": n_values,
            "n_below": n_below,
            "n_above": n_above,
            "frac_out_of_range": float(frac),
            "min_seen": min_seen,
            "max_seen": max_seen,
        }
        if frac > max_out_of_range_frac:
            errors.append(
                f"{feature!r}: {n_below + n_above}/{n_values} recorded values outside "
                f"[{lo:g}, {hi:g}] ({100 * frac:.1f}% > {100 * max_out_of_range_frac:g}%) — "
                f"suspected unit or mapping bug (min_seen={min_seen:g}, max_seen={max_seen:g}); "
                f"fix upstream, do not null away"
            )
            continue
        if n_below or n_above:
            df[feature] = values.where(~(below | above), np.nan)

    report = {
        "pass": not errors,
        "n_rows": int(len(df)),
        "n_nulled_total": int(sum(c["n_below"] + c["n_above"] for c in columns.values())),
        "max_out_of_range_frac": float(max_out_of_range_frac),
        "errors": errors,
        "columns": columns,
    }
    if errors:
        raise ValueError("plausibility check FAILED:\n  " + "\n  ".join(errors))
    return df, report


def format_out_of_range(report: dict) -> str:
    """Printable plausibility report: header + one line per feature that had values nulled."""
    lines = [
        f"[ranges] pass={report['pass']} nulled_total={report['n_nulled_total']} "
        f"(gate: > {100 * report['max_out_of_range_frac']:g}% of a feature's recorded values fails)"
    ]
    for feature, c in report["columns"].items():
        if c["n_below"] or c["n_above"]:
            lines.append(
                f"[ranges] {feature:<34} [{c['lo']:g}, {c['hi']:g}] n={c['n_values']:>5} "
                f"below={c['n_below']:>4} above={c['n_above']:>4} "
                f"min_seen={c['min_seen']:g} max_seen={c['max_seen']:g}"
            )
    if len(lines) == 1:
        lines.append("[ranges] no out-of-range values")
    return "\n".join(lines)
