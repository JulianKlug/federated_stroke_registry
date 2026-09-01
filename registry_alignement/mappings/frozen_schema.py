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
# raw column (registry .xlsx or EHR-derived, post unit-conversion) -> frozen name.
# Registry columns are identity today (the frozen names ARE the Geneva registry
# spellings); EHR-derived features get real renames (e.g. "pv.pulse" -> ...).
GVA_TO_FROZEN: dict[str, str] = {
    "Age (calc.)": "Age (calc.)",
    "NIH on admission": "NIH on admission",
    "3M Death": "3M Death",
    # TODO(frozen-v2): EHR-derived features, e.g.
    # "pv.pulse": "<frozen name>",
    # "lab.result.sang.creatinine": "<frozen name>",
}

# --- Shenzhen ----------------------------------------------------------------------
# raw Shenzhen export column -> frozen name. Filled with the partner at schema
# sign-off (source names per GVA_TO_SHENZEN, read in reverse); run remotely by
# the partner's preprocessing, never on Geneva infrastructure.
SHENZHEN_TO_FROZEN: dict[str, str] = {
    # "Age": "Age (calc.)",
    # "Admission NIHSS Score": "NIH on admission",
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
