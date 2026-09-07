"""Geneva raw-value vocabularies and declared units feeding the frozen table (mapping-as-data).

Applied on RAW column names, BEFORE the GVA_TO_FROZEN rename, by
preprocessing.frozen_table (encode_binaries / convert_to_frozen_units). Site-specific
('Biological', 'started before admission', ...) — a shenzhen_encodings.py sibling with the
same shape follows when the partner's export is known.

Rules: any raw value not listed makes encode_binaries raise (never a silent NaN). NaN stays
NaN unless the column is in GVA_BINARY_FILLNA. `wake_up_stroke` is already int {0,1}
(preprocessing.registry_cohort.preprocess_features) and is deliberately absent here.
The binary OUTCOMES ('3M Death', 'Death in hospital') are encoded exactly like binary
features and are NEVER used to drop rows — the label is chosen in fed_stroke.schema.

Decisions (2026-09-02): sex 1 = female; 'IVT with rtPA' == 'started before admission'
(drip-and-ship) counts as IVT; a blank 'MedHist Prost. heart valves' means no prosthesis (0).
"""
from __future__ import annotations

from .frozen_schema import FROZEN_UNITS, GVA_TO_FROZEN, is_binary_unit
from .units import UNITS

_YES_NO: dict[str, int] = {"yes": 1, "no": 0}

GVA_BINARY_ENCODINGS: dict[str, dict[str, int]] = {
    "Sex": {"Female": 1, "Male": 0},          # FROZEN_UNITS['sex'] == 'binary (1 = female)'
    "IVT with rtPA": {"yes": 1, "started before admission": 1, "no": 0},
    "IAT": _YES_NO,                           # -> EVT
    "MedHist Stroke": _YES_NO,
    "MedHist TIA": _YES_NO,
    "MedHist ICH": _YES_NO,
    "MedHist Hypertension": _YES_NO,
    "MedHist Diabetes": _YES_NO,
    "MedHist Hyperlipidemia": _YES_NO,
    "MedHist Atrial Fibr.": _YES_NO,
    "MedHist CHD": _YES_NO,
    "MedHist Prost. heart valves": {"Biological": 1, "Mechanical": 1},  # blank -> 0 (GVA_BINARY_FILLNA)
    "MedHist PAD": _YES_NO,
    "MedHist Smoking": _YES_NO,
    # binary outcomes (OPSUM-reconciled by registry_cohort.preprocess_outcome); NaN stays NaN
    "3M Death": _YES_NO,                      # -> death_3m
    "Death in hospital": _YES_NO,             # -> death_in_hospital
}

# Columns where a blank means "no", not "unknown".
GVA_BINARY_FILLNA: dict[str, int] = {"MedHist Prost. heart valves": 0}

# Binary raw columns that arrive already {0,1}-encoded (skip encode_binaries; cast to float later).
GVA_PRE_ENCODED_BINARIES: tuple[str, ...] = ("wake_up_stroke",)

# Registry-sourced numeric columns carry no per-row unit label; their unit is the registry's
# documented one, taken from mappings.units.UNITS (single source — KeyError here if UNITS ever
# loses one). EHR-derived *_first_value columns are NOT listed: they carry a *_first_unit
# sibling (preprocessing.first_values.assemble_wide) that is read per row. Includes the
# ordinal outcome '3M mRS' (unit 'mRS', unitless).
GVA_REGISTRY_NUMERIC_COLS: tuple[str, ...] = (
    "Age (calc.)",
    "Prestroke disability (Rankin)",
    "1st syst. bp",
    "1st diast. bp",
    "NIH on admission",
    "1st glucose",
    "GCS on admission",
    "ODT",
    "ONT",
    "DNT",
    "OPT",
    "3M mRS",
)
GVA_DECLARED_UNITS: dict[str, str] = {c: UNITS[c] for c in GVA_REGISTRY_NUMERIC_COLS}
# -> years, mRS, mmHg, mmHg, NIHSS points, mmol/L, GCS points, min, min, min, min, mRS

# --- import-time consistency (loud) ---------------------------------------------------
assert set(GVA_BINARY_ENCODINGS) <= set(GVA_TO_FROZEN), "encoded columns must be mapped"
assert set(GVA_BINARY_FILLNA) <= set(GVA_BINARY_ENCODINGS), "fillna only for encoded columns"
assert set(GVA_PRE_ENCODED_BINARIES) <= set(GVA_TO_FROZEN)
assert set(GVA_REGISTRY_NUMERIC_COLS) <= set(GVA_TO_FROZEN), "declared-unit columns must be mapped"
_binary_raw = {raw for raw, fz in GVA_TO_FROZEN.items() if is_binary_unit(FROZEN_UNITS[fz])}
assert _binary_raw == set(GVA_BINARY_ENCODINGS) | set(GVA_PRE_ENCODED_BINARIES), \
    "every binary frozen column (feature or outcome) needs a GVA encoding (or be pre-encoded), " \
    "and nothing else does"
_numeric_raw = set(GVA_TO_FROZEN) - _binary_raw
assert all(c.endswith("_first_value") or c in GVA_REGISTRY_NUMERIC_COLS for c in _numeric_raw), \
    "every numeric raw column needs a per-row *_first_unit sibling or a declared unit"
assert all(v in (0, 1) for m in GVA_BINARY_ENCODINGS.values() for v in m.values())
del _binary_raw, _numeric_raw
