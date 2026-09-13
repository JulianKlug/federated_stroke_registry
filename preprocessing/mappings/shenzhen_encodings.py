"""Shenzhen raw-value vocabularies and declared units feeding the frozen table (mapping-as-data).

- applied on RAW column names, BEFORE the SHENZHEN_TO_FROZEN rename, by
  preprocessing.frozen_table (encode_binaries / convert_to_frozen_units);
- any raw value not listed makes encode_binaries RAISE — never a silent NaN;
- NaN stays NaN unless the column is in SHENZHEN_BINARY_FILLNA.

encoding conventions:
- `sex: 1 = female`
- thrombolysis started before arrival (drip-and-ship) counts as IVT.
"""
from __future__ import annotations

from .frozen_schema import (
    FROZEN_FEATURES,
    FROZEN_OUTCOMES,
    FROZEN_UNITS,
    SHENZHEN_TO_FROZEN,
    is_binary_unit,
)

# TODO(shenzhen): raw value -> {0, 1} per binary column, incl. the binary outcomes.
# Example shape: {"Sex": {"F": 1, "M": 0}, "Prior hypertension": {"是": 1, "否": 0}}
SHENZHEN_BINARY_ENCODINGS: dict[str, dict[str, int]] = {}

# TODO(shenzhen): columns where a BLANK means "no" (0), not "unknown". 
SHENZHEN_BINARY_FILLNA: dict[str, int] = {}

# TODO(shenzhen): binary raw columns already exported as {0, 1} (skip encode_binaries).
SHENZHEN_PRE_ENCODED_BINARIES: tuple[str, ...] = ()

# TODO(shenzhen): unit of record per NUMERIC raw column that carries no per-row unit label.
# through UNIT_ALIASES into FROZEN_UNITS, so a wrong entry silently rescales a feature.
SHENZHEN_DECLARED_UNITS: dict[str, str] = {
    # mg/L -> ng/ml is prefix algebra (x 1e3); the ASSAY convention is not. If Shenzhen
    # reports DDU where Geneva reports FEU (FEU ~ 2 x DDU), a residual ~2x survives this
    # conversion — a sign-off question, not expressible in UNIT_ALIASES.
    "D-dimer": "mg/L",
}


def _binary_raw_columns() -> set[str]:
    """Raw columns whose frozen target is binary — exactly those needing an encoding."""
    return {raw for raw, frozen in SHENZHEN_TO_FROZEN.items()
            if is_binary_unit(FROZEN_UNITS[frozen])}


def validate_shenzhen_encodings() -> None:
    """Geneva's import-time assertions, deferred: raise ValueError listing EVERY gap at once.

    Checks the mapping reaches all 44 frozen columns, that each binary raw column has an
    encoding (or is declared pre-encoded), that each numeric raw column has a declared unit,
    and that encodings only ever produce {0, 1}.
    """
    problems: list[str] = []

    unmapped = sorted({*FROZEN_FEATURES, *FROZEN_OUTCOMES} - set(SHENZHEN_TO_FROZEN.values()))
    if unmapped:
        problems.append(f"SHENZHEN_TO_FROZEN reaches no source for: {unmapped}")

    binary_raw = _binary_raw_columns()
    encoded = set(SHENZHEN_BINARY_ENCODINGS) | set(SHENZHEN_PRE_ENCODED_BINARIES)
    if binary_raw - encoded:
        problems.append(f"binary columns without an encoding: {sorted(binary_raw - encoded)}")
    if encoded - binary_raw:
        problems.append(f"encodings for non-binary / unmapped columns: {sorted(encoded - binary_raw)}")

    numeric_raw = set(SHENZHEN_TO_FROZEN) - binary_raw
    if numeric_raw - set(SHENZHEN_DECLARED_UNITS):
        problems.append(
            "numeric columns without a declared unit (or a per-row unit label): "
            f"{sorted(numeric_raw - set(SHENZHEN_DECLARED_UNITS))}"
        )

    if set(SHENZHEN_BINARY_FILLNA) - set(SHENZHEN_BINARY_ENCODINGS):
        problems.append("SHENZHEN_BINARY_FILLNA covers columns that are not encoded")
    bad = {v for m in SHENZHEN_BINARY_ENCODINGS.values() for v in m.values()} - {0, 1}
    if bad:
        problems.append(f"encodings must produce 0 or 1; got {sorted(bad)}")

    if problems:
        raise ValueError("Shenzhen mapping incomplete:\n  - " + "\n  - ".join(problems))
