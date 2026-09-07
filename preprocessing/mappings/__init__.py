"""Mapping tables shared by the registry-alignment and architecture layers.

Keeping mappings in their own modules lets them be edited without touching
preprocessing/summarization logic, and makes each mapping easy to review/diff.
"""
from .frozen_schema import (
    FROZEN_FEATURES,
    FROZEN_OUTCOME_RANGES,
    FROZEN_OUTCOMES,
    FROZEN_RANGES,
    FROZEN_UNITS,
    GVA_TO_FROZEN,
    ID_COL,
    SHENZHEN_TO_FROZEN,
    UNITLESS,
    base_unit,
    is_binary_unit,
    validate_frozen_columns,
)
from .gva_encodings import (
    GVA_BINARY_ENCODINGS,
    GVA_BINARY_FILLNA,
    GVA_DECLARED_UNITS,
    GVA_PRE_ENCODED_BINARIES,
    GVA_REGISTRY_NUMERIC_COLS,
)
from .gva_to_shenzen import GVA_TO_SHENZEN
from .unit_aliases import UNIT_ALIASES, normalize_unit_label
from .unit_conversions import UNIT_CONVERSIONS
from .units import UNITS

__all__ = [
    # frozen contract
    "FROZEN_FEATURES", "FROZEN_OUTCOMES", "FROZEN_OUTCOME_RANGES", "FROZEN_RANGES",
    "FROZEN_UNITS", "ID_COL", "UNITLESS", "base_unit", "is_binary_unit",
    "validate_frozen_columns",
    # per-site raw -> frozen
    "GVA_TO_FROZEN", "SHENZHEN_TO_FROZEN",
    # Geneva vocabularies / declared units
    "GVA_BINARY_ENCODINGS", "GVA_BINARY_FILLNA", "GVA_DECLARED_UNITS",
    "GVA_PRE_ENCODED_BINARIES", "GVA_REGISTRY_NUMERIC_COLS",
    # units
    "UNIT_ALIASES", "normalize_unit_label", "UNITS",
    # legacy comparison-plot helpers (not on the frozen path)
    "GVA_TO_SHENZEN", "UNIT_CONVERSIONS",
]
