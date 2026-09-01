"""Mapping tables shared by the registry-alignment and architecture layers.

Keeping mappings in their own modules lets them be edited without touching
preprocessing/summarization logic, and makes each mapping easy to review/diff.
"""
from .frozen_schema import GVA_TO_FROZEN, SHENZHEN_TO_FROZEN, validate_frozen_columns
from .gva_to_shenzen import GVA_TO_SHENZEN
from .unit_conversions import UNIT_CONVERSIONS
from .units import UNITS

__all__ = ["GVA_TO_FROZEN", "GVA_TO_SHENZEN", "SHENZHEN_TO_FROZEN",
           "UNITS", "UNIT_CONVERSIONS", "validate_frozen_columns"]
