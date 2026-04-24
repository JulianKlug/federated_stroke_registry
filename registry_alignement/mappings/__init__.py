"""Mapping tables used by build_gva_summary_table.py.

Keeping mappings in their own modules lets them be edited without touching
summarization logic, and makes each mapping easy to review/diff.
"""
from .gva_to_shenzen import GVA_TO_SHENZEN
from .units import UNITS

__all__ = ["GVA_TO_SHENZEN", "UNITS"]
