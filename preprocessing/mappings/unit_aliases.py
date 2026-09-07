"""Observed unit label → multiplicative factor INTO the frozen unit (mapping-as-data).

    UNIT_ALIASES[normalize_unit_label(frozen_unit)][normalize_unit_label(observed_label)] = factor
    frozen_value = observed_value * factor

Site-agnostic: a label is a label whether Geneva or Shenzhen writes it. Adding a synonym here
is a site convenience, NOT a contract change (the contract is frozen_schema.FROZEN_UNITS) —
which is why this lives outside frozen_schema.py.

Scope: metric-prefix algebra and spelling synonyms ONLY. Mass↔molar conversions (mg/dl →
µmol/l) are substance-specific and deliberately NOT expressible here — if such a label ever
appears, preprocessing.frozen_table.convert_to_frozen_units fails loudly and a per-column
override table becomes a follow-up. Every table contains its own frozen unit at 1.0. Every
label observed in the GVA EHR (out/gva_first_values_after_admission.csv) and every registry
declared unit (mappings.units.UNITS) is present; anything else raises at conversion time.
Keep the set small: observed labels + their immediate metric-prefix neighbours.
"""
from __future__ import annotations

import re

_GREEK_MU = "μ"   # μ GREEK SMALL LETTER MU — defensive only
_MICRO = "µ"      # µ MICRO SIGN — what the GVA EHR and frozen_feature_names.xlsx actually use


def normalize_unit_label(label) -> str:
    """Canonical spelling of a unit label.

    Strip + collapse whitespace; μ → µ; drop a trailing '.' ('puls./min.' → 'puls./min');
    litre 'L' → 'l' at word end ('mmol/L' → 'mmol/l', 'ng/mL' → 'ng/ml', 'G/L' → 'G/l').
    Case is otherwise PRESERVED: 'G/l' (giga per litre, cell counts) and 'g/l' (gram per
    litre, fibrinogen) are different frozen units and must not collapse.
    """
    s = " ".join(str(label).split())
    s = s.replace(_GREEK_MU, _MICRO)
    s = s.rstrip(".")
    s = re.sub(r"L\b", "l", s)
    return s


UNIT_ALIASES: dict[str, dict[str, float]] = {
    "years":   {"years": 1.0},
    "no unit": {"no unit": 1.0, "NIHSS points": 1.0, "GCS points": 1.0, "mRS": 1.0},
    "°C":      {"°C": 1.0},
    "bpm":     {"bpm": 1.0, "/min": 1.0, "puls./min": 1.0, "cycles/min": 1.0},
    "mmHg":    {"mmHg": 1.0},
    "mmol/l":  {"mmol/l": 1.0, "µmol/l": 1e-3},          # serves glucose 'mmol/L' and LDL/urea 'mmol/l'
    "G/l":     {"G/l": 1.0},                              # giga per litre (cell counts); NOT 'g/l'
    "mg/l":    {"mg/l": 1.0, "µg/l": 1e-3, "g/l": 1e3},
    "g/l":     {"g/l": 1.0, "mg/l": 1e-3},                # gram per litre (fibrinogen); NOT 'G/l'
    "ng/ml":   {"ng/ml": 1.0, "µg/l": 1.0, "mg/l": 1e3, "µg/ml": 1e3},
    "%":       {"%": 1.0},
    "U/l":     {"U/l": 1.0},
    "µmol/l":  {"µmol/l": 1.0, "mmol/l": 1e3, "nmol/l": 1e-3},
    "min":     {"min": 1.0},
}

for _unit, _table in UNIT_ALIASES.items():
    assert _unit == normalize_unit_label(_unit), f"UNIT_ALIASES key {_unit!r} is not normalized"
    assert _table.get(_unit) == 1.0, f"UNIT_ALIASES[{_unit!r}] must contain itself at 1.0"
    for _label, _factor in _table.items():
        assert _label == normalize_unit_label(_label), f"UNIT_ALIASES[{_unit!r}]: {_label!r} not normalized"
        assert _factor > 0, f"UNIT_ALIASES[{_unit!r}][{_label!r}]: non-positive factor"
del _unit, _table, _label, _factor
