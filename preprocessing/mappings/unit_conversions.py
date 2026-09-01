"""Unit conversions between Geneva (GVA) and Shenzhen reporting conventions.

Keyed by Geneva variable name. Each entry describes the multiplier to apply
to the Geneva value to express it in the Shenzhen-side unit, so that the
two cohorts can be plotted on a common scale.

    shenzhen_value = geneva_value * factor

Add a `note` whenever the conversion carries an assumption (e.g. assay
reporting convention) that downstream readers should be aware of.
"""
from __future__ import annotations

UNIT_CONVERSIONS: dict[str, dict] = {
    # D-dimer: Geneva reports ng/mL, Shenzhen reports mg/L.
    # 1 mg/L = 1000 ng/mL  ->  divide Geneva by 1000.
    # Assumes both cohorts use the same reporting convention (FEU vs DDU);
    # FEU ≈ 2 × DDU, so a residual ~2× factor may remain if conventions differ.
    "d_dimeres_first_value": {
        "from_unit": "ng/mL",
        "to_unit": "mg/L",
        "factor": 1 / 1000,
        "note": "Assumes both cohorts report in the same convention (FEU vs DDU).",
    },
}
