"""Units for Geneva variables (continuous / ordinal / time fields).

Only variables with a meaningful unit appear here; anything not listed is
treated as unitless when the summary table is built.
"""
from __future__ import annotations

UNITS: dict[str, str] = {
    "Age (calc.)": "years",
    "Height": "cm",
    "Weight": "kg",
    "BMI": "kg/m²",
    "1st syst. bp": "mmHg",
    "1st diast. bp": "mmHg",
    "1st glucose": "mmol/L",
    "1st cholesterol total": "mmol/L",
    "1st cholesterol LDL": "mmol/L",
    "1st creatinine": "µmol/L",
    "Vit. K ag INR": "INR",
    "NIH on admission": "NIHSS points",
    "NIHSS 24h": "NIHSS points",
    "3M NIHSS": "NIHSS points",
    "GCS on admission": "GCS points",
    "Prestroke disability (Rankin)": "mRS",
    "3M mRS": "mRS",
    "Total rtPA dose": "mg",
    "IAT rtPA dose": "mg",
    "IAT urokinase dose": "IU",
    "Onset to treatment (min.)": "min",
    "Door to treatment (min.)": "min",
    "Onset to groin puncture (min.)": "min",
    "Door to groin puncture (min.)": "min",
    "Door to image (min.)": "min",
    "Duration of hospital stay (days)": "days",
    "FU Holter days": "days",
    "Average sleep": "hours",
    "Last night sleep": "hours",
}
