"""Geneva -> Shenzen semantic variable mapping.

Only Geneva variables that overlap with a Shenzen variable appear here. Edits here flow through to the
summary table's `variable_overlap` / `shenzhen_variable_name` columns.

When a Geneva concept maps to multiple Shenzen fields, the value is a single
string listing them separated by " / ".
"""
from __future__ import annotations

GVA_TO_SHENZEN: dict[str, str] = {
    "Case ID": "Outpatient Number / ID Number / Medical Record Number",
    "Last name": "Name",
    "First name": "Name",
    "DOB": "Date of Birth",
    "Age (calc.)": "Age",
    "Sex": "Gender",
    "Ethnicity": "Ethnicity",
    "Insurance": "Insurance Type",
    "Prestroke living situation": "Living Situation Before Onset",
    "Prestroke disability (Rankin)": "Premorbid mRS Score",
    "Time of symptom onset known": "Exact Onset Time",
    "Onset date": "Exact Onset Time / Estimated Onset Time",
    "Onset time": "Exact Onset Time / Estimated Onset Time",
    "Wake-up date": "Is Wake-Up Stroke / Last Known Well Time",
    "Wake-up time": "Is Wake-Up Stroke / Last Known Well Time",
    "Arrival at hospital": "Arrival Time at Hospital",
    "Arrival time": "Arrival Time at Hospital",
    "Transport": "Mode of Arrival",
    "Referral": "Transfer During Transport",
    "Patient referred from": "Transfer During Transport",
    "Reason of referral": "Transfer During Transport",
    "NIH on admission": "Admission NIHSS Score",
    "GCS on admission": "First GCS Score",
    "1st syst. bp": "Admission Systolic Blood Pressure",
    "1st diast. bp": "Admission Diastolic Blood Pressure",
    "Height": "Height",
    "Weight": "Weight",
    "1st brain imaging type": "Head CT Performed / MRI Completion Time",
    "1st brain imaging date": "Head CT Time / MRI Completion Time",
    "1st brain imaging time": "Head CT Time / MRI Completion Time",
    "Acute perf. imaging type": "CTP / PWI / ASL Completion Time",
    "1st vascular imaging type": "CTA / MRA / Carotid imaging Completion Time",
    "IVT with rtPA": "Intravenous Thrombolysis Performed at This Hospital",
    "IAT": "Vascular Intervention or Surgery Performed",
    "MedHist Hypertension": "Past Medical History / Years with Hypertension",
    "MedHist Diabetes": "Past Medical History / Years with Diabetes Mellitus",
    "MedHist Hyperlipidemia": "Past Medical History",
    "MedHist Stroke": "Past Medical History",
    "MedHist TIA": "Past Medical History",
    "MedHist ICH": "Past Medical History",
    "MedHist Atrial Fibr.": "Past Medical History",
    "MedHist CHD": "Past Medical History",
    "MedHist Smoking": "Smoking History",
    "1st glucose": "Fasting Blood Glucose / Rapid Blood Glucose Value",
    "1st cholesterol LDL": "LDL-C",
    "1st creatinine": "Serum Creatinine",
    "Discharge destination": "Discharge Status",
    "Discharge date": "Discharge Date",
    "Duration of hospital stay (days)": "Discharged Within 7 Days of Onset (partial)",
}
