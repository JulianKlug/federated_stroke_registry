"""case_admission_id derivation — '<patient_id>_<EDS last 4 digits>'.

The registry 'Case ID' is a 12-char string; chars [8:-4] are the patient ID,
the last 4 are the EDS admission suffix. The EHR extraction carries the two
parts as separate columns. All three helpers land on the same id format, which
is also the loader-side derivation (architecture/fed_stroke/task.py).
"""
import pandas as pd


def create_ehr_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['eds_end_4digit'].str.zfill(4).astype(str)
    return case_identification_column


def create_registry_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    df = df.copy()
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['Case ID'].apply(lambda x: x[8:-4]).astype(str)
    if 'EDS_last_4_digits' not in df.columns:
        df['EDS_last_4_digits'] = df['Case ID'].apply(lambda x: x[-4:]).astype(str)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['EDS_last_4_digits'].str.zfill(4).astype(str)
    return case_identification_column


def build_case_admission_id(case_id: pd.Series) -> pd.Series:
    """Vectorized twin of create_registry_case_identification_column for a bare
    'Case ID' series (mirror of task.py:71-80)."""
    patient_id = case_id.str[8:-4].astype(str)
    eds_last_4 = case_id.str[-4:].astype(str).str.zfill(4)
    return patient_id + "_" + eds_last_4
