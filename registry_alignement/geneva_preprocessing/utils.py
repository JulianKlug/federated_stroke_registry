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