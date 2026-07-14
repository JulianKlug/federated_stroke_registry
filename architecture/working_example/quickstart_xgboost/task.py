"""quickstart_xgboost: A Flower / XGBoost app."""
import pandas as pd
from datasets import Dataset
import numpy as np
import xgboost as xgb
from pathlib import Path
from flwr.app import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.model_selection import train_test_split

def generate_splits(data, outcome, test_size, seed,
                    test_pids_path=None, train_pids_path=None):
    """
    Splits the input data into training and testing sets based on patient IDs and a specified outcome.

    Args:
        data: The input dataset.
        outcome: The specific outcome label to consider for splitting.
        test_size (float): The proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.

    If using predefined test and train patient ids:
        test_pids_path (str): Path to the test patient IDs file.
        train_pids_path (str): Path to the train patient IDs file.

    Returns:
        list: A list of tuples containing the patient IDs for training and validation sets.

    Raises:
        ValueError: If the input data is corrupted or does not meet the required format.
    """

    data['patient_id'] = data['case_admission_id'].apply(lambda x: x.split('_')[0])
    
    """
    SPLITTING DATA
    Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
    would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
    """
    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids = data.patient_id.unique().tolist()
    all_outcomes = data[data.patient_id.isin(all_pids)]
    # keep only maximum outcome per patient (in case of multiple admissions per patient)
    all_outcomes = all_outcomes.groupby('patient_id')[outcome].max().reset_index()
    # let all outcomes be an array of outcomes corresponding to the unique patient ids
    all_outcomes = all_outcomes[outcome].values.tolist()

    # Using predefined test and train patient ids
    if test_pids_path is not None:
        pid_test = pd.read_csv(test_pids_path, dtype=str).patient_id.tolist()
        pid_train = pd.read_csv(train_pids_path, dtype=str).patient_id.tolist()

        y_pid_test = [1 if pid in data.patient_id.values else 0 for pid in pid_test]
        y_pid_train = [1 if pid in data.patient_id.values else 0 for pid in pid_train]

    else:
        pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids,
                                                                        all_outcomes,
                                                                        stratify=all_outcomes,
                                                                        test_size=test_size,
                                                                        random_state=seed)

    train_set = data[data.patient_id.isin(pid_train)]        
    test_set = data[data.patient_id.isin(pid_test)]
    num_train = len(train_set)
    num_test = len(test_set)

    return train_set, test_set, num_train, num_test

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


def load_data_gva(context: Context):
    """Load GVA data."""
    data_path = Path('/mnt/hdd1/datasets/GVA_stroke_registry/stroke_registry_post_hoc_modified.xlsx')
    # data_path = Path(context.node_config["data-path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {data_path}")

    df = pd.read_excel(data_path)   
    df['case_admission_id'] = create_registry_case_identification_column(df)

    # todo: read features and target from config instead of hardcoding
    feature_cols = ['Age (calc.)', 'NIH on admission']
    target_col = '3M Death'
    data_df = df[['case_admission_id', *feature_cols, target_col]].copy()
   
    # todo: features & labels should be preprocessed before, along with dropping of duplicates
    # preprocess target col to binary labels from 'yes'/'no' to 1/0
    data_df[target_col] = data_df[target_col].apply(lambda x: 1 if x == 'yes' else 0 if x == 'no' else np.nan)
        
    n_pre_drop = data_df.case_admission_id.nunique()
    # drop rows with NaN values in the target column
    data_df = data_df.dropna(subset=[target_col])
    n_post_drop = data_df.case_admission_id.nunique()
    print(f"Dropped {n_pre_drop - n_post_drop} cids with NaN values in the target column '{target_col}', for a remaining of {n_post_drop} unique cases.")

    # Train/test splitting - split around ids
    train_df, valid_df, num_train, num_val = generate_splits(data_df, outcome=target_col, test_size=0.2, seed=42)    
    # drop columns except features and target
    train_df = train_df[feature_cols + [target_col]]
    valid_df = valid_df[feature_cols + [target_col]]

    # Reformat data to DMatrix for xgboost
    train_dmatrix = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col])
    valid_dmatrix = xgb.DMatrix(valid_df[feature_cols], label=valid_df[target_col])

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
