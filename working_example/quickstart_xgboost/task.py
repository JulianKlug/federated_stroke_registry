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

def load_data_gva(context: Context):
    """Load GVA data from the parquet file at the SuperNode's configured path.

    Contract: read the file at `context.node_config["data-path"]`. No
    cross-SuperNode partitioning inside — the file is authoritative. A local
    train/valid holdout is still built here because `client_app.py`'s
    `@evaluate()` needs a `valid_dmatrix`.
    """
    data_path = Path(context.node_config["data-path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {data_path}")

    feature_cols = ['Age (calc.)', 'NIH on admission']
    target_col = '3M Death'

    data_df = pd.read_parquet(data_path)

    n_rows = len(data_df)
    n_patients = data_df['case_admission_id'].str.split('_').str[0].nunique()
    label_balance = data_df[target_col].mean()
    print(
        f"load_data_gva({data_path.name}): rows={n_rows}, "
        f"unique_patients={n_patients}, label_balance={label_balance:.4f}"
    )

    train_df, valid_df, num_train, num_val = generate_splits(
        data_df, outcome=target_col, test_size=0.2, seed=42
    )
    train_df = train_df[feature_cols + [target_col]]
    valid_df = valid_df[feature_cols + [target_col]]

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
