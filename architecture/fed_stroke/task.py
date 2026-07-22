"""fed_stroke: shared task utilities (data loading, config helpers)."""
from pathlib import Path

import pandas as pd
import xgboost as xgb
from flwr.app import Context
from sklearn.model_selection import train_test_split

from fed_stroke.schema import FEATURE_COLS, TARGET_COL

# The patient-disjoint hold-out partition is pinned to a FIXED seed, deliberately
# NOT a config key: it must never vary, or the HELD patient set would shift between
# the HPO search runs and the winner's hold-out report, silently leaking selection
# patients into the "hold-out" (spec 1.1.a §4.5/§4.6, Decision 11).
HOLDOUT_PARTITION_SEED = 42


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


def resolve_run_split(data, outcome, split_seed=42, holdout_frac=0.0,
                      holdout_eval=False):
    """The single train/valid split contract, shared by the federated client, the
    offline scorer, and the pooled baseline (spec 1.1.a §4.5).

    Three modes, selected by (holdout_frac, holdout_eval):

    - holdout_frac == 0.0 -> FLAT (today's behavior): a plain
      generate_splits(test_size=0.2, seed=split_seed) train/valid split. At the
      defaults (split_seed=42, holdout_frac=0.0) this is byte-identical to the old
      hardcoded `generate_splits(..., test_size=0.2, seed=42)` call, so every
      existing federated run and test is unchanged.

    - holdout_frac > 0.0, holdout_eval=False -> SEARCH repeat: first reserve a
      patient-disjoint HELD set by partitioning DEV/HELD with the FIXED
      HOLDOUT_PARTITION_SEED (never the search seed), then sub-split DEV into
      train/valid by split_seed. HELD is never returned here — it is excluded from
      every search repeat by the fixed partition, not by a seed convention.

    - holdout_frac > 0.0, holdout_eval=True -> HOLD-OUT report: same fixed DEV/HELD
      partition; train = all of DEV, valid = HELD. Because HELD sat outside every
      search repeat, this is a genuine patient-disjoint hold-out.

    Returns (train_df, valid_df) with the full column set (incl. patient_id), exactly
    like generate_splits — callers subset to FEATURE_COLS/TARGET_COL as needed.
    Patient-level partitioning is delegated to generate_splits, so the DEV/HELD
    boundary respects the multiple-admission (patient_id) guard.
    """
    if holdout_frac == 0.0:
        train_df, valid_df, _, _ = generate_splits(
            data, outcome=outcome, test_size=0.2, seed=split_seed
        )
        return train_df, valid_df

    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(
            f"holdout_frac must be in [0.0, 1.0); got {holdout_frac}"
        )

    # Reserve the patient-disjoint HELD set with the FIXED partition seed.
    dev_df, held_df, _, _ = generate_splits(
        data, outcome=outcome, test_size=holdout_frac, seed=HOLDOUT_PARTITION_SEED
    )

    if holdout_eval:
        # HOLD-OUT report: train on all of DEV, evaluate on the disjoint HELD set.
        return dev_df, held_df

    # SEARCH repeat: sub-split DEV into train/valid by the (varying) search seed.
    train_df, valid_df, _, _ = generate_splits(
        dev_df, outcome=outcome, test_size=0.2, seed=split_seed
    )
    return train_df, valid_df


def _resolve_context_split(context: Context):
    """Shared load + split contract for the SuperNode's configured data file.

    Reads `context.node_config["data-path"]` (no cross-SuperNode partitioning — the file is
    authoritative), then applies the one `resolve_run_split` contract driven by run_config (all
    default to today's flat seed-42 split, so an unconfigured run is byte-identical). Returns
    `(train_df, valid_df, num_train, num_val)` already subset to FEATURE_COLS + TARGET_COL, so
    both the DMatrix path (`load_data_gva`) and the raw-array DP path (`load_data_arrays`) train
    and evaluate on EXACTLY the same split at matched split-seed/holdout-frac/holdout-eval."""
    data_path = Path(context.node_config["data-path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {data_path}")

    feature_cols = FEATURE_COLS
    target_col = TARGET_COL

    data_df = pd.read_parquet(data_path)

    n_rows = len(data_df)
    n_patients = data_df['case_admission_id'].str.split('_').str[0].nunique()
    label_balance = data_df[target_col].mean()
    print(
        f"load_data_gva({data_path.name}): rows={n_rows}, "
        f"unique_patients={n_patients}, label_balance={label_balance:.4f}"
    )

    # Split contract driven by run_config (all default to today's flat seed-42
    # split, so an unconfigured run is byte-identical). HPO varies split-seed for
    # repeated CV and sets holdout-frac to reserve a patient-disjoint HELD set;
    # holdout-eval flips the winner's report run onto HELD (spec 1.1.a §4.5/§4.6).
    split_seed = context.run_config.get("split-seed", 42)
    holdout_frac = context.run_config.get("holdout-frac", 0.0)
    holdout_eval = context.run_config.get("holdout-eval", False)
    train_df, valid_df = resolve_run_split(
        data_df, outcome=target_col, split_seed=split_seed,
        holdout_frac=holdout_frac, holdout_eval=holdout_eval,
    )
    num_train = len(train_df)
    num_val = len(valid_df)
    train_df = train_df[feature_cols + [target_col]]
    valid_df = valid_df[feature_cols + [target_col]]
    return train_df, valid_df, num_train, num_val


def load_data_gva(context: Context):
    """Load GVA data from the parquet file at the SuperNode's configured path.

    Contract: read the file at `context.node_config["data-path"]`. No
    cross-SuperNode partitioning inside — the file is authoritative. A local
    train/valid holdout is still built here because `client_app.py`'s
    `@evaluate()` needs a `valid_dmatrix`.
    """
    feature_cols = FEATURE_COLS
    target_col = TARGET_COL
    train_df, valid_df, num_train, num_val = _resolve_context_split(context)

    train_dmatrix = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col])
    valid_dmatrix = xgb.DMatrix(valid_df[feature_cols], label=valid_df[target_col])

    return train_dmatrix, valid_dmatrix, num_train, num_val


def load_data_arrays(context: Context):
    """Raw-numpy analog of load_data_gva for the DP learner (§4.5), which consumes `X, y` arrays
    (NOT a DMatrix). Returns `(X_train, y_train, X_valid, y_valid, num_train, num_val)` via the
    SAME `_resolve_context_split` contract, so the DP arm trains/evaluates on exactly the split
    the XGB arm does at matched split-seed/holdout-frac/holdout-eval. X columns are exactly
    FEATURE_COLS order (== FEATURE_RANGES order, §3.9); y is TARGET_COL. Additive — the DMatrix
    path is untouched."""
    feature_cols = FEATURE_COLS
    target_col = TARGET_COL
    train_df, valid_df, num_train, num_val = _resolve_context_split(context)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_valid = valid_df[feature_cols].to_numpy(dtype=float)
    y_valid = valid_df[target_col].to_numpy(dtype=float)

    return X_train, y_train, X_valid, y_valid, num_train, num_val


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
