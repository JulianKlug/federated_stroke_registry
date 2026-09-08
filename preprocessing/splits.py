"""Patient-level disjoint splits of per-admission tables."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .mappings.frozen_schema import ID_COL


def stratified_patient_split(
    df: pd.DataFrame, seed: int, target_col: str = "3M Death", id_col: str = ID_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """50/50 split at the patient level, stratified on ``target_col``. Reduces
    each patient to their max outcome to handle rare multi-admission patients
    (same trick as task.py:41-47). ``id_col`` is '<patient key>_<admission>':
    the pseudonymous ID_COL of a node table, or case_ids.RAW_ID_COL upstream."""
    df = df.copy()
    df["patient_id"] = df[id_col].str.split("_").str[0]

    per_patient_outcome = df.groupby("patient_id")[target_col].max().reset_index()
    pids = per_patient_outcome["patient_id"].tolist()
    outcomes = per_patient_outcome[target_col].tolist()

    pids_a, pids_b, _, _ = train_test_split(
        pids,
        outcomes,
        stratify=outcomes,
        test_size=0.5,
        random_state=seed,
    )
    half_a = df[df["patient_id"].isin(pids_a)].drop(columns=["patient_id"])
    half_b = df[df["patient_id"].isin(pids_b)].drop(columns=["patient_id"])
    return half_a, half_b
