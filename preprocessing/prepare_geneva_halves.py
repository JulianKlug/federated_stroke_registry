"""One-off script: split the Geneva stroke registry into two patient-ID-stratified
50/50 parquet halves.

Roadmap step 1.a.ii. Produces `out/geneva_half_A.parquet` and
`out/geneva_half_B.parquet` from the raw Geneva Excel. Halves are disjoint at
the patient level (no patient appears in both), stratified on the `3M Death`
outcome, and deterministic under a fixed seed.

Run from the repo root: `python preprocessing/prepare_geneva_halves.py`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_XLSX = Path(
    "/mnt/hdd1/datasets/GVA_stroke_registry/stroke_registry_post_hoc_modified.xlsx"
)
OUT_DIR = REPO_ROOT / "out"
FEATURE_COLS = ["Age (calc.)", "NIH on admission"]
TARGET_COL = "3M Death"


def build_case_admission_id(case_id: pd.Series) -> pd.Series:
    """Mirror of task.py:71-80. `Case ID` is a 12-char string; middle 4 chars
    are the patient ID, last 4 are the EDS admission suffix."""
    patient_id = case_id.str[8:-4].astype(str)
    eds_last_4 = case_id.str[-4:].astype(str).str.zfill(4)
    return patient_id + "_" + eds_last_4


def stratified_patient_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """50/50 split at the patient level, stratified on TARGET_COL. Reduces
    each patient to their max outcome to handle rare multi-admission patients
    (same trick as task.py:41-47)."""
    df = df.copy()
    df["patient_id"] = df["case_admission_id"].str.split("_").str[0]

    per_patient_outcome = df.groupby("patient_id")[TARGET_COL].max().reset_index()
    pids = per_patient_outcome["patient_id"].tolist()
    outcomes = per_patient_outcome[TARGET_COL].tolist()

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


def summarize(name: str, df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_patients = df["case_admission_id"].str.split("_").str[0].nunique()
    label_balance = df[TARGET_COL].mean()
    print(
        f"{name}: rows={n_rows}, unique_patients={n_patients}, "
        f"label_balance(3M Death=1)={label_balance:.4f}"
    )


def main() -> None:
    if not SOURCE_XLSX.exists():
        raise FileNotFoundError(f"Geneva Excel not found: {SOURCE_XLSX}")

    df = pd.read_excel(SOURCE_XLSX)
    df["case_admission_id"] = build_case_admission_id(df["Case ID"])

    df = df[["case_admission_id", *FEATURE_COLS, TARGET_COL]].copy()
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0}).astype(float)

    n_pre = df["case_admission_id"].nunique()
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    n_post = df["case_admission_id"].nunique()
    print(
        f"Dropped {n_pre - n_post} cids with NaN target '{TARGET_COL}'; "
        f"{n_post} unique cases remain."
    )

    half_a, half_b = stratified_patient_split(df, seed=SEED)

    pids_a = set(half_a["case_admission_id"].str.split("_").str[0])
    pids_b = set(half_b["case_admission_id"].str.split("_").str[0])
    overlap = pids_a & pids_b
    if overlap:
        raise AssertionError(
            f"Patient sets are not disjoint: {len(overlap)} shared patient IDs"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_a = OUT_DIR / "geneva_half_A.parquet"
    out_b = OUT_DIR / "geneva_half_B.parquet"
    half_a.to_parquet(out_a, index=False)
    half_b.to_parquet(out_b, index=False)

    summarize("half_A", half_a)
    summarize("half_B", half_b)
    print(f"Wrote {out_a}")
    print(f"Wrote {out_b}")


if __name__ == "__main__":
    main()
