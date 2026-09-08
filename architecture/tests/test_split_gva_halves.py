"""split_gva_halves: patient-disjoint 50/50 partition of the frozen GVA table.

Synthetic frames only. Import mechanics: see test_preprocess_gva.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCH_DIR.parent
for _p in (REPO_ROOT, ARCH_DIR / "preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocessing.mappings import FROZEN_FEATURES, FROZEN_OUTCOMES  # noqa: E402
from preprocessing.mappings.frozen_schema import ID_COL  # noqa: E402
from split_gva_halves import SPLIT_SEED, split_patient_disjoint_halves  # noqa: E402

from fed_stroke.schema import TARGET_COL  # noqa: E402

N_PATIENTS = 200
MULTI_ADMISSION_EVERY = 10   # every 10th patient gets a second admission
NAN_OUTCOME_EVERY = 7        # every 7th patient has no recorded outcome


def _frozen_frame(seed: int = 0) -> pd.DataFrame:
    """Minimal valid frozen-schema table: pseudonymous ids, float features,
    outcomes with NaN gaps, some multi-admission patients."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(N_PATIENTS):
        pid = f"{p:016x}"
        n_adm = 2 if p % MULTI_ADMISSION_EVERY == 0 else 1
        outcome = np.nan if p % NAN_OUTCOME_EVERY == 0 else float(p % 2)
        for adm in range(1, n_adm + 1):
            rows.append((f"{pid}_{adm}", outcome))
    df = pd.DataFrame(rows, columns=[ID_COL, TARGET_COL])
    for col in FROZEN_FEATURES:
        df[col] = rng.uniform(0.0, 1.0, len(df))
    for col in FROZEN_OUTCOMES:
        if col != TARGET_COL:
            df[col] = 0.0
    return df[[ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES]]


def _pids(half: pd.DataFrame) -> set[str]:
    return set(half[ID_COL].str.split("_").str[0])


def test_halves_are_patient_disjoint_and_lossless():
    df = _frozen_frame()
    half_a, half_b = split_patient_disjoint_halves(df)

    assert not (_pids(half_a) & _pids(half_b))
    # no admission dropped — NaN-outcome rows included (label drop is loader-side)
    assert len(half_a) + len(half_b) == len(df)
    assert sorted([*half_a[ID_COL], *half_b[ID_COL]]) == sorted(df[ID_COL])
    assert half_a[TARGET_COL].isna().any() and half_b[TARGET_COL].isna().any()


def test_multi_admission_patient_stays_in_one_half():
    df = _frozen_frame()
    half_a, half_b = split_patient_disjoint_halves(df)
    multi = df[ID_COL].str.split("_").str[0].value_counts()
    for pid in multi[multi > 1].index:
        in_a = (half_a[ID_COL].str.split("_").str[0] == pid).any()
        in_b = (half_b[ID_COL].str.split("_").str[0] == pid).any()
        assert in_a != in_b


def test_deterministic_and_no_strat_column_leak():
    df = _frozen_frame()
    a1, b1 = split_patient_disjoint_halves(df, seed=SPLIT_SEED)
    a2, b2 = split_patient_disjoint_halves(df, seed=SPLIT_SEED)
    pd.testing.assert_frame_equal(a1.reset_index(drop=True), a2.reset_index(drop=True))
    pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))
    # column contract unchanged (write_node_parquet checks order)
    assert list(a1.columns) == list(df.columns)


def test_recorded_outcome_roughly_balanced():
    df = _frozen_frame()
    half_a, half_b = split_patient_disjoint_halves(df)
    rate_a = half_a[TARGET_COL].mean()
    rate_b = half_b[TARGET_COL].mean()
    assert abs(rate_a - rate_b) < 0.1
