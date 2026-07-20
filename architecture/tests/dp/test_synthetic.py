"""fed_stroke.dp: synthetic single-site generator tests (spec 1.1 §4.6/§4.7).

Plain-pytest idiom of tests/test_metrics.py. Pins determinism, label balance in range, both
classes present in train AND valid (so compute_binary_metrics never returns NaN AUC), and the
frame's round-trippable schema.
"""
import numpy as np

from fed_stroke.dp.boost import FEATURE_RANGES
from fed_stroke.dp.synthetic import (
    make_synthetic_frame,
    make_synthetic_site,
    synthetic_train_valid,
)
from fed_stroke.schema import FEATURE_COLS, TARGET_COL


def test_generator_determinism():
    X1, y1 = make_synthetic_site(n=500, seed=7)
    X2, y2 = make_synthetic_site(n=500, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_label_balance_and_feature_ranges():
    X, y = make_synthetic_site(n=2000, seed=0)
    frac = y.mean()
    assert 0.0 < frac < 1.0                         # not single-class
    assert 0.02 < frac < 0.5                         # a plausible rare-outcome balance
    # drawn from a SUBSET of the public ranges (intentional gap; do not "align")
    age_lo, age_hi = FEATURE_RANGES[FEATURE_COLS[0]]
    nih_lo, nih_hi = FEATURE_RANGES[FEATURE_COLS[1]]
    assert age_lo <= X[:, 0].min() and X[:, 0].max() <= age_hi
    assert nih_lo <= X[:, 1].min() and X[:, 1].max() <= nih_hi


def test_both_classes_in_train_and_valid():
    X, y = make_synthetic_site(n=1000, seed=3)
    X_tr, X_va, y_tr, y_va = synthetic_train_valid(X, y, seed=42)
    for split in (y_tr, y_va):
        assert set(np.unique(split)) == {0, 1}


def test_frame_schema_roundtrips():
    df = make_synthetic_frame(n=300, prefix="A", seed=1)
    assert list(df.columns) == ["case_admission_id", FEATURE_COLS[0], FEATURE_COLS[1], TARGET_COL]
    # unique case_admission_id of the f"{prefix}{i}_1" form (patient_id = split('_')[0])
    assert df["case_admission_id"].is_unique
    assert df["case_admission_id"].iloc[0] == "A0_1"
    assert df[TARGET_COL].isin([0, 1]).all()
