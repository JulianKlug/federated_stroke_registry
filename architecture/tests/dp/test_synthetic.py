"""fed_stroke.dp: synthetic single-site generator tests (spec 1.1 §4.6/§4.7).

Plain-pytest idiom of tests/test_metrics.py. Pins determinism, label balance in range, both
classes present in train AND valid (so compute_binary_metrics never returns NaN AUC), the
frame's round-trippable frozen schema, and — since the 41-feature freeze — that the (age,
NIHSS, y) signal is untouched while every background column sits inside its public range.
"""
import numpy as np
import pytest

from fed_stroke.dp.boost import FEATURE_RANGES
from fed_stroke.dp.synthetic import (
    AGE_IDX,
    NIHSS_IDX,
    assemble_site_frame,
    assemble_site_matrix,
    background_matrix,
    make_synthetic_frame,
    make_synthetic_site,
    synthetic_train_valid,
)
from fed_stroke.schema import FEATURE_COLS, ID_COL, TARGET_COL, is_binary_feature


def test_generator_determinism():
    X1, y1 = make_synthetic_site(n=500, seed=7)
    X2, y2 = make_synthetic_site(n=500, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_label_balance_and_feature_ranges():
    X, y = make_synthetic_site(n=2000, seed=0)
    assert X.shape == (2000, len(FEATURE_COLS))
    frac = y.mean()
    assert 0.0 < frac < 1.0                         # not single-class
    assert 0.02 < frac < 0.5                         # a plausible rare-outcome balance
    # signal columns drawn from a SUBSET of the public ranges (intentional gap; do not "align")
    age_lo, age_hi = FEATURE_RANGES["age"]
    nih_lo, nih_hi = FEATURE_RANGES["NIHSS"]
    assert age_lo <= X[:, AGE_IDX].min() and X[:, AGE_IDX].max() <= age_hi
    assert nih_lo <= X[:, NIHSS_IDX].min() and X[:, NIHSS_IDX].max() <= nih_hi
    # every column sits inside its public range; binaries are exactly {0, 1}
    for j, name in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_RANGES[name]
        assert lo <= X[:, j].min() and X[:, j].max() <= hi, name
        if is_binary_feature(name):
            assert set(np.unique(X[:, j])) <= {0.0, 1.0}, name


def test_signal_identical_to_pre_freeze_two_feature_generator():
    """Widening to the frozen schema must not move the (age, NIHSS) draw — same rng stream."""
    n, seed = 500, 3
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 90.0, size=n)
    nih = rng.uniform(0.0, 30.0, size=n)
    X, _ = make_synthetic_site(n=n, seed=seed)
    assert np.array_equal(X[:, AGE_IDX], age)
    assert np.array_equal(X[:, NIHSS_IDX], nih)


def test_background_is_seeded_and_label_free():
    b1 = background_matrix(100, seed=5)
    b2 = background_matrix(100, seed=5)
    b3 = background_matrix(100, seed=6)
    assert b1.shape == (100, len(FEATURE_COLS))
    assert np.array_equal(b1, b2) and not np.array_equal(b1, b3)


def test_assemble_site_matrix_and_frame():
    X = assemble_site_matrix([70.0, 80.0], [5.0, 20.0], seed=0)
    assert X.shape == (2, len(FEATURE_COLS))
    assert X[:, AGE_IDX].tolist() == [70.0, 80.0] and X[:, NIHSS_IDX].tolist() == [5.0, 20.0]
    df = assemble_site_frame(["a_1", "b_1"], [70.0, 80.0], [5.0, 20.0], [0, 1], seed=0)
    assert list(df.columns) == [ID_COL, *FEATURE_COLS, TARGET_COL]
    assert df["age"].tolist() == [70.0, 80.0] and df[TARGET_COL].tolist() == [0, 1]
    with pytest.raises(ValueError, match="aligned"):
        assemble_site_matrix([1.0], [1.0, 2.0])


def test_both_classes_in_train_and_valid():
    X, y = make_synthetic_site(n=1000, seed=3)
    X_tr, X_va, y_tr, y_va = synthetic_train_valid(X, y, seed=42)
    for split in (y_tr, y_va):
        assert set(np.unique(split)) == {0, 1}


def test_frame_schema_roundtrips():
    df = make_synthetic_frame(n=300, prefix="A", seed=1)
    assert list(df.columns) == [ID_COL, *FEATURE_COLS, TARGET_COL]
    # unique case_admission_id of the f"{prefix}{i}_1" form (patient_id = split('_')[0])
    assert df[ID_COL].is_unique
    assert df[ID_COL].iloc[0] == "A0_1"
    assert df[TARGET_COL].isin([0, 1]).all()
