"""Derived features (fed_stroke.schema.DERIVED_COLS) are computed at LOAD time, in the wheel
both sites run — never delivered by a site's preprocessing.

These tests pin the ratio itself and the ONE thing the placement buys: the derivation sees the
raw delivered values, not the missing sentinel generate_splits writes over them.
"""
import numpy as np
import pandas as pd
import pytest

from fed_stroke import task
from fed_stroke.dp.synthetic import assemble_site_frame
from fed_stroke.schema import DERIVED_COLS, MISSING_SENTINEL, TARGET_COL

NLR, NEUTROPHILS, LYMPHOCYTES = task.NLR, task.NEUTROPHILS, task.LYMPHOCYTES


def _frame(labels, seed=0):
    n = len(labels)
    rng = np.random.default_rng(seed)
    return assemble_site_frame([f"p{i}_1" for i in range(n)], rng.uniform(40, 90, n),
                               rng.uniform(0, 30, n), labels, seed=seed)


def test_nlr_is_the_count_ratio():
    df = pd.DataFrame({NEUTROPHILS: [6.0, 2.5, 9.0], LYMPHOCYTES: [2.0, 5.0, 0.9]})
    assert task.add_derived_features(df)[NLR].tolist() == [3.0, 0.5, 10.0]
    assert NLR not in df.columns                               # input untouched


def test_no_ratio_without_both_counts():
    """A zero lymphocyte count is an unknown ratio, not an infinite one; so is a missing count."""
    df = pd.DataFrame({NEUTROPHILS: [6.0, np.nan, 6.0, 6.0],
                       LYMPHOCYTES: [0.0, 2.0, np.nan, 3.0]})
    out = task.add_derived_features(df)[NLR]
    assert out.iloc[:3].isna().all() and out.iloc[3] == 2.0
    assert np.isfinite(out.dropna()).all()


def test_derivation_is_authoritative_over_a_delivered_column():
    """assemble_site_frame hands over a full FEATURE_COLS frame, nlr included — a site's value
    for a derived feature is never trusted."""
    df = _frame([1, 0, 1])
    df[NEUTROPHILS], df[LYMPHOCYTES], df[NLR] = 8.0, 2.0, -7.0
    assert task.add_derived_features(df)[NLR].tolist() == [4.0, 4.0, 4.0]


def test_missing_operand_column_fails_loudly():
    with pytest.raises(ValueError, match=f"{NEUTROPHILS}.*full frozen schema"):
        task.add_derived_features(_frame([1, 0]).drop(columns=[NEUTROPHILS]))


@pytest.mark.parametrize("holdout_frac", [0.0, 0.3])
def test_resolve_run_split_derives_before_the_sentinel(holdout_frac):
    """generate_splits runs twice in the hold-out modes and turns a missing count into
    MISSING_SENTINEL; deriving there would read -1/-1 as a ratio of 1.0."""
    df = _frame([1, 0] * 8)
    df[NEUTROPHILS], df[LYMPHOCYTES] = 4.0, 2.0
    df.loc[df.index[:4], [NEUTROPHILS, LYMPHOCYTES]] = np.nan

    train_df, valid_df = task.resolve_run_split(df, outcome=TARGET_COL,
                                                holdout_frac=holdout_frac)
    both = pd.concat([train_df, valid_df])
    assert set(DERIVED_COLS) <= set(both.columns)
    assert set(both[NLR]) == {2.0, MISSING_SENTINEL}
