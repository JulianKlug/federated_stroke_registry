"""The training label is chosen at LOAD time (fed_stroke.schema TARGET_COL / TARGET_RULE ->
task.select_labelled_rows), never in a site's preprocessing.

The node parquet carries every cohort admission with all OUTCOME_COLS (NaN when not recorded);
these tests pin that the loader derives the label once, drops the unlabelled rows, counts
them, and refuses anything that is not {0, 1}.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fed_stroke import schema, task
from fed_stroke.dp.synthetic import assemble_site_frame
from fed_stroke.schema import FEATURE_COLS, ID_COL, OUTCOME_COLS, TARGET_COL, mrs_at_most


def _frame(labels, seed=0):
    n = len(labels)
    rng = np.random.default_rng(seed)
    return assemble_site_frame([f"p{i}_1" for i in range(n)], rng.uniform(40, 90, n),
                               rng.uniform(0, 30, n), labels, seed=seed)


def test_target_is_a_frozen_outcome_and_label_id():
    assert TARGET_COL in OUTCOME_COLS
    assert schema.label_id() == (TARGET_COL if schema.TARGET_RULE is None
                                 else f"{TARGET_COL}|{schema.TARGET_RULE.__name__}")


def test_derive_label_identity_and_validation():
    s = pd.Series([0.0, 1.0, np.nan, 1.0], name="death_3m")
    out = task.derive_label(s)
    assert out.tolist()[:2] == [0.0, 1.0] and np.isnan(out.iloc[2]) and out.dtype == "float64"
    with pytest.raises(ValueError, match=r"\{0, 1\}.*TARGET_RULE"):
        task.derive_label(pd.Series([0.0, 2.0], name="death_3m"))


def test_derive_label_with_mrs_rule_keeps_nan():
    mrs = pd.Series([0.0, 2.0, 3.0, 6.0, np.nan], name="mrs_3m")
    out = task.derive_label(mrs, mrs_at_most(2))
    assert out.tolist()[:4] == [1.0, 1.0, 0.0, 0.0] and np.isnan(out.iloc[4])
    assert mrs_at_most(2).__name__ == "mrs_at_most_2"


def test_select_labelled_rows_drops_unlabelled_and_casts_int():
    df = _frame([1, 0, np.nan, 1, np.nan])
    out, n_dropped = task.select_labelled_rows(df, TARGET_COL)
    assert n_dropped == 2 and len(out) == 3
    assert out[TARGET_COL].dtype == "int64" and out[TARGET_COL].tolist() == [1, 0, 1]
    assert out[ID_COL].tolist() == ["p0_1", "p1_1", "p3_1"]
    assert len(df) == 5                                     # input untouched
    with pytest.raises(ValueError, match="absent"):
        task.select_labelled_rows(df.drop(columns=[TARGET_COL]), TARGET_COL)


def test_select_labelled_rows_applies_the_schema_rule_to_the_target_only(monkeypatch):
    monkeypatch.setattr(task, "TARGET_COL", "mrs_3m")
    monkeypatch.setattr(task, "TARGET_RULE", mrs_at_most(2))
    df = pd.DataFrame({ID_COL: ["a_1", "b_1", "c_1"], "mrs_3m": [0.0, 3.0, np.nan],
                       "death_3m": [0.0, 1.0, 0.0]})
    out, n_dropped = task.select_labelled_rows(df, "mrs_3m")
    assert out["mrs_3m"].tolist() == [1, 0] and n_dropped == 1
    # another (binary) outcome column gets no rule
    out, n_dropped = task.select_labelled_rows(df, "death_3m")
    assert out["death_3m"].tolist() == [0, 1, 0] and n_dropped == 0


def test_resolve_run_split_drops_unlabelled_before_dedup_and_split():
    labels = np.array([1, 0] * 20, dtype=float)
    labels[[3, 8, 15, 22, 30, 37]] = np.nan
    df = _frame(labels, seed=1)
    tr, va = task.resolve_run_split(df, TARGET_COL, split_seed=1)
    both = pd.concat([tr, va])
    assert len(both) == 34 and both[TARGET_COL].dtype == "int64"
    assert both[TARGET_COL].isin([0, 1]).all()
    assert not set(both[ID_COL]) & {f"p{i}_1" for i in (3, 8, 15, 22, 30, 37)}


def test_loader_counts_rows_without_label(tmp_path, capsys):
    labels = np.array([1, 0] * 20, dtype=float)
    labels[:6] = np.nan
    df = _frame(labels, seed=2)
    df["mrs_3m"] = 2.0                                       # the other outcomes ride along untouched
    df["death_in_hospital"] = 0.0
    path = tmp_path / "geneva_half_A.parquet"
    df.to_parquet(path)
    ctx = SimpleNamespace(node_config={"data-path": str(path)}, run_config={})
    train_df, valid_df, n_train, n_val, pids = task._resolve_context_split(ctx)
    assert n_train + n_val == 34 and len(pids) == n_train
    assert list(train_df.columns) == [*FEATURE_COLS, TARGET_COL]      # outcomes are not features
    assert np.isfinite(train_df[FEATURE_COLS].to_numpy(dtype=float)).all()
    out = capsys.readouterr().out
    assert f"label={schema.label_id()}" in out and "rows_without_label=6" in out

    # a parquet without the chosen label column fails loudly and names what it carries
    df.drop(columns=[TARGET_COL]).to_parquet(path)
    with pytest.raises(ValueError, match=r"absent.*mrs_3m"):
        task._resolve_context_split(ctx)
