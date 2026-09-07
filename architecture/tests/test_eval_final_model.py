"""[T1] Regression guard: the §4.4 refactor of eval_final_model._site_metrics onto
the shared baseline.score_booster_on_half is behavior-preserving.

A silent drift here would corrupt BOTH the offline eval table and the 1.d pooled
gate, since both now depend on this one scorer.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from fed_stroke.baseline import score_booster_on_half
from fed_stroke.dp.synthetic import assemble_site_frame
from fed_stroke.metrics import compute_binary_metrics
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.task import generate_splits


def _make_half(path, seed=1, n=80):
    rng = np.random.RandomState(seed)
    age = rng.uniform(40, 90, n)
    nih = rng.uniform(0, 30, n)
    logits = 0.05 * (age - 65) + 0.1 * (nih - 15) + rng.normal(0, 1, n)
    y = (logits > np.quantile(logits, 0.65)).astype(int)
    df = assemble_site_frame([f"H{i}_1" for i in range(n)], age, nih, y, seed=seed)
    df.to_parquet(path)
    return df


def _pre_refactor_site_metrics(bst, data_path, operating_point, n_boot, boot_seed):
    """Verbatim reproduction of the deleted eval_final_model._site_metrics body."""
    data_df = pd.read_parquet(data_path)
    _, valid_df, _, _ = generate_splits(
        data_df, outcome=TARGET_COL, test_size=0.2, seed=42
    )
    valid_dmatrix = xgb.DMatrix(valid_df[FEATURE_COLS], label=valid_df[TARGET_COL])
    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    return compute_binary_metrics(
        y_true, y_prob, operating_point, n_boot=n_boot, boot_seed=boot_seed
    )


def test_shared_scorer_matches_pre_refactor(tmp_path):
    half = tmp_path / "geneva_half_A.parquet"
    df = _make_half(half)
    dm = xgb.DMatrix(df[FEATURE_COLS], label=df[TARGET_COL])
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 3, "seed": 0},
                    dm, num_boost_round=10)

    new = score_booster_on_half(bst, half, operating_point=0.5, n_boot=200, boot_seed=0)
    old = _pre_refactor_site_metrics(bst, half, 0.5, 200, 0)

    assert new.keys() == old.keys()
    for k in old:
        assert new[k] == old[k], f"metric {k} drifted: {new[k]} != {old[k]}"
