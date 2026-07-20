"""fed_stroke.dp: a controllable single-site synthetic generator (spec 1.1 §4.6).

A logistic-linear ground truth at real per-node scale (n≈1000, the GVA-half binding case; the
cohorts are disbalanced — GVA >2000 total split ~1000/node, Shenzhen ~40000) over the frozen
2-feature schema, so the DP prototype's three-arm sanity harness has data before DP touches real
Geneva. DP utility is scale-sensitive: at this n the mechanism shows a clean monotone ε→AUC
erosion, whereas at n≈380 the honest 2·D·T-release accounting drove ε≤5 to chance. Labels are
Bernoulli (not a hard threshold) so the Bayes-optimal AUC is < 1 and DP has real headroom to erode
as ε shrinks. Mirrors the `_two_feature_data` / `_make_half` idioms already in the tests.

`FEATURE_RANGES` has ONE home — `fed_stroke.dp.boost` (it is a DP-safety artifact, §3.7) — and this
module imports it, never redefines it. The generator draws from a SUBSET of those public ranges
(Age∼U[40,90] ⊂ [0,120]; NIHSS∼U[0,30] ⊂ [0,42]); that gap is INTENTIONAL — real values sit inside
the public clinical range — so do not "align" them.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fed_stroke.dp.boost import FEATURE_RANGES  # single home; do NOT redefine  # noqa: F401
from fed_stroke.schema import FEATURE_COLS, TARGET_COL


def make_synthetic_site(n=1000, coef=(0.05, 0.12), intercept=-3.0, noise=0.5, seed=0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Age ~ U[40,90], NIHSS ~ U[0,30]; logit = intercept + coef.Age·(age−65)
    + coef.NIHSS·(nih−15) + noise·N(0,1); y ~ Bernoulli(σ(logit)). n=1000 ≈ GVA per-node scale.

    Returns (X, y): X is (n, 2) as [Age, NIHSS] (FEATURE_COLS order); y is 0/1.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 90.0, size=n)
    nih = rng.uniform(0.0, 30.0, size=n)
    logit = (intercept
             + coef[0] * (age - 65.0)
             + coef[1] * (nih - 15.0)
             + noise * rng.standard_normal(n))
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    X = np.column_stack([age, nih])
    return X, y


def make_synthetic_frame(n=1000, coef=(0.05, 0.12), intercept=-3.0, noise=0.5,
                         prefix="S", seed=0) -> pd.DataFrame:
    """Same signal as `make_synthetic_site`, framed with FEATURE_COLS + TARGET_COL + a unique
    `case_admission_id` (f'{prefix}{i}_1'), so it round-trips through split_half /
    score_booster_on_half if the demo's --via-parquet path is used."""
    X, y = make_synthetic_site(n=n, coef=coef, intercept=intercept, noise=noise, seed=seed)
    return pd.DataFrame({
        "case_admission_id": [f"{prefix}{i}_1" for i in range(n)],
        FEATURE_COLS[0]: X[:, 0],
        FEATURE_COLS[1]: X[:, 1],
        TARGET_COL: y,
    })


def synthetic_train_valid(X, y, test_size=0.2, seed=42, stratify=True):
    """Stratified split: BOTH classes guaranteed in train AND valid (no NaN AUC, §8.3).

    Returns (X_train, X_valid, y_train, y_valid).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if stratify else None,
    )
