"""fed_stroke.dp: a controllable single-site synthetic generator (spec 1.1 §4.6).

A logistic-linear ground truth at real per-node scale (n≈1000, the GVA-half binding case; the
cohorts are disbalanced — GVA >2000 total split ~1000/node, Shenzhen ~40000) over the frozen
41-feature schema. The label depends on TWO signal features only — `age` and `NIHSS`, exactly
the pre-freeze generator — while the other 39 frozen columns are label-independent BACKGROUND
drawn inside their public FEATURE_RANGES. That keeps the ground truth interpretable and gives
the DP prototype's three-arm harness the REAL d (and therefore the real σ·√d histogram noise)
before DP touches real Geneva. DP utility is scale-sensitive: at this n the mechanism shows a
clean monotone ε→AUC erosion, whereas at n≈380 the honest 2·D·T-release accounting drove ε≤5
to chance. Labels are Bernoulli (not a hard threshold) so the Bayes-optimal AUC is < 1 and DP
has real headroom to erode as ε shrinks.

`FEATURE_RANGES` has ONE home — `fed_stroke.dp.boost` (it is a DP-safety artifact, §3.7) — and
this module imports it, never redefines it. The generator draws from a SUBSET of those public
ranges (age∼U[40,90] ⊂ [0,120]; NIHSS∼U[0,30] ⊂ [0,42]; background in the central half of each
range, binaries ~ Bernoulli(0.3)); that gap is INTENTIONAL — real values sit inside the public
clinical range — so do not "align" them.

`assemble_site_matrix` / `assemble_site_frame` are the ONE place tests and fixtures widen a
hand-rolled (age, NIHSS, y) signal into a full frozen-schema X / DataFrame.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fed_stroke.dp.boost import FEATURE_RANGES  # single home; do NOT redefine  # noqa: F401
from fed_stroke.schema import FEATURE_COLS, ID_COL, TARGET_COL, is_binary_feature

SIGNAL_FEATURES = ("age", "NIHSS")
AGE_IDX = FEATURE_COLS.index("age")
NIHSS_IDX = FEATURE_COLS.index("NIHSS")
BACKGROUND_BINARY_RATE = 0.3
_BACKGROUND_STREAM = 0xB6   # SeedSequence spawn key: background noise never shares the signal's stream


def background_matrix(n: int, seed: int = 0) -> np.ndarray:
    """(n, d) label-independent background, one column per FEATURE_COLS entry, deterministic
    under `seed`: binaries ~ Bernoulli(BACKGROUND_BINARY_RATE), continuous ~ U over the
    central half of the feature's public range. Signal columns are drawn too (callers
    overwrite them), so column j is always FEATURE_COLS[j] and always inside FEATURE_RANGES."""
    rng = np.random.default_rng([int(seed), _BACKGROUND_STREAM])
    X = np.empty((n, len(FEATURE_COLS)), dtype=float)
    for j, name in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_RANGES[name]
        if is_binary_feature(name):
            X[:, j] = (rng.uniform(size=n) < BACKGROUND_BINARY_RATE).astype(float)
        else:
            width = hi - lo
            X[:, j] = rng.uniform(lo + 0.25 * width, lo + 0.75 * width, size=n)
    return X


def assemble_site_matrix(age, nih, seed: int = 0) -> np.ndarray:
    """Widen an (age, NIHSS) signal into a full frozen-schema X: background_matrix with the
    two signal columns overwritten. Column order == FEATURE_COLS == FEATURE_RANGES."""
    age = np.asarray(age, dtype=float)
    nih = np.asarray(nih, dtype=float)
    if age.shape != nih.shape or age.ndim != 1:
        raise ValueError(f"age and nih must be 1-D and aligned; got {age.shape} vs {nih.shape}")
    X = background_matrix(len(age), seed)
    X[:, AGE_IDX] = age
    X[:, NIHSS_IDX] = nih
    return X


def assemble_site_frame(case_ids, age, nih, y, seed: int = 0) -> pd.DataFrame:
    """assemble_site_matrix framed for the loader: [ID_COL, *FEATURE_COLS, TARGET_COL].
    The real node parquet also carries the other schema.OUTCOME_COLS; the loader
    only needs TARGET_COL, so fixtures stay minimal."""
    X = assemble_site_matrix(age, nih, seed)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, ID_COL, list(case_ids))
    df[TARGET_COL] = np.asarray(y)
    return df


def make_synthetic_site(n=1000, coef=(0.05, 0.12), intercept=-3.0, noise=0.5, seed=0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """age ~ U[40,90], NIHSS ~ U[0,30]; logit = intercept + coef.age·(age−65)
    + coef.NIHSS·(nih−15) + noise·N(0,1); y ~ Bernoulli(σ(logit)). n=1000 ≈ GVA per-node scale.

    Returns (X, y): X is (n, len(FEATURE_COLS)) in FEATURE_COLS order — age at AGE_IDX, NIHSS at
    NIHSS_IDX, the other columns label-independent background (background_matrix); y is 0/1.
    The signal draw (and hence y) is byte-identical to the pre-freeze 2-feature generator.
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
    return assemble_site_matrix(age, nih, seed), y


def make_synthetic_frame(n=1000, coef=(0.05, 0.12), intercept=-3.0, noise=0.5,
                         prefix="S", seed=0) -> pd.DataFrame:
    """Same signal as `make_synthetic_site`, framed as [ID_COL, *FEATURE_COLS, TARGET_COL]
    with a unique id (f'{prefix}{i}_1'), so it round-trips through the parquet loader
    (split_half / score_booster_on_half)."""
    X, y = make_synthetic_site(n=n, coef=coef, intercept=intercept, noise=noise, seed=seed)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, ID_COL, [f"{prefix}{i}_1" for i in range(n)])
    df[TARGET_COL] = y
    return df


def synthetic_train_valid(X, y, test_size=0.2, seed=42, stratify=True):
    """Stratified split: BOTH classes guaranteed in train AND valid (no NaN AUC, §8.3).

    Returns (X_train, X_valid, y_train, y_valid).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if stratify else None,
    )
