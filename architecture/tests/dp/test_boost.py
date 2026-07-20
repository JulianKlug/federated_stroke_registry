"""fed_stroke.dp: mechanism + learner tests (spec 1.1 §4.7, §7.2/§7.3).

Plain-pytest idiom of tests/test_metrics.py. Pins the sensitivity constants, the level/release
COUNTS (the two-releases-per-level trap the Opacus gate cannot see, §3.2/§3.3), the L1 Laplace
scale, the post-processing invariance of clip_bound, the noised-hessian floor, and the learner's
behaviour (no-noise ≈ stock XGBoost, monotone ε erosion, determinism, bounded output).
"""
import numpy as np
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from flwr.common.config import unflatten_dict

from fed_stroke.dp import (
    BoostParams,
    DPConfig,
    make_mechanism,
    num_gaussian_releases,
    num_histogram_queries,
    train_dp_gbdt,
)
from fed_stroke.dp import accounting as A
from fed_stroke.dp import boost as B
from fed_stroke.dp.synthetic import make_synthetic_site, synthetic_train_valid
from fed_stroke.task import replace_keys

pytestmark = pytest.mark.filterwarnings("ignore:Optimal RDP order")

DELTA = 1e-5
DEMO_BOOST = BoostParams(max_depth=3, num_boost_round=20, eta=0.1,
                         min_child_weight=5.0, base_score=0.5, seed=0)


def _fit_auc(eps, seed, n=1000, boost=None):
    """Train one arm (eps=None -> noise off) and return valid AUC."""
    boost = boost or DEMO_BOOST
    X, y = make_synthetic_site(n=n, seed=seed)
    X_tr, X_va, y_tr, y_va = synthetic_train_valid(X, y, seed=42)
    dp = DPConfig(enabled=eps is not None, target_epsilon=(eps if eps is not None else 5.0))
    b = train_dp_gbdt(X_tr, y_tr, boost, dp, rng=np.random.default_rng(seed))
    return roc_auc_score(y_va, b.predict(X_va))


# ---- sensitivity (§3.2) --------------------------------------------------------

def test_gradient_hessian_bounds():
    """g = p − y ∈ [−1, 1]; h = p(1−p) ∈ (0, 0.25]."""
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-6, 1 - 1e-6, size=10_000)
    y = rng.integers(0, 2, size=10_000)
    g = p - y
    h = p * (1 - p)
    assert g.min() >= -1.0 and g.max() <= 1.0
    assert h.min() > 0.0 and h.max() <= 0.25 + 1e-12


def test_histogram_sensitivity_bound():
    """Adding one record changes exactly ONE bin per feature; |ΔG| ≤ 1, |ΔH| ≤ 0.25 (§3.2)."""
    max_bins = 16
    edges = B.fixed_bin_edges(B.FEATURE_RANGES, max_bins)
    rng = np.random.default_rng(1)
    n, d = 200, 2
    X = np.column_stack([rng.uniform(40, 90, n), rng.uniform(0, 30, n)])
    g = rng.uniform(-1, 1, n)
    h = rng.uniform(0, 0.25, n)

    def hist(Xa, ga, ha):
        binned = B._binize(Xa, edges, max_bins)
        G = np.stack([np.bincount(binned[:, f], weights=ga, minlength=max_bins)[:max_bins]
                      for f in range(d)])
        H = np.stack([np.bincount(binned[:, f], weights=ha, minlength=max_bins)[:max_bins]
                      for f in range(d)])
        return G, H

    G0, H0 = hist(X, g, h)
    # append one extra record with the maximal-magnitude contribution
    x_new = np.array([[65.0, 15.0]])
    g_new, h_new = np.array([1.0]), np.array([0.25])
    G1, H1 = hist(np.vstack([X, x_new]), np.append(g, g_new), np.append(h, h_new))

    for f in range(d):
        changed = np.flatnonzero(~np.isclose(G0[f], G1[f]))
        assert changed.size == 1                          # exactly one bin per feature
        assert abs((G1[f] - G0[f]).sum()) <= 1.0 + 1e-9   # |ΔG| ≤ G_L1_PER_FEATURE
        assert abs((H1[f] - H0[f]).sum()) <= 0.25 + 1e-9  # |ΔH| ≤ H_L1_PER_FEATURE


# ---- composition counts (§3.3, the Opacus gate cannot see these) ---------------

def test_num_queries_is_depth_times_rounds():
    assert num_histogram_queries(DEMO_BOOST) == 3 * 20
    assert num_histogram_queries(BoostParams(max_depth=4, num_boost_round=20)) == 80


def test_gaussian_releases_is_two_times_levels():
    """G AND H are two Gaussian releases per level -> 2·D·T (pins the ~2× under-report, §3.2)."""
    assert num_gaussian_releases(DEMO_BOOST) == 2 * num_histogram_queries(DEMO_BOOST)
    assert num_gaussian_releases(BoostParams(max_depth=4, num_boost_round=20)) == 160


def test_gaussian_epsilon_uses_release_count():
    """The mechanism charges ε on 2·D·T (G+H), not one release per level."""
    boost = DEMO_BOOST
    sigma = 4.0
    dp = DPConfig(enabled=True, mechanism="gaussian", noise_multiplier=sigma, delta=DELTA)
    mech = make_mechanism(dp, boost, num_features=2)
    n_rel = num_gaussian_releases(boost)              # = 2·D·T = 120
    assert mech.num_releases == n_rel
    assert mech.reported_epsilon == pytest.approx(A.account_run(n_rel, sigma, 1.0, DELTA))
    # ... and that is materially larger than charging only D·T (the bug this guards):
    assert mech.reported_epsilon > A.account_run(n_rel // 2, sigma, 1.0, DELTA)


def test_laplace_scale_uses_l1_not_l2():
    """Laplace per-feature scale grows with d (L1), NOT √d (L2): d=2→d=4 ratio is 2, not √2."""
    boost = DEMO_BOOST
    dp = DPConfig(enabled=True, mechanism="laplace", target_epsilon=5.0)
    m2 = make_mechanism(dp, boost, num_features=2)
    m4 = make_mechanism(dp, boost, num_features=4)
    ratio = m4.b_g / m2.b_g
    assert ratio == pytest.approx(2.0)                 # d-linear (L1)
    assert ratio != pytest.approx(np.sqrt(2))          # NOT √d (L2)


def test_epsilon_independent_of_clip_bound():
    """clip_bound is post-processing -> reported ε invariant to it (§3.8)."""
    boost = DEMO_BOOST
    eps = []
    for c in (0.1, 1.0, 10.0):
        dp = DPConfig(enabled=True, mechanism="gaussian", noise_multiplier=3.0,
                      clip_bound=c, delta=DELTA)
        eps.append(make_mechanism(dp, boost, num_features=2).reported_epsilon)
    assert eps[0] == pytest.approx(eps[1]) == pytest.approx(eps[2])


# ---- learner behaviour ---------------------------------------------------------

def test_negative_noised_hessian_is_floored():
    """A noised child hessian H+λ < 0 must not divide by ≤0 nor produce NaN/inf; training with
    huge pinned noise still completes with finite, bounded predictions (§3.3/§8.3 DENOM_FLOOR)."""
    # crafted histogram with all-negative hessian
    Gh = np.array([[1.0, -2.0, 3.0, -1.0]])
    Hh = np.array([[-5.0, -4.0, -6.0, -3.0]])
    best = B._best_split(Gh, Hh, reg_lambda=1.0, min_child_weight=-1e9)
    assert best is None or np.isfinite(best[0])

    X, y = make_synthetic_site(n=400, seed=0)
    dp = DPConfig(enabled=True, mechanism="gaussian", noise_multiplier=50.0)  # swamping noise
    b = train_dp_gbdt(X, y, DEMO_BOOST, dp, rng=np.random.default_rng(0))
    p = b.predict(X)
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))


def test_noise_multiplier_monotonic_in_epsilon():
    """σ chosen by make_mechanism is strictly decreasing in target ε."""
    boost = DEMO_BOOST
    sig = []
    for e in (1, 3, 5, 10):
        dp = DPConfig(enabled=True, mechanism="gaussian", target_epsilon=e, delta=DELTA)
        sig.append(make_mechanism(dp, boost, num_features=2).noise_multiplier)
    assert all(sig[i] > sig[i + 1] for i in range(len(sig) - 1))


def test_no_noise_equals_reasonable_baseline():
    """DP-off learner tracks stock xgb.train (A→B = learner cost, not init/threading/subsample).
    The anchor pins base_score=0.5, nthread=1, subsample=1.0 (F5)."""
    n, seed = 1200, 0
    X, y = make_synthetic_site(n=n, seed=seed)
    X_tr, X_va, y_tr, y_va = synthetic_train_valid(X, y, seed=42)

    bst = xgb.train(
        {"objective": "binary:logistic", "eta": 0.1, "max_depth": 3, "min_child_weight": 5,
         "tree_method": "hist", "subsample": 1.0, "base_score": 0.5, "nthread": 1, "seed": seed},
        xgb.DMatrix(X_tr, label=y_tr), num_boost_round=20,
    )
    auc_a = roc_auc_score(y_va, bst.predict(xgb.DMatrix(X_va, label=y_va)))
    b = train_dp_gbdt(X_tr, y_tr, DEMO_BOOST, DPConfig(enabled=False))
    auc_b = roc_auc_score(y_va, b.predict(X_va))

    assert auc_a >= 0.72 and auc_b >= 0.72            # both materially better than chance
    assert abs(auc_a - auc_b) <= 0.08                 # our learner tracks stock XGBoost


def test_dp_auc_monotonicity():
    """Averaged over seeds (single-seed DP AUC is high-variance): no-noise ≳ high-ε ≫ low-ε.
    ε spans the noise-floor transition — at n≈1000 ε=1 is ~chance while ε=100 ≈ no-noise (§7.2).
    Asserting the ε=5≥ε=1 ordering directly would live below the noise floor, so this pins the
    robust ends instead."""
    seeds = range(8)
    a_off = np.mean([_fit_auc(None, s) for s in seeds])
    a_lo = np.mean([_fit_auc(1, s) for s in seeds])
    a_hi = np.mean([_fit_auc(100, s) for s in seeds])
    assert a_hi > a_lo + 0.08          # erosion is real across the transition
    assert a_off > a_lo + 0.08         # privacy at tight ε has a real utility cost
    assert a_off >= a_hi - 0.05        # no-noise is at least as good as loose-ε (within variance)


def test_determinism_pinned_seed():
    X, y = make_synthetic_site(n=600, seed=0)
    dp = DPConfig(enabled=True, mechanism="gaussian", target_epsilon=5.0)
    b1 = train_dp_gbdt(X, y, DEMO_BOOST, dp, rng=np.random.default_rng(0))
    b2 = train_dp_gbdt(X, y, DEMO_BOOST, dp, rng=np.random.default_rng(0))
    assert b1.trees == b2.trees                       # bitwise-identical trees
    assert np.array_equal(b1.predict(X), b2.predict(X))


def test_fixed_bin_edges_data_independent():
    """Fixed-range edges depend only on feature_ranges + max_bins, never on X (§3.7)."""
    e1 = B.fixed_bin_edges(B.FEATURE_RANGES, 32)
    e2 = B.fixed_bin_edges(B.FEATURE_RANGES, 32)
    for a, b in zip(e1, e2):
        assert np.array_equal(a, b)
    assert np.array_equal(e1[0], np.linspace(0.0, 120.0, 33))
    assert np.array_equal(e1[1], np.linspace(0.0, 42.0, 33))


def test_dp_mode_rejects_quantile_bins():
    """DP mode must refuse data-dependent (quantile) edges (§3.7)."""
    X, y = make_synthetic_site(n=200, seed=0)
    dp = DPConfig(enabled=True, bin_strategy="quantile")
    with pytest.raises(ValueError, match="fixed_range"):
        train_dp_gbdt(X, y, DEMO_BOOST, dp)


def test_leaf_clip_bounds_weight():
    """A tiny clip_bound forces every leaf weight into [−c, c]."""
    X, y = make_synthetic_site(n=600, seed=0)
    c = 0.05
    dp = DPConfig(enabled=True, mechanism="gaussian", target_epsilon=5.0, clip_bound=c)
    b = train_dp_gbdt(X, y, DEMO_BOOST, dp, rng=np.random.default_rng(0))

    def leaves(node):
        if "leaf" in node:
            return [node["leaf"]]
        return leaves(node["left"]) + leaves(node["right"])

    weights = [w for tree in b.trees for w in leaves(tree)]
    assert all(abs(w) <= c + 1e-12 for w in weights)


def test_predict_range():
    X, y = make_synthetic_site(n=400, seed=0)
    b = train_dp_gbdt(X, y, DEMO_BOOST, DPConfig(enabled=False))
    p = b.predict(X)
    assert np.all((p > 0) & (p < 1))


def test_dpconfig_from_run_config():
    """A pyproject-style flat dict round-trips through unflatten_dict + replace_keys +
    from_run_config to the expected dataclass (mirrors 1.d's idempotency pin, §4.3)."""
    flat = {
        "dp.enabled": True,
        "dp.mechanism": "gaussian",
        "dp.target-epsilon": 3.0,
        "dp.delta": 1e-5,
        "dp.clip-bound": 0.5,
        "dp.max-bins": 16,
        "dp.bin-strategy": "fixed-range",
    }
    cfg = replace_keys(unflatten_dict(flat))
    dp = DPConfig.from_run_config(cfg)
    assert dp == DPConfig(enabled=True, mechanism="gaussian", target_epsilon=3.0, delta=1e-5,
                          clip_bound=0.5, max_bins=16, bin_strategy="fixed_range")
