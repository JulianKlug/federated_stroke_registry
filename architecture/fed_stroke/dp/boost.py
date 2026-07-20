"""fed_stroke.dp: the DP-GBDT mechanism, learner, and plug-point seam (spec 1.1 §4.2).

A parallel, minimal-but-real histogram gradient-boosting learner in NumPy that we fully own.
Stock XGBoost's `hist` builder computes the gradient/hessian histograms in C++ and never
exposes them to Python (spec §3.1), so "add noise to the histograms" cannot be done by
configuring `xgboost`; in DP mode the future federated client **swaps the whole learner** for
this one (§4.5). Noise-off (ε=∞) it is an ordinary shallow GBDT.

The seam a downstream `client_app` DP branch imports is `DPConfig`, the `HistogramNoiseMechanism`
protocol, and `train_dp_gbdt` (re-exported from `fed_stroke.dp`).

**Two accounting facts pinned here that the Opacus gate cannot see (spec §3.2/§3.3, risk #10):**
- Composition is per TREE LEVEL, not per node: siblings partition the records (parallel
  composition), so `num_histogram_queries = max_depth × num_boost_round` (NOT 2^D−1 per tree).
- Each level noises the gradient AND the hessian histogram as TWO independent Gaussian
  mechanisms, so `num_gaussian_releases = 2 × num_histogram_queries`. Feeding the accountant
  `D·T` instead of `2·D·T` under-reports ε ~2×.

Geometric leaf clipping (`|w| ≤ clip_bound`) is POST-PROCESSING of the already-noised histogram
(§3.8) — a utility/stability guard, NOT a privacy knob; it costs ZERO budget and ε is invariant
to `clip_bound`.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fed_stroke.dp.accounting import (
    account_run,
    account_run_laplace,
    noise_multiplier_for_epsilon,
)

# Sensitivity constants (§3.2). L2 drives the Gaussian arm, L1 the Laplace arm.
# For binary:logistic, per-example g = p − y ∈ [−1, 1] and h = p(1−p) ∈ (0, 0.25], so
# adding/removing one record moves exactly ONE bin of ONE feature -> these ARE the sensitivities
# (no per-example gradient clipping needed, unlike DP-SGD).
G_L2_PER_FEATURE = 1.0      # |p - y| <= 1              (Gaussian, L2)
H_L2_PER_FEATURE = 0.25     # p(1-p) <= 0.25            (Gaussian, L2)
G_L1_PER_FEATURE = 1.0      # per-feature L1 == per-feature magnitude (Laplace)
H_L1_PER_FEATURE = 0.25     # per-feature L1                          (Laplace)
DENOM_FLOOR = 1e-3          # min positive (H+λ) after noise, so gain never divides by <=0 (§3.3/§8.3)

# Public, data-INDEPENDENT clinical ranges -> DP-safe fixed bin edges (§3.7). ONE home; imported
# (never redefined) by fed_stroke.dp.synthetic. Keyed to the frozen FEATURE_COLS.
FEATURE_RANGES = {"Age (calc.)": (0.0, 120.0), "NIH on admission": (0.0, 42.0)}


# ------------------------------------------------------------------------ config

@dataclass(frozen=True)
class DPConfig:
    enabled: bool = False
    mechanism: str = "gaussian"        # "gaussian" (RDP, primary) | "laplace" (secondary)
    target_epsilon: float = 5.0        # arch §6.3 starting budget
    delta: float = 1e-5
    clip_bound: float = 1.0            # geometric leaf clip: |w| <= clip_bound (post-processing)
    max_bins: int = 32
    bin_strategy: str = "fixed_range"  # "fixed_range" (DP-safe) | "quantile" (non-DP arms only)
    noise_multiplier: float | None = None   # if set, ε is REPORTED not targeted

    @classmethod
    def from_run_config(cls, cfg: dict) -> "DPConfig":
        """Map cfg["dp"] (already through unflatten_dict + task.replace_keys) to a DPConfig.

        Keys arrive with `-`→`_` already applied to KEYS by replace_keys; enum-like VALUES
        (e.g. bin-strategy = "fixed-range") keep their dash, so normalize the string values
        `-`→`_` here so `"fixed-range"` maps to the `"fixed_range"` field value.
        """
        dp = cfg.get("dp", {})

        def norm(v):
            return v.replace("-", "_") if isinstance(v, str) else v

        return cls(
            enabled=dp.get("enabled", cls.enabled),
            mechanism=norm(dp.get("mechanism", cls.mechanism)),
            target_epsilon=float(dp.get("target_epsilon", cls.target_epsilon)),
            delta=float(dp.get("delta", cls.delta)),
            clip_bound=float(dp.get("clip_bound", cls.clip_bound)),
            max_bins=int(dp.get("max_bins", cls.max_bins)),
            bin_strategy=norm(dp.get("bin_strategy", cls.bin_strategy)),
            noise_multiplier=dp.get("noise_multiplier", cls.noise_multiplier),
        )


@dataclass(frozen=True)
class BoostParams:                     # XGBoost-aligned names so from_xgb_params is trivial
    max_depth: int = 3
    num_boost_round: int = 20
    eta: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: float = 5.0
    base_score: float = 0.5
    seed: int = 0

    @classmethod
    def from_xgb_params(cls, params: dict, num_boost_round: int | None = None) -> "BoostParams":
        """Build from an xgb params dict (post replace_keys). `reg_lambda` reads xgb's `lambda`
        or `reg_lambda`; `num_boost_round` is separate from params (derived from the tree budget
        upstream) and may be overridden by the caller."""
        return cls(
            max_depth=int(params.get("max_depth", cls.max_depth)),
            num_boost_round=int(num_boost_round if num_boost_round is not None
                                else params.get("num_boost_round", cls.num_boost_round)),
            eta=float(params.get("eta", cls.eta)),
            reg_lambda=float(params.get("reg_lambda", params.get("lambda", cls.reg_lambda))),
            min_child_weight=float(params.get("min_child_weight", cls.min_child_weight)),
            base_score=float(params.get("base_score", cls.base_score)),
            seed=int(params.get("seed", cls.seed)),
        )


def num_histogram_queries(boost: BoostParams) -> int:
    """Split-finding LEVELS = max_depth × num_boost_round (§3.3). NOT the node count (2^D−1)."""
    return boost.max_depth * boost.num_boost_round


def num_gaussian_releases(boost: BoostParams) -> int:
    """2 × levels — the gradient AND hessian histograms are TWO Gaussian releases per level
    (§3.2/§3.3). This is the count fed to the accountant; D·T would under-report ε ~2×."""
    return 2 * num_histogram_queries(boost)


# ------------------------------------------------------------------------ binning

def fixed_bin_edges(feature_ranges: dict, max_bins: int) -> list[np.ndarray]:
    """Data-INDEPENDENT bin edges over public ranges (§3.7). One edge array per feature, in the
    dict's insertion order (which must match the X column order = FEATURE_COLS)."""
    return [
        np.linspace(lo, hi, max_bins + 1)
        for (lo, hi) in feature_ranges.values()
    ]


def quantile_bin_edges(X: np.ndarray, max_bins: int) -> list[np.ndarray]:
    """Empirical-quantile edges — NON-DP arms only (leaks the feature distribution, §3.7)."""
    qs = np.linspace(0.0, 1.0, max_bins + 1)
    return [np.quantile(X[:, j], qs) for j in range(X.shape[1])]


def _binize(X: np.ndarray, edges: list[np.ndarray], max_bins: int) -> np.ndarray:
    """Map each column to an integer bin index in [0, max_bins-1] using its edges."""
    out = np.empty(X.shape, dtype=np.int64)
    for j, e in enumerate(edges):
        # interior edges e[1:-1]; searchsorted -> bin index, clipped into range.
        idx = np.searchsorted(e[1:-1], X[:, j], side="right")
        out[:, j] = np.clip(idx, 0, max_bins - 1)
    return out


# ------------------------------------------------------------------------ mechanisms

class HistogramNoiseMechanism(Protocol):
    def add_noise(self, G: np.ndarray, H: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ...


class _IdentityMechanism:
    """Noise OFF (ε = ∞): an ordinary GBDT. Post-processing invariants still hold."""
    noise_multiplier = float("inf")
    reported_epsilon = float("inf")
    num_releases = 0

    def add_noise(self, G, H, rng):
        return G, H


class _GaussianMechanism:
    """Per-stat Gaussian noise (Decision 7, §9): G and H are TWO releases per level, each at
    multiplier σ, with DIFFERENT per-stat scales (H lives on a 4× smaller scale, so a single
    isotropic noise would swamp it). Per-entry std = σ·√d·stat_L2."""
    def __init__(self, sigma, std_g, std_h, reported_epsilon, num_releases):
        self.noise_multiplier = sigma
        self.std_g = std_g
        self.std_h = std_h
        self.reported_epsilon = reported_epsilon
        self.num_releases = num_releases

    def add_noise(self, G, H, rng):
        return (G + rng.normal(0.0, self.std_g, size=G.shape),
                H + rng.normal(0.0, self.std_h, size=H.shape))


class _LaplaceMechanism:
    """Secondary cruder arm: L1-calibrated Laplace, basic composition (§3.2, §8.6). Per-feature
    scale uses `d` (L1), NOT `√d` (L2). Noises G and H (⇒ 2·D·T releases; ε split evenly)."""
    def __init__(self, b_g, b_h, reported_epsilon, num_releases):
        self.b_g = b_g
        self.b_h = b_h
        self.noise_multiplier = float("nan")   # Laplace has no Gaussian multiplier
        self.reported_epsilon = reported_epsilon
        self.num_releases = num_releases

    def add_noise(self, G, H, rng):
        return (G + rng.laplace(0.0, self.b_g, size=G.shape),
                H + rng.laplace(0.0, self.b_h, size=H.shape))


def make_mechanism(dp: DPConfig, boost: BoostParams, num_features: int) -> HistogramNoiseMechanism:
    """Factory. dp.enabled=False -> identity (no-op, ε=∞).

    Gaussian (primary): n_rel = num_gaussian_releases(boost) = 2·D·T. σ is calibrated to
    dp.target_epsilon at q=1.0 (or pinned via dp.noise_multiplier, in which case ε is REPORTED).
    Per-feature std = σ·√d·G_L2 (G) and σ·√d·H_L2 (H). Feeding n_rel (not D·T) is load-bearing
    (§3.2) and is NOT covered by the Opacus gate.

    Laplace (secondary, L1): split dp.target_epsilon evenly across n_rel = 2·D·T releases;
    per-feature scales b_G = n_rel·d·G_L1/ε, b_H = n_rel·d·H_L1/ε (uses `d`, NOT `√d`).
    """
    if not dp.enabled:
        return _IdentityMechanism()

    d = num_features
    if dp.mechanism == "gaussian":
        n_rel = num_gaussian_releases(boost)
        if dp.noise_multiplier is not None:
            sigma = float(dp.noise_multiplier)
        else:
            sigma = noise_multiplier_for_epsilon(dp.target_epsilon, n_rel, 1.0, dp.delta)
        std_g = sigma * np.sqrt(d) * G_L2_PER_FEATURE
        std_h = sigma * np.sqrt(d) * H_L2_PER_FEATURE
        reported = account_run(n_rel, sigma, 1.0, dp.delta)
        return _GaussianMechanism(sigma, std_g, std_h, reported, n_rel)

    if dp.mechanism == "laplace":
        n_rel = num_gaussian_releases(boost)          # G and H both noised -> 2·D·T releases
        eps_per = dp.target_epsilon / n_rel
        b_g = d * G_L1_PER_FEATURE / eps_per          # = n_rel·d·G_L1 / target_epsilon
        b_h = d * H_L1_PER_FEATURE / eps_per
        k = num_histogram_queries(boost)              # per-group release count (G group, H group)
        reported = (account_run_laplace(k, d * G_L1_PER_FEATURE, b_g)
                    + account_run_laplace(k, d * H_L1_PER_FEATURE, b_h))
        return _LaplaceMechanism(b_g, b_h, reported, n_rel)

    raise ValueError(f"unknown dp.mechanism={dp.mechanism!r} (expected 'gaussian' | 'laplace')")


# ------------------------------------------------------------------------ learner

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _leaf_weight(g_total, h_total, eta, reg_lambda, clip_bound):
    """w = -η·G / D(H), then geometric clip |w| <= clip_bound (post-processing; ZERO ε, §3.8)."""
    denom = max(h_total + reg_lambda, DENOM_FLOOR)
    w = -eta * g_total / denom
    return float(np.clip(w, -clip_bound, clip_bound))


def _best_split(Gh, Hh, reg_lambda, min_child_weight):
    """Best (feature, bin) by noised gain over already-noised per-feature histograms Gh, Hh
    (shape (d, B)). Returns (gain, feature, bin, GL, HL, GR, HR) or None if no valid candidate.

    gain = ½[G_L²/D(H_L) + G_R²/D(H_R) − G_tot²/D(H_tot)],  D(h) = max(h+λ, DENOM_FLOOR),
    so a noise-driven negative hessian can never flip the gain sign or divide by <=0 (§3.3/§8.3).
    Candidates whose NOISED child hessian < min_child_weight are rejected.
    """
    d, B = Gh.shape
    best = None
    for f in range(d):
        cumG = np.cumsum(Gh[f])
        cumH = np.cumsum(Hh[f])
        totG, totH = cumG[-1], cumH[-1]
        GL, HL = cumG[:-1], cumH[:-1]          # left = bins [0..b], b in 0..B-2
        GR, HR = totG - GL, totH - HL
        DL = np.maximum(HL + reg_lambda, DENOM_FLOOR)
        DR = np.maximum(HR + reg_lambda, DENOM_FLOOR)
        Dtot = max(totH + reg_lambda, DENOM_FLOOR)
        gain = 0.5 * (GL ** 2 / DL + GR ** 2 / DR - totG ** 2 / Dtot)
        valid = (HL >= min_child_weight) & (HR >= min_child_weight)
        gain = np.where(valid, gain, -np.inf)
        b = int(np.argmax(gain))
        if not np.isfinite(gain[b]):
            continue
        if best is None or gain[b] > best[0]:
            best = (float(gain[b]), f, b, float(GL[b]), float(HL[b]), float(GR[b]), float(HR[b]))
    return best


class DPBooster:
    """A trained DP-GBDT ensemble. Serialization + a DP-aware server aggregator are the
    downstream FL-integration item (§4.5); this prototype only trains + predicts."""
    def __init__(self, trees, base_margin, edges, max_bins):
        self.trees = trees                # list of nested-dict trees
        self.base_margin = base_margin
        self.edges = edges
        self.max_bins = max_bins

    def _tree_predict(self, tree, binned):
        out = np.zeros(binned.shape[0])

        def recurse(node, mask):
            if "leaf" in node:
                out[mask] = node["leaf"]
                return
            col = binned[:, node["feature"]]
            recurse(node["left"], mask & (col <= node["bin"]))
            recurse(node["right"], mask & (col > node["bin"]))

        recurse(tree, np.ones(binned.shape[0], dtype=bool))
        return out

    def predict_margin(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        binned = _binize(X, self.edges, self.max_bins)
        margin = np.full(X.shape[0], self.base_margin, dtype=float)
        for tree in self.trees:
            margin += self._tree_predict(tree, binned)
        return margin

    def predict(self, X) -> np.ndarray:
        return _sigmoid(self.predict_margin(X))


def train_dp_gbdt(X, y, boost: BoostParams, dp: DPConfig,
                  feature_ranges: dict | None = None,
                  mechanism: HistogramNoiseMechanism | None = None,
                  rng: np.random.Generator | None = None) -> DPBooster:
    """Minimal histogram GBDT with the DP mechanism as the ONLY place noise enters.

    Per node (depth < max_depth): bin rows -> accumulate (G,H) per feature -> mechanism.add_noise
    -> pick split by argmax noised gain -> reject children with noised H < min_child_weight.
    Leaf weight is summed from the already-noised parent histogram (never re-queried), then
    geometric-clipped (post-processing, ZERO ε; §3.8). base_margin = logit(base_score) is a
    constant (data-independent). DP mode requires fixed-range (data-independent) bins and forces
    q=1.0 accounting (§3.5/§3.7).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    if rng is None:
        rng = np.random.default_rng(boost.seed)
    if mechanism is None:
        mechanism = make_mechanism(dp, boost, d)

    # Bin edges: data-independent (fixed range) is mandatory under DP; quantile only off-DP.
    if dp.enabled and dp.bin_strategy != "fixed_range":
        raise ValueError(
            f"DP mode requires bin_strategy='fixed_range' (data-independent, DP-safe); "
            f"got {dp.bin_strategy!r} (§3.7)."
        )
    if dp.bin_strategy == "quantile" and not dp.enabled:
        edges = quantile_bin_edges(X, dp.max_bins)
    else:
        fr = feature_ranges if feature_ranges is not None else FEATURE_RANGES
        edges = fixed_bin_edges(fr, dp.max_bins)

    binned = _binize(X, edges, dp.max_bins)
    base_margin = _logit(boost.base_score)
    lam, mcw, eta, clip = boost.reg_lambda, boost.min_child_weight, boost.eta, dp.clip_bound
    max_depth, max_bins = boost.max_depth, dp.max_bins

    def build(rows, depth, inherited):
        # inherited = (G_total, H_total) from the parent's noised histogram, or None at the root.
        if depth >= max_depth or rows.shape[0] == 0:
            g_t, h_t = inherited if inherited is not None else (0.0, 0.0)
            return {"leaf": _leaf_weight(g_t, h_t, eta, lam, clip)}
        # Per-feature (G, H) histograms over this node's rows -> ONE level release (noised).
        Gh = np.empty((d, max_bins))
        Hh = np.empty((d, max_bins))
        for f in range(d):
            Gh[f] = np.bincount(binned[rows, f], weights=g[rows], minlength=max_bins)[:max_bins]
            Hh[f] = np.bincount(binned[rows, f], weights=h[rows], minlength=max_bins)[:max_bins]
        Gh, Hh = mechanism.add_noise(Gh, Hh, rng)

        own_total = (float(Gh[0].sum()), float(Hh[0].sum()))
        node_total = inherited if inherited is not None else own_total

        best = _best_split(Gh, Hh, lam, mcw)
        if best is None or best[0] <= 0.0:
            g_t, h_t = node_total
            return {"leaf": _leaf_weight(g_t, h_t, eta, lam, clip)}

        _, f, b, GL, HL, GR, HR = best
        col = binned[rows, f]
        left_rows = rows[col <= b]
        right_rows = rows[col > b]
        return {
            "feature": f, "bin": b,
            "left": build(left_rows, depth + 1, (GL, HL)),
            "right": build(right_rows, depth + 1, (GR, HR)),
        }

    all_rows = np.arange(n)
    margin = np.full(n, base_margin, dtype=float)
    trees = []
    for _ in range(boost.num_boost_round):
        p = _sigmoid(margin)
        g = p - y                 # gradient  ∈ [-1, 1]
        h = p * (1.0 - p)         # hessian   ∈ (0, 0.25]
        tree = build(all_rows, 0, None)
        trees.append(tree)
        booster_step = DPBooster([tree], 0.0, edges, max_bins)
        margin = margin + booster_step._tree_predict(tree, binned)

    booster = DPBooster(trees, base_margin, edges, max_bins)
    # Expose what the mechanism accounted, so the demo/tests can read σ, ε, and the release count.
    booster.mechanism = mechanism
    return booster
