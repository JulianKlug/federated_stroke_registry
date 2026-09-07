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
import json
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fed_stroke.dp.accounting import (
    account_run,
    account_run_laplace,
    noise_multiplier_for_epsilon,
)
from fed_stroke.schema import FEATURE_COLS, MISSING_SENTINEL, is_binary_feature

# Auto-detect marker written into every serialized DPBooster payload (§4.1/§4.8). The offline
# loader (eval_final_model.py) sniffs this to route a model file to from_json_bytes vs XGBoost.
# v2 (2026-07-23): fixed_bin_edges reserves bin 0 for the missing sentinel (REMOVE-IF-NO-DP),
# so v1 models' stored (feature_ranges, max_bins) would reconstruct DIFFERENT edges than they
# were trained with — the version bump makes stale v1 artifacts fail LOUDLY at load instead of
# silently predicting through shifted bins. All v1 artifacts predate the 1.1.a″ remediation
# (public-seeded noise, non-deduped cohort) and must not be scored anyway.
# v3 (2026-09-04): the frozen schema grew 2 → 41 features (fed_stroke.schema, FEATURE_RANGES
# below). A v2 artifact's stored 2-entry feature_ranges + integer tree feature indices are only
# interpretable against the OLD column order; scored on 41-column X it would silently read
# column 1 (sex) as NIHSS. The bump (plus the width check in _binize) makes v2 artifacts fail
# LOUDLY at load. No v2 artifact was ever trained on frozen-schema data.
DP_MODEL_FORMAT = "dp-gbdt-v3"

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
# (never redefined) by fed_stroke.dp.synthetic. Keyed to the frozen FEATURE_COLS — SAME keys in
# the SAME order (asserted below: column j of X == FEATURE_COLS[j] == the j-th edge array), and
# expressed in fed_stroke.schema.FEATURE_UNITS. These stay the REAL clinical ranges — the
# missing-sentinel bin is reserved structurally by fixed_bin_edges (REMOVE-IF-NO-DP), never by
# widening these values. Every lower bound must exceed MISSING_SENTINEL (-1.0); a value BELOW
# its lower bound bins into the sentinel bin, which is why out-of-range cleaning belongs in the
# site preprocessing, not here. Binaries are (0, 1).

FEATURE_RANGES = {
    "age": (0.0, 120.0),                        # years
    "sex": (0.0, 1.0),                          # binary (1 = female)
    "wake_up_stroke": (0.0, 1.0),               # binary
    "pre_stroke_mrs": (0.0, 5.0),               # mRS 0-5 (6 = dead is impossible pre-stroke)
    "temperature": (25.0, 45.0),                # °C
    "heart_rate": (0.0, 300.0),                 # bpm
    "respiratory_rate": (0.0, 100.0),           # /min
    "systolic_blood_pressure": (0.0, 300.0),    # mmHg
    "diastolic_blood_pressure": (0.0, 200.0),   # mmHg
    "NIHSS": (0.0, 42.0),                       # points
    "glucose": (0.0, 60.0),                     # mmol/L
    "GCS": (3.0, 15.0),                         # points
    "white_blood_cell_count": (0.0, 100.0),     # G/l
    "neutrophil_count": (0.0, 100.0),           # G/l
    "lymphocyte_count": (0.0, 50.0),            # G/l
    "CRP": (0.0, 600.0),                        # mg/l
    "INR": (0.0, 20.0),                         # ratio
    "fibrinogen": (0.0, 20.0),                  # g/l
    "d_dimer": (0.0, 100000.0),                  # ng/ml
    "hba1c": (0.0, 20.0),                       # %
    "alt": (0.0, 50000.0),                       # U/l
    "LDL": (0.0, 15.0),                         # mmol/l
    "creatinine": (0.0, 3000.0),                # µmol/l
    "urea": (0.0, 200.0),                       # mmol/l
    "med_hist_stroke": (0.0, 1.0),              # binary
    "med_hist_tia": (0.0, 1.0),                 # binary
    "med_hist_ich": (0.0, 1.0),                 # binary
    "med_hist_hta": (0.0, 1.0),                 # binary
    "med_hist_diabetes": (0.0, 1.0),            # binary
    "med_hist_hyperlipidemia": (0.0, 1.0),      # binary
    "med_hist_af": (0.0, 1.0),                  # binary
    "med_hist_coronary_heart_disease": (0.0, 1.0),     # binary
    "med_hist_valv_heart_disease": (0.0, 1.0),         # binary
    "med_hist_peripheral_artery_disease": (0.0, 1.0),  # binary
    "med_hist_smoking": (0.0, 1.0),             # binary
    "ODT": (0.0, 10080.0),                       # min — onset-to-door, capped at 7d
    "ONT": (0.0, 2880.0),                       # min — onset-to-needle, capped at 48 h
    "DNT": (0.0, 10080.0),                        # min — door-to-needle, capped at 7d
    "OPT": (0.0, 2880.0),                       # min — onset-to-puncture, capped at 48 h
    "IVT": (0.0, 1.0),                          # binary
    "EVT": (0.0, 1.0),                          # binary
}
assert list(FEATURE_RANGES) == list(FEATURE_COLS), \
    "FEATURE_RANGES must mirror fed_stroke.schema.FEATURE_COLS — same keys, same ORDER"
for _name, (_lo, _hi) in FEATURE_RANGES.items():
    assert MISSING_SENTINEL < _lo < _hi, f"FEATURE_RANGES[{_name!r}]: need sentinel < lo < hi"
    assert not is_binary_feature(_name) or (_lo, _hi) == (0.0, 1.0), \
        f"FEATURE_RANGES[{_name!r}]: binary features are (0, 1)"
del _name, _lo, _hi


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
    # noise_seed: NEVER a production knob. DP noise draws from OS entropy (R2); a config that
    # seeds the noise stream fail-closes in from_run_config unless the insecure_test escape
    # hatch is set AND the run is not on real frozen-schema data.
    noise_seed: int | None = None
    insecure_test: bool = False

    @classmethod
    def from_run_config(cls, cfg: dict, node_provenance: str | None = None) -> "DPConfig":
        """Map cfg["dp"] (already through unflatten_dict + task.replace_keys) to a DPConfig.

        Keys arrive with `-`→`_` already applied to KEYS by replace_keys; enum-like VALUES
        (e.g. bin-strategy = "fixed-range") keep their dash, so normalize the string values
        `-`→`_` here so `"fixed-range"` maps to the `"fixed_range"` field value.

        Fail-closed (R2): a `dp.noise-seed` key with `dp.enabled = true` is REFUSED — noise
        seeded from any config input breaks the DP inequality at every ε (reviewer A, finding 1;
        Opacus secure mode likewise prohibits user seeds). The only escape is the explicit
        `dp.insecure-test = true` hatch, itself rejected when `data-provenance` is
        "real-frozen-schema": no run whose ε could be claimed on real patients may ever draw
        deterministic noise.

        `node_provenance` (the NODE-OWNED `node_config["data-provenance"]`) is authoritative
        when given (reviewer A C1 / B F8): the submitter's cfg string cannot unlock the hatch
        on a real node. The cfg fallback only serves callers without a node context (tests,
        the single-site prototype).
        """
        dp = cfg.get("dp", {})

        def norm(v):
            return v.replace("-", "_") if isinstance(v, str) else v

        if dp.get("noise_seed") is not None and dp.get("enabled", cls.enabled):
            insecure = bool(dp.get("insecure_test", False))
            provenance = (node_provenance if node_provenance is not None
                          else cfg.get("data_provenance", "example-halves"))
            if not insecure:
                raise ValueError(
                    "dp.noise-seed with dp.enabled = true is refused: DP noise must draw fresh "
                    "OS entropy (R2). Deterministic noise is injection-only (tests) or requires "
                    "the explicit dp.insecure-test = true escape hatch on non-real data."
                )
            if provenance == "real-frozen-schema":
                raise ValueError(
                    "dp.insecure-test cannot seed DP noise on real-frozen-schema data: a run "
                    "whose ε could be claimed on real patients must never draw deterministic "
                    "noise (R2)."
                )

        return cls(
            enabled=dp.get("enabled", cls.enabled),
            mechanism=norm(dp.get("mechanism", cls.mechanism)),
            target_epsilon=float(dp.get("target_epsilon", cls.target_epsilon)),
            delta=float(dp.get("delta", cls.delta)),
            clip_bound=float(dp.get("clip_bound", cls.clip_bound)),
            max_bins=int(dp.get("max_bins", cls.max_bins)),
            bin_strategy=norm(dp.get("bin_strategy", cls.bin_strategy)),
            noise_multiplier=dp.get("noise_multiplier", cls.noise_multiplier),
            noise_seed=dp.get("noise_seed", cls.noise_seed),
            insecure_test=bool(dp.get("insecure_test", cls.insecure_test)),
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
    dict's insertion order (which must match the X column order = FEATURE_COLS).

    REMOVE-IF-NO-DP (missingness policy variant 1, 2026-07-23): bin 0 is RESERVED for the
    public MISSING_SENTINEL — edges are [sentinel, linspace(lo, hi, max_bins)], so the sentinel
    lands in bin 0 and every real value ≥ lo lands in bins 1..max_bins-1, for ANY max_bins
    (the bin-0/bin-1 boundary is exactly lo, never a fraction of the range). Still fully
    data-independent and public; a record still lands in exactly ONE bin per feature, so the
    sensitivity bounds and the accounting are untouched. Cost: max_bins-1 (not max_bins) bins
    of real-value resolution."""
    for name, (lo, hi) in feature_ranges.items():
        if lo <= MISSING_SENTINEL:
            raise ValueError(
                f"feature range for {name!r} starts at {lo}, not above the missing sentinel "
                f"{MISSING_SENTINEL} — bin 0 could not separate missing from real values."
            )
    return [
        np.concatenate(([MISSING_SENTINEL], np.linspace(lo, hi, max_bins)))
        for (lo, hi) in feature_ranges.values()
    ]


def quantile_bin_edges(X: np.ndarray, max_bins: int) -> list[np.ndarray]:
    """Empirical-quantile edges — NON-DP arms only (leaks the feature distribution, §3.7)."""
    qs = np.linspace(0.0, 1.0, max_bins + 1)
    return [np.quantile(X[:, j], qs) for j in range(X.shape[1])]


def _binize(X: np.ndarray, edges: list[np.ndarray], max_bins: int) -> np.ndarray:
    """Map each column to an integer bin index in [0, max_bins-1] using its edges.

    Fails loudly on a width mismatch: an edge grid built for a different feature set (a stale
    artifact from before the 41-feature freeze, or schema drift) must never silently bin the
    first len(edges) columns and leave the rest uninitialised."""
    if X.shape[1] != len(edges):
        raise ValueError(
            f"_binize: X has {X.shape[1]} columns but {len(edges)} edge arrays — the bin grid "
            f"was built for a different feature set (stale model artifact or schema drift)."
        )
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

    dp.mechanism="identity" WITH dp.enabled=True is the R9 comparator arm B: the DP learner
    (same fixed binning, same q=1.0, same tree code as arm C) with the noise turned off. It
    releases nothing under a DP claim (ε=∞ reported, ZERO releases, never enters the R6
    ledger); the 1.1.b privacy-cost headline is B→C, which differs ONLY in this mechanism.

    Gaussian (primary): n_rel = num_gaussian_releases(boost) = 2·D·T. σ is calibrated to
    dp.target_epsilon at q=1.0 (or pinned via dp.noise_multiplier, in which case ε is REPORTED).
    Per-feature std = σ·√d·G_L2 (G) and σ·√d·H_L2 (H). Feeding n_rel (not D·T) is load-bearing
    (§3.2) and is NOT covered by the Opacus gate.

    Laplace (secondary, L1): split dp.target_epsilon evenly across n_rel = 2·D·T releases;
    per-feature scales b_G = n_rel·d·G_L1/ε, b_H = n_rel·d·H_L1/ε (uses `d`, NOT `√d`).
    """
    if not dp.enabled or dp.mechanism == "identity":
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

    raise ValueError(
        f"unknown dp.mechanism={dp.mechanism!r} (expected 'gaussian' | 'laplace' | 'identity')"
    )


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


def _json_finite(obj):
    """Recursive pre-pass run BEFORE json.dumps: replace every non-finite float (inf/nan) with
    None, and cast numpy scalars (np.integer/np.floating) to native py int/float (§4.1, C-B1).

    REQUIRED, not decorative: json.dumps(allow_nan=False) RAISES on a native float inf/nan, and
    the `default=` hook is only invoked for types the encoder does NOT recognize — it never fires
    for a recognized float — so inf/nan -> null cannot be done by default= alone. meta legitimately
    carries inf (identity ε = ∞) and nan (Laplace σ), so those must be scrubbed here first; the
    subsequent allow_nan=False is then only a belt-and-suspenders assert that nothing slipped
    through (matches the repo artifact contract, server_app.py:195)."""
    if isinstance(obj, dict):
        return {k: _json_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_finite(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


class DPBooster:
    """A trained DP-GBDT ensemble. Serialization (to_json_bytes/from_json_bytes) + a DP-aware
    server aggregator (strategies.DPFedXgbBagging) wire it into the live federation (§4.1/§4.6)."""
    def __init__(self, trees, base_margin, edges, max_bins,
                 feature_ranges: dict | None = None, meta: dict | None = None):
        self.trees = trees                # list of nested-dict trees
        self.base_margin = base_margin
        self.edges = edges
        self.max_bins = max_bins
        # feature_ranges: the {name:(lo,hi)} dict `edges` were built from — first-class because it
        #   is UNRECOVERABLE from the numpy `edges` (names gone), and reaching for the module
        #   FEATURE_RANGES at serialize time would drift edges for a custom-range booster (§4.1, C-B2).
        # meta: accounting provenance (ε/σ/release-count + config block) that must survive the whole
        #   dp_local_boost -> serialize -> merge -> re-serialize -> rebuild -> log/save chain, since a
        #   reloaded/merged booster has NO mechanism to derive it from (Decision 11).
        # Both DEFAULT to None so the transient per-round margin booster inside _grow_trees
        # (DPBooster([tree], 0.0, edges, max_bins), never serialized) keeps its 4-arg construction.
        # to_json_bytes RAISES if feature_ranges is None — a booster meant to cross the wire must
        # carry it.
        self.feature_ranges = feature_ranges
        self.meta = meta

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

    def to_json_bytes(self) -> bytes:
        """Serialize to JSON bytes (mirrors xgb Booster.save_raw('json')) so a DP ensemble crosses
        the Flower transport as bytes in ArrayRecord["0"] exactly like an XGB model (§4.1).

        Emits self.meta VERBATIM (not re-derived from .mechanism — a merged/reloaded booster has
        no mechanism). Edges are NOT stored: self.feature_ranges + max_bins reconstruct them on
        load via the SAME fixed_bin_edges the writer used (no float drift at bin boundaries,
        byte-identical edges across sites/rounds). The FULL payload passes through _json_finite()
        FIRST (inf/nan -> null, numpy scalars -> py), THEN json.dumps(allow_nan=False) as a
        belt-and-suspenders assert that nothing non-finite slipped through."""
        if self.feature_ranges is None:
            raise ValueError(
                "DPBooster.to_json_bytes requires feature_ranges (a wire-bound booster must carry "
                "the {name:(lo,hi)} dict its edges were built from); got None."
            )
        payload = {
            "format": DP_MODEL_FORMAT,
            "trees": self.trees,
            "base_margin": float(self.base_margin),
            "feature_ranges": {k: [float(lo), float(hi)]
                               for k, (lo, hi) in self.feature_ranges.items()},
            "max_bins": int(self.max_bins),
            "meta": self.meta or {},
        }
        return json.dumps(_json_finite(payload), allow_nan=False).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "DPBooster":
        """Inverse of to_json_bytes: parse JSON, reconstruct edges via
        fixed_bin_edges(feature_ranges, max_bins), rebuild the DPBooster with feature_ranges +
        meta restored. Raises ValueError if data is empty or the format marker is absent/unknown
        (a wrong-format model must fail loudly, never predict garbage)."""
        if not data:
            raise ValueError("DPBooster.from_json_bytes: empty bytes (no model).")
        payload = json.loads(bytes(data).decode("utf-8"))
        if payload.get("format") != DP_MODEL_FORMAT:
            raise ValueError(
                f"DPBooster.from_json_bytes: unknown/absent format marker "
                f"{payload.get('format')!r} (expected {DP_MODEL_FORMAT!r})."
            )
        feature_ranges = {k: (float(lo), float(hi))
                          for k, (lo, hi) in payload["feature_ranges"].items()}
        max_bins = int(payload["max_bins"])
        edges = fixed_bin_edges(feature_ranges, max_bins)
        return cls(
            trees=payload["trees"],
            base_margin=float(payload["base_margin"]),
            edges=edges,
            max_bins=max_bins,
            feature_ranges=feature_ranges,
            meta=payload.get("meta") or {},
        )


def _grow_trees(binned, y, edges, boost: BoostParams, dp: DPConfig, mechanism, rng,
                init_margin: np.ndarray, num_rounds: int) -> list:
    """The boost loop + nested split-finder, lifted VERBATIM from train_dp_gbdt so the noise
    logic lives in exactly ONE place (§4.2, Decision 3). Grows `num_rounds` trees starting from
    `init_margin` (a constant base_margin for a fresh fit, or the incoming global model's margins
    for a federated continuation — DP sensitivity is margin-invariant, §3.2). The ONLY place noise
    enters stays mechanism.add_noise inside build(). Returns the list of new nested-dict trees.

    `build` closes over the per-round `g`/`h` (reassigned each iteration) exactly as before, so
    trees are bitwise-identical to the pre-refactor loop at a fixed rng (pinned, §4.9 case 8)."""
    y = np.asarray(y, dtype=float)
    n, d = binned.shape
    lam, mcw, eta, clip = boost.reg_lambda, boost.min_child_weight, boost.eta, dp.clip_bound
    max_depth, max_bins = boost.max_depth, dp.max_bins

    def build(rows, depth, inherited):
        # inherited = (G_total, H_total) from the parent's noised histogram, or None at the root.
        # ONLY the depth cap may leaf-exit before the query: whether the noised histogram is
        # issued must never depend on the raw partition (rows.shape[0] == 0 skipping the query
        # made "query issued at all" a data-dependent event — reviewer B F4, spec 1.1.a″ R1).
        # Empty rows -> np.bincount over empty arrays -> all-zero histograms -> the release is
        # pure noise; 2·D·T already charges for it, and parallel composition is unaffected.
        if depth >= max_depth:
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
    margin = np.asarray(init_margin, dtype=float).copy()
    trees = []
    for _ in range(num_rounds):
        p = _sigmoid(margin)
        g = p - y                 # gradient  ∈ [-1, 1]
        h = p * (1.0 - p)         # hessian   ∈ (0, 0.25]
        tree = build(all_rows, 0, None)
        trees.append(tree)
        booster_step = DPBooster([tree], 0.0, edges, max_bins)
        margin = margin + booster_step._tree_predict(tree, binned)
    return trees


def _mechanism_meta(mechanism, dp: DPConfig, boost: BoostParams, train_method: str,
                    per_site_trees: int | None = None) -> dict:
    """Accounting provenance for DPBooster.meta, read at save/log time from a reloaded booster
    that no longer has a mechanism (Decision 11, §4.1). num_releases/noise_multiplier/
    reported_epsilon come straight off the RUN-CALIBRATED mechanism; the dp block records the
    knobs. Non-finite values (identity ε=∞, Laplace σ=nan) are scrubbed to null on serialize."""
    if per_site_trees is None:
        if mechanism.num_releases == 0:
            # Identity (arm B): no releases, so there is no per-site accounting budget to
            # report — None (-> null on serialize), never a misleading 0.
            per_site_trees = None
        else:
            # 2·D·per_site releases -> per_site = releases // (2·max_depth) (§3.3).
            per_site_trees = (mechanism.num_releases // (2 * boost.max_depth)
                              if boost.max_depth else None)
    return {
        "train_method": train_method,
        "per_site_trees": per_site_trees,
        "num_releases": int(mechanism.num_releases),
        "noise_multiplier": float(mechanism.noise_multiplier),
        "reported_epsilon": float(mechanism.reported_epsilon),
        "dp": {
            "enabled": dp.enabled,
            "mechanism": dp.mechanism,
            "target_epsilon": dp.target_epsilon,
            "delta": dp.delta,
            "clip_bound": dp.clip_bound,
            "max_bins": dp.max_bins,
            "bin_strategy": dp.bin_strategy,
        },
        "fed_run_config": {},
    }


def train_dp_gbdt(X, y, boost: BoostParams, dp: DPConfig,
                  feature_ranges: dict | None = None,
                  mechanism: HistogramNoiseMechanism | None = None,
                  rng: np.random.Generator | None = None) -> DPBooster:
    """Minimal histogram GBDT with the DP mechanism as the ONLY place noise enters.

    Now a THIN wrapper over _grow_trees (init_margin = logit(base_score), num_rounds =
    boost.num_boost_round), so behavior is byte-identical to the pre-refactor loop (§4.2 case 8);
    the prototype/demo/single-site tests are untouched.

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
        # DP mode defaults to FRESH OS ENTROPY (R2): no seed — not boost.seed, not any public
        # input — may feed the noise stream, or the DP inequality fails at every ε for δ < 1
        # (reviewer A, finding 1). Deterministic noise is injection-only (pass rng explicitly,
        # tests). Off-DP the rng is inert for noise (identity mechanism), so the seeded default
        # keeps the non-DP learner reproducible.
        rng = np.random.default_rng(None if dp.enabled else boost.seed)
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
        fr = None
    else:
        fr = feature_ranges if feature_ranges is not None else FEATURE_RANGES
        edges = fixed_bin_edges(fr, dp.max_bins)

    binned = _binize(X, edges, dp.max_bins)
    base_margin = _logit(boost.base_score)
    init_margin = np.full(n, base_margin, dtype=float)
    trees = _grow_trees(binned, y, edges, boost, dp, mechanism, rng,
                        init_margin, boost.num_boost_round)

    booster = DPBooster(trees, base_margin, edges, dp.max_bins,
                        feature_ranges=fr,
                        meta=_mechanism_meta(mechanism, dp, boost, train_method="single-site"))
    # Expose what the mechanism accounted, so the demo/tests can read σ, ε, and the release count.
    booster.mechanism = mechanism
    return booster


def per_site_tree_budget(train_method: str, num_rounds: int, num_sites: int,
                         local_epochs: int) -> int:
    """Trees the BUSIEST site grows across the WHOLE run — the per-patient release count that
    calibrates σ once for the run (§3.3/§3.4).

    bagging: num_rounds * local_epochs  (every site trains every round; == total_trees//num_sites)
    cyclic : ceil(num_rounds / num_sites) * local_epochs  (one site/round; the site that trains
             the most rounds releases the most). NEVER total_trees//num_sites for cyclic: when
             num_rounds % num_sites != 0 that under-reports the busiest site's ε (§3.4, Risk 1).
    """
    if train_method == "bagging":
        return num_rounds * local_epochs
    if train_method == "cyclic":
        return math.ceil(num_rounds / num_sites) * local_epochs
    raise ValueError(f"Unknown train-method: {train_method!r}")


def dp_local_boost(global_booster, X, y, boost: BoostParams, dp: DPConfig, mechanism, rng,
                   num_local_round: int, train_method: str) -> DPBooster:
    """DP analog of client_app._local_boost: resume boosting from the incoming global model's
    margins and grow only THIS round's trees (§4.2).

    `boost` is the GROWTH params (num_boost_round == num_local_round). `mechanism` is the
    RUN-CALIBRATED mechanism (σ fixed to n_rel = 2·D·T_site, §4.3) — passed in, never rebuilt
    here. `rng` is the per-round, per-site generator the caller seeds from public
    (base_seed, global_round, site) (§3.5) — passed in and forwarded verbatim to _grow_trees, so
    noise is independent across rounds and sites. Both are NEVER defaulted: a missing rng would
    silently collapse to boost.seed and draw identical noise every round (the C-A1 failure this
    signature exists to prevent).

    Binning/edges are data-independent, so continuation reuses the global's edges/feature_ranges
    (identical to a fresh build). round 1 (global_booster is None): fresh fit from logit(base_score).
    Bagging returns only the NEW trees; cyclic returns global.trees + new trees (mirrors
    _local_boost's slice-vs-full, client_app.py:34-42)."""
    # Parity with train_dp_gbdt: DP mode is fixed-range only (§4.2). dp_local_boost always builds
    # fixed_bin_edges, so guard explicitly or an illegal dp.bin-strategy='quantile' runs
    # silently-safe here while train_dp_gbdt raises — a behavioral split for the same config.
    if dp.enabled and dp.bin_strategy != "fixed_range":
        raise ValueError(
            f"DP mode requires bin_strategy='fixed_range' (data-independent, DP-safe); "
            f"got {dp.bin_strategy!r} (§3.7)."
        )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    base_margin = _logit(boost.base_score)

    if global_booster is None:
        fr = FEATURE_RANGES
        edges = fixed_bin_edges(fr, dp.max_bins)
        init_margin = np.full(X.shape[0], base_margin, dtype=float)
    else:
        fr = global_booster.feature_ranges
        edges = global_booster.edges
        init_margin = global_booster.predict_margin(X)

    binned = _binize(X, edges, dp.max_bins)
    new_trees = _grow_trees(binned, y, edges, boost, dp, mechanism, rng,
                           init_margin, num_local_round)

    if train_method == "cyclic" and global_booster is not None:
        trees = list(global_booster.trees) + new_trees   # full ensemble adopted wholesale
    else:
        trees = new_trees                                 # bagging: this round's new trees only

    meta = _mechanism_meta(mechanism, dp, boost, train_method=train_method)
    booster = DPBooster(trees, base_margin, edges, dp.max_bins, feature_ranges=fr, meta=meta)
    booster.mechanism = mechanism
    return booster


def assert_dp_roundtrip(booster: "DPBooster", X) -> None:
    """Tests-only tripwire (mirrors baseline.assert_prediction_roundtrip): a booster reloaded from
    its own to_json_bytes must predict bitwise-identically on X. NOT used by the server aggregator
    — the server has no feature data X, so it does the structural check (feature_ranges/base_margin/
    max_bins equality across replies) instead (§4.1, C-A3)."""
    reloaded = DPBooster.from_json_bytes(booster.to_json_bytes())
    y_before = booster.predict(X)
    y_after = reloaded.predict(X)
    if not np.array_equal(y_before, y_after):
        n_diff = int(np.sum(y_before != y_after))
        raise ValueError(
            f"DPBooster serialization round-trip not a fixed point: {n_diff} predictions differ."
        )
