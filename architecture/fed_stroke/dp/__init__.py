"""fed_stroke.dp: the DP plug-point prototype (roadmap Phase v1.1/v1.2 prerequisite).

Re-exports the plug-point SEAM the downstream federated `client_app` DP branch imports
(spec 1.1 §4.5): a single `from fed_stroke.dp import ...` reaches the config, the mechanism
protocol, the learner, and the DP-safe feature ranges.

Only `accounting`, `boost`, and `preconditions` are imported here — all torch-free — so
`import fed_stroke.dp` stays a minimal, torch-free seam (acceptance §7.6). The synthetic
generator (`fed_stroke.dp.synthetic`) is a dev/test helper and the run ledger
(`fed_stroke.dp.ledger`) a harness helper; both are imported explicitly where needed.
"""
from fed_stroke.dp.boost import (
    DP_MODEL_FORMAT,
    DPBooster,
    DPConfig,
    BoostParams,
    FEATURE_RANGES,
    HistogramNoiseMechanism,
    dp_local_boost,
    make_mechanism,
    num_gaussian_releases,
    num_histogram_queries,
    per_site_tree_budget,
    train_dp_gbdt,
)
from fed_stroke.dp.preconditions import validate_dp_preconditions

__all__ = [
    "DP_MODEL_FORMAT",
    "DPBooster",
    "DPConfig",
    "BoostParams",
    "FEATURE_RANGES",
    "HistogramNoiseMechanism",
    "dp_local_boost",
    "make_mechanism",
    "num_gaussian_releases",
    "num_histogram_queries",
    "per_site_tree_budget",
    "train_dp_gbdt",
    "validate_dp_preconditions",
]
