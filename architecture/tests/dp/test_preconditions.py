"""validate_dp_preconditions gate tests — one test per condition (spec 1.1.a″ R7)."""
import numpy as np
import pytest

from fed_stroke.dp import (
    BoostParams,
    DPConfig,
    make_mechanism,
    validate_dp_preconditions,
)
from fed_stroke.dp import boost as B
from fed_stroke.dp.synthetic import background_matrix

pytestmark = pytest.mark.filterwarnings("ignore:Optimal RDP order")

DELTA = 1e-5


def _valid_inputs(n=50, mechanism_name="gaussian"):
    """A complete, gate-passing input set; tests mutate ONE condition each."""
    rng = np.random.default_rng(0)
    X = background_matrix(n, seed=0)          # every frozen column inside its public range
    y = rng.integers(0, 2, n).astype(float)
    pids = np.array([f"P{i:04d}" for i in range(n)])
    dp = DPConfig(enabled=True, mechanism=mechanism_name, target_epsilon=5.0, delta=DELTA)
    boost = BoostParams(max_depth=4, num_boost_round=20, base_score=0.5, seed=0)
    mechanism = make_mechanism(dp, boost, X.shape[1])
    params = {"max_depth": 4, "base_score": 0.5, "seed": 0}
    return dict(X=X, y=y, patient_ids=pids, dp=dp, params=params, mechanism=mechanism,
                global_round=1, num_rounds=20)


def test_gate_passes_on_valid_inputs():
    validate_dp_preconditions(**_valid_inputs())                       # gaussian
    validate_dp_preconditions(**_valid_inputs(mechanism_name="laplace"))


def test_gate_rejects_nonfinite_features():
    kw = _valid_inputs()
    kw["X"][3, 1] = np.nan
    with pytest.raises(ValueError, match="features"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_nonfinite_or_nonbinary_labels():
    kw = _valid_inputs()
    kw["y"][0] = np.inf
    with pytest.raises(ValueError, match="labels must be finite"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["y"][0] = 2.0
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_duplicate_patients():
    """The R3 learner-side belt: a load path that skipped generate_splits' dedup fails here."""
    kw = _valid_inputs()
    kw["patient_ids"][1] = kw["patient_ids"][0]
    with pytest.raises(ValueError, match="one row per patient"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_misaligned_patient_ids():
    kw = _valid_inputs()
    kw["patient_ids"] = kw["patient_ids"][:-1]
    with pytest.raises(ValueError, match="align"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_missing_or_invalid_base_score():
    kw = _valid_inputs()
    del kw["params"]["base_score"]
    with pytest.raises(ValueError, match="EXPLICITLY"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["params"]["base_score"] = 1.0        # boundary: logit(1) is inf
    with pytest.raises(ValueError, match="base_score"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_non_fixed_range_bins_and_feature_mismatch():
    kw = _valid_inputs()
    kw["dp"] = DPConfig(enabled=True, target_epsilon=5.0, delta=DELTA,
                        bin_strategy="quantile")
    with pytest.raises(ValueError, match="fixed_range"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["X"] = kw["X"][:, :-1]               # frozen-schema width drift (one column short)
    with pytest.raises(ValueError, match="columns"):
        validate_dp_preconditions(**kw)


def test_gate_rejects_nonfinite_feature_ranges_and_bad_clip():
    kw = _valid_inputs()
    kw["feature_ranges"] = {**B.FEATURE_RANGES, "age": (0.0, np.inf)}
    with pytest.raises(ValueError, match="bin edges"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["dp"] = DPConfig(enabled=True, target_epsilon=5.0, delta=DELTA, clip_bound=0.0)
    kw["mechanism"] = make_mechanism(kw["dp"], BoostParams(max_depth=4, num_boost_round=20),
                                     len(B.FEATURE_RANGES))
    with pytest.raises(ValueError, match="clip_bound"):
        validate_dp_preconditions(**kw)


def test_gate_gaussian_budget_checks():
    kw = _valid_inputs()
    kw["mechanism"] = B._GaussianMechanism(0.0, 0.0, 0.0, 5.0, 160)     # σ = 0
    with pytest.raises(ValueError, match="σ"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["dp"] = DPConfig(enabled=True, target_epsilon=5.0, delta=0.0, noise_multiplier=12.0)
    kw["mechanism"] = B._GaussianMechanism(12.0, 1.0, 0.25, 5.0, 160)
    with pytest.raises(ValueError, match="δ"):
        validate_dp_preconditions(**kw)
    kw = _valid_inputs()
    kw["mechanism"] = B._GaussianMechanism(12.0, 1.0, 0.25, float("inf"), 160)  # ε not finite
    with pytest.raises(ValueError, match="ε"):
        validate_dp_preconditions(**kw)


def test_gate_laplace_uses_scales_never_the_nan_multiplier():
    """The Laplace trap: noise_multiplier is nan BY CONSTRUCTION for Laplace — a blanket σ > 0
    check would reject every legitimate Laplace run. The gate validates b_g/b_h instead."""
    kw = _valid_inputs(mechanism_name="laplace")
    assert np.isnan(kw["mechanism"].noise_multiplier)
    validate_dp_preconditions(**kw)                    # legitimate laplace run passes
    kw["mechanism"] = B._LaplaceMechanism(0.0, 1.0, 5.0, 240)   # broken b_g
    with pytest.raises(ValueError, match="laplace scales"):
        validate_dp_preconditions(**kw)


def test_gate_identity_skips_budget_but_not_data_checks():
    kw = _valid_inputs(mechanism_name="identity")
    assert kw["mechanism"].num_releases == 0
    validate_dp_preconditions(**kw)                    # ε=∞ is fine: nothing is spent
    kw["y"][0] = 2.0                                   # ... but data checks still bind
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        validate_dp_preconditions(**kw)


def test_gate_enforces_authorized_round_count():
    kw = _valid_inputs()
    kw["global_round"] = 21                            # σ was calibrated for 20 rounds
    with pytest.raises(ValueError, match="authorized round count"):
        validate_dp_preconditions(**kw)


# --- run provenance gate (reviewer A C1/C2, reviewer B F8) --------------------
from fed_stroke.dp.preconditions import (  # noqa: E402
    PROVENANCE_EXAMPLE,
    PROVENANCE_REAL,
    validate_dp_run_provenance,
)

_REAL_NODE = {"data-provenance": PROVENANCE_REAL, "dp-ledger-path": "/abs/dp_ledger.jsonl"}
_EXAMPLE_NODE = {"data-provenance": PROVENANCE_EXAMPLE}


def test_run_provenance_is_node_owned_and_fail_closed():
    # No node declaration -> refuse (never default to "example").
    with pytest.raises(ValueError, match="data-provenance"):
        validate_dp_run_provenance({"data-path": "x.parquet"}, {}, "bagging")
    # Unknown vocabulary -> refuse.
    with pytest.raises(ValueError, match="data-provenance"):
        validate_dp_run_provenance({"data-provenance": "prod"}, {}, "bagging")
    assert validate_dp_run_provenance(_EXAMPLE_NODE, {}, "bagging") == PROVENANCE_EXAMPLE


def test_run_provenance_submitter_cannot_override_node():
    # The default run-config ("example-halves") on a REAL node is refused, never silently
    # downgraded into an unledgered spend (B F8).
    with pytest.raises(ValueError, match="disagrees"):
        validate_dp_run_provenance(_REAL_NODE, {"data-provenance": PROVENANCE_EXAMPLE},
                                   "bagging")
    # Nor can a submitter promote example data to a real-ε claim.
    with pytest.raises(ValueError, match="disagrees"):
        validate_dp_run_provenance(_EXAMPLE_NODE, {"data-provenance": PROVENANCE_REAL},
                                   "bagging")
    # Absent or agreeing declaration -> the node's value.
    assert validate_dp_run_provenance(_REAL_NODE, {}, "bagging") == PROVENANCE_REAL
    assert validate_dp_run_provenance(_REAL_NODE, {"data-provenance": PROVENANCE_REAL},
                                      "bagging") == PROVENANCE_REAL


def test_run_provenance_real_requires_node_ledger_path():
    with pytest.raises(ValueError, match="dp-ledger-path"):
        validate_dp_run_provenance({"data-provenance": PROVENANCE_REAL}, {}, "bagging")


def test_run_provenance_refuses_cyclic_on_real_data():
    # Cyclic ledgers only the round-1 site (A C2): refused on real data, still runnable as a
    # rehearsal on example halves.
    with pytest.raises(ValueError, match="cyclic"):
        validate_dp_run_provenance(_REAL_NODE, {}, "cyclic")
    assert validate_dp_run_provenance(_EXAMPLE_NODE, {}, "cyclic") == PROVENANCE_EXAMPLE
