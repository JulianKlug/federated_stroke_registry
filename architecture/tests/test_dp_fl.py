"""DP → FL integration tests (spec 1.1.a′ §4.9).

Pins the correctness guards the Opacus equivalence gate alone cannot provide: the run-level
per-busiest-site accounting (cases 1-3), per-round/per-site noise independence + rng threading
(case 4), serialization round-trip + non-finite meta (case 5), the bagging tree-concat merge
(case 6), the empty-accumulator/round-1 sentinel (case 7), the train_dp_gbdt byte-identity of the
_grow_trees refactor (case 8), offline dispatch (case 10), meta persistence through save (case 11),
and the @evaluate DP contract (case 12). Case 9 (non-DP byte-identity) lives in test_client_boost.

Reuses the tiny-synthetic-data idiom of tests/dp/test_boost.py. Cases 6/11 build REAL flwr
Message/RecordDict/ArrayRecord/MetricRecord replies — a SimpleNamespace fake fails flwr's
validate_message_reply_consistency (the reply must be a real RecordDict with one ArrayRecord +
one MetricRecord carrying num-examples).
"""
import math

import numpy as np
import pytest
import xgboost as xgb
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from fed_stroke.client_app import _dp_train_round, _site_hash
from fed_stroke.dp import (
    DP_MODEL_FORMAT,
    BoostParams,
    DPBooster,
    DPConfig,
    dp_local_boost,
    make_mechanism,
    num_gaussian_releases,
    per_site_tree_budget,
    train_dp_gbdt,
)
from fed_stroke.dp import accounting as A
from fed_stroke.dp import boost as B
from fed_stroke.metrics import REQUIRED_METRIC_KEYS, compute_binary_metrics
from fed_stroke.schema import FEATURE_COLS, TARGET_COL
from fed_stroke.server_app import derive_num_rounds, save_final_model
from fed_stroke.strategies import DPFedXgbBagging

pytestmark = pytest.mark.filterwarnings("ignore:Optimal RDP order")

DELTA = 1e-5


def _data(n=200, seed=0):
    """Two-feature synthetic site in the FEATURE_RANGES columns/order (Age, NIH)."""
    gen = np.random.default_rng(seed)
    X = np.column_stack([gen.uniform(0, 120, n), gen.uniform(0, 42, n)])
    y = ((X[:, 0] / 120 + X[:, 1] / 42 + gen.normal(0, 0.3, n)) > 1.0).astype(float)
    return X, y


def _params(base_score=0.5, seed=0):
    return {"max_depth": 4, "base_score": base_score, "seed": seed}


def _train_reply(booster: DPBooster, num_examples: int, node: int) -> Message:
    """A REAL flwr train reply carrying the DPBooster bytes (what aggregate_train validates)."""
    arr = ArrayRecord([np.frombuffer(booster.to_json_bytes(), dtype=np.uint8)])
    content = RecordDict({"arrays": arr,
                          "metrics": MetricRecord({"num-examples": num_examples})})
    instruction = Message(content=RecordDict(), dst_node_id=node, message_type="train")
    return Message(content=content, reply_to=instruction)


# --- Case 1: release-count / σ calibration -----------------------------------

def test_release_count_and_sigma_reconstruction():
    # 40 trees / 2 sites / depth 4 / 1 epoch -> bagging num_rounds=20, per_site=20 -> n_rel=160.
    num_rounds = derive_num_rounds("bagging", 40, 2, 1)
    per_site = per_site_tree_budget("bagging", num_rounds, 2, 1)
    assert per_site == 20
    acct = BoostParams.from_xgb_params(_params(), num_boost_round=per_site)
    assert num_gaussian_releases(acct) == 2 * 4 * 20 == 160

    dp = DPConfig(enabled=True, mechanism="gaussian", target_epsilon=30.0, delta=DELTA)
    m1 = make_mechanism(dp, acct, 2)
    m2 = make_mechanism(dp, acct, 2)
    assert m1.num_releases == 160
    # σ is a pure function of config -> every stateless round reconstructs the identical σ (§3.5).
    assert m1.noise_multiplier == m2.noise_multiplier
    # ε cross-checks the Opacus-gated accountant at that n_rel (q=1.0).
    assert m1.reported_epsilon == pytest.approx(
        A.account_run(160, m1.noise_multiplier, 1.0, DELTA), rel=1e-9
    )


# --- Case 2: cyclic uneven participation (highest value, pins §3.4) ----------

def test_cyclic_budget_uses_ceil_never_floor():
    # 41 rounds / 2 sites: busiest site trains ceil(41/2)=21 rounds, NOT 41//2=20.
    assert per_site_tree_budget("cyclic", 41, 2, 1) == 21
    # bagging budget == total_trees // num_sites (exact).
    assert per_site_tree_budget("bagging", 20, 2, 1) == 20 == 40 // 2
    # The reported ε at the true budget 21 is >= ε at the under-counted 20 (never under-report).
    sigma = 2.46
    eps20 = A.account_run(2 * 4 * 20, sigma, 1.0, DELTA)
    eps21 = A.account_run(2 * 4 * 21, sigma, 1.0, DELTA)
    assert eps21 >= eps20

    with pytest.raises(ValueError):
        per_site_tree_budget("sideways", 10, 2, 1)


# --- Case 3: two-BoostParams separation (pins §4.3, drives pure _dp_train_round)

def test_two_boostparams_grows_local_epochs_at_run_level_sigma():
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    total_trees, num_sites, local_epochs = 8, 2, 2
    num_rounds = derive_num_rounds("bagging", total_trees, num_sites, local_epochs)  # 2
    per_site = per_site_tree_budget("bagging", num_rounds, num_sites, local_epochs)  # 4
    expected_releases = 2 * 4 * per_site

    grown = 0
    global_bytes = None
    running = None
    sigmas = set()
    for r in range(1, num_rounds + 1):
        booster = _dp_train_round(
            _params(), dp, r, local_epochs, "bagging",
            total_trees, num_sites, X, y, global_bytes, site="node_A",
        )
        # Exactly local_epochs NEW trees per round (growth params), never per_site.
        assert len(booster.trees) == local_epochs
        # σ stays the RUN-level value every round (accounting params), not local_epochs-level.
        assert booster.meta["num_releases"] == expected_releases
        sigmas.add(round(booster.meta["noise_multiplier"], 12))
        grown += len(booster.trees)
        # Feed a growing global so the next round continues (mirrors bagging concat).
        running = booster if running is None else DPBooster(
            list(running.trees) + list(booster.trees), booster.base_margin,
            booster.edges, booster.max_bins,
            feature_ranges=booster.feature_ranges, meta=booster.meta,
        )
        global_bytes = running.to_json_bytes()
    assert grown == per_site           # total grown over the run == busiest-site budget
    assert len(sigmas) == 1            # one run-level σ across all rounds


# --- Case 4: cross-round + cross-site noise independence (pins §3.5 + rng threading)

def test_noise_independent_across_rounds_and_sites_and_reproducible():
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    kw = dict(local_epochs=1, train_method="bagging", total_trees=40, num_sites=2, X=X, y=y)

    # A round-2 global model to continue from (same for every continuation below).
    g = _dp_train_round(_params(), dp, 1, 1, "bagging", 40, 2, X, y, None, site="node_A")
    gb = DPBooster(list(g.trees), g.base_margin, g.edges, g.max_bins,
                   feature_ranges=g.feature_ranges, meta=g.meta).to_json_bytes()

    r2_A = _dp_train_round(_params(), dp, 2, global_model_bytes=gb, site="node_A", **kw)
    r3_A = _dp_train_round(_params(), dp, 3, global_model_bytes=gb, site="node_A", **kw)
    r2_B = _dp_train_round(_params(), dp, 2, global_model_bytes=gb, site="node_B", **kw)
    r2_A_again = _dp_train_round(_params(), dp, 2, global_model_bytes=gb, site="node_A", **kw)

    # Consecutive rounds draw DIFFERENT noise (would be identical if rng didn't reach _grow_trees).
    assert r2_A.trees != r3_A.trees
    # Distinct sites in the SAME round draw DIFFERENT noise.
    assert r2_A.trees != r2_B.trees
    # Fully reproducible from the same public (base_seed, round, site) SeedSequence inputs.
    assert r2_A.trees == r2_A_again.trees
    # The seed derives only from public metadata.
    assert _site_hash("node_A") != _site_hash("node_B")
    assert _site_hash("node_A") == _site_hash("node_A")


# --- Case 5: serialization round-trip + non-finite meta + bad bytes ----------

def test_serialization_roundtrip_and_edges_and_bad_bytes():
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    boost = BoostParams(max_depth=3, num_boost_round=6, base_score=0.5, seed=0)
    b = train_dp_gbdt(X, y, boost, dp, rng=np.random.default_rng(0))

    B.assert_dp_roundtrip(b, X)  # predict fixed point
    reb = DPBooster.from_json_bytes(b.to_json_bytes())
    for e_new, e_old in zip(reb.edges, b.edges):     # edges reconstructed from feature_ranges
        assert np.array_equal(e_new, e_old)
    assert DP_MODEL_FORMAT in b.to_json_bytes().decode()

    # Non-finite meta: identity ε=∞ and Laplace σ=nan serialize to null, reload as None.
    b.meta["reported_epsilon"] = float("inf")
    b.meta["noise_multiplier"] = float("nan")
    reloaded = DPBooster.from_json_bytes(b.to_json_bytes())
    assert reloaded.meta["reported_epsilon"] is None
    assert reloaded.meta["noise_multiplier"] is None

    with pytest.raises(ValueError):
        DPBooster.from_json_bytes(b"")
    with pytest.raises(ValueError):
        DPBooster.from_json_bytes(b'{"format": "not-dp", "trees": []}')
    # A wire-bound booster must carry feature_ranges.
    with pytest.raises(ValueError):
        DPBooster([{"leaf": 0.0}], 0.0, b.edges, b.max_bins).to_json_bytes()


# --- Case 6: bagging merge equivalence (pins §3.6, base_score != 0.5) --------

def _bagging_reply(X, y, mech, growth, dp, site_hash, node, global_booster=None):
    b = dp_local_boost(global_booster, X, y, growth, dp, mech,
                       np.random.default_rng([0, 1, site_hash]), 1, "bagging")
    return b, _train_reply(b, len(y), node)


def test_bagging_merge_sums_contributions_and_rejects_drift():
    X, y = _data()
    # base_score=0.6 -> base_margin = logit(0.6) != 0, so "counted once" vs "twice" is observable.
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    acct = BoostParams.from_xgb_params(_params(base_score=0.6), num_boost_round=20)
    mech = make_mechanism(dp, acct, 2)
    growth = BoostParams.from_xgb_params(_params(base_score=0.6), num_boost_round=1)

    bA, replyA = _bagging_reply(X, y, mech, growth, dp, 111, 10)
    bB, replyB = _bagging_reply(X, y, mech, growth, dp, 222, 20)
    assert bA.base_margin == pytest.approx(math.log(0.6 / 0.4))

    # Direct concat with ONE shared base_margin == base_margin + A's trees + B's trees.
    merged = DPBooster(list(bA.trees) + list(bB.trees), bA.base_margin, bA.edges, bA.max_bins,
                       feature_ranges=bA.feature_ranges, meta=bA.meta)
    contribA = bA.predict_margin(X) - bA.base_margin
    contribB = bB.predict_margin(X) - bB.base_margin
    assert np.allclose(merged.predict_margin(X), bA.base_margin + contribA + contribB)

    # A 2-round DPFedXgbBagging continuation reproduces the hand-computed trajectory.
    strat = DPFedXgbBagging(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2)
    strat.current_bst = b""
    arrays1, _ = strat.aggregate_train(1, [replyA, replyB])
    g1 = DPBooster.from_json_bytes(bytes(arrays1["0"].numpy().tobytes()))
    assert len(g1.trees) == 2
    assert np.allclose(g1.predict_margin(X), bA.base_margin + contribA + contribB)

    # Round 2: both sites continue from g1; aggregate concatenates onto g1's trees.
    strat.current_bst = g1.to_json_bytes()
    c2, replyA2 = _bagging_reply(X, y, mech, growth, dp, 111, 10, global_booster=g1)
    d2, replyB2 = _bagging_reply(X, y, mech, growth, dp, 222, 20, global_booster=g1)
    arrays2, _ = strat.aggregate_train(2, [replyA2, replyB2])
    g2 = DPBooster.from_json_bytes(bytes(arrays2["0"].numpy().tobytes()))
    assert len(g2.trees) == 4  # g1's 2 + one new tree per site
    ref = (g1.predict_margin(X)
           + (c2.predict_margin(X) - c2.base_margin)
           + (d2.predict_margin(X) - d2.base_margin))
    assert np.allclose(g2.predict_margin(X), ref)

    # Mismatched feature_ranges / max_bins across replies raises.
    bad_fr = DPBooster(bA.trees, bA.base_margin, bA.edges, bA.max_bins,
                       feature_ranges={"X": (0.0, 1.0)}, meta=bA.meta)
    with pytest.raises(ValueError):
        DPFedXgbBagging._assert_mergeable(bA, bad_fr)


# --- Case 7: empty-sentinel + round-1 ----------------------------------------

def test_empty_accumulator_adopts_round1_replies():
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    acct = BoostParams.from_xgb_params(_params(), num_boost_round=20)
    mech = make_mechanism(dp, acct, 2)
    growth = BoostParams.from_xgb_params(_params(), num_boost_round=1)
    _, replyA = _bagging_reply(X, y, mech, growth, dp, 111, 10)
    _, replyB = _bagging_reply(X, y, mech, growth, dp, 222, 20)

    strat = DPFedXgbBagging(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2)
    strat.current_bst = b""   # the initial server seed
    arrays, metrics = strat.aggregate_train(1, [replyA, replyB])
    merged = DPBooster.from_json_bytes(bytes(arrays["0"].numpy().tobytes()))
    assert len(merged.trees) == 2   # adopted both replies, no double-count of an empty base
    assert metrics is not None

    # Client maps server-round == 1 to fresh training (global_dp is None), never byte-sniffing:
    # _dp_train_round with global_round=1 and global_model_bytes=None must not deserialize.
    fresh = _dp_train_round(_params(), dp, 1, 1, "bagging", 40, 2, X, y, None, site="node_A")
    assert len(fresh.trees) == 1


# --- Case 8: train_dp_gbdt byte-identity of the _grow_trees refactor ----------

def test_train_dp_gbdt_is_thin_wrapper_over_grow_trees():
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    boost = BoostParams(max_depth=3, num_boost_round=8, base_score=0.5, seed=0)

    # Determinism at a fixed rng.
    b1 = train_dp_gbdt(X, y, boost, dp, rng=np.random.default_rng(0))
    b2 = train_dp_gbdt(X, y, boost, dp, rng=np.random.default_rng(0))
    assert b1.trees == b2.trees
    assert np.array_equal(b1.predict(X), b2.predict(X))

    # The wrapper == the shared core: train_dp_gbdt trees equal a direct _grow_trees call with the
    # same init_margin / num_rounds, so the refactor cannot silently diverge from the prototype.
    edges = B.fixed_bin_edges(B.FEATURE_RANGES, dp.max_bins)
    binned = B._binize(np.asarray(X, float), edges, dp.max_bins)
    mech = make_mechanism(dp, boost, X.shape[1])
    init = np.full(X.shape[0], B._logit(boost.base_score))
    core = B._grow_trees(binned, y, edges, boost, dp, mech, np.random.default_rng(0),
                         init, boost.num_boost_round)
    assert core == b1.trees


# --- Case 10: offline dispatch (score_booster_on_half + eval loader sniff) ----

def test_score_booster_on_half_dispatches_on_type(tmp_path):
    from fed_stroke.baseline import score_booster_on_half
    import pandas as pd

    X, y = _data(n=240, seed=3)
    df = pd.DataFrame({FEATURE_COLS[0]: X[:, 0], FEATURE_COLS[1]: X[:, 1],
                       "case_admission_id": [f"{i}_a" for i in range(len(y))],
                       TARGET_COL: y.astype(int)})
    half = tmp_path / "geneva_half_A.parquet"
    df.to_parquet(half)

    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    dp_bst = train_dp_gbdt(X, y, BoostParams(max_depth=3, num_boost_round=8, seed=0), dp,
                           rng=np.random.default_rng(0))
    xgb_bst = xgb.train({"objective": "binary:logistic", "max_depth": 3, "seed": 0},
                        xgb.DMatrix(df[FEATURE_COLS], label=df[TARGET_COL]), num_boost_round=8)

    m_dp = score_booster_on_half(dp_bst, half, 0.5, 100, 0)
    m_xgb = score_booster_on_half(xgb_bst, half, 0.5, 100, 0)
    assert m_dp.keys() == m_xgb.keys()   # identical metric contract regardless of learner


def test_eval_final_model_loader_sniffs_format(tmp_path):
    import json
    X, y = _data(n=120, seed=5)
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    dp_bst = train_dp_gbdt(X, y, BoostParams(max_depth=3, num_boost_round=7, seed=0), dp,
                           rng=np.random.default_rng(0))
    dp_path = tmp_path / "dp_model.json"
    dp_path.write_bytes(dp_bst.to_json_bytes())
    # DP file: format marker present -> route to from_json_bytes, tree count = len(trees).
    assert json.loads(dp_path.read_bytes().decode()).get("format") == DP_MODEL_FORMAT
    assert len(DPBooster.from_json_bytes(dp_path.read_bytes()).trees) == 7

    # XGB file: no dp marker -> the sniff falls through to the XGBoost loader.
    xgb_bst = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 0},
                        xgb.DMatrix(X, label=y), num_boost_round=5)
    xgb_path = tmp_path / "xgb_model.json"
    xgb_bst.save_model(str(xgb_path))
    try:
        is_dp = json.loads(xgb_path.read_bytes().decode()).get("format") == DP_MODEL_FORMAT
    except (ValueError, UnicodeDecodeError):
        is_dp = False
    assert is_dp is False


# --- Case 11: meta persistence + logging contract (A2 chain) ------------------

def test_meta_survives_save_and_merge(tmp_path):
    X, y = _data()
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    total_trees, num_sites = 40, 2
    per_site = per_site_tree_budget("bagging", derive_num_rounds("bagging", total_trees, num_sites, 1),
                                    num_sites, 1)
    booster = _dp_train_round(_params(), dp, 1, 1, "bagging", total_trees, num_sites, X, y,
                              None, site="node_A")
    params = {"max_depth": 4, "seed": 0}
    out = save_final_model(booster, tmp_path / "models", params, total_trees)

    reloaded = DPBooster.from_json_bytes(out.read_bytes())
    assert reloaded.meta["reported_epsilon"] is not None
    assert reloaded.meta["noise_multiplier"] is not None
    assert reloaded.meta["num_releases"] == 2 * 4 * per_site
    assert reloaded.meta["fed_run_config"]["params"] == params
    assert reloaded.meta["fed_run_config"]["total_trees"] == total_trees
    # It matches what make_mechanism computed for that config.
    acct = BoostParams.from_xgb_params(_params(), num_boost_round=per_site)
    assert reloaded.meta["num_releases"] == make_mechanism(dp, acct, 2).num_releases

    # Merge carries meta from the replies through re-serialize.
    acctm = BoostParams.from_xgb_params(_params(), num_boost_round=per_site)
    mech = make_mechanism(dp, acctm, 2)
    growth = BoostParams.from_xgb_params(_params(), num_boost_round=1)
    _, ra = _bagging_reply(X, y, mech, growth, dp, 111, 10)
    _, rb = _bagging_reply(X, y, mech, growth, dp, 222, 20)
    strat = DPFedXgbBagging(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2)
    strat.current_bst = b""
    arrays, _ = strat.aggregate_train(1, [ra, rb])
    merged = DPBooster.from_json_bytes(bytes(arrays["0"].numpy().tobytes()))
    assert merged.meta["num_releases"] == 2 * 4 * per_site


# --- Case 12: @evaluate DP branch contract -----------------------------------

def test_evaluate_dp_branch_metric_contract():
    X, y = _data(n=180, seed=2)
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=DELTA)
    booster = train_dp_gbdt(X, y, BoostParams(max_depth=3, num_boost_round=8, seed=0), dp,
                            rng=np.random.default_rng(0))

    # Reconstruct what @evaluate does: deserialize from ArrayRecord["0"], predict on raw X,
    # score through the UNCHANGED compute_binary_metrics, apply the n->num-examples rename.
    arr = ArrayRecord([np.frombuffer(booster.to_json_bytes(), dtype=np.uint8)])
    reloaded = DPBooster.from_json_bytes(bytes(arr["0"].numpy().tobytes()))
    y_prob = reloaded.predict(X)
    metrics = compute_binary_metrics(y, y_prob, 0.5, n_boot=100, boot_seed=0)
    metrics["num-examples"] = metrics.pop("n")
    assert REQUIRED_METRIC_KEYS <= set(metrics)
    assert metrics["num-examples"] == len(y)
