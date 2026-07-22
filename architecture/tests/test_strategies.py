"""Unit tests for OrderedFedXgbCyclic's site-name ordering + site map (§4.3), plus the DP
bagging aggregator's tree-concat merge and meta carry (spec 1.1.a′ §4.6/§4.9 cases 6/11)."""
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from fed_stroke.dp import (
    BoostParams,
    DPBooster,
    DPConfig,
    dp_local_boost,
    make_mechanism,
)
from fed_stroke.strategies import DPFedXgbBagging, OrderedFedXgbCyclic


# ---- fakes -----------------------------------------------------------------


class FakeReply:
    """Minimal stand-in for a flwr reply Message.

    `_ensure_site_map` only touches `.metadata.src_node_id`, `.has_error()`, and
    `.content["config"]["site"]`, so a plain object with those is enough.
    """

    def __init__(self, src_node_id, site=None, error=False):
        self.metadata = SimpleNamespace(src_node_id=src_node_id)
        self._error = error
        self.content = {"config": {"site": site}}

    def has_error(self):
        return self._error


class FakeGrid:
    def __init__(self, node_ids, replies):
        self._node_ids = node_ids
        self._replies = replies
        self.calls = 0  # number of send_and_receive invocations
        self.last_messages = None

    def get_node_ids(self):
        return list(self._node_ids)

    def send_and_receive(self, messages, timeout=None):
        self.calls += 1
        self.last_messages = list(messages)
        return list(self._replies)


def _strategy(order="forward"):
    return OrderedFedXgbCyclic(
        order=order, fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2
    )


# ---- order validation ------------------------------------------------------


def test_order_forward_and_reverse_ok():
    assert _strategy("forward").order == "forward"
    assert _strategy("reverse").order == "reverse"


def test_order_invalid_raises():
    with pytest.raises(ValueError):
        _strategy("sideways")


# ---- _reorder_nodes --------------------------------------------------------


def test_reorder_forward_is_lexicographic_by_site():
    s = _strategy("forward")
    s._node_sites = {10: "geneva_half_B.parquet", 20: "geneva_half_A.parquet"}
    # A sorts before B → node 20 first
    assert s._reorder_nodes([10, 20]) == [20, 10]


def test_reorder_reverse_is_exact_reversal_of_forward():
    fwd = _strategy("forward")
    rev = _strategy("reverse")
    sites = {10: "geneva_half_B.parquet", 20: "geneva_half_A.parquet", 30: "c.parquet"}
    fwd._node_sites = dict(sites)
    rev._node_sites = dict(sites)
    node_ids = [30, 10, 20]
    assert rev._reorder_nodes(node_ids) == list(reversed(fwd._reorder_nodes(node_ids)))


def test_reorder_unmapped_node_raises_keyerror():
    s = _strategy("forward")
    s._node_sites = {10: "a.parquet"}
    with pytest.raises(KeyError):
        s._reorder_nodes([10, 99])


# ---- _ensure_site_map ------------------------------------------------------


def test_ensure_site_map_builds_from_replies():
    s = _strategy()
    grid = FakeGrid(
        node_ids=[10, 20],
        replies=[FakeReply(10, "geneva_half_B.parquet"),
                 FakeReply(20, "geneva_half_A.parquet")],
    )
    s._ensure_site_map(grid)
    assert s._node_sites == {10: "geneva_half_B.parquet", 20: "geneva_half_A.parquet"}
    assert grid.calls == 1


def test_ensure_site_map_caches_no_second_query():
    s = _strategy()
    grid = FakeGrid(
        node_ids=[10, 20],
        replies=[FakeReply(10, "b.parquet"), FakeReply(20, "a.parquet")],
    )
    s._ensure_site_map(grid)
    s._ensure_site_map(grid)  # all known now → must not query again
    assert grid.calls == 1


def test_ensure_site_map_raises_on_error_reply():
    s = _strategy()
    grid = FakeGrid(
        node_ids=[10, 20],
        replies=[FakeReply(10, error=True), FakeReply(20, "a.parquet")],
    )
    with pytest.raises(RuntimeError):
        s._ensure_site_map(grid)


def test_ensure_site_map_raises_on_missing_reply():
    # node 20 never answers (fewer replies than messages sent)
    s = _strategy()
    grid = FakeGrid(node_ids=[10, 20], replies=[FakeReply(10, "a.parquet")])
    with pytest.raises(RuntimeError):
        s._ensure_site_map(grid)


# ---- DPFedXgbBagging aggregate_train (cases 6/11) ---------------------------
# NOTE: the SimpleNamespace/FakeReply idiom above only fakes the site-query path; it does NOT
# satisfy aggregate_train, which inherits flwr's validate_message_reply_consistency and needs a
# REAL RecordDict with one ArrayRecord + one MetricRecord carrying num-examples per reply.


def _dp_replies(node_hashes):
    gen = np.random.default_rng(0)
    n = 160
    X = np.column_stack([gen.uniform(0, 120, n), gen.uniform(0, 42, n)])
    y = ((X[:, 0] / 120 + X[:, 1] / 42) > 1.0).astype(float)
    dp = DPConfig(enabled=True, target_epsilon=30.0, delta=1e-5)
    acct = BoostParams.from_xgb_params({"max_depth": 3, "base_score": 0.6}, num_boost_round=20)
    mech = make_mechanism(dp, acct, 2)
    growth = BoostParams.from_xgb_params({"max_depth": 3, "base_score": 0.6}, num_boost_round=1)
    replies, boosters = [], []
    for h, node in node_hashes:
        b = dp_local_boost(None, X, y, growth, dp, mech, np.random.default_rng([0, 1, h]),
                           1, "bagging")
        arr = ArrayRecord([np.frombuffer(b.to_json_bytes(), dtype=np.uint8)])
        content = RecordDict({"arrays": arr, "metrics": MetricRecord({"num-examples": n})})
        instr = Message(content=RecordDict(), dst_node_id=node, message_type="train")
        replies.append(Message(content=content, reply_to=instr))
        boosters.append(b)
    return replies, boosters, mech


def test_dp_bagging_aggregate_train_merges_and_returns_arrays_and_metrics():
    replies, boosters, mech = _dp_replies([(111, 10), (222, 20)])
    strat = DPFedXgbBagging(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2)
    strat.current_bst = b""   # the initial server seed, as inherited configure_train would set

    arrays, metrics = strat.aggregate_train(1, replies)
    assert arrays is not None and metrics is not None   # C-B3: both, not a bare ArrayRecord
    merged = DPBooster.from_json_bytes(bytes(arrays["0"].numpy().tobytes()))
    # round-1 adopts both replies' trees (one per site under bagging).
    assert len(merged.trees) == len(boosters)
    # meta survives merge -> re-serialize (case 11): num_releases is the run-level release count.
    assert merged.meta["num_releases"] == mech.num_releases
