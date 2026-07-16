"""Unit tests for OrderedFedXgbCyclic's site-name ordering + site map (§4.3)."""
from types import SimpleNamespace

import pytest

from fed_stroke.strategies import OrderedFedXgbCyclic


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
