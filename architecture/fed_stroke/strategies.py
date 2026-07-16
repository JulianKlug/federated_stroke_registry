"""fed_stroke: custom server strategies.

`OrderedFedXgbCyclic` gives flwr's `FedXgbCyclic` a deterministic, site-name
driven node order so the forward/reverse run pair of roadmap 1.b's last-site
bias check is reproducible across SuperNode restarts and reruns.
"""
from logging import INFO

from flwr.app import Message, MessageType, RecordDict
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedXgbCyclic


class OrderedFedXgbCyclic(FedXgbCyclic):
    """FedXgbCyclic with a deterministic, site-name-driven node order.

    On first configure, queries every connected node for its site name (the
    client answers with its data file name, see client_app.site_info) and caches
    a node_id -> site map. Cycling order is the lexicographic sort of site names
    ("forward") or its exact reversal ("reverse") — stable across SuperNode
    restarts and reruns, as required by roadmap 1.b's last-site-bias check.

    flwr's `FedXgbCyclic` cycles nodes in registration (connection) order, which
    varies between runs. Sorting by raw node ID is deterministic within a run but
    node IDs are random per SuperNode process — a restart between the forward and
    reverse run could map the ID sort to sites differently and both runs could
    finish on the *same* site. Pinning order to site identity (which only the
    client knows, via its `node_config["data-path"]`) removes that failure mode.
    """

    # generous: only guards against a node that never answers the query
    # (e.g. version skew / missing @app.query handler), not against slow ones.
    SITE_QUERY_TIMEOUT_S = 60.0

    def __init__(self, order: str = "forward", **kwargs):
        super().__init__(**kwargs)
        if order not in ("forward", "reverse"):
            raise ValueError(f"cyclic-order must be forward|reverse, got: {order}")
        self.order = order
        self._node_sites: dict[int, str] = {}  # node_id -> site (data file name)

    def _ensure_site_map(self, grid: Grid) -> None:
        node_ids = list(grid.get_node_ids())
        unknown = [n for n in node_ids if n not in self._node_sites]
        if not unknown:
            return
        messages = [
            Message(
                content=RecordDict(),
                message_type=MessageType.QUERY,
                dst_node_id=nid,
            )
            for nid in unknown
        ]
        answered: set[int] = set()
        # timeout is mandatory: with timeout=None a node that never replies
        # hangs the whole run silently. send_and_receive returns only the
        # replies received within the window (see flwr grid.py docstring).
        for reply in grid.send_and_receive(
            messages, timeout=self.SITE_QUERY_TIMEOUT_S
        ):
            src = reply.metadata.src_node_id
            if reply.has_error():
                raise RuntimeError(
                    f"Site query failed for node {src}; "
                    "cannot establish deterministic cyclic order."
                )
            self._node_sites[src] = str(reply.content["config"]["site"])
            answered.add(src)
        missing = set(unknown) - answered
        if missing:  # hard gate: never fall back to node-ID order
            raise RuntimeError(
                f"Site query got no reply from node(s) {sorted(missing)} "
                f"within {self.SITE_QUERY_TIMEOUT_S}s (missing @app.query "
                "handler or unreachable node); aborting cyclic run."
            )
        log(INFO, "Cyclic site map (order=%s): %s", self.order, self._node_sites)

    def configure_train(self, server_round, arrays, config, grid):
        self._ensure_site_map(grid)
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for msg in messages:  # authoritative round->site evidence in server log
            log(
                INFO,
                "Round %s: training site %s",
                server_round,
                self._node_sites[msg.metadata.dst_node_id],
            )
        return messages

    def configure_evaluate(self, server_round, arrays, config, grid):
        self._ensure_site_map(grid)
        return super().configure_evaluate(server_round, arrays, config, grid)

    def _reorder_nodes(self, node_ids: list[int]) -> list[int]:
        return sorted(
            node_ids,
            key=lambda nid: self._node_sites[nid],  # KeyError = unmapped node: fail loud
            reverse=(self.order == "reverse"),
        )
