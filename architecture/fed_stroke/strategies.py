"""fed_stroke: custom server strategies.

`OrderedFedXgbCyclic` gives flwr's `FedXgbCyclic` a deterministic, site-name
driven node order so the forward/reverse run pair of roadmap 1.b's last-site
bias check is reproducible across SuperNode restarts and reruns.

`DPFedXgbBagging` is the DP analog of flwr's `FedXgbBagging`: it merges DPBooster
models by CONCATENATING trees (the DP trees are parallel additive corrections on
one shared base_margin — §3.6/§4.6) rather than rewriting the XGBoost JSON schema.
"""
from collections.abc import Iterable
from logging import INFO
from typing import cast

import numpy as np
from flwr.app import ArrayRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic

from fed_stroke.dp import DPBooster


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


class DPFedXgbBagging(FedXgbBagging):
    """Bagging over DPBooster models (§4.6). Overrides `aggregate_train` only; `configure_train`,
    the `min_available_nodes`/single-array contract, and the MetricRecord aggregation are inherited.

    The inherited `configure_train` caches `self.current_bst = arrays["0"].tobytes()` each round —
    `b""` on round 1, else last round's merged DP JSON — so the accumulator base is always in
    `self.current_bst`, exactly as the XGB parent uses it.

    Merge (§3.6): DP trees from all sites this round are parallel additive corrections on the SAME
    shared starting margin, so the merge is a plain concatenation of `.trees` lists with a SINGLE
    shared `base_margin` (counted once); leaf weights already carry η, so concatenation sums
    η-scaled corrections exactly as XGB bagging does — no id/indptr bookkeeping. `edges`/
    `base_margin`/`max_bins`/`feature_ranges` are data-independent (identical across sites), so the
    aggregator ASSERTS they agree across replies — the server-side structural tripwire that
    replaces the predict-based round-trip check (the server has no feature data X, §4.1/C-A3)."""

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate DPBooster ArrayRecords + MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            array_record_key = next(iter(reply_contents[0].array_records.keys()))

            # 1. Deserialize each reply's DPBooster.
            boosters = []
            for content in reply_contents:
                self._ensure_single_array(cast(ArrayRecord, content[array_record_key]))
                data = content[array_record_key]["0"].numpy().tobytes()
                boosters.append(DPBooster.from_json_bytes(data))

            # 2. Structural tripwire: data-independent fields must agree across replies (and, when
            #    the accumulator is non-empty, with it too); else a node/config drift is merging
            #    incompatible bin grids — fail loud.
            ref = boosters[0]
            for b in boosters[1:]:
                self._assert_mergeable(ref, b)

            # 3./4. Accumulator base = self.current_bst (inherited configure_train). On server_round
            #    1 (== current_bst b"") adopt this round's replies (empty-accumulator sentinel,
            #    mirroring aggregate_bagging's `if bst_prev == b""`); else concatenate onto the
            #    accumulated global's trees. base_margin counted ONCE (§3.6).
            if server_round == 1 or not self.current_bst:
                merged_trees = []
            else:
                accumulator = DPBooster.from_json_bytes(self.current_bst)
                self._assert_mergeable(ref, accumulator)
                merged_trees = list(accumulator.trees)
            for b in boosters:
                merged_trees = merged_trees + list(b.trees)

            # 5. meta survives merge->re-serialize (identical across replies by construction, §3.5).
            merged = DPBooster(
                trees=merged_trees,
                base_margin=ref.base_margin,
                edges=ref.edges,
                max_bins=ref.max_bins,
                feature_ranges=ref.feature_ranges,
                meta=boosters[0].meta,
            )

            # 6. Re-serialize -> ArrayRecord([uint8]) at ["0"]; store bytes back so next round's
            #    inherited configure_train re-caches it as current_bst.
            self.current_bst = merged.to_json_bytes()
            arrays = ArrayRecord([np.frombuffer(self.current_bst, dtype=np.uint8)])

            # 7. Aggregate MetricRecords exactly like the parent (a bare ArrayRecord return would
            #    unpack wrong in strategy.start — C-B3).
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays, metrics

    @staticmethod
    def _assert_mergeable(a: DPBooster, b: DPBooster) -> None:
        """Raise if two DPBoosters disagree on any data-independent field (feature_ranges /
        base_margin / max_bins) — a config/node drift that would merge incompatible bin grids."""
        if a.feature_ranges != b.feature_ranges:
            raise ValueError(
                f"DP bagging merge: feature_ranges disagree across replies "
                f"({a.feature_ranges} vs {b.feature_ranges})."
            )
        if a.max_bins != b.max_bins:
            raise ValueError(
                f"DP bagging merge: max_bins disagree ({a.max_bins} vs {b.max_bins})."
            )
        if a.base_margin != b.base_margin:
            raise ValueError(
                f"DP bagging merge: base_margin disagree ({a.base_margin} vs {b.base_margin})."
            )
