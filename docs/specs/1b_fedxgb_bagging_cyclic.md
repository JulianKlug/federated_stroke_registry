# Spec 1.b — `FedXgbBagging` and `FedXgbCyclic` with matched hyperparameters and matched tree budget

Implements roadmap item 1.b ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Implement both `FedXgbBagging` and `FedXgbCyclic` with matched
> hyperparameters (architecture §3) and matched total tree budget.
> Alternate cyclic order across runs to check for last-site bias.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§2 and §3) and the installed framework (`flwr==1.31.0`, message-based API).

## 1. Goal & scope

**In scope**

- Both aggregation strategies selectable from run config, running on the
  existing `architecture/fed_stroke/` app against the two Geneva halves.
- Hyperparameters matched to architecture §3 for both strategies.
- Matched total tree budget: both strategies produce a **40-tree** final
  ensemble, with the round count **derived in code** from a single
  `total-trees` budget so a mismatched run is impossible (§4.6).
- Deterministic, site-driven cyclic order that can be alternated across
  runs, plus server-side round→site logging (§4.3).
- A local evaluation script that scores each run's saved final model on
  **both** halves' validation splits — the only way the R2-vs-R3
  comparison measures last-site bias rather than site difficulty (§4.7).
- Unit tests for all new logic, including a regression test pinning the
  existing bagging behavior of `_local_boost` (§4.8).

**Out of scope** (later roadmap items)

- Site-stratified AUC-ROC / AUC-PR / Brier / confusion-matrix harness → 1.c.
  §4.7's script is a debugging-grade sanity tool on the Geneva halves (local
  data access is allowed for debugging), not the 1.c harness.
- Federated-vs-pooled correctness check → 1.d.
- Choosing a winning strategy — deferred to Phase v1.3 on real cross-site
  data; both strategies are co-equal candidates and stay maintained.

## 2. Dependencies

- 1.a complete: `geneva_half_A.parquet` / `geneva_half_B.parquet` exist and a
  1 SuperLink / 2 SuperNode topology runs locally, each SuperNode configured
  with its own `data-path` (consumed by `load_data_gva` in
  [architecture/fed_stroke/task.py](../../architecture/fed_stroke/task.py)).
- No new runtime dependencies. `FedXgbCyclic` ships with the pinned
  `flwr` (`flwr.serverapp.strategy.FedXgbCyclic`). `pytest` is added as a
  dev-only dependency (§4.8).

## 3. How the two strategies differ in flwr 1.31 (design constraints)

These framework facts drive the design; they were verified against the
installed `flwr` sources.

| | `FedXgbBagging` | `FedXgbCyclic` |
|---|---|---|
| Nodes trained per round | all sampled nodes (both sites) | **exactly one** node: index `(server_round - 1) % n_nodes` |
| Trees added per round | `n_clients × local-epochs` = 2 | `local-epochs` = 1 |
| What the client must return | **only the newly boosted trees** (server merges them into the global ensemble via `aggregate_bagging`) | **the full booster** (server adopts the reply wholesale as the new global model) |
| Node ordering | n/a | registration order (`registered_nodes`), i.e. connection order — **not reproducible across runs** |
| `configure_evaluate` | all nodes | the **same node that just trained** (`_make_sampling` uses the identical `(server_round - 1) % n` index for train and evaluate) |

Three consequences:

1. The client-side tree extraction must switch on the strategy: the current
   `_local_boost` last-N-trees slice is correct for bagging but would throw
   away the whole prior ensemble under cyclic.
2. "Alternate cyclic order across runs" cannot be done by just re-running:
   the order depends on SuperNode connection timing. It needs an explicit,
   config-driven order control (§4.3).
3. Under cyclic, every per-round AUC is a **self-evaluation**: the site that
   just boosted evaluates the resulting model on its own validation split.
   The final-round AUCs of a forward and a reverse run therefore come from
   *different* validation sets with a self-eval bias — comparing them does
   not measure last-site bias. The final models must be scored on both
   halves outside the FL loop (§4.7).

## 4. Design

### 4.1 Run-config keys

Add to `[tool.flwr.app.config]` in
[architecture/pyproject.toml](../../architecture/pyproject.toml):

```toml
train-method = "bagging"   # "bagging" | "cyclic"
cyclic-order = "forward"   # "forward" | "reverse"; only read when train-method = "cyclic"
total-trees = 40           # final ensemble size; num-server-rounds is DERIVED from this (§4.6)
num-sites = 2              # participating SuperNodes; used for round derivation and min_available_nodes
```

`num-server-rounds` is **removed** from the config: it was the one knob that
could silently break the matched budget (a cyclic run at the bagging default
would end at half the trees with no error). The server derives it (§4.2).

Naming mirrors Flower's comprehensive XGBoost example, which uses the same
`train-method` switch for the same purpose.

### 4.2 Server: round derivation + strategy selection (`server_app.py`)

Two small, unit-testable functions, called from `main()`:

```python
def derive_num_rounds(
    train_method: str, total_trees: int, num_sites: int, local_epochs: int
) -> int:
    """Compute the round count that yields exactly `total_trees` trees."""
    if train_method == "bagging":
        trees_per_round = num_sites * local_epochs
    elif train_method == "cyclic":
        trees_per_round = local_epochs
    else:
        raise ValueError(f"Unknown train-method: {train_method}")
    if total_trees % trees_per_round:
        raise ValueError(
            f"total-trees={total_trees} is not divisible by "
            f"{trees_per_round} trees/round ({train_method})"
        )
    return total_trees // trees_per_round


def build_strategy(run_config) -> FedXgbBagging | OrderedFedXgbCyclic:
    train_method = run_config["train-method"]
    # fraction values pass through from config for BOTH strategies: the
    # config must never silently lie. flwr's own FedXgbCyclic constructor
    # raises on any value other than 0.0/1.0 (1.0 is the only useful one).
    if train_method == "bagging":
        return FedXgbBagging(
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
        )
    if train_method == "cyclic":
        return OrderedFedXgbCyclic(
            order=run_config["cyclic-order"],
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
        )
    raise ValueError(f"Unknown train-method: {train_method}")
```

`main()` logs the derivation once at startup
(`train-method=cyclic: total-trees=40 → num-server-rounds=40`), then calls
`strategy.start(..., num_rounds=derived)`. Everything else in `main()`
(empty initial `ArrayRecord`, optional `save-model`) is strategy-agnostic
and stays as is.

### 4.3 Cyclic order control: `OrderedFedXgbCyclic`

New module `architecture/fed_stroke/strategies.py`.

flwr's `FedXgbCyclic` cycles nodes in registration (connection) order, which
varies between runs. Sorting by raw node ID would be deterministic *within*
a run, but node IDs are random per SuperNode process — if a SuperNode
restarts between the forward and the reverse run, the ID sort can map to
sites differently and both runs could finish on the **same** site. The order
must therefore be pinned to **site identity**, which only the client knows
(its `node_config["data-path"]`). Mechanism: a one-time `QUERY` round-trip.

```python
class OrderedFedXgbCyclic(FedXgbCyclic):
    """FedXgbCyclic with a deterministic, site-name-driven node order.

    On first configure, queries every connected node for its site name
    (the client answers with its data file name, §4.4) and caches a
    node_id → site map. Cycling order is the lexicographic sort of site
    names ("forward") or its exact reversal ("reverse") — stable across
    SuperNode restarts and reruns, as required by roadmap 1.b's
    last-site-bias check.
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
            Message(content=RecordDict(), message_type=MessageType.QUERY,
                    dst_node_id=nid)
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
        for msg in messages:  # authoritative round→site evidence in server log
            log(INFO, "Round %s: training site %s",
                server_round, self._node_sites[msg.metadata.dst_node_id])
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
```

Notes:

- `_make_sampling` calls `_reorder_nodes` and then picks index
  `(server_round - 1) % n`, so overriding `_reorder_nodes` remains the
  single, narrow cycling hook; `configure_train`/`configure_evaluate` are
  wrapped only to populate the site map (they delegate to `super()`).
- The reply carries the site name in a `ConfigRecord` because
  `MetricRecord` values are numeric-only in flwr.
- The server log now carries the authoritative round→site mapping
  (`Round N: training site geneva_half_X.parquet`); the `load_data_gva`
  print on the client side remains as corroborating evidence.
- Defense-in-depth: even though site-name ordering makes SuperNode restarts
  between runs harmless, still verify from the logs that R2 and R3 trained
  **opposite** sites in their final round (acceptance §7); if they somehow
  did not, re-run the second one with the opposite `cyclic-order`.

### 4.4 Client: site query + strategy-dependent model return (`client_app.py`)

**Site query handler** (new) — answers §4.3's one-time query:

```python
@app.query()
def site_info(msg: Message, context: Context) -> Message:
    site = Path(context.node_config["data-path"]).name
    return Message(
        content=RecordDict({"config": ConfigRecord({"site": site})}),
        reply_to=msg,
    )
```

**Training reply** — in `train()`, read `train-method` from run config and
branch after local boosting:

- `bagging` — unchanged: `_local_boost` slices the last `num_local_round`
  trees; the server merges them into the global ensemble.
- `cyclic` — return the **full** updated booster (no slice):
  `FedXgbCyclic.aggregate_train` adopts the client's reply as the new global
  model, so the reply must carry the entire ensemble built so far.

Concretely, `_local_boost` gains a `train_method` parameter:

```python
def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    for _ in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    if train_method == "bagging":
        # extract only the newly added trees for server-side merging
        return bst_input[
            bst_input.num_boosted_rounds() - num_local_round
            : bst_input.num_boosted_rounds()
        ]
    return bst_input  # cyclic: full model
```

The `global_round == 1` bootstrap (train from scratch) needs no change: under
cyclic, the second site first trains in round 2, where it correctly loads the
round-1 global model and boosts on top.

`evaluate()` is strategy-agnostic and stays unchanged.

### 4.5 Matched hyperparameters (architecture §3)

Update `[tool.flwr.app.config]` in `architecture/pyproject.toml` to the §3
values. Both strategies read the identical `params.*` block — that is what
"matched hyperparameters" means operationally.

| Key | Current | §3 value |
|---|---|---|
| `num-server-rounds` | 3 | **removed** — derived from `total-trees = 40` (§4.2, §4.6) |
| `local-epochs` | 1 | 1 (unchanged) |
| `params.eta` | 0.1 | 0.1 (unchanged) |
| `params.max-depth` | 8 | **4** |
| `params.min-child-weight` | — (absent) | **5** |
| `params.subsample` | 1 | **0.8** |
| `params.colsample-bytree` | — (absent) | **0.8** |
| `params.tree-method` | `hist` | `hist` (unchanged) |
| `params.num-parallel-tree` | 1 | 1 (unchanged) |
| `fraction-train` / `fraction-evaluate` | 1.0 | 1.0 (unchanged) |

§3 marks these as starting points; HPO happens in Phase v1.1/v1.2. 1.b only
needs both strategies to run at the *same* values.

### 4.6 Matched tree budget

Target ensemble: **40 trees** (§3: `num_rounds × n_clients × local_epochs` =
`20 × 2 × 1`), carried by `total-trees = 40` in config. The server derives
the round count per strategy (§4.2):

- Bagging: `num_sites × local-epochs` = 2 trees/round → 20 derived rounds.
- Cyclic: `local-epochs` = 1 tree/round → 40 derived rounds.

Because derivation is in code, the run commands need no rounds override —
the budget cannot be mismatched by a forgotten flag:

```bash
# bagging (defaults)
flwr run . <federation> --run-config "save-model=true"
# cyclic, forward order
flwr run . <federation> --run-config "train-method='cyclic' save-model=true"
# cyclic, reverse order
flwr run . <federation> --run-config "train-method='cyclic' cyclic-order='reverse' save-model=true"
```

### 4.7 Final-model evaluation script

New `architecture/scripts/eval_final_model.py` (debugging-grade; local Geneva
data access is allowed for debugging per project constraints). Rationale in
§3 consequence 3: in-run cyclic AUC is self-evaluation on alternating sites,
so R1/R2/R3 can only be compared by scoring each **saved** final model on the
**same two** validation sets.

```bash
python scripts/eval_final_model.py final_model.json \
    --data geneva_half_A.parquet geneva_half_B.parquet \
    --expected-trees 40
```

Behavior:

1. Load the model; **assert `num_boosted_rounds() == --expected-trees`**
   (automates acceptance check 2 and backstops the budget derivation
   against e.g. a failed round producing a short ensemble).
2. For each data file: rebuild the *identical* validation split used in-run
   (`generate_splits`, `test_size=0.2`, `seed=42`) and report AUC.
3. Print one table row per (model, site) — these numbers go in the logbook.

To keep the split/feature definitions in one place (DRY), `task.py` exposes
`FEATURE_COLS` / `TARGET_COL` as module constants and the script imports
them plus `generate_splits` — no duplicated column lists.

### 4.8 Tests

New `architecture/tests/` (pytest, added to a `[dependency-groups] dev`
group; no runtime dependency). E2E behavior is covered by the run matrix
(§6); everything below is fast pure-logic testing.

| File | Cases |
|---|---|
| `tests/test_strategies.py` | `order` validation (forward/reverse ok, anything else → `ValueError`); `_reorder_nodes` forward = lexicographic by site, reverse = exact reversal of forward on the same inputs; unmapped node id → `KeyError`; `_ensure_site_map` builds the map from stubbed query replies, caches (no second query), raises on an error reply, and raises when a queried node returns **no** reply (stub `send_and_receive` yields fewer replies than messages sent). Stub `Grid` = small fake with `get_node_ids` / `send_and_receive`. |
| `tests/test_server_config.py` | `derive_num_rounds`: bagging 40/2/1 → 20; cyclic 40/2/1 → 40; indivisible (e.g. 39 bagging) → `ValueError`; unknown method → `ValueError`. `build_strategy`: bagging → `FedXgbBagging`; cyclic → `OrderedFedXgbCyclic` with the configured order; unknown → `ValueError`. |
| `tests/test_client_boost.py` | **CRITICAL — regression** (this change touches working bagging code): on a tiny synthetic `DMatrix`, `_local_boost(..., "bagging")` returns a booster with exactly `num_local_round` trees. Cyclic branch: returns the *same* booster object with `num_boosted_rounds` grown by `num_local_round`. |

## 5. Files to change

| File | Change |
|---|---|
| `architecture/fed_stroke/strategies.py` | **new** — `OrderedFedXgbCyclic` with site-map ordering (§4.3) |
| `architecture/fed_stroke/server_app.py` | `derive_num_rounds` + `build_strategy`, startup derivation log (§4.2) |
| `architecture/fed_stroke/client_app.py` | `@app.query` site handler; `train_method` branch in `_local_boost` (§4.4) |
| `architecture/fed_stroke/task.py` | hoist `FEATURE_COLS` / `TARGET_COL` to module constants (§4.7) |
| `architecture/scripts/eval_final_model.py` | **new** — per-site AUC + tree-count assert on saved models (§4.7) |
| `architecture/tests/` | **new** — `test_strategies.py`, `test_server_config.py`, `test_client_boost.py` (§4.8) |
| `architecture/pyproject.toml` | §3 param values; `train-method` / `cyclic-order` / `total-trees` / `num-sites` keys; drop `num-server-rounds`; `pytest` dev group (§4.1, §4.5) |

## 6. Run matrix

All runs on the 1.a topology (2 SuperNodes, one Geneva half each), identical
`params.*`, `total-trees = 40`:

| Run | `train-method` | Derived rounds | `cyclic-order` | Final trees |
|---|---|---|---|---|
| R1 | bagging | 20 | — | 40 |
| R2 | cyclic | 40 | forward | 40 |
| R3 | cyclic | 40 | reverse | 40 |

## 7. Acceptance criteria

1. `pytest` suite (§4.8) passes, including the bagging regression test.
2. All three runs complete end-to-end with no errors; server startup log
   shows the correct derived round count (20 / 40 / 40).
3. Matched budget verified: `eval_final_model.py --expected-trees 40`
   passes for each run's saved `final_model.json`.
4. Cyclic schedule verified from the server's `Round N: training site …`
   log: exactly one SuperNode trains per round, sites strictly alternate,
   and R2/R3 finish on **opposite** sites. (If not — should be impossible
   with site-name ordering — re-run with the opposite `cyclic-order`.)
5. Bagging schedule verified from logs: both SuperNodes train every round.
6. Per-site AUC of each run's final model on **both** halves (from
   `eval_final_model.py`) recorded in [docs/logbook.md](../logbook.md),
   together with which site trained last in R2 and R3.
7. Sanity, not a gate: all six per-site AUCs in a plausible band relative
   to the 1.a single-node baseline (formal correctness bound is 1.d's
   ±3 AUC-point check against pooled).

## 8. Risks & notes

- **In-run cyclic AUC is self-evaluation** (§3 consequence 3): the per-round
  series alternates sites *and* each point is the just-trained site scoring
  itself — read it as a training-progress trace only. All cross-run
  comparisons use §4.7's both-halves evaluation. The 1.c harness later
  replaces this with proper site-stratified evaluation.
- **Last-site bias**: 1.b delivers the forward/reverse run pair, the
  round→site server log, and per-site final-model AUCs on both halves.
  Quantifying bias formally (per-site metric deltas between R2 and R3 with
  1.c's full metric set) stays in 1.c; if the §4.7 per-site AUCs of R2 and
  R3 already diverge grossly on the *same* validation half, flag it in the
  logbook.
- **Site query is a hard gate**: a node that returns an error reply *or*
  never answers within `SITE_QUERY_TIMEOUT_S` aborts the cyclic run
  (`RuntimeError`) rather than falling back to node-ID order — a silent
  fallback could invalidate the run pair, and a no-timeout wait would hang
  the run with no diagnostic (the exact failure mode when a node lacks the
  `@app.query` handler, e.g. version skew with Shenzhen later). Bagging runs
  never send the query.
- **`min_available_nodes`**: driven by `num-sites` (=2) for both strategies;
  `FedXgbCyclic` additionally hard-requires `min_train_nodes = 2` in flwr
  1.31, matching our topology; a run started with one SuperNode down will
  wait rather than silently train single-site.
- **`fraction-train`/`fraction-evaluate` pass through from config** for both
  strategies; flwr's `FedXgbCyclic` constructor rejects anything other than
  0.0/1.0 (and 0.0 disables the phase, so 1.0 is the only useful value).
  The config never silently lies about what ran.
- **Serialization size under cyclic** grows with the ensemble (full model
  shipped every round, up to 40 shallow trees) — negligible at
  `max_depth = 4` and N=2; noted only because DP work in v1.1+ touches
  what crosses the wire.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found (claude) | 1 finding (query timeout/hang), fixed |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 6 issues, 0 critical gaps, all resolved |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE VOICE:** codex CLI failed (401 auth) and the Claude subagent hit the org spend limit; the independent adversarial pass was run in-loop instead, verifying the Issue-1B QUERY mechanism against installed flwr sources. It confirmed feasibility and surfaced one new finding (silent-hang on unanswered site query), now fixed as Issue 7.
- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement.
