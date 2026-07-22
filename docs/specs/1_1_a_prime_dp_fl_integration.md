# Spec 1.1.a′ — DP → FL integration (wire the DP seam into the live federation)

Implements roadmap item 1.1.a′ ([architecture/roadmap.md](../../architecture/roadmap.md)):

> 1.1.a′ DP → FL integration. Wire the single-site DP seam (`fed_stroke/dp/`: `DPConfig`,
> `HistogramNoiseMechanism`, `train_dp_gbdt`, the RDP accountant) into the real federated path:
> client-side Laplace/Gaussian noise on the gradient AND hessian histograms + geometric leaf
> clipping (architecture §6.3), a DP-aware aggregator, and `DPBooster` serialization across
> rounds, with the accountant composing the `2·D·T` releases per run (spec `1_1_prereq` §3.3).
> `dp.enabled = true` forces `subsample = 1.0` for honest q = 1.0 accounting. The `dp.*` config
> keys and the 1.1.a harness already thread these via `--run-config`, so this makes those knobs
> bite and turns 1.1.b into "the 1.1.a sweep + DP arms" with no new orchestration. Built and
> validated on the loopback `local-deployment` topology (synthetic/example data) — it does NOT
> run on real patient data (that is gated in 1.1.b), so no real-ε claim is made here. Closes the
> downstream integration item flagged in docs/specs/1_1_prereq_dp_plugpoint.md §4.5. Prerequisite
> for 1.1.b.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§6.3) and the installed framework (`flwr==1.31.0` message-based API, `xgboost>=2.0`, Python
`>=3.12`, `uv`-managed). This spec closes the FL-integration item the DP prototype
([1_1_prereq_dp_plugpoint.md](1_1_prereq_dp_plugpoint.md) §4.5) designed but did not build. It
reuses the trained DP learner + accountant that prototype shipped
([`fed_stroke/dp/`](../../architecture/fed_stroke/dp/)), the per-round client seam built by 1.b
([client_app.py](../../architecture/fed_stroke/client_app.py)), the strategies from 1.b
([strategies.py](../../architecture/fed_stroke/strategies.py)), the single scorer from 1.c
([metrics.py](../../architecture/fed_stroke/metrics.py)), and the split contract from 1.1.a
([task.py](../../architecture/fed_stroke/task.py) `resolve_run_split`).

**This task ships the spec only.** The roadmap 1.1.a′ checkbox stays `[ ]` until the code lands
in a follow-up — mirroring how 1.e/1.f and the DP prototype shipped their spec ahead of
implementation.

**Framing (read before §1).** DP is a **measurement instrument**, not a shipping commitment
(DP prototype spec, "Framing"). 1.1.a′ is a pure **wiring** task: the mechanism, the accountant,
and the Opacus-validated ε math already exist and are validated single-site; this task connects
them to the live 2-SuperNode federation so the same `2·D·T` accounting that the prototype proved
on synthetic data now runs across a federated run. It deliberately runs on **synthetic/example
data only** — no ε is claimed about real patients here. The real-Geneva DP sweep and the first
real-ε numbers are **1.1.b**, gated behind the independent DP-accountant review
([dp_accountant_review_packet.md](../reviews/dp_accountant_review_packet.md)). 1.1.a′ makes the
`dp.*` knobs bite so that 1.1.b is "the 1.1.a sweep + DP arms" with no new orchestration.

## 1. Goal & scope

**In scope**

- **`DPBooster` serialization** (`dp/boost.py`): `to_json_bytes` / `from_json_bytes`, so a DP
  ensemble crosses the Flower transport as bytes in an `ArrayRecord["0"]` exactly like an
  XGBoost `save_raw("json")` model — the transport plumbing is unchanged (§4.1).
- **A continuation learner** (`dp/boost.py`): `dp_local_boost`, which resumes boosting from an
  incoming global `DPBooster`'s margins and grows only this round's trees — the DP analog of
  `client_app._local_boost` (§4.2). Built by extracting the prototype's boost loop into a shared
  `_grow_trees` helper so `train_dp_gbdt` stays byte-identical (§4.2).
- **Run-level DP accounting** (`dp/boost.py` + `client_app.py`): the noise multiplier σ is
  calibrated **once per run** over the whole run's `2·D·T_site` releases, where `T_site` is the
  **busiest** site's per-run tree count; each stateless round reconstructs the identical σ and
  draws **independent per-round, per-site** noise (§4.3).
- **Client DP branch** (`client_app.py`): `@train` and `@evaluate` gain a `dp.enabled` branch
  that swaps the whole learner (train), and deserializes + predicts a `DPBooster` (evaluate),
  reading raw numpy `X, y` via a new `task.load_data_arrays` (§4.4, §4.5).
- **A DP-aware bagging aggregator** (`strategies.py`): `DPFedXgbBagging`, concatenating
  `DPBooster` trees round-over-round (the DP analog of flwr's `aggregate_bagging`); **cyclic
  reuses `OrderedFedXgbCyclic` unchanged** (§4.6).
- **Server DP branch** (`server_app.py`): DP-aware strategy selection in `build_strategy`, and a
  DP branch in `main()`/`save_final_model` that reconstructs + persists a `DPBooster` and logs
  the reported ε, σ, and release count (§4.7).
- **`subsample = 1.0` enforcement in DP mode** (`client_app.py`): honest q = 1.0 accounting
  (§4.4, §3.5).
- **Minimal offline DP scoring** (user-chosen scope, §4.8): `baseline.score_booster_on_half` and
  `scripts/eval_final_model.py` auto-detect the DP model format and score a `DPBooster` on raw
  X — the objective path 1.1.b consumes, and what makes the §6 verification re-scorable.
- **Tests** pinning the accounting release count, cyclic uneven-participation, cross-round /
  cross-site noise independence, serialization round-trip, bagging-merge equivalence, and the
  **non-DP byte-identical regression** (§4.9).

**Out of scope** (named + owned by a later item — never silently dropped)

- **The ε ∈ {1, 3, 5, 10}, δ = 1e-5 sweep and the DP-vs-classic HPO on real Geneva → 1.1.b.**
  1.1.a′ makes the knobs bite and validates on synthetic/example data; it runs **no** real-patient
  data and makes **no** real-ε claim. 1.1.b is gated behind the independent DP review
  ([dp_accountant_review_packet.md](../reviews/dp_accountant_review_packet.md)).
- **`check_fed_vs_pooled.py` DP support** (a federated-DP vs pooled-DP correctness gate). The DP
  learner's noised `min_child_weight` gate makes 1.d's ±0.03 tolerance inapplicable, so a DP
  gate needs its own tolerance design → later item (§8).
- **Private-quantile bin edges on real data** and **subsampled-RDP amplification credit** — both
  already downstream-owned by the prototype spec ([1_1_prereq_dp_plugpoint.md](1_1_prereq_dp_plugpoint.md)
  §8.1/§8.2). 1.1.a′ keeps `bin_strategy="fixed_range"` and q = 1.0.
- **HPO orchestration changes.** `run_hpo.py` already threads arbitrary `--run-config` overrides
  (1.1.a §4.4), so `dp.*` keys flow through unchanged; adding DP *arms* to the grid is 1.1.b.
- **Any change to the accountant math** (`dp/accounting.py`). It is Opacus-validated and frozen;
  this task only *feeds* it the correct release count.

## 2. Dependencies

- **DP prototype complete** (it is — [docs/logbook.md](../logbook.md) 2026-07-20): the
  `fed_stroke/dp/` package (`boost.py`, `accounting.py`, `synthetic.py`, `__init__.py`) exists,
  the Opacus gate passes, and the seam symbols are exported from `fed_stroke.dp`
  (`DPConfig`, `BoostParams`, `DPBooster`, `HistogramNoiseMechanism`, `make_mechanism`,
  `train_dp_gbdt`, `num_histogram_queries`, `num_gaussian_releases`, `FEATURE_RANGES`).
- **1.b complete**: both strategies selectable via
  [`build_strategy`](../../architecture/fed_stroke/server_app.py) and the per-round client seam
  (`_train_round`/`_local_boost`/`round_seed`,
  [client_app.py:29-80](../../architecture/fed_stroke/client_app.py)).
- **1.c complete**: the shared scorer
  [`compute_binary_metrics`](../../architecture/fed_stroke/metrics.py) and
  [`score_booster_on_half`](../../architecture/fed_stroke/baseline.py).
- **1.e complete**: the loopback 2-SuperNode `local-deployment` federation stood up by
  [`scripts/run_local_federation.sh`](../../architecture/scripts/run_local_federation.sh) — the
  substrate the §6 verification runs on.
- **1.1.a complete**: the split contract `task.resolve_run_split` and the config keys
  `split-seed`/`holdout-frac`/`holdout-eval`.
- **`dp.*` config keys present** in `pyproject.toml` `[tool.flwr.app.config]` (from the DP
  prototype). **No new config keys are added by this task.**
- **No new runtime dependencies.** `numpy`/`scipy` are already explicit runtime deps (DP
  prototype); the FL wiring uses only the standard library + `flwr` + the existing DP package.
  **opacus stays dev-only** — the runtime must stay torch-free (§7.6).

## 3. Framework / DP facts that drive the design

Verified against the installed `flwr==1.31.0` sources and the current `fed_stroke` package. Each
is load-bearing; a wrong constant here silently corrupts the reported ε or breaks the transport.

- **3.1 The DP learner is a NumPy learner over raw `X, y`, not an XGBoost booster.**
  `train_dp_gbdt(X, y, boost, dp, ...)` ([dp/boost.py:322](../../architecture/fed_stroke/dp/boost.py))
  consumes numpy arrays and returns a `DPBooster` whose state is
  `(trees: list[nested dict], base_margin: float, edges: list[np.ndarray], max_bins: int)`
  ([dp/boost.py:287-320](../../architecture/fed_stroke/dp/boost.py)). It has **no
  serialization** and **only** `predict_margin`/`predict`. The live path today serializes with
  `bst.save_raw("json")` and reloads with `xgb.Booster().load_model`
  ([client_app.py:114-120, 78-79](../../architecture/fed_stroke/client_app.py)) — neither works
  on a `DPBooster`. ⇒ DP mode **swaps the whole learner** (prototype §3.1/§4.5) and needs its
  own serializer (§4.1) and its own load path everywhere the XGB one is used.

- **3.2 `train_dp_gbdt` has no continuation mode.** Its boost loop
  ([dp/boost.py:391-402](../../architecture/fed_stroke/dp/boost.py)) hardcodes
  `margin = np.full(n, base_margin)` and loops `boost.num_boost_round` times. Federated boosting
  requires resuming from the *incoming global model's* margins and growing only this round's
  trees (the XGB path does this via `xgb.Booster().load_model` + `bst.update`,
  [client_app.py:78-80, 29-42](../../architecture/fed_stroke/client_app.py)). ⇒ Extract the boost
  loop into a shared `_grow_trees(..., init_margin, num_rounds)` and add `dp_local_boost` (§4.2).
  **DP sensitivity is invariant to the starting margin** — `g = p − y ∈ [−1,1]`,
  `h = p(1−p) ∈ (0,0.25]` regardless of `p` (prototype §3.2) — so continuation does not perturb
  noise calibration.

- **3.3 DP accounting is per-run and per-site: `num_gaussian_releases = 2 · D · T_site`.**
  Composition is per tree *level* (`D` levels/tree), two Gaussian releases per level (gradient
  AND hessian), across all `T_site` trees a single site grows in the whole run (prototype §3.3).
  `T_site` is the number of trees the *busiest* site grows across all rounds — the per-patient
  guarantee is calibrated to the site that releases the most (§3.4). σ must therefore be
  calibrated **once per run** to `n_rel = 2 · D · T_site`, not per round. `make_mechanism`
  already computes `n_rel = num_gaussian_releases(boost) = 2 · max_depth · boost.num_boost_round`
  ([dp/boost.py:114-122, 201](../../architecture/fed_stroke/dp/boost.py)) — so the BoostParams fed
  to `make_mechanism` must carry `num_boost_round = T_site` (§4.3).

- **3.4 `T_site` (busiest site) differs by strategy — and `total_trees // num_sites`
  under-reports ε for cyclic.** From
  [`derive_num_rounds`](../../architecture/fed_stroke/server_app.py) and the two strategies:
  - **bagging**: every site trains every round, growing `local_epochs` trees/round ⇒
    `T_site = num_rounds · local_epochs = total_trees // num_sites` (exact).
  - **cyclic**: exactly one site trains per round; a site trains in `⌈num_rounds / num_sites⌉`
    rounds worst-case ⇒ `T_site = ⌈num_rounds / num_sites⌉ · local_epochs`.

  With `total-trees=41, local-epochs=1, num-sites=2`: `derive_num_rounds` returns 41 (41 % 1 = 0,
  no guard trips), site A grows **21** trees, site B **20**. Using `total_trees // num_sites = 20`
  would calibrate σ for `2·D·20` releases while site A actually emits `2·D·21` — **site A's true
  ε exceeds the reported ε**. ⇒ Never use `total_trees // num_sites` for cyclic; always
  `⌈num_rounds / num_sites⌉ · local_epochs`. Pinned by a helper `per_site_tree_budget` and a
  dedicated test (§4.9 case 2, the highest-value new test).

- **3.5 σ is a pure function of config, so stateless rounds reconstruct it identically; but noise
  must be independent per round AND per site.** `noise_multiplier_for_epsilon`
  ([dp/accounting.py:194](../../architecture/fed_stroke/dp/accounting.py)) is deterministic
  bisection with no RNG — every round (a fresh process) recomputes the same σ from the same
  config. Composition models the per-release Gaussians as **independent**, so the client RNG must
  vary per round and per site: seed from the public `(base_seed, global_round, site)` only —
  never from data. If two sites in the same round shared a seed they would draw identical noise
  vectors (each marginal is still `N(0,σ²)`, so per-site privacy holds, but the correlation is
  undesirable and collapses the sampling diversity the `round_seed` fix,
  [client_app.py:45-60](../../architecture/fed_stroke/client_app.py), exists to preserve). ⇒
  `rng = np.random.default_rng(np.random.SeedSequence([base_seed, global_round, site_hash]))`,
  `site_hash` derived from the public site name (§4.3/§4.4).

- **3.6 Bagging = concatenate DP trees; the merge is simpler than XGB's.** In XGB bagging the
  server holds an accumulated `current_bst` and appends each round's new trees from all clients,
  rewriting `num_trees`/`iteration_indptr`/tree ids in the XGBoost JSON schema
  (`aggregate_bagging`, flwr `strategy_utils`). For DP the trees are **parallel additive
  corrections on the same shared starting margin** that round, so the merge is a plain
  concatenation of `.trees` lists with a **single** shared `base_margin` (counted once); leaf
  weights already carry η ([dp/boost.py:250-254](../../architecture/fed_stroke/dp/boost.py)), so
  concatenation sums η-scaled corrections exactly as XGB does. No id/indptr bookkeeping. The
  `edges`/`base_margin`/`max_bins` are **data-independent** (`fixed_bin_edges(FEATURE_RANGES,
  max_bins)` and `logit(base_score)`), so they are identical across sites and can be asserted
  equal on merge (§4.6). **Cyclic** adopts `reply[0]` bytes wholesale (`FedXgbCyclic`), so it is
  format-agnostic and needs **no** server change beyond the DP bytes round-tripping (§4.6).

- **3.7 The initial global model is `b""` (empty bytes).** `server_app.main` seeds
  `ArrayRecord([np.frombuffer(b"", dtype=np.uint8)])`
  ([server_app.py:166-169](../../architecture/fed_stroke/server_app.py)) and the client maps
  round 1 to `None` ([client_app.py:106-108](../../architecture/fed_stroke/client_app.py)). ⇒ The
  DP round-1 branch must treat "no global model" as fresh training (init from `base_margin`), and
  the DP bagging aggregator must treat an empty accumulator as "adopt this round's replies"
  (mirroring `aggregate_bagging`'s `if bst_prev == b""` sentinel). Round-1 detection stays on
  `server-round == 1`, never on byte-sniffing (§4.4/§4.6).

- **3.8 Config reaches code as `cfg["dp"]` and `cfg["params"]` after
  `replace_keys(unflatten_dict(...))`.** `DPConfig.from_run_config(cfg)`
  ([dp/boost.py:62](../../architecture/fed_stroke/dp/boost.py)) parses `cfg["dp"]`; the DP knobs
  live in a sibling `dp.*` table, never under `params.*` (prototype §3.6). `cfg` is already built
  at [client_app.py:101](../../architecture/fed_stroke/client_app.py) and
  [server_app.py:161](../../architecture/fed_stroke/server_app.py), so the DP branch reads
  `DPConfig.from_run_config(cfg)` right there. **No new config keys.**

- **3.9 The feature count `d` drives the noise scale.** `make_mechanism` sets
  `std_g = σ·√d·G_L2`, `std_h = σ·√d·H_L2` with `d = num_features`
  ([dp/boost.py:201-226](../../architecture/fed_stroke/dp/boost.py)), and `_binize` needs one
  edge array per feature. `FEATURE_RANGES` has exactly the two frozen `FEATURE_COLS`
  ([schema.py](../../architecture/fed_stroke/schema.py) == [dp/boost.py:46]). ⇒ The raw-array
  loader must return X in **exactly** the `FEATURE_RANGES` columns/order, and the client asserts
  `X.shape[1] == len(FEATURE_RANGES)` so a future feature-set growth fails loudly rather than
  mis-scaling `√d` noise (§4.4).

- **3.10 `subsample` is a no-op in the DP learner.** `BoostParams.from_xgb_params` does not read
  `subsample` and `train_dp_gbdt` never samples rows. Forcing `subsample = 1.0` in DP mode is
  therefore **documentary** — it makes the honest q = 1.0 accounting (prototype §3.5) explicit
  and guards against a stray `params.subsample=0.8` being read as if it applied. It must not
  mutate the shared `params` dict the non-DP arm reuses (§4.4).

## 4. Design

All DP behavior is gated on `dp.enabled` (a config value), never on sniffing model bytes. When
`dp.enabled = false` every code path below is the current code verbatim — the non-DP run is
byte-identical (§4.9 regression, §7.7).

### 4.1 `dp/boost.py` — `DPBooster` serialization + first-class `meta`

**`meta` is a first-class `DPBooster` attribute** (Decision 11), not a value derived on the fly
from `.mechanism`. This is load-bearing: the ε/σ/release-count provenance must survive the full
chain `dp_local_boost → to_json_bytes → wire → from_json_bytes → aggregator merge → re-serialize
→ server rebuild → log/save`, and after `from_json_bytes` there is **no mechanism** to derive it
from. So `DPBooster.__init__` gains `meta: dict | None = None`; every producer sets it and every
serializer round-trips it.

```python
DP_MODEL_FORMAT = "dp-gbdt-v1"      # module constant; the auto-detect marker (§4.8)

def _json_finite(obj):
    """Recursive pre-pass: replace every non-finite float (inf/nan) with None BEFORE
    json.dumps. REQUIRED — json.dumps(allow_nan=False) RAISES on a native float inf/nan,
    and `default=` is only invoked for types the encoder does NOT recognize, so it never
    fires for a recognized float. So inf/nan → null cannot be done by default= alone; it
    must be scrubbed here first. Also casts np.integer/np.floating -> int/float so a future
    np.int64 bin cannot break transport. Applied to `meta` (which holds inf ε / nan σ) and,
    defensively, to base_margin."""

class DPBooster:
    def __init__(self, trees, base_margin, edges, max_bins,
                 feature_ranges: dict | None = None, meta: dict | None = None):
        ...                       # trees / base_margin / edges / max_bins unchanged
        self.feature_ranges = feature_ranges   # NEW: {name:(lo,hi)} the edges were built from
        self.meta = meta                        # NEW: accounting provenance (None for a bare ensemble)
        # Both DEFAULT to None so the transient per-round margin booster
        # (`DPBooster([tree], 0.0, edges, max_bins)` inside _grow_trees, never serialized)
        # keeps its current 4-arg construction unchanged. to_json_bytes RAISES if
        # feature_ranges is None — a booster meant to cross the wire must carry it.

    def to_json_bytes(self) -> bytes:
        """Serialize to JSON bytes (mirrors xgb Booster.save_raw('json')).
        Payload: {"format": DP_MODEL_FORMAT, "trees": self.trees,
                  "base_margin": float(self.base_margin),
                  "feature_ranges": self.feature_ranges, "max_bins": int(self.max_bins),
                  "meta": self.meta or {}}.
        Emits self.meta verbatim (NOT re-derived from .mechanism — the merged/reloaded booster
        has no mechanism). The FULL payload passes through _json_finite() FIRST (inf/nan → null,
        numpy scalars → py), THEN json.dumps(..., allow_nan=False) as a belt-and-suspenders
        assert that nothing non-finite slipped through (matches server_app.py:195). Edges are
        NOT stored — self.feature_ranges + max_bins reconstruct them on load via the SAME
        fixed_bin_edges the writer used (avoids float drift at bin boundaries, guarantees
        byte-identical edges across sites/rounds)."""

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "DPBooster":
        """Inverse: parse JSON, reconstruct edges via fixed_bin_edges(payload["feature_ranges"],
        max_bins), rebuild DPBooster(..., feature_ranges=payload["feature_ranges"],
        meta=payload["meta"]). Raises ValueError if data is empty or the format marker is
        absent/unknown (a wrong-format model must fail loudly)."""
```

**`feature_ranges` is a first-class field, not the module global.** `DPBooster` never stored the
`{name:(lo,hi)}` dict, and it is **unrecoverable from `self.edges`** (numpy arrays — the names are
gone). Reaching for the module `FEATURE_RANGES` at serialize time would silently drift edges for
any booster trained via `train_dp_gbdt(..., feature_ranges=custom)`
([boost.py:323](../../architecture/fed_stroke/dp/boost.py)). So `feature_ranges` is a constructor
argument: `train_dp_gbdt` and `dp_local_boost` pass the exact dict they built `edges` from, and
`from_json_bytes` restores it. This is what actually guarantees the §4.9-case-5 invariant
("reconstructed edges == writer's edges"), which nothing in the prior draft enforced.

- `feature_ranges` is serialized (not the numpy `edges`) so the reader reconstructs edges with
  the same `fixed_bin_edges` the writer used. `self.edges` reconstructed on load must equal the
  writer's edges (tripwire, §4.9 case 5).
- `meta` carries accounting provenance for `save_final_model` and the logbook:
  `{"train_method", "total_trees", "per_site_trees", "num_releases", "noise_multiplier",
  "reported_epsilon", "dp": {enabled, mechanism, target_epsilon, delta, clip_bound, max_bins,
  bin_strategy}, "fed_run_config": {...}}`. Values that are `inf`/`nan` (identity ε = ∞; Laplace
  σ = nan) serialize as `null`.
- **`meta` provenance chain (Decision 11).** `dp_local_boost` builds `meta` from its
  run-calibrated `mechanism` and sets it on the reply booster (§4.2); `to_json_bytes` emits it;
  `from_json_bytes` restores it. The bagging aggregator, after merging trees, sets
  `merged.meta = replies[0].meta` (every reply's meta is identical by construction — σ is a pure
  function of config, §3.5 — but the aggregator **asserts** they agree, §4.6). `train_dp_gbdt`
  still attaches `.mechanism` post-hoc for the prototype/demo, and additionally sets `.meta` so
  the byte-identity wrapper (§4.2) and the offline path see the same field; `predict` never reads
  either.
- **DP round-trip tripwire (tests-only)** — a helper `assert_dp_roundtrip(booster, X)` mirroring
  `baseline.assert_prediction_roundtrip`
  ([baseline.py:188-207](../../architecture/fed_stroke/baseline.py)): `from_json_bytes(
  booster.to_json_bytes())` predicts `np.array_equal` on `X`. **Used by tests only** — the server
  aggregator has no feature data `X`, so it does the structural check in §4.6 step 2 instead
  (feature_ranges/base_margin/max_bins equality across replies), never the predict-based tripwire.

### 4.2 `dp/boost.py` — continuation learner (`_grow_trees` + `dp_local_boost`)

Refactor the prototype's boost loop into a shared core, then add the federated continuation
entry point.

```python
def _grow_trees(binned, y, edges, boost, dp, mechanism, rng,
                init_margin: np.ndarray, num_rounds: int) -> list[dict]:
    """The boost loop lifted verbatim from train_dp_gbdt (dp/boost.py:391-402), but
    starting from init_margin (not a constant base_margin) and running num_rounds.
    Returns the list of new nested-dict trees. The ONLY place noise enters stays
    mechanism.add_noise inside build() (unchanged)."""

def dp_local_boost(global_booster, X, y, boost, dp, mechanism, rng, num_local_round,
                   train_method) -> DPBooster:
    """DP analog of client_app._local_boost. `boost` here is the GROWTH BoostParams
    (num_boost_round == num_local_round). `mechanism` is the RUN-CALIBRATED mechanism
    (its σ was fixed to n_rel = 2·D·T_site, §4.3) — passed in, never rebuilt here.
    `rng` is the per-round, per-site generator the caller seeds from public
    (base_seed, global_round, site) (§3.5/§4.4) — passed in and forwarded verbatim to
    _grow_trees, so the noise is independent across rounds and sites. **Never defaulted
    here**: a missing rng would silently collapse to boost.seed and draw identical noise
    every round (the A1 failure this signature exists to prevent).

    Binning: edges + feature_ranges are data-independent. round 1 uses
    fixed_bin_edges(FEATURE_RANGES, dp.max_bins) and fr = FEATURE_RANGES; continuation reuses
    global_booster.edges and fr = global_booster.feature_ranges (restored by from_json_bytes,
    §4.1) — identical either way (§3.6). It bins X once, then calls _grow_trees(..., rng, ...).
    Every returned DPBooster carries feature_ranges=fr and meta (below).

    round 1 (global_booster is None): init_margin = np.full(len(X), base_margin); grow
        num_local_round trees. Return DPBooster(new_trees, base_margin, edges, max_bins,
        feature_ranges=fr, meta=...) (all new).
    later: init_margin = global_booster.predict_margin(X); grow num_local_round NEW trees.
        bagging -> return DPBooster(new_trees, ..., feature_ranges=fr, meta=...)  # new trees only
        cyclic  -> return DPBooster(global.trees + new_trees, ..., feature_ranges=fr, meta=...)
    Mirrors _local_boost's slice-vs-full return (client_app.py:34-42). Sets .meta on the
    returned booster from `mechanism` (noise_multiplier / reported_epsilon / num_releases
    + the run config block, §4.1) so provenance travels with the reply. `train_dp_gbdt`
    (the thin wrapper) likewise passes feature_ranges=fr into its DPBooster."""
```

- `train_dp_gbdt` is rewritten as a **thin wrapper** over `_grow_trees` (init_margin =
  `logit(base_score)`, num_rounds = `boost.num_boost_round`) — behavior **byte-identical** to
  today, pinned by a regression test (§4.9 case 8) so the prototype/demo/tests are untouched.
- **Do not re-implement** the noised `build`/`_best_split`/`_leaf_weight` inside `dp_local_boost`
  — divergence there is a silent privacy/utility bug. All noise logic stays in the shared core.
- New symbols exported from `fed_stroke.dp.__init__` (§5): `dp_local_boost`,
  `per_site_tree_budget`, `DP_MODEL_FORMAT`, and (already exported) the rest.

### 4.3 `dp/boost.py` — run-level accounting (`per_site_tree_budget` + the two-BoostParams rule)

```python
def per_site_tree_budget(train_method, num_rounds, num_sites, local_epochs) -> int:
    """Trees the BUSIEST site grows across the whole run — the per-patient release count.
    bagging: num_rounds * local_epochs
    cyclic : ceil(num_rounds / num_sites) * local_epochs   # never total_trees//num_sites (§3.4)
    Raises on unknown train_method."""
```

**The two-BoostParams rule (load-bearing).** Two distinct `BoostParams` objects with different
`num_boost_round`:

1. **Accounting params** → `make_mechanism`. `num_boost_round = per_site_tree_budget(...)`, so
   `n_rel = num_gaussian_releases = 2 · D · per_site_budget` and σ is calibrated to the whole
   run. (For `total-trees=40, num-sites=2, max-depth=4, local-epochs=1`: `per_site_budget = 20`
   → `n_rel = 160`.)
2. **Growth params** → `dp_local_boost`. `num_boost_round = local_epochs` (trees grown *this*
   round).

Reusing one object either calibrates σ for `local_epochs` releases (massive under-noise, ε
under-reported ~`num_rounds×`) or grows `per_site_budget` trees per round. Pinned by §4.9 cases
1 & 3.

The client computes `num_rounds` via the same `server_app.derive_num_rounds` (imported, not
duplicated) so the client and server agree, then builds the accounting params and the run
mechanism. Because σ depends only on config, every round reconstructs the identical σ (§3.5).

### 4.4 `client_app.py` — the client DP branch

**A pure `_dp_train_round` mirrors `_train_round` (Decision 12).** `_train_round`
([client_app.py:63-80](../../architecture/fed_stroke/client_app.py)) was deliberately extracted
to be **Context/Message-free** so the regression test drives the exact per-round sequence without
faking Flower objects. The DP path gets the **same treatment**: a pure
`_dp_train_round(...)` that takes only arrays + config primitives + the deserialized global, so
§4.9 case 3 (two-BoostParams separation) drives it directly with no fake `Context`. `@train` only
marshals `Context`/`Message` → arrays, then dispatches on `dp.enabled`.

```python
def _dp_train_round(params, dp, global_round, local_epochs, train_method,
                    total_trees, num_sites, X, y, global_model_bytes, site) -> DPBooster:
    """Pure DP analog of _train_round (no Context/Message). Returns the reply DPBooster.

    global_model_bytes: raw bytes of the current global DPBooster, or None on round 1.
    Round-1 detection is on `global_round == 1` (never byte-sniffing, §3.7). `local_epochs`
    is BOTH the growth tree count and dp_local_boost's num_local_round (they are equal by
    construction) — ONE source, no redundant param."""
    assert X.shape[1] == len(FEATURE_RANGES)                 # (§3.9) fail loud on schema drift
    num_rounds = derive_num_rounds(train_method, total_trees, num_sites, local_epochs)
    per_site   = per_site_tree_budget(train_method, num_rounds, num_sites, local_epochs)
    acct_boost = BoostParams.from_xgb_params(params, num_boost_round=per_site)     # σ calibration
    growth     = BoostParams.from_xgb_params(params, num_boost_round=local_epochs)
    mechanism  = make_mechanism(dp, acct_boost, X.shape[1])   # run-calibrated σ (§4.3)
    rng        = np.random.default_rng(
                     np.random.SeedSequence([int(params.get("seed", 0)), global_round,
                                             _site_hash(site)]))       # §3.5 independence
    global_dp  = None if global_round == 1 else DPBooster.from_json_bytes(global_model_bytes)
    return dp_local_boost(global_dp, X, y, growth, dp, mechanism, rng,       # rng THREADED (A1)
                          local_epochs, train_method)

@app.train()  # @train: DP branch after cfg is built (client_app.py:101)
def train(msg, context):
    ...
    num_local_round = context.run_config["local-epochs"]   # existing var (client_app.py:98)
    dp = DPConfig.from_run_config(cfg)
    if dp.enabled:
        X_tr, y_tr, _, _, num_train, _ = load_data_arrays(context)   # raw arrays (§4.5)
        site  = Path(context.node_config["data-path"]).name
        gbytes = None if global_round == 1 else \
                 bytes(msg.content["arrays"]["0"].numpy().tobytes())
        booster = _dp_train_round(
            params, dp, global_round, num_local_round, train_method,
            context.run_config["total-trees"], context.run_config["num-sites"],
            X_tr, y_tr, gbytes, site)
        model_np = np.frombuffer(booster.to_json_bytes(), dtype=np.uint8)
        model_record = ArrayRecord([model_np])
        ...  # same MetricRecord({"num-examples": num_train}) + Message as the XGB path
    else:
        ... existing XGB path, verbatim (client_app.py:95-126) ...
```

- **`dp_local_boost` keeps the `fixed_range` guard for parity with `train_dp_gbdt`.**
  `train_dp_gbdt` raises on `dp.enabled and bin_strategy != "fixed_range"`
  ([boost.py:344-348](../../architecture/fed_stroke/dp/boost.py)); `dp_local_boost` must raise the
  same way (it always builds `fixed_bin_edges`, so an illegal `dp.bin-strategy='quantile'` would
  otherwise run silently-safe instead of failing loud — a behavioral split between the two
  learners for the same illegal config).

- **The XGB `save_raw`/`load_model` path is left untouched** — the DP path has its **own**
  wrap/unwrap (no shared helper), so the non-DP bytes cannot be perturbed (§7.7). The only shared
  shape is `ArrayRecord([np.frombuffer(bytes, uint8)])` at index `["0"]`, which is what keeps the
  Flower transport identical.
- **`rng` is threaded end-to-end** — created in `_dp_train_round`, passed to `dp_local_boost`,
  forwarded to `_grow_trees`, and consumed only inside `mechanism.add_noise` (§3.5). No function
  in the chain defaults it; §4.9 case 4 pins that consecutive rounds and distinct sites draw
  different noise.
- `_site_hash(site)` is a small, stable, non-negative int from the public site name (e.g.
  `int.from_bytes(hashlib.sha1(site.encode()).digest()[:4], "big")`) — public metadata only,
  never data (§3.5).
- **`subsample = 1.0` guard**: if `dp.enabled and float(params.get("subsample", 1.0)) != 1.0`,
  log a warning and use `1.0` for the DP arm **without mutating** `params` (§3.10). The DP learner
  ignores it regardless; the guard is documentary honesty for q = 1.0 accounting.
- `@evaluate` DP branch: `booster = DPBooster.from_json_bytes(...)`; load raw valid arrays via
  `load_data_arrays`; `y_prob = booster.predict(X_valid)`; score through the **unchanged**
  `compute_binary_metrics` ([client_app.py:146-152](../../architecture/fed_stroke/client_app.py)).
  The site `ConfigRecord` and `num-examples` rename are identical to the XGB branch.

### 4.5 `task.py` — raw-array data path

```python
def load_data_arrays(context) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """(X_train, y_train, X_valid, y_valid, num_train, num_val) via the SAME resolve_run_split
    path load_data_gva uses (task.py). X columns are exactly FEATURE_COLS order (== FEATURE_RANGES
    order, §3.9); y is TARGET_COL. Additive — load_data_gva (DMatrix path) is untouched."""
```

Both `load_data_gva` and `load_data_arrays` delegate to the one `resolve_run_split` contract
(1.1.a §4.5), so the DP arm trains/evaluates on exactly the same split as the XGB arm at matched
`split-seed`/`holdout-frac`/`holdout-eval`.

### 4.6 `strategies.py` — DP-aware bagging aggregator (cyclic reused)

```python
class DPFedXgbBagging(FedXgbBagging):
    """Bagging over DPBooster models. Overrides aggregate_train only; configure_train and
    the min_available_nodes/single-array contracts are inherited. Inherited configure_train
    sets self.current_bst = arrays["0"].tobytes() each round (fedxgb_bagging.py:79-86): b""
    on round 1, else last round's merged DP JSON — so the accumulator base is always in
    self.current_bst, exactly as the XGB parent uses it."""
    def aggregate_train(self, server_round, replies):
        # 0. Base accumulator = self.current_bst (set by inherited configure_train). Empty
        #    (b"") on round 1 -> the empty-accumulator sentinel (step 4); else deserialize it.
        # 1. Deserialize each reply's DPBooster (DPBooster.from_json_bytes).
        # 2. Assert feature_ranges, base_margin, max_bins agree across replies (and, when the
        #    accumulator is non-empty, with it too) — data-independent, must match; else raise
        #    (catches node/config drift). This is the server-side structural tripwire that
        #    replaces the predict-based assert_dp_roundtrip (server has no X, §4.1).
        # 3. Concatenate each reply's NEW .trees onto the accumulated global's .trees
        #    (base_margin counted ONCE, leaves already η-scaled — §3.6).
        # 4. Empty-accumulator sentinel: on server_round 1 (self.current_bst == b""),
        #    adopt this round's concatenated replies as the new global (mirrors aggregate_bagging's
        #    `if bst_prev == b""`). server_round == 1 and current_bst == b"" are equivalent here
        #    (round-1 seed is b""); key off server_round for symmetry with the client (§3.7).
        # 5. merged.meta = replies[0].meta (identical across replies by construction, §3.5;
        #    asserted consistent in step 2's spirit) — so ε/σ/release-count survive to save/log (§4.7).
        # 6. Re-serialize the merged DPBooster -> ArrayRecord([uint8]) at ["0"]; store its bytes
        #    back so the next round's configure_train re-caches it as current_bst.
        # 7. RETURN (arrays, metrics) — the signature is
        #    tuple[ArrayRecord|None, MetricRecord|None] (fedxgb_bagging.py:88-117). Aggregate the
        #    MetricRecords via self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        #    exactly like the parent, and return BOTH. Returning a bare ArrayRecord unpacks
        #    wrong in strategy.start. Route empty/invalid replies through the inherited
        #    self._check_and_log_replies (as the parent does) before any of the above.
```

- **Cyclic** reuses `OrderedFedXgbCyclic` **unchanged**: `FedXgbCyclic.aggregate_train` adopts
  `reply[0]` bytes wholesale (no XGB parsing), and the client already returns the **full**
  DPBooster ensemble under cyclic (§4.2). The site-query ordering logic is format-agnostic.
- `build_strategy` ([server_app.py:51-74](../../architecture/fed_stroke/server_app.py)) gains a
  DP branch: `train_method == "bagging"` and `dp.enabled` → `DPFedXgbBagging(...)` (same
  `fraction_*`/`min_available_nodes`/`evaluate_metrics_aggr_fn` args as `FedXgbBagging`); cyclic
  → `OrderedFedXgbCyclic` regardless of `dp.enabled`. When `dp.enabled = false`, `build_strategy`
  returns exactly today's strategies.

### 4.7 `server_app.py` — DP branch in `main()` / `save_final_model`

- **`main()` reads `dp = DPConfig.from_run_config(cfg)`** right after `cfg` is built
  ([server_app.py:160-161](../../architecture/fed_stroke/server_app.py)) so both the rebuild and
  the log branch on `dp.enabled`.
- **Final-model rebuild** ([server_app.py:212-226](../../architecture/fed_stroke/server_app.py)):
  the XGB `xgb.Booster().load_model(...)` is DP-specific-branched — `if dp.enabled: bst =
  DPBooster.from_json_bytes(bytes(result.arrays["0"].numpy().tobytes()))` (which restores `.meta`,
  §4.1); else the current XGB rebuild verbatim.
- **`save_final_model`** ([server_app.py:121-139](../../architecture/fed_stroke/server_app.py)):
  **dispatch on `isinstance(bst, DPBooster)`** (Decision 9 — no signature change, so the
  `test_server_config.py` import and the non-DP path stay byte-identical). DP branch writes
  `bst.to_json_bytes()` to `<model-dir>/final_model.json` — **no** `bst.set_attr`, **no**
  `bst.save_model` (those are XGB-only). The provenance the XGB path stamps via
  `set_attr(fed_run_config=...)` travels in the DPBooster `meta` block instead (§4.1); the DP
  branch merges `{"params": params, "total_trees": total_trees}` into `bst.meta["fed_run_config"]`
  before writing, so the saved JSON carries the same provenance the XGB attribute would.
- **Logging**: on a DP run, log `reported_epsilon`, `noise_multiplier`, and
  `num_releases = 2·D·per_site_budget` **read from the rebuilt booster's `.meta`** (a first-class
  attribute restored by `from_json_bytes`, §4.1 — not re-derived from a mechanism the reloaded
  model no longer has) so the operator and the logbook see the ε the run operated at. Because it
  is synthetic/example data, the log line is explicitly marked **not a real-ε claim** (§8).
- The initial `b""` seed and `derive_num_rounds` are unchanged (the DP client interprets round 1
  as fresh; the DP aggregator interprets the empty accumulator as adopt-replies, §3.7).

### 4.8 Minimal offline DP scoring (user-chosen scope)

The auto-detect is deliberately **split in two** (Decision 10), because
`score_booster_on_half(bst, data_path, ...)`
([baseline.py:161](../../architecture/fed_stroke/baseline.py)) receives an **already-loaded
booster object**, not a file path — it never reads the model file, so the format sniff cannot
live inside it. Instead:

- **`baseline.score_booster_on_half` — object-type dispatch, signature unchanged.** Branch on
  `isinstance(bst, DPBooster)`: DP → `y_prob = bst.predict(X_valid)` on the **raw X** from the
  same `resolve_run_split`-derived `valid_df[FEATURE_COLS]` (a `np.ndarray`, no `DMatrix`); else
  the current `xgb.DMatrix` + `bst.predict(dmatrix)` path verbatim. `y_true` is
  `valid_df[TARGET_COL].to_numpy()` either way; the metric math (`compute_binary_metrics`) is
  identical for both. **Keeping the `(bst, ...)` signature is what leaves all 10 call sites
  untouched** (`check_fed_vs_pooled.py` ×4, `run_hpo.py` ×2, `eval_final_model.py` ×1, and 3
  tests) — they keep passing whatever booster object they already loaded.
- **`scripts/eval_final_model.py` — file-format sniff at the loader** (this is where the file is
  read, [eval_final_model.py:59-68](../../architecture/scripts/eval_final_model.py)): read the
  model file's bytes; if the JSON carries `"format": DP_MODEL_FORMAT`, `bst =
  DPBooster.from_json_bytes(data)` and the tree count is `len(bst.trees)`; else the current
  `xgb.Booster().load_model(...)` + `bst.num_boosted_rounds()` path. `--expected-trees` asserts
  against whichever count. Then pass the loaded object straight into `score_booster_on_half`
  (which dispatches on type, above). Without this branch the hard-coded
  `xgb.Booster().load_model` throws on a DP JSON — the V-DP step (§6) fails.
  **Move the `bst.set_param({"eval_metric": "auc"})` line
  ([eval_final_model.py:61](../../architecture/scripts/eval_final_model.py)) INTO the XGB branch**
  — `DPBooster` has no `set_param`, so leaving it unconditional between load and count raises
  `AttributeError` on a DP model. (`eval_metric` is irrelevant to `DPBooster.predict` anyway.)
- **`check_fed_vs_pooled.py` is explicitly NOT changed** (§1 out-of-scope; §8 names why: the DP
  learner's noised `min_child_weight` gate voids the ±0.03 tolerance). Its call to
  `score_booster_on_half` still passes an `xgb.Booster`, so the object-type dispatch leaves it on
  the unchanged path.

### 4.9 Tests (`architecture/tests/test_dp_fl.py`, new; + extensions)

Fast pure-logic where possible; the live-federation E2E path is the §6 run matrix. Reuse the
tiny-booster idiom of `tests/test_client_boost.py`. **The `SimpleNamespace`/`FakeReply` idiom in
`tests/test_strategies.py` only fakes the site-query path — it does NOT satisfy
`aggregate_train`.** `DPFedXgbBagging.aggregate_train` inherits `FedAvg._check_and_log_replies`
→ `validate_message_reply_consistency` (flwr `strategy_utils.py`), which requires each reply's
`.content` to be a real `RecordDict` with exactly one `ArrayRecord` + one `MetricRecord` carrying
`num-examples`. So cases 6/11 must build **real** `flwr.app` `Message`/`RecordDict`/`ArrayRecord`/
`MetricRecord` objects (cheap — the DP bytes are small); a `SimpleNamespace` reply raises
`InconsistentMessageReplies`.

| # | Case |
|---|---|
| 1 | **Release-count / σ calibration:** for a run config, `make_mechanism(acct_boost, ...).num_releases == 2·D·per_site_budget` (=160 for 40/2/depth-4/epochs-1); σ reconstructed independently in rounds 1..R is bit-identical; ε cross-checks `account_run(n_rel, σ, 1.0, δ)` at that `n_rel` (rides the existing Opacus q=1.0 gate). |
| 2 | **Cyclic uneven participation (pins §3.4 — highest value):** `per_site_tree_budget("cyclic", num_rounds=41, num_sites=2, local_epochs=1) == 21` (not 20); the reported ε at budget 21 ≥ ε at budget 20 (never under-report). Bagging budget == `total_trees // num_sites`. |
| 3 | **Two-BoostParams separation (pins §4.3, drives the pure `_dp_train_round`):** stepping `_dp_train_round` over R rounds (no `Context`, §4.4/Decision 12) grows exactly `local_epochs` trees per round while the mechanism's σ stays the run-level value; the grown-tree count over R rounds equals `per_site_budget`. |
| 4 | **Cross-round + cross-site noise independence (pins §3.5 AND the A1 rng-threading):** consecutive `global_round`s draw different noise; two distinct `site`s in the same round draw different noise; all reproducible from the same `SeedSequence` inputs. Drive `_dp_train_round` at fixed data with varying `(global_round, site)` and assert the returned trees differ — this is what catches a regression where `rng` stops reaching `_grow_trees` (the noise would then be identical every round). |
| 5 | **Serialization round-trip (pins §4.1):** `DPBooster.from_json_bytes(b.to_json_bytes()).predict(X)` equals `b.predict(X)`, and the reloaded `.edges` equal the writer's (reconstructed from the serialized `feature_ranges`+`max_bins`). **Non-finite `meta`:** a booster whose `meta` carries `reported_epsilon=inf` (identity arm) or `noise_multiplier=nan` (Laplace arm) serializes without raising (the `_json_finite` pre-pass maps them to `null`, then `allow_nan=False` passes) and reloads with those keys `None`. `from_json_bytes(b"")` and an unknown/absent format marker raise `ValueError`. |
| 6 | **Bagging merge equivalence (pins §3.6):** concatenating two boosters' `.trees` with one shared `base_margin` predicts the sum of their individual margin contributions; a 2-round `DPFedXgbBagging.aggregate_train` continuation reproduces a hand-computed reference trajectory; mismatched `feature_ranges`/`max_bins` across replies raises. **Build the test boosters with `base_score != 0.5`** (so `base_margin = logit(base_score) ≠ 0`) — at the config default `base_score=0.5 → base_margin=0`, where "counted once" and "counted twice" are numerically identical and the double-count regression this case exists to catch would slip through. |
| 7 | **Empty-sentinel + round-1 (pins §3.7):** `DPFedXgbBagging.aggregate_train` on the initial `b""` accumulator adopts round-1 replies; the client maps `server-round == 1` → fresh (`global_dp is None`), not byte-sniffing. |
| 8 | **`train_dp_gbdt` byte-identity (pins §4.2 refactor):** the `_grow_trees`-based `train_dp_gbdt` produces bitwise-identical trees + predictions to a pinned pre-refactor reference at a fixed seed (protects the prototype/demo). |
| 9 | **Non-DP regression (pins §7.7):** with `dp.enabled = false`, `_train_round` reply bytes and `save_final_model`'s `final_model.json` are byte-identical to current `main` (extend `test_client_boost.py` / the server config test). |
| 10 | **Offline dispatch (pins §4.8):** `score_booster_on_half` given a `DPBooster` predicts on raw X; given an `xgb.Booster` uses the DMatrix path; both return the same metric keys. Separately, `eval_final_model.py`'s loader routes a `dp-gbdt-v1` file to `from_json_bytes` (tree count `len(trees)`) and an XGB file to `load_model` (`num_boosted_rounds`). |
| 11 | **`meta` persistence + logging contract (pins §4.1/§4.7 — the A2 chain):** after `save_final_model` on a DP model, reload `final_model.json` via `from_json_bytes` and assert `.meta` carries `reported_epsilon`, `noise_multiplier`, `num_releases == 2·D·per_site_budget`, and `fed_run_config.{params,total_trees}`, matching what `make_mechanism` computed for that config. Also: a `DPFedXgbBagging.aggregate_train` over 2 replies sets `merged.meta` from the replies (survives merge→re-serialize). |
| 12 | **`@evaluate` DP branch (pins §4.4 evaluate path):** a thin unit test — deserialize a known `DPBooster` from `ArrayRecord["0"]` bytes, predict on `load_data_arrays`-shaped raw X, and confirm `compute_binary_metrics` returns the full key set with the `num-examples` rename, matching the XGB branch's contract. |

## 5. Files to change

- **Modify** `architecture/fed_stroke/dp/boost.py` — `DPBooster.__init__` gains `feature_ranges`
  + `meta` (both default `None`); module-level `_json_finite` scrubber; `to_json_bytes`/
  `from_json_bytes` (round-trip `feature_ranges`+`meta`, scrub non-finite) + tests-only
  `assert_dp_roundtrip` (§4.1);
  `_grow_trees(..., rng, ...)` extraction + `train_dp_gbdt` thin-wrapper (sets `.meta`) +
  `dp_local_boost(..., rng, ...)` (§4.2); `per_site_tree_budget` + `DP_MODEL_FORMAT` (§4.3).
- **Modify** `architecture/fed_stroke/dp/__init__.py` — export `dp_local_boost`,
  `per_site_tree_budget`, `DP_MODEL_FORMAT` (and keep the existing seam exports).
- **Modify** `architecture/fed_stroke/client_app.py` — pure `_dp_train_round` helper (mirrors
  `_train_round`, threads `rng`) + `@train` DP dispatch; `@evaluate` DP branch; `_site_hash`;
  `subsample=1.0` guard; `X.shape[1]` assertion (§4.4). Imports `derive_num_rounds` from
  `server_app` (same as `hpo.py:20`), `per_site_tree_budget`/`dp_local_boost`/`DPBooster`/
  `BoostParams`/`make_mechanism`/`FEATURE_RANGES` from `fed_stroke.dp`.
- **Modify** `architecture/fed_stroke/task.py` — `load_data_arrays` (§4.5).
- **Modify** `architecture/fed_stroke/strategies.py` — `DPFedXgbBagging` (overrides
  `aggregate_train` only; copies `replies[0].meta`) (§4.6).
- **Modify** `architecture/fed_stroke/server_app.py` — `main()` reads `dp`; DP branch in
  `build_strategy`, `main()` rebuild (`from_json_bytes`), `save_final_model`
  (`isinstance(bst, DPBooster)` dispatch, no signature change); DP logging from `.meta` (§4.7).
- **Modify** `architecture/fed_stroke/baseline.py` — `score_booster_on_half` object-type dispatch
  (`isinstance(bst, DPBooster)`, signature unchanged) (§4.8).
- **Modify** `architecture/scripts/eval_final_model.py` — file-format sniff at the loader
  (`from_json_bytes` vs `load_model`; `len(trees)` vs `num_boosted_rounds`); move the XGB-only
  `set_param({"eval_metric": "auc"})` into the XGB branch (§4.8).
- **New** `architecture/tests/test_dp_fl.py` — §4.9 cases 1–8, 10, 11, 12; **modify**
  `architecture/tests/test_client_boost.py` (non-DP regression, case 9) and
  `architecture/tests/test_strategies.py` (DP bagging merge + `meta` carry, cases 6/11).
- **Modify** `docs/logbook.md` — dated entry when the code lands (gate result, the DP-run ε/σ/
  release-count on example/synthetic data, marked **not a real-ε claim**).
- **Reused unchanged**: `fed_stroke/dp/accounting.py` (frozen, Opacus-validated),
  `fed_stroke/dp/synthetic.py`, `fed_stroke/metrics.py`, `fed_stroke/schema.py`,
  `scripts/run_local_federation.sh`, `pyproject.toml` (`dp.*` keys already present — **no config
  change**).
- **Not changed here** (named, deferred): `scripts/check_fed_vs_pooled.py` (§8);
  `scripts/run_hpo.py` (already threads `dp.*`; DP arms are 1.1.b).
- **Not checked here**: the roadmap 1.1.a′ checkbox stays `[ ]` until the code lands.

## 6. Verification (end to end)

Run from `architecture/`. The 2-SuperNode `local-deployment` federation must be up
(`scripts/run_local_federation.sh start`). All runs use **example/synthetic data only** — no real
patient data, no real-ε claim (§8).

| Step | Command | Expect |
|---|---|---|
| V1 | `uv run pytest` | green, incl. `test_dp_fl.py` and the **non-DP byte-identity** regression (case 9). |
| V2 | `uv sync --no-dev && uv run python -c "import fed_stroke.dp, fed_stroke.strategies"` | succeeds with opacus/torch absent — **runtime stays torch-free** (re-`uv sync --group dev` after). |
| V3 | `flwr run . local-deployment --stream` (default `dp.enabled=false`) | completes; `final_model.json` byte-identical to a pre-change run (the non-DP path is untouched). |
| V4-bag | `flwr run . local-deployment --stream --run-config "dp.enabled=true dp.mechanism='gaussian' dp.target-epsilon=30 train-method='bagging' save-model=true"` | both rounds complete over the TLS channel; server logs reported ε / σ / `num_releases=2·D·per_site` (marked not-a-real-ε); DP `final_model.json` written. |
| V5-cyc | same as V4 with `train-method='cyclic'` | completes via `OrderedFedXgbCyclic` (unchanged) round-tripping DP bytes; per-site budget uses `⌈num_rounds/num_sites⌉`. |
| V-DP | `python scripts/eval_final_model.py out/<dp-model-dir>/final_model.json --data out/geneva_half_A.parquet out/geneva_half_B.parquet --expected-trees <t>` | auto-detects `dp-gbdt-v1`, reconstructs the DPBooster, scores both halves — matches the run's evaluate metrics to fp tolerance (shared `compute_binary_metrics`). |

V4/V5 are **plumbing + accounting** proofs on synthetic/example data. The real-Geneva DP sweep
and the first real-ε numbers are **1.1.b**, gated behind the independent DP review.

## 7. Acceptance criteria

1. **DP models cross the wire and round-trip.** A `DPBooster` serializes to bytes, travels in
   `ArrayRecord["0"]`, and deserializes to a booster that predicts identically (edges
   reconstructed from `feature_ranges`+`max_bins`; `meta` inf/nan → null). Empty / unknown-format
   bytes raise.
2. **A DP federated run completes for both strategies** over the loopback TLS topology (V4/V5):
   bagging via `DPFedXgbBagging` (tree concatenation), cyclic via unchanged `OrderedFedXgbCyclic`.
3. **Accounting is run-level and per-busiest-site.** σ is calibrated once to
   `n_rel = 2·D·per_site_budget`; `per_site_budget` uses `⌈num_rounds/num_sites⌉·local_epochs`
   for cyclic (never `total_trees//num_sites`); each stateless round reconstructs the identical σ;
   the reported ε at the busiest-site budget is a valid upper bound for every site. Pinned by
   tests (§4.9 cases 1–3), including the cyclic-uneven-participation case.
4. **Noise is independent per round and per site** (**within a run** — cross-run independence is a
   named 1.1.b item, §8), seeded from public `(base_seed, round, site)` only, and reproducible
   (§4.9 case 4).
5. **`dp.enabled = true` forces q = 1.0 accounting**: `subsample` is treated as `1.0` for the DP
   arm (documentary; the learner never samples), without mutating the shared `params`.
6. **Runtime stays torch-free** (V2): `fed_stroke.dp` and `fed_stroke.strategies` import under
   `uv sync --no-dev`; opacus stays dev-only.
7. **The non-DP path is byte-identical** (V1 case 9, V3): with `dp.enabled = false` the client
   reply bytes and the saved `final_model.json` match current `main` exactly.
8. **A saved DP model is re-scorable offline** (V-DP): `eval_final_model.py` /
   `score_booster_on_half` auto-detect the DP format and score on raw X, matching the run's
   evaluate metrics — the objective path 1.1.b consumes.
9. **No real-ε claim.** Every V-run uses synthetic/example data; the server ε log line and the
   logbook entry are marked as such.
10. `pytest` (full suite) stays green; new tests follow the repo's fixture idiom.

## 8. Risks & notes

- **The cyclic budget is the subtlest ε trap.** `total_trees // num_sites` silently under-reports
  ε whenever `num_rounds % num_sites != 0` (§3.4). Fixed by `per_site_tree_budget` (busiest site)
  and pinned by the highest-value new test (§4.9 case 2). Consider a warning in DP cyclic mode
  when `num_rounds % num_sites != 0` so the uneven case is explicit.
- **The two-BoostParams rule is the second trap.** Feeding the growth params (`local_epochs`) to
  `make_mechanism` under-noises by ~`num_rounds×`; feeding the accounting params to
  `dp_local_boost` grows the whole budget every round. Two distinct objects, pinned by §4.9
  cases 1 & 3.
- **DP utility is scale-sensitive; ε ≳ 30 is where GVA-scale utility appears.** The prototype
  (logbook 2026-07-20) found that at GVA per-node scale (~1000 rows) the honest `2·D·T`
  accounting drives ε ≤ 5 toward chance and clean monotone utility needs ε ≳ 30. 1.1.a′ makes no
  utility claim (synthetic/example only); the V-runs use ε = 30 to exercise a non-degenerate arm.
  Real-scale utility is 1.1.b's finding.
- **DP tree shapes differ from XGB** (noised `min_child_weight` gate, prototype §8.5), so a
  federated-DP vs pooled-DP correctness gate cannot reuse 1.d's ±0.03 tolerance — that is why
  `check_fed_vs_pooled.py` DP support is deferred (§1) and needs its own tolerance design.
- **Effective learning rate under bagging is inherited, not introduced.** Appending both sites'
  trees each round doubles the per-round step — a property of `FedXgbBagging` that the DP path
  reproduces faithfully (§3.6), not a new DP bug.
- **Bagging merge assumes data-independent `edges`/`base_margin`/`max_bins`.** True today
  (`fixed_bin_edges(FEATURE_RANGES, max_bins)`, `logit(0.5)=0`); the aggregator **asserts**
  equality across replies so a config/node drift fails loudly rather than merging incompatible
  bin grids (§4.6, §4.9 case 6).
- **Feature-count coupling.** `√d` noise scaling and `_binize` edge count both key off
  `d = len(FEATURE_RANGES)`; the client asserts `X.shape[1] == len(FEATURE_RANGES)` (§3.9) so a
  future frozen-schema growth fails loudly instead of mis-scaling noise.
- **`base_score`.** `pyproject.toml` sets no `params.base_score`, so `BoostParams` defaults to
  0.5 → `base_margin = 0`; deterministic and carried once in the serialized DPBooster. Any DP-vs-
  baseline comparison (1.1.b) must agree on `base_score`.
- **JSON non-finite tokens.** `meta` can hold `inf` (identity ε) / `nan` (Laplace σ). These are
  turned into `null` by the `_json_finite` recursive pre-pass **before** `json.dumps`; `allow_nan=
  False` alone does NOT do this — it *raises* on a native `inf`/`nan`, and `default=` never fires
  for a recognized float (C-B1). `allow_nan=False` stays only as a belt-and-suspenders assert that
  the pre-pass caught everything, honoring the repo's artifact contract (§4.1, server_app.py:195).
- **`meta` provenance is fragile across the merge.** ε/σ/release-count live in `DPBooster.meta`,
  which must be set by `dp_local_boost`, round-tripped by both serializers, AND copied
  `replies[0].meta → merged.meta` by the aggregator — a reloaded/merged booster has **no
  mechanism** to fall back on, so any dropped hop makes the server's ε log line and the saved
  provenance silently empty (Decision 11). Pinned by §4.9 case 11.
- **`rng` threading is the third accounting trap** (after the cyclic budget and the two
  BoostParams). It must reach `_grow_trees` from the client's public-seed generator with no
  default anywhere in the chain, or noise repeats across rounds/sites (§3.5, C-A1). Pinned by
  §4.9 case 4.
- **Cross-RUN noise correlation — a 1.1.b hazard the seed scheme bakes in here (owned + gated).**
  The RNG seeds from `SeedSequence([base_seed, global_round, site_hash])` with
  `base_seed = params.seed = 0` (a fixed config constant) and **no per-run entropy**. Within one
  run this is exactly right (distinct `(round, site)` → independent streams, §3.5). But **two runs
  on the same data draw the identical standard-normal sequence `Z`**: arm-i noise is `σ_i·Z` and
  arm-j is `σ_j·Z`, so their difference cancels the data-independent part and leaves noise-reduced
  signal — an adversary holding two released arms can partially denoise. 1.1.a′ is synthetic-only
  and makes **no ε claim**, so this does not affect any V-run here. But §1 states 1.1.b is "the
  1.1.a sweep + DP arms **with no new orchestration**", i.e. it inherits this seed scheme verbatim,
  and the ε∈{1,3,5,10} sweep (plus forward/reverse cyclic pairs) releases multiple models on the
  **same Geneva patients**. Two things 1.1.b + the independent DP-accountant review must settle,
  flagged now so they are not lost: (1) fold a **run-unique public nonce** (e.g. a logged run id
  or the arm's `target_epsilon`+`mechanism`) into the `SeedSequence` so cross-run noise is
  independent; (2) decide whether the sweep's **joint** release is accounted by sequential
  composition (ε's add) rather than claiming each arm's ε in isolation. Both are DP-soundness
  questions, correctly downstream of 1.1.a′'s wiring scope, but the hazard originates in this
  spec's seed design.
- **Layering: the client imports `derive_num_rounds` from `server_app`.** This is the established
  pattern (`hpo.py:20` already does it) and pulls only torch-free flwr symbols, so it does not
  affect the §7.6 torch-free runtime check (which imports `fed_stroke.dp`/`fed_stroke.strategies`,
  not `client_app`). Flagged so a future reader does not mistake it for a new cycle.

## 9. Review decisions — audit trail

Recorded so the reasoning is not lost; if this section disagrees with §1–§8, the body wins.

- **Decision 1 — DP mode swaps the learner; `DPBooster` gets its own serializer, the XGB path is
  untouched.** Stock `hist` histograms are unreachable (prototype §3.1) so DP cannot intercept
  `bst.update`; the DP learner is a separate NumPy learner with a separate model object. A shared
  XGB/DP wrap/unwrap helper was rejected — it risks perturbing the non-DP bytes for no correctness
  gain (§4.4/§7.7). The only shared shape is the `ArrayRecord([uint8])` transport envelope.
- **Decision 2 — accounting is per-run, calibrated to the busiest site.** σ is fixed once to
  `2·D·per_site_budget`; the per-patient guarantee tracks the site that releases the most, so
  cyclic uses `⌈num_rounds/num_sites⌉` (never `total_trees//num_sites`). This is the honest,
  never-under-report choice (§3.4, Risk 1).
- **Decision 3 — continuation via a shared `_grow_trees`, not a re-implemented loop.** Refactor
  the prototype's boost loop into one core that both `train_dp_gbdt` (byte-identical wrapper) and
  `dp_local_boost` call, so the noise logic lives in exactly one place (§4.2). Sensitivity is
  margin-invariant, so continuation does not touch calibration.
- **Decision 4 — cyclic reuses `OrderedFedXgbCyclic` unchanged; only bagging gets a DP
  aggregator.** `FedXgbCyclic` adopts `reply[0]` bytes wholesale (format-agnostic); only bagging
  parses+merges the model, so only bagging needs `DPFedXgbBagging` (§3.6/§4.6).
- **Decision 5 — per-round, per-site independent noise from public seeds only.** Composition
  models independent Gaussians; seeding from `(base_seed, round, site_hash)` (public metadata,
  never data) gives independence + reproducibility and avoids the two sites drawing identical
  noise in a round (§3.5).
- **Decision 6 — synthetic/example data only; no real-ε claim.** 1.1.a′ is wiring + accounting
  validation. The real-Geneva sweep and the first real-ε numbers are 1.1.b, gated behind the
  independent DP-accountant review. Every V-run and the logbook entry are marked accordingly (§8).
- **Decision 7 — minimal offline scoring (with the user).** Only `score_booster_on_half` and
  `eval_final_model.py` gain DP auto-detect — the objective path 1.1.b needs and what makes §6
  re-scorable. `check_fed_vs_pooled.py` DP support is deferred: the noised `min_child_weight` gate
  voids the 1.d tolerance and a DP gate needs its own design (§8).
- **Decision 8 — no new config keys.** The `dp.*` table already exists from the prototype; this
  task makes those knobs bite. `run_hpo.py` already threads `--run-config`, so DP arms are a
  grid change in 1.1.b, not orchestration here (§4.9 framing, roadmap 1.1.a′).
- **Decision 9 — `save_final_model` dispatches on `isinstance(bst, DPBooster)`, not a new arg.**
  Keeps the `(bst, model_dir, params, total_trees)` signature, so `test_server_config.py` and the
  non-DP save path are untouched; the DP branch writes `to_json_bytes()` and folds
  `fed_run_config` into `meta` (§4.7). (Eng review 2026-07-22.)
- **Decision 10 — the offline auto-detect is split: object-type dispatch in
  `score_booster_on_half`, file-format sniff in `eval_final_model.py`.** `score_booster_on_half`
  receives a loaded booster, not a path, so it cannot read the file; it branches on
  `isinstance(bst, DPBooster)` and keeps its signature (leaving all 9 call sites, incl. 1.d and
  1.1.a, untouched). The file sniff lives at `eval_final_model.py`'s loader, the one place that
  opens the model file (§4.8). (Eng review 2026-07-22.)
- **Decision 11 — `meta` is a first-class `DPBooster` attribute threaded through the whole
  chain.** ε/σ/release-count must survive `dp_local_boost → serialize → deserialize → merge →
  re-serialize → rebuild → log/save`, but a reloaded/merged booster has no `mechanism` to derive
  it from. So `DPBooster.meta` is set by every producer, round-tripped by `to_json_bytes`/
  `from_json_bytes`, and copied `replies[0].meta → merged.meta` by the aggregator (§4.1/§4.6/§4.7).
  (Eng review 2026-07-22.)
- **Decision 12 — the client DP path is a pure `_dp_train_round`, mirroring `_train_round`.** The
  repo already extracted `_train_round` to be Context/Message-free for testability; the DP path
  gets the same shape so §4.9 case 3/4 drive the per-round accounting + noise sequence with no
  faked `Context` (§4.4). (Eng review 2026-07-22.)

**Corrections folded in by the 2026-07-22 eng review (bugs in the prior draft, not choices):**

- **C-A1 — the seeded `rng` was created in the client but never passed into `dp_local_boost`,
  which had no `rng` parameter to forward to `_grow_trees`.** As drafted, noise would have fallen
  back to `boost.seed` and been identical across every round and site — silently voiding the
  per-round/per-site independence of §3.5 and acceptance #4. `rng` is now an explicit,
  never-defaulted parameter of `dp_local_boost`, threaded end-to-end (§4.2/§4.4); §4.9 case 4
  pins it.
- **C-A3 — `assert_dp_roundtrip(booster, X)` was described as "used by the aggregator", but the
  server aggregator has no feature data `X`.** It is now tests-only; the aggregator does the
  data-independent structural check (feature_ranges/base_margin/max_bins equality) instead
  (§4.1/§4.6).

**Corrections from the 2026-07-22 outside-voice pass (two independent adversarial reviewers, DP
lens + FL-integration lens; no code changes to `dp/accounting.py`):**

- **C-B1 — `allow_nan=False` + `default=` does NOT turn inf/nan into `null`; it raises.**
  `json.dumps(allow_nan=False)` raises `ValueError` on a native float `inf`/`nan`, and `default=`
  is only called for types the encoder does not recognize (never for a float). The prior draft's
  serialization would have killed every Laplace run (`noise_multiplier=nan`) and the identity arm
  (`reported_epsilon=inf`), and made §4.9 cases 5/11 unwritable. Fixed by a module-level
  `_json_finite` recursive pre-pass (inf/nan → None, numpy scalars → py) applied BEFORE
  `json.dumps`, which stays as a belt-and-suspenders assert (§4.1). **Both reviewers found this
  independently** — the highest-value catch of the pass.
- **C-B2 — `to_json_bytes` serialized `feature_ranges`, but `DPBooster` had no such field and it
  is unrecoverable from the numpy `edges`.** The reconstruct-edges invariant (§4.9 case 5) was
  therefore unenforced, and reaching for the module `FEATURE_RANGES` would silently drift edges
  for a booster trained with a custom `feature_ranges`. Fixed by making `feature_ranges` a
  first-class constructor field carried through serialize/deserialize (§4.1/§4.2).
- **C-B3 — the `§4.6` aggregate_train pseudocode omitted the `(arrays, metrics)` return** and the
  `MetricRecord` aggregation the signature requires; a verbatim implementer would return a bare
  `ArrayRecord` and break `strategy.start`. Fixed with step 7 (§4.6).
- **C-B4 — smaller verbatim-implementation traps:** `eval_final_model.py`'s unconditional
  `set_param({"eval_metric":"auc"})` (no `DPBooster.set_param` → `AttributeError`) moved into the
  XGB branch (§4.8); the `§4.9` test idiom corrected — `aggregate_train` needs real
  `flwr` `Message`/`RecordDict` replies, not the site-query `SimpleNamespace` fake (M3); case 6
  must use `base_score != 0.5` or the double-count check is blind at `base_margin=0` (L5); the
  redundant `num_local_round`/`local_epochs` pair collapsed and the `local_epochs` `NameError` in
  the `@train` snippet fixed (§4.4); `dp_local_boost` keeps `train_dp_gbdt`'s `fixed_range` guard
  for parity (§4.2).
- **Flagged, not silently patched — cross-run noise correlation (§8).** Both the seed scheme and
  the "1.1.b = the sweep with no new orchestration" framing originate here, but the fix (a
  run-unique nonce + joint-composition accounting) is a DP-soundness decision correctly owned by
  1.1.b and its independent DP-accountant review. Recorded as a named §8 risk so it is not lost.
  1.1.a′ makes no ε claim, so no V-run here is affected.

**Confirmed CORRECT by the outside voice (independently re-derived / re-run, not merely agreed):**
`n_rel = 2·D·T_site` and the two-BoostParams feed; the cyclic busiest-site budget
`⌈num_rounds/num_sites⌉·local_epochs` (traced through flwr's `_make_sampling` + the numeric
under-report it prevents: reported ε 30.0 vs true 30.99 at budget 20 for 21 emitted trees);
per-run σ reconstruction; within-run per-round/per-site noise independence (post C-A1);
margin-invariant sensitivity for continuation; `subsample`→q=1.0 as an unbypassable no-op; the
Flower transport envelope; `current_bst` accumulator + `b""` sentinel semantics; cyclic reuse of
the stock strategy; JSON-safety of the tree dicts; `_dp_train_round` purity (case 3 drivable with
no `Context`); `score_booster_on_half` object-dispatch leaving all call sites intact; and the
absence of any import cycle (runtime stays torch-free).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside voice | 2 independent adversarial subagents (codex unavailable) | Independent 2nd opinion | 2 (DP-correctness lens + FL-integration lens, 2026-07-22) | CLEAR (fixes applied) | Both re-derived the accounting from scratch and confirmed the core sound. Caught 1 defect **both** found independently — the `allow_nan=False`+`default=` inf/nan→null mechanism raises instead of nulling (kills Laplace/identity, C-B1) — plus `feature_ranges` had no backing field (C-B2), the aggregate_train `(arrays,metrics)` return was omitted (C-B3), the `set_param`/test-idiom/base_score-0/`local_epochs` traps (C-B4), and the cross-run noise-correlation hazard (flagged to 1.1.b, §8). Folded into §4/§5/§8/§9. |
| Eng Review | `/plan-eng-review` | Architecture & tests | 2 (Plan agent spec stage; full eng review 2026-07-22 against the installed sources) | CLEAR (fixes applied) | Run 1 caught the cyclic ε under-report (`total_trees//num_sites` → `⌈num_rounds/num_sites⌉`), the two-BoostParams σ trap, cross-site noise seeding, the `_grow_trees` refactor need, the `b""` sentinel + XGB-specific `main()`/scorer lines, and the edges-reconstruction / inf-nan serialization pitfalls. Run 2 verified every file/line-ref against the tree and caught 5 compose-into-code defects: **(A1)** `rng` created but never threaded into `dp_local_boost`/`_grow_trees` → noise would repeat every round/site; **(A2)** `meta` derived from `.mechanism` but a reloaded/merged booster has none → ε/σ logging + save could not work, fixed by first-class `DPBooster.meta` (Decision 11); **(A3)** `assert_dp_roundtrip` "used by the aggregator" is impossible (server has no X) → tests-only; **(C1)** DP logic inlined in `@train` broke the repo's pure-`_train_round` test idiom → pure `_dp_train_round` (Decision 12); **(offline scoring)** `score_booster_on_half` "reads the model file" contradicted its `(bst, …)` signature with 9 call sites → object-type dispatch + loader-level file sniff (Decision 10). Also fixed `save_final_model` DP dispatch (Decision 9) and the `baseline.py` line-ref (156→161). All folded into §4/§5/§8/§9. |
| Design Review | `/plan-design-review` | UI/UX | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience | 0 | — | — |

- **CROSS-MODEL:** codex unavailable in this environment; substituted **two independent
  adversarial subagent reviewers** (2026-07-22, DP-correctness + FL-integration lenses), run
  without being shown the eng-review findings so their pass was genuinely independent. They
  converged: core accounting/integration **sound** (each re-derived `n_rel`, the cyclic budget,
  σ reconstruction, `current_bst` semantics), and surfaced the C-B1..C-B4 defects + the cross-run
  seed hazard now folded into §4/§8/§9 — one defect (C-B1, the inf/nan serialization) found by
  both independently. Combined with the two earlier eng passes (Plan-agent spec stage; the
  2026-07-22 line-by-line source verification that fixed the `baseline.py` 156→161 drift), the
  spec has had four independent reviews.
- **UNRESOLVED:** none at spec stage. Open items are downstream-owned and named in §1/§8: the real-
  Geneva ε sweep (1.1.b, gated on the independent DP review), `check_fed_vs_pooled.py` DP support,
  private-quantile bins on real data, and subsampled-RDP amplification credit.
- **VERDICT:** READY — spec complete and corrected (12 decisions + 2 pure corrections recorded in
  §9); implement behind the §6 verification, with the non-DP byte-identity regression (§4.9 case
  9), the release-count / cyclic-budget tests (§4.9 cases 1–3), the rng-independence test (case 4),
  and the `meta`-persistence test (case 11) as the correctness guards the Opacus gate alone cannot
  provide.

NO UNRESOLVED DECISIONS
