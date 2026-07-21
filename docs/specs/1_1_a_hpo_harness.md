# Spec 1.1.a — Systematic HPO harness (bagging + cyclic), first-pass on Geneva

Implements roadmap item 1.1.a ([architecture/roadmap.md](../../architecture/roadmap.md)):

> 1.1.a Systematic HPO on bagging: `num_server_rounds`, `max_depth`, `eta`,
> `min_child_weight`, `subsample`, `colsample_bytree` (architecture §3 defaults
> are starting points, not endpoints). Cyclic revalidation happens in Phase
> v1.3 once the strategy winner is picked.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§3 and §4) and the installed framework (`flwr==1.31.0`, message-based API,
`xgboost>=2.0`, `scikit-learn>=1.3`, Python `>=3.12`, `uv`). Builds directly on
1.b ([1b_fedxgb_bagging_cyclic.md](1b_fedxgb_bagging_cyclic.md)), 1.c
([1c_evaluation_harness.md](1c_evaluation_harness.md)), and 1.e
([1e_secure_topology.md](1e_secure_topology.md)).

**Framing.** Phase v1.1/v1.2 is the *DP + HPO development phase*; its outputs are
**development artifacts, not the headline**. 1.1.a therefore has two distinct
deliverables: (1) a **reusable, strategy-agnostic HPO harness** that drives the
real FL pipeline, and (2) a **first-pass Geneva-only bagging sweep** that yields
**approximate hyperparameter ranges** and a provisional tuned config. These GVA
ranges are explicitly provisional — Shenzhen's ~40k-patient cohort (vs ~1k/node
at Geneva, per [docs/logbook.md](../logbook.md) 2026-07-20) and its different
distribution will move the optimum, so the **same harness is re-run cross-site in
Phase v1.3** to re-tune within these ranges (§4.8, new roadmap item). 1.1.a is
also the **no-DP baseline** that 1.1.b measures its DP arms against.

## 1. Goal & scope

**In scope**

- A reusable HPO harness (`fed_stroke/hpo.py` + `scripts/run_hpo.py`) that runs
  each candidate config through the **real federated pipeline** via repeated
  `flwr run . <federation>`, federation-parameterized so the identical harness
  serves the Geneva-dev loopback topology now and the cross-site link in v1.3
  (§4.4, §3.2).
- **Strategy-agnostic:** `--strategy bagging|cyclic` selects the aggregation
  strategy. Per the roadmap the **1.1.a headline run is bagging**; cyclic is
  supported by the same harness (its formal revalidation stays Phase v1.3).
- A declarative **search space** over the six roadmap knobs, with
  `num_server_rounds` realized as the derived **`total-trees`** budget (§3.1),
  and divisibility pre-validated against `server_app.derive_num_rounds` (§4.1).
- A **selection objective** = mean of the two Geneva halves' site-stratified
  **AUC-ROC**, tie-broken by **lower cross-site AUC variance** (mirrors the v1.3
  winner rule, roadmap 1.3.d), computed by **offline both-halves scoring** of
  each trial's saved model via `baseline.score_booster_on_half` (§3.4, §4.2).
- A **repeated multi-seed CV** protocol for selection on a DEV remainder, with a
  **patient-disjoint hold-out** reserved for the untouched final report (§4.5),
  requiring three new config keys (`split-seed`, `holdout-frac`, `holdout-eval`,
  §4.6) all defaulting to today's flat split.
- Result artifacts: a ranked **leaderboard**, a **results JSON**, a
  **narrowed-range summary**, and a ready-to-paste **tuned-config TOML** (§4.3,
  §4.7) — the concrete hand-off to 1.1.b and the v1.3 re-run.
- Unit tests for all pure logic (§4.9).

**Out of scope** (later roadmap items)

- DP noise on the histograms and the ε ∈ {1, 3, 5, 10} arms → **1.1.b** (itself
  gated on the independent DP review, [dp_accountant_review_packet.md](../reviews/dp_accountant_review_packet.md)).
  1.1.a produces the no-DP baseline those arms are compared to; it does not touch
  `fed_stroke/dp/`.
- **Adaptive search (Optuna / TPE), parallelism, and compute-budget matching →
  1.1.c.** 1.1.a ships a coarse grid plus a `--max-trials` random-subsample
  escape hatch only; the search *method* upgrade is 1.1.c's explicit charter.
- **Cyclic revalidation as a decision**, and the **cross-site HPO re-run** →
  **Phase v1.3** (the re-run item is added by this spec, §4.8).
- Choosing a winning strategy → Phase v1.3 on real cross-site data; bagging and
  cyclic stay co-equal candidates.
- Any change to the DP modules, the secure topology (1.e), or Docker (1.f).

## 2. Dependencies

- **1.b complete** ([1b_fedxgb_bagging_cyclic.md](1b_fedxgb_bagging_cyclic.md)):
  both strategies selectable from config via
  [`build_strategy`](../../architecture/fed_stroke/server_app.py) and the tree
  budget derived by
  [`derive_num_rounds`](../../architecture/fed_stroke/server_app.py).
- **1.c complete** ([1c_evaluation_harness.md](1c_evaluation_harness.md)): the
  site-stratified harness and the shared scorer
  [`score_booster_on_half`](../../architecture/fed_stroke/baseline.py) /
  [`compute_binary_metrics`](../../architecture/fed_stroke/metrics.py) exist and
  are the single source of metric math.
- **1.e complete** ([1e_secure_topology.md](1e_secure_topology.md)): the loopback
  `local-deployment` federation stood up by
  [`scripts/run_local_federation.sh`](../../architecture/scripts/run_local_federation.sh)
  — the standing 2-SuperNode topology trials fire at (§3.2).
- **Roadmap phase blocker (stated, not resolved here).** The *headline* 1.1.a
  sweep is blocked by the roadmap on real Geneva data at the frozen schema plus a
  published v0.a local sanity baseline (preprocessing track). The **harness
  itself** is built and validated against the current two Geneva halves and runs
  unchanged the moment real frozen-schema data lands behind the SuperNode
  `data-path`s — no code change, only different parquet files.
- **No new runtime dependencies.** Grid expansion and orchestration use the
  standard library (`itertools`, `subprocess`, `json`, `argparse`) plus the
  already-present `pandas` / `xgboost` / `scikit-learn`. `pytest` is already a
  dev-only dependency.

## 3. Framework / design facts that drive the design

Verified against the installed `flwr==1.31.0` sources and the current
`fed_stroke` package.

1. **`num_server_rounds` is derived, never set.**
   [`server_app.derive_num_rounds`](../../architecture/fed_stroke/server_app.py)
   computes the round count from a single `total-trees` budget per strategy
   (bagging: `total_trees / (num_sites × local_epochs)`; cyclic: `total_trees /
   local_epochs`), and `num-server-rounds` was deliberately removed from the
   config (1.b §4.1). ⇒ The roadmap's `num_server_rounds` knob is realized as
   the **`total-trees`** sweep axis, and every candidate `total-trees` is
   pre-validated for divisibility by calling `derive_num_rounds` **before** a run
   is launched — the harness never fires a run the server would reject.
2. **2-site runs require the deployment topology, not simulation.** The
   `local-simulation` federation is `num-supernodes=1` (see `~/.flwr/config.toml`
   and `working_example/pyproject.toml`), and giving two SuperNodes **distinct**
   data files is a property only the deployment topology has — runtime
   partitioning inside the load function was deliberately rejected (roadmap 1.a).
   ⇒ Trials fire `flwr run . local-deployment` at the standing loopback
   federation; the **federation name is the only thing that changes** for the
   v1.3 cross-site re-run (§4.8).
3. **Repeated `flwr run` against a standing federation is stateless per run.**
   [`server_app.main`](../../architecture/fed_stroke/server_app.py) re-inits an
   empty global model (`global_model = b""`) each run, and clients reload data
   every round via
   [`load_data_gva`](../../architecture/fed_stroke/task.py). ⇒ Many trials can
   fire at one long-lived federation; `--run-config` overrides (`params.*`,
   `total-trees`, `train-method`, `split-seed`, `holdout-frac`, `holdout-eval`,
   `n-boot`, `save-model`, and per-trial absolute `model-dir` / `metrics-dir`)
   fully parameterize a run, and runs do not leak state into one another.
4. **Cyclic per-round evaluation is a self-evaluation** (1.b §3 consequence 3;
   1.c §8). Under cyclic, each round evaluates only the site that just trained,
   so the persisted per-round artifact is **not** cross-strategy comparable. ⇒
   The HPO objective must **not** read the per-round metrics artifact; it scores
   each trial's **saved final model** on **both** halves offline via
   [`score_booster_on_half`](../../architecture/fed_stroke/baseline.py) — the
   only bagging-vs-cyclic-comparable measure, exactly as established in 1.b §4.7
   / `eval_final_model.py`.
5. **The validation split is hardcoded to `seed=42`.**
   [`load_data_gva`](../../architecture/fed_stroke/task.py:98) and
   [`baseline.split_half`](../../architecture/fed_stroke/baseline.py:89) both
   call `generate_splits(..., test_size=0.2, seed=42)`. ⇒ To run repeated CV the
   split logic moves into one helper (`task.resolve_run_split`) driven by three
   config keys — `split-seed` (the train/valid seed, default 42), `holdout-frac`
   (patient-disjoint hold-out fraction, default `0.0` = disabled), `holdout-eval`
   (default `false`) — and `split_half` / `score_booster_on_half` gain matching
   `split_seed=42, holdout_frac=0.0, holdout_eval=False` parameters. At those
   defaults the behavior is byte-identical to today's single `seed=42` split
   (§4.5). The final report is a genuine **patient-disjoint** hold-out (a reserved
   patient set, partitioned once by a fixed `HOLDOUT_PARTITION_SEED = 42`), **not**
   a held-out seed — Decision 11 (§9) explains why a held-out seed leaked.
6. **`colsample-bytree` / `subsample` bite weakly at the current schema.**
   [`schema.py`](../../architecture/fed_stroke/schema.py) freezes only two
   features (`Age (calc.)`, `NIH on admission`); over two columns
   `colsample-bytree < 1` is coarse (the very interaction the `round_seed` fix in
   [client_app.py:45](../../architecture/fed_stroke/client_app.py) exists to
   handle). ⇒ The grid keeps both knobs — the real feature set grows on the
   preprocessing track, and DP forces `subsample = 1.0` for honest `q = 1.0`
   accounting (1.1 prereq §3.5, [1_1_prereq_dp_plugpoint.md](1_1_prereq_dp_plugpoint.md))
   so their landscape matters downstream — but their weak effect at two features
   is noted (§8).

Consequence: the change is a **pure-logic library** (grid, objective, ranking,
artifacts, `build_signal`), a **thin orchestration driver** (subprocess `flwr run`
with a per-run timeout + a preflight canary, then read saved models), **one split
helper** (`resolve_run_split`) that the client and the offline scorer both delegate
to, **three backward-compatible `split-*` config keys** (all defaulting to today's
flat split), **three reconstruction flags** on `eval_final_model.py`, and a
**roadmap update**. No FL protocol code, no metric math, and no strategy code is
rewritten — all reused.

## 4. Design

### 4.1 `architecture/fed_stroke/hpo.py` — search space + trial generation

Pure, importable, no I/O and no `flwr` app context (mirrors how `baseline.py`
splits testable core from `scripts/`). Default grid is a module constant,
overridable by a JSON file passed to the driver (§4.4). Coarse by design — the
search *method* upgrade is 1.1.c.

```python
# Coarse first-pass grid (architecture §3 defaults sit inside each range).
# Overridable via run_hpo.py --grid <file.json>. num_server_rounds is realized
# as total-trees (§3.1). Adaptive/finer search is 1.1.c, not here.
DEFAULT_GRID = {
    "total-trees":            [40, 80, 160],
    "params.max-depth":       [3, 4, 6],
    "params.eta":             [0.05, 0.1, 0.3],
    "params.min-child-weight":[1, 5, 20],
    "params.subsample":       [0.8, 1.0],
    "params.colsample-bytree":[0.8, 1.0],
}  # 3·3·3·3·2·2 = 324 cells
# total-trees realizes the num_server_rounds knob (§3.1) — the roadmap's
# first-named and highest-impact axis — so it carries three points, not two:
# with only the {40,160} endpoints summarize_top_ranges (§4.3) could not narrow
# the round count, it could only pick an endpoint. All three are divisible by 2
# (bagging trees/round at num_sites=2, local_epochs=1), so none is rejected by
# valid_trials at the default topology (Decision 8, §9).


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of the grid → ordered list of flat override dicts.

    Each dict maps a `flwr run --run-config` dotted key to a value, e.g.
    {"total-trees": 40, "params.max-depth": 4, ...}. Deterministic ordering
    (keys sorted, itertools.product) so trial ids are stable across runs.
    """


def valid_trials(trials, train_method, num_sites, local_epochs) -> tuple[list, list]:
    """Split trials into (runnable, rejected) by tree-budget divisibility.

    Reuses server_app.derive_num_rounds: a trial whose `total-trees` is not
    divisible by trees/round for this strategy is rejected here (fail fast) rather
    than launched and rejected by the server. Returns rejected trials with the
    reason so the driver can log them — never silently drop.
    """
```

`trial_id` is a short deterministic hash of the override dict, used for the
per-trial output directory (`out/hpo/<trial_id>/…`). It **must** be computed with
`hashlib` (e.g. `sha1(json.dumps(override, sort_keys=True).encode()).hexdigest()[:10]`)
— **never** Python's builtin `hash()`, which is per-process salted by
`PYTHONHASHSEED` and would give a trial a *different* directory on a resume or a
re-run, silently re-running everything and orphaning the first run's models
(Decision 10, §9). Canonical serialization (`sort_keys=True`) makes the id
invariant to dict insertion order. A unit test pins cross-process stability
(§4.9).

### 4.2 `architecture/fed_stroke/hpo.py` — objective + selection

The objective is computed from **offline both-halves scoring** (§3.4), not the
per-round artifact.

```python
def trial_objective(per_repeat_site_aucs: list[dict], k_search: int) -> dict:
    """Aggregate one trial's per-repeat per-site AUC-ROC into a scalar objective.

    per_repeat_site_aucs: [{site: auc_or_None, ...}, ...] — one dict per search
    seed, each holding both halves' offline AUC-ROC (None = single-class split).

    Per repeat: objective_r = mean(site AUCs), variance_r = population variance
    across sites — ONLY if every site AUC is present. A repeat with any None site
    AUC is DROPPED (single-class split, §8). The trial's objective is the mean of
    objective_r over valid repeats; variance is the mean of variance_r.

    NaN policy: if fewer than ceil(k_search / 2) valid repeats remain, the trial
    is INVALID (objective = NaN) — surfaced in the leaderboard, excluded from
    ranking, never silently averaged from too little signal.

    Returns {objective, variance, n_valid_repeats, valid: bool, per_repeat: [...]}.
    """


def rank_trials(results: list[dict]) -> list[dict]:
    """Sort trials by (objective desc, variance asc); invalid trials sink to the
    bottom; ties broken deterministically by trial_id. Mirrors the v1.3 winner
    rule (roadmap 1.3.d): higher mean site AUC, then lower cross-site variance."""


def select_winner(ranked: list[dict]) -> dict | None:
    """Top valid trial, or None if every trial is invalid (all-degenerate sweep —
    the driver then errors loudly rather than emit a meaningless 'winner')."""
```

The mean-of-two-halves objective and the variance tie-break are the exact shape
of the v1.3 winner rule, exercised here on the two Geneva halves as a proxy (see
the §8 note that on IID halves the variance is mostly split noise).

### 4.3 `architecture/fed_stroke/hpo.py` — results artifact + leaderboard + narrowed ranges

```python
def build_results(meta, ranked, holdout) -> dict:
    """Assemble the results artifact (schema below); pure, unit-testable."""


def validate_results(results: dict) -> None:
    """Structural contract check before write; raises ValueError on bad shape
    (mirrors metrics.validate_metrics_artifact). Required top-level keys: meta,
    signal, trials, winner, holdout. `meta` MUST carry `data_provenance` ∈
    {"example-halves", "real-frozen-schema"} (the throwaway gate, §4.7) and
    `holdout_frac` / `holdout_partition_seed`; `signal` MUST carry
    `top_objective`, `median_objective`, `top_minus_median`,
    `winner_vs_runnerup_ci_overlap` (the resolving-power check, §4.3). A missing or
    unknown-valued key raises ValueError — the artifact never ships without its
    provenance stamp or its noise-tier signal."""


def render_leaderboard(results: dict) -> str:
    """Markdown leaderboard (mirrors metrics.render_run_report style): rank,
    trial_id, the six knobs, objective, variance, n_valid_repeats, and a ⚠ flag
    on invalid/degenerate trials so they stay visible."""


def summarize_top_ranges(ranked: list[dict], top_frac: float = 0.1) -> dict:
    """Per-hyperparameter min / max / mode across the top-decile valid trials —
    the narrowed search space handed to 1.1.b and the v1.3 re-run. Rendered to
    narrowed_ranges.md by the driver."""


def build_signal(ranked: list[dict]) -> dict:
    """Resolving-power check (outside-voice #2): does the ranking carry signal, or
    is the 'winner' noise? Over valid trials: top objective, median objective, their
    gap, and whether the winner's hold-out/objective CI overlaps the runner-up's.
    Feeds results.json["signal"] and a bold WARNING banner in narrowed_ranges.md when
    top≈median or the CIs overlap — so a human never reads a noise-tier winner as a
    real optimum. Small 2-feature schema + rare outcome makes this a real risk (§8)."""
```

**Results artifact schema** (`out/hpo/results.json`; enforced by
`validate_results`):

```
{
  "meta": {
    "strategy": "bagging",
    "federation": "local-deployment",
    "search_seeds": [1, 2, 3],
    "holdout_frac": 0.2,
    "holdout_partition_seed": 42,
    "operating_point": 0.5,
    "data_provenance": "example-halves",   // "example-halves" (V3, THROWAWAY) | "real-frozen-schema" (deliverable)
    "grid": { ...the grid actually swept... }
  },
  "signal": {                              // spread check (outside-voice #2): is the ranking above noise?
    "top_objective": 0.78, "median_objective": 0.71,
    "top_minus_median": 0.07,
    "winner_vs_runnerup_ci_overlap": true  // true ⇒ treat the "winner" as noise-tier; see narrowed_ranges.md
  },
  "trials": [
    {
      "trial_id": "<hash>",
      "params": { "total-trees": 40, "params.max-depth": 4, ... },
      "objective": 0.78, "variance": 0.0009,
      "n_valid_repeats": 3, "valid": true,
      "per_repeat": [
        { "split_seed": 1, "site_aucs": { "geneva_half_A.parquet": 0.79,
                                          "geneva_half_B.parquet": 0.77 } },
        ...
      ]
    },
    ...
  ],
  "winner": { "trial_id": "<hash>", "params": { ... } },
  "holdout": { "params": { ... }, "holdout_frac": 0.2, "holdout_partition_seed": 42,
               "site_metrics": { "<site>": { full compute_binary_metrics dict } } }
}
```

Round/site conventions match 1.c: site keys are parquet file names; a `null` AUC
means a single-class split.

### 4.4 `architecture/scripts/run_hpo.py` — orchestration driver

The only component that touches the process boundary. CLI:

```bash
python scripts/run_hpo.py \
    --federation local-deployment \      # swap for the cross-site federation in v1.3
    --strategy bagging \                 # bagging (headline) | cyclic
    --search-seeds 1 2 3 \               # repeated-CV seeds (train/valid sub-split of DEV, §4.5)
    --holdout-frac 0.2 \                 # patient-disjoint hold-out fraction (§4.5); 0.0 disables
    --grid grid.json \                   # optional; defaults to hpo.DEFAULT_GRID
    --max-trials 0 \                     # 0 = full grid; >0 = random subsample (escape hatch)
    --seed 0 \                           # RNG seed for the --max-trials random subsample (reproducible)
    --n-boot 0 \                         # OFFLINE bootstrap resamples during search (0 = skip, fast)
    --run-timeout 900 \                  # per-run wall-clock cap (s); a run exceeding it is killed → degenerate repeat (§4.4 Flow)
    --out-dir out/hpo \                  # resolved to an ABSOLUTE path before any run (§4.4 Flow)
    --resume \                           # skip any (trial, seed) whose final_model.json already exists
    --dry-run                            # print the trial plan + run count, launch nothing
```

Flow:

0. **Resolve `--out-dir` to an absolute path** (`Path(out_dir).resolve()`) as the
   very first step, and derive every per-trial directory from it. This is
   load-bearing, not cosmetic: the ServerApp process runs with a **different CWD**
   than the driver (it is launched by `run_local_federation.sh`, not from
   `architecture/`), and `save_final_model` resolves a relative `model-dir`
   against *the ServerApp's* CWD — so a relative `model-dir` writes the model
   somewhere the driver cannot find it. `run_local_federation.sh` documents this
   exact hazard for both `model-dir` and `metrics-dir` (its header, and the
   `${MODELS_DIR}` absolute-path example it prints on `start`). **All paths the
   driver passes into a run are absolute** (Decision 7, §9).
1. **Preflight.** `expand_grid` → `valid_trials`; log rejected cells with reason.
   Validate `0.0 < holdout-frac < 1.0` (a search needs a non-empty DEV *and* a
   non-empty HELD — hard error otherwise). A TCP connect to the SuperLink
   Control-API address (`[tool.fed_stroke.superlink].address` in `pyproject.toml`,
   `socket.create_connection((host, port), timeout=…)`) proves only the SuperLink
   listener is up — **not** that both SuperNodes are alive and correctly pinned. So
   preflight also fires **one canary run** (the smallest valid trial at the first
   search seed, subject to `--run-timeout`) before the sweep: a canary that fails
   or times out means a dead / mis-pinned node topology, and the driver aborts with
   a hint to run `scripts/run_local_federation.sh start` rather than launching ~972
   trials that would each fail or hang (outside-voice #1/#3, Decision 12 §9). On
   `--dry-run`, print the plan + total run count (`|valid trials| × |search-seeds|`)
   and skip the canary.
2. **Search.** For each valid trial × search seed:
   - **Resume (`--resume`):** if `<abs-out>/<trial_id>/seed<seed>/final_model.json`
     already exists, skip the `flwr run` and score the existing model directly.
     Models save per-trial-dir already, so a driver crash mid-sweep costs nothing
     to resume from (Decision 9, §9).
   - Otherwise fire one run (all paths ABSOLUTE, in-run CIs OFF):
     ```bash
     flwr run . <federation> --run-config \
       "train-method='<strategy>' total-trees=<t> params.max-depth=<d> \
        params.eta=<e> params.min-child-weight=<m> params.subsample=<s> \
        params.colsample-bytree=<c> split-seed=<seed> save-model=true n-boot=0 \
        model-dir='<abs-out>/<trial_id>/seed<seed>' \
        metrics-dir='<abs-out>/<trial_id>/seed<seed>'"
     ```
     - `n-boot=0` overrides the config default (1000): the HPO objective reads the
       **saved model offline** (§3.4), never the per-round metrics artifact, so the
       in-run bootstrap CIs would be pure discarded compute (per 160-tree bagging
       trial: ~80 rounds × 2 sites × 1000 resamples × 3 metrics). Offline CIs are
       controlled separately by `--n-boot` and are 0 during search, 1000 for the
       hold-out (Decision 7, §9).
     - **Per-trial `metrics-dir`** stops every run clobbering the shared default
       `out/metrics/bagging.json` (the canonical non-HPO artifact the logbook
       reads) and keeps the (discarded) per-run JSON isolated per trial.
   - The `--run-config` string is built by a pure helper (`build_run_config`,
     unit-tested §4.9) that quotes string values as TOML (`key='value'`), emits
     absolute paths, and produces **no shell metacharacters**. The driver invokes
     `subprocess.run([... , "--run-config", cfg_str], shell=False, timeout=<--run-timeout>)`
     — a **list, not a shell string** — so flwr's TOML parser (not a shell) consumes
     the quotes. `shell=True` would let the shell strip the quotes and flwr would
     then see bare tokens (Decision 7, §9).
   - **Timeout, not hang (outside-voice #1, Decision 12 §9).** `build_strategy`
     sets `min_available_nodes=num-sites` (server_app.py:63), so a `flwr run` whose
     federation lost a SuperNode mid-sweep **blocks indefinitely** waiting for node
     availability — it does not exit non-zero. Without a cap, the first dead node
     stalls the whole overnight sweep and `--resume` cannot help (nothing crashed).
     `--run-timeout` (default 900s ≫ a healthy run) bounds every run; on
     `TimeoutExpired` the driver **kills the process group** (the run's children,
     mirroring `run_local_federation.sh`'s `kill -- -pgid`), marks that repeat
     degenerate, logs it, and continues.
   - On success (or resume), load
     `<abs-out>/<trial_id>/seed<seed>/final_model.json` and score **both** halves
     with `score_booster_on_half(bst, half, operating_point, n_boot=<--n-boot>,
     boot_seed=0, split_seed=<seed>, holdout_frac=<--holdout-frac>,
     holdout_eval=False)` (§4.5). A **failed `flwr run`** (non-zero exit, timeout,
     or missing model) marks that repeat degenerate, is logged, and does **not**
     abort the sweep.
3. **Select.** `trial_objective` per trial → `rank_trials` → `select_winner`.
4. **Hold-out.** Re-run the winner once in HOLD-OUT mode
   (`holdout-frac=<--holdout-frac> holdout-eval=true`, `save-model=true`,
   `model-dir='<abs-out>/holdout'`, same absolute-path + `n-boot=0` in-run rules as
   step 2), so it **trains on DEV** and the saved model has never seen the HELD
   patients. Then score both halves
   **offline** with **full CIs** via `score_booster_on_half(..., n_boot=1000,
   holdout_frac=<--holdout-frac>, holdout_eval=True)` — the honest, patient-disjoint
   tuned-baseline number. The full CI is the one place bootstrap runs (§8:
   rank-comparable during search, calibrated only at the hold-out).
5. **Write** `leaderboard.md`, `results.json`, `narrowed_ranges.md`, and
   `tuned_<strategy>.toml` (§4.7) under `--out-dir`.

Sequential by default. Parallelism across trials is a compute-budget concern
deferred to 1.1.c (contending `flwr run`s on one loopback federation would also
muddy timing); noted, not built.

### 4.5 Split protocol — patient-disjoint hold-out + repeated CV on the remainder

**Why not "held-out seed 42".** An earlier design held out *seed 42* (a
train/valid split seed disjoint from the selection seeds). Eng review (outside
voice, Decision 11) showed that is **not leakage-free**: MC-CV splits overlap by
design, so seed-42's validation patients also appear in the selection seeds'
*train* sets — selection therefore saw the very patients scored at "hold-out", and
the reported no-DP baseline (which 1.1.b compares its DP arms against) is
optimistically biased. Seed-disjoint ≠ patient-disjoint. The fix reserves a
patient set, not a seed.

**Single split contract — one helper.** All three consumers
([`task.load_data_gva`](../../architecture/fed_stroke/task.py:72),
[`baseline.split_half`](../../architecture/fed_stroke/baseline.py:89),
[`baseline.score_booster_on_half`](../../architecture/fed_stroke/baseline.py:156))
delegate to a new `task.resolve_run_split(data, outcome, split_seed, holdout_frac,
holdout_eval)` so the train/valid contract lives in exactly one place (removes the
current duplication where `load_data_gva` and `split_half` each call
`generate_splits` independently). Its three modes:

```
holdout_frac == 0.0                    → FLAT (today's behavior):
                                         generate_splits(test_size=0.2, seed=split_seed)
holdout_frac  > 0.0, holdout_eval=False → SEARCH repeat:
                                         1. partition patients DEV / HELD by a FIXED
                                            HOLDOUT_PARTITION_SEED = 42 (module constant,
                                            never varied), test_size=holdout_frac, stratified
                                         2. sub-split DEV into train/valid by split_seed
                                            (test_size=0.2). HELD is never touched.
holdout_frac  > 0.0, holdout_eval=True  → HOLD-OUT report:
                                         train = all of DEV, valid = HELD (disjoint patients)
```

Patient-level partitioning reuses `generate_splits`' existing patient-ID logic
(split by `patient_id`, stratified on the reduced-per-patient outcome), so the
DEV/HELD boundary respects the multiple-admission guard already in `task.py:34-47`.

- **`holdout_frac` defaults to `0.0`** in config, so every existing federated run
  and test stays **byte-identical** (FLAT mode ≡ today's `generate_splits(...,
  seed=split_seed)`; with `split-seed` defaulting to 42, that is exactly the
  current split). `train_pooled_booster`'s internal `split_half` calls keep the
  defaults (`holdout_frac=0.0`), so 1.d is unaffected.
- **Selection** fires each of the K search seeds (default `1 2 3`) in SEARCH mode
  at `holdout_frac=0.2` → repeated (Monte-Carlo) CV *on DEV only*; the objective
  is the mean over repeats (§4.2). The search seeds no longer carry any
  hold-out meaning — the HELD patients are excluded from every one of them by the
  fixed partition, not by a seed convention.
- **Held-out report** runs the winner once in HOLD-OUT mode (`holdout_frac=0.2,
  holdout_eval=true`): train on DEV, score on HELD with full bootstrap CIs. HELD
  patients entered **no** selection run, so this is a genuine patient-disjoint
  hold-out.

Note: within DEV, the MC-CV sub-splits still overlap (deliberate variance
reduction for a small cohort — disjoint K-fold would leave each validation set
tiny and frequently single-class). Overlap inside DEV is fine; what makes the
final number honest is that HELD sits entirely outside DEV (§8).

### 4.6 Config keys (`pyproject.toml`)

Add three keys to `[tool.flwr.app.config]`. All three default to today's flat
single-split behavior, so no existing run changes (§4.5):

```toml
split-seed = 42       # train/valid split seed; HPO varies it for repeated CV on DEV (§4.5)
holdout-frac = 0.0    # patient-disjoint hold-out fraction; 0.0 = disabled (flat split, byte-identical to today)
holdout-eval = false  # true only for the winner's hold-out report run: train on DEV, evaluate on HELD (§4.5)
```

`resolve_run_split` reads these via `context.run_config` (the same channel
`client_app` already uses for `local-epochs` / `operating-point` / `n-boot`), so
they reach both `train` and `evaluate`. No other config surface: all six
hyperparameters already exist under `params.*` / `total-trees` and are overridden
per trial via `--run-config` (§4.4). `HOLDOUT_PARTITION_SEED` is a **module
constant**, not a config key — it must never vary, or the hold-out set would shift
between the search and the report.

### 4.7 Tuned-config emission

The driver writes `out/hpo/tuned_<strategy>.toml` — the winning `params` block +
`total-trees` as a ready-to-paste `[tool.flwr.app.config]` fragment — plus a
provenance header (strategy, search seeds, `holdout-frac` + `holdout_partition_seed`,
objective, the patient-disjoint hold-out per-site AUC-ROC with CIs, **and
`data_provenance`**). The header MUST carry `data_provenance`: only a
`real-frozen-schema` run is the deliverable input that 1.1.b re-runs with DP and
that the v1.3 cross-site re-run narrows around. An `example-halves` TOML (V3) is
plumbing-validation only and must never be consumed as tuned ranges (outside-voice
#5, Decision 13 §9).

### 4.8 Roadmap update (`architecture/roadmap.md`)

Decision 3 (user-approved): GVA HPO yields *provisional ranges*, re-tuned
cross-site once Shenzhen is online. The current v1.3 wording implies no re-tune.

- **Reword 1.3.c** so it references the **cross-site re-tuned** hyperparameters
  rather than "the tuned hyperparameters from v1.1/v1.2".
- **Add a new v1.3 item** immediately before 1.3.c — *Cross-site HPO re-run:
  using the same 1.1.a harness with `--federation` pointed at the real
  GVA↔Shenzhen deployment, re-tune within the narrowed ranges from 1.1.a on real
  cross-site data. Runs on the deployment link (Shenzhen data never leaves), so
  it is bounded to the narrowed ranges to keep the real-round budget tractable.*
- **Append a clause to 1.1.a** noting the harness is strategy-agnostic and its
  Geneva ranges are provisional pending this cross-site re-run.

### 4.9 Tests (`architecture/tests/test_hpo.py`, new)

All fast pure-logic (no `flwr run`; the E2E path is the §6 run matrix). Reuse the
`SimpleNamespace`/fake-record style from `tests/test_strategies.py` and the tiny
booster from `tests/test_client_boost.py`.

| Cases |
|---|
| **`expand_grid`:** cartesian product count (`len == prod of list lengths`); deterministic order; a single-value grid → one trial. |
| **`valid_trials`:** bagging with `num_sites=2, local_epochs=1` rejects odd `total-trees` (e.g. 39) and keeps even; cyclic keeps all; rejected trials carry a reason. Reuses `derive_num_rounds`. |
| **`build_run_config`:** override dict → exact `--run-config` string (dotted keys, string values quoted TOML-style, `save-model=true`, `n-boot=0`, ABSOLUTE per-trial `model-dir` **and** `metrics-dir`); assert the emitted paths are absolute and the string carries **no shell metacharacters** (guards the `shell=False` contract). |
| **`trial_id` determinism:** the same override dict yields the same id within a process AND across processes (compute it under two different `PYTHONHASHSEED` values via a subprocess, or assert it equals a pinned `hashlib` digest) — proves builtin `hash()` was not used; dict insertion order does not change the id. |
| **`--max-trials` subsample:** with a fixed `--seed`, the random subsample is reproducible (same seed → same trial set); a different seed generally differs; `max_trials >= |grid|` returns the full grid. |
| **resume skip (pure helper):** given a `tmp_path` where `<trial_id>/seed<seed>/final_model.json` exists for some (trial, seed), the resume predicate reports those as skippable and the rest as runnable. |
| **`trial_objective`:** known per-repeat site-AUC vectors → correct mean objective + mean variance; a repeat with a `None` site AUC is dropped; trial marked **invalid** when `< ceil(k/2)` valid repeats; a fully-valid trial reports `n_valid_repeats == k`. |
| **`rank_trials` / `select_winner`:** ordered by objective desc then variance asc; invalid trials sink; deterministic tie-break; all-invalid → `select_winner` returns `None`. |
| **`validate_results` + `render_leaderboard`:** a well-formed results dict passes and renders both valid and ⚠-flagged rows; a missing top-level key raises `ValueError`; **a dict missing `meta.data_provenance` (or with an unknown value), missing `meta.holdout_frac`, or missing any `signal.*` key raises `ValueError`** (guards the §4.7 throwaway gate and the §4.3 resolving-power stamp). |
| **`summarize_top_ranges` + `build_signal`:** on a hand-built ranking, `summarize_top_ranges` picks the correct min/max/mode per knob over the top decile; `build_signal` reports top>median with no CI overlap on a well-separated ranking, and flags overlap/`top≈median` on a flat one. |
| **`resolve_run_split` (regression + new — the §4.5 core):** (a) **byte-identity:** `holdout_frac=0.0, split_seed=42` reproduces today's split exactly (guards the refactor); (b) distinct `split_seed` values yield distinct DEV train/valid patient-ID sets; (c) **patient-disjointness:** at `holdout_frac=0.2`, the HELD set (holdout_eval=True) shares **no** `patient_id` with ANY search-seed's DEV train OR valid set — the leakage the design exists to prevent; (d) `HELD ∪ DEV` covers all patients and `HELD ∩ DEV = ∅`; (e) `score_booster_on_half`/`split_half` with matching params reconstruct the identical split the run used. |

## 5. Files to change

| File | Change |
|---|---|
| `architecture/fed_stroke/hpo.py` | **new** — grid + trial generation (§4.1), objective + selection (§4.2), results/leaderboard/narrowed-ranges (§4.3) |
| `architecture/scripts/run_hpo.py` | **new** — orchestration driver: subprocess `flwr run`, offline scoring, artifact writing (§4.4) |
| `architecture/fed_stroke/task.py` | **new** `resolve_run_split(...)` + `HOLDOUT_PARTITION_SEED` (the single split contract, §4.5); `load_data_gva` reads `split-seed` / `holdout-frac` / `holdout-eval` from run config and delegates to it |
| `architecture/fed_stroke/baseline.py` | `split_half` and `score_booster_on_half` gain `split_seed=42, holdout_frac=0.0, holdout_eval=False` params, delegating to `resolve_run_split` (§4.5); `train_pooled_booster` keeps defaults so 1.d is byte-identical |
| `architecture/scripts/eval_final_model.py` | add `--split-seed` / `--holdout-frac` / `--holdout-eval` args threaded into `score_booster_on_half` so it reconstructs the hold-out (HELD) split; V4 depends on this (§6). `--expected-trees` is already required |
| `architecture/pyproject.toml` | add `split-seed`, `holdout-frac`, `holdout-eval` config keys (§4.6) |
| `architecture/tests/test_hpo.py` | **new** — pure-logic tests + orchestration guarantees (paths/timeout/resume/trial_id) + the `resolve_run_split` byte-identity **and** patient-disjointness guards (§4.9) |
| `architecture/roadmap.md` | reword 1.3.c, add the cross-site HPO re-run item, annotate 1.1.a (§4.8) |

## 6. Run matrix / verification

Runs from `architecture/`. The 2-SuperNode `local-deployment` federation must be
up (`scripts/run_local_federation.sh start`).

| Step | Command | Expect |
|---|---|---|
| V1 | `uv run pytest` | green, incl. `test_hpo.py` and the seed-42 regression guard |
| V2 | `python scripts/run_hpo.py --strategy bagging --dry-run` | trial plan + run count (324 cells × 3 seeds = 972) printed; nothing launched; **0 rejected cells** at the default topology (every `total-trees` ∈ {40,80,160} is divisible by 2 — the rejection path itself is exercised by the §4.9 `valid_trials` test with `total-trees=39`, not by the default grid) |
| V3 | `python scripts/run_hpo.py --strategy bagging --max-trials 4 --search-seeds 1 2 3` | completes; writes `leaderboard.md`, `results.json`, `narrowed_ranges.md`, `tuned_bagging.toml` all stamped `data_provenance="example-halves"` (THROWAWAY — plumbing proof only, **not** the deliverable ranges); invalid/degenerate trials visible, not dropped |
| V4 | `python scripts/eval_final_model.py out/hpo/holdout/final_model.json --data out/geneva_half_A.parquet out/geneva_half_B.parquet --expected-trees <t> --holdout-frac 0.2 --holdout-eval` | per-site AUC-ROC matches the `holdout` block in `results.json` (shared scorer + same split params ⇒ exact match). `--expected-trees` is **required** by the script; the reconstruction flags make `eval_final_model.py` rebuild the **HELD** split (§5) |
| V5 | `python scripts/run_hpo.py --strategy cyclic --max-trials 4` | completes — proves the strategy-agnostic path; objective populated from both-halves scoring |

V1–V5 run against the **example** halves and prove the plumbing only — their
artifacts are `data_provenance="example-halves"` and are throwaway (§4.7). The
deliverable leaderboard, tuned config, hold-out numbers, and narrowed ranges come
from the same commands re-run on the **real frozen-schema** Geneva data (the
roadmap phase blocker, §2); only those are recorded in
[docs/logbook.md](../logbook.md) as the hand-off to 1.1.b.

## 7. Acceptance criteria

1. `pytest` suite passes, including `test_hpo.py`; at `holdout-frac=0.0,
   split-seed=42` (the defaults) `resolve_run_split` leaves the existing split
   byte-identical (regression guard green).
2. A coarse bagging sweep runs end-to-end against the standing federation and
   writes all four artifacts; invalid/degenerate trials appear in the leaderboard
   with a ⚠ flag rather than being silently dropped.
3. The objective is computed by **offline both-halves scoring** (not the
   per-round artifact); selection = mean site AUC-ROC, tie-break lower cross-site
   variance.
4. Selection uses repeated CV on the DEV remainder; the winner's
   **patient-disjoint** hold-out per-site AUC-ROC (with CIs, on HELD) is reported
   and reproduced by `eval_final_model.py` (with `--holdout-frac`/`--holdout-eval`)
   on the same saved model to floating-point tolerance. A test asserts HELD shares
   no `patient_id` with any selection split.
5. The cyclic sweep runs through the same harness (strategy-agnostic support
   proven); the headline 1.1.a run stays bagging.
6. `narrowed_ranges.md` and `tuned_bagging.toml` are produced as the hand-off to
   1.1.b and the v1.3 re-run.
7. The roadmap shows the new cross-site HPO re-run item, a reworded 1.3.c, and
   the 1.1.a provisional-ranges annotation.
8. First-pass results recorded in [docs/logbook.md](../logbook.md).
9. Every path the driver passes into a run is absolute; the search run-config
   carries `n-boot=0` and a per-trial `metrics-dir` (the shared
   `out/metrics/bagging.json` is never touched by the sweep); `--resume` skips
   completed `(trial, seed)` pairs; and `trial_id` is identical across processes
   (regression-guarded in `test_hpo.py`, §4.9).
10. No single run can stall the sweep: every `flwr run` is bounded by
    `--run-timeout` and a timeout/failed run degrades one repeat (killed process
    group), never blocks; a preflight **canary run** aborts the sweep up front if
    the node topology is dead/mis-pinned (§4.4).
11. Only a `data_provenance="real-frozen-schema"` run's artifacts are treated as
    the deliverable hand-off; `example-halves` (V1–V5) artifacts are throwaway
    plumbing proofs and are never consumed as tuned ranges (§4.7).

## 8. Risks & notes

- **Geneva ranges are provisional.** Shenzhen's ~40k cohort (vs ~1k/node at
  Geneva) and different distribution will move the optimum; the same harness is
  re-run cross-site in v1.3 (§4.8). This is the central framing, not a footnote —
  1.1.a narrows the space, it does not finalize hyperparameters.
- **Small N + rare outcome → NaN AUCs, and the patient-disjoint hold-out shrinks
  the pool further.** Reserving `holdout-frac=0.2` of patients as HELD leaves ~80%
  as DEV; a search repeat then validates on ~20% of DEV (~16% of a ~1k-row half)
  of the rare `3M Death` outcome — more likely to land single-class (null AUC, per
  1.c) than the old flat 20%. The NaN policy (§4.2) drops degenerate repeats and
  marks a trial invalid below quorum, keeping degeneracy **visible** in the
  leaderboard rather than averaging a decision out of too little signal; repeated
  multi-seed CV (§4.5) means one unlucky split does not decide a trial. This is the
  honest cost of a leakage-free hold-out (Decision 11) on a small cohort — accepted
  because 1.1.a only *narrows* ranges (provisional, re-tuned at Shenzhen scale).
- **Resolving power may be below noise — surfaced, not hidden.** Over two features
  and ~1k rows, AUC gaps across the 324 cells can be smaller than MC-CV split
  noise. `build_signal` (§4.3) reports top-vs-median objective and winner/runner-up
  CI overlap, and `narrowed_ranges.md` carries a bold warning when the ranking is
  noise-tier — so the "winner" and narrowed ranges handed to 1.1.b are never read
  as a real optimum when they are not (outside-voice #2). The ranges are
  provisional regardless (re-tuned cross-site).
- **Two IID Geneva halves** ⇒ the cross-site variance used in the tie-break is
  mostly split noise here, not genuine site heterogeneity (the 1.c §8 trap). The
  variance tie-break is a v1.3-shaped rule exercised on a proxy; its real bite
  arrives with Shenzhen. Reading a small A-vs-B gap as a site effect is the trap;
  the objective's job here is ranking hyperparameters, not measuring sites.
- **Two-feature schema weakens `subsample` / `colsample-bytree`.** Their effect
  is coarse over two columns (§3.6); the grid keeps them for the richer real
  schema and because DP pins `subsample = 1.0` (1.1 prereq §3.5), so their
  landscape matters to 1.1.b even if it is flat now.
- **Runtime.** The coarse default grid (324 cells) × K search seeds is ~972
  loopback runs; at small Geneva data each run is seconds-to-low-tens-of-seconds,
  i.e. an overnight-scale one-off. Three levers keep it tractable: `n-boot=0`
  during search removes the dominant per-round cost (the discarded in-run
  bootstrap — §4.4 step 2); `--resume` makes a crash mid-sweep free to restart;
  and `--max-trials` (seeded random subsample) caps the run count directly.
  Matching the search *method* to the compute budget (Optuna/TPE) is **1.1.c**,
  explicitly not this spec.
- **Standing-federation coupling + the hang failure mode.** Bringing the
  federation up/down stays `run_local_federation.sh`'s job (no lifecycle
  management inside `run_hpo.py`). The dangerous case is not a *down* federation
  (caught by the preflight canary) but one that **loses a SuperNode mid-sweep**:
  `min_available_nodes=num-sites` makes the next `flwr run` block forever, not
  fail. `--run-timeout` converts that hang into a degenerate repeat (kill the
  process group, log, continue), so no single dead node can stall ~972 runs
  (Decision 12). A failed/timed-out individual run degrades one repeat, never the
  sweep.
- **Execution substrate reaffirmed over a faster shortcut.** Eng review raised
  tuning on the in-process pooled booster (ms/trial) for this no-DP single-site
  pass instead of ~972 federated subprocess runs. Rejected (Decision 4 kept): the
  harness exists to *be* the v1.3 cross-site/DP code path (roadmap "no throwaway
  code"), and 1.d's 3-AUC-point federated≈pooled tolerance is too loose to trust
  pooled *ranking* to reproduce federated ranking. The compute cost is mitigated by
  `n-boot=0` / `--resume` / `--max-trials`; a pooled fast-path is captured as a
  1.1.c compute-budget question (Decision 14).
- **Cross-site substrate cost.** The v1.3 re-run drives the **real deployment
  link** (Shenzhen data never leaves its site) — many real federated rounds,
  partner-in-the-loop, internationally. The harness is substrate-agnostic so no
  rewrite is needed, but the re-run is deliberately bounded to 1.1.a's narrowed
  ranges to keep the real-round budget tractable (§4.8).
- **Objective is rank-comparable, not calibrated.** During search `--n-boot 0`
  skips CIs for speed; only the hold-out report computes full CIs. Trials are
  compared by point AUC, which is adequate for ranking; the hold-out CI is what
  carries uncertainty into the logbook.

## 9. Decisions locked with the user (audit trail)

Recorded so the reasoning is not lost; if this section ever disagrees with
§1–§8, the body wins.

1. **Reusable, strategy-agnostic harness** (bagging + cyclic) from day one; the
   1.1.a *headline run* stays bagging per the roadmap, cyclic revalidation stays
   v1.3.
2. **Geneva first pass yields provisional ranges**, not final hyperparameters.
3. **Cross-site HPO re-run added to v1.3** with a roadmap edit (reword 1.3.c +
   new re-run item), because GVA-tuned hyperparameters will move at Shenzhen's
   scale/distribution.
4. **Execution substrate = subprocess `flwr run`, federation-parameterized**, on
   the deployment topology (loopback `local-deployment` now, real cross-site
   link in v1.3). Simulation was rejected: `local-simulation` is
   `num-supernodes=1` and 2-site distinct-file loading only exists in the
   deployment topology (§3.2).
5. **Objective = mean site AUC-ROC, tie-break lower cross-site variance**,
   mirroring the v1.3 winner rule, computed by offline both-halves scoring of the
   saved model (§3.4) so bagging and cyclic are comparable.
6. **Repeated multi-seed CV for selection + a patient-disjoint hold-out** for the
   final report (superseded from the original "held-out seed 42" by Decision 11).
   The pipeline change is three `split-*` config keys and a single split helper
   `resolve_run_split`, all defaulting to today's flat split so existing runs are
   byte-identical (§4.5).
7. **Output-path & in-run-CI run-config contract** (eng review, §4.4). Every path
   the driver passes into a `flwr run` is **absolute** — the ServerApp CWD differs
   from the driver's, and `save_final_model` resolves a relative `model-dir`
   against the ServerApp CWD, so a relative path writes the model where the driver
   cannot find it (`run_local_federation.sh` documents this for both `model-dir`
   and `metrics-dir`). `metrics-dir` is per-trial so runs never clobber the shared
   `out/metrics/bagging.json`. The search run-config sets `n-boot=0`: the HPO
   objective reads the saved model offline (Decision 5), never the per-round
   artifact, so in-run bootstrap CIs are pure discarded compute. `build_run_config`
   emits no shell metacharacters and the driver runs `subprocess` with a **list and
   `shell=False`** so flwr's TOML parser (not a shell) consumes the value quotes.
8. **`total-trees` grid = [40, 80, 160]** (eng review, §4.1). Three points, not
   two, on the axis that realizes the `num_server_rounds` knob — the roadmap's
   first-named and highest-impact knob — so `summarize_top_ranges` can narrow the
   round count instead of only choosing an endpoint. All three divide evenly at the
   default bagging topology, so none is rejected by `valid_trials`.
9. **Idempotent `--resume`** (eng review, §4.4). A driver crash mid-sweep (the
   default is ~972 unattended runs) must not lose finished work; a (trial, seed)
   whose `final_model.json` already exists is skipped and scored directly. Free,
   because models already save to per-trial dirs.
10. **`trial_id` via `hashlib` over canonical JSON; seeded `--max-trials`** (eng
    review, §4.1/§4.4). Builtin `hash()` is `PYTHONHASHSEED`-salted per process, so
    it would give a trial a different directory on a resume and silently re-run
    everything; a `hashlib` digest of `json.dumps(..., sort_keys=True)` is stable
    across processes. The `--max-trials` random subsample takes a `--seed` so the
    sampled trial set is reproducible.
11. **Patient-disjoint hold-out, not a held-out seed** (eng review outside voice,
    §4.5 — supersedes the seed-42 hold-out in Decision 6). MC-CV splits overlap, so
    a "held-out seed" still shares patients with the selection seeds' train sets →
    the no-DP baseline 1.1.b compares against would be optimistically biased. A
    fixed-partition patient set (`holdout-frac`, `HOLDOUT_PARTITION_SEED=42`) is
    reserved before any selection and reused via `generate_splits`' patient-ID
    logic; accepted cost is a smaller DEV selection pool on a small cohort (§8).
12. **Per-run timeout + preflight canary** (eng review outside voice, §4.4). Because
    `min_available_nodes=num-sites` makes a run *hang* (not fail) when a SuperNode
    dies mid-sweep, `--run-timeout` bounds every run and converts a hang into a
    degenerate repeat; a preflight canary run aborts the sweep up front on a
    dead/mis-pinned topology (the SuperLink TCP probe alone cannot see node
    liveness). This is what makes the "unattended overnight" claim real.
13. **`data_provenance` stamp gates the deliverable** (eng review outside voice,
    §4.7). The V1–V5 example-halves artifacts are throwaway plumbing proofs; only a
    `real-frozen-schema` run's `tuned_*.toml` / `narrowed_ranges.md` may be consumed
    by 1.1.b, so example-subset ranges cannot silently leak into the DP sweep.
14. **Kept federated substrate over pooled-tuning shortcut; deferred the fast-path
    to 1.1.c** (eng review outside voice tension #6, §8). Pooled in-process tuning
    would be far faster for this no-DP pass but builds a second code path not reused
    in v1.3 and the federated≈pooled tolerance is too loose to trust its ranking.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 7 issues (2 P1, 5 P2); 7 resolved, 0 unresolved; 0 critical gaps |
| Outside Voice | Claude subagent (Codex unavailable) | Independent 2nd opinion | 1 | issues_found | 8 findings; 7 actioned, 1 (variance tie-break) already documented; 0 unresolved |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

Eng review verified every code anchor against the installed sources (`flwr==1.31.0`,
`fed_stroke`). All signatures, line numbers, and behavioral claims (§3) checked out.
The 7 primary findings were all in the orchestration layer (§4.4) and
reproducibility, not the FL/metric core: P1 relative `model-dir` (would break V3/V4
across the ServerApp CWD boundary), P1 `metrics-dir` clobbering the canonical
artifact, P2 discarded in-run bootstrap CIs, P2 subprocess quoting, P2 undefined
reachability check, P2 `trial_id` non-determinism, P2 no crash-resume. Folded into
§4.1/§4.4/§4.9 (Decisions 7–10). `total-trees` grid widened to three points
(Decision 8).

- **OUTSIDE VOICE (Claude subagent):** 8 findings. Actioned: per-run
  timeout + preflight canary against the `min_available_nodes` **hang** (Decision
  12); **patient-disjoint hold-out** replacing the leaky held-out-seed (Decision
  11, the substantive statistical fix); `data_provenance` throwaway gate (Decision
  13); resolving-power `build_signal` check (§4.3/§8); V4 command + `eval_final_model.py`
  reconstruction flags (§5/§6). One finding (variance tie-break = split noise on 2
  IID halves) was already documented in §8. 
- **CROSS-MODEL TENSION — execution substrate (resolved):** outside voice proposed
  pooled in-process tuning over 972 federated runs; user kept the federated
  substrate (Decision 4/14) — the harness must be the v1.3 code path and the
  federated≈pooled tolerance is too loose to trust pooled ranking. Fast-path
  deferred to 1.1.c.
- **VERDICT:** ENG CLEARED — ready to implement. CEO/Design/DX not run (Design n/a — no UI).

NO UNRESOLVED DECISIONS
