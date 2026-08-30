# Spec 1.1.b — DP sweep driver (`run_dp_sweep.py`): the A→B→C comparator sweep at ε ∈ {1, 3, 5, 10}

Implements the run-and-report machinery for roadmap item 1.1.b
([architecture/roadmap.md](../../architecture/roadmap.md)):

> 1.1.b Run the sweep both without DP and with DP at each pilot ε ∈ {1, 3, 5, 10},
> δ = 1e-5 (builds on the wired DP path from 1.1.a′).

and for acceptance items 4–5 of the remediation roadmap
([1_1_a_doubleprime_dp_remediation.md](1_1_a_doubleprime_dp_remediation.md)):

> 4. Only then does 1.1.b run on real frozen-schema Geneva data — in the **federated
>    2-node topology but Geneva-only** […] producing the A→B→C decomposition (R9).
> 5. The **DP-vs-no-DP go/no-go decision** (is the B→C privacy cost acceptable at a
>    usable ε?) is made on that Geneva-only federation and recorded in `docs/logbook.md`
>    **before Shenzhen is integrated into the network**.

Builds directly on the landed 1.1.a″ remediation (R1–R7 + R9, logbook 2026-07-23), the
1.1.a HPO harness ([1_1_a_hpo_harness.md](1_1_a_hpo_harness.md)) whose driver pattern it
mirrors, and the report template
[docs/templates/1_1_b_report_skeleton.md](../templates/1_1_b_report_skeleton.md).

**Framing.** All of the hard machinery already exists: the three comparator arms are
reachable via run-config (`dp.enabled` / `dp.mechanism`), the ledger appends and composes,
the precondition gate fail-closes, the offline scorer handles both model formats. What is
missing is the *orchestration + reporting* layer: one driver that fires the six matched
federated runs on one shared config, re-scores every saved model on both halves, extracts
the accounting from the models and the ledger, and emits the filled 1.1.b report the
go/no-go decision is made on. This spec is the sole reference for that driver.

## 1. Goal & scope

**In scope**

- `architecture/scripts/run_dp_sweep.py` — orchestration driver (the process boundary:
  fires `flwr run`, reads files, writes artifacts).
- `architecture/fed_stroke/dpsweep.py` — all pure logic (arm plan, run-config
  construction, delta computation, accounting extraction/validation, report rendering),
  unit-testable without a federation, mirroring the `hpo.py` / `run_hpo.py` split.
- `architecture/scripts/flwr_proc.py` — shared driver commons extracted from
  `run_hpo.py` (`_flwr_bin`, `_run_flwr`, the pyproject fact helpers, **and `_json_safe`**);
  `run_hpo.py` is refactored to import it, behavior-identical. `_json_safe` is extended
  in the extraction to map **non-finite floats — `NaN` *and* `±inf` — to `None`** (today it
  handles NaN only). This is a behavior-identical superset for run_hpo (its artifacts never
  contain inf) and a **defense-in-depth** layer for the sweep: real saved DPBooster meta is
  already inf-scrubbed by `to_json_bytes` (§4.6), so this widening guards only hand-built
  fixtures and any future non-scrubbed path — cheap insurance, not the primary fix.
- A small extraction in `fed_stroke/baseline.py`: the format-sniffing model loader
  currently inlined in `eval_final_model.py` becomes `load_saved_booster()` so the sweep
  driver, `eval_final_model.py`, and tests share one loader. **Return contract** (pinned so
  this spec is self-contained): `load_saved_booster(path) -> LoadedModel`, a
  `NamedTuple(booster, fmt: str, n_trees: int)` where — mirroring the current
  `eval_final_model.py` sniff — a payload whose top-level `format` starts with `dp-gbdt-`
  is loaded via `DPBooster.from_json_bytes` (`fmt` = that verbatim string, e.g.
  `"dp-gbdt-v2"`; `n_trees = len(booster.trees)`; **no** `set_param` — DPBooster has none),
  and anything else via `xgb.Booster().load_model` with `set_param({"eval_metric": "auc"})`
  applied (`fmt = "xgboost-json"`; `n_trees = booster.num_boosted_rounds()`). `eval_final_model.py`
  loses its inline sniff and consumes the tuple; the sweep driver's §4.4 checks read
  `.fmt` / `.n_trees` off it.
- The **run matrix**: sequential federated runs of arm A (stock XGBoost), arm B (DP
  learner, identity mechanism), and arm C (DP learner, gaussian) at each
  ε ∈ {1, 3, 5, 10}, all on one shared config, against a standing federation.
- **Offline re-scoring** of every saved model on both halves' validation splits with
  bootstrap CIs, through the single shared scorer `score_booster_on_half`.
- **Report emission**: a filled instance of the 1.1.b skeleton (all computable fields),
  carrying the B→C privacy-cost deltas, the A→B learner-cost delta, and — per the R6
  reporting rule — both the per-run ε and the composed ledger-total ε, plus
  `results.json` (machine-readable sweep artifact).
- Tests (`architecture/tests/test_dpsweep.py`).

**Out of scope**

- **DP-aware hyperparameter re-tuning.** The roadmap's observation that "DP changes the
  utility landscape — `max_depth` and `min_child_weight` trade off differently under
  histogram noise" is handled by re-running the existing 1.1.a harness with `dp.*`
  overrides in its grid; that exploration belongs to 1.1.c/1.1.d and is deliberately NOT
  this driver. This driver runs **one shared config** — a grid × ε product on real data
  would explode the R6 ledger composition (every C cell spends ε on the same patients)
  and is exactly what the fixed-config A→B→C design avoids.
- Paired-bootstrap CIs on the B→C / A→B *deltas* (per-arm CIs are reported; a paired
  bootstrap over the shared validation rows is a cheap future upgrade if the deltas land
  near the CI width — noted in §8, not built now).
- Privatizing validation metrics (R5 keeps them inside the trust boundary; deferred to
  the earliest metric that must cross, likely v1.3).
- Cyclic revalidation (Phase v1.3), Shenzhen anything, changes to `accounting.py`
  (frozen), changes to the DP mechanism/learner/ledger (all landed in 1.1.a″).

**Gates this driver does NOT lift.** 1.1.b on real patient data stays blocked until the
R8 packet re-review is signed off by both reviewers and recorded in `docs/logbook.md`,
and until real frozen-schema Geneva data exists (preprocessing track). The driver is
built and rehearsed on the example halves now (provenance-gated, §4.2); the real run is
a flag flip after the gates clear.

## 2. Dependencies

- Standing federation: `scripts/run_local_federation.sh start` (1 SuperLink /
  2 SuperNodes over TLS, node_A/node_B pinned to the two Geneva halves via
  `[tool.fed_stroke.nodes]`). The driver never manages federation lifecycle (same
  contract as `run_hpo.py`).
- Landed 1.1.a″ machinery (all in-tree as of 2026-07-23):
  - arm selectors: `dp.enabled=false` → stock `xgb.train`; `dp.enabled=true` +
    `dp.mechanism='identity'` → arm B; `+ dp.mechanism='gaussian'` +
    `dp.target-epsilon` → arm C (`client_app.py` train branch).
  - `fed_stroke/dp/ledger.py`: `append_entry` fires on each site's first train call of a
    real-frozen-schema C run; `ledger_total()` composes per site; `config_hash()`.
  - `validate_dp_preconditions()` (R7) fail-closes every DP round.
  - `score_booster_on_half` scores both `xgb.Booster` and `DPBooster` via object-type
    dispatch; `eval_final_model.py` carries the `dp-gbdt-*` format sniff.
  - Missingness variant 1 (sentinel/missing-bin, `REMOVE-IF-NO-DP`) applied at the
    loader for all arms; model format `dp-gbdt-v2`.
  - Report template `docs/templates/1_1_b_report_skeleton.md`.
- `run_hpo.py` provides the driver idioms this spec reuses verbatim: `--stream`ed
  `flwr run` with wall-clock timeout + process-group kill, TCP preflight probe,
  absolute artifact paths, `hpo.build_run_config` for run-config string construction,
  `--resume`, `--dry-run`.

## 3. Facts that drive the design

1. **Six runs, one config.** The comparator set is A, B, C@10, C@5, C@3, C@1 — 2 + |ε|
   federated runs total. At ~4 min/run polling overhead (project memory) plus training,
   the whole sweep is well under an hour; sequential execution against the standing
   federation is the right shape (no parallelism machinery, no run-queue).
2. **Matched hyperparameters are a hard R9 requirement.** The DP learner forces
   q = 1.0 and ignores `subsample`/`colsample_bytree`; arm A must therefore run at
   `subsample = colsample_bytree = 1.0` or A→B conflates sampling with learner cost.
   `base_score` is already explicit and identical everywhere (pyproject
   `params.base-score = 0.5`, R7-enforced).
3. **The production ε path is FLAT-mode.** The R4 hash split at `holdout_frac = 0.0`,
   `test_size = 0.2` is the accounted configuration (spec 1.1.a″ R4); the skeleton's
   cohort section assumes it. The driver pins `holdout-frac=0.0`, `holdout-eval=false`
   for every arm and threads one `--split-seed` (default 42) into both the runs and the
   offline scoring, so all six arms train and validate on byte-identical row sets.
4. **ε provenance is the model, composition is the ledger.** Each DP model carries its
   run-calibrated accounting in `meta` (`reported_epsilon`, `noise_multiplier`,
   `num_releases`, `per_site_trees`) — read per-run ε and σ back from the artifact, never
   from the intended config (the `check_fed_vs_pooled` idiom: checked invariant, not
   assumption). The ledger (`out/dp_ledger.jsonl`, repo-root-relative = the loopback
   SuperNodes' CWD) is the source for composition; `ledger_total()` is the only composer.
   Ledger `site` keys are `Path(data-path).name` (i.e. `geneva_half_A.parquet`), not
   `node_A` — the report maps them verbatim with the node label alongside.
5. **Ledger entries are intent-to-spend.** A C run that crashes after round 1 has
   ledgered (and spent) its budget; a `--resume` re-run appends again. Both entries are
   correct and both compose (conservative in the right direction). The report's composed
   total therefore covers *all* recorded real-data spend to date, including failed
   attempts and any prior sweeps — that is the R6 semantics, not a bug.
6. **The provenance key gates everything sharp.** `data-provenance='real-frozen-schema'`
   is what arms the ledger append and rejects the insecure-test hatch. A rehearsal run at
   the default `example-halves` exercises the full driver path except ledger writes —
   so ledger assertions in the driver must be conditional on provenance, and the
   rehearsal report must be unmistakably marked as not-a-real-ε-claim.
7. **The skeleton's σ column is config-dependent.** Its constants
   (51.171/18.888/12.050/6.699) assume k = 2·D·T_site = 160 (max_depth 4, 20 trees/site).
   A tuned config with different depth or tree budget calibrates different σ. The driver
   fills σ from each model's meta — it must never copy the skeleton constants.
8. **`build_strategy` sets `min_available_nodes = num-sites`** — a run against a
   degraded federation blocks forever rather than failing; the wall-clock timeout +
   process-group kill from `run_hpo._run_flwr` is load-bearing and is reused.

## 4. Design

### 4.1 Arm plan (`fed_stroke/dpsweep.py`)

```python
ARM_A = "A"          # stock xgb.train           (dp.enabled=false)
ARM_B = "B"          # DP learner, identity      (no noise, no ε)
ARM_C = "C"          # DP learner, gaussian at ε

def arm_plan(epsilons: list[float]) -> list[dict]:
    """The ordered run matrix. Each entry:
    {"arm": "A"|"B"|"C", "label": str, "epsilon": float|None, "dp_overrides": dict}
    Order: A, B, then C at DESCENDING ε.
    """
```

- Labels double as artifact directory names: `armA`, `armB`, `armC_eps10`, `armC_eps5`,
  `armC_eps3`, `armC_eps1` (ε formatted with `g`, so 0.5 would be `armC_eps0.5`).
- `dp_overrides` per arm:
  - A: `{"dp.enabled": False}`
  - B: `{"dp.enabled": True, "dp.mechanism": "identity"}`
  - C@ε: `{"dp.enabled": True, "dp.mechanism": "gaussian", "dp.target-epsilon": ε,
    "dp.delta": δ}`
- **Order rationale:** A and B spend no ε, so every infrastructure failure mode
  (federation down, config typo, model-write race) surfaces before any privacy budget is
  spent. If arm A fails, the driver aborts the whole sweep (arm A is the built-in canary
  — no separate canary run, unlike `run_hpo.py`, because here the first run is already
  the cheapest and spends nothing). C order is immaterial to composition (RDP composition
  is commutative); descending matches the skeleton table's row order.

### 4.2 Shared config and run-config construction

One shared hyperparameter set for all six runs, resolved in this precedence:

1. **Base:** `[tool.flwr.app.config]` from `pyproject.toml` (the committed defaults).
2. **Optional `--config <tuned.toml>`:** a `tuned_bagging.toml` emitted by the 1.1.a
   real-data HPO run. The driver parses its `[tool.flwr.app.config]` table and adopts
   `total-trees`, `params.max-depth`, `params.eta`, `params.min-child-weight`. Its
   `params.subsample` / `params.colsample-bytree` are read but **discarded** (see next
   point), with a printed note when they differ from 1.0.
3. **Forced, always, for every arm:** `params.subsample = 1.0`,
   `params.colsample-bytree = 1.0` (fact §3.2 / R9 matching). Never optional, never a
   flag.

The per-arm `--run-config` string is built by reusing `hpo.build_run_config(override,
strategy, split_seed, model_dir, metrics_dir, holdout_frac=0.0, holdout_eval=False,
save_model=True, n_boot=0)` with
`override = {**shared_overrides, **arm["dp_overrides"], "data-provenance": provenance}`,
where `shared_overrides` carries the tuned/forced keys above. This inherits, for free:
TOML-quoted scalars, sorted dotted keys, absolute `model-dir`/`metrics-dir`
(`<out_dir>/<label>/`), `save-model=true`, and `n-boot=0` in-run (offline scoring owns
the CIs — in-run bootstrap would be discarded compute).

Notes:
- `dp.delta` is threaded from `--delta` (default 1e-5, the roadmap's pinned value) into
  both the C run configs and the `ledger_total()` conversion, so run and composition use
  one δ by construction.
- **`dp.ledger-path` is threaded as an ABSOLUTE path** into every arm's run-config
  override (from `--ledger-path`, resolved to absolute against the repo root in Step 0).
  This is load-bearing: `client_app` writes the ledger to
  `context.run_config.get("dp.ledger-path", "out/dp_ledger.jsonl")` **relative to the
  SuperNode's CWD**, while the driver *reads* the ledger at `--ledger-path` — the two
  coincide only at the default. Threading one absolute path makes the SuperNode write
  exactly where the driver reads, independent of the SuperNode's CWD, so a non-default
  `--ledger-path` (e.g. isolating a rehearsal ledger) cannot silently split write from
  read. A and B carry the same override but never append (identity/non-DP are
  provenance-and-mechanism gated in `client_app`), so threading is harmless for them.
- `total-trees` divisibility is validated upfront via
  `derive_num_rounds(strategy, total_trees, num_sites, local_epochs)` (imported from
  `fed_stroke.server_app`, same as `hpo.py` does) — fail before run 1, not at run 3.
- The identity arm deliberately carries no `dp.target-epsilon` override: the mechanism
  ignores it, and its absence keeps the arm-B run-config string minimal and readable.

### 4.3 Driver flow (`scripts/run_dp_sweep.py`)

```
Step 0  parse args; resolve absolute out_dir; load pyproject facts
        (half paths, operating point, superlink host:port, num-sites, local-epochs)
        via scripts/flwr_proc.py
Step 1  build shared config (§4.2); build arm plan (§4.1); validate divisibility;
        --dry-run: print the 6-run plan with the EXACT run-config strings, exit
Step 2  preflight: TCP probe the SuperLink Control API; snapshot ledger entry count
Step 3  for each arm, in plan order:
          if --resume and <out_dir>/<label>/final_model.json exists: skip the run
          else: fire flwr run . <federation> --stream --run-config <cfg>
                (wall-clock --run-timeout, process-group kill on expiry)
          load the saved model (load_saved_booster); run per-arm assertions (§4.4)
          if arm A failed → abort the sweep (nothing spent yet)
          if a C arm failed → record the arm as DEGENERATE, continue (its ledger
                entry may already exist — intent-to-spend; the report shows it)
          offline-score the model on BOTH halves (§4.5)
Step 4  accounting: read the ledger delta (real provenance only, §4.6);
        ledger_total(ledger_path, delta) → composed ε per site
Step 5  compute deltas (§4.5); assemble results dict; write
        <out_dir>/results.json  and  <out_dir>/report_1_1_b.md (§4.7)
```

- The driver runs from anywhere; every path it hands a run or opens itself is absolute
  (Decision 7 of the HPO spec, inherited).
- Sequential only. No parallel runs — the federation is a single shared topology and the
  ledger append is per-run-first-round; interleaving buys nothing and risks confusion.
- **Real-provenance guard:** `--data-provenance real-frozen-schema` additionally
  requires `--operator <name>` and `--gate-ack "<free text>"`. The driver cannot verify
  the R8 sign-off; `--gate-ack` forces the operator to type an affirmative statement
  (e.g. `"R8 sign-off recorded in docs/logbook.md 2026-08-XX"`) which is stamped
  verbatim into the report and `results.json`. Refuse to start without it. On the
  default `example-halves` both flags are optional (`--operator` defaults to
  `getpass.getuser()`).

### 4.4 Per-arm post-run assertions (fail loud, in the driver)

After each run, before scoring. Every check reads the `LoadedModel` tuple from
`load_saved_booster` (§1): `.fmt`, `.n_trees`, `.booster`. Let `D = fed_run_config
params.max-depth` (read back from the model) and `per_site =
per_site_tree_budget(strategy, num_rounds, num_sites, local_epochs)` — imported verbatim
from `fed_stroke.dp.boost`, with `num_rounds = derive_num_rounds(strategy, total-trees,
num-sites, local-epochs)` (§4.2). This is the *independent* source for the gaussian
release-count invariant (a real check, not a restatement of the meta's own
`per_site_trees`). **Do NOT use `total-trees // num-sites`**: it equals `per_site` only for
bagging; for cyclic with `num_rounds % num-sites != 0` it under-reports the busiest site's
release count and the check false-fails (`boost.per_site_tree_budget` docstring, Risk 1).
**Meta key locations** (verified against
`dp/boost._mechanism_meta`): `num_releases`, `noise_multiplier`, `reported_epsilon`,
`per_site_trees` are **top-level** meta keys; `mechanism` is **nested** at
`meta["dp"]["mechanism"]`. For identity, `noise_multiplier`/`reported_epsilon` are already
`null` on disk (scrubbed by `DPBooster.to_json_bytes`, §4.6) — read them as `None`, never `inf`.

| Check | A | B | C@ε |
|---|---|---|---|
| `final_model.json` exists, loads via `load_saved_booster` | ✓ | ✓ | ✓ |
| `.n_trees == total-trees` | ✓ | ✓ | ✓ |
| `.fmt == "xgboost-json"` | ✓ | — | — |
| `.fmt == "dp-gbdt-v2"` | — | ✓ | ✓ |
| `fed_run_config` read-back shows `subsample == colsample_bytree == 1.0` | ✓ | ✓ | ✓ |
| meta: `num_releases == 0`; `reported_epsilon` and `noise_multiplier` are `None` (identity's `inf` is scrubbed to `null` by `to_json_bytes`) | — | ✓ | — |
| meta: `meta["dp"]["mechanism"] == "gaussian"`, σ (`noise_multiplier`) finite > 0, `per_site_trees == per_site`, `num_releases == 2·D·per_site` | — | — | ✓ |
| meta: `reported_epsilon` within 1e-3 relative of the target ε | — | — | ✓ |

(`fed_run_config` rides in `set_attr` for XGB models and in `meta` for DPBoosters —
`save_final_model` already stamps both.)

Ledger checks (only when provenance is `real-frozen-schema`) are **strategy-aware**,
because the append gate fires on each site's round-1 train call (`client_app.py:160`) and
the two strategies schedule round 1 differently:

- **bagging** — every site trains every round, so both see round 1. After each C run:
  exactly **2 new entries** (one per site) since the previous check; the two share one
  `config_hash`; each entry's `epsilon`/`noise_multiplier` equals the model meta's
  `reported_epsilon`/σ. After the sweep: total new entries == 2 × (C runs fired).
- **cyclic** — one site trains per round (`(server_round-1) % num_sites`), so **only the
  round-1 site (site A) appends**; sites that first train on round ≥ 2 (site B) **never
  ledger their spend**, though they do add noisy trees. After each C run: exactly **1 new
  entry** (site A); its `epsilon`/`noise_multiplier` matches the meta. After the sweep:
  total new entries == 1 × (C runs fired). **This is a landed-code accounting gap, not a
  driver choice** — see the mandatory unledgered-site handling below and Risk §8.
- after A and B runs (either strategy): **0 new entries** (identity/non-DP must never
  ledger — the ledger module fail-louds on identity; this catches silent gating
  regressions).

**Unledgered-site handling (cyclic, mandatory).** The expected site set is the half
basenames — `{Path(data-path).name for each node}` (fact §3.4), the same keys
`ledger_total()` uses; `unledgered_sites = expected − set(ledger_total().keys())`. When it
is non-empty on a real run, the driver MUST mark each missing site in the report's
per-site privacy table as **`UNLEDGERED — cyclic round-1 append gap (Risk §8)`**, never as
blank or `0` (both read as "spent nothing", a false privacy claim — this node did spend ε).
`results.json` records them under `ledger.unledgered_sites`. (On **bagging** the §4.4
"exactly 2 entries/C run" assertion already fails loud if a site is missing, so a bagging
`unledgered_sites` is only ever populated by an aborted run, not silently rendered.) The
go/no-go section stays operator-only; the operator sees the gap explicitly.

On `example-halves` provenance all ledger checks assert **zero** new entries for every arm
and both strategies (the append is provenance-gated; a rehearsal that writes the ledger is
a bug).

### 4.5 Offline re-scoring and deltas

Every saved model is scored on **both** halves through the single shared scorer:

```python
score_booster_on_half(bst, half_path, operating_point,
                      n_boot=args.n_boot, boot_seed=0,
                      split_seed=args.split_seed, holdout_frac=0.0,
                      holdout_eval=False)
```

- `--n-boot` default **1000** (the report needs the 95% CIs; this is the terminal
  scoring pass, not a search loop — the HPO harness's `n_boot=0` speed trick does not
  apply).
- Same `split_seed`/FLAT mode as the runs, so the scored validation split is
  byte-identical to the in-run one, per half, for all six arms (fact §3.3).
- A NaN AUC (single-class split) is recorded as `None` and every delta involving it is
  `None` (the §4.3-of-1.1.a convention).

Deltas, per half `h` and metric `m ∈ {auc_roc, auc_pr, brier}`:

- **B→C@ε (privacy cost, HEADLINE):** `Δ_priv(ε, h, m) = m(C@ε, h) − m(B, h)`
- **A→B (learner cost, context):** `Δ_learn(h, m) = m(B, h) − m(A, h)`

Sign convention, stated in the report: negative ΔAUC-ROC/ΔAUC-PR = utility lost,
positive ΔBrier = calibration worsened. A mean-across-halves column accompanies the
per-half values for the headline sentence; per-half is primary (site-stratified
reporting is the 1.c contract).

### 4.6 Accounting extraction

Pure functions in `dpsweep.py`, driven by the driver:

- `extract_dp_meta(bst) -> dict` — `{mechanism, sigma, epsilon, num_releases,
  per_site_trees}` from a DPBooster's meta. Sources (verified against
  `dp/boost._mechanism_meta`): `sigma`←`meta["noise_multiplier"]`,
  `epsilon`←`meta["reported_epsilon"]`, `num_releases`/`per_site_trees` top-level,
  `mechanism`←**`meta["dp"]["mechanism"]`** (nested, not top-level).
  **Non-finite handling.** `DPBooster.to_json_bytes` already runs `_json_finite` over the
  whole payload (`dp/boost.py:352`), scrubbing every `inf`/`NaN` → `null` **on serialize** —
  so for any model loaded from disk, arm B's identity `sigma`/`epsilon` arrive as `None`,
  **never `inf`**. `extract_dp_meta` therefore must simply **tolerate and pass through
  `None`** (a `math.isfinite(None)` would `TypeError`); it maps any *residual* `±inf`/`NaN`
  → `None` too, but that path only fires for in-memory mechanisms or hand-built JSON
  fixtures that inject `Infinity`, not real artifacts. The report renderer reads this dict
  and formats a `None` epsilon/sigma as `∞ (none spent)` for arm B. `per_site_trees` is
  already `None` for identity (`num_releases == 0`). The shared `_json_safe` (§1, extended
  to catch `±inf`) is the belt-and-suspenders second layer at the `json.dumps(allow_nan=False)`
  call — it guards fixtures and any future non-scrubbed path, not a live `inf` from a real
  saved model.
- `ledger_delta(before: list, after: list) -> list` — the entries appended during this
  sweep (list-suffix check; the ledger is append-only so a suffix is guaranteed).
- `validate_sweep_ledger(delta, c_arms, provenance)` — the §4.4 ledger assertions.
- Composition: the driver calls `dp_ledger.ledger_total(ledger_path, delta=args.delta)`
  — never a re-implementation (R6: `accounting.py` fns verbatim, one conversion). The
  returned `{site: ε}` dict is embedded **verbatim** (repr) in the report, as the
  skeleton's "`ledger_total()` output pasted verbatim" field demands.
- Per-run ε per site comes from the ledger entries of this sweep (cross-checked equal to
  the model meta, §4.4); the composed total covers the whole ledger to date (fact §3.5)
  — the report states this explicitly.

### 4.7 Report + results artifacts

**`<out_dir>/report_1_1_b.md`** — rendered programmatically by
`dpsweep.render_report(results) -> str` (house style: `metrics.render_run_report`,
`hpo.render_leaderboard` — never string-substitution into the template file, which is
brittle against whitespace). The renderer mirrors the skeleton's section structure
exactly; a test pins that every skeleton section heading appears in the output (drift
tripwire between template and renderer, §4.9).

Field-by-field fill contract (template → source):

| Skeleton field | Filled from |
|---|---|
| Date / operator | run date (UTC) / `--operator` |
| Data provenance | `--data-provenance` |
| rows / patients per half after dedup | `cohort_stats()` (below) |
| split line (`test_size`, `split_seed`, hash rule) | constants + `--split-seed` |
| % missing per feature per half | `cohort_stats()` — NaN rate per `FEATURE_COLS` on the **raw** parquet (pre-sentinel) |
| Config hash(es) | **C arms only** — the `config_hash` carried on this sweep's ledger entries (the two per-site entries of a C run share one hash). A and B show `n/a — no DP, not ledgered`. Rationale: `config_hash` hashes the per-arm run-identity **including the `dp` block**, so A/B/C hashes never match anyway — the hash is per-arm-config identity, not a cross-arm matching proof. Matching is proven by the §4.4 `subsample == colsample_bytree == 1.0` read-back; the skeleton's field is explicitly scoped to `dp_ledger` entries. |
| Comparator arm table | one row per (arm, half): Arm, half, mechanism + actual σ (from meta, never the skeleton constants — fact §3.7), ε, AUC-ROC (95% CI), AUC-PR, Brier |
| B→C / A→B lines | §4.5 delta tables (per half + mean), sign convention stated |
| Privacy accounting table | per site: the per-run ε of each C run of this sweep, and the composed ledger-total; ledger site keys shown verbatim with node labels. On cyclic, sites absent from `ledger_total()` render as `UNLEDGERED — cyclic round-1 append gap (Risk §8)`, never blank/0 (§4.4 unledgered-site handling) |
| `ledger_total()` verbatim | §4.6 |
| Release boundary box 2 (`dp-site-weight`) | auto-checked (code-enforced + test-pinned) |
| Release boundary box 1, Go/no-go section | **left blank — operator-only.** The driver fills every computable field; the boundary attestation and the go/no-go decision are human judgments recorded manually, then logged in `docs/logbook.md` |

- `cohort_stats(half_paths, split_seed)` (pure, in `dpsweep.py`): per half, post-dedup
  row count and unique-patient count via the same `split_half`/`resolve_run_split` chain
  the runs use (train + valid row totals), plus per-feature raw missingness. Local reads
  of GVA data are allowed (project constraint: GVA local access for debugging); these
  counts are **inside-boundary** values — the report must not leave the project (R5),
  which the report itself states.
- **Rehearsal banner:** when provenance ≠ `real-frozen-schema`, the report opens with a
  loud block: *"REHEARSAL — example halves. NOT a real-ε claim. This artifact does not
  fill roadmap 1.1.b."* and the privacy-accounting section notes that no ledger entries
  were written.
- Degenerate arms (failed/timed-out C runs) appear in the arm table with `DEGENERATE`
  in the metric cells and a note that their ledger spend (if the entry landed) still
  composes.

**`<out_dir>/results.json`** — machine-readable superset, strict JSON
(`json.dumps(allow_nan=False)`; non-finite floats — `NaN` **and** `±inf` — → `null` via
the extended `_json_safe` from `flwr_proc.py`, §1): `meta` (date, operator, gate-ack, provenance, federation, strategy,
split-seed, delta, epsilons, shared config incl. forced 1.0s, tuned-config path if any),
`cohort`, `arms` (per arm: label, run-config string, model path, format, tree count,
dp-meta extract, per-half metrics dicts), `deltas`, `ledger` (entries added this sweep,
`ledger_total` output, ledger path, and `unledgered_sites` — sites that spent ε but did
not ledger, non-empty only on cyclic, §4.4). This is the artifact 1.1.e's notebook
skeleton will ingest.

### 4.8 CLI

```
python scripts/run_dp_sweep.py
  --federation local-deployment        # swap for the cross-site link in v1.3
  --strategy {bagging,cyclic}          # default bagging (the 1.1.b deliverable). cyclic is
                                       #   supported but its ledger under-accounts the
                                       #   non-round-1 site (§4.4 strategy-aware asserts +
                                       #   unledgered-site handling; Risk §8) — bagging is
                                       #   the strategy the go/no-go should rest on
  --epsilons 1 3 5 10                  # pilot grid; run DESCENDING regardless of input order
  --delta 1e-5
  --config <tuned_bagging.toml>        # optional; §4.2 precedence
  --split-seed 42
  --n-boot 1000
  --out-dir out/dp_sweep               # relative → repo root; per-arm subdirs inside
  --ledger-path out/dp_ledger.jsonl    # relative → repo root; resolved ABSOLUTE and
                                       #   threaded into every run as dp.ledger-path so
                                       #   the SuperNode writes where the driver reads (§4.2)
  --data-provenance {example-halves,real-frozen-schema}   # default example-halves
  --operator <name>                    # required for real provenance
  --gate-ack "<text>"                  # required for real provenance (§4.3)
  --run-timeout 900
  --resume                             # skip arms whose final_model.json exists (re-score only)
  --dry-run                            # print the 6-run plan + exact run-config strings
```

`--resume` semantics: an existing model is trusted and re-scored; a *missing* model
re-fires the run. For C arms this can double-append the ledger — correct (intent-to-
spend, fact §3.5) and called out in the driver's log line when it happens.

### 4.9 Tests (`architecture/tests/test_dpsweep.py`)

Pure-logic tests (no federation, no subprocess):

1. `arm_plan([1,3,5,10])` → 6 arms, order `A, B, C@10, C@5, C@3, C@1`; labels; overrides
   exactly as §4.1 (B has no `target-epsilon`; C carries the threaded δ).
2. Run-config strings: every arm carries `params.subsample=1.0`,
   `params.colsample-bytree=1.0`, `holdout-frac=0.0`, `n-boot=0`, `save-model=true`,
   absolute dirs, an absolute `dp.ledger-path` (§4.2 threading), and the `data-provenance`
   token; `dp.enabled`/`dp.mechanism` appear for B and C; `dp.target-epsilon`/`dp.delta`
   appear for C only (arm B carries neither — §4.1).
3. Shared-config resolution: tuned TOML adopted for the four tuned knobs; its
   subsample/colsample discarded (forced 1.0) with the divergence note; divisibility
   failure raises before any run.
4. Delta computation: values, sign conventions, and `None` propagation for a NaN AUC.
5. `extract_dp_meta`: gaussian meta round-trips; `mechanism` is read from
   `meta["dp"]["mechanism"]`; an identity meta **loaded from a real saved model** (σ/ε
   already `None`) passes through as `None` without `TypeError`; a fixture that injects
   `inf` σ/ε is also mapped to `None`; and a `results`-shaped dict with an arm-B block
   survives `json.dumps(_json_safe(d), allow_nan=False)` without raising (the inf-crash
   guard — both defensive layers, §4.6/§1, exercised together).
6. `ledger_delta` + `validate_sweep_ledger` (strategy-aware, §4.4): **bagging** — 2
   entries/C-run pass, config-hash mismatch across the site pair fails, extra entry on an
   A/B run fails, ε/σ mismatch vs meta fails; **cyclic** — 1 entry/C-run (site A) passes,
   2 entries would fail, and `unledgered_sites` == the sites missing from `ledger_total()`
   (site B) is computed and non-empty; any entry on `example-halves` provenance fails for
   either strategy.
7. `render_report`: every section heading of
   `docs/templates/1_1_b_report_skeleton.md` (read at test time) appears in the render —
   the template-drift tripwire; both ε numbers present for every C arm (R6 rule); all
   three arms present with B→C labelled headline (R9); rehearsal banner present iff
   provenance is example-halves; box 2 checked, box 1 and go/no-go blank. A cyclic
   `results` fixture with a site missing from `ledger_total()` renders that site's
   privacy-table cell as `UNLEDGERED …` — never blank or `0` (§4.4 unledgered-site guard).
8. `cohort_stats` on a synthetic parquet fixture: dedup counts, missingness rates.
9. `load_saved_booster` contract (`fed_stroke.baseline`, covered in `test_baseline.py`
   alongside the existing loader tests, or here): a 2-tree XGB JSON fixture →
   `fmt == "xgboost-json"`, `n_trees == 2`, booster predicts; a `dp-gbdt-v2` fixture →
   `fmt == "dp-gbdt-v2"`, `n_trees == len(trees)`, DPBooster predicts. Regression pin that
   `eval_final_model.py` (refactored onto the shared loader) still reports the same tree
   count — its `test_eval_final_model.py` stays green with zero assertion changes.

Driver integration test (monkeypatched runner, `test_hpo.py` pattern): a fake
`_run_flwr` that drops pre-built tiny models (a 2-tree XGB JSON with the
`fed_run_config` attr; identity and gaussian `dp-gbdt-v2` fixtures with plausible meta)
into the arm dirs and appends fake ledger lines **at the `dp.ledger-path` parsed out of
the run-config string it was handed** (proving the driver reads the same absolute path it
threaded, §4.2) for C arms — assert: plan order honored, abort-on-arm-A-failure, C-arm
failure recorded as DEGENERATE without aborting, resume skips completed arms, both
artifacts written and `results.json` validates (including an arm-B block with `epsilon`/
`sigma` = `null`, not a raised `ValueError`). **Parametrized over strategy**: the fake
runner writes 2 ledger lines/C-run for bagging and 1 (site A) for cyclic, and the test
asserts the strategy-aware ledger validation passes in each case and that the cyclic run
populates `ledger.unledgered_sites` (site B) with the report cell marked `UNLEDGERED`
(§4.4).

Existing suites untouched; `run_hpo.py`'s refactor onto `scripts/flwr_proc.py` must keep
`test_hpo.py` green with zero assertion changes.

## 5. Files to change

| File | Change |
|---|---|
| `architecture/fed_stroke/dpsweep.py` | new — all pure logic (§4.1, §4.2 helpers, §4.5, §4.6, §4.7 renderers) |
| `architecture/scripts/run_dp_sweep.py` | new — orchestration driver (§4.3, §4.8) |
| `architecture/scripts/flwr_proc.py` | new — `_flwr_bin`/`_run_flwr` + pyproject fact helpers + `_json_safe` extracted from `run_hpo.py`; `_json_safe` widened to map `±inf` (not just `NaN`) → `None` (§1) |
| `architecture/scripts/run_hpo.py` | refactor onto `flwr_proc.py`, behavior-identical (its `_json_safe`/`_run_flwr`/fact helpers become imports); docstring's "the ONLY component that touches the process boundary" updated to name both drivers |
| `architecture/fed_stroke/baseline.py` | add `load_saved_booster(path) -> LoadedModel(booster, fmt, n_trees)` (§1 return contract), extracted from `eval_final_model.py` |
| `architecture/scripts/eval_final_model.py` | drop inline sniff; consume the `LoadedModel` tuple; behavior-identical (`test_eval_final_model.py` unchanged) |
| `architecture/tests/test_dpsweep.py` | new (§4.9) |
| `architecture/roadmap.md` | 1.1.b gains a pointer to this spec (at implementation time) |

`flwr_proc.py` lives in `scripts/`, not `fed_stroke/`: it is driver-side process
infrastructure that must not ship in the wheel to the Shenzhen SuperNode (the hatch
wheel packages `fed_stroke` only). Both drivers reach it via a `sys.path` insert of the
scripts dir, mirroring how they already insert `ARCH_DIR`.

## 6. Run matrix / verification

1. **Unit suite** — `pytest tests/test_dpsweep.py` plus the full existing suite
   (235/235 as of 2026-07-23) green.
2. **Dry-run** — `python scripts/run_dp_sweep.py --dry-run` prints the six-run plan with
   exact run-config strings; nothing launched.
3. **Rehearsal E2E (example halves)** — standing TLS loopback up
   (`run_local_federation.sh start`), then
   `python scripts/run_dp_sweep.py --out-dir out/dp_sweep_rehearsal`. Expect: all six
   runs complete; six models scored on both halves; `report_1_1_b.md` carries the
   rehearsal banner, all arm rows, both deltas, and an empty privacy-ledger section
   (zero entries written — asserted); `results.json` validates. This is the acceptance
   rehearsal and is runnable **now**, before the R8 gate clears.
4. **`--resume` check** — delete one C arm's model, re-run with `--resume`: only that
   arm re-fires.
5. **Real run (post-gate)** — after both reviewers' R8 sign-off is in `docs/logbook.md`
   AND real frozen-schema Geneva halves exist:
   `python scripts/run_dp_sweep.py --data-provenance real-frozen-schema
   --config <tuned toml> --operator <name> --gate-ack "..."`. Expect (bagging, the
   default/deliverable): **8 new ledger entries** (4 ε × 2 sites); report fully filled.
   On cyclic it would instead be **4 entries** (4 ε × site A only) with site B rendered
   `UNLEDGERED` (§4.4, Risk §8) — which is why the go/no-go rests on bagging. Operator
   completes box 1 + go/no-go and records the decision in `docs/logbook.md` — **before any
   Shenzhen integration** (remediation acceptance 5).

## 7. Acceptance criteria

1. `arm_plan`/run-config/report tests green; full suite green; no change to any
   `fed_stroke/dp/` module or to `accounting.py` (frozen).
2. Rehearsal E2E (§6.3) passes on the untouched example halves over TLS, producing both
   artifacts with the rehearsal banner and zero ledger writes.
3. The rendered report satisfies, mechanically (test-pinned): R6 — every per-run ε is
   accompanied by the composed ledger-total ε; R9 — all three arms on the identical
   cohort/split with B→C flagged as the privacy-cost headline and A→B as learner cost;
   every skeleton section present.
4. All six runs share one config with `subsample = colsample_bytree = 1.0` forced, one
   split seed, FLAT mode — verified by read-back from the saved models, not from intent.
5. The driver fail-closes correctly: aborts on arm-A failure; refuses real provenance
   without `--operator` + `--gate-ack`; ledger assertions per §4.4.
6. Roadmap 1.1.b annotated with the spec pointer; the real-run procedure (§6.5) is the
   documented, single-command path from "gates cleared" to "filled report".

## 8. Risks & notes

- **The go/no-go is not automatable.** The driver produces the evidence; the decision
  (remediation acceptance 5) is the user's, recorded manually. The report deliberately
  leaves those fields blank rather than pre-filling a judgment.
- **Delta significance.** At n≈1000/node the per-arm 95% CIs may be wider than the B→C
  deltas at large ε. The report shows CIs next to every delta so this is visible, and a
  paired bootstrap over the shared validation rows is the named upgrade if point deltas
  land inside CI width (out of scope now, §1).
- **Ledger totals grow monotonically across re-runs.** Re-running the real sweep (e.g.
  after a bug) composes on top of the previous spend — by design. If the composed total
  becomes the story's weak point, the trusted-experimentation alternative (release only
  one selected model) is already reserved for the v1.3 headline (R6 note); nothing in
  this driver forecloses it.
- **Crashed C runs.** The ledger entry lands on round 1; the model may not. The report
  shows such arms as DEGENERATE with their spend composed — honest but unsatisfying;
  the mitigation is the A-first ordering (infra failures die before ε is spent).
- **Cyclic under-accounts the non-round-1 site (landed-code gap).** The ledger append
  gate fires on each site's round-1 train call (`client_app.py:160`), but cyclic trains
  one site per round (`(server_round-1) % num_sites`), so only site A (round 1) ledgers;
  site B trains on round ≥ 2, spends ε (adds noisy trees), and **never ledgers it**. This
  driver cannot fix it (the ledger, `client_app`, and `accounting.py` are all out of scope,
  §1) — it **surfaces** it: strategy-aware assertions expect 1 entry/C run on cyclic, and
  the report marks the missing site `UNLEDGERED` rather than blank/0 (§4.4). **Consequence
  for the decision:** a cyclic sweep cannot support a per-site B→C privacy claim for the
  unledgered site, so the go/no-go should rest on the **bagging** sweep (the 1.1.b
  deliverable and default). Fixing the cyclic append timing is a v1.3 item (§1, "Cyclic
  revalidation"). This risk was accepted deliberately to keep `--strategy cyclic` runnable
  now (Decision 15).
- **Skeleton drift.** The renderer mirrors the template rather than substituting into
  it; the §4.9 heading tripwire fails the suite if the template is edited without
  updating the renderer (and vice versa).
- **`--epsilons` is a flag, not hardcoded** — but the pilot grid {1, 3, 5, 10} at
  δ = 1e-5 is the roadmap contract; deviating on the real run should be a deliberate,
  logged act.

## 9. Decisions locked in this spec (audit trail)

1. **Fixed-config A→B→C, not grid × ε** — the 1.1.b decision artifact per remediation
   acceptance 4/5; DP-aware re-tuning stays with the 1.1.a harness under 1.1.c/1.1.d
   (ledger-composition blow-up avoided).
2. **Run order A → B → C(desc ε)**; arm A doubles as the canary; abort on A, continue
   (as DEGENERATE) on a C failure.
3. **`hpo.build_run_config` reused** for run-config construction; `subsample` /
   `colsample-bytree` forced to 1.0 for every arm, unconditionally.
4. **ε/σ read back from model meta; composition only via `ledger_total()`**; per-run ε
   cross-checked equal between model meta and ledger entries.
5. **Report rendered programmatically** (mirroring the skeleton) with a heading-presence
   tripwire test — no template string-substitution.
6. **Driver fills every computable field; boundary box 1 and go/no-go stay
   operator-only.** Box 2 (`dp-site-weight`) auto-checked (code-enforced).
7. **Real-provenance runs require `--operator` + `--gate-ack`** — an affirmation stamped
   into the artifacts, standing in for the non-machine-checkable R8 sign-off.
8. **Shared driver commons extracted to `scripts/flwr_proc.py`** (not `fed_stroke/` — it
   must not ship in the Shenzhen wheel); model-loader sniff extracted to
   `baseline.load_saved_booster`.
9. **In-run `n-boot=0`, offline `--n-boot 1000`** — CIs computed once, at the terminal
   scoring pass, on the identical split via `score_booster_on_half`.
10. **δ threaded from one `--delta` flag** into run configs and `ledger_total()` alike.
11. **`--ledger-path` resolved absolute and threaded as `dp.ledger-path`** into every run's
    override, so the SuperNode writes exactly where the driver reads — removing the
    default-only coincidence and the SuperNode-CWD dependency (§4.2). A/B carry it but
    never append.
12. **Arm-B non-finite meta arrives as `None`, not `inf`.** `DPBooster.to_json_bytes`
    already scrubs identity's `inf` σ/ε to `null` on serialize (`dp/boost.py:352`), so
    every model loaded from disk has `noise_multiplier`/`reported_epsilon` = `None`. The
    §4.4 arm-B assertion is `is None` (NOT `== inf`), `extract_dp_meta` tolerates/passes
    `None`, and `mechanism` is read from the nested `meta["dp"]["mechanism"]`. The `_json_safe`
    inf-widening (extracted to `flwr_proc.py`) plus the `extract_dp_meta` inf→None map are
    two defensive layers for fixtures / non-scrubbed paths, not the live path (§1, §4.6). A
    regression test drives an arm-B `results` block through `json.dumps(allow_nan=False)`.
13. **`load_saved_booster(path) -> LoadedModel(booster, fmt, n_trees)`** is the one pinned
    loader contract: XGB → `fmt="xgboost-json"` with `eval_metric` set and
    `n_trees=num_boosted_rounds()`; `dp-gbdt-*` → `fmt` the verbatim format string, no
    `set_param`, `n_trees=len(trees)`. `eval_final_model.py` and the sweep driver both
    consume the tuple; §4.4 checks read `.fmt`/`.n_trees` (§1).
14. **A/B carry no report `config_hash`** (`n/a — no DP, not ledgered`); only C arms show
    the hash from their ledger entries. `config_hash` includes the per-arm `dp` block so it
    was never a cross-arm matching proof — the matching guarantee is the §4.4
    subsample/colsample read-back (§4.7).
15. **`--strategy cyclic` is kept and runnable, with strategy-aware ledger accounting.**
    Ledger assertions branch on strategy (bagging ⇒ 2 entries/C run; cyclic ⇒ 1, site A
    only), the gaussian release check uses `boost.per_site_tree_budget` (not
    `total//num-sites`, which is cyclic-wrong), and any site missing from `ledger_total()`
    on a real cyclic run is rendered `UNLEDGERED` in the report and listed in
    `results.json.ledger.unledgered_sites` — never blank/0. The underlying cyclic append
    gap is a landed-code limitation the driver surfaces, not fixes; the go/no-go should
    rest on bagging (§4.4, §4.8, Risk §8).

## Addendum 2026-08-30 — re-review conditions A-C1/C2, B-F8 (supersedes parts of §4.2, §4.4, §4.8, Decisions 11/15)

Reviewer A's re-review (`docs/reviews/dp_accountant_review_findings_A_rereview.md`, C1/C2)
and reviewer B's round 2 (`dp_accountant_review_findings_B_rev2.md`, F8) found the DP rails
keyed off submitter-controlled `run_config`. Landed fixes:

- **Provenance and ledger path are node-owned.** `node_config` carries `data-provenance` and
  `dp-ledger-path` (materialized by `run_local_federation.sh` from
  `[tool.fed_stroke.nodes]`, ledger path resolved absolute). `validate_dp_run_provenance()`
  (`fed_stroke/dp/preconditions.py`) fail-closes the DP branch without them, refuses a
  `run_config` declaration that disagrees with the node's, and requires a node ledger path on
  real data. `_maybe_append_ledger` and the `DPConfig` insecure-test hatch read the node value.
- **Decision 11 superseded:** the driver no longer threads `dp.ledger-path` into the run
  config and `--ledger-path` is gone; it reads the nodes' declared path
  (`flwr_proc._node_ledger_path`) and refuses to start when `--data-provenance` disagrees
  with the nodes' declaration (`_node_provenance`). `data-provenance` in the run config
  remains the submitter's declaration, cross-checked by the client.
- **Decision 15 narrowed:** `--strategy cyclic` + DP is refused on real-frozen-schema data by
  both the client gate and the driver; it stays runnable as a rehearsal on example halves. The
  §4.4 cyclic ledger assertions and the `UNLEDGERED` rendering remain as defense in depth.
