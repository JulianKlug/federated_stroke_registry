# Spec 1.d — Federated-vs-pooled correctness check on the Geneva 50/50 partition

Implements roadmap item 1.d ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Federated-vs-pooled correctness check on Geneva 50/50 partition. Federated
> result must be within 3 AUC points of pooled Geneva xgboost. Purpose: catch
> silent data-partitioning, DMatrix, or tree-serialization bugs before any
> downstream DP result is measured against them.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§4) and the installed framework (`flwr==1.31.0`, `xgboost>=2.0`). Builds directly
on 1.b ([1b_fedxgb_bagging_cyclic.md](1b_fedxgb_bagging_cyclic.md)) and 1.c
([1c_evaluation_harness.md](1c_evaluation_harness.md)), whose §1 explicitly
deferred this ±3-AUC-point bound to here. Every split/metric seam this spec needs
was built by 1.b/1.c to be reused; this spec adds one thing they lack — a
**pooled** (single-process) reference to compare the federated numbers against.

## 1. Goal & scope

1.d asks one question: **does the distributed pipeline produce a model as good as
training one XGBoost on the same rows in one process?** The federated path does a
lot that can silently corrupt the model — patient-ID-stratified per-node splits,
per-node `DMatrix` construction, per-round tree aggregation (bagging merges
trees; cyclic passes the whole booster), and repeated `save_raw`/`load_model`
round-trips. If a site trains on the wrong rows, a `DMatrix` is built with
mismatched columns/labels, or a tree is dropped in serialize/merge/save, the
per-site AUC drifts away from a centrally-trained reference. This check is the
tripwire for that bug class, run **before** any DP-utility number is measured
against the federated baseline (a DP regression is only interpretable if the
no-DP federated baseline is itself trustworthy).

**In scope**

- A **pooled baseline trainer**: one XGBoost fit in a single process on the
  union of the two Geneva halves' *training* splits, with params and tree budget
  matched to the federation, trained **deterministically** (pinned seed).
- A **per-site AUC gate on bagging** — the ensemble most directly comparable to
  pooled: for each site, `|AUC_ROC_bagging − AUC_ROC_pooled| ≤ tol`, where `tol`
  is **calibrated empirically** from the no-bug delta distribution (§6 step 0),
  not asserted. Cyclic per-site deltas are computed and reported but **do not
  gate** — cyclic last-site bias is a known non-bug (§8.2), so hard-gating it
  would contradict the spec's own risk analysis.
- A **serialization round-trip tripwire**: every saved model, re-serialized and
  reloaded, must predict **per-row-identically**. AUC-ROC is rank-only and blind
  to a monotonic-distorting serialize/merge bug (delta stays 0); the round-trip
  catches that bug class directly, which the AUC gate cannot.
- A **shared scoring seam** so the federated model, the pooled model, and 1.c's
  `eval_final_model.py` all score a booster on a half through identical code.
- A CLI check that runs over **both strategies** (bagging + cyclic
  forward/reverse), prints a per-site comparison table, writes a traceable
  artifact, and exits non-zero on any **bagging** gate failure, any round-trip
  failure, a NaN AUC, or unverified config provenance.
- Unit tests for the gate logic, the pooled trainer, and the round-trip tripwire.

**Out of scope** (later roadmap items)

- Choosing a winning strategy or quantifying last-site bias → Phase v1.3 / 1.3.d,
  on real cross-site data.
- DP noise on the histograms / DP-utility curve → Phase v1.1/v1.2.
- The **v0.a local sanity baseline** (roadmap §"Local sanity baselines"). That is
  a *different* artifact: a single XGBoost on the *full* real Geneva set at the
  frozen schema with its own held-out split, reported for external comparison.
  1.d's pooled model is instead pinned to the *50/50-halves union* with per-site
  evaluation so it is directly comparable to the federated run. They share the
  trainer idea but not the data contract; 1.d stays self-contained. (When v0.a is
  built it may reuse `baseline.train_pooled_booster`.)
- Any server-side/global evaluation dataset — the server never sees data (§4);
  all scoring is on per-site local validation splits.

## 2. Dependencies

- **1.b complete**: both strategies selectable from config; the two Geneva halves
  (`out/geneva_half_{A,B}.parquet`) and the 2-SuperNode topology exist.
- **1.c complete**: `metrics.compute_binary_metrics` is the single source of
  metric math; `eval_final_model.py` already scores a saved federated model on a
  half via `generate_splits(test_size=0.2, seed=42)` + `compute_binary_metrics`.
- **No new runtime dependencies.** `tomllib` is stdlib on the app's Python 3.12
  (`requires-python >=3.12`); `xgboost` and `scikit-learn` are already project
  deps ([architecture/pyproject.toml](../../architecture/pyproject.toml)).

## 3. Framework facts that drive the design

- **The federation has no global test set — evaluation is per-site.** Each
  SuperNode does its own 80/20 patient-stratified split in `load_data_gva`
  ([task.py:98-105](../../architecture/fed_stroke/task.py)) and
  `client_app.evaluate` scores on *that half's* 20% holdout
  ([client_app.py:100-139](../../architecture/fed_stroke/client_app.py)). So "the
  federated result" is a pair of per-site AUCs, not one pooled number. The pooled
  reference must therefore also be scored **per site**, on the same two splits, or
  the comparison is not apples-to-apples.
- **The pooled training set must be the union of the two halves' train splits —
  not a fresh split of the raw Excel.** The two federated clients collectively
  train on `train_A ∪ train_B`, where `(train_i, valid_i) =
  generate_splits(half_i, test_size=0.2, seed=42)`. Re-splitting the raw registry
  would give the pooled model a *different* row sample, so any AUC gap would
  conflate "federated vs pooled" with "sample A vs sample B." Concatenating the
  two halves' `generate_splits` train parts fixes the rows and isolates the one
  variable this check exists to measure. It also guarantees the pooled model
  never sees any validation row.
- **Bagging-merged trees ≠ sequential boosting.** `FedXgbBagging` builds 40 trees
  by merging 1 tree from each of 2 sites across 20 rounds; a pooled `xgb.train(...,
  num_boost_round=40)` boosts 40 trees sequentially on the union. These are
  legitimately different ensembles, so the bound is a **loose sanity gate, not an
  equality test**, and its width is *calibrated* from the measured no-bug delta
  (§6 step 0), not asserted. `subsample`/`colsample_bytree` add further
  stochasticity; the pooled fit pins `seed` explicitly (§8.10) so it is
  reproducible run-to-run rather than relying on XGBoost's implicit `seed=0`.
- **The trained model — not `pyproject.toml` — is the source of truth for the
  params + budget the pooled reference must match.** The FL run reconstructs its
  params via `replace_keys(unflatten_dict(context.run_config))`
  ([server_app.py:139-140](../../architecture/fed_stroke/server_app.py)), where
  `context.run_config` is `[tool.flwr.app.config]` **plus any `--run-config`
  overrides applied at launch** (the §6 matrix overrides `train-method` /
  `cyclic-order` this way). So `pyproject.toml` alone is *not* what a given
  federated model was trained with — an override to `params.eta` or `total-trees`
  would silently diverge, and the pooled reference, if it re-read only
  `pyproject.toml`, would train against the wrong baseline and quietly
  invalidate the whole gate. To close this, **`server_app` embeds the
  resolved run config into the saved model** as an XGBoost booster attribute
  (`bst.set_attr(fed_run_config=json.dumps({"params": params, "total_trees":
  ...}))`); the attribute is stored inside the model's JSON, survives
  `save_model`/`load_model` and any `mv` rename (verified), and travels with the
  model wherever the run matrix moves it. The check reads params + budget **from
  each model's own embedded config**, asserts all federated models carry
  identical params + `total_trees` (the "matched budget" invariant, now checked
  at runtime instead of assumed), and trains the pooled reference with exactly
  those. `pyproject.toml` is a **fallback only**, used with a loud warning for a
  legacy model that predates the embedded attribute.
- **The `unflatten_dict` machinery is equivalence-tested, not hand-copied.**
  `tomllib` parses TOML dotted keys (`params.eta = 0.1`) into an *already-nested*
  `{"params": {"eta": ...}}`, whereas `context.run_config` is *flat*
  (`{"params.eta": ...}`) and only becomes nested after `unflatten_dict`. The two
  shapes converge to byte-identical params because flwr's `unflatten_dict`
  preserves an already-nested value unchanged (verified:
  `unflatten_dict(cfg_table) == unflatten_dict(flatten_dict(cfg_table))`). This
  non-obvious idempotency is pinned by a unit test that asserts
  `load_matched_config`'s params equal the FL-path derivation
  (`replace_keys(unflatten_dict(flatten_dict(cfg_table)))`), *not* a hand-written
  expected dict — a hand-copy would reintroduce exactly the drift 1.d exists to
  catch.

## 4. Design

### 4.0 Params/budget provenance — the one thing this check must get right

The pooled reference is only a valid baseline if it is trained with the *same*
params and tree budget the federated models actually used. That provenance now
flows through the saved model itself, not through a re-read of `pyproject.toml`:

```
 pyproject.toml [tool.flwr.app.config]
        │  (+ optional --run-config overrides at launch)
        ▼
 context.run_config  ──unflatten_dict+replace_keys──►  params, total_trees
        │                                                     │
        │  (train federation)                                 │  bst.set_attr(
        ▼                                                     ▼   fed_run_config=…)
 aggregated global model  ───────────────────────────►  final_model.json
                                                              │  (embeds resolved
                                                              │   params+total_trees;
                                                              │   survives mv rename)
                                                              ▼
                       check_fed_vs_pooled.py: read embedded config from EACH model
                                                              │
                          assert all fed models share identical params+total_trees
                                                              │  (matched-budget invariant,
                                                              ▼   checked not assumed)
                       train_pooled_booster(params, num_boost_round=total_trees)
                                                              │
                          ┌───────────────────────────────────┴──────────────┐
                          ▼                                                    ▼
        round-trip tripwire (all models,               bagging AUC gate (hard):
        per-row exact, gates independently)            |auc_bagging − auc_pooled| ≤ tol_calibrated
                                                        cyclic AUC: INFO, non-gating
```

Legacy model with no `fed_run_config` attribute → fall back to
`load_matched_config(pyproject.toml)` with a loud warning that params provenance
is unverified.

### 4.1 New module `architecture/fed_stroke/baseline.py`

The reusable, testable core. No CLI, no printing — pure functions the script and
the tests both call.

```python
def read_model_config(bst, pyproject_path) -> tuple[dict, int, str]:
    """Return (params, total_trees, provenance) for a saved federated model.

    Primary source: the model's own embedded `fed_run_config` attribute
    (bst.attr("fed_run_config")), written by server_app at save time — so the
    pooled reference matches the params/budget THIS model was actually trained
    with, including any --run-config overrides. provenance="model".
    Fallback: if the attribute is absent (legacy model), read load_matched_config
    (pyproject) and return provenance="pyproject-fallback" so the caller can warn.
    """

def assert_matched_across_models(configs) -> tuple[dict, int]:
    """Given the (params, total_trees) read from every federated model, assert
    they are identical across all models and return the single agreed
    (params, total_trees). Divergence is a HARD FAIL: it means the strategies
    were not run at a matched budget, so no single pooled reference is valid."""

def load_matched_config(pyproject_path) -> tuple[dict, int]:
    """FALLBACK reader: [tool.flwr.app.config] -> (params, total_trees).

    Uses tomllib + unflatten_dict + task.replace_keys. The unit test pins its
    params to the FL-path derivation replace_keys(unflatten_dict(flatten_dict(
    cfg_table))) — NOT a hand-written dict — so the equivalence is guarded, not
    hand-copied. Used only when a model carries no embedded config.
    """

def split_half(data_path):
    """(train_df, valid_df) for one half via generate_splits(test_size=0.2, seed=42).

    The exact split contract load_data_gva uses, so valid_df here IS the split
    client_app.evaluate scored on."""

def train_pooled_booster(half_paths, params, num_boost_round) -> xgb.Booster:
    """Concatenate every half's train_df, build ONE DMatrix, xgb.train once.

    Selects DMatrix columns as `FEATURE_COLS` (features) + `TARGET_COL` (label),
    exactly as `load_data_gva` does — generate_splits returns the FULL frame
    (incl. patient_id, case_admission_id, target), so feeding it raw would leak ID
    columns in as features. Concatenate the halves' train_dfs, then subset.

    Asserts pooled-train patient IDs are disjoint from every half's valid_df
    (no validation row leaks into the reference). Forces a DETERMINISTIC fit so the
    pooled reference is reproducible run-to-run (a ±0.01 wobble can flip PASS/FAIL
    at the calibrated tolerance): params must carry a pinned `seed`, AND the pooled
    fit overrides `nthread=1` (XGBoost `hist` can carry tiny float nondeterminism
    across threads on some versions; single-thread removes that axis). The
    config's `nthread` is a federation-throughput knob, not a correctness one, so
    overriding it for the reference is safe."""

def score_booster_on_half(bst, data_path, operating_point, n_boot, boot_seed) -> dict:
    """Rebuild the half's valid split and return compute_binary_metrics(...).

    This is eval_final_model._site_metrics lifted into the package: the federated
    model, the pooled model, and the 1.c offline scorer all score through this
    one function, so their AUCs match by construction."""

def assert_prediction_roundtrip(bst, dmatrix) -> None:
    """Serialization tripwire (independent of the AUC gate). Predict; re-serialize
    the booster (save_raw("json")); reload into a fresh Booster; predict again;
    assert bitwise-identical per-row output (np.array_equal). A save/load that is
    not a fixed point — a dropped or reordered tree, a truncated node — surfaces
    here even when it leaves AUC-ROC unchanged, because AUC is rank-only and blind
    to monotonic score distortion. Raises on any mismatch. Run on every federated
    model and the pooled model."""

def compare_auc(pooled_by_site, federated_by_site, tol) -> dict:
    """Pure gate: per-site delta = |auc_fed - auc_pooled|; pass if delta <= tol.

    NaN on either side (single-class split) is a HARD FAIL, not a skip. Returns a
    structured result {site: {pooled, federated, delta, passed}} + strategy-overall
    bool. No I/O — directly unit-testable. Whether a strategy's result *gates the
    exit code* is the CLI's call (bagging gates; cyclic is informational), not this
    function's — it scores every strategy identically."""
```

### 4.2 New script `architecture/scripts/check_fed_vs_pooled.py`

Thin CLI over `baseline.py`, mirroring `eval_final_model.py`'s argument style.
**There is no `--expected-trees` flag** — the tree budget is read from the models'
embedded config, so there is a single source of truth and no operator-supplied
number can silently disagree with what the models were trained at.

```
python scripts/check_fed_vs_pooled.py \
    --data ../out/geneva_half_A.parquet ../out/geneva_half_B.parquet \
    --fed bagging=../out/models/bagging.json \
          cyclic_forward=../out/models/cyclic_forward.json \
          cyclic_reverse=../out/models/cyclic_reverse.json \
    --max-auc-delta 0.03
```

Flow:
1. Load every `--fed LABEL=PATH` model; `read_model_config` each →
   `(params, total_trees, provenance)`. **`pyproject-fallback` provenance is a
   HARD FAIL** (exit non-zero) unless `--allow-pyproject-fallback` is passed — a
   warning would be silently ignored in CI, letting a `set_attr` regression
   quietly revert to the exact `pyproject.toml`-reread path A1 exists to
   eliminate while the gate still exits 0.
2. `assert_matched_across_models(...)` → the single agreed `(params,
   total_trees)`. Divergent params/budget across models is a hard fail (the
   strategies were not run at a matched budget; no single pooled reference is
   valid).
3. `train_pooled_booster(--data, params, num_boost_round=total_trees)`
   (deterministic — pinned `seed`); assert the pooled model has exactly
   `total_trees` boosted rounds.
4. **Serialization round-trip tripwire.** `assert_prediction_roundtrip` on the
   pooled model and on each federated model (probe DMatrix = each half's valid
   split). Any per-row mismatch is a hard fail — this is the direct check for the
   serialize/merge bug class, run regardless of the AUC deltas.
5. `score_booster_on_half` the pooled model on each half → `pooled_by_site`.
   Also score on `concat(valid_A, valid_B)` → a pooled-combined AUC (printed,
   **informational only**; the gate is per-site).
6. For each fed model: assert `bst.num_boosted_rounds() == total_trees` (budget
   check against the *derived* budget, not an operator flag), score per site,
   then `compare_auc(pooled_by_site, fed_by_site, tol=--max-auc-delta)`.
7. Print a per-site table (site · pooled AUC · fed AUC · delta · PASS/FAIL) per
   strategy; write `out/metrics/fed_vs_pooled.{json,md}`.
8. **Exit non-zero** if: any **bagging** (site) delta > tol, any round-trip
   mismatch, any NaN AUC, the models' configs diverge, provenance is unverified
   (step 1), or the pooled tree count ≠ `total_trees`. **Cyclic per-site deltas
   are printed and written to the artifact but do NOT affect the exit code**
   (cyclic last-site bias is a known non-bug, §8.2); a cyclic over-tol delta is
   surfaced in the table as `INFO`, not `FAIL`.

A `--calibrate N` mode (§6 step 0) trains the pooled reference across `N` seeds,
prints the per-site no-bug `|bagging_auc − pooled_auc|` distribution, and exits 0
without gating — run once to choose the tolerance from evidence rather than
assertion.

`--max-auc-delta` defaults to the value **calibrated in §6 step 0** (the measured
no-bug delta plus margin), not a hand-picked constant; 0.03 is a provisional
placeholder until that run is done. Defaults for `--operating-point` (0.5),
`--n-boot` (1000), `--boot-seed` (0) mirror `pyproject.toml` so the numbers match
the FL run exactly, same as `eval_final_model.py:65-71`. (These are scoring-only
knobs — they do not affect the trained model — so reading them from
`pyproject.toml` is safe even though params come from the model.)

### 4.3 Modify `architecture/fed_stroke/server_app.py` — embed the resolved run config

In the `save-model` block ([server_app.py:191-205](../../architecture/fed_stroke/server_app.py)),
before `bst.save_model(str(out_path))`, stamp the model with the resolved config
it was trained under:

```python
bst.set_attr(fed_run_config=json.dumps({
    "params": params,                               # already resolved at line 140
    "total_trees": context.run_config["total-trees"],
}))
```

`params` is the exact dict the clients trained with (`replace_keys(unflatten_dict(
context.run_config))`), so the attribute captures any `--run-config` override, not
just the `pyproject.toml` defaults. The attribute rides inside the model JSON,
survives `save_model`/`load_model` and the `mv` renames in §6 (verified round-trip),
and is what `read_model_config` reads back. This is the only change to 1.b/1.c
code, it is additive (a new attribute; nothing else in the save path changes), and
it is what turns 1.d's "matched budget" from an assumption into a checked
invariant.

### 4.4 Refactor `architecture/scripts/eval_final_model.py`

Replace its private `_site_metrics` with a call to
`baseline.score_booster_on_half`. Behaviour is unchanged; the duplicate split +
scoring logic is deleted so the two paths cannot drift. `eval_final_model.py`
keeps its own `--expected-trees` flag (it scores a single arbitrary model and has
no matched-budget invariant to derive from); only the new check drops it.

### 4.5 Artifact `out/metrics/fed_vs_pooled.json`

```json
{
  "max_auc_delta": 0.03,
  "total_trees": 40,
  "params": { "objective": "binary:logistic", "eta": 0.1, "max_depth": 4, "...": "..." },
  "config_provenance": "model",
  "pooled": { "geneva_half_A.parquet": { ...full metric dict... }, "geneva_half_B.parquet": {...} },
  "federated": {
    "bagging":        { "geneva_half_A.parquet": {...}, "geneva_half_B.parquet": {...} },
    "cyclic_forward": { ... }, "cyclic_reverse": { ... }
  },
  "gate": {
    "bagging":        { "gating": true,  "geneva_half_A.parquet": {"pooled":0.71,"federated":0.69,"delta":0.02,"passed":true}, ... },
    "cyclic_forward": { "gating": false, "geneva_half_A.parquet": {"pooled":0.71,"federated":0.66,"delta":0.05,"passed":false}, ... },
    "cyclic_reverse": { "gating": false, ... }
  },
  "roundtrip_ok": true,
  "passed": true
}
```

Each strategy carries `"gating"`: `true` for bagging (its `passed` drives the exit
code), `false` for cyclic (its `passed` is recorded for diagnosis but ignored by
the gate — so a `false` cyclic entry with top-level `"passed": true` is expected,
not a contradiction). `roundtrip_ok` records the serialization tripwire result
(gates independently of AUC). `params`, `total_trees`, and `config_provenance`
(`"model"` or `"pyproject-fallback"`) are recorded so the artifact is
self-documenting: a reader can see exactly what the pooled reference was trained
with and whether that came from the models or a fallback. A companion `.md` renders the same table for humans
(regenerated each run; JSON is the source of truth), consistent with 1.c's
`render_run_report` discipline.

### 4.6 Tests

**`architecture/tests/test_baseline.py` (new)** — pure-logic and sub-second,
matching the fakes/`SimpleNamespace` idiom of `tests/test_metrics.py`:

- `compare_auc`: within tol → pass; over tol → fail; NaN on either side → hard
  fail (not skipped); overall bool is the AND of per-site results.
- `train_pooled_booster` on a synthetic two-half fixture yields a booster with
  exactly `num_boost_round` trees.
- Leakage guard: pooled-train patient IDs ∩ each half's valid patient IDs is
  empty.
- `load_matched_config` on a tiny fixture `pyproject` returns `total_trees` and a
  params dict **equal to the FL-path derivation**
  `replace_keys(unflatten_dict(flatten_dict(cfg_table)))` (C1 — pins the two
  paths together; not a hand-copied expected dict).
- `read_model_config`: a booster stamped with `fed_run_config` returns
  `provenance="model"` and the embedded params/budget; a booster with no
  attribute returns `provenance="pyproject-fallback"` and the pyproject values.
- `assert_matched_across_models`: identical configs → returns the single
  `(params, total_trees)`; params mismatch OR `total_trees` mismatch across
  models → raises (hard fail).
- **[T2] `score_booster_on_half`** on a synthetic half returns a full
  `compute_binary_metrics` dict for the correct (`test_size=0.2, seed=42`) split.
- **`assert_prediction_roundtrip`**: an intact booster round-trips (passes); a
  booster whose serialized bytes are then tampered (or a stub returning different
  predictions on the second predict) raises. Confirms the tripwire actually
  catches a non-fixed-point serialization.
- **`train_pooled_booster` determinism**: training twice on the same fixture with
  the pinned seed yields per-row-identical predictions (guards the boundary
  flakiness the calibrated tolerance would otherwise mask).

**[T1 — regression, `tests/test_eval_final_model.py` new or added to an existing
test] the §4.4 refactor is behavior-preserving.** Score a fixed booster on a
fixture half through `score_booster_on_half` and assert the metric dict is
identical to the pre-refactor `_site_metrics` computation (same split, same
`compute_binary_metrics` call). Guards the shared scorer that *both* the offline
table and the pooled gate now depend on — a silent drift here would corrupt both.

**[T3 — end-to-end, `tests/test_check_fed_vs_pooled.py` new] the CLI's exit-code
contract** (acceptance criteria 1 & 6), driven by `subprocess` on a synthetic
fixture (two tiny halves + a couple of stamped boosters):
- all deltas within tol → exit 0, artifact written, `"passed": true`.
- an injected over-tol **bagging** model → exit ≠ 0, `"passed": false`.
- an injected over-tol **cyclic** model → exit **0** (informational), row marked
  `INFO` not `FAIL`, `"passed": true` — the bagging-gates/cyclic-informational
  split (outside-voice #3).
- a forced single-class validation split (NaN AUC) → exit ≠ 0 (hard fail, not a
  silent skip).
- two fed models stamped with divergent params → exit ≠ 0 before any pooled
  training (matched-budget invariant).
- a model with **no `fed_run_config`** (pyproject-fallback provenance) → exit ≠ 0
  by default; exit 0 only with `--allow-pyproject-fallback` (outside-voice #6).
- a model whose serialized bytes don't round-trip → exit ≠ 0 (round-trip
  tripwire, independent of AUC).

**`tests/test_server_config.py` (extend)** — after a `save-model` run, the saved
model carries a `fed_run_config` attribute whose `params` equal the resolved
config and whose `total_trees` equals `total-trees` (A1 embed, verified at the
producer side so a broken stamp can't slip through to the check as a silent
fallback).

## 5. Files to change

- **New** `architecture/fed_stroke/baseline.py` — §4.1 core.
- **New** `architecture/scripts/check_fed_vs_pooled.py` — §4.2 CLI.
- **New** `architecture/tests/test_baseline.py` — §4.6 pure-logic tests.
- **New** `architecture/tests/test_check_fed_vs_pooled.py` — §4.6 [T3] CLI
  exit-code e2e.
- **New** `architecture/tests/test_eval_final_model.py` — §4.6 [T1] refactor
  regression (or fold into an existing eval test if one is added later).
- **Modify** `architecture/scripts/eval_final_model.py` — §4.4 refactor onto the
  shared scorer (`_site_metrics` → `score_booster_on_half`).
- **Modify** `architecture/fed_stroke/server_app.py` — §4.3, embed the resolved
  run config into the saved model (3-line additive `set_attr`).
- **Modify** `architecture/tests/test_server_config.py` — §4.6, assert the
  saved model carries the `fed_run_config` attribute.
- **Modify** `architecture/pyproject.toml` — add `params.seed = 0` to
  `[tool.flwr.app.config]` (make the currently-implicit XGBoost default explicit
  and pinned). It then flows into every model's embedded config → the pooled
  reference trains deterministically. Additive; the default is already 0, so
  federated numbers do not change.
- **Modify** `TODOS.md` — add a short closure note that 1.d unified the per-site
  scorer (`_site_metrics` → `score_booster_on_half`). Note: the pre-existing 1.c
  reuse/consistency item is *already* marked DONE, so this is a fresh note, not a
  re-mark of that item.
- **Reused unchanged**: `fed_stroke/schema.py` (`FEATURE_COLS`, `TARGET_COL`),
  `fed_stroke/task.py` (`generate_splits`, `replace_keys`), `fed_stroke/metrics.py`
  (`compute_binary_metrics`), `out/geneva_half_{A,B}.parquet`.

> **File-count note (Step-0 complexity):** this touches ~9 files, above the
> 8-file smell threshold, but the *production* surface is small — one new module
> (`baseline.py`), one thin CLI, a 3-line additive `server_app` stamp, and a
> behavior-preserving `eval_final_model` refactor. The rest are test files, added
> by the deliberate "full coverage" decision (regression + scorer unit + CLI
> e2e). The growth is traceable to explicit review decisions, not scope creep.

**Build order.** `baseline.py` (§4.1) is the foundation — the CLI (§4.2) and the
`eval_final_model` refactor (§4.4) both import it — so implement and test it first.
The `server_app` embed (§4.3) + `pyproject.toml` seed are independent and can land
in parallel (they're what makes a real §6 run produce `provenance="model"`; the
unit tests use synthetic stamped boosters and don't need a live federation).

## 6. Run matrix

Produce one saved model per strategy, then gate all three in one check. Run from
`architecture/` with the local 2-SuperNode topology up
(`scripts/run_local_federation.sh start`):

```bash
# 1. Save one final model per strategy (rename each so it isn't overwritten).
#    MUST run after the §4.3 server_app change so each model carries fed_run_config.
MD="model-dir='${PWD}/../out/models'"
flwr run . local-deployment --run-config "save-model=true ${MD}"
mv ../out/models/final_model.json ../out/models/bagging.json
flwr run . local-deployment --run-config "train-method='cyclic' save-model=true ${MD}"
mv ../out/models/final_model.json ../out/models/cyclic_forward.json
flwr run . local-deployment --run-config "train-method='cyclic' cyclic-order='reverse' save-model=true ${MD}"
mv ../out/models/final_model.json ../out/models/cyclic_reverse.json

# 2. CALIBRATE the tolerance (one-time, before trusting the gate) — needs the
#    bagging model from step 1. Train the pooled reference across N seeds, score
#    per site, and report the no-bug |bagging_auc - pooled_auc| distribution. Set
#    --max-auc-delta from it (e.g. max observed delta + a small margin); record the
#    number and the run in docs/logbook.md. Until this is done, 0.03 is PROVISIONAL.
python scripts/check_fed_vs_pooled.py \
    --data ../out/geneva_half_A.parquet ../out/geneva_half_B.parquet \
    --fed bagging=../out/models/bagging.json \
    --calibrate 5          # trains pooled over 5 seeds, prints delta spread, exits 0

# 3. Gate every strategy per site with the calibrated tolerance
#    (no --expected-trees: the budget is read from each model's embedded config)
python scripts/check_fed_vs_pooled.py \
    --data ../out/geneva_half_A.parquet ../out/geneva_half_B.parquet \
    --fed bagging=../out/models/bagging.json \
          cyclic_forward=../out/models/cyclic_forward.json \
          cyclic_reverse=../out/models/cyclic_reverse.json \
    --max-auc-delta <calibrated value from step 2>
```

**Calibration measures pooled-side variance only.** `--calibrate N` varies the
*pooled* seed across N fits; the bagging model is a single fixed run, so the
reported spread does **not** include bagging's own run-to-run variance. It
therefore *understates* the true no-bug delta spread. Set the margin with that in
mind (don't pick the bare max observed delta — add headroom), or, for a fuller
calibration, re-run the federation across several seeds too (expensive: N full
federated runs) and measure the delta across both axes. Document which was done in
`docs/logbook.md`.

**Re-run requirement:** the three models must be saved *after* the §4.3
`server_app` change so they carry the `fed_run_config` attribute. Models saved by
an earlier build have no attribute; the check still runs but falls back to
`pyproject.toml` with a `config_provenance="pyproject-fallback"` warning — re-run
step 1 to get verified provenance. (The current `out/models/*.json` predate the
attribute, so a first real run must regenerate them.)

Log the pooled per-site AUCs and the pass/fail table in `docs/logbook.md`
alongside the 1.c run entries.

## 7. Acceptance criteria

1. `check_fed_vs_pooled.py` prints, per strategy and per site, the pooled
   AUC-ROC, federated AUC-ROC, delta, and PASS/FAIL (bagging) or INFO (cyclic);
   exits 0 iff every **bagging** delta ≤ the calibrated `--max-auc-delta`, all
   round-trips pass, provenance is verified, and no AUC is NaN.
2. The **bagging** gate passes on the Geneva 50/50 partition, and cyclic per-site
   deltas are reported (informational). This is the correctness reading of the v1
   line "both strategies implemented and pass 1.d": both strategies are
   implemented and run through the check; **bagging is the hard correctness gate**
   (the ensemble comparable to pooled), while cyclic is reported without gating
   because its last-site bias is a known non-bug (§8.2). This reinterpretation is
   explicit, not silent.
3. The tolerance is **calibrated** (§6 step 0: measured no-bug delta + margin,
   recorded in `docs/logbook.md`), not asserted; 0.03 stands only until that run.
4. The pooled model uses params + budget read from the federated models' embedded
   `fed_run_config` (not re-derived from `pyproject.toml`); all federated models
   are asserted to carry identical params + `total_trees`; the pooled model has
   exactly that `total_trees` (40) boosted rounds and is trained with a pinned
   `seed` (reproducible run-to-run). No hand-copied params, no operator-supplied
   tree count — drift-guarded by construction.
5. Every saved model (federated and pooled) passes the `assert_prediction_roundtrip`
   serialization tripwire; a non-round-tripping model exits non-zero regardless of
   its AUC delta.
6. Pooled training rows are patient-disjoint from every validation split
   (asserted at runtime and in a unit test) — no leakage inflates the reference.
7. `pytest tests/test_baseline.py tests/test_check_fed_vs_pooled.py` passes,
   including the [T3] exit-code cases (pass→0, bagging-fail→≠0, cyclic-over-tol→0,
   NaN→≠0, divergent configs→≠0, pyproject-fallback→≠0, round-trip-fail→≠0); the
   [T1] refactor-regression test passes; `eval_final_model.py` still produces its
   table via the shared `score_booster_on_half`.
8. `out/metrics/fed_vs_pooled.json` is written, valid JSON, and carries the
   per-site result for every strategy plus `params`, `total_trees`,
   `config_provenance`, and the calibrated `max_auc_delta`.
9. A `save-model` run stamps the saved model with a `fed_run_config` attribute
   (asserted in `test_server_config.py`); `check_fed_vs_pooled.py` reports
   `config_provenance="model"`, and a model lacking the attribute exits non-zero
   unless `--allow-pyproject-fallback` is passed.

## 8. Risks & notes

1. **The bound is a sanity gate, not equality — and it is calibrated, not
   asserted.** Bagging merges 20×2 independently-boosted trees; the pooled model
   boosts 40 sequentially; `subsample=0.8`/`colsample_bytree=0.8` add variance. So
   a no-bug delta is genuinely nonzero. At n~380 with ~0.15-wide marginal AUC CIs,
   a hand-guessed 0.03 could false-fail a clean run. The tolerance is therefore
   **measured** (§6 step 0) from the actual no-bug delta distribution across seeds
   and recorded in `docs/logbook.md`, not picked by assertion. The AUC gate is
   deliberately coarse; the fine-grained serialization detector is the round-trip
   tripwire (note 9), not this bound.
2. **Cyclic is reported, not gated.** Cyclic can leave one site's AUC
   systematically lower than pooled — a known non-bug (1.3.d's concern). Hard-
   gating it would red the CI on expected behavior, so cyclic per-site deltas are
   printed and written as `INFO` and do NOT affect the exit code. Bagging is the
   ensemble most directly comparable to pooling and is the hard gate; a bagging
   failure is the strongest infra-bug signal. The per-site table still makes a
   cyclic outlier diagnosable (every strategy → infra; only cyclic → bias).
3. **Single-class validation split.** A 20% holdout with only one class present
   makes AUC undefined (`compute_binary_metrics` returns NaN). The check treats a
   NaN on either side as a **hard fail** rather than a silent skip — it means the
   check could not actually be performed and the data needs a look. Unlikely at
   this split size on `3M Death`, but must not pass quietly.
4. **Schema-agnostic.** The check reads `FEATURE_COLS`/`TARGET_COL` from
   `schema.py`, so it keeps working when the frozen feature set grows beyond
   today's two columns (`Age (calc.)`, `NIH on admission`).
5. **Expected neighbourhood.** Current federated bagging AUC-ROC is 0.6898
   (half_A) / 0.6608 (half_B) (`out/metrics/bagging.md`). The pooled reference is
   expected in the same range; this is the sanity anchor the §6 step-0 calibration
   starts from before it measures the actual no-bug delta and sets the tolerance.
6. **`tomllib` needs Python 3.12.** Run the script under the app's `.venv`
   (`requires-python >=3.12`), not a system Python 3.9. `tomllib` is only reached
   on the `pyproject-fallback` path now, but keep it available.
7. **Params provenance is the model, not `pyproject.toml` (A1 resolution).** flwr's
   effective params = `pyproject.toml` + any `--run-config` override at launch, so
   re-reading `pyproject.toml` could silently give the pooled reference *different*
   params than a model was actually trained with. `server_app` therefore stamps the
   resolved config into the saved model (`set_attr("fed_run_config", …)`), and the
   check reads params + budget from there, asserting all federated models agree.
   The §6 matrix overrides only `train-method`/`cyclic-order` today, so this is a
   latent gap being closed before it can bite. Cost: 3 additive lines in
   `server_app` + one producer-side test. The `fed_run_config` attribute is also a
   useful audit record of exactly how each saved model was trained.
8. **Single source for the tree budget (A2 resolution).** The old plan had two
   numbers — config `total-trees` (pooled) and CLI `--expected-trees` (fed check) —
   with nothing asserting they matched, so a mismatch would silently train the
   pooled reference at a different budget. `--expected-trees` is dropped; the
   budget comes from the models' embedded `total_trees`, used for both pooled
   training and the fed budget assert. One number, no operator drift.
9. **AUC is rank-only; the round-trip is the real serialization tripwire
   (outside-voice #1).** AUC-ROC is invariant to any monotonic score transform, so
   a serialize/merge bug that preserves ranking (a wrong `base_score`, a scaling
   defect) gives delta 0 and would *pass* the AUC gate — precisely the bug class
   1.d names. `assert_prediction_roundtrip` (predict → re-serialize → reload →
   predict, per-row exact) catches that class directly and gates independently of
   the AUC deltas. Note: this is a same-model round-trip, not a fed-vs-pooled
   per-row diff — bagging and pooled are different ensembles and *should* predict
   differently per row, so per-row equality between them would false-fail.
10. **Determinism + no silent provenance fallback (outside-voice #4, #6).** The
    pooled reference is retrained on the check machine every run, so it must be
    deterministic: `seed` is pinned in `pyproject.toml` (flows into the embedded
    config) **and `train_pooled_booster` overrides `nthread=1`** (removes the
    multi-thread float-nondeterminism axis); its determinism is unit-tested.
    `xgboost>=2.0` is unpinned — if a future version wobbles AUC by ±0.01 the
    calibrated tolerance must be re-measured (consider pinning `xgboost` or freezing
    the pooled model as a checked artifact if this ever bites). And
    `pyproject-fallback` provenance is a HARD FAIL, not a warning: a `set_attr`
    persistence regression must not silently revert A1 to `pyproject.toml`-reread
    while the gate stays green.

## 9. Review decisions — audit trail

**plan-eng-review (2026-07-18):** grounded against the code (verified the split
contract, the `unflatten_dict` idempotency, patient-disjoint halves, cited AUCs,
and `set_attr` JSON round-trip). Three decisions taken:

- **A1 — params/budget provenance.** `load_matched_config` reading only
  `pyproject.toml` could diverge from a model's actual `--run-config`-resolved
  params. **Decision: persist the resolved run config into the saved model**
  (`server_app.set_attr("fed_run_config", …)`); the check reads params + budget
  from each model and asserts they match. `pyproject` is fallback-only. (§3, §4.0,
  §4.1, §4.3, §8.7)
- **A2 — tree-budget source.** Config `total-trees` and CLI `--expected-trees`
  were two unsynchronised numbers. **Decision: drop `--expected-trees`; derive the
  budget from the models' embedded config** — one source. (§4.2, §8.8)
- **Tests — full coverage.** **Decision: add the [T1] refactor-regression test,
  the [T2] `score_booster_on_half` unit test, and the [T3] CLI exit-code e2e**, on
  top of the original pure-logic set; and pin the `load_matched_config` test to
  the FL-path derivation rather than a hand-copied dict (C1). (§4.6)

Minor: §5 TODOS note reframed (the 1.c reuse item is already DONE — C2); the
pooled-combined AUC stays as informational-only (small non-essential complexity,
kept for diagnostics).

**Outside voice (independent Claude subagent, 2026-07-18; Codex was unavailable —
401 auth).** Four further decisions, all accepted, folded in:

- **Tolerance is calibrated, not asserted.** At n~380 with ~0.15-wide CIs a
  guessed 0.03 risks false-fail; §6 step 0 measures the no-bug delta distribution
  across seeds and sets the tolerance from it. (§1, §4.2 `--calibrate`, §6, §8.1)
- **Bagging hard-gates; cyclic informational.** The old plan hard-failed on any
  strategy/site over tol, contradicting §8.2's "cyclic bias is a known non-bug."
  Cyclic deltas are now `INFO`, not gating. (§1, §4.2, §7.2, §8.2)
- **Serialization round-trip tripwire added.** AUC is rank-only and blind to a
  monotonic-distorting serialize/merge bug; `assert_prediction_roundtrip` (same-
  model predict → re-serialize → reload → predict, per-row exact) catches that
  class directly. (§1, §4.1, §4.2 step 4, §8.9)
- **Determinism + hard-fail on fallback.** Pinned `seed` for a reproducible pooled
  reference; `pyproject-fallback` provenance is a hard fail (not a warning) so a
  `set_attr` regression can't silently revert A1. (§4.1, §4.2 step 1, §5, §8.10)

Reframed one outside-voice claim: its proposed *fed-vs-pooled per-row equality*
check is wrong (the two are different ensembles by design); per-row equality
belongs to the same-model round-trip, which is what was added.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | unavailable | 401 auth — fell back to Claude subagent |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 2 arch (A1,A2) + 2 quality (C1,C2) + 3 test gaps, all folded in |
| Outside Voice | Claude subagent | Independent challenge | 1 | issues_found→resolved | 4 accepted: calibrate tol, bagging-only gate, round-trip tripwire, determinism/fallback |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **CROSS-MODEL:** Codex unavailable (401); Claude subagent supplied the outside
  voice. It surfaced the tolerance-calibration and rank-only-AUC blind spots the
  eng review missed; one of its proposed fixes was reframed (see §9).
- **UNRESOLVED:** none — all eng-review (A1, A2, tests) and outside-voice
  (calibration, gating, round-trip, determinism) decisions taken and applied.
- **VERDICT:** ENG CLEARED — spec revised, ready to implement.
