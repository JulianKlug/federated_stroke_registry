# Spec 1.c — Evaluation harness: site-stratified AUC-ROC, AUC-PR, Brier, confusion matrix

Implements roadmap item 1.c ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Evaluation harness: site-stratified AUC-ROC, AUC-PR, Brier, and confusion
> matrix at the operating point, per architecture §4.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§4) and the installed framework (`flwr==1.31.0`, message-based API). Builds
directly on 1.b ([1b_fedxgb_bagging_cyclic.md](1b_fedxgb_bagging_cyclic.md)),
whose §4.7 script and §8 notes explicitly deferred the real metric harness to
here.

## 1. Goal & scope

Architecture §4 requires:

> Report AUC-ROC, AUC-PR, calibration (Brier score), and confusion matrix at
> the operating point, **stratified by site**.

Today the harness reports one number. `client_app.evaluate()`
([architecture/fed_stroke/client_app.py:99](../../architecture/fed_stroke/client_app.py))
computes a single AUC via `bst.eval_set` and returns it in a `MetricRecord`;
the strategy's default `evaluate_metrics_aggr_fn`
(`flwr…strategy_utils.aggregate_metricrecords`) then **weighted-averages that
AUC across sites** — which is exactly the cross-site collapse §4 forbids.

**In scope**

- Client-side computation of the full §4 metric set (AUC-ROC, AUC-PR, Brier,
  confusion matrix) on each site's local validation split, inside the existing
  `@app.evaluate()`. The confusion matrix is reported at **two** operating
  points: a shared fixed one (cross-site comparable) and a per-site data-driven
  one (Youden-J, meaningful for the imbalanced outcome) — §4.1.
- **Bootstrap confidence intervals** on the three scalar metrics, so tiny per-site
  splits are not over-read (§4.1).
- A custom evaluate-metrics aggregator that **preserves per-site
  stratification** (no averaging) for **both** strategies.
- Per-round, per-site metrics **logged** and **persisted to a schema-validated,
  valid-JSON artifact** under `out/metrics/`, ready for the v1.1.e reporting
  notebook (§4.3).
- Upgrade of 1.b's debugging-grade `eval_final_model.py` to the same shared
  metric code, turning it into a Geneva-local correctness cross-check.
- Unit tests for the metric computation, the aggregator, the artifact
  encode/decode + validator, and the `evaluate()` wiring.

**Out of scope** (later roadmap items)

- Federated-vs-pooled correctness check (the ±3 AUC-point bound) → 1.d.
- Choosing a winning strategy or quantifying last-site bias formally → Phase
  v1.3 / 1.3.d, on real cross-site data.
- DP noise on the metrics/histograms → Phase v1.1/v1.2.
- Any centralized/server-side evaluation dataset — the server never sees data
  (§4); all computation is client-side.

## 2. Dependencies

- 1.b complete: both strategies (`FedXgbBagging`, `OrderedFedXgbCyclic`)
  selectable from config; the two Geneva halves and the 2-SuperNode topology
  exist.
- **No new runtime dependencies.** `scikit-learn>=1.3` is already a project
  dependency ([architecture/pyproject.toml](../../architecture/pyproject.toml))
  and supplies every metric (`roc_auc_score`, `average_precision_score`,
  `brier_score_loss`, `confusion_matrix`).

## 3. Framework facts that drive the design

Verified against the installed `flwr==1.31.0` sources.

- **The default evaluate aggregator collapses sites.**
  `aggregate_metricrecords` (`strategy_utils.py`) weighted-averages every
  metric key across all replies by `num-examples`. Feeding it richer metrics
  would just average AUC-PR and Brier too — still one number, no
  stratification. The aggregator itself must be replaced.
- **The aggregator receives the full reply `RecordDict`.**
  `FedAvg.aggregate_evaluate` (`fedavg.py`) calls
  `self.evaluate_metrics_aggr_fn([msg.content for msg in valid_replies], key)`.
  `msg.content` is the entire reply `RecordDict` — **config records included**
  — so a custom aggregator can read a site name the client attaches in a
  `ConfigRecord`. This is the same channel 1.b's `site_info` query already uses
  (`MetricRecord` values are numeric-only in flwr, so the site *string* must
  travel in a `ConfigRecord`).
- **Both strategies accept the hook.** `FedXgbBagging.__init__` (and its
  `FedAvg` base, which `OrderedFedXgbCyclic` also derives from) take
  `evaluate_metrics_aggr_fn: Callable[[list[RecordDict], str], MetricRecord]`.
  One aggregator wires into both via `build_strategy`.
- **Per-round aggregated metrics surface on the result.**
  `strategy.start(...)` returns a `Result` whose
  `evaluate_metrics_clientapp: dict[int, MetricRecord]` is round-indexed —
  this is where the persisted JSON is read from.
- **`xgboost` gives probabilities directly.** With `binary:logistic`,
  `bst.predict(dmatrix)` returns per-row probabilities; `dmatrix.get_label()`
  returns the labels. Both AUC-PR and Brier need these probabilities, so the
  harness computes from `predict`, not `eval_set`.

Consequence: the change is three coordinated pieces — (a) a shared metric
function, (b) a client that returns the full set plus its site tag, (c) a
site-preserving aggregator — plus persistence and the offline cross-check.

## 4. Design

### 4.1 New module `architecture/fed_stroke/metrics.py`

Single source of truth for metric math, imported by both the federated client
(§4.2) and the offline scorer (§4.4) so their numbers are identical by
construction. Imports: `numpy`, `math`, and from `sklearn.metrics`
`roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
roc_curve`.

```python
# Metric keys every valid per-site artifact entry must contain (§4.3 validator).
REQUIRED_METRIC_KEYS = {
    "auc_roc", "auc_pr", "brier", "tn", "fp", "fn", "tp",
    "op_j", "tn_j", "fp_j", "fn_j", "tp_j", "n_pos", "num-examples",
}


def _bootstrap_ci(y_true, y_prob, metric_fn, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CI for metric_fn; (nan, nan) if it can't be formed.

    Resamples that land single-class are skipped (the AUC metrics are undefined
    there); if every resample is single-class, returns (nan, nan).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():          # single-class resample → skip
            continue
        stats.append(metric_fn(yt, yp))
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _youden_threshold(y_true, y_prob) -> float:
    """Threshold maximizing TPR − FPR (Youden's J) on the site's own ROC."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_binary_metrics(y_true, y_prob, operating_point=0.5,
                           n_boot=1000, boot_seed=0) -> dict:
    """Full §4 metric set (+ bootstrap CIs + data-driven point) for one split.

    Flat, MetricRecord-safe values:
      auc_roc, auc_roc_lo, auc_roc_hi   # + bootstrap 95% CI
      auc_pr,  auc_pr_lo,  auc_pr_hi
      brier,   brier_lo,   brier_hi
      tn, fp, fn, tp                    # confusion @ fixed operating_point
                                        #   (shared threshold → cross-site comparable)
      op_j, tn_j, fp_j, fn_j, tp_j      # confusion @ per-site Youden-J point
                                        #   (per-site threshold → NOT cross-site comparable)
      n, n_pos
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= operating_point).astype(int)

    # confusion matrix over the fixed {0,1} label space so counts are stable
    # even when a site's split is single-class (labels=[0, 1] avoids a 1x1 cm)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    single_class = n_pos == 0 or n_pos == len(y_true)

    out = {
        "auc_roc": float("nan"), "auc_roc_lo": float("nan"), "auc_roc_hi": float("nan"),
        "auc_pr":  float("nan"), "auc_pr_lo":  float("nan"), "auc_pr_hi":  float("nan"),
        "brier":   float(brier_score_loss(y_true, y_prob)),
        "brier_lo": float("nan"), "brier_hi": float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # data-driven point: undefined single-class → op_j NaN, cells mirror fixed
        "op_j": float("nan"),
        "tn_j": int(tn), "fp_j": int(fp), "fn_j": int(fn), "tp_j": int(tp),
        "n": int(len(y_true)), "n_pos": n_pos,
    }
    if single_class:
        # AUCs, their CIs, and Youden's J are all undefined with one class
        # present — leave them NaN, do not raise. brier and the fixed-point
        # confusion are still valid and already set above.
        return out

    out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    out["auc_pr"] = float(average_precision_score(y_true, y_prob))
    out["auc_roc_lo"], out["auc_roc_hi"] = _bootstrap_ci(
        y_true, y_prob, roc_auc_score, n_boot, boot_seed)
    out["auc_pr_lo"], out["auc_pr_hi"] = _bootstrap_ci(
        y_true, y_prob, average_precision_score, n_boot, boot_seed)
    out["brier_lo"], out["brier_hi"] = _bootstrap_ci(
        y_true, y_prob, brier_score_loss, n_boot, boot_seed)

    op_j = _youden_threshold(y_true, y_prob)
    tn_j, fp_j, fn_j, tp_j = confusion_matrix(
        y_true, (y_prob >= op_j).astype(int), labels=[0, 1]).ravel()
    out["op_j"] = op_j
    out["tn_j"], out["fp_j"], out["fn_j"], out["tp_j"] = (
        int(tn_j), int(fp_j), int(fn_j), int(tp_j))
    return out
```

- **Single-class guard (§8).** The small Geneva halves' 20% validation split can
  land all-negative for a rare outcome (`3M Death`). `roc_auc_score` /
  `average_precision_score` / `roc_curve` are undefined there; the harness
  returns `NaN` for the AUCs, their CIs, and `op_j` so the round still completes
  and the gap is visible in the artifact. Confusion counts and `brier` are still
  well-defined and computed. `labels=[0, 1]` forces the 2×2 shape even
  single-class, so `.ravel()` into `(tn, fp, fn, tp)` never misaligns.
- **Two operating points.** The **fixed** point (`operating_point`, default
  `0.5`, §4.5) gives a confusion matrix that is directly comparable across sites
  because every site thresholds identically. But `0.5` is degenerate for a rare
  outcome (`tp≈fp≈0`), so the harness *also* reports a **per-site data-driven**
  point (`op_j`, Youden's J on that site's ROC) with its own cells
  (`*_j`) — meaningful for imbalanced outcomes. The `_j` cells are **not**
  cross-site comparable (each site uses a different threshold); the fixed cells
  are. Both travel in the artifact, clearly separated by the `_j` suffix.
- **Bootstrap CIs.** `auc_roc`/`auc_pr`/`brier` each carry a percentile 95% CI
  (`_lo`/`_hi`) so tiny-`n_pos` per-site numbers are not over-read (§8). `n_boot`
  and `boot_seed` come from config (§4.5) — deterministic given the seed. Cost
  is ~1k cheap resamples over a few-hundred-row split, sub-second per evaluate.

### 4.2 Client: full metric set + site tag (`client_app.py`)

Replace the `bst.eval_set` single-AUC block in `evaluate()` (lines ~112–124):

```python
@app.evaluate()
def evaluate(msg, context):
    _, valid_dmatrix, _, num_val = load_data_gva(context)
    cfg = replace_keys(unflatten_dict(context.run_config))
    bst = xgb.Booster(params=cfg["params"])
    bst.load_model(bytearray(msg.content["arrays"]["0"].numpy().tobytes()))

    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    metrics = compute_binary_metrics(
        y_true, y_prob,
        context.run_config["operating-point"],
        n_boot=context.run_config["n-boot"],
        boot_seed=context.run_config["boot-seed"],
    )
    # `n` (row count from compute_binary_metrics) IS the weighting quantity; rename
    # it to the key flwr's weighting path expects rather than carry a duplicate.
    metrics["num-examples"] = metrics.pop("n")

    site = Path(context.node_config["data-path"]).name   # same id as site_info
    return Message(
        content=RecordDict({
            "metrics": MetricRecord(metrics),
            "config": ConfigRecord({"site": site}),
        }),
        reply_to=msg,
    )
```

- The `site` expression is exactly the one already used by `site_info`
  (`client_app.py:47`) — reuse, do not re-derive.
- **One row-count key.** `compute_binary_metrics` returns `n`; the client renames
  it in place to `num-examples` (the key flwr's consistency check and weighting
  path expect), so the reply carries exactly one row-count value, not a duplicate
  `n` + `num-examples`. The custom aggregator (§4.3) re-tags it per site rather
  than averaging on it. The offline scorer (§4.4) calls `compute_binary_metrics`
  directly and keeps the `n` name.
- `train()` and the `site_info` query are untouched.

### 4.3 Server: site-preserving aggregator + persistence (`metrics.py` + `server_app.py`)

**Aggregator** (in `metrics.py`, signature matches flwr's hook):

```python
def site_stratified_evaluate_metrics(records, weighting_metric_name) -> MetricRecord:
    """Emit per-site metrics with NO cross-site averaging.

    Each reply RecordDict carries a MetricRecord + a ConfigRecord{"site": …}.
    Output keys are site-suffixed: "auc_roc/geneva_half_A.parquet", etc.
    """
    out = MetricRecord()
    seen = set()
    for rec in records:
        site = str(next(iter(rec.config_records.values()))["site"])
        if site in seen:                     # two nodes claiming one file → misconfig
            raise ValueError(f"Duplicate site in evaluate replies: {site}")
        seen.add(site)
        mr = next(iter(rec.metric_records.values()))
        # Every key is site-suffixed, num-examples included: it becomes the
        # per-site row count in the artifact rather than an averaging weight.
        for k, v in mr.items():
            out[f"{k}/{site}"] = v
    return out
```

- `/` separator keeps keys readable and parses back cleanly (see the decoder
  below). Site names are parquet file names (no `/`), so `rsplit("/", 1)`
  recovers the site unambiguously even if a metric name ever contained a `/`.
- `num-examples` is deliberately re-tagged per site (`num-examples/<site>`), not
  dropped: the per-site row count is useful context for reading a NaN AUC. This
  differs from flwr's default aggregator, which excludes the weighting key.
- Raising on a duplicate site defends against a federation where two SuperNodes
  are misconfigured to the same data file — silently overwriting would report
  one site twice and hide the other.

**Decoder + schema validator** (also in `metrics.py`, beside the encoder so the
two never drift):

```python
def nest_site_metrics(evaluate_metrics_clientapp) -> dict:
    """Inverse of the aggregator's f"{metric}/{site}" encoding.

    Returns {round(int): {site(str): {metric(str): value}}}. NaN → None so the
    artifact is standards-valid JSON; pandas reads null back as NaN, so the
    v1.1.e notebook is unaffected.
    """
    nested = {}
    for rnd, mr in evaluate_metrics_clientapp.items():
        for key, value in mr.items():
            metric, site = key.rsplit("/", 1)
            v = None if isinstance(value, float) and math.isnan(value) else value
            nested.setdefault(int(rnd), {}).setdefault(site, {})[metric] = v
    return nested


def validate_metrics_artifact(nested) -> None:
    """Structural contract check before the artifact is written.

    Guards the interface the v1.1.e reporting notebook consumes; raises
    ValueError on any shape it cannot rely on. Cheap — runs once per run.
    """
    if not nested:
        raise ValueError("empty metrics artifact — no evaluate rounds captured")
    for rnd, sites in nested.items():
        if not isinstance(rnd, int):
            raise ValueError(f"round key {rnd!r} must be int")
        if not sites:
            raise ValueError(f"round {rnd} has no sites")
        for site, metrics in sites.items():
            missing = REQUIRED_METRIC_KEYS - metrics.keys()
            if missing:
                raise ValueError(
                    f"round {rnd} site {site}: missing metric keys {sorted(missing)}"
                )
```

**Artifact schema** (the single contract v1.1.e depends on — documented here and
enforced by `validate_metrics_artifact`):

```
{ <round:int> : { <site:str> : { <metric:str> : <number|null> } } }
```

- **Round keys** are ints (JSON stringifies them on write; the notebook must
  cast back). **Site keys** are parquet file names (e.g. `geneva_half_A.parquet`).
- **The site set varies by round under cyclic** (one site per round, alternating)
  and is both sites every round under bagging — consumers must not assume a fixed
  site set per round (§8).
- **AUC-family values may be `null`** (single-class split → NaN → null). `brier`
  and all confusion cells are always present and numeric.
- Each site entry contains `REQUIRED_METRIC_KEYS` (§4.1): the fixed-point cells
  (`tn,fp,fn,tp`), the Youden-J point and cells (`op_j,tn_j,…`), the three AUCs +
  their `_lo/_hi` CIs, `n_pos`, and `num-examples`.

**Wiring** — in `build_strategy` (`server_app.py:44`), pass the aggregator to
both branches:

```python
FedXgbBagging(..., evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics)
OrderedFedXgbCyclic(..., evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics)
```

**Wiring** — in `build_strategy` (`server_app.py:44`), pass the aggregator to
both branches:

```python
FedXgbBagging(..., evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics)
OrderedFedXgbCyclic(..., evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics)
```

**Persistence** — after `strategy.start(...)` in `main()`, decode, validate, and
write. All three decode/validate/encode steps are library calls into `metrics.py`;
`main()` only orchestrates and writes the file:

```python
nested = nest_site_metrics(result.evaluate_metrics_clientapp)  # {round:{site:{metric}}}, NaN→None
validate_metrics_artifact(nested)                              # contract check, raises on bad shape

metrics_dir = Path(context.run_config["metrics-dir"])
metrics_dir.mkdir(parents=True, exist_ok=True)
tag = train_method + (f"_{context.run_config['cyclic-order']}" if train_method == "cyclic" else "")
# allow_nan=False: NaNs are already None; any stray NaN must raise, never write
# the invalid bare `NaN` token that jq / JS / strict parsers reject.
(metrics_dir / f"{tag}.json").write_text(json.dumps(nested, indent=2, allow_nan=False))
```

- `<run-tag>` encodes `train-method` and, for cyclic, `cyclic-order`, so a
  forward and a reverse run do not overwrite each other (mirrors 1.b's R2/R3
  distinction). Directory from a new `metrics-dir` config key, mirroring the
  existing `model-dir` pattern (`server_app.py:117`).
- `main()` also `log()`s a per-site table for the final round (the headline
  numbers), so the run is legible without opening the JSON. The table flags an
  all-zero fixed-point confusion matrix as a degenerate-threshold warning (§7),
  since `0.5` on a rare outcome yields `tp≈fp≈0` (the Youden-J cells stay
  informative).

Under **cyclic**, a round evaluates only the site that just trained (§1.b §3),
so a given round's entry may hold one site; the final artifact still contains
both sites across the run. This is a property of cyclic evaluation, not a bug —
noted in §8. The bagging artifact has both sites every round.

### 4.4 Offline cross-check (`scripts/eval_final_model.py`)

Replace the `bst.eval_set` AUC (`eval_final_model.py:39`) with
`compute_binary_metrics` on `bst.predict`, and print the full metric row per
site (AUC-ROC with its CI, AUC-PR, Brier, both confusion points, `n_pos`). This
upgrades 1.b's debugging tool into a correctness check: on a saved model, each
site's offline AUC-ROC must match that model's federated final-round
`auc_roc/<site>`, because both now call the *same* function on the *same* split
(`generate_splits`, `test_size=0.2`, `seed=42`). Pass the same `operating-point`,
`n-boot`, and `boot-seed` so the numbers are reproducible to full precision, not
just "close". Only usable on Geneva (debug data access); never on Shenzhen —
hence it is a check, not the harness.

### 4.5 Config keys (`pyproject.toml`)

Add to `[tool.flwr.app.config]`:

```toml
operating-point = 0.5      # shared FIXED threshold for the cross-site-comparable confusion matrix (§4.1)
metrics-dir = "out/metrics"  # where the per-run site-stratified JSON is written (relative = ServerApp CWD)
n-boot = 1000              # bootstrap resamples for per-metric 95% CIs (§4.1); determinism via boot-seed
boot-seed = 0              # RNG seed for reproducible bootstrap CIs
```

The **single, shared** `operating-point` (not per-site) is deliberate: it is the
one threshold at which the two sites' confusion matrices are directly comparable.
`0.5` is the smallest defensible default; a clinically-motivated value can be set
later by overriding this one key. The **data-driven** point (Youden-J, §4.1) is
computed per site from data, needs no config, and is reported *alongside* the
fixed one for imbalanced outcomes — it is explicitly not cross-site comparable.
`n-boot`/`boot-seed` are in config (not hard-coded) so the CIs are reproducible
and the config never lies about what produced a number.

### 4.6 Tests (`tests/test_metrics.py`, new)

Mostly fast pure-logic; reuse the `SimpleNamespace`/fake-record style from
`tests/test_strategies.py`. The `evaluate()` case fits a tiny real booster
(reuse `test_client_boost._synthetic_dmatrix`) — still sub-second.

| Cases |
|---|
| **`compute_binary_metrics` — core:** hand-built vector with known AUC-ROC / AUC-PR / Brier and known `(tn, fp, fn, tp)` at threshold 0.5. |
| **Single-class** (all-negative and all-positive): `auc_roc`/`auc_pr`, their CIs, and `op_j` are `nan`; `brier` and both fixed + `_j` confusion cells still correct and equal; no exception. |
| **Non-default fixed operating point** shifts `(tn,fp,fn,tp)` as expected. |
| **Youden-J point:** on a vector where 0.5 is degenerate (`tp=fp=0`), `op_j` differs from 0.5 and the `_j` cells are non-degenerate; `op_j` reproduces on the same input. |
| **Bootstrap CIs:** `_lo ≤ point ≤ _hi` for each of `auc_roc`/`auc_pr`/`brier`; identical `boot_seed` → identical CIs (determinism); `n_boot=0` → CIs `nan`. |
| **`site_stratified_evaluate_metrics`:** two fake replies (`MetricRecord` + `ConfigRecord{site}`) → merged record has both sites' suffixed keys, values **unaveraged**, `num-examples/<site>` present per site; duplicate-site input raises `ValueError`. |
| **`nest_site_metrics` round-trip:** feed aggregator output back through the decoder → `{round:{site:{metric}}}`; a `NaN` metric decodes to `None` (JSON-null), non-NaN passes through; round keys are `int`. |
| **`validate_metrics_artifact`:** a well-formed nested dict passes; missing a `REQUIRED_METRIC_KEYS` entry, an empty artifact, and a non-int round key each raise `ValueError`. |
| **`client_app.evaluate()` wiring** (light integration): monkeypatch `load_data_gva` to return a synthetic `(valid_dmatrix, num_val)` (no real GVA parquet needed), fit a tiny booster and serialize it into `arrays["0"]`, call `evaluate()` with a fake msg/context (run_config carries `params`/`operating-point`/`n-boot`/`boot-seed`; node_config a `data-path` whose `.name` is the expected site). Assert the reply `MetricRecord` has all `REQUIRED_METRIC_KEYS` (incl. `num-examples`, no bare `n`) and the reply `ConfigRecord` carries that `site`. |

## 5. Files to change

| File | Change |
|---|---|
| `architecture/fed_stroke/metrics.py` | **new** — `REQUIRED_METRIC_KEYS`, `compute_binary_metrics` (fixed + Youden-J points, bootstrap CIs), `site_stratified_evaluate_metrics`, `nest_site_metrics`, `validate_metrics_artifact`, plus the documented artifact schema (§4.1, §4.3) |
| `architecture/fed_stroke/client_app.py` | `evaluate()` computes full metric set via shared fn, renames `n`→`num-examples`, attaches site `ConfigRecord`, passes `n-boot`/`boot-seed` (§4.2) |
| `architecture/fed_stroke/server_app.py` | wire aggregator into both strategies; `nest_site_metrics` + `validate_metrics_artifact` + `json.dumps(allow_nan=False)`; log final-round per-site table with degenerate-matrix flag (§4.3) |
| `architecture/scripts/eval_final_model.py` | use shared `compute_binary_metrics`, print full metric row per site; pass matching `operating-point`/`n-boot`/`boot-seed` (§4.4) |
| `architecture/pyproject.toml` | add `operating-point`, `metrics-dir`, `n-boot`, `boot-seed` config keys (§4.5) |
| `architecture/tests/test_metrics.py` | **new** — `compute_binary_metrics` (core, single-class, Youden, CIs), aggregator, `nest_site_metrics` round-trip, `validate_metrics_artifact`, and `client_app.evaluate()` wiring (§4.6) |

## 6. Run matrix

Reuses 1.b's topology and runs; 1.c changes only what `evaluate` returns and
how it is aggregated/persisted.

| Run | `train-method` | `cyclic-order` | Artifact written |
|---|---|---|---|
| R1 | bagging | — | `out/metrics/bagging.json` |
| R2 | cyclic | forward | `out/metrics/cyclic_forward.json` |
| R3 | cyclic | reverse | `out/metrics/cyclic_reverse.json` |

## 7. Acceptance criteria

1. `pytest` suite passes, including the new `test_metrics.py` (metrics core,
   single-class, Youden-J, bootstrap CIs, aggregator, `nest_site_metrics`
   round-trip, `validate_metrics_artifact`, `evaluate()` wiring) and the existing
   `test_strategies.py`.
2. R1 (bagging) completes; server log shows a per-site table with AUC-ROC (+CI),
   AUC-PR, Brier, and **both** confusion points for **both** halves; `bagging.json`
   contains two site entries in every round. An all-zero fixed-point confusion
   matrix is logged as a **degenerate-threshold warning**, not silently passed.
3. **Stratification proof:** the two sites' `auc_roc` values in `bagging.json`
   differ — evidence they were not averaged into one number. (Note §8: on the
   IID Geneva halves this difference may be sampling noise; it proves *no
   averaging*, not genuine site heterogeneity — the latter needs real cross-site
   data.)
4. R2/R3 (cyclic) complete; each writes its own tagged artifact; across the run
   both sites appear (per-round single-site evaluation under cyclic is expected,
   §8).
5. **Offline cross-check:** for R1's saved model,
   `eval_final_model.py --expected-trees 40` reports per-site AUC-ROC matching
   the federated final-round `auc_roc/<site>` within floating-point tolerance
   (same `operating-point`/`n-boot`/`boot-seed` → exact match, not just close).
6. Single-class handling (NaN AUCs/CIs/`op_j`, valid Brier + confusion) is
   covered by the unit test (no live run needed).
7. **Artifact is valid JSON** and passes `validate_metrics_artifact`: readable by
   a strict parser (`jq .`), NaNs rendered as `null`.
8. Final-round per-site metrics for all three runs recorded in
   [docs/logbook.md](../logbook.md).

## 8. Risks & notes

- **Cyclic evaluation is single-site per round.** `FedXgbCyclic` evaluates the
  same node that just trained (1.b §3), so a cyclic round's artifact entry
  holds one site, and any given round's per-site metric is a self-evaluation.
  This is inherent to cyclic and unchanged by 1.c; the harness records it
  faithfully rather than papering over it. Cross-run / cross-site comparison
  (last-site bias, strategy choice) is 1.d / Phase v1.3 work, using both-halves
  scoring as in 1.b §4.7.
- **Single-class validation splits → NaN AUCs.** On the small Geneva halves a
  rare outcome can yield an all-negative 20% split. The harness returns `NaN`
  for the AUCs, their bootstrap CIs, and the Youden-J point (confusion counts
  and Brier still valid) so the round completes and the gap is explicit in the
  artifact (as JSON `null`); it is a data-size signal, not a code failure.
  Stratifying the split by outcome is a preprocessing-track concern, not 1.c.
- **The two "sites" are IID halves of Geneva, so stratification runs against
  data with no real cross-site shift.** Per-site metrics on IID halves are
  near-identical by construction; acceptance criterion 3 therefore proves the
  aggregator does not average, but says nothing about whether the harness would
  *surface* genuine heterogeneity. The real target — Geneva-vs-Shenzhen
  divergence — is only exercised once Shenzhen connects (Phase v1.3). Reading a
  small A-vs-B gap here as a site effect is a trap; the bootstrap CIs are the
  guard against over-reading it.
- **Cyclic per-round metrics are in-distribution self-evaluations.** A cyclic
  round evaluates the site that just trained, on a model that just ingested that
  site's own data, so its AUC is optimistic relative to a bagging round (merged
  model, both sites). The two artifacts (`cyclic_*.json` vs `bagging.json`) are
  therefore **not** directly comparable for strategy selection — that comparison
  needs both-halves scoring of each saved final model (1.b §4.7 / `eval_final_model.py`),
  which is 1.d / v1.3 work. 1.c records the raw per-round numbers faithfully; it
  does not assert cross-strategy comparability.
- **Per-site confusion cells + `n_pos` cross the wire every round.** This is a
  richer disclosure than 1.b's single AUC scalar. Reviewed and accepted for the
  Geneva-local pilot (both halves are debug-accessible). Small-cell suppression
  (masking cells below a governance threshold) is a candidate **before Shenzhen
  connects**, tracked with the DP work (Phase v1.1/v1.2), not built in 1.c.
- **`num-examples` no longer averages anything.** It stays in the reply because
  the strategy's consistency/weighting path expects the key, but the custom
  aggregator re-tags it per site rather than using it as a weight — deliberate,
  since stratified reporting must not weight-average.
- **Config never lies.** The single shared `operating-point` is written in
  config and used verbatim by every site; there is no hidden per-site
  threshold.
- **Shenzhen readiness.** All metric computation is client-side and only
  aggregate numbers cross the wire (§4 / architecture privacy stack). The
  offline `eval_final_model.py` is Geneva-debug-only and never part of the
  Shenzhen path — the federated harness is the sole production route, which is
  the whole reason metrics are computed in `@app.evaluate()` rather than by
  scoring a pulled model.

## 9. Review decisions — audit trail (plan-eng-review, 2026-07-16)

**Audit trail only — not a second source of truth.** Every decision below is now
woven into §3–§8 above, which are authoritative for implementation. This section
records *what changed and why* so the reasoning is not lost; if it ever disagrees
with the body, the body wins. Each item points to where it now lives.

**Framework facts (§3) re-verified against installed `flwr==1.31.0`:** default
aggregator collapses sites (`strategy_utils.py:101`); `aggregate_evaluate`
passes full `RecordDict`s incl. config records (`fedavg.py:312`); both
strategies accept `evaluate_metrics_aggr_fn` and default
`weighted_by_key="num-examples"`; `result.evaluate_metrics_clientapp` is
round-indexed (`result.py:56`). `MetricRecord` accepts `NaN`; single-class
`brier_score_loss` returns valid values in sklearn 1.9 (no raise). All correct.

1. **NaN → valid JSON (§4.3).** Before `json.dumps`, walk `nested` replacing
   `float('nan')` with `None`, and dump with `allow_nan=False` so any stray NaN
   raises instead of writing the invalid bare `NaN` token. pandas reads `null`
   back as `NaN`, so notebook semantics are unchanged. (Verified: default
   `json.dumps` emits `NaN`, which `jq`/JS/strict parsers reject.)

2. **Extract the flatten→nest reconstruction (§4.3).** Move the
   `result.evaluate_metrics_clientapp` → `{round:{site:{metric}}}` logic out of
   `server_app.main()` into `metrics.py` (e.g. `nest_site_metrics(...)`), beside
   the aggregator that produces the encoding. Unit-test the aggregator→nest
   round-trip, including a NaN→null case. `main()` shrinks to call + write.

3. **Aggregator: drop the dead branch (§4.3).** Removed the no-op
   `if k == weighting_metric_name: pass`; the aggregator emits every key
   site-suffixed, `num-examples/<site>` included (per-site row count). The
   `n`/`num-examples` duplicate is collapsed in the client (§4.2:
   `metrics["num-examples"] = metrics.pop("n")`).

4. **Test `evaluate()` wiring (§4.6).** Add a light integration test: fit a tiny
   booster on a synthetic DMatrix (reuse `test_client_boost._synthetic_dmatrix`),
   call `evaluate()` with a fake msg/context (msg carries the model bytes in
   `arrays["0"]`), assert the reply has a `MetricRecord` with all §4 keys +
   `num-examples` and a `ConfigRecord{site}`.

5. **Small-cell disclosure — reviewed, NOT addressed in 1.c.** The full
   confusion cells + `n_pos` per site per round on the fixed seed=42 split are a
   richer wire disclosure than today's single AUC. Decision: accept for the
   Geneva-local pilot (both halves are debug-accessible anyway); no suppression
   this cycle. Small-cell suppression remains a candidate before Shenzhen joins.

6. **Threshold-free confusion summary (§4.1, §4.5, §7).** The fixed 0.5 point is
   degenerate for the rare `3M Death` outcome (`tp≈fp≈0`). `compute_binary_metrics`
   now also emits cells at a per-site data-driven point (`op_j`, Youden-J on the
   ROC), alongside the fixed 0.5 cells. The `_j` cells are per-site and **not**
   cross-site comparable; the fixed cells stay the comparable pair. §7 criterion
   2 flags an all-zero fixed matrix as a degenerate-threshold warning.

7. **JSON schema contract + validator (§4.1 `REQUIRED_METRIC_KEYS`, §4.3).** The
   artifact schema (round→site→metric; variable site sets under cyclic; `null`
   AUCs) is documented in §4.3 and enforced by `validate_metrics_artifact` before
   write. Prevents a silent KeyError/mislead when the v1.1.e notebook consumes it.

8. **Bootstrap confidence intervals (§4.1, §4.5).** `compute_binary_metrics` adds
   `_lo`/`_hi` percentile CIs for `auc_roc`/`auc_pr`/`brier` so tiny-`n_pos`
   per-site metrics carry uncertainty and are not over-read. `n-boot`/`boot-seed`
   are config keys (§4.5) for determinism; cost is sub-second per evaluate.

9. **`TODOS.md`.** The existing item ("1.c should reuse 1.b's split + site helpers")
   is satisfied by this spec; mark it done in the 1.c implementation commit. (Not
   a spec-body element — an action for the implementation PR.)

**Scope note.** Decisions 6–8 grew 1.c from metric plumbing into an evaluation
layer (second operating point, schema validation, bootstrap inference), now fully
reflected in §4.1/§4.3/§4.5/§4.6/§5/§7. Deliberate, per "thoroughness > speed."

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES RESOLVED | 3 review issues + 2 outside-voice issues folded in; 2 deferred |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE VOICE:** Codex unavailable (401 auth); Claude subagent ran — 6 new findings, 2 folded into scope (#2 threshold, #5 schema), 1 rejected (#1 small-cell), 3 noted as downstream/risk.
- **UNRESOLVED:** none — all 10 decisions resolved.
- **VERDICT:** ENG CLEARED — spec is implementation-ready with §9 decisions applied.
