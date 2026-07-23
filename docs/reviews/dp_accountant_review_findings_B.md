# DP accountant review — findings report

- **Subject:** [`dp_accountant_review_packet.md`](dp_accountant_review_packet.md) (gate for roadmap 1.1.b)
- **Code reviewed:** `architecture/fed_stroke/dp/accounting.py`, `architecture/fed_stroke/dp/boost.py`,
  `architecture/tests/dp/test_accounting.py`, `architecture/tests/dp/test_boost.py`
- **Date:** 2026-07-22
- **Method:** independent re-derivation of every math claim in packet §2; full read of the two
  modules and both test files; test suites executed (48/48 pass); §3 σ table reproduced and
  cross-checked against a fresh Opacus 1.6.0 `RDPAccountant`; one targeted Monte-Carlo experiment
  on the mechanism's control flow (F4 below).
- **Verdict:** **Approve with conditions.** The *accounting* (counts, sensitivities, calibration,
  RDP composition, conversion, Laplace algebra) is correct and gate-verified. The *mechanism* has
  one un-noised data-dependent branch (F4) that breaks the post-processing argument and therefore
  the packet's top-level certification as written. Fix is a one-line change with zero accounting
  impact. Conditions: F4 fixed + regression test, F5 added to packet §7, F6/F7 doc/test nits.

---

## 1. Claims verified as CORRECT

| Packet claim | Verification | Result |
|---|---|---|
| 1 — g/h bounds are the sensitivity | Re-derived: `g = p−y ∈ [−1,1]`, `h = p(1−p) ∈ (0,0.25]` exact for `binary:logistic`; one record → one bin per feature; concatenated L2 `= √d·stat`, L1 `= d·stat`. Constants at `boost.py:44-47` match. | ✅ |
| 2 — composition per level, `D·T` | Siblings partition rows ⇒ parallel composition; partition depends on raw data only through *previously noised* releases (legitimate adaptive query selection). `D·T` is additionally an upper bound (early leaves issue fewer queries). `boost.py:120-122`. | ✅ (except F4 carve-out) |
| 3 — `2·D·T` releases (G and H) | Whitening check re-derived: combined Mahalanobis shift `2/σ²` ⇒ per-level RDP `α/σ²`; feeding `2·D·T` at per-step `α/(2σ²)` is equivalent. `boost.py:125-128, 223, 227`. Note: `2/σ²` is an upper bound — g and h cannot be simultaneously extremal (`max_p[(p−y)² + (p(1−p)/0.25)²] ≈ 1.28`, at p≈0.4, y=1), so the count is valid and slightly conservative. Safe direction. | ✅ |
| 4 — noise calibration realizes σ | Per-entry std `σ·√d·stat_L2` against concatenated L2 `√d·stat` ⇒ multiplier σ. Independent per-sibling noise is the correct realization of parallel composition. `boost.py:228-229, 187-189`. | ✅ |
| 5 — RDP composition + improved Balle conversion | Faithful Opacus port: closed form `α/(2σ²)` at q=1 (`accounting.py:111`), composition in linear RDP space *before* the min-over-α (`accounting.py:186-191`), Balle conversion term-for-term (`accounting.py:169-173`), `DEFAULT_ORDERS == RDPAccountant.DEFAULT_ALPHAS` confirmed against installed Opacus 1.6.0 (integer run starts at 12). | ✅ |
| 6 — q = 1.0, no amplification credit | The learner has no subsampling at all (`_grow_trees` uses `all_rows`); factory passes `sample_rate=1.0`. q<1 path exists but is only tested in the conservative direction. | ✅ |
| 7 — Laplace L1 calibration | Algebra re-derived end-to-end: `eps_per = ε/(2DT)`, `b_G = d·G_L1/eps_per = 2DT·d·G_L1/ε`; reported `ε = k·d·G_L1/b_G + k·d·H_L1/b_H` with `k = D·T` sums back to exactly the target. Uses `d`, not `√d`. Pure-DP parallel composition within a level holds. `boost.py:233-241`, `accounting.py:229-240`. | ✅ |
| 8 — leaf clipping is post-processing | Leaf weights summed from already-noised parent histograms; `clip_bound` invariance pinned by test. **Correct as far as it goes — but the broader "everything downstream of the noised histogram is post-processing" premise fails at one branch: see F4.** | ⚠️ see F4 |
| 9 — per-site ε = per-patient guarantee | Sound under the stated assumption (each patient's records at exactly one site). Single-site scope honestly declared. | ✅ |

**Numerical reproduction (k = 160, q = 1, δ = 1e-5).** `noise_multiplier_for_epsilon` returns
σ = 51.171 / 18.888 / 12.050 / 6.699 for ε = 1 / 3 / 5 / 10 — matching the packet's ≈51 / ≈19 /
≈12 / ≈6.6. Each σ fed to a fresh Opacus 1.6.0 `RDPAccountant` (160 steps, q=1.0) returns ε
identical to `account_run` to 9 decimal places (measured relative difference 0.0). Round-trip
`account_run(160, σ, 1.0, 1e-5)` ≤ target in all cases (never over-spends).

**Tests.** Both suites pass (48/48). All test line references in the packet are accurate and the
cited tests assert what the packet says they assert. §5's list of what the Opacus gate does NOT
cover is accurate.

---

## 2. Findings

Numbering continues the packet's §6 (F1–F3 were caught pre-review).

### F4 — BLOCKING: the empty-node early exit is an un-noised data-dependent branch

`boost.py:434` (inside `_grow_trees.build`):

```python
if depth >= max_depth or rows.shape[0] == 0:
    return {"leaf": ...}
```

`rows.shape[0] == 0` reads the **raw** partition and controls the **released** tree structure
without passing through any noised query. Under add/remove adjacency: a node holding zero records
becomes a leaf with probability **1** (the histogram query is never issued); the neighbouring
dataset, where the added record routes to that node, issues the query and can split.

**Quantified.** Monte-Carlo estimate of P(a noise-only histogram yields a valid split) at the
calibrated σ values (d=2, B=32, `min_child_weight=5`, λ=1, 20 000 trials each):

| σ (target ε at k=160) | P(split from pure noise) |
|---|---|
| 6.70 (ε=10) | 0.376 |
| 12.05 (ε=5) | 0.542 |
| 51.17 (ε=1) | 0.686 |

So the structurally distinguishing output event ("this node is internal") has conditional
probability 0 in one world and 0.38–0.69 in the neighbouring world. No (ε, δ=1e-5) bound covers a
0-vs-0.4 gap; for the Laplace arm (pure ε, no δ) the guarantee is broken outright. The noised
`min_child_weight` check makes the trigger *common* in practice, not exotic: under DP-scale noise
(std_h ≈ 4.3 at ε=5) splits routinely send zero real rows to a child.

**Why the proof breaks.** RDP composition + post-processing requires that *whether a query is
issued* be a measurable function of prior releases only. Here issuance depends on the raw data.
This contradicts Claim 8's premise and therefore the packet's headline statement that the reported
ε "is a sound (ε, δ)-DP guarantee for the histograms + tree ensemble".

**Fix (one line, zero budget).** Remove the `rows.shape[0] == 0` early exit — always issue the
noised histogram query at `depth < max_depth`. `np.bincount` over empty rows returns zeros, so the
release is pure noise; the `2·D·T` count already charges for it (D·T was an upper bound assuming
every level queries); parallel composition still holds (the record still perturbs only its own
sibling). Everything downstream then genuinely is post-processing. Behavioral consequence: empty
nodes may split into noise-only subtrees — a utility non-event at depth ≤ 4, and bounded by
`max_depth`. Add a regression test pinning that an empty node still issues exactly one
`mechanism.add_noise` call (e.g. via a counting mechanism stub).

### F5 — assumption gap: `base_score` data-independence is assumed, not enforced

`base_margin = logit(base_score)` is released in every serialized model
(`boost.py:381, 545-546`). The prototype pins `base_score=0.5` (data-independent), but
`BoostParams.from_xgb_params` (`boost.py:104-117`) ingests whatever `base_score` arrives in the
params dict — and XGBoost ≥ 2.0 **auto-derives** `base_score` from the label mean unless explicitly
set. If the FL wiring ever forwards an auto-derived value, label prevalence leaks outside the
accounting. Not a bug in the reviewed code today; a landmine for §4.5 wiring.

**Fix.** Add to packet §7 as an explicit assumption, and (better) assert in DP mode that
`base_score` was explicitly configured rather than defaulted from upstream.

### F6 — doc: `boost.py` line references in the packet are stale (~+6 offset)

All `boost.py` citations in packet §2 are off by about 6 lines (a later edit shifted the file):
`num_histogram_queries` is at 120-122 (packet: 114-116), `num_gaussian_releases` at 125-128
(packet: 119-122), factory σ/std lines at 223-231 (packet: 217-224), Laplace branch at 233-241
(packet: 227-235), `_leaf_weight` at 256 (packet: 250), `_best_split` at 263 (packet: 257),
`DENOM_FLOOR` at 48 (packet: 42). The `accounting.py` and test references are accurate. Cosmetic,
but this packet exists to be checked line-by-line — refresh before circulating.

### F7 — test nit: the H-side single-bin invariant is unpinned

`test_histogram_sensitivity_bound` (`test_boost.py:58-86`) asserts `changed.size == 1` only for G,
not for H, and checks the *summed* delta rather than the per-bin delta. Equivalent as long as
exactly one bin changes, but the H half of the invariant Claim 1 rests on is not directly pinned.
One extra assertion closes it.

---

## 3. Conditions for sign-off

1. **F4 fixed** (always-query) **+ regression test**, before any DP run on real Geneva data.
   Re-run both suites and the demo after the change; the accounting and σ table are unaffected.
2. **F5 recorded in packet §7** (or enforced with an assertion in DP mode).
3. **F6 line references refreshed; F7 assertion added** (non-blocking).

Nothing in the review contradicts the accountant itself: `accounting.py` can be considered
verified as-is (Opacus-equivalent at q=1, machine precision; conservative at q<1; Laplace algebra
exact). The condition is confined to the mechanism's control flow in `boost.py`.
