# DP accountant review — joint summary of both independent reports

- **Subject:** `dp_accountant_review_packet.md` (gate for roadmap 1.1.b)
- **Reviewer A verdict:** **BLOCK**
- **Reviewer B verdict:** **Approve with conditions** (one blocking mechanism bug, F4)
- **Joint verdict:** **Do not sign off 1.1.b for real Geneva data yet.** The accountant
  mathematics is independently confirmed correct by both reviewers; the blockers are all in the
  *mechanism and pipeline* around it. Six blocking items must be fixed (list below).

## Where the reviewers agree: the accounting itself is correct

Both reviewers independently re-derived the math and reproduced the σ table, with essentially
identical results:

- Gradient/hessian sensitivity bounds (`g ∈ [−1,1]`, `h ∈ (0,0.25]`), √d (L2) / d (L1)
  aggregation — correct.
- Composition count `2·D·T` Gaussian releases, parallel composition across siblings — correct
  (B notes it is even slightly conservative).
- Gaussian calibration (`std_G = σ√d`, `std_H = 0.25σ√d`) realizes multiplier σ — correct.
- RDP composition + improved Balle conversion is a faithful Opacus port. A verified the order
  grid against current official Opacus source (couldn't install opacus locally); B verified
  against an installed Opacus 1.6.0 `RDPAccountant` to machine precision (0.0 relative diff over
  160 steps).
- No subsampling credit taken (q=1) — correct.
- Laplace L1 calibration uses `d` not `√d` and sums back exactly to target ε — correct.
- σ table reproduced identically by both at k=160, δ=1e-5:
  σ = 51.17 / 18.89 / 12.05 / 6.70 for ε = 1 / 3 / 5 / 10, round-trip never over-spends.

`accounting.py` can be considered verified as-is.

## Blocking findings (union of both reports — largely disjoint, all must be fixed)

| # | Source | Finding | Fix direction |
|---|--------|---------|----------------|
| 1 | A-1 | **DP noise is deterministic from public inputs.** RNG seeded from `(seed, round, site_hash)` (`client_app.py:126-128`, `boost.py:526-527`). Reproduced: neighboring data + same seed → distinguishable models with probability 1; no (ε,δ) holds. | Fresh secret CSPRNG entropy in production; deterministic seeding confined to test-only code. |
| 2 | B-F4 | **Empty-node early exit is an un-noised data-dependent branch.** `rows.shape[0] == 0` (`boost.py:434`) skips the histogram query based on raw data; Monte-Carlo shows a 0-vs-0.38–0.69 distinguishing event. Breaks the post-processing argument (and the Laplace pure-ε guarantee outright). | One line, zero budget: always issue the noised query at `depth < max_depth`; add regression test that an empty node still calls `add_noise` exactly once. |
| 3 | A-2 | **Train/validation split is not adjacency-stable.** Stratified `train_test_split` on the full private dataset (`task.py:40-72`); adding one patient changed membership of six existing patients in reproduction, invalidating the one-row-difference sensitivity argument. | Dataset-independent split rule (e.g. keyed hash of patient ID), or split fixed before the DP input is defined, or separately accounted. |
| 4 | A-3 | **Adjacency unit is admission rows, not patients.** All admission rows retained (`task.py:70-72, 207-208`); a patient with m admissions inflates sensitivity by up to m. Claim 9 ("per-site ε = per-patient") fails for the actual data representation. (B passed Claim 9 only *under the stated one-record assumption*, which A shows is not enforced.) | Aggregate to one contribution per patient, or enforce+account a contribution cap, or relabel as admission-level DP. |
| 5 | A-4 | **Federated transcript leaks un-accounted exact statistics:** exact `num-examples` (`client_app.py:176-179`), exact validation metrics (`client_app.py:209-255`), confusion-matrix cells and data-derived Youden threshold (`metrics.py:88-125`). | Remove, privatize under extra budget, or move explicitly outside the claimed DP release boundary; add a privacy statement for validation patients. |
| 6 | A-5 | **ε sweeps need cross-run composition.** Releasing multiple models over the same patients composes; per-run ε is not the guarantee for the collection. | RDP ledger composing all released runs, or a trusted experimentation boundary releasing one selected model. |

Note the reviews are complementary, not contradictory: A audited the system/pipeline level
(randomness, splits, adjacency unit, transcript, sweeps), B audited the mechanism's control flow
and cross-checked against installed Opacus. Their blocking sets are disjoint except that A-3
concretely falsifies the assumption under which B passed Claim 9.

## Non-blocking findings (Reviewer B)

- **F5** — `base_score` data-independence is assumed, not enforced. XGBoost ≥ 2.0 auto-derives it
  from the label mean; if FL wiring ever forwards that, label prevalence leaks. Record in packet
  §7 and assert explicit configuration in DP mode.
- **F6** — packet's `boost.py` line references are stale (~+6 offset); refresh before circulating.
- **F7** — `test_histogram_sensitivity_bound` pins the single-bin invariant only for G, not H; add
  the H-side assertion.

## Additional safeguards (Reviewer A, fail-closed validation)

Labels finite and exactly {0,1}; per-unit contribution cap enforced; finite features/parameters;
`0<δ<1`, `ε>0`, `σ>0`; bin edges public or separately privatized; client-side round-count
enforcement and local privacy ledger; patient disjointness across sites enforced or composed.

## Test-execution status

- Reviewer A: standalone accounting suite 12 passed / 19 skipped; could not run the full DP/FL
  suite (`flwr`/`opacus` install failed, mirror HTTP 503) — Opacus formulas checked by source
  inspection + independent recomputation instead.
- Reviewer B: both DP suites executed, 48/48 pass, plus live Opacus 1.6.0 cross-check and a
  Monte-Carlo experiment for F4.

## Suggested order of work

1. F4 fix + regression test (one line, no accounting impact).
2. Production RNG sourcing (A-1) — likewise no accounting impact.
3. Per-patient aggregation / contribution cap (A-3) — may change sensitivities, re-derive Claim 1.
4. Adjacency-stable split (A-2).
5. Release-boundary cleanup for counts/metrics (A-4) and cross-run ledger (A-5).
6. Packet hygiene: F5 assumption, F6 line refs, F7 assertion; then re-issue the packet and re-review.
