# TODOS

## ✅ DONE (1.d, 2026-07-18) — unified the per-site scorer

1.d extracted the per-site validation split + scoring seam into
`fed_stroke.baseline.score_booster_on_half`. `scripts/eval_final_model.py`'s
private `_site_metrics` was replaced by a call to it, so the offline eval table
and the new federated-vs-pooled gate (`scripts/check_fed_vs_pooled.py`) now score
through one implementation and cannot drift. (Distinct from the 1.c reuse item
below, which was already DONE.)

## ✅ DONE (1.c, 2026-07-17) — 1.c should reuse 1.b's split + site-mapping helpers (do not re-derive)

Resolved by the 1.c evaluation harness: `fed_stroke/metrics.py` and both the
client (`client_app.evaluate`) and offline scorer (`scripts/eval_final_model.py`)
share one split via `task.generate_splits` (`test_size=0.2`, `seed=42`) and the
`schema.py` column contract; the offline scorer was extended, not rewritten; the
server reads the site label through the existing `@app.query` site channel.

<details><summary>original item</summary>

**What:** When implementing roadmap 1.c (site-stratified AUC-ROC / AUC-PR /
Brier / confusion-matrix harness), reuse the seams 1.b introduces rather than
rebuilding them:
- `schema.py` frozen column contract (`FEATURE_COLS`, `TARGET_COL`) and
  `task.py`'s `generate_splits` (with `test_size=0.2`, `seed=42`).
- `scripts/eval_final_model.py` per-site scoring of a saved `final_model.json`.
- `OrderedFedXgbCyclic`'s node→site query (`@app.query` handler returning the
  data-file name) if 1.c needs a server-side site label.

**Why:** 1.b and 1.c must share one definition of "the validation split." Two
copies will drift, produce divergent numbers, and defeat 1.d's federated-vs-
pooled correctness check (±3 AUC points) — the exact silent-partitioning class
of bug 1.d exists to catch.

**Pros:** One source of truth for the split; faster 1.c; protects 1.d.
**Cons:** Slight coupling of 1.c's harness to 1.b's helper shapes (acceptable —
the split definition *should* be shared).

**Context:** 1.b (docs/specs/1b_fedxgb_bagging_cyclic.md) deliberately keeps the
eval script "debugging-grade" and out of 1.c's scope, but its split logic and
site mapping are the reusable core. Start 1.c by importing from `task.py` and
extending `eval_final_model.py`, not by writing a fresh split.

**Depends on / blocked by:** 1.b implemented and merged.

</details>
