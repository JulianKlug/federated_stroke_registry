# DP accountant re-review (reviewer B, round 2) — findings

- **Subject:** [`dp_accountant_review_packet.md`](dp_accountant_review_packet.md), revision
  2026-07-23 (post 1.1.a″ remediation, commit `7c43b16`).
- **Prior report:** [`dp_accountant_review_findings_B.md`](dp_accountant_review_findings_B.md)
  (approve with conditions; F4 blocking).
- **Date:** 2026-08-29
- **Method:** re-read of packet §2/§7 against current code; verification of each of my
  conditions (F4–F7) and of the six joint blockers as wired; full read of the depth guard,
  `_best_split`, `_grow_trees`, `train_dp_gbdt`, `preconditions.py`, `ledger.py`, the DP branch
  of `client_app.py`, `task.py` dedup/keyed split, sentinel binning; suites executed
  (`tests/dp`: 75/75 pass).
- **Verdict:** **Approve with conditions** (all non-blocking for the math; one new
  process-level condition, F8). The accountant is unchanged and stays verified. The mechanism's
  control flow is now data-independent (F4 closed). The pipeline blockers are remediated as
  described; their safety rails, however, hang on a self-declared config string (F8).

---

## 1. Status of prior conditions

| # | Prior finding | Status | Evidence |
|---|---|---|---|
| F4 | Un-noised empty-node early exit | **Closed** | `boost.py:498` guard is `depth >= max_depth` only; `build` has no other raw-data branch (`_best_split` reads noised `Gh/Hh` only; `best[0] <= 0.0` is on noised gain). Regression `test_empty_node_still_issues_histogram_query` pins 2^D−1 queries on an all-empty frontier, 4 of them with raw mass 0. σ table pinned byte-identical. |
| F5 | `base_score` data-independence assumed | **Closed** | `preconditions.py` refuses a missing `params.base_score` and requires finite `(0,1)`; `client_app.py:230` passes the raw run-config `params` (not a defaulted `BoostParams`), so the check is not vacuous. `test_gate_rejects_missing_or_invalid_base_score`. |
| F6 | Stale `boost.py` line refs | **Recurs** — see F6′ | |
| F7 | H-side single-bin invariant unpinned | **Closed** | `test_boost.py:84-90`: one H bin per feature, same bin as G, per-bin ΔH ≤ 0.25. Mislabelled "F6" in test comment and packet Claim 1 — see F6′. |

## 2. Joint blockers as wired (spot-check, not a re-audit of reviewer A's scope)

| Joint # | Remediation | Observation |
|---|---|---|
| 1 (A-1) | R2 OS entropy | `client_app.py:142` `default_rng()`; `boost.py:601` `default_rng(None if dp.enabled …)`. `_site_hash` gone. Seeded path only via `dp.noise-seed` + `dp.insecure-test`, refused when `data-provenance == "real-frozen-schema"`. **But** the refusal keys on a config string that defaults to `"example-halves"` (F8). |
| 2 (B-F4) | R1 | Closed, above. |
| 3 (A-2) | R4 keyed hash | `_hash_split_assign`: HMAC(`SPLIT_KEY‖role‖seed`, pid). Per-pid, input-set independent — adjacency-stable. DP mode refuses predefined pid lists. Correct. |
| 4 (A-3) | R3 dedup | `dedup_one_row_per_patient`: max-outcome + lexicographic tie-break, per patient only. Removing one patient removes exactly one training row. Belt at `_resolve_context_split`, braces at R7 gate. Correct. |
| 5 (A-4) | R5 boundary | DP train reply sends `dp-site-weight` (node_config), never the count; `num_train` not unpacked in the branch. Evaluate reply still carries exact validation `n` and metrics — declared inside-boundary by §7.2. Acceptable given the trust model; the packet is explicit that nothing inside-boundary may be published. |
| 6 (A-5) | R6 ledger | RDP composition across runs, single conversion at δ; Laplace ε added linearly. Math correct (no new math; reuses frozen fns). Intent-to-spend on round 1 is the right direction. **Gated on the same self-declared string** (F8). Server-side total only readable in loopback — documented. |

## 3. New findings

### F8 — CONDITION: R2 and R6 rails are honour-system on `data-provenance`

`_maybe_append_ledger` (`client_app.py:162`) returns without writing when
`run_config["data-provenance"] != "real-frozen-schema"`; `DPConfig.from_run_config`
(`boost.py:103`) permits seeded noise under `insecure-test` under the same test. The key
**defaults to `"example-halves"`** (`pyproject.toml:65`, and the `.get` default in
`from_run_config`). Nothing ties the string to what is actually loaded from
`node_config["data-path"]`.

Consequence: a real-Geneva run launched with the default config spends ε with **no ledger
entry**, and — if a stale `insecure-test`/`noise-seed` override is still in the overrides —
with **deterministic noise**. Neither is caught by any test or gate. The R6 reporting rule
("every per-run ε is reported with the composed total") is then silently wrong, in the
under-report direction.

**Fix direction (either suffices for sign-off):**
1. Invert the default: append to the ledger **unless** provenance is explicitly declared
   synthetic; refuse `insecure-test` unless provenance is explicitly declared synthetic.
   Missing key ⇒ treated as real.
2. Better: make provenance a **node-side** fact (`node_config`, next to `data-path`, or a
   marker column/metadata in the frozen-schema parquet) and have the R7 gate refuse a run
   whose run-config provenance disagrees with the node's.

Add a test: default run-config + gaussian mechanism ⇒ ledger entry written.

### F9 — assumption 11 has no in-repo check

Packet §7.1 #11 calls cross-site patient disjointness "directly verifiable in the Geneva-only
2-node phase". The half files are produced outside this repo (no generator under `scripts/`),
and no code path intersects `patient_id` across node files. Per-site ε = per-patient ε
(Claim 9) fails silently if a patient appears in both halves — the R3 dedup is per file and
cannot see it.

**Fix.** One-off check recorded in the logbook before the 1.1.b determination (intersection of
`patient_id` across the two node parquets must be empty), or a `scripts/` helper the sweep
driver runs first. Not a code change to the mechanism.

### F6′ — doc: `boost.py` references stale again; F6/F7 mislabel

The packet was refreshed against an intermediate `boost.py`. Current positions: sensitivity
constants `50-53` (packet 44-47), `DENOM_FLOOR` `54` (48), `num_histogram_queries` `158`
(150-153), `num_gaussian_releases` `163` (155-158), factory `n_rel`/`std_g`/`std_h`
`280/285-286` (258, 262-264), Gaussian `add_noise` `239` (217), Laplace branch `290-298`
(268-276), `_leaf_weight` `315` (293), `_best_split` `322` (300), depth guard `498`.
`accounting.py` and every test reference are accurate.

Also: packet Claim 1 and `test_boost.py:84` credit the H-side assertion to "reviewer B F6";
it was **F7** (F6 was the line-reference finding). The packet's "What is NOT in scope: the FL
wiring (not yet built)" bullet contradicts the revision note that puts the wiring in scope —
delete it.

### F10 — nit: sentinel bin absorbs any sub-range value

`fixed_bin_edges` reserves bin 0 for `MISSING_SENTINEL`; `_binize` sends every value `< lo`
there, so a non-sentinel out-of-range value (e.g. a negative age) is indistinguishable from
missing. No privacy impact (still one bin per feature); worth a gate assertion
`X[X != SENTINEL] >= lo` or a note.

---

## 4. Conditions for sign-off

1. **F8** — provenance rail made fail-closed (default ⇒ real) or node-bound, with a test.
2. **F9** — cross-site `patient_id` disjointness checked and logged before the 1.1.b
   determination.
3. **F6′** — refresh `boost.py` references, fix the F6→F7 attribution, drop the stale
   scope bullet (non-blocking).

`accounting.py` remains verified as-is. The mechanism (`boost.py`) is now sound under the
packet's model: every histogram query is issued as a function of public configuration and
prior noised releases only. With F8 fixed, I sign off on 1.1.b running on real Geneva data.
