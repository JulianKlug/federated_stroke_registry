# Independent DP review (reviewer A) — re-review of the 1.1.a″ packet

**Subject:** `dp_accountant_review_packet.md`, revision 2026-07-23, at commit `6eb1b25`.
**Prior verdict:** BLOCK (`dp_accountant_review_findings_A.md`).
**Verdict now:** **Approve with conditions.** Conditions C1–C3 must hold before the first
DP run on real Geneva patients; C4–C6 before the packet circulates outside the project.

Scope reviewed: Claims 1–9, §5 gate coverage, §7 assumption table and release boundary, and
the remediation code (`client_app.py` DP branch, `boost.py`, `task.py`, `preconditions.py`,
`ledger.py`, `server_app.py`, `pyproject.toml`). `accounting.py` is byte-unchanged since the
first review (verified by diff); my first-round verification of Claims 5 and 7 stands.

## Test execution

Unlike round 1, the full DP/FL suites ran locally: `tests/dp`, `tests/test_dp_fl.py`,
`tests/test_client_boost.py` — **101 passed**, including the Opacus equivalence gate.

## Disposition of my five blocking findings

### A-1 — deterministic noise from public inputs → **closed**

- `client_app._dp_train_round` leaves `rng=None` → `np.random.default_rng()` (OS entropy).
  `train_dp_gbdt` defaults to entropy whenever `dp.enabled`. `_site_hash` is gone.
- `DPConfig.from_run_config` refuses `dp.noise-seed` with `dp.enabled`; the `insecure-test`
  hatch is refused on `data-provenance = real-frozen-schema`.
- `test_no_public_seed_feeds_production_noise` is a source-level tripwire;
  `test_noise_reproducible_only_under_injected_rng_and_fresh_in_production` pins behaviour.

Residual: the hatch's guard keys off a *run_config* string — see C1.

### A-2 — adjacency-unstable split → **closed**

`generate_splits` assigns by `HMAC(SPLIT_KEY|role|seed, patient_id) mod 2^32 < 2^32·test_size`.
Assignment depends only on the patient's own id; role tags domain-separate the DEV/HELD and
train/valid stages; DP mode refuses predefined pid lists.

Independent reproduction (500 synthetic patients, 1–3 admissions each, 5 % missing NIHSS;
add one patient; flat and hold-out modes): symmetric difference of training rows = `{NEW_0}`,
of validation rows = `∅`. Exactly one row moves. The Claim 1 sensitivity argument now attaches
to the actual pipeline.

Note the split is unstratified by design; that is a utility cost, not a privacy issue.

### A-3 — admission-row adjacency → **closed** (Geneva-only phase)

`dedup_one_row_per_patient` runs before any split, for all arms; the loader re-asserts
uniqueness per partition; the R7 gate re-asserts it on the training rows each round. The dedup
rule (max outcome, then lexicographic `case_admission_id`) is a function of the patient's own
rows only, so it is adjacency-stable. Reproduction above: every partition one row per patient.

Cross-site disjointness stays a stated assumption (§7 row 11). Acceptable while both nodes are
Geneva halves cut by the same keyed hash; it becomes load-bearing at Shenzhen integration.

### A-4 — un-accounted exact releases → **closed by boundary declaration**, with C3

The DP train reply no longer carries the exact count (`dp-site-weight`, a fixed node_config
constant). Validation metrics, confusion matrices, Youden thresholds and validation counts are
now declared *inside* the trusted federation boundary (§7.2) rather than privatized. That is a
legitimate resolution **provided the boundary is real**: the only object crossing it is the
selected model plus post-processing of it. Two observations:

- `server_app` writes `out/metrics/*.json|md` with per-site exact statistics. These files are
  inside-boundary artifacts by the §7.2 rule; nothing enforces it. See C3.
- Once Shenzhen joins, its validation statistics flow to a Geneva-operated aggregator. That is
  an institution-to-institution disclosure, outside the DP claim, and needs Shenzhen's explicit
  agreement. Record it in §7.2 now so the trust model is not silently widened later.

### A-5 — cross-run composition → **closed in design, incomplete in mechanism** (C1, C2)

`ledger.py` composes per-site RDP curves via the frozen `compose_rdp`/`rdp_to_epsilon` and
adds Laplace ε linearly — correct and conservative. Intent-to-spend on first participation is
the right direction. Two gaps remain in *when* and *where* an entry lands:

1. **Cyclic under-ledgers.** `_maybe_append_ledger` fires only at `global_round == 1`. Under
   `OrderedFedXgbCyclic` one site trains per round, so the second site's first call is round 2
   and it is never ledgered while still spending its full `2·D·T_site`. The sweep driver
   documents this as a risk and asks the operator to prefer bagging; the client does not refuse.
2. **Gating on submitter-controlled config.** Whether a run is ledgered (`data-provenance`),
   *where* it is ledgered (`dp.ledger-path`), and whether the seeded-noise hatch is legal, all
   derive from `run_config`, which the `flwr run` submitter sets — not the node that owns the
   data. The default is `example-halves`. A real-data run launched without the override spends
   ε with no ledger entry and permits `insecure-test` seeding. In the Geneva-only phase
   submitter = node operator, so this is a misconfiguration risk rather than an adversary; at
   Shenzhen it becomes a trust hole (Shenzhen's node has no say over its own ledger).

## Conditions

**Before the first real-Geneva DP run:**

- **C1 — node-owned provenance and ledger.** Move `data-provenance` and `dp.ledger-path` to
  `node_config` (beside `data-path`, `dp-site-weight`); the DP branch fails closed if the node
  declares neither. The submitter must not be able to switch a node's data to "example" or
  redirect its ledger. `DPConfig.from_run_config`'s hatch check must read the node-side value.
- **C2 — cyclic.** Either ledger on the site's first *participation* (not `global_round == 1`),
  or make the client refuse `train-method = cyclic` with `dp.enabled` on real data. A CLI note is
  not an enforcement.
- **C3 — boundary enforcement for metrics artifacts.** `out/metrics/` and `out/dp_ledger.jsonl`
  are inside-boundary. Add them to the reporting template's "never leaves the project" list and
  to the release checklist; the 1.1.b report must cite only aggregate figures derived from the
  released model or explicitly declared non-protected data. Record the Shenzhen disclosure point
  above in §7.2.

**Before the packet circulates:**

- **C4 — stale docstring.** `boost.py:dp_local_boost` still says the rng is "seeded from public
  (base_seed, global_round, site)". A future maintainer reading it will reinstate A-1. Rewrite
  to match `_dp_train_round`.
- **C5 — ledger integrity.** Append-only is by convention (`open("a")`). Have the 1.1.b report
  record the ledger file's sha256 and entry count at sweep start and end so a truncated ledger
  is detectable. A tamper-evident chain is not required for the pilot.
- **C6 — `dp-site-weight` provenance.** The packet calls it "approximate cohort size rounded to
  hundreds". State explicitly that it is fixed *before* any DP run and never recomputed from the
  data thereafter; a weight re-derived per run is a coarse count release.

## Claim-by-claim (changes from round 1 only)

| Claim | Round 1 | Now | Basis |
|---|---|---|---|
| 1 sensitivity | conditional | **pass** | binary labels and one row/patient enforced (R7, R3); reproduction |
| 4 calibration | pass | pass | entropy source now sound (R2) |
| 6 q = 1 | conditional | **pass** | split stability reproduced (R4) |
| 8 post-processing | pass | pass | R1 premise: query count is data-independent (`test_empty_node_still_issues_histogram_query`) |
| 9 per-site = per-patient | fail | **pass, Geneva-only** | R3/R4; cross-site disjointness remains §7 row 11 |
| others | pass | pass | `accounting.py` unchanged |

Missingness sentinel (§7 row 14): a record still occupies exactly one bin per feature; no
accounting impact. Confirmed `fixed_bin_edges` reserves bin 0 and asserts sentinel < every lower
bound.

## Safeguards requested in round 1

All implemented in `preconditions.py` except client-side *persistent* round enforcement: the
gate refuses `global_round > num_rounds`, but `num_rounds` is derived from the same run_config
the submitter controls. Under per-run calibration this is self-consistent (a larger budget
recalibrates σ), so it is not a leak; the cross-run guard is the ledger — hence C1.

## Gate decision

> **Approve with conditions.** The accountant mathematics was already sound; the pipeline now
> realises its preconditions: fresh entropy, one row per patient, an adjacency-stable split,
> a data-independent query count, no exact count on the wire. Sign-off for the Geneva-only
> 1.1.b sweep is granted once C1–C3 are in place. Shenzhen integration re-opens §7 row 11 and
> the A-4 disclosure point and needs a short addendum review, not a full one.
