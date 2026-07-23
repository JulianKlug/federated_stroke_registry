# Spec 1.1.a″ — DP remediation roadmap (response to the 1.1.b gate reviews)

Responds to the independent DP-accountant review gate for roadmap 1.1.b
([architecture/roadmap.md](../../architecture/roadmap.md)). Two independent reviewers examined
[dp_accountant_review_packet.md](../reviews/dp_accountant_review_packet.md):

- **Reviewer A** ([findings](../reviews/dp_accountant_review_findings_A.md)) — verdict **BLOCK**
  (system/pipeline level: randomness, split stability, adjacency unit, transcript, sweeps).
- **Reviewer B** ([findings](../reviews/dp_accountant_review_findings_B.md)) — verdict
  **approve with conditions** (mechanism control flow, cross-checked against Opacus 1.6.0).
- **Joint summary:** [dp_accountant_review_joint_summary.md](../reviews/dp_accountant_review_joint_summary.md).

**What is settled and does NOT reopen:** both reviewers independently confirmed the accounting
math — sensitivity bounds, the `2·D·T` release count, Gaussian/Laplace calibration, RDP
composition + Balle conversion, and the σ table (51.17 / 18.89 / 12.05 / 6.70 for
ε = 1/3/5/10 at k=160, δ=1e-5). `fed_stroke/dp/accounting.py` is considered verified as-is.
Nothing in this roadmap touches it.

**What blocks:** six findings, all in the mechanism/pipeline around the accountant. Until they
are fixed and re-reviewed, the reported ε is not a sound per-patient (ε, δ)-DP claim for real
Geneva data, and 1.1.b stays gated. One additional item (R9) was surfaced during the engineering
review of this spec (2026-07-23) — not a reviewer finding, but required for a *sound* privacy-cost
comparison, so it is folded into the same landing.

**Framing.** Same as the DP prototype: DP is a measurement instrument. The remediation is
organized so the cheap, zero-accounting-impact fixes land first (Phase A), the fixes that may
change sensitivities land second with their re-derivations (Phase B), and the release-boundary
decisions — which are policy, not math — land third (Phase C). Re-review is the exit (Phase D).

---

## Phase A — mechanism correctness (no accounting impact)

### R1 — Always issue the histogram query on empty nodes  *(Reviewer B, F4 — BLOCKING)*

- **Defect:** `boost.py:434` (`_grow_trees.build`) early-exits to a leaf when
  `rows.shape[0] == 0`, so *whether the noised query is issued* depends on the raw partition.
  A zero-record node is a leaf with probability 1; in the neighbouring dataset the query is
  issued and splits with probability 0.38–0.69 at calibrated σ (B's Monte-Carlo). No (ε, δ=1e-5)
  covers that gap; the Laplace pure-ε arm is broken outright.
- **Fix:** split the guard — keep the `depth >= max_depth` leaf exit, drop the
  `rows.shape[0] == 0` arm. `np.bincount` over empty `rows` returns zeros, so the release is
  pure noise; `2·D·T` already charges for it (D·T was an upper bound assuming every level
  queries); parallel composition is unaffected (each record still perturbs only its own
  sibling). Everything downstream then genuinely is post-processing.
- **Consequence accepted:** empty nodes may split into noise-only subtrees — a utility
  non-event at `max_depth ≤ 4`, bounded by `max_depth`.
- **Acceptance:** regression test pinning that an empty node still issues exactly one
  `mechanism.add_noise` call per level (counting-mechanism stub); full DP suite green;
  σ table unchanged (assert round-trip values byte-for-byte).

### R2 — Fresh secret entropy for DP noise in production  *(Reviewer A, finding 1 — BLOCKING)*

- **Defect:** production noise RNG is deterministically derived from public inputs —
  `SeedSequence([params.seed, global_round, _site_hash(site)])` at `client_app.py:126-128`, and
  the standalone default `np.random.default_rng(boost.seed)` at `boost.py:526-527`. A verified
  empirically: neighbouring data + same seed → distinguishable models with probability 1, so
  the DP inequality fails at any ε for δ < 1.
- **Design note:** this was a *deliberate* 1.1.a′ choice ("noise seeded from public
  `(base_seed, round, site)` only", logbook 2026-07-22) made for stateless-round
  reproducibility. The reproducibility goal survives only for **tests**, never for runs whose ε
  is claimed.
- **Fix:**
  1. In DP mode, default the noise RNG to OS entropy (`np.random.default_rng()` /
     `secrets`-seeded generator) at both call sites. No experiment seed, round number, or site
     identifier may enter the noise stream.
  2. Keep deterministic RNG **injection-only**: `dp_local_boost` / `train_dp_gbdt` accept an
     explicit `rng` for tests; the FL client constructs no seeded RNG in DP mode.
  3. Rework `tests/test_dp_fl.py:144-167`
     (`test_noise_independent_across_rounds_and_sites_and_reproducible`). Today it asserts
     reproducibility from the public `SeedSequence([seed, round, _site_hash(site)])` **and**
     cross-round / cross-site independence *derived from that SeedSequence*. After this fix
     `_site_hash` no longer feeds production noise, so those assertions are vestigial and must be
     re-expressed: (a) **reproducibility** is asserted only under an explicitly *injected* `rng`
     (the test constructs two identical generators and expects identical models); (b)
     **independence** across rounds/sites is asserted by injecting *distinct* RNGs, not by
     `SeedSequence` inputs; (c) add the inverse guard — two **production-mode** runs (no injected
     rng, OS entropy) on identical data must produce **different** serialized models. Drop the
     `_site_hash` distinctness assertions, or relocate them to a unit test of `_site_hash` itself
     if that helper survives for non-noise uses.
  4. Fail closed: if a config tries to seed DP noise (e.g. a `dp.noise_seed` key ever appears),
     refuse to run with `dp.enabled = true` outside an explicit `--insecure-dp-test` escape
     hatch that is rejected when `--data-provenance real-frozen-schema`.
- **Acceptance:** inverse-determinism test green; grep-level audit that no
  `SeedSequence`/`default_rng(seed)` feeds `mechanism.add_noise` in production paths; Opacus
  precedent (secure mode prohibits user seeds) cited in the packet.
- **Audit allowlist (verified 2026-07-22):** `mechanism.add_noise` (`boost.py:443`) receives
  *only* the `rng` threaded from the two sites named above (`client_app.py:126-127` →
  `dp_local_boost` → `build`; `boost.py:527` → `build`). Two other seeded generators exist and
  are **known-cleared** — they must **not** be changed by this fix: `metrics.py:49`
  (`_bootstrap_ci`, validation-metric confidence intervals — inside the R5 trust boundary, never
  a noise seed) and `dp/synthetic.py:31` (synthetic-data generator for tests/demos, not a
  production path). The R2 grep audit will surface both; treat them as expected false positives,
  not violations.

---

## Phase B — adjacency unit and split stability (may change sensitivities)

### R3 — One contribution per patient  *(Reviewer A, finding 3 — BLOCKING; falsifies packet Claim 9)*

- **Defect:** `task.py` splits by `patient_id` but trains on **all admission rows**
  (`task.py:70-72`, `207-208`). A patient with m admissions contributes m histogram rows, so
  worst-case sensitivity is `ΔG = m`, `ΔH = 0.25·m` — the accountant's `ΔG = 1` is per
  admission row, not per patient. Reviewer B passed Claim 9 only under the one-record
  assumption A showed is unenforced.
- **Decision (user, 2026-07-22):** deduplicate to **one admission row per patient**, rule =
  **max-outcome + tie-break**: keep an admission whose label equals the patient's max outcome
  (matches the existing stratification reduction at `task.py:48-53`); among ties, keep the
  lexicographically smallest `case_admission_id`. Scope = **whole pipeline**: dedup applied at
  the top of `generate_splits` (`task.py:18`), before any split, for the DP **and** non-DP arms
  alike — otherwise the 1.1.b DP-vs-no-DP comparison would confound noise cost with a cohort
  change. Validation therefore also becomes one-event-per-patient.
- **Insertion point (decision, 2026-07-22):** dedup lives in `generate_splits`, **not** in
  `_resolve_context_split`. `_resolve_context_split` (`task.py:133`) is the Flower SuperNode
  loader only — it is called exclusively by `load_data_gva` (`task.py:188`) and
  `load_data_arrays` (`task.py:205`). The pooled baseline reaches data via
  `baseline.py:split_half (task.py:90/:103) → resolve_run_split`, and the offline scorer via
  `eval_final_model → score_booster_on_half → split_half → resolve_run_split`, both **bypassing**
  `_resolve_context_split`. `generate_splits` (`task.py:18`) is the single chokepoint every path
  converges on (`_resolve_context_split → resolve_run_split → generate_splits`; `baseline.py →
  resolve_run_split → generate_splits`; `eval_final_model → score_booster_on_half → split_half →
  resolve_run_split → generate_splits`). Deduping there is **idempotent** — `resolve_run_split`
  calls `generate_splits` once in FLAT/HOLD-OUT mode and **twice** in SEARCH mode (the DEV/HELD
  partition at `task.py:118` then the DEV sub-split at `task.py:127`), and deduping
  already-deduped data is a no-op, so the repeated calls are safe. **Exact insertion point:** the
  dedup must run *after* `patient_id` is derived (`task.py:40`, `data['patient_id'] =
  case_admission_id.split('_')[0]`), since both the group key and the tie-break column come from
  `case_admission_id` — "top of `generate_splits`" means "immediately after patient_id
  derivation," not literally line 18. Placing dedup only in `_resolve_context_split` would leave
  the non-DP pooled baseline on all-admission rows, reintroducing the cohort confound this fix
  exists to remove and making the "non-DP baseline re-run on the deduped halves" acceptance
  criterion (below) unmeetable.
- **Empirical grounding (both Geneva halves = full registry subset, 2026-07-22):** 3674 rows →
  3382 patients; dedup drops 292 rows (7.9%). 243 patients (7.2%) are multi-admission with m
  up to **6** — so the un-fixed inflation is up to ΔG = 6, not a corner case. Only 16 patients
  have disagreeing labels across admissions, so the rule choice barely moves label balance
  (all-rows 0.0882 → dedup-max 0.0923). The rule mostly decides which admission's *features*
  represent the patient; the label is nearly rule-invariant.
- **Rejected alternatives:** contribution cap m > 1 (multiplies σ by m for the same ε — a
  utility disaster at n≈1000/node); relabeling to admission-level DP (weakens the guarantee we
  want to claim); first/index admission (cleanest clinical framing but **not computable
  today** — the EDS suffix in `case_admission_id` is not chronological: sorted by suffix, age
  is non-decreasing for only 56% of multi-admission patients; revisit if the frozen schema
  lands an admission-date column); keyed-hash pick (computable but clinically arbitrary).
- **Fix:** dedup applied at the top of `generate_splits` (see insertion-point decision above)
  *before* any split and before the DP mechanism's input; enforced fail-closed by an assertion
  (`n_rows == n_unique_patients`) inside the DP learner, not just the loader (belt-and-suspenders:
  the loader dedup guarantees correctness, the learner assertion catches any future load path
  that forgets to route through `generate_splits`).
- **Accounting impact:** none once enforced — `ΔG = 1, ΔH = 0.25` become true per-patient
  bounds and Claim 1/Claim 9 hold as written. Re-state Claim 9 in the packet as *enforced*, not
  assumed. Cross-site patient disjointness stays an assumption (GVA/Shenzhen populations),
  now stated explicitly in the packet's assumption table.
- **Acceptance:** test that a synthetic multi-admission patient yields exactly one row with
  the max-outcome label and tie-break admission; DP learner assertion test; non-DP baseline
  (1.d-style, = arm A of R9) re-run once on the deduped halves and the delta logged — this
  re-baselines every downstream comparison, since the cohort shrinks 7.9% in rows. Note the
  1.1.b privacy-cost comparison itself is arm B→C, not this A baseline (see R9).

### R4 — Adjacency-stable train/validation split  *(Reviewer A, finding 2 — BLOCKING)*

- **Defect:** the stratified `train_test_split` over all patient IDs (`task.py:54-68`) is not
  adjacency-stable: adding one patient reshuffled six existing patients in A's reproduction, so
  neighbouring raw datasets do not yield training tables differing by one row, and the
  histogram sensitivity argument doesn't attach to the actual pipeline.
- **Fix (chosen):** dataset-independent split rule — assignment by **keyed hash of
  `patient_id`**. The rule replaces the `train_test_split` at `task.py:64-68` inside
  `generate_splits`, i.e. the same chokepoint R3 dedups at.
  - **Key derivation (decision, 2026-07-22):** `assign(pid) = HMAC(key, pid) mod K`; a patient
    is in **validation** iff `assign(pid) < K · test_size` (`K` a fixed large modulus, e.g.
    `2**32`). The HMAC key is **`key = split_key ‖ role ‖ seed`**, where `split_key` is a public
    per-project constant, `role ∈ {"partition", "subsplit"}` domain-separates the two split
    stages, and `seed` is the run's public seed for that stage (`HOLDOUT_PARTITION_SEED` for the
    DEV/HELD `partition`; `resolve_run_split`'s `split_seed` for the `subsplit`).
  - **Why `role` is mandatory, not cosmetic:** unlike today's `train_test_split` (input-set
    dependent, so seed reuse across the two stages is harmless), the hash rule is input-set
    *independent*. Without the `role` tag, a run with `split_seed == HOLDOUT_PARTITION_SEED`
    (=42) would give the DEV/HELD partition and the DEV sub-split the **identical** key → the
    sub-split's validation band collapses (empty at `holdout_frac=0.2`, a fixed zero-variance
    band otherwise), silently breaking HPO CV. The production ε path is FLAT-mode
    (`holdout_frac=0.0`, no partition) so its soundness is unaffected, but domain-separating the
    role removes the footgun for any HPO run at no cost. `assign` also depends on `test_size`
    only via the threshold, never the key, so the FLAT (0.2) and partition (`holdout_frac`)
    thresholds do not alias.
    This key satisfies both DP and HPO constraints simultaneously:
    - **Adjacency stability** (the DP requirement): within a single run the seed is fixed and
      public, so each patient's assignment depends only on their own `patient_id` — adding or
      removing one patient never moves another. No cohort-derived quantity enters the key.
    - **HPO variation** (the existing requirement): folding `split_seed` into the key means
      distinct seeds still yield distinct splits, so `resolve_run_split`'s SEARCH-mode CV
      repeats keep varying (preserves `test_hpo.py:369`), and the DEV/HELD partition keyed by
      `HOLDOUT_PARTITION_SEED` stays patient-disjoint from every search sub-split (preserves
      `test_hpo.py:376/388/397`): HELD = `{assign_HELD(pid) < K·holdout_frac}` from the full
      cohort; the sub-split runs only on the complement DEV, so HELD ∩ search-valid = ∅ holds.
  - **Scope = whole pipeline (decision, 2026-07-22):** the hash rule replaces the stratified
    split for the DP **and** non-DP arms alike — **not** DP-mode-only. Rationale: R3 already
    forces whole-pipeline dedup so the 1.1.b DP-vs-no-DP comparison isn't confounded by a cohort
    change; a DP-mode-only hash split would reintroduce the identical class of confound one layer
    up (DP arm splits by hash, non-DP arm by stratification → different validation membership →
    different metrics). Keeping one split rule across both arms is the only internally consistent
    choice. The stratification loss (below) is therefore borne symmetrically by both arms and
    cancels in the comparison.
  - The existing `test_pids_path` predefined-list mechanism (`task.py:56-62`) remains supported
    for frozen historical splits but is not the DP-mode default (a list derived from the private
    cohort has no rule for the adjacent dataset's extra patient). DP mode still fail-closes if
    the active rule is neither the hash rule nor an explicitly declared frozen public list with a
    documented adjacency statement (see Acceptance).
- **Cost accepted:** loses outcome stratification, now for **both** arms (see whole-pipeline
  scope above). Stratification **cannot** be kept: its balance comes from an exact per-label
  count over the whole cohort, and that cohort-dependence is precisely what destroys the
  per-patient adjacency stability the DP sensitivity bound requires (a label-keyed "stratified
  hash" is adjacency-stable but gives zero variance reduction, so it is not worth the
  complexity). At GVA per-node scale (~1000, per project memory) the balance jitter of a hash
  split is small; the harness's noise-tier banners already contextualize metric noise; and
  because both arms lose it identically, the DP-vs-no-DP delta is unaffected.
- **Rejected / deferred balance mitigations (discussed, not chosen — recorded so they are not
  re-litigated):**
  1. *Exact stratification* — incompatible with adjacency stability (see Cost accepted); a
     label-keyed "stratified hash" is stable but gives zero variance reduction. Rejected.
  2. *Decoupled split* — pick training membership by stable hash, then stratify **only** the
     validation set (validation is inside the R5 boundary, not DP-released, so it need not be
     adjacency-stable). Rejected: train and valid partition the deduped cohort, so a stratified
     valid must either subsample (wastes validation patients) or move patients across the line
     (destabilizes train). Not worth it at n≈1000.
  3. *Class-weighted validation metrics* — leave the split alone and correct the hash split's
     label-balance jitter at metric-computation time (inverse-prevalence weighting on the full
     validation set). A metrics change, not a split change. **Deferred, kept open:** a cheap
     future option if PR-AUC / Brier jitter proves distracting, but not required for 1.1.b since
     both arms share the identical split so the jitter cancels in the B→C / A→B deltas.
- **Accounting impact:** none — the split becomes free preprocessing as Claim 1 assumes.
- **Acceptance:** property test — for random cohorts, adding/removing any single patient
  changes no other patient's assignment; split-fraction tolerance test; DP mode fail-closed if
  `dp.enabled` and the split rule is not the hash rule (or an explicitly declared frozen
  public list with a documented adjacency statement).

---

## Phase C — release boundary and cross-run composition (policy + plumbing)

### R5 — Define and enforce the DP release boundary  *(Reviewer A, finding 4 — BLOCKING)*

- **Defect:** the client releases exact values the accountant does not cover: exact
  `num-examples` (`client_app.py:176-179`), exact validation metrics
  (`client_app.py:209-255`), confusion-matrix cells / counts / data-derived Youden threshold
  (`metrics.py:88-125`). Under add/remove adjacency even the training count is data-dependent.
- **Decision (two-tier boundary, matching the pilot's trust model):**
  1. **Inside the trusted federation boundary** (mTLS channel between the two sites and the
     Geneva-operated aggregator, per architecture §6.1): validation metrics and confusion
     matrices may continue to flow for experiment steering. They are declared **inside** the
     boundary — not part of the DP release — and are never published or exported.
  2. **Crossing the boundary (published / leaves the federation):** only the selected model
     (DP-accounted) and figures derived from it. Any per-site metric that appears in a
     publication or report must either be computed on data the packet declares non-protected,
     or be privatized under an explicitly accounted extra budget (deferred until actually
     needed — likely only final headline metrics in v1.3).
  3. **`num-examples` specifically:** stop sending the exact count in DP mode. Replace with a
     fixed public per-site weight (configured constant, e.g. the site's approximate cohort
     size rounded to hundreds) — bagging aggregation only needs relative weights; exactness
     buys nothing.
  4. **Validation patients:** add an explicit statement to the packet — validation data is
     used for steering inside the boundary; the DP guarantee claimed publicly covers training
     patients' contribution to the released model, and validation patients' exposure is
     confined to the trusted boundary.
- **Acceptance:** DP-mode client sends no exact `num-examples` (test); packet §7 gains the
  boundary definition with an explicit diagram of what crosses; logbook rule recorded: nothing
  inside-boundary appears in any artifact that leaves the project.

### R6 — Cross-run privacy ledger for the ε sweep  *(Reviewer A, finding 5 — BLOCKING for 1.1.b's sweep design)*

- **Defect:** 1.1.b releases models at ε ∈ {1, 3, 5, 10} over the *same* patients. Per-run ε
  is not the guarantee for the collection; composition applies. Reusing deterministic noise
  across runs (fixed by R2) would have made this strictly worse.
- **Fix:** an append-only RDP ledger (`out/dp_ledger.jsonl`, one line per real-data DP run:
  date, config hash, k, σ, mechanism, per-run ε) plus a `ledger_total()` helper that composes
  all recorded runs' RDP curves and converts **once** — reusing `accounting.py`'s existing
  compose/convert functions verbatim (verified code; no new math). The harness appends
  automatically whenever `dp.enabled` and `--data-provenance real-frozen-schema`.
- **Reporting rule:** every artifact reporting a per-run ε also states the composed
  ledger-total ε to date. The trusted-experimentation alternative (release only one selected
  model) is kept open for the v1.3 headline, where it may give the cleaner story.
- **Acceptance:** ledger unit tests (compose-of-one equals per-run ε; compose-of-N matches
  Opacus on the same N-run schedule); harness integration test; reporting template
  (1.1.e skeleton) carries both numbers.

### R9 — Unconfounded privacy-cost comparator (identity-mechanism arm)  *(eng review, 2026-07-23; not a reviewer finding)*

- **Defect:** the headline claim of 1.1.b is the *cost of privacy*, but the non-DP arm the code
  routes to when `dp.enabled = False` (`client_app.py:182`, stock `xgb.train`) — the same
  "1.d-style baseline" R3's acceptance names — is a **different learner** from the DP arm. It
  uses XGBoost's quantile-sketch binning (vs the DP learner's fixed 32-bin `linspace`) and
  honours the `subsample`/`colsample_bytree = 0.8` values swept by the HPO grid
  (`hpo.py:30-31`), whereas the DP learner forces `q = 1.0` and ignores them
  (`client_app.py:163-165`). So `stock-XGB → noised-DP-learner` conflates *learner cost* with
  *privacy cost*. R3/R4 remove the split and cohort confounds; a stock-XGBoost comparator would
  reintroduce a first-order confound plausibly larger than the ones removed.
- **Decision (user, 2026-07-23): full A→B→C decomposition.** Three arms, all trained and
  evaluated on the **same deduped halves (R3) and the same hash split (R4)**, same
  hyperparameters:
  - **A** = stock XGBoost (`xgb.train`, today's `dp.enabled = False` path).
  - **B** = the DP learner with the **identity mechanism** (noise off) — same fixed binning,
    same `q = 1.0`, same tree code as C, differing only in that no noise is added.
  - **C** = the DP learner with noise on, at each ε.
  - **Headline "cost of privacy" = B→C** (B and C share everything except the mechanism, so the
    delta is noise and nothing else). **A→B is reported as "learner cost"** for context (does the
    from-scratch DP learner track XGBoost at all). This matches the decomposition
    `dp_prototype_demo.py:8,136-137` already established; 1.1.b promotes it from demo to the
    real-data headline.
- **Execution (decision, 2026-07-23):** all three arms run in the **federated** setup — the real
  2-node Flower topology, matching the 1.1.b deployment — but on **Geneva data only**: the two
  nodes are the two patient-disjoint Geneva halves, **not** Geneva + Shenzhen. The whole
  DP-vs-no-DP determination (which ε is acceptable, whether DP is viable at all) is made on this
  Geneva-only federation **before Shenzhen is integrated into the network**. Rationale: Shenzhen
  has no local data access for debugging (project constraint), so the methodology and the go/no-go
  must be settled where both nodes are inspectable. Bearing on the DP claim: with both nodes
  Geneva-owned we *hold* both datasets, so cross-node patient disjointness (R7's stated
  assumption) is directly **verifiable** here — unlike the eventual GVA/Shenzhen boundary, where
  no-data-sharing makes it uncheckable. That assumption only becomes load-bearing when Shenzhen
  joins, which is explicitly downstream of this decision.
- **Fix:** make arm B reachable in the federated 1.1.b run path (the DP-learner client route),
  not only in the demo/tests. Route the DP learner (`_dp_train_round` / `train_dp_gbdt`)
  with the identity mechanism via an explicit comparator selector (e.g. a `dp.mechanism =
  "identity"` config or a `--comparator-arm {A,B,C}` flag) — reusing the existing
  `DPConfig(enabled=False)`-through-`train_dp_gbdt` no-noise path the demo already exercises. All
  three arms must share the R3 dedup + R4 hash split and matched hyperparameters; the DP learner's
  forced `q = 1.0` means arm A must also run at `subsample = colsample = 1.0` so A→B isolates
  learner+binning and nothing else.
- **Accounting impact:** none. Arm B releases nothing under a DP claim (identity mechanism, no ε
  spent); it is a utility reference computed inside the trust boundary (R5). Only arm C's models
  are DP-accounted and enter the R6 ledger.
- **Acceptance:** arm B produced by the same code path as arm C differing only in the mechanism
  (unit test: arm B == arm C when C's σ is forced to 0 / noise disabled); arm B ≠ arm A on
  identical data (different learner); 1.1.b reporting template (R6 / 1.1.e skeleton) carries all
  three arms with B→C flagged as the privacy-cost headline and A→B as learner cost; every arm
  reported on the identical deduped + hash-split cohort.

---

## Phase D — hardening, packet refresh, re-review (exit gate)

### R7 — Fail-closed validation + `base_score` enforcement  *(Reviewer A safeguards; Reviewer B F5)*

DP mode refuses to run unless all hold (single `validate_dp_preconditions()` gate, unit-tested
per condition):

- labels finite and exactly in {0, 1};
- one row per patient (R3 assertion);
- features and mechanism parameters finite; `0 < δ < 1`, `ε > 0`, `σ > 0`;
- bin edges fixed/public (already the 1.1.a′ contract — now asserted, not assumed);
- `base_score` explicitly configured, never defaulted from upstream — XGBoost ≥ 2.0
  auto-derives it from label mean, which would leak prevalence outside the accounting
  (`boost.py:104-117` ingests whatever arrives);
- client enforces the authorized round count for the run's calibrated σ (refuse rounds beyond
  `T_site` used at calibration) and appends to the local ledger (R6).

Cross-site patient disjointness is recorded as a stated assumption (packet §7) with a v1.3
follow-up: a privacy-preserving overlap check or an explicit composed-budget fallback if
overlap cannot be excluded.

### R8 — Packet hygiene + re-issue + re-review  *(Reviewer B F6/F7; both reviewers' exit condition)*

- Refresh the packet's stale `boost.py` line references (~+6 offset; `accounting.py` and test
  refs are accurate).
- Add the H-side single-bin assertion to `test_histogram_sensitivity_bound`
  (`tests/dp/test_boost.py:58-90`). Today it asserts `changed.size == 1` for **G** but for **H**
  only checks the summed delta `|ΔH| ≤ 0.25`; add the symmetric `changed_H.size == 1` (exactly
  one H bin per feature moves), pinning the per-bin H delta, not just the summed delta.
- Update packet §2/§7: Claim 8's premise now holds via R1; Claim 9 re-stated as enforced via
  R3; new assumptions (bin edges, `base_score`, cross-site disjointness, release boundary,
  ledger) tabulated.
- Re-issue the packet and obtain sign-off from **both** reviewers (A's verdict was BLOCK, so
  A's re-review is required, not optional). Record reviewer, date, and scope in
  `docs/logbook.md`, per the 1.1.b gate text.

---

## Ordering, dependencies, and effort

| Item | Blocks | Depends on | Size | Accounting impact |
|---|---|---|---|---|
| R1 empty-node query | 1.1.b | — | XS (guard split + 1 test) | none (count already conservative) |
| R2 fresh entropy | 1.1.b | — | S (2 call sites + test rework) | none |
| R3 per-patient dedup | 1.1.b | — | S (loader + assertion + tests) | none once enforced; packet Claim 9 re-stated |
| R4 hash split | 1.1.b | R3 (shares the `generate_splits` chokepoint) | S | none (split becomes free preprocessing) |
| R5 release boundary | 1.1.b | decision recorded | S code / M policy write-up | none (boundary declaration) |
| R6 run ledger | 1.1.b sweep | R2 (no shared noise streams) | S (reuses verified accounting fns) | adds *composed* ε reporting |
| R7 fail-closed gate | 1.1.b | R3, R4 | S | none |
| R9 identity-mechanism arm | 1.1.b (privacy-cost claim) | R3, R4 (shared cohort/split); relates to R6 reporting | S (reuses no-noise DP learner + comparator selector) | none (arm B spends no ε) |
| R8 packet + re-review | 1.1.b (exit) | R1–R7 + R9 landed | M (external reviewers' latency) | — |

Suggested landing order: **R1 → R2** (independent, zero-impact, unblock everything else's
re-testing) → **R3 → R4** (one loader change-set) → **R5 + R6 + R9** (boundary + ledger +
comparator arm, one change-set — R9 rides on the same R3/R4 cohort/split surface and feeds R6
reporting) → **R7** (gate over all of it) → **R8** (re-issue, re-review, sign-off).

**Effort caveat:** the per-item sizes above are each fix *in isolation*. R3, R4, and R9 all land
on the same surface (`generate_splits` / `resolve_run_split` / the DP-learner routing in
`client_app.py`). Taken together — dedup + the hash-split rewrite that drops `stratify` from
`generate_splits` + the identity-mechanism comparator wiring + rebaselining the split tests R3/R4
intentionally invalidate — that shared change-set is the **single largest block** of the
remediation. Plan it as one **M**-sized change-set, not three independent **S** items; the "S"
labels undersell the combined loader/split/routing rework.

## Acceptance for 1.1.a″ (reopens 1.1.b)

1. All R1–R7 + R9 landed; full test suite green (was 198/198 pre-remediation; grows with the new
   regression/property tests); σ table reproduced unchanged.
   - **Tests intentionally invalidated by R3/R4 (must be rebaselined, not treated as
     regressions):** the whole-pipeline dedup (R3) and the hash split replacing stratified
     `train_test_split` (R4) deliberately change split membership and cohort size, so these
     existing green assertions *will* break and are expected to be updated in the same
     change-set: `test_hpo.py:359` (`test_resolve_run_split_byte_identical_to_legacy_seed42` —
     the seed-42 byte-identity no longer holds once the split rule changes), the non-DP
     "byte-identical to pre-change main" path exercised in `test_dp_fl` (§7.7 case 9), and any
     `generate_splits`-based fixture in `test_eval_final_model.py` / `test_baseline.py` that
     pins pre-dedup row counts. Each must be re-recorded against the deduped + hash-split
     baseline; the diff in counts/metrics is logged per R3's acceptance. Distinguish these from
     R4's HPO-preserving tests (`test_hpo.py:369/376/388/397`), which must **stay green** because
     the seed-folded key keeps distinct-seed-distinct-split and DEV/HELD disjointness intact.
2. Loopback DP demo (1.1.a′ V4/V5) re-run green with production entropy — two runs on
   identical data produce different models, same reported ε.
3. Packet re-issued (R8); **both reviewers signed off**; sign-off (reviewer, date, scope)
   recorded in `docs/logbook.md`.
4. Only then does 1.1.b run on real frozen-schema Geneva data — in the **federated 2-node
   topology but Geneva-only** (both nodes are Geneva halves; Shenzhen is not in the network yet),
   producing the A→B→C decomposition (R9).
5. The **DP-vs-no-DP go/no-go decision** (is the B→C privacy cost acceptable at a usable ε?) is
   made on that Geneva-only federation and recorded in `docs/logbook.md` **before Shenzhen is
   integrated into the network**. Shenzhen integration is downstream of, and gated on, this
   decision — at which point R7's cross-site disjointness assumption becomes load-bearing and its
   v1.3 follow-up (overlap check or composed-budget fallback) applies.

## Explicitly out of scope

- Any change to `accounting.py` — verified by both reviewers; frozen.
- Privatizing validation metrics with extra budget — deferred until a metric must cross the
  release boundary (earliest v1.3 headline); R5's boundary declaration covers 1.1.b.
- Cross-site overlap *enforcement* — stated assumption + v1.3 follow-up (R7).
- Subsampling amplification (q < 1) — the learner remains q = 1 by design (1.1.a′).
