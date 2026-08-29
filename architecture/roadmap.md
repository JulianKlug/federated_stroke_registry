# Roadmap

Two tracks: **federated architecture** (sequential) and **preprocessing**
(parallel). Architecture is a linear engineering flow — infrastructure on the
Geneva example subset, then DP + HPO on real Geneva data, then Shenzhen
onboarding + headline. Preprocessing runs alongside and produces the frozen
aligned schema per site: Geneva first (unblocking DP + HPO development),
Shenzhen later (unblocking the headline). Both aggregation strategies
(bagging + cyclic) are implemented in parallel from the start — the winner
is chosen only on real cross-site data in Phase v1.3, not on any Geneva-only
proxy.

Reconciled with [../docs/automated_review/architecture_federated_xgboost.md](../docs/automated_review/architecture_federated_xgboost.md).

## Constraints

- No data leaves a supernode.
- Each supernode should be launchable by a non-engineer (doctor-operator).
- Research pilot — avoid over-engineering. Ship the smallest version that
  answers the research question.

---

## Federated architecture track (sequential)

### Warmup ✅

- [x] Flower + XGBoost quickstart on HIGGS (`working_example/`). Framework
  smoke test.

### Phase v1 — Distributed FL infrastructure on Geneva

Engineering work on the Geneva example subset. Data is split once into two
independent files (patient-ID-stratified 50/50, fixed seed) and each SuperNode
reads its own file from a configured path — mirroring the production shape
where Geneva and Shenzhen each read local-only data. Runtime partitioning
inside the load function is deliberately avoided: it would be throwaway code
that does not match the final workflow. Both aggregation strategies are
implemented and kept in parallel — no winner is picked here.

- [x] 1.a Bootstrap the **functional** distributed topology on Geneva
  (insecure/loopback — the mTLS hardening is split out to 1.e). Ordered
  sub-steps so that each intermediate is independently verifiable:
  1. **Verify the current working example runs on this dev machine.**
     Update the hardcoded path in `task.py:85` to the local Geneva Excel
     (`/mnt/hdd1/datasets/GVA_stroke_registry`, per `CLAUDE.md`); run
     `flwr run .`; confirm an AUC number lands. Establishes the baseline
     before any structural change — any later bug you introduce is then
     distinguishable from preexisting state.
  2. **Data preparation.** One-off script produces `geneva_half_A.parquet`
     and `geneva_half_B.parquet` via patient-ID-stratified 50/50 split
     with fixed seed (`preprocessing/prepare_geneva_halves.py`). Parquet,
     not Excel — the per-round reload cost in the working example is
     otherwise wasteful. Both files are inspectable artifacts before any
     FL code changes.
  3. **Config-driven data path.** `load_data_gva` reads from
     `context.node_config["data-path"]`; no hardcoded path, no
     partitioning inside — the load contract is "read the file at the
     configured path." Runs across a 1-SuperLink / 2-SuperNode local
     topology (`scripts/run_local_federation.sh`), each SuperNode pinned
     to one half; each prints its row count, unique patient count, and
     label balance, and a federated round completes end to end.
- [x] 1.b Implement both `FedXgbBagging` and `FedXgbCyclic` with matched
  hyperparameters (architecture §3) and matched total tree budget.
  Alternate cyclic order across runs to check for last-site bias.
- [x] 1.c Evaluation harness: site-stratified AUC-ROC, AUC-PR, Brier, and
  confusion matrix at the operating point, per architecture §4.
- [x] 1.d Federated-vs-pooled correctness check on Geneva 50/50 partition.
  Federated result must be within 3 AUC points of pooled Geneva xgboost.
  Purpose: catch silent data-partitioning, DMatrix, or tree-serialization
  bugs before any downstream DP result is measured against them.
- [x] 1.e Secure the topology (architecture §6.1). Replace the insecure
  local run with 1 `SuperLink` / 2 `SuperNode`s over **mTLS**, with
  **certificate pinning** and **short-lived credentials** (node
  authentication). Move each SuperNode's `data-path` into `pyproject.toml`
  under `[tool.flwr.federations.<name>]` (it currently lives in a CLI
  `--node-config` arg + `~/.flwr/config.toml`). This is the same secure
  contract Shenzhen will follow (1.3.a/1.3.b), practised locally to
  de-risk onboarding; nothing here is throwaway. Verify: disjoint patient
  sets across the two halves, roughly equal size, similar label balance,
  and a federated round completes end to end over the mTLS channel.
  *(Note on the literal wording, superseded by
  docs/specs/1e_secure_topology.md Decision 1-3: "mTLS" is realized as
  Flower-native server-side TLS + CA pinning + EC node authentication —
  flwr 1.31 has no X.509 client-cert mTLS. And `data-path` in
  `[tool.flwr.federations.<name>]` is infeasible twice over: `data-path`
  is per-SuperNode `node_config` while a federation block is a single
  connection entry, and `flwr run` migrates any committed
  `[tool.flwr.federations]` into `~/.flwr/config.toml` and comments it
  out of the tracked pyproject.toml. The committed config lives in
  `[tool.fed_stroke.superlink]`/`[tool.fed_stroke.nodes]` instead, spec
  §4.5-§4.6.)*
- [x] 1.f Docker smoke build. Dockerfile compiles, container runs the
  pipeline end-to-end on the dev machine. Full Shenzhen-ready packaging
  (version pinning, clean-bootstrap test on a non-dev machine) is
  deferred to Phase v1.3, but building the image once here catches
  Dockerfile bugs long before Shenzhen go-live.

**Acceptance for v1:** both strategies implemented and pass 1.d; evaluation
harness produces the three metrics; topology runs over mTLS (1.e); Dockerfile
builds and runs the pipeline locally (1.f).

### Phase v1.1/v1.2 — DP + HPO development phase (real Geneva data)

Blockers: v1 complete; DP plug-point prototype passes on single-site synthetic
(see prerequisite below); Geneva side of the preprocessing track has produced
the frozen schema on real Geneva data; local sanity baseline v0.a (Geneva)
is published.

Runs on real Geneva data with a default strategy (bagging, since that is what
`working_example/` uses). The DP-utility curve produced here is a
**development artifact**, not the headline — it validates the DP mechanism
and gives a first look at the utility landscape on real Geneva data. The
headline curve on real cross-site data with the winning strategy is produced
in Phase v1.3.

- [x] Prerequisite — DP plug-point prototype on single-site synthetic.
  Custom client-side DP-XGBoost-style Laplace/Gaussian noise on
  gradient/hessian histograms before aggregation, plus geometric clipping
  of leaf outputs to bound sensitivity (architecture §6.3). Flower's
  built-in DP wrappers do not apply — they target weight averaging in
  FedAvg, not tree-structure aggregation. Includes DP accounting
  verification: on a fixed toy setting (known rounds, known noise scale),
  compute total ε with the accountant and compare to Opacus's
  `RDPAccountant` or TensorFlow Privacy's `compute_rdp` on the same
  setting. Numbers must match within floating-point tolerance before any
  downstream sweep uses this accountant.
- [ ] 1.1.a Systematic HPO on bagging: `num_server_rounds`, `max_depth`,
  `eta`, `min_child_weight`, `subsample`, `colsample_bytree` (architecture
  §3 defaults are starting points, not endpoints). Cyclic revalidation
  happens in Phase v1.3 once the strategy winner is picked. The harness
  built here is strategy-agnostic (runs bagging and cyclic) and
  substrate-agnostic (federation-parameterized), so the same code drives the
  Geneva-dev loopback topology now and the cross-site re-run later; see
  docs/specs/1_1_a_hpo_harness.md. The Geneva-only ranges are **provisional**
  — re-tuned on real cross-site data in Phase v1.3 (1.3.b′).
  **Status (2026-07-21): harness built + validated end-to-end (bagging + cyclic)
  on the example halves (184/184 tests; `fed_stroke/hpo.py`, `scripts/run_hpo.py`).
  The provisional Geneva ranges — the item's second deliverable — remain blocked on
  real frozen-schema Geneva data (phase blocker), so this stays open until the sweep
  is re-run on real data with `--data-provenance real-frozen-schema` and recorded in
  the logbook.**
- [x] 1.1.a′ DP → FL integration. Wire the single-site DP seam
  (`fed_stroke/dp/`: `DPConfig`, `HistogramNoiseMechanism`, `train_dp_gbdt`, the
  RDP accountant) into the real federated path: client-side Laplace/Gaussian noise
  on the gradient AND hessian histograms + geometric leaf clipping (architecture
  §6.3), a DP-aware aggregator, and `DPBooster` serialization across rounds, with
  the accountant composing the `2·D·T` releases per run (spec `1_1_prereq` §3.3).
  `dp.enabled = true` forces `subsample = 1.0` for honest q = 1.0 accounting. The
  `dp.*` config keys and the 1.1.a harness already thread these via `--run-config`,
  so this makes those knobs bite and turns 1.1.b into "the 1.1.a sweep + DP arms"
  with no new orchestration. Built and validated on the loopback
  `local-deployment` topology (synthetic/example data) — it does NOT run on real
  patient data (that is gated in 1.1.b), so no real-ε claim is made here. Closes
  the downstream integration item flagged in
  docs/specs/1_1_prereq_dp_plugpoint.md §4.5. Prerequisite for 1.1.b.
- [ ] 1.1.a″ DP remediation (response to the 1.1.b gate reviews). The
  independent review ran on 2026-07-22 with two reviewers: verdicts BLOCK
  (reviewer A, system/pipeline level) and approve-with-conditions (reviewer B,
  mechanism level). Both independently confirmed the accountant math and the
  σ table — `fed_stroke/dp/accounting.py` is verified and frozen. Six blocking
  findings sit in the mechanism/pipeline around it: empty-node un-noised
  branch (B-F4), deterministic public-seeded DP noise (A-1), non-adjacency-
  stable train/val split (A-2), admission-rows-not-patients adjacency unit
  (A-3), un-accounted exact transcript releases (A-4), missing cross-run
  sweep composition (A-5). Fix roadmap with decisions, ordering (R1–R8), and
  acceptance: docs/specs/1_1_a_doubleprime_dp_remediation.md. Joint review summary:
  docs/reviews/dp_accountant_review_joint_summary.md. Exit = packet re-issued
  and **both** reviewers signed off, recorded in docs/logbook.md.
- [ ] 1.1.b Run the sweep both without DP and with DP at each pilot
  ε ∈ {1, 3, 5, 10}, δ = 1e-5 (builds on the wired DP path from 1.1.a′). DP
  changes the utility landscape —
  `max_depth` and `min_child_weight` in particular trade off differently
  under histogram noise.
  Driver + report machinery: docs/specs/1_1_b_dp_sweep_driver.md
  (`scripts/run_dp_sweep.py`, the A→B→C comparator sweep on one shared
  config; rehearsed on the example halves — the real run stays behind the
  gate below and is the single command documented in that spec §6.5).
  - **GATE (blocking) — independent DP/privacy review of the accountant
    before this runs on real patient data.** 1.1.b is the first task that
    reports ε as a claim about real Geneva patients, not a synthetic sanity
    number. Before it runs, a reviewer with DP expertise *independent of the
    accountant's author* must sign off on the accounting: the per-level
    composition, the `2·D·T` Gaussian release count (both the gradient and
    hessian histograms are noised — see spec `1_1_prereq` §3.3), the Laplace
    L1 (`∝ d`, not `√d`) scale, and the Balle RDP→(ε,δ) conversion. The
    Opacus equivalence gate and the prototype's own tests do NOT satisfy this
    — the gate is blind to the release count, and author-written tests share
    the author's blind spots (both errors above were caught only by an
    independent re-derivation at spec stage). Record the sign-off (reviewer,
    date, scope) in `docs/logbook.md`.
    **Status (2026-07-22): review round 1 complete — gate NOT passed.**
    Accountant math confirmed by both reviewers; six blocking pipeline/
    mechanism findings must be remediated (1.1.a″,
    docs/specs/1_1_a_doubleprime_dp_remediation.md) and the packet re-reviewed
    before this item runs on real patient data.
- [ ] 1.1.c Search method matched to compute budget: Optuna / TPE if
  evaluation runs are cheap, coarse grid otherwise. Rényi-DP composition
  accounting across rounds throughout.
- [ ] 1.1.d Geneva-only DP-utility curve at tuned hyperparameters
  (development artifact, not headline).
- [ ] 1.1.e Reporting artifact skeleton: versioned notebook layout + figure
  template ready to receive the v1.3 headline numbers.

**Acceptance for v1.1/v1.2:** DP mechanism validated end-to-end; accountant
matches reference; Geneva-only development curve produced at all four ε
values; notebook skeleton in place.

### Phase v1.3 — Shenzhen onboarding + headline experiment

Blockers: v1.1/v1.2 complete; Shenzhen side of the preprocessing track
complete (partner has produced frozen schema + smoke-test artifact + all
divergence gates cleared); local sanity baseline v0.b (Shenzhen) published.

- [ ] 1.3.a SuperNode packaging shipped to Shenzhen. Docker image +
  one-command bootstrap so a non-engineer at Shenzhen can launch the
  SuperNode. Dockerfile pins exact versions of Python, `flwr`, `xgboost`,
  `numpy`, and OS-level OpenMP runtime — architecture §8 names matched
  versions as a blocking dependency. Clean-bootstrap acceptance test:
  build the image, run it end to end on a machine that is not the dev
  machine (fresh VM or colleague's laptop) before it is shipped.
- [ ] 1.3.b Cross-site smoke test. mTLS handshake works between Geneva and
  Shenzhen SuperNodes. One federated round completes end to end. No
  science yet — just wires.
- [ ] 1.3.b′ Cross-site HPO re-run. The Geneva-only hyperparameter ranges
  from 1.1.a are provisional — Shenzhen's ~40k cohort and different
  distribution move the optimum. Using the **same** 1.1.a harness with
  `--federation` pointed at the real GVA↔Shenzhen deployment, re-tune on real
  cross-site data. Bounded to the narrowed ranges 1.1.a produced (not a wide
  search): the re-run drives real federated rounds over the international link
  with the partner's node in the loop, so the round budget must stay small.
  Produces the cross-site re-tuned hyperparameters 1.3.c consumes.
- [ ] 1.3.c Real cross-site bake-off: run both `FedXgbBagging` and
  `FedXgbCyclic` on real data from both sites at the **cross-site re-tuned**
  hyperparameters (1.3.b′) — not the provisional Geneva-only ranges from
  v1.1/v1.2. Alternate cyclic order across runs.
- [ ] 1.3.d Pick winner by site-stratified AUC (not pooled). Report both.
  Tie-break rule: if bagging wins one site and cyclic wins the other,
  prefer the strategy with lower cross-site AUC variance. If variance is
  comparable (within 0.01), report both curves and let the write-up carry
  both.
- [ ] 1.3.e Headline DP-utility sweep on real cross-site data with the
  winning strategy. Report AUC-ROC / AUC-PR / Brier deltas versus the
  tuned no-DP baseline at each ε. This is the pilot's contribution to
  the DP-utility tradeoff literature for clinical GBDT.

**Acceptance for v1.3:** Shenzhen SuperNode online; cross-site round
completes; winner picked; headline DP-utility curve on real cross-site data
drawn at all four ε values with tuned hyperparameters; notebook + figure
exportable.

### Phase v2 — Conditional (trigger-driven reopen)

Reopen §6.6 of the architecture doc only if one of its five triggers fires:
untrusted aggregator, DPO/IRB flag on aggregation traffic, third site under
weaker DPA, move to vertical FL, or richer-than-histogram per-round artifacts
crossing the border. If any fires, the upgrade path is NVFlare + Paillier HE,
not TenSEAL/CKKS on Flower.

### Phase v2.1 — Third site + SecAgg+

Only when a third clinical partner is confirmed. SecAgg+ is low-value at N=2
and only becomes meaningful with a third participant or a non-trusted
aggregator.

---

## Preprocessing track (parallel)

Runs alongside the architecture track. Geneva side finishes first, unblocking
v1.1/v1.2 development on real Geneva data. Shenzhen side finishes later,
unblocking Phase v1.3.

- [ ] Pick the primary label — one, not both: `mRS ≤ 2 at 3 months` or
  `in-hospital mortality`.
- [ ] Freeze the aligned feature set. Lock
  `registry_alignement/mappings/gva_to_shenzen.py` and `unit_conversions.py`.
  Publish `registry_alignement/FROZEN_FEATURES.md` naming every included
  variable, its Geneva source, its Shenzhen source, and its unit after
  conversion.
- [ ] Geneva preprocessing: EHR → tabular in the frozen schema. Extends
  `registry_alignement/geneva_preprocessing/`. Completion unblocks v0.a
  (local sanity baseline for Geneva) and Phase v1.1/v1.2.
- [ ] Shenzhen preprocessing: same schema, run remotely by partner. Needs
  partner sign-off on the frozen schema before code moves. Completion
  (with the smoke-test artifact and divergence gates below) unblocks
  v0.b and Phase v1.3.
- [ ] Cross-site smoke-test artifact. Preprocessing script emits a summary
  report at each site: row counts, label balance, per-feature
  median / min / max / missing rate, unit-check pass/fail. Only aggregate
  statistics leave the site — no raw records. The Geneva and Shenzhen
  reports are compared side by side before any cross-site training touches
  the data.
- [ ] Divergence go/no-go gates. Written thresholds that block v1.3 from
  starting until cleared. Starting proposals (revise once real distributions
  are seen):
  - Label balance drift > 3× between sites → block, investigate labelling.
  - Per-feature median > 10× off between sites → block, suspected unit
    conversion bug.
  - Missing-rate divergence > 25 percentage points on any feature → block,
    suspected schema-mapping bug.

**Acceptance for the preprocessing track:** frozen schema published; both
sites produce their summary report; all divergence gates cleared or their
failure explained.

---

## Local sanity baselines

Two milestones, each triggered by its site's preprocessing completing. Maps
to architecture doc §8's v0 milestone.

- [ ] v0.a Local xgboost on real Geneva data at the frozen schema. Report
  AUC-ROC, AUC-PR, Brier on a held-out Geneva split. Unblocks v1.1/v1.2.
- [ ] v0.b Local xgboost on real Shenzhen data at the frozen schema, run
  remotely by partner. Same three metrics on their held-out split. Only
  the metric numbers leave the site — no data, no model. Unblocks v1.3.

**Acceptance for the local sanity baselines:** both per-site AUC-ROC /
AUC-PR / Brier reported on the real frozen schema; numbers logged in the
same versioned notebook that carries the v1.3 headline curve.

---

## Explicitly not on this roadmap

Cross-reference architecture §9.

- **Homomorphic encryption.** Out of scope per architecture §6.6 — HE does
  not mitigate TimberStrike (the actual reconstruction risk) and the 2-site
  threat model does not need it.
- **Vertical FL.** Same schema on both sides means horizontal FL. Vertical
  FL is the setting where encrypted-gradient schemes (SecureBoost line) become
  structurally necessary; that is a different problem and not this pilot.
- **FedAvg / gradient averaging strategies.** Wrong strategy family for
  tree-structure aggregation.
- **Deep tabular networks.** GBDT is the appropriate baseline for clinical
  tabular data.
- **Per-site personalization heads.** Reserve for v2 only if v1 site-stratified
  metrics diverge substantially.
- **Pooling of raw data at any point.**

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 12 issues found, 12 resolved; 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement
