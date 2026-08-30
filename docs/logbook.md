# Logbook

- 2026-07-14 — working_example flwr xgboost run on GVA data; 3 rounds, AUC 0.748 / 0.730 / 0.734

- 2026-07-16 — 1.b bagging vs cyclic, matched §3 params + 40-tree budget; final-model AUC on half A / B (last-trained site):
  R1 bagging 0.690 / 0.661; R2 cyclic-fwd 0.701 / 0.663 (last B); R3 cyclic-rev 0.696 / 0.670 (last A).
  Cyclic alternates strictly, R2/R3 end on opposite sites; no gross last-site bias (R2-vs-R3 Δ≤0.008 same half). 18/18 tests pass.

- 2026-07-17 — 1.c site-stratified evaluation harness: per-site AUC-ROC / AUC-PR / Brier + confusion
  at a fixed (0.5) and a per-site Youden-J operating point, bootstrap 95% CIs, a site-preserving
  evaluate aggregator (no cross-site averaging), and a schema-validated JSON artifact per run under
  `out/metrics/`. Final-round per-site AUC-ROC (95% CI) on the seed=42, 20% split:
  R1 bagging — A 0.690 [0.609,0.769], B 0.661 [0.563,0.756] (both sites every round; the two AUCs
  differ ⇒ proof of no averaging). R2 cyclic-fwd — last-site B 0.663 [0.564,0.756]; R3 cyclic-rev —
  last-site A 0.696 [0.611,0.774] (one site per round under cyclic, opposite last sites). AUC-ROC
  reproduces 1.b's eval_set numbers (0.690 / 0.661 / 0.696). Fixed-0.5 confusion is degenerate
  (tp=fp=0) on the 8.9%-positive `3M Death` outcome ⇒ logged as a degenerate-threshold warning;
  Youden-J cells stay informative (R1 A: tp=24, fp=108). Offline `eval_final_model.py` on the saved
  R1 model matches the federated `auc_roc/<site>` to full precision (shared `compute_binary_metrics`).
  All three artifacts are strict-valid JSON (jq), 0 nulls this run. 37/37 tests pass.

- 2026-07-18 — 1.d federated-vs-pooled correctness check implemented and run. New pooled
  reference (`baseline.train_pooled_booster`, deterministic: pinned `seed=0` + `nthread=1`),
  per-site shared scorer, serialization round-trip tripwire, and matched-budget provenance read
  from each model's embedded `fed_run_config` (server_app now stamps it). All three models were
  regenerated so provenance = `model`. 71/71 tests pass. **Finding: the check FAILS — federated
  is far below pooled.** Pooled per-site AUC-ROC: A 0.783, B 0.777 (pooled-combined 0.780).
  Federated: bagging A 0.692 / B 0.660; cyclic-fwd A 0.701 / B 0.663; cyclic-rev A 0.696 / B 0.670.
  Per-site |Δ| ≈ 0.08–0.12 for **every** strategy — an order of magnitude above the ±0.03
  provisional bound. Calibration (`--calibrate 5`, pooled-side only): pooled AUC very stable
  (spread ≤0.008), so the gap is systematic, not seed noise. Diagnostic: a CENTRALIZED model on
  only ONE half (40 trees, same params) scores 0.75 (half-A model) / 0.77 (half-B model) on the
  held-out splits — i.e. a single site's data trained centrally beats ALL federated runs, which
  use BOTH sites. So the gap is NOT a data-volume effect; the federated training path (1.b) is
  losing information (likely boosting-continuation / global-model-accumulation defect, affecting
  bagging AND cyclic alike). 1.d did its job as a tripwire. Root-causing the 1.b federated pipeline
  is a separate follow-up (out of 1.d scope); the ±3-AUC bound must NOT be relaxed to 0.13 to mask
  it. Artifact: `out/metrics/fed_vs_pooled.{json,md}` (gate `passed: false`, `roundtrip_ok: true`).

- 2026-07-18 — 1.d gap ROOT-CAUSED and FIXED. Not a boosting-continuation defect (the report's
  guess): `xgb.train(40)` and an in-process 40x `.update()` loop are identical, so continuation is
  fine. Real cause: `client_app` rebuilds a fresh `Booster` + `load_model` every round (the ensemble
  crosses the network), which RE-SEEDS XGBoost's column-subsample RNG from `params.seed=0` each round.
  With `colsample_bytree=0.8` over 2 features that deterministically draws the SAME single column
  every round, so all 40 trees split on `Age` and NIH on admission is never used (0 splits vs pooled's
  170). Confirmed by standalone repro that matched the saved cyclic model to the digit, a param sweep
  (colsample=1.0 → lossless; colsample=0.8 → collapse), and dumping split features. Fix: advance the
  seed per round in `client_app` (`round_seed(params, r)` = base + round; `train()` refactored over a
  new testable `_train_round`). Regenerated all three models on the live 2-SuperNode federation — split
  usage now balanced (bagging 110/166, cyclic-fwd 106/178, cyclic-rev 114/173 Age/NIH). Gate re-run
  **PASS** at the unchanged ±0.03: bagging ΔA 0.003 / ΔB 0.017; cyclic-fwd ΔA 0.006 / ΔB 0.007;
  cyclic-rev ΔA 0.000 / ΔB 0.016. 75/75 tests pass (4 new per-round-seed regressions). Write-up:
  `out/1d_solution.md`. Artifact: `out/metrics/fed_vs_pooled.{json,md}` (`passed: true`).

- 2026-07-19 — 1.e secure topology implemented and verified (docs/specs/1e_secure_topology.md). The
  insecure loopback run is replaced by server-side TLS + CA pinning + EC P-384 node authentication —
  flwr 1.31's supported realization of the roadmap's "mTLS" (spec §3.6; no X.509 client certs).
  New `scripts/gen_certs.sh` (local CA, 90-day loopback-SAN server cert, per-node OpenSSH keys, all
  in gitignored `.secrets/` — also excluded from the FAB via architecture/.gitignore);
  `run_local_federation.sh` refactored: all ports/paths now come from the committed
  `[tool.fed_stroke.superlink]`/`[tool.fed_stroke.nodes]` tables in pyproject.toml (NOT
  `[tool.flwr.federations]`, which `flwr run` would migrate + comment out, spec §3.9), the
  `[superlink.local-deployment]` entry in ~/.flwr/config.toml is generated on every `start` via
  flwr's own merge-upsert (other connections preserved, `insecure` dropped), readiness is polled
  (no more sleep 3), and both node keys are registered on every `start`. Verified end to end:
  bagging AND cyclic rounds complete over the TLS channel; halves roughly equal (A 1845/B 1829 rows,
  1691 unique patients each, label balance 0.0889/0.0875; disjointness asserted by
  prepare_geneva_halves.py). Negative checks are an asserting harness
  (`scripts/verify_negative_security.sh`, `run_local_federation.sh verify`), all fail closed with
  live-pinned signatures: unregistered key → SuperLink "[Fleet.ActivateNode] Activation failed: No
  SuperNode found with the given public key."; wrong CA → SuperNode "SSL/TLS handshake error
  detected."; insecure `flwr run` → exit 1 "Connection to the SuperLink is unavailable"; plus
  pyproject.toml stays byte-identical after `flwr run`. §8 residuals resolved empirically: the
  SQLite `--database` DOES persist registrations, but they cannot be re-bound to static keys on
  restart (killed nodes stay 'online' until heartbeat expiry ~1 min → activate fails; and
  node.public_key is UNIQUE with rows kept after unregister → a key can never be re-registered in
  the same DB). So `start` recreates the LinkState and registers fresh — quick stop/start restart
  proven robust — and key rotation always mints a NEW key (procedure in gen_certs.sh header).
  86/86 tests pass (11 new in tests/test_secure_topology.py: cert/key/SAN/validity invariants,
  flwr-parseability of the OpenSSH keys, no-federations-block tripwire, config-generation
  merge-not-truncate via FLWR_HOME, gitignore hygiene).

- 2026-07-20 — 1.f Docker smoke build implemented and verified on the dev machine
  (docs/specs/1f_docker_smoke_build.md). New `architecture/Dockerfile` (python:3.12-slim + three
  apt packages — `openssh-client` for gen_certs.sh's `ssh-keygen` plus `libgomp1`/`openssl` as
  defense-in-depth, see the finding below — then `uv sync --locked --no-dev` from the committed
  uv.lock), `architecture/.dockerignore`, `architecture/scripts/smoke_pipeline.sh`
  (the container CMD and the local smoke command both), and `tests/test_docker_smoke.py`. The
  container REPRODUCES the host tree (/workspace/architecture + /workspace/out, venv at
  /workspace/architecture/.venv) so `run_local_federation.sh`/`gen_certs.sh` run VERBATIM,
  unforked — a single loopback container fits the whole 1.e topology (127.0.0.1) with no docker
  networking. Build context is `architecture/` (NOT the repo root) so Docker actually reads
  `architecture/.dockerignore` and never bakes host `.secrets/`/`.venv/`/`.federation/`/`out/`.
  Real Geneva halves are bind-mounted at run time (`-v <repo>/out:/workspace/out`), never baked in.
  Verified end to end: image builds (1.63 GB); build-time no-secrets check passes (`.secrets/`,
  `.federation/` absent in the image); `docker run --init` brings up the secure topology (TLS +
  node auth, both nodes registered), streams one 20-round bagging run to completion over the TLS
  channel (`flwr run . local-deployment --stream`), and the smoke gate's OWN assertion passes —
  both sites present in the last round with finite AUC (final-round auc_roc A 0.779 / B 0.794),
  then teardown. **Verification finding that corrected the spec's §3.1 premise:** dropping
  `libgomp1` did NOT break the run — `xgboost==3.3.0`'s manylinux wheel vendors its own
  `libgomp-*.so.1` under `xgboost.libs/` (confirmed via `ldd libxgboost.so`), so `import xgboost`
  works without the system OpenMP; and `openssl` already ships in `python:3.12-slim`. The ONE
  package both required and missing from the slim base is `openssh-client` — dropping it makes the
  container fail fast at `gen_certs.sh:99` (`ssh-keygen: command not found`, exit 127) before any
  training. That is the real Dockerfile bug 1.f catches; `libgomp1`/`openssl` are kept as
  defense-in-depth (insurance against xgboost wheel / base-image changes), documented as such in
  the Dockerfile. Two further spec/impl bugs caught pre-merge: (1) a `.dockerignore` with trailing
  inline comments + an em-dash crashed BuildKit's exclude-patterns parser (fixed: bare ASCII
  pattern lines — Docker `.dockerignore` supports neither inline comments nor non-ASCII bytes in a
  pattern); (2) the spec's §6.2 no-secrets check tested `.venv/pyvenv.cfg` absence, but `uv sync`
  rebuilds the venv inside the image so that path always exists — the check could never pass;
  corrected to assert `.secrets`/`.federation` absence (the host-only dirs nothing recreates).
  `run_local_federation.sh`/`gen_certs.sh`/architecture/.gitignore reused unchanged.
  Digest pin, exact OS-OpenMP pin, non-root USER, and the non-dev-machine clean-bootstrap test
  are deferred to 1.3.a (roadmap). 99/99 tests pass (13 new static invariants in
  tests/test_docker_smoke.py; the live build+run is a scripted dev-machine check, not a pytest).
- 2026-07-20 — DP plug-point prototype (Phase v1.1/v1.2 prerequisite) implemented and verified
  (docs/specs/1_1_prereq_dp_plugpoint.md). New `fed_stroke/dp/` subpackage: a torch-free RDP
  accountant (`accounting.py`), a self-contained NumPy DP-GBDT learner + mechanism + seam
  (`boost.py`), a single-site synthetic generator (`synthetic.py`), a three-arm demo
  (`scripts/dp_prototype_demo.py`), and `tests/dp/`. numpy/scipy became explicit runtime deps;
  opacus>=1.5,<2 is dev-only (the gate reference; the runtime stays torch-free — verified: under
  `uv sync --no-dev`, opacus/torch absent, `import fed_stroke.dp` succeeds).
  **THE GATE PASSES:** our `account_run` ε equals Opacus `RDPAccountant` within rel=1e-6 at every
  q=1.0 toy setting (σ∈{0.5,1,2}×steps∈{1,20,100}); `DEFAULT_ORDERS == RDPAccountant.DEFAULT_ALPHAS`;
  the inverse `noise_multiplier_for_epsilon` round-trips to target ε (ε∈{1,3,5,10}, m∈{80,160})
  without over-spending. 52 dp tests + 151 full-suite green.
  **Spec correction caught by the gate:** the spec's order grid `range(11,64)` is wrong — Opacus
  uses `range(12,64)` (integer 11 is deliberately absent; the fractional run tops out at 10.9).
  `range(11,64)` both fails `test_default_orders_match_opacus` and injects an extra α=11 that can
  win the min and break the rel=1e-6 match. Implemented `range(12,64)`.
  **Composition pinned (the Opacus gate cannot see these, §3.2/§3.3):** `num_histogram_queries =
  D×T` (levels, NOT 2^D−1 nodes) and `num_gaussian_releases = 2×D×T` (gradient AND hessian are two
  Gaussian releases per level — feeding D×T under-reports ε ~2×). Leaf clipping is post-processing:
  reported ε is invariant to `clip_bound`. Laplace calibrates to L1 (`d`), not L2 (`√d`).
  **Scale correction (from the user):** the cohorts are disbalanced — GVA >2000 total (~1000/node),
  Shenzhen ~40000 — NOT the spec's assumed "~380 per-site". DP utility is scale-sensitive: at n≈380
  the honest 120-release accounting drove ε≤5 to chance; at n≈1000 (GVA per-node, the binding case)
  the mechanism shows a clean monotone ε→AUC erosion. Three-arm synthetic sanity (n=1000, seed=0,
  max_depth=3, rounds=20, gaussian, 60 levels / 120 releases, δ=1e-5, q=1.0):
  A classic-xgb AUC 0.834 ≈ B numpy-nonoise 0.822 (A→B = learner cost); then B→C = privacy cost —
  ε=1 (σ=44.3) 0.529, ε=5 (σ=10.4) 0.648, ε=30 (σ=2.46) 0.848, ε=100 (σ=1.07) 0.804, ε=1000
  (σ=0.27) 0.821. Takeaway: at GVA per-node scale, meaningful DP utility needs ε≳30 under this
  honest 2·D·T accounting. Seam (`DPConfig`/`HistogramNoiseMechanism`/`train_dp_gbdt`) + `dp.*`
  config are defined; FL wiring (DPBooster serialization, DP-aware aggregator) stays a downstream
  v1.1 integration item (§4.5).

- 2026-07-21 — 1.1.a HPO harness implemented (docs/specs/1_1_a_hpo_harness.md): pure core
  `fed_stroke/hpo.py` + driver `scripts/run_hpo.py` (`flwr run --stream`, offline both-halves
  scoring, winner hold-out re-run, 4 artifacts). Objective = mean two-half AUC-ROC, tie-break lower
  cross-site variance, scored on the SAVED model (bagging/cyclic comparable). Split unified into
  `task.resolve_run_split` (3 new config keys default to today's flat seed-42 split); final report
  is a **patient-disjoint** hold-out (fixed `HOLDOUT_PARTITION_SEED=42`, not a leaky held-out seed).
  **Validated on EXAMPLE halves only (throwaway):** V1 pytest 184/184 (+33, incl. seed-42
  byte-identity + HELD-disjointness guards); V2 dry-run 972 runs / 0 rejected; V3/V5 bagging+cyclic
  E2E on the live federation (reduced {4,8}-tree grid — full-grid run ~4 min via deployment polling)
  → 4 artifacts, ⚠ noise-tier banners; V4 `eval_final_model.py --holdout-eval` reproduced the HELD
  AUC-ROC to fp identity (0.8005 / 0.7705). Deliverable ranges await the same commands on real
  frozen-schema data (`--data-provenance real-frozen-schema`) — the hand-off to 1.1.b / v1.3 (1.3.b′).

- 2026-07-22 — 1.1.a′ DP → FL integration landed (docs/specs/1_1_a_prime_dp_fl_integration.md):
  the single-site DP seam is now wired into the live 2-SuperNode federation. `DPBooster` gains JSON
  serialization (`dp-gbdt-v1`); the boost loop is refactored into a shared `_grow_trees` core so
  `train_dp_gbdt` stays byte-identical and a new `dp_local_boost` resumes from the global's margins;
  `DPFedXgbBagging` concatenates DP trees, cyclic reuses `OrderedFedXgbCyclic` unchanged. Accounting
  is run-level per busiest site: σ calibrated once to `n_rel = 2·D·per_site` (cyclic uses
  `⌈num_rounds/num_sites⌉`); noise seeded from public `(base_seed, round, site)` only. No config keys
  added, accountant math untouched.
  **Validated on the EXAMPLE Geneva halves only — NOT a real-ε claim (that sweep is 1.1.b):** V1
  `pytest` 198/198; V2 torch-free import under `uv sync --no-dev`; V3 non-DP path byte-identical; V4-bag
  + V5-cyc DP gaussian (ε=30) complete over TLS — server logs ε 29.99998917 / σ 2.83742 /
  `num_releases=160`; V-DP `eval_final_model.py` auto-detects `dp-gbdt-v1` and reproduces the
  federated evaluate AUC to fp identity. Turns 1.1.b into "the 1.1.a sweep + DP arms" with no new
  orchestration.

- 2026-07-22 — 1.1.b gate, review round 1: two independent DP reviews of
  docs/reviews/dp_accountant_review_packet.md returned (A: BLOCK, system/pipeline level;
  B: approve-with-conditions, mechanism level; joined in
  docs/reviews/dp_accountant_review_joint_summary.md). **Accountant math confirmed by both**
  — sensitivities, 2·D·T count, Gaussian/Laplace calibration, RDP+Balle conversion, σ table
  (51.17/18.89/12.05/6.70 for ε=1/3/5/10 at k=160, δ=1e-5; B cross-checked installed Opacus
  1.6.0 to machine precision) — `accounting.py` verified and frozen. **Gate NOT passed:** six
  blocking findings in the surrounding mechanism/pipeline — un-noised empty-node branch
  (boost.py:434), deterministic public-seeded DP noise (client_app.py:126-128, a deliberate
  1.1.a′ design now reversed for production), non-adjacency-stable stratified split (task.py),
  admission-row (not per-patient) adjacency unit, exact num-examples/validation-metric releases
  outside the accounting, and missing cross-run sweep composition. Remediation roadmap 1.1.a″
  written (docs/specs/1_1_a_doubleprime_dp_remediation.md): R1 always-issue empty-node query, R2 OS
  entropy for DP noise (deterministic RNG injection-only in tests), R3 one-row-per-patient
  dedup, R4 keyed-hash split, R5 two-tier release boundary + public site weights, R6 append-only
  RDP run ledger with composed-ε reporting, R7 fail-closed precondition gate (incl. explicit
  base_score), R8 packet refresh + re-review. Exit = both reviewers sign off; 1.1.b stays
  blocked for real Geneva data until then. No sign-off recorded.

- 2026-07-22 — R3 dedup rule decided (user): one row per patient via **max-outcome +
  lexicographic case_admission_id tie-break**, applied at load for the **whole pipeline**
  (DP and non-DP arms), before the split. Grounding on the real halves: 3674 rows → 3382
  patients (−7.9%), 243 multi-admission patients, m ≤ 6, only 16 with disagreeing labels
  (balance 0.0882 → 0.0923). First/index-admission rejected for now — the EDS suffix is not
  chronological (age monotone in only 56% of multi-admission patients); revisit if the frozen
  schema includes an admission date. Non-DP baselines must be re-run once on the deduped
  cohort. Spec updated: docs/specs/1_1_a_doubleprime_dp_remediation.md §R3.

- 2026-07-23 — 1.1.a″ remediation implemented (all code items R1–R7 + R9;
  docs/specs/1_1_a_doubleprime_dp_remediation.md). R1 depth-only leaf guard — every node at
  depth < D issues its noised query (empty ⇒ pure-noise release, already charged by 2·D·T). R2
  DP noise draws OS entropy at both call sites; deterministic rng is injection-only;
  `dp.noise-seed` fail-closes (insecure-test hatch rejected on real-frozen-schema); `_site_hash`
  deleted; source-audit tripwire test. R3 one-row-per-patient dedup in `generate_splits`
  (max-outcome + tie-break), loader assertion + learner-side gate check. R4 stratified
  `train_test_split` replaced by `HMAC(SPLIT_KEY‖role‖seed, patient_id)` assignment
  (role ∈ {partition, subsplit} domain-separated; stratification loss accepted symmetrically);
  DP mode refuses predefined pid lists. R5 DP train reply sends fixed public `dp-site-weight`
  (node_config; pyproject 1000/1000), never the exact count; evaluate metrics declared
  inside-boundary. **Boundary rule: nothing inside-boundary appears in any artifact that leaves
  the project.** R6 `fed_stroke/dp/ledger.py` — per-site append-only jsonl (intent-to-spend on
  each site's FIRST round of a real-frozen-schema DP run), `ledger_total()` composes per site
  via the frozen accounting fns (compose-of-N pinned against Opacus); server logs composed
  total next to per-run ε; reporting template docs/templates/1_1_b_report_skeleton.md. R7
  `validate_dp_preconditions()` gate in the client DP branch (labels/features finite, one row
  per patient, per-mechanism budget checks — laplace validated on b_g/b_h not the nan σ —
  fixed public bins, EXPLICIT `params.base-score` (new pyproject key, also pins the non-DP arm
  init: deliberate, matches R9 arms), authorized round count). R9 identity mechanism reachable
  with `dp.enabled` + `dp.mechanism="identity"` = comparator arm B (0 releases, ε=∞→null, never
  ledgered); arm B == σ→0 arm C pinned; A→B→C decomposition promoted to the 1.1.b template with
  B→C as the privacy-cost headline (arm A at subsample=colsample=1.0). R8 packet refreshed
  (line refs, Claim 8 premise via R1, Claim 9 re-stated as enforced, §7 assumption table +
  boundary diagram, H-side single-bin assertion added to `test_histogram_sensitivity_bound`).
  **Suite: 233/233 (was 198); σ table byte-identical (test-pinned); non-DP `_train_round`
  byte-identity hashes untouched.** Rebaseline audit: the spec-predicted test breaks from R3/R4
  (`test_hpo.py` seed-42 byte-identity, eval/baseline fixtures) did NOT fail — they compare the
  shared split code to itself and the fixtures are single-admission, so they passed mechanically;
  the seed-42 test was renamed (`test_resolve_run_split_flat_delegates_to_generate_splits`) to
  stop claiming a legacy identity that no longer exists. Only intentional behavior rework:
  test_dp_fl case 4 (entropy). **Still open for 1.1.a″ acceptance:** non-DP arm-A re-run on the
  deduped halves with the delta logged (real-data step, after sign-off), packet re-issue to
  BOTH reviewers (A's re-review required — verdict was BLOCK), sign-off recorded here, and only
  then the Geneva-only federated A→B→C sweep + go/no-go BEFORE Shenzhen integration.

- 2026-07-23 — 1.1.a″ loopback E2E verification (acceptance 2 + fail-closed rails exercised
  for real). Live TLS 2-node federation: DP gaussian (ε=30) and identity (arm B) runs both
  complete under **production OS entropy**; server logs the accounting line
  (`reported_epsilon=29.9999992 σ=0.897 num_releases=16` on the short run) and the arm-B
  annotation (`IDENTITY mechanism … no ε spent, not a DP release`); no ledger written
  (provenance=example-halves — correct gating); two unseeded `train_dp_gbdt` runs on identical
  data produce different serialized models at identical reported ε. **The new rails caught two
  REAL data defects in the example halves on the first E2E attempt:** (1) half A contains an
  exactly-duplicated `case_admission_id` (REDACTED, twice, same label) — the R3 loader
  assertion fail-closed; dedup hardened to stable-sort + head(1) so exact-duplicate ids
  deterministically reduce to one row (regression test added). (2) Both halves contain missing
  features (NIH on admission: 84/1845 in A, 72/1829 in B; Age: 1 in B) — the R7 gate refuses
  them. **Previously the DP learner silently binned NaN into the TOP bin (searchsorted
  fallthrough → max-NIHSS artifact), so the 1.1.a′ V4/V5 DP numbers quietly trained on that;
  the non-DP XGBoost arm handles NaN natively, so the arms also diverged on missingness.** E2E
  was verified on NaN-free copies (dropna; originals restored, sha256-verified). ⇒ **New open
  decision before any real-data DP run:** a missingness policy for the frozen schema (impute /
  explicit missing indicator / drop) — applied identically to ALL R9 arms so A→B→C stays
  unconfounded.

- 2026-07-23 — Missingness policy decided (user) and implemented: **variant 1, sentinel /
  missing bin, applied at the loader for ALL arms** — flagged `REMOVE-IF-NO-DP` at every touch
  point (grep token) because it exists ONLY for the DP learner; if the project later moves
  forward without DP, remove it and let XGBoost's native NaN handling take over. Mechanics:
  `schema.MISSING_SENTINEL = -1.0` (public); `generate_splits` (the R3 chokepoint, so the
  pooled baseline and offline scorer inherit it too) fills NaN features with the sentinel;
  `fixed_bin_edges` reserves **bin 0** structurally (edges = [sentinel, linspace(lo, hi,
  max_bins)]) so missing/real separation holds exactly at ANY max_bins and FEATURE_RANGES keep
  their clinical meaning; arm A's XGBoost sees the sentinel as an ordinary splittable value
  (DMatrix missing stays NaN), so A→B→C shares one encoding. **Zero accounting impact** — a
  record still lands in exactly one bin per feature; σ table pins unchanged. Cost: max_bins−1
  real-value bins (32→31 at default). **Model format bumped `dp-gbdt-v1` → `dp-gbdt-v2`**:
  v1 artifacts' stored (feature_ranges, max_bins) would reconstruct shifted edges under the
  new function and silently mispredict — the bump makes every pre-remediation artifact (all
  invalid anyway: public-seeded noise, non-deduped cohort) fail loudly at load; the offline
  loader now routes any `dp-gbdt-*` prefix to DPBooster so stale files get the clear version
  error. Suite 235/235. Loopback E2E re-run on the ORIGINAL example halves (NaNs and the
  duplicate caid included, no preprocessing): non-DP arm A, identity arm B, and gaussian arm C
  all complete over TLS — acceptance 2 now holds on the untouched example data. Packet §7
  assumption table gains row 14; the 1.1.b template records the encoding + per-half missing %.

- 2026-08-30 — R8 re-review returned: **A: BLOCK → approve with conditions; B: approve with
  conditions** (docs/reviews/dp_accountant_review_findings_A_rereview.md,
  dp_accountant_review_findings_B_rev2.md — committed from reviewer worktrees). Both confirm the
  accountant math and the six round-1 blockers closed. Blocking conditions fixed today:
  **A-C1 ≡ B-F8** — DP rails (R2 hatch, R6 ledger) keyed off submitter-set
  `run_config["data-provenance"]`; now node-owned (`node_config` data-provenance +
  dp-ledger-path, `validate_dp_run_provenance()` fail-closed, disagreeing run-config refused;
  driver `--ledger-path` removed, `--data-provenance` must match the nodes). **A-C2** — cyclic
  ledgers only the round-1 site; cyclic + DP now refused on real data (client + driver),
  rehearsal on example halves still allowed. Suite 282/282. **Still open before real run:**
  A-C3 (boundary list for out/metrics + ledger, Shenzhen disclosure in packet §7.2),
  B-F9 (cross-half patient_id disjointness check, logged). **Before packet circulates:** A-C4
  (dp_local_boost docstring), A-C5 (ledger sha256 + count in report), A-C6 (dp-site-weight
  fixed-before-run statement), B-F6′ (boost.py line refs, F6→F7 attribution, stale scope
  bullet), B-F10 (sentinel-bin nit). No sign-off recorded yet — both are conditional on the
  above.

- 2026-08-30 — Decision (user): re-review conditions **A-C3** (boundary list for out/metrics +
  ledger, Shenzhen disclosure note in packet §7.2) and **B-F9** (cross-half patient_id
  disjointness check) are SKIPPED for the Geneva-only 1.1.b run. Both remain formally listed by
  the reviewers as sign-off conditions; the deviation must be agreed with them at sign-off or
  the sign-off stays conditional. Both nodes are cut from one registry by one keyed hash
  (prepare_geneva_halves.py), which is the informal basis for skipping F9. Revisit both at
  Shenzhen integration (A-C3's disclosure point becomes load-bearing there).

