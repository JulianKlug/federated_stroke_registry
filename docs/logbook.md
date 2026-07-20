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
