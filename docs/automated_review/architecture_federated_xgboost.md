# Architecture — federated XGBoost on Flower for the Geneva ↔ Shenzhen stroke registry

Companion to [literature_review_federated_xgboost.md](literature_review_federated_xgboost.md).
Every decision below cites the source that motivates it.

## 0. Setting

- Two clinical sites, both always-online, both with substantial local
  compute (cross-silo). This matches the cross-silo regime described by
  Kairouz et al. and the FL-in-healthcare guidance ([Kairouz et
  al.][fl-open-problems]; [FL in healthcare review, 2025][fl-healthcare-review]).
- Horizontal partitioning: same schema after alignment
  (`registry_alignement/`), different patients.
- Task: binary or multi-class stroke-outcome model on aligned tabular
  features (starting target could be e.g. 3-month favorable outcome
  mRS ≤ 2 or in-hospital mortality — outside this doc's scope).

## 1. Framework choice — Flower + `flwr-xgboost`

**Decision.** Use Flower's `flwr-xgboost` stack in horizontal mode.

**Why.**

- First-party integration with XGBoost 2.x with well-documented server
  strategies and a comprehensive HIGGS example that mirrors our data
  shape ([Flower comprehensive example][flower-comprehensive]; [Flower
  blog 2024][flower-blog-2024]).
- Both aggregation strategies (bagging, cyclic) are available out of the
  box; we can A/B them without reimplementing infrastructure.
- Flower supports SecAgg+ ([Salvia][salvia]) and integrates with DP
  wrappers, giving us upgrade paths without a framework rewrite.
- Alternative — NVIDIA FLARE — is a strong choice if we later decide we
  need HE (see §6.3), but its default setup targets larger deployments;
  Flower is lighter for a two-site pilot.

## 2. Training strategy — both bagging and cyclic, chosen empirically

**Decision.** Treat `FedXgbBagging` and `FedXgbCyclic` as **co-equal
candidates** for v1, not primary/ablation. Choose the winner on
site-stratified AUC (§4) rather than *a priori*.

**Why.**

- Geneva and Shenzhen cohorts are **imbalanced in size** — the strategy
  choice is not neutral in that setting.
- Bagging: on each round every client boosts the same number of trees,
  so a site with 10× more patients still contributes exactly one tree
  per round in the default configuration. This under-uses the larger
  cohort. Bagging works best when clients are balanced, per Flower's own
  AUC comparison on HIGGS ([Flower blog on bagging
  aggregation][flower-blog-bagging]; [Flower comprehensive
  example][flower-comprehensive]).
- Cyclic: each site refines the ensemble on top of the previous site's
  model. When one cohort is much larger, cyclic lets the larger site do
  correspondingly more useful boosting work, and gives the smaller site
  a full model to fine-tune against rather than a diluted average. In a
  2-client setup this is closer to sequential transfer than to
  ensembling, which can be an advantage under imbalance.
- Neither strategy dominates the other in the literature; the Flower
  comprehensive example shows both converging to comparable AUC on
  balanced HIGGS partitions, and neither has been benchmarked on strong
  cross-silo imbalance in a published clinical setting.

**How to decide in v1.**

- Run both `FedXgbBagging` and `FedXgbCyclic` with matched hyperparameters
  (§3) and matched total tree budget.
- Compare on **site-stratified AUC** (Geneva-test, Shenzhen-test), not
  just pooled AUC, because pooled AUC can hide a strategy that only
  works well on the larger site.
- Also compare **calibration per site** — cyclic ordering can leave the
  final model biased toward the last site trained on; alternate the
  order across runs to check.

## 3. Starting hyperparameters

Grounded in the Flower comprehensive example (defaults for HIGGS,
binary classification, ~11M rows) ([Flower comprehensive
example][flower-comprehensive]) and adjusted for a much smaller clinical
tabular dataset where regularization and shallow trees matter more
([DP-XGBoost][dp-xgboost]; [Maddock et al.][fed-dp-boost]):

| Parameter                   | Value                | Rationale |
| --------------------------- | -------------------- | --------- |
| `num_server_rounds`         | **20** (initial)     | Cross-silo FL uses tens of rounds, not thousands ([Kairouz et al.][fl-open-problems]; [Flower comprehensive][flower-comprehensive]). Sweep 10, 20, 50. |
| `local_epochs`              | **1** tree / round   | Flower's canonical setting for bagging ([Flower blog 2024][flower-blog-2024]). Total ensemble ≈ `num_rounds × n_clients × 1` = 40 trees. |
| `params.objective`          | `binary:logistic`    | Task-dependent; start with binary outcome. |
| `params.eval_metric`        | `auc`                | Class-imbalance-robust metric aligned with the HIGGS example ([Flower comprehensive][flower-comprehensive]). Also track `aucpr`, calibration, and site-stratified metrics. |
| `params.eta`                | **0.1**              | Flower default ([Flower comprehensive][flower-comprehensive]). |
| `params.max_depth`          | **4** (start), 6 max | Shallower than the Flower HIGGS default of 8 — reduces variance on small clinical data and cuts the DP privacy budget significantly ([DP-XGBoost §5][dp-xgboost]; [Maddock et al.][fed-dp-boost]). |
| `params.min_child_weight`   | **5**                | Prevents leaves supported by a handful of patients — same reasoning as reducing depth. |
| `params.subsample`          | **0.8**              | Standard XGBoost bagging noise; also mildly reduces reconstruction fidelity by making tree structure less deterministic ([TimberStrike §5, mitigations][timberstrike]). |
| `params.colsample_bytree`   | **0.8**              | Same rationale. |
| `params.tree_method`        | `hist`               | Flower default and prerequisite for DP histogram noise later ([DP-XGBoost][dp-xgboost]). |
| `params.num_parallel_tree`  | **1**                | Flower canonical setup ([Flower comprehensive][flower-comprehensive]). |
| `fraction_train`            | **1.0**              | Both sites participate every round (cross-silo, always-available). |
| `fraction_evaluate`         | **1.0**              | Both sites evaluate every round for site-stratified monitoring. |

Data partitioning: **no artificial partitioning** — each site is its own
partition. This matches the "linear partitioner + N=2 clients" scenario in
the Flower comprehensive example ([Flower comprehensive][flower-comprehensive]).

## 4. Evaluation protocol

- **Centralized evaluation on each site's held-out set** — the server never
  sees the data; each client returns a fixed set of metrics after each
  round.
- Report AUC-ROC, AUC-PR, calibration (Brier score), and confusion matrix
  at the operating point, **stratified by site**. This is standard for FL
  healthcare studies ([FL in healthcare review, 2025][fl-healthcare-review]).
- Compare against three baselines: (a) local-only Geneva model on Geneva
  test, (b) local-only Shenzhen model on Shenzhen test, (c) hypothetical
  pooled model (only feasible if a DUA covers a small joint validation set;
  otherwise report the local-only baselines and skip the pool).

## 5. Distribution-shift handling

- Lock the aligned schema (see `registry_alignement/`) before any FL run.
  Include unit conversions ([Geneva/Shenzhen unit map][unit-conversions])
  and definition harmonization (ODT/ONT/DNT/DPT/OPT already implemented
  in `build_gva_summary_table.py`).
- Keep a `site` indicator **outside** the model (for monitoring), not as a
  feature. Adding it as a feature would give any downstream user a shortcut
  the model shouldn't be allowed to use, per the cross-silo iterative-alignment
  work ([Cross-Silo FL with Iterative Parameter Alignment,
  2024][crosssilo-align]).
- If site-stratified metrics diverge substantially, revisit the alignment
  layer first, then consider a lightweight per-site calibration head
  ([Liu et al., NeurIPS 2022][crosssilo-dp]) before touching the core
  model.

## 6. Privacy stack

The literature review's key finding is that "no raw data leaves the site"
is **not** sufficient — TimberStrike reconstructs 73–95% of a target
client's records on federated GBDT frameworks (Flower, NVFlare, FedTree)
including on a stroke prediction dataset ([TimberStrike, PoPETs
2025][timberstrike]). The stack below is picked with that attack in
mind, and its load-bearing layer is **DP on the histograms** — not HE.
See §6.6 for why HE is explicitly out of scope.

Attack-surface / defence map:

| Surface                                              | Attacker                          | Defence in this stack               |
| ---------------------------------------------------- | --------------------------------- | ----------------------------------- |
| Transport (client ↔ aggregator)                      | Network eavesdropper              | mTLS (§6.1)                         |
| Per-site histograms readable at aggregator/peer      | Honest-but-curious server/peer    | Not addressed in v1 — see §6.6      |
| **Final tree ensemble (TimberStrike surface)**       | **Honest-but-curious peer client**| **Cohort/tree hygiene + DP (§6.2–§6.3)** |
| Model published externally                           | Any downstream reader             | DP (§6.3) + governance (§6.4)       |
| Peer institution reading its own decrypted model     | Peer                              | Governance (§6.4)                   |

### 6.1 Baseline (mandatory)

- Mutual TLS between clients and the Flower `SuperLink`; certificate
  pinning; short-lived credentials.
- All model artifacts and telemetry at rest encrypted (site-controlled
  KMS).
- Feature-schema freeze: only the aligned columns are exposed to the
  Flower client; nothing else on the site's disk is reachable from the
  training process.
- Node-parquet de-identification (`preprocessing.anonymise`, anon-v1,
  schema frozen-v2): the id column is `pseudo_admission_id` =
  HMAC-SHA256(site key, patient id) + admission ordinal — no hospital
  identifier leaves the build; age is completed years plus a per-patient
  keyed jitter (±2 y), top-coded at 90 (SPHN de-identification guidance
  §5.1.9 / HIPAA Safe Harbor). The key is the data provider's, kept
  outside the repo and away from the artefacts (HRO Art. 26 coded
  data), immutable for the project (the R4 split hashes the pseudonym).
  The parquet carries no frame attrs (they held raw per-feature
  extremes) and the cross-site smoke report publishes p05/p95, never
  a single row's min/max.

### 6.2 Cohort-size and tree-shape hygiene (free, first-line)

With Geneva- and Shenzhen-scale cohorts, most per-record signal in
individual trees can be suppressed at zero utility cost by shaping the
tree ensemble so that no leaf, split, or threshold is uniquely tied to
a small group of patients.

- `max_depth ≤ 4` (§3). Shallow trees have coarser thresholds — a
  boundary "NIHSS ≤ 7.5" sitting between hundreds of patients on each
  side identifies no one; sitting between two patients does.
- `min_child_weight ≥ 20` (relax the v1 default of 5 upward as cohorts
  grow). This directly enforces that every leaf is supported by a
  meaningful patient count; small leaves are the highest-leak part of a
  tree.
- `subsample=0.8`, `colsample_bytree=0.8` (§3). Non-deterministic tree
  structure makes reconstruction fidelity drop.
- Large N per site is the second free lever: it makes DP (§6.3) much
  cheaper, since histogram sensitivity is roughly constant while signal
  grows with N.

These do not eliminate TimberStrike — rare feature combinations and
outliers remain identifying, and membership-inference is easier than
full reconstruction — but they are the first cut and cost nothing.

### 6.3 Central-DP on split statistics (load-bearing defence against TimberStrike)

**Decision.** Adopt a DP-XGBoost-style noise mechanism on the histogram
statistics that flow into the aggregator. This is the **v1.1 milestone**
and the load-bearing privacy layer.

**Why.**

- DP is the only mitigation TimberStrike acknowledges as effective
  against tree-reconstruction attacks ([TimberStrike §6][timberstrike]),
  but naive application collapses utility. The DP-XGBoost / DP-TR /
  DP-XGB line of work ([Grislain & Gonzalvez, 2021][dp-xgboost];
  [Maddock et al., 2022][fed-dp-boost]) shows that with **shallow trees
  and few rounds** — exactly our starting configuration (§3) — DP-XGBoost
  can retain most of its utility.
- Starting budget: **ε ≈ 5**, δ = 1e-5 over the entire training run
  (this is on the loose end; healthcare pilots often report ε ≈ 1–10).
  Report ε per round × rounds accounted with Rényi-DP composition.
- Concretely: per-round Laplace/Gaussian noise on the gradient/hessian
  histograms shipped by each client before aggregation, plus geometric
  clipping of leaf outputs to bound sensitivity.

**Implementation note — Flower's built-in DP wrappers do not apply directly.**
Flower ships `DifferentialPrivacyClientSideFixedClipping` /
`Adaptive` (and server-side twins) as strategy wrappers ([Flower DP
how-to][flower-dp-howto]; [Flower DP explanation][flower-dp-explanation]).
Those wrappers clip and noise **model-parameter updates** on top of
FedAvg-style aggregation — they assume the strategy is averaging real-valued
weight tensors. `FedXgbBagging` aggregates tree structures, not weight
tensors, so we need DP-XGBoost-style noise on the **histograms before
tree construction**, not on the tree parameters after. That is a custom
plug-point on the client, informed by [Grislain & Gonzalvez][dp-xgboost]
and [Maddock et al.][fed-dp-boost]

### 6.4 Governance (mandatory, non-technical)

- Data-processing agreement + data-use agreement between the two
  institutions; local IRB / ethics approval on each side.
- DPO review at HUG; equivalent review at the Shenzhen partner
  (including any CAC self-assessment required under PIPL / DSL for
  cross-border transfer of derived model artifacts — see §6.6).
- In a two-party setting the DPA is the actual reason to trust the peer:
  neither DP nor HE removes the fact that the peer sees the final model.
  Governance is not a "soft" layer, it is doing real work.

### 6.5 Secure aggregation (SecAgg+) — low value for N=2

- Flower supports SecAgg+ via Salvia ([Li et al., 2022][salvia];
  [Flower SecAgg example][flower-secagg-example]).
- With only two sites, a masking-based two-party scheme reveals each
  input to the other side; SecAgg becomes useful only if we introduce a
  third participant or a non-trusted aggregator. Track this as a
  precondition for adding a third site.

### 6.6 Homomorphic encryption — explicitly out of scope for v1

**Decision.** HE (TenSEAL/CKKS, Paillier via NVFlare, or otherwise) is
**not** part of the v1 privacy stack.

**Why out of scope.**

- **HE does not mitigate TimberStrike.** The attack targets the *final
  tree ensemble*, which both sites must hold in plaintext to run
  predictions. HE can only encrypt intermediate transport and
  aggregation; the load-bearing defence against tree reconstruction is
  DP (§6.3), not encryption.
- **The 2-site cross-silo threat model doesn't need it.** HE would
  protect against a third-party aggregator reading per-site histograms.
  Our SuperLink is expected to be hosted on one of the two institutions'
  infrastructure with the other having read-only observer access.
  Whatever the aggregator sees, the peer also sees in the final model —
  encrypting the intermediate step buys nothing against the actual
  adversary.
- **Legal review has not (yet) demanded it.** Cross-border transfer
  regimes (Chinese PIPL / DSL / CAC outbound-transfer review, Swiss
  FADP, EU GDPR) require "appropriate safeguards" but do not name HE.
  In horizontal FL with `FedXgbBagging`, no per-record data crosses the
  border at all — only tree structures and aggregate histograms —
  making the "transfer of personal data" trigger legally arguable.
- **Engineering cost is high, and Flower has no first-party XGBoost HE
  plugin.** A TenSEAL-based integration would need custom histogram
  encryption at the client side (BFV, not CKKS, for near-integer
  counts) and a custom aggregator — essentially reimplementing what
  NVFlare provides with Paillier ([Wang et al., 2025][nvflare-secure-xgb]).

**Triggers that would move HE back in scope.** Any of the following
should reopen this decision:

1. The Flower `SuperLink` is hosted on infrastructure neither
   institution controls (a third-party academic hub, public cloud in a
   third jurisdiction). Aggregator becomes untrusted.
2. A DPO, IRB, or CAC reviewer flags the aggregation traffic as covered
   by outbound-transfer rules and requires a supplementary technical
   measure beyond DP + mTLS.
3. We add a third clinical site under a weaker DPA or a different
   jurisdiction than the Geneva-Shenzhen pair.
4. We move to vertical FL — e.g. labels held by only one site,
   features distributed across the others. Encrypted gradients then
   become structurally necessary (SecureBoost setting;
   [Chen et al.][secureboostplus]).
5. We ship per-round artifacts richer than aggregate histograms
   (unaggregated gradient statistics, small-bucket histograms,
   embeddings) across the border, which fires the transfer trigger
   more cleanly.

If any of 1–5 fires, the upgrade path is **NVFlare + Paillier HE**
([NVFlare Secure XGBoost user guide][nvflare-guide]), not TenSEAL/CKKS
glued onto Flower — NVFlare has a first-party plugin, TenSEAL would be
a from-scratch integration.

### 6.7 Threat model, explicitly

- Adversary: **honest-but-curious partner site**. This is exactly
  TimberStrike's threat model, and matches what the DPA/DUA between the
  two institutions is designed to accommodate.
- Also in scope: passive network adversary (mitigated by mTLS in §6.1).
- Not in scope for v1: malicious clients (they could poison the model —
  handle with model-level robust aggregation later), untrusted
  aggregator (see §6.6 triggers), side channels on compute
  infrastructure, or nation-state-level adversaries.
- All decisions above should be revisited if the threat model widens.

## 7. Reference architecture

```
┌────────────────────────┐                    ┌────────────────────────┐
│   Geneva site (HUG)    │                    │  Shenzhen site         │
│                        │                    │                        │
│  Aligned tabular EHR ──┼──► flwr ClientApp  │  Aligned tabular EHR ──┼──► flwr ClientApp
│  (registry_alignement) │      + XGBoost     │  (registry_alignement) │     + XGBoost
│                        │      + DP noise    │                        │     + DP noise
└──────────┬─────────────┘                    └──────────┬─────────────┘
           │                                             │
           │  mTLS, gRPC                                 │  mTLS, gRPC
           └─────────────────────┬───────────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Flower SuperLink (v1.x) │
                    │  Strategy: FedXgbBagging │
                    │  Aggregator: histograms  │
                    │    (DP-composed)         │
                    │  Metric aggregation      │
                    └──────────────┬───────────┘
                                   ▼
                       Global XGBoost ensemble
```

Neutral hosting for the SuperLink (e.g. a jointly-controlled VM in a
region acceptable to both institutions, or one site as host with
observer read-only access for the other) is a governance decision, not a
technical one — but it is a decision that must be made before v1.

## 8. Milestones

| Milestone | Content | Blocking dependencies |
| --- | --- | --- |
| **v0 — local sanity** | Train XGBoost locally at each site on aligned features; publish per-site AUC. | `registry_alignement` frozen for a defined feature set. |
| **v1 — Flower bagging + cyclic bake-off + baseline privacy** | Both `FedXgbBagging` and `FedXgbCyclic` with matched hyperparameters, mTLS, DPA/IRB, per-site + global metrics; pick winner by site-stratified AUC. | v0; hosting/governance decision; matched Python/XGBoost/Flower versions. |
| **v1.1 — DP-XGBoost noise** | Add histogram noise on the winning v1 strategy; sweep ε ∈ {1, 3, 5, 10}; report utility curve. | v1. |
| **v2 — Trigger-driven reopen (conditional)** | Reopen §6.6 HE decision *only if* any of the five triggers there fires. If so, migrate to NVFlare + Paillier HE, not TenSEAL. | §6.6 trigger observed. |
| **v2.1 — third site / SecAgg** | Add a third participant; enable SecAgg+ meaningfully. | Third-site partner. |

## 9. What we are explicitly not doing (v1)

- **No homomorphic encryption.** Out of scope for v1 per §6.6; HE does
  not mitigate TimberStrike (the actual reconstruction risk) and would
  harden a threat surface — untrusted aggregator or vertical FL — that
  we do not have. Five explicit triggers in §6.6 would reopen the
  decision.
- No vertical FL — same schema on both sides means horizontal
  ([SecureBoost line of work][secureboostplus] is not applicable).
- No deep tabular network — GBDT is the appropriate baseline for
  clinical tabular data.
- No FedAvg / gradient-averaging strategies — we are aggregating tree
  structures, not weights. Flower's built-in DP wrappers
  ([DifferentialPrivacyClientSideAdaptiveClipping][flower-dp-howto])
  target weight-averaging and do not directly apply — we need
  histogram-level DP noise instead (§6.3).
- No client-level personalization layer in v1 — reserve for v2 if
  site-stratified metrics diverge.
- No pooling of raw data at any point.

---

## References

- <a id="flower-blog-2024"></a>[flower-blog-2024] Flower Labs. *Federated XGBoost: Flower is all you need.* Flower blog, 2024-02-14. https://flower.ai/blog/2024-02-14-federated-xgboost-with-flower/
- <a id="flower-blog-bagging"></a>[flower-blog-bagging] Flower Labs. *Federated XGBoost with bagging aggregation.* Flower blog, 2023-11-29. https://flower.ai/blog/2023-11-29-federated-xgboost-with-bagging-aggregation/
- <a id="flower-comprehensive"></a>[flower-comprehensive] *Federated Learning with XGBoost and Flower (Comprehensive Example).* Flower Examples 1.29. https://flower.ai/docs/examples/xgboost-comprehensive.html
- <a id="secureboostplus"></a>[secureboostplus] Chen, W., et al. *SecureBoost+: A High Performance Gradient Boosting Tree Framework for Large Scale Vertical FL.* arXiv:2110.10927. https://arxiv.org/html/2110.10927v5
- <a id="nvflare-secure-xgb"></a>[nvflare-secure-xgb] Wang, Y., et al. *Secure Federated XGBoost with CUDA-accelerated Homomorphic Encryption via NVIDIA FLARE.* arXiv:2504.03909 (2025). https://arxiv.org/pdf/2504.03909
- <a id="nvflare-guide"></a>[nvflare-guide] *NVFlare XGBoost User Guide (Secure XGBoost).* NVIDIA FLARE 2.6 docs. https://nvflare.readthedocs.io/en/2.6/user_guide/federated_xgboost/secure_xgboost_user_guide.html
- <a id="dp-xgboost"></a>[dp-xgboost] Grislain, N., Gonzalvez, J. *DP-XGBoost: Private Machine Learning at Scale.* arXiv:2110.12770 (2021). https://arxiv.org/pdf/2110.12770
- <a id="fed-dp-boost"></a>[fed-dp-boost] Maddock, S., Cormode, G., Wang, T., Maple, C., Jha, S. *Federated Boosted Decision Trees with Differential Privacy.* CCS 2022. arXiv:2210.02910. https://arxiv.org/pdf/2210.02910
- <a id="salvia"></a>[salvia] Li, K. H., Cormode, G., et al. *Secure Aggregation for Federated Learning in Flower (Salvia).* DistributedML 2022; arXiv:2205.06117. https://arxiv.org/pdf/2205.06117
- <a id="flower-secagg-example"></a>[flower-secagg-example] *Secure aggregation with Flower (SecAgg+ protocol).* Flower Examples 1.29. https://flower.ai/docs/examples/flower-secure-aggregation.html
- <a id="flower-dp-howto"></a>[flower-dp-howto] *Use Differential Privacy — Flower Framework how-to.* https://flower.ai/docs/framework/how-to-use-differential-privacy.html
- <a id="flower-dp-explanation"></a>[flower-dp-explanation] *Differential Privacy — Flower Framework explanation.* https://flower.ai/docs/framework/explanation-differential-privacy.html
- <a id="timberstrike"></a>[timberstrike] Di Gennaro, M., De Lucia, G., Longari, S., Zanero, S., Carminati, M. *TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems.* PoPETs 2025(4). arXiv:2506.07605. https://petsymposium.org/popets/2025/popets-2025-0145.pdf
- <a id="fl-open-problems"></a>[fl-open-problems] Kairouz, P., et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in ML, 2021. https://arxiv.org/pdf/1912.04977
- <a id="crosssilo-dp"></a>[crosssilo-dp] Liu, Z., et al. *On Privacy and Personalization in Cross-Silo Federated Learning.* NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/2788b4cdf421e03650868cc4184bfed8-Paper-Conference.pdf
- <a id="crosssilo-align"></a>[crosssilo-align] *Cross-Silo Federated Learning Across Divergent Domains with Iterative Parameter Alignment.* arXiv:2311.04818, 2024. https://arxiv.org/html/2311.04818v4
- <a id="fl-healthcare-review"></a>[fl-healthcare-review] *From challenges and pitfalls to recommendations and opportunities: Implementing federated learning in healthcare.* Medical Image Analysis, 2025. https://www.sciencedirect.com/science/article/pii/S1361841525000453
- <a id="unit-conversions"></a>[unit-conversions] `registry_alignement/mappings/unit_conversions.py` — Geneva/Shenzhen unit conversion map (D-dimer, etc.).

[flower-blog-2024]: #flower-blog-2024
[flower-blog-bagging]: #flower-blog-bagging
[flower-comprehensive]: #flower-comprehensive
[secureboostplus]: #secureboostplus
[nvflare-secure-xgb]: #nvflare-secure-xgb
[nvflare-guide]: #nvflare-guide
[dp-xgboost]: #dp-xgboost
[fed-dp-boost]: #fed-dp-boost
[salvia]: #salvia
[flower-secagg-example]: #flower-secagg-example
[flower-dp-howto]: #flower-dp-howto
[flower-dp-explanation]: #flower-dp-explanation
[timberstrike]: #timberstrike
[fl-open-problems]: #fl-open-problems
[crosssilo-dp]: #crosssilo-dp
[crosssilo-align]: #crosssilo-align
[fl-healthcare-review]: #fl-healthcare-review
[unit-conversions]: #unit-conversions
