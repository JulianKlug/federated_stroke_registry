# Literature review — federated XGBoost with Flower

Scope: horizontal, cross-silo federated learning of a gradient-boosted trees
(GBDT) classifier on the joint Geneva + Shenzhen stroke registry (two sites,
tabular EHR features, shared schema after alignment). Focus is on
Flower's `flwr-xgboost` stack, its two aggregation strategies, and the
privacy toolbox that surrounds it. Each subsection ends with the concrete
implication for our two-site setup.

## 1. Why XGBoost is a strong baseline for tabular clinical FL

Boosted trees remain state of the art for heterogeneous tabular EHR data,
outperform deep tabular models on most benchmarks, natively handle missing
values, and are far cheaper to communicate than dense neural weights. The
Flower authors motivate `flwr-xgboost` on exactly this observation ([Flower
blog, 2024][flower-blog-2024]).

Practical implication: for this project we do not need to invent a neural
tabular architecture; a federated GBDT on the aligned Geneva+Shenzhen
features is the right first target.

## 2. Flower's two horizontal aggregation strategies

Flower ships two ready-made server strategies ([Flower docs, XGBoost
quickstart][flower-quickstart]; [Flower docs, comprehensive
example][flower-comprehensive]):

**Bagging — `FedXgbBagging`.** In each round every client boosts a small
number of local trees on its own data and the server concatenates them
into the global ensemble. Client trees at round *t* are all trained on the
same global ensemble from round *t-1*. Introduced in the Flower blog on
bagging aggregation ([Flower blog, 2023][flower-blog-bagging]) and
promoted as the default in the 2024 write-up ([Flower blog,
2024][flower-blog-2024]).

**Cyclic — `FedXgbCyclic`.** Only one client boosts per round; the updated
model is passed to the next client in the next round, round-robin. Useful
when clients are highly heterogeneous or when we want the second site to
literally "refine" what the first learned.

Empirical AUC comparisons on HIGGS in the Flower comprehensive example
show both strategies converge to comparable AUC over 20–50 rounds; bagging
is generally more stable when clients are balanced ([Flower docs,
comprehensive example][flower-comprehensive]).

Practical implication: with **two roughly balanced sites** and IID-ish
features after alignment, bagging is the safer default. Cyclic is a
useful ablation if one site turns out to dominate in sample size or
signal.

## 3. FedXGBllr — gradient-less federated boosting

Ma et al. ([Gradient-less Federated Gradient Boosting Trees with Learnable
Learning Rates, 2023][fedxgbllr]) propose FedXGBllr: each client trains a
full local XGBoost model, ships only the trees (never gradients/hessians),
and the server learns per-tree learning rates via a tiny 1-D CNN. The
communication overhead is 25×–700× lower than gradient-sharing baselines,
and gradients — which are known to leak features and labels — are never
transmitted. Released as a Flower baseline ([Flower baseline
`hfedxgboost`][flower-baseline-hfedxgboost]).

Practical implication: FedXGBllr is an attractive fall-back if
gradient-sharing turns out to leak too much (see §7) or if we later add
more sites and communication becomes a bottleneck.

## 4. SecureBoost and vertical FL — why it is not our setting

SecureBoost and SecureBoost+ ([Cheng et al., SecureBoost+,
2021/2024][secureboostplus]) target **vertical** FL: parties hold
different columns for the same patients, one active party holds labels,
and gradients are sent under Paillier homomorphic encryption. NVIDIA
FLARE's Secure Federated XGBoost extends this with CUDA-accelerated
homomorphic encryption ([Wang et al., 2025][nvflare-secure-xgb]).

Our two sites hold **different patients with the same schema** — this is
horizontal FL, so SecureBoost's split-finding protocol does not apply.
However, the HE machinery in NVIDIA FLARE also protects horizontal
histogram aggregation and is currently the strongest off-the-shelf option
if we later decide gradient statistics themselves need encryption in
transit ([NVIDIA developer blog, 2024][nvidia-he-blog]; [NVFlare Secure
XGBoost User Guide][nvflare-guide]).

## 5. Differential privacy for federated trees

DP-XGBoost ([Grislain & Gonzalvez, 2021][dp-xgboost]) adapts XGBoost's
histogram-based split finding to the (ε, δ)-DP model: it privatizes both
the histogram used to score splits and the leaf values, and shows that
default XGBoost hyperparameters (`max_depth=6`, `n_estimators=100`) burn
privacy budget very fast — shallower trees and fewer rounds are strongly
preferred under DP.

Maddock et al. ([Federated Boosted Decision Trees with DP,
2022][fed-dp-boost]) build DP-TR/DP-XGB in a federated setting and reach
the same conclusion: **shallow trees + few rounds** are the design lever
that keeps utility acceptable under DP. PrivBoost ([Han et al.,
2025][privboost]) extends this to a full FL framework with per-tree DP
accounting.

**Flower's built-in DP support — what it does and does not cover.**
Flower ships four DP strategy wrappers —
`DifferentialPrivacyClientSideFixedClipping`,
`DifferentialPrivacyClientSideAdaptiveClipping`, and their server-side
twins ([Flower DP how-to][flower-dp-howto];
[Flower DP explanation][flower-dp-explanation]). These wrap a base
strategy (typically FedAvg) and add per-client clipping + Gaussian noise
on **model-parameter updates**. That is exactly the right primitive for
neural-network FL, but it does not fit `FedXgbBagging`: the aggregation
step there is tree concatenation, not weight averaging, and the leak
surface is the histogram used for split scoring, not the tree
parameters after the fact. Consequently, a DP-XGBoost implementation on
top of Flower has to inject noise on the **client side, at histogram
construction**, before boosting — as [DP-XGBoost][dp-xgboost] and
[Maddock et al.][fed-dp-boost] describe — rather than by wrapping the
server strategy.

Practical implication: if we adopt DP, we should not port centralized
XGBoost defaults. Start with `max_depth ∈ {3,4}`, 20–40 rounds total, and
budget ε per round. And expect to write the client-side histogram
noise ourselves — Flower's DP wrappers are for a different aggregation
shape.

## 6. Secure aggregation in Flower

Flower supports SecAgg / SecAgg+ via Salvia ([Li et al., Secure Aggregation
for FL in Flower, 2022][salvia]) — a masking protocol against
honest-but-curious servers, robust to client dropout. There is a
first-party example ([Flower secure aggregation
example][flower-secagg-example]).

Practical implication: with only two sites, classic SecAgg is
low-value (a two-party masking scheme reveals your input to the other
party). Its main utility here would be if we later add a third site or if
we introduce a coordinator/aggregator that we do not want to see the raw
tree statistics.

## 7. Attacks: horizontal federated trees leak

Di Gennaro et al. ([TimberStrike, PoPETs 2025][timberstrike]) show an
optimization-based **dataset reconstruction attack** carried out by a
single honest-but-curious client against horizontally federated GBDT.
Evaluated against Flower, NVFlare, and FedTree — the three frameworks we
would realistically pick from — the attack reconstructs **73.05% to
95.63%** of a target client's records. Crucially, the paper evaluates on
a **publicly available stroke prediction dataset**, which is essentially
our threat surface.

Key takeaways from TimberStrike:

- Split values and decision paths embedded in shared trees are the leak.
  The threat exists even without gradient sharing.
- DP mitigates the attack **but degrades utility significantly** at
  privacy budgets that actually block reconstruction.
- The authors explicitly call out that tree-specific privacy defences are
  underdeveloped compared to the neural-network literature.
- **HE does not mitigate TimberStrike.** The paper evaluates the attack
  against NVFlare — which uses Paillier HE for histogram aggregation —
  and still reaches 73–95% reconstruction. HE encrypts intermediate
  transport and aggregation; TimberStrike operates on the *final tree
  ensemble*, which every client must hold in plaintext to run
  predictions. HE and DP defend orthogonal surfaces (HE: aggregator /
  transport; DP: the trees themselves).

Practical implication: this is the most important finding for our
project. We cannot treat "no raw data leaves the site" as sufficient
privacy for a two-site GBDT. Every design choice below is scored against
its exposure to a TimberStrike-style attack — and specifically, HE is
not part of that scoring because it addresses a different threat
surface.

## 8. Cross-silo FL with very few clients

Cross-silo FL is defined as ≥2 always-available institutional clients
with large local datasets ([Kairouz et al., "Advances and Open Problems in
FL"][fl-open-problems]; also see recent cross-silo DP work like [Liu et
al., NeurIPS 2022][crosssilo-dp]). Practical guides ([FedEff, 2025][fedeff];
[Cross-Silo FL with Iterative Parameter Alignment, 2024][crosssilo-align])
consistently note:

- Round counts are much smaller than cross-device FL (tens, not
  thousands).
- Local computation per round can be substantial (large batches, more
  local iterations) because each site has the compute.
- Non-IID is the dominant failure mode; site-specific calibration and
  personalization layers help.

Practical implication: with two sites we expect to run on the order of
**20–50 server rounds**, do meaningful local work each round, and pay
close attention to Geneva/Shenzhen distribution shift (see §9).

## 9. Distribution shift, Geneva vs. Shenzhen

Cross-silo FL in healthcare consistently reports that raw pooled training
is often outperformed by strategies that acknowledge site heterogeneity:
personalization layers, site as a covariate, or per-site calibration
heads ([Cross-Silo FL with Iterative Parameter Alignment, 2024][crosssilo-align];
review by [Rieke et al. / follow-ups in Nature Digital Medicine and
Medical Image Analysis 2025][fl-healthcare-review]). Our own
`registry_alignement` work is already addressing schema and unit
differences (e.g. D-dimer unit conversion, ODT/ONT/DNT/DPT/OPT harmonization),
which is a prerequisite to any FL run.

Practical implication: unit- and definition-alignment must be locked
before any federated run, and we should log a site indicator in
evaluation (not necessarily as a feature) to detect performance gaps.

## 10. Summary of what the literature tells us

1. Both `FedXgbBagging` and `FedXgbCyclic` are viable and neither
   dominates in the published record. With imbalanced site cohorts,
   run both and decide empirically on site-stratified metrics; cyclic
   often benefits imbalanced settings, bagging balanced ones.
2. Keep trees shallow (`max_depth` 4–6) and total rounds modest (20–50).
3. Do not rely on "no raw data leaves the site" for privacy — the
   TimberStrike attack directly targets our exact setup on a stroke
   dataset. Cohort-size and tree-shape hygiene (`min_child_weight`,
   shallow depth, subsampling) are the free first line; DP is the
   load-bearing defence.
4. HE and DP defend **orthogonal** surfaces. HE protects
   aggregator/transport; DP protects the trees themselves. TimberStrike
   penetrates HE-based frameworks (evaluated on NVFlare), so HE is not
   a substitute for DP.
5. Flower's built-in DP wrappers are FedAvg-shaped and do not directly
   apply to `FedXgbBagging`; client-side histogram noise is the correct
   plug-point.
6. Alignment (units, definitions, missingness conventions) is a
   pre-condition, not a co-project.

---

## References

- <a id="flower-blog-2024"></a>[flower-blog-2024] Flower Labs. *Federated XGBoost: Flower is all you need.* Flower blog, 2024-02-14. https://flower.ai/blog/2024-02-14-federated-xgboost-with-flower/
- <a id="flower-blog-bagging"></a>[flower-blog-bagging] Flower Labs. *Federated XGBoost with bagging aggregation.* Flower blog, 2023-11-29. https://flower.ai/blog/2023-11-29-federated-xgboost-with-bagging-aggregation/
- <a id="flower-quickstart"></a>[flower-quickstart] *Quickstart XGBoost.* Flower Framework docs. https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
- <a id="flower-comprehensive"></a>[flower-comprehensive] *Federated Learning with XGBoost and Flower (Comprehensive Example).* Flower Examples 1.29. https://flower.ai/docs/examples/xgboost-comprehensive.html
- <a id="fedxgbllr"></a>[fedxgbllr] Ma, C., et al. *Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates.* arXiv:2304.07537 (2023). https://arxiv.org/pdf/2304.07537
- <a id="flower-baseline-hfedxgboost"></a>[flower-baseline-hfedxgboost] Flower Labs. *Baseline: `hfedxgboost` (FedXGBllr).* https://flower.ai/docs/baselines/hfedxgboost.html
- <a id="secureboostplus"></a>[secureboostplus] Chen, W., et al. *SecureBoost+: A High Performance Gradient Boosting Tree Framework for Large Scale Vertical FL.* arXiv:2110.10927. https://arxiv.org/html/2110.10927v5
- <a id="nvflare-secure-xgb"></a>[nvflare-secure-xgb] Wang, Y., et al. *Secure Federated XGBoost with CUDA-accelerated Homomorphic Encryption via NVIDIA FLARE.* arXiv:2504.03909 (2025). https://arxiv.org/pdf/2504.03909
- <a id="nvidia-he-blog"></a>[nvidia-he-blog] NVIDIA Developer Blog. *Security for Data Privacy in Federated Learning with CUDA-Accelerated Homomorphic Encryption in XGBoost.* 2024. https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/
- <a id="nvflare-guide"></a>[nvflare-guide] *NVFlare XGBoost User Guide (Secure XGBoost).* NVIDIA FLARE 2.6 docs. https://nvflare.readthedocs.io/en/2.6/user_guide/federated_xgboost/secure_xgboost_user_guide.html
- <a id="dp-xgboost"></a>[dp-xgboost] Grislain, N., Gonzalvez, J. *DP-XGBoost: Private Machine Learning at Scale.* arXiv:2110.12770 (2021). https://arxiv.org/pdf/2110.12770
- <a id="fed-dp-boost"></a>[fed-dp-boost] Maddock, S., Cormode, G., Wang, T., Maple, C., Jha, S. *Federated Boosted Decision Trees with Differential Privacy.* CCS 2022. arXiv:2210.02910. https://arxiv.org/pdf/2210.02910
- <a id="privboost"></a>[privboost] Han, R., et al. *PrivBoost: A federated learning framework for differentially private tree boosting.* Computer Networks, 2025. https://www.sciencedirect.com/science/article/abs/pii/S1389128625007832
- <a id="salvia"></a>[salvia] Li, K. H., Cormode, G., et al. *Secure Aggregation for Federated Learning in Flower (Salvia).* DistributedML 2022; arXiv:2205.06117. https://arxiv.org/pdf/2205.06117
- <a id="flower-secagg-example"></a>[flower-secagg-example] *Secure aggregation with Flower (SecAgg+ protocol).* Flower Examples 1.29. https://flower.ai/docs/examples/flower-secure-aggregation.html
- <a id="flower-dp-howto"></a>[flower-dp-howto] *Use Differential Privacy — Flower Framework how-to.* https://flower.ai/docs/framework/how-to-use-differential-privacy.html
- <a id="flower-dp-explanation"></a>[flower-dp-explanation] *Differential Privacy — Flower Framework explanation.* https://flower.ai/docs/framework/explanation-differential-privacy.html
- <a id="timberstrike"></a>[timberstrike] Di Gennaro, M., De Lucia, G., Longari, S., Zanero, S., Carminati, M. *TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems.* PoPETs 2025(4). arXiv:2506.07605. https://petsymposium.org/popets/2025/popets-2025-0145.pdf
- <a id="fl-open-problems"></a>[fl-open-problems] Kairouz, P., et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in ML, 2021. https://arxiv.org/pdf/1912.04977
- <a id="crosssilo-dp"></a>[crosssilo-dp] Liu, Z., et al. *On Privacy and Personalization in Cross-Silo Federated Learning.* NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/2788b4cdf421e03650868cc4184bfed8-Paper-Conference.pdf
- <a id="fedeff"></a>[fedeff] *FedEff: Efficient federated learning with optimal local epochs for heterogeneous clients.* 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12592536/
- <a id="crosssilo-align"></a>[crosssilo-align] *Cross-Silo Federated Learning Across Divergent Domains with Iterative Parameter Alignment.* arXiv:2311.04818, 2024. https://arxiv.org/html/2311.04818v4
- <a id="fl-healthcare-review"></a>[fl-healthcare-review] *From challenges and pitfalls to recommendations and opportunities: Implementing federated learning in healthcare.* Medical Image Analysis, 2025. https://www.sciencedirect.com/science/article/pii/S1361841525000453

[flower-blog-2024]: #flower-blog-2024
[flower-blog-bagging]: #flower-blog-bagging
[flower-quickstart]: #flower-quickstart
[flower-comprehensive]: #flower-comprehensive
[fedxgbllr]: #fedxgbllr
[flower-baseline-hfedxgboost]: #flower-baseline-hfedxgboost
[secureboostplus]: #secureboostplus
[nvflare-secure-xgb]: #nvflare-secure-xgb
[nvidia-he-blog]: #nvidia-he-blog
[nvflare-guide]: #nvflare-guide
[dp-xgboost]: #dp-xgboost
[fed-dp-boost]: #fed-dp-boost
[privboost]: #privboost
[salvia]: #salvia
[flower-secagg-example]: #flower-secagg-example
[flower-dp-howto]: #flower-dp-howto
[flower-dp-explanation]: #flower-dp-explanation
[timberstrike]: #timberstrike
[fl-open-problems]: #fl-open-problems
[crosssilo-dp]: #crosssilo-dp
[fedeff]: #fedeff
[crosssilo-align]: #crosssilo-align
[fl-healthcare-review]: #fl-healthcare-review
