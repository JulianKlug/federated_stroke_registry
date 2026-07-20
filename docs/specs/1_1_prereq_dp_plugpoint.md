# Spec 1.1 (prerequisite) — DP plug-point prototype on single-site synthetic

Implements the **Phase v1.1/v1.2 prerequisite** ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Prerequisite — DP plug-point prototype on single-site synthetic. Custom client-side
> DP-XGBoost-style Laplace/Gaussian noise on gradient/hessian histograms before aggregation,
> plus geometric clipping of leaf outputs to bound sensitivity (architecture §6.3). Flower's
> built-in DP wrappers do not apply — they target weight averaging in FedAvg, not tree-structure
> aggregation. Includes DP accounting verification: on a fixed toy setting (known rounds, known
> noise scale), compute total ε with the accountant and compare to Opacus's `RDPAccountant` or
> TensorFlow Privacy's `compute_rdp` on the same setting. Numbers must match within
> floating-point tolerance before any downstream sweep uses this accountant.

Grounded in the architecture doc
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md),
§6.3) and the literature review
([docs/automated_review/literature_review_federated_xgboost.md](../automated_review/literature_review_federated_xgboost.md),
§5), and the **installed** framework (`flwr==1.31.0` target, `xgboost>=2.0`, Python `>=3.12`,
`uv`-managed). It reuses the client training seam built by 1.b
([1b_fedxgb_bagging_cyclic.md](1b_fedxgb_bagging_cyclic.md)) and the single scorer built by 1.c
([1c_evaluation_harness.md](1c_evaluation_harness.md)); it mirrors the pure-functions-plus-thin-CLI
shape of 1.d's `baseline.py` + `check_fed_vs_pooled.py`
([1d_federated_vs_pooled.md](1d_federated_vs_pooled.md)).

**This task ships the spec only.** The roadmap prerequisite checkbox stays `[ ]` until the code
lands in a follow-up task — mirroring how 1.e/1.f shipped their spec ahead of implementation.

**Framing (why this exists — read before §1).** DP is *not* what makes the two-center
collaboration function; the DPA/IRB governance layer is (§6.4 of the architecture doc). DP has
exactly one job here: bound how much a curious **peer institution**, or a **downstream reader of a
published model**, can reconstruct about individual patients from the tree ensemble and the
histograms that build it — the **TimberStrike** surface (lit review §7; the attack is demonstrated
*on Flower*, our stack). HE and SecAgg do not mitigate it at N=2 (§6.5/§6.6). Whether that threat is
in-scope is a legal/publication decision, not a technical one, and DP carries a real utility cost at
our small per-site N (~380) and current 2-feature schema. **So this prototype is deliberately
cheap: its purpose is to become the instrument that *measures* the DP-utility cost** — first on
synthetic here, then on real Geneva in 1.1.d — so the keep-or-drop-DP decision is made on evidence.
It is not a commitment to ship DP.

## 1. Goal & scope

The prerequisite answers two questions, in isolation, before DP touches real data or the live
federation:

1. **Does the DP mechanism work end-to-end?** A client-side DP-XGBoost-style learner — Gaussian
   (primary) / Laplace (secondary) noise on the gradient/hessian split histograms, plus geometric
   clipping of leaf outputs — trains on a single synthetic site and produces a sane AUC that erodes
   sensibly as ε shrinks.
2. **Is the ε-accountant trustworthy?** On a fixed toy setting (known queries, known noise scale),
   our accountant's total ε **matches Opacus's `RDPAccountant` within floating-point tolerance**.
   This is the roadmap's hard gate: *numbers must match before any downstream sweep uses the
   accountant.*

**Why a self-contained NumPy learner (not a hook into XGBoost).** Stock XGBoost's `hist` builder
computes the gradient/hessian histograms in C++ and never exposes them to Python — there is no
callback, and `bst.update` is opaque (verified against the `_train_round`/`_local_boost` seam in
[client_app.py:29-80](../../architecture/fed_stroke/client_app.py)). "Add noise to the histograms"
therefore cannot be done by configuring stock `xgboost`. Both the architecture doc (§6.3: "a custom
plug-point on the client") and the literature review (§5: "expect to write the client-side histogram
noise ourselves") already concluded this. The prototype is a **parallel, minimal-but-real histogram
gradient-boosting learner in NumPy** that we fully own — it becomes the executable reference for the
eventual production integration.

**In scope**

- A **NumPy DP-GBDT learner** (`dp_boost.py`): fixed-bin gradient/hessian histograms, Gaussian
  (RDP-accounted) and Laplace (basic-composition) noise on the split histograms, noised-gain split
  selection, and **geometric leaf clipping** (`|w| ≤ clip_bound`) to bound each leaf's contribution
  (a post-processing utility guard, **not** a privacy knob — §3.8).
  Shallow trees + few rounds — the DP-friendly regime (lit review §5). Noise off (ε=∞) yields an
  ordinary GBDT.
- An **own lightweight RDP accountant** (`dp_accounting.py`, numpy/scipy only) — the Gaussian
  mechanism under Rényi-DP composition, converted to (ε, δ) — plus an inverse `σ`-for-target-ε
  calibrator the mechanism uses. Kept torch-free so the federated client runtime never pulls a
  heavyweight DP library.
- The **accountant equivalence gate**: a test that reproduces Opacus's per-step RDP, order grid,
  and RDP→(ε,δ) conversion and asserts our ε equals Opacus's within tolerance on a fixed toy matrix.
  **Opacus is a dev/test-only dependency.**
- A **single-site synthetic generator** (`synthetic.py`): a controllable logistic-linear ground
  truth at real per-site scale (n≈380) over the frozen 2-feature schema, with a stratified split so
  neither train nor valid is single-class.
- A **three-arm sanity harness** (`dp_prototype_demo.py`) scoring, through the *same*
  `metrics.compute_binary_metrics`: **(A)** classic stock `xgb.train`, **(B)** our learner with
  noise OFF, **(C)** our learner with DP ON at ε ∈ {1, 3, 5}. The three arms make the DP-vs-classic
  gap **decompose**: A→B = cost of our simpler learner vs stock XGBoost; B→C = cost of privacy.
- The **plug-point seam**: the exact symbols (`DPConfig`, the `HistogramNoiseMechanism` protocol,
  `train_dp_gbdt`) a future federated client will import, and DP config keys under a `dp.*` table in
  `pyproject.toml`.
- Unit tests for the sensitivity bounds, the accountant (incl. the Opacus gate + inverse
  round-trip), the learner (determinism, no-noise≈classic, ε-monotonicity), and the generator.

**Out of scope** (named + owned by a later roadmap item — never silently dropped)

- **Wiring DP into the live federation** — the `client_app._train_round` DP branch, `DPBooster`
  serialization into the `ArrayRecord`, and a DP-aware server aggregator. The seam is *designed*
  here (§4.5) but *not built*. → a downstream Phase v1.1 integration item, gated on this prototype.
- **The ε ∈ {1, 3, 5, 10}, δ=1e-5 DP sweep and the DP-vs-classic HPO on real Geneva** → **1.1.b**
  (sweep) / **1.1.d** (Geneva-only DP-utility curve) / **1.1.e** (reporting skeleton). This spec
  produces only a synthetic sanity look; the classic-XGB anchor here is the seed of that head-to-head
  comparison (§4.4, §8.7).
- **Differentially-private bin edges on real data.** The prototype's DP arm uses fixed *public
  clinical ranges* (data-independent, DP-safe). A private-quantile mechanism (or a public reference
  grid) for real Geneva feature distributions → downstream, before 1.1.b runs DP on real data (§8.1).
- **Privacy-amplification credit from subsampling.** The prototype forces `subsample=1.0` in DP mode
  and reports ε at sampling rate q=1.0 (an honest upper bound). Honest amplification requires genuine
  Poisson record sampling + the subsampled-RDP bound → downstream (§8.2, §8.6).
- **A hardened accountant** (tight subsampled amplification in production, adaptive order grids).
  The prototype accountant is real and Opacus-validated at q=1.0, sufficient to produce defensible ε
  for a synthetic sanity run.

## 2. Dependencies

- **1.b/1.c complete** (they are): the client training seam
  ([client_app.py](../../architecture/fed_stroke/client_app.py)) and the single scorer
  `metrics.compute_binary_metrics`
  ([metrics.py:70](../../architecture/fed_stroke/metrics.py)) exist and are the shapes this spec
  reuses/extends.
- **Frozen schema** ([schema.py](../../architecture/fed_stroke/schema.py)): `FEATURE_COLS =
  ['Age (calc.)', 'NIH on admission']`, `TARGET_COL = '3M Death'`. The generator and bin ranges are
  keyed to these two columns and stay correct as the frozen set grows (§8.4).
- **New runtime dependencies:** `numpy>=1.26` and `scipy>=1.11` become **explicit** in
  `[project].dependencies` (today only transitive via xgboost/pandas/scikit-learn). `dp_accounting.py`
  imports both directly.
- **New dev/test-only dependency:** `opacus>=1.5,<2` in `[dependency-groups].dev` (today
  `["pytest>=8"]`). Opacus pulls `torch` (~heavy) and is the equivalence-gate reference **only** —
  it MUST NOT enter `[project].dependencies` or the Shenzhen container (§8.8).

## 3. Framework / DP facts that drive the design

These are the load-bearing facts; the design in §4 follows from them. Each is stated precisely
because a wrong constant or a miscounted composition silently corrupts the reported ε.

- **3.1 Stock `hist` histograms are unreachable from Python.** `bst.update(dmatrix, ...)`
  ([client_app.py:32](../../architecture/fed_stroke/client_app.py)) runs the split-finding
  histogram build in C++; there is no Python hook and no callback exposing per-bin gradient/hessian
  sums. Hence the DP mechanism is a parallel NumPy learner, not an interception of XGBoost (§1, §4.2).
  In DP mode the future client **swaps the whole learner**, it does not noise `bst.update` (§4.5).

- **3.2 For `binary:logistic`, per-example gradient and hessian are *naturally bounded* — that
  bound IS the sensitivity, no gradient clipping needed.** With raw margin `F_i`, `p_i = σ(F_i)`:
  - gradient `g_i = p_i − y_i ∈ [−1, 1]` (since `p_i ∈ (0,1)`, `y_i ∈ {0,1}`),
  - hessian `h_i = p_i(1−p_i) ∈ (0, 0.25]` (max at `p_i = 0.5`).

  A per-feature histogram over `B` bins accumulates `G_b = Σ_{i∈b} g_i` and `H_b = Σ_{i∈b} h_i`.
  Adding/removing one record changes **exactly one bin of one feature** (the bin its value falls in),
  so the change vector has a single nonzero entry ⇒ L1 = L2 sensitivity **per feature**:
  `Δ(G) = max|g| = 1.0`, `Δ(H) = max h = 0.25`. Across `d` features released together at one node a
  record touches one bin *per feature* ⇒ concatenated L2 sensitivity `√d · 1.0` (G) and
  `√d · 0.25` (H). With today's `d = 2`: `√2` and `0.25√2`. Constants in code:
  `G_L2_PER_FEATURE = 1.0`, `H_L2_PER_FEATURE = 0.25`. This is why DP-GBDT needs no per-example
  gradient clipping (unlike DP-SGD).

  **Both histograms are released, and that is TWO Gaussian mechanisms, not one (drives §3.3's release
  count).** At each split-finding node the learner noises the gradient histogram `G` *and* the
  hessian histogram `H` — with *different* per-stat noise scales (`σ·√d·1.0` on G, `σ·√d·0.25` on H,
  §4.2), because H lives on a 4× smaller scale and a single isotropic noise would swamp it. Two
  independent Gaussian mechanisms, each at multiplier `σ`, compose: their joint RDP is `2·α/(2σ²) =
  α/σ²` per node-level, **not** `α/(2σ²)`. Whitening confirms it: a ±1-record change moves G by L2
  `√d·1.0` and H by L2 `√d·0.25`, so `‖ΔG/(σ√d·1.0)‖² + ‖ΔH/(σ√d·0.25)‖² = 1/σ² + 1/σ² = 2/σ²`,
  giving Gaussian RDP `α·(2/σ²)/2 = α/σ²`. Charging one release per level (as a naive reading would)
  under-reports ε by ~2×. Counted correctly in §3.3.

  **L1 sensitivity for the Laplace arm (L1, not L2).** The Laplace mechanism calibrates to L1, not
  L2. A record touches one bin per feature, so the L1 change across `d` features is `d·s` (L1 sums
  the per-feature entries; it does **not** take a root). Constants:
  `G_L1_PER_FEATURE = 1.0`, `H_L1_PER_FEATURE = 0.25`; per-release L1 is `d·1.0` (G) and `d·0.25`
  (H). Using `√d` here (an L2 factor) would under-noise the Laplace arm and understate its ε (§4.2,
  §8.2). The Laplace arm noises G and H too ⇒ it is likewise `2·D·T` releases, not `D·T`.

- **3.3 Composition is PER TREE LEVEL (not per node), and each level is TWO Gaussian releases (G and
  H).** Two independent counts, both load-bearing:

  1. **Levels, not nodes (the classic DP-GBDT error).** A record flows to **exactly one node per
     depth level** of each tree. Sibling nodes at a level partition the records ⇒ releasing all their
     histograms costs the RDP of a *single* query (**parallel composition**), not the sum over nodes.
     Only *depth levels*, *trees*, and *rounds* compose sequentially:

     ```
     num_histogram_queries  = D × T          # split-finding LEVELS; NOT (number of nodes)
     ```

     Counting nodes (`2^D − 1` per tree — 15 at D=4) instead of levels (`D` — 4 at D=4)
     over-charges by `(2^D − 1)/D ≈ 3.75×` at depth 4 (harmless direction, but wrong).

  2. **Two statistics per level ⇒ double the Gaussian releases (§3.2).** Each level noises the
     gradient AND the hessian histogram, each its own Gaussian mechanism, so the accountant is fed:

     ```
     num_gaussian_releases  = 2 × num_histogram_queries = 2 × D × T
     ```

     Counting one release per level under-charges ~2×. Getting *either* count wrong makes every
     reported ε meaningless — hence both are pinned by tests (§4.7).

  With the committed federated config (`total-trees=40`, `num-sites=2`, `max-depth=4`,
  `local-epochs=1`), each site grows `trees_per_site = 20`, so `D×T = 4 × 20 = 80` levels ⇒
  **`num_gaussian_releases = 160`** per site. (The synthetic demo defaults to `max-depth=3`,
  `rounds=20` ⇒ `60` levels ⇒ **`120`** releases — §4.4.) Split-argmax on the noised gain and
  leaf-weight computation from the already-noised node totals are **post-processing** of the noised
  release ⇒ they cost **zero** additional budget (report-noisy-max, §3.8). Pinned by the module
  docstring and tests (§4.7).

- **3.4 The accountant must reproduce Opacus *exactly* on three internals or the gate fails.**
  (i) per-step Gaussian RDP `ε_RDP(α) = α/(2σ²)` at q=1.0 (Opacus special-cases q==1 to this closed
  form → machine-precision match); (ii) the **order grid** `DEFAULT_ORDERS = [1+x/10 for x in
  1..99] + [11..63]`, identical to `RDPAccountant.DEFAULT_ALPHAS`; (iii) the **improved
  Balle-et-al RDP→(ε,δ) conversion** `ε(α) = ρ(α) − (ln δ + ln α)/(α−1) + ln((α−1)/α)`, min over the
  grid — **not** the classic Mironov `ρ(α) + log(1/δ)/(α−1)`, which is looser and produces a
  systematic offset that fails the tolerance. (§4.1, §8 risk #1–2.)

- **3.5 XGBoost `subsample=0.8` is without-replacement, once per tree — NOT Poisson per-record.**
  The subsampled-Gaussian amplification bound assumes Poisson inclusion. Claiming amplification credit
  for without-replacement sampling is not sound. So the prototype forces `subsample=1.0` in DP mode
  and accounts at q=1.0 (no credit) — the reported ε is a clean k-fold Gaussian composition and an
  honest upper bound. `subsample` stays an XGBoost *utility* knob for the non-DP arms only (§8.2, §8.6).

- **3.6 Config reaches code as `params.*` fed verbatim to `xgb.train`.** `context.run_config` (flat,
  dashed) → `unflatten_dict` → `task.replace_keys` (`-`→`_`) → `cfg["params"]`
  ([client_app.py:101-102](../../architecture/fed_stroke/client_app.py)). XGBoost rejects unknown
  params, so **DP knobs must live in a sibling `dp.*` table, never under `params.*`** (§4.3, §8.5).

- **3.7 Data-dependent bin edges are not DP.** Quantile edges computed from `X` leak the empirical
  feature distribution. The DP arm must use **fixed, data-independent** edges over public clinical
  ranges (Age ∈ [0,120], NIHSS ∈ [0,42]); quantile edges are permitted only for the non-DP arms
  (A/B) or a future private-quantile mechanism (§8.1).

- **3.8 Geometric leaf clipping is POST-PROCESSING here — a utility guard, NOT a privacy knob.** In
  *this* learner the only place noise enters is the histogram (§4.2); a leaf's `(G,H)` totals are
  summed from the *already-noised* parent histogram, never re-queried from the data. So the leaf
  weight `w = -η·G/(H+λ)` and its clip `w ← clip(w, −c, +c)` are post-processing of a DP release ⇒
  by post-processing immunity they cost **zero** budget and **`clip_bound` has no effect on ε**. Its
  job is utility/stability: bounding a single leaf's contribution so histogram noise cannot blow a
  weight up, and matching the bounded-output invariant a production release would want. The roadmap /
  architecture §6.3 phrase this as "clipping to bound sensitivity"; that wording comes from the
  *alternative* mechanism (lit review §5: "privatize both the histogram **and** the leaf values"),
  where leaf values are a **separate noised release** and clipping bounds *their* sensitivity. This
  prototype deliberately does **not** take that path (Decision 8, §9) — leaves stay post-processing.
  A test asserts reported ε is invariant to `clip_bound` (§4.7).

- **3.9 The reported ε is PER-SITE, which is the per-patient guarantee.** Each patient's records live
  at exactly one site and are touched only by that site's histograms, so the single-site ε computed
  here IS the guarantee an individual patient receives — there is no cross-site composition to add. A
  2-site federation gives each site its own independent ε at the same level; the prototype measures
  one site's, which is the number that matters for the patient.

## 4. Design

Pure functions in the package + thin CLIs in `scripts/`, matching `baseline.py` /
`check_fed_vs_pooled.py`. Two new package modules (`dp_accounting.py`, `dp_boost.py`), one generator
(`synthetic.py`), one demo script, three test files.

### 4.1 `architecture/fed_stroke/dp_accounting.py` — the accountant (the gate's subject)

numpy/scipy only; no torch/tf/flwr import. Sensitivity-agnostic: it works purely in
`noise_multiplier` (σ) units, so the gate never needs a sensitivity and the mechanism (§4.2) owns the
sensitivity→σ mapping.

```python
DEFAULT_ORDERS: tuple[float, ...]   # = tuple([1+x/10 for x in range(1,100)] + list(range(11,64)))
                                    #   MUST equal opacus RDPAccountant.DEFAULT_ALPHAS (§3.4)

def rdp_gaussian(alpha, noise_multiplier, sample_rate=1.0):
    """Per-step Rényi-DP ε at order(s) α for one (subsampled) Gaussian release.
    q == 1.0  -> alpha / (2 * noise_multiplier**2)   (exact; matches Opacus's special case)
    q  < 1.0  -> subsampled-Gaussian (Mironov-Wang) log_a bound, log-space (scipy.special
                 gammaln/betaln/log_ndtr). Behind the same param; the GATE rides q=1.0."""

def compose_rdp(rdp_per_step, num_steps):
    """Additive RDP composition of identical steps: total ρ(α) = num_steps * rdp_per_step.
    (Overload: accept [(rdp_vec, count), ...] and sum for heterogeneous steps.)"""

def rdp_to_epsilon(rdp, orders, delta) -> tuple[float, float]:
    """Improved Balle-et-al conversion (§3.4), min over orders -> (epsilon, best_order).
    Warn if best_order is the grid boundary (grid too narrow) — mirrors Opacus."""

def account_run(num_queries, noise_multiplier, sample_rate, delta,
                orders=DEFAULT_ORDERS) -> float:
    """Total ε for num_queries identical Gaussian releases: compose then convert."""

def noise_multiplier_for_epsilon(target_epsilon, num_queries, sample_rate, delta,
                                 orders=DEFAULT_ORDERS,
                                 sigma_bounds=(1e-3, 1e3), tol=1e-6) -> float:
    """Inverse calibration. ε is strictly decreasing in σ -> monotone bisection on
    account_run(...) == target_epsilon. Returns the smallest σ with ε <= target.
    Raises if target unreachable within sigma_bounds. Consumed by make_mechanism (§4.2)."""

def account_run_laplace(num_queries, l1_sensitivity, laplace_scale) -> float:
    """SECONDARY pure-ε basic composition for ONE homogeneous group of Laplace releases:
    ε = num_queries * l1_sensitivity / laplace_scale.
    `l1_sensitivity` is the per-release L1 = d * stat_L1 (L1 sums over features, NO root — §3.2);
    e.g. d*G_L1_PER_FEATURE for the gradient group. The G and H groups are accounted SEPARATELY and
    their ε summed (basic composition): ε_total = account_run_laplace(k, d*G_L1, b_G)
    + account_run_laplace(k, d*H_L1, b_H), with k = D*T. No δ, no RDP; NOT part of the gate."""
```

### 4.2 `architecture/fed_stroke/dp_boost.py` — mechanism, learner, seam

```python
# Sensitivity constants (§3.2). L2 drives the Gaussian arm, L1 the Laplace arm.
G_L2_PER_FEATURE = 1.0      # |p - y| <= 1              (Gaussian, L2)
H_L2_PER_FEATURE = 0.25     # p(1-p) <= 0.25            (Gaussian, L2)
G_L1_PER_FEATURE = 1.0      # per-feature L1 == per-feature magnitude (Laplace)
H_L1_PER_FEATURE = 0.25     # per-feature L1                          (Laplace)
DENOM_FLOOR = 1e-3          # min positive (H+λ) after noise, so gain never divides by <=0 (§3.3, §8.3)

@dataclass(frozen=True)
class DPConfig:
    enabled: bool = False
    mechanism: str = "gaussian"        # "gaussian" (RDP, primary) | "laplace" (secondary)
    target_epsilon: float = 5.0        # arch §6.3 starting budget
    delta: float = 1e-5
    clip_bound: float = 1.0            # geometric leaf clip: |w| <= clip_bound
    max_bins: int = 32
    bin_strategy: str = "fixed_range"  # "fixed_range" (DP-safe) | "quantile" (non-DP arms only)
    noise_multiplier: float | None = None   # if set, ε is REPORTED not targeted
    @classmethod
    def from_run_config(cls, cfg: dict) -> "DPConfig": ...   # maps cfg["dp"] (post unflatten+replace)

@dataclass(frozen=True)
class BoostParams:                     # XGBoost-aligned names so from_xgb_params is trivial
    max_depth: int = 3
    num_boost_round: int = 20
    eta: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: float = 5.0
    base_score: float = 0.5
    seed: int = 0
    @classmethod
    def from_xgb_params(cls, params: dict) -> "BoostParams": ...

def num_histogram_queries(boost: BoostParams) -> int:      # = max_depth * num_boost_round (§3.3, levels)
def num_gaussian_releases(boost: BoostParams) -> int:      # = 2 * num_histogram_queries (G AND H; §3.2/§3.3)

FEATURE_RANGES = {"Age (calc.)": (0.0, 120.0), "NIH on admission": (0.0, 42.0)}   # public, DP-safe
def fixed_bin_edges(feature_ranges, max_bins) -> list[np.ndarray]      # data-INDEPENDENT
def quantile_bin_edges(X, max_bins) -> list[np.ndarray]                # NON-DP arms only (§3.7)

class HistogramNoiseMechanism(Protocol):
    def add_noise(self, G: np.ndarray, H: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]: ...

def make_mechanism(dp: DPConfig, boost: BoostParams, num_features: int) -> HistogramNoiseMechanism:
    """Factory. dp.enabled=False -> identity (no-op, ε=∞). d = num_features.
    Gaussian (primary, per-stat noise kept — Decision 7, §9):
      n_rel = num_gaussian_releases(boost)   # = 2*D*T; G and H are TWO releases per level (§3.2/§3.3)
      σ = dp_accounting.noise_multiplier_for_epsilon(dp.target_epsilon, n_rel, 1.0, dp.delta)
          (unless dp.noise_multiplier is pinned, in which case ε is REPORTED via
           account_run(n_rel, σ, 1.0, dp.delta));
      per-feature noise std = σ * √d * G_L2_PER_FEATURE (G) and σ * √d * H_L2_PER_FEATURE (H).
      NOTE: feeding n_rel (not D*T) is load-bearing — the two per-stat mechanisms compose to α/σ²
      per level, so D*T would under-report ε ~2× (§3.2). The Opacus gate does NOT cover this; it is
      pinned by test_gaussian_releases_is_two_times_levels (§4.7).
    Laplace (secondary, L1 not L2 — §3.2): split dp.target_epsilon equally across n_rel=2*D*T
      releases; per-release ε = dp.target_epsilon / n_rel. Per-feature scales (basic composition):
        b_G = d * G_L1_PER_FEATURE / (dp.target_epsilon / n_rel) = n_rel * d * G_L1_PER_FEATURE / ε
        b_H = n_rel * d * H_L1_PER_FEATURE / dp.target_epsilon
      (Uses `d`, NOT `√d` — Laplace calibrates to L1; √d would understate ε, §8.2.)"""

def train_dp_gbdt(X, y, boost, dp, feature_ranges=None, mechanism=None, rng=None) -> "DPBooster":
    """Minimal histogram GBDT. Per node: bin rows -> accumulate (G,H) -> mechanism.add_noise
    (the ONLY place noise enters) -> pick split by argmax noised gain
    (½[G_L²/D(H_L) + G_R²/D(H_R) − (G_L+G_R)²/D(H_L+H_R)]) where D(h) = max(h + λ, DENOM_FLOOR)
    so a noise-driven negative hessian can never flip the gain sign or divide by <=0 (§3.3, §8.3);
    reject any candidate whose NOISED child hessian < min_child_weight -> leaf w = -eta·G/D(H)
    -> GEOMETRIC CLIP w = clip(w, -c, +c) (post-processing / utility guard; ZERO ε effect, §3.8).
    base_margin = logit(base_score) (constant, data-independent)."""

class DPBooster:                       # .trees, .base_margin, .predict_margin(X), .predict(X)->(0,1)
    ...
```

Gaussian is the RDP-accounted primary; Laplace is a cruder pure-ε sanity path (expected worse at
matched ε — itself a useful signal). The accountant lives in `dp_accounting.py` and is *imported*
here — one accountant, one gate.

### 4.3 Config — `architecture/pyproject.toml` `[tool.flwr.app.config]`

Add a `dp.*` sibling table (NEVER under `params.*`, §3.6). Existing `params.tree-method="hist"`,
`params.max-depth=4`, `params.min-child-weight=5`, `params.subsample=0.8` are untouched.

```toml
dp.enabled = false
dp.mechanism = "gaussian"        # "gaussian" (RDP, primary) | "laplace" (secondary)
dp.target-epsilon = 5.0          # arch §6.3 starting budget; sweep {1,3,5,10} is downstream (1.1.b)
dp.delta = 1e-5
dp.clip-bound = 1.0              # geometric leaf clip |w| <= clip_bound
dp.max-bins = 32
dp.bin-strategy = "fixed-range"  # "fixed-range" (DP-safe) | "quantile" (NOT DP; non-DP arms only)
```

Flow: `unflatten_dict` → `{"dp": {"target-epsilon": ...}}`; `replace_keys` → `dp.target_epsilon`,
`dp.clip_bound`, `dp.bin_strategy`, `dp.max_bins`; `DPConfig.from_run_config(cfg)` maps `cfg["dp"]`.
Comment that `dp.enabled=true` forces `subsample=1.0` in the DP learner (§3.5).

### 4.4 `architecture/scripts/dp_prototype_demo.py` — three-arm sanity harness

Thin CLI mirroring `eval_final_model.py` (sys.path insert for `fed_stroke`, argparse, table print,
no file writes by default). All three arms scored through **the same** `compute_binary_metrics` on
the same synthetic split so AUCs are directly comparable and the gap decomposes:

- **Arm A — classic anchor:** `xgb.train({objective, eta, max_depth, min_child_weight,
  tree_method:"hist", subsample:1.0, base_score:0.5, nthread:1}, DMatrix(X_tr,y_tr),
  num_boost_round)`; predict on valid; `compute_binary_metrics`. **`base_score=0.5` and `nthread=1`
  are pinned deliberately** (F5, §9): XGBoost 2.x otherwise auto-estimates `base_score` from the
  label mean, and multi-thread `hist` sums non-deterministically — either would inject a difference
  into A→B that is *not* "our learner", polluting the gap decomposition and the logbook table. Arm B
  reads the same values via `BoostParams` (`base_score=0.5`, single-threaded NumPy).
- **Arm B — our learner, noise OFF:** `train_dp_gbdt(..., dp=DPConfig(enabled=False))`.
- **Arm C — our learner, DP ON** at ε ∈ {1,3,5} (`--epsilons`): `DPConfig(enabled=True,
  target_epsilon=ε, ...)`.

**Gap decomposition (the user's HPO requirement, seeded here):** `(A classic XGB) → (B NumPy
no-noise) → (C DP)`. A→B isolates "our simpler learner vs stock XGBoost"; B→C isolates "cost of
privacy." The same three-arm decomposition is what 1.1.b/1.1.d/1.1.e will run on real Geneva as the
DP-vs-classic head-to-head (§8.7).

CLI: `--n --seed --epsilons 1 3 5 --max-depth 3 --rounds 20 --eta 0.1 --clip-bound 1.0
--max-bins 32 --mechanism gaussian --n-boot --boot-seed`. Table columns: `arm | epsilon |
noise_mult | levels | releases | auc_roc | auc_pr | brier | n | n_pos`, where `levels = D×T` and
`releases = 2×D×T` (§3.3) — printing both makes the 2× accounting visible in the output. With demo
defaults (`max-depth 3`, `rounds 20`): `levels = 60`, `releases = 120`. Print the derived
`noise_multiplier` per ε, and a one-line banner: DP arm uses fixed public-range bins (not data
quantiles) and forces `subsample=1.0` (§3.5).

### 4.5 The plug-point seam (designed here, wired downstream)

The seam is the three symbols `DPConfig`, `HistogramNoiseMechanism`/`train_dp_gbdt` in `dp_boost.py`.
Future FL wiring — **not built in this task** — is a branch in
[`client_app._train_round`](../../architecture/fed_stroke/client_app.py):

```python
cfg_dp = DPConfig.from_run_config(cfg)         # cfg already built in client_app.train
if cfg_dp.enabled:
    booster = train_dp_gbdt(X, y, BoostParams.from_xgb_params(params), cfg_dp,
                            feature_ranges=FEATURE_RANGES)     # swaps the whole learner (§3.1)
    # -> serialize DPBooster into the ArrayRecord; a DP-aware server aggregator merges it
else:
    ... existing xgb.train / _local_boost path ...
```

Because stock `hist` histograms are unreachable (§3.1), DP mode **replaces the learner** rather than
intercepting `bst.update`. That has two consequences owned by the downstream integration item, not
here: the `ArrayRecord` currently carries `bst.save_raw("json")`
([client_app.py:114-120](../../architecture/fed_stroke/client_app.py)) — a `DPBooster` needs its own
serialization — and the server strategy (`FedXgbBagging` / `OrderedFedXgbCyclic`) must learn to merge
`DPBooster`s (or the sites must exchange noised histograms per round). The prototype makes that a
*wiring* exercise, not a research one.

### 4.6 `architecture/fed_stroke/synthetic.py` — single-site generator

`FEATURE_RANGES` has ONE home — `dp_boost.py` (it is a DP-safety artifact, §3.7) — and `synthetic.py`
`imports` it, never redefines it (DRY). The generator draws from a *subset* of those public ranges
(Age∼U[40,90] ⊂ [0,120]; NIHSS∼U[0,30] ⊂ [0,42]); that gap is **intentional** — real values sit
inside the public clinical range — so do not "align" them.

```python
from fed_stroke.dp_boost import FEATURE_RANGES   # single home; do NOT redefine
def make_synthetic_site(n=380, coef=(0.05, 0.12), intercept=-3.0, noise=0.5, seed=0)
                        -> tuple[np.ndarray, np.ndarray]:
    """Age ~ U[40,90], NIHSS ~ U[0,30]; logit = intercept + coef.Age*(age-65)
    + coef.NIHSS*(nih-15) + noise*N(0,1); y ~ Bernoulli(σ(logit)). n=380 ≈ real per-site."""
def make_synthetic_frame(..., prefix="S", seed=0) -> pd.DataFrame:
    """Same signal as FEATURE_COLS + TARGET_COL + unique case_admission_id (f'{prefix}{i}_1'),
    so it round-trips through split_half / score_booster_on_half if --via-parquet is used."""
def synthetic_train_valid(X, y, test_size=0.2, seed=42, stratify=True):
    """Stratified split: BOTH classes guaranteed in train AND valid (no NaN AUC, §8.3)."""
```

Bernoulli (not a hard threshold) so Bayes AUC < 1 and DP has real headroom to erode. Mirrors the
`_two_feature_data` / `_make_half` idioms already in the tests.

### 4.7 Tests

**`architecture/tests/test_dp_accounting.py` — THE GATE** (plain-pytest idiom of `test_metrics.py`):
- `test_default_orders_match_opacus`: `DEFAULT_ORDERS == tuple(RDPAccountant.DEFAULT_ALPHAS)`.
- **Equivalence gate**, parametrized over σ ∈ {0.5,1.0,2.0} × steps ∈ {1,20,100} × q ∈ {1.0,0.8},
  δ=1e-5. Reference:
  ```python
  opacus = pytest.importorskip("opacus")
  from opacus.accountants import RDPAccountant
  acc = RDPAccountant()
  for _ in range(steps): acc.step(noise_multiplier=sigma, sample_rate=q)
  eps_ref = acc.get_epsilon(delta=delta)
  eps_ours = account_run(steps, sigma, q, delta)
  ```
  q=1.0 (**hard gate**): `eps_ours == pytest.approx(eps_ref, rel=1e-6, abs=1e-9)`. **q<1 decision
  (F6, pinned — do NOT leave to implementation):** the prototype forces q=1.0 in DP mode (§3.5), so
  the SGM bound is NOT required to match Opacus exactly. The q=0.8 case asserts only the conservative
  direction `eps_ours >= eps_ref` (never under-report), with a comment. The q=1.0 gate is the sole
  exact gate and is never loosened.
- **Inverse-calibration round-trip**, ε ∈ {1,3,5,10}, q=1.0, δ=1e-5:
  `account_run(m, noise_multiplier_for_epsilon(ε,m,1.0,1e-5), 1.0, 1e-5) ≈ ε` (rel 1e-4) and
  `≤ ε + 1e-6` (never over-spend); σ decreasing in ε. Run at `m ∈ {80, 160}` — `160` is the real
  federated Gaussian-release count (2×D×T at depth-4×20, §3.3); `80` exercises the level count too.
  (These are accountant *unit* inputs; the mechanism always feeds `num_gaussian_releases`.)
- **Closed-form pure tests** (no opacus): `rdp_gaussian(α,σ)==α/(2σ²)`; `account_run` monotone in σ
  and in k.
- **CI guard**: `if os.environ.get("CI"): assert importlib.util.find_spec("opacus")` — skip locally
  (importorskip), hard-fail in CI so a missing dev group can't mask a broken gate.

**`architecture/tests/test_dp_boost.py`:**
- `test_gradient_hessian_bounds` — `g=p−y ∈ [−1,1]`, `h=p(1−p) ∈ (0,0.25]`.
- `test_histogram_sensitivity_bound` — build on N vs N+1 rows: exactly one bin per feature changes,
  `|ΔG| ≤ 1`, `|ΔH| ≤ 0.25` (empirically pins §3.2).
- `test_num_queries_is_depth_times_rounds` — `num_histogram_queries == max_depth*num_boost_round`
  (pins §3.3 level count against the per-node error).
- `test_gaussian_releases_is_two_times_levels` — `num_gaussian_releases == 2*num_histogram_queries`
  (pins the §3.2/§3.3 two-stat count against the ~2× under-report; the Opacus gate cannot see this).
- `test_gaussian_epsilon_uses_release_count` — for fixed σ, the ε `make_mechanism` reports (via
  `account_run(n_rel, σ, …)`) equals `account_run(2*D*T, σ, …)`, i.e. the mechanism charges G+H, not
  one release per level.
- `test_laplace_scale_uses_l1` — Laplace per-feature scale scales with `d`, NOT `√d`: at `d=2` vs a
  hypothetical `d=4` (same ε,k) the scale ratio is `4/2=2`, not `√2`; and doubling `d` doubles ε for
  fixed scale (pins §3.2/§8.2).
- `test_epsilon_independent_of_clip_bound` — reported ε is identical across `clip_bound ∈
  {0.1, 1.0, 10.0}` at fixed (ε-target, k, σ) — clipping is post-processing (pins §3.8).
- `test_negative_noised_hessian_is_floored` — force large hessian noise (tiny ε / pinned σ) so a
  child's noised `H+λ < 0`; assert no split divides by ≤0, gain stays finite and non-negative at the
  chosen split, and training completes (pins the §3.3/§8.3 `DENOM_FLOOR` guard).
- `test_noise_multiplier_monotonic_in_epsilon` — σ strictly decreasing in ε.
- `test_no_noise_equals_reasonable_baseline` — DP-off learner AUC within ~0.05–0.08 of `xgb.train`,
  both ≥ ~0.75 on separable synthetic (proves the learner is real). The anchor pins `base_score=0.5`,
  `nthread=1`, `subsample=1.0` so the gap is "learner", not init/threading/subsample (F5, §4.4).
- `test_dp_auc_monotonicity` — averaged over seeds: `AUC(no-noise) ≥ AUC(ε=5) ≥ AUC(ε=1)` with a
  margin (soft trend; averaged to avoid single-seed flakiness).
- `test_determinism_pinned_seed` — same seed → bitwise-identical trees + predictions.
- `test_fixed_bin_edges_data_independent` — edges depend only on `feature_ranges`+`max_bins`.
- `test_leaf_clip_bounds_weight` — tiny `clip_bound` → all `|w| ≤ clip_bound`.
- `test_predict_range` — probabilities ∈ (0,1).
- `test_dpconfig_from_run_config` — a pyproject-style flat dict round-trips through
  `unflatten_dict`+`replace_keys`+`from_run_config` to the expected dataclass (mirrors 1.d's
  idempotency pin).

**`architecture/tests/test_synthetic.py`** (or folded into `test_dp_boost.py`): generator
determinism; label balance in range; both classes present in train and valid.

## 5. Files to change

- **New** `architecture/fed_stroke/dp_accounting.py` — §4.1 accountant.
- **New** `architecture/fed_stroke/dp_boost.py` — §4.2 mechanism/learner/seam.
- **New** `architecture/fed_stroke/synthetic.py` — §4.6 generator.
- **New** `architecture/scripts/dp_prototype_demo.py` — §4.4 three-arm CLI.
- **New** `architecture/tests/test_dp_accounting.py` — §4.7 gate + inverse round-trip.
- **New** `architecture/tests/test_dp_boost.py` — §4.7 mechanism/learner tests.
- **New** `architecture/tests/test_synthetic.py` — §4.7 generator tests (or fold in).
- **Modify** `architecture/pyproject.toml` — §4.3 `dp.*` config table; add `numpy>=1.26`,
  `scipy>=1.11` to `[project].dependencies`; add `opacus>=1.5,<2` to `[dependency-groups].dev`.
  Regenerate `uv.lock` with the dev group (production resolution ignores it).
- **Modify** `docs/logbook.md` — dated entry when the code lands (format: `- YYYY-MM-DD — DP
  plug-point prototype …`, with the gate result and the three-arm synthetic AUCs).
- **Reused unchanged**: `fed_stroke/metrics.py` (`compute_binary_metrics`), `fed_stroke/schema.py`
  (`FEATURE_COLS`, `TARGET_COL`), optionally `fed_stroke/baseline.py`
  (`split_half`/`score_booster_on_half` via `--via-parquet`).
- **Not modified here** (designed only, §4.5): `fed_stroke/client_app.py`, `fed_stroke/server_app.py`,
  the strategies — the FL DP branch is a downstream integration item.
- **Not checked here**: the roadmap prerequisite checkbox stays `[ ]` until the code lands.

## 6. Verification (end to end)

Run from `architecture/`. The env split is the point: the gate needs the dev group; the runtime must
not.

1. **Install dev group** (pulls opacus/torch for the gate only):
   `uv sync --group dev`
2. **The gate** — our accountant matches Opacus and the inverse calibrator round-trips:
   `uv run pytest tests/test_dp_accounting.py -q`
   This MUST pass before any downstream sweep trusts the accountant (roadmap acceptance).
3. **Mechanism / learner / generator**:
   `uv run pytest tests/test_dp_boost.py tests/test_synthetic.py -q`
4. **Three-arm sanity run**:
   `uv run python scripts/dp_prototype_demo.py --epsilons 1 3 5`
   Confirm **A ≈ B** (our no-noise learner tracks stock XGBoost) and **monotone AUC erosion B→C** as
   ε shrinks. Record the printed table + derived σ/k in `docs/logbook.md`.
5. **Runtime stays torch-free** — the federated client never needs opacus:
   `uv sync --no-dev && uv run python -c "import fed_stroke.dp_accounting, fed_stroke.dp_boost"`
   succeeds with opacus absent. (Re-`uv sync --group dev` afterwards to restore the gate.)

## 7. Acceptance criteria

1. **The gate passes.** `test_dp_accounting.py` shows our `account_run` ε equals Opacus
   `RDPAccountant` ε within `rel=1e-6` at every q=1.0 toy setting (σ×steps matrix), `DEFAULT_ORDERS`
   equals `RDPAccountant.DEFAULT_ALPHAS`, and the inverse `noise_multiplier_for_epsilon` round-trips
   to the target ε (never over-spending). This is the roadmap's hard gate.
2. **The mechanism trains and behaves.** On single-site synthetic: DP-off AUC is within tolerance of
   the `xgb.train` anchor and both are materially better than chance; DP-on AUC erodes monotonically
   as ε shrinks (averaged over seeds); leaf weights respect `clip_bound`; predictions ∈ (0,1);
   pinned-seed runs are bitwise reproducible.
3. **Sensitivity + composition are pinned to code, not prose.** Tests assert `Δ(G)≤1`, `Δ(H)≤0.25`
   empirically; `num_histogram_queries == max_depth × num_boost_round` (per-level, not per-node);
   **`num_gaussian_releases == 2 × D × T`** and the mechanism charges ε on that count (G and H are
   two releases per level — the ~2× under-report is pinned out); the Laplace scale uses `d` (L1),
   not `√d`; and the noised-hessian denominator is floored (no divide-by-≤0).
4. **DP is data-independent where it must be.** The DP arm uses fixed public-range bin edges (test:
   edges independent of X) and forces `subsample=1.0` / accounts at q=1.0 (honest upper bound). No
   data-dependent quantiles and no unearned amplification credit.
5. **The three-arm gap decomposes.** `dp_prototype_demo.py` prints A/B/C with `noise_mult` and `k`,
   scored through the shared `compute_binary_metrics`, so (classic XGB → no-noise learner → DP) is
   readable — the DP-vs-classic comparison HPO will carry forward.
6. **Runtime stays torch-free.** `fed_stroke.dp_accounting` and `fed_stroke.dp_boost` import under
   `uv sync --no-dev` (opacus absent); opacus is only in `[dependency-groups].dev`.
7. **The seam is defined.** `DPConfig`, `HistogramNoiseMechanism`/`train_dp_gbdt`, and the `dp.*`
   config keys exist and are the exact symbols the downstream `client_app` DP branch will import;
   `DPConfig.from_run_config` round-trips a pyproject-style config.
8. `pytest` (full suite) stays green; new tests are sub-second and follow the repo's fixture idiom.

## 8. Risks & notes

1. **Data-dependent bin edges leak (highest).** Quantile edges from `X` are not DP (§3.7). The DP arm
   uses fixed public clinical ranges (Age 0–120, NIHSS 0–42); real Geneva needs a private-quantile
   mechanism or a public reference grid before 1.1.b runs DP on real data. Enforced by
   `bin_strategy="fixed_range"` in DP mode.
2. **Subsample amplification is not claimed (§3.5).** XGBoost `subsample=0.8` is without-replacement,
   not Poisson; the prototype forces `subsample=1.0` in DP mode and reports ε at q=1.0. The `q<1` SGM
   path exists behind `sample_rate` but is used honestly only if the mechanism performs genuine
   Poisson record sampling (downstream). Reported ε is a conservative upper bound.
3. **Single-class / empty nodes.** Noise or empty bins → tiny/zero hessian; guard `H+λ` division and
   apply `min_child_weight` on the noised H. The generator's stratified split keeps both classes in
   train and valid so `compute_binary_metrics` doesn't return NaN AUC.
4. **Schema-agnostic.** Sensitivity constants are per-feature and the L2 scales as `√d`; the generator
   and `FEATURE_RANGES` read the frozen `FEATURE_COLS`, so the mechanism keeps working as the feature
   set grows beyond the two current columns — the ε cost scales with `d` explicitly.
5. **`min_child_weight` under noise is approximate.** Noise can push a child's hessian below the
   threshold or negative; the guard is applied on the noised sum and the count guarantee is
   probabilistic under DP. Documented, not silently ignored.
6. **The Laplace path is cruder, and calibrates to L1 (`d`), not L2 (`√d`).** Basic composition, L1
   sensitivity `d·stat_L1` per release (§3.2), no δ; noises G and H (⇒ `2·D·T` releases, ε split
   evenly); expected worse than Gaussian at matched ε. A `√d` factor here (an L2 slip) would
   understate ε — pinned out by `test_laplace_scale_uses_l1` (§4.7). Secondary sanity arm, explicitly
   **not** the cross-checked accountant path.
7. **Classic-XGB anchor is the seed of the HPO DP-vs-classic comparison (user requirement).** The
   three-arm decomposition here is the synthetic dress rehearsal; 1.1.b/1.1.d/1.1.e run the same
   A/B/C head-to-head on real Geneva. The anchor must use the same hyperparameters as the learner
   (via `BoostParams.from_xgb_params`) so A→B measures only "learner", not a hyperparameter mismatch.
8. **Opacus is dev-only; the runtime must stay torch-free.** The wheel packages only `fed_stroke`
   and the container installs `[project].dependencies` (dependency-groups are excluded from the built
   artifact and from `uv sync --no-dev`). Verified by acceptance criterion 6.
9. **The conversion formula is load-bearing.** Classic Mironov vs improved Balle (§3.4): using the
   wrong one produces a systematic ε offset that fails the gate. Pinned to the Balle form + the exact
   Opacus order grid. The opacus dep is a *range* (`>=1.5,<2`), not an exact pin; the real guard is
   `test_default_orders_match_opacus`, which fails loudly if any 1.x release changes `DEFAULT_ALPHAS`
   (a deliberate canary, not silent drift). Pin exactly only if that canary proves noisy in CI.
10. **The two-releases-per-level count is the subtlest ε trap, and the Opacus gate does NOT cover
    it (F1, highest of this review).** The gate validates the accountant in (σ, steps) units; it is
    blind to how many releases the *mechanism* issues per level. Because each level noises G **and**
    H (two Gaussian mechanisms, §3.2), feeding `D·T` instead of `2·D·T` under-reports ε ~2× while the
    gate stays green. Guarded by `test_gaussian_releases_is_two_times_levels` and
    `test_gaussian_epsilon_uses_release_count` (§4.7), and pinned in the module docstring.
11. **FL integration is out of scope (§4.5).** DP mode swaps the whole learner, so `DPBooster`
    serialization and a DP-aware server aggregator are a distinct downstream item. This prototype
    proves the mechanism + accountant + a synthetic utility look so that integration is wiring, not
    research.

## 9. Review decisions — audit trail

- **Decision 1 — mechanism = self-contained NumPy DP-GBDT (not a hook, not a fork).** Stock `hist`
  histograms are unreachable from Python (§3.1, verified against the client seam), so noise must be
  injected in a learner we own. A `dp-xgboost` fork was rejected: it pins an old XGBoost incompatible
  with `xgboost>=2.0`/`flwr>=1.28`, is effectively unmaintained, and would make the accountant gate
  "trust the fork" instead of "validate our math." Custom-objective gradient extraction collapses
  into the NumPy path anyway (you still must build the histogram and pick the split yourself). The
  docs already reached this conclusion (arch §6.3; lit review §5).
- **Decision 2 — own lightweight RDP accountant, validated against Opacus (dev-only).** Keeps the
  federated client runtime torch-free (§8.8) and makes the downstream ε numbers *ours* and auditable,
  which is exactly what the roadmap gate ("compute ε with the accountant and compare to Opacus")
  presumes. Using Opacus at runtime was rejected (pulls torch into the Shenzhen container and leaves
  nothing to compare). Google `dp_accounting` was considered as the reference but Opacus is the
  roadmap-named one with the cleaner API; the accountant is written to match Opacus internals exactly
  (§3.4).
- **Decision 3 — full DP-utility sanity on synthetic, three arms (not primitives-only).** Running the
  mechanism end-to-end (A classic / B no-noise / C DP) exercises the whole plug-point and gives cheap,
  decisive evidence the plumbing is right before real data — the point of a *prototype*. The synthetic
  generator + demo cost little. The ε∈{1,3,5,10} sweep proper stays downstream (1.1.b) on real Geneva.
- **Decision 4 — composition is per tree level (k = D×T), and split-argmax/leaf-weights are free
  post-processing.** Parallel composition across disjoint sibling nodes and across features (§3.3);
  the standard DP-GBDT treatment (Grislain & Gonzalvez; Maddock et al., both cited in arch §6.3).
  Rejected the per-node count (over-charges ≈3.75× at depth 4 — `(2^D−1)/D = 15/4`). Pinned by test.
- **Decision 5 — honest privacy accounting over amplification credit.** Force `subsample=1.0` in DP
  mode and account at q=1.0 because XGBoost's subsampling is without-replacement, not Poisson (§3.5).
  A too-clever amplification claim would *understate* the true ε — the opposite of what a privacy
  guarantee must do.
- **Decision 6 — DP framed as a measurement instrument, not a commitment (with the user).** DP is
  optional for a governed 2-center collaboration; this prototype exists to *quantify* the DP-utility
  cost so the keep/drop decision is evidence-based (see Framing above and §1). This is why the spec is
  deliberately cheap and single-site.
- **User requirement folded in — DP must be evaluated head-to-head against classic XGBoost during
  HPO.** Realized as the Arm A classic anchor and the A→B→C decomposition (§4.4), carried forward as
  an explicit note to 1.1.b/1.1.d/1.1.e (§8.7).

### Decisions from the plan-eng-review pass (2026-07-20)

- **Decision 7 — the Gaussian mechanism charges BOTH histograms: `num_gaussian_releases = 2·D·T`,
  with per-stat noise kept (with the user).** Each level noises the gradient AND hessian histogram as
  two independent Gaussian mechanisms (different scales, §3.2), which compose to `α/σ²` per level.
  The original spec fed `k = D·T` to the accountant, silently under-reporting ε by ~2×. Alternatives
  rejected: a single isotropic joint (G,H) query (would swamp the small-scale hessian → worse
  utility, less faithful to production) and noising only the gradient (deviates from the roadmap's
  explicit "gradient/hessian histograms"). Per-stat noise + a doubled release count is the honest
  fix that preserves utility. This is *not* covered by the Opacus gate; pinned by dedicated tests
  (§4.7, §8 note 10).
- **Decision 8 — leaves stay post-processing (histogram-only noise); `clip_bound` is a utility guard,
  not a privacy knob (with the user).** Leaf `(G,H)` are summed from the already-noised histogram, so
  by post-processing immunity they cost zero budget and ε is invariant to `clip_bound` (§3.8). This
  diverges from the lit-review §5 variant that *also* privatizes leaf values as a separate noised
  release; that path is deliberately not taken — in this learner the leaf totals are already noisy,
  so a separate leaf noise would double-count. Recorded so the arch §6.3 "clip to bound sensitivity"
  wording is not mistaken for a budgeted mechanism. Pinned by `test_epsilon_independent_of_clip_bound`.
- **Decision 9 — correctness/clarity fixes applied this review (no design fork):**
  - **F2 — Laplace calibrates to L1 (`d`), not L2 (`√d`).** The original `b = k·√d·… /ε` understated
    ε; corrected to `d·stat_L1` per release with G/H accounted separately (§3.2, §4.1, §4.2, §8.6).
  - **F4 — the noised-hessian denominator is floored** (`D(h)=max(h+λ, DENOM_FLOOR)`) and splits with
    noised child hessian `< min_child_weight` are rejected, so noise cannot flip the gain sign or
    divide by ≤0 (§4.2, §8.3). Previously "guard H+λ" was named but unspecified.
  - **F5 — Arm A pins `base_score=0.5` and `nthread=1`** so XGBoost's auto-`base_score` and
    multi-thread nondeterminism do not pollute the A→B gap or the logbook table (§4.4).
  - **F6 — the q<1 accountant test is pinned to conservative-only** (`eps_ours ≥ eps_ref`); q=1.0
    stays the sole exact gate, since the prototype forces q=1.0 (§4.7).
  - **F7 — `FEATURE_RANGES` has one home** (`dp_boost.py`), imported by `synthetic.py` (§4.6, DRY).
  - **F8 — the demo prints both `levels` and `releases`** (60 / 120 at defaults) so the 2× is visible
    (§4.4).
  - **F9 — the opacus order-grid guard is the `test_default_orders_match_opacus` canary,** not the
    (ranged) dep version (§8.9).
  - **F10 — the reported ε is stated to be per-site (= per-patient)** since patients do not cross
    sites (§3.9).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 (inline) | CLEAR | DP-necessity re-examined with the user before writing; reframed as an evidence-gathering instrument, not a commitment |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | not run at spec stage |
| Eng Review | `/plan-eng-review` | Architecture & tests | 2 (Plan agents) + 1 (2026-07-20 critical pass) | CLEAR (fixes applied) | 2nd pass caught 2 silent-ε-undercount bugs and fixed them: F1 Gaussian charged 1 release/level but noises G+H (~2× under-report) → `num_gaussian_releases=2·D·T`; F2 Laplace used `√d` (L2) not `d` (L1). +8 corrections (F3–F10) folded in; all decisions in §9 |
| Outside Voice | — | Independent challenge | 0 | — | deferred to the implementation PR |
| Design Review | `/plan-design-review` | UI/UX | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience | 0 | — | — |

- **CROSS-MODEL:** not run; two independent Plan passes (mechanism, accountant) cross-checked each
  other and were reconciled in §4. The 2026-07-20 critical pass then verified every code/config
  citation against the tree and re-derived the DP math from first principles, catching F1/F2.
- **UNRESOLVED:** none at spec stage. Two design forks surfaced this pass were resolved *with the
  user* — Decision 7 (charge G+H, keep per-stat noise) and Decision 8 (leaves post-processing).
  Open items remain *downstream-owned* and named in §1 (out of scope) and §8: FL integration,
  private-quantile bins on real data, subsampled-RDP credit, the real-Geneva ε sweep.
- **VERDICT:** READY — spec complete and corrected; implement behind the §6 verification, with the
  accountant gate (§7.1) green AND `test_gaussian_releases_is_two_times_levels` / `test_laplace_scale_uses_l1`
  as the release-count and L1 guards that the gate alone cannot provide.

NO UNRESOLVED DECISIONS
