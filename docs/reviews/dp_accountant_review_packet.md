# DP accountant — independent review packet

**Purpose.** This packet is the briefing for the **blocking GATE on roadmap 1.1.b**: an independent
DP/privacy review of the ε-accountant *before it reports ε as a claim about real Geneva patients*.
It is self-contained — a reviewer should be able to check every load-bearing claim from the math
here plus a targeted look at the cited code, without reading the whole spec.

- **What is being certified:** the ε (and the Laplace ε) that the DP-GBDT prototype reports for a
  single site are a sound (ε, δ)-DP guarantee for the histograms + tree ensemble released by that
  site, under the stated assumptions (§7).
- **What is NOT in scope:** the utility of the mechanism, the FL wiring (not yet built), and the
  real-data private-quantile binning (downstream). Only the *accounting* is under review.
- **Reviewer independence requirement:** the reviewer must have DP expertise and be **independent of
  the accountant's author** (the author is the same agent that wrote the code and its tests, so the
  author's tests share the author's blind spots — see §6).
- **Reference spec:** [`docs/specs/1_1_prereq_dp_plugpoint.md`](../specs/1_1_prereq_dp_plugpoint.md)
  (§3 = "framework/DP facts that drive the design"; the section tags below point into it).
- **Code under review:** [`architecture/fed_stroke/dp/accounting.py`](../../architecture/fed_stroke/dp/accounting.py)
  and [`architecture/fed_stroke/dp/boost.py`](../../architecture/fed_stroke/dp/boost.py).

---

## 1. Setting and notation

- Task: `binary:logistic` gradient boosting. Per example, with raw margin `F_i`, `p_i = σ(F_i)`,
  label `y_i ∈ {0,1}`:
  - gradient `g_i = p_i − y_i`
  - hessian `h_i = p_i(1 − p_i)`
- The learner is a **histogram** GBDT: at each split-finding node it bins each of `d` features into
  `B` fixed bins and accumulates, per feature, `G_b = Σ_{i∈b} g_i` and `H_b = Σ_{i∈b} h_i`.
- Configuration that fixes the release count: tree depth `D` (`max_depth`), trees per site `T`
  (`num_boost_round`). Committed federated config: `D=4`, `T=20`. Synthetic demo: `D=3`, `T=20`.
- `σ` = Gaussian noise multiplier (Opacus convention: noise std / L2-sensitivity). `δ = 1e-5`.
- Neighbouring datasets: add/remove one patient record ("unbounded"/add-remove adjacency).

**Neighbouring-record effect (the fact all sensitivities rest on).** A record has one value per
feature, so it lands in **exactly one bin of each feature's histogram**. Adding/removing it changes
one bin per feature by `g_i` (in `G`) and `h_i` (in `H`), and nothing else.

---

## 2. Load-bearing claims to verify

Each claim states the math, then the code + test that pin it. Please check the math independently;
the code/test pointers are for confirming the implementation matches.

### Claim 1 — the gradient/hessian bounds ARE the sensitivity (no gradient clipping)
`g_i = p_i − y_i ∈ [−1, 1]` and `h_i = p_i(1−p_i) ∈ (0, 0.25]`. Hence per **feature**:
- L2 (and L1, single nonzero entry) sensitivity of `G`: `Δ(G) = max|g| = 1.0`.
- L2 (and L1) sensitivity of `H`: `Δ(H) = max h = 0.25`.

Across `d` features **concatenated** (one bin touched per feature): **L2** `= √d · 1.0` (G),
`√d · 0.25` (H); **L1** `= d · 1.0` (G), `d · 0.25` (H). No per-example gradient clipping is needed
(unlike DP-SGD) because the objective already bounds `g, h`.

- Code: constants `G_L2_PER_FEATURE=1.0`, `H_L2_PER_FEATURE=0.25`, `G_L1_PER_FEATURE=1.0`,
  `H_L1_PER_FEATURE=0.25` — `boost.py:38-41`.
- Test: `test_histogram_sensitivity_bound` (N vs N+1 rows → exactly one bin/feature changes,
  `|ΔG|≤1`, `|ΔH|≤0.25`) — `tests/dp/test_boost.py:58`; `test_gradient_hessian_bounds:47`. (§3.2)

### Claim 2 — composition is PER TREE LEVEL, not per node
A record flows to exactly one node per depth level; sibling nodes at a level **partition** the
records, so releasing all their histograms is **parallel composition** — the RDP of a *single*
query, not the sum over nodes. Only depth levels, trees, and rounds compose **sequentially**:

```
num_histogram_queries = D × T            # split-finding LEVELS, NOT (2^D − 1) nodes per tree
```

Counting nodes (`2^D−1 = 15` at `D=4`) instead of levels (`D=4`) would over-charge by
`(2^D−1)/D ≈ 3.75×` (safe direction, but wrong).

- Code: `num_histogram_queries = max_depth * num_boost_round` — `boost.py:114-116`.
- Test: `test_num_queries_is_depth_times_rounds` — `tests/dp/test_boost.py:91`. (§3.3)

### Claim 3 — each level is TWO Gaussian releases (G and H) → `2·D·T` (highest-risk item)
Each level noises the **gradient AND the hessian** histogram, as **two independent Gaussian
mechanisms**, each at multiplier `σ` (with different per-stat scales, Claim 4). Two independent
Gaussian mechanisms at the same `σ` compose to per-level RDP `2·α/(2σ²) = α/σ²`, i.e. the accountant
must be fed **twice** the level count:

```
num_gaussian_releases = 2 × num_histogram_queries = 2·D·T
```

Whitening check (a ±1-record change): it moves `G` by L2 `√d·1.0` and `H` by L2 `√d·0.25`, so the
combined squared-Mahalanobis shift is
`‖ΔG/(σ√d·1.0)‖² + ‖ΔH/(σ√d·0.25)‖² = 1/σ² + 1/σ² = 2/σ²`, giving Gaussian RDP
`α·(2/σ²)/2 = α/σ²` per level. Charging one release per level under-reports ε by ~2×.

**This is the single subtlest ε trap, and the Opacus equivalence gate is BLIND to it** (§5): the gate
validates ε *given* a release count; it cannot see that the mechanism issues `2·D·T`, not `D·T`.

- Code: `num_gaussian_releases = 2 * num_histogram_queries` — `boost.py:119-122`; the factory feeds
  `n_rel = num_gaussian_releases(boost)` to the accountant — `boost.py:217, 224`.
- Tests: `test_gaussian_releases_is_two_times_levels:96`; `test_gaussian_epsilon_uses_release_count:102`
  (asserts the reported ε equals `account_run(2·D·T, σ, …)`, not the `D·T` value). (§3.2/§3.3)

### Claim 4 — Gaussian noise calibration realizes multiplier `σ`
Per-entry noise std added to the histograms:
- `G`: `std_G = σ · √d · G_L2 = σ·√d·1.0`
- `H`: `std_H = σ · √d · H_L2 = σ·√d·0.25`

For the concatenated `d`-feature `G` release (L2 sensitivity `√d·1.0`), adding per-coordinate noise
`N(0, (σ·√d·1.0)²)` is the Gaussian mechanism at noise multiplier `σ` → per-release RDP `α/(2σ²)`.
Same for `H`. Independent per-sibling noise at these scales is the correct realization of the
per-level parallel composition in Claim 2 (a record perturbs one sibling only).

- Code: `std_g/std_h` — `boost.py:222-223`; `_GaussianMechanism.add_noise` — `boost.py:~180`.

### Claim 5 — RDP composition + the improved Balle RDP→(ε,δ) conversion
Composition of `k = 2·D·T` identical Gaussian releases (linear RDP space):
`ρ_total(α) = k · α/(2σ²)`. Conversion (improved Balle et al. 2020), min over the order grid:

```
ε(α) = ρ(α) − (ln δ + ln α)/(α − 1) + ln((α − 1)/α)          # NOT Mironov ρ(α) + ln(1/δ)/(α−1)
ε    = min over α in DEFAULT_ORDERS
DEFAULT_ORDERS = [1 + x/10 for x in 1..99] + [12..63]        # == Opacus RDPAccountant.DEFAULT_ALPHAS
```

The classic Mironov conversion is looser and would give a systematic offset; the grid must be
Opacus's exactly (note the integer run starts at **12**, not 11 — 11 is absent).

- Code: `rdp_to_epsilon` (`eps = rdp − (log δ + log α)/(α−1) + log((α−1)/α)`, `np.nanargmin`, no
  negative clamp) — `accounting.py:158-184`; `DEFAULT_ORDERS` — `accounting.py:37`;
  `rdp_gaussian` q==1 closed form `α/(2σ²)` — `accounting.py:110`; `account_run` — `accounting.py:186`.
- Tests (the **Opacus equivalence gate**): `test_default_orders_match_opacus:22`,
  `test_epsilon_matches_opacus_q1:31` (rel=1e-6 across σ×steps), `test_rdp_gaussian_closed_form:77`,
  `test_account_run_monotone_in_sigma_and_k:84`, and the inverse round-trip
  `test_inverse_calibration_roundtrip:62`. (§3.4)

### Claim 6 — q = 1.0, no subsampling-amplification credit
XGBoost `subsample=0.8` is **without-replacement, once per tree** — NOT Poisson per-record — so the
subsampled-Gaussian amplification bound does not apply. DP mode forces `subsample=1.0` and accounts
at `q=1.0`; the reported ε is a clean `k`-fold Gaussian composition and an **honest upper bound**. A
`q<1` subsampled path exists behind `sample_rate` but is used only conservatively (see §6), never to
claim credit.

- Code: `rdp_gaussian(..., sample_rate=1.0)` default; the factory passes `1.0` — `boost.py:219`.
- Test (conservative direction only for q<1): `test_epsilon_conservative_when_subsampled:45`. (§3.5)

### Claim 7 — Laplace arm calibrates to L1 (`∝ d`), not L2 (`∝ √d`)
Secondary/cruder arm: basic (not RDP) composition, no δ. Per-release **L1** sensitivity is `d·stat_L1`
(L1 sums the per-feature entries; **no root**). It noises G and H, so `2·D·T` releases, budget split
evenly. Per-feature scales: `b_G = 2·D·T · d · G_L1 / ε`, `b_H = 2·D·T · d · H_L1 / ε`. Total ε is
the sum over the two homogeneous groups (each `k = D·T`):
`ε = k·(d·G_L1)/b_G + k·(d·H_L1)/b_H`. Using `√d` here (an L2 factor) would **understate** ε.

- Code: laplace branch of `make_mechanism` — `boost.py:227-235` (uses `d * *_L1_PER_FEATURE`);
  `account_run_laplace = num_queries · l1_sensitivity / laplace_scale` — `accounting.py:229`.
- Test: `test_laplace_scale_uses_l1_not_l2` (d=2→d=4 scale ratio is 2, not √2) — `tests/dp/test_boost.py:115`.
  (§3.2, §8.6)

### Claim 8 — leaf clipping is POST-PROCESSING (zero ε; `clip_bound` is not a privacy knob)
The only place noise enters is the histogram. A leaf's `(G,H)` totals are summed from the
**already-noised** parent histogram, never re-queried from the data. So `w = −η·G/D(H)` and its clip
`w ← clip(w, −c, +c)` are post-processing of a DP release ⇒ **zero** additional budget, and reported ε
is **invariant to `clip_bound`**. (`D(h) = max(h+λ, DENOM_FLOOR)` floors the denominator so noise
cannot divide by ≤0 or flip the gain sign; `min_child_weight` is applied on the *noised* hessian, so
that constraint is probabilistic under noise — documented, not a privacy claim.)

- Code: `_leaf_weight` — `boost.py:250`; `_best_split` floored denom — `boost.py:257`;
  `DENOM_FLOOR` — `boost.py:42`.
- Tests: `test_epsilon_independent_of_clip_bound:126`; `test_negative_noised_hessian_is_floored:139`.
  (§3.8)

### Claim 9 — reported ε is PER-SITE = the per-patient guarantee
Each patient's records live at exactly one site and are touched only by that site's histograms, so
the single-site ε **is** the guarantee an individual patient receives; there is no cross-site
composition to add. (§3.9)

---

## 3. Concrete numbers to sanity-check

For the committed federated config (`D=4`, `T=20`): `D·T = 80` levels ⇒
**`num_gaussian_releases = 160`** per site. Synthetic demo (`D=3`, `T=20`): `60` levels ⇒ **`120`**.

Reference σ our calibrator returns (q=1, δ=1e-5, k=160), which the reviewer can cross-check against
Opacus `get_noise_multiplier` / `RDPAccountant`:

| target ε | k (releases) | σ (our `noise_multiplier_for_epsilon`) |
|---|---|---|
| 1  | 160 | ≈ 51 |
| 3  | 160 | ≈ 19 |
| 5  | 160 | ≈ 12 |
| 10 | 160 | ≈ 6.6 |

(Exact values reproducible via `noise_multiplier_for_epsilon(ε, 160, 1.0, 1e-5)`; the demo prints the
`k=120` variants.)

**Independent cross-check recipe (Opacus, k=160, q=1):**
```python
from opacus.accountants import RDPAccountant
acc = RDPAccountant()
for _ in range(160):
    acc.step(noise_multiplier=SIGMA, sample_rate=1.0)
eps_ref = acc.get_epsilon(delta=1e-5)     # compare to fed_stroke.dp.accounting.account_run(160, SIGMA, 1.0, 1e-5)
```

---

## 4. How to reproduce the checks

From `architecture/` (dev group installs opacus/torch — the reference only):
```
uv sync --group dev
uv run pytest tests/dp/test_accounting.py -q      # the equivalence gate + inverse round-trip
uv run pytest tests/dp/test_boost.py -q           # sensitivity/composition/L1/clip invariants
uv run python scripts/dp_prototype_demo.py --epsilons 1 5 30 100 1000   # end-to-end σ, ε, k
```

---

## 5. What the Opacus equivalence gate does and does NOT cover

**Covers (machine-precision, rel=1e-6 at q=1.0):** the per-step Gaussian RDP `α/(2σ²)`, the order
grid, the RDP→(ε,δ) conversion, and composition-by-count. I.e. *"given k releases at multiplier σ, is
the reported ε correct?"* — **yes.**

**Does NOT cover (must be checked by this review):**
1. **The release count `k = 2·D·T`** — the gate is fed whatever `k` the mechanism computes; the ~2×
   two-releases-per-level fact (Claim 3) is invisible to it.
2. **The sensitivity → σ mapping** (Claims 1, 4): whether `√d·stat_L2` is the right per-entry std.
3. **The Laplace L1 scale** (Claim 7): basic composition is not exercised by the Opacus gate at all.
4. **The subsampling / adjacency assumptions** (Claim 6) and the per-site = per-patient framing
   (Claim 9).

These four are exactly why an independent human review is required in addition to the passing gate.

---

## 6. Errors already caught (transparency — do not assume they are the only ones)

Surfaced by an independent re-derivation **at spec stage**, before code — evidence that the gate and
author-written tests are not sufficient on their own:

- **F1 — release count.** The original spec fed `k = D·T` (one release per level); corrected to
  `2·D·T` because each level noises **both** G and H (Claim 3). This was a ~2× ε **under-report**.
- **F2 — Laplace scale.** Originally used `√d` (an L2 factor); corrected to `d` (L1, Claim 7). Using
  `√d` **understates** the Laplace ε.

Surfaced at **implementation stage** (caught by the gate itself):

- **F3 — order grid.** The spec text wrote `range(11, 64)`; Opacus uses `range(12, 64)` (integer 11
  is deliberately absent). `range(11,64)` fails `test_default_orders_match_opacus` and can shift the
  `min`-over-α and break the rel=1e-6 match. Implemented `range(12, 64)`.

The reviewer should treat these as a prior on where further errors may hide (composition counting and
sensitivity roots), not as an all-clear.

---

## 7. Assumptions & limitations the sign-off is conditioned on

1. **Add/remove-one-record adjacency**, unbounded DP.
2. **q = 1.0** (no subsampling amplification); `subsample` is forced to 1.0 in DP mode.
3. **δ = 1e-5** default (pilot). The reported ε is an **upper bound**.
4. **Fixed, data-independent bin edges** over public clinical ranges (Age 0–120, NIHSS 0–42). On real
   Geneva, private-quantile (or public-reference-grid) edges are a **downstream prerequisite before
   1.1.b runs DP on real data** — data-dependent quantile edges are NOT DP.
5. **`min_child_weight` under noise is probabilistic** (applied on the noised hessian).
6. **Gaussian is the RDP-accounted primary; Laplace is a secondary basic-composition sanity path**,
   not cross-checked against a reference accountant.
7. The prototype is **single-site synthetic**; this review certifies the **accountant math** that
   will carry to real data, not the FL wiring (unbuilt).

---

## 8. Sign-off

> An independent DP-expert review of the items in §2 and §5 is required before roadmap 1.1.b runs the
> DP sweep on real Geneva patients. Record the outcome below **and** as a dated entry in
> [`docs/logbook.md`](../logbook.md) (reviewer, date, scope, verdict).

- **Reviewer (name / affiliation):** ________________________
- **Date:** ____________
- **Scope reviewed:** Claims 1–9 (§2) · gate coverage (§5) · assumptions (§7)  — strike any not reviewed.
- **Verdict:** ☐ Approve ☐ Approve with conditions ☐ Reject
- **Conditions / findings (if any):**

  ______________________________________________________________________

  ______________________________________________________________________

- **Signature:** ________________________
