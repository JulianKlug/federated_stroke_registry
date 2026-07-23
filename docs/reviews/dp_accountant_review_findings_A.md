# Independent DP review verdict: **BLOCK**

The accountant’s core Gaussian and Laplace mathematics is largely correct **for a fixed training table containing exactly one protected row per individual and using fresh private randomness**.

The current implementation does not meet those conditions. It therefore cannot presently report its ε as a sound per-patient guarantee for real Geneva data.

## Blocking findings

### 1. DP noise is deterministically generated from public inputs

The federated client constructs the noise RNG from:

```python
SeedSequence([params.seed, global_round, _site_hash(site)])
```

at `fed_stroke/client_app.py:126-128`. The standalone learner also defaults to the configured public seed at `fed_stroke/dp/boost.py:526-527`. Tests explicitly require identical models when these values are reused (`tests/test_dp_fl.py:144-167`).

This is fatal to the claimed DP guarantee. For neighboring datasets (D,D'), the released model is a deterministic, nonconstant function. If (M(D)\ne M(D')), choose the event (S={M(D)}):

[
\Pr[M(D)\in S]=1,\qquad \Pr[M(D')\in S]=0.
]

The DP inequality would require (1\le\delta), which is false for (\delta=10^{-5}).

I reproduced this behavior locally:

* Same data and seed: byte-identical serialized model.
* Neighboring data and same seed: different serialized model.

Production DP randomness must come from fresh, secret OS/CSPRNG entropy and must not be derived from an experiment seed, round number, or site identifier. Deterministic RNG injection can remain only in test-only code.

This is consistent with Opacus’s production guidance: its secure mode uses a cryptographically strong RNG, warns that insecure RNG is for experimentation, and prohibits supplying a seed in secure mode. ([opacus.ai][1])

### 2. The private train/validation split is not adjacency-stable

`fed_stroke/task.py:40-72` derives a patient-level, outcome-stratified split from the complete private dataset using `train_test_split`. Adding or removing one patient can change the assignment of several existing patients.

In a reproduction using the implementation’s splitting pattern, adding one patient changed the train/validation membership of **six existing patients**.

Consequently, neighboring raw site datasets do not necessarily produce training tables differing by one row. The histogram sensitivity arguments in Claim 1 then no longer apply to the actual pipeline.

The split must instead be one of:

* Fixed before the DP mechanism’s input is defined.
* Derived through a dataset-independent rule, such as a public keyed hash of patient ID.
* Included in a separate privacy analysis.

Private-label stratification cannot be treated as free preprocessing.

### 3. The implementation has admission-row privacy, not patient privacy

The hand-off assumes one patient record is the adjacency unit. The loader does not enforce that.

`fed_stroke/task.py:40-53` recognizes that a patient may have multiple admissions, but `task.py:70-72` and `task.py:207-208` retain all admission rows for training.

A patient with (m) admissions may therefore affect (m) histogram contributions. The worst-case per-feature sensitivity can rise from:

[
\Delta G=1,\quad \Delta H=0.25
]

to as much as:

[
\Delta G=m,\quad \Delta H=0.25m.
]

Thus Claim 9—“per-site ε is the per-patient guarantee”—is false for the current data representation.

Before sign-off, the pipeline must aggregate to one contribution per patient, enforce and account for a strict contribution cap, or explicitly relabel the protection as admission-row-level DP.

### 4. The federated transcript contains additional non-DP releases

The accountant covers the tree ensemble generated from noised histograms. The client also releases:

* Exact `num-examples` values in `client_app.py:176-179`.
* Exact validation metrics in `client_app.py:209-255`.
* Confusion-matrix cells, sample counts, positive counts, and a data-derived Youden threshold through `fed_stroke/metrics.py:88-125`.

Under add/remove adjacency, even the exact training count is itself data-dependent. The joint output of “DP model + exact count + exact validation statistics” is not protected by the model accountant.

These values must be removed, privatized under an additional budget, or placed outside the claimed DP release boundary. Validation patients also need an explicit privacy statement.

### 5. An ε sweep needs cross-run composition

The reported ε is calculated for one model-training run. If multiple models from an ε sweep are released for the same patients, the privacy guarantee is for the **collection of all releases**, not each model in isolation.

The project needs either:

* An RDP ledger that composes all released runs and converts the aggregate once.
* A trusted experimentation boundary from which only one selected model is released.

Using the same deterministic noise stream across sweep runs makes this issue more severe and must not be used.

## Claim-by-claim result

| Claim                              | Result                                            | Review finding                                                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Gradient/hessian sensitivity    | **Conditional pass**                              | Bounds and (\sqrt d)/(d) aggregation are correct for binary labels and one row per privacy unit. Neither condition is fully enforced.                                                         |
| 2. Composition per tree level      | **Pass**                                          | Counting `D × T` levels is valid: records occupy one node per level and sibling datasets are disjoint.                                                                                        |
| 3. Two Gaussian releases per level | **Pass**                                          | `2 × D × T` is correctly computed and supplied to the accountant.                                                                                                                             |
| 4. Gaussian calibration            | **Pass**                                          | `std_G=σ√d` and `std_H=0.25σ√d` correctly realize multiplier (σ) for the separate releases.                                                                                                   |
| 5. RDP accounting/conversion       | **Pass by inspection and numerical reproduction** | Formula, composition, and order grid match the current official Opacus implementation. Official Opacus still uses fractional orders through 10.9 followed by integers 12–63. ([opacus.ai][2]) |
| 6. No subsampling credit           | **Conditional pass**                              | The learner uses all training rows and accounts at (q=1). This does not repair the unstable private train/validation split.                                                                   |
| 7. Laplace L1 calibration          | **Pass**                                          | The code correctly uses (d), not (\sqrt d), and composes the G and H groups.                                                                                                                  |
| 8. Leaf clipping post-processing   | **Pass**                                          | Leaf values and clipping are computed from already-noised histograms and add no privacy cost.                                                                                                 |
| 9. Per-site equals per-patient     | **Fail**                                          | Multiple admissions per patient are retained, and cross-site patient disjointness is assumed rather than enforced.                                                                            |

## Independent numerical checks

For (D=4), (T=20):

[
D T=80,\qquad k=2DT=160.
]

Using the implementation’s accountant and inverse calibration at (\delta=10^{-5}):

| Target ε | Computed σ | Round-trip ε |
| -------: | ---------: | -----------: |
|        1 |   51.17053 |   0.99999999 |
|        3 |   18.88773 |   2.99999985 |
|        5 |   12.05005 |   4.99999970 |
|       10 |    6.69895 |   9.99999851 |

These agree with the hand-off’s rounded sanity values.

## Additional safeguards required

The implementation should fail closed unless all of the following are validated locally:

* Labels are finite and exactly in ({0,1}).
* Each privacy unit contributes at most the calibrated number of rows.
* Feature values and mechanism parameters are finite.
* (0<\delta<1), (ε>0), and (σ>0).
* Bin edges are fixed/public or separately privatized.
* The client enforces the authorized number of rounds and maintains a local privacy ledger.
* Patients are disjoint across sites, or cross-site participation is composed.

## Test execution note

The standalone accounting suite completed with **12 passed and 19 skipped** in the available environment. I could not execute the full DP/federated suite because `flwr` and `opacus` were not installed; dependency installation failed due a package-mirror HTTP 503. The Opacus-sensitive formulas and grid were therefore checked through source inspection, official current Opacus source, and independent numerical recomputation rather than a local Opacus execution.

## Gate decision

> **Do not sign off roadmap 1.1.b for real Geneva patients.**
> The fixed-table accountant mathematics passes conditionally, but the public deterministic RNG, unstable private split, multiple-admission contribution mismatch, and unaccounted federated outputs invalidate the implementation’s current per-patient ((ε,\delta))-DP claim.

[1]: https://opacus.ai/api/_modules/opacus/privacy_engine.html "https://opacus.ai/api/_modules/opacus/privacy_engine.html"
[2]: https://opacus.ai/api/_modules/opacus/accountants/rdp.html "https://opacus.ai/api/_modules/opacus/accountants/rdp.html"

