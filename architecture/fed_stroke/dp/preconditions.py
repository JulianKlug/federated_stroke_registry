"""fed_stroke.dp.preconditions: the fail-closed DP precondition gate (spec 1.1.a″ R7).

DP mode refuses to run unless EVERY condition holds — one `validate_dp_preconditions()` gate,
called by the client's DP train branch each round, unit-tested per condition:

- labels finite and exactly in {0, 1} (the g/h sensitivity bounds assume binary:logistic);
- one row per patient (the R3 adjacency unit, asserted at the learner side — the loader dedup
  guarantees it, this catches any future load path that forgets to route through
  generate_splits);
- features and mechanism parameters finite; per-mechanism budget checks (see below);
- bin edges fixed/public (the 1.1.a′ contract — now asserted, not assumed);
- `base_score` EXPLICITLY configured, never defaulted from upstream (XGBoost ≥ 2.0
  auto-derives it from the label mean, which would leak prevalence outside the accounting);
- the authorized round count for the run's calibrated σ (refuse rounds beyond the T_site used
  at calibration).

Per-mechanism budget checks (NEVER a blanket σ > 0 — Laplace's `noise_multiplier` is nan by
construction and `nan > 0` is False, which would reject every legitimate Laplace run):
- gaussian: σ finite > 0, 0 < δ < 1, reported ε finite > 0 (validated on
  `mechanism.reported_epsilon`, which also covers pinned-σ runs where ε is reported, not
  targeted);
- laplace: scales b_g, b_h finite > 0, reported ε finite > 0 (pure ε; δ not applicable);
- identity (R9 arm B): no ε/σ/δ checks — nothing is spent — but every data/binning/base_score
  condition still applies.
"""
import numpy as np

from fed_stroke.dp.boost import DPConfig, FEATURE_RANGES, fixed_bin_edges


def _fail(msg: str) -> None:
    raise ValueError(f"DP preconditions not met (spec 1.1.a″ R7) — {msg}")


def validate_dp_preconditions(*, X, y, patient_ids, dp: DPConfig, params: dict,
                              mechanism, global_round: int, num_rounds: int,
                              feature_ranges: dict | None = None) -> None:
    """Fail-closed gate; raises ValueError naming the violated condition, returns None if all
    hold. `mechanism` is the RUN-CALIBRATED mechanism the round will train with."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # -- data ----------------------------------------------------------------
    if not np.all(np.isfinite(X)):
        _fail("features must be finite (no NaN/inf reaches the histogram).")
    if not np.all(np.isfinite(y)):
        _fail("labels must be finite.")
    if not np.isin(y, (0.0, 1.0)).all():
        _fail("labels must be exactly in {0, 1} (the g ∈ [−1,1], h ∈ (0,0.25] sensitivity "
              "bounds assume binary:logistic).")

    pids = np.asarray(patient_ids)
    if pids.shape[0] != X.shape[0] or pids.shape[0] != y.shape[0]:
        _fail(f"patient_ids ({pids.shape[0]}) must align 1:1 with rows "
              f"(X: {X.shape[0]}, y: {y.shape[0]}).")
    if len(np.unique(pids)) != len(pids):
        _fail(f"one row per patient (R3) violated: {len(pids)} rows for "
              f"{len(np.unique(pids))} unique patients — the adjacency unit is the patient; "
              f"route loading through generate_splits.")

    # -- base_score (never defaulted) -----------------------------------------
    if "base_score" not in params:
        _fail("base_score must be EXPLICITLY configured (params.base-score): XGBoost ≥ 2.0 "
              "otherwise derives it from the label mean, leaking prevalence outside the "
              "accounting.")
    base_score = float(params["base_score"])
    if not (np.isfinite(base_score) and 0.0 < base_score < 1.0):
        _fail(f"base_score must be finite in (0, 1); got {base_score!r}.")

    # -- binning (fixed/public, asserted not assumed) --------------------------
    if dp.bin_strategy != "fixed_range":
        _fail(f"bin_strategy must be 'fixed_range' (data-independent); got "
              f"{dp.bin_strategy!r}.")
    if not (isinstance(dp.max_bins, int) and dp.max_bins > 0):
        _fail(f"max_bins must be a positive int; got {dp.max_bins!r}.")
    fr = feature_ranges if feature_ranges is not None else FEATURE_RANGES
    if X.shape[1] != len(fr):
        _fail(f"X has {X.shape[1]} columns but the public feature_ranges declare {len(fr)} "
              f"— √d noise scaling would be wrong.")
    for name, (lo, hi) in fr.items():
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            _fail(f"bin edges for {name!r} are not derivable: feature range ({lo!r}, {hi!r}) "
                  f"must be public finite constants with lo < hi.")
    for name, edges in zip(fr, fixed_bin_edges(fr, dp.max_bins)):
        if not np.all(np.isfinite(edges)):
            _fail(f"bin edges for {name!r} are not finite — feature ranges must be public "
                  f"finite constants.")
    if not (np.isfinite(dp.clip_bound) and dp.clip_bound > 0):
        _fail(f"clip_bound must be finite > 0; got {dp.clip_bound!r}.")

    # -- mechanism budget (per-mechanism, never a blanket σ > 0) ---------------
    if dp.mechanism == "identity":
        pass  # arm B spends nothing; no budget to validate.
    elif dp.mechanism == "gaussian":
        sigma = mechanism.noise_multiplier
        if not (np.isfinite(sigma) and sigma > 0):
            _fail(f"gaussian σ must be finite > 0; got {sigma!r}.")
        if not 0.0 < dp.delta < 1.0:
            _fail(f"δ must satisfy 0 < δ < 1; got {dp.delta!r}.")
        eps = mechanism.reported_epsilon
        if not (np.isfinite(eps) and eps > 0):
            _fail(f"reported ε must be finite > 0; got {eps!r}.")
    elif dp.mechanism == "laplace":
        if not (np.isfinite(mechanism.b_g) and mechanism.b_g > 0
                and np.isfinite(mechanism.b_h) and mechanism.b_h > 0):
            _fail(f"laplace scales must be finite > 0; got b_g={mechanism.b_g!r}, "
                  f"b_h={mechanism.b_h!r}.")
        eps = mechanism.reported_epsilon
        if not (np.isfinite(eps) and eps > 0):
            _fail(f"reported ε must be finite > 0; got {eps!r}.")
    else:
        _fail(f"unknown dp.mechanism {dp.mechanism!r}.")

    # -- authorized round count (σ was calibrated for num_rounds; more rounds would spend
    #    unaccounted releases) ---------------------------------------------------------
    if global_round > num_rounds:
        _fail(f"round {global_round} exceeds the authorized round count {num_rounds} the "
              f"run's σ was calibrated for — refusing to spend unaccounted releases.")
