"""fed_stroke.dp: a lightweight, torch-free RDP accountant (spec 1.1 §4.1).

The Gaussian mechanism under Rényi-DP composition, converted to (ε, δ) via the improved
Balle-et-al bound, plus an inverse σ-for-target-ε calibrator. numpy/scipy only — no
torch/tf/flwr — so the federated client runtime never pulls a heavyweight DP library
(spec §8.8). This module is *sensitivity-agnostic*: it works purely in `noise_multiplier`
(σ) units, so the equivalence gate never needs a sensitivity and the mechanism
(`fed_stroke.dp.boost`) owns the sensitivity→σ mapping.

**The hard gate (roadmap acceptance).** On a fixed toy setting this accountant's total ε
MUST equal Opacus's `RDPAccountant` within floating-point tolerance at q=1.0. To achieve
that this module reproduces three Opacus internals *exactly* (spec §3.4):

  (i)   per-step Gaussian RDP `α/(2σ²)` at q=1.0 (Opacus special-cases q==1 to this closed
        form → machine-precision match);
  (ii)  the order grid `DEFAULT_ORDERS`, identical to `RDPAccountant.DEFAULT_ALPHAS`;
  (iii) the improved Balle-et-al RDP→(ε,δ) conversion
        `ε(α) = ρ(α) − (ln δ + ln α)/(α−1) + ln((α−1)/α)`, min over the grid — NOT the
        classic Mironov `ρ(α) + ln(1/δ)/(α−1)`, which is looser and fails the tolerance.

The subsampled path (q<1) mirrors Opacus's Mironov-Wang `log_a` bound but is NOT gate-exact:
the prototype forces q=1.0 (§3.5), so q<1 is only required to be conservative (never
under-report). See `fed_stroke/dp/boost.py` for how the mechanism feeds this accountant
`2·D·T` Gaussian releases (the two-releases-per-level count the Opacus gate cannot see, §3.3).
"""
import math
import warnings

import numpy as np
from scipy import special

# Order grid. MUST equal opacus RDPAccountant.DEFAULT_ALPHAS (spec §3.4, pinned by
# test_default_orders_match_opacus). NOTE: the integer run is range(12, 64) — 11 is
# DELIBERATELY absent (the fractional comprehension tops out at 10.9). Opacus 1.5.x uses
# exactly this grid; using range(11, 64) both fails the equality test and injects an extra
# α=11 candidate that can win the min and break the rel=1e-6 gate (spec 1.1 correction).
DEFAULT_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))


def _log_add(log_a: float, log_b: float) -> float:
    """log(exp(log_a) + exp(log_b)), stable. Mirrors opacus _log_add."""
    if log_a < log_b:
        log_a, log_b = log_b, log_a
    if log_b == -np.inf:
        return log_a
    return math.log1p(math.exp(log_b - log_a)) + log_a


def _log_sub(log_a: float, log_b: float) -> float:
    """log(exp(log_a) − exp(log_b)), stable; requires log_a >= log_b. Mirrors opacus."""
    if log_a < log_b:
        raise ValueError("_log_sub: negative result (log_a < log_b)")
    if log_b == -np.inf:
        return log_a
    return math.log1p(-math.exp(log_b - log_a)) + log_a


def _log_erfc(x: float) -> float:
    """log(erfc(x)) via scipy.log_ndtr — stable in the tails. Mirrors opacus _log_erfc."""
    return math.log(2) + special.log_ndtr(-x * (2 ** 0.5))


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log(A_alpha) for integer alpha (opacus _compute_log_a_for_int_alpha)."""
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * i - i) / (2 * sigma ** 2)
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log(A_alpha) for fractional alpha (opacus _compute_log_a_for_frac_alpha)."""
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma ** 2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma ** 2) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def _compute_rdp_scalar(q: float, sigma: float, alpha: float) -> float:
    """Per-step subsampled-Gaussian RDP at ONE order (opacus _compute_rdp branch order)."""
    if q == 0:
        return 0.0
    if sigma == 0:
        return np.inf
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    if np.isinf(alpha):
        return np.inf
    # Detect integer-valued orders by .is_integer() (e.g. 2.0, 3.0 route to the int path),
    # NOT by dtype — matches opacus (spec pitfall, only matters for q<1).
    if float(alpha).is_integer():
        log_a = _compute_log_a_int(q, sigma, int(alpha))
    else:
        log_a = _compute_log_a_frac(q, sigma, alpha)
    return float(log_a) / (alpha - 1)


def rdp_gaussian(alpha, noise_multiplier, sample_rate=1.0):
    """Per-step Rényi-DP ε at order(s) α for one (subsampled) Gaussian release.

    q == 1.0 -> α / (2·σ²)   (exact closed form; matches Opacus's q==1 special case)
    q  < 1.0 -> subsampled-Gaussian (Mironov-Wang) log_a bound, log-space. Behind the same
                param; the GATE rides q=1.0, so q<1 is required only to be conservative.

    Accepts a scalar or array-like `alpha`; returns the same shape.
    """
    orders = np.asarray(alpha, dtype=float)
    if sample_rate == 1.0:
        out = orders / (2 * noise_multiplier ** 2)
    else:
        out = np.array(
            [_compute_rdp_scalar(sample_rate, noise_multiplier, a) for a in orders.ravel()]
        ).reshape(orders.shape)
    return out if out.ndim else float(out)


def compose_rdp(rdp_per_step, num_steps=None):
    """Additive RDP composition (done in LINEAR RDP space, not log space).

    Two call shapes:
      compose_rdp(rdp_vec, num_steps)  -> num_steps * rdp_vec   (identical steps)
      compose_rdp([(rdp_vec, count), ...]) -> Σ count_k * rdp_vec_k   (heterogeneous)
    """
    if num_steps is not None:
        return np.asarray(rdp_per_step, dtype=float) * num_steps
    total = None
    for rdp_vec, count in rdp_per_step:
        term = np.asarray(rdp_vec, dtype=float) * count
        total = term if total is None else total + term
    return total


def rdp_to_epsilon(rdp, orders, delta) -> tuple[float, float]:
    """Improved Balle-et-al RDP→(ε,δ) conversion, min over orders -> (epsilon, best_order).

    ε(α) = ρ(α) − (ln δ + ln α)/(α−1) + ln((α−1)/α)   [NOT the looser Mironov form, §3.4].
    Uses np.nanargmin (some entries can be NaN/inf at large σ) and applies NO negative clamp
    (must match Opacus, which returns slightly-negative ε directly). Warns — does not raise —
    when the optimum sits on a grid boundary (grid too narrow), mirroring Opacus.
    """
    rdp = np.asarray(rdp, dtype=float)
    orders_vec = np.asarray(orders, dtype=float)
    # Exact term/operation order matters for the rel=1e-6 match.
    eps = (
        rdp
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )
    if np.all(np.isnan(eps)):
        return float("inf"), float("nan")
    idx = int(np.nanargmin(eps))
    if idx == 0 or idx == len(eps) - 1:
        warnings.warn(
            "Optimal RDP order is at the boundary of the grid; the reported ε may be loose. "
            "Widen the order grid.",
            stacklevel=2,
        )
    return float(eps[idx]), float(orders_vec[idx])


def account_run(num_queries, noise_multiplier, sample_rate, delta,
                orders=DEFAULT_ORDERS) -> float:
    """Total ε for `num_queries` identical Gaussian releases: compose then convert."""
    per_step = rdp_gaussian(orders, noise_multiplier, sample_rate)
    total_rdp = compose_rdp(per_step, num_queries)
    return rdp_to_epsilon(total_rdp, orders, delta)[0]


def noise_multiplier_for_epsilon(target_epsilon, num_queries, sample_rate, delta,
                                 orders=DEFAULT_ORDERS,
                                 sigma_bounds=(1e-3, 1e3), tol=1e-6) -> float:
    """Inverse calibration: smallest σ whose accounted ε <= target_epsilon.

    ε is strictly decreasing in σ (for q=1.0 every per-order term is α·num_queries/(2σ²),
    and the min of strictly-decreasing functions is strictly decreasing) -> monotone
    bisection. Raises if the target is unreachable within `sigma_bounds`. Consumed by
    `make_mechanism` (§4.2).
    """
    lo, hi = sigma_bounds
    # Suppress the boundary warning during search: intermediate probes at extreme σ routinely
    # land on the grid edge; only a caller's own account_run/rdp_to_epsilon should warn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps_lo = account_run(num_queries, lo, sample_rate, delta, orders)  # large ε (little noise)
        eps_hi = account_run(num_queries, hi, sample_rate, delta, orders)  # small ε (much noise)
        if eps_lo < target_epsilon:
            # Even the smallest σ already satisfies the target (target very loose).
            return lo
        if eps_hi > target_epsilon:
            raise ValueError(
                f"target ε={target_epsilon} unreachable within σ∈{sigma_bounds}: "
                f"ε(σ={hi})={eps_hi:.4f} > target. Widen sigma_bounds."
            )
        # Bisect for eps(σ) == target; ε decreasing in σ. Smallest σ with ε <= target.
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if account_run(num_queries, mid, sample_rate, delta, orders) > target_epsilon:
                lo = mid          # still over budget -> need more noise
            else:
                hi = mid          # within budget -> can try less noise
    return hi


def account_run_laplace(num_queries, l1_sensitivity, laplace_scale) -> float:
    """SECONDARY pure-ε basic composition for ONE homogeneous group of Laplace releases.

    ε = num_queries · l1_sensitivity / laplace_scale.

    `l1_sensitivity` is the per-release L1 = d · stat_L1 (L1 sums over features, NO root —
    §3.2); e.g. d·G_L1_PER_FEATURE for the gradient group. The G and H groups are accounted
    SEPARATELY and their ε summed by the caller (basic composition):
        ε_total = account_run_laplace(k, d·G_L1, b_G) + account_run_laplace(k, d·H_L1, b_H),
    with k = D·T. No δ, no RDP; NOT part of the Opacus gate.
    """
    return num_queries * l1_sensitivity / laplace_scale
