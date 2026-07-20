"""fed_stroke.dp: THE GATE — our RDP accountant must equal Opacus (spec 1.1 §4.7, §7.1).

Plain-pytest idiom of tests/test_metrics.py. q=1.0 is the SOLE exact gate (rel=1e-6); q<1 is
asserted only in the conservative direction (never under-report), since the prototype forces
q=1.0 (§3.5, decision F6). The order grid and the improved-Balle conversion are what make the
match exact (§3.4) — a wrong grid (e.g. range(11,64)) or the looser Mironov form fails here.
"""
import importlib.util
import os

import pytest

from fed_stroke.dp import accounting as A

# The accountant legitimately warns when the optimal RDP order hits the grid boundary (extreme ε);
# that is not a failure and would otherwise clutter the run.
pytestmark = pytest.mark.filterwarnings("ignore:Optimal RDP order")

DELTA = 1e-5


def test_default_orders_match_opacus():
    """DEFAULT_ORDERS must equal RDPAccountant.DEFAULT_ALPHAS exactly (range(12,64), NOT 11)."""
    pytest.importorskip("opacus")
    from opacus.accountants import RDPAccountant
    assert tuple(A.DEFAULT_ORDERS) == tuple(RDPAccountant.DEFAULT_ALPHAS)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("steps", [1, 20, 100])
def test_epsilon_matches_opacus_q1(sigma, steps):
    """HARD GATE: q=1.0, our ε == Opacus ε within rel=1e-6."""
    pytest.importorskip("opacus")
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=1.0)
    eps_ref = acc.get_epsilon(delta=DELTA)
    eps_ours = A.account_run(steps, sigma, 1.0, DELTA)
    assert eps_ours == pytest.approx(eps_ref, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("steps", [1, 20, 100])
def test_epsilon_conservative_when_subsampled(sigma, steps):
    """q<1: assert ONLY the conservative direction (never under-report), NOT exact equality (F6).
    The prototype forces q=1.0, so the subsampled-Gaussian bound is not gate-critical — it just
    must never understate ε versus Opacus."""
    pytest.importorskip("opacus")
    from opacus.accountants import RDPAccountant
    q = 0.8
    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=q)
    eps_ref = acc.get_epsilon(delta=DELTA)
    eps_ours = A.account_run(steps, sigma, q, DELTA)
    assert eps_ours >= eps_ref - 1e-9


@pytest.mark.parametrize("eps", [1, 3, 5, 10])
@pytest.mark.parametrize("m", [80, 160])
def test_inverse_calibration_roundtrip(eps, m):
    """noise_multiplier_for_epsilon round-trips to the target ε and never over-spends.
    m ∈ {80, 160}: 160 is the real federated 2·D·T Gaussian-release count (§3.3); 80 exercises
    the level count. These are accountant UNIT inputs; the mechanism always feeds 2·D·T."""
    sigma = A.noise_multiplier_for_epsilon(eps, m, 1.0, DELTA)
    back = A.account_run(m, sigma, 1.0, DELTA)
    assert back <= eps + 1e-6                       # never over-spend
    assert back == pytest.approx(eps, rel=1e-4)


def test_noise_multiplier_decreasing_in_epsilon():
    sig = [A.noise_multiplier_for_epsilon(e, 160, 1.0, DELTA) for e in (1, 3, 5, 10)]
    assert all(sig[i] > sig[i + 1] for i in range(len(sig) - 1))


def test_rdp_gaussian_closed_form():
    """Per-step Gaussian RDP at q=1.0 is exactly α/(2σ²)."""
    for sigma in (0.5, 1.0, 2.0):
        for a in (2.0, 5.0, 32.0):
            assert A.rdp_gaussian(a, sigma, 1.0) == pytest.approx(a / (2 * sigma ** 2))


def test_account_run_monotone_in_sigma_and_k():
    e_sigma = [A.account_run(60, s, 1.0, DELTA) for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(e_sigma[i] > e_sigma[i + 1] for i in range(len(e_sigma) - 1))
    e_k = [A.account_run(k, 1.0, 1.0, DELTA) for k in (10, 60, 120)]
    assert all(e_k[i] < e_k[i + 1] for i in range(len(e_k) - 1))


def test_ci_requires_opacus():
    """In CI a missing dev group must HARD-FAIL so a broken gate can't be masked; locally the
    importorskip guards above just skip (spec §4.7)."""
    if os.environ.get("CI"):
        assert importlib.util.find_spec("opacus") is not None, \
            "opacus (dev group) missing in CI — the equivalence gate cannot run"
