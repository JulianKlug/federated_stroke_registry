"""Cross-run privacy ledger tests (spec 1.1.a″ R6).

Pins: append-only jsonl round-trip, the identity-refusal guard, compose-of-one == the per-run
ε (accounting.account_run), compose-of-N == Opacus on the same heterogeneous schedule (the
same equivalence gate as tests/dp/test_accounting.py), per-site grouping (patient-disjoint
sites never compose across), and linear (basic-composition) addition of Laplace entries.
"""
import json

import pytest

from fed_stroke.dp import accounting as A
from fed_stroke.dp import ledger as L

pytestmark = pytest.mark.filterwarnings("ignore:Optimal RDP order")

DELTA = 1e-5


def _append_gaussian(path, site="node_A", sigma=12.05, k=160):
    eps = A.account_run(k, sigma, 1.0, DELTA)
    return L.append_entry(path, site=site, mechanism="gaussian", num_releases=k,
                          noise_multiplier=sigma, epsilon=eps, delta=DELTA,
                          run_config_hash="cfg-" + site)


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "dp_ledger.jsonl"
    e1 = _append_gaussian(path, sigma=12.05, k=160)
    e2 = _append_gaussian(path, sigma=18.89, k=160)
    entries = L.read_entries(path)
    assert entries == [e1, e2]                       # append order preserved
    assert len(path.read_text().strip().splitlines()) == 2
    # jsonl: every line independently parseable (append-only, no rewrite).
    for line in path.read_text().strip().splitlines():
        json.loads(line)


def test_append_refuses_identity(tmp_path):
    path = tmp_path / "dp_ledger.jsonl"
    with pytest.raises(ValueError, match="identity"):
        L.append_entry(path, site="node_A", mechanism="identity", num_releases=0,
                       noise_multiplier=float("inf"), epsilon=float("inf"), delta=DELTA,
                       run_config_hash="x")
    assert not path.exists()                         # refused BEFORE any write


def test_empty_or_missing_ledger_totals_empty(tmp_path):
    assert L.ledger_total(tmp_path / "nope.jsonl", DELTA) == {}


def test_compose_of_one_equals_per_run_epsilon(tmp_path):
    path = tmp_path / "dp_ledger.jsonl"
    sigma, k = 12.050053853763266, 160               # the ε=5 row of the calibrated σ table
    _append_gaussian(path, sigma=sigma, k=k)
    totals = L.ledger_total(path, DELTA)
    assert totals["node_A"] == pytest.approx(A.account_run(k, sigma, 1.0, DELTA), rel=1e-9)


def test_compose_of_n_matches_opacus_schedule(tmp_path):
    """The composed ledger total equals Opacus's RDPAccountant on the same heterogeneous
    (σ_i, k_i) schedule at q=1.0 — the R6 acceptance gate."""
    pytest.importorskip("opacus")
    from opacus.accountants import RDPAccountant

    path = tmp_path / "dp_ledger.jsonl"
    schedule = [(51.170527747745616, 160), (18.88773447119351, 160),
                (12.050053853763266, 80)]
    acc = RDPAccountant()
    for sigma, k in schedule:
        _append_gaussian(path, sigma=sigma, k=k)
        for _ in range(k):
            acc.step(noise_multiplier=sigma, sample_rate=1.0)
    eps_ref = acc.get_epsilon(delta=DELTA)
    totals = L.ledger_total(path, DELTA)
    assert totals["node_A"] == pytest.approx(eps_ref, rel=1e-6, abs=1e-9)


def test_totals_are_composed_per_site_never_across(tmp_path):
    """Sites hold patient-disjoint cohorts: node_B's spend must not inflate node_A's total
    (a shared loopback ledger file would otherwise double-count per-patient ε)."""
    path = tmp_path / "dp_ledger.jsonl"
    _append_gaussian(path, site="node_A", sigma=12.05, k=160)
    _append_gaussian(path, site="node_B", sigma=6.7, k=80)
    totals = L.ledger_total(path, DELTA)
    assert totals["node_A"] == pytest.approx(A.account_run(160, 12.05, 1.0, DELTA), rel=1e-9)
    assert totals["node_B"] == pytest.approx(A.account_run(80, 6.7, 1.0, DELTA), rel=1e-9)


def test_laplace_entries_add_linearly(tmp_path):
    path = tmp_path / "dp_ledger.jsonl"
    _append_gaussian(path, sigma=12.05, k=160)
    L.append_entry(path, site="node_A", mechanism="laplace", num_releases=120,
                   noise_multiplier=float("nan"), epsilon=2.0, delta=DELTA,
                   run_config_hash="lap")
    totals = L.ledger_total(path, DELTA)
    gaussian_part = A.account_run(160, 12.05, 1.0, DELTA)
    assert totals["node_A"] == pytest.approx(gaussian_part + 2.0, rel=1e-9)
    # Laplace's nan multiplier is stored as null, never a JSON-breaking NaN.
    lap = [e for e in L.read_entries(path) if e["mechanism"] == "laplace"][0]
    assert lap["noise_multiplier"] is None


def test_config_hash_is_canonical(tmp_path):
    a = L.config_hash({"dp": {"target_epsilon": 5.0}, "params": {"max_depth": 4}})
    b = L.config_hash({"params": {"max_depth": 4}, "dp": {"target_epsilon": 5.0}})
    c = L.config_hash({"params": {"max_depth": 3}, "dp": {"target_epsilon": 5.0}})
    assert a == b != c
