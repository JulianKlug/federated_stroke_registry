"""fed_stroke.dp.ledger: append-only cross-run privacy ledger (spec 1.1.a″ R6).

The 1.1.b sweep releases models at several ε over the SAME patients; the per-run ε is not the
guarantee for the collection — composition applies. This ledger records every real-data DP run
(one JSON line per run) and `ledger_total` composes all recorded runs' RDP curves and converts
ONCE, reusing `accounting.py`'s verified compose/convert functions verbatim (no new math).

Semantics:
- One entry per (site, run). Entries record INTENT-TO-SPEND: the client appends on its FIRST
  DP train call of a run, when the full authorized (k, σ, ε) is already known (σ is
  run-calibrated) — a run that crashes mid-way has still spent noise, so recording upfront is
  conservative in exactly the right direction. Appending at the end would also miss cyclic's
  non-final site.
- Composition is PER SITE: sites hold patient-disjoint cohorts, so the per-patient budget
  composes within a site, never across sites. A shared ledger file (e.g. the loopback
  deployment's common CWD) must therefore never be summed blindly.
- Gaussian runs compose in RDP space (compose_rdp over heterogeneous (rdp_vec, count) pairs,
  one rdp_to_epsilon conversion at the end). Laplace runs add their pure ε linearly (basic
  composition) on top — valid and conservative. Identity (arm B) runs spend nothing and are
  never appended.

Reporting rule (R6): every artifact reporting a per-run ε also states the composed
ledger-total ε to date.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fed_stroke.dp.accounting import (
    DEFAULT_ORDERS,
    compose_rdp,
    rdp_gaussian,
    rdp_to_epsilon,
)

_LEDGER_MECHANISMS = {"gaussian", "laplace"}


def config_hash(run_identity: dict) -> str:
    """Stable sha1 over the canonical-JSON run identity (same idiom as hpo.trial_id)."""
    return hashlib.sha1(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def append_entry(path, *, site: str, mechanism: str, num_releases: int,
                 noise_multiplier: float, epsilon: float, delta: float,
                 run_config_hash: str, date: str | None = None) -> dict:
    """Append one run's spend to the jsonl ledger (append-only; creates parents/file).

    `mechanism` must be a real noise mechanism — identity spends no ε and must never be
    appended (fail loud rather than silently pollute the composition).
    """
    if mechanism not in _LEDGER_MECHANISMS:
        raise ValueError(
            f"ledger entries record real DP spend only; got mechanism={mechanism!r} "
            f"(expected one of {sorted(_LEDGER_MECHANISMS)})."
        )
    entry = {
        "date": date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": str(site),
        "config_hash": str(run_config_hash),
        "mechanism": str(mechanism),
        "num_releases": int(num_releases),
        # Laplace has no Gaussian multiplier (nan) -> stored as None; composition for laplace
        # entries uses `epsilon` directly.
        "noise_multiplier": (None if noise_multiplier is None
                             or not np.isfinite(noise_multiplier)
                             else float(noise_multiplier)),
        "epsilon": float(epsilon),
        "delta": float(delta),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_entries(path) -> list[dict]:
    """All ledger entries, in append order. Missing file -> empty ledger."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ledger_total(path, delta: float, orders=DEFAULT_ORDERS) -> dict[str, float]:
    """Composed ε to date, PER SITE (see module docstring for why never across sites).

    Gaussian entries: Σ_runs k_i · rdp_gaussian(orders, σ_i) composed in RDP space, converted
    once via rdp_to_epsilon at `delta`. Laplace entries: pure-ε basic composition, added
    linearly. Returns {site: composed ε}; empty ledger -> {}.
    """
    entries = read_entries(path)
    totals: dict[str, float] = {}
    for site in sorted({e["site"] for e in entries}):
        site_entries = [e for e in entries if e["site"] == site]
        eps = 0.0
        gaussian = [
            (rdp_gaussian(orders, float(e["noise_multiplier"]), 1.0), int(e["num_releases"]))
            for e in site_entries if e["mechanism"] == "gaussian"
        ]
        if gaussian:
            eps += rdp_to_epsilon(compose_rdp(gaussian), orders, delta)[0]
        eps += sum(float(e["epsilon"])
                   for e in site_entries if e["mechanism"] == "laplace")
        totals[site] = eps
    return totals
