"""Shared driver commons for the process-boundary scripts (spec 1.1.b §1).

`run_hpo.py` (roadmap 1.1.a) and `run_dp_sweep.py` (roadmap 1.1.b) both drive the
real federated pipeline by firing `flwr run . <federation>` subprocesses against a
STANDING federation and reading pyproject-derived fixed facts. This module holds
the pieces they share verbatim: the flwr subprocess runner (wall-clock timeout +
process-group kill), the pyproject fact readers, the SuperLink TCP preflight, and
the strict-JSON sanitizer.

Lives in `scripts/`, NOT `fed_stroke/`: this is driver-side process infrastructure
that must not ship in the wheel to the Shenzhen SuperNode (the hatch wheel packages
`fed_stroke` only). Drivers reach it via a `sys.path` insert of the scripts dir.
"""
import math
import os
import signal
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

ARCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ARCH_DIR.parent
PYPROJECT = ARCH_DIR / "pyproject.toml"


# --------------------------------------------------------------------------- #
# pyproject-derived fixed facts (single committed source)
# --------------------------------------------------------------------------- #
def _load_pyproject():
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _superlink_hostport(cfg):
    addr = cfg["tool"]["fed_stroke"]["superlink"]["address"]
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _half_paths(cfg):
    """Absolute parquet paths for each pinned SuperNode half, resolved against the
    repo root (the ServerApp/SuperNode CWD, where `out/geneva_half_*.parquet` live).
    """
    nodes = cfg["tool"]["fed_stroke"]["nodes"]
    return [(REPO_ROOT / node["data-path"]).resolve() for node in nodes.values()]


def _operating_point(cfg):
    return cfg["tool"]["flwr"]["app"]["config"]["operating-point"]


def preflight_superlink(cfg):
    """TCP probe of the SuperLink Control API. Proves the listener is up, but NOT
    that both SuperNodes are alive/pinned — drivers follow with their own cheapest
    run (the HPO canary; the sweep's arm A). Raises SystemExit when unreachable.
    """
    host, port = _superlink_hostport(cfg)
    try:
        with socket.create_connection((host, port), timeout=5):
            pass
    except OSError as exc:
        raise SystemExit(
            f"cannot reach SuperLink Control-API {host}:{port} ({exc}). "
            "Bring the federation up: scripts/run_local_federation.sh start")


# --------------------------------------------------------------------------- #
# subprocess: one `flwr run`, bounded by a wall-clock timeout, group-killed
# --------------------------------------------------------------------------- #
def _flwr_bin():
    cand = Path(sys.executable).parent / "flwr"
    return str(cand) if cand.exists() else "flwr"


def _run_flwr(federation, run_config, timeout):
    """Fire one `flwr run . <federation> --stream --run-config <cfg>` with
    `shell=False` (a list, so flwr's TOML parser — not a shell — consumes the value
    quotes) and a wall-clock cap. Returns (ok, timed_out).

    `--stream` is load-bearing: plain `flwr run` submits the run and returns
    immediately (flwr 1.31), so without it the driver would race the ServerApp's
    model write. `--stream` blocks until the run reaches a terminal state (mirroring
    smoke_pipeline.sh).

    `build_strategy` sets `min_available_nodes=num-sites`, so a run whose federation
    lost a SuperNode BLOCKS indefinitely rather than exiting non-zero — hence the
    hard timeout. On timeout the child's whole process group is killed (it is its own
    session leader via `start_new_session=True`), mirroring
    `run_local_federation.sh`'s `kill -- -pgid`, so no orphaned superexec lingers.
    """
    cmd = [_flwr_bin(), "run", ".", federation, "--stream", "--run-config", run_config]
    proc = subprocess.Popen(
        cmd, cwd=str(ARCH_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            print(f"    flwr run exited {proc.returncode}; tail:\n"
                  + "\n".join(f"      {ln}" for ln in (out or "").splitlines()[-8:]))
        return proc.returncode == 0, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"    flwr run exceeded --run-timeout={timeout}s; killed process group")
        return False, True


# --------------------------------------------------------------------------- #
# JSON sanitation (strict artifacts: NaN and ±inf -> null)
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """Non-finite floats (NaN AND ±inf) -> None, recursively, so artifacts survive
    json.dumps(allow_nan=False). The ±inf case is defense-in-depth for the sweep:
    real saved DPBooster meta is already inf-scrubbed by to_json_bytes, so it only
    guards hand-built fixtures and any future non-scrubbed path (spec 1.1.b §1).
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
