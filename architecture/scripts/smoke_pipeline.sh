#!/usr/bin/env bash
# End-to-end smoke run of the federated stroke pipeline (roadmap 1.f,
# docs/specs/1f_docker_smoke_build.md §4.4). This is BOTH the container CMD and
# the local smoke command — it runs in and out of Docker.
#
# It does NOT fork any orchestration: it drives the existing secure-topology
# launcher (run_local_federation.sh, roadmap 1.e) unchanged, then submits one
# bagging run over the TLS + node-auth channel and asserts the metrics artifact.
#
# Fail-fast (set -e), always-clean (trap … stop). Every path is derived from this
# script's own location, so the CMD works regardless of the caller's CWD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ARCH_DIR}/.." && pwd)"     # in-container: /workspace
VENV_BIN="${ARCH_DIR}/.venv/bin"              # same idiom as run_local_federation.sh:50
LAUNCHER="${SCRIPT_DIR}/run_local_federation.sh"
PYPROJECT="${ARCH_DIR}/pyproject.toml"

# --- 1. Preflight: the mounted Geneva halves must be present -----------------
# Half names are read from the committed [tool.fed_stroke.nodes].*.data-path so
# this check can never drift from the node config the SuperNodes actually use.
# Emit space-separated basenames (e.g. "geneva_half_A.parquet geneva_half_B.parquet").
HALVES="$(PYPROJECT="${PYPROJECT}" "${VENV_BIN}/python" - <<'PY'
import os, tomllib
from pathlib import Path
with open(os.environ["PYPROJECT"], "rb") as f:
    nodes = tomllib.load(f)["tool"]["fed_stroke"]["nodes"]
print(" ".join(Path(n["data-path"]).name for n in nodes.values()))
PY
)"

missing=0
for name in ${HALVES}; do
    [ -f "${REPO_ROOT}/out/${name}" ] || { echo "MISSING: ${REPO_ROOT}/out/${name}" >&2; missing=1; }
done
if [ "${missing}" -eq 1 ]; then
    cat >&2 <<EOF

ERROR: the Geneva data halves are not present at ${REPO_ROOT}/out (see roadmap 1.a).
The image never bakes patient data — mount the halves at run time:

    docker run --rm --init -v "<repo>/out:/workspace/out" fed-stroke-smoke

EOF
    exit 1
fi

# --- 2. Always tear the topology down (success, failure, or signal) ----------
trap '"${LAUNCHER}" stop' EXIT

# --- 3. Bring up the 1.e secure topology (TLS + node auth) -------------------
# ensure_certs → gen_certs.sh mints the CA/server cert + node keys on first run;
# SuperLink comes up over TLS, both SuperNodes register and reach READY.
"${LAUNCHER}" start

# --- 4. One bagging run — the smoke gate -------------------------------------
# --stream blocks until the run reaches a terminal state: `flwr run` submits and
# returns immediately by default (flwr 1.31), so without it step 5 would race the
# artifact write. model-dir/metrics-dir default to the ServerApp CWD (not this
# dir), so pass them ABSOLUTE. Default train-method is bagging → the ServerApp
# writes ${REPO_ROOT}/out/metrics/bagging.json.
cd "${ARCH_DIR}"
"${VENV_BIN}/flwr" run . local-deployment --stream --run-config \
    "save-model=true model-dir='${REPO_ROOT}/out/models' metrics-dir='${REPO_ROOT}/out/metrics'"

# --- 5. Assert the artifact — the smoke gate's OWN assertions ----------------
# NOT validate_metrics_artifact (it requires int round keys; JSON round-trips them
# to strings, so it would raise on a valid file, metrics.py:290). We assert the
# file landed and carries a real two-site result: both expected sites in the last
# round, each auc_roc a finite float (null = a degenerate single-class site).
PYPROJECT="${PYPROJECT}" METRICS="${REPO_ROOT}/out/metrics/bagging.json" \
    "${VENV_BIN}/python" - <<'PY'
import json, math, os, sys, tomllib
from pathlib import Path

artifact = Path(os.environ["METRICS"])
if not artifact.is_file():
    sys.exit(f"FAIL: metrics artifact not written: {artifact}")
try:
    data = json.loads(artifact.read_text())
except json.JSONDecodeError as exc:
    sys.exit(f"FAIL: {artifact} is not valid JSON: {exc}")
if not data:
    sys.exit(f"FAIL: {artifact} has no rounds")

# Expected site keys = the parquet basenames the SuperNodes report as their site
# id (Path(node_config["data-path"]).name, client_app.py:86,159). Derived from the
# committed node config, not hardcoded.
with open(os.environ["PYPROJECT"], "rb") as f:
    nodes = tomllib.load(f)["tool"]["fed_stroke"]["nodes"]
expected = {Path(n["data-path"]).name for n in nodes.values()}

last = data[max(data, key=lambda k: int(k))]  # JSON round keys are strings
missing = expected - last.keys()
if missing:
    sys.exit(f"FAIL: last round missing site(s) {sorted(missing)}; got {sorted(last)}")

aucs = {}
for site in expected:
    auc = last[site].get("auc_roc")
    if not isinstance(auc, (int, float)) or isinstance(auc, bool) or math.isnan(auc):
        sys.exit(f"FAIL: site {site} auc_roc is not a finite float: {auc!r}")
    aucs[site] = auc

summary = ", ".join(f"{site}={aucs[site]:.4f}" for site in sorted(expected))
print(f"OK: {artifact} — both sites present, finite AUC ({summary})")
PY

# --- 6. Success (the trap stops the federation) ------------------------------
echo "Smoke run passed."
