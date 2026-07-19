#!/usr/bin/env bash
# Negative security checks for the 1.e secure topology (spec §4.9/§6 step 4):
# prove security is actually ON by asserting that three attacks FAIL CLOSED
# within a bounded timeout. Exit non-zero if any attack succeeds or hangs.
#
#   1. SuperNode with an UNREGISTERED key  → rejected by the SuperLink
#      (asserted on the SuperLink's rejection log line + the rogue node never
#      reaching READY — not on exit codes, which gRPC retry loops make unreliable)
#   2. SuperNode pinning the WRONG CA      → TLS handshake fails
#   3. `flwr run` with insecure = true     → connection refused, and the tracked
#      pyproject.toml stays byte-identical (spec §3.9/§8 residual)
#
# Precondition: the secure topology is up (run_local_federation.sh start).
# Run via: scripts/run_local_federation.sh verify
#
# Log-line signatures are pinned against flwr 1.31 source AND a live rejection:
#   fleet_servicer.py:129  "[Fleet.ActivateNode] Activation failed: No SuperNode
#                           found with the given public key."
#   retry_invoker.py:370   "SSL/TLS handshake error detected."
set -euo pipefail

ARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ARCH_DIR}/.." && pwd)"
VENV_PY="${ARCH_DIR}/.venv/bin"
RUN_DIR="${ARCH_DIR}/.federation"
PYPROJECT="${ARCH_DIR}/pyproject.toml"
SUPERLINK_LOG="${RUN_DIR}/superlink.log"
DEADLINE=30  # seconds per check; a hang past this reads as FAIL, not PASS

# flower-supernode spawns flower-superexec by bare name → venv bin on PATH.
export PATH="${VENV_PY}:${PATH}"

READY_MARKER="SuperNode ID:"
REJECT_LINE="Activation failed: No SuperNode found with the given public key"
HANDSHAKE_LINE="SSL/TLS handshake error detected"

# Committed connection values (same single source as the launcher, spec §4.5).
eval "$(PYPROJECT="${PYPROJECT}" REPO_ROOT="${REPO_ROOT}" "${VENV_PY}/python" - <<'PY'
import os, tomllib
from pathlib import Path
with open(os.environ["PYPROJECT"], "rb") as f:
    sl = tomllib.load(f)["tool"]["fed_stroke"]["superlink"]
print(f'SL_NAME="{sl["name"]}"')
print(f'SL_ADDR="{sl["address"]}"')
print(f'SL_FLEET="{sl["fleet-address"]}"')
ca = (Path(os.environ["REPO_ROOT"]) / sl["ca-cert"]).resolve()
print(f'SL_CA="{ca}"')
PY
)"

# Precondition: SuperLink up.
ctrl_host="${SL_ADDR%:*}"; ctrl_port="${SL_ADDR##*:}"
if ! "${VENV_PY}/python" -c "import socket; socket.create_connection(('${ctrl_host}', ${ctrl_port}), 2).close()" 2>/dev/null; then
    echo "ERROR: SuperLink not reachable at ${SL_ADDR} — run 'run_local_federation.sh start' first." >&2
    exit 2
fi
[ -f "${SUPERLINK_LOG}" ] || { echo "ERROR: ${SUPERLINK_LOG} not found." >&2; exit 2; }

TMP="$(mktemp -d)"
ROGUE_PIDS=()
cleanup() {
    for pid in "${ROGUE_PIDS[@]:-}"; do
        [ -n "${pid}" ] && kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    rm -rf "${TMP}"
}
trap cleanup EXIT

FAILURES=0
result() {  # name PASS|FAIL detail
    if [ "$2" = PASS ]; then
        echo "PASS  $1  ($3)"
    else
        echo "FAIL  $1  ($3)" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

# Wait until CONDITION-cmd succeeds or the rogue log goes READY or deadline.
# Echoes: "ok" | "ready" | "timeout".
wait_reject() {  # rogue_log condition_cmd...
    local rogue_log="$1"; shift
    local end=$((SECONDS + DEADLINE))
    while [ "${SECONDS}" -lt "${end}" ]; do
        if grep -q "${READY_MARKER}" "${rogue_log}" 2>/dev/null; then echo ready; return; fi
        if "$@" >/dev/null 2>&1; then echo ok; return; fi
        sleep 0.5
    done
    echo timeout
}

echo "Negative security checks against ${SL_NAME} (${SL_ADDR} / fleet ${SL_FLEET}), ${DEADLINE}s deadline each"
echo

# --- Check 1: unregistered node-auth key must be rejected by the SuperLink ---
ssh-keygen -q -t ecdsa -b 384 -N "" -C "rogue unregistered" -f "${TMP}/rogue_key"
offset=$(wc -c < "${SUPERLINK_LOG}")
setsid "${VENV_PY}/flower-supernode" \
    --root-certificates "${SL_CA}" \
    --superlink "${SL_FLEET}" \
    --auth-supernode-private-key "${TMP}/rogue_key" \
    --clientappio-api-address 127.0.0.1:9096 \
    > "${TMP}/rogue_unregistered.log" 2>&1 &
ROGUE_PIDS+=($!)
verdict=$(wait_reject "${TMP}/rogue_unregistered.log" \
    bash -c "tail -c +$((offset + 1)) '${SUPERLINK_LOG}' | grep -q '${REJECT_LINE}'")
kill -TERM -- "-${ROGUE_PIDS[-1]}" 2>/dev/null || true
case "${verdict}" in
    ok)      result "unregistered-key" PASS "SuperLink logged: ${REJECT_LINE}; node never READY" ;;
    ready)   result "unregistered-key" FAIL "rogue SuperNode reached READY with an unregistered key" ;;
    timeout) result "unregistered-key" FAIL "no rejection line in ${SUPERLINK_LOG} within ${DEADLINE}s" ;;
esac

# --- Check 2: wrong CA in --root-certificates must fail the TLS handshake ---
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 -nodes \
    -keyout "${TMP}/wrong_ca.key" -out "${TMP}/wrong_ca.crt" \
    -days 1 -subj "/CN=unrelated CA" 2>/dev/null
setsid "${VENV_PY}/flower-supernode" \
    --root-certificates "${TMP}/wrong_ca.crt" \
    --superlink "${SL_FLEET}" \
    --auth-supernode-private-key "${TMP}/rogue_key" \
    --clientappio-api-address 127.0.0.1:9097 \
    > "${TMP}/rogue_wrong_ca.log" 2>&1 &
ROGUE_PIDS+=($!)
verdict=$(wait_reject "${TMP}/rogue_wrong_ca.log" \
    grep -q "${HANDSHAKE_LINE}" "${TMP}/rogue_wrong_ca.log")
kill -TERM -- "-${ROGUE_PIDS[-1]}" 2>/dev/null || true
case "${verdict}" in
    ok)      result "wrong-ca" PASS "SuperNode logged: ${HANDSHAKE_LINE}; never READY" ;;
    ready)   result "wrong-ca" FAIL "rogue SuperNode reached READY under the wrong CA" ;;
    timeout) result "wrong-ca" FAIL "no TLS handshake error in log within ${DEADLINE}s" ;;
esac

# --- Check 3: insecure `flwr run` against the TLS SuperLink must be refused ---
# (flwr also refuses insecure+root-certificates combined at config-parse time,
# config_utils — a half-secure connection can't even be constructed.)
mkdir -p "${TMP}/flwrhome"
cat > "${TMP}/flwrhome/config.toml" <<EOF
[superlink]
default = "insecure-test"

[superlink.insecure-test]
address = "${SL_ADDR}"
insecure = true
EOF
pyproject_before=$(sha256sum "${PYPROJECT}")
set +e
run_out=$(cd "${ARCH_DIR}" && FLWR_HOME="${TMP}/flwrhome" timeout "${DEADLINE}" \
    "${VENV_PY}/flwr" run . insecure-test 2>&1)
run_rc=$?
set -e
UNAVAILABLE_LINE="Connection to the SuperLink is unavailable"
if [ "${run_rc}" -eq 0 ]; then
    result "insecure-flwr-run" FAIL "flwr run over plaintext against the TLS port succeeded"
elif [ "${run_rc}" -eq 124 ]; then
    result "insecure-flwr-run" FAIL "flwr run hung past ${DEADLINE}s instead of failing closed"
elif grep -q "${UNAVAILABLE_LINE}" <<<"${run_out}"; then
    result "insecure-flwr-run" PASS "exit ${run_rc}: ${UNAVAILABLE_LINE}"
else
    result "insecure-flwr-run" FAIL "exit ${run_rc} but without the expected '${UNAVAILABLE_LINE}' message"
fi

# §8 residual: `flwr run` must leave the tracked pyproject.toml byte-identical
# (guards against a future [tool.flwr.federations] block re-triggering migrate).
if [ "$(sha256sum "${PYPROJECT}")" = "${pyproject_before}" ]; then
    result "pyproject-untouched" PASS "flwr run left architecture/pyproject.toml byte-identical"
else
    result "pyproject-untouched" FAIL "flwr run mutated the tracked architecture/pyproject.toml"
fi

echo
if [ "${FAILURES}" -eq 0 ]; then
    echo "All negative checks failed closed — security is ON."
else
    echo "${FAILURES} check(s) did NOT fail closed." >&2
    exit 1
fi
