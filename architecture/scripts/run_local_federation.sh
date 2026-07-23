#!/usr/bin/env bash
# Stand up the SECURE local 1-SuperLink / 2-SuperNode Flower topology
# (roadmap 1.e, docs/specs/1e_secure_topology.md): server-side TLS with CA
# pinning + EC public-key node authentication — Flower 1.31's supported
# realization of "mTLS" (spec §3.6): the channel is encrypted, the SuperNodes
# verify the SuperLink (pinned CA), and the SuperLink authenticates each
# SuperNode (registered signing key).
#
# Each SuperNode is pinned to one Geneva half via its own node-config
# `data-path`, which `load_data_gva` reads. Simulation mode can't do this
# (shared node-config), so deployment mode is required.
#
# All ports/addresses/paths come from ONE committed source: the
# [tool.fed_stroke.superlink] and [tool.fed_stroke.nodes] tables in
# architecture/pyproject.toml (spec §4.5/§4.6). On every `start` this script
# materializes the [superlink.local-deployment] entry in ~/.flwr/config.toml
# from those values (merge-upsert — other connections in that file are
# preserved) and (re-)registers both node keys with the SuperLink.
#
# Secret material lives in the gitignored architecture/.secrets/ and is
# generated on demand by scripts/gen_certs.sh (rotation procedure in its
# header). Never commit anything from that directory.
#
# Usage:
#   scripts/run_local_federation.sh start   # certs → config → superlink → register → supernodes
#   scripts/run_local_federation.sh stop    # kill everything started by 'start'
#   scripts/run_local_federation.sh status  # show tracked PIDs
#   scripts/run_local_federation.sh verify  # negative security checks (fail-closed proofs)
#
# After 'start', run the matrix from architecture/ against the
# `local-deployment` connection, e.g.:
#   flwr run . local-deployment --run-config "save-model=true"
#
# NOTE: `flwr run` bundles this app dir into a FAB and reads THIS dir's
# .gitignore (not the repo-root one) to decide what to include, then rejects any
# path deeper than 10. The `.venv/` entry in architecture/.gitignore is what
# keeps the venv out of the bundle — without it the FAB build fails with
# "exceeds the maximum directory depth of 10". The `.secrets/` entry likewise
# keeps the keys out of the FAB. See that file's header.
#
# Also pass metrics-dir as an ABSOLUTE path (like model-dir): both are relative
# to the ServerApp CWD by default, which is not this directory, so a relative
# value writes the artifact somewhere unexpected. e.g.:
#   flwr run . local-deployment --run-config \
#     "save-model=true model-dir='${PWD}/../out/models' metrics-dir='${PWD}/../out/metrics'"
set -euo pipefail

ARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ARCH_DIR}/.." && pwd)"
VENV_PY="${ARCH_DIR}/.venv/bin"
RUN_DIR="${ARCH_DIR}/.federation"
PID_FILE="${RUN_DIR}/pids"
PYPROJECT="${ARCH_DIR}/pyproject.toml"
DB_PATH="${RUN_DIR}/state.db"   # SQLite LinkState; recreated on every `start` (see start())
MODELS_DIR="${REPO_ROOT}/out/models"  # saved final_model.json lands here (absolute → CWD-independent)

# Read the committed [tool.fed_stroke.*] tables and export them as shell vars.
# Single source of truth for every port/address/path (spec §4.5): SL_* for the
# SuperLink connection, NODE_<name>_{DATA,APPIO,KEY} per node, SECRETS_DIR
# derived from the committed ca-cert path.
load_committed_config() {
    eval "$(PYPROJECT="${PYPROJECT}" REPO_ROOT="${REPO_ROOT}" "${VENV_PY}/python" - <<'PY'
import os, tomllib
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
with open(os.environ["PYPROJECT"], "rb") as f:
    cfg = tomllib.load(f)["tool"]["fed_stroke"]

sl = cfg["superlink"]
for k in ("name", "address", "fleet-address", "ca-cert"):
    if k not in sl:
        raise SystemExit(f"pyproject [tool.fed_stroke.superlink] missing key: {k}")
ca = (repo_root / sl["ca-cert"]).resolve()
ctrl_host, ctrl_port = sl["address"].rsplit(":", 1)
fleet_host, fleet_port = sl["fleet-address"].rsplit(":", 1)

print(f'SL_NAME="{sl["name"]}"')
print(f'SL_ADDR="{sl["address"]}"')
print(f'SL_FLEET="{sl["fleet-address"]}"')
print(f'SL_CTRL_HOST="{ctrl_host}"'); print(f'SL_CTRL_PORT="{ctrl_port}"')
print(f'SL_FLEET_HOST="{fleet_host}"'); print(f'SL_FLEET_PORT="{fleet_port}"')
print(f'SL_CA="{ca}"')
print(f'SECRETS_DIR="{ca.parent}"')

nodes = cfg["nodes"]
print(f'NODES="{" ".join(nodes)}"')
for name, node in nodes.items():
    for k in ("data-path", "appio", "key"):
        if k not in node:
            raise SystemExit(f"pyproject [tool.fed_stroke.nodes.{name}] missing key: {k}")
    data = (repo_root / node["data-path"]).resolve()
    print(f'NODE_{name}_DATA="{data}"')
    print(f'NODE_{name}_APPIO="{node["appio"]}"')
    print(f'NODE_{name}_KEY="{node["key"]}"')
    # dp-site-weight (spec 1.1.a″ R5): fixed public per-site weight the DP train reply
    # sends instead of the exact training count. Optional; client defaults to 1.
    print(f'NODE_{name}_DPWEIGHT="{node.get("dp-site-weight", 1)}"')
PY
)"
}

# Generate the secret material if any of it is missing (one-command operator
# flow, spec §8 — the same contract Shenzhen inherits in 1.3.a).
ensure_certs() {
    local missing=0
    for f in ca.crt server.pem server.key; do
        [ -f "${SECRETS_DIR}/${f}" ] || missing=1
    done
    for n in ${NODES}; do
        local kvar="NODE_${n}_KEY"
        [ -f "${SECRETS_DIR}/${!kvar}" ] && [ -f "${SECRETS_DIR}/${!kvar}.pub" ] || missing=1
    done
    if [ "${missing}" -eq 1 ]; then
        echo "Secret material incomplete → running gen_certs.sh"
        "${ARCH_DIR}/scripts/gen_certs.sh" "${SECRETS_DIR}"
    fi
}

# Materialize [superlink.<SL_NAME>] in ~/.flwr/config.toml (or $FLWR_HOME) from
# the committed values, via flwr's own merge-upsert helper — it replaces only
# this one entry and preserves any other [superlink.*] connections (spec §4.5).
# flwr.cli.flower_config/typing are internal APIs; acceptable because the
# project pins flwr 1.31.0 and the spec is verified against that exact source.
write_flwr_connection() {
    PYPROJECT="${PYPROJECT}" REPO_ROOT="${REPO_ROOT}" "${VENV_PY}/python" - <<'PY'
import os, tomllib
from pathlib import Path
from flwr.cli.flower_config import write_superlink_connection
from flwr.cli.typing import SuperLinkConnection

repo_root = Path(os.environ["REPO_ROOT"])
with open(os.environ["PYPROJECT"], "rb") as f:
    sl = tomllib.load(f)["tool"]["fed_stroke"]["superlink"]
ca = str((repo_root / sl["ca-cert"]).resolve())  # must be absolute (typing.py:166)
write_superlink_connection(SuperLinkConnection(
    name=sl["name"], address=sl["address"], root_certificates=ca))
print(f'Wrote [superlink.{sl["name"]}] (address={sl["address"]}, root-certificates={ca})')
PY
}

# Bounded TCP-connect poll; fails fast if the daemon died. TLS ports accept TCP
# before the handshake, so a plain connect proves the listener is up.
wait_for_port() {  # host port deadline_s name pid
    local host="$1" port="$2" deadline="$3" name="$4" pid="$5"
    local end=$((SECONDS + deadline))
    until "${VENV_PY}/python" -c "import socket; socket.create_connection(('${host}', ${port}), 1).close()" 2>/dev/null; do
        kill -0 "${pid}" 2>/dev/null || { echo "ERROR: ${name} died during startup; see ${RUN_DIR}/${name}.log" >&2; exit 1; }
        [ "${SECONDS}" -ge "${end}" ] && { echo "ERROR: ${name} not listening on ${host}:${port} after ${deadline}s" >&2; exit 1; }
        sleep 0.5
    done
}

# Bounded wait for a log line; fails fast if the daemon died.
wait_for_log() {  # logfile pattern deadline_s name pid
    local logfile="$1" pattern="$2" deadline="$3" name="$4" pid="$5"
    local end=$((SECONDS + deadline))
    until grep -q "${pattern}" "${logfile}" 2>/dev/null; do
        kill -0 "${pid}" 2>/dev/null || { echo "ERROR: ${name} died during startup; see ${logfile}" >&2; exit 1; }
        [ "${SECONDS}" -ge "${end}" ] && { echo "ERROR: ${name} not ready ('${pattern}') after ${deadline}s; see ${logfile}" >&2; exit 1; }
        sleep 0.5
    done
}

# Register every node key with the running SuperLink — unconditionally on
# every `start` (spec §4.3). NB: `--format json` exits 0 even on errors; the
# JSON body is the verdict, so parse it instead of the exit code.
register_nodes() {
    local n kvar out verdict
    for n in ${NODES}; do
        kvar="NODE_${n}_KEY"
        out=$("${VENV_PY}/flwr" supernode register "${SECRETS_DIR}/${!kvar}.pub" "${SL_NAME}" --format json 2>&1) || true
        verdict=$("${VENV_PY}/python" -c '
import json, sys
try: d = json.load(sys.stdin)
except Exception: print("unparseable register output"); sys.exit()
print(d.get("node-id", "?") if d.get("success") else d.get("error-message", "unknown error").strip())' <<<"${out}")
        if [[ "${verdict}" =~ ^[0-9]+$ ]]; then
            echo "  ${n}: registered (node id ${verdict})"
        else
            echo "ERROR: registering ${n} failed: ${verdict}" >&2
            exit 1
        fi
    done
}

start() {
    mkdir -p "${RUN_DIR}"
    : > "${PID_FILE}"

    # flower-superlink/-supernode spawn helper subprocesses (flower-superexec,
    # the clientapp runner) by bare name, so the venv bin must be on PATH.
    export PATH="${VENV_PY}:${PATH}"

    load_committed_config
    for n in ${NODES}; do
        local dvar="NODE_${n}_DATA"
        [ -f "${!dvar}" ] || { echo "ERROR: data file missing: ${!dvar}" >&2; exit 1; }
    done
    ensure_certs
    write_flwr_connection

    # Fresh LinkState on every start. Empirical findings against flwr 1.31
    # (spec §3.5/§8 residual, resolved): the SQLite --database DOES persist
    # registered node keys across restarts, but persisted registrations cannot
    # be safely re-bound to our static per-node keys: (a) a node killed by
    # `stop` stays 'online' until heartbeat expiry (~1 min), and activate_node
    # only succeeds from registered/offline status, so a quick restart's
    # SuperNodes die with "could not be activated"; (b) node.public_key is
    # UNIQUE and `flwr supernode unregister` keeps the row (status
    # 'unregistered'), so a once-unregistered key can NEVER be re-registered in
    # the same database. Fresh state + the unconditional register below is the
    # robust restart contract; run artifacts live in out/, so nothing durable
    # is lost. (Key rotation therefore always generates a NEW key, §4.8.)
    rm -f "${DB_PATH}"

    # setsid detaches each daemon into its own session so it survives the shell
    # that launched this script (the daemons must outlive `start`).
    echo "Starting SuperLink (TLS + node auth) → logs ${RUN_DIR}/superlink.log"
    setsid "${VENV_PY}/flower-superlink" \
        --ssl-ca-certfile "${SECRETS_DIR}/ca.crt" \
        --ssl-certfile "${SECRETS_DIR}/server.pem" \
        --ssl-keyfile "${SECRETS_DIR}/server.key" \
        --enable-supernode-auth \
        --database "${DB_PATH}" \
        > "${RUN_DIR}/superlink.log" 2>&1 &
    local sl_pid=$!
    echo "superlink ${sl_pid}" >> "${PID_FILE}"
    wait_for_port "${SL_CTRL_HOST}" "${SL_CTRL_PORT}" 30 superlink "${sl_pid}"
    wait_for_port "${SL_FLEET_HOST}" "${SL_FLEET_PORT}" 30 superlink "${sl_pid}"

    echo "Registering SuperNode keys with the SuperLink:"
    register_nodes

    local n dvar avar kvar pid
    for n in ${NODES}; do
        dvar="NODE_${n}_DATA"; avar="NODE_${n}_APPIO"; kvar="NODE_${n}_KEY"; wvar="NODE_${n}_DPWEIGHT"
        echo "Starting SuperNode ${n} ($(basename "${!dvar}")) → logs ${RUN_DIR}/supernode_${n}.log"
        setsid "${VENV_PY}/flower-supernode" \
            --root-certificates "${SECRETS_DIR}/ca.crt" \
            --superlink "${SL_FLEET}" \
            --auth-supernode-private-key "${SECRETS_DIR}/${!kvar}" \
            --clientappio-api-address "${!avar}" \
            --node-config "data-path=\"${!dvar}\" dp-site-weight=${!wvar}" \
            > "${RUN_DIR}/supernode_${n}.log" 2>&1 &
        pid=$!
        echo "supernode_${n} ${pid}" >> "${PID_FILE}"
    done

    while read -r name pid; do
        [ "${name}" = superlink ] && continue
        wait_for_log "${RUN_DIR}/${name}.log" "SuperNode ID:" 30 "${name}" "${pid}"
    done < "${PID_FILE}"

    echo
    echo "Secure topology up (TLS + node auth). Tracked PIDs:"
    cat "${PID_FILE}"
    echo
    echo "Models will be saved to ${MODELS_DIR}"
    echo "Run the matrix from ${ARCH_DIR}:"
    local md="model-dir='${MODELS_DIR}'"
    echo "  flwr run . ${SL_NAME} --run-config \"save-model=true ${md}\""
    echo "  flwr run . ${SL_NAME} --run-config \"train-method='cyclic' save-model=true ${md}\""
    echo "  flwr run . ${SL_NAME} --run-config \"train-method='cyclic' cyclic-order='reverse' save-model=true ${md}\""
}

stop() {
    [ -f "${PID_FILE}" ] || { echo "No PID file at ${PID_FILE}"; return 0; }
    while read -r name pid; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "Killing ${name} (${pid}) and its process group"
            # setsid made each daemon a group leader (pgid == pid); killing the
            # group also reaps the flower-superexec helper subprocesses.
            kill -TERM -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        fi
    done < "${PID_FILE}"
    rm -f "${PID_FILE}"
}

status() {
    [ -f "${PID_FILE}" ] || { echo "No tracked processes"; return 0; }
    while read -r name pid; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "${name} ${pid} RUNNING"
        else
            echo "${name} ${pid} DEAD"
        fi
    done < "${PID_FILE}"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    verify) exec "${ARCH_DIR}/scripts/verify_negative_security.sh" ;;
    # Internal: materialize the ~/.flwr/config.toml connection entry only
    # (used by tests via FLWR_HOME to prove the merge-not-truncate invariant).
    write-config) load_committed_config; write_flwr_connection ;;
    *) echo "Usage: $0 {start|stop|status|verify}" >&2; exit 1 ;;
esac
