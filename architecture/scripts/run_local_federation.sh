#!/usr/bin/env bash
# Stand up a local 1-SuperLink / 2-SuperNode Flower topology for roadmap 1.b.
#
# Insecure, no mTLS — debugging-grade local run only (the mTLS topology is
# roadmap 1.a step 4). Each SuperNode is pinned to one Geneva half via its own
# node-config `data-path`, which `load_data_gva` reads. Simulation mode can't do
# this (shared node-config), so deployment mode is required.
#
# Usage:
#   scripts/run_local_federation.sh start   # launch superlink + 2 supernodes
#   scripts/run_local_federation.sh stop    # kill everything started by 'start'
#   scripts/run_local_federation.sh status  # show tracked PIDs
#
# After 'start', run the matrix from architecture/ against the `local-deployment`
# federation, e.g.:
#   flwr run . local-deployment --run-config "save-model=true"
#
# NOTE: `flwr run` bundles this app dir into a FAB and reads THIS dir's
# .gitignore (not the repo-root one) to decide what to include, then rejects any
# path deeper than 10. The `.venv/` entry in architecture/.gitignore is what
# keeps the venv out of the bundle — without it the FAB build fails with
# "exceeds the maximum directory depth of 10". See that file's header.
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

# Ports (insecure, loopback only)
FLEET_API="127.0.0.1:9092"      # SuperNodes connect here
SUPERLINK_ADDR="127.0.0.1:9092" # what SuperNodes pass to --superlink
NODE_A_APPIO="127.0.0.1:9094"
NODE_B_APPIO="127.0.0.1:9095"

DATA_A="${REPO_ROOT}/out/geneva_half_A.parquet"
DATA_B="${REPO_ROOT}/out/geneva_half_B.parquet"
MODELS_DIR="${REPO_ROOT}/out/models"  # saved final_model.json lands here (absolute → CWD-independent)

start() {
    mkdir -p "${RUN_DIR}"
    : > "${PID_FILE}"

    # flower-superlink/-supernode spawn helper subprocesses (flower-superexec,
    # the clientapp runner) by bare name, so the venv bin must be on PATH.
    export PATH="${VENV_PY}:${PATH}"

    for f in "${DATA_A}" "${DATA_B}"; do
        [ -f "${f}" ] || { echo "ERROR: data file missing: ${f}" >&2; exit 1; }
    done

    # setsid detaches each daemon into its own session so it survives the shell
    # that launched this script (the daemons must outlive `start`).
    echo "Starting SuperLink (insecure) → logs ${RUN_DIR}/superlink.log"
    setsid "${VENV_PY}/flower-superlink" --insecure \
        > "${RUN_DIR}/superlink.log" 2>&1 &
    echo "superlink $!" >> "${PID_FILE}"
    sleep 3  # let the Fleet API bind before SuperNodes dial in

    echo "Starting SuperNode A (${DATA_A##*/}) → logs ${RUN_DIR}/supernode_A.log"
    setsid "${VENV_PY}/flower-supernode" --insecure \
        --superlink "${SUPERLINK_ADDR}" \
        --clientappio-api-address "${NODE_A_APPIO}" \
        --node-config "data-path=\"${DATA_A}\"" \
        > "${RUN_DIR}/supernode_A.log" 2>&1 &
    echo "supernode_A $!" >> "${PID_FILE}"

    echo "Starting SuperNode B (${DATA_B##*/}) → logs ${RUN_DIR}/supernode_B.log"
    setsid "${VENV_PY}/flower-supernode" --insecure \
        --superlink "${SUPERLINK_ADDR}" \
        --clientappio-api-address "${NODE_B_APPIO}" \
        --node-config "data-path=\"${DATA_B}\"" \
        > "${RUN_DIR}/supernode_B.log" 2>&1 &
    echo "supernode_B $!" >> "${PID_FILE}"

    sleep 3
    echo
    echo "Topology up. Tracked PIDs:"
    cat "${PID_FILE}"
    echo
    echo "Models will be saved to ${MODELS_DIR}"
    echo "Run the matrix from ${ARCH_DIR}:"
    local md="model-dir='${MODELS_DIR}'"
    echo "  flwr run . local-deployment --run-config \"save-model=true ${md}\""
    echo "  flwr run . local-deployment --run-config \"train-method='cyclic' save-model=true ${md}\""
    echo "  flwr run . local-deployment --run-config \"train-method='cyclic' cyclic-order='reverse' save-model=true ${md}\""
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
    *) echo "Usage: $0 {start|stop|status}" >&2; exit 1 ;;
esac
