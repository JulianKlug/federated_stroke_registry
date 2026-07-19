#!/usr/bin/env bash
# Generate the secret material for the secure local topology (roadmap 1.e,
# docs/specs/1e_secure_topology.md §4.1):
#
#   ca.key / ca.crt          local CA (self-signed, EC P-384)
#   server.key / server.pem  SuperLink TLS cert, CA-signed, SAN = loopback
#                            names, short validity (${DAYS_SERVER} days)
#   node_A / node_A.pub      per-SuperNode OpenSSH EC P-384 signing keypair
#   node_B / node_B.pub      (what `flwr supernode register` / the SuperNode's
#                            --auth-supernode-private-key parse)
#
# Everything lands in a gitignored secrets dir. NEVER commit ca.key,
# server.key, or any node private key. ca.crt/server.pem are regenerable:
# a fresh checkout just runs this script (run_local_federation.sh start does
# it automatically when material is missing).
#
# Usage:
#   scripts/gen_certs.sh [SECRETS_DIR] [--force]
#     SECRETS_DIR  defaults to architecture/.secrets
#     --force      regenerate even if material already exists
#
# Rotation (spec §4.8; Flower has no built-in credential expiry):
#   server cert:  scripts/gen_certs.sh --force  (or delete server.*, re-run),
#                 then run_local_federation.sh stop && start.
#   node key:     delete node_X and node_X.pub, re-run this script (generates a
#                 fresh keypair), then run_local_federation.sh stop && start —
#                 `start` recreates the SuperLink state and registers whatever
#                 keys are present, so no manual unregister step is needed.
#                 Rotation MUST mint a new key: flwr 1.31 binds each public key
#                 permanently to its node row (UNIQUE, kept after unregister),
#                 so a key can never be re-registered into the same database.
set -euo pipefail

ARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SECRETS_DIR="${ARCH_DIR}/.secrets"
FORCE=0
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE=1 ;;
        *) SECRETS_DIR="${arg}" ;;
    esac
done

DAYS_CA=180
DAYS_SERVER=90   # short-lived transport cert (spec §4.8, Decision 2)
NODES="node_A node_B"

EXPECTED=(ca.key ca.crt server.key server.pem)
for n in ${NODES}; do EXPECTED+=("${n}" "${n}.pub"); done

all_present() {
    for f in "${EXPECTED[@]}"; do
        [ -f "${SECRETS_DIR}/${f}" ] || return 1
    done
}

print_fingerprints() {
    echo
    echo "Secret material in ${SECRETS_DIR}:"
    openssl x509 -in "${SECRETS_DIR}/ca.crt" -noout -fingerprint -sha256 -enddate \
        | sed 's/^/  ca.crt      /'
    openssl x509 -in "${SECRETS_DIR}/server.pem" -noout -fingerprint -sha256 -enddate \
        | sed 's/^/  server.pem  /'
    for n in ${NODES}; do
        ssh-keygen -lf "${SECRETS_DIR}/${n}.pub" | sed "s/^/  ${n}      /"
    done
}

if all_present && [ "${FORCE}" -eq 0 ]; then
    echo "Secret material already present (use --force to regenerate)."
    print_fingerprints
    exit 0
fi

mkdir -p "${SECRETS_DIR}"
# ssh-keygen prompts interactively on overwrite, so clear targets first.
for f in "${EXPECTED[@]}"; do rm -f "${SECRETS_DIR}/${f}"; done

umask 077

echo "Generating CA (EC P-384, ${DAYS_CA} days)..."
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 -nodes \
    -keyout "${SECRETS_DIR}/ca.key" -out "${SECRETS_DIR}/ca.crt" \
    -days "${DAYS_CA}" -subj "/CN=fed-stroke local CA" 2>/dev/null

echo "Generating SuperLink server cert (loopback SAN, ${DAYS_SERVER} days)..."
openssl req -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 -nodes \
    -keyout "${SECRETS_DIR}/server.key" -out "${SECRETS_DIR}/server.csr" \
    -subj "/CN=localhost" 2>/dev/null
openssl x509 -req -in "${SECRETS_DIR}/server.csr" \
    -CA "${SECRETS_DIR}/ca.crt" -CAkey "${SECRETS_DIR}/ca.key" -CAcreateserial \
    -out "${SECRETS_DIR}/server.pem" -days "${DAYS_SERVER}" \
    -extfile <(printf 'subjectAltName=IP:127.0.0.1,DNS:localhost,IP:::1') 2>/dev/null
rm -f "${SECRETS_DIR}/server.csr" "${SECRETS_DIR}/ca.srl"

for n in ${NODES}; do
    echo "Generating SuperNode signing key ${n} (OpenSSH ECDSA P-384)..."
    ssh-keygen -q -t ecdsa -b 384 -N "" -C "fed-stroke ${n}" -f "${SECRETS_DIR}/${n}"
done

chmod 600 "${SECRETS_DIR}/ca.key" "${SECRETS_DIR}/server.key"
for n in ${NODES}; do chmod 600 "${SECRETS_DIR}/${n}"; done
chmod 644 "${SECRETS_DIR}/ca.crt" "${SECRETS_DIR}/server.pem"
for n in ${NODES}; do chmod 644 "${SECRETS_DIR}/${n}.pub"; done

print_fingerprints
