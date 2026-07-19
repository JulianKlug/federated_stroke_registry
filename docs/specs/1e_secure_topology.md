# Spec 1.e — Secure the local Geneva topology: TLS + node authentication

Implements roadmap item 1.e ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Secure the topology (architecture §6.1). Replace the insecure local run with 1
> `SuperLink` / 2 `SuperNode`s over **mTLS**, with **certificate pinning** and
> **short-lived credentials** (node authentication). Move each SuperNode's
> `data-path` into `pyproject.toml` under `[tool.flwr.federations.<name>]` (it
> currently lives in a CLI `--node-config` arg + `~/.flwr/config.toml`). This is the
> same secure contract Shenzhen will follow (1.3.a/1.3.b), practised locally to
> de-risk onboarding; nothing here is throwaway. Verify: disjoint patient sets
> across the two halves, roughly equal size, similar label balance, and a federated
> round completes end to end over the mTLS channel.

Grounded in the architecture doc §6.1
([docs/automated_review/architecture_federated_xgboost.md](../automated_review/architecture_federated_xgboost.md))
and the **installed** framework (`flwr==1.31.0`). Builds on the insecure 1.a topology
([architecture/scripts/run_local_federation.sh](../../architecture/scripts/run_local_federation.sh))
and the data halves produced by
[preprocessing/prepare_geneva_halves.py](../../preprocessing/prepare_geneva_halves.py).
This is the exact secure contract Shenzhen follows in 1.3.a/1.3.b, practised locally —
nothing here is throwaway.

The framework facts in §3 were verified against the **installed flwr 1.31 source**
(`.venv/.../flwr/server/app.py`, `flwr/cli/supernode/register.py`,
`flwr/supernode/cli/flower_supernode.py`), not from memory — the roadmap's literal
"mTLS / static credentials" wording does not match what flwr 1.31 provides, and §3
records the gap.

## 1. Goal & scope

Replace the `--insecure` loopback topology with one where the channel is
**encrypted**, the SuperNodes **verify** the SuperLink (certificate pinning), and the
SuperLink **authenticates** each SuperNode (node auth) — and where the secure topology
config is version-controlled instead of living in the untracked `~/.flwr/config.toml`.

**In scope**

- A **certificate/key generation script** (`scripts/gen_certs.sh`): a local CA, a
  SuperLink server cert (SAN covering `127.0.0.1`/`localhost`/`::1`, short validity),
  and one EC (P-384) signing key pair per SuperNode. Writes to a **gitignored**
  secrets directory; prints fingerprints.
- **SuperLink over TLS + node auth**:
  `--ssl-ca-certfile/--ssl-certfile/--ssl-keyfile` + `--enable-supernode-auth`, on a
  persistent `--database` (intended to survive restarts; unconditional re-register on
  every `start` is the safety net until persistence is source-proven — §4.3, §8).
- **SuperNodes over TLS with CA pinning** (`--root-certificates`) **+ an EC signing
  key** (`--auth-supernode-private-key`; the public key is derived from it in 1.31 —
  `--auth-supernode-public-key` is deprecated, §3.7), registered via
  `flwr supernode register`.
- **One secure connection surface, script-generated.** In flwr 1.31 *both*
  `flwr run` and `flwr supernode register` read the connection from
  `~/.flwr/config.toml` `[superlink.<name>]` (§3.4); the committed values (address
  `:9093`, CA-cert path) live in a **`[tool.fed_stroke.superlink]`** table the script
  reads and materializes into `~/.flwr/config.toml` on `start`. We deliberately do
  **not** commit `[tool.flwr.federations.local-deployment]` — `flwr run` would migrate
  it into `~/.flwr/config.toml` and then comment it out of the committed
  `pyproject.toml` (§3.9), mutating a tracked file.
- **Committed node→data-path map** + a `run_local_federation.sh` refactor that reads
  it (removing the hardcoded `DATA_A`/`DATA_B` strings and the ad-hoc `--node-config`).
- **Secret hygiene**: secrets dir excluded from both git and the FAB bundle; private
  keys and the CA key are never committed.
- A documented **credential lifetime & rotation** procedure.
- **Tests** for the mechanically-verifiable pieces + a scripted **end-to-end secure
  round** verification, including negative checks that prove security is actually on.

**Out of scope** (later roadmap / non-technical)

- DP noise on histograms → Phase v1.1/v1.2 prerequisite.
- SecAgg+ → Phase v2.1 (low value at N=2, architecture §6.5).
- Homomorphic encryption → explicitly out of scope (architecture §6.6).
- The **real cross-site** Geneva↔Shenzhen handshake → 1.3.b. 1.e is loopback only.
- At-rest artifact encryption / site-controlled KMS (architecture §6.1 names it): a
  site-ops concern, documented here as deferred, not implemented in the FL code.
- Governance/DPA/IRB (architecture §6.4): non-technical, tracked elsewhere.

## 2. Dependencies

- **1.a complete**: `scripts/run_local_federation.sh` + the two halves
  (`out/geneva_half_{A,B}.parquet`) exist; `load_data_gva` reads
  `node_config["data-path"]` ([task.py:80](../../architecture/fed_stroke/task.py)).
- **flwr 1.31.0** as installed (§3 facts are verified against this exact version).
- **`ssh-keygen`** (EC keypairs) and **`openssl`** (CA + server cert) on PATH.
  `cryptography` is already a transitive flwr dep (used by the register CLI).
- No new Python runtime dependencies.

## 3. Framework facts that drive the design (flwr 1.31, verified against source)

These are the load-bearing facts; the design in §4 follows from them.

- **3.1 TLS is server-authenticated only.** SuperLink takes
  `--ssl-ca-certfile/--ssl-certfile/--ssl-keyfile`; SuperNode takes
  `--root-certificates`, whose `--help` states verbatim *"This is not a client
  certificate for mTLS."* So the transport gives encryption + SuperNode→SuperLink
  trust (**certificate pinning** = the SuperNode pins the CA that signed the server
  cert). It does **not** give X.509 client-cert mTLS.
- **3.2 Node authentication is a dynamic registration flow; the static allowlist is
  gone.** `--auth-list-public-keys` **hard-exits** in 1.31
  (`flwr/server/app.py:366`, `SUPERLINK_INVALID_ARGS`: *"no longer supported… use
  `--enable-supernode-auth` and use the Flower CLI to register SuperNodes"*). The
  supported flow:
  1. SuperLink launched with `--enable-supernode-auth`.
  2. Generate an EC key pair per SuperNode.
  3. `flwr supernode register <public_key_file> <superlink-connection>` registers the
     public key with the running SuperLink over its Control API.
  4. SuperNode launched with `--auth-supernode-private-key` **only** — the public key is
     derived from it; `--auth-supernode-public-key` is deprecated (§3.7).
- **3.3 Node-auth keys are OpenSSH-format NIST EC (P-384).** `register.py:125-137`
  loads the public key via `serialization.load_ssh_public_key`, requires
  `ec.EllipticCurvePublicKey`, and calls `uses_nist_ec_curve(...)`; the arg help says
  *"P-384 (or any other NIST EC curve)"*. The SuperNode side loads the private key with
  `load_ssh_private_key` (`flwr/supernode/cli/flower_supernode.py:283`) and derives the
  public key from it (§3.7). → generate with `ssh-keygen -t ecdsa -b 384`. (Note: the
  `load_ssh_public_key` at `flower_supernode.py:333` is the **Ed25519 trusted-entities**
  path, a different feature — not node auth.)
- **3.4 BOTH `flwr run` and `flwr supernode register` resolve the connection from
  `~/.flwr/config.toml` `[superlink.<name>]` — there is one runtime surface, not two.**
  `register.py:76` and `run/run.py:141` both call `read_superlink_connection(<name>)` →
  `read_flower_config()` (`flower_config.py:263,380`), which loads
  `get_flwr_home()/config.toml` (= `~/.flwr/config.toml`) and looks the connection up by
  name under `[superlink.<name>]` (`flower_config.py:266,298`). Neither reads
  `pyproject.toml`'s `[tool.flwr.federations.<name>]` at *runtime*. So the setup script
  **generates** the single `[superlink.local-deployment]` entry in `~/.flwr/config.toml`
  on `start` from committed values (§4.5), and both CLIs read it. (An earlier draft of
  this spec wrongly split this into "pyproject for run, config.toml for register" — see
  §3.9 for why committing the pyproject block is actively harmful.)
- **3.5 Registration persistence follows the SuperLink `--database`.** With the
  default in-memory state, registered keys are lost on SuperLink restart. → the launch
  script either uses a persistent `--database` path or (re)registers inside `start` on
  every launch (§4.3).
- **3.6 "mTLS" reconciliation.** The roadmap word "mTLS" is realized here as
  **(server-side TLS + CA pinning) + (EC public-key node authentication)**. Net
  security property matches mTLS's intent — channel encrypted, both ends authenticated
  — via Flower's supported mechanism rather than X.509 client certs.
- **3.7 The SuperNode public-key flag is deprecated; the private key is authoritative.**
  `--auth-supernode-public-key` is deprecated in 1.31
  (`flower_supernode.py:264-266,297-303`): passing it logs a WARN and is redundant
  because `_try_setup_client_authentication` derives the public key from the private key
  (`ssh_private_key.public_key()`). → the SuperNode launch passes **only**
  `--auth-supernode-private-key` (§4.4). The `node_X.pub` file is still generated — the
  *register* step (§4.3) needs it — it just isn't handed to `flower-supernode`.
- **3.8 `--enable-supernode-auth` hard-requires TLS + the gRPC-rere Fleet transport.**
  `app.py:335-343` hard-exits under `--insecure`; `app.py:344-350` hard-exits unless
  `--fleet-api-type == grpc-rere`. Both are satisfied by this design: the launch is TLS
  (§4.2) and `--fleet-api-type` **defaults** to `grpc-rere` (`app.py:920`), so §4.2
  needs no extra flag. Recorded because the coupling is invisible: flipping the
  transport later would break auth with a non-obvious error.
- **3.9 Committing `[tool.flwr.federations]` makes `flwr run` mutate the tracked
  `pyproject.toml`.** `flwr run` calls `migrate(app, [], ignore_legacy_usage=True)`
  (`run/run.py:128`) *before* resolving the connection. `[tool.flwr.federations]` is a
  **legacy** surface: if present, `_is_migratable` is true (`config_migration.py:108`),
  so migrate copies the block into `~/.flwr/config.toml` via `write_superlink_connection`
  (`config_migration.py:135`) **and then unconditionally comments the block out of the
  committed `pyproject.toml`** (`_comment_out_legacy_toml_config`,
  `config_migration.py:270`; verified against the migrate body, lines 231-280). Net: the
  first `flwr run` leaves `pyproject.toml` dirty in git and silently overwrites any
  script-written `[superlink.local-deployment]` with the migrated values. → do **not**
  commit `[tool.flwr.federations]`; put the connection values in a non-flwr
  `[tool.fed_stroke.superlink]` table `migrate` ignores (§4.5). With no
  `[tool.flwr.federations]`, `migrate` is a no-op (`_is_migratable` false → early
  return, `config_migration.py:239-247`) and nothing is mutated.

## 4. Design

### 4.0 Terminology reconciliation (Decision 1 = A)

State the §3.6 mapping up front so "mTLS" in the roadmap and "TLS + node auth" in the
code never read as a gap. Operator-facing docs use the phrase "secure channel".

### 4.1 New `architecture/scripts/gen_certs.sh`

Generates, into a gitignored secrets dir (§4.7):

- **CA**: `ca.key` (private, `chmod 600`) + `ca.crt` (self-signed).
- **Server cert**: `server.key` + `server.pem`, CA-signed, `subjectAltName =
  IP:127.0.0.1, DNS:localhost, IP:::1`, **short validity** (e.g. `-days 90`) per
  Decision 2.
- **Per-node EC keys**: `node_A`/`node_A.pub`, `node_B`/`node_B.pub` via
  `ssh-keygen -t ecdsa -b 384 -N "" -f node_X` (OpenSSH format, §3.3).
- Idempotent (skip-if-exists or `--force`); prints cert fingerprints + key comments so
  the operator can eyeball what was made.

### 4.2 SuperLink launch (refactor `run_local_federation.sh`)

Replace `flower-superlink --insecure` with:

```
flower-superlink \
  --ssl-ca-certfile <secrets>/ca.crt \
  --ssl-certfile   <secrets>/server.pem \
  --ssl-keyfile    <secrets>/server.key \
  --enable-supernode-auth \
  --database <arch>/.federation/state.db      # persistent → registrations survive (§3.5)
```

Ports unchanged (Fleet `:9092`, Control `:9093`).

### 4.3 Node registration step (inside `start`, after SuperLink is up)

For each node: `flwr supernode register <secrets>/node_X.pub local-deployment`. This
requires the `[superlink.local-deployment]` connection to exist in `~/.flwr/config.toml`
first (§3.4) — the script writes it (§4.5) **before** the first register call. Register
runs **on every `start`, unconditionally** (idempotent: tolerate the already-registered
duplicate error, or `unregister` → re-register), so a fresh `start` re-establishes auth
regardless of whether the `--database` actually persisted registrations across the
restart (§3.5 — persistence is asserted but not yet source-proven, §8). Gate on the
SuperLink Control API being reachable first (the script already `sleep`s for the Fleet
API to bind; extend to the Control API).

### 4.4 SuperNode launch (refactor)

Replace `flower-supernode --insecure` with:

```
flower-supernode \
  --root-certificates <secrets>/ca.crt \       # CA pinning (§3.1)
  --superlink 127.0.0.1:9092 \
  --auth-supernode-private-key <secrets>/node_X \   # public key derived from this (§3.7)
  --clientappio-api-address 127.0.0.1:909X \
  --node-config "data-path=\"<half_X>\""        # sourced from the committed node map (§4.6)
```

Do **not** pass `--auth-supernode-public-key`: it's deprecated in 1.31 and only warns
(§3.7).

### 4.5 Secure connection config — one generated surface (Decision 3 = A, part 1)

Both CLIs read the connection from `~/.flwr/config.toml` `[superlink.<name>]` (§3.4), and
committing `[tool.flwr.federations]` would mutate the tracked `pyproject.toml` (§3.9). So:

**Committed values live in a non-flwr `[tool.fed_stroke.superlink]` table** (which
`migrate` never touches) in `architecture/pyproject.toml`:

```
[tool.fed_stroke.superlink]
name = "local-deployment"
address = "127.0.0.1:9093"     # Control API; also the Fleet dial target for flwr run
ca-cert = "<repo-relative>/ca.crt"
```

**The script generates `~/.flwr/config.toml` `[superlink.local-deployment]` on `start`**
from that table (resolving `ca-cert` to the path flwr needs), giving the single runtime
surface both `flwr run . local-deployment` and `flwr supernode register … local-deployment`
read:

```
[superlink.local-deployment]
address = "127.0.0.1:9093"
root-certificates = "<resolved>/ca.crt"     # no insecure = true
```

Two operational musts:

- **Merge, do not truncate.** `~/.flwr/config.toml` may already exist with other
  connections (flwr auto-seeds `[superlink.local]`/`[superlink.supergrid]` and a
  `default` on first CLI use, `cli/constant.py`). The script must upsert only the
  `[superlink.local-deployment]` key (flwr's own `write_superlink_connection` merges;
  match that), never overwrite the whole file.
- **Path portability.** `ca-cert` is committed repo-relative; the script resolves it to
  an absolute path when writing `~/.flwr/config.toml` (documented, not a hardcoded home
  dir). See §8.

**Single source for ports/addresses.** `[tool.fed_stroke.superlink].address` and the
`[tool.fed_stroke.nodes]` appio ports (§4.6) are the committed source of truth; the
script derives its own `FLEET_API`/`SUPERLINK_ADDR` constants and the generated
`~/.flwr/config.toml` from them, so `:9092`/`:9093` can't drift across surfaces (the
drift F5 flagged is now structurally prevented, not just cautioned).

### 4.6 Committed node→data-path map + script refactor (Decision 3 = A, part 2)

The roadmap's literal "put `data-path` in the federation block" is **not possible** —
`data-path` is *per-SuperNode* `node_config`, while a `[tool.flwr.federations]` block is a
*single* connection entry (and one we don't commit at all, §3.9). Instead, commit the
per-node mapping in a **`[tool.fed_stroke.nodes]` table in `architecture/pyproject.toml`**:

```
[tool.fed_stroke.nodes]
node_A = { data-path = "out/geneva_half_A.parquet", appio = "127.0.0.1:9094", key = "node_A" }
node_B = { data-path = "out/geneva_half_B.parquet", appio = "127.0.0.1:9095", key = "node_B" }
```

`run_local_federation.sh` reads this (single source of truth) instead of hardcoded
`DATA_A`/`DATA_B`, honoring the roadmap's intent (topology config in version control).

### 4.7 Secret hygiene

- Secrets dir (e.g. `architecture/.secrets/` or repo-root `secrets/`) added to
  **`architecture/.gitignore`** (which is also read by the FAB packager — keeps keys
  out of the bundle, per that file's load-bearing header) **and** root `.gitignore`.
- **Never commit** `ca.key`, `server.key`, or any `node_X` private key. Only the
  *path* to `ca.crt` is referenced from committed config.
- Document that `ca.crt` and `server.pem` are regenerable from `gen_certs.sh`, so a
  fresh checkout runs that script before `start`.

### 4.8 Credential lifetime & rotation (Decision 2 = A)

Node-auth EC keys are static allowlist entries (Flower has no built-in expiry) — the
"short-lived" intent is honored on the TLS server cert (bounded `-days`) plus a written
rotation procedure: regenerate the server cert, restart the SuperLink; rotate a node
key by `unregister` → regenerate → `register`. Documented in the spec and the script
header. **Known deviation from the roadmap's literal "short-lived credentials":** only
the *transport* cert is short-lived; the node-auth *identity* is a static key with no
expiry, rotated by procedure. This is the honest ceiling of what 1.31 provides at N=2;
governance (architecture §6.4), not credential churn, is the load-bearing trust layer (Decision 2).

### 4.9 Tests

Mechanically verifiable (unit/integration under `architecture/tests/`):

- `gen_certs.sh` output invariants: files exist with expected perms; node keys parse
  as `ec.EllipticCurvePublicKey` on P-384; server cert SAN includes the loopback names;
  server cert validity is bounded (≤ configured days).
- Config invariants: `pyproject.toml` has **no** `[tool.flwr.federations]` block (its
  presence would trigger the §3.9 migrate/comment-out); the committed
  `[tool.fed_stroke.superlink]` table has an `address` and `ca-cert` and no `insecure`;
  the `[tool.fed_stroke.nodes]` map parses and its `data-path`s point at the two halves.
- Generation invariant: after a dry-run of the config-generation step, the resulting
  `~/.flwr/config.toml` has `[superlink.local-deployment]` with `root-certificates` and
  no `insecure`, and any pre-existing `[superlink.*]` connections in that file are
  preserved (merge, not truncate — §4.5).

The **end-to-end positive round** is a scripted verification (§6), not a pytest.

**Negative security checks are automated, not eyeballed.** The three checks in §6 step 4 are
the whole point of 1.e — they prove security is actually *on* and are the regression
guard the Shenzhen onboarding (1.3.a) inherits. Ship them as an asserting harness
(`architecture/scripts/verify_negative_security.sh`, exit non-zero on any check that
does *not* fail closed), each asserting a concrete signature rather than a vibe:

- **Unregistered/no key → rejected.** Launch a SuperNode with an unregistered EC key.
  Assert on **(never-READY within a bounded timeout) + the SuperLink's auth-rejection
  log line** — *not* on the SuperNode's exit code. An unregistered node typically enters
  gRPC reconnect/retry rather than exiting non-zero, so exit code is an unreliable
  signal; the timeout-never-READY + server-side rejection log is the real assertion. Pin
  the exact log string during implementation against a real rejection.
- **Wrong CA → TLS handshake fails.** Launch a SuperNode with a `--root-certificates`
  pointing at an unrelated CA; assert the gRPC/TLS handshake error (not a generic
  timeout).
- **Insecure `flwr run` → refused.** `flwr run` with `insecure = true` / no CA against
  the TLS SuperLink; assert connection refused. (flwr also rejects `insecure` +
  `root-certificates` set together at config-parse time, `config_utils.py:204-208`, so a
  half-secure connection can't even be constructed — the check is framework-enforced.)

Each check must fail closed within a bounded timeout (so a hang reads as FAIL, not
PASS). Wire this harness into `run_local_federation.sh` (a `verify` subcommand) or CI.

## 5. Files to change

- **New** `architecture/scripts/gen_certs.sh` — CA + server cert + per-node EC keys.
- **Modify** `architecture/scripts/run_local_federation.sh` — TLS flags, node-auth
  private keys, `~/.flwr/config.toml` `[superlink.local-deployment]` generation (merge,
  §4.5), unconditional registration step, persistent `--database`, node-map sourcing.
- **Modify** `architecture/pyproject.toml` — add `[tool.fed_stroke.superlink]` and
  `[tool.fed_stroke.nodes]`. Do **not** add `[tool.flwr.federations.*]` (§3.9 — `flwr
  run` would migrate + comment it out).
- **Modify** `architecture/.gitignore` **and** root `.gitignore` — secrets dir.
- **New** `architecture/tests/test_secure_topology.py` — cert/key + config invariants.
- **New** `architecture/scripts/verify_negative_security.sh` — the three §6 step 4
  negative checks as asserting, fail-closed tests (§4.9).
- **Modify** `architecture/roadmap.md` (1.e item) — annotate that "put `data-path` in
  `[tool.flwr.federations.<name>]`" is infeasible and superseded by the §4.6 split (so a
  future reader doesn't re-hit the contradiction, §4.6/Decision 3).
- **Docs** — `docs/logbook.md` entry; roadmap 1.e checkbox left unchecked until the
  code lands (this task ships the spec only).

## 6. Verification (end to end)

0. `scripts/gen_certs.sh` → CA + server cert + node keys in the secrets dir.
1. `scripts/run_local_federation.sh start` → SuperLink (TLS + auth) + 2 SuperNodes
   (pinned CA + signing keys), nodes registered.
2. Data invariants already asserted by `prepare_geneva_halves.py` (disjoint patient
   sets, ~equal size, similar label balance) — the roadmap's first three verify items.
3. `flwr run . local-deployment` → a federated round completes over TLS; logs show the
   secure Fleet channel + successful node auth. (Run both strategies as in 1.b.)
4. **Negative checks (prove security is on)** — run via
   `scripts/verify_negative_security.sh`, which asserts each fails closed within a
   bounded timeout (§4.9), not by eyeballing logs:
   - SuperNode with **no / unregistered** key → rejected by the SuperLink.
   - SuperNode with the **wrong CA** in `--root-certificates` → TLS handshake fails.
   - `flwr run` against the SuperLink with `insecure`/no CA → connection refused.

## 7. Acceptance criteria

- CA, server cert (loopback SAN, bounded validity), and two EC node keypairs are
  generated by the script into a gitignored dir; no secret is tracked by git or bundled
  into the FAB.
- SuperLink runs with TLS + `--enable-supernode-auth`; both SuperNodes connect with CA
  pinning + a registered EC key each (private key only on the SuperNode CLI, §3.7); a
  federated round completes end to end over the encrypted channel for **both**
  strategies.
- The three negative checks (§6, step 4) all fail closed, asserted by
  `verify_negative_security.sh` (§4.9), not by manual log inspection.
- The secure connection values are committed in `pyproject.toml`'s
  `[tool.fed_stroke.superlink]` table (no `insecure`), and the per-node `data-path` comes
  from the committed `[tool.fed_stroke.nodes]` map. `pyproject.toml` contains **no**
  `[tool.flwr.federations]` block (§3.9). The single runtime surface
  (`~/.flwr/config.toml` `[superlink.local-deployment]`, read by both `flwr run` and
  `flwr supernode register`) is **generated by the script from those committed values**
  on `start`, merged into any existing connections — so no *hand-maintained* or committed
  `~/.flwr/config.toml` is required, and running `flwr run` never dirties `pyproject.toml`
  (§3.4, §4.5).
- Tests in §4.9 pass; rotation procedure documented.

## 8. Risks & notes

- **Connection resolution (§3.4) — RESOLVED.** Both `flwr run` and `flwr supernode
  register` read `~/.flwr/config.toml` `[superlink.<name>]` (verified against
  `run/run.py:141`, `register.py:76`, `flower_config.py`). The script generates that one
  entry from the committed `[tool.fed_stroke.superlink]` table.
- **Do not commit `[tool.flwr.federations]` (§3.9) — RESOLVED in design.** `flwr run`
  migrates it into `~/.flwr/config.toml` and comments it out of the tracked
  `pyproject.toml` (`config_migration.py:270`). Verified. Design avoids it by using the
  non-flwr `[tool.fed_stroke.superlink]` table. Residual: an implementation test should
  assert `flwr run` leaves `pyproject.toml` byte-identical (guards against a future
  reader re-adding a federation block).
- **`~/.flwr/config.toml` generation must merge, not truncate (§4.5)** — the file may
  hold other connections (flwr auto-seeds some); upsert only `[superlink.local-deployment]`.
- **Registration persistence (§3.5)** — validate the persistent `--database` path
  actually preserves *registered node keys* (not just run/task state) across
  `start`/`stop`; until proven, the §4.3 unconditional re-register is the safety net.
  **Implementation outcome (2026-07-19):** the SQLite `--database` *does* persist
  registrations, but they cannot be re-bound to static keys on restart: (a) a killed
  node stays `online` until heartbeat expiry (~1 min) and `activate_node` only succeeds
  from `registered`/`offline`, so quick-restart SuperNodes die with "could not be
  activated"; (b) `node.public_key` is UNIQUE and `unregister` keeps the row, so a
  once-unregistered key can never be re-registered in the same DB. Resolution shipped:
  `start` recreates the LinkState (`rm -f state.db`) and registers fresh — the §4.3
  unconditional register is the contract, not a safety net — and key rotation always
  mints a NEW key (§4.8).
- **`ca-cert` path portability (§4.5)** — commit repo-relative; script resolves to
  absolute when writing `~/.flwr/config.toml`. No hardcoded home path.
- **`ssh-keygen` EC format** — confirm the OpenSSH keys it emits load cleanly via
  `load_ssh_public_key` on this box (expected yes; it's the format register.py parses).
- **FAB build** — the secrets dir must be gitignored *in `architecture/.gitignore`* or
  `flwr run`'s FAB packager may bundle/reject it (that .gitignore is load-bearing for
  the build, per its header).
- **At-rest encryption / KMS (architecture §6.1)** — deferred to site-ops; documented, not coded.
- **Doctor-operator UX** — keep cert-gen + registration inside the `start` flow so the
  operator runs one command; this is the contract Shenzhen inherits (1.3.a).

## 9. Review decisions — audit trail

- **Decision 1 — secure channel = A (Flower-native TLS + node auth).** A true X.509
  mTLS reverse proxy adds a non-native daemon to ship to Shenzhen for no gain against
  the honest-but-curious threat model (architecture §6.7). Flower's TLS + EC node auth gives the same
  net property (encrypted, both-ends-authenticated).
- **Decision 2 — credential lifetime = A (static EC keys + short-expiry TLS + rotation
  doc).** Token/ephemeral auth is Enterprise-tier overkill for N=2 local; governance
  (architecture §6.4), not credential churn, is the load-bearing trust layer.
- **Decision 3 — config = A (committed values table + script-generated single surface).**
  Two roadmap literals are infeasible in flwr 1.31: "data-path in the federation block"
  (per-node `node_config` vs. a single federation entry) and "commit the federation
  block" (`flwr run` migrates + comments it out, §3.9). Resolution: commit the connection
  *values* in a non-flwr `[tool.fed_stroke.superlink]` table + the node map in
  `[tool.fed_stroke.nodes]`; the script materializes the one runtime surface
  (`~/.flwr/config.toml` `[superlink.local-deployment]`, read by both CLIs, §3.4). Honors
  the roadmap's real intent — topology config in version control — without a mutated
  tracked file.
- **Framework note (auth-list)** — `--auth-list-public-keys` was removed in flwr 1.31;
  the spec uses the `flwr supernode register` flow instead (verified against installed
  source).
- **Framework note (public-key flag, §3.7)** — `--auth-supernode-public-key` is
  deprecated in 1.31; the SuperNode public key is derived from the private key. The
  launch passes only `--auth-supernode-private-key`. (Caught in review; the earlier draft
  passed both.)
- **Framework note (connection resolution, §3.4)** — BOTH `flwr run` (`run/run.py:141`)
  and `flwr supernode register` (`register.py:76`) read the connection from
  `~/.flwr/config.toml` `[superlink.<name>]`; neither reads pyproject federations at
  runtime (verified against `flower_config.py`). One runtime surface, script-generated.
- **Framework note (migrate/comment-out, §3.9)** — committing `[tool.flwr.federations]`
  makes `flwr run` copy it into `~/.flwr/config.toml` and comment it out of the tracked
  `pyproject.toml` (`config_migration.py:270`, verified). The spec deliberately avoids
  the flwr-native federation block for this reason.
- **Roadmap supersession (§4.6, §5)** — the roadmap's literal "`data-path` in
  `[tool.flwr.federations.<name>]`" is infeasible on two counts (per-node config; and the
  migrate/comment-out trap); this spec supersedes it and the roadmap item is annotated
  (§5) so the contradiction isn't rediscovered.
- **External review (fresh-context subagent)** — an independent reviewer, verifying
  against the same installed source, refuted the earlier draft's "two config surfaces"
  framing (F1): `flwr run` also reads `~/.flwr/config.toml`, and committing the pyproject
  federation block mutates a tracked file (§3.9). Findings X1-X3 (below) were folded in;
  the config design was reworked to a single generated surface (Decision 3, revised).

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|--------|---------|------|--------|----------|
| Eng Review | `/plan-eng-review` | 1 | Applied | 6 (F1-F6) |
| External Review | fresh-context subagent | 1 | Applied | 3 P1 + 4 P2 (X1-X7) |
| Codex Review | `/codex review` | 0 | Blocked (auth 401) | — |

All findings verified against installed flwr 1.31 source (`.venv/.../flwr/`).

**Eng review (F1-F6):**
- **F1 [HIGH] — connection resolution.** register reads `~/.flwr/config.toml`, not
  pyproject (`flower_config.py`). *Superseded by X1: `flwr run` reads it too — see below.*
- **F2 [MED] — `--auth-supernode-public-key` deprecated** (`flower_supernode.py:264,297`).
  Dropped from the launch (§3.7, §4.4); `node_X.pub` kept for the register step.
- **F3 [LOW] — `--enable-supernode-auth` ⟹ TLS + grpc-rere** (`app.py:335,344,920`).
  Default satisfies it; documented §3.8.
- **F4 [MED] — negative checks were manual.** Asserting fail-closed harness (§4.9/§5/§6).
- **F5 [LOW] — port drift across surfaces.** Now structurally prevented: one committed
  `[tool.fed_stroke.superlink]`/`[tool.fed_stroke.nodes]` source, script derives the rest (§4.5).
- **F6 [LOW] — roadmap states infeasible instruction.** Annotation task (§5), supersession (§9).

**External review (X1-X7) — refuted F1's framing and hardened the rest:**
- **X1 [P1] — "two config surfaces" was wrong.** `flwr run` also reads
  `~/.flwr/config.toml` `[superlink.<name>]` (`run/run.py:141`), not pyproject
  federations. Reworked to one script-generated surface (§3.4, §4.5).
- **X2 [P1] — committing `[tool.flwr.federations]` mutates tracked `pyproject.toml`.**
  `flwr run`'s migrate comments the block out (`config_migration.py:270`). Design now uses
  a non-flwr `[tool.fed_stroke.superlink]` table (§3.9, §4.5, §5).
- **X3 [P1] — script-written vs migrate-written `[superlink]` collision.** Eliminated by
  X2's fix (no migrate runs when no federation block is committed).
- **X4 [P2] — negative check #1 relied on exit code.** Now asserts never-READY-in-timeout
  + server auth-rejection log (§4.9).
- **X5 [P2] — `--database` persistence asserted, not proven.** §4.3 re-register made
  unconditional; §8 residual.
- **X6 [P2] — "short-lived credentials" only partial.** Stated as a known deviation (§4.8).
- **X7 [P2] — config.toml must merge, not truncate.** §4.5 + §8; generation-merge test (§4.9).
- Also: stale §3.3 citation (`flower_supernode.py:333` is the Ed25519 path) corrected.

VERDICT: ENG + EXTERNAL CLEARED — spec is source-accurate and internally consistent after
F1-F6 + X1-X7; the config design was reworked from "two surfaces" to one generated
surface. CODEX not absorbed (auth 401 — re-run after `codex login` for a third opinion).
Residual items in §8 are implementation-time confirmations (`--database` key persistence,
exact rejection log strings, `flwr run` leaves pyproject byte-identical), not open spec
decisions.

NO UNRESOLVED DECISIONS
