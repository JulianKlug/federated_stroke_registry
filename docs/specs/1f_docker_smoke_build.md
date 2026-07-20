# Spec 1.f — Docker smoke build

Implements roadmap item 1.f ([architecture/roadmap.md](../../architecture/roadmap.md)):

> Docker smoke build. Dockerfile compiles, container runs the pipeline end-to-end
> on the dev machine. Full Shenzhen-ready packaging (version pinning,
> clean-bootstrap test on a non-dev machine) is deferred to Phase v1.3, but
> building the image once here catches Dockerfile bugs long before Shenzhen
> go-live.

Grounded in the **installed** framework (`flwr==1.31.0`, `xgboost>=2.0`) and the
1.e secure-topology machinery this container reuses **verbatim**:
[architecture/scripts/run_local_federation.sh](../../architecture/scripts/run_local_federation.sh),
[architecture/scripts/gen_certs.sh](../../architecture/scripts/gen_certs.sh),
[docs/specs/1e_secure_topology.md](1e_secure_topology.md), and the committed
`[tool.fed_stroke.*]` tables in
[architecture/pyproject.toml](../../architecture/pyproject.toml). The design's central
move is that the container does **not** fork any of that orchestration — it reproduces the
host directory layout so the existing scripts run unchanged (§3.4, §4.1).

This task **ships the spec only.** The roadmap 1.f checkbox stays `[ ]` until the code
lands in a follow-up task — mirroring how 1.e's spec shipped ahead of its implementation.

## 1. Goal & scope

Prove, once, on the dev machine, that the federated stroke pipeline builds into a Docker
image and runs end to end inside a container — so a Dockerfile bug (missing OpenMP runtime,
missing `ssh-keygen`, wrong path layout) is caught here rather than during Shenzhen
onboarding (1.3.a). This is a **smoke build**, not the shippable Shenzhen package.

**In scope**

- A **`architecture/Dockerfile`** that builds on the dev machine from `python:3.12-slim`,
  installs the OS packages the slim base lacks (§3.1, §3.2), and installs the Python
  environment from the committed `architecture/uv.lock` via `uv sync --locked` (reproducible
  Python layer for free — flwr/xgboost/numpy are already pinned there).
- A **single-container, loopback run** of the real secure pipeline: the container's entry
  point runs the existing `run_local_federation.sh start` (1-SuperLink / 2-SuperNode over
  server-side TLS + CA pinning + EC node auth on `127.0.0.1`, spec 1.e), then one
  `flwr run . local-deployment`, then asserts the metrics artifact, then `stop`.
- An **`architecture/scripts/smoke_pipeline.sh`** orchestrator (runnable in *and* out of
  Docker) that is the container `CMD` and the local smoke command both.
- A **`architecture/.dockerignore`** that keeps `.venv/`, `.secrets/`, `.federation/`,
  `out/` out of the build context (mirrors `architecture/.gitignore`; keeps host keys and
  data out of the image, §4.3).
- **Real Geneva data mounted, never baked in** (§4.5, Decision 2): the two halves are
  bind-mounted at run time; the image contains no patient data.
- **Static tests** (`architecture/tests/test_docker_smoke.py`) asserting the Dockerfile /
  `.dockerignore` / entrypoint invariants; the live build+run is the §6 scripted check.

**Out of scope** (deferred, with the roadmap item that owns each)

- **Multi-container / cross-host topology** (separate SuperLink + SuperNode containers on a
  docker network, real hostname SANs) → Phase v1.3 **1.3.b** (cross-site smoke test). 1.f is
  loopback inside one container only.
- **Shenzhen-ready packaging** → Phase v1.3 **1.3.a**: base image pinned by **digest**, exact
  **OS OpenMP** version pin, and the **clean-bootstrap acceptance test on a non-dev machine**.
  The roadmap explicitly parks these; 1.f uses a base *tag* + `uv.lock` and builds only on the
  dev machine (§4.6, Decision 3).
- **Non-root container user** → a 1.3.a hardening item (§8). 1.f runs as root — acceptable for
  a dev smoke build; the secure topology needs no privileged ops.
- **Synthetic smoke data** — not built; 1.f mounts the real halves (Decision 2, §8).
- **CI integration** of the live build+run — the build+run is dev-machine scoped by design
  (needs the data mount); only the static invariants (§4.7) run in a data-less CI.

## 2. Dependencies

- **1.a complete**: `out/geneva_half_{A,B}.parquet` exist on the host (produced by
  [preprocessing/prepare_geneva_halves.py](../../preprocessing/prepare_geneva_halves.py)).
  These are the mount source; the container does **not** need the raw Geneva Excel or the
  preprocessing step.
- **1.e complete**: `run_local_federation.sh`, `gen_certs.sh`,
  `verify_negative_security.sh`, and the committed `[tool.fed_stroke.superlink]` /
  `[tool.fed_stroke.nodes]` tables exist and work on the host.
- **`architecture/uv.lock`** current against `architecture/pyproject.toml` (the Docker build
  installs from it with `--locked`, which fails the build if the lock is stale — a useful guard).
- **`docker`** installed on the dev machine. No new Python runtime dependencies.

## 3. Framework / runtime facts that drive the design

These are the load-bearing facts; the design in §4 follows from them. They are why a naive
`FROM python:3.12-slim; pip install .; CMD flwr run` would fail — 1.f exists to surface exactly
these.

- **3.1 xgboost's OpenMP runtime (`libgomp1`) — a *design hypothesis that 1.f falsified for the
  pinned version*.** The original premise was that xgboost links the system `libgomp.so.1`, absent
  from `python:3.12-slim`, so `import xgboost` would fail with
  `libgomp.so.1: cannot open shared object file` unless the image `apt-get install`s `libgomp1`.
  **The 1.f verification (2026-07-20) disproved this for `xgboost==3.3.0`:** its manylinux wheel
  **vendors its own** `libgomp-*.so.1` under `site-packages/xgboost.libs/`, and `ldd libxgboost.so`
  resolves OpenMP to that bundled copy — so `import xgboost` and every client boost work with **no**
  system `libgomp1`. `libgomp1` is therefore kept as *defense-in-depth* (insurance against a future
  xgboost that links the system libgomp — a source build or non-manylinux wheel), **not** the load-
  bearing catch. The real bug 1.f catches is §3.2's missing `ssh-keygen` (empirically the only apt
  package both required and absent from the slim base; §6.5 negative check). 1.3.a pins the exact
  xgboost + OS-OpenMP versions.
- **3.2 `gen_certs.sh` needs `openssl` and `ssh-keygen`; the slim base lacks
  `openssh-client`.** The cert/key generator builds the CA + server cert with `openssl`
  (`gen_certs.sh:83-93`) and mints the OpenSSH-format EC P-384 node-auth keypairs with
  `ssh-keygen -t ecdsa -b 384` (`gen_certs.sh:99`) — the exact format flwr's node-auth
  register step parses (1.e §3.3). `openssl` is usually present on slim; `ssh-keygen`
  (package `openssh-client`) is **not**, so the image must install it or `ensure_certs`
  aborts before the SuperLink starts.
- **3.3 `flwr run` builds a FAB at run time honoring the *app-dir* `.gitignore` and rejecting
  paths deeper than 10.** The packager reads `architecture/.gitignore` (not the repo root's)
  and fails on any bundled path past `MAX_DIR_DEPTH=10`; the `.venv/` line in that file is what
  keeps the vendored venv out of the bundle
  ([architecture/.gitignore](../../architecture/.gitignore) header). Inside the container the
  venv lives under the app dir (§4.1), so this same `.gitignore` line makes `flwr run`'s FAB
  build succeed in the container too — **no `.gitignore` change is needed**, and the container
  must not delete or override it.
- **3.4 `run_local_federation.sh` derives every path from its own location, so the container
  layout — not the script — is what must change.** The script computes
  `ARCH_DIR = scripts/..`, `REPO_ROOT = ARCH_DIR/..`, `VENV_PY = ARCH_DIR/.venv/bin`
  (`run_local_federation.sh:48-55`); node data resolves to `REPO_ROOT/out/<half>.parquet` and
  the secrets dir to `REPO_ROOT/architecture/.secrets` (via the committed `ca-cert` path,
  `:74,84,92`). Therefore the container must reproduce the host's **`<repo>/architecture` +
  `<repo>/out`** shape and place the venv at `<repo>/architecture/.venv`; then the whole 1.e
  orchestration runs **verbatim**, unforked (§4.1). Any other layout would force a script
  rewrite — avoided by construction.
- **3.5 Deployment mode is required; simulation cannot pin per-node data.** Simulation shares
  one `node_config` across nodes, so it cannot give each SuperNode its own `data-path`
  (`run_local_federation.sh:9-11`). The smoke run therefore targets the running SuperLink with
  `flwr run . local-deployment` (deployment mode), exactly as 1.b/1.c/1.e do — not bare
  `flwr run .` (simulation).
- **3.6 The 1.e topology is loopback-only (`127.0.0.1`), which is why one container suffices.**
  SuperLink Fleet `:9092` / Control `:9093` and the per-node appio ports are all on `127.0.0.1`
  (committed `[tool.fed_stroke.*]` tables). A single container has one loopback namespace, so
  the entire topology fits with **no docker networking and no cross-container TLS SANs** —
  those first appear when the topology spans hosts (1.3.b).

## 4. Design

### 4.1 Container layout — reproduce the host tree so 1.e runs unforked (Decision 4)

The image reproduces `<repo>/architecture` + `<repo>/out`:

- App source copied to **`/workspace/architecture`**; **`WORKDIR /workspace/architecture`**.
- `uv sync` creates the venv at **`/workspace/architecture/.venv`** → `VENV_PY` resolves
  (§3.4).
- **`/workspace/out`** is the bind-mount target for the data halves (and where metrics/models
  are written) → `REPO_ROOT/out` resolves.
- `~/.flwr/config.toml` is materialized inside the container's HOME by
  `run_local_federation.sh` on `start` (ephemeral, per-container — fine).

Net: `smoke_pipeline.sh` calls the existing `run_local_federation.sh` with **zero edits** to
that script or `gen_certs.sh`.

### 4.2 `architecture/Dockerfile`

```dockerfile
FROM python:3.12-slim            # tag, not digest — digest pin is 1.3.a (§4.6)

# System deps (§3.1, §3.2). Only openssh-client is required AND absent from the slim base;
# libgomp1/openssl are defense-in-depth (see §3.1 — xgboost's wheel vendors libgomp, openssl
# ships in the base). The real, committed Dockerfile carries the full rationale in comments.
#   openssh-client — ssh-keygen for EC node-auth keys (gen_certs.sh:99); MISSING from slim
#   libgomp1       — xgboost's OpenMP runtime; insurance (wheel vendors its own copy)
#   openssl        — CA + server cert (gen_certs.sh); already present in slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 openssh-client openssl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv   # version tag, not :latest — the one
                                                             # floating input in an otherwise uv.lock-
                                                             # reproducible layer (digest pin is 1.3.a)

WORKDIR /workspace/architecture
COPY . /workspace/architecture/                # context IS architecture/ (§6.1); .dockerignore
                                               # (§4.3) excludes .venv/.secrets/.federation/out
RUN uv sync --locked --no-dev                  # venv at /workspace/architecture/.venv (§4.1)

ENV PATH="/workspace/architecture/.venv/bin:${PATH}"
CMD ["scripts/smoke_pipeline.sh"]
```

Notes: `--locked` fails the build if `uv.lock` is stale (guard, §2); `--no-dev` drops the
pytest group (the smoke run needs the app, not the test tools). **The build context is
`architecture/`, not the repo root** (§6.1) — that is what makes `architecture/.dockerignore`
the context-root ignore file Docker actually reads (§4.3), so `.secrets/` and `.venv/` are
excluded from the `COPY .` above. Docker only reads `.dockerignore` from the **root of the
build context** (or a Dockerfile-adjacent `Dockerfile.dockerignore` under BuildKit); a
`.dockerignore` in some other subdirectory is silently ignored. `out/` sits at the repo root,
outside this context, so it can never be copied in — it is bind-mounted at run time (§4.5).

### 4.3 `architecture/.dockerignore`

This file is the **build-context root** ignore file — it works only because the build context
is `architecture/` (§4.2, §6.1), so Docker reads `architecture/.dockerignore` as `<context>/.dockerignore`.
The patterns below are context-root-relative and mirror `architecture/.gitignore`. **This is
load-bearing for the "no secret key material in the image" acceptance criterion (§7):** without
it honored, `COPY .` would bake the host `.secrets/` (CA + node PRIVATE keys) and `.venv/` into
the image. Exclude from the build context:

```
.venv/          # rebuilt by uv sync; never copy the host venv
.secrets/       # host CA + node PRIVATE keys — must never enter the image (security)
.federation/    # run logs + state.db from host runs
out/            # data + artifacts are mounted, never baked in (Decision 2)
__pycache__/
*.py[cod]
```

The **FAB** build inside the container is a separate mechanism governed by the app-dir
`.gitignore` (§3.3), which is copied in and left intact — so `flwr run` still excludes `.venv`
and passes the depth-10 check inside the container.

### 4.4 `architecture/scripts/smoke_pipeline.sh` (entrypoint; runs in and out of Docker)

Orchestration, fail-fast, always-clean:

1. **Preflight.** Resolve `REPO_ROOT` from the script location; assert the two half files exist
   — if not, print a clear "mount the Geneva halves at /workspace/out (see 1.a)" message and exit
   non-zero. (Guards the "forgot `-v`" case; `run_local_federation.sh:193-196` also checks, but its
   message is not mount-aware.) The half **names are not hardcoded here** — read them from
   pyproject `[tool.fed_stroke.nodes].data-path` (today `out/geneva_half_A.parquet` /
   `out/geneva_half_B.parquet`) so this check can never drift from the node config the SuperNodes
   actually use.
   Resolve `SCRIPT_DIR` from `${BASH_SOURCE[0]}` (same idiom the reused scripts use); invoke the
   sibling launcher as `"${SCRIPT_DIR}/run_local_federation.sh"` everywhere below so the entrypoint
   works regardless of the caller's CWD (container `CMD` and local invocation both).
2. **`trap` cleanup.** Install `trap '"${SCRIPT_DIR}/run_local_federation.sh" stop' EXIT` so the
   topology is torn down on success, failure, or signal.
3. **Bring up the secure topology.** `"${SCRIPT_DIR}/run_local_federation.sh" start` — certs
   auto-generated by `ensure_certs` → `gen_certs.sh`, SuperLink up over TLS + node auth, both
   nodes registered and READY.
4. **One bagging run (the smoke gate).** From `/workspace/architecture` (the `WORKDIR`), invoke
   the run **with `--stream` so the command blocks until the run reaches a terminal state** —
   `flwr run` submits and returns immediately by default (flwr 1.31: "logs are not streamed by
   default"), so without `--stream` step 5 would race the ServerApp and read a not-yet-written
   artifact. Use absolute `model-dir` / `metrics-dir` (both default to the ServerApp CWD, which
   is not this dir, `run_local_federation.sh:41-45`); in the container `REPO_ROOT=/workspace`, so:

   ```bash
   flwr run . local-deployment --stream --run-config \
     "save-model=true model-dir='${REPO_ROOT}/out/models' metrics-dir='${REPO_ROOT}/out/metrics'"
   ```

   `${REPO_ROOT}` is the same value `smoke_pipeline.sh` resolved in step 1 (in-container it is
   `/workspace`). One bagging round is the gate — the full bagging+cyclic×2 matrix is 1.b/1.c/1.d
   verification, not a smoke concern. The default `train-method` is `bagging`, so the ServerApp
   writes `${REPO_ROOT}/out/metrics/bagging.json` (`server_app.py:189-192`, `tag = train_method`).

   **Completion detection.** `--stream` returning is the wait; the artifact assertion in step 5 is
   the actual pass/fail gate (a failed run leaves no valid artifact → step 5 exits non-zero). If
   `--stream` ever proves unreliable, the robust fallback is to submit with `--format json`, capture
   the `run-id`, and poll `flwr ls --run-id <id> --format json` until `status == "finished"` (bounded
   timeout, fail on a died SuperLink) before step 5 — but `--stream` is the primary path.
5. **Assert the artifact — with the smoke check's OWN assertions, not the write-path validator.**
   `server_app.py:184` already runs `validate_metrics_artifact` on the in-memory nested dict
   *before* the file is written, so a structurally broken run fails inside `flwr run` and never
   reaches this step. What the smoke gate must add is that the **file landed and carries a
   real two-site result.** Load `${REPO_ROOT}/out/metrics/bagging.json` and assert, non-zero exit
   on any failure:
   - the file exists and parses as JSON;
   - the last round contains **both** expected sites. The site key is the parquet **file name**,
     `Path(node_config["data-path"]).name` (`client_app.py:86,159`) — i.e. the JSON has site keys
     `"geneva_half_A.parquet"` and `"geneva_half_B.parquet"`. Derive the expected pair from the
     pyproject `data-path` basenames (step 1), do not hardcode, so it tracks the node config.
   - each site's `auc_roc` is a **finite float** (not `null` — `nest_site_metrics` maps NaN→None,
     so a single-class/degenerate site surfaces as `null`, which the smoke run must reject).

   Do **not** hand the reloaded JSON to `validate_metrics_artifact`: that validator requires
   **int** round keys (`metrics.py:290`), but JSON serializes them as strings, so it would raise
   on a perfectly valid artifact. It also checks required metric *keys* only — it does not verify
   two sites or finite AUC — so it is the wrong tool for this gate even after int-coercion.
6. Exit 0 on success (the `trap` stops the federation).

### 4.5 Data mount contract (Decision 2)

`docker run --rm --init -v "<repo>/out:/workspace/out" fed-stroke-smoke` (`--init` for PID-1
signal handling / zombie reaping, §8). The halves are only read;
metrics land in `out/metrics/` and the saved model in `out/models/` on the same mount, so
results survive the container. **No patient data is ever in the image** — the mount is the only
data path. The mount is read-write (for the artifact subdirs) but the halves themselves are
never written.

### 4.6 Scope line vs 1.3.a — forward-compatible, not gold-plated

1.f deliberately stops at "builds + runs on the dev machine". The Dockerfile is written so
1.3.a only *tightens* it: pin `FROM` by digest, pin the exact `libgomp1` version, add a
non-root `USER`, and add the clean-bootstrap acceptance run on a foreign machine. None of those
are added here (roadmap defers them); nothing here is throwaway — 1.3.a edits this same file.

### 4.7 Tests

Static invariants in `architecture/tests/test_docker_smoke.py` (pytest, no docker needed —
mirrors 1.e keeping the live e2e round out of pytest, §6):

- **Dockerfile**: `apt-get install` line contains `libgomp1`, `openssh-client`, `openssl`;
  uses `uv sync --locked`; `WORKDIR` is the architecture dir; `CMD` invokes
  `smoke_pipeline.sh`; there is **no** `COPY` of `out` or `.secrets` (guards against baking
  data/keys into the image).
- **`.dockerignore`**: excludes `.venv`, `.secrets`, `.federation`, `out`.
- **`smoke_pipeline.sh`**: exists and is executable; contains the fail-fast halves check and a
  `trap … stop` cleanup; invokes `flwr run` **with `--stream`** (proves the run is awaited, §4.4
  step 4) and with **absolute** `model-dir`/`metrics-dir` (`${REPO_ROOT}/out/...`, not relative);
  resolves the half names and the sibling `run_local_federation.sh` rather than hardcoding a CWD.

**Coverage caveat (important):** these static tests grep the `.dockerignore` *content*, which
does **not** prove Docker honors it — a misplaced ignore file (the A1 failure mode) would leave
these tests green while the built image still bakes `.secrets/`. The static suite guards the
source files, not the image. The one assertion that actually protects the "no secret key
material" criterion (§7) is a **build-time** check, so it belongs in the §6 scripted run: after
`docker build`, assert `.secrets/` (and `.federation/`) are absent from `/workspace/architecture`
inside the image (e.g. `docker run --rm --entrypoint sh fed-stroke-smoke -c '! test -e .secrets'`).
(Not `.venv/`: `uv sync` rebuilds it inside the image, so its absence is not a valid assertion.)

The live **image build + container run** is the §6 scripted verification, not a pytest (it
needs docker + the data mount) — exactly the split 1.e uses for its e2e round.

## 5. Files to change (for the follow-up implementation task)

- **New** `architecture/Dockerfile` — §4.2.
- **New** `architecture/.dockerignore` — §4.3.
- **New** `architecture/scripts/smoke_pipeline.sh` — §4.4 (executable).
- **New** `architecture/tests/test_docker_smoke.py` — §4.7 static invariants.
- **Modify** `architecture/roadmap.md` — check 1.f `[x]` only **after** the code lands; this
  spec task leaves it `[ ]`.
- **Docs** — `docs/logbook.md` entry when the code lands.

No change to `run_local_federation.sh`, `gen_certs.sh`, `architecture/.gitignore`, or the
`[tool.fed_stroke.*]` tables — the container reuses them unchanged (§3.4, §4.1).

## 6. Verification (end to end)

1. **Build.** From the repo root, with the context set to the app dir:
   `docker build -f architecture/Dockerfile -t fed-stroke-smoke architecture/` → image builds; the
   apt layer installs OpenMP + ssh-keygen + openssl; `uv sync --locked` succeeds. (Context =
   `architecture/` is what makes `architecture/.dockerignore` effective, §4.2/§4.3.)
2. **No-secrets check (proves A1's fix holds).**
   `docker run --rm --entrypoint sh fed-stroke-smoke -c '! test -e .secrets && ! test -e .federation'`
   → exits **0**, proving the `.dockerignore` was honored and no host CA/node private keys were
   baked into the image (§4.3, §7). This is the check the static §4.7 tests structurally cannot make.
   **NB:** the test dirs must be host-only ones nothing in the Dockerfile recreates. `.secrets/`
   (the security criterion) and `.federation/` qualify; `.venv/` does **not** — `uv sync` builds a
   fresh venv inside the image, so `.venv/pyvenv.cfg` always exists and testing its absence would
   fail on every correct build. (Its presence is desirable — it is the uv-built venv, not a host
   copy: `.dockerignore` excludes `.venv/` from `COPY .`, then `uv sync` recreates it.)
3. **Run.** `docker run --rm -v "$(pwd)/out:/workspace/out" fed-stroke-smoke` (add `--init` for
   clean PID-1 signal handling, §8) → `smoke_pipeline.sh` brings up the secure topology, completes
   one bagging round over the TLS channel, writes `out/metrics/bagging.json`, asserts it, tears
   down, and exits **0**.
4. **Artifact check.** `out/metrics/bagging.json` on the host has both sites and a finite AUC
   (the smoke gate's own assertion, §4.4 step 5).
5. **Negative check (proves the guard is real).** Temporarily drop **`openssh-client`** from the
   apt line, rebuild, run → the container must **fail fast** at `gen_certs.sh:99`
   (`ssh-keygen: command not found`, exit 127) before any training, demonstrating 1.f actually
   catches the bug it exists for. Restore the line.
   **NB (1.f finding, 2026-07-20):** dropping `libgomp1` instead does **not** fail — `xgboost==3.3.0`
   vendors its own libgomp in the wheel (§3.1), so the run completes normally. `openssh-client` is
   the one apt package both required and missing from the slim base; it is the real guard.

## 7. Acceptance criteria

- `docker build` succeeds on the dev machine from `python:3.12-slim` with the three system
  packages and `uv sync --locked` (§4.2).
- `docker run` with the `out/` mount completes **one bagging round end to end over the secure
  (TLS + node-auth) channel** (the run is invoked with `--stream` so the container blocks until it
  finishes, §4.4 step 4) and writes `out/metrics/bagging.json`. The smoke script's own assertions
  pass: file loads, **both** sites `geneva_half_A.parquet` / `geneva_half_B.parquet` present in the
  last round, each `auc_roc` a finite float (§4.4 step 5). (Note: `validate_metrics_artifact` runs
  inside the ServerApp pre-write, `server_app.py:184`, and is **not** re-used on the reloaded JSON —
  it requires int round keys that JSON round-trips to strings; see §4.4 step 5.)
- The image contains **no patient data and no secret key material** — data is mounted, and the
  `architecture/.dockerignore` (honored because the build context is `architecture/`, §4.2)
  excludes `out/`, `.secrets/`, and `.venv/`. **Proven at build time** by the §6.2 no-secrets
  check, not merely by the static content grep (§4.3, §4.5, §4.7 caveat).
- `run_local_federation.sh` / `gen_certs.sh` are reused **unchanged** (§4.1).
- The §4.7 static tests pass.
- Version pinning by digest, OS-OpenMP pin, non-root user, and the non-dev-machine
  clean-bootstrap test are **documented as deferred to 1.3.a**, not silently skipped (§4.6, §8).

## 8. Risks & notes

- **Unpinned base image + OS OpenMP (deferred to 1.3.a).** 1.f uses the `python:3.12-slim`
  *tag* and an unversioned `libgomp1`; the Python layer is reproducible via `uv.lock`, but the
  OS layer is not byte-reproducible until 1.3.a pins the digest + OpenMP version. Stated, not
  hidden.
- **Image size.** `flwr[simulation]` pulls Ray and friends, so the image is large. Acceptable
  for a smoke build; a slimmer runtime (drop the simulation extra, multi-stage build) is a
  1.3.a optimization, not a 1.f blocker.
- **Runs as root.** No privileged ops are needed by the secure topology; a non-root `USER` is a
  1.3.a hardening step (§4.6).
- **Data mount required — dev-machine scoped by design.** With no `-v out:` mount the preflight
  fails fast (§4.4). The live build+run therefore can't run in a data-less CI; that is
  intentional (roadmap: "on the dev machine"). Only the §4.7 static invariants run in CI. If a
  data-less CI smoke is later wanted, a synthetic-half generator would be the enabler — noted as
  a *possible* 1.3.a/CI enhancement, explicitly **not** built here (Decision 2).
- **Mount is read-write but never mutates the halves.** Metrics/models are written into
  `out/metrics` and `out/models`; the parquet halves are only read.
- **`uv.lock` staleness.** `--locked` turns a stale lock into a build failure (guard); if the
  build fails there, run `uv lock` in `architecture/` and commit before rebuilding.
- **Do not strip the app-dir `.gitignore` in the image (§3.3).** It is load-bearing for the
  in-container FAB build; the `.dockerignore` is a *separate* file and does not replace it.
- **PID 1 / zombie reaping.** `smoke_pipeline.sh` is the container `CMD`, so it runs as PID 1;
  bash-as-PID-1 does not reap zombies, and `run_local_federation.sh` detaches `setsid` daemons
  plus `flower-superexec` helpers that are `kill`ed on teardown. For a one-shot container that
  exits immediately this is harmless, but run with `docker run --init` (tini) for correct signal
  forwarding and reaping — cheap and idiomatic.
- **Base-image runtime deps beyond the three apt packages.** The reused 1.e scripts also need
  `bash`, `setsid` (util-linux), and coreutils/`grep`/`sed`. All ship in `python:3.12-slim`
  (Debian bookworm), so no action for 1.f — but a future base swap for size (e.g. Alpine, which
  drops `setsid` and ships busybox) would silently break `register_nodes`/teardown. Enumerated
  here so the dependency is not invisible.

## 9. Review decisions — audit trail

- **Decision 1 — topology = A (single container, loopback).** One container runs the full
  1.e secure topology on `127.0.0.1` then `flwr run`. Smallest thing that proves the image +
  pipeline; the loopback-only 1.e topology (§3.6) fits one container with no docker networking.
  Multi-container / cross-host (real SANs, container hostnames) is genuinely different work and
  is the substance of 1.3.b — pulling it into 1.f would be over-engineering a smoke build.
- **Decision 2 — smoke data = A (mount real Geneva halves, never bake).** Truest end-to-end
  run; keeps patient data out of the image (hygiene + "no data leaves a supernode"). Dev-machine
  scoped, matching the roadmap. A synthetic fallback (portable/CI) was considered and declined
  for 1.f to keep the surface minimal (§8).
- **Decision 3 — pinning = A (`uv.lock` Python layer now; base digest + OS-OpenMP pin +
  clean-room to 1.3.a).** `uv sync --locked` makes the Python deps reproducible essentially for
  free without over-building; the roadmap explicitly parks base-image/OS pinning and the
  non-dev-machine acceptance test in 1.3.a. Draws an honest line so 1.f is neither
  under-delivered nor gold-plated (§4.6).
- **Decision 4 — reuse `run_local_federation.sh` unchanged via container layout, no script
  fork.** Because that script derives all paths from its own location (§3.4), reproducing the
  host `<repo>/architecture` + `<repo>/out` tree lets the entire 1.e orchestration run verbatim.
  A Docker-specific fork of the launch logic was rejected — it would double the maintenance
  surface and drift from the path Shenzhen inherits.
- **Decision 5 — await the run with `flwr run --stream`, gate on the artifact (§4.4 steps 4-5).**
  `flwr run` submits and returns immediately in flwr 1.31 (logs off by default), so the smoke
  script must block on completion before asserting the artifact. `--stream` (follow logs until the
  run terminates) was chosen over polling `flwr ls --run-id --format json` because the artifact
  assertion is already the pass/fail gate — a failed run leaves no valid `bagging.json`, so no
  separate status-string parsing is needed. Polling is documented as the fallback if streaming
  proves flaky. The smoke gate asserts its **own** invariants (both sites present, finite AUC) and
  deliberately does **not** re-use `validate_metrics_artifact` on the reloaded JSON — that
  pre-write validator requires int round keys that JSON serializes to strings and would raise on a
  valid file (found in this review; see §4.4 step 5).

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|--------|---------|------|--------|----------|
| Eng Review | `/plan-eng-review` | 2 | PASS (revised) | 3 substantive + 4 nits + completeness pass, all folded in |
| Codex Review | `/codex review` | 0 | — | — |

**Findings (all grounded against the cited source and resolved in this revision):**

- **A1 — CRITICAL (security + correctness).** `.dockerignore` at `architecture/.dockerignore`
  with build context = repo root is read by neither Docker nor BuildKit, so `COPY architecture/`
  baked host `.secrets/` (CA + node PRIVATE keys) and `.venv/` into the image — contradicting the
  §7 "no secret key material" criterion. **Fixed:** build context set to `architecture/`, `COPY .`,
  §4.2/§4.3/§6.1 rewritten; new §6.2 build-time no-secrets check added.
- **A2 — CORRECTNESS.** §4.4 step 5 fed reloaded JSON to `validate_metrics_artifact`, which
  requires **int** round keys (`metrics.py:290`; JSON keys are strings → always raises) and checks
  only required metric keys — not "both sites present / finite AUC" as claimed. **Fixed:** step 5
  now names the artifact (`bagging.json`, `server_app.py:189`) and specifies the smoke gate's own
  assertions (file loads, both sites in last round, finite `auc_roc`).
- **T1 — TEST GAP.** §4.7 static content-grep cannot catch A1 (Docker never reads the file), so it
  would go green on a secret-baking image. **Fixed:** §4.7 caveat added; the real guard is the
  build-time §6.2 check.
- **Nits (folded in):** A3 pinned `uv:0.8` (was `:latest`, §4.2); P1 `docker run --init` for PID-1
  reaping (§8, §6.3, §4.5); C1 preflight reads half names from pyproject instead of hardcoding
  (§4.4.1); C2 base-image deps beyond the 3 apt packages enumerated (§8).

**Completeness pass (2nd run — make the spec a self-contained implementation source):**

- **A4 — run completion was unspecified (would have raced).** `flwr run` submits and returns
  immediately in flwr 1.31; step 5 asserted the artifact with no wait. **Fixed:** step 4 now uses
  `flwr run --stream` to block until the run terminates, with the `flwr ls --run-id --format json`
  poll as the documented fallback; recorded as Decision 5 (§9).
- **Site-key names pinned.** Verified `site = Path(node_config["data-path"]).name`
  (`client_app.py:86,159`); step 5 now names the exact keys `geneva_half_A.parquet` /
  `geneva_half_B.parquet` (derived from pyproject, not hardcoded).
- **Concrete paths + entrypoint resolution.** Step 4 replaces the `<abs>` placeholder with
  `${REPO_ROOT}/out/...` (in-container `/workspace`); steps 1-3 resolve `SCRIPT_DIR` and invoke the
  sibling `run_local_federation.sh` by path so the `CMD` is CWD-independent.
- **§7 acceptance criterion 2 re-aligned** to the smoke script's own assertions (it still cited
  `validate_metrics_artifact` as the gate, now corrected); §4.5 run command aligned with `--init`;
  §4.7 test extended to assert `--stream` + absolute dirs.

Verified accurate, no change needed: path derivation (§3.4 vs `run_local_federation.sh:48-55`),
FAB `.gitignore` mechanism (`.secrets/`/`.federation/`/`.venv/` all excluded), `local-deployment`
connection name (pyproject), site-key derivation, and every cited line number.

- **UNRESOLVED:** 0
- **VERDICT:** READY — spec revised under `/plan-eng-review`; all findings resolved in-document.
  Implementation task may proceed from this revision (roadmap 1.f stays `[ ]` until code lands).

NO UNRESOLVED DECISIONS
