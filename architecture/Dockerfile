# Docker smoke build for the federated stroke pipeline (roadmap 1.f,
# docs/specs/1f_docker_smoke_build.md). Proves ONCE, on the dev machine, that the
# image builds and runs the real 1.e secure topology end to end inside a single
# loopback container — so a Dockerfile bug is caught here, not during Shenzhen
# onboarding (1.3.a).
#
# The image reproduces the host tree (<repo>/architecture + <repo>/out, venv at
# <repo>/architecture/.venv) so run_local_federation.sh / gen_certs.sh run
# VERBATIM, unforked — they derive every path from their own location (§3.4/§4.1).
#
# Build context is architecture/ (NOT the repo root) — that is what makes
# architecture/.dockerignore the context-root ignore file Docker actually reads
# (§4.2/§4.3), so host .secrets/ and .venv/ never enter the image:
#   docker build -f architecture/Dockerfile -t fed-stroke-smoke architecture/
#
# Base image is pinned by TAG, not digest; libgomp1 is unversioned. Digest pin +
# exact OS-OpenMP pin + non-root USER + clean-bootstrap on a foreign machine are
# deferred to 1.3.a (§4.6, §8), which tightens this same file.
FROM python:3.12-slim

# System deps for the pipeline (§3.1, §3.2). Empirically (1.f verification, 2026-07-20),
# only openssh-client is BOTH required AND missing from python:3.12-slim; the other two
# are defense-in-depth (kept deliberately, not load-bearing for the pinned deps):
#   openssh-client — REQUIRED & MISSING: ssh-keygen, for gen_certs.sh's EC P-384 node-auth
#                    keypairs (gen_certs.sh:99). The slim base has no ssh-keygen, so without
#                    this the container fails fast at cert generation ("ssh-keygen: command
#                    not found"). This is the real Dockerfile bug 1.f catches.
#   libgomp1       — xgboost's OpenMP runtime. NOT strictly needed for xgboost==3.3.0: its
#                    manylinux wheel vendors its own libgomp (xgboost.libs/libgomp-*.so.1),
#                    so `import xgboost` works without the system copy. Kept as insurance
#                    against a future xgboost that links the system libgomp (source build,
#                    non-manylinux wheel). 1.3.a pins the exact version.
#   openssl        — CA + server cert in gen_certs.sh. Already present in python:3.12-slim
#                    today; declared explicitly so the dependency survives a base-image change.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 openssh-client openssl \
    && rm -rf /var/lib/apt/lists/*

# uv pinned to a minor tag (not :latest) — the one floating input in an otherwise
# uv.lock-reproducible Python layer; digest pin is 1.3.a.
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /workspace/architecture
# Context IS architecture/ (see header); .dockerignore excludes .venv/.secrets/
# .federation/out. out/ lives at the repo root, outside this context, so it can
# never be copied in — it is bind-mounted at run time (§4.5).
COPY . /workspace/architecture/
# venv at /workspace/architecture/.venv → VENV_PY resolves (§4.1). --locked fails
# the build if uv.lock is stale (guard, §2); --no-dev drops the pytest group.
RUN uv sync --locked --no-dev

ENV PATH="/workspace/architecture/.venv/bin:${PATH}"
CMD ["scripts/smoke_pipeline.sh"]
