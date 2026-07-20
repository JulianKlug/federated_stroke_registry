"""Static invariants of the 1.f Docker smoke build (spec §4.7).

Sub-second, no docker required (matching the house pytest idiom, same split 1.e
uses): these assert the source files — Dockerfile, .dockerignore, and the
smoke_pipeline.sh entrypoint — carry the invariants the smoke build depends on.

Coverage caveat (spec §4.7): these grep source *content*. They CANNOT prove
Docker honors .dockerignore — a misplaced ignore file (the A1 failure mode) would
leave these green while the built image still bakes .secrets/. The guarantee that
no secret key material enters the image is the BUILD-TIME no-secrets check in the
§6 scripted run (docker build, then assert .secrets/ and .venv/ are absent inside
the image), not this pytest. The live image build + container run is likewise a
scripted verification (needs docker + the data mount), not a test here.
"""

import os
import re
from pathlib import Path

ARCH_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = ARCH_DIR / "Dockerfile"
DOCKERIGNORE = ARCH_DIR / ".dockerignore"
SMOKE = ARCH_DIR / "scripts" / "smoke_pipeline.sh"


def _dockerfile() -> str:
    return DOCKERFILE.read_text()


def _apt_install_line(text: str) -> str:
    """The `apt-get install ... \\` line and its continuation (the package list)."""
    # The install spans two lines (backslash continuation); join then collapse.
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "apt-get install" in line:
            return line
    return ""


# --- Dockerfile invariants ---------------------------------------------------


def test_dockerfile_exists():
    assert DOCKERFILE.is_file(), "architecture/Dockerfile missing"


def test_dockerfile_installs_the_three_system_packages():
    # §3.2: openssh-client (ssh-keygen) is the load-bearing one — required by
    # gen_certs.sh and absent from python:3.12-slim; the §6.5 negative check proves
    # dropping it fails the container fast. libgomp1/openssl are kept as
    # defense-in-depth (§3.1: xgboost's wheel vendors its own libgomp; openssl ships
    # in the base), so this asserts all three stay declared.
    apt = _apt_install_line(_dockerfile())
    for pkg in ("libgomp1", "openssh-client", "openssl"):
        assert pkg in apt, f"Dockerfile apt-get install line missing {pkg}: {apt!r}"


def test_dockerfile_uv_sync_is_locked():
    # --locked turns a stale uv.lock into a build failure (reproducibility guard).
    assert re.search(r"uv sync\b[^\n]*--locked", _dockerfile()), "uv sync must use --locked"


def test_dockerfile_workdir_is_architecture_dir():
    # §4.1: venv must land at /workspace/architecture/.venv so VENV_PY resolves.
    assert re.search(r"^\s*WORKDIR\s+/workspace/architecture\s*$", _dockerfile(), re.M)


def test_dockerfile_cmd_invokes_smoke_pipeline():
    assert "smoke_pipeline.sh" in _dockerfile(), "CMD must invoke smoke_pipeline.sh"


def test_dockerfile_does_not_bake_data_or_secrets():
    # Guards against a COPY that would bake patient data / private keys into the
    # image. `COPY .` (whole context) is fine — .dockerignore excludes them; an
    # explicit `COPY out` / `COPY .secrets` is not.
    for line in _dockerfile().splitlines():
        stripped = line.strip()
        if stripped.startswith("COPY") and "--from=" not in stripped:
            assert not re.search(r"\b(out|\.secrets)\b", stripped), (
                f"Dockerfile must not COPY data/secrets into the image: {stripped!r}"
            )


# --- .dockerignore invariants ------------------------------------------------


def test_dockerignore_excludes_venv_secrets_federation_out():
    # §4.3: load-bearing for the "no secret key material in the image" criterion.
    entries = {
        line.split("#", 1)[0].strip().rstrip("/")
        for line in DOCKERIGNORE.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    }
    for pat in (".venv", ".secrets", ".federation", "out"):
        assert pat in entries, f".dockerignore must exclude {pat!r}; got {sorted(entries)}"


# --- smoke_pipeline.sh invariants --------------------------------------------


def test_smoke_pipeline_exists_and_is_executable():
    assert SMOKE.is_file(), "scripts/smoke_pipeline.sh missing"
    assert os.access(SMOKE, os.X_OK), "scripts/smoke_pipeline.sh must be executable"


def test_smoke_pipeline_has_failfast_halves_check():
    text = SMOKE.read_text()
    # Preflight resolves the half names from pyproject (not hardcoded) and checks
    # them under REPO_ROOT/out before starting anything.
    assert "fed_stroke" in text and "nodes" in text, "must read node config from pyproject"
    assert "data-path" in text, "must derive half names from node data-path"
    assert re.search(r"REPO_ROOT.*/out/", text), "must check halves under REPO_ROOT/out"


def test_smoke_pipeline_traps_cleanup():
    text = SMOKE.read_text()
    assert re.search(r"trap\b.*stop", text), "must install a `trap … stop` cleanup"


def test_smoke_pipeline_awaits_the_run_with_stream():
    # §4.4 step 4: --stream blocks until the run terminates, else step 5 races the
    # artifact write.
    text = SMOKE.read_text()
    assert re.search(r"flwr\b[^\n]*run\b[^\n]*--stream", text), "flwr run must use --stream"


def test_smoke_pipeline_uses_absolute_metrics_and_model_dirs():
    # Both default to the ServerApp CWD; a relative value writes the artifact
    # somewhere unexpected. Must be ${REPO_ROOT}/out/... (absolute at run time).
    text = SMOKE.read_text()
    assert "model-dir='${REPO_ROOT}/out/models'" in text
    assert "metrics-dir='${REPO_ROOT}/out/metrics'" in text


def test_smoke_pipeline_resolves_sibling_launcher_not_a_fixed_cwd():
    # The CMD runs regardless of caller CWD: resolve SCRIPT_DIR from BASH_SOURCE
    # and invoke run_local_federation.sh by path.
    text = SMOKE.read_text()
    assert "BASH_SOURCE" in text, "must resolve SCRIPT_DIR from BASH_SOURCE"
    assert "run_local_federation.sh" in text, "must invoke the sibling launcher by path"
