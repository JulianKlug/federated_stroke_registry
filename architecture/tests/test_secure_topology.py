"""Mechanically-verifiable invariants of the 1.e secure topology (spec §4.9).

Sub-second, no live federation (matching the house pytest idiom): cert/key
material generated once into a tmp dir by the real gen_certs.sh, committed
pyproject invariants, and the ~/.flwr/config.toml generation step run
hermetically via FLWR_HOME. The end-to-end secure round and the negative
security checks are scripted verifications (run_local_federation.sh start /
verify), not pytests.
"""

import datetime
import ipaddress
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_ssh_private_key,
    load_ssh_public_key,
)

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCH_DIR.parent
GEN_CERTS = ARCH_DIR / "scripts" / "gen_certs.sh"
LAUNCHER = ARCH_DIR / "scripts" / "run_local_federation.sh"
NODES = ("node_A", "node_B")


@pytest.fixture(scope="session")
def secrets_dir(tmp_path_factory):
    """Run the real gen_certs.sh once into a tmp dir (doubles as its test)."""
    out = tmp_path_factory.mktemp("secrets")
    subprocess.run([str(GEN_CERTS), str(out)], check=True, capture_output=True)
    return out


def _pyproject():
    with open(ARCH_DIR / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


# --- gen_certs.sh output invariants -----------------------------------------


def test_gen_certs_creates_expected_files_with_tight_perms(secrets_dir):
    private = ["ca.key", "server.key", *NODES]
    public = ["ca.crt", "server.pem", *(f"{n}.pub" for n in NODES)]
    for name in private + public:
        assert (secrets_dir / name).is_file(), f"missing {name}"
    for name in private:
        mode = (secrets_dir / name).stat().st_mode & 0o777
        assert mode & 0o077 == 0, f"{name} readable by group/other: {oct(mode)}"


def test_gen_certs_is_idempotent_without_force(secrets_dir):
    before = {p.name: p.read_bytes() for p in secrets_dir.iterdir()}
    res = subprocess.run(
        [str(GEN_CERTS), str(secrets_dir)], check=True, capture_output=True, text=True
    )
    assert "already present" in res.stdout
    after = {p.name: p.read_bytes() for p in secrets_dir.iterdir()}
    assert before == after


def test_node_public_keys_are_openssh_p384(secrets_dir):
    # Mirrors flwr's register CLI (register.py:125-137): load_ssh_public_key
    # → EllipticCurvePublicKey on a NIST curve.
    for n in NODES:
        key = load_ssh_public_key((secrets_dir / f"{n}.pub").read_bytes())
        assert isinstance(key, ec.EllipticCurvePublicKey)
        assert key.curve.name == "secp384r1"


def test_node_private_keys_load_like_flower_supernode_does(secrets_dir):
    # Mirrors flower_supernode.py:283-302: load_ssh_private_key, then the
    # public key is DERIVED from it (--auth-supernode-public-key deprecated).
    for n in NODES:
        priv = load_ssh_private_key((secrets_dir / n).read_bytes(), password=None)
        assert isinstance(priv, ec.EllipticCurvePrivateKey)
        derived = priv.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        on_disk = load_ssh_public_key((secrets_dir / f"{n}.pub").read_bytes())
        assert derived == on_disk.public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)


def test_server_cert_san_covers_loopback(secrets_dir):
    cert = x509.load_pem_x509_certificate((secrets_dir / "server.pem").read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    ips = set(san.get_values_for_type(x509.IPAddress))
    names = set(san.get_values_for_type(x509.DNSName))
    assert ipaddress.ip_address("127.0.0.1") in ips
    assert ipaddress.ip_address("::1") in ips
    assert "localhost" in names


def test_server_cert_validity_bounded_and_ca_signed(secrets_dir):
    cert = x509.load_pem_x509_certificate((secrets_dir / "server.pem").read_bytes())
    ca = x509.load_pem_x509_certificate((secrets_dir / "ca.crt").read_bytes())
    lifetime = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert lifetime <= datetime.timedelta(days=91), "server cert not short-lived"
    assert cert.issuer == ca.subject
    ca.public_key().verify(
        cert.signature, cert.tbs_certificate_bytes, ec.ECDSA(cert.signature_hash_algorithm)
    )


# --- committed pyproject invariants ------------------------------------------


def test_pyproject_has_no_flwr_federations_block():
    # §3.9 tripwire: a committed [tool.flwr.federations] block would make
    # `flwr run` migrate it into ~/.flwr/config.toml and comment it out of the
    # tracked pyproject.toml.
    assert "federations" not in _pyproject()["tool"]["flwr"]


def test_fed_stroke_superlink_table_invariants():
    sl = _pyproject()["tool"]["fed_stroke"]["superlink"]
    for key in ("name", "address", "fleet-address", "ca-cert"):
        assert key in sl, f"[tool.fed_stroke.superlink] missing {key}"
    assert "insecure" not in sl
    ca = Path(sl["ca-cert"])
    assert not ca.is_absolute(), "ca-cert must be repo-relative (portability, §4.5)"
    assert ".secrets" in ca.parts, "ca-cert must live in the gitignored secrets dir"


def test_fed_stroke_nodes_map_points_at_the_halves():
    nodes = _pyproject()["tool"]["fed_stroke"]["nodes"]
    assert len(nodes) == 2
    data_paths = [n["data-path"] for n in nodes.values()]
    assert sorted(Path(p).name for p in data_paths) == [
        "geneva_half_A.parquet",
        "geneva_half_B.parquet",
    ]
    appios = [n["appio"] for n in nodes.values()]
    keys = [n["key"] for n in nodes.values()]
    assert len(set(appios)) == len(appios)
    assert len(set(keys)) == len(keys)
    sl = _pyproject()["tool"]["fed_stroke"]["superlink"]
    reserved = {sl["address"], sl["fleet-address"]}
    assert not reserved & set(appios), "appio ports collide with SuperLink ports"


# --- ~/.flwr/config.toml generation: merge, not truncate (§4.5, X7) ----------


def test_connection_generation_merges_not_truncates(tmp_path):
    flwr_home = tmp_path / "flwrhome"
    flwr_home.mkdir()
    (flwr_home / "config.toml").write_text(
        '[superlink]\ndefault = "local"\n\n'
        '[superlink.local]\naddress = ":local:"\n\n'
        '[superlink.local-deployment]\naddress = "127.0.0.1:9093"\ninsecure = true\n'
    )
    subprocess.run(
        [str(LAUNCHER), "write-config"],
        check=True,
        capture_output=True,
        env={**os.environ, "FLWR_HOME": str(flwr_home)},
    )
    with open(flwr_home / "config.toml", "rb") as f:
        cfg = tomllib.load(f)["superlink"]
    # pre-existing connections preserved (merge, not truncate)
    assert cfg["default"] == "local"
    assert cfg["local"]["address"] == ":local:"
    # the generated entry is secure: pinned CA (absolute), no insecure flag
    ld = cfg["local-deployment"]
    assert "insecure" not in ld
    assert Path(ld["root-certificates"]).is_absolute()
    assert Path(ld["root-certificates"]).name == "ca.crt"
    sl = _pyproject()["tool"]["fed_stroke"]["superlink"]
    assert ld["address"] == sl["address"]


# --- secret hygiene -----------------------------------------------------------


def test_secrets_dir_is_gitignored_for_git_and_fab():
    # architecture/.gitignore is also the FAB packager's exclusion list — the
    # entry there is what keeps keys out of the bundle (spec §4.7).
    assert ".secrets/" in (ARCH_DIR / ".gitignore").read_text().splitlines()
    assert "architecture/.secrets/" in (REPO_ROOT / ".gitignore").read_text().splitlines()
