"""Site-agnostic node artifacts: the frozen table → node parquet + smoke report.

The last two steps of EVERY site's preprocessing, shared so the two sites cannot drift:
Geneva binds them in architecture/preprocessing/preprocess_gva.py, Shenzhen in
architecture/preprocessing/preprocess_shenzhen.py. The parquet stays at its site; the smoke
report is the ONLY artifact designed to cross the border (aggregate-only, §1.3 hand-off spec).

Deliberately free of any fed_stroke import: this module ships in the partner bundle, which
carries no architecture layer. The schema version it stamps therefore comes from the
preprocessing-side mirror (mappings.frozen_schema.SCHEMA_VERSION); the architecture tests
assert the two mirrors agree.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .mappings.frozen_schema import (
    FROZEN_FEATURES,
    FROZEN_OUTCOMES,
    FROZEN_UNITS,
    ID_COL,
    SCHEMA_VERSION,
    is_binary_unit,
    validate_frozen_columns,
)

# Provenance of a node file: real site data at the frozen schema. The nodes' own
# `data-provenance` key (pyproject [tool.fed_stroke.nodes]) is flipped to match, by hand,
# only once the file exists — it arms the DP ledger.
PROVENANCE = "real-frozen-schema"

METADATA_PREFIX = "fed_stroke."   # key prefix of the provenance stamp in the parquet key-value metadata
_JSON_STAMP_KEYS = ("source_files", "feature_cols", "outcome_cols", "anonymisation")
# Per-feature spread in the smoke report: percentiles, never a single row's min / max — the report
# crosses sites, and an extreme value is one identifiable patient's value.
SMOKE_QUANTILES = (0.05, 0.95)


def _quantile_key(q: float) -> str:
    return f"p{round(q * 100):02d}"


def sha256_of(path: Path) -> str:
    """Hex sha256 of one file, streamed (the EHR CSVs are gigabytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def smoke_report(df: pd.DataFrame, source_files: dict[str, str], created: str,
                 node_file: str) -> dict:
    """AGGREGATE-ONLY summary of a frozen table — the roadmap smoke-test artifact, the one
    report designed to cross sites for the Geneva/Shenzhen comparison (divergence gates:
    per-feature median > 10x apart, missing-rate > 25 pp apart, label balance > 3x apart).

    Contents: row and distinct-patient counts; per outcome column recorded/missing and the
    positive rate (the candidate label balance — the label itself is fed_stroke.schema's
    choice, not fixed in the table); per feature median / SMOKE_QUANTILES / missing rate; the
    anonymisation stamp; the unit check and the plausibility (out-of-range) report from
    wide_to_frozen when the frame still carries them in df.attrs (attrs are NOT written to the
    parquet, so a frame read back from disk reports them as unavailable); the cohort exclusion
    chain. Never an id, never a value of an individual row.
    """
    attrs = df.attrs
    n_rows = int(len(df))
    pids = df[ID_COL].astype(str).str.split("_").str[0]

    def _stat(fn, s, n):
        return float(fn(s)) if n else None

    features = {}
    for c in FROZEN_FEATURES:
        s = df[c]
        n = int(s.notna().sum())
        features[c] = {
            "n_recorded": n,
            "missing_rate": float(s.isna().mean()) if n_rows else None,
            "median": _stat(pd.Series.median, s, n),
            **{_quantile_key(q): _stat(functools.partial(pd.Series.quantile, q=q), s, n)
               for q in SMOKE_QUANTILES},
        }
    outcomes = {}
    for c in FROZEN_OUTCOMES:
        s = df[c]
        n = int(s.notna().sum())
        entry = {"n_recorded": n, "n_missing": int(s.isna().sum()),
                 "missing_rate": float(s.isna().mean()) if n_rows else None}
        if is_binary_unit(FROZEN_UNITS[c]):
            entry["n_positive"] = int(s.sum()) if n else 0
            entry["positive_rate"] = _stat(pd.Series.mean, s, n)   # candidate label balance
        else:
            entry["median"] = _stat(pd.Series.median, s, n)
        outcomes[c] = entry

    unit_check = attrs.get("unit_check")
    out_of_range = attrs.get("out_of_range")
    unavailable = {"pass": None,
                   "note": "not available: the frame did not come from wide_to_frozen (no attrs)"}
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": PROVENANCE,
        "created": created,
        "node_file": node_file,
        "source_files": dict(source_files),
        "n_rows": n_rows,
        "n_patients": int(pids.nunique()),
        "n_features": len(FROZEN_FEATURES),
        "n_outcomes": len(FROZEN_OUTCOMES),
        "anonymisation": attrs.get("anonymisation"),
        "label": ("not fixed in the table — fed_stroke.schema.TARGET_COL / TARGET_RULE; "
                  "positive_rate per binary outcome is the candidate label balance"),
        "outcomes": outcomes,
        "features": features,
        "unit_check": ({"pass": unit_check["pass"],
                        "n_columns_checked": unit_check["n_columns_checked"],
                        "n_assumed_unit_total": unit_check["n_assumed_unit_total"],
                        "warnings": list(unit_check.get("warnings", []))}
                       if unit_check else dict(unavailable)),
        "out_of_range": ({"pass": out_of_range["pass"],
                          "n_nulled_total": out_of_range["n_nulled_total"],
                          "max_out_of_range_frac": out_of_range["max_out_of_range_frac"],
                          "nulled": {f: {"n_below": c["n_below"], "n_above": c["n_above"],
                                         "n_values": c["n_values"]}
                                     for f, c in out_of_range["columns"].items()
                                     if c["n_below"] or c["n_above"]}}
                         if out_of_range else dict(unavailable)),
        "exclusions": attrs.get("exclusions"),
        "build_log": attrs.get("build_log_path"),
    }


def read_node_metadata(path: Path) -> dict:
    """The fed_stroke.* provenance stamp of a node parquet (JSON-valued fields decoded):
    schema_version, provenance, created, source_files, feature_cols, outcome_cols, n_rows,
    anonymisation (spec, age_jitter_years, age_top_coded_n, key_fingerprint)."""
    meta = pq.read_schema(str(path)).metadata or {}
    stamp = {}
    for key, value in meta.items():
        key = key.decode("utf-8")
        if not key.startswith(METADATA_PREFIX):
            continue
        field = key[len(METADATA_PREFIX):]
        text = value.decode("utf-8")
        if field in _JSON_STAMP_KEYS:
            stamp[field] = json.loads(text)
        elif field == "n_rows":
            stamp[field] = int(text)
        else:
            stamp[field] = text
    return stamp


def write_node_parquet(df: pd.DataFrame, out_path: Path,
                       source_files: dict[str, str]) -> dict:
    """Write ONE site's node parquet + provenance metadata; return the smoke summary.

    This is the artifact a SuperNode's node_config data-path points at — for the
    GVA node in the v1.3 cross-site topology, and (via split_gva_halves.py) for
    each loopback half in the Geneva-only phase.

    Input:
        df: a DE-IDENTIFIED frozen-schema table (build_frozen_gva_table output or a half of
            it): columns EXACTLY [ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES] in that ORDER
            (the loader emits X in FEATURE_COLS order), features and outcomes float64, ids
            unique, df.attrs['anonymisation'] present (anonymise_frozen's stamp; a half read
            back from parquet copies it from read_node_metadata). Anything else raises BEFORE
            any file is written — a raw wide_to_frozen frame fails on its column name.
        out_path: parquet destination (must match the node's data-path).
        source_files: {label: sha256} of the inputs (hash_inputs), stamped for audit.
    Output:
        dict, the roadmap smoke-test artifact (smoke_report) — AGGREGATE-ONLY: row and
        patient counts, per outcome availability / positive rate, per feature median /
        p05 / p95 / missing rate, the anonymisation stamp, unit-check and out-of-range
        pass/fail, exclusion chain. Also written next to the parquet as
        <stem>_smoke_report.json (the returned dict carries its path under 'smoke_report_path').

    Side effects:
        - parquet key-value metadata stamped under the 'fed_stroke.' prefix: SCHEMA_VERSION,
          PROVENANCE, source hashes, creation date, the frozen column lists, the anonymisation
          stamp — the parquet-side provenance marker (B F8); read it back with
          read_node_metadata. Frame attrs are NOT written: pandas would serialise them, and
          they carry raw single-row extremes (out_of_range min/max_seen) and a local path;
        - after the node file exists, flip that node's data-provenance to
          'real-frozen-schema' in pyproject [tool.fed_stroke.nodes] (manual,
          deliberate — it arms the ledger).
    """
    out_path = Path(out_path)
    expected = [ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES]
    validate_frozen_columns(df.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=ID_COL)
    if list(df.columns) != expected:
        raise ValueError(
            f"write_node_parquet: column ORDER must be [{ID_COL}, *FROZEN_FEATURES, "
            "*FROZEN_OUTCOMES] (the loader emits X in this order) — run finalize_frozen_table."
        )
    if df[ID_COL].isna().any() or not df[ID_COL].is_unique:
        raise ValueError(f"write_node_parquet: {ID_COL} must be present and unique per row.")
    non_float = [c for c in expected[1:] if str(df[c].dtype) != "float64"]
    if non_float:
        raise ValueError(f"write_node_parquet: features and outcomes must be float64; got {non_float}")
    if "anonymisation" not in df.attrs:
        raise ValueError(
            "write_node_parquet: frame carries no anonymisation stamp — run "
            "preprocessing.anonymise.anonymise_frozen (a half read back from parquet copies the "
            "stamp from read_node_metadata)."
        )

    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    report_path = out_path.with_name(out_path.stem + "_smoke_report.json")
    report = smoke_report(df, source_files, created, out_path.name)
    report["smoke_report_path"] = str(report_path)

    stamp = {
        "schema_version": SCHEMA_VERSION,
        "provenance": PROVENANCE,
        "created": created,
        "source_files": json.dumps(dict(source_files), sort_keys=True),
        "feature_cols": json.dumps(list(FROZEN_FEATURES)),
        "outcome_cols": json.dumps(list(FROZEN_OUTCOMES)),
        "n_rows": str(len(df)),
        "anonymisation": json.dumps(df.attrs["anonymisation"], sort_keys=True),
    }
    # attrs stay OUT of the file: pandas would serialise them (out_of_range min/max_seen = raw
    # single-row extremes, the local build-log path); the fed_stroke.* stamp is the only channel
    plain = df.copy(deep=False)
    plain.attrs = {}
    table = pa.Table.from_pandas(plain, preserve_index=False)
    metadata = {**(table.schema.metadata or {}),
                **{(METADATA_PREFIX + k).encode("utf-8"): v.encode("utf-8") for k, v in stamp.items()}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), str(out_path))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
