"""Shenzhen preprocessing: the site export → the frozen-schema table the Shenzhen node trains on.


Contract with the architecture layer — identical to Geneva's:
- one row per admission — EVERY cohort admission, whatever its outcome; columns EXACTLY
  [ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES] — no PII, no extras;
- DE-IDENTIFIED (preprocessing.anonymise, the last step before the write): ID_COL is
  '<hmac-sha256(key, patient_id)[:16]>_<admission ordinal>'; age is completed years + a
  per-patient keyed jitter, top-coded at 90. The key is Shenzhen's own, generated once, kept
  outside the repo and away from the artefacts; only its fingerprint is stamped;
- the training LABEL is NOT decided here: fed_stroke.schema picks it among the outcome columns
  and the loader drops the unlabelled rows at load time;
- values IN the frozen units of record (FROZEN_UNITS); binaries encoded per
  preprocessing.mappings.shenzhen_encodings (sex: 1 = female);
- missing values stay NaN; values outside FROZEN_RANGES become NaN and are counted; a feature
  out of range wholesale fails the build (unit / mapping bug);
- NO one-row-per-patient dedup, NO train/valid split, NO sentinel encoding, NO label drop —
  all four are loader-side (fed_stroke/task.py).

Usage:
    python preprocess_shenzhen.py \
        --export /.../shenzhen_export.xlsx \
        --out    out/shenzhen_frozen.parquet \
        --pseudonym-key ~/.fed_stroke/shenzhen_pseudonym.key
    # key, once, immutable for the project (the loader's split hashes the pseudonym):
    #   python -c "import secrets,sys; sys.stdout.buffer.write(secrets.token_bytes(32))" > KEY
    #   chmod 600 KEY
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.anonymise import anonymise_frozen, read_pseudonym_key  # noqa: E402
from preprocessing.build_log import format_build_log, patient_ids, write_build_log  # noqa: E402
from preprocessing.frozen_table import (  # noqa: E402
    convert_to_frozen_units,
    encode_binaries,
    finalize_frozen_table,
    format_out_of_range,
    format_unit_check,
    nullify_out_of_range,
    rename_to_frozen,
    select_mapped_columns,
)
from preprocessing.mappings import (  # noqa: E402
    FROZEN_FEATURES,
    FROZEN_OUTCOME_RANGES,
    FROZEN_OUTCOMES,
    FROZEN_RANGES,
    FROZEN_UNITS,
    RAW_ID_COL,
    SCHEMA_VERSION,
    SHENZHEN_BINARY_ENCODINGS,
    SHENZHEN_BINARY_FILLNA,
    SHENZHEN_DECLARED_UNITS,
    SHENZHEN_TO_FROZEN,
    UNIT_ALIASES,
    is_binary_unit,
    validate_frozen_columns,
    validate_shenzhen_encodings,
)
from preprocessing.node_parquet import PROVENANCE, sha256_of, write_node_parquet  # noqa: E402

# Timing features are DERIVED from timestamps, never exported ready-made (Geneva does the same
# in preprocessing.registry_cohort). These are the raw names SHENZHEN_TO_FROZEN expects back.
TIMING_COLS = ("ODT", "ONT", "DNT", "OPT")

DEFAULT_BUILD_LOG = REPO_ROOT / "out" / "shenzhen_frozen.build.log"


# ---------------------------------------------------------------- site-specific (TODO shenzhen)

def load_shenzhen_export(export_path: Path) -> pd.DataFrame:
    """Read the site export into ONE wide frame, one row per admission, raw column names.

    Raw names must be exactly the keys of mappings.frozen_schema.SHENZHEN_TO_FROZEN — rename
    here if the export headers differ, rather than editing the mapping (the mapping is the
    reviewed, signed-off artifact).
    """
    raise NotImplementedError("TODO(shenzhen): read the site export")


def build_shenzhen_cohort(raw: pd.DataFrame, exclusions: list[dict]) -> pd.DataFrame:
    """Apply the agreed cohort filter, recording EVERY stage in `exclusions`.

    Matches Geneva's definition: exact-duplicate rows dropped,
    ischemic stroke only, one index admission per hospital stay. Use
    preprocessing.build_log.record_exclusion(exclusions, stage, reason, before, after) per
    stage — the build log and the smoke report both read that chain, and it is how the two
    sites' cohorts are compared.

    NEVER drop a row for a missing outcome: the label is chosen at training time.
    """
    raise NotImplementedError("TODO(shenzhen): cohort filter + exclusion chain")



def derive_timings(df: pd.DataFrame) -> pd.DataFrame:
    """The four timing intervals in MINUTES, as columns named exactly TIMING_COLS.

    ODT onset→door, ONT onset→needle, DNT door→needle, OPT onset→puncture. Unknown onset (and
    any interval it feeds) is NaN, never 0. 
    Negative or implausible values may be left as they are: nullify_out_of_range catches them against FROZEN_RANGES and counts them.
    """
    raise NotImplementedError("TODO(shenzhen): ODT / ONT / DNT / OPT derivation")


def derive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """The three outcome columns, under the raw names added to SHENZHEN_TO_FROZEN (Q1).

    mrs_3m (0-6), death_3m, death_in_hospital. 
    Not recorded stays NaN — no row is dropped.
    """
    raise NotImplementedError("TODO(shenzhen): outcome columns + reconciliation")


# ---------------------------------------------------------------- pre-wired (do not edit)

def hash_inputs(export_paths: list[Path]) -> dict[str, str]:
    """{label: sha256} of every input file the build read — stamped into the parquet for audit."""
    return {f"export/{Path(p).name}": sha256_of(Path(p)) for p in export_paths}


def assemble_wide(cohort: pd.DataFrame) -> pd.DataFrame:
    """Cohort + the derived columns → the wide raw frame wide_to_frozen consumes."""
    wide = cohort.copy()
  
    timings = derive_timings(cohort)
    _check_derived(timings, TIMING_COLS, "derive_timings")
    outcomes = derive_outcomes(cohort)
    _check_derived(outcomes, tuple(_raw_names_of(FROZEN_OUTCOMES)), "derive_outcomes")

    return wide.join(timings).join(outcomes)


def _raw_names_of(frozen_names: list[str]) -> list[str]:
    """The raw column names SHENZHEN_TO_FROZEN maps onto these frozen names."""
    inverse = {frozen: raw for raw, frozen in SHENZHEN_TO_FROZEN.items()}
    missing = [f for f in frozen_names if f not in inverse]
    if missing:
        raise ValueError(f"SHENZHEN_TO_FROZEN has no source column for {missing}")
    return [inverse[f] for f in frozen_names]


def _check_derived(df: pd.DataFrame, expected: tuple[str, ...], who: str) -> None:
    """A derivation must produce exactly the raw names the mapping expects — fail here rather
    than let select_mapped_columns report it as a missing frozen column three steps later."""
    if tuple(df.columns) == expected:
        return
    raise ValueError(f"{who}: expected columns {list(expected)}, got {list(df.columns)}")


def wide_to_frozen(wide_df: pd.DataFrame) -> pd.DataFrame:
    """The wide raw frame → the frozen-schema table."""
    # 1. per-row unit check into FROZEN_UNITS (raw names; binaries skipped)
    df, unit_check = convert_to_frozen_units(
        wide_df, SHENZHEN_TO_FROZEN, FROZEN_UNITS, SHENZHEN_DECLARED_UNITS, UNIT_ALIASES
    )
    print(format_unit_check(unit_check))

    # 2. binaries incl. the binary outcomes → 3. rename → 4. select → 5. validate → 6. finalize
    df = encode_binaries(df, SHENZHEN_BINARY_ENCODINGS, fillna=SHENZHEN_BINARY_FILLNA)
    df = rename_to_frozen(df, SHENZHEN_TO_FROZEN)
    df = select_mapped_columns(df, SHENZHEN_TO_FROZEN, id_col=RAW_ID_COL)
    validate_frozen_columns(df.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID_COL)
    df = finalize_frozen_table(df, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID_COL,
                               outcome_ranges=FROZEN_OUTCOME_RANGES)

    # 7. public plausibility ranges: outside -> NaN, counted; fails on a wholesale miss
    df, out_of_range = nullify_out_of_range(df, FROZEN_RANGES, FROZEN_FEATURES)
    print(format_out_of_range(out_of_range))

    # 8. outcome availability — informational, nothing is dropped
    outcomes: dict[str, dict] = {}
    for col in FROZEN_OUTCOMES:
        s = df[col]
        entry = {"n_recorded": int(s.notna().sum()), "n_missing": int(s.isna().sum())}
        if is_binary_unit(FROZEN_UNITS[col]):
            entry["n_positive"] = int(s.sum()) if entry["n_recorded"] else 0
        else:
            entry["median"] = float(s.median()) if entry["n_recorded"] else None
        outcomes[col] = entry

    # 9. aggregate-only side data for the build log and write_node_parquet's smoke report
    df.attrs["unit_check"] = unit_check
    df.attrs["out_of_range"] = out_of_range
    df.attrs["outcomes"] = outcomes
    return df


def assemble_build_log(df: pd.DataFrame, export_path: Path) -> str:
    """Render the build log (aggregate-only: counts and percentages, no ids, no values)."""
    pids = patient_ids(df)
    header = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "provenance": PROVENANCE,
        "anonymisation": df.attrs.get("anonymisation", {}).get("spec", "none"),
        "export": str(export_path),
        "output": (f"{len(df)} rows x {df.shape[1]} cols "
                   f"({pids.nunique() if pids is not None else 'n/a'} patients; "
                   f"{len(FROZEN_FEATURES)} features + {len(FROZEN_OUTCOMES)} outcomes)"),
        "label": ("not part of this table — chosen in fed_stroke.schema (TARGET_COL / "
                  "TARGET_RULE); unlabelled rows are dropped at load time"),
    }
    return format_build_log("Shenzhen frozen-table build log", header,
                            df.attrs.get("exclusions", []),
                            df.attrs.get("out_of_range"), df.attrs.get("unit_check"),
                            outcomes=df.attrs.get("outcomes"))


def build_frozen_shenzhen_table(export_path: Path, pseudonym_key: bytes,
                                log_path: Path | None = None) -> pd.DataFrame:
    """Site export → ONE tidy, DE-IDENTIFIED per-admission table in the frozen schema."""
    exclusions: list[dict] = []
    raw = load_shenzhen_export(export_path)
    cohort = build_shenzhen_cohort(raw, exclusions=exclusions)
    print(f"[cohort] raw={len(raw)} cohort={len(cohort)}")

    frozen = wide_to_frozen(assemble_wide(cohort))
    frozen.attrs["exclusions"] = exclusions

    # de-identification is the LAST step: write_node_parquet refuses an unstamped frame
    frozen = anonymise_frozen(frozen, pseudonym_key)

    path = write_build_log(log_path if log_path is not None else DEFAULT_BUILD_LOG,
                           assemble_build_log(frozen, export_path))
    frozen.attrs["build_log_path"] = str(path)
    print(f"[log] build log written to {path}")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="the Shenzhen site export")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "out" / "shenzhen_frozen.parquet",
                        help="the Shenzhen node's parquet (node_config data-path)")
    parser.add_argument("--log", type=Path, default=None,
                        help="build log; default: next to --out as <stem>.build.log")
    parser.add_argument("--pseudonym-key", type=Path, required=True,
                        help="a file of >= 32 random bytes kept OUTSIDE the repo and away from "
                             "--out; never printed or stamped (only its fingerprint is)")
    parser.add_argument("--skip-hash", action="store_true",
                        help="do not sha256 the inputs into the parquet stamp (iteration only)")
    args = parser.parse_args()

    # the full mapping to-do list up front, rather than one failing column per run
    validate_shenzhen_encodings()

    log_path = args.log if args.log is not None else args.out.with_name(args.out.stem + ".build.log")
    pseudonym_key = read_pseudonym_key(args.pseudonym_key, args.out, REPO_ROOT)
    df = build_frozen_shenzhen_table(args.export, pseudonym_key, log_path=log_path)
    source_files = {} if args.skip_hash else hash_inputs([args.export])

    summary = write_node_parquet(df, args.out, source_files=source_files)
    print(f"[node] {args.out}: {summary['n_rows']} rows, {summary['n_patients']} patients, "
          f"schema {summary['schema_version']}, {len(source_files)} input hashes stamped; "
          f"key fingerprint {summary['anonymisation']['key_fingerprint']}; "
          f"unit_check pass={summary['unit_check']['pass']}, "
          f"out_of_range pass={summary['out_of_range']['pass']}")
    print(f"[node] smoke report (the only artifact that leaves the site): "
          f"{summary['smoke_report_path']}")


if __name__ == "__main__":
    main()
