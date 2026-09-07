"""GVA preprocessing: registry + EHR → the frozen-schema table the GVA node trains on.


Contract with the architecture layer:
- one row per case_admission_id — EVERY cohort admission, whatever its outcome; columns EXACTLY
  ['case_admission_id', *FROZEN_FEATURES, *FROZEN_OUTCOMES] (preprocessing.mappings.
  frozen_schema, the verbatim mirror of frozen_feature_names.xlsx plus the three standardized
  outcome columns) — no PII, no extras;
- the training LABEL is NOT decided here: fed_stroke.schema picks TARGET_COL (and an optional
  cut-off rule) among the outcome columns and the loader drops the unlabelled rows at load time;
- values IN the frozen units of record (FROZEN_UNITS): per-row unit labels are checked and
  factored via preprocessing.mappings.unit_aliases; binaries are encoded per
  preprocessing.mappings.gva_encodings (sex: 1 = female);
- missing values stay NaN (the loader applies MISSING_SENTINEL); values outside the public
  plausibility ranges (preprocessing.mappings.frozen_schema.FROZEN_RANGES == the DP bin grid)
  are set to NaN and counted per feature (df.attrs['out_of_range']); a feature out of range
  wholesale fails the build (unit / mapping bug);
- outcomes stay NaN when not recorded; their availability is counted (df.attrs['outcomes']);
  the aggregate unit-check report is df.attrs['unit_check'] (feeds the smoke report);
- a build log is written next to the parquet (<stem>.build.log; aggregate-only): every cohort
  exclusion stage with its reason, rows and distinct patients removed, outcome availability,
  the nulled values per feature, the unit check — the same data is on df.attrs;
- NO one-row-per-patient dedup, NO train/valid split, NO sentinel encoding, NO label drop —
  R3/R4, the sentinel and the label are loader-side (task.py) and accountant-reviewed there.

fed_stroke.schema mirrors the preprocessing-side frozen list (FEATURE_COLS == FROZEN_FEATURES,
same order). If the two ever drift this script prints a loud stderr warning: the loader could
not consume the parquet.

Usage:
    python architecture/preprocessing/preprocess_gva.py \
        --registry /.../stroke_registry_post_hoc_modified.xlsx \
        --ehr-dir  /.../Extraction_YYYYMMDD \
        --out      out/gva_frozen.parquet          # + out/gva_frozen.build.log
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve().parent
ARCH_DIR = _HERE.parent
REPO_ROOT = ARCH_DIR.parent
# fed_stroke (frozen schema) + repo root (the shared `preprocessing` library).
sys.path.insert(0, str(ARCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

import fed_stroke.schema as arch_schema  # noqa: E402  — SCHEMA_VERSION + the drift warning
from preprocessing.build_log import (  # noqa: E402
    format_build_log,
    patient_ids,
    write_build_log,
)
from preprocessing.case_ids import (  # noqa: E402
    create_ehr_case_identification_column,
    create_registry_case_identification_column,
)
from preprocessing.first_values import (  # noqa: E402
    LAB_FILE_PREFIX,
    PV_FILE_PREFIX,
    assemble_wide,
    extract_lab_dosage_first_values,
    extract_pv_lab_first_values,
    extract_pv_vital_first_values,
    load_concat_csvs,
)
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
    GVA_BINARY_ENCODINGS,
    GVA_BINARY_FILLNA,
    GVA_DECLARED_UNITS,
    GVA_TO_FROZEN,
    ID_COL,
    UNIT_ALIASES,
    is_binary_unit,
    validate_frozen_columns,
)
from preprocessing.registry_cohort import (  # noqa: E402
    build_cohort,
    parse_yyyymmdd,
    preprocess_features,
    preprocess_outcome,
)


# ONE home for the schema version: fed_stroke.schema (the parquet stamp must name the contract
# the loader was built against). Bump there on any FROZEN_FEATURES/FROZEN_OUTCOMES/FROZEN_UNITS change.
SCHEMA_VERSION = arch_schema.SCHEMA_VERSION
PROVENANCE = "real-frozen-schema"


def warn_if_architecture_schema_not_frozen() -> bool:
    """Compare fed_stroke.schema to the preprocessing-side frozen list; banner to stderr on mismatch.

    Not an exception: this script must keep working while an architecture-side change is in
    flight. Returns True when the two agree (the normal state since the 2026-09 freeze) — the
    loader can consume the parquet. A False here means fed_stroke.schema and
    preprocessing.mappings.frozen_schema were edited out of step.
    """
    same = (list(arch_schema.FEATURE_COLS) == list(FROZEN_FEATURES)
            and list(arch_schema.OUTCOME_COLS) == list(FROZEN_OUTCOMES))
    if not same:
        rule = "=" * 78
        print(
            f"{rule}\n"
            "WARNING: architecture schema not yet frozen; loader cannot consume this parquet "
            "until fed_stroke.schema is expanded.\n"
            f"  fed_stroke.schema.FEATURE_COLS (n={len(arch_schema.FEATURE_COLS)}): "
            f"{list(arch_schema.FEATURE_COLS)}\n"
            f"  preprocessing FROZEN_FEATURES  (n={len(FROZEN_FEATURES)}): "
            f"first={FROZEN_FEATURES[:3]} ... last={FROZEN_FEATURES[-3:]}\n"
            f"  fed_stroke.schema.OUTCOME_COLS: {list(arch_schema.OUTCOME_COLS)} vs "
            f"preprocessing FROZEN_OUTCOMES: {FROZEN_OUTCOMES}\n"
            f"{rule}",
            file=sys.stderr, flush=True,
        )
    return same


def wide_to_frozen(wide_df: pd.DataFrame) -> pd.DataFrame:
    """assemble_wide output (raw registry + EHR columns) → the frozen-schema table.

    Pure (no I/O) — callable from the exploration notebook on a cached wide_df. Binds the
    Geneva mapping tables (preprocessing.mappings) to the site-agnostic steps in
    preprocessing.frozen_table, in the contract order:
        convert_to_frozen_units → encode_binaries → rename_to_frozen → select_mapped_columns
        → validate_frozen_columns → finalize_frozen_table → nullify_out_of_range.
    Every step fails loudly; nothing is silently NaN'd, and NO ROW IS DROPPED: the table keeps
    every cohort admission with all FROZEN_OUTCOMES (NaN when not recorded). The training
    label — which outcome, which cut-off, and therefore which rows are trainable — is
    fed_stroke.schema's decision, applied by the loader (task.select_labelled_rows).

    Output: columns EXACTLY [ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES]; features float64 in
    FROZEN_UNITS (NaN = missing or implausible), outcomes float64 (NaN = not recorded), ids
    unique. Aggregate-only side data on df.attrs: 'unit_check' (the smoke report's unit-check
    pass/fail), 'out_of_range' (per-feature below/above counts), 'outcomes' (per-outcome
    recorded/missing counts), 'architecture_schema_frozen'.
    """
    # 1. per-row unit check into FROZEN_UNITS (raw names; binaries skipped, mrs_3m unitless)
    df, unit_check = convert_to_frozen_units(
        wide_df, GVA_TO_FROZEN, FROZEN_UNITS, GVA_DECLARED_UNITS, UNIT_ALIASES
    )
    print(format_unit_check(unit_check))

    # 2. binaries incl. the binary outcomes (raw names) → 3. rename → 4. select → 5. validate
    #    → 6. finalize (outcome value ranges enforced)
    df = encode_binaries(df, GVA_BINARY_ENCODINGS, fillna=GVA_BINARY_FILLNA)
    df = rename_to_frozen(df, GVA_TO_FROZEN)
    df = select_mapped_columns(df, GVA_TO_FROZEN, id_col=ID_COL)
    validate_frozen_columns(df.columns, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=ID_COL)
    df = finalize_frozen_table(df, FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=ID_COL,
                               outcome_ranges=FROZEN_OUTCOME_RANGES)

    # 7. public plausibility ranges (frozen names, frozen units): outside -> NaN, counted;
    #    fails if any feature is out of range wholesale (unit / mapping bug)
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
    print("[outcomes] " + " | ".join(
        f"{c}: recorded {o['n_recorded']}/{len(df)}"
        + (f", 1 = {o['n_positive']}" if "n_positive" in o else
           (f", median {o['median']:g}" if o.get("median") is not None else ""))
        for c, o in outcomes.items()))
    print(f"[schema] OK: {len(df)} rows x {df.shape[1]} cols "
          f"(= {ID_COL} + {len(FROZEN_FEATURES)} features + {len(FROZEN_OUTCOMES)} outcomes); "
          f"ids unique; no row dropped")

    # 9. aggregate-only side data for the build log and write_node_parquet's smoke report
    df.attrs["unit_check"] = unit_check
    df.attrs["out_of_range"] = out_of_range
    df.attrs["outcomes"] = outcomes
    df.attrs["architecture_schema_frozen"] = warn_if_architecture_schema_not_frozen()
    return df


DEFAULT_BUILD_LOG = REPO_ROOT / "out" / "gva_frozen.build.log"


def assemble_build_log(df: pd.DataFrame, registry_xlsx: Path, ehr_dir: Path,
                       ehr_rows: dict[str, int] | None = None) -> str:
    """Render the build log for a wide_to_frozen output whose attrs carry the cohort exclusion
    chain ('exclusions'), the outcome availability ('outcomes'), the plausibility report
    ('out_of_range') and the unit check. Aggregate-only: counts and percentages, no ids,
    no values."""
    pids = patient_ids(df)
    header = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "provenance": PROVENANCE,
        "registry": str(registry_xlsx),
        "ehr_dir": str(ehr_dir),
        "ehr_rows_loaded": ", ".join(f"{k}={v}" for k, v in (ehr_rows or {}).items()) or "n/a",
        "output": (f"{len(df)} rows x {df.shape[1]} cols "
                   f"({pids.nunique() if pids is not None else 'n/a'} patients; "
                   f"{len(FROZEN_FEATURES)} features + {len(FROZEN_OUTCOMES)} outcomes)"),
        "label": ("not part of this table — chosen in fed_stroke.schema (TARGET_COL / "
                  "TARGET_RULE); unlabelled rows are dropped at load time"),
    }
    return format_build_log("GVA frozen-table build log", header,
                            df.attrs.get("exclusions", []),
                            df.attrs.get("out_of_range"), df.attrs.get("unit_check"),
                            outcomes=df.attrs.get("outcomes"))


def build_frozen_gva_table(registry_xlsx: Path, ehr_dir: Path,
                           log_path: Path | None = None) -> pd.DataFrame:
    """Registry + EHR → ONE tidy per-admission table in the frozen schema.

    Also writes the build log (`log_path`, default DEFAULT_BUILD_LOG; main() puts it next to
    the parquet as <stem>.build.log): every cohort exclusion stage with its reason and the rows
    and distinct patients it removed, the outcome availability, the nulled values per feature,
    the unit check. Aggregate-only. The same data stays on df.attrs ('exclusions', 'outcomes',
    'out_of_range', 'unit_check', 'build_log_path') for write_node_parquet's smoke report.

    Steps to populate (reuse, don't re-derive):
    - cohort: preprocessing.registry_cohort (exact-duplicate rows dropped,
      'Type of event' == 'Ischemic stroke'); each stage recorded as an exclusion;
    - outcomes: the OPSUM 3M Death / 3M mRS / in-hospital death reconciliation already
      coded there, carried as the three FROZEN_OUTCOMES columns (never used to drop rows);
    - case_admission_id: preprocessing.case_ids
      (== the loader's patient_id + '_' + EDS-last-4 derivation, task.py);
    - EHR features: preprocessing.first_values extraction, joined on
      case_admission_id;
    - wide_to_frozen: target encode/drop, per-row unit check INTO FROZEN_UNITS
      (preprocessing.mappings.unit_aliases + GVA_DECLARED_UNITS), binary
      encodings (preprocessing.mappings.gva_encodings), GVA_TO_FROZEN rename,
      preprocessing.mappings.validate_frozen_columns, dtype finalisation.

    Input:
        registry_xlsx: the Geneva stroke-registry export (.xlsx).
        ehr_dir: the EHR extraction directory (patientvalue + lab CSVs).
    Output:
        DataFrame, one row per case_admission_id (every cohort admission), columns EXACTLY
        ['case_admission_id', *FROZEN_FEATURES, *FROZEN_OUTCOMES]:
        - case_admission_id: str, '<patient_id>_<eds_last_4>';
        - features: float64, IN FROZEN_UNITS, missing = NaN (no sentinel).
          Values outside the public plausibility ranges (FROZEN_RANGES ==
          the DP bin grid; e.g. negative ODT, DBP 889) are set to NaN and
          counted in df.attrs['out_of_range'] — a data-quality step, not a
          privacy question. Zeros inside a range that starts at 0 (age 0,
          glucose 0) are NOT caught: that is the lower bounds' job;
        - outcomes: float64, NaN when not recorded (mrs_3m 0-6; death_3m and
          death_in_hospital {0, 1}); NO row is dropped for a missing outcome —
          fed_stroke.schema chooses the label and the loader drops the unlabelled
          rows (counted there).
    """
    # preprocess registry
    exclusions: list[dict] = []
    df = pd.read_excel(registry_xlsx)
    df, n_raw, n_filtered = build_cohort(df, exclusions=exclusions)
    print(f"[cohort] raw={n_raw} ischemic-stroke cohort={n_filtered}")
    # derive case_admission_id
    df['case_admission_id'] = create_registry_case_identification_column(df)
    df = preprocess_outcome(df)
    df = preprocess_features(df)

    df["admission_date"] = parse_yyyymmdd(df["Arrival at hospital"])

    # preprocess EHR data
    print(f"[df] n_patients={len(df)}")
    print(f"[load]   PV files ({PV_FILE_PREFIX}*.csv) from {ehr_dir}")
    pv_df = load_concat_csvs(ehr_dir, PV_FILE_PREFIX)
    pv_df["case_admission_id"] = create_ehr_case_identification_column(pv_df)
    print(f"[load]   PV rows={len(pv_df)}")

    print(f"[load]   lab files ({LAB_FILE_PREFIX}*.csv) from {ehr_dir}")
    lab_df = load_concat_csvs(ehr_dir, LAB_FILE_PREFIX)
    lab_df["case_admission_id"] = create_ehr_case_identification_column(lab_df)
    print(f"[load]   lab rows={len(lab_df)}")

    per_var: dict[str, pd.DataFrame] = {}
    per_var.update(extract_pv_vital_first_values(pv_df, df))
    per_var.update(extract_pv_lab_first_values(pv_df, df))
    per_var.update(extract_lab_dosage_first_values(lab_df, df))

    wide_df = assemble_wide(df, per_var)
    frozen = wide_to_frozen(wide_df)

    # build log: cohort exclusion stages, outcome availability, nulled values, unit check
    frozen.attrs["exclusions"] = exclusions
    text = assemble_build_log(frozen, registry_xlsx, ehr_dir,
                              ehr_rows={PV_FILE_PREFIX: len(pv_df), LAB_FILE_PREFIX: len(lab_df)})
    path = write_build_log(log_path if log_path is not None else DEFAULT_BUILD_LOG, text)
    frozen.attrs["build_log_path"] = str(path)
    print(f"[log] build log written to {path}")
    return frozen
# TODO implement anonymisation of GVA data


def write_node_parquet(df: pd.DataFrame, out_path: Path,
                       source_files: dict[str, str]) -> dict:
    """Write ONE site's node parquet + provenance metadata; return the smoke summary.

    This is the artifact a SuperNode's node_config data-path points at — for the
    GVA node in the v1.3 cross-site topology, and (via split_gva_halves.py) for
    each loopback half in the Geneva-only phase.

    Input:
        df: a frozen-schema table (build_frozen_gva_table output or a half of it).
        out_path: parquet destination (must match the node's data-path).
        source_files: {filename: sha256} of the inputs, stamped for audit.
    Output:
        dict, the roadmap smoke-test artifact — AGGREGATE-ONLY (the one report
        designed to cross sites for the Geneva/Shenzhen comparison): row count,
        label balance, per feature median / min / max / missing rate + unit-check
        pass/fail (df.attrs['unit_check'], from wide_to_frozen; plus
        df.attrs['out_of_range'] per-feature below/above counts and
        df.attrs['outcomes'] per-outcome availability). Also written next to the
        parquet as <stem>_smoke_report.json.

    Side effects:
        - parquet key-value metadata stamped: SCHEMA_VERSION, PROVENANCE,
          source hashes, creation date — the parquet-side provenance marker (B F8);
        - after the node file exists, flip that node's data-provenance to
          'real-frozen-schema' in pyproject [tool.fed_stroke.nodes] (manual,
          deliberate — it arms the ledger).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True,
                        help="Geneva stroke-registry .xlsx")
    parser.add_argument("--ehr-dir", type=Path, required=True,
                        help="EHR extraction dir (patientvalue + lab CSVs)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "gva_frozen.parquet",
                        help="the GVA node's parquet (node_config data-path)")
    parser.add_argument("--log", type=Path, default=None,
                        help="build log (aggregate-only: exclusions with reasons, nulled values); "
                             "default: next to --out as <stem>.build.log")
    args = parser.parse_args()

    log_path = args.log if args.log is not None else args.out.with_name(args.out.stem + ".build.log")
    df = build_frozen_gva_table(args.registry, args.ehr_dir, log_path=log_path)
    summary = write_node_parquet(df, args.out,
                                 source_files={})  # TODO: sha256 of the two inputs
    print(summary)


if __name__ == "__main__":
    main()
