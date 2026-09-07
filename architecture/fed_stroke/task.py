"""fed_stroke: shared task utilities (data loading, config helpers)."""
import hashlib
import hmac
from pathlib import Path

import pandas as pd
import xgboost as xgb
from flwr.app import Context

from fed_stroke.schema import (
    FEATURE_COLS,
    MISSING_SENTINEL,
    OUTCOME_COLS,
    TARGET_COL,
    TARGET_RULE,
    label_id,
)

# The patient-disjoint hold-out partition is pinned to a FIXED seed, deliberately
# NOT a config key: it must never vary, or the HELD patient set would shift between
# the HPO search runs and the winner's hold-out report, silently leaking selection
# patients into the "hold-out" (spec 1.1.a §4.5/§4.6, Decision 11).
HOLDOUT_PARTITION_SEED = 42

# Public per-project constant for the keyed-hash split rule (spec 1.1.a″ R4). Public by
# design: adjacency stability comes from each patient's assignment depending ONLY on their
# own patient_id (plus the run's public seed/role), never from any cohort-derived quantity.
SPLIT_KEY = b"fed-stroke-split-v1"
_HASH_MODULUS = 2 ** 32


def _hash_split_assign(pid: str, role: str, seed: int) -> int:
    """assign(pid) = HMAC(key, pid) mod 2^32, key = SPLIT_KEY ‖ role ‖ seed (R4).

    `role` ∈ {"partition", "subsplit"} domain-separates the two split stages: the hash rule is
    input-set INDEPENDENT (unlike the old stratified train_test_split), so without the role tag
    a run with split_seed == HOLDOUT_PARTITION_SEED would give the DEV/HELD partition and the
    DEV sub-split the identical key and collapse the sub-split's validation band. `test_size`
    enters only the caller's threshold, never the key, so FLAT and partition fractions cannot
    alias. `|`-separated so (role, seed) pairs cannot collide under concatenation."""
    key = b"|".join([SPLIT_KEY, role.encode(), str(int(seed)).encode()])
    digest = hmac.new(key, str(pid).encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big")


def encode_missing_as_sentinel(data):
    """REMOVE-IF-NO-DP — missingness policy variant 1 (sentinel / missing bin, 2026-07-23).

    Map NaN in the feature columns to the PUBLIC schema.MISSING_SENTINEL, so "not recorded"
    becomes an explicit ordinary value every arm sees identically: the DP learner bins it into
    the reserved bin 0 (fixed_bin_edges), and stock XGBoost sees a distinguishable low value it
    can split on (DMatrix's missing marker stays NaN, so the sentinel is NEVER treated as
    missing by XGBoost). Applied at the generate_splits chokepoint — NOT the DP branch — because
    a DP-only encoding would confound the A→B→C comparison (spec 1.1.a″ R9) with a missingness
    difference. Idempotent (the sentinel is not NaN). Registry reality this handles: the
    example halves carried ~4.5% missing NIHSS; before this, the DP learner silently binned
    NaN into the TOP bin (max-NIHSS artifact). On the real frozen table several labs are
    mostly missing (GCS entirely), so every one of the 41 columns goes through here.

    Fails loudly if a frozen column is absent: a table missing a FEATURE_COLS entry is a
    schema-contract violation (preprocess_gva.validate_frozen_columns guarantees the full
    set), and silently skipping it would only surface later as a KeyError on the subset.

    Exists ONLY because the DP learner's fixed public bins cannot represent NaN and the R7 gate
    refuses non-finite features. If the project later moves forward WITHOUT DP: remove this
    (grep REMOVE-IF-NO-DP) and let XGBoost's native NaN handling take over.
    """
    missing = [col for col in FEATURE_COLS if col not in data.columns]
    if missing:
        raise ValueError(
            f"frozen-schema columns absent from the loaded table ({len(missing)} of "
            f"{len(FEATURE_COLS)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}"
        )
    for col in FEATURE_COLS:
        data[col] = data[col].fillna(MISSING_SENTINEL)
    return data


def derive_label(outcome: pd.Series, rule=None) -> pd.Series:
    """Frozen outcome column -> training label as float64 in {0, 1, NaN}.

    `rule` None: the column must already be binary. Otherwise `rule(values)` (e.g.
    schema.mrs_at_most(2)) is applied to the RECORDED values only — NaN stays NaN so the row
    can be dropped, never silently labelled. Anything outside {0, 1} after the rule raises.
    """
    s = pd.to_numeric(outcome, errors="raise").astype("float64")
    if rule is not None:
        derived = rule(s)
        if not isinstance(derived, pd.Series):
            derived = pd.Series(list(derived), index=s.index)
        s = derived.astype("float64").where(s.notna())
    bad = sorted(set(s.dropna().unique()) - {0.0, 1.0})
    if bad:
        raise ValueError(
            f"label derived from {outcome.name!r} must be in {{0, 1}}; found {bad} — set "
            f"schema.TARGET_RULE to binarize an ordinal outcome (e.g. mrs_at_most(2))."
        )
    return s


def select_labelled_rows(data, outcome=TARGET_COL):
    """Derive the label of `outcome` and DROP the rows that have none. Returns (frame, n_dropped).

    The frozen table keeps every cohort admission with all OUTCOME_COLS (NaN when not
    recorded); which rows are trainable depends on the label chosen in fed_stroke.schema, so the
    drop happens HERE, once, at load (resolve_run_split) — never in a site's preprocessing. A
    per-record rule on public constants: adjacency-safe for the DP arm (a record without a
    label contributes to no histogram in any neighbouring dataset either). The label column is
    cast to int64 so the R7 gate sees exactly {0, 1}. TARGET_RULE applies only when `outcome`
    IS the schema's TARGET_COL; any other outcome column must already be binary.
    """
    if outcome not in data.columns:
        carried = [c for c in OUTCOME_COLS if c in data.columns]
        raise ValueError(
            f"label column {outcome!r} absent from the loaded table (frozen outcome columns "
            f"present: {carried}); fed_stroke.schema.TARGET_COL must name a column the site "
            f"preprocessing delivers."
        )
    rule = TARGET_RULE if outcome == TARGET_COL else None
    data = data.copy()
    data[outcome] = derive_label(data[outcome], rule)
    keep = data[outcome].notna()
    out = data.loc[keep].copy()
    out[outcome] = out[outcome].astype("int64")
    return out, int((~keep).sum())


def dedup_one_row_per_patient(data, outcome):
    """R3: reduce to ONE admission row per patient — the DP adjacency unit is the PATIENT.

    Rule (user decision, 2026-07-22): keep an admission whose label equals the patient's max
    outcome (matches the historical stratification reduction); among ties, keep the
    lexicographically smallest case_admission_id. Exact-duplicate case_admission_id rows (the
    registry contains e.g. one admission recorded twice, same id) tie on every key — the
    stable sort + head(1) then keeps the first such row deterministically, so the result is
    always EXACTLY one row per patient. Idempotent (deduping deduped data is a no-op), so
    resolve_run_split's repeated generate_splits calls in SEARCH mode are safe. Applied to
    the WHOLE pipeline (DP and non-DP arms alike), or the 1.1.b DP-vs-no-DP comparison would
    confound noise cost with a cohort change."""
    max_out = data.groupby('patient_id')[outcome].transform('max')
    candidates = data[data[outcome] == max_out]
    return (candidates.sort_values('case_admission_id', kind='stable')
                      .groupby('patient_id', sort=False)
                      .head(1)
                      .sort_index())


def generate_splits(data, outcome, test_size, seed,
                    test_pids_path=None, train_pids_path=None,
                    role="subsplit", dp_mode=False):
    """Patient-level train/test split — the single chokepoint every load path converges on
    (`_resolve_context_split`, `baseline.split_half`, `eval_final_model` all route here).

    Two 1.1.a″ invariants live here, for the DP AND non-DP arms alike:

    - R3 dedup: one admission row per patient (max-outcome rule + tie-break), applied
      immediately after patient_id derivation, BEFORE any split. Worst-case histogram
      sensitivity is per contributed row; without dedup a patient with m admissions
      contributes m rows and the accountant's ΔG = 1 is per-admission, not per-patient.
    - R4 adjacency-stable split: assignment by keyed hash of patient_id (see
      _hash_split_assign), replacing the stratified train_test_split. Adding/removing one
      patient never moves another patient's assignment, so neighbouring raw datasets yield
      training tables differing by exactly one row (the sensitivity argument attaches to the
      actual pipeline). Cost accepted: no outcome stratification — exact stratification is
      cohort-dependent and precisely what breaks adjacency stability.

    Args:
        data: input frame with case_admission_id, features, outcome.
        outcome: label column (drives the dedup rule; no longer used for stratification).
        test_size (float): validation fraction — a patient is in validation iff
            assign(pid) < 2^32 · test_size.
        seed (int): public per-stage seed, folded into the HMAC key (distinct seeds keep
            yielding distinct splits for HPO CV repeats).
        test_pids_path / train_pids_path: optional predefined frozen patient-id lists
            (historical splits). REFUSED in dp_mode — a list derived from the private cohort
            has no rule for the adjacent dataset's extra patient.
        role: "subsplit" (default) or "partition" (the fixed DEV/HELD stage) — domain
            separation of the two split stages, see _hash_split_assign.
        dp_mode: fail-closed guard — True refuses any split rule other than the hash rule.

    Returns:
        (train_set, test_set, num_train, num_test); frames keep the full column set
        (including the derived patient_id).
    """

    data['patient_id'] = data['case_admission_id'].apply(lambda x: x.split('_')[0])

    # REMOVE-IF-NO-DP: missing features -> public sentinel (variant 1), for ALL arms alike.
    data = encode_missing_as_sentinel(data)

    # R3: one row per patient, before ANY split and before the DP mechanism's input.
    data = dedup_one_row_per_patient(data, outcome)

    # Using predefined test and train patient ids
    if test_pids_path is not None:
        if dp_mode:
            raise ValueError(
                "DP mode refuses predefined patient-id lists (spec 1.1.a″ R4): a frozen list "
                "derived from the private cohort has no assignment rule for the adjacent "
                "dataset's extra patient. Use the keyed-hash split, or declare a frozen PUBLIC "
                "list with a documented adjacency statement (not implemented)."
            )
        pid_test = pd.read_csv(test_pids_path, dtype=str).patient_id.tolist()
        pid_train = pd.read_csv(train_pids_path, dtype=str).patient_id.tolist()
    else:
        # R4: keyed-hash assignment. Each patient's side depends only on (pid, role, seed) —
        # never on any other patient, any count, or any label.
        threshold = _HASH_MODULUS * float(test_size)
        all_pids = data['patient_id'].unique()
        in_test = {pid: _hash_split_assign(pid, role, seed) < threshold for pid in all_pids}
        pid_test = [pid for pid in all_pids if in_test[pid]]
        pid_train = [pid for pid in all_pids if not in_test[pid]]

    train_set = data[data.patient_id.isin(pid_train)]
    test_set = data[data.patient_id.isin(pid_test)]
    num_train = len(train_set)
    num_test = len(test_set)

    return train_set, test_set, num_train, num_test


def resolve_run_split(data, outcome, split_seed=42, holdout_frac=0.0,
                      holdout_eval=False, dp_mode=False):
    """The single train/valid split contract, shared by the federated client, the
    offline scorer, and the pooled baseline (spec 1.1.a §4.5).

    Three modes, selected by (holdout_frac, holdout_eval):

    - holdout_frac == 0.0 -> FLAT: a plain generate_splits(test_size=0.2,
      seed=split_seed, role="subsplit") train/valid split.

    - holdout_frac > 0.0, holdout_eval=False -> SEARCH repeat: first reserve a
      patient-disjoint HELD set by partitioning DEV/HELD with the FIXED
      HOLDOUT_PARTITION_SEED (never the search seed) under role="partition", then
      sub-split DEV into train/valid by split_seed under role="subsplit". HELD is
      never returned here — it is excluded from every search repeat by the fixed
      partition, not by a seed convention. The role tag domain-separates the two
      stages (see _hash_split_assign): HELD is a threshold set over the FULL cohort
      and the sub-split runs only on its complement DEV, so HELD stays
      patient-disjoint from every search sub-split by construction.

    - holdout_frac > 0.0, holdout_eval=True -> HOLD-OUT report: same fixed DEV/HELD
      partition; train = all of DEV, valid = HELD. Because HELD sat outside every
      search repeat, this is a genuine patient-disjoint hold-out.

    Since 1.1.a″ (R3/R4) every mode returns DEDUPED (one row per patient),
    hash-split frames — for the DP and non-DP arms alike. `dp_mode` threads the R4
    fail-closed guard down to generate_splits.

    Returns (train_df, valid_df) with the full column set (incl. patient_id), exactly
    like generate_splits — callers subset to FEATURE_COLS/TARGET_COL as needed.
    Patient-level partitioning is delegated to generate_splits, so the DEV/HELD
    boundary respects the multiple-admission (patient_id) guard.

    Label first, ONCE per load: the frozen table carries every cohort admission and all
    OUTCOME_COLS; select_labelled_rows derives `outcome`'s label (schema.TARGET_RULE) and
    drops the rows without one before any dedup or split. Called exactly once per load path
    (the federated loader and baseline.split_half both enter here once), so an ordinal rule
    is never applied twice.
    """
    data, _ = select_labelled_rows(data, outcome)

    if holdout_frac == 0.0:
        train_df, valid_df, _, _ = generate_splits(
            data, outcome=outcome, test_size=0.2, seed=split_seed,
            role="subsplit", dp_mode=dp_mode,
        )
        return train_df, valid_df

    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(
            f"holdout_frac must be in [0.0, 1.0); got {holdout_frac}"
        )

    # Reserve the patient-disjoint HELD set with the FIXED partition seed.
    dev_df, held_df, _, _ = generate_splits(
        data, outcome=outcome, test_size=holdout_frac, seed=HOLDOUT_PARTITION_SEED,
        role="partition", dp_mode=dp_mode,
    )

    if holdout_eval:
        # HOLD-OUT report: train on all of DEV, evaluate on the disjoint HELD set.
        return dev_df, held_df

    # SEARCH repeat: sub-split DEV into train/valid by the (varying) search seed.
    train_df, valid_df, _, _ = generate_splits(
        dev_df, outcome=outcome, test_size=0.2, seed=split_seed,
        role="subsplit", dp_mode=dp_mode,
    )
    return train_df, valid_df


def _resolve_context_split(context: Context):
    """Shared load + split contract for the SuperNode's configured data file.

    Reads `context.node_config["data-path"]` (no cross-SuperNode partitioning — the file is
    authoritative), then applies the one `resolve_run_split` contract driven by run_config.
    Returns `(train_df, valid_df, num_train, num_val, train_pids)` with the frames already
    subset to FEATURE_COLS + TARGET_COL, so both the DMatrix path (`load_data_gva`) and the
    raw-array DP path (`load_data_arrays`) train and evaluate on EXACTLY the same split at
    matched split-seed/holdout-frac/holdout-eval. `train_pids` (the training rows' patient
    ids, extracted BEFORE the column subset) feeds the DP precondition gate's one-row-per-
    patient check (R3/R7)."""
    data_path = Path(context.node_config["data-path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {data_path}")

    feature_cols = FEATURE_COLS
    target_col = TARGET_COL

    data_df = pd.read_parquet(data_path)

    n_rows = len(data_df)
    n_patients = data_df['case_admission_id'].str.split('_').str[0].nunique()
    # The parquet carries every cohort admission and all OUTCOME_COLS; the label is chosen in
    # fed_stroke.schema and the unlabelled rows are dropped ONCE inside resolve_run_split
    # (select_labelled_rows). Counted here up front so the loader log shows it.
    if target_col not in data_df.columns:
        raise ValueError(
            f"{data_path.name}: label column {target_col!r} absent; outcome columns present: "
            f"{[c for c in OUTCOME_COLS if c in data_df.columns]}"
        )
    n_unlabelled = int(data_df[target_col].isna().sum())
    recorded = data_df[target_col].dropna()
    label_balance = derive_label(recorded, TARGET_RULE).mean() if len(recorded) else float("nan")
    print(
        f"load_data_gva({data_path.name}): rows={n_rows}, unique_patients={n_patients}, "
        f"label={label_id()}, rows_without_label={n_unlabelled}, label_balance={label_balance:.4f}"
    )

    # Split contract driven by run_config (all default to today's flat seed-42
    # split, so an unconfigured run is byte-identical). HPO varies split-seed for
    # repeated CV and sets holdout-frac to reserve a patient-disjoint HELD set;
    # holdout-eval flips the winner's report run onto HELD (spec 1.1.a §4.5/§4.6).
    split_seed = context.run_config.get("split-seed", 42)
    holdout_frac = context.run_config.get("holdout-frac", 0.0)
    holdout_eval = context.run_config.get("holdout-eval", False)
    # run_config is the FLAT dotted-key dict here (unflatten_dict happens in client_app), so
    # dp.enabled is read under its flat key. Threads the R4 fail-closed guard.
    dp_mode = bool(context.run_config.get("dp.enabled", False))
    train_df, valid_df = resolve_run_split(
        data_df, outcome=target_col, split_seed=split_seed,
        holdout_frac=holdout_frac, holdout_eval=holdout_eval, dp_mode=dp_mode,
    )
    # R3 loader belt: post-dedup, EVERY returned partition is one row per patient. This can
    # only fire if generate_splits' dedup regresses — the learner-side belt is the DP
    # precondition gate (R7), fed by train_pids below.
    for part_name, part_df in (("train", train_df), ("valid", valid_df)):
        if part_df['patient_id'].nunique() != len(part_df):
            raise ValueError(
                f"{part_name} split has {len(part_df)} rows for "
                f"{part_df['patient_id'].nunique()} patients — one-row-per-patient dedup "
                f"(spec 1.1.a″ R3) was bypassed."
            )
    train_pids = train_df['patient_id'].to_numpy()
    num_train = len(train_df)
    num_val = len(valid_df)
    train_df = train_df[feature_cols + [target_col]]
    valid_df = valid_df[feature_cols + [target_col]]
    return train_df, valid_df, num_train, num_val, train_pids


def load_data_gva(context: Context):
    """Load GVA data from the parquet file at the SuperNode's configured path.

    Contract: read the file at `context.node_config["data-path"]`. No
    cross-SuperNode partitioning inside — the file is authoritative. A local
    train/valid holdout is still built here because `client_app.py`'s
    `@evaluate()` needs a `valid_dmatrix`.
    """
    feature_cols = FEATURE_COLS
    target_col = TARGET_COL
    train_df, valid_df, num_train, num_val, _ = _resolve_context_split(context)

    train_dmatrix = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col])
    valid_dmatrix = xgb.DMatrix(valid_df[feature_cols], label=valid_df[target_col])

    return train_dmatrix, valid_dmatrix, num_train, num_val


def load_data_arrays(context: Context):
    """Raw-numpy analog of load_data_gva for the DP learner (§4.5), which consumes `X, y` arrays
    (NOT a DMatrix). Returns `(X_train, y_train, X_valid, y_valid, num_train, num_val,
    train_pids)` via the SAME `_resolve_context_split` contract, so the DP arm trains/evaluates
    on exactly the split the XGB arm does at matched split-seed/holdout-frac/holdout-eval.
    X columns are exactly FEATURE_COLS order (== FEATURE_RANGES order, §3.9); y is TARGET_COL.
    `train_pids` are the training rows' patient ids, for the DP precondition gate's
    one-row-per-patient check (R3/R7)."""
    feature_cols = FEATURE_COLS
    target_col = TARGET_COL
    train_df, valid_df, num_train, num_val, train_pids = _resolve_context_split(context)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_valid = valid_df[feature_cols].to_numpy(dtype=float)
    y_valid = valid_df[target_col].to_numpy(dtype=float)

    return X_train, y_train, X_valid, y_valid, num_train, num_val, train_pids


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
