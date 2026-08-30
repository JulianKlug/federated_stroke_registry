"""fed_stroke.dpsweep: pure logic for the 1.1.b A→B→C DP sweep driver.

Everything `scripts/run_dp_sweep.py` needs that does NOT touch the process
boundary: the arm plan, shared-config resolution, run-config construction,
model read-backs and per-arm assertions, ledger-delta validation, the
B→C / A→B delta math, cohort stats, and the report/results assembly. All
unit-testable without a federation (mirrors the `hpo.py` / `run_hpo.py` split).

Spec: docs/specs/1_1_b_dp_sweep_driver.md.
"""
import json
import math
import tomllib
from pathlib import Path

import pandas as pd

from fed_stroke import hpo
from fed_stroke.baseline import LoadedModel, split_half
from fed_stroke.dp import DP_MODEL_FORMAT, DPBooster, per_site_tree_budget
from fed_stroke.schema import FEATURE_COLS
from fed_stroke.server_app import derive_num_rounds

ARM_A = "A"          # stock xgb.train           (dp.enabled=false)
ARM_B = "B"          # DP learner, identity      (no noise, no ε)
ARM_C = "C"          # DP learner, gaussian at ε

XGB_FORMAT = "xgboost-json"

# The four knobs a tuned TOML may override; subsample/colsample are NOT among them —
# they are forced to 1.0 for every arm (R9: the DP learner forces q=1.0 and ignores
# both, so arm A must match or A→B conflates sampling with learner cost).
TUNED_KEYS = ("total-trees", "params.max-depth", "params.eta",
              "params.min-child-weight")
FORCED_KEYS = ("params.subsample", "params.colsample-bytree")

DELTA_METRICS = ("auc_roc", "auc_pr", "brier")

SIGN_CONVENTION = ("negative ΔAUC-ROC/ΔAUC-PR = utility lost; "
                   "positive ΔBrier = calibration worsened")

UNLEDGERED_MARK = "UNLEDGERED — cyclic round-1 append gap (Risk §8)"

REHEARSAL_BANNER = ("**REHEARSAL — example halves. NOT a real-ε claim. "
                    "This artifact does not fill roadmap 1.1.b.**")


# --------------------------------------------------------------------------- #
# §4.1  arm plan
# --------------------------------------------------------------------------- #
def arm_plan(epsilons, delta) -> list[dict]:
    """The ordered run matrix. Each entry:
    {"arm": "A"|"B"|"C", "label": str, "epsilon": float|None, "dp_overrides": dict}

    Order: A, B, then C at DESCENDING ε (regardless of input order). A and B spend
    no ε, so every infrastructure failure mode surfaces before any privacy budget
    is spent; arm A doubles as the canary. The identity arm deliberately carries no
    `dp.target-epsilon` (the mechanism ignores it; its absence keeps the arm-B
    run-config string minimal).
    """
    plan = [
        {"arm": ARM_A, "label": "armA", "epsilon": None,
         "dp_overrides": {"dp.enabled": False}},
        {"arm": ARM_B, "label": "armB", "epsilon": None,
         "dp_overrides": {"dp.enabled": True, "dp.mechanism": "identity"}},
    ]
    for eps in sorted((float(e) for e in epsilons), reverse=True):
        plan.append({
            "arm": ARM_C, "label": f"armC_eps{eps:g}", "epsilon": eps,
            "dp_overrides": {"dp.enabled": True, "dp.mechanism": "gaussian",
                             "dp.target-epsilon": eps, "dp.delta": float(delta)},
        })
    return plan


# --------------------------------------------------------------------------- #
# §4.2  shared config + run-config construction
# --------------------------------------------------------------------------- #
def _get_dotted(table, dotted_key):
    """Read a dotted key (`params.max-depth`) out of a NESTED toml table (tomllib
    parses dotted keys into nested dicts). Returns None when absent."""
    cur = table
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def parse_tuned_table(text: str) -> dict:
    """Parse a tuned TOML (e.g. tuned_bagging.toml from the 1.1.a harness) and
    return its [tool.flwr.app.config] table (nested). Raises when the table is
    absent — a tuned file without it is the wrong file, not an empty override."""
    data = tomllib.loads(text)
    table = _get_dotted(data, "tool.flwr.app.config")
    if table is None:
        raise ValueError("tuned config carries no [tool.flwr.app.config] table")
    return table


def resolve_shared_config(app_cfg: dict, tuned_table: dict | None = None):
    """One shared hyperparameter set for all six runs -> (overrides, notes).

    Precedence: pyproject [tool.flwr.app.config] base -> optional tuned TOML for
    the four TUNED_KEYS -> params.subsample = params.colsample-bytree = 1.0 FORCED,
    always, for every arm (never optional, never a flag). A tuned subsample/
    colsample differing from 1.0 is read but discarded, with a note.
    """
    shared = {}
    notes = []
    for key in TUNED_KEYS:
        val = _get_dotted(app_cfg, key)
        if val is None:
            raise ValueError(f"pyproject [tool.flwr.app.config] is missing {key!r}")
        shared[key] = val
    if tuned_table is not None:
        for key in TUNED_KEYS:
            val = _get_dotted(tuned_table, key)
            if val is not None:
                shared[key] = val
        for key in FORCED_KEYS:
            val = _get_dotted(tuned_table, key)
            if val is not None and float(val) != 1.0:
                notes.append(f"tuned {key}={val} read but DISCARDED — forced to "
                             "1.0 for every arm (R9 matching)")
    for key in FORCED_KEYS:
        shared[key] = 1.0
    return shared, notes


def build_arm_run_config(arm, shared_overrides, strategy, split_seed, out_dir,
                         provenance) -> str:
    """The per-arm `--run-config` string, via hpo.build_run_config (inherits
    TOML-quoted scalars, sorted dotted keys, absolute model/metrics dirs,
    save-model=true, n-boot=0 — offline scoring owns the CIs).

    `data-provenance` rides along as the SUBMITTER's declaration only: the node's
    own `node_config` value is authoritative and the client refuses a run whose
    declaration disagrees (reviewer A C1 / B F8). The ledger path is node-owned
    too and is deliberately NOT threaded here (supersedes spec Decision 11).
    """
    arm_dir = Path(out_dir) / arm["label"]
    override = {
        **shared_overrides,
        **arm["dp_overrides"],
        "data-provenance": provenance,
    }
    return hpo.build_run_config(
        override, strategy, split_seed, arm_dir, arm_dir,
        holdout_frac=0.0, holdout_eval=False, save_model=True, n_boot=0)


def validate_divisibility(strategy, total_trees, num_sites, local_epochs) -> int:
    """total-trees divisibility, upfront — fail before run 1, not at run 3.
    Returns num_rounds (raises ValueError via derive_num_rounds otherwise)."""
    return derive_num_rounds(strategy, int(total_trees), int(num_sites),
                             int(local_epochs))


# --------------------------------------------------------------------------- #
# §4.6  accounting extraction (model side)
# --------------------------------------------------------------------------- #
def _finite_or_none(x):
    """None passes through (identity σ/ε arrive from disk as None — to_json_bytes
    scrubs inf on serialize); residual ±inf/NaN (in-memory mechanisms, hand-built
    fixtures) map to None too. Never math.isfinite(None) — that TypeErrors."""
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def extract_dp_meta(bst) -> dict:
    """{mechanism, sigma, epsilon, num_releases, per_site_trees} off a DPBooster's
    meta. `mechanism` is NESTED at meta["dp"]["mechanism"]; the accounting numbers
    are top-level (verified against dp/boost._mechanism_meta)."""
    meta = getattr(bst, "meta", None) or {}
    return {
        "mechanism": (meta.get("dp") or {}).get("mechanism"),
        "sigma": _finite_or_none(meta.get("noise_multiplier")),
        "epsilon": _finite_or_none(meta.get("reported_epsilon")),
        "num_releases": meta.get("num_releases"),
        "per_site_trees": meta.get("per_site_trees"),
    }


def read_back_fed_run_config(loaded: LoadedModel) -> dict:
    """The `fed_run_config` save_final_model stamped into the model itself:
    {"params": {...underscored keys...}, "total_trees": int}. XGB carries it
    JSON-encoded in set_attr; a DPBooster carries it in meta. Read back from the
    artifact, never from the intended config (checked invariant, not assumption)."""
    bst = loaded.booster
    if isinstance(bst, DPBooster):
        frc = (bst.meta or {}).get("fed_run_config")
        if not frc:
            raise ValueError("DP model carries no meta['fed_run_config']")
        return frc
    raw = bst.attr("fed_run_config")
    if raw is None:
        raise ValueError("XGB model carries no fed_run_config attr")
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# §4.4  per-arm post-run assertions (pure; the driver fails loud on non-empty)
# --------------------------------------------------------------------------- #
def check_arm_model(arm, loaded: LoadedModel, total_trees, per_site) -> list[str]:
    """The §4.4 assertion table for one arm's saved model. Returns a list of
    violation strings (empty = pass). `per_site` must come from
    dp.boost.per_site_tree_budget — NOT total_trees // num_sites, which is
    cyclic-wrong when num_rounds % num_sites != 0."""
    problems = []
    if loaded.n_trees != int(total_trees):
        problems.append(f"n_trees {loaded.n_trees} != total-trees {total_trees}")

    expected_fmt = XGB_FORMAT if arm["arm"] == ARM_A else DP_MODEL_FORMAT
    if loaded.fmt != expected_fmt:
        problems.append(f"format {loaded.fmt!r} != expected {expected_fmt!r}")
        return problems  # wrong object type — the meta checks below would misfire

    frc = read_back_fed_run_config(loaded)
    params = frc.get("params", {})
    for key in ("subsample", "colsample_bytree"):
        val = params.get(key)
        if val is None or float(val) != 1.0:
            problems.append(f"read-back params.{key}={val!r} != 1.0 (R9 matching)")

    if arm["arm"] == ARM_A:
        return problems

    dp_meta = extract_dp_meta(loaded.booster)
    if arm["arm"] == ARM_B:
        if dp_meta["num_releases"] != 0:
            problems.append(f"identity num_releases={dp_meta['num_releases']} != 0")
        for key in ("epsilon", "sigma"):
            if dp_meta[key] is not None:
                problems.append(f"identity {key}={dp_meta[key]!r} — expected None "
                                "(inf is scrubbed to null by to_json_bytes)")
        return problems

    # arm C
    if dp_meta["mechanism"] != "gaussian":
        problems.append(f"meta['dp']['mechanism']={dp_meta['mechanism']!r} != 'gaussian'")
    sigma = dp_meta["sigma"]
    if not (isinstance(sigma, (int, float)) and sigma is not None
            and math.isfinite(sigma) and sigma > 0):
        problems.append(f"sigma={sigma!r} not finite > 0")
    if dp_meta["per_site_trees"] != per_site:
        problems.append(f"per_site_trees={dp_meta['per_site_trees']} != "
                        f"per_site_tree_budget {per_site}")
    depth = params.get("max_depth")
    if depth is None:
        problems.append("read-back params.max_depth missing — cannot check the "
                        "release-count invariant")
    else:
        expected_releases = 2 * int(depth) * per_site
        if dp_meta["num_releases"] != expected_releases:
            problems.append(f"num_releases={dp_meta['num_releases']} != "
                            f"2·D·per_site = {expected_releases}")
    eps = dp_meta["epsilon"]
    target = arm["epsilon"]
    if eps is None or not math.isclose(eps, target, rel_tol=1e-3):
        problems.append(f"reported_epsilon={eps!r} not within 1e-3 relative of "
                        f"target ε={target}")
    return problems


# --------------------------------------------------------------------------- #
# §4.4/§4.6  ledger-delta validation (strategy-aware)
# --------------------------------------------------------------------------- #
def ledger_delta(before: list, after: list) -> list:
    """Entries appended between two reads. The ledger is append-only, so `before`
    must be a prefix of `after` — anything else means the file was rewritten."""
    if after[:len(before)] != before:
        raise ValueError("ledger is not append-only: existing entries changed "
                         "between reads")
    return after[len(before):]


def entries_per_c_run(strategy: str) -> int:
    """bagging: every site trains round 1 -> 2 entries/C run. cyclic: one site per
    round, only the round-1 site appends -> 1 entry/C run (the landed-code
    accounting gap the report surfaces as UNLEDGERED; Risk §8)."""
    if strategy == "bagging":
        return 2
    if strategy == "cyclic":
        return 1
    raise ValueError(f"Unknown strategy: {strategy!r}")


def validate_sweep_ledger(delta, c_runs, provenance, strategy) -> list[str]:
    """The §4.4 ledger assertions over `delta` (entries appended during the runs
    covered by this check). `c_runs` are the successfully-completed C runs those
    entries belong to, in run order, each {"label", "epsilon", "sigma"} carrying
    the MODEL META values (the cross-check is entry == meta). Pass c_runs=[] to
    assert an A/B run appended nothing. Returns violation strings (empty = pass).

    On example-halves provenance the append is gated off entirely, so ANY entry —
    any arm, either strategy — is a bug.
    """
    problems = []
    if provenance != "real-frozen-schema":
        if delta:
            problems.append(f"{len(delta)} ledger entries written on {provenance!r} "
                            "provenance (append is provenance-gated; a rehearsal "
                            "that writes the ledger is a bug)")
        return problems

    per_run = entries_per_c_run(strategy)
    expected = per_run * len(c_runs)
    if len(delta) != expected:
        problems.append(f"expected {expected} new ledger entries "
                        f"({per_run}/C-run × {len(c_runs)} on {strategy}), "
                        f"found {len(delta)}")
        return problems

    for i, run in enumerate(c_runs):
        chunk = delta[i * per_run:(i + 1) * per_run]
        hashes = {e["config_hash"] for e in chunk}
        if len(hashes) != 1:
            problems.append(f"{run['label']}: its {per_run} entries carry "
                            f"{len(hashes)} distinct config_hash values")
        for e in chunk:
            if e["mechanism"] != "gaussian":
                problems.append(f"{run['label']}: entry mechanism "
                                f"{e['mechanism']!r} != 'gaussian'")
            if run.get("epsilon") is not None and not math.isclose(
                    float(e["epsilon"]), float(run["epsilon"]), rel_tol=1e-9):
                problems.append(f"{run['label']}: entry epsilon {e['epsilon']} != "
                                f"model meta reported_epsilon {run['epsilon']}")
            if run.get("sigma") is not None and not math.isclose(
                    float(e["noise_multiplier"]), float(run["sigma"]), rel_tol=1e-9):
                problems.append(f"{run['label']}: entry noise_multiplier "
                                f"{e['noise_multiplier']} != model meta σ "
                                f"{run['sigma']}")
    return problems


def unledgered_sites(expected_sites, ledger_totals) -> list[str]:
    """Sites that spent ε but never appended (cyclic round-1 gap): the half
    basenames missing from ledger_total()'s keys. The report MUST mark these
    UNLEDGERED — never blank or 0, both of which read as 'spent nothing'."""
    return sorted(set(expected_sites) - set(ledger_totals))


# --------------------------------------------------------------------------- #
# §4.5  deltas
# --------------------------------------------------------------------------- #
def _metric_or_none(metrics, key):
    if metrics is None:
        return None
    val = metrics.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _delta_map(minuend_by_half, subtrahend_by_half, halves) -> dict:
    """Per-half {metric: minuend − subtrahend} + a 'mean' row across halves.
    Any None operand makes that delta None (the §4.3-of-1.1.a convention);
    a None per-half delta makes the mean None (per-half is primary anyway)."""
    out = {}
    for h in halves:
        out[h] = {}
        for m in DELTA_METRICS:
            a = _metric_or_none(minuend_by_half.get(h), m)
            b = _metric_or_none(subtrahend_by_half.get(h), m)
            out[h][m] = None if a is None or b is None else a - b
    mean = {}
    for m in DELTA_METRICS:
        vals = [out[h][m] for h in halves]
        mean[m] = (None if any(v is None for v in vals)
                   else sum(vals) / len(vals))
    out["mean"] = mean
    return out


def compute_deltas(arms, halves) -> dict:
    """B→C@ε (privacy cost, HEADLINE) and A→B (learner cost, context) per half
    per metric, from the arms' offline metrics. `arms` is the results-shaped list
    (each {"arm", "label", "metrics": {half: dict|None}})."""
    by_label = {a["label"]: a for a in arms}
    metrics_a = by_label["armA"]["metrics"]
    metrics_b = by_label["armB"]["metrics"]
    privacy = {}
    for a in arms:
        if a["arm"] != ARM_C:
            continue
        privacy[a["label"]] = _delta_map(a["metrics"], metrics_b, halves)
    return {
        "privacy_cost": privacy,                      # Δ_priv = C@ε − B
        "learner_cost": _delta_map(metrics_b, metrics_a, halves),  # Δ_learn = B − A
        "sign_convention": SIGN_CONVENTION,
    }


# --------------------------------------------------------------------------- #
# §4.7  cohort stats
# --------------------------------------------------------------------------- #
def cohort_stats(half_paths, split_seed) -> dict:
    """Per half: post-dedup row/patient counts via the same split chain the runs
    use (FLAT mode: train + valid totals), plus per-feature RAW missingness (NaN
    rate on the raw parquet, pre-sentinel). Inside-boundary values (R5): local GVA
    reads are allowed for debugging; the report must not leave the project."""
    stats = {}
    for path in half_paths:
        path = Path(path)
        raw = pd.read_parquet(path)
        train_df, valid_df = split_half(path, split_seed=split_seed,
                                        holdout_frac=0.0, holdout_eval=False)
        pids = set(train_df["patient_id"]) | set(valid_df["patient_id"])
        stats[path.name] = {
            "raw_rows": int(len(raw)),
            "rows_after_dedup": int(len(train_df) + len(valid_df)),
            "patients_after_dedup": int(len(pids)),
            "missingness": {c: float(raw[c].isna().mean()) for c in FEATURE_COLS},
        }
    return stats


# --------------------------------------------------------------------------- #
# §4.7  results assembly + report rendering
# --------------------------------------------------------------------------- #
_RESULTS_KEYS = ("meta", "cohort", "arms", "deltas", "ledger")


def build_results(meta, cohort, arms, deltas, ledger) -> dict:
    """The machine-readable sweep artifact (1.1.e's notebook ingests this)."""
    results = {"meta": meta, "cohort": cohort, "arms": arms, "deltas": deltas,
               "ledger": ledger}
    validate_results(results)
    return results


def validate_results(results) -> None:
    for key in _RESULTS_KEYS:
        if key not in results:
            raise ValueError(f"results artifact missing {key!r}")
    for field in ("date", "operator", "data_provenance", "strategy", "delta",
                  "epsilons", "shared_config", "split_seed"):
        if field not in results["meta"]:
            raise ValueError(f"results meta missing {field!r}")
    labels = [a["label"] for a in results["arms"]]
    for required in ("armA", "armB"):
        if required not in labels:
            raise ValueError(f"results arms missing {required!r}")
    for field in ("path", "entries_added", "ledger_total", "unledgered_sites"):
        if field not in results["ledger"]:
            raise ValueError(f"results ledger missing {field!r}")


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _fmt_ci(metrics, key):
    if metrics is None:
        return "DEGENERATE"
    val = _metric_or_none(metrics, key)
    lo = _metric_or_none(metrics, f"{key}_lo")
    hi = _metric_or_none(metrics, f"{key}_hi")
    if val is None:
        return "—"
    if lo is None or hi is None:
        return _fmt(val)
    return f"{_fmt(val)} [{_fmt(lo)}, {_fmt(hi)}]"


def _fmt_cell(metrics, key):
    return "DEGENERATE" if metrics is None else _fmt(_metric_or_none(metrics, key))


def _arm_mechanism_cell(arm):
    if arm["arm"] == ARM_A:
        return "—"
    dp_meta = arm.get("dp_meta") or {}
    if arm["arm"] == ARM_B:
        return "identity (noise off)"
    sigma = dp_meta.get("sigma")
    return f"gaussian σ={_fmt(sigma, 3)}" if sigma is not None else "gaussian"


def _arm_epsilon_cell(arm):
    if arm["arm"] == ARM_A:
        return "—"
    if arm["arm"] == ARM_B:
        return "∞ (none spent)"
    dp_meta = arm.get("dp_meta") or {}
    eps = dp_meta.get("epsilon")
    return _fmt(eps) if eps is not None else _fmt(arm.get("epsilon"))


def _arm_learner_cell(arm):
    return "stock `xgb.train`" if arm["arm"] == ARM_A else "DP learner"


def _delta_lines(delta_map, halves) -> list[str]:
    lines = ["| half | ΔAUC-ROC | ΔAUC-PR | ΔBrier |",
             "|---|---|---|---|"]
    for h in list(halves) + ["mean"]:
        row = delta_map.get(h, {})
        lines.append(f"| {h} | {_fmt(row.get('auc_roc'))} "
                     f"| {_fmt(row.get('auc_pr'))} | {_fmt(row.get('brier'))} |")
    return lines


def render_report(results) -> str:
    """The filled 1.1.b report, rendered programmatically (house style:
    metrics.render_run_report / hpo.render_leaderboard — never string-substitution
    into the template). Mirrors every skeleton section heading exactly; the §4.9
    heading tripwire test pins the two in sync. Box 1 of the release boundary and
    the whole go/no-go section stay operator-only (Decision 6)."""
    meta = results["meta"]
    ledger = results["ledger"]
    arms = results["arms"]
    deltas = results["deltas"]
    halves = list(results["cohort"].keys())
    rehearsal = meta["data_provenance"] != "real-frozen-schema"
    site_labels = ledger.get("site_labels", {})
    c_arms = [a for a in arms if a["arm"] == ARM_C]

    lines = [f"# 1.1.b DP sweep report — {meta['strategy']} @ "
             f"{meta.get('federation', 'local-deployment')}", ""]
    if rehearsal:
        lines += [f"> {REHEARSAL_BANNER}", ""]

    # ----- Run identification ------------------------------------------------
    lines += ["## Run identification", ""]
    lines.append(f"- Date / operator: {meta['date']} / {meta['operator']}")
    if meta.get("gate_ack"):
        lines.append(f"- Gate acknowledgement (R8): {meta['gate_ack']}")
    lines.append(f"- Data provenance: `{meta['data_provenance']}` (federated "
                 "2-node topology, **Geneva-only** — Shenzhen is NOT in the "
                 "network; spec 1.1.a″ acceptance 4)")
    lines.append("- Cohort: deduped one-row-per-patient (R3), keyed-hash split "
                 "(R4) — identical for ALL arms")
    for half, st in results["cohort"].items():
        lines.append(f"  - {half}: {st['rows_after_dedup']} rows / "
                     f"{st['patients_after_dedup']} patients after dedup "
                     f"(raw rows {st['raw_rows']})")
    lines.append(f"  - split: FLAT `test_size=0.2`, `split_seed={meta['split_seed']}`; "
                 "hash rule `HMAC(SPLIT_KEY‖role‖seed, pid)`")
    lines.append("  - missingness: variant 1 sentinel/missing-bin encoding "
                 "(`REMOVE-IF-NO-DP`), applied at the loader for ALL arms — "
                 "% missing per feature per half:")
    for half, st in results["cohort"].items():
        miss = "; ".join(f"`{feat}` {100 * rate:.1f}%"
                         for feat, rate in st["missingness"].items())
        lines.append(f"    - {half}: {miss}")
    hash_cells = []
    for arm in c_arms:
        h = arm.get("config_hash")
        hash_cells.append(f"{arm['label']}: `{h}`" if h else f"{arm['label']}: —")
    lines.append("- Config hash(es) (`dp_ledger` entries): "
                 + ("; ".join(hash_cells) if hash_cells else "—")
                 + "; A/B: n/a — no DP, not ledgered")
    lines.append("")

    # ----- Comparator arms ---------------------------------------------------
    lines += ["## Comparator arms (R9 — all on the SAME deduped halves, SAME "
              "hash split, matched hyperparameters)", ""]
    lines.append("Hyperparameter matching: every arm ran at `subsample = "
                 "colsample_bytree = 1.0` (the DP learner forces q = 1.0 and "
                 "ignores both), `base_score` explicitly configured and identical "
                 "everywhere — verified by read-back from every saved model, not "
                 "from intent.")
    lines.append("")
    lines += ["| Arm | Half | Learner | Mechanism | ε | AUC-ROC (95% CI) "
              "| AUC-PR | Brier |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for half in halves:
            m = (arm.get("metrics") or {}).get(half)
            m = None if arm.get("degenerate") else m
            arm_name = (arm["label"] if arm["arm"] != ARM_C
                        else f"C @ ε={arm['epsilon']:g}")
            arm_name = {"armA": "A", "armB": "B"}.get(arm_name, arm_name)
            lines.append(
                f"| {arm_name} | {half} | {_arm_learner_cell(arm)} "
                f"| {_arm_mechanism_cell(arm)} | {_arm_epsilon_cell(arm)} "
                f"| {_fmt_ci(m, 'auc_roc')} | {_fmt_cell(m, 'auc_pr')} "
                f"| {_fmt_cell(m, 'brier')} |")
    degenerate = [a["label"] for a in arms if a.get("degenerate")]
    if degenerate:
        lines.append("")
        lines.append(f"DEGENERATE arms (run failed/timed out): "
                     f"{', '.join(degenerate)} — their ledger spend (if the "
                     "entry landed) still composes below (intent-to-spend).")
    lines.append("")
    lines.append(f"Sign convention for the deltas below: {SIGN_CONVENTION}.")
    lines.append("")
    lines.append("- **B→C = cost of privacy (HEADLINE)** — B and C share "
                 "everything except the mechanism:")
    for label, dmap in deltas["privacy_cost"].items():
        lines.append("")
        lines.append(f"  {label} (Δ = C − B):")
        lines.append("")
        lines += ["  " + ln for ln in _delta_lines(dmap, halves)]
    lines.append("")
    lines.append("- A→B = learner cost (context only — does the from-scratch DP "
                 "learner track XGBoost; Δ = B − A):")
    lines.append("")
    lines += ["  " + ln for ln in _delta_lines(deltas["learner_cost"], halves)]
    lines.append("")

    # ----- Privacy accounting ------------------------------------------------
    lines += ["## Privacy accounting (R6 — both numbers, always)", ""]
    if rehearsal:
        lines.append("No ledger entries were written: the append is provenance-"
                     "gated to `real-frozen-schema`, and this is a rehearsal "
                     "(zero new entries asserted). The table below is therefore "
                     "empty of real spend.")
        lines.append("")
    totals = ledger["ledger_total"]
    per_run_eps = ledger.get("per_run_epsilon", {})
    delta_val = meta["delta"]
    lines += ["| Site | This run's ε (per-run) | Composed ledger-total ε to date "
              "| δ |",
              "|---|---|---|---|"]
    all_sites = sorted(set(ledger.get("expected_sites", [])) | set(totals))
    for site in all_sites:
        label = site_labels.get(site)
        site_cell = f"{site} ({label})" if label else site
        if site in ledger["unledgered_sites"]:
            per_run_cell = total_cell = UNLEDGERED_MARK
        else:
            runs = per_run_eps.get(site, {})
            per_run_cell = ("; ".join(f"{lbl}: {_fmt(eps)}"
                                      for lbl, eps in runs.items())
                            if runs else "—")
            total_cell = _fmt(totals.get(site))
        lines.append(f"| {site_cell} | {per_run_cell} | {total_cell} "
                     f"| {delta_val:g} |")
    lines.append("")
    lines.append(f"- Ledger file(s): `{ledger['path']}` (site-local; totals "
                 "composed PER SITE, never across)")
    lines.append(f"- `ledger_total()` output pasted verbatim: `{totals!r}`")
    lines.append("- The composed total covers ALL recorded real-data spend to "
                 "date — including failed attempts and prior sweeps (intent-to-"
                 "spend, R6 semantics), not just this sweep's runs.")
    if ledger["unledgered_sites"]:
        lines.append(f"- UNLEDGERED sites: {', '.join(ledger['unledgered_sites'])} "
                     "— these nodes DID spend ε (they add noisy trees) but the "
                     "cyclic round-1 append gap means their spend was never "
                     "recorded (landed-code limitation, Risk §8). A per-site B→C "
                     "privacy claim for them is NOT supported; the go/no-go "
                     "should rest on the bagging sweep.")
    lines.append("")

    # ----- Release boundary --------------------------------------------------
    lines += ["## Release boundary (R5)", ""]
    lines.append("- [ ] Nothing inside-boundary (validation metrics, confusion "
                 "matrices, exact counts) appears in this artifact if it leaves "
                 "the project; only the selected model (DP-accounted) and figures "
                 "derived from it cross. *(operator attestation — this report "
                 "contains inside-boundary values and must not leave the "
                 "project)*")
    lines.append("- [x] DP train replies carried `dp-site-weight` (fixed public "
                 "constants), never exact counts. *(code-enforced + test-pinned)*")
    lines.append("")

    # ----- Go/no-go ----------------------------------------------------------
    lines += ["## Go/no-go (spec 1.1.a″ acceptance 5)", ""]
    lines.append("- Is the B→C privacy cost acceptable at a usable ε? Decision + "
                 "rationale: ______ *(operator-only — the driver fills every "
                 "computable field; this judgment is recorded manually)*")
    lines.append("- Recorded in `docs/logbook.md` on: ______ (BEFORE any Shenzhen "
                 "integration)")
    lines.append("")
    return "\n".join(lines)
