"""fed_stroke: HPO harness core (roadmap 1.1.a).

Pure, importable logic behind `scripts/run_hpo.py`: no I/O, no `flwr` app context,
no subprocess. Everything here is unit-tested directly (tests/test_hpo.py); the
process boundary (spawning `flwr run`, reading saved models) lives in the driver.

Search space + trial generation (§4.1), objective + selection (§4.2), and the
results artifact / leaderboard / narrowed ranges (§4.3). Coarse by design — the
search *method* upgrade (Optuna/TPE) is roadmap 1.1.c, not here.
"""
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from fed_stroke.server_app import derive_num_rounds

# Coarse first-pass grid (architecture §3 defaults sit inside each range).
# Overridable via run_hpo.py --grid <file.json>. num_server_rounds is realized as
# total-trees (§3.1). Adaptive/finer search is 1.1.c, not here.
DEFAULT_GRID = {
    "total-trees":             [40, 80, 160],
    "params.max-depth":        [3, 4, 6],
    "params.eta":              [0.05, 0.1, 0.3],
    "params.min-child-weight": [1, 5, 20],
    "params.subsample":        [0.8, 1.0],
    "params.colsample-bytree": [0.8, 1.0],
}  # 3·3·3·3·2·2 = 324 cells

# The six roadmap knobs, in leaderboard/summary column order (total-trees first —
# it realizes the num_server_rounds axis, §3.1).
KNOBS = [
    "total-trees",
    "params.max-depth",
    "params.eta",
    "params.min-child-weight",
    "params.subsample",
    "params.colsample-bytree",
]

_REQUIRED_TOP_KEYS = {"meta", "signal", "trials", "winner", "holdout"}
_REQUIRED_SIGNAL_KEYS = {
    "top_objective", "median_objective", "top_minus_median",
    "winner_vs_runnerup_ci_overlap",
}
_VALID_PROVENANCE = {"example-halves", "real-frozen-schema"}


# --------------------------------------------------------------------------- #
# §4.1  search space + trial generation
# --------------------------------------------------------------------------- #
def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of the grid -> ordered list of flat override dicts.

    Each dict maps a `flwr run --run-config` dotted key to a value, e.g.
    {"total-trees": 40, "params.max-depth": 4, ...}. Deterministic ordering (keys
    sorted, itertools.product) so trial ids are stable across runs.
    """
    keys = sorted(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def trial_id(override: dict) -> str:
    """Short, cross-process-stable id for a trial override dict.

    A sha1 digest of the canonical (`sort_keys=True`) JSON, NOT Python's builtin
    `hash()`, which is per-process salted by PYTHONHASHSEED and would give a trial a
    *different* directory on a resume or re-run — silently re-running everything and
    orphaning the first run's models (Decision 10, §4.1). Canonical serialization
    makes the id invariant to dict insertion order.
    """
    canon = json.dumps(override, sort_keys=True).encode()
    return hashlib.sha1(canon).hexdigest()[:10]


def valid_trials(trials, train_method, num_sites, local_epochs):
    """Split trials into (runnable, rejected) by tree-budget divisibility.

    Reuses server_app.derive_num_rounds: a trial whose `total-trees` is not divisible
    by trees/round for this strategy is rejected here (fail fast) rather than launched
    and rejected by the server. Rejected trials carry the reason so the driver can log
    them — never silently dropped.
    """
    runnable, rejected = [], []
    for trial in trials:
        try:
            derive_num_rounds(
                train_method, trial["total-trees"], num_sites, local_epochs
            )
        except ValueError as exc:
            rejected.append({"trial": trial, "reason": str(exc)})
        else:
            runnable.append(trial)
    return runnable, rejected


def subsample_trials(trials, max_trials, seed):
    """Random subsample escape hatch (§4.4). `max_trials <= 0` or `>= len(trials)`
    returns the full grid unchanged; otherwise a reproducible `random.Random(seed)`
    sample. Result is ordered by trial_id so iteration is stable given the seed.
    """
    trials = list(trials)
    if max_trials <= 0 or max_trials >= len(trials):
        return trials
    sample = random.Random(seed).sample(trials, max_trials)
    return sorted(sample, key=trial_id)


def _toml_scalar(value) -> str:
    """Render a Python scalar as a TOML token for a `--run-config` string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    # strings (incl. paths) are TOML single-quoted so flwr's parser keeps them whole
    return f"'{value}'"


def build_run_config(override, strategy, split_seed, model_dir, metrics_dir,
                     holdout_frac=0.0, holdout_eval=False, save_model=True,
                     n_boot=0) -> str:
    """Override dict -> the exact `flwr run --run-config` string for one trial.

    Emits dotted keys, TOML-quoted string values (`key='value'`), ABSOLUTE
    `model-dir`/`metrics-dir` (Decision 7 — the ServerApp CWD differs from the
    driver's), `save-model=true`, and `n-boot=0` (the objective scores the saved
    model offline, so in-run bootstrap CIs would be pure discarded compute). Produces
    no shell metacharacters; the driver passes it as ONE argument with `shell=False`,
    so flwr's TOML parser (not a shell) consumes the quotes.
    """
    tokens = [f"train-method={_toml_scalar(strategy)}"]
    for key in sorted(override):
        tokens.append(f"{key}={_toml_scalar(override[key])}")
    tokens.append(f"split-seed={_toml_scalar(int(split_seed))}")
    tokens.append(f"holdout-frac={_toml_scalar(float(holdout_frac))}")
    tokens.append(f"holdout-eval={_toml_scalar(bool(holdout_eval))}")
    tokens.append(f"save-model={_toml_scalar(bool(save_model))}")
    tokens.append(f"n-boot={_toml_scalar(int(n_boot))}")
    tokens.append(f"model-dir={_toml_scalar(str(model_dir))}")
    tokens.append(f"metrics-dir={_toml_scalar(str(metrics_dir))}")
    return " ".join(tokens)


def trial_seed_dir(out_dir, tid, seed) -> Path:
    """Per-(trial, seed) output directory: <out_dir>/<trial_id>/seed<seed>."""
    return Path(out_dir) / tid / f"seed{seed}"


def final_model_path(out_dir, tid, seed) -> Path:
    """Path of the saved model for a (trial, seed)."""
    return trial_seed_dir(out_dir, tid, seed) / "final_model.json"


def is_completed(out_dir, tid, seed) -> bool:
    """Resume predicate: True if this (trial, seed)'s saved model already exists, so
    the driver can skip the `flwr run` and score the existing model directly (§4.4).
    """
    return final_model_path(out_dir, tid, seed).exists()


# --------------------------------------------------------------------------- #
# §4.2  objective + selection
# --------------------------------------------------------------------------- #
def trial_objective(per_repeat, k_search: int) -> dict:
    """Aggregate one trial's per-repeat per-site AUC-ROC into a scalar objective.

    `per_repeat`: [{"site_aucs": {site: auc_or_None, ...}, ...}, ...] — one dict per
    search seed (any extra keys, e.g. `split_seed`, are passed through untouched).

    Per repeat: objective_r = mean(site AUCs), variance_r = population variance across
    sites — ONLY if every site AUC is present. A repeat with any None site AUC (or no
    sites) is DROPPED (single-class split, §8). The trial's objective is the mean of
    objective_r over valid repeats; variance is the mean of variance_r.

    NaN policy: if fewer than ceil(k_search / 2) valid repeats remain, the trial is
    INVALID (objective = NaN) — surfaced in the leaderboard, excluded from ranking,
    never silently averaged from too little signal.

    Returns {objective, variance, n_valid_repeats, valid, per_repeat} where the
    returned per_repeat echoes the input enriched with per-repeat objective/variance
    and a `dropped` flag.
    """
    enriched, objs, vars_ = [], [], []
    for rep in per_repeat:
        site_aucs = rep["site_aucs"]
        aucs = list(site_aucs.values())
        dropped = len(aucs) == 0 or any(a is None for a in aucs)
        if dropped:
            enriched.append({**rep, "objective": None, "variance": None,
                             "dropped": True})
        else:
            obj_r = statistics.fmean(aucs)
            var_r = statistics.pvariance(aucs)
            objs.append(obj_r)
            vars_.append(var_r)
            enriched.append({**rep, "objective": obj_r, "variance": var_r,
                             "dropped": False})
    n_valid = len(objs)
    valid = n_valid >= math.ceil(k_search / 2)
    objective = statistics.fmean(objs) if valid else float("nan")
    variance = statistics.fmean(vars_) if valid else float("nan")
    return {
        "objective": objective,
        "variance": variance,
        "n_valid_repeats": n_valid,
        "valid": valid,
        "per_repeat": enriched,
    }


def rank_trials(results: list[dict]) -> list[dict]:
    """Sort trials by (objective desc, variance asc); invalid trials sink to the
    bottom; ties broken deterministically by trial_id. Mirrors the v1.3 winner rule
    (roadmap 1.3.d): higher mean site AUC, then lower cross-site variance.
    """
    valid = [r for r in results if r.get("valid")]
    invalid = [r for r in results if not r.get("valid")]
    valid.sort(key=lambda r: (-r["objective"], r["variance"], r["trial_id"]))
    invalid.sort(key=lambda r: r["trial_id"])
    return valid + invalid


def select_winner(ranked: list[dict]):
    """Top valid trial, or None if every trial is invalid (all-degenerate sweep — the
    driver then errors loudly rather than emit a meaningless 'winner').
    """
    for r in ranked:
        if r.get("valid"):
            return r
    return None


# --------------------------------------------------------------------------- #
# §4.3  results artifact + leaderboard + narrowed ranges + signal
# --------------------------------------------------------------------------- #
def _mode(values):
    """Most common value; ties broken by the smallest value (deterministic)."""
    counts = Counter(values)
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def summarize_top_ranges(ranked: list[dict], top_frac: float = 0.1) -> dict:
    """Per-hyperparameter min / max / mode across the top-decile valid trials — the
    narrowed search space handed to 1.1.b and the v1.3 re-run. Rendered to
    narrowed_ranges.md by the driver. Empty dict if there are no valid trials.
    """
    valid = [r for r in ranked if r.get("valid")]
    if not valid:
        return {}
    k = max(1, math.ceil(len(valid) * top_frac))
    top = valid[:k]
    out = {"n_top": k}
    for knob in KNOBS:
        vals = [t["params"][knob] for t in top]
        out[knob] = {
            "min": min(vals),
            "max": max(vals),
            "mode": _mode(vals),
            "values": sorted(set(vals)),
        }
    return out


def _repeat_interval(trial):
    """(lo, hi) = mean ± population-stdev of a trial's per-repeat objectives — a
    coarse spread proxy for the resolving-power check (no in-run CIs during search).
    """
    vals = [r["objective"] for r in trial["per_repeat"] if not r.get("dropped")]
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return (m - s, m + s)


def build_signal(ranked: list[dict]) -> dict:
    """Resolving-power check (outside-voice #2): does the ranking carry signal, or is
    the 'winner' noise? Over valid trials: top objective, median objective, their gap,
    and whether the winner's objective spread overlaps the runner-up's. Feeds
    results.json["signal"] and a bold WARNING banner in narrowed_ranges.md when
    top≈median or the spreads overlap — so a human never reads a noise-tier winner as
    a real optimum (a small cohort with a rare outcome makes this a real risk, §8).
    """
    valid = [t for t in ranked if t.get("valid")]
    if not valid:
        return {
            "top_objective": None,
            "median_objective": None,
            "top_minus_median": None,
            "winner_vs_runnerup_ci_overlap": True,
        }
    objs = [t["objective"] for t in valid]
    top = objs[0]
    median = statistics.median(objs)
    if len(valid) < 2:
        # No runner-up to separate from — cannot establish resolving power.
        overlap = True
    else:
        w_lo, w_hi = _repeat_interval(valid[0])
        r_lo, r_hi = _repeat_interval(valid[1])
        overlap = (w_lo <= r_hi) and (r_lo <= w_hi)
    return {
        "top_objective": top,
        "median_objective": median,
        "top_minus_median": top - median,
        "winner_vs_runnerup_ci_overlap": overlap,
    }


def _nan_to_none(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _trial_public(t: dict) -> dict:
    """Trim an internal trial-result dict to the results.json trial schema (§4.3),
    converting NaN -> None so the artifact is strict-JSON (allow_nan=False).
    """
    return {
        "trial_id": t["trial_id"],
        "params": t["params"],
        "objective": _nan_to_none(t["objective"]),
        "variance": _nan_to_none(t["variance"]),
        "n_valid_repeats": t["n_valid_repeats"],
        "valid": t["valid"],
        "per_repeat": [
            {"split_seed": r.get("split_seed"), "site_aucs": r["site_aucs"]}
            for r in t["per_repeat"]
        ],
    }


def build_results(meta: dict, ranked: list[dict], holdout) -> dict:
    """Assemble the results artifact (schema in spec §4.3); pure, unit-testable."""
    winner = select_winner(ranked)
    signal = {k: _nan_to_none(v) for k, v in build_signal(ranked).items()}
    return {
        "meta": meta,
        "signal": signal,
        "trials": [_trial_public(t) for t in ranked],
        "winner": ({"trial_id": winner["trial_id"], "params": winner["params"]}
                   if winner else None),
        "holdout": holdout,
    }


def validate_results(results: dict) -> None:
    """Structural contract check before write; raises ValueError on bad shape (mirrors
    metrics.validate_metrics_artifact). Required top-level keys: meta, signal, trials,
    winner, holdout. `meta` MUST carry `data_provenance` ∈ {"example-halves",
    "real-frozen-schema"} (the throwaway gate, §4.7) and `holdout_frac` /
    `holdout_partition_seed`; `signal` MUST carry all four resolving-power keys (§4.3).
    A missing or unknown-valued key raises ValueError — the artifact never ships
    without its provenance stamp or its noise-tier signal.
    """
    missing = _REQUIRED_TOP_KEYS - results.keys()
    if missing:
        raise ValueError(f"results missing top-level keys: {sorted(missing)}")

    meta = results["meta"]
    provenance = meta.get("data_provenance")
    if provenance not in _VALID_PROVENANCE:
        raise ValueError(
            f"meta.data_provenance must be one of {sorted(_VALID_PROVENANCE)}; "
            f"got {provenance!r}"
        )
    for key in ("holdout_frac", "holdout_partition_seed"):
        if key not in meta:
            raise ValueError(f"meta missing required key: {key}")

    missing_signal = _REQUIRED_SIGNAL_KEYS - results["signal"].keys()
    if missing_signal:
        raise ValueError(f"signal missing keys: {sorted(missing_signal)}")


def _fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_leaderboard(results: dict) -> str:
    """Markdown leaderboard (mirrors metrics.render_run_report style): rank, trial_id,
    the six knobs, objective, variance, n_valid_repeats, and a ⚠ flag on
    invalid/degenerate trials so they stay visible (never silently dropped).
    """
    meta = results["meta"]
    lines = [
        f"# HPO leaderboard — {meta.get('strategy', '?')} "
        f"({meta.get('federation', '?')})",
        "",
        f"Search seeds: {meta.get('search_seeds')} · holdout-frac: "
        f"{meta.get('holdout_frac')} · provenance: `{meta.get('data_provenance')}`",
        "",
        "| rank | trial | total-trees | max-depth | eta | min-child-weight "
        "| subsample | colsample-bytree | objective | variance | n_valid | flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(results["trials"], start=1):
        p = t["params"]
        flag = "" if t["valid"] else "⚠"
        lines.append(
            f"| {i} | `{t['trial_id']}` | {p['total-trees']} "
            f"| {p['params.max-depth']} | {p['params.eta']} "
            f"| {p['params.min-child-weight']} | {p['params.subsample']} "
            f"| {p['params.colsample-bytree']} | {_fmt(t['objective'])} "
            f"| {_fmt(t['variance'])} | {t['n_valid_repeats']} | {flag} |"
        )
    return "\n".join(lines) + "\n"
