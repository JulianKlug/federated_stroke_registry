"""fed_stroke: the site-stratified evaluation harness (roadmap 1.c).

Single source of truth for metric math, imported by both the federated client
(`client_app.evaluate`) and the offline scorer (`scripts/eval_final_model.py`)
so their numbers are identical by construction.

Architecture §4 requires AUC-ROC, AUC-PR, calibration (Brier score), and a
confusion matrix at the operating point, **stratified by site** — never
weight-averaged across sites the way flwr's default evaluate aggregator would.

The artifact this module produces is the single contract the v1.1.e reporting
notebook consumes:

    { <round:int> : { <site:str> : { <metric:str> : <number|null> } } }

- Round keys are ints (JSON stringifies them on write; the notebook must cast
  back). Site keys are parquet file names (e.g. `geneva_half_A.parquet`).
- The site set varies by round under cyclic (one site per round, alternating)
  and is both sites every round under bagging — consumers must NOT assume a
  fixed site set per round.
- AUC-family values may be `null` (single-class split -> NaN -> null). `brier`
  and all confusion cells are always present and numeric.
- Each site entry contains `REQUIRED_METRIC_KEYS`.
"""
import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Metric keys every valid per-site artifact entry must contain (§4.3 validator).
REQUIRED_METRIC_KEYS = {
    "auc_roc", "auc_pr", "brier", "tn", "fp", "fn", "tp",
    "op_j", "tn_j", "fp_j", "fn_j", "tp_j", "n_pos", "num-examples",
}


def _bootstrap_ci(y_true, y_prob, metric_fn, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CI for metric_fn; (nan, nan) if it can't be formed.

    Resamples that land single-class are skipped (the AUC metrics are undefined
    there); if every resample is single-class, returns (nan, nan).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():          # single-class resample -> skip
            continue
        stats.append(metric_fn(yt, yp))
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _youden_threshold(y_true, y_prob) -> float:
    """Threshold maximizing TPR - FPR (Youden's J) on the site's own ROC."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_binary_metrics(y_true, y_prob, operating_point=0.5,
                           n_boot=1000, boot_seed=0) -> dict:
    """Full §4 metric set (+ bootstrap CIs + data-driven point) for one split.

    Flat, MetricRecord-safe values:
      auc_roc, auc_roc_lo, auc_roc_hi   # + bootstrap 95% CI
      auc_pr,  auc_pr_lo,  auc_pr_hi
      brier,   brier_lo,   brier_hi
      tn, fp, fn, tp                    # confusion @ fixed operating_point
                                        #   (shared threshold -> cross-site comparable)
      op_j, tn_j, fp_j, fn_j, tp_j      # confusion @ per-site Youden-J point
                                        #   (per-site threshold -> NOT cross-site comparable)
      n, n_pos
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= operating_point).astype(int)

    # confusion matrix over the fixed {0,1} label space so counts are stable
    # even when a site's split is single-class (labels=[0, 1] avoids a 1x1 cm)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    single_class = n_pos == 0 or n_pos == len(y_true)

    out = {
        "auc_roc": float("nan"), "auc_roc_lo": float("nan"), "auc_roc_hi": float("nan"),
        "auc_pr":  float("nan"), "auc_pr_lo":  float("nan"), "auc_pr_hi":  float("nan"),
        "brier":   float(brier_score_loss(y_true, y_prob)),
        "brier_lo": float("nan"), "brier_hi": float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # data-driven point: undefined single-class -> op_j NaN, cells mirror fixed
        "op_j": float("nan"),
        "tn_j": int(tn), "fp_j": int(fp), "fn_j": int(fn), "tp_j": int(tp),
        "n": int(len(y_true)), "n_pos": n_pos,
    }
    if single_class:
        # AUCs, their CIs, and Youden's J are all undefined with one class
        # present -- leave them NaN, do not raise. brier and the fixed-point
        # confusion are still valid and already set above.
        return out

    out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    out["auc_pr"] = float(average_precision_score(y_true, y_prob))
    out["auc_roc_lo"], out["auc_roc_hi"] = _bootstrap_ci(
        y_true, y_prob, roc_auc_score, n_boot, boot_seed)
    out["auc_pr_lo"], out["auc_pr_hi"] = _bootstrap_ci(
        y_true, y_prob, average_precision_score, n_boot, boot_seed)
    out["brier_lo"], out["brier_hi"] = _bootstrap_ci(
        y_true, y_prob, brier_score_loss, n_boot, boot_seed)

    op_j = _youden_threshold(y_true, y_prob)
    tn_j, fp_j, fn_j, tp_j = confusion_matrix(
        y_true, (y_prob >= op_j).astype(int), labels=[0, 1]).ravel()
    out["op_j"] = op_j
    out["tn_j"], out["fp_j"], out["fn_j"], out["tp_j"] = (
        int(tn_j), int(fp_j), int(fn_j), int(tp_j))
    return out


def site_stratified_evaluate_metrics(records, weighting_metric_name):
    """Emit per-site metrics with NO cross-site averaging.

    Matches flwr's `evaluate_metrics_aggr_fn` hook signature: each reply
    RecordDict carries a MetricRecord + a ConfigRecord{"site": ...}. Output keys
    are site-suffixed: "auc_roc/geneva_half_A.parquet", etc.

    Unlike flwr's default aggregator, `num-examples` is deliberately re-tagged
    per site (`num-examples/<site>`) rather than used as an averaging weight:
    the per-site row count is useful context for reading a NaN AUC, and
    stratified reporting must not weight-average.
    """
    # Local import keeps this module importable without a live flwr app context
    # for the pure-math parts, matching how the rest of fed_stroke imports flwr.
    from flwr.app import MetricRecord

    out = MetricRecord()
    seen = set()
    for rec in records:
        site = str(next(iter(rec.config_records.values()))["site"])
        if site in seen:                     # two nodes claiming one file -> misconfig
            raise ValueError(f"Duplicate site in evaluate replies: {site}")
        seen.add(site)
        mr = next(iter(rec.metric_records.values()))
        # Every key is site-suffixed, num-examples included: it becomes the
        # per-site row count in the artifact rather than an averaging weight.
        for k, v in mr.items():
            out[f"{k}/{site}"] = v
    return out


def nest_site_metrics(evaluate_metrics_clientapp) -> dict:
    """Inverse of the aggregator's f"{metric}/{site}" encoding.

    Returns {round(int): {site(str): {metric(str): value}}}. NaN -> None so the
    artifact is standards-valid JSON; pandas reads null back as NaN, so the
    v1.1.e notebook is unaffected. Site names are parquet file names (no "/"),
    so rsplit("/", 1) recovers the site unambiguously even if a metric name ever
    contained a "/".
    """
    nested = {}
    for rnd, mr in evaluate_metrics_clientapp.items():
        for key, value in mr.items():
            metric, site = key.rsplit("/", 1)
            v = None if isinstance(value, float) and math.isnan(value) else value
            nested.setdefault(int(rnd), {}).setdefault(site, {})[metric] = v
    return nested


def summarize_final_round(nested: dict) -> list:
    """Per-site headline rows for a run's LAST round.

    Returns a list of dicts (one per site, site-sorted), each a metric dict with
    `round` and `site` added. The last round is the natural at-a-glance summary:
    bagging's final merged model on both sites; cyclic's last-trained site.
    """
    final_round = max(nested)
    return [
        {"round": final_round, "site": site, **nested[final_round][site]}
        for site in sorted(nested[final_round])
    ]


def _overview_num(value, spec=".4f") -> str:
    """Format a possibly-null (single-class → NaN → None) value for the table."""
    return "—" if value is None else format(value, spec)


def render_run_report(tag: str, nested: dict, operating_point=None) -> str:
    """Render a human-readable Markdown report for ONE run's nested artifact.

    Written beside the run's `<tag>.json` as `<tag>.md`. Two sections: a
    final-round headline table (the full metric set per site) and a per-round
    AUC-ROC trajectory (one column per site — the informative view for cyclic,
    where each round trains/evaluates a different site). The JSON stays the
    source of truth; this is a convenience view regenerated on every run.
    """
    out = [
        f"# `{tag}` — evaluation metrics",
        "",
        f"_Auto-generated by `fed_stroke.server_app`. Source of truth: `{tag}.json` "
        "in this directory; this file is a convenience summary, regenerated on every run._",
        "",
    ]
    if operating_point is not None:
        out += [
            f"Fixed operating point: **{operating_point}** — shared across sites, so the "
            "fixed confusion matrix is cross-site comparable. The Youden-J point (`op_J`) "
            "is chosen per site from its own ROC and is **not** comparable across sites.",
            "",
        ]
    if not nested:
        out += ["_No evaluate rounds captured._", ""]
        return "\n".join(out)

    # -- final-round headline -------------------------------------------------
    final_round = max(nested)
    out += [
        f"## Final round ({final_round}) — headline",
        "",
        "| site | n | n_pos | AUC-ROC (95% CI) | AUC-PR | Brier "
        "| fixed (tn,fp,fn,tp) | Youden (tn,fp,fn,tp) | op_J |",
        "|---|---:|---:|---|---:|---:|---|---|---:|",
    ]
    any_degenerate = False
    for m in summarize_final_round(nested):
        ci = f"[{_overview_num(m.get('auc_roc_lo'))}, {_overview_num(m.get('auc_roc_hi'))}]"
        fixed = f"({m['tn']},{m['fp']},{m['fn']},{m['tp']})"
        if m["fp"] == 0 and m["tp"] == 0:   # degenerate fixed threshold
            fixed += " ⚠"
            any_degenerate = True
        youden = f"({m['tn_j']},{m['fp_j']},{m['fn_j']},{m['tp_j']})"
        out.append(
            f"| {m['site']} | {m['num-examples']} | {m['n_pos']} "
            f"| {_overview_num(m.get('auc_roc'))} {ci} | {_overview_num(m.get('auc_pr'))} "
            f"| {_overview_num(m.get('brier'))} | {fixed} | {youden} "
            f"| {_overview_num(m.get('op_j'))} |"
        )

    # -- per-round AUC-ROC trajectory ----------------------------------------
    sites = sorted({s for sites in nested.values() for s in sites})
    out += [
        "",
        "## Per-round AUC-ROC",
        "",
        "| round | " + " | ".join(sites) + " |",
        "|---:|" + "|".join(["---"] * len(sites)) + "|",
    ]
    for rnd in sorted(nested):
        cells = [
            _overview_num(nested[rnd][s].get("auc_roc")) if s in nested[rnd] else "·"
            for s in sites
        ]
        out.append(f"| {rnd} | " + " | ".join(cells) + " |")

    # -- notes ----------------------------------------------------------------
    out += ["", "**Notes**", ""]
    if any_degenerate:
        out.append(
            "- ⚠ fixed-point confusion is degenerate (no positives predicted at the fixed "
            "threshold — expected for a rare outcome); read the Youden-J cells instead."
        )
    out += [
        "- Under **cyclic** each round trains/evaluates a single site (a `·` marks a round "
        "where that site was not evaluated); **bagging** evaluates both sites every round.",
        "- Cells are `—` when a site's split was single-class that round (AUC undefined).",
        "",
    ]
    return "\n".join(out)


def validate_metrics_artifact(nested) -> None:
    """Structural contract check before the artifact is written.

    Guards the interface the v1.1.e reporting notebook consumes; raises
    ValueError on any shape it cannot rely on. Cheap -- runs once per run.
    """
    if not nested:
        raise ValueError("empty metrics artifact -- no evaluate rounds captured")
    for rnd, sites in nested.items():
        if not isinstance(rnd, int):
            raise ValueError(f"round key {rnd!r} must be int")
        if not sites:
            raise ValueError(f"round {rnd} has no sites")
        for site, metrics in sites.items():
            missing = REQUIRED_METRIC_KEYS - metrics.keys()
            if missing:
                raise ValueError(
                    f"round {rnd} site {site}: missing metric keys {sorted(missing)}"
                )
