"""fed_stroke: ServerApp — orchestrates FedXgbBagging / FedXgbCyclic across SuperNodes."""

import json
from logging import INFO, WARNING
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common import log
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from fed_stroke.dp import DPBooster, DPConfig
from fed_stroke.dp import ledger as dp_ledger
from fed_stroke.metrics import (
    nest_site_metrics,
    render_run_report,
    site_stratified_evaluate_metrics,
    validate_metrics_artifact,
)
from fed_stroke.strategies import DPFedXgbBagging, OrderedFedXgbCyclic
from fed_stroke.task import replace_keys

# Loopback-only: where the two local SuperNodes (pyproject [tool.fed_stroke.nodes]
# dp-ledger-path, repo-root-relative) write their shared ledger.
LOOPBACK_LEDGER_PATH = "out/dp_ledger.jsonl"

# Create ServerApp
app = ServerApp()


def derive_num_rounds(
    train_method: str, total_trees: int, num_sites: int, local_epochs: int
) -> int:
    """Compute the round count that yields exactly `total_trees` trees.

    The single `total-trees` budget is the source of truth; the round count is
    derived per strategy so a mismatched run (e.g. a cyclic run at the bagging
    default ending at half the trees) is impossible.
    """
    if train_method == "bagging":
        trees_per_round = num_sites * local_epochs
    elif train_method == "cyclic":
        trees_per_round = local_epochs
    else:
        raise ValueError(f"Unknown train-method: {train_method}")
    if total_trees % trees_per_round:
        raise ValueError(
            f"total-trees={total_trees} is not divisible by "
            f"{trees_per_round} trees/round ({train_method})"
        )
    return total_trees // trees_per_round


def build_strategy(run_config):
    """Select the aggregation strategy from run config.

    fraction values pass through from config for BOTH strategies: the config must
    never silently lie. flwr's own FedXgbCyclic constructor raises on any
    fraction other than 0.0/1.0 (1.0 is the only useful one).
    """
    train_method = run_config["train-method"]
    dp = DPConfig.from_run_config(replace_keys(unflatten_dict(run_config)))
    if train_method == "bagging":
        # DP bagging concatenates DPBooster trees (§4.6); non-DP keeps flwr's stock strategy so
        # dp.enabled=false is byte-identical.
        bagging_cls = DPFedXgbBagging if dp.enabled else FedXgbBagging
        return bagging_cls(
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
            evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics,
        )
    if train_method == "cyclic":
        # Cyclic reuses OrderedFedXgbCyclic unchanged regardless of dp.enabled: FedXgbCyclic adopts
        # reply[0] bytes wholesale (format-agnostic) and the DP client returns the full ensemble.
        return OrderedFedXgbCyclic(
            order=run_config["cyclic-order"],
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
            evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics,
        )
    raise ValueError(f"Unknown train-method: {train_method}")


def _fmt(value, spec=".4f"):
    """Format an artifact value for the log table (None <- NaN -> 'n/a')."""
    return "n/a" if value is None else format(value, spec)


def log_final_round_table(nested: dict) -> None:
    """Log the final round's per-site headline metrics so the run is legible
    without opening the JSON.

    Flags an all-zero fixed-point confusion matrix as a degenerate-threshold
    warning (§7): 0.5 on a rare outcome yields tp≈fp≈0, so the Youden-J cells
    are the informative pair there.
    """
    if not nested:
        return
    final_round = max(nested)
    log(INFO, "Final round (%s) per-site metrics:", final_round)
    log(
        INFO,
        "  %-26s %8s %8s %8s %-19s %-19s %6s %6s",
        "site", "auc_roc", "auc_pr", "brier",
        "fixed(tn,fp,fn,tp)", "youden(tn,fp,fn,tp)", "n_pos", "n",
    )
    for site, m in sorted(nested[final_round].items()):
        auc_roc = _fmt(m.get("auc_roc"))
        ci = f"[{_fmt(m.get('auc_roc_lo'))},{_fmt(m.get('auc_roc_hi'))}]"
        fixed = f"({m['tn']},{m['fp']},{m['fn']},{m['tp']})"
        youden = f"({m['tn_j']},{m['fp_j']},{m['fn_j']},{m['tp_j']})"
        log(
            INFO,
            "  %-26s %8s %8s %8s %-19s %-19s %6s %6s  ci=%s",
            site, auc_roc, _fmt(m.get("auc_pr")), _fmt(m.get("brier")),
            fixed, youden, m["n_pos"], m["num-examples"], ci,
        )
        if m["fp"] == 0 and m["tp"] == 0:
            log(
                WARNING,
                "  degenerate fixed-point confusion for %s: no positives "
                "predicted at the fixed threshold (rare outcome); read the "
                "Youden-J cells instead.",
                site,
            )


def save_final_model(bst, model_dir, params, total_trees):
    """Stamp the resolved run config into the model, then save it to disk.

    Embedding `fed_run_config` (the exact params + tree budget this model was
    trained under, including any `--run-config` override) turns 1.d's "matched
    budget" from an assumption into a checked invariant: `check_fed_vs_pooled.py`
    reads params + budget back from the model itself, not a re-read of
    `pyproject.toml`. The attribute rides inside the model JSON and survives
    `save_model`/`load_model` and any `mv` rename.

    Dispatch on the object type (Decision 9), NOT a new argument, so the signature and the non-DP
    save path stay byte-identical. A DPBooster has no set_attr/save_model (those are XGB-only), so
    its provenance rides in the `meta` block: fold `fed_run_config` in, then write to_json_bytes().
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "final_model.json"
    print(f"\nSaving final model to {out_path}...")
    if isinstance(bst, DPBooster):
        bst.meta = dict(bst.meta or {})
        bst.meta["fed_run_config"] = {"params": params, "total_trees": total_trees}
        out_path.write_bytes(bst.to_json_bytes())
        return out_path
    bst.set_attr(fed_run_config=json.dumps({
        "params": params,
        "total_trees": total_trees,
    }))
    bst.save_model(str(out_path))
    return out_path


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    train_method = context.run_config["train-method"]
    num_rounds = derive_num_rounds(
        train_method,
        context.run_config["total-trees"],
        context.run_config["num-sites"],
        context.run_config["local-epochs"],
    )
    log(
        INFO,
        "train-method=%s: total-trees=%s → num-server-rounds=%s",
        train_method,
        context.run_config["total-trees"],
        num_rounds,
    )
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    dp = DPConfig.from_run_config(cfg)

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize the selected strategy
    strategy = build_strategy(context.run_config)

    # Start strategy for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Persist the site-stratified metrics artifact (§4.3). All decode/validate
    # steps are library calls into metrics.py; main() only orchestrates + writes.
    nested = nest_site_metrics(result.evaluate_metrics_clientapp)  # {round:{site:{metric}}}, NaN->None
    validate_metrics_artifact(nested)                              # contract check, raises on bad shape
    log_final_round_table(nested)                                  # headline numbers into the server log

    metrics_dir = Path(context.run_config["metrics-dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag = train_method + (
        f"_{context.run_config['cyclic-order']}" if train_method == "cyclic" else ""
    )
    metrics_path = metrics_dir / f"{tag}.json"
    # allow_nan=False: NaNs are already None; any stray NaN must raise, never
    # write the invalid bare `NaN` token that jq / JS / strict parsers reject.
    metrics_path.write_text(json.dumps(nested, indent=2, allow_nan=False))
    log(INFO, "Wrote site-stratified metrics artifact to %s", metrics_path)

    # Write a human-readable per-run report beside the JSON. Convenience only —
    # the JSON stays the source of truth — so a failure here must not sink a
    # finished run.
    try:
        report_path = metrics_dir / f"{tag}.md"
        report_path.write_text(
            render_run_report(
                tag, nested, operating_point=context.run_config["operating-point"]
            )
        )
        log(INFO, "Wrote per-run metrics report to %s", report_path)
    except Exception as exc:  # noqa: BLE001 - report is best-effort
        log(WARNING, "Could not write per-run metrics report: %s", exc)

    if dp.enabled:
        # DP: rebuild the merged DPBooster (restores .meta, §4.1) and log the accounting the run
        # operated at. SYNTHETIC/EXAMPLE data only — this is NOT a real-ε claim (Decision 6, §8);
        # the real-Geneva ε sweep is 1.1.b behind the independent DP-accountant review.
        dp_bst = DPBooster.from_json_bytes(bytes(result.arrays["0"].numpy().tobytes()))
        m = dp_bst.meta or {}
        if dp.mechanism == "identity":
            # R9 comparator arm B: the DP learner with noise OFF — a utility reference inside
            # the trust boundary, NOT a DP release. No ε is spent; nothing enters the ledger.
            log(
                INFO,
                "DP-learner run, IDENTITY mechanism (R9 comparator arm B): no noise added, "
                "no ε spent, not a DP release. num_releases=%s",
                m.get("num_releases"),
            )
        else:
            log(
                INFO,
                "DP run (synthetic/example data — NOT a real-ε claim): mechanism=%s "
                "reported_epsilon=%s noise_multiplier=%s num_releases=%s (=2·D·per_site) "
                "per_site_trees=%s delta=%s",
                dp.mechanism, m.get("reported_epsilon"), m.get("noise_multiplier"),
                m.get("num_releases"), m.get("per_site_trees"), dp.delta,
            )
            # R6 reporting rule: every artifact reporting a per-run ε also states the composed
            # ledger-total ε to date. The ledger path is NODE-owned (node_config, reviewer A
            # C1) — the ServerApp has no node_config, so this is a loopback-only convenience
            # read at the committed default (the nodes' pyproject dp-ledger-path, resolved
            # against the shared repo-root CWD). The authoritative composition is the sweep
            # driver's, which reads the nodes' declared path.
            totals = dp_ledger.ledger_total(LOOPBACK_LEDGER_PATH, dp.delta)
            if totals:
                log(INFO, "Composed ledger-total ε to date (R6, per site, δ=%s): %s",
                    dp.delta, totals)

    if context.run_config["save-model"]:
        # Rebuild the final global booster from the aggregated arrays.
        if dp.enabled:
            bst = DPBooster.from_json_bytes(bytes(result.arrays["0"].numpy().tobytes()))
        else:
            bst = xgb.Booster(params=params)
            global_model = bytearray(result.arrays["0"].numpy().tobytes())
            bst.load_model(global_model)

        # Stamp the resolved run config into the model and save it. `model-dir`
        # as an absolute path is CWD-independent; a relative one resolves against
        # the ServerApp working directory.
        save_final_model(
            bst,
            Path(context.run_config["model-dir"]),
            params,
            context.run_config["total-trees"],
        )
