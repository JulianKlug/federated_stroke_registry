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

from fed_stroke.metrics import (
    nest_site_metrics,
    render_run_report,
    site_stratified_evaluate_metrics,
    validate_metrics_artifact,
)
from fed_stroke.strategies import OrderedFedXgbCyclic
from fed_stroke.task import replace_keys

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
    if train_method == "bagging":
        return FedXgbBagging(
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
            evaluate_metrics_aggr_fn=site_stratified_evaluate_metrics,
        )
    if train_method == "cyclic":
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

    if context.run_config["save-model"]:
        # Save final model to disk
        bst = xgb.Booster(params=params)
        global_model = bytearray(result.arrays["0"].numpy().tobytes())

        # Load global model into booster
        bst.load_model(global_model)

        # Save model under `model-dir` (an absolute path is CWD-independent;
        # a relative one resolves against the ServerApp working directory).
        model_dir = Path(context.run_config["model-dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "final_model.json"
        print(f"\nSaving final model to {out_path}...")
        bst.save_model(str(out_path))
