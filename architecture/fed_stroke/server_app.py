"""fed_stroke: ServerApp — orchestrates FedXgbBagging / FedXgbCyclic across SuperNodes."""

from logging import INFO
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common import log
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

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
        )
    if train_method == "cyclic":
        return OrderedFedXgbCyclic(
            order=run_config["cyclic-order"],
            fraction_train=run_config["fraction-train"],
            fraction_evaluate=run_config["fraction-evaluate"],
            min_available_nodes=run_config["num-sites"],
        )
    raise ValueError(f"Unknown train-method: {train_method}")


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
