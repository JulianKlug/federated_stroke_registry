"""fed_stroke: ClientApp — local training and evaluation on each SuperNode."""

import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from fed_stroke.metrics import compute_binary_metrics
from fed_stroke.task import load_data_gva, replace_keys

warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    # Update trees based on local training data.
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    if train_method == "bagging":
        # Bagging: extract only the newly added trees for server-side merging.
        return bst_input[
            bst_input.num_boosted_rounds()
            - num_local_round : bst_input.num_boosted_rounds()
        ]
    # Cyclic: the server adopts the reply wholesale as the new global model,
    # so return the full updated booster (the whole ensemble built so far).
    return bst_input


def round_seed(params, global_round):
    """params with a per-round XGBoost `seed` (base seed + global round).

    WHY (see out/1d_solution.md): every round rebuilds a fresh
    `xgb.Booster(params=params)` and `load_model`s the global model, because the
    ensemble crosses the network. Each reload RE-SEEDS XGBoost's row/column
    subsampling RNG from `params["seed"]`. A fixed seed therefore makes every
    round draw the SAME subsample — with `colsample_bytree < 1` over few features
    that collapses to one column, so all trees split on a single feature and the
    ensemble underfits (the ~0.10 AUC gap 1.d caught). Advancing the seed per
    round restores the cross-round sampling diversity an in-process
    `xgb.train(num_boost_round=N)` gets for free from its advancing RNG, while
    staying a pure function of the round so cyclic forward/reverse stays
    reproducible. Never mutates the caller's dict.
    """
    return {**params, "seed": int(params.get("seed", 0)) + global_round}


def _train_round(
    params, global_round, num_local_round, train_dmatrix, train_method, global_model
):
    """Run one client training round; return the booster to reply with.

    `global_model`: raw bytes of the current global model, or None on round 1
    (no global model exists yet, so train from scratch). Bagging replies with
    only the newly boosted trees, cyclic with the full grown ensemble — see
    `_local_boost`. Pure (no Message/Context), so the regression test can drive
    the exact per-round sequence `train()` runs.
    """
    params = round_seed(params, global_round)
    if global_model is None:
        # First round: no global model yet — train the local seed trees.
        return xgb.train(params, train_dmatrix, num_boost_round=num_local_round)
    bst = xgb.Booster(params=params)
    bst.load_model(global_model)
    return _local_boost(bst, num_local_round, train_dmatrix, train_method)


@app.query()
def site_info(msg: Message, context: Context) -> Message:
    """Answer OrderedFedXgbCyclic's one-time site query with this node's site name."""
    site = Path(context.node_config["data-path"]).name
    return Message(
        content=RecordDict({"config": ConfigRecord({"site": site})}),
        reply_to=msg,
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    train_dmatrix, _, num_train, _ = load_data_gva(context)

    # Read from run config
    num_local_round = context.run_config["local-epochs"]
    train_method = context.run_config["train-method"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    global_round = msg.content["config"]["server-round"]
    # Round 1 has no global model yet; later rounds continue the received one.
    global_model = None
    if global_round != 1:
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst = _train_round(
        params, global_round, num_local_round, train_dmatrix, train_method, global_model
    )

    # Save model
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Construct reply message
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": num_train,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    _, valid_dmatrix, _, num_val = load_data_gva(context)

    # Load config
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Load global model
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    # Full §4 metric set from raw probabilities (AUC-PR and Brier need
    # probabilities, so compute from predict, not eval_set).
    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    metrics = compute_binary_metrics(
        y_true,
        y_prob,
        context.run_config["operating-point"],
        n_boot=context.run_config["n-boot"],
        boot_seed=context.run_config["boot-seed"],
    )
    # `n` (row count from compute_binary_metrics) IS the weighting quantity;
    # rename it in place to the key flwr's consistency/weighting path expects so
    # the reply carries exactly one row-count value, not a duplicate.
    metrics["num-examples"] = metrics.pop("n")

    # Same site id the site_info query already reports (client_app.py:47).
    site = Path(context.node_config["data-path"]).name

    # Construct and return reply Message. The site string travels in a
    # ConfigRecord (MetricRecord values are numeric-only in flwr); the custom
    # server aggregator reads it to keep per-site metrics stratified.
    content = RecordDict({
        "metrics": MetricRecord(metrics),
        "config": ConfigRecord({"site": site}),
    })
    return Message(content=content, reply_to=msg)
