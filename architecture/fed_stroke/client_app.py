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
    if global_round == 1:
        # First round local training
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # Load global model into booster
        bst.load_model(global_model)

        # Local training
        bst = _local_boost(bst, num_local_round, train_dmatrix, train_method)

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

    # Run evaluation
    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    auc = float(eval_results.split("\t")[1].split(":")[1])

    # Construct and return reply Message
    metrics = {
        "auc": auc,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
