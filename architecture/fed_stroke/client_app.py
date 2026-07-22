"""fed_stroke: ClientApp — local training and evaluation on each SuperNode."""

import hashlib
import warnings
from logging import WARNING
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
from flwr.common import log
from flwr.common.config import unflatten_dict

from fed_stroke.dp import (
    FEATURE_RANGES,
    BoostParams,
    DPBooster,
    DPConfig,
    dp_local_boost,
    make_mechanism,
    per_site_tree_budget,
)
from fed_stroke.metrics import compute_binary_metrics
# derive_num_rounds lives in server_app (torch-free flwr symbols only); the client imports it so
# client and server agree on the round count without duplicating the rule (same as hpo.py:20; the
# §7.6 torch-free check imports fed_stroke.dp/strategies, not client_app, so this is not a cycle).
from fed_stroke.server_app import derive_num_rounds
from fed_stroke.task import load_data_arrays, load_data_gva, replace_keys

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


def _site_hash(site: str) -> int:
    """Stable non-negative int from the PUBLIC site name (never data), for the per-site noise
    stream (§3.5). Public metadata only — this is the sole per-site entropy in the RNG seed."""
    return int.from_bytes(hashlib.sha1(site.encode()).digest()[:4], "big")


def _dp_train_round(params, dp, global_round, local_epochs, train_method,
                    total_trees, num_sites, X, y, global_model_bytes, site) -> DPBooster:
    """Pure DP analog of _train_round (no Message/Context, so §4.9 case 3/4 drive it directly).

    Returns the reply DPBooster. `global_model_bytes` is the raw bytes of the current global
    DPBooster, or None on round 1 (round-1 detection is on global_round == 1, never byte-sniffing,
    §3.7). `local_epochs` is BOTH the growth tree count and dp_local_boost's num_local_round (equal
    by construction). The TWO-BoostParams rule (§4.3): the accounting params carry per_site so σ is
    calibrated once to n_rel = 2·D·T_site; the growth params carry local_epochs (trees grown this
    round). The rng is seeded from public (base_seed, global_round, site) ONLY and threaded end to
    end into dp_local_boost -> _grow_trees, so noise is independent per round and per site (§3.5)."""
    assert X.shape[1] == len(FEATURE_RANGES), (
        f"DP noise scale keys off d = len(FEATURE_RANGES) = {len(FEATURE_RANGES)}; got "
        f"X with {X.shape[1]} columns (§3.9 — a frozen-schema growth must fail loud, not "
        f"mis-scale √d noise)."
    )
    num_rounds = derive_num_rounds(train_method, total_trees, num_sites, local_epochs)
    per_site = per_site_tree_budget(train_method, num_rounds, num_sites, local_epochs)
    acct_boost = BoostParams.from_xgb_params(params, num_boost_round=per_site)   # σ calibration
    growth = BoostParams.from_xgb_params(params, num_boost_round=local_epochs)   # trees this round
    mechanism = make_mechanism(dp, acct_boost, X.shape[1])                       # run-calibrated σ
    rng = np.random.default_rng(
        np.random.SeedSequence([int(params.get("seed", 0)), int(global_round), _site_hash(site)])
    )
    global_dp = None if global_round == 1 else DPBooster.from_json_bytes(global_model_bytes)
    booster = dp_local_boost(global_dp, X, y, growth, dp, mechanism, rng,
                             local_epochs, train_method)
    booster.meta["total_trees"] = int(total_trees)   # completes the accounting provenance (§4.1)
    return booster


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
    # Read from run config (data-independent, so build cfg before touching data).
    num_local_round = context.run_config["local-epochs"]
    train_method = context.run_config["train-method"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    global_round = msg.content["config"]["server-round"]
    dp = DPConfig.from_run_config(cfg)

    if dp.enabled:
        # DP branch: swap the whole learner (§4.4). Raw numpy X,y — NOT a DMatrix — and a
        # DPBooster serialized with its own JSON. The Flower transport envelope
        # (ArrayRecord([uint8]) at ["0"]) is identical to the XGB path; nothing else is shared.
        # subsample=1.0 guard: honest q=1.0 accounting (§3.10). The DP learner never samples rows,
        # so this is documentary — warn but do NOT mutate the shared `params` the non-DP arm reuses.
        if float(params.get("subsample", 1.0)) != 1.0:
            log(WARNING, "DP mode: params.subsample=%s ignored; using q=1.0 for honest "
                "accounting (the DP learner never subsamples rows).", params.get("subsample"))
        X_tr, y_tr, _, _, num_train, _ = load_data_arrays(context)
        site = Path(context.node_config["data-path"]).name
        global_model_bytes = None
        if global_round != 1:
            global_model_bytes = bytes(msg.content["arrays"]["0"].numpy().tobytes())
        booster = _dp_train_round(
            params, dp, global_round, num_local_round, train_method,
            context.run_config["total-trees"], context.run_config["num-sites"],
            X_tr, y_tr, global_model_bytes, site,
        )
        model_np = np.frombuffer(booster.to_json_bytes(), dtype=np.uint8)
        model_record = ArrayRecord([model_np])
        metric_record = MetricRecord({"num-examples": num_train})
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

    # ---- non-DP path: byte-identical to pre-change main (§4.9 case 9 / §7.7) ----
    train_dmatrix, _, num_train, _ = load_data_gva(context)

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
    # Load config
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    dp = DPConfig.from_run_config(cfg)

    if dp.enabled:
        # DP branch: deserialize a DPBooster and predict on raw X (§4.4). Scoring goes through
        # the UNCHANGED compute_binary_metrics, so the metric contract matches the XGB branch.
        _, _, X_valid, y_valid, _, num_val = load_data_arrays(context)
        booster = DPBooster.from_json_bytes(bytes(msg.content["arrays"]["0"].numpy().tobytes()))
        y_prob = booster.predict(X_valid)
        y_true = y_valid
    else:
        _, valid_dmatrix, _, num_val = load_data_gva(context)
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
