"""Unit tests for the site-stratified evaluation harness (spec §4.6).

Mostly fast pure-logic. The evaluate() case fits a tiny real booster (reuses the
synthetic-DMatrix idiom from test_client_boost) — still sub-second.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb
from flwr.app import ConfigRecord, Message, MetricRecord, RecordDict

from fed_stroke import client_app
from fed_stroke.metrics import (
    REQUIRED_METRIC_KEYS,
    compute_binary_metrics,
    nest_site_metrics,
    render_run_report,
    site_stratified_evaluate_metrics,
    summarize_final_round,
    validate_metrics_artifact,
)

PARAMS = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3}


def _synthetic_dmatrix(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = (X[:, 0] + rng.rand(n) * 0.1 > 0.5).astype(int)
    return xgb.DMatrix(X, label=y)


# ---- compute_binary_metrics: core ------------------------------------------


def test_core_known_values():
    # Perfectly separated: AUC-ROC = AUC-PR = 1.0; at 0.5 -> (tn,fp,fn,tp)=(2,0,0,2).
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_binary_metrics(y_true, y_prob, n_boot=200)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["auc_pr"] == pytest.approx(1.0)
    # brier = mean((p - y)^2) = (0.01+0.04+0.04+0.01)/4
    assert m["brier"] == pytest.approx(0.025)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)
    assert m["n"] == 4 and m["n_pos"] == 2


def test_all_required_keys_present_minus_num_examples():
    # compute_binary_metrics emits `n`; the client renames it to `num-examples`.
    m = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_boot=50)
    expected = (REQUIRED_METRIC_KEYS - {"num-examples"}) | {"n"}
    assert expected <= m.keys()


def test_non_default_operating_point_shifts_confusion():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_binary_metrics(y_true, y_prob, operating_point=0.85, n_boot=50)
    # only 0.9 >= 0.85 -> (tn,fp,fn,tp)=(2,0,1,1)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 1, 1)


# ---- compute_binary_metrics: single-class ----------------------------------


@pytest.mark.parametrize(
    "y_true,y_prob,fixed",
    [
        # all-negative: at 0.5 the 0.6 prob predicts positive -> (3,1,0,0)
        ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.6], (3, 1, 0, 0)),
        # all-positive: at 0.5 two probs >= 0.5 -> (0,0,2,2)
        ([1, 1, 1, 1], [0.1, 0.4, 0.6, 0.9], (0, 0, 2, 2)),
    ],
)
def test_single_class_nan_aucs_valid_rest(y_true, y_prob, fixed):
    m = compute_binary_metrics(y_true, y_prob, n_boot=100)
    assert math.isnan(m["auc_roc"]) and math.isnan(m["auc_pr"])
    assert math.isnan(m["auc_roc_lo"]) and math.isnan(m["auc_roc_hi"])
    assert math.isnan(m["auc_pr_lo"]) and math.isnan(m["auc_pr_hi"])
    assert math.isnan(m["op_j"])
    # brier is well-defined; fixed confusion valid; _j cells mirror fixed
    assert not math.isnan(m["brier"])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == fixed
    assert (m["tn_j"], m["fp_j"], m["fn_j"], m["tp_j"]) == fixed


def test_single_class_does_not_raise():
    # smoke: no exception on an all-negative split
    compute_binary_metrics([0, 0, 0], [0.2, 0.3, 0.4], n_boot=10)


# ---- compute_binary_metrics: Youden-J point --------------------------------


def test_youden_point_non_degenerate_when_fixed_is_degenerate():
    # positives carry higher probs than negatives, but all < 0.5 -> fixed 0.5 is
    # degenerate (tp=fp=0); Youden separates them.
    y_true = [0, 0, 0, 1, 1]
    y_prob = [0.05, 0.10, 0.15, 0.20, 0.30]
    m = compute_binary_metrics(y_true, y_prob, n_boot=100)
    assert (m["fp"], m["tp"]) == (0, 0)          # fixed 0.5 is degenerate
    assert m["op_j"] != 0.5
    assert m["tp_j"] == 2 and m["fp_j"] == 0     # Youden recovers the positives


def test_youden_point_reproducible():
    y_true = [0, 0, 0, 1, 1]
    y_prob = [0.05, 0.10, 0.15, 0.20, 0.30]
    a = compute_binary_metrics(y_true, y_prob, n_boot=10)
    b = compute_binary_metrics(y_true, y_prob, n_boot=10)
    assert a["op_j"] == b["op_j"]


# ---- compute_binary_metrics: bootstrap CIs ---------------------------------


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.RandomState(1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
    y_prob = np.clip(y_true * 0.4 + rng.rand(len(y_true)) * 0.5, 0, 1)
    m1 = compute_binary_metrics(y_true, y_prob, n_boot=500, boot_seed=7)
    m2 = compute_binary_metrics(y_true, y_prob, n_boot=500, boot_seed=7)
    for key in ("auc_roc", "auc_pr", "brier"):
        lo, hi = m1[f"{key}_lo"], m1[f"{key}_hi"]
        assert lo <= m1[key] <= hi
        # identical seed -> identical CI
        assert (lo, hi) == (m2[f"{key}_lo"], m2[f"{key}_hi"])


def test_bootstrap_ci_nan_when_n_boot_zero():
    m = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_boot=0)
    for key in ("auc_roc", "auc_pr", "brier"):
        assert math.isnan(m[f"{key}_lo"]) and math.isnan(m[f"{key}_hi"])


# ---- site_stratified_evaluate_metrics --------------------------------------


def _reply(site, metrics):
    return RecordDict({
        "metrics": MetricRecord(metrics),
        "config": ConfigRecord({"site": site}),
    })


def test_aggregator_stratifies_without_averaging():
    a = _reply("geneva_half_A.parquet", {"auc_roc": 0.70, "num-examples": 100})
    b = _reply("geneva_half_B.parquet", {"auc_roc": 0.60, "num-examples": 200})
    out = site_stratified_evaluate_metrics([a, b], "num-examples")
    # both sites present, suffixed, values verbatim (NOT weight-averaged)
    assert out["auc_roc/geneva_half_A.parquet"] == pytest.approx(0.70)
    assert out["auc_roc/geneva_half_B.parquet"] == pytest.approx(0.60)
    # num-examples re-tagged per site, not consumed as a weight
    assert out["num-examples/geneva_half_A.parquet"] == 100
    assert out["num-examples/geneva_half_B.parquet"] == 200


def test_aggregator_raises_on_duplicate_site():
    a = _reply("geneva_half_A.parquet", {"auc_roc": 0.70, "num-examples": 100})
    b = _reply("geneva_half_A.parquet", {"auc_roc": 0.65, "num-examples": 100})
    with pytest.raises(ValueError):
        site_stratified_evaluate_metrics([a, b], "num-examples")


# ---- nest_site_metrics round-trip ------------------------------------------


def test_nest_round_trip_nan_to_none_and_int_rounds():
    a = _reply("a.parquet", {"auc_roc": float("nan"), "brier": 0.2})
    b = _reply("b.parquet", {"auc_roc": 0.6, "brier": 0.3})
    agg = site_stratified_evaluate_metrics([a, b], "num-examples")
    nested = nest_site_metrics({3: agg})
    assert set(nested) == {3}
    assert isinstance(next(iter(nested)), int)
    # NaN decodes to None (JSON null); real values pass through
    assert nested[3]["a.parquet"]["auc_roc"] is None
    assert nested[3]["a.parquet"]["brier"] == pytest.approx(0.2)
    assert nested[3]["b.parquet"]["auc_roc"] == pytest.approx(0.6)


# ---- validate_metrics_artifact ---------------------------------------------


def _full_site_entry():
    return {k: (0 if k != "op_j" else 0.4) for k in REQUIRED_METRIC_KEYS}


def test_validate_accepts_well_formed():
    validate_metrics_artifact({1: {"a.parquet": _full_site_entry()}})


def test_validate_rejects_empty():
    with pytest.raises(ValueError):
        validate_metrics_artifact({})


def test_validate_rejects_missing_metric_key():
    entry = _full_site_entry()
    del entry["auc_roc"]
    with pytest.raises(ValueError):
        validate_metrics_artifact({1: {"a.parquet": entry}})


def test_validate_rejects_non_int_round_key():
    with pytest.raises(ValueError):
        validate_metrics_artifact({"1": {"a.parquet": _full_site_entry()}})


def test_validate_rejects_round_with_no_sites():
    with pytest.raises(ValueError):
        validate_metrics_artifact({1: {}})


# ---- summarize_final_round / render_run_report -----------------------------


def _degenerate_entry():
    # rare-outcome shape: fixed 0.5 predicts no positives (fp=tp=0), Youden does
    e = {k: 0 for k in REQUIRED_METRIC_KEYS}
    e.update({"auc_roc": 0.69, "auc_roc_lo": 0.60, "auc_roc_hi": 0.77,
              "auc_pr": 0.19, "brier": 0.088, "op_j": 0.12,
              "tn": 343, "fp": 0, "fn": 39, "tp": 0,
              "tn_j": 235, "fp_j": 108, "fn_j": 15, "tp_j": 24,
              "n_pos": 39, "num-examples": 382})
    return e


def test_summarize_final_round_picks_last_round():
    nested = {1: {"a.parquet": _full_site_entry()},
              20: {"a.parquet": _degenerate_entry(), "b.parquet": _degenerate_entry()}}
    rows = summarize_final_round(nested)
    assert {r["round"] for r in rows} == {20}
    assert [r["site"] for r in rows] == ["a.parquet", "b.parquet"]  # site-sorted


def test_render_run_report_headline_and_trajectory():
    nested = {
        1: {"geneva_half_A.parquet": _degenerate_entry()},
        2: {"geneva_half_B.parquet": _degenerate_entry()},
    }
    md = render_run_report("cyclic_forward", nested, operating_point=0.5)
    assert "`cyclic_forward`" in md            # run named in the title
    assert "Final round (2)" in md             # headline uses the last round
    assert "Per-round AUC-ROC" in md           # trajectory section present
    assert "geneva_half_A.parquet" in md and "geneva_half_B.parquet" in md
    assert "0.5" in md                         # fixed operating point stated
    assert "⚠" in md                           # degenerate fixed matrix flagged
    # cyclic: round 1 has no B, round 2 has no A -> a "·" absence marker appears
    assert "·" in md


def test_render_run_report_handles_single_class_none_values():
    # single-class round: AUC-family decoded to None -> rendered as em dash, no crash
    entry = _degenerate_entry()
    entry.update({"auc_roc": None, "auc_roc_lo": None, "auc_roc_hi": None,
                  "auc_pr": None, "op_j": None})
    md = render_run_report("bagging", {20: {"a.parquet": entry}}, operating_point=0.5)
    assert "—" in md


def test_render_run_report_empty():
    assert "No evaluate rounds captured" in render_run_report("bagging", {}, operating_point=0.5)


# ---- client_app.evaluate() wiring (light integration) ----------------------


def test_evaluate_returns_full_metrics_and_site(monkeypatch):
    valid_dm = _synthetic_dmatrix(n=60, seed=3)
    num_val = valid_dm.num_row()

    # No real GVA parquet: monkeypatch the loader to hand back the synthetic split.
    monkeypatch.setattr(
        client_app, "load_data_gva",
        lambda ctx: (None, valid_dm, None, num_val),
    )

    # A tiny booster serialized into the msg's arrays["0"], as train() would.
    bst = xgb.train(PARAMS, valid_dm, num_boost_round=3)
    model_np = np.frombuffer(bst.save_raw("json"), dtype=np.uint8)
    msg = Message(
        content=RecordDict({"arrays": client_app.ArrayRecord([model_np])}),
        dst_node_id=1,
        message_type="evaluate",
    )

    context = SimpleNamespace(
        run_config={
            "params.objective": "binary:logistic",
            "params.max-depth": 2,
            "operating-point": 0.5,
            "n-boot": 50,
            "boot-seed": 0,
        },
        node_config={"data-path": "/data/geneva_half_A.parquet"},
    )

    reply = client_app.evaluate(msg, context)

    mr = next(iter(reply.content.metric_records.values()))
    assert REQUIRED_METRIC_KEYS <= mr.keys()   # incl. num-examples
    assert "n" not in mr                        # renamed, no duplicate bare `n`
    assert mr["num-examples"] == num_val
    cr = next(iter(reply.content.config_records.values()))
    assert cr["site"] == "geneva_half_A.parquet"
