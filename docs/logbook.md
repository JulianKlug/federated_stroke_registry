# Logbook

- 2026-07-14 — working_example flwr xgboost run on GVA data; 3 rounds, AUC 0.748 / 0.730 / 0.734

- 2026-07-16 — 1.b bagging vs cyclic, matched §3 params + 40-tree budget; final-model AUC on half A / B (last-trained site):
  R1 bagging 0.690 / 0.661; R2 cyclic-fwd 0.701 / 0.663 (last B); R3 cyclic-rev 0.696 / 0.670 (last A).
  Cyclic alternates strictly, R2/R3 end on opposite sites; no gross last-site bias (R2-vs-R3 Δ≤0.008 same half). 18/18 tests pass.

- 2026-07-17 — 1.c site-stratified evaluation harness: per-site AUC-ROC / AUC-PR / Brier + confusion
  at a fixed (0.5) and a per-site Youden-J operating point, bootstrap 95% CIs, a site-preserving
  evaluate aggregator (no cross-site averaging), and a schema-validated JSON artifact per run under
  `out/metrics/`. Final-round per-site AUC-ROC (95% CI) on the seed=42, 20% split:
  R1 bagging — A 0.690 [0.609,0.769], B 0.661 [0.563,0.756] (both sites every round; the two AUCs
  differ ⇒ proof of no averaging). R2 cyclic-fwd — last-site B 0.663 [0.564,0.756]; R3 cyclic-rev —
  last-site A 0.696 [0.611,0.774] (one site per round under cyclic, opposite last sites). AUC-ROC
  reproduces 1.b's eval_set numbers (0.690 / 0.661 / 0.696). Fixed-0.5 confusion is degenerate
  (tp=fp=0) on the 8.9%-positive `3M Death` outcome ⇒ logged as a degenerate-threshold warning;
  Youden-J cells stay informative (R1 A: tp=24, fp=108). Offline `eval_final_model.py` on the saved
  R1 model matches the federated `auc_roc/<site>` to full precision (shared `compute_binary_metrics`).
  All three artifacts are strict-valid JSON (jq), 0 nulls this run. 37/37 tests pass.
