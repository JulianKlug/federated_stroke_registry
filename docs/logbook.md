# Logbook

- 2026-07-14 — working_example flwr xgboost run on GVA data; 3 rounds, AUC 0.748 / 0.730 / 0.734

- 2026-07-16 — 1.b bagging vs cyclic, matched §3 params + 40-tree budget; final-model AUC on half A / B (last-trained site):
  R1 bagging 0.690 / 0.661; R2 cyclic-fwd 0.701 / 0.663 (last B); R3 cyclic-rev 0.696 / 0.670 (last A).
  Cyclic alternates strictly, R2/R3 end on opposite sites; no gross last-site bias (R2-vs-R3 Δ≤0.008 same half). 18/18 tests pass.
