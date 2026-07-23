# 1.1.b DP sweep report — skeleton (spec 1.1.a″ R6/R9 reporting template)

> Template for the real-data ε sweep (roadmap 1.1.b) and the 1.1.e write-up. Every 1.1.b
> artifact MUST carry (a) the per-run ε **and** the composed ledger-total ε to date (R6
> reporting rule), and (b) all three comparator arms with **B→C flagged as the privacy-cost
> headline** (R9). Copy, fill, and check every box.

## Run identification

- Date / operator:
- Data provenance: `real-frozen-schema` (Geneva halves; federated 2-node topology, **Geneva-only** — Shenzhen is NOT in the network; spec 1.1.a″ acceptance 4)
- Cohort: deduped one-row-per-patient (R3), keyed-hash split (R4) — identical for ALL arms
  - rows / patients per half after dedup:
  - split: FLAT `test_size=0.2`, `split_seed=`; hash rule `HMAC(SPLIT_KEY‖role‖seed, pid)`
  - missingness: variant 1 sentinel/missing-bin encoding (`REMOVE-IF-NO-DP`), applied at the
    loader for ALL arms — % missing per feature per half:
- Config hash(es) (`dp_ledger` entries):

## Comparator arms (R9 — all on the SAME deduped halves, SAME hash split, matched hyperparameters)

Hyperparameter matching: arm A runs at `subsample = colsample_bytree = 1.0` (the DP learner
forces q = 1.0 and ignores both), `base_score` explicitly configured and identical everywhere.

| Arm | Learner | Mechanism | ε | AUC-ROC (95% CI) | AUC-PR | Brier |
|---|---|---|---|---|---|---|
| A | stock `xgb.train` | — | — | | | |
| B | DP learner | identity (noise off) | ∞ (none spent) | | | |
| C @ ε=10 | DP learner | gaussian σ=6.699 | 10 | | | |
| C @ ε=5 | DP learner | gaussian σ=12.050 | 5 | | | |
| C @ ε=3 | DP learner | gaussian σ=18.888 | 3 | | | |
| C @ ε=1 | DP learner | gaussian σ=51.171 | 1 | | | |

- **B→C = cost of privacy (HEADLINE)** — B and C share everything except the mechanism:
- A→B = learner cost (context only — does the from-scratch DP learner track XGBoost):

## Privacy accounting (R6 — both numbers, always)

| Site | This run's ε (per-run) | Composed ledger-total ε to date | δ |
|---|---|---|---|
| node_A (Geneva half A) | | | 1e-5 |
| node_B (Geneva half B) | | | 1e-5 |

- Ledger file(s): `out/dp_ledger.jsonl` (site-local; totals composed PER SITE, never across)
- `ledger_total()` output pasted verbatim:

## Release boundary (R5)

- [ ] Nothing inside-boundary (validation metrics, confusion matrices, exact counts) appears
      in this artifact if it leaves the project; only the selected model (DP-accounted) and
      figures derived from it cross.
- [ ] DP train replies carried `dp-site-weight` (fixed public constants), never exact counts.

## Go/no-go (spec 1.1.a″ acceptance 5)

- Is the B→C privacy cost acceptable at a usable ε? Decision + rationale:
- Recorded in `docs/logbook.md` on: ______ (BEFORE any Shenzhen integration)
