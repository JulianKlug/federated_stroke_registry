"""Three-arm DP plug-point sanity harness on single-site synthetic (roadmap 1.1 prereq).

Scores three arms through the SAME `metrics.compute_binary_metrics` on the SAME synthetic split
so the DP-vs-classic gap DECOMPOSES:

  A  classic stock `xgb.train`      -> the anchor (base_score=0.5, nthread=1 pinned)
  B  our NumPy learner, noise OFF   -> A→B isolates "our simpler learner vs stock XGBoost"
  C  our learner, DP ON at each ε   -> B→C isolates "cost of privacy"

The same A/B/C decomposition is what 1.1.b/1.1.d/1.1.e will run on real Geneva as the DP-vs-classic
head-to-head. This script only reads/prints — no file writes.

NOTE (scale): default --n=1000 ≈ GVA per-node scale (cohorts are disbalanced: GVA >2000 total split
~1000/node, Shenzhen ~40000). At this n the DP arm shows a clean monotone ε→AUC erosion; at n≈380 the
honest 2·D·T-release accounting drives ε≤5 to chance (a real DP-utility-cost finding, not a bug).

Usage:
    uv run python scripts/dp_prototype_demo.py --epsilons 1 5 30 100 1000
    uv run python scripts/dp_prototype_demo.py --n 40000 --epsilons 1 3 5   # Shenzhen scale
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

# At extreme ε the optimal RDP order sits on the grid boundary; the accountant warns (honest, but
# noisy for a demo). The gate (tests/dp/test_accounting.py) validates the accountant separately.
warnings.filterwarnings("ignore", message="Optimal RDP order.*")

# Make `fed_stroke` importable regardless of CWD (script lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fed_stroke.dp import (  # noqa: E402
    BoostParams,
    DPConfig,
    num_gaussian_releases,
    num_histogram_queries,
    train_dp_gbdt,
)
from fed_stroke.dp.synthetic import (  # noqa: E402
    make_synthetic_site,
    synthetic_train_valid,
)
from fed_stroke.metrics import compute_binary_metrics  # noqa: E402


def _row(arm, epsilon, noise_mult, levels, releases, m):
    """One fixed-width table row. `epsilon`/`noise_mult` may be None -> '—'."""
    eps = "—" if epsilon is None else f"{epsilon:.4g}"
    nm = "—" if noise_mult is None else f"{noise_mult:.3f}"
    return (
        f"{arm:<16} {eps:>7} {nm:>10} {levels:>6} {releases:>8} "
        f"{m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} {m['brier']:>8.4f} "
        f"{m['n']:>6} {m['n_pos']:>6}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000,
                        help="Synthetic site size (default 1000 ≈ GVA per-node)")
    parser.add_argument("--seed", type=int, default=0, help="Generator + noise seed")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 5, 30, 100, 1000],
                        help="Target ε values for the DP arm (default spans the noise-floor "
                             "transition so the erosion is visible)")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20, help="num_boost_round")
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--clip-bound", type=float, default=1.0)
    parser.add_argument("--max-bins", type=int, default=32)
    parser.add_argument("--mechanism", choices=["gaussian", "laplace"], default="gaussian")
    # Scoring knobs mirror pyproject config so numbers line up with the FL run.
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()

    X, y = make_synthetic_site(n=args.n, seed=args.seed)
    X_tr, X_va, y_tr, y_va = synthetic_train_valid(X, y, seed=42)

    boost = BoostParams(max_depth=args.max_depth, num_boost_round=args.rounds, eta=args.eta,
                        min_child_weight=5.0, base_score=0.5, seed=args.seed)
    levels = num_histogram_queries(boost)        # = D×T
    releases = num_gaussian_releases(boost)       # = 2×D×T

    def score(y_prob):
        return compute_binary_metrics(y_va, y_prob, operating_point=0.5,
                                      n_boot=args.n_boot, boot_seed=args.boot_seed)

    print(f"DP plug-point synthetic sanity — n={args.n}, seed={args.seed}, "
          f"max_depth={args.max_depth}, rounds={args.rounds}, mechanism={args.mechanism}")
    print(f"levels (D×T) = {levels}   gaussian releases (2×D×T) = {releases}   "
          f"δ=1e-5, q=1.0")
    print("Banner: the DP arm uses FIXED public-range bins (Age 0–120, NIHSS 0–42; NOT data "
          "quantiles) and forces subsample=1.0 (honest ε upper bound).")
    print()

    header = (
        f"{'arm':<16} {'epsilon':>7} {'noise_mult':>10} {'levels':>6} {'releases':>8} "
        f"{'auc_roc':>8} {'auc_pr':>8} {'brier':>8} {'n':>6} {'n_pos':>6}"
    )
    print(header)
    print("-" * len(header))

    # Arm A — classic anchor. base_score=0.5 + nthread=1 pinned so XGBoost's auto-base_score and
    # multi-thread nondeterminism do not pollute the A→B gap (F5).
    xgb_params = {
        "objective": "binary:logistic", "eta": args.eta, "max_depth": args.max_depth,
        "min_child_weight": 5, "tree_method": "hist", "subsample": 1.0,
        "base_score": 0.5, "nthread": 1, "seed": args.seed,
    }
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    bst = xgb.train(xgb_params, dtr, num_boost_round=args.rounds)
    print(_row("A classic-xgb", None, None, levels, releases, score(bst.predict(dva))))

    # Arm B — our learner, noise OFF (ε = ∞).
    booster_b = train_dp_gbdt(X_tr, y_tr, boost, DPConfig(enabled=False))
    print(_row("B numpy-nonoise", None, None, levels, releases, score(booster_b.predict(X_va))))

    # Arm C — our learner, DP ON at each ε. The injected rng makes this demo REPRODUCIBLE and
    # therefore an INSECURE-TEST path (R2): production DP runs never seed noise (OS entropy);
    # deterministic noise is injection-only. Synthetic data only — never point this at real data.
    for eps in args.epsilons:
        dp = DPConfig(enabled=True, mechanism=args.mechanism, target_epsilon=float(eps),
                      clip_bound=args.clip_bound, max_bins=args.max_bins)
        booster_c = train_dp_gbdt(X_tr, y_tr, boost, dp, rng=np.random.default_rng(args.seed))
        mech = booster_c.mechanism
        nm = None if np.isnan(mech.noise_multiplier) or np.isinf(mech.noise_multiplier) \
            else mech.noise_multiplier
        print(_row(f"C dp-{args.mechanism}", mech.reported_epsilon, nm,
                   levels, releases, score(booster_c.predict(X_va))))

    print()
    print("Read: A ≈ B confirms our no-noise learner tracks stock XGBoost (A→B = learner cost); "
          "AUC eroding as ε shrinks is B→C = cost of privacy.")


if __name__ == "__main__":
    main()
