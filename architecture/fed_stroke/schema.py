"""fed_stroke: the frozen cross-site data contract.

The column names are defined and derived upstream by the registry-alignment
layer — see `registry_alignement/mappings/`
"""

FEATURE_COLS = ['Age (calc.)', 'NIH on admission']
TARGET_COL = '3M Death'

# --- Missingness policy: variant 1, sentinel / missing bin (decided 2026-07-23) --------------
# REMOVE-IF-NO-DP: this constant and every use of it (grep for REMOVE-IF-NO-DP) exist ONLY
# because the DP learner's fixed public bins cannot represent NaN and the R7 precondition gate
# refuses non-finite features. Missing feature values are encoded at the loader as this PUBLIC
# out-of-range sentinel, and fixed_bin_edges reserves bin 0 for it, so "not recorded" is an
# explicit ordinary value that ALL comparator arms (stock XGBoost included) see identically —
# a DP-arm-only policy would confound the A→B→C decomposition with a missingness-encoding
# difference. If the project later moves forward WITHOUT DP, remove the sentinel encoding and
# the reserved bin, and let XGBoost's native NaN handling (learned per-split default
# direction) take over.
# Must stay strictly below every FEATURE_RANGES lower bound (fixed_bin_edges asserts this).
MISSING_SENTINEL = -1.0
