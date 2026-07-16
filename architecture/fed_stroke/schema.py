"""fed_stroke: the frozen cross-site data contract.

The column names are defined and derived upstream by the registry-alignment
layer — see `registry_alignement/mappings/`
"""

FEATURE_COLS = ['Age (calc.)', 'NIH on admission']
TARGET_COL = '3M Death'
