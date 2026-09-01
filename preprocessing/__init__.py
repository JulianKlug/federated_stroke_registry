"""Shared GVA preprocessing library.

The single home for preprocessing logic used by BOTH layers:
  - registry_alignement/  (summary tables, cross-site comparison)
  - architecture/preprocessing/  (frozen-schema node tables)

Modules:
  - case_ids:        case_admission_id derivation (registry + EHR)
  - registry_cohort: registry cohort filter, OPSUM outcome reconciliation,
                     timing derivation
  - first_values:    first-after-admission lab/vital extraction from the EHR
  - splits:          patient-level disjoint splits
  - mappings:        mapping tables (frozen schema, GVA<->Shenzhen, units)

Scripts here (prepare_geneva_halves.py) are one-off CLIs, not library code.
"""
from . import mappings  # noqa: F401
from .case_ids import (  # noqa: F401
    build_case_admission_id,
    create_ehr_case_identification_column,
    create_registry_case_identification_column,
)
from .first_values import (  # noqa: F401
    assemble_wide,
    extract_lab_dosage_first_values,
    extract_pv_lab_first_values,
    extract_pv_vital_first_values,
    load_concat_csvs,
)
from .registry_cohort import (  # noqa: F401
    build_cohort,
    compute_timings,
    parse_yyyymmdd,
    preprocess,
    preprocess_features,
    preprocess_outcome,
)
from .splits import stratified_patient_split  # noqa: F401
