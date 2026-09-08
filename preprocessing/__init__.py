"""Shared GVA preprocessing library.

The single home for preprocessing logic used by BOTH layers:
  - registry_alignement/  (summary tables, cross-site comparison)
  - architecture/preprocessing/  (frozen-schema node tables)

Modules:
  - anonymise:       de-identification of the frozen table (pseudonymous ids, age
                     floor + keyed jitter + top-code) — the last step before a node parquet
  - build_log:       aggregate-only build log (exclusions with reasons, nulled values)
  - case_ids:        raw case_admission_id derivation (registry + EHR)
  - registry_cohort: registry cohort filter, OPSUM outcome reconciliation,
                     timing derivation
  - first_values:    first-after-admission lab/vital extraction from the EHR
  - frozen_table:    wide raw table -> frozen-schema table (unit check, binary
                     encoding, rename, column selection, dtype finalisation,
                     out-of-range -> NaN against the public plausibility ranges)
  - splits:          patient-level disjoint splits
  - mappings:        mapping tables (frozen schema + units, per-site raw->frozen,
                     GVA vocabularies, unit aliases, GVA<->Shenzhen)

Scripts here (prepare_geneva_halves.py) are one-off CLIs, not library code.
"""
from . import mappings  # noqa: F401
from .anonymise import (  # noqa: F401
    ANONYMISATION_SPEC,
    anonymise_age,
    anonymise_frozen,
    keyed_age_jitter,
    pseudonymise_case_ids,
)
from .build_log import (  # noqa: F401
    format_build_log,
    patient_ids,
    record_exclusion,
    write_build_log,
)
from .case_ids import (  # noqa: F401
    RAW_ID_COL,
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
from .frozen_table import (  # noqa: F401
    convert_to_frozen_units,
    encode_binaries,
    finalize_frozen_table,
    format_out_of_range,
    format_unit_check,
    nullify_out_of_range,
    rename_to_frozen,
    select_mapped_columns,
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
