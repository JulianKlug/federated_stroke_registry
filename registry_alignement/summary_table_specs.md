# Summary table specs
## Preprocessing
- Drop duplicates
- Filter to the target cohort (e.g. ischemic stroke only)
- Apply any dataset-specific outcome preprocessing
- Infer variable type: binary, ordinal, continuous, date, or time

## Output
1. Summary table (CSV)
   - Columns: `variable_name`, `summary_stat`, `unit`, `missing`, `variable_overlap`, `reference_variable_name`
   - `summary_stat` by type:
     - binary: n (%)
     - ordinal / continuous: median (Q1-Q3)
     - date: median date (Q1-Q3)
     - time: median time (Q1-Q3) [HH:MM]
   - `unit`: variable unit if applicable
   - `missing`: n_missing/n_total (%)
   - `variable_overlap`: true/false — overlap with reference dataset
   - `reference_variable_name`: matching variable name in the reference dataset, if any

2. Metadata table
   - Number of patients
   - Start year, end year
