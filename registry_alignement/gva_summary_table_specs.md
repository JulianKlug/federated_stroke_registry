# Summary table specs

Create a script to create a summary table of the database taking as input a database excel file. 

# Preprocessing
- The script should start with dropping duplicates. 
- Then filter to have only ischemic stroke: Type of event == Ischemic stroke
- outcomes should be preprocessed as done in ref 1
- each variable should be checked to see if it is binary, ordinal, continuous, date or time

# Output
1. Summary table, csv format
- Columns: original variable name, summary statistic string, unit, missing, variable_overlap, shenzhen_variable_name
- column details:
-- summary statistic string: depends on variable type
--- binary: n (%)
--- ordinal: median (Q1-Q3)
--- continous: median (Q1-Q3)
--- date: median date (Q1-Q3)
--- time: median time (Q1-Q3)  [HH:MM timestamps]
-- unit: unit of the variable (if exists)
-- missing: n_missing/n_total_patients (%)
-- variable_overlap: true/false if overlap between the two datasets
-- shenzhen_variable_name: equivalent variable name in shenzen dataset if overlap present   

2. meta data table
- number of patients
- start year, end year


# References
1. https://github.com/JulianKlug/OPSUM/blob/main/meta_data/geneva_stroke_unit_patient_characteristics.py

