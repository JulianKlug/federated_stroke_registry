# Spec 1.3.0 — Shenzhen preprocessing hand-off

Prepares the preprocessing-track item that gates Phase v1.3
([architecture/roadmap.md](../../architecture/roadmap.md)):

> Shenzhen preprocessing: same schema, run remotely by partner. Needs partner
> sign-off on the frozen schema before code moves. Completion (with the
> smoke-test artifact and divergence gates below) unblocks v0.b and Phase v1.3.

The partner's first deliverable is `preprocess_shenzhen.py`: their raw export →
the frozen-schema node parquet, built and run entirely on Shenzhen
infrastructure. This spec enumerates what must exist on **our** side before that
file can be written, and what the partner must answer before the mapping tables
can be completed.

Grounded in the shipped Geneva path: [preprocessing/mappings/frozen_schema.py](../../preprocessing/mappings/frozen_schema.py)
(the contract), [preprocessing/frozen_table.py](../../preprocessing/frozen_table.py)
(the site-agnostic steps), [architecture/preprocessing/preprocess_gva.py](../../architecture/preprocessing/preprocess_gva.py)
(the reference site binding), [architecture/fed_stroke/schema.py](../../architecture/fed_stroke/schema.py)
(the loader-side mirror).

This task ships the spec, the partner questionnaire
([docs/handoff/shenzhen_preprocessing_questionnaire.md](../handoff/shenzhen_preprocessing_questionnaire.md)),
the implementation checklist
([docs/handoff/shenzhen_implementation_checklist.md](../handoff/shenzhen_implementation_checklist.md)),
W1–W2 below, and the two skeletons (W3, W7). **No mapping table is completed
here** — five of the holes below are questions only Shenzhen can answer.

## 1. Goal & scope

Make `preprocess_shenzhen.py` a **site-binding exercise, not a contract
exercise**. The partner should write the reader for their export, fill in the
vocabularies of their own data, and call the same seven shared functions Geneva
calls — never re-derive the schema, the plausibility ranges, the
de-identification, the parquet stamp or the smoke report.

**In scope**

- The **gap list** (§4): every contract hole that makes a conformant Shenzhen
  table impossible to write today.
- The **shippable bundle** definition (§5): which modules travel, which stay,
  and the one refactor needed to make the node-parquet writer site-agnostic.
- A **`preprocess_shenzhen.py` skeleton** + synthetic fixture + contract test
  suite, so the partner can prove conformance without our data and we can review
  without theirs (§6, W7–W9).
- The **sign-off artifact** `registry_alignement/FROZEN_FEATURES.md` (§6, W10) —
  the roadmap's unchecked "publish the frozen feature set" item.
- The **smoke-report exchange + divergence-gate comparator** (§6, W12–W13).

**Out of scope**

- SuperNode packaging / Docker for Shenzhen → roadmap **1.3.a**.
- mTLS onboarding of their node → roadmap **1.3.b**.
- The DP accountant's Shenzhen addendum review → owed separately
  ([docs/logbook.md](../logbook.md) 2026-08-30, scope line "Geneva-only").
- Any change to `FROZEN_FEATURES`, `FROZEN_RANGES` or `SCHEMA_VERSION` — those
  are contract changes requiring the freeze discipline in §8.

## 2. The contract `preprocess_shenzhen.py` must satisfy

Identical to Geneva's, restated so the partner document can quote it:

- **One row per cohort admission.** Every admission, whatever its outcome. No
  dedup to one row per patient, no train/valid split, no label drop — all three
  are loader-side (`fed_stroke/task.py` R3/R4) and accountant-reviewed there.
- **Columns EXACTLY** `[ID_COL, *FROZEN_FEATURES, *FROZEN_OUTCOMES]`, in that
  order: 1 + 41 + 3 = 45. Column order is part of the contract — the loader
  emits `X` in `FEATURE_COLS` order and the DP bin grid is indexed by position.
- **Values in `FROZEN_UNITS`.** Binaries encoded `{0,1}` (`sex: 1 = female`).
- **Missing stays NaN.** The loader applies `MISSING_SENTINEL = -1.0`; the
  preprocessing never encodes it.
- **Out-of-range → NaN, counted.** `FROZEN_RANGES` is the public DP bin grid; a
  feature out of range wholesale fails the build (unit or mapping bug).
- **Outcomes are columns, never labels.** NaN when not recorded, no row dropped.
  `fed_stroke.schema.TARGET_COL` picks the label at training time.
- **De-identified before the write.** `ID_COL = '<patient pseudonym>_<admission
  ordinal>'`; the loader derives the patient key as `id.split('_')[0]`
  (`task.py:188`) for R3 dedup and the R4 adjacency-stable split, so an id
  without exactly this shape silently breaks patient-level disjointness.
- **Two artifacts out, one of them crosses the border**: the node parquet
  (stays at the site, path referenced by their SuperNode's `data-path`) and
  `<stem>_smoke_report.json` (aggregate-only; the only file that travels).
  A `<stem>.build.log` stays local.

## 3. The seven shared steps

`preprocess_shenzhen.wide_to_frozen` binds Shenzhen tables to the same
site-agnostic functions, in the same order
([preprocessing/frozen_table.py](../../preprocessing/frozen_table.py)):

```
convert_to_frozen_units(SHENZHEN_TO_FROZEN, FROZEN_UNITS, SHENZHEN_DECLARED_UNITS, UNIT_ALIASES)
  → encode_binaries(SHENZHEN_BINARY_ENCODINGS, fillna=SHENZHEN_BINARY_FILLNA)
  → rename_to_frozen(SHENZHEN_TO_FROZEN)
  → select_mapped_columns(SHENZHEN_TO_FROZEN, id_col=RAW_ID_COL)
  → validate_frozen_columns(FROZEN_FEATURES, FROZEN_OUTCOMES, id_col=RAW_ID_COL)
  → finalize_frozen_table(..., outcome_ranges=FROZEN_OUTCOME_RANGES)
  → nullify_out_of_range(FROZEN_RANGES, FROZEN_FEATURES)
then anonymise_frozen(key) → write_node_parquet(df, out_path, source_files)
```

Everything the partner writes sits **before** step 1 (read the export, derive
timings, build the cohort, construct the raw id) and **inside** the four
Shenzhen mapping tables. Nothing else.

## 4. Gap analysis — what blocks the first line of code

### H1 — `SHENZHEN_TO_FROZEN` is incomplete (blocking)

[frozen_schema.py:340](../../preprocessing/mappings/frozen_schema.py) carries five
`TODO(shenzhen sign-off)` entries: `IVT`, `EVT`, `mrs_3m`, `death_3m`,
`death_in_hospital`. The import-time assertion that Geneva passes —
`set(GVA_TO_FROZEN.values()) == {*FROZEN_FEATURES, *FROZEN_OUTCOMES}` — has no
Shenzhen twin precisely because the dict is incomplete, so nothing fails today
and the hole is invisible until their build runs.

**No outcome source = no label = no training.** `TARGET_COL = "death_3m"`; if
Shenzhen has no 3-month follow-up, either the label changes for both sites or
the pilot's headline changes. This is the single highest-risk unknown and must
be answered first (questionnaire Q1).

### H2 — the four timing features are mapped as if they were raw columns (blocking)

`SHENZHEN_TO_FROZEN` maps `"ODT" -> "ODT"`, `"ONT" -> "ONT"`, `"DNT" -> "DNT"`,
`"OPT" -> "OPT"`. Geneva does not have these as export columns either — they are
derived in `preprocessing/registry_cohort.py` from onset/arrival/needle/puncture
timestamps. The Shenzhen variable inventory
([gva_to_shenzen.py](../../preprocessing/mappings/gva_to_shenzen.py)) lists
`Exact Onset Time`, `Estimated Onset Time`, `Last Known Well Time`,
`Arrival Time at Hospital` — i.e. the ingredients, not the results.

Needs a written derivation spec: which timestamp per interval, what to do when
onset is estimated vs exact, the wake-up rule (Geneva uses last-known-well),
and the unknown-onset policy (NaN, not 0). `wake_up_stroke` has the same shape:
Geneva derives it, Shenzhen has `Is Wake-Up Stroke`.

### H3 — no `shenzhen_encodings.py` (blocking)

Geneva has [gva_encodings.py](../../preprocessing/mappings/gva_encodings.py):
binary vocabularies, a fill-na table, pre-encoded binaries, declared units.
`encode_binaries` **raises on any unlisted value** — by design, never a silent
NaN — so all 16 binary columns fail without the sibling. Needed:

| table | what it holds |
|---|---|
| `SHENZHEN_BINARY_ENCODINGS` | raw value → `{0,1}` per binary column, incl. binary outcomes |
| `SHENZHEN_BINARY_FILLNA` | columns where blank means `0`, not unknown |
| `SHENZHEN_PRE_ENCODED_BINARIES` | columns already `{0,1}` in the export |
| `SHENZHEN_DECLARED_UNITS` | unit per numeric column lacking a per-row unit label |

Two encoding conventions must be reproduced exactly, not re-chosen:
`sex: 1 = female`, and "thrombolysis started before admission (drip-and-ship)
counts as IVT" (Geneva decision, 2026-09-02).

### H4 — lab units will fail loudly on plausible Shenzhen conventions (blocking)

[unit_aliases.py](../../preprocessing/mappings/unit_aliases.py) is
metric-prefix algebra and spelling synonyms **only**; mass↔molar is deliberately
inexpressible and `convert_to_frozen_units` raises when it meets one. Chinese
lab reporting plausibly diverges on at least:

| frozen column | frozen unit | plausible Shenzhen unit | conversion |
|---|---|---|---|
| `glucose` | mmol/L | mg/dL | molar, ×0.0555 |
| `creatinine` | µmol/l | mg/dL | molar, ×88.4 |
| `LDL` | mmol/l | mg/dL | molar, ×0.02586 |
| `urea` | mmol/l | mg/dL (BUN) | molar, ×0.357 |
| `d_dimer` | ng/ml | mg/L FEU or µg/mL DDU | **assay-dependent, ×1000 and/or ÷2** |
| `CRP` | mg/l | mg/dL | prefix, ×10 (already expressible) |
| `white_blood_cell_count` | G/l | ×10⁹/L, ×10³/µL | spelling synonyms, ×1 |

`d_dimer` is the dangerous one: FEU↔DDU is a factor-of-2 assay convention, not
arithmetic, and a wrong choice lands inside the plausibility range — no error,
silently divergent feature. Needs a per-column override table
(`SUBSTANCE_CONVERSIONS: {frozen_column: {label: factor}}`) plus an explicit
FEU/DDU declaration from the partner.

### H5 — missing-feature policy undecided (blocking)

The contract demands all 41 feature columns. If Shenzhen records no respiratory
rate, no fibrinogen, no pre-stroke mRS, what is legal?

**Proposed D3**: an all-NaN column is legal and must be declared up front.
Verify `finalize_frozen_table` (float64 cast on an empty column),
`nullify_out_of_range` (no recorded values → no error) and `smoke_report`
(median of an empty series → must not emit NaN into JSON; `write_node_parquet`
uses `allow_nan=False`) all survive it — **this is untested today**. A declared-
absent feature is then exempt from the missing-rate divergence gate and listed in
the v1.3 write-up as a site-asymmetric feature.

### H6 — the cohort definition exists only as Geneva code (blocking)

`registry_cohort.build_cohort` is the Geneva registry's filter (dedup + ischemic
stroke, with the OPSUM outcome reconciliation). The partner cannot read Geneva
code as a specification of *their* cohort. Needs the criteria in prose:
diagnosis inclusion, index-admission rule, multi-admission handling, date
window, and every exclusion Geneva applies with its reason — so their build log
has comparable stages.

### H7 — the node-parquet writer lived inside a Geneva-named script (RESOLVED, W1)

`smoke_report`, `write_node_parquet`, `read_node_metadata`, `hash_inputs`,
`sha256_of`, `METADATA_PREFIX`, `SMOKE_QUANTILES`, `PROVENANCE` sit in
[architecture/preprocessing/preprocess_gva.py:354-570](../../architecture/preprocessing/preprocess_gva.py).
Their content is already site-agnostic (they read `FROZEN_FEATURES` /
`FROZEN_OUTCOMES` / `ID_COL` / `SCHEMA_VERSION`). Shipping as-is forces the
partner to import from a file named `preprocess_gva` — or, far likelier,
reimplement the stamp and the smoke report and drift from ours, defeating the
cross-site comparison the report exists for. Moved to
[preprocessing/node_parquet.py](../../preprocessing/node_parquet.py); `preprocess_gva`
imports them back, `hash_inputs` stays Geneva-specific.

### H8 — `frozen_table` imported `RAW_ID_COL` from Geneva-specific `case_ids` (RESOLVED, W2)

`preprocessing/frozen_table.py:16` takes its default `id_col` from
`preprocessing/case_ids.py`, a module of Geneva registry `Case ID` string
surgery. `RAW_ID_COL` now sits next to `ID_COL` in `frozen_schema.py` (`case_ids`
re-exports it), `SCHEMA_VERSION` is mirrored there too — the bundle stamps the
parquet without importing `fed_stroke` — and `read_pseudonym_key` moved to
`anonymise.py`, next to the anonymisation it feeds.

### H9 — the label decision is still formally open (blocking for W10 sign-off)

The roadmap's "Pick the primary label — one, not both" is unchecked, while
`schema.TARGET_COL = "death_3m"`. Settle it before the partner builds outcome
columns and before `FROZEN_FEATURES.md` is signed: the answer determines which
outcome they must chase hardest, and H1's answer may force it.

## 5. The bundle

| module | ships | why |
|---|---|---|
| `preprocessing/mappings/frozen_schema.py` | ✅ | the contract |
| `preprocessing/mappings/unit_aliases.py` | ✅ | site-agnostic label algebra |
| `preprocessing/mappings/shenzhen_encodings.py` | ✅ (new, W3) | their vocabularies |
| `preprocessing/frozen_table.py` | ✅ | the seven steps |
| `preprocessing/anonymise.py` | ✅ | de-identification + the id contract |
| `preprocessing/build_log.py` | ✅ | exclusion-chain log |
| `preprocessing/node_parquet.py` | ✅ (new, W1) | stamp + smoke report |
| `preprocessing/splits.py` | ❌ | Geneva loopback halves only |
| `preprocessing/case_ids.py` | ❌ | Geneva registry id surgery (see H8) |
| `preprocessing/first_values.py` | ❌ | Geneva EHR CSV extraction |
| `preprocessing/registry_cohort.py` | ❌ | Geneva registry cohort (replaced by §4 H6 prose) |
| `registry_alignement/**` | ❌ | Geneva summary tables |
| `architecture/fed_stroke/**` | ❌ here | ships separately as the 1.3.a wheel |

Root `pyproject.toml` currently packages `preprocessing*` **and**
`registry_alignement*` as one distribution. The bundle needs its own extra or
its own minimal dependency set: Python 3.12, `pandas>=2.0`, `pyarrow>=15`,
`numpy>=1.26`. No `flwr`, no `xgboost`, no `scikit-learn` (that dependency
enters only via `splits.py`, which does not ship), no network access.

## 6. Work items

Ordered; W1–W2 unblock the skeleton, W3–W6 wait on partner answers.

| # | item | blocked on |
|---|---|---|
| **W1** ✅ | Extract `preprocessing/node_parquet.py` from `preprocess_gva.py` (H7): `smoke_report`, `write_node_parquet`, `read_node_metadata`, `sha256_of`, constants. `preprocess_gva` imports them back and keeps its own `hash_inputs` — behaviour-identical, 356 tests green. | — |
| **W2** ✅ | `RAW_ID_COL` + `SCHEMA_VERSION` move to `frozen_schema.py` (H8), `read_pseudonym_key` to `anonymise.py`; `case_ids` re-exports. | — |
| **W3** 🟡 | `preprocessing/mappings/shenzhen_encodings.py` (H3): the four tables + `validate_shenzhen_encodings()`, which reports every gap at once (Geneva's import-time asserts cannot run while the tables are empty). Skeleton shipped; the tables themselves are the partner's. | Q2–Q4, Q6 |
| **W4** | Complete `SHENZHEN_TO_FROZEN` (H1) and add the twin import-time assertion Geneva has. | Q1, Q5 |
| **W5** | `SUBSTANCE_CONVERSIONS` per-column mass↔molar override table + wire into `convert_to_frozen_units` (H4), with the FEU/DDU decision recorded. | Q6 |
| **W6** | Timing-derivation spec + `derive_timings()` helper the partner calls pre-step-1 (H2). | Q5 |
| **W7** ✅ | `architecture/preprocessing/preprocess_shenzhen.py` skeleton: the seven steps pre-wired, `main()` + CLI mirroring Geneva's, six site stubs raising `NotImplementedError`. | W1–W3 |
| **W8** | Synthetic Shenzhen export generator (fake 200-row table in their raw column names, no real values) so the partner can run end-to-end on day one. | W7 |
| **W9** | `architecture/tests/test_shenzhen_contract.py` mirroring the Geneva contract tests: mapping covers exactly the frozen names, every binary column has an encoding, declared units resolve, end-to-end `wide_to_frozen` on the fixture, `write_node_parquet` round-trip. Runs on both sides. | W8 |
| **W10** | `registry_alignement/FROZEN_FEATURES.md` — the sign-off artifact: per column its Geneva source, Shenzhen source, unit, encoding convention, plausibility range, and the three outcomes. Generated from the mapping tables, not hand-maintained. | W4, H9 |
| **W11** | Partner key instructions: their own 32-byte key, generation, storage outside the repo, immutability for the project (the R4 split hashes the pseudonym), and the `ID_COL` value contract. | — |
| **W12** | Smoke-report exchange protocol: what travels, how, to whom; explicit statement that no other artifact leaves. | — |
| **W13** | `scripts/check_divergence_gates.py` — compares two smoke reports against the roadmap thresholds (label balance > 3×, per-feature median > 10×, missing rate > 25 pp), exempting declared-absent features (H5). **No such script exists today.** | W10, D3 |
| **W14** | Decide the missing-feature policy (H5) and add the all-NaN-column test. | D3 |

## 7. Decisions

- **D1 (proposed)** — the partner writes **only** `preprocess_shenzhen.py` plus
  their four mapping tables. Every shared step stays in the bundle, unmodified.
  A needed change to a shared function comes back to us as a request, not a
  local patch. *Why:* a forked `frozen_table.py` is undetectable from the smoke
  report and would surface as an unexplained divergence months later.
- **D2 (proposed)** — Shenzhen runs the **same anonymisation** as Geneva with
  **their own key**, even though their parquet never leaves their site. *Why:*
  the id format is a loader contract (`split('_')[0]`), and the age top-code
  keeps the smoke reports comparable.
- **D3 (proposed, needs the user)** — an all-NaN feature column is **legal when
  declared in advance**; it is exempt from the missing-rate gate and reported as
  a site-asymmetric feature. *Alternative:* drop the feature from the frozen
  schema for both sites — a `SCHEMA_VERSION` bump and a re-run of Geneva's
  preprocessing. Recommend all-NaN: XGBoost handles a constant column as a
  no-split feature, and the sentinel encoding makes it explicit for the DP arm.
- **D4 (proposed)** — `d_dimer` FEU/DDU is declared by the partner in writing
  and stamped into the smoke report, not inferred from the value distribution.
- **D5 (open, needs the user + H1's answer)** — the primary label. Current code
  says `death_3m`.

## 8. Freeze discipline

Any of W4, W5, W14 that changes `FROZEN_FEATURES`, `FROZEN_UNITS`,
`FROZEN_RANGES`, `FROZEN_OUTCOMES` or `ID_COL` is a **cross-site contract
change**: bump `SCHEMA_VERSION` in both mirrors
(`preprocessing/mappings/frozen_schema.py`, `architecture/fed_stroke/schema.py`),
bump `DP_MODEL_FORMAT` if a feature moves or its range changes, re-obtain
partner sign-off, and record it in [docs/logbook.md](../logbook.md). Adding a
unit synonym or a substance override is **not** a contract change — the contract
is the target unit, not the labels accepted on the way in.

## 9. Acceptance

- The partner can run `preprocess_shenzhen.py` on the synthetic fixture, in
  their own environment, with no access to Geneva data or infrastructure, and
  get a stamped node parquet + smoke report.
- `test_shenzhen_contract.py` passes on both sides.
- `FROZEN_FEATURES.md` is signed off by the partner, with every one of the 45
  columns sourced or declared absent.
- `check_divergence_gates.py` runs on the two real smoke reports and reports
  every gate as pass, or fail-with-explanation.
- Roadmap: the preprocessing track's "Shenzhen preprocessing" item can then be
  checked, and v0.b / 1.3.b unblock.

## 10. Risks

| risk | impact | mitigation |
|---|---|---|
| No 3-month follow-up at Shenzhen | pilot's label changes or headline weakens | Q1 first; fallback label `death_in_hospital` (Geneva: 5.7 % positive, 0 % missing) |
| `d_dimer` FEU/DDU wrong | silent 2× divergence, inside range, no error | D4 written declaration + the 10× median gate (a 2× error still passes — flag in the write-up) |
| Timing derivations diverge | ODT/ONT/DNT/OPT incomparable across sites | W6 spec + per-interval medians in the smoke report |
| Partner forks `frozen_table.py` | undetectable drift | D1 + ship the bundle with a version stamp; `test_shenzhen_contract.py` pins behaviour |
| ~40 k vs ~3.4 k rows | 12× imbalance dominates bagging aggregation | out of scope here; 1.3.b′ re-tunes, 1.3.d reports site-stratified |
