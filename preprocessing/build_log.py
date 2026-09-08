"""Build log for a site's frozen-table run: exclusions with reasons + nulled feature counts.

Aggregate-only by construction — stage names, counts and percentages, never ids or values —
so the file can sit next to the node parquet and travel with the smoke report across sites.

Two halves:
  - record_exclusion: called after every step that drops rows (registry_cohort.build_cohort,
    preprocess_gva.wide_to_frozen); records rows AND distinct patients before/after.
  - format_build_log / write_build_log: render the collected stages, the plausibility report
    (frozen_table.nullify_out_of_range) and the unit check into one human-readable text.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .case_ids import RAW_ID_COL, build_case_admission_id
from .mappings.frozen_schema import ID_COL


def patient_ids(df: pd.DataFrame) -> pd.Series | None:
    """Distinct-patient key of a frame at any pipeline stage: the prefix of the pseudonymous
    ID_COL (de-identified table) or of the raw RAW_ID_COL ('<patient_id>_<eds>') when present,
    else from the registry 'Case ID'; None if none of them."""
    for id_col in (ID_COL, RAW_ID_COL):
        if id_col in df.columns:
            return df[id_col].astype(str).str.split("_").str[0]
    if "Case ID" in df.columns:
        return build_case_admission_id(df["Case ID"].astype(str)).str.split("_").str[0]
    return None


def record_exclusion(exclusions: list[dict], stage: str, reason: str,
                     before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Append one exclusion stage to `exclusions` and return it.

    rows_* count admissions (frame rows); patients_* count distinct patient ids (None when the
    frame carries no patient key). A patient counts as excluded only when NONE of their rows
    survive the stage — e.g. dropping a patient's TIA admission while their ischemic-stroke
    admission stays excludes a row, not a patient.
    """
    pb, pa = patient_ids(before), patient_ids(after)
    entry = {
        "stage": stage,
        "reason": reason,
        "rows_before": int(len(before)),
        "rows_excluded": int(len(before) - len(after)),
        "rows_after": int(len(after)),
        "patients_before": int(pb.nunique()) if pb is not None else None,
        "patients_after": int(pa.nunique()) if pa is not None else None,
    }
    entry["patients_excluded"] = (
        entry["patients_before"] - entry["patients_after"]
        if pb is not None and pa is not None else None
    )
    exclusions.append(entry)
    return entry


def _na(v) -> str:
    return "n/a" if v is None else str(v)


def format_build_log(title: str, header: dict, exclusions: list[dict],
                     out_of_range: dict | None = None, unit_check: dict | None = None,
                     outcomes: dict | None = None) -> str:
    """Render the build log. `header` is an ordered dict of provenance lines (date, inputs,
    schema version, output shape); `exclusions` from record_exclusion; `outcomes` the per-outcome
    availability dict (n_recorded / n_missing / n_positive or median); `out_of_range` from
    frozen_table.nullify_out_of_range; `unit_check` from frozen_table.convert_to_frozen_units."""
    lines = [title, "=" * max(len(title), 24), ""]
    width = max((len(k) for k in header), default=0)
    for key, value in header.items():
        lines.append(f"{key:<{width}}  {value}")

    # --- exclusions -----------------------------------------------------------------
    lines += ["", "Exclusions (rows = registry admissions; patients = distinct patient ids)"]
    if not exclusions:
        lines.append("no exclusion stages recorded")
    else:
        lines.append(f"{'#':>2} {'stage':<48} {'rows_before':>11} {'excluded':>9} {'rows_after':>10}"
                     f" {'pts_before':>10} {'pts_excl':>8} {'pts_after':>9}")
        for i, e in enumerate(exclusions, 1):
            pct = (f" ({100 * e['rows_excluded'] / e['rows_before']:.1f}% of rows)"
                   if e["rows_before"] else "")
            lines.append(f"{i:>2} {e['stage']:<48} {e['rows_before']:>11} {e['rows_excluded']:>9} "
                         f"{e['rows_after']:>10} {_na(e['patients_before']):>10} "
                         f"{_na(e['patients_excluded']):>8} {_na(e['patients_after']):>9}")
            lines.append(f"   reason: {e['reason']}{pct}")
        first, last = exclusions[0], exclusions[-1]
        rows_total = first["rows_before"] - last["rows_after"]
        total = (f"total excluded: {rows_total} of {first['rows_before']} rows"
                 + (f" ({100 * rows_total / first['rows_before']:.1f}%)" if first["rows_before"] else ""))
        if first["patients_before"] is not None and last["patients_after"] is not None:
            pts_total = first["patients_before"] - last["patients_after"]
            total += f"; {pts_total} of {first['patients_before']} patients"
            if first["patients_before"]:
                total += f" ({100 * pts_total / first['patients_before']:.1f}%)"
        total += f" -> {last['rows_after']} rows, {_na(last['patients_after'])} patients kept"
        lines.append(total)

    # --- outcome availability ---------------------------------------------------------
    if outcomes is not None:
        lines += ["", "Outcome availability (no row is excluded for a missing outcome — the training "
                      "label is chosen in fed_stroke.schema and unlabelled rows are dropped at load time)",
                  f"{'outcome':<22} {'recorded':>8} {'missing':>8} {'%missing':>8}  note"]
        for name, o in outcomes.items():
            total = o["n_recorded"] + o["n_missing"]
            pct = 100 * o["n_missing"] / total if total else 0.0
            if "n_positive" in o:
                share = 100 * o["n_positive"] / o["n_recorded"] if o["n_recorded"] else 0.0
                note = f"1 = {o['n_positive']} ({share:.1f}% of recorded)"
            else:
                note = f"median {o['median']:g}" if o.get("median") is not None else "no recorded values"
            lines.append(f"{name:<22} {o['n_recorded']:>8} {o['n_missing']:>8} {pct:>7.1f}%  {note}")

    # --- nulled feature values --------------------------------------------------------
    lines += ["", "Nulled feature values (outside the public plausibility ranges -> NaN)"]
    if out_of_range is None:
        lines.append("plausibility step not run")
    else:
        gate = out_of_range["max_out_of_range_frac"]
        lines.append(f"gate: a feature with > {100 * gate:g}% of its recorded values out of range "
                     f"fails the build; pass={out_of_range['pass']}")
        lines.append(f"{'feature':<34} {'range':>18} {'recorded':>8} {'below':>6} {'above':>6} "
                     f"{'nulled':>6} {'%rec':>6} {'min_seen':>12} {'max_seen':>12}")
        n_feats = 0
        for feature, c in out_of_range["columns"].items():
            n = c["n_below"] + c["n_above"]
            if not n:
                continue
            n_feats += 1
            rng = f"[{c['lo']:g}, {c['hi']:g}]"
            pct = 100 * n / c["n_values"] if c["n_values"] else 0.0
            lines.append(f"{feature:<34} {rng:>18} {c['n_values']:>8} {c['n_below']:>6} "
                         f"{c['n_above']:>6} {n:>6} {pct:>5.1f}% {c['min_seen']:>12g} "
                         f"{c['max_seen']:>12g}")
        untouched = len(out_of_range["columns"]) - n_feats
        lines.append(f"total nulled: {out_of_range['n_nulled_total']} values across {n_feats} "
                     f"features; {untouched} features untouched")

    # --- unit check (one line + warnings) ---------------------------------------------
    if unit_check is not None:
        lines += ["", f"Unit check: pass={unit_check['pass']}; {unit_check['n_columns_checked']} "
                      f"numeric features checked; {unit_check['n_assumed_unit_total']} values had "
                      f"no unit label and were assumed already in the frozen unit"]
        for w in unit_check.get("warnings", []):
            lines.append(f"  - {w}")
    return "\n".join(lines) + "\n"


def write_build_log(path: Path, text: str) -> Path:
    """Write the log text (creating parent directories) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
