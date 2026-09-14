"""Clinical-score (scale file) first-value extraction — the GVA GCS source.

GCS is 100 % missing in the registry export ('GCS on admission' is never filled in), and it
is NOT among the patientvalue keys either: the Geneva EHR records it in the scale*.csv files
under several Glasgow form names, all scoring the same 3-15 total. These tests pin that
extraction. Synthetic frames only — no real registry / EHR data.
"""
import sys
from pathlib import Path

import pandas as pd

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.case_ids import RAW_ID_COL  # noqa: E402
from preprocessing.first_values import (  # noqa: E402
    SCALE_SCORES,
    extract_scale_first_values,
)

RAW_ID = RAW_ID_COL
ADMISSION = pd.Timestamp("2020-03-10")


def _cohort(*case_ids: str) -> pd.DataFrame:
    return pd.DataFrame({RAW_ID: list(case_ids), "admission_date": [ADMISSION] * len(case_ids)})


def _scale_rows(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(case_admission_id, scale, event_date 'DD.MM.YYYY HH:MM', score) -> a scale*.csv frame."""
    return pd.DataFrame(rows, columns=[RAW_ID, "scale", "event_date", "score"])


def test_gcs_is_a_declared_scale_score():
    assert dict(SCALE_SCORES)["gcs"], "the scale file must yield a 'gcs' variable"


def test_gcs_first_value_is_earliest_on_or_after_admission():
    scale_df = _scale_rows([
        ("p1_0001", "Glasgow", "09.03.2020 22:00", "11.0"),   # day before admission — ignored
        ("p1_0001", "Glasgow", "10.03.2020 14:30", "13.0"),   # admission day, later
        ("p1_0001", "Glasgow", "10.03.2020 08:15", "15.0"),   # admission day, earliest -> wins
        ("p1_0001", "Glasgow", "12.03.2020 08:00", "7.0"),
    ])
    out = extract_scale_first_values(scale_df, _cohort("p1_0001"))["gcs"]
    assert out["first_value"].tolist() == [15.0]
    assert out["first_datetime"].tolist() == [pd.Timestamp("2020-03-10 08:15")]


def test_every_glasgow_form_name_feeds_the_same_variable():
    """The nurses' forms differ (bare, + pupils, + pupils/motor, the 'urgence' typo with a
    double space) but every one of them scores the same 3-15 total."""
    forms = [
        "Glasgow",
        "Glasgow + pupilles",
        "Glasgow + pupilles + sensibilité/motricité",
        "Glasgow  urgence ",
    ]
    scale_df = _scale_rows([
        (f"p{i}_000{i}", form, "10.03.2020 09:00", "12.0") for i, form in enumerate(forms)
    ])
    cohort = _cohort(*(f"p{i}_000{i}" for i in range(len(forms))))
    out = extract_scale_first_values(scale_df, cohort)["gcs"]
    assert len(out) == len(forms)
    assert out["first_value"].unique().tolist() == [12.0]


def test_other_scales_never_leak_into_gcs():
    scale_df = _scale_rows([
        ("p1_0001", "NIHSS - National Institute of Health Stroke Scale", "10.03.2020 08:00", "22.0"),
        ("p1_0001", "Braden - Echelle de risque d'escarre", "10.03.2020 08:30", "18.0"),
        ("p1_0001", "Glasgow", "10.03.2020 09:00", "14.0"),
    ])
    out = extract_scale_first_values(scale_df, _cohort("p1_0001"))["gcs"]
    assert out["first_value"].tolist() == [14.0]


def test_scores_carry_no_unit_label():
    """The scale file has no unit column: a score is unitless by construction, so the frozen
    unit check must see a blank label (assumed 'no unit') rather than an unknown one."""
    scale_df = _scale_rows([("p1_0001", "Glasgow", "10.03.2020 09:00", "14.0")])
    out = extract_scale_first_values(scale_df, _cohort("p1_0001"))["gcs"]
    assert out["first_unit"].isna().all()


def test_patient_without_a_score_gets_no_row():
    scale_df = _scale_rows([("p1_0001", "Glasgow", "10.03.2020 09:00", "14.0")])
    out = extract_scale_first_values(scale_df, _cohort("p1_0001", "p2_0002"))["gcs"]
    assert out[RAW_ID].tolist() == ["p1_0001"]


def test_empty_scale_file_yields_the_empty_frame_shape():
    out = extract_scale_first_values(_scale_rows([]), _cohort("p1_0001"))["gcs"]
    assert out.empty
    assert list(out.columns) == [RAW_ID, "first_value", "first_datetime", "first_unit"]
