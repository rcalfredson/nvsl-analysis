import csv
from types import SimpleNamespace

import numpy as np
import pytest

from src.exporting.rpid_auc_audit import write_rpid_auc_audit_csv
from src.utils.common import store_auc_entries


def _va(name, fly):
    return SimpleNamespace(fn=name, f=fly, flies=(0, 1), saved_auc={})


def test_rpid_auc_audit_reconciles_inputs_terms_and_equation(tmp_path):
    va = _va("/data/video.avi", 3)
    trns = [SimpleNamespace(n=1)]
    raw = np.array([[[[0.2, 0.4, 0.6], [0.1, 0.1, 0.2]]]])
    differences = raw[:, :, 0, :] - raw[:, :, 1, :]
    store_auc_entries([va], "rpid", differences, n_flies=1)
    out_csv = tmp_path / "rpid_auc_audit.csv"

    row_count, mismatch_count = write_rpid_auc_audit_csv(
        [va],
        trns,
        raw,
        out_csv=str(out_csv),
        sync_bucket_minutes=10,
    )

    with out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert row_count == 3
    assert mismatch_count == 0
    assert rows[0]["video"] == "video.avi"
    assert rows[0]["fly"] == "3"
    assert float(rows[0]["difference"]) == pytest.approx(0.1)
    assert float(rows[0]["trapezoid_term_to_next"]) == pytest.approx(0.2)
    assert float(rows[1]["trapezoid_term_to_next"]) == pytest.approx(0.35)
    assert float(rows[1]["cumulative_auc_through_next"]) == pytest.approx(0.55)
    assert float(rows[0]["recomputed_auc"]) == pytest.approx(0.55)
    assert float(rows[0]["learning_stats_auc"]) == pytest.approx(0.55)
    assert rows[0]["matches_learning_stats"] == "True"
    assert rows[0]["auc_x_spacing"] == "1"
    assert rows[0]["auc_units"] == "PI*bucket_interval"
    assert rows[0]["equation"].startswith("AUC = (")
    assert rows[0]["equation"].endswith("= 0.55")


def test_rpid_auc_audit_matches_production_trailing_nan_rule(tmp_path):
    vas = [_va("v1.avi", 0), _va("v2.avi", 1)]
    trns = [SimpleNamespace(n=1)]
    raw = np.array(
        [
            [[[0.0, 1.0, np.nan], [0.0, 0.0, np.nan]]],
            [[[2.0, 4.0, np.nan], [1.0, 1.0, np.nan]]],
        ]
    )
    differences = raw[:, :, 0, :] - raw[:, :, 1, :]
    store_auc_entries(vas, "rpid", differences, n_flies=1)
    out_csv = tmp_path / "rpid_auc_audit.csv"

    _, mismatch_count = write_rpid_auc_audit_csv(
        vas,
        trns,
        raw,
        out_csv=str(out_csv),
        sync_bucket_minutes=10,
    )

    with out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert mismatch_count == 0
    assert len(rows) == 6
    assert {row["effective_bucket_count"] for row in rows} == {"2"}
    assert {row["trailing_all_nan_bucket_dropped"] for row in rows} == {"True"}
    trailing_rows = [row for row in rows if row["bucket"] == "3"]
    assert all(row["included_in_auc"] == "False" for row in trailing_rows)
    assert all(row["trapezoid_term_to_next"] == "" for row in trailing_rows)


def test_rpid_auc_audit_flags_stored_value_mismatch(tmp_path):
    va = _va("video.avi", 0)
    trns = [SimpleNamespace(n=1)]
    raw = np.array([[[[0.0, 1.0], [0.0, 0.0]]]])
    va.saved_auc["rpid"] = [{"training": 0, "exp": 99.0, "ctrl": np.nan}]
    out_csv = tmp_path / "rpid_auc_audit.csv"

    _, mismatch_count = write_rpid_auc_audit_csv(
        [va],
        trns,
        raw,
        out_csv=str(out_csv),
        sync_bucket_minutes=10,
    )

    with out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert mismatch_count == 1
    assert {row["matches_learning_stats"] for row in rows} == {"False"}
