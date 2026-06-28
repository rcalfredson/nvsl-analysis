import csv

import numpy as np
import pytest

from src.plotting.first_n_reward_diagnostics import (
    FIRST_N_REWARD_DIAGNOSTIC_FIELDS,
    load_first_n_reward_diagnostics_npz,
    validate_first_n_reward_diagnostics_bundle,
    validate_first_n_reward_diagnostics_csv,
)


def _row(**overrides):
    row = {
        "video_basename": "video_a",
        "video_path": "/data/video_a.avi",
        "va_tag": 0,
        "fly_idx": 0,
        "selected_subset_label": "All flies",
        "selected_trainings_label": "T1",
        "skip_first_sync_buckets": 0,
        "keep_first_sync_buckets": 0,
        "nth_reward_target": 10,
        "has_selected_window": True,
        "actual_reward_count_in_selected_window": 10,
        "eligible_for_nth_reward_cutoff": True,
        "sli": 0.25,
        "cutoff_frame": 950.0,
        "cutoff_training": 1.0,
        "cutoff_time_since_selected_window_start_s": 95.0,
        "cutoff_time_since_cutoff_training_start_s": 95.0,
        "time_to_first_actual_reward_s": 5.0,
        "time_to_nth_actual_reward_s": 95.0,
        "first_n_reward_span_s": 90.0,
        "actual_reward_count_by_cutoff": 10.0,
        "control_reward_count_by_cutoff": 0.0,
        "actual_circle_entry_count_by_cutoff": 10.0,
        "control_circle_entry_count_by_cutoff": 0.0,
        "reward_pi_by_cutoff": 1.0,
        "actual_entry_minus_reward_count_by_cutoff": 0.0,
        "control_entry_minus_reward_count_by_cutoff": -10.0,
        "control_to_actual_entry_ratio_by_cutoff": 0.0,
        "control_to_actual_reward_ratio_by_cutoff": 0.0,
        "reward_event_type": "calc",
        "selected_reward_count_in_selected_window": 10,
        "time_to_first_selected_reward_s": 5.0,
        "time_to_nth_selected_reward_s": 95.0,
        "first_n_selected_reward_span_s": 90.0,
        "selected_reward_rate_to_nth_per_min": 6.0,
        "first_n_selected_reward_distance_traveled_mm": 900.0,
        "selected_reward_rate_to_nth_per_m": 10.0,
    }
    row.update(overrides)
    return row

def _bundle_from_rows(rows):
    return {
        key: np.asarray([row[key] for row in rows], dtype=object)
        for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS
    }

def test_validate_first_n_reward_diagnostics_bundle_accepts_valid_rows(tmp_path):
    bundle = _bundle_from_rows([_row()])
    validate_first_n_reward_diagnostics_bundle(bundle)

    path = tmp_path / "first_n_reward_diagnostics.npz"
    np.savez(path, **bundle)
    loaded = load_first_n_reward_diagnostics_npz(str(path))
    assert loaded['selected_reward_rate_to_nth_per_min'][0] == 6.0
    assert loaded['selected_reward_rate_to_nth_per_m'][0] == 10.0

def test_validate_first_n_reward_diagnostics_bundle_rejects_bad_rate():
    bundle = _bundle_from_rows(
        [_row(selected_reward_rate_to_nth_per_min=10 * 60.0 / 95.0)]
    )

    with pytest.raises(ValueError, match="inconsistent selected reward rate"):
        validate_first_n_reward_diagnostics_bundle(bundle)

def test_validate_first_n_reward_diagnostics_bundle_rejects_bad_distance_rate():
    bundle = _bundle_from_rows(
        [_row(selected_reward_rate_to_nth_per_m=10.0 / 0.9)]
    )

    with pytest.raises(ValueError, match="inconsistent selected reward distance rate"):
        validate_first_n_reward_diagnostics_bundle(bundle)

def test_validate_first_n_reward_diagnostics_bundle_rejects_shape_mismatch():
    bundle = _bundle_from_rows([_row()])
    bundle['sli'] = np.asarray([0.25, 0.5])

    with pytest.raises(ValueError, match="inconsistent column lengths"):
        validate_first_n_reward_diagnostics_bundle(bundle)

def test_validate_first_n_reward_diagnostics_csv_accepts_valid_rows(tmp_path):
    path = tmp_path / "first_n_reward_diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIRST_N_REWARD_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerow(_row())

    validate_first_n_reward_diagnostics_csv(str(path))

def test_validate_first_n_reward_diagnostics_csv_rejects_missing_column(tmp_path):
    path = tmp_path / "first_n_reward_diagnostics.csv"
    fieldnames = [
        key
        for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS
        if key != "selected_reward_rate_to_nth_per_min"
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        row = _row()
        del row["selected_reward_rate_to_nth_per_min"]
        writer.writerow(row)

    with pytest.raises(ValueError, match="missing columns"):
        validate_first_n_reward_diagnostics_csv(str(path))
