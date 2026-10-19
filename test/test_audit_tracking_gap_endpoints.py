"""Portable synthetic checks; no Synology recordings are needed."""

import numpy as np
import pytest

from scripts import audit_tracking_gap_endpoints as audit


def context(x, y=None):
    return dict(
        video_path="example.avi",
        pair=1,
        role="yok",
        physical_id=3,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y if y is not None else [50] * len(x), dtype=float),
        fps=2.0,
        floor=((0.0, 0.0), (100.0, 100.0)),
        px_per_mm=10.0,
        border_px=1.0,
        geometries=[
            dict(
                training=1,
                active_start_frame=0,
                active_stop_frame=100,
                training_start_frame=2,
                training_stop_frame=80,
                reward=(50.0, 50.0, 10.0),
                control=(50.0, 50.0, 30.0),
            )
        ],
    )


def test_missing_runs_preserve_initial_and_trailing_gaps():
    assert audit.missing_runs([np.nan, 1, np.nan, np.nan, 2, np.nan]) == [
        (0, 1),
        (2, 4),
        (5, 6),
    ]
    assert audit.missing_runs([1, 2]) == []
    assert audit.missing_runs([np.nan, np.nan]) == [(0, 2)]


def test_gap_indices_timing_and_raw_endpoints():
    c = context([np.nan, 20, np.nan, np.nan, 70, np.nan])
    before = c["x"].copy()
    initial, internal, trailing = audit.gap_rows(c)
    assert initial["before_frame"] is None
    assert initial["after_frame"] == 1
    assert not initial["counted_in_analyzer_loss_fraction"]
    assert internal["gap_start_frame"] == 2
    assert internal["gap_stop_frame_exclusive"] == 4
    assert internal["last_missing_frame"] == 3
    assert internal["lost_frames"] == 2
    assert internal["missing_duration_s"] == 1.0
    assert internal["observation_bracket_s"] == 1.5
    assert internal["before_frame"] == 1 and internal["before_x_px"] == 20
    assert internal["after_frame"] == 4 and internal["after_x_px"] == 70
    assert internal["before_phase"] == "pre"
    assert internal["after_phase"] == "training"
    assert trailing["after_frame"] is None and trailing["after_x_px"] is None
    assert trailing["after_status"] == "no_observed_endpoint"
    assert trailing["counted_in_analyzer_loss_fraction"]
    np.testing.assert_array_equal(c["x"], before)


def measure(c, frame=0):
    return audit.endpoint_fields(
        frame,
        c["x"],
        c["y"],
        c["floor"],
        c["px_per_mm"],
        c["geometries"],
        c["border_px"],
    )


def test_wall_sign_and_circle_boundary_sign_have_separate_meanings():
    center = measure(context([50]))
    assert center["nearest_wall_signed_mm"] == 5
    assert center["reward_boundary_signed_mm"] == -1
    assert center["reward_outer_border_signed_mm"] == -1.1
    edge = measure(context([60]))
    assert edge["reward_boundary_signed_mm"] == 0
    assert edge["reward_outer_border_signed_mm"] == -0.1
    outside = measure(context([-2]))
    assert outside["nearest_wall"] == "left"
    assert outside["nearest_wall_signed_mm"] == -0.2
    assert outside["outside_floor"]


def test_invalid_y_is_not_silently_interpolated_or_used():
    result = measure(context([20], [np.nan]))
    assert result["status"] == "invalid_observed_xy"
    assert result["nearest_wall_signed_mm"] is None
    assert result["reward_boundary_signed_mm"] is None


def test_geometry_boundaries_and_wake_intervals_are_explicit():
    geometries = [
        dict(training=1, active_start_frame=2, active_stop_frame=4),
        dict(training=2, active_start_frame=6, active_stop_frame=10),
    ]
    assert audit.geometry_at_frame(geometries, 2)["training"] == 1
    assert audit.geometry_at_frame(geometries, 4) is None
    assert audit.geometry_at_frame(geometries, 5) is None
    assert audit.geometry_at_frame(geometries, 6)["training"] == 2
    c = context([20])
    c["geometries"] = []
    result = measure(c)
    assert result["nearest_wall_signed_mm"] == 2
    assert result["phase"] == "outside_circle_analysis_ranges"
    assert result["control_boundary_signed_mm"] is None


def test_circle_wall_clearance_and_border_share_same_units():
    rows = audit.geometry_rows(context([20]))
    reward, control = rows
    assert reward["circle_wall_clearance_mm"] == 4
    assert reward["outer_border_wall_clearance_mm"] == 3.9
    assert control["circle_wall_clearance_mm"] == 2
    assert control["outer_border_wall_clearance_mm"] == 1.9


def test_all_missing_keeps_both_endpoints_unknown():
    (row,) = audit.gap_rows(context([np.nan, np.nan]))
    assert row["initial_gap"] and row["trailing_gap"]
    assert row["before_frame"] is None and row["after_frame"] is None
    assert row["before_nearest_wall_signed_mm"] is None


def test_raw_xy_length_mismatch_fails():
    with pytest.raises(ValueError, match="same nonzero length"):
        audit.gap_rows(context([1, np.nan], [1]))
