from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.trajectory_turn_helpers import add_circle_turn_fields


def _run_circle_turn_detector(angles_deg, *, threshold_deg=90):
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    frame_count = len(angles) + 1
    trj = SimpleNamespace(
        velAngles=angles,
        sp=np.full(frame_count, 3.0, dtype=float),
    )
    va = SimpleNamespace(fps=10)
    opts = SimpleNamespace(
        turn_duration_thresh=0.75,
        min_vel_angle_delta=threshold_deg,
        min_turn_speed=2,
    )
    stats = {
        "boundary_contact": np.zeros(frame_count, dtype=bool),
        "boundary_contact_regions": [slice(1, len(angles))],
    }

    add_circle_turn_fields(trj, va, stats, opts)
    return stats


def test_circle_turn_angle_deltas_preserve_sign_and_cancel():
    stats = _run_circle_turn_detector([0, 100, 0])

    assert stats["turning_indices"] == []
    assert stats["rejection_reasons"] == ["too_little_velocity_angle_change"]
    assert stats["total_vel_angle_deltas"] == pytest.approx([0])

    diagnostics = stats["turn_angle_diagnostics"][0]
    assert np.degrees(diagnostics["vel_angle_deltas"]) == pytest.approx([100, -100])
    assert np.degrees(diagnostics["signed_total_vel_angle_delta"]) == pytest.approx(0)


def test_circle_turn_angle_deltas_wrap_across_pi_with_sign():
    stats = _run_circle_turn_detector([170, -170, -150], threshold_deg=30)

    assert stats["turning_indices"] == [0]
    assert stats["rejection_reasons"] == ["turn"]
    assert np.degrees(stats["total_vel_angle_deltas"]) == pytest.approx([40])

    diagnostics = stats["turn_angle_diagnostics"][0]
    assert np.degrees(diagnostics["vel_angle_deltas"]) == pytest.approx([20, 20])
    assert np.degrees(diagnostics["signed_total_vel_angle_delta"]) == pytest.approx(40)
