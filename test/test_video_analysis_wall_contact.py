from types import SimpleNamespace

import pytest

pytest.importorskip("cv2")

from src.analysis.video_analysis import _apply_boundary_contact_results


def test_parallel_wall_result_keeps_original_trajectory_index():
    bad_exp = SimpleNamespace()
    valid_yoked = SimpleNamespace()
    va = SimpleNamespace(trx=[bad_exp, valid_yoked])

    _apply_boundary_contact_results(
        va,
        trajectory_indices=[1],
        results=[
            {
                "boundary_event_stats": {"wall": "yoked-result"},
                "wall_orientations": ["all"],
            }
        ],
    )

    assert not hasattr(bad_exp, "boundary_event_stats")
    assert valid_yoked.boundary_event_stats == {"wall": "yoked-result"}
    assert va.wall_orientations == ["all"]
