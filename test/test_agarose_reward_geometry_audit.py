import numpy as np
import pytest
import scripts.plot_agarose_reward_geometry_audit as geometry_plot

from src.analysis.agarose_reward_geometry_audit import (
    _count_complete_draws,
    audit_site_candidates,
    circle_outside_rectangle_fraction,
    rotate_about,
    signed_wall_clearances_mm,
)
from scripts.plot_agarose_reward_geometry_audit import _select_complete_draw


def test_reward_centered_rotation_preserves_reward_distance():
    reward = np.asarray((3.0, 4.0))
    site = np.asarray((13.0, 4.0))

    rotated = rotate_about(site, reward, 73.0)

    assert np.linalg.norm(rotated - reward) == pytest.approx(
        np.linalg.norm(site - reward)
    )


def test_analytical_candidates_are_reward_matched_and_neighbor_balanced():
    rows = audit_site_candidates(
        physical_centers=((-10.0, 0.0), (0.0, -10.0), (10.0, 0.0), (0.0, 10.0)),
        physical_site_index=0,
        reward_center=(2.0, 1.0),
        floor_bounds=(-30.0, -30.0, 30.0, 30.0),
        nominal_radius_px=0.1,
        outer_radius_px=0.2,
        px_per_mm=1.0,
        agarose_buffer_mm=0.0,
        candidate_method="analytical",
    )

    assert rows
    assert {row["candidate_method"] for row in rows} == {"analytical"}
    assert max(abs(row["reward_distance_error_mm"]) for row in rows) < 1e-10
    assert max(row["nearest_two_gap_imbalance_mm"] for row in rows) < 1e-10


def test_signed_wall_clearance_allows_wall_truncated_analysis_circle():
    clearances = signed_wall_clearances_mm(
        center=(4.0, 10.0),
        floor_bounds=(0.0, 0.0, 20.0, 20.0),
        outer_radius_px=4.5,
        px_per_mm=1.0,
    )

    assert clearances[0] == pytest.approx(-0.5)
    assert clearances[1] == pytest.approx(5.5)


def test_circle_outside_area_fraction_handles_walls_and_corners():
    bounds = (0.0, 0.0, 20.0, 20.0)

    assert circle_outside_rectangle_fraction((10.0, 10.0), 2.0, bounds) == pytest.approx(
        0.0, abs=2e-6
    )
    assert circle_outside_rectangle_fraction((0.0, 10.0), 2.0, bounds) == pytest.approx(
        0.5, abs=2e-6
    )
    assert circle_outside_rectangle_fraction((0.0, 0.0), 2.0, bounds) == pytest.approx(
        0.75, abs=2e-6
    )


def test_candidate_audit_uses_outer_to_nominal_agarose_surface_gap():
    rows = audit_site_candidates(
        physical_centers=((4.0, 10.0), (10.0, 4.0), (16.0, 10.0), (10.0, 16.0)),
        physical_site_index=0,
        reward_center=(8.0, 8.0),
        floor_bounds=(0.0, 0.0, 20.0, 20.0),
        nominal_radius_px=1.0,
        outer_radius_px=1.5,
        px_per_mm=1.0,
        angle_step_deg=90.0,
        agarose_buffer_mm=1.0,
        candidate_method="grid",
    )

    original = rows[0]
    assert original["min_physical_agarose_gap_mm"] == pytest.approx(-2.5)
    assert not original["passes_agarose_buffer"]
    assert all(abs(row["reward_distance_error_mm"]) < 1e-12 for row in rows)
    assert any(row["passes_hard_geometry"] for row in rows[1:])


def test_complete_draw_count_rejects_overlapping_virtual_circles():
    def row(x, y):
        return {"candidate_x_px": x, "candidate_y_px": y}

    candidates = [
        [row(0.0, 0.0)],
        [row(1.0, 0.0), row(4.0, 0.0)],
    ]

    count, capped = _count_complete_draws(candidates, outer_radius_px=1.0)

    assert count == 1
    assert not capped


def test_nearest_wall_diagnostic_is_separate_from_two_wall_match():
    rows = audit_site_candidates(
        physical_centers=((4.0, 10.0),),
        physical_site_index=0,
        reward_center=(10.0, 10.0),
        floor_bounds=(0.0, 0.0, 30.0, 30.0),
        nominal_radius_px=1.0,
        outer_radius_px=1.0,
        px_per_mm=1.0,
        angle_step_deg=180.0,
        agarose_buffer_mm=0.0,
        candidate_method="grid",
        wall_tiers_mm={"diagnostic": (10.0, 0.0)},
    )

    rotated = rows[1]
    assert rotated["passes_diagnostic_nearest_only"]
    assert not rotated["passes_diagnostic"]


def test_primary_geometry_limits_outer_circle_area_outside_floor():
    rows = audit_site_candidates(
        physical_centers=((5.0, 10.0),),
        physical_site_index=0,
        reward_center=(10.0, 10.0),
        floor_bounds=(0.0, 0.0, 20.0, 20.0),
        nominal_radius_px=0.1,
        outer_radius_px=4.0,
        px_per_mm=1.0,
        angle_step_deg=180.0,
        agarose_buffer_mm=0.0,
        max_outside_area_fraction=0.25,
        candidate_method="grid",
    )

    assert rows[0]["passes_primary_geometry"] is False  # original site fails buffer
    assert rows[1]["outside_floor_area_fraction"] == pytest.approx(0.0, abs=2e-6)
    assert rows[1]["passes_primary_geometry"] is True


def test_visualizer_selects_one_nonoverlapping_candidate_per_site():
    rows = []
    for site, x in enumerate((0.0, 4.0, 8.0, 12.0), 1):
        rows.append(
            {
                "physical_site": str(site),
                "candidate_x_px": str(x),
                "candidate_y_px": "0",
                "passes_primary_geometry": "True",
                "outer_radius_px": "1",
            }
        )

    selected = _select_complete_draw(rows, seed=101, strategy="random")

    assert sorted(selected) == [1, 2, 3, 4]


def test_visualizer_maximin_draw_maximizes_worst_agarose_gap():
    rows = []
    for site, base_x in enumerate((0.0, 10.0), 1):
        for offset, gap in ((0.0, 1.0), (3.0, 2.0)):
            rows.append(
                {
                    "physical_site": str(site),
                    "candidate_x_px": str(base_x + offset),
                    "candidate_y_px": "0",
                    "passes_primary_geometry": "True",
                    "outer_radius_px": "1",
                    "min_physical_agarose_gap_mm": str(gap),
                    "outside_floor_area_fraction": "0",
                }
            )

    selected = _select_complete_draw(rows, seed=101, strategy="maximin")

    assert {float(row["min_physical_agarose_gap_mm"]) for row in selected.values()} == {
        2.0
    }


def test_visualizer_lists_configs_without_requiring_output(monkeypatch, capsys):
    monkeypatch.setattr(
        geometry_plot,
        "_read_rows",
        lambda _path: [
            {
                "video": "video.avi",
                "video_index": "0",
                "va_fly": "0",
                "trajectory_fly": "0",
                "training": "1",
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_agarose_reward_geometry_audit.py",
            "--candidates",
            "unused.csv",
            "--list-configs",
        ],
    )

    assert geometry_plot.main() == 0
    assert "video.avi" in capsys.readouterr().out
