import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.trajectory import Trajectory
from src.analysis.video_analysis import VideoAnalysis
from src.exporting.agarose_sli_bundle import (
    _extract_agarose_arrays,
    export_agarose_sli_bundle,
)
from src.utils.common import CT
from src.utils.agarose_debug import _debug_image_candidates


class _IdentityXformer:
    fctr = 1.0
    frameSize = (244, 244)

    def t2f(self, x, y, **_kwargs):
        return x, y


class _Training:
    def __init__(self, *, start, stop, name):
        self.start = int(start)
        self.stop = int(stop)
        self._name = name

    def name(self):
        return self._name


class _TrajectoryEpisodes:
    def __init__(self, episodes, *, bad=False):
        self._episodes = list(episodes)
        self._bad = bool(bad)
        self.calls = []

    def calc_agarose_dual_circle_episodes(self, **kwargs):
        self.calls.append(kwargs)
        self.agarose_dual_circle_episodes = list(self._episodes)


def _trajectory_at_well_distances(distances_px):
    xf = _IdentityXformer()
    _radius, centers = CT.large.arenaWells(xf, 0)
    cx, cy = centers[0]
    trj = object.__new__(Trajectory)
    trj.x = cx + np.asarray(distances_px, dtype=float)
    trj.y = np.full_like(trj.x, cy)
    trj.f = 0
    trj.va = SimpleNamespace(ct=CT.large, xf=xf, trxf={0: None})
    return trj


def _video_analysis_stub(**attrs):
    va = object.__new__(VideoAnalysis)
    for name, value in attrs.items():
        setattr(va, name, value)
    return va


def test_agarose_debug_image_candidates_follow_active_window_and_balance_geometry():
    def row(geometry, avoids, bucket, *, included=True, role="exp"):
        return {
            "geometry": geometry,
            "avoids_inner": avoids,
            "fly_role": role,
            "included_by_active_filters": included,
            "sync_training_idx_1based": 2,
            "sync_bucket_idx_1based": bucket,
        }

    va = SimpleNamespace(
        agarose_dual_circle_debug_rows=[
            row("physical_agarose", False, 1),
            row("physical_agarose", False, 2),
            row("physical_agarose", True, 3),
            row("virtual_control", False, 4),
            row("virtual_control", True, 5),
            row("virtual_control", True, 3, included=False),
            row("virtual_control", True, 3, role="ctrl"),
        ]
    )

    selected = _debug_image_candidates(
        [va],
        training_index=2,
        sync_bucket_start_index=2,
        sync_bucket_end_index=5,
    )

    assert [candidate[1]["sync_bucket_idx_1based"] for candidate in selected] == [
        2,
        3,
        4,
        5,
    ]
    assert {candidate[1]["geometry"] for candidate in selected} == {
        "physical_agarose",
        "virtual_control",
    }


def test_agarose_debug_gallery_includes_rejected_reward_arc_entries():
    retained = {
        "geometry": "physical_agarose",
        "avoids_inner": True,
        "fly_role": "exp",
        "included_by_active_filters": True,
        "exclude_reward_facing_arc_entries": True,
        "reward_arc_entry_rejected": False,
        "sync_training_idx_1based": 2,
        "sync_bucket_idx_1based": 3,
    }
    rejected = {
        **retained,
        "included_by_active_filters": False,
        "reward_arc_entry_rejected": True,
    }
    va = SimpleNamespace(agarose_dual_circle_debug_rows=[retained, rejected])

    selected = _debug_image_candidates([va], training_index=2)

    assert len(selected) == 2
    assert {
        row["reward_arc_entry_rejected"] for _va, row in selected
    } == {False, True}


def test_agarose_dual_circle_classifies_outer_only_and_inner_contact_episodes():
    trj = _trajectory_at_well_distances(
        [41.0, 31.0, 33.0, 41.0, 31.0, 20.0, 31.0, 41.0]
    )

    trj.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    assert [
        (ep["start"], ep["stop"], ep["avoids_inner"], ep["entered_inner_frame"])
        for ep in trj.agarose_dual_circle_episodes
    ] == [(1, 3, True, None), (4, 7, False, 5)]
    assert [
        ep["wall_facing_entry"] for ep in trj.agarose_dual_circle_episodes
    ] == [False, False]
    assert all(
        ep["entry_wall_alignment"] < 0
        for ep in trj.agarose_dual_circle_episodes
    )
    label, x, y, method = trj.agarose_dual_circle_episodes[0][
        "entry_outer_intersections"
    ][0]
    center = np.asarray(
        trj.agarose_dual_circle_geometry["centers_px"][0], dtype=float
    )
    assert label == "well1"
    assert method == "segment_intersection"
    assert np.linalg.norm(np.asarray((x, y)) - center) == pytest.approx(
        trj.agarose_dual_circle_geometry["outer_radius_px"]
    )


def test_agarose_entry_alignment_identifies_outward_wall_facing_semicircle():
    xf = _IdentityXformer()
    _radius, centers = CT.large.arenaWells(xf, 0)
    cx, cy = centers[0]
    trj = object.__new__(Trajectory)
    trj.x = cx - np.asarray([41.0, 31.0, 33.0, 41.0])
    trj.y = np.full_like(trj.x, cy)
    trj.f = 0
    trj.va = SimpleNamespace(ct=CT.large, xf=xf, trxf={0: None})

    trj.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    assert len(trj.agarose_dual_circle_episodes) == 1
    episode = trj.agarose_dual_circle_episodes[0]
    assert episode["wall_facing_entry"] is True
    assert episode["entry_wall_alignment"] == pytest.approx(1.0)


def test_agarose_dual_circle_outer_requires_full_border_crossings():
    outer_only = _trajectory_at_well_distances([36.0, 35.4, 34.5, 35.4, 36.0])

    outer_only.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    assert [
        (ep["start"], ep["stop"], ep["avoids_inner"])
        for ep in outer_only.agarose_dual_circle_episodes
    ] == [(2, 4, True)]

    enters_inner = _trajectory_at_well_distances(
        [36.0, 34.0, 28.4, 27.5, 28.4, 34.0, 36.0]
    )

    enters_inner.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    assert [
        (ep["start"], ep["stop"], ep["avoids_inner"], ep["entered_inner_frame"])
        for ep in enters_inner.agarose_dual_circle_episodes
    ] == [(1, 6, False, 3)]


def test_agarose_dual_circle_uses_border_only_for_outer_circle(monkeypatch):
    border_widths = []
    original_calc_in_circle = Trajectory.calc_in_circle

    def recording_calc_in_circle(self, *args, **kwargs):
        border_widths.append(kwargs["border_width_px"])
        return original_calc_in_circle(self, *args, **kwargs)

    monkeypatch.setattr(Trajectory, "calc_in_circle", recording_calc_in_circle)
    trj = _trajectory_at_well_distances([41.0, 34.0, 27.5, 34.0, 41.0])

    trj.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    expected_outer_width = 0.1 * CT.large.pxPerMmFloor()
    assert border_widths
    np.testing.assert_allclose(border_widths[::2], expected_outer_width)
    np.testing.assert_allclose(border_widths[1::2], 0.0)


def test_agarose_dual_circle_legacy_policy_reproduces_pre_hysteresis_borders(
    monkeypatch,
):
    border_widths = []
    original_calc_in_circle = Trajectory.calc_in_circle

    def recording_calc_in_circle(self, *args, **kwargs):
        border_widths.append(kwargs["border_width_px"])
        return original_calc_in_circle(self, *args, **kwargs)

    monkeypatch.setattr(Trajectory, "calc_in_circle", recording_calc_in_circle)
    trj = _trajectory_at_well_distances([36.0, 35.4, 34.5, 35.4, 36.0])

    trj.calc_agarose_dual_circle_episodes(
        delta_mm=1.0, boundary_policy="legacy"
    )

    expected_width = 0.1 * CT.large.pxPerMmFloor()
    np.testing.assert_allclose(border_widths, expected_width)
    assert [
        (ep["start"], ep["stop"], ep["avoids_inner"])
        for ep in trj.agarose_dual_circle_episodes
    ] == [(1, 4, True)]
    assert trj.agarose_dual_circle_geometry["boundary_policy"] == "legacy"


def test_agarose_dual_circle_rejects_unknown_boundary_policy():
    trj = _trajectory_at_well_distances([41.0, 34.0, 41.0])

    with pytest.raises(ValueError, match="boundary policy must be one of"):
        trj.calc_agarose_dual_circle_episodes(boundary_policy="unknown")


def test_agarose_dual_circle_legacy_policy_preserves_lost_frame_splitting():
    distances = [36.0, 34.0, np.nan, 34.0, 36.0]
    legacy = _trajectory_at_well_distances(distances)
    current = _trajectory_at_well_distances(distances)

    legacy.calc_agarose_dual_circle_episodes(
        delta_mm=1.0, boundary_policy="legacy"
    )
    current.calc_agarose_dual_circle_episodes(
        delta_mm=1.0, boundary_policy="hysteretic"
    )

    assert [
        (episode["start"], episode["stop"])
        for episode in legacy.agarose_dual_circle_episodes
    ] == [(1, 2), (3, 4)]
    assert [
        (episode["start"], episode["stop"])
        for episode in current.agarose_dual_circle_episodes
    ] == [(1, 4)]


def test_agarose_virtual_control_rotates_sites_without_changing_radial_distance():
    trj = _trajectory_at_well_distances([100.0, 100.0])
    physical_radius, physical_centers = CT.large.arenaWells(
        trj.va.xf, trj.va.trxf[trj.f]
    )
    floor_tl, floor_br = CT.large.floor(trj.va.xf, f=trj.va.trxf[trj.f])
    arena_center = 0.5 * (np.asarray(floor_tl) + np.asarray(floor_br))
    trj.calc_agarose_dual_circle_episodes(
        delta_mm=1.0, center_rotation_deg=45.0
    )

    virtual_centers = np.asarray(
        trj.agarose_dual_circle_geometry["centers_px"], dtype=float
    )
    physical_centers = np.asarray(physical_centers, dtype=float)
    np.testing.assert_allclose(
        np.linalg.norm(virtual_centers - arena_center, axis=1),
        np.linalg.norm(physical_centers - arena_center, axis=1),
    )
    np.testing.assert_allclose(
        trj.agarose_dual_circle_geometry["inner_radius_px"], physical_radius
    )
    np.testing.assert_allclose(
        trj.agarose_dual_circle_geometry["nominal_agarose_radius_px"],
        physical_radius,
    )
    # No rotated site lies on a physical agarose center.
    assert np.min(
        np.linalg.norm(
            virtual_centers[:, np.newaxis, :] - physical_centers[np.newaxis, :, :],
            axis=2,
        )
    ) > physical_radius


def test_agarose_dual_circle_accepts_explicit_site_and_frame_window():
    trj = _trajectory_at_well_distances(
        [41.0, 31.0, 33.0, 41.0, 31.0, 33.0, 41.0]
    )
    _radius, centers = CT.large.arenaWells(trj.va.xf, trj.va.trxf[trj.f])

    trj.calc_agarose_dual_circle_episodes(
        delta_mm=1.0,
        centers_override=(centers[0],),
        site_labels=("virtual_site1",),
        frame_range=(3, 7),
    )

    assert [
        (ep["start"], ep["stop"], ep["start_well_labels"])
        for ep in trj.agarose_dual_circle_episodes
    ] == [(4, 6, ("virtual_site1",))]
    assert trj.agarose_dual_circle_geometry["centers_px"] == (centers[0],)


def test_agarose_dual_circle_radius_offsets_preserve_concentric_centers():
    trj = _trajectory_at_well_distances([100.0, 100.0])
    physical_radius, physical_centers = CT.large.arenaWells(
        trj.va.xf, trj.va.trxf[trj.f]
    )
    trj.calc_agarose_dual_circle_episodes(
        delta_mm=2.0, inner_radius_offset_mm=1.0
    )

    analysis_centers = np.asarray(
        trj.agarose_dual_circle_geometry["centers_px"], dtype=float
    )
    physical_centers = np.asarray(physical_centers, dtype=float)
    np.testing.assert_allclose(analysis_centers, physical_centers)
    np.testing.assert_allclose(
        trj.agarose_dual_circle_geometry["inner_radius_px"],
        physical_radius + CT.large.pxPerMmFloor(),
    )
    np.testing.assert_allclose(
        trj.agarose_dual_circle_geometry["outer_radius_px"],
        physical_radius + 2.0 * CT.large.pxPerMmFloor(),
    )
    assert trj.agarose_dual_circle_geometry["inner_radius_offset_mm"] == 1.0
    assert trj.agarose_dual_circle_geometry["outer_radius_offset_mm"] == 2.0


def test_agarose_analysis_rejects_negative_inner_radius_offset():
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(agarose_inner_radius_offset_mm=-1.0),
        sync_bucket_ranges=[],
        trx=[],
        trns=[],
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)


def test_agarose_analysis_requires_outer_offset_beyond_inner_offset():
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_inner_radius_offset_mm=1.0,
            agarose_outer_delta_mm=1.0,
        ),
        sync_bucket_ranges=[],
        trx=[],
        trns=[],
    )

    with pytest.raises(ValueError, match="greater than"):
        VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)


def test_agarose_analysis_rejects_combining_semicircle_and_arc_filters():
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_wall_facing_entry_only=True,
            agarose_exclude_reward_facing_arc_entries=True,
        ),
        sync_bucket_ranges=[],
        trx=[],
        trns=[],
    )

    with pytest.raises(ValueError, match="cannot be enabled together"):
        VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)


def test_reward_reference_entry_alignment_is_training_specific():
    va = _video_analysis_stub(
        trns=[
            SimpleNamespace(circles=lambda _fi: [(0.0, 0.0, 2.0)]),
            SimpleNamespace(circles=lambda _fi: [(20.0, 0.0, 2.0)]),
        ]
    )
    trj = SimpleNamespace(
        agarose_dual_circle_geometry={
            "center_rotation_deg": 0.0,
            "centers_px": ((10.0, 0.0),),
        }
    )
    episode = {
        "entry_point": (11.0, 0.0),
        "start_well_labels": ("well1",),
        "entry_wall_alignment": 1.0,
        "wall_facing_entry": True,
    }

    t1_alignment, t1_wall_facing = VideoAnalysis._agaroseEpisodeWallFacingEntry(
        va, episode, trj, 0, 0, reference="reward"
    )
    t2_alignment, t2_wall_facing = VideoAnalysis._agaroseEpisodeWallFacingEntry(
        va, episode, trj, 0, 1, reference="reward"
    )

    assert t1_alignment == pytest.approx(1.0)
    assert t1_wall_facing is True
    assert t2_alignment == pytest.approx(-1.0)
    assert t2_wall_facing is False


def test_reward_facing_arc_uses_nominal_agarose_edge_and_boundary_crossing():
    va = _video_analysis_stub(
        trns=[SimpleNamespace(circles=lambda _fi: [(0.0, 0.0, 1.0)])]
    )
    trj = SimpleNamespace(
        agarose_dual_circle_geometry={
            "center_rotation_deg": 0.0,
            "nominal_centers_px": ((10.0, 0.0),),
            "centers_px": ((10.0, 0.0),),
            "nominal_agarose_radius_px": 2.0,
            "inner_radius_px": 3.0,
            "outer_radius_px": 4.0,
        }
    )
    rejected_episode = {
        "start_well_labels": ("well1",),
        "entry_outer_intersections": (
            ("well1", 6.0, 0.0, "segment_intersection"),
        ),
    }
    kept_episode = {
        "start_well_labels": ("well1",),
        "entry_outer_intersections": (
            ("well1", 14.0, 0.0, "segment_intersection"),
        ),
    }

    rejected = VideoAnalysis._agaroseEpisodeRewardArcEntry(
        va, rejected_episode, trj, 0, 0
    )
    kept = VideoAnalysis._agaroseEpisodeRewardArcEntry(
        va, kept_episode, trj, 0, 0
    )

    assert rejected["rejected"] is True
    assert rejected["kept"] is False
    assert kept["rejected"] is False
    assert kept["kept"] is True
    assert rejected["site_results"][0]["gate_radius_px"] == pytest.approx(8.0)
    assert rejected["site_results"][0]["arc_width_deg"] == pytest.approx(
        98.9168, abs=1e-4
    )


def test_agarose_ratio_uses_avoid_over_total_and_min_total_masks_ratio():
    exp = _TrajectoryEpisodes(
        [
            {"start": 1, "stop": 4, "avoids_inner": True},
            {"start": 6, "stop": 8, "avoids_inner": False},
            {"start": 7, "stop": 9, "avoids_inner": True},
        ]
    )
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            min_agarose_episodes=2,
            agarose_dual_circle_min_total=1,
            agarose_dual_circle_debug_csv=None,
            agarose_dual_circle_boundary_policy="legacy",
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10)]],
        trx=[exp],
        trns=[_Training(start=10, stop=20, name="T1")],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    counts = va.agarose_dual_circle_counts
    np.testing.assert_array_equal(counts["avoid"], [[[1, 1]]])
    np.testing.assert_array_equal(counts["total"], [[[1, 2]]])
    np.testing.assert_allclose(counts["ratio"], [[[np.nan, 0.5]]])
    assert exp.calls[0]["delta_mm"] == 1.0
    assert exp.calls[0]["boundary_policy"] == "legacy"


def test_agarose_episode_assignment_uses_episode_start_frame_for_sync_bucket():
    exp = _TrajectoryEpisodes(
        [
            {"start": 4, "stop": 8, "avoids_inner": True},
            {"start": 5, "stop": 6, "avoids_inner": False},
            {"start": 10, "stop": 12, "avoids_inner": True},
        ]
    )

    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10)], [(10, 15)]],
        trx=[exp],
        trns=[
            _Training(start=10, stop=20, name="T1"),
            _Training(start=30, stop=40, name="T2"),
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    counts = va.agarose_dual_circle_counts
    np.testing.assert_array_equal(counts["avoid"], [[[1, 0]], [[1, 0]]])
    np.testing.assert_array_equal(counts["total"], [[[1, 1]], [[1, 0]]])
    np.testing.assert_allclose(counts["ratio"], [[[1.0, 0.0]], [[1.0, np.nan]]])


def test_wall_facing_option_filters_sync_counts_before_ratio():
    exp = _TrajectoryEpisodes(
        [
            {
                "start": 1,
                "stop": 3,
                "avoids_inner": True,
                "wall_facing_entry": True,
            },
            {
                "start": 2,
                "stop": 4,
                "avoids_inner": False,
                "wall_facing_entry": False,
            },
        ]
    )
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            agarose_wall_facing_entry_only=True,
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5)]],
        trx=[exp],
        trns=[_Training(start=10, stop=20, name="T1")],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    np.testing.assert_array_equal(va.agarose_dual_circle_counts["total"], [[[1]]])
    np.testing.assert_array_equal(va.agarose_dual_circle_counts["avoid"], [[[1]]])
    np.testing.assert_allclose(va.agarose_dual_circle_counts["ratio"], [[[1.0]]])


def test_reward_referenced_wall_filter_uses_each_training_reward_center():
    exp = _TrajectoryEpisodes(
        [
            {
                "start": 1,
                "stop": 3,
                "avoids_inner": True,
                "entry_point": (11.0, 0.0),
                "start_well_labels": ("well1",),
            },
            {
                "start": 6,
                "stop": 8,
                "avoids_inner": True,
                "entry_point": (11.0, 0.0),
                "start_well_labels": ("well1",),
            },
        ]
    )
    exp.agarose_dual_circle_geometry = {
        "center_rotation_deg": 0.0,
        "centers_px": ((10.0, 0.0),),
    }
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            agarose_wall_facing_entry_only=True,
            agarose_wall_facing_reference="reward",
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5)], [(5, 10)]],
        trx=[exp],
        trns=[
            SimpleNamespace(
                start=10,
                stop=20,
                circles=lambda _fi: [(0.0, 0.0, 2.0)],
            ),
            SimpleNamespace(
                start=30,
                stop=40,
                circles=lambda _fi: [(20.0, 0.0, 2.0)],
            ),
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    np.testing.assert_array_equal(
        va.agarose_dual_circle_counts["total"], [[[1]], [[0]]]
    )
    np.testing.assert_array_equal(
        va.agarose_dual_circle_counts["avoid"], [[[1]], [[0]]]
    )


def test_reward_facing_arc_filter_excludes_only_entries_on_intersection_arc():
    exp = _TrajectoryEpisodes(
        [
            {
                "start": 1,
                "stop": 3,
                "avoids_inner": False,
                "start_well_labels": ("well1",),
                "entry_outer_intersections": (
                    ("well1", 6.0, 0.0, "segment_intersection"),
                ),
            },
            {
                "start": 2,
                "stop": 4,
                "avoids_inner": True,
                "start_well_labels": ("well1",),
                "entry_outer_intersections": (
                    ("well1", 14.0, 0.0, "segment_intersection"),
                ),
            },
        ]
    )
    exp.agarose_dual_circle_geometry = {
        "center_rotation_deg": 0.0,
        "nominal_centers_px": ((10.0, 0.0),),
        "centers_px": ((10.0, 0.0),),
        "nominal_agarose_radius_px": 2.0,
        "inner_radius_px": 3.0,
        "outer_radius_px": 4.0,
    }
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_inner_radius_offset_mm=1.0,
            agarose_outer_delta_mm=2.0,
            agarose_exclude_reward_facing_arc_entries=True,
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5)]],
        trx=[exp],
        trns=[
            SimpleNamespace(
                start=10,
                stop=20,
                circles=lambda _fi: [(0.0, 0.0, 1.0)],
            )
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    np.testing.assert_array_equal(
        va.agarose_dual_circle_counts["total"], [[[1]]]
    )
    np.testing.assert_array_equal(
        va.agarose_dual_circle_counts["avoid"], [[[1]]]
    )


def test_virtual_agarose_analysis_uses_separate_paired_result_attributes():
    exp = _TrajectoryEpisodes(
        [{"start": 1, "stop": 4, "avoids_inner": True}]
    )
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5)]],
        trx=[exp],
        trns=[_Training(start=10, stop=20, name="T1")],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(
        va, center_rotation_deg=45.0, result_prefix="agarose_virtual"
    )

    np.testing.assert_allclose(
        va.agarose_virtual_dual_circle_counts["ratio"], [[[1.0]]]
    )
    assert exp.calls == [
        {
            "delta_mm": 1.0,
            "center_rotation_deg": 45.0,
            "inner_radius_offset_mm": 0.0,
            "boundary_policy": "hysteretic",
        }
    ]
    assert va.agarose_virtual_dual_circle_geometry == {
        "center_rotation_deg": 45.0,
        "outer_delta_mm": 1.0,
        "inner_radius_offset_mm": 0.0,
        "boundary_policy": "hysteretic",
        "farthest_from_reward_only": False,
        "wall_facing_entry_only": False,
        "wall_facing_reference": "arena",
        "exclude_reward_facing_arc_entries": False,
    }


def test_reward_analytical_virtual_control_pools_one_selected_site_per_source():
    class RewardTraining(_Training):
        postStop = 10

        def circles(self, _fly):
            return [(170.0, 70.0, 10.0)]

    exp = _TrajectoryEpisodes(
        [{"start": 2, "stop": 4, "avoids_inner": True}]
    )
    exp.x = np.zeros(10)
    va = _video_analysis_stub(
        ct=CT.large,
        xf=_IdentityXformer(),
        trxf={0: None},
        opts=SimpleNamespace(
            agarose_outer_delta_mm=0.5,
            agarose_inner_radius_offset_mm=0.0,
            agarose_reward_audit_buffer_mm=1.0,
            agarose_reward_audit_max_outside_area_frac=0.25,
            agarose_reward_control_seed=101,
            min_agarose_episodes=1,
        ),
        sync_bucket_ranges=[[(0, 5)]],
        trx=[exp],
        trns=[RewardTraining(start=5, stop=8, name="T1")],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseRewardMatchedVirtualControl(va)

    assert len(exp.calls) == 4
    assert all(len(call["centers_override"]) == 1 for call in exp.calls)
    assert {call["site_labels"][0] for call in exp.calls} == {
        "virtual_site1",
        "virtual_site2",
        "virtual_site3",
        "virtual_site4",
    }
    np.testing.assert_array_equal(
        va.agarose_virtual_dual_circle_counts["total"], [[[4]]]
    )
    np.testing.assert_array_equal(
        va.agarose_virtual_dual_circle_counts["avoid"], [[[4]]]
    )
    np.testing.assert_allclose(
        va.agarose_virtual_dual_circle_counts["ratio"], [[[1.0]]]
    )
    assert va.agarose_virtual_dual_circle_geometry["method"] == (
        "reward_analytical_maximin"
    )


def test_farthest_reward_site_selection_retains_physical_tie_and_one_virtual_site():
    reward = (3.0, -3.0, 2.0)
    va = _video_analysis_stub(
        ct=CT.large2,
        xf=SimpleNamespace(fctr=1.0),
        trns=[SimpleNamespace(circles=lambda _fi: [reward])],
    )
    trj = SimpleNamespace(
        agarose_dual_circle_geometry={
            "center_rotation_deg": 0.0,
            "centers_px": ((-10, 0), (0, -10), (10, 0), (0, 10)),
        }
    )

    physical = VideoAnalysis._agaroseFarthestRewardSiteLabels(va, trj, 0)
    assert physical == [{"well1", "well4"}]

    d = 10.0 / np.sqrt(2.0)
    trj.agarose_dual_circle_geometry = {
        "center_rotation_deg": 45.0,
        "centers_px": ((-d, -d), (d, -d), (d, d), (-d, d)),
    }
    virtual = VideoAnalysis._agaroseFarthestRewardSiteLabels(va, trj, 0)
    assert virtual == [{"virtual_site4"}]


def test_farthest_reward_episode_filter_uses_training_specific_labels():
    labels = [{"well1", "well4"}, {"well2"}]
    left = {"start_well_labels": ("well1",)}
    top = {"start_well_labels": ("well2",)}

    assert VideoAnalysis._agaroseEpisodeUsesAllowedSite(left, labels, 0)
    assert not VideoAnalysis._agaroseEpisodeUsesAllowedSite(top, labels, 0)
    assert VideoAnalysis._agaroseEpisodeUsesAllowedSite(top, labels, 1)
    assert VideoAnalysis._agaroseEpisodeUsesAllowedSite(left, None, 0)


def test_wall_facing_entry_filter_is_independent_of_site_subset():
    inward = {"start_well_labels": ("well1",), "wall_facing_entry": False}
    outward = {"start_well_labels": ("well1",), "wall_facing_entry": True}

    assert not VideoAnalysis._agaroseEpisodeUsesAllowedSite(
        inward, None, 0, wall_facing_entry_only=True
    )
    assert VideoAnalysis._agaroseEpisodeUsesAllowedSite(
        outward, None, 0, wall_facing_entry_only=True
    )


def test_agarose_pre_windows_use_episode_start_frame():
    episodes = [
        {"start": 9, "stop": 12, "avoids_inner": True},
        {"start": 10, "stop": 12, "avoids_inner": False},
        {"start": 29, "stop": 31, "avoids_inner": False},
    ]
    va = _video_analysis_stub(
        opts=SimpleNamespace(min_agarose_episodes=1),
        trx=[SimpleNamespace(_bad=False)],
        trns=[
            _Training(start=10, stop=20, name="T1"),
            _Training(start=30, stop=40, name="T2"),
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    pre = VideoAnalysis._calcAgaroseDualCirclePreCounts(va, [episodes])
    training_pre = VideoAnalysis._calcAgaroseDualCircleTrainingPreCounts(va, [episodes])

    np.testing.assert_array_equal(pre["avoid"], [1])
    np.testing.assert_array_equal(pre["total"], [1])
    np.testing.assert_allclose(pre["ratio"], [1.0])
    np.testing.assert_array_equal(training_pre["avoid"], [[1], [0]])
    np.testing.assert_array_equal(training_pre["total"], [[1], [1]])
    np.testing.assert_allclose(training_pre["ratio"], [[1.0], [0.0]])


def test_agarose_bundle_extraction_keeps_exp_ctrl_axes_and_counts():
    ratio = np.asarray([[[1.0, 0.5], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[3, 2], [1, 0]]], dtype=int)
    avoid = np.asarray([[[3, 1], [0, 0]]], dtype=int)
    va = _video_analysis_stub(
        fn="fake-video",
        trns=[_Training(start=0, stop=10, name="T1")],
        agarose_dual_circle_counts={"ratio": ratio, "total": total, "avoid": avoid},
        _numRewardsMsg=lambda *args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 3, None),
    )

    ratio_exp, ratio_ctrl, total_exp, total_ctrl, avoid_exp, avoid_ctrl = (
        _extract_agarose_arrays([va])
    )

    np.testing.assert_allclose(ratio_exp, [[[1.0, 0.5, np.nan]]])
    np.testing.assert_allclose(ratio_ctrl, [[[0.0, np.nan, np.nan]]])
    np.testing.assert_array_equal(total_exp, [[[3, 2, 0]]])
    np.testing.assert_array_equal(total_ctrl, [[[1, 0, 0]]])
    np.testing.assert_array_equal(avoid_exp, [[[3, 1, 0]]])
    np.testing.assert_array_equal(avoid_ctrl, [[[0, 0, 0]]])


def test_agarose_bundle_export_records_defaults_and_min_total_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    ratio = np.asarray([[[1.0, np.nan], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[2, 1], [2, 0]]], dtype=int)
    avoid = np.asarray([[[2, 1], [0, 0]]], dtype=int)
    va = _video_analysis_stub(
        fn="fake-video",
        f=7,
        _skipped=False,
        noyc=False,
        trns=[_Training(start=0, stop=10, name="T1")],
        agarose_dual_circle_counts={"ratio": ratio, "total": total, "avoid": avoid},
        agarose_virtual_dual_circle_counts={
            "ratio": 1.0 - ratio,
            "total": total,
            "avoid": total - avoid,
        },
        _numRewardsMsg=lambda *_args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 2, None),
    )
    opts = SimpleNamespace(
        export_group_label="Intact Control>Kir",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=None,
        sli_select_keep_first_sync_buckets=None,
        min_agarose_episodes=2,
        agarose_dual_circle_min_total=1,
        agarose_sli_include_pre=False,
    )
    out = tmp_path / "agarose_bundle.npz"

    export_agarose_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert int(bundle["sli_select_skip_first_sync_buckets"]) == 0
        assert int(bundle["sli_select_keep_first_sync_buckets"]) == 0
        assert int(bundle["min_agarose_episodes"]) == 2
        assert int(bundle["agarose_dual_circle_min_total"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_min_episodes"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_unit_count"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_included_count"]) == 1
        np.testing.assert_allclose(
            bundle["agarose_virtual_ratio_exp"], [[[0.0, np.nan]]], equal_nan=True
        )
        np.testing.assert_array_equal(
            bundle["agarose_virtual_total_exp"], [[[2, 1]]]
        )
        assert float(bundle["agarose_virtual_rotation_deg"]) == 45.0
        assert not bool(bundle["agarose_wall_facing_entry_only"])
        assert float(bundle["agarose_inner_radius_offset_mm"]) == 0.0
        assert float(bundle["agarose_outer_radius_offset_mm"]) == 0.5
        assert (
            str(bundle["agarose_dual_circle_boundary_policy"].item())
            == "hysteretic"
        )
        assert str(bundle["agarose_wall_facing_reference"].item()) == "arena"
        assert not bool(bundle["agarose_exclude_reward_facing_arc_entries"])
        assert int(bundle["episode_filter_agarose_sync_exp_excluded_count"]) == 1
        np.testing.assert_array_equal(
            bundle["episode_filter_agarose_sync_exp_episode_counts"], [2, 1]
        )
        np.testing.assert_array_equal(
            bundle["episode_filter_agarose_sync_exp_excluded_episode_counts"], [1]
        )


def test_agarose_bundle_export_masks_exp_target_sync_bucket_failures(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    monkeypatch.setattr(
        "src.exporting.agarose_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.25], dtype=float),
            np.asarray([[[0.25, 0.5]]], dtype=float),
        ),
    )
    ratio = np.asarray([[[1.0, 0.5], [0.0, 1.0]]], dtype=float)
    total = np.asarray([[[2, 2], [2, 2]]], dtype=int)
    avoid = np.asarray([[[2, 1], [0, 2]]], dtype=int)
    va = _video_analysis_stub(
        fn="fake-video",
        f=7,
        _skipped=False,
        noyc=False,
        flies=[object(), object()],
        reward_exclusion_mask=[[[True, False], [False, False]]],
        sync_bucket_ranges=[[]],
        trns=[_Training(start=0, stop=10, name="T1")],
        agarose_dual_circle_counts={"ratio": ratio, "total": total, "avoid": avoid},
        _numRewardsMsg=lambda *_args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 2, None),
    )
    opts = SimpleNamespace(
        export_group_label="Intact Control>Kir",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_agarose_episodes=1,
        agarose_dual_circle_min_total=1,
        agarose_sli_include_pre=False,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
    )
    out = tmp_path / "agarose_bundle_target_sync_bucket_filtered.npz"

    export_agarose_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_reason"], ["target_sync_bucket_missing"]
        )
        assert bool(bundle["exp_target_sync_bucket_filter_enabled"])
        assert np.isnan(bundle["agarose_ratio_exp"]).all()
        np.testing.assert_allclose(bundle["agarose_ratio_ctrl"], [[[0.0, 1.0]]])
        np.testing.assert_array_equal(bundle["agarose_total_exp"], [[[2, 2]]])
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
