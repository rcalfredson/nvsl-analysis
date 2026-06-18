import json
from types import SimpleNamespace

import numpy as np

from src.exporting.turnback_home_vector_alignment_sli_bundle import (
    cosine_alignment,
    episode_home_vector_alignment,
    export_turnback_home_vector_alignment_sli_bundle,
    heading_vector_at_reentry,
    home_vector_from_reentry_to_center,
)


class _CircleTraining:
    def __init__(self, *, start=0, stop=10, circle=True, radius_px=10.0, name="T1"):
        self.start = int(start)
        self.stop = int(stop)
        self._circle = bool(circle)
        self._radius_px = float(radius_px)
        self._name = str(name)

    def isCircle(self):
        return self._circle

    def circles(self, _fly_idx):
        return [(0.0, 0.0, self._radius_px)]

    def name(self):
        return self._name

    def sname(self):
        return self.name()


class _TrajectoryEpisodes:
    def __init__(self, *, x, y, episodes, bad=False, fly_idx=0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._episodes = list(episodes)
        self._bad = bool(bad)
        self.f = int(fly_idx)
        self.calls = []

    def reward_turnback_dual_circle_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


def _blank_xy(n):
    x = np.full(int(n), np.nan, dtype=float)
    y = np.full(int(n), np.nan, dtype=float)
    return x, y


def _set_xy(x, y, coords):
    for idx, xy in coords.items():
        x[int(idx)] = float(xy[0])
        y[int(idx)] = float(xy[1])


def _episode(
    event_frame,
    *,
    start=None,
    turns_back=True,
    end_reason="reenter_inner",
    inner_radius_px=10.0,
):
    event_frame = int(event_frame)
    return {
        "start": int(event_frame - 3 if start is None else start),
        # Existing turnback episode convention: stop is one past outcome frame,
        # so the re-entry frame is stop - 1.
        "stop": int(event_frame + 1),
        "turns_back": bool(turns_back),
        "end_reason": str(end_reason),
        "inner_radius_px": float(inner_radius_px),
    }


def _default_opts(**overrides):
    opts = dict(
        export_group_label="Synthetic Group",
        min_turnback_episodes=1,
        turnback_home_vector_alignment_trainings=None,
        turnback_home_vector_alignment_skip_first_sync_buckets=None,
        turnback_home_vector_alignment_keep_first_sync_buckets=None,
        turnback_home_vector_alignment_last_sync_buckets=None,
        turnback_home_vector_alignment_window_radius_frames=1,
        turnback_home_vector_alignment_exclude_wall_contact=False,
        turnback_home_vector_alignment_inner_radius_mm=None,
        turnback_home_vector_alignment_inner_delta_mm=None,
        turnback_home_vector_alignment_outer_radius_mm=None,
        turnback_home_vector_alignment_outer_delta_mm=None,
        turnback_home_vector_alignment_border_width_mm=None,
        turnback_home_vector_alignment_inner_radius_offset_px=None,
        turnback_inner_radius_mm=None,
        turnback_inner_delta_mm=0.0,
        turnback_outer_radius_mm=None,
        turnback_outer_delta_mm=2.0,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
        turnback_home_vector_alignment_sli_group="all",
        top_sli_fraction=None,
        bottom_sli_fraction=None,
        best_worst_trn=2,
        sli_use_training_mean=False,
        sli_select_skip_first_sync_buckets=None,
        sli_select_keep_first_sync_buckets=None,
        sli_select_bucket=None,
    )
    opts.update(overrides)
    return SimpleNamespace(**opts)


def _make_export_va(*, episodes, coords, sync_bucket_ranges=None, n_frames=140):
    x, y = _blank_xy(n_frames)
    _set_xy(x, y, coords)

    trj = _TrajectoryEpisodes(x=x, y=y, episodes=episodes)
    trns = [
        _CircleTraining(start=0, stop=50, name="T1"),
        _CircleTraining(start=100, stop=130, name="T2"),
    ]

    return SimpleNamespace(
        fn="fake-video.avi",
        _skipped=False,
        noyc=True,
        trns=trns,
        trx=[trj],
        sync_bucket_ranges=(
            sync_bucket_ranges
            if sync_bucket_ranges is not None
            else [[], [(100, 110), (110, 120), (120, 130)]]
        ),
    )


def test_home_vector_points_from_reentry_position_to_reward_center():
    vec = home_vector_from_reentry_to_center(
        cx=0.0,
        cy=0.0,
        x=12.0,
        y=0.0,
    )
    assert vec == (-12.0, 0.0)


def test_cosine_alignment_reports_homeward_tangent_and_away_cases():
    np.testing.assert_allclose(cosine_alignment((-2.0, 0.0), (-10.0, 0.0)), 1.0)
    np.testing.assert_allclose(cosine_alignment((0.0, 2.0), (-10.0, 0.0)), 0.0)
    np.testing.assert_allclose(cosine_alignment((2.0, 0.0), (-10.0, 0.0)), -1.0)


def test_heading_vector_uses_windowed_displacement_and_skips_nans():
    x, y = _blank_xy(8)

    # event_frame = 3, radius = 2
    # preferred before target is frame 1 and after target is frame 5.
    # The frame near the event is NaN, but the selected window endpoints are finite.
    _set_xy(x, y, {1: (2.0, 0.0), 3: (0.0, 0.0), 4: (np.nan, np.nan), 5: (-2.0, 0.0)})
    trj = SimpleNamespace(x=x, y=y)

    heading = heading_vector_at_reentry(
        trj, event_frame=3, window_radius_frames=2, training_start=0, training_stop=8
    )

    np.testing.assert_allclose(heading, (-4.0, 0.0))


def test_heading_vector_radius_zero_falls_back_to_distinct_pre_and_post_samples():
    x, y = _blank_xy(7)
    _set_xy(x, y, {2: (2.0, 0.0), 3: (1.0, 0.0), 4: (-1.0, 0.0)})
    trj = SimpleNamespace(x=x, y=y)

    heading = heading_vector_at_reentry(
        trj, event_frame=3, window_radius_frames=0, training_start=0, training_stop=7
    )

    np.testing.assert_allclose(heading, (-3.0, 0.0))


def test_episode_home_vector_alignment_for_radial_and_tangent_reentry():
    trn = _CircleTraining(start=0, stop=10, radius_px=10.0)

    # Re-entry at right perimeter: home vector points left.
    # Heading leftward across the event gives alignment 1.
    x1, y1 = _blank_xy(10)
    _set_xy(x1, y1, {4: (12.0, 0.0), 5: (10.0, 0.0), 6: (8.0, 0.0)})
    trj_homeward = SimpleNamespace(x=x1, y=y1, f=0)
    val_homeward = episode_home_vector_alignment(
        trj_homeward, trn, _episode(5), window_radius_frames=1
    )
    np.testing.assert_allclose(val_homeward, 1.0)

    # Same re-entry point, but heading upward across the event: targent, alignment 0.
    x2, y2 = _blank_xy(10)
    _set_xy(
        x2,
        y2,
        {4: (10.0, -1.0), 5: (10.0, 0.0), 6: (10.0, 1.0)},
    )
    trj_tangent = SimpleNamespace(x=x2, y=y2, f=0)
    val_tangent = episode_home_vector_alignment(
        trj_tangent,
        trn,
        _episode(5),
        window_radius_frames=1,
    )
    np.testing.assert_allclose(val_tangent, 0.0)


def test_export_schema_uses_selected_successful_reentry_episodes_only(tmp_path):
    episodes = [
        # This success is in the first T2 sync bucket and should be skipped by
        # the metric default: skip first sync bucket, keep next four.
        _episode(105),
        # Included: radially homeward at re-entry, value 1.
        _episode(115),
        # Ignored: failure episode, despite lying in selected window.
        _episode(118, turns_back=False, end_reason="exit_outer"),
        # Included: tangent at re-entry, value 0.
        _episode(125),
    ]

    coords = {
        # Skipped sync-bucket success.
        104: (12.0, 0.0),
        105: (10.0, 0.0),
        106: (8.0, 0.0),
        # Included homeward success.
        114: (12.0, 0.0),
        115: (10.0, 0.0),
        116: (8.0, 0.0),
        # Failure episode coordinates, should not contribute.
        117: (12.0, 0.0),
        118: (10.0, 0.0),
        119: (8.0, 0.0),
        # Included tangent success.
        124: (10.0, -1.0),
        125: (10.0, 0.0),
        126: (10.0, 1.0),
    }

    va = _make_export_va(episodes=episodes, coords=coords)
    opts = _default_opts(min_turnback_episodes=2)
    out = tmp_path / "home_vector_alignment.npz"

    export_turnback_home_vector_alignment_sli_bundle(
        [va], opts, gls=None, out_fn=str(out)
    )

    with np.load(out, allow_pickle=True) as bundle:
        for key in (
            "panel_labels",
            "per_unit_values_panel",
            "per_unit_ids_panel",
            "mean",
            "ci_lo",
            "ci_hi",
            "n_units_panel",
            "meta_json",
        ):
            assert key in bundle.files

        values = bundle["per_unit_values_panel"][0]
        ids = bundle["per_unit_ids_panel"][0]

        values = np.asarray(bundle["per_unit_values_panel"][0], dtype=float)
        np.testing.assert_allclose(values, [0.5])
        np.testing.assert_allclose(bundle["mean"], [0.5])
        np.testing.assert_array_equal(bundle["n_units_panel"], [1])
        np.testing.assert_array_equal(ids, ["fake-video:fly0"])
        np.testing.assert_array_equal(
            bundle["turnback_home_vector_alignment_episode_counts"],
            [2],
        )

        meta = json.loads(bundle["meta_json"].item())
        assert meta["metric"] == "turnback_home_vector_alignment"
        assert meta["training_selection"]["trainings_effective"] == [2]
        assert meta["skip_first_sync_buckets"] == 1
        assert meta["keep_first_sync_buckets"] == 4
        assert meta["window_radius_frames"] == 1
        assert "does not use Trajectory.theta" in meta["heading_convention"]

        # The actual episode iterator should receive the same default turnback
        # geometry used by the exporter.
        trj = va.trx[0]
        assert trj.calls[0]["inner_delta_mm"] == 0.0
        assert trj.calls[0]["outer_delta_mm"] == 2.0


def test_export_min_episode_filter_excludes_low_count_fly(tmp_path):
    episodes = [
        _episode(115),
    ]
    coords = {
        114: (12.0, 0.0),
        115: (10.0, 0.0),
        116: (8.0, 0.0),
    }

    va = _make_export_va(episodes=episodes, coords=coords)
    opts = _default_opts(min_turnback_episodes=2)
    out = tmp_path / "home_vector_alignment_min_filter.npz"

    export_turnback_home_vector_alignment_sli_bundle(
        [va], opts, gls=None, out_fn=str(out)
    )

    with np.load(out, allow_pickle=True) as bundle:
        assert bundle["per_unit_values_panel"][0].size == 0
        assert bundle["per_unit_ids_panel"][0].size == 0
        np.testing.assert_array_equal(bundle["n_units_panel"], [0])
        assert np.isnan(bundle["mean"][0])


def test_export_records_top_sli_subset_metadata(tmp_path):
    episodes = [
        _episode(115),
        _episode(125),
    ]
    coords = {
        114: (12.0, 0.0),
        115: (10.0, 0.0),
        116: (8.0, 0.0),
        124: (10.0, -1.0),
        125: (10.0, 0.0),
        126: (10.0, 1.0),
    }

    va = _make_export_va(episodes=episodes, coords=coords)
    opts = _default_opts(
        turnback_home_vector_alignment_sli_group="top",
        top_sli_fraction=0.25,
        best_worst_trn=2,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=1,
        sli_select_keep_first_sync_buckets=4,
    )
    out = tmp_path / "home_vector_alignment_top25.npz"

    export_turnback_home_vector_alignment_sli_bundle(
        [va], opts, gls=None, out_fn=str(out)
    )

    with np.load(out, allow_pickle=True) as bundle:
        meta = json.loads(bundle["meta_json"].item())
        assert meta["sli_group"] == "top"
        assert meta["top_sli_fraction"] == 0.25
        assert meta["subset_label"] == "top 25% SLI flies"
        assert meta["sli_selection"]["training"] == 2
        assert meta["sli_selection"]["use_training_mean"] is True
        assert meta["sli_selection"]["skip_first_sync_buckets"] == 1
        assert meta["sli_selection"]["keep_first_sync_buckets"] == 4
