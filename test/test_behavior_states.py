from types import SimpleNamespace

import numpy as np

from src.analysis.behavior_states import (
    BehaviorState,
    BehaviorStateConfig,
    DEFAULT_BEHAVIOR_STATE_CONFIG,
    SchmittButterworthConfig,
    _reference_savgol_filter,
    analyze_trajectory_behavior_states,
    absorb_sharp_turn_gaps,
    behavior_state_config_from_opts,
    clean_schmitt_bout_durations,
    classify_behavior_states,
    consolidate_schmitt_butterworth_peaks,
    extend_schmitt_stops,
    filter_short_turn_segments,
    find_turns,
    path_angular_speed_rad_s,
    pivot_pause_turn_mask,
    raw_segment_speed_mm_s,
    schmitt,
    schmitt_butterworth_path_angular_velocity_deg_s,
    schmitt_walking_mask,
    signed_turn_angle_deg,
    strict_turn_peak_indices,
)
from src.analysis.random_frame_windows import (
    sample_non_overlapping_frame_windows,
    sample_non_overlapping_frame_windows_from_domains,
)
from src.plotting.event_chain_plotter import EventChainPlotter


def _reference_find_turns(angular_speed, head_speed, body_speed):
    cfg = DEFAULT_BEHAVIOR_STATE_CONFIG
    turns_mask1 = (
        (np.abs(angular_speed) > cfg.angular_small_turn_rad_s).astype(float)
        + 1.5 * (np.abs(angular_speed) > cfg.angular_large_turn_rad_s).astype(float)
    )
    turns_mask1 = np.roll(turns_mask1, -2)
    turns_mask1[-2:] = 0

    hbdiff = np.roll(head_speed - body_speed, -4)
    hbdiff[-4:] = 0
    turns_mask2 = (
        0.5 * (hbdiff > cfg.head_body_small_turn_mm_s).astype(float)
        + (hbdiff > cfg.head_body_large_turn_mm_s).astype(float)
    )
    return (
        _reference_savgol_filter(turns_mask1)
        + _reference_savgol_filter(turns_mask2)
        >= cfg.turn_score_threshold
    )


def test_behavior_state_plot_window_expands_clipped_turn_boundaries():
    states = np.array(
        [
            BehaviorState.RUN,
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.RUN,
            BehaviorState.RUN,
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.RUN,
        ],
        dtype=np.int8,
    )

    start, stop = EventChainPlotter._expand_behavior_state_turn_window(
        states, 2, 6, turn_boundary_pad=1
    )

    assert (start, stop) == (0, 8)


def test_behavior_state_plot_window_does_not_expand_unclipped_boundaries():
    states = np.array(
        [
            BehaviorState.RUN,
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.RUN,
            BehaviorState.RUN,
        ],
        dtype=np.int8,
    )

    start, stop = EventChainPlotter._expand_behavior_state_turn_window(
        states, 0, 4, turn_boundary_pad=2
    )

    assert (start, stop) == (0, 4)


def test_behavior_state_plot_expansion_is_clamped_to_sampling_domain():
    plotter = object.__new__(EventChainPlotter)
    plotter.va = SimpleNamespace(
        opts=SimpleNamespace(behavior_state_plot_turn_boundary_pad=2),
        trns=[],
    )
    states = np.array(
        [
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.TURN,
            BehaviorState.RUN,
            BehaviorState.RUN,
            BehaviorState.RUN,
            BehaviorState.RUN,
        ],
        dtype=np.int8,
    )

    resolved = plotter._resolve_behavior_state_plot_window(
        states,
        len(states),
        start_frame=2,
        stop_frame=5,
        frame_domain=(2, 6),
    )

    assert resolved == (2, 5)


def test_random_frame_windows_are_seeded_and_non_overlapping():
    kwargs = dict(
        domain_start=10,
        domain_stop=1000,
        window_span=100,
        count=6,
    )

    windows = sample_non_overlapping_frame_windows(
        **kwargs, rng=np.random.default_rng(42)
    )
    repeated = sample_non_overlapping_frame_windows(
        **kwargs, rng=np.random.default_rng(42)
    )

    assert windows == repeated
    assert all(stop - start == 100 for start, stop in windows)
    assert all(left[1] < right[0] for left, right in zip(windows, windows[1:]))


def test_random_base_windows_do_not_depend_on_post_sampling_expansion():
    kwargs = dict(
        domain_start=10,
        domain_stop=500,
        window_span=40,
        count=5,
    )
    windows_a = sample_non_overlapping_frame_windows(
        **kwargs, rng=np.random.default_rng(19)
    )
    windows_b = sample_non_overlapping_frame_windows(
        **kwargs, rng=np.random.default_rng(19)
    )

    rendered_a = [(start - 2, stop + 2) for start, stop in windows_a]
    rendered_b = [(start - 7, stop + 7) for start, stop in windows_b]

    assert windows_a == windows_b
    assert rendered_a != rendered_b


def test_random_frame_windows_respect_expanded_output_bounds():
    def resolve(start, stop):
        return start - 3, stop + 3

    windows = sample_non_overlapping_frame_windows(
        domain_start=10,
        domain_stop=500,
        window_span=40,
        count=8,
        rng=np.random.default_rng(7),
        resolve_bounds=resolve,
    )
    resolved = [resolve(*window) for window in windows]

    assert all(left[1] < right[0] for left, right in zip(resolved, resolved[1:]))


def test_random_frame_windows_stay_wholly_within_separate_trainings():
    trainings = [(100, 250), (500, 750)]

    windows = sample_non_overlapping_frame_windows_from_domains(
        frame_domains=trainings,
        window_span=50,
        count=5,
        rng=np.random.default_rng(11),
        resolve_bounds=lambda start, stop: (start - 2, stop + 2),
    )

    for start, stop in windows:
        expanded = (start - 2, stop + 2)
        assert any(
            training_start <= expanded[0] and expanded[1] <= training_stop
            for training_start, training_stop in trainings
        )


def test_random_frame_windows_report_when_too_many_are_requested():
    with np.testing.assert_raises_regex(ValueError, "at most 2 fit"):
        sample_non_overlapping_frame_windows(
            domain_start=0,
            domain_stop=21,
            window_span=10,
            count=3,
            rng=np.random.default_rng(1),
        )


def test_find_turns_matches_reference_two_signal_scoring():
    n = 24
    angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)

    angular_speed[8:11] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    head_speed[10:14] = body_speed[10:14] + 2.0

    expected = _reference_find_turns(angular_speed, head_speed, body_speed)

    np.testing.assert_array_equal(
        find_turns(angular_speed, head_speed, body_speed), expected
    )
    assert np.any(expected)


def test_find_turns_uses_path_angular_score_when_theta_is_flat():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    path_angular_speed[5:8] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        savgol_window=99,
        angular_mask_shift_frames=0,
        turn_min_segments=1,
        turn_score_threshold=2.0,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    expected = np.zeros(n, dtype=bool)
    expected[5:8] = True
    np.testing.assert_array_equal(turns, expected)


def test_find_turns_can_shift_path_angular_score_when_requested():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    path_angular_speed[5:8] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        savgol_window=99,
        angular_mask_shift_frames=0,
        path_angular_mask_shift_frames=2,
        turn_min_segments=1,
        turn_score_threshold=2.0,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    expected = np.zeros(n, dtype=bool)
    expected[3:6] = True
    np.testing.assert_array_equal(turns, expected)


def test_find_turns_path_source_ignores_theta_angular_score():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    angular_speed[5:8] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        turn_angular_source="path",
        savgol_window=99,
        angular_mask_shift_frames=0,
        turn_score_threshold=2.0,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    np.testing.assert_array_equal(turns, np.zeros(n, dtype=bool))


def test_find_turns_path_no_head_body_ignores_head_body_score():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n) * 5.0
    body_speed = np.ones(n)
    config = BehaviorStateConfig(
        turn_angular_source="path_no_head_body",
        savgol_window=99,
        angular_mask_shift_frames=0,
        turn_score_threshold=1.0,
        turn_min_segments=1,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    np.testing.assert_array_equal(turns, np.zeros(n, dtype=bool))


def test_find_turns_can_expand_turn_to_largest_path_vertex_incoming_segment():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    path_angular_speed[5] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 1.0
    path_angular_speed[6] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        turn_angular_source="path_no_head_body",
        savgol_window=99,
        angular_mask_shift_frames=0,
        path_angular_mask_shift_frames=0,
        turn_score_threshold=2.0,
        turn_min_segments=2,
        turn_expand_largest_vertex=True,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    expected = np.zeros(n, dtype=bool)
    expected[4:7] = True
    np.testing.assert_array_equal(turns, expected)


def test_find_turns_can_expand_turn_to_sharp_start_boundary_vertex():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    path_angular_speed[5] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    path_angular_speed[6] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 1.0
    config = BehaviorStateConfig(
        turn_angular_source="path_no_head_body",
        savgol_window=99,
        angular_mask_shift_frames=0,
        path_angular_mask_shift_frames=0,
        turn_score_threshold=2.0,
        turn_min_segments=2,
        turn_expand_largest_vertex=True,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    expected = np.zeros(n, dtype=bool)
    expected[4:7] = True
    np.testing.assert_array_equal(turns, expected)


def test_absorb_sharp_turn_gaps_fills_short_gap_between_turns():
    turn_mask = np.zeros(12, dtype=bool)
    turn_mask[3:5] = True
    turn_mask[6:8] = True
    path_angular_speed = np.zeros(12)
    path_angular_speed[3:5] = 2.0
    path_angular_speed[5] = 2.5
    path_angular_speed[6:8] = 2.1

    turns = absorb_sharp_turn_gaps(
        turn_mask,
        path_angular_speed,
        max_gap_segments=1,
        min_peak_ratio=1.0,
        min_segments=2,
    )

    expected = np.zeros(12, dtype=bool)
    expected[3:8] = True
    np.testing.assert_array_equal(turns, expected)


def test_absorb_sharp_turn_gaps_fills_gap_attached_to_sharp_boundary_vertex():
    turn_mask = np.zeros(12, dtype=bool)
    turn_mask[3:5] = True
    turn_mask[6:9] = True
    path_angular_speed = np.zeros(12)
    path_angular_speed[3:5] = 1.5
    path_angular_speed[5] = 0.2
    path_angular_speed[6] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    path_angular_speed[7:9] = 1.5

    turns = absorb_sharp_turn_gaps(
        turn_mask,
        path_angular_speed,
        max_gap_segments=1,
        min_peak_ratio=1.0,
        min_segments=2,
    )

    expected = np.zeros(12, dtype=bool)
    expected[3:9] = True
    np.testing.assert_array_equal(turns, expected)


def test_absorb_sharp_turn_gaps_splits_opposing_direction_island_at_weakest_vertex():
    turn_mask = np.zeros(14, dtype=bool)
    turn_mask[3:5] = True
    turn_mask[6:9] = True
    path_angular_speed = np.zeros(14)
    path_angular_speed[3:5] = 2.0
    path_angular_speed[5] = 2.5
    path_angular_speed[6] = -0.2
    path_angular_speed[7:9] = -2.1

    turns = absorb_sharp_turn_gaps(
        turn_mask,
        path_angular_speed,
        max_gap_segments=1,
        min_peak_ratio=1.0,
        min_segments=2,
    )

    expected = np.zeros(14, dtype=bool)
    expected[3:5] = True
    expected[6:9] = True
    np.testing.assert_array_equal(turns, expected)


def test_find_turns_path_source_still_uses_head_body_score():
    n = 16
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n) * 5.0
    body_speed = np.ones(n)
    config = BehaviorStateConfig(
        turn_angular_source="path",
        savgol_window=99,
        angular_mask_shift_frames=0,
        head_body_shift_frames=0,
        turn_score_threshold=1.0,
        turn_min_segments=1,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )

    assert np.any(turns)


def test_pivot_pause_turn_mask_includes_incoming_short_and_outgoing_segments():
    x = np.array([0.0, 10.0, 10.5, 10.5, 20.5])
    y = np.array([0.0, 0.0, 0.5, 10.5, 10.5])
    segment_speed = raw_segment_speed_mm_s(x, y, fps=10.0, px_per_mm=10.0)
    body_speed = np.full_like(x, 3.0)
    config = BehaviorStateConfig(
        turn_rescue_pivot_pauses=True,
        turn_path_min_segment_speed_mm_s=2.0,
        turn_pivot_min_angle_rad=np.deg2rad(80.0),
    )

    mask = pivot_pause_turn_mask(
        x, y, segment_speed, body_speed_mm_s=body_speed, config=config
    )

    expected = np.zeros_like(mask, dtype=bool)
    expected[0:3] = True
    np.testing.assert_array_equal(mask, expected)


def test_pivot_pause_turn_mask_can_include_two_short_segments():
    x = np.array([0.0, 10.0, 10.5, 11.0, 11.0, 21.0])
    y = np.array([0.0, 0.0, 0.5, 1.0, 11.0, 11.0])
    segment_speed = raw_segment_speed_mm_s(x, y, fps=10.0, px_per_mm=10.0)
    body_speed = np.full_like(x, 3.0)
    config = BehaviorStateConfig(
        turn_rescue_pivot_pauses=True,
        turn_path_min_segment_speed_mm_s=2.0,
        turn_pivot_min_angle_rad=np.deg2rad(80.0),
        turn_pivot_max_short_segments=2,
    )

    mask = pivot_pause_turn_mask(
        x, y, segment_speed, body_speed_mm_s=body_speed, config=config
    )

    expected = np.zeros_like(mask, dtype=bool)
    expected[0:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_pivot_pause_turn_mask_rejects_rest_like_body_speed():
    x = np.array([0.0, 10.0, 10.5, 11.0, 11.0, 21.0])
    y = np.array([0.0, 0.0, 0.5, 1.0, 11.0, 11.0])
    segment_speed = raw_segment_speed_mm_s(x, y, fps=10.0, px_per_mm=10.0)
    body_speed = np.full_like(x, 0.5)
    config = BehaviorStateConfig(
        turn_rescue_pivot_pauses=True,
        turn_path_min_segment_speed_mm_s=2.0,
        turn_pivot_min_angle_rad=np.deg2rad(80.0),
        turn_pivot_max_short_segments=2,
    )

    mask = pivot_pause_turn_mask(
        x, y, segment_speed, body_speed_mm_s=body_speed, config=config
    )

    np.testing.assert_array_equal(mask, np.zeros_like(mask, dtype=bool))


def test_find_turns_can_rescue_pivot_pause_mask():
    n = 8
    angular_speed = np.zeros(n)
    path_angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    pivot_mask = np.zeros(n, dtype=bool)
    pivot_mask[2:5] = True
    config = BehaviorStateConfig(
        turn_angular_source="path_no_head_body",
        turn_rescue_pivot_pauses=True,
    )

    turns = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
        pivot_turn_mask=pivot_mask,
    )

    np.testing.assert_array_equal(turns, pivot_mask)


def test_filter_short_turn_segments_removes_one_segment_turns_by_default():
    turn_mask = np.zeros(8, dtype=bool)
    turn_mask[2] = True
    turn_mask[4:6] = True

    filtered = filter_short_turn_segments(turn_mask)

    expected = np.zeros(8, dtype=bool)
    expected[4:6] = True
    np.testing.assert_array_equal(filtered, expected)


def test_find_turns_can_keep_one_segment_turns_when_configured():
    n = 12
    angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    angular_speed[5] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        savgol_window=99,
        angular_mask_shift_frames=0,
        turn_min_segments=1,
        turn_score_threshold=2.0,
    )

    turns = find_turns(angular_speed, head_speed, body_speed, config=config)

    expected = np.zeros(n, dtype=bool)
    expected[5] = True
    np.testing.assert_array_equal(turns, expected)


def test_find_turns_drops_one_segment_turns_by_default():
    n = 12
    angular_speed = np.zeros(n)
    head_speed = np.ones(n)
    body_speed = np.ones(n)
    angular_speed[5] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    config = BehaviorStateConfig(
        savgol_window=99,
        angular_mask_shift_frames=0,
        turn_score_threshold=2.0,
    )

    turns = find_turns(angular_speed, head_speed, body_speed, config=config)

    np.testing.assert_array_equal(turns, np.zeros(n, dtype=bool))


def test_behavior_state_config_from_opts_sets_core_turn_thresholds():
    class Opts:
        behavior_state_turn_angular_source = "path_no_head_body"
        behavior_state_turn_path_angular_alignment = "segment_end"
        behavior_state_turn_path_min_speed_mm_s = 3.0
        behavior_state_turn_path_min_segment_speed_mm_s = 2.5
        behavior_state_turn_path_angular_shift_frames = 1
        behavior_state_turn_angular_small_deg_s = 100.0
        behavior_state_turn_angular_large_deg_s = 150.0
        behavior_state_turn_score_threshold = 1.4
        behavior_state_turn_min_segments = 3
        behavior_state_turn_expand_largest_vertex = True
        behavior_state_turn_absorb_sharp_gaps = True
        behavior_state_turn_sharp_gap_max_segments = 4
        behavior_state_turn_sharp_gap_min_peak_ratio = 0.9
        behavior_state_turn_rescue_pivot_pauses = True
        behavior_state_turn_pivot_min_angle_deg = 100.0
        behavior_state_turn_pivot_max_short_segments = 2
        behavior_state_turn_pivot_short_max_speed_mm_s = 1.0
        behavior_state_turn_pivot_flank_min_speed_mm_s = 2.0

    config = behavior_state_config_from_opts(Opts())

    assert config.turn_angular_source == "path_no_head_body"
    assert config.turn_path_angular_alignment == "segment_end"
    assert config.turn_path_min_speed_mm_s == 3.0
    assert config.turn_path_min_segment_speed_mm_s == 2.5
    assert config.path_angular_mask_shift_frames == 1
    assert np.isclose(config.angular_small_turn_rad_s, np.deg2rad(100.0))
    assert np.isclose(config.angular_large_turn_rad_s, np.deg2rad(150.0))
    assert config.turn_score_threshold == 1.4
    assert config.turn_min_segments == 3
    assert config.turn_expand_largest_vertex is True
    assert config.turn_absorb_sharp_gaps is True
    assert config.turn_sharp_gap_max_segments == 4
    assert config.turn_sharp_gap_min_peak_ratio == 0.9
    assert config.turn_rescue_pivot_pauses is True
    assert np.isclose(config.turn_pivot_min_angle_rad, np.deg2rad(100.0))
    assert config.turn_pivot_max_short_segments == 2
    assert config.turn_pivot_short_segment_max_speed_mm_s == 1.0
    assert config.turn_pivot_flank_min_speed_mm_s == 2.0


def test_classify_behavior_states_uses_turns_before_run_schmitt():
    n = 32
    body_speed = np.r_[np.zeros(8), np.full(16, 4.0), np.zeros(8)]
    head_speed = body_speed.copy()
    angular_speed = np.zeros(n)
    angular_speed[12:15] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    head_speed[14:18] = body_speed[14:18] + 2.0

    states = classify_behavior_states(body_speed, angular_speed, head_speed)
    turn_mask = find_turns(angular_speed, head_speed, body_speed)
    walking = schmitt(_reference_savgol_filter(body_speed), high=3.0, low=1.0) > 0

    np.testing.assert_array_equal(states == BehaviorState.TURN, turn_mask)
    np.testing.assert_array_equal(states == BehaviorState.RUN, walking & ~turn_mask)
    np.testing.assert_array_equal(states == BehaviorState.REST, ~(walking | turn_mask))


def test_path_angular_speed_detects_movement_bearing_change():
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    body_speed = np.full_like(x, 2.0)
    config = BehaviorStateConfig(savgol_window=99)

    path_speed = path_angular_speed_rad_s(
        x, y, fps=10.0, body_speed_mm_s=body_speed, config=config
    )

    assert np.isfinite(path_speed[2])
    assert path_speed[2] > 0
    assert path_speed[3] == 0


def test_path_angular_speed_can_use_segment_end_alignment():
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    body_speed = np.full_like(x, 2.0)
    config = BehaviorStateConfig(
        savgol_window=99, turn_path_angular_alignment="segment_end"
    )

    path_speed = path_angular_speed_rad_s(
        x, y, fps=10.0, body_speed_mm_s=body_speed, config=config
    )

    assert path_speed[2] == 0
    assert np.isfinite(path_speed[3])
    assert path_speed[3] > 0


def test_path_angular_speed_uses_core_path_min_speed_gate():
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    body_speed = np.full_like(x, 1.5)
    config = BehaviorStateConfig(savgol_window=99, turn_path_min_speed_mm_s=2.0)

    path_speed = path_angular_speed_rad_s(
        x, y, fps=10.0, body_speed_mm_s=body_speed, config=config
    )

    assert np.all(~np.isfinite(path_speed))


def test_path_angular_speed_requires_adjacent_raw_segments_when_px_per_mm_available():
    x = np.array([0.0, 10.0, 20.0, 20.5, 20.5])
    y = np.array([0.0, 0.0, 0.0, 0.5, 10.5])
    body_speed = np.full_like(x, 10.0)
    config = BehaviorStateConfig(
        savgol_window=99,
        turn_path_min_speed_mm_s=2.0,
    )

    path_speed = path_angular_speed_rad_s(
        x,
        y,
        fps=10.0,
        px_per_mm=10.0,
        body_speed_mm_s=body_speed,
        config=config,
    )

    assert not np.isfinite(path_speed[2])


def test_analyze_trajectory_behavior_states_attaches_reference_signals():
    class ChamberType:
        def pxPerMmFloor(self):
            return 10.0

    class Xformer:
        fctr = 1.0

    class Video:
        fps = 10.0
        ct = ChamberType()
        xf = Xformer()

    class Trajectory:
        va = Video()
        x = np.arange(20, dtype=float)
        y = np.zeros(20, dtype=float)
        h = np.full(20, 4.0)
        theta = np.r_[np.zeros(10), np.full(10, 90.0)]

        def bad(self):
            return False

    trj = Trajectory()

    states = analyze_trajectory_behavior_states(trj)

    assert states is trj.behavior_state
    assert states.shape == trj.x.shape
    assert trj.behavior_body_speed_mm_s.shape == trj.x.shape
    assert trj.behavior_head_speed_mm_s.shape == trj.x.shape
    assert trj.behavior_angular_speed_rad_s.shape == trj.x.shape
    assert trj.behavior_path_angular_speed_rad_s.shape == trj.x.shape
    assert trj.behavior_turn_mask.dtype == bool


def test_schmitt_butterworth_config_from_opts_uses_low_fps_defaults_and_overrides():
    class Opts:
        behavior_state_detector = "schmitt_butterworth"
        behavior_state_sb_butterworth_cutoff_hz = 1.5
        behavior_state_sb_angular_moving_average_frames = 2

    config = behavior_state_config_from_opts(Opts())

    assert isinstance(config, SchmittButterworthConfig)
    assert config.position_savgol_window == 3
    assert config.position_savgol_order == 1
    assert config.butterworth_cutoff_hz == 1.5
    assert config.angular_moving_average_frames == 2
    assert config.turn_flank_frames == 2
    assert config.turn_peak_threshold_deg_s == 120.0


def test_schmitt_walking_mask_uses_inclusive_thresholds_and_stopped_initial_state():
    speed = np.array([1.5, 2.0, 1.5, 1.0, 1.5])

    walking = schmitt_walking_mask(speed)

    np.testing.assert_array_equal(walking, [False, True, True, False, False])


def test_schmitt_stop_extension_is_fixed_anchor_contiguous_and_single_pass():
    walking = np.array([True, True, False, False, True, True])
    x = np.array([0.0, 0.4, 0.4, 0.4, 0.8, 1.5])
    y = np.zeros_like(x)

    extended = extend_schmitt_stops(
        walking,
        x,
        y,
        fps=2.0,
        px_per_mm=1.0,
    )

    np.testing.assert_array_equal(extended, [False, False, False, False, False, True])


def test_schmitt_short_bout_cleanup_merges_and_terminates():
    config = SchmittButterworthConfig(
        stopped_min_duration_s=0.2,
        walking_min_duration_s=0.2,
    )
    walking = np.array([True, True, False, True, True])

    cleaned = clean_schmitt_bout_durations(walking, fps=10.0, config=config)

    np.testing.assert_array_equal(cleaned, np.ones(5, dtype=bool))


def test_schmitt_bout_cleanup_uses_valid_sample_durations():
    config = SchmittButterworthConfig(
        stopped_min_duration_s=0.2,
        walking_min_duration_s=0.05,
    )
    walking = np.array([True, False, True])

    fixed_rate = clean_schmitt_bout_durations(
        walking, fps=10.0, config=config
    )
    timestamped = clean_schmitt_bout_durations(
        walking,
        fps=10.0,
        dt_s=np.array([0.1, 0.25, 0.1]),
        config=config,
    )

    np.testing.assert_array_equal(fixed_rate, [True, True, True])
    np.testing.assert_array_equal(timestamped, walking)


def test_schmitt_path_angular_velocity_uses_cartesian_cw_ccw_sign():
    ccw = schmitt_butterworth_path_angular_velocity_deg_s(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        fps=1.0,
    )
    cw = schmitt_butterworth_path_angular_velocity_deg_s(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        fps=1.0,
    )

    assert np.isclose(ccw[1], 90.0)
    assert np.isclose(cw[1], -90.0)


def test_schmitt_peak_detection_is_inclusive_but_rejects_plateaus():
    signal = np.array([0.0, 100.0, 120.0, 100.0, 130.0, 130.0, 0.0])

    peaks = strict_turn_peak_indices(
        signal, threshold_deg_s=120.0, flank_frames=1
    )

    np.testing.assert_array_equal(peaks, [2])


def test_schmitt_signed_turn_angle_uses_forward_parametric_flanks():
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, -1.0, -2.0])

    angle = signed_turn_angle_deg(
        x, y, first_peak=2, last_peak=2, flank_frames=2
    )

    assert np.isclose(angle, 90.0)


def test_schmitt_peak_consolidation_is_transitive_and_uses_largest_peak():
    x = np.zeros(9)
    y = np.zeros(9)
    x[[2, 4, 6]] = [0.0, 0.4, 0.8]
    angular = np.zeros(9)
    angular[[2, 4, 6]] = [130.0, -180.0, 150.0]
    config = SchmittButterworthConfig(
        turn_flank_frames=2, turn_merge_radius_mm=0.5
    )

    events = consolidate_schmitt_butterworth_peaks(
        np.array([2, 4, 6]),
        x,
        y,
        angular,
        px_per_mm=1.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0]["peaks"] == (2, 4, 6)
    assert events[0]["representative_peak"] == 4
    assert (events[0]["start"], events[0]["stop"]) == (0, 8)


def test_schmitt_butterworth_analysis_requires_and_uses_wall_regions():
    class ChamberType:
        def pxPerMmFloor(self):
            return 1.0

    class Xformer:
        fctr = 1.0

    class Video:
        fps = 7.5
        ct = ChamberType()
        xf = Xformer()

    class Trajectory:
        va = Video()
        x = np.arange(30, dtype=float)
        y = np.zeros(30, dtype=float)
        boundary_event_stats = {
            "wall": {"all": {"edge": {"boundary_contact_regions": []}}}
        }

        def bad(self):
            return False

    trj = Trajectory()
    states = analyze_trajectory_behavior_states(
        trj, config=SchmittButterworthConfig()
    )

    np.testing.assert_array_equal(states, np.full(30, BehaviorState.RUN))
    assert trj.behavior_detector == "schmitt_butterworth"
    assert trj.behavior_sb_angular_velocity_deg_s.shape == (30,)
    assert trj.behavior_sb_turn_events == []


def test_schmitt_butterworth_rejects_turn_interval_at_wall():
    class ChamberType:
        def pxPerMmFloor(self):
            return 1.0

    class Xformer:
        fctr = 1.0

    class Video:
        fps = 7.5
        ct = ChamberType()
        xf = Xformer()

    class Trajectory:
        va = Video()
        x = np.r_[np.arange(15, dtype=float), np.full(15, 14.0)]
        y = np.r_[np.zeros(15), -np.arange(1, 16, dtype=float)]

        def bad(self):
            return False

    config = SchmittButterworthConfig(
        butterworth_cutoff_hz=3.0,
        turn_peak_threshold_deg_s=30.0,
    )
    away = Trajectory()
    away.boundary_event_stats = {
        "wall": {"all": {"edge": {"boundary_contact_regions": []}}}
    }
    at_wall = Trajectory()
    at_wall.boundary_event_stats = {
        "wall": {
            "all": {"edge": {"boundary_contact_regions": [slice(13, 17)]}}
        }
    }

    away_states = analyze_trajectory_behavior_states(away, config=config)
    wall_states = analyze_trajectory_behavior_states(at_wall, config=config)

    assert np.count_nonzero(away_states == BehaviorState.TURN) == 5
    assert np.count_nonzero(wall_states == BehaviorState.TURN) == 0
    assert at_wall.behavior_sb_turn_events[0]["rejection_reason"] == "wall_contact"
