import numpy as np

from src.analysis.behavior_states import (
    BehaviorState,
    BehaviorStateConfig,
    DEFAULT_BEHAVIOR_STATE_CONFIG,
    _reference_savgol_filter,
    analyze_trajectory_behavior_states,
    absorb_sharp_turn_gaps,
    behavior_state_config_from_opts,
    classify_behavior_states,
    filter_short_turn_segments,
    find_turns,
    path_angular_speed_rad_s,
    schmitt,
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
