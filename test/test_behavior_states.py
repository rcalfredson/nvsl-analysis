import numpy as np

from src.analysis.behavior_states import (
    BehaviorState,
    DEFAULT_BEHAVIOR_STATE_CONFIG,
    _savgol_like_reference,
    analyze_trajectory_behavior_states,
    classify_behavior_states,
    find_turns,
    schmitt,
)


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
        _savgol_like_reference(turns_mask1)
        + _savgol_like_reference(turns_mask2)
        >= cfg.turn_score_threshold
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


def test_classify_behavior_states_uses_turns_before_run_schmitt():
    n = 32
    body_speed = np.r_[np.zeros(8), np.full(16, 4.0), np.zeros(8)]
    head_speed = body_speed.copy()
    angular_speed = np.zeros(n)
    angular_speed[12:15] = DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s + 0.2
    head_speed[14:18] = body_speed[14:18] + 2.0

    states = classify_behavior_states(body_speed, angular_speed, head_speed)
    turn_mask = find_turns(angular_speed, head_speed, body_speed)
    walking = schmitt(_savgol_like_reference(body_speed), high=3.0, low=1.0) > 0

    np.testing.assert_array_equal(states == BehaviorState.TURN, turn_mask)
    np.testing.assert_array_equal(states == BehaviorState.RUN, walking & ~turn_mask)
    np.testing.assert_array_equal(states == BehaviorState.REST, ~(walking | turn_mask))


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
    assert trj.behavior_turn_mask.dtype == bool
