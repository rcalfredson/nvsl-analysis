from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.trajectory import Trajectory


def test_calc_in_circle_distinguishes_inside_border_and_outside():
    states = Trajectory.calc_in_circle(
        None,
        np.array([0, 9.999, 10, 10.999, 11]),
        np.zeros(5),
        0,
        0,
        10,
    )

    assert states.tolist() == [2, 2, 1, 1, 0]


def test_hysteretic_circle_mask_requires_crossing_the_full_border():
    states = np.array([2, 1, 2, 1, 0, 1, 0, 1, 2])

    inside = Trajectory._hysteretic_circle_inside(states)

    assert inside.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_hysteretic_circle_mask_holds_state_across_missing_samples():
    states = np.array([2, np.nan, 1, 0, np.nan, 1, 2])

    inside = Trajectory._hysteretic_circle_inside(states)

    assert inside.tolist() == [True, True, True, False, False, False, True]


def test_custom_circle_radius_is_converted_from_mm_to_pixels():
    captured = {}

    def calc_in_circle(x, y, cx, cy, radius):
        captured["radius"] = radius
        return np.zeros(len(x), dtype=int)

    trajectory = SimpleNamespace(
        opts=SimpleNamespace(pctTimeCircleRad=2.5),
        va=SimpleNamespace(
            ct=SimpleNamespace(pxPerMmFloor=lambda: 10),
            xf=SimpleNamespace(fctr=1.0),
        ),
        f=0,
        xy=lambda start, stop: (np.zeros(stop - start), np.zeros(stop - start)),
        calc_in_circle=calc_in_circle,
    )
    training = SimpleNamespace(
        n=2,
        start=10,
        postStop=20,
        circles=lambda fly: [(5, 6, 1)],
    )

    custom, custom_pre = Trajectory._calculate_custom_circle(
        trajectory, training, None
    )

    assert captured["radius"] == 25
    assert custom.shape == (10,)
    assert custom_pre is None


def test_fraction_inside_circle_excludes_invalid_frames_from_both_terms():
    fraction = Trajectory._fraction_inside_circle(
        np.array([2, 2, 1, 2]),
        np.array([True, False, True, True]),
    )

    assert fraction == pytest.approx(2 / 3)
    assert 0 <= fraction <= 1


def test_custom_and_reward_ranges_include_finite_interpolated_positions():
    trajectory = SimpleNamespace(
        nan=np.array([False, True, False, False]),
        x=np.array([0.0, 1.0, np.nan, 3.0]),
        y=np.array([0.0, 1.0, np.nan, 3.0]),
        pctInC={"rwd": [], "custom": []},
        _fraction_inside_circle=Trajectory._fraction_inside_circle,
    )
    training = SimpleNamespace(n=2, start=0)
    interpolated_states = np.array([2, 2, 2, 0])

    Trajectory._calculate_circle_percentages_for_ranges(
        trajectory,
        training,
        [interpolated_states],
        [slice(0, 4)],
        interpolated_states,
        None,
        0,
    )

    assert trajectory.pctInC["rwd"] == pytest.approx([2 / 3])
    assert trajectory.pctInC["custom"] == pytest.approx([2 / 3])


def test_fraction_inside_circle_rejects_misaligned_masks():
    with pytest.raises(ValueError, match="matching shapes"):
        Trajectory._fraction_inside_circle(
            np.array([2, 1]),
            np.array([True]),
        )


@pytest.mark.parametrize("label", ["slideConc", "slideShift", "custom"])
def test_circle_sync_buckets_include_finite_interpolated_positions(label):
    trajectory = SimpleNamespace(
        f=0,
        pctInC_SB={},
        nan=np.array([False, True, True, False]),
        x=np.array([0.0, 1.0, np.nan, 3.0]),
        y=np.array([0.0, 1.0, np.nan, 3.0]),
        va=SimpleNamespace(
            _numRewardsMsg=lambda *_args, **_kwargs: 4,
            _syncBucket=lambda _training, _frames: (0, 1, None),
        ),
    )
    training = SimpleNamespace(start=0, stop=4)
    circle_states = np.array([2, 2, 0, 0])

    Trajectory._percent_per_sync_bucket(
        trajectory, training, circle_states, label
    )

    assert trajectory.pctInC_SB[label][0] == pytest.approx([100 * 2 / 3])


def test_reward_occupancy_uses_interpolated_geometry_without_creating_entries():
    captured = {}
    circle_states = iter(
        [
            np.array([0, 2, 0]),
            np.array([0, 0, 0]),
        ]
    )
    training = SimpleNamespace(
        n=1,
        start=0,
        stop=3,
        postStop=3,
        circles=lambda _fly: ((0, 0, 1), (0, 0, 1)),
    )
    trajectory = SimpleNamespace(
        va=SimpleNamespace(circle=True, trns=[training], startPre=0),
        opts=SimpleNamespace(cTurnAnlyz=False, showRewardMismatch=False),
        f=0,
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 1.0, 2.0]),
        nan=np.array([False, True, False]),
        _bad=False,
        xy=lambda start, stop: (
            trajectory.x[start:stop],
            trajectory.y[start:stop],
        ),
        calc_in_circle=lambda *_args: next(circle_states).copy(),
        _circle_masks=lambda *_args: {},
        _calcEnEx=lambda states, start, mode="en": Trajectory._calcEnEx(
            states, start, mode
        ),
        _calcPercentInCircle=lambda _training, states, _pre: captured.update(
            occupancy=states.copy()
        ),
        _checkRewards=lambda *_args: None,
    )

    Trajectory._calcRewards(trajectory)

    assert captured["occupancy"].tolist() == [0, 2, 0]
    assert trajectory.en[0].size == 0


def test_pre_circle_ranges_align_when_pre_period_is_shorter_than_fixed_windows():
    trajectory = SimpleNamespace(
        va=SimpleNamespace(startPre=3),
        nan=np.zeros(10, dtype=bool),
        x=np.arange(10, dtype=float),
        y=np.arange(10, dtype=float),
        pctInC={"rwd": [], "custom": []},
        _fraction_inside_circle=Trajectory._fraction_inside_circle,
    )
    training = SimpleNamespace(n=1, start=5, stop=9)
    in_circle = np.arange(10)
    in_circle_pre = np.array([2, 1])

    circle_ranges, valid_ranges = Trajectory._append_pct_circle_pre_ranges(
        trajectory,
        training,
        in_circle,
        in_circle_pre,
        bl_3_min=3,
        bl_10_min=10,
    )

    assert [len(values) for values in circle_ranges] == [2, 2, 4]
    assert valid_ranges == [slice(3, 5), slice(3, 5), slice(5, 9)]

    Trajectory._calculate_circle_percentages_for_ranges(
        trajectory,
        training,
        circle_ranges,
        valid_ranges,
        inC_custom=None,
        inCPre_custom=None,
        bl_3_min=3,
    )

    assert trajectory.pctInC["rwd"] == pytest.approx([0.5, 0.5, 0.25])
