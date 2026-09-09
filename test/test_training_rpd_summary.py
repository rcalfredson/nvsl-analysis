from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t

from src.analysis.training_rpd_summary import (
    report_training_rewards_per_distance,
    training_rewards_per_distance,
)


def _video(rewards=(2, 0), distances=(1000.0, 1000.0), start=3):
    trn = SimpleNamespace(n=1, start=0, stop=27)
    calls = []

    def count(fi, la, *, calc, ctrl, f):
        assert calc and not ctrl
        calls.append(("count", f, fi, la))
        return rewards[f]

    def distance(f, fi, la):
        calls.append(("distance", f, fi, la))
        return distances[f]

    def sync(training, *, skip):
        assert training is trn and skip == 0
        return start, 2, []

    va = SimpleNamespace(
        trns=[trn],
        trx=[
            SimpleNamespace(
                bad=lambda: False,
                distTrav=lambda fi, la, f=f: distance(f, fi, la),
            )
            for f in range(len(rewards))
        ],
        xf=SimpleNamespace(fctr=2.0),
        ct=SimpleNamespace(pxPerMmFloor=lambda: 5.0),
        _syncBucket=sync,
        _countOn=count,
        # These filters must not affect the RPM-matched training summary.
        reward_exclusion_mask=[[[True], [True]]],
        opts=SimpleNamespace(rpd_pooled_min_rewards=5, piTh=10),
    )
    return va, calls


def test_training_rpd_matches_rpm_window_and_converts_distance_to_meters():
    va, calls = _video()
    exp, diff = training_rewards_per_distance(va, va.trns[0])
    assert exp == pytest.approx(20.0)
    assert diff == pytest.approx(20.0)
    assert calls == [
        ("distance", 0, 3, 27), ("count", 0, 3, 27),
        ("distance", 1, 3, 27), ("count", 1, 3, 27),
    ]


@pytest.mark.parametrize("distance", [0, -1, np.nan, np.inf])
def test_invalid_yoked_distance_preserves_experimental_value(distance):
    va, _ = _video(distances=(1000, distance))
    exp, diff = training_rewards_per_distance(va, va.trns[0])
    assert exp == pytest.approx(20.0)
    assert np.isnan(diff)


@pytest.mark.parametrize("start", [None, np.nan, 27, 30])
def test_missing_or_empty_rpm_window_is_not_a_zero_reward_observation(start):
    va, calls = _video(start=start)
    assert np.all(np.isnan(training_rewards_per_distance(va, va.trns[0])))
    assert calls == []


def test_valid_zero_reward_experimental_only_fly_is_retained():
    va, _ = _video(rewards=(0,), distances=(1000,))
    exp, diff = training_rewards_per_distance(va, va.trns[0])
    assert exp == 0
    assert np.isnan(diff)


@pytest.mark.parametrize("bad_fly", [0, 1])
def test_bad_trajectory_does_not_contribute(bad_fly):
    va, _ = _video()
    va.trx[bad_fly].bad = lambda: True
    exp, diff = training_rewards_per_distance(va, va.trns[0])
    assert np.isnan(diff)
    if bad_fly == 0:
        assert np.isnan(exp)
    else:
        assert exp == pytest.approx(20.0)


def test_summary_uses_per_fly_ratios_and_paired_difference_ci(capsys):
    vas = [
        _video(rewards=(2, 1))[0],
        _video(rewards=(6, 4), distances=(2000, 2000))[0],
        _video(rewards=(0,), distances=(1000,))[0],
    ]
    values = report_training_rewards_per_distance(vas, vas[0].trns, [0, 0, 0])
    assert values[:, 0, 0] == pytest.approx([20, 30, 0])
    assert values[:2, 0, 1] == pytest.approx([10, 10])
    assert np.isnan(values[2, 0, 1])
    output = capsys.readouterr().out
    exp_values = np.array([20, 30, 0])
    delta = t.ppf(0.975, 2) * np.std(exp_values, ddof=1) / np.sqrt(3)
    assert f"t1, exp fly: 16.67 ±{delta:.2f}" in output
    assert "t1, exp-minus-yoked: 10.00 ±0.00 (2)" in output
    assert 'n = 3  (in "()" below if different)' in output


def test_summary_keeps_groups_separate_and_reports_missing_samples(capsys):
    vas = [_video(rewards=(0, 0))[0], _video(start=None)[0]]
    report_training_rewards_per_distance(vas, vas[0].trns, [0, 1], ["control", "test"])
    output = capsys.readouterr().out
    assert output.count("t1, control: 0.00 ±nan") == 2
    assert output.count("t1, test: nan ±nan (0)") == 2
