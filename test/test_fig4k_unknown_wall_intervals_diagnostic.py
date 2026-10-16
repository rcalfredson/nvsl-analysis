import numpy as np
import pytest

from scripts.diagnose_fig4k_unknown_wall_intervals import (
    interval_observation_stats,
    summarize,
)
from src.plotting.wall_contacts_per_reward_interval_totals import (
    WallContactsPerRewardIntervalPerFlyCollector,
)


def test_missing_reward_endpoint_belongs_only_to_following_interval():
    trace = np.asarray([0, 1, 0, 0, np.nan, 0, 0, 0, np.nan])
    observed, fraction, eligible = interval_observation_stats(trace, [0, 4], [4, 8])
    np.testing.assert_array_equal(observed, [4, 3])
    np.testing.assert_allclose(fraction, [1, .75])
    np.testing.assert_array_equal(eligible, [True, False])


def test_strict_exclusion_can_discard_a_complete_observed_episode():
    trace = np.asarray([0, 1, 0, np.nan, 0, 0, 1, 0], dtype=float)
    # Use existing episode counting, including its start-only interval assignment.
    regions = [slice(1, 2), slice(6, 7)]
    counts = WallContactsPerRewardIntervalPerFlyCollector._count_regions_by_reward_interval_start(
        regions, np.asarray([0, 4, 8])
    )
    _, _, eligible = interval_observation_stats(trace, [0, 4], [4, 8])
    np.testing.assert_array_equal(counts, [1, 1])
    assert counts[~eligible].sum() == 1
    assert counts[eligible].mean() == counts.mean()


def test_entirely_unknown_and_zero_intervals():
    observed, fraction, eligible = interval_observation_stats([np.nan] * 4, [0], [4])
    assert observed[0] == 0 and fraction[0] == 0 and not eligible[0]
    observed, fraction, eligible = interval_observation_stats([0, 1], [], [])
    assert observed.size == fraction.size == eligible.size == 0


@pytest.mark.parametrize("trace,starts,ends", [
    ([0, 1], [-1], [1]), ([0, 1], [0], [3]), ([0, 1], [1], [1]),
    ([0, 255], [0], [2]), ([[0, 1]], [0], [1]),
])
def test_rejects_invalid_trace_or_interval(trace, starts, ends):
    with pytest.raises(ValueError):
        interval_observation_stats(trace, starts, ends)


def test_no_eligible_intervals_is_separate_from_five_interval_threshold():
    flies = [dict(unit_id="a", n_eligible=0, baseline_mean=1., strict_mean=np.nan),
             dict(unit_id="b", n_eligible=2, baseline_mean=1., strict_mean=2.)]
    intervals = [dict(unit_id="a", strict_eligible=False, classifiable_frames=0,
                      classifiable_fraction=0., wall_contact_event_count=1),
                 dict(unit_id="b", strict_eligible=True, classifiable_frames=2,
                      classifiable_fraction=1., wall_contact_event_count=2)]
    result = summarize(flies, intervals)
    assert result["no_eligible_interval_flies"] == 1
    assert result["below_five_eligible_flies"] == 2
    assert result["episodes_discarded"] == 1
    assert result["changed_mean_flies"] == result["paired_flies"] == 1
