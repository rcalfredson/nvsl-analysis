import pytest

from src.plotting.sync_bucket_axis_limits import default_sync_bucket_ylim


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("commag", (0.0, 10.0)),
        ("rrd_mean_dist", (0.0, 220.0)),
        ("between_reward_return_leg_dist", (0.0, 220.0)),
    ],
)
def test_shared_sync_bucket_axis_limits(metric, expected):
    assert default_sync_bucket_ylim(metric) == expected
