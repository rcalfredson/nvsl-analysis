import numpy as np

from scripts.inspect_openloop_walking import period_masks, walking_trajectory
from src.utils.common import CT


def test_led_periods_exclude_pre_post_and_partition_each_training():
    frames = dict(startTrain=[2, 8], startPost=[6, 12],
                  v7=[2, 5, 8, 11], v0=[0, 4, 6, 10, 12])
    masks = period_masks(frames, 14)
    assert np.flatnonzero(masks['session']).tolist() == [2, 3, 4, 5, 8, 9, 10, 11]
    assert np.flatnonzero(masks['led_on']).tolist() == [2, 3, 5, 8, 9, 11]
    assert np.flatnonzero(masks['led_off']).tolist() == [4, 10]
    assert np.array_equal(masks['led_on'] | masks['led_off'], masks['session'])
    assert not np.any(masks['led_on'] & masks['led_off'])


def test_walking_interpolates_gaps_and_uses_strict_threshold():
    # HTL scale is 8 px/mm. At 8 fps, a 2 px step is exactly
    # 2 mm/s and must not count as walking; a 3 px step must count.
    x = [0, 2, np.nan, 6, 9]
    y = [0, 0, np.nan, 0, 0]
    trj = walking_trajectory(x, y, fps=8, chamber=CT.htl)
    np.testing.assert_array_equal(trj.x, [0, 2, 4, 6, 9])
    np.testing.assert_array_equal(trj.walking, [False, False, False, False, True])
    assert np.isnan(x[2])  # Source data remains unchanged.


def test_absent_tracking_does_not_become_zero_walking():
    assert walking_trajectory(np.full(100, np.nan), np.full(100, np.nan), 30, CT.htl) is None
