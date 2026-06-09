from types import SimpleNamespace

import numpy as np

from src.exporting.speed_sli_bundle import _extract_speed_arrays


class FakeTraining:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class FakeTrajectory:
    def __init__(self, speeds_px_s, px_per_mm=2.0):
        self.sp = np.asarray(speeds_px_s, dtype=float)
        self.pxPerMmFloor = float(px_per_mm)

    def bad(self):
        return False


class FakeVA:
    fps = 1.0

    def __init__(self):
        self.trns = [FakeTraining(0, 25)]
        self.trx = [
            FakeTrajectory(np.full(30, 4.0), px_per_mm=2.0),
            FakeTrajectory(np.full(30, 8.0), px_per_mm=2.0),
        ]

    def _min2f(self, minutes):
        return int(minutes * 10)

    def _syncBucket(self, trn, df, skip=1):
        return 1, 3, np.asarray([0])


def test_extract_speed_arrays_by_sync_bucket_keeps_incomplete_bucket_nan():
    opts = SimpleNamespace(syncBucketLenMin=1.0, excl_wall_for_spd=False)

    out = _extract_speed_arrays([FakeVA()], opts)

    assert out["speed_exp"].shape == (1, 1, 3)
    assert out["speed_ctrl"].shape == (1, 1, 3)
    np.testing.assert_allclose(out["speed_exp"][0, 0, :2], [2.0, 2.0])
    np.testing.assert_allclose(out["speed_ctrl"][0, 0, :2], [4.0, 4.0])
    assert np.isnan(out["speed_exp"][0, 0, 2])
    assert np.isnan(out["speed_ctrl"][0, 0, 2])
    np.testing.assert_array_equal(out["speedN_exp"][0, 0], [10, 10, 0])
    np.testing.assert_array_equal(out["speedN_ctrl"][0, 0], [10, 10, 0])
