from types import SimpleNamespace

import numpy as np

import src.exporting.speed_sli_bundle as speed_sli_bundle
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


def test_speed_bundle_keeps_metric_when_sli_target_filter_fails(monkeypatch):
    bundle = {
        "speed_exp": np.asarray([[[2.0, 3.0]]]),
        "sli": np.asarray([0.5]),
        "sli_ts": np.asarray([[[0.4, 0.6]]]),
    }
    monkeypatch.setattr(
        speed_sli_bundle,
        "build_metric_plus_sli_bundle",
        lambda *args, **kwargs: bundle.copy(),
    )
    monkeypatch.setattr(
        speed_sli_bundle,
        "exp_target_sync_bucket_eligibility_mask",
        lambda vas, opts: np.asarray([False]),
    )
    monkeypatch.setattr(
        speed_sli_bundle,
        "exp_target_sync_bucket_filter_payload",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(speed_sli_bundle, "normalize_sli_bundle", lambda value: value)

    va = SimpleNamespace(_skipped=False)
    out = speed_sli_bundle.build_speed_sli_bundle(
        [va], SimpleNamespace(excl_wall_for_spd=False), ["group"]
    )

    np.testing.assert_allclose(out["speed_exp"], [[[2.0, 3.0]]])
    assert np.isnan(out["sli"][0])
    assert np.isnan(out["sli_ts"][0]).all()
