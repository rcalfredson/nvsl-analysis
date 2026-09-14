import sys
import types

import numpy as np

import src.exporting.bundle_utils as bundle_utils
import src.exporting.com_sli_bundle as com_sli_bundle
from src.analysis.sli_tools import compute_sli_per_fly


class _DummyTraining:
    def __init__(self, name="T1"):
        self._name = name

    def name(self):
        return self._name


class _DummyVA:
    def __init__(self, fn="video_a", *, skipped=False):
        self.fn = fn
        self._skipped = skipped
        self.trns = [_DummyTraining()]


def _opts(**overrides):
    values = {
        "best_worst_trn": 1,
        "sli_use_training_mean": True,
        "sli_select_skip_first_sync_buckets": 0,
        "sli_select_keep_first_sync_buckets": 0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _build_bundle(monkeypatch, opts):
    sli = np.array([0.25], dtype=float)
    sli_ts = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=float)

    monkeypatch.setattr(
        bundle_utils,
        "_compute_sli_scalar_and_timeseries_from_rpid",
        lambda vas, opts: (sli, sli_ts),
    )
    monkeypatch.setattr(
        bundle_utils, "_safe_group_label", lambda opts, gls: "test-group"
    )
    monkeypatch.setattr(
        bundle_utils, "min_between_reward_sync_bucket_trajectories", lambda opts: 1
    )

    fake_analyze = types.ModuleType("analyze")
    fake_analyze.bucketLenForType = lambda bucket_type: (10.0, None)
    monkeypatch.setitem(sys.modules, "analyze", fake_analyze)

    metric = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=float)

    return bundle_utils.build_metric_plus_sli_bundle(
        [_DummyVA()],
        opts,
        ["test-group"],
        extract_metric_arrays=lambda vas: {"metric": metric},
        bucket_type="test",
        print_label="test",
    )


def test_build_metric_plus_sli_bundle_uses_default_sli_min_valid_sync_buckets(
    monkeypatch,
):
    bundle = _build_bundle(monkeypatch, _opts())

    assert bundle["sli_min_valid_sync_buckets"] == 3


def test_build_metric_plus_sli_bundle_stores_configured_sli_min_valid_sync_buckets(
    monkeypatch,
):
    bundle = _build_bundle(monkeypatch, _opts(sli_min_valid_sync_buckets=2))

    assert bundle["sli_min_valid_sync_buckets"] == 2


def test_build_metric_plus_sli_bundle_preserves_computed_sli_with_minimum_metadata(
    monkeypatch,
):
    bundle = _build_bundle(monkeypatch, _opts(sli_min_valid_sync_buckets=2))

    np.testing.assert_allclose(bundle["sli"], [0.25])
    np.testing.assert_allclose(bundle["sli_ts"], [[[0.1, 0.2, 0.3, 0.4]]])


def test_shared_bundle_sli_computation_applies_configured_minimum(monkeypatch):
    class _RpidVA(_DummyVA):
        def __init__(self, values):
            super().__init__()
            self.flies = (0, 1)
            self.values = np.asarray(values, dtype=float)

    fake_analyze = types.ModuleType("analyze")
    fake_analyze.compute_sli_per_fly = compute_sli_per_fly
    fake_analyze.typeCalc = lambda metric: (metric, None)
    fake_analyze.trnsForType = lambda va, metric: va.trns
    fake_analyze.vaVarForType = lambda va, metric, calc: va.values.ravel()
    monkeypatch.setitem(sys.modules, "analyze", fake_analyze)

    vas = [
        _RpidVA([[[10.0, 10.0, np.nan, np.nan], [0.0, 0.0, 0.0, 0.0]]]),
        _RpidVA([[[1.0, 2.0, 3.0, np.nan], [0.0, 0.0, 0.0, 0.0]]]),
    ]
    opts = _opts(sli_min_valid_sync_buckets=3)

    sli, sli_ts = com_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid(
        vas, opts
    )

    np.testing.assert_allclose(sli, [np.nan, 2.0], equal_nan=True)
    np.testing.assert_allclose(
        sli_ts,
        [
            [[10.0, 10.0, np.nan, np.nan]],
            [[1.0, 2.0, 3.0, np.nan]],
        ],
        equal_nan=True,
    )
