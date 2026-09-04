from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.analysis.video_analysis import VideoAnalysis
from src.utils.common import CT, Xformer


class _FakeTrajectory:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


def _make_htl_va(abs_fly: int) -> VideoAnalysis:
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.ct = CT.htl
    va.trxf = (abs_fly,)
    va.xf = Xformer(
        {"fctr": 1.0, "x": 0.0, "y": 0.0},
        CT.htl,
        np.zeros((720, 720), dtype=np.uint8),
        True,
    )
    return va


def test_htl_heatmap_bounds_use_padded_floor_geometry():
    va = _make_htl_va(abs_fly=0)

    xym, xyM = va._heatmapBounds(0)

    np.testing.assert_allclose(xym, (-15.0, -15.0))
    np.testing.assert_allclose(xyM, (95.0, 143.0))


def test_htl_heatmap_coords_recover_canonical_local_points_across_grid_positions():
    canonical_x = np.array([10.0, 40.0, 70.0, 15.0, 65.0])
    canonical_y = np.array([10.0, 64.0, 120.0, 100.0, 25.0])

    for abs_fly in (0, 9, 17, 19):
        va = _make_htl_va(abs_fly=abs_fly)

        frame_pts = [va.xf.t2f(x, y, f=abs_fly) for x, y in zip(canonical_x, canonical_y)]
        trx = _FakeTrajectory(
            x=[pt[0] for pt in frame_pts],
            y=[pt[1] for pt in frame_pts],
        )

        recovered_x, recovered_y = va._heatmapCoords(trx, 0, 0, len(canonical_x))

        np.testing.assert_allclose(recovered_x, canonical_x, atol=1e-6)
        np.testing.assert_allclose(recovered_y, canonical_y, atol=1e-6)

        cx_frame, cy_frame = va.xf.t2f(*CT.htl.center(), f=abs_fly)
        cx_local, cy_local = va.xf.f2t(cx_frame, cy_frame, f=abs_fly)
        np.testing.assert_allclose((cx_local, cy_local), CT.htl.center(), atol=1e-6)


class _FakeTraining:
    start = 100
    stop = 1000


def _make_heatmap_window_va(bucket, *, head_minutes=None, tail_minutes=None):
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.opts = SimpleNamespace(
        hm_sync_bucket=bucket,
        hm_sync_bucket_head_minutes=head_minutes,
        hm_sync_bucket_tail_minutes=tail_minutes,
        syncBucketLenMin=10,
        hm_pre_minutes=10,
    )
    va._min2f = lambda minutes: int(minutes * 10)
    va._syncBucket = lambda training, df: (150, 9, np.array([149]))
    return va


def test_heatmap_training_frame_range_defaults_to_full_training():
    va = _make_heatmap_window_va(None)

    assert va._heatmapTrainingFrameRange(_FakeTraining()) == (100, 1000)


def test_heatmap_training_frame_range_selects_one_based_sync_bucket():
    va = _make_heatmap_window_va(5)

    assert va._heatmapTrainingFrameRange(_FakeTraining()) == (550, 650)


def test_heatmap_training_frame_range_selects_bucket_head():
    va = _make_heatmap_window_va(5, head_minutes=5)

    assert va._heatmapTrainingFrameRange(_FakeTraining()) == (550, 600)


def test_heatmap_training_frame_range_selects_bucket_tail():
    va = _make_heatmap_window_va(5, tail_minutes=5)

    assert va._heatmapTrainingFrameRange(_FakeTraining()) == (600, 650)


def test_heatmap_bucket_portion_requires_bucket_selection():
    va = _make_heatmap_window_va(None, tail_minutes=5)

    with pytest.raises(ValueError, match="require --hm-sync-bucket"):
        va._heatmapTrainingFrameRange(_FakeTraining())


def test_heatmap_bucket_portion_cannot_exceed_bucket_duration():
    va = _make_heatmap_window_va(5, head_minutes=11)

    with pytest.raises(ValueError, match="must not exceed --sb"):
        va._heatmapTrainingFrameRange(_FakeTraining())


def test_heatmap_training_frame_range_rejects_incomplete_bucket():
    va = _make_heatmap_window_va(9)

    assert va._heatmapTrainingFrameRange(_FakeTraining()) is None


def test_heatmap_training_frame_range_requires_positive_bucket():
    va = _make_heatmap_window_va(0)

    with pytest.raises(ValueError, match="--hm-sync-bucket must be >= 1"):
        va._heatmapTrainingFrameRange(_FakeTraining())


def test_heatmap_pre_training_frame_range_uses_trailing_fixed_window():
    va = _make_heatmap_window_va(None)
    va.trns = [_FakeTraining()]
    va.startPre = 0

    assert va._heatmapPreTrainingFrameRange() == (0, 100)


def test_heatmap_pre_training_frame_range_rejects_incomplete_window():
    va = _make_heatmap_window_va(None)
    va.trns = [_FakeTraining()]
    va.startPre = 1

    assert va._heatmapPreTrainingFrameRange() is None


def test_heatmap_pre_training_frame_range_requires_training_metadata():
    va = _make_heatmap_window_va(None)
    va.trns = []

    assert va._heatmapPreTrainingFrameRange() is None


def test_heatmap_periods_share_histogram_calculation_and_post_masking():
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.heatmapOOB = False
    va._heatmapCoords = lambda trx, f, fi, la: [
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    ]
    va._debugHeatmapAlignment = lambda **kwargs: None
    trx = SimpleNamespace(walking=np.ones(4, dtype=bool))
    kwargs = dict(
        trx=trx,
        t=_FakeTraining(),
        f=0,
        fi=0,
        la=4,
        xym=np.array([0.0, 0.0]),
        xyM=np.array([2.0, 2.0]),
        bins=[2, 2],
        rng=np.array([[0.0, 2.0], [0.0, 2.0]]),
    )

    pre_map, pre_length, _ = va._calculateHeatmapForFrameRange(
        **kwargs, period="pre"
    )
    training_map, _, _ = va._calculateHeatmapForFrameRange(
        **kwargs, period="training"
    )
    post_map, post_length, _ = va._calculateHeatmapForFrameRange(
        **kwargs, period="post", fiRi=2
    )

    assert pre_length == post_length == 4
    np.testing.assert_array_equal(pre_map, training_map)
    assert np.sum(pre_map) == 4
    assert np.sum(post_map) == 2
