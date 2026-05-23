import numpy as np
import pytest

from src.analysis.sli_bundle_utils import normalize_sli_bundle


def _bundle(**overrides):
    bundle = {
        "sli": np.asarray([0.1, np.nan], dtype=float),
        "sli_ts": np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[np.nan, 0.1, 0.2], [0.3, np.nan, 0.5]],
            ],
            dtype=float,
        ),
        "group_label": np.asarray("Control", dtype=object),
        "bucket_len_min": np.array(10.0, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a", "video_b"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(2, dtype=int),
    }
    bundle.update(overrides)
    return bundle


def test_normalize_sli_bundle_accepts_valid_sli_shapes_and_metadata():
    normalized = normalize_sli_bundle(_bundle())

    assert normalized["sli"].shape == (2,)
    assert normalized["sli_ts"].shape == (2, 2, 3)
    assert normalized["sli_training_idx"] == 1
    assert normalized["sli_select_skip_first_sync_buckets"] == 1
    assert normalized["sli_select_keep_first_sync_buckets"] == 2


def test_normalize_sli_bundle_rejects_video_id_sli_length_mismatch():
    with pytest.raises(ValueError, match=r"len\(video_ids\)=1 but len\(sli\)=2"):
        normalize_sli_bundle(_bundle(video_ids=np.asarray(["video_a"], dtype=object)))


def test_normalize_sli_bundle_rejects_non_3d_sli_ts():
    with pytest.raises(ValueError, match="non-3D sli_ts shape"):
        normalize_sli_bundle(_bundle(sli_ts=np.asarray([[0.1, 0.2]], dtype=float)))


def test_normalize_sli_bundle_rejects_sli_ts_sli_length_mismatch():
    with pytest.raises(ValueError, match=r"sli_ts.shape\[0\]=1 but len\(sli\)=2"):
        normalize_sli_bundle(
            _bundle(sli_ts=np.asarray([[[0.1, 0.2, 0.3]]], dtype=float))
        )


def test_normalize_sli_bundle_rejects_training_name_sli_ts_mismatch():
    with pytest.raises(ValueError, match=r"len\(training_names\)=1"):
        normalize_sli_bundle(
            _bundle(training_names=np.asarray(["T1"], dtype=object))
        )


def test_normalize_sli_bundle_rejects_out_of_range_sli_training_idx():
    with pytest.raises(ValueError, match="sli_training_idx=2 but sli_ts has 2 trainings"):
        normalize_sli_bundle(_bundle(sli_training_idx=np.array(2, dtype=int)))


def test_normalize_sli_bundle_allows_empty_sli_ts_training_axis():
    normalized = normalize_sli_bundle(
        _bundle(
            sli_ts=np.empty((2, 0, 0), dtype=float),
            training_names=np.asarray([], dtype=object),
            sli_training_idx=np.array(3, dtype=int),
            sli_select_skip_first_sync_buckets=np.array(0, dtype=int),
            sli_select_keep_first_sync_buckets=np.array(0, dtype=int),
        )
    )

    assert normalized["sli_ts"].shape == (2, 0, 0)
    assert normalized["sli_training_idx"] == 3


def test_normalize_sli_bundle_rejects_negative_sli_window_metadata():
    with pytest.raises(ValueError, match="negative sli_select_skip_first_sync_buckets"):
        normalize_sli_bundle(
            _bundle(sli_select_skip_first_sync_buckets=np.array(-1, dtype=int))
        )

    with pytest.raises(ValueError, match="negative sli_select_keep_first_sync_buckets"):
        normalize_sli_bundle(
            _bundle(sli_select_keep_first_sync_buckets=np.array(-1, dtype=int))
        )


def test_normalize_sli_bundle_rejects_sli_window_beyond_sync_buckets():
    with pytest.raises(ValueError, match="skips 4 SLI sync buckets"):
        normalize_sli_bundle(
            _bundle(sli_select_skip_first_sync_buckets=np.array(4, dtype=int))
        )

    with pytest.raises(ValueError, match="keeps SLI sync-bucket window"):
        normalize_sli_bundle(
            _bundle(
                sli_select_skip_first_sync_buckets=np.array(2, dtype=int),
                sli_select_keep_first_sync_buckets=np.array(2, dtype=int),
            )
        )
