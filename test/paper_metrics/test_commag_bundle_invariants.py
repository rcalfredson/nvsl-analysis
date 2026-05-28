import numpy as np
import pytest

from src.analysis.sli_bundle_utils import normalize_sli_bundle, validate_commag_bundle


def _bundle(**overrides):
    exp = np.asarray(
        [
            [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]],
            [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
        ],
        dtype=float,
    )
    ctrl = np.asarray(
        [
            [[1.0, np.nan, 2.0], [3.0, 4.0, np.nan]],
            [[np.nan, np.nan, 1.0], [2.0, 2.5, 3.0]],
        ],
        dtype=float,
    )
    bundle = {
        "sli": np.asarray([0.1, np.nan], dtype=float),
        "sli_ts": np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[np.nan, 0.1, 0.2], [0.3, np.nan, 0.5]],
            ],
            dtype=float,
        ),
        "commag_exp": exp,
        "commag_ctrl": ctrl,
        "commagN_exp": np.asarray(np.isfinite(exp), dtype=int),
        "commagN_ctrl": np.asarray(np.isfinite(ctrl), dtype=int),
        "group_label": np.asarray("Control", dtype=object),
        "bucket_len_min": np.array(10.0, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a", "video_b"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(0, dtype=int),
        "btw_rwd_sync_bucket_min_trajectories": np.array(1, dtype=int),
    }
    bundle.update(overrides)
    return bundle


def test_validate_commag_bundle_accepts_valid_shapes_counts_and_metadata():
    validate_commag_bundle(_bundle())

    normalized = normalize_sli_bundle(_bundle())
    assert normalized["commag_exp"].shape == (2, 2, 3)


def test_validate_commag_bundle_rejects_missing_metric_key():
    bundle = _bundle()
    del bundle["commag_ctrl"]

    with pytest.raises(ValueError, match="missing COM-magnitude keys"):
        validate_commag_bundle(bundle)


def test_validate_commag_bundle_rejects_metric_shape_mismatch():
    with pytest.raises(ValueError, match=r"commag_exp\.shape=\(2, 2, 2\)"):
        validate_commag_bundle(_bundle(commag_exp=np.ones((2, 2, 2), dtype=float)))


def test_validate_commag_bundle_rejects_bad_magnitudes():
    with pytest.raises(ValueError, match="negative values"):
        validate_commag_bundle(
            _bundle(
                commag_exp=np.asarray(
                    [
                        [[1.0, -2.0, np.nan], [3.0, 4.0, 5.0]],
                        [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
                    ]
                )
            )
        )

    with pytest.raises(ValueError, match="infinite values"):
        validate_commag_bundle(
            _bundle(
                commag_exp=np.asarray(
                    [
                        [[1.0, np.inf, np.nan], [3.0, 4.0, 5.0]],
                        [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
                    ]
                )
            )
        )


def test_validate_commag_bundle_rejects_bad_counts():
    with pytest.raises(ValueError, match="negative values"):
        validate_commag_bundle(
            _bundle(
                commagN_exp=np.asarray(
                    [[[1, -1, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                )
            )
        )

    with pytest.raises(ValueError, match="non-integer values"):
        validate_commag_bundle(
            _bundle(
                commagN_exp=np.asarray(
                    [[[1, 1.5, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                )
            )
        )


def test_validate_commag_bundle_rejects_finite_value_with_insufficient_count():
    with pytest.raises(ValueError, match="finite commag_exp where count is below 1"):
        validate_commag_bundle(
            _bundle(
                commag_exp=np.asarray(
                    [
                        [[1.0, 2.0, 2.0], [3.0, 4.0, 5.0]],
                        [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
                    ]
                ),
                commagN_exp=np.asarray(
                    [[[1, 1, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                ),
            )
        )

    with pytest.raises(ValueError, match="finite commag_ctrl where count is below 2"):
        validate_commag_bundle(
            _bundle(
                btw_rwd_sync_bucket_min_trajectories=np.array(2, dtype=int),
                commagN_exp=np.asarray(
                    [[[2, 2, 0], [2, 2, 2]], [[0, 2, 2], [2, 0, 2]]]
                ),
                commag_ctrl=np.asarray(
                    [
                        [[1.0, np.nan, 2.0], [3.0, 4.0, np.nan]],
                        [[np.nan, np.nan, 1.0], [2.0, 2.5, 3.0]],
                    ],
                    dtype=float,
                ),
                commagN_ctrl=np.asarray(
                    [[[2, 0, 1], [2, 2, 0]], [[0, 0, 2], [2, 2, 2]]]
                ),
            )
        )


def test_validate_commag_bundle_allows_nan_values_explained_by_zero_or_low_counts():
    validate_commag_bundle(
        _bundle(
            btw_rwd_sync_bucket_min_trajectories=np.array(2, dtype=int),
            commag_exp=np.asarray(
                [
                    [[1.0, np.nan, np.nan], [3.0, 4.0, 5.0]],
                    [[np.nan, np.nan, 2.5], [2.0, np.nan, 3.0]],
                ]
            ),
            commagN_exp=np.asarray([[[2, 1, 0], [2, 2, 2]], [[0, 1, 2], [2, 0, 2]]]),
            commag_ctrl=np.asarray(
                [
                    [[1.0, np.nan, np.nan], [3.0, 4.0, np.nan]],
                    [[np.nan, np.nan, 1.0], [2.0, 2.5, 3.0]],
                ]
            ),
            commagN_ctrl=np.asarray([[[2, 0, 1], [2, 2, 0]], [[0, 0, 2], [2, 2, 2]]]),
        )
    )


def test_validate_commag_bundle_rejects_bad_bucket_metadata():
    with pytest.raises(ValueError, match="invalid bucket_len_min"):
        validate_commag_bundle(_bundle(bucket_len_min=np.array(np.nan)))

    with pytest.raises(
        ValueError, match="negative btw_rwd_sync_bucket_min_trajectories"
    ):
        validate_commag_bundle(
            _bundle(btw_rwd_sync_bucket_min_trajectories=np.array(-1))
        )
