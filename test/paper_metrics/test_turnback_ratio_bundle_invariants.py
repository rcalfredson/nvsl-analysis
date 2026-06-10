import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    normalize_sli_bundle,
    validate_turnback_ratio_bundle,
)


def _bundle(**overrides):
    bundle = {
        "sli": np.asarray([0.1, np.nan], dtype=float),
        "sli_ts": np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, np.nan]],
                [[np.nan, 0.1, 0.2], [0.3, np.nan, 0.5]],
            ]
        ),
        "group_label": np.asarray("Intact Control>Kir", dtype=object),
        "bucket_len_min": np.array(10, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a", "video_b"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(0, dtype=int),
        "turnback_ratio_exp": np.asarray(
            [
                [[1.0, 0.5, np.nan], [0.0, 2.0 / 3.0, np.nan]],
                [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
            ],
            dtype=float,
        ),
        "turnback_ratio_ctrl": np.asarray(
            [
                [[0.0, 0.25, np.nan], [0.5, np.nan, 1.0]],
                [[np.nan, 0.0, 0.75], [1.0, 0.5, np.nan]],
            ],
            dtype=float,
        ),
        "turnback_total_exp": np.asarray(
            [[[1, 2, 0], [1, 3, 0]], [[0, 1, 4], [2, 0, 1]]], dtype=int
        ),
        "turnback_total_ctrl": np.asarray(
            [[[1, 4, 0], [2, 0, 1]], [[0, 1, 4], [1, 2, 0]]], dtype=int
        ),
        "turnback_inner_delta_mm": np.array(4.0, dtype=float),
        "turnback_outer_delta_mm": np.array(8.0, dtype=float),
        "turnback_inner_radius_offset_px": np.array(0.0, dtype=float),
    }
    bundle.update(overrides)
    return bundle


def test_validate_turnback_ratio_bundle_accepts_valid_shapes_and_metadata():
    validate_turnback_ratio_bundle(_bundle())

    normalized = normalize_sli_bundle(_bundle())
    assert normalized["turnback_ratio_exp"].shape == (2, 2, 3)


def test_validate_turnback_ratio_bundle_allows_target_sync_bucket_filtered_exp_ratio():
    validate_turnback_ratio_bundle(
        _bundle(
            turnback_ratio_exp=np.asarray(
                [
                    [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                    [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                ],
                dtype=float,
            ),
            exp_target_sync_bucket_filter_eligible=np.asarray([False, True]),
        )
    )


def test_validate_turnback_ratio_bundle_accepts_legacy_pi_filter_key():
    validate_turnback_ratio_bundle(
        _bundle(
            turnback_ratio_exp=np.asarray(
                [
                    [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                    [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                ],
                dtype=float,
            ),
            exp_pi_threshold_filter_eligible=np.asarray([False, True]),
        )
    )


def test_validate_turnback_ratio_bundle_rejects_missing_metric_key():
    bundle = _bundle()
    del bundle["turnback_outer_delta_mm"]

    with pytest.raises(ValueError, match="missing turnback ratio keys"):
        validate_turnback_ratio_bundle(bundle)


def test_validate_turnback_ratio_bundle_rejects_metric_shape_mismatch():
    with pytest.raises(ValueError, match=r"turnback_ratio_exp\.shape=\(2, 2, 2\)"):
        validate_turnback_ratio_bundle(
            _bundle(turnback_ratio_exp=np.ones((2, 2, 2), dtype=float))
        )


def test_validate_turnback_ratio_bundle_rejects_bad_totals():
    with pytest.raises(ValueError, match="negative values"):
        validate_turnback_ratio_bundle(
            _bundle(
                turnback_total_exp=np.asarray(
                    [[[1, -1, 0], [1, 3, 0]], [[0, 1, 4], [2, 0, 1]]]
                )
            )
        )

    with pytest.raises(ValueError, match="non-integer values"):
        validate_turnback_ratio_bundle(
            _bundle(
                turnback_total_ctrl=np.asarray(
                    [[[1, 4.5, 0], [2, 0, 1]], [[0, 1, 4], [1, 2, 0]]]
                )
            )
        )


def test_validate_turnback_ratio_bundle_rejects_bad_probabilities():
    with pytest.raises(ValueError, match="out-of-range probabilities"):
        validate_turnback_ratio_bundle(
            _bundle(
                turnback_ratio_exp=np.asarray(
                    [
                        [[1.2, 0.5, np.nan], [0.0, 2.0 / 3.0, np.nan]],
                        [
                            [np.nan, 1.0, 0.25],
                            [0.5, np.nan, 1.0],
                        ],
                    ]
                )
            )
        )

    with pytest.raises(
        ValueError, match="finite turnback_ratio_exp.*below reporting threshold"
    ):
        validate_turnback_ratio_bundle(
            _bundle(
                turnback_ratio_exp=np.asarray(
                    [
                        [[1.0, 0.5, 0.0], [0.0, 2.0 / 3.0, np.nan]],
                        [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                    ]
                )
            )
        )


def test_validate_turnback_ratio_bundle_rejects_ratio_inconsistent_with_counts():
    with pytest.raises(ValueError, match="inconsistent with integer counts"):
        validate_turnback_ratio_bundle(
            _bundle(
                turnback_ratio_ctrl=np.asarray(
                    [
                        [[0.0, 0.3, np.nan], [0.5, np.nan, 1.0]],
                        [[np.nan, 0.0, 0.75], [1.0, 0.5, np.nan]],
                    ]
                )
            )
        )


def test_validate_turnback_ratio_bundle_rejects_bad_radius_metadata():
    with pytest.raises(ValueError, match="turnback_outer_delta_mm=.*<="):
        validate_turnback_ratio_bundle(_bundle(turnback_outer_delta_mm=np.array(4.0)))

    with pytest.raises(ValueError, match="negative turnback_inner_delta_mm"):
        validate_turnback_ratio_bundle(_bundle(turnback_inner_delta_mm=np.array(-1.0)))

    with pytest.raises(ValueError, match="non-finite turnback_inner_radius_offset_px"):
        validate_turnback_ratio_bundle(
            _bundle(turnback_inner_radius_offset_px=np.array(np.nan))
        )
