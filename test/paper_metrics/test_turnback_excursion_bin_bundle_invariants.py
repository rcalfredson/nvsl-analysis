import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    normalize_sli_bundle,
    validate_turnback_excursion_bin_bundle,
)


def _bundle(**overrides):
    bundle = {
        "sli": np.asarray([0.1], dtype=float),
        "sli_ts": np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=float),
        "group_label": np.asarray("Control", dtype=object),
        "bucket_len_min": np.array(np.nan, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(0, dtype=int),
        "turnback_excursion_bin_ratio_exp": np.asarray([[0.5, np.nan]]),
        "turnback_excursion_bin_ratio_ctrl": np.asarray([[0.2, 0.6]]),
        "turnback_excursion_bin_turn_exp": np.asarray([[2.0, 0.0]]),
        "turnback_excursion_bin_turn_ctrl": np.asarray([[1.0, 3.0]]),
        "turnback_excursion_bin_total_exp": np.asarray([[4, 0]], dtype=int),
        "turnback_excursion_bin_total_ctrl": np.asarray([[5, 5]], dtype=int),
        "turnback_excursion_bin_edges_mm": np.asarray([2.0, 8.0, 16.0]),
        "turnback_excursion_bin_requested_edges_mm": np.asarray([2.0, 8.0, 16.0]),
        "turnback_excursion_bin_open_ended_upper_bin": np.asarray(False),
        "turnback_excursion_bin_trainings": np.asarray([1], dtype=int),
        "turnback_excursion_bin_skip_first_sync_buckets": np.array(0, dtype=int),
        "turnback_excursion_bin_keep_first_sync_buckets": np.array(0, dtype=int),
        "turnback_excursion_bin_last_sync_buckets": np.array(0, dtype=int),
        "turnback_excursion_bin_inner_delta_mm": np.array(2.0, dtype=float),
        "turnback_excursion_bin_inner_radius_offset_px": np.array(0.0, dtype=float),
        "turnback_excursion_bin_border_width_mm": np.array(0.1, dtype=float),
        "turnback_excursion_bin_window_summary": np.asarray(
            ["T2 training 2[0,10)"], dtype=object
        ),
    }
    bundle.update(overrides)
    return bundle


def test_validate_turnback_excursion_bin_bundle_accepts_valid_shapes_and_metadata():
    validate_turnback_excursion_bin_bundle(_bundle())

    normalized = normalize_sli_bundle(_bundle())
    assert normalized["turnback_excursion_bin_ratio_exp"].shape == (1, 2)


def test_validate_turnback_excursion_bin_bundle_accepts_pair_mode_metadata():
    validate_turnback_excursion_bin_bundle(
        _bundle(
            turnback_excursion_bin_edges_mm=np.asarray([]),
            turnback_excursion_bin_requested_edges_mm=np.asarray([]),
            turnback_excursion_bin_pair_inner_deltas_mm=np.asarray([2.0, 6.0]),
            turnback_excursion_bin_pair_outer_deltas_mm=np.asarray([4.0, 8.0]),
            turnback_excursion_bin_pair_mode=np.asarray(True),
        )
    )


def test_validate_turnback_excursion_bin_bundle_rejects_missing_metric_key():
    bundle = _bundle()
    del bundle["turnback_excursion_bin_ratio_exp"]

    with pytest.raises(ValueError, match="missing turnback excursion-bin"):
        validate_turnback_excursion_bin_bundle(bundle)


def test_validate_turnback_excursion_bin_bundle_rejects_metric_shape_mismatch():
    with pytest.raises(
        ValueError, match=r"turnback_excursion_bin_ratio_exp\.shape=\(1, 1\)"
    ):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_ratio_exp=np.asarray([[0.5]]))
        )


def test_validate_turnback_excursion_bin_bundle_rejects_bad_edges():
    with pytest.raises(ValueError, match="fewer than two"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_edges_mm=np.asarray([2.0]))
        )

    with pytest.raises(ValueError, match="non-increasing"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_edges_mm=np.asarray([2.0, 8.0, 8.0]))
        )

    with pytest.raises(ValueError, match="non-finite resolved"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_edges_mm=np.asarray([2.0, 8.0, np.inf]))
        )


def test_validate_turnback_excursion_bin_bundle_allows_requested_inf_last_edge():
    validate_turnback_excursion_bin_bundle(
        _bundle(
            turnback_excursion_bin_edges_mm=np.asarray([2.0, 8.0, 20.0]),
            turnback_excursion_bin_requested_edges_mm=np.asarray([2.0, 8.0, np.inf]),
            turnback_excursion_bin_open_ended_upper_bin=np.asarray(True),
        )
    )


def test_validate_turnback_excursion_bin_bundle_rejects_bad_totals():
    with pytest.raises(ValueError, match="negative values"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_total_exp=np.asarray([[4, -1]]))
        )

    with pytest.raises(ValueError, match="non-integer values"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_total_exp=np.asarray([[4.5, 0.0]]))
        )


def test_validate_turnback_excursion_bin_bundle_rejects_bad_probabilities():
    with pytest.raises(ValueError, match="out-of-range probabilities"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_ratio_exp=np.asarray([[1.2, np.nan]]))
        )

    with pytest.raises(ValueError, match="finite .* where total == 0"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_ratio_exp=np.asarray([[0.5, 0.0]]))
        )


def test_validate_turnback_excursion_bin_bundle_rejects_inconsistent_ratio():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_ratio_exp=np.asarray([[0.25, np.nan]]))
        )


def test_validate_turnback_excursion_bin_bundle_rejects_turn_mass_above_total():
    with pytest.raises(ValueError, match="non-finite values"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_turn_exp=np.asarray([[2.0, np.nan]]))
        )

    with pytest.raises(ValueError, match="greater than totals"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_turn_exp=np.asarray([[5.0, 0.0]]))
        )


def test_validate_turnback_excursion_bin_bundle_rejects_bad_metadata():
    with pytest.raises(
        ValueError, match="len\\(turnback_excursion_bin_window_summary\\)=2"
    ):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_window_summary=np.asarray(["a", "b"]))
        )

    with pytest.raises(
        ValueError, match="negative turnback_excursion_bin_inner_delta_mm"
    ):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_inner_delta_mm=np.asarray(-1.0))
        )

    with pytest.raises(
        ValueError, match="non-finite turnback_excursion_bin_inner_radius_offset_px"
    ):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_inner_radius_offset_px=np.asarray(np.nan))
        )

    with pytest.raises(ValueError, match="negative turnback_excursion_bin_trainings"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_trainings=np.asarray([-1]))
        )

    with pytest.raises(ValueError, match="outer deltas <= inner"):
        validate_turnback_excursion_bin_bundle(
            _bundle(
                turnback_excursion_bin_edges_mm=np.asarray([]),
                turnback_excursion_bin_requested_edges_mm=np.asarray([]),
                turnback_excursion_bin_pair_inner_deltas_mm=np.asarray([2.0]),
                turnback_excursion_bin_pair_outer_deltas_mm=np.asarray([2.0]),
            )
        )

    with pytest.raises(ValueError, match="requested last edge is not inf"):
        validate_turnback_excursion_bin_bundle(
            _bundle(turnback_excursion_bin_open_ended_upper_bin=np.asarray(True))
        )
