import numpy as np
import pytest

from src.plotting.between_reward_conditioned_disttrav import (
    BetweenRewardConditionedDistTravResult,
)


def _result(**overrides):
    payload = {
        "x_edges": np.asarray([2.0, 8.0, 16.0], dtype=float),
        "x_centers": np.asarray([5.0, 12.0], dtype=float),
        "mean_total": np.asarray([4.0, np.nan], dtype=float),
        "ci_lo_total": np.asarray([3.5, np.nan], dtype=float),
        "ci_hi_total": np.asarray([4.5, np.nan], dtype=float),
        "mean_tail": np.asarray([2.0, np.nan], dtype=float),
        "ci_lo_tail": np.asarray([1.5, np.nan], dtype=float),
        "ci_hi_tail": np.asarray([2.5, np.nan], dtype=float),
        "n_units": np.asarray([2, 0], dtype=int),
        "meta": {"units": "mm"},
        "per_unit_total": np.asarray([[3.0, np.nan], [5.0, np.nan]], dtype=float),
        "per_unit_tail": np.asarray([[1.0, np.nan], [3.0, np.nan]], dtype=float),
        "per_unit_ids": np.asarray(["fly_a", "fly_b"], dtype=object),
    }
    payload.update(overrides)
    return BetweenRewardConditionedDistTravResult(**payload)


def test_conditioned_disttrav_result_validate_accepts_consistent_result():
    _result().validate()


def test_conditioned_disttrav_result_validate_rejects_bad_edges_and_centers():
    with pytest.raises(ValueError, match="strictly increasing"):
        _result(x_edges=np.asarray([2.0, 8.0, 8.0])).validate()

    with pytest.raises(ValueError, match="bin midpoints"):
        _result(x_centers=np.asarray([5.0, 10.0])).validate()


def test_conditioned_disttrav_result_validate_rejects_bad_counts():
    with pytest.raises(ValueError, match="n_units must be nonnegative"):
        _result(n_units=np.asarray([2, -1])).validate()

    with pytest.raises(ValueError, match="n_units must contain integer values"):
        _result(n_units=np.asarray([2.5, 0.0])).validate()


def test_conditioned_disttrav_result_validate_rejects_negative_or_infinite_distances():
    with pytest.raises(ValueError, match="mean_total must be nonnegative"):
        _result(mean_total=np.asarray([-1.0, np.nan])).validate()

    with pytest.raises(ValueError, match="ci_hi_tail must not contain infinite"):
        _result(ci_hi_tail=np.asarray([np.inf, np.nan])).validate()

    with pytest.raises(ValueError, match="per_unit_total must be nonnegative"):
        _result(per_unit_total=np.asarray([[-1.0, np.nan], [5.0, np.nan]])).validate()


def test_conditioned_disttrav_result_validate_rejects_inverted_confidence_interval():
    with pytest.raises(ValueError, match="ci_lo_total must be <= mean_total"):
        _result(ci_lo_total=np.asarray([4.5, np.nan])).validate()

    with pytest.raises(ValueError, match="mean_tail must be <= ci_hi_tail"):
        _result(ci_hi_tail=np.asarray([1.5, np.nan])).validate()


def test_conditioned_disttrav_result_validate_rejects_finite_summary_for_empty_bin():
    with pytest.raises(ValueError, match="mean_total must be NaN where n_units == 0"):
        _result(mean_total=np.asarray([4.0, 1.0])).validate()


def test_conditioned_disttrav_result_validate_rejects_per_unit_shape_mismatch():
    with pytest.raises(ValueError, match="per_unit_total and per_unit_tail shapes"):
        _result(per_unit_tail=np.asarray([[1.0, np.nan]])).validate()

    with pytest.raises(ValueError, match="per_unit_ids length"):
        _result(per_unit_ids=np.asarray(["fly_a"], dtype=object)).validate()
