import numpy as np
import pytest

from src.analysis.agarose_frame_policy import agarose_percentage_masks


@pytest.mark.parametrize(
    ("policy", "expected_numerator", "expected_denominator"),
    [
        ("corrected", [False, False, True, False], [True, False, True, True]),
        ("legacy", [False, True, True, False], [True, False, True, True]),
        (
            "interpolated-inclusive",
            [False, True, True, False],
            [True, True, True, True],
        ),
    ],
)
def test_agarose_percentage_masks(policy, expected_numerator, expected_denominator):
    numerator, denominator = agarose_percentage_masks(
        contact=np.array([False, False, True, False]),
        interpolated_contact=np.array([False, True, True, False]),
        lost=np.array([False, True, False, False]),
        policy=policy,
    )

    assert numerator.tolist() == expected_numerator
    assert denominator.tolist() == expected_denominator


def test_agarose_percentage_masks_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="matching shapes"):
        agarose_percentage_masks(
            contact=np.zeros(2, dtype=bool),
            interpolated_contact=np.zeros(3, dtype=bool),
            lost=np.zeros(2, dtype=bool),
        )


def test_agarose_percentage_masks_do_not_treat_nan_as_contact():
    numerator, denominator = agarose_percentage_masks(
        contact=np.array([0.0, np.nan, 1.0]),
        interpolated_contact=np.array([0.0, np.nan, 1.0]),
        lost=np.array([False, True, False]),
        policy="interpolated-inclusive",
    )

    assert numerator.tolist() == [False, False, True]
    assert denominator.tolist() == [True, True, True]
