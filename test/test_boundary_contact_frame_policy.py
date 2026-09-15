import numpy as np

from src.analysis.boundary_contact_frame_policy import (
    complete_contact_regions,
    interpolation_classifiable_contact,
)


def _bounds(regions):
    return [(region.start, region.stop) for region in regions]


def test_internal_interpolation_is_classifiable():
    contact = interpolation_classifiable_contact(
        [0, 1, 1, 1, 0], [False, False, True, False, False]
    )

    np.testing.assert_equal(contact, [0, 1, 1, 1, 0])
    assert _bounds(complete_contact_regions(contact)) == [(1, 4)]


def test_leading_and_trailing_lost_spans_remain_unknown():
    contact = interpolation_classifiable_contact(
        [1, 1, 0, 1, 1], [True, False, False, False, True]
    )

    np.testing.assert_equal(contact, [np.nan, 1, 0, 1, np.nan])
    assert complete_contact_regions(contact) == []


def test_contact_after_known_noncontact_is_not_left_censored():
    contact = interpolation_classifiable_contact(
        [0, 0, 1, 0], [True, False, False, False]
    )

    assert _bounds(complete_contact_regions(contact)) == [(2, 3)]


def test_fully_observed_contact_is_complete():
    assert _bounds(complete_contact_regions([0, 1, 1, 0])) == [(1, 3)]
