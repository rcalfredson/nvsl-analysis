import numpy as np

from src.analysis.turn_probability import turn_probabilities_in_range


def test_turn_probability_uses_its_own_contact_opportunities():
    probabilities = turn_probabilities_in_range(
        np.asarray([110, 120, 130, 250]),
        {110: True, 120: False},
        slice(100, 200),
        min_contact_events=3,
    )

    np.testing.assert_allclose(probabilities, [1 / 3, 1 / 3])


def test_turn_probability_requires_configured_contact_opportunities():
    probabilities = turn_probabilities_in_range(
        np.asarray([110, 120]),
        {110: True},
        slice(100, 200),
        min_contact_events=3,
    )

    assert np.isnan(probabilities).all()


def test_turn_probability_requires_at_least_one_opportunity_when_threshold_is_zero():
    probabilities = turn_probabilities_in_range(
        np.asarray([]),
        {},
        slice(100, 200),
        min_contact_events=0,
    )

    assert np.isnan(probabilities).all()


def test_turn_probability_rejects_missing_fixed_window_without_fallback():
    probabilities = turn_probabilities_in_range(
        np.asarray([110, 120, 130]),
        {110: True},
        slice(np.nan, np.nan),
        min_contact_events=1,
    )

    assert np.isnan(probabilities).all()
