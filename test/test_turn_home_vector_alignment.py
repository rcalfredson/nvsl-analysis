import numpy as np
import pytest

from src.exporting.turn_home_vector_alignment_sli_bundle import (
    ANCHOR_FRAME,
    ANCHOR_SEGMENT_MIDPOINT,
    VALUE_MODE_EXP,
    VALUE_MODE_EXP_MINUS_YOK,
    _combine_role_panel_means,
    _turn_fully_within_radius_range_mm,
    parse_radius_range_mm,
    parse_radius_ranges_mm,
    radius_range_slug,
    turn_home_vector_alignment_delta,
)


def test_combine_role_panel_means_supports_exp_and_paired_exp_minus_yok():
    role_results = [(12.5, 8), (4.0, 6)]

    assert _combine_role_panel_means(role_results, VALUE_MODE_EXP) == 12.5
    assert (
        _combine_role_panel_means(role_results, VALUE_MODE_EXP_MINUS_YOK) == 8.5
    )


class DummyTrajectory:
    f = 0

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.pxPerMmFloor = 1.0


class DummyTraining:
    def __init__(self, cx, cy, r=10.0):
        self._circle = (float(cx), float(cy), float(r))

    def circles(self, _fly):
        return [self._circle]


def test_turn_home_vector_alignment_delta_positive_when_turn_improves_alignment():
    trj = DummyTrajectory(
        x=[0.0, 1.0, 1.0, 2.0, 3.0],
        y=[0.0, 0.0, 1.0, 1.0, 1.0],
    )
    trn = DummyTraining(cx=10.0, cy=1.0)

    delta = turn_home_vector_alignment_delta(
        trj, trn, turn_start=1, turn_stop=2, anchor=ANCHOR_FRAME
    )

    assert np.isfinite(delta)
    assert delta > 0.0
    assert delta == pytest.approx(38.6598082541)


def test_turn_home_vector_alignment_delta_supports_segment_midpoint_anchor():
    trj = DummyTrajectory(
        x=[0.0, 1.0, 1.0, 2.0, 3.0],
        y=[0.0, 0.0, 1.0, 1.0, 1.0],
    )
    trn = DummyTraining(cx=10.0, cy=1.0)

    delta = turn_home_vector_alignment_delta(
        trj,
        trn,
        turn_start=1,
        turn_stop=2,
        anchor=ANCHOR_SEGMENT_MIDPOINT,
    )

    assert np.isfinite(delta)
    assert delta == pytest.approx(41.8201698801)


def test_parse_radius_range_accepts_hyphen_and_colon():
    assert parse_radius_range_mm("3-5") == (3.0, 5.0)
    assert parse_radius_range_mm("8:10") == (8.0, 10.0)
    assert parse_radius_range_mm("") is None
    assert parse_radius_ranges_mm("3-5,8:10") == [(3.0, 5.0), (8.0, 10.0)]
    assert radius_range_slug((3.0, 5.5)) == "r3_5p5mm"


def test_turn_radius_filter_requires_full_metric_span_inside_band():
    trj = DummyTrajectory(
        x=[4.0, 4.5, 4.6, 4.7, 5.1],
        y=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    trn = DummyTraining(cx=0.0, cy=0.0)

    assert _turn_fully_within_radius_range_mm(trj, trn, 1, 2, (4.0, 5.0)) is False
    assert _turn_fully_within_radius_range_mm(trj, trn, 1, 2, (4.0, 5.2)) is True
