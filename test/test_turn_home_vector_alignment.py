import numpy as np
import pytest

from src.exporting.turn_home_vector_alignment_sli_bundle import (
    ANCHOR_FRAME,
    ANCHOR_SEGMENT_MIDPOINT,
    turn_home_vector_alignment_delta,
)


class DummyTrajectory:
    f = 0

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


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
