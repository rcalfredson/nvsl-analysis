from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis


def test_agarose_percentage_excludes_interpolated_frames_from_both_terms():
    trj = SimpleNamespace(
        nan=np.array([False, True, False, False]),
        boundary_event_stats={
            "agarose": {
                "tb": {
                    "ctr": {
                        "original_boundary_contact": np.array(
                            [False, True, True, False]
                        )
                    },
                    "edge": {
                        "original_boundary_contact": np.zeros(4, dtype=bool)
                    },
                }
            }
        },
        bad=lambda: False,
    )
    va = object.__new__(VideoAnalysis)
    va.trx = [trj]
    va.trns = []
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([100 / 3])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])
