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
    va.opts = SimpleNamespace(agarose_time_lost_frame_policy="corrected")
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([100 / 3])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])


def test_agarose_percentage_can_reproduce_legacy_lost_frame_numerator():
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
    va.opts = SimpleNamespace(agarose_time_lost_frame_policy="legacy")
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([200 / 3])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])


def test_agarose_percentage_can_include_interpolated_frames_in_both_terms():
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
    va.opts = SimpleNamespace(
        agarose_time_lost_frame_policy="interpolated-inclusive"
    )
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([50])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])


def test_agarose_percentage_includes_interpolated_frames_by_default():
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
    va.opts = SimpleNamespace()
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([50])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])
