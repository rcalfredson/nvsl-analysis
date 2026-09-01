from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis
from src.analysis.motion import DataCombiner


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


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (None, 50),
        ("corrected", 100 / 3),
        ("legacy", 200 / 3),
        ("interpolated-inclusive", 50),
    ],
)
def test_large_chamber_agarose_percentage_uses_interpolated_contact_mask(
    policy, expected
):
    contact_stats = {
        "original_boundary_contact": np.array([False, False, True, False]),
        "interpolated_boundary_contact": np.array([False, True, True, False]),
    }
    trj = SimpleNamespace(
        nan=np.array([False, True, False, False]),
        boundary_event_stats={
            "agarose": {
                "tb": {
                    "ctr": contact_stats,
                    "edge": {
                        "original_boundary_contact": np.zeros(4, dtype=bool),
                        "interpolated_boundary_contact": np.zeros(4, dtype=bool),
                    },
                }
            }
        },
        bad=lambda: False,
    )
    va = object.__new__(VideoAnalysis)
    va.trx = [trj]
    va.trns = []
    va.opts = (
        SimpleNamespace()
        if policy is None
        else SimpleNamespace(agarose_time_lost_frame_policy=policy)
    )
    va.reward_ranges = [slice(0, 4)]
    va.pair_exclude = [False]
    va._min2f = lambda _minutes: 0

    VideoAnalysis.calcOnRegionProportionsForCsv(va, "agarose")

    assert va.regionPercentagesCsv["agarose"]["ctr"] == pytest.approx([expected])
    assert va.regionPercentagesCsv["agarose"]["edge"] == pytest.approx([0])


def test_sync_bucket_agarose_percentage_uses_explicit_denominator_mask():
    training = SimpleNamespace(start=0, stop=4)
    va = SimpleNamespace(
        trx=[SimpleNamespace(bad=lambda: False)],
        flies=[0],
        trns=[training],
        nf=4,
        _numRewardsMsg=lambda *_args, **_kwargs: 4,
        _syncBucket=lambda _training, _df: (0, 1, None),
        _append=lambda target, values, _start, _count: target.append(values),
    )
    combiner = DataCombiner(va, post_bucket_len_min=1)

    combiner.combineResults(
        attr=None,
        key=None,
        sv_name="onAgaroseCtr",
        silent=True,
        asPct=True,
        predictions=[np.array([False, True, True, False])],
        denominator_masks=[np.ones(4, dtype=bool)],
    )

    assert va.onAgaroseCtrSB[0][0] == pytest.approx([50])
