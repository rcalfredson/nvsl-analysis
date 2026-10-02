from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis


@pytest.mark.parametrize("training_sync", ["reward", "midline", "control"])
@pytest.mark.parametrize("preceding_buckets", [0, 1])
@pytest.mark.parametrize("control_entries", [(), (130, 159, 160, 200), (100, 130, 159, 160, 200)])
def test_post_pi_counts_from_onset_and_matches_no_sync(
    training_sync, preceding_buckets, control_entries,
):
    va = object.__new__(VideoAnalysis)
    va.opts = SimpleNamespace(
        rpiPostBucketLenMin=1, piTh=1, reward_pi_sync=training_sync,
    )
    va.fps = 1
    va.flies = [0, 1]
    va.rpiNumNonPostBuckets = preceding_buckets
    va.rpiNumPostBuckets = preceding_buckets + 2
    va.numRewardsTotPrePost = [[], [], [], []]
    va.trns = [SimpleNamespace(
        n=1, start=0, stop=100, postStop=220, name=lambda: "training 1",
    )]
    va.trx = [
        SimpleNamespace(
            en=[np.array([39, 40, 99, 100, 101, 159, 160, 219, 220]),
                np.array(control_entries)],
            bad=lambda: False,
        ),
        SimpleNamespace(
            en=[np.array([99, 100, 145, 160, 220]),
                np.array([99, 110, 160, 219, 220])],
            bad=lambda: False,
        ),
    ]
    va._idxFirstOn = lambda *args, **kwargs: pytest.fail("Post PI must not wait for an entry")
    va._printBucketVals = lambda *args, **kwargs: None

    va.rewardPiPost()

    first_control_count = sum(100 <= frame < 160 for frame in control_entries)
    second_control_count = sum(160 <= frame < 220 for frame in control_entries)
    exp_pi = [
        (3 - first_control_count) / (3 + first_control_count),
        (2 - second_control_count) / (2 + second_control_count),
    ]
    yoked_pi = [1 / 3, -1 / 3]
    exp_crossings = [3 + first_control_count, 2 + second_control_count]
    yoked_crossings = [3, 3]
    if preceding_buckets:
        exp_pi.insert(0, 1)
        yoked_pi.insert(0, 0)
        exp_crossings.insert(0, 2)
        yoked_crossings.insert(0, 2)

    assert va.rewardPiPst[0] == pytest.approx(exp_pi)
    assert va.rewardPiPst[1] == pytest.approx(yoked_pi)
    assert va.numPostCrossings == [tuple(exp_crossings), tuple(yoked_crossings)]
    assert va.rewardPiPst == va.rewardPiPstNoSync
    assert va.numPostCrossings == va.numPostCrossingsNoSync
