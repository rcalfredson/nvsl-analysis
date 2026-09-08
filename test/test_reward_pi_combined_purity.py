from types import SimpleNamespace

import numpy as np

from src.analysis.video_analysis import VideoAnalysis


class _Trajectory:
    def __init__(self):
        self._bad = False

    def bad(self, value=None):
        if value is not None:
            self._bad = value
        return self._bad


def _video():
    trajectory = _Trajectory()
    training = SimpleNamespace(start=0, stop=600)
    va = SimpleNamespace(
        fn="recording.avi",
        f=0,
        flies=(0,),
        noyc=True,
        trx=[trajectory],
        trns=[training],
        rewardPIPre=[np.nan],
        rewardPI=[[np.nan]],
        rewardPiPst=[[np.nan, np.nan]],
        rewardPiPstNoSync=[[np.nan, np.nan]],
        buckets=np.array([np.nan]),
    )
    return va


def test_all_nan_combined_reward_pi_does_not_mark_trajectory_bad():
    va = _video()

    result = VideoAnalysis.rewardPiCombined(va)

    assert np.all(np.isnan(result))
    assert not va.trx[0].bad()
