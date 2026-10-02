from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis
from src.utils.constants import ST


def _analysis(symmetric=True):
    va = object.__new__(VideoAnalysis)
    va._numRewardsMsg = lambda sync, silent: 80
    va._syncBucket = lambda trn, df: (100, 5, [])
    va._idxFirstOn = lambda fi, la, **kwargs: fi + 10
    va._idxFirstCtrlSide = lambda fi, la, trn: fi + 20
    trn = SimpleNamespace(start=0, stop=500, hasSymCtrl=lambda: symmetric)
    return va, trn


@pytest.mark.parametrize("sync_type,symmetric,expected", [
    (ST.midline, True, 120),
    (ST.midline, False, 111),
    (ST.control, True, 111),
])
def test_training_sync_preserves_initial_qualification(sync_type, symmetric, expected):
    va, trn = _analysis(symmetric)
    assert va._idxSync(sync_type, trn, 100, 500) == expected


@pytest.mark.parametrize("start", [0, 101, 200, 499])
def test_training_sync_rejects_origins_other_than_first_bucket(start):
    va, trn = _analysis()
    with pytest.raises(AssertionError, match="first sync-bucket origin"):
        va._idxSync(ST.midline, trn, start, 500)


@pytest.mark.parametrize("start", [500, 550])
def test_post_training_sync_does_not_require_training_bucket_origin(start):
    va, trn = _analysis()
    va._syncBucket = lambda *args: pytest.fail("Post synchronization queried training origin")
    assert va._idxSync(ST.control, trn, start, 600) == start + 11


def test_fixed_and_missing_origins_do_not_require_behavioral_qualification():
    va, trn = _analysis()
    va._syncBucket = lambda *args: pytest.fail("No behavioral synchronization required")
    assert va._idxSync(ST.fixed, trn, 200, 500) == 200
    assert va._idxSync(ST.midline, trn, None, 500) is None
    assert np.isnan(va._idxSync(ST.midline, trn, np.nan, 500))

