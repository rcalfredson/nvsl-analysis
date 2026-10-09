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
    (ST.reward, True, 100),
    (ST.reward, False, 100),
])
def test_training_sync_preserves_initial_qualification(sync_type, symmetric, expected):
    va, trn = _analysis(symmetric)
    assert va._idxSync(sync_type, trn, 100, 500) == expected


@pytest.mark.parametrize("sync_type", [ST.midline, ST.reward])
@pytest.mark.parametrize("start", [0, 101, 200, 499])
def test_training_sync_rejects_origins_other_than_first_bucket(start, sync_type):
    va, trn = _analysis()
    with pytest.raises(AssertionError, match="first sync-bucket origin"):
        va._idxSync(sync_type, trn, start, 500)


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
    assert va._idxSync(ST.reward, trn, None, 500) is None
    assert np.isnan(va._idxSync(ST.reward, trn, np.nan, 500))


@pytest.mark.parametrize("symmetric", [False, True])
def test_reward_sync_never_waits_for_control(symmetric):
    va, trn = _analysis(symmetric)
    va._idxFirstOn = lambda *args, **kwargs: pytest.fail("Reward already skipped")
    va._idxFirstCtrlSide = lambda *args: pytest.fail("No control-side wait")
    assert va._idxSync(ST.reward, trn, 100, 500) == 100


def test_post_reward_sync_skips_first_target_entry():
    va, trn = _analysis()
    def first(fi, la, *, calc, ctrl):
        assert calc and not ctrl
        return 520
    va._idxFirstOn = first
    assert va._idxSync(ST.reward, trn, 500, 600) == 521
    va._idxFirstOn = lambda *args, **kwargs: None
    assert va._idxSync(ST.reward, trn, 500, 600) is None


def _counting_analysis(policy=None, control_entries=(40, 80)):
    va = object.__new__(VideoAnalysis)
    va.opts = SimpleNamespace(reward_pi_sync=policy, piTh=1)
    va.on = np.array([10, 20, 30, 50, 60, 70, 90, 110])
    va.trx = [
        SimpleNamespace(en=[va.on, np.array(control_entries)], bad=lambda: False),
        SimpleNamespace(
            en=[np.array([5, 15, 25, 55, 75]), np.array([7, 35, 65, 85])],
            bad=lambda: False,
        ),
    ]
    va.trns = [SimpleNamespace(
        n=1, start=0, stop=130, len=lambda: 130,
        name=lambda: "training 1", hasSymCtrl=lambda: False,
    )]
    va.flies = [0, 1]
    va.noyc = False
    va._numRewardsMsg = lambda *args, **kwargs: 50
    va._printBucketVals = lambda *args, **kwargs: None
    return va


@pytest.mark.parametrize("policy", [None, "reward"])
def test_default_pi_skips_only_initial_reward_and_preserves_buckets(policy):
    va = _counting_analysis(policy)
    va.bySyncBucket()
    # Actual and calculated target entries at frame 10 are skipped; frame 20
    # remains even though the first control entry is not until frame 40.
    assert va.sync_bucket_ranges == [[(11, 61), (61, 111)]]
    assert va.numRewards[0][0] == [(4, 3)]
    assert va.numRewards[1][0] == [(4, 3), (3, 1)]
    assert va.numRewards[1][1] == [(1, 1), (1, 2)]
    assert va.rewardPI[0] == pytest.approx((3 / 5, 1 / 2, np.nan), nan_ok=True)
    assert va.rewardPI[1] == pytest.approx((1 / 2, -1 / 3, np.nan), nan_ok=True)


def test_legacy_policy_remains_selectable():
    va = _counting_analysis("midline")
    va.bySyncBucket()
    assert va.sync_bucket_ranges == [[(11, 61), (61, 111)]]
    assert va.numRewards[1][0] == [(2, 3), (1, 1)]
    assert va.numRewards[1][1] == [(0, 1), (0, 2)]


def test_default_counts_target_entries_without_any_control_entry():
    va = _counting_analysis(control_entries=())
    va.bySyncBucket()
    assert va.rewardPI[0][:2] == (1, 1)


def test_no_actual_reward_leaves_pi_missing():
    va = _counting_analysis()
    va.on = np.array([])
    va.bySyncBucket()
    assert va.sync_bucket_ranges == [[]]
    assert np.all(np.isnan(va.rewardPI))


@pytest.mark.parametrize("policy,counts", [(None, (4, 3)), ("midline", (2, 1))])
def test_per_bucket_reward_metrics_use_selected_pi_policy(policy, counts):
    va = _counting_analysis(policy)
    va.xf = SimpleNamespace(fctr=1)
    va.ct = SimpleNamespace(pxPerMmFloor=lambda: 1)
    va._f2min = lambda frames: frames / 50
    va.is_excluded_pair = lambda *args: False
    for f, trx in enumerate(va.trx):
        trx.f = f
        trx._bad = False
        trx.distTrav = lambda start, stop: 1000  # one meter per bucket
    va._rewards_per_distance(silent=True)
    va._rewards_per_minute_by_sync_bucket(silent=True)
    for attr in ("rwdsPerDist", "rwdsPerMinBySyncBucket"):
        rows = getattr(va, attr)
        assert rows[0][:2] == pytest.approx((counts[0], 3))
        assert rows[1][:2] == pytest.approx((counts[1], 1))


@pytest.mark.parametrize("policy", [None, "reward", "midline", "control", "fixed"])
def test_training_rpm_excludes_t0_and_counts_through_training_end(policy):
    va = _counting_analysis(policy, control_entries=())
    va.fps = 1
    # Include a reward immediately after T0 and one in the final partial bucket;
    # exclude entries before T0, at T0, and at the exclusive training stop.
    va.trx[0].en[0] = np.array([5, 10, 11, 20, 120, 129, 130])
    va.rewardsPerMinute()
    assert va.rewardsPerMin == pytest.approx([(4 / ((130 - 11) / 60),)])


def test_training_rpm_with_only_t0_reward_is_zero():
    va = _counting_analysis()
    va.fps = 1
    va.on = va.trx[0].en[0] = np.array([10])
    va.rewardsPerMinute()
    assert va.rewardsPerMin == [(0,)]


@pytest.mark.parametrize("actual_rewards", [(), (129,)])
def test_training_rpm_missing_or_empty_window_is_nan(actual_rewards):
    va = _counting_analysis()
    va.fps = 1
    va.on = np.array(actual_rewards)
    va.rewardsPerMinute()
    assert np.isnan(va.rewardsPerMin[0][0])
