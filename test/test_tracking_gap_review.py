from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_tracking_gap_endpoints import gap_rows
from src.analysis.tracking_gap_review import (
    add_observed_controls, compare_annotations, finalize_episodes, gap_candidates,
    match_events, merge_review_windows, reconstructed_entries, sample_candidates,
)


def context(x, *, first=False):
    return dict(video_path="example.avi", pair=1, role="exp", physical_id=1,
                x=np.asarray(x, float), y=np.zeros(len(x)), fps=2., px_per_mm=1.,
                floor=((-30,-30),(30,30)), border_px=1.,
                geometries=[dict(training=1 if first else 2, active_start_frame=0,
                                 active_stop_frame=len(x), training_start_frame=2 if first else 0,
                                 training_stop_frame=len(x), reward=(0,0,2), control=(0,0,4),
                                 sync_buckets=[[0,4],[4,len(x)]])])


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("x", [
    [6,np.nan,np.nan,0,6,0,6],
    [6,2.5,np.nan,0,6,np.nan,6],
    [np.nan,np.nan,0,6,np.nan,np.nan,0],
    [0,np.nan,0,6,0,np.nan,np.nan],
])
def test_reconstruction_matches_actual_calc_rewards_with_interpolated_gaps(x, first):
    from src.analysis.trajectory import Trajectory
    c = context(x, first=first)
    expected = reconstructed_entries(c)
    trj = Trajectory.__new__(Trajectory)
    trj.nan = np.isnan(c["x"])
    # Deliberately place every interpolated point inside both circles. The actual
    # production routine must remove these fabricated inside states during gaps.
    trj.x = np.where(trj.nan, 0., c["x"])
    trj.y = c["y"].copy()
    trj.f, trj._bad = 1, True
    trj.opts = SimpleNamespace(cTurnAnlyz=False, showRewardMismatch=False)
    g = c["geometries"][0]
    # Production branches on training n, and explicitly splits pre for n=1.
    t = SimpleNamespace(n=g["training"], start=g["training_start_frame"],
                        stop=g["training_stop_frame"], postStop=g["active_stop_frame"],
                        circles=lambda _: [g["reward"], g["control"]])
    trj.va = SimpleNamespace(circle=True, trns=[t], startPre=0)
    trj._circle_masks = lambda *args: {}
    trj._checkRewards = lambda *args: None
    with redirect_stdout(io.StringIO()):
        trj._calcRewards()
    for i,name in enumerate(("reward", "control")):
        np.testing.assert_array_equal(expected[g["training"],name], trj.en[i])


def test_border_after_recovery_can_delay_an_entry():
    c = context([6,np.nan,2.5,2.5,0,6])
    assert reconstructed_entries(c)[2,"reward"].tolist() == [4]


def test_outside_outside_candidate_is_kept_and_classification_is_not_safety():
    c = context([6,np.nan,np.nan,6,0,6])
    rows = gap_candidates(c, gap_rows(c), near_mm=.1, long_seconds=.5)
    reward = next(r for r in rows if r["circle"]=="reward")
    assert reward["before_state"] == reward["after_state"] == "outside"
    assert reward["stratum"] == "long_gap_same_state"


def test_stratified_sample_is_reproducible_and_reports_population():
    c = context([6,np.nan,0,6,np.nan,0,6,np.nan,0])
    candidates = gap_candidates(c, gap_rows(c))
    a = sample_candidates(candidates, 1, 4)
    assert a == sample_candidates(candidates, 1, 4)
    assert all(r["stratum_population"]>=r["stratum_sample"]==1 for r in a)


def test_merge_overlapping_windows_prevents_duplicate_scoring():
    rows = [dict(video_path="v",fly=1,role="exp",training=2,circle="reward",
                 review_start_frame=a,review_stop_frame_exclusive=b,gap_id=i,stratum="state_change")
            for i,(a,b) in enumerate([(5,10),(8,15),(15,18)])]
    episodes = merge_review_windows(rows)
    assert [(r["review_start_frame"],r["review_stop_frame_exclusive"]) for r in episodes] == [(5,15),(15,18)]
    assert episodes[0]["selected_gap_ids"] == [0,1]


def test_observed_controls_are_finite_and_do_not_overlap_scored_gaps():
    c = context([6]*50)
    c["x"][20:24] = np.nan
    candidates = gap_candidates(c,gap_rows(c))
    episodes = merge_review_windows(candidates)
    all_episodes = add_observed_controls(episodes,[c],2,10)
    for e in all_episodes:
        if e["source"] != "observed_control":
            continue
        a,b=e["review_start_frame"],e["review_stop_frame_exclusive"]
        assert np.isfinite(c["x"][a-1:b]).all()
        for other in all_episodes:
            if other is not e and other["circle"]==e["circle"]:
                assert min(b,other["review_stop_frame_exclusive"]) <= max(a,other["review_start_frame"])


def example_episode():
    return dict(review_id="a",video_path="v",fly=1,role="exp",circle="reward",source="gap_review",
                review_start_frame=10,review_stop_frame_exclusive=20,calculated_entry_frames="[12]")


def annotation(status="pending", frames="", confidence=""):
    return dict(review_id="a",review_status=status,observed_entry_frames=frames,confidence=confidence,notes="")


def test_pending_and_unscorable_are_not_counted_as_zero_errors():
    for status in ("pending", "unscorable"):
        row, = compare_annotations([example_episode()],[annotation(status)])
        assert row["observed_count"] is None
        assert row["observed_minus_calculated"] is None
        assert row["reviewed_frames"] == 0


def test_explicit_zero_and_timing_only_disagreement_are_distinguished():
    row, = compare_annotations([example_episode()],[annotation("reviewed","[]","high")])
    assert row["observed_minus_calculated"] == -1
    row, = compare_annotations([example_episode()],[annotation("reviewed","[17]","high")],tolerance=2)
    assert row["observed_minus_calculated"] == 0
    assert row["matched_events"] == 0
    assert row["unmatched_calculated_frames"] == "[12]"
    assert row["unmatched_observed_frames"] == "[17]"


@pytest.mark.parametrize("row", [annotation("reviewed","","high"),annotation("reviewed","[20]","high"),
                                  annotation("reviewed","[12,12]","high"),annotation("reviewed","[true]","high"),
                                  annotation("reviewed","[12]",""),annotation("pending","[]")])
def test_invalid_annotations_fail(row):
    with pytest.raises(ValueError):
        compare_annotations([example_episode()],[row])


def test_missing_or_duplicate_review_ids_fail():
    with pytest.raises(ValueError):
        compare_annotations([example_episode()],[])
    with pytest.raises(ValueError):
        compare_annotations([example_episode()],[annotation(),annotation()])


def test_temporal_matching_keeps_count_differences_separate():
    matches,extra,missed=match_events([10,20],[11,15,20],1)
    assert matches==[(10,11),(20,20)] and extra==[] and missed==[15]


def test_finalized_review_ids_and_events_are_stable():
    c=context([6,np.nan,0,6,0])
    selected=gap_candidates(c,gap_rows(c))
    episodes=merge_review_windows(selected)
    eventmap={(c["video_path"],c["pair"],c["role"]):reconstructed_entries(c)}
    rows=finalize_episodes(episodes,[c],eventmap)
    assert rows==finalize_episodes(episodes,[c],eventmap)
    for row in rows:
        frames=json.loads(row["calculated_entry_frames"])
        assert all(row["review_start_frame"]<=f<row["review_stop_frame_exclusive"] for f in frames)
