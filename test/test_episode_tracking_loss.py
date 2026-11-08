"""Synthetic checks for interval support, selection, and clustered comparisons."""
import numpy as np
import pytest

from src.analysis.episode_tracking_loss import GapIndex, reduce_episodes
from scripts.audit_episode_tracking_loss import (
    POLICIES, bootstrap_stats, close_value, episode_row, scalar_values,
    subject_rows, identities, cohort_contrasts, summarize, main,
    index_wall_episodes, Sources,
)


def test_gap_intersection_half_open_edges_and_terminal_runs():
    gaps = GapIndex([True,False,True,True,True,False,True],2)
    assert gaps.exposure(1,2)["lost_frames"] == 0
    x = gaps.exposure(3,6)
    assert x["lost_frames"] == 2
    assert x["gap_count"] == 1
    assert x["max_gap_overlap_s"] == 1
    assert x["max_intersecting_gap_s"] == 1.5
    assert gaps.exposure(0,7)["lost_frames"] == 5
    for start,stop in [(-1,2),(1,1),(3,8)]:
        with pytest.raises(ValueError):
            gaps.exposure(start,stop)


def test_missing_reentry_at_exclusive_stop_is_in_decision_support():
    gaps = GapIndex([False]*5+[True]+[False]*4,10)
    ep = episode_row({},gaps,metric="turnback",variant="3/5",period="T2",
                     start=2,stop=5,value=1,event=5)
    assert ep["episode_lost_frames"] == 0
    assert ep["support_lost_frames"] == 1
    assert ep["outcome_frame_missing"]
    assert reduce_episodes([ep],policy="complete_episode",minimum=1)["value"] == 1
    assert np.isnan(reduce_episodes([ep],policy="complete_support",minimum=1)["value"])


def test_heading_support_extends_past_episode_and_requires_original_mask():
    gaps = GapIndex([False]*6+[True]+[False]*3,10)
    ep = episode_row({},gaps,metric="alignment",variant="3/5",period="T2",start=2,stop=5,
                     value=.7,event=5,support_stop=7,heading_frames=[3,4,5,6])
    assert ep["episode_lost_frames"] == 0
    assert ep["heading_missing_frames"] == 1
    assert ep["support_lost_frames"] == 1


def test_ratio_denominator_includes_failures_and_minimum_is_reapplied():
    eps = [dict(value=v,episode_lost_frames=lost,support_lost_frames=lost)
           for v,lost in [(1,0),(1,0),(0,0),(0,0),(0,0),(1,1)]]
    assert reduce_episodes(eps,policy="baseline")["value"] == .5
    assert reduce_episodes(eps,policy="complete_support")["value"] == .4
    assert np.isnan(reduce_episodes(eps,policy="complete_support",minimum=6)["value"])


def test_frozen_final_sli_gate_prevents_readmission_and_checks_baseline():
    ep = dict(value=1.,episode_lost_frames=0,support_lost_frames=0,
              outcome_frame_missing=False,heading_missing_frames=0,
              episode_max_intersecting_gap_s=0.,episode_frames=3)
    base = dict(cohort="ctrl",video_path="a.avi",pair=0)
    rows = subject_rows(base,[ep]*5,metric="ratio",variant="x",period="T2",
                        expected=1.,eligible=False,reason="undefined paired final SLI")
    assert all(not r["included"] and np.isnan(r["value"]) for r in rows)
    with pytest.raises(ValueError,match="Baseline mismatch"):
        subject_rows(base,[ep]*5,metric="ratio",variant="x",period="T2",
                     expected=.5,eligible=True,reason="")


def test_scalar_identity_formats_and_ambiguous_basename_rejected():
    keys = [("/a/video.avi",0),("/a/video.avi",1)]
    z = dict(per_unit_ids_panel=np.array([["/a/video.avi|fly_id=0|trx_idx=0","video:fly1"]]),
             per_unit_values_panel=np.array([[.2,.3]]))
    assert scalar_values(z,keys) == {keys[0]:.2,keys[1]:.3}
    with pytest.raises(ValueError,match="Ambiguous"):
        scalar_values(z,keys+[("/b/video.avi",0)])
    with pytest.raises(ValueError,match="Duplicate"):
        identities(dict(video_fns=["a","a"],fly_ids=[0,0]))


def test_common_fly_change_separates_attrition_from_measurement_change():
    rows = []
    for video,pair,baseline,clean in [("a",0,0.,1.),("a",1,2.,3.),("b",0,10.,np.nan)]:
        for policy in POLICIES:
            value = baseline if policy=="baseline" else clean
            rows.append(dict(video_path=video,pair=pair,policy=policy,value=value,
                             included=bool(np.isfinite(value)),baseline_n_episodes=10,n_episodes=5,
                             episodes_with_loss=2,episodes_with_support_loss=3))
    stats = bootstrap_stats(rows,300,7)
    base,_,clean = stats
    assert base["mean"] == 4
    assert clean["mean"] == 2
    assert clean["mean_shift_from_full_baseline"] == -2
    assert clean["common_fly_shift"] == 1
    assert clean["n_flies"] == 2 and clean["n_videos"] == 1
    assert clean["common_fly_shift_ci_low"] == clean["common_fly_shift_ci_high"] == 1


def test_nonfinite_baseline_values_must_also_agree():
    close_value(np.nan,np.nan,"empty")
    with pytest.raises(ValueError):
        close_value(np.nan,.2,"missing")


def test_zero_retained_and_zero_baseline_samples_stay_undefined():
    rows = []
    for policy in POLICIES:
        rows.append(dict(video_path="a",pair=0,policy=policy,value=1. if policy=="baseline" else np.nan,
                         included=policy=="baseline",baseline_n_episodes=5,n_episodes=0,
                         episodes_with_loss=5,episodes_with_support_loss=5))
    stats = bootstrap_stats(rows,20,1)
    assert stats[2]["n_flies"] == 0
    assert np.isnan(stats[2]["mean"]) and np.isnan(stats[2]["ci_low"])
    for r in rows:
        r["included"] = False
    stats = bootstrap_stats(rows,20,1)
    assert all(r["baseline_n_flies"]==0 and np.isnan(r["mean"]) for r in stats)


def test_cohort_contrast_and_paired_change_intervals():
    rows = []
    for cohort,base,change in [("ctrl",1.,.1),("pfnd_kir",3.,.4)]:
        for policy in POLICIES:
            rows.append(dict(cohort=cohort,metric="ratio",variant="x",period="T2",video_path=cohort,
                             pair=0,policy=policy,value=base if policy=="baseline" else base+change,
                             included=True,baseline_n_episodes=10,n_episodes=5,
                             episodes_with_loss=2,episodes_with_support_loss=2))
    summaries = summarize(rows,30,42)
    contrasts = cohort_contrasts(summaries,rows,30,42)
    baseline,_,clean = contrasts
    assert baseline["difference"] == 2.
    assert clean["difference"] == pytest.approx(2.3)
    assert clean["change_in_difference"] == pytest.approx(.3)
    assert clean["change_ci_low"] == pytest.approx(.3)
    assert clean["change_ci_high"] == pytest.approx(.3)
    assert clean["bootstrap_valid_draws"] == 30


def test_cli_refuses_to_overwrite_existing_results(tmp_path):
    (tmp_path/"annotations.csv").write_text("preserve me")
    with pytest.raises(SystemExit):
        main(["--output-dir",str(tmp_path)])
    assert (tmp_path/"annotations.csv").read_text() == "preserve me"


def wall_episode(start=2, stop=4, count=2):
    return dict(video_path="a.avi",va_tag="0",training_index="2",fly_role="exp",
                trajectory_index="0",start_reward_frame=str(start),end_reward_frame=str(stop),
                window_start_frame="1",window_end_frame="10",pooled_episode_count=str(count))


def test_wall_episode_validation_rejects_duplicates_truncation_and_gaps():
    keys = [("a.avi",0)]
    first,second = wall_episode(),wall_episode(4,6)
    assert len(index_wall_episodes([first,second],keys)[keys[0]]) == 2
    for rows,match in [([first,first],"Duplicate"),([first],"count mismatch"),
                       ([first,wall_episode(5,6)],"Nonconsecutive"),
                       ([wall_episode(2,10,1)],"outside"),
                       ([dict(first,fly_role="yok")],"Unexpected")]:
        with pytest.raises(ValueError,match=match):
            index_wall_episodes(rows,keys)


def test_duplicate_subject_policy_cannot_silently_replace_results():
    row = dict(video_path="a.avi",pair=0,policy="baseline")
    with pytest.raises(ValueError,match="Duplicate subject/policy"):
        bootstrap_stats([row,row],20,1)


def test_cli_rejects_duplicate_cohort_before_creating_output(tmp_path):
    output = tmp_path/"new_results"
    with pytest.raises(SystemExit):
        main(["--output-dir",str(output),"--cohort","ctrl","--cohort","ctrl"])
    assert not output.exists()


def test_source_snapshot_detects_files_changed_after_read(tmp_path):
    source = tmp_path/"input.txt"
    source.write_text("original")
    sources = Sources(tmp_path)
    sources.track(source)
    sources.verify_unchanged()
    source.write_text("changed contents")
    with pytest.raises(ValueError,match="Source changed during analysis"):
        sources.verify_unchanged()
