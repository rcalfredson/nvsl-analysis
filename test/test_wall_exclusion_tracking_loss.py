import numpy as np
import pandas as pd
import pytest
from src.analysis.wall_exclusion_tracking_loss import classify,frame_counts,reduce_cell


def episodes():
    return pd.DataFrame([dict(metric=m,start_frame=3,stop_frame_exclusive=7,
        support_start_frame=2,support_stop_frame_exclusive=8,value=1.,
        support_lost_frames=1,episode_lost_frames=1) for m in ['tortuosity','turnback_ratio']])


def test_boundary_contact_is_secondary_not_primary_exclusion():
    eps=episodes(); contact=np.zeros(12,dtype=bool);contact[7]=True
    zeros=np.zeros(12,dtype=bool)
    result=classify(eps,contact,~zeros,zeros,zeros,(0,12))
    assert list(result['wall_contact_frames'])==[0,0]
    assert list(result['support_wall_contact_frames'])==[1,1]


def test_tortuosity_uses_production_window_clip_bouts_use_full_interval():
    eps=episodes(); contact=np.zeros(12,dtype=bool);contact[3]=True
    zeros=np.zeros(12,dtype=bool)
    result=classify(eps,contact,~zeros,zeros,zeros,(4,10))
    assert list(result['wall_contact_frames'])==[0,1]
    assert list(result['wall_filter_start'])==[4,3]


def test_wall_exclusion_removes_whole_episodes_and_reapplies_minimum():
    eps=pd.DataFrame([dict(value=float(i),wall_contact_frames=(i==4),
        support_wall_contact_frames=(i>=3),support_lost_frames=(i>=2),episode_lost_frames=0,
        support_observed_near_wall_frames=(i>=1),support_lost_frames_with_near_wall_gap_endpoint=(i==4)) for i in range(5)])
    before=reduce_cell(eps,'wall_allowed'); after=reduce_cell(eps,'exclude_episode_contact')
    assert before['n_episodes']==5 and before['measured_value']==2.
    assert before['exposed_episode_fraction']==.6
    assert after['n_episodes']==4 and np.isnan(after['measured_value'])
    assert after['episode_value_sum']==6. and after['exposed_episodes']==2
    assert reduce_cell(eps,'exclude_support_contact')['n_episodes']==3
    assert len(eps)==5


def test_missing_frames_and_near_wall_endpoints_remain_distinct():
    eps=episodes(); no=np.zeros(12,dtype=bool)
    observed=~no; observed[4:6]=False
    near=no.copy();near[3]=True
    lost_near=no.copy();lost_near[4:6]=True
    result=classify(eps,no,observed,near,lost_near,(0,12))
    assert list(result['support_observed_frames'])==[4,4]
    assert list(result['support_observed_near_wall_frames'])==[1,1]
    assert list(result['support_lost_frames_with_near_wall_gap_endpoint'])==[2,2]
    assert not result['wall_contact_frames'].any()


def test_invalid_intervals_fail_instead_of_wrapping():
    with pytest.raises(ValueError): frame_counts([False]*10,[-1],[2])
    with pytest.raises(ValueError): frame_counts([False]*10,[2],[11])
    with pytest.raises(ValueError): frame_counts([False]*10,[2],[2])
    assert frame_counts([True]*10,[0,4],[4,10]).tolist()==[4,6]


def test_empty_episode_cells_stay_undefined():
    eps=pd.DataFrame(columns=['value','wall_contact_frames','support_wall_contact_frames',
        'support_lost_frames','episode_lost_frames','support_observed_near_wall_frames',
        'support_lost_frames_with_near_wall_gap_endpoint'])
    result=reduce_cell(eps,'exclude_episode_contact')
    assert result['n_episodes']==0 and np.isnan(result['measured_value'])
    assert np.isnan(result['exposed_episode_fraction'])


def test_summary_separates_episode_selection_from_fly_dropout():
    from scripts.audit_wall_exclusion_tracking_loss import summarize
    rows=[]
    for cohort,video,bn,be,bv,an,ae,av in [
        ('ctrl','a',6,2,10.,4,0,np.nan),('ctrl','b',5,0,20.,5,0,20.),
        ('pfnd_kir','c',5,1,5.,5,1,5.)]:
        for policy in ('wall_allowed','exclude_episode_contact','exclude_support_contact'):
            baseline=policy=='wall_allowed'
            n,exposed,value=(bn,be,bv) if baseline else (an,ae,av)
            rows.append(dict(cohort=cohort,video_path=video,pair=0,metric='tortuosity',variant='full',
                period='T2_SB2-5',policy=policy,n_episodes=n,exposed_episodes=exposed,
                value=value,included=bool(np.isfinite(value)),episodes_with_observed_wall_proximity=0,
                exposed_episodes_with_near_wall_gap_endpoint=0))
    s,c=summarize(pd.DataFrame(rows),draws=10,seed=1)
    r=s[s.cohort.eq('ctrl')&s.policy.eq('exclude_episode_contact')].iloc[0]
    assert (r.baseline_flies,r.retained_flies)==(2,1)
    assert r.baseline_exposure==pytest.approx(2/11)
    assert r.episodes_after_wall_filter_before_attrition==9
    assert r.common_fly_baseline_exposure==r.common_fly_after_exposure==0
    assert r.baseline_mean==15 and r.retained_mean==20 and r.common_fly_change==0
    q=c[c.policy.eq('exclude_episode_contact')&c.contrast.eq('pfnd_kir - ctrl')].iloc[0]
    assert q.common_fly_change==0 and q.composition_change==-5 and q.total_change==-5
