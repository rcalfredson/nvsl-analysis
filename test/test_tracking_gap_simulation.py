"""Portable tests for empirical-gap placement and counterfactual comparisons."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.tracking_gap_simulation import (
    true_runs,candidate_starts,duration_bin,endpoint_features,matching_candidates,
    compare_episodes,exposed_candidates,
)
from scripts import simulate_tracking_gaps as simulation
from src.analysis.trajectory import Trajectory


def test_candidates_never_bridge_real_gaps_and_preserve_context():
    mask = np.ones(30,dtype=bool)
    mask[10:15] = False
    runs = true_runs(mask,2,28)
    assert runs == [(2,10),(15,28)]
    starts,parents = candidate_starts(runs,3,2)
    assert list(starts) == [18,19,20,21,22]
    for start,index in zip(starts,parents):
        a,b = runs[index]
        assert a<=start-3 and start+3+2<b
        assert mask[start-3:start+3+3].all()
    assert not candidate_starts(runs,50,0)[0].size


def test_duration_bins_retain_boundary_values():
    assert [duration_bin(x) for x in [.5,.5001,2,2.01,10,10.1]] == ["<=0.5s","0.5-2s","0.5-2s","2-10s","2-10s",">10s"]
    with pytest.raises(ValueError):
        duration_bin(0)


def test_endpoint_features_do_not_use_the_hidden_interior():
    x=np.array([1.,2.,3.,4.,5.,6.,7.]);y=np.ones(7)*4
    kwargs=dict(floor=((0,0),(10,10)),scale=1.,reward_center=(5,5),wells=(1.,[(2,2),(8,8)]))
    before=endpoint_features(x,y,[2],3,**kwargs)
    x[2:5]=999.;y[2:5]=-999.
    after=endpoint_features(x,y,[2],3,**kwargs)
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])
    assert before['endpoint_chord_mm'][0] == 4


def test_matching_keeps_wall_categories_and_rejects_unsupported_geometry():
    features=dict(before_wall_bin=np.array([0,1,0]),after_wall_bin=np.array([0,0,0]),
                  endpoint_chord_mm=np.array([3.,3.,7.]))
    donor=dict(before_wall_bin=0,after_wall_bin=0,endpoint_chord_mm=3.)
    assert list(matching_candidates(features,donor,2.)) == [0]
    donor['endpoint_chord_mm']=100.
    assert not matching_candidates(features,donor,2.).size


def ep(a,b,v):
    return dict(start=a,stop=b,value=v)


def test_comparison_exposes_splits_merges_and_timing_changes():
    split=compare_episodes([ep(2,10,1)],[ep(2,5,1),ep(6,10,0)])
    assert split['episode_count_change']==1 and split['reference_overlapping_multiple']==1
    merge=compare_episodes([ep(2,5,1),ep(6,10,0)],[ep(2,10,1)])
    assert merge['episode_count_change']==-1 and merge['simulated_overlapping_multiple']==1
    timed=compare_episodes([ep(2,10,1)],[ep(3,11,0)])
    assert timed['one_to_one_overlap_pairs']==1
    assert timed['paired_value_mean_error']==-1
    assert timed['paired_start_mean_absolute_shift_frames']==1
    assert timed['reference_without_overlap']==timed['simulated_without_overlap']==0


def test_disjoint_and_empty_cells_are_not_zero_error_measurements():
    stats=compare_episodes([ep(2,5,1)],[ep(5,10,0)])
    assert stats['reference_without_overlap']==stats['simulated_without_overlap']==1
    assert np.isnan(stats['paired_value_mean_error'])
    empty=compare_episodes([],[])
    assert empty['reference_episodes']==0 and np.isnan(empty['paired_value_mean_error'])


def test_targeted_placement_is_explicit_support_overlap():
    support=np.zeros(20,dtype=bool);support[5:8]=True
    assert list(exposed_candidates(np.array([1,3,7,8]),3,support))==[1,2]


def test_all_tracker_fields_are_masked_without_mutating_source(monkeypatch):
    class WallStub:
        def __init__(self,trj,va):
            self.return_data={'boundary_event_stats':{}}
        def calc_dist_boundary_to_ellipse(self,**kwargs):
            assert kwargs['exclude_lost_frames']
    monkeypatch.setattr(simulation,'EllipseToBoundaryDistCalculator',WallStub)
    raw={f:[np.arange(10,dtype=float)+2 if f=='x' else np.ones(10)*2] for f in simulation.FIELDS}
    va=SimpleNamespace(trxRw=raw,opts=SimpleNamespace(),ct=SimpleNamespace(pxPerMmFloor=lambda:1.),
                       openLoop=False,on=np.array([],dtype=int))
    context=dict(analysis=va,physical_id=0,geometries=[])
    _,trj=simulation.build_trajectory(context,(3,6))
    assert trj.nan.tolist()==[False]*3+[True]*3+[False]*4
    np.testing.assert_allclose(trj.x,np.arange(10)+2)
    assert np.isnan(trj.w[3:6]).all() and np.isnan(trj.h[3:6]).all()
    assert np.isfinite(trj.theta[3:6]).all()
    assert all(np.isfinite(raw[f][0]).all() for f in simulation.FIELDS)
    for gap in [(0,3),(3,10),(5,5)]:
        with pytest.raises(ValueError):simulation.build_trajectory(context,gap)


def test_linear_interpolation_can_hide_an_entire_turnback():
    class Training:
        start,stop=0,6
        def isCircle(self):return True
        def circles(self,f):return [(0.,0.,1.)]
    trn=Training()
    va=SimpleNamespace(ct=SimpleNamespace(pxPerMmFloor=lambda:1.),xf=SimpleNamespace(fctr=1.))
    def trajectory(x):
        t=Trajectory.__new__(Trajectory)
        t.x=np.array(x,dtype=float);t.y=np.zeros(6);t.f=0;t.va=va
        return t
    reference=trajectory([2,2,4,4,2,2])
    hidden=trajectory([2,2,2,2,2,2]) # chord after hiding frames 2 and 3
    before=reference.reward_turnback_dual_circle_episodes_for_training(trn=trn,inner_radius_mm=3,outer_radius_mm=5)
    after=hidden.reward_turnback_dual_circle_episodes_for_training(trn=trn,inner_radius_mm=3,outer_radius_mm=5)
    assert len(before)==1 and before[0]['turns_back']
    assert after==[]


def test_output_folder_refuses_overwrite(tmp_path):
    (tmp_path/'annotations.csv').write_text('preserve')
    with pytest.raises(ValueError):simulation.empty_folder(tmp_path)
