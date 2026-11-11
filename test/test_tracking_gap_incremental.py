"""Full-window rules and contrast decomposition for incremental loss."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.analysis.tracking_gap_incremental import (
    aggregate, contrast_components, episode_changes, stable_rng, video_bootstrap_weights,
)


def row(cohort, video, before, after, repeat=0, eligible=True):
    return dict(cohort=cohort, video_path=video, before_value=before,
                after_value=after, repeat=repeat, after_eligible=eligible)


def test_minimum_is_applied_to_whole_window_and_undefined_is_retained():
    assert not aggregate([dict(value=2.)]*4)['eligible']
    assert np.isnan(aggregate([dict(value=2.)]*4)['value'])
    assert aggregate([dict(value=2.)]*5) == dict(
        n_episodes=5, episode_value_sum=10., eligible=True, value=2.)
    with pytest.raises(ValueError):
        aggregate([dict(value=np.nan)])


def test_contrast_separates_numeric_and_composition_effects():
    rows = [row('p', 'a', 10., 11.), row('p', 'b', 20., np.nan, eligible=False),
            row('n', 'c', 4., 6.)]
    r = contrast_components(rows, 'p', 'n')
    assert r['before_contrast'] == 11.
    assert r['before_common_contrast'] == 6.
    assert r['after_contrast'] == 5.
    assert r['common_fly_change'] == -1.
    assert r['composition_change'] == -5.
    assert r['total_change'] == r['common_fly_change']+r['composition_change'] == -6.


def test_newly_eligible_fly_is_kept_without_inventing_its_baseline():
    rows = [row('p', 'a', 10., 11.), row('p', 'b', 20., np.nan, eligible=False),
            row('n', 'c', 4., 6.), row('p', 'd', np.nan, 30.)]
    r = contrast_components(rows, 'p', 'n')
    assert r['before_contrast'] == 11.
    assert r['after_contrast'] == 14.5
    assert r['common_fly_change'] == -1.
    assert r['eligibility_loss_shift'] == -5.
    assert r['eligibility_gain_shift'] == 9.5
    assert r['composition_change'] == 4.5
    assert r['total_change'] == 3.5


def test_repeats_have_equal_weight_and_cluster_draw_reuses_video_weight():
    rows = [row('p', 'a', 10., 12.), row('p', 'a', 20., 24.), row('n', 'b', 5., 5.),
            row('p', 'a', 10., 14., repeat=1), row('p', 'a', 20., 28., repeat=1),
            row('n', 'b', 5., 5., repeat=1)]
    assert contrast_components(rows, 'p', 'n')['total_change'] == 4.5
    assert contrast_components(rows, 'p', 'n', {'a': 2, 'b': 1})['total_change'] == 4.5
    assert np.isnan(contrast_components(rows, 'p', 'n', {'a': 1})['total_change'])


def test_undefined_repeat_is_not_silently_dropped():
    rows = [row('p', 'a', 10., 12.), row('n', 'b', 5., 5.),
            row('p', 'a', 10., np.nan, repeat=1, eligible=False), row('n', 'b', 5., 5., repeat=1)]
    assert np.isnan(contrast_components(rows, 'p', 'n')['after_contrast'])
    assert np.isnan(contrast_components([], 'p', 'n')['total_change'])


def test_episode_membership_does_not_call_boundary_changes_misses():
    a = [dict(start=0, stop=10, value=1.), dict(start=20, stop=30, value=4.)]
    b = [dict(start=0, stop=10, value=2.), dict(start=21, stop=30, value=4.)]
    assert episode_changes(a, b) == dict(episodes_no_longer_present=1,
        episodes_newly_present=1, same_boundary_episodes=1, same_boundary_values_changed=1)
    with pytest.raises(ValueError):
        episode_changes(a+a, b)


def test_rng_is_identity_stable_and_does_not_reuse_repeat_streams():
    a = stable_rng(1, 'fly-a', 0).integers(100000, size=10)
    np.testing.assert_array_equal(a, stable_rng(1, 'fly-a', 0).integers(100000, size=10))
    assert not np.array_equal(a, stable_rng(1, 'fly-a', 1).integers(100000, size=10))


def test_video_bootstrap_keeps_strata_sizes_and_ignores_repeat_multiplicity():
    rows = [row('p', 'a', 1., 1.), row('p', 'a', 2., 2.),
            row('p', 'b', 1., 1.), row('n', 'c', 1., 1.)]
    weights = video_bootstrap_weights(rows, stable_rng(7))
    assert weights.get('a', 0)+weights.get('b', 0) == 2
    assert weights['c'] == 1
    assert weights == video_bootstrap_weights(rows+rows, stable_rng(7))
    with pytest.raises(ValueError):
        video_bootstrap_weights(rows+[row('n', 'a', 1., 1.)], stable_rng(7))


def test_report_keeps_entirely_unsupported_cells_undefined(tmp_path):
    import json
    from scripts.simulate_incremental_tracking_loss import summaries, report
    from scripts.report_incremental_tracking_loss import readout
    coverage = pd.DataFrame([dict(arm='empirical_spatial', metric='tortuosity',
        cohort='ctrl', video_path='a', pair=0, supported=False, baseline_eligible=True,
        baseline_value=3., donor_gaps=2, supported_donors=0,
        donor_lost_frames=100, supported_donor_lost_frames=0)])
    frozen = pd.DataFrame([dict(metric='tortuosity', cohort='ctrl', included=True, value=3.)])
    cohorts, contrasts = summaries(coverage, pd.DataFrame(), frozen, draws=2, seed=1)
    assert cohorts.supported_flies.eq(0).all()
    assert contrasts.total_change.isna().all()
    assert contrasts.total_change_bootstrap_defined.eq(0).all()
    report(tmp_path, cohorts, contrasts, SimpleNamespace(repeats=4))
    assert 'undefined' in (tmp_path/'report.html').read_text()
    for name, data in [('cohort_summary', cohorts), ('contrasts', contrasts),
                       ('coverage', coverage), ('trials', pd.DataFrame())]:
        data.to_csv(tmp_path/f'{name}.csv', index=False)
    (tmp_path/'provenance.json').write_text(json.dumps(dict(subjects=1, settings=dict(repeats=4))))
    readout(tmp_path, tmp_path/'readout')
    assert '0 artificial-gap trials' in (tmp_path/'readout/report.html').read_text()


def test_report_counts_unique_eligibility_changes_not_mask_repeats():
    from scripts.simulate_incremental_tracking_loss import summaries
    coverage = pd.DataFrame([dict(arm='empirical_spatial', metric='tortuosity',
        cohort='ctrl', video_path='a', pair=0, supported=True, baseline_eligible=True,
        baseline_value=3., donor_gaps=2, supported_donors=2,
        donor_lost_frames=100, supported_donor_lost_frames=100)])
    frozen = pd.DataFrame([dict(metric='tortuosity', cohort='ctrl', included=True, value=3.)])
    trials = pd.DataFrame([dict(arm='empirical_spatial', metric='tortuosity',
        cohort='ctrl', video_path='a', pair=0, repeat=i, before_value=3., after_value=np.nan,
        before_eligible=True, after_eligible=False, value_change=np.nan, added_seconds=1.) for i in range(4)])
    cohorts, _ = summaries(coverage, trials, frozen, draws=2, seed=1)
    ctrl = cohorts[cohorts.cohort.eq('ctrl')].iloc[0]
    assert ctrl.flies_losing_eligibility == 1
    assert ctrl.undefined_after_trials == 4
    assert ctrl.flies_gaining_eligibility == 0
