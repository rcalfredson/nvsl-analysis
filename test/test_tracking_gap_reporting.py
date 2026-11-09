"""Reporting regressions: denominators, eligibility attrition and missing support."""
import numpy as np
import pandas as pd
import pytest

from scripts import summarize_tracking_gap_pilot as report
from scripts import report_tracking_gap_simulation as combined_report
from scripts.tracking_gap_report_tables import (
    REDETECTION_COLUMNS, FIXED_COLUMNS,
    load_pilot, trial_audit,
)


def write_fixture(root, *, unsupported=False, spatial_only=False):
    pilot = root/'pilot'
    old = root/'old'
    pilot.mkdir(); old.mkdir()
    arm = 'spatial_matched' if spatial_only else 'episode_targeted:agarose_avoidance'
    trials, evaluations, episodes = [], [], []
    # A well-supported comparison, the small-denominator maximum, and attrition.
    specs = [('enough', 6, 3, 5, 2), ('small', 2, 1, 1, 0), ('lost', 5, 3, 4, 2)]
    for trial, n, num, ns, nums in specs:
        trials.append(dict(trial_id=trial, cohort='ctrl', video_path='/video.avi', pair=0,
            placement_arm=arm, status='unsupported' if unsupported else 'matched',
            duration_bin='0.5-2s', donor_gap_id=n, duration_s=1., gap_frames=10,
            evaluation_start_frame=np.nan if unsupported else 100,
            evaluation_stop_frame_exclusive=np.nan if unsupported else 500))
        if unsupported:
            continue
        row = dict.fromkeys(REDETECTION_COLUMNS, 0)
        row.update(trial_id=trial, cohort='ctrl', duration_bin='0.5-2s',
            metric='agarose_avoidance', variant='physical_1mm',
            reference_mean_unfiltered=num/n, simulated_mean_unfiltered=nums/ns,
            mean_change=nums/ns-num/n, reference_episodes=n, simulated_episodes=ns,
            reference_meets_five_episode_minimum=n >= 5,
            simulated_meets_five_episode_minimum=ns >= 5, episode_count_change=ns-n)
        evaluations.append(row)
        for condition, count, numerator in [('reference', n, num), ('simulated', ns, nums)]:
            for index in range(count):
                episodes.append(dict(trial_id=trial, condition=condition, metric='agarose_avoidance',
                    variant='physical_1mm', start_frame=110+index*10,
                    stop_frame_exclusive=115+index*10, value=int(index < numerator)))
    pd.DataFrame(trials).to_csv(pilot/'trials.csv', index=False)
    # Match the runner: files for empty row lists may be absent.
    if evaluations:
        pd.DataFrame(evaluations).to_csv(pilot/'redetection.csv', index=False)
        pd.DataFrame(episodes).to_csv(pilot/'episodes.csv', index=False)
    pd.DataFrame(columns=FIXED_COLUMNS).to_csv(pilot/'fixed_boundaries.csv', index=False)
    pd.DataFrame([dict(primary_donor_gaps=100)]).to_csv(old/'real_gap_summary.csv', index=False)
    return pilot, old


def agarose(table):
    return table[table.metric.eq('agarose_avoidance')].iloc[0]


def test_trial_numerators_and_eligibility_are_reconstructed_and_validated(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    t, r, _, ep = load_pilot(pilot)
    details = trial_audit(t, r, ep)
    small = details[(details.trial_id == 'small') & (details.metric == 'agarose_avoidance')].iloc[0]
    assert (small.reference_numerator, small.reference_episodes) == (1, 2)
    assert (small.simulated_numerator, small.simulated_episodes) == (0, 1)
    assert small.displayed_change == -50
    assert small.eligibility_transition == 'below minimum in both'
    assert small.evaluation_duration_s == 40
    assert small.reference_episode_intervals == '110:115|120:125'
    # An absent evaluated metric is unknown, not a fabricated zero denominator.
    absent = details[(details.trial_id == 'small') & (details.metric == 'tortuosity')].iloc[0]
    assert absent.evaluation_status == 'no evaluation rows'
    assert np.isnan(absent.reference_episodes)
    with pytest.raises(ValueError, match='Episode count mismatch'):
        trial_audit(t, r, ep.iloc[1:])


def test_minimum_filter_excludes_small_cases_but_preserves_attrition_counts(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    t, r, _, _ = load_pilot(pilot)
    all_means = agarose(report.detection_table(r, t))
    filtered = agarose(report.detection_table(r, t, eligible_only=True))
    assert all_means['Trials with both means'] == 3
    assert all_means['Maximum absolute change'] == 50
    assert filtered['Trials with both means'] == 1
    assert filtered['Mean signed change'] == pytest.approx(-10)
    assert filtered['Lost eligibility'] == 1
    assert filtered['Below minimum in both'] == 1
    assert filtered['Eligible in both'] == 1
    assert filtered['Matched trials'] == 3


@pytest.mark.parametrize('unsupported,spatial_only', [(True, False), (False, True)])
def test_companion_renders_unsupported_and_spatial_only_pilots(tmp_path, unsupported, spatial_only):
    pilot, old = write_fixture(tmp_path, unsupported=unsupported, spatial_only=spatial_only)
    out = tmp_path/'out'
    report.main(['--pilot-dir', str(pilot), '--existing-report-dir', str(old), '--output-dir', str(out)])
    targeted = pd.read_csv(out/'targeted_detection.csv')
    assert len(targeted) == 10
    if spatial_only:
        assert targeted['Status'].eq('arm not requested').all()
    else:
        assert agarose(targeted)['Status'] == 'no matched placements'
    assert targeted['Mean signed change'].isna().all()
    highlights = pd.read_csv(out/'at_a_glance.csv')
    assert len(highlights) == 6
    assert highlights['Observed response'].str.contains('undefined').all()
    if unsupported:
        details = pd.read_csv(out/'trial_details.csv')
        assert details.reference_episodes.isna().all()
        assert details.evaluation_status.eq('unsupported placement').all()
    assert 'Trial-level denominators' in (out/'report.html').read_text()


def test_no_exposure_is_not_zero_error(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    t, _, f, _ = load_pilot(pilot)
    # A requested/matched numeric arm with no fixed measurements.
    t['placement_arm'] = 'episode_targeted:home_vector_alignment'
    table = report.fixed_table(f, t)
    assert table['Median absolute error'].isna().all()
    assert table.loc[table.metric.eq('home_vector_alignment'), 'Status'].eq('no evaluation rows').all()


def test_inconsistent_minimum_flag_is_rejected(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    rows = pd.read_csv(pilot/'redetection.csv')
    rows.loc[0, 'reference_meets_five_episode_minimum'] = False
    rows.to_csv(pilot/'redetection.csv', index=False)
    with pytest.raises(ValueError, match='episode-minimum flag'):
        load_pilot(pilot)


def test_denominator_strata_preserve_the_small_case_and_eligibility_loss(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    t, r, _, ep = load_pilot(pilot)
    table = report.denominator_table(trial_audit(t, r, ep)).set_index('Reference episode count')
    assert table.loc['1–4', 'Maximum absolute change'] == 50
    assert table.loc['5–9', 'Evaluated trials'] == 2
    assert table.loc['5–9', 'Lost eligibility'] == 1


def test_unavailable_exposed_measurement_does_not_become_zero_error(tmp_path):
    pilot, _ = write_fixture(tmp_path)
    row = dict.fromkeys(FIXED_COLUMNS, 0)
    row.update(trial_id='enough', cohort='ctrl', duration_bin='0.5-2s',
        metric='home_vector_alignment', variant='13/15mm', available=False,
        support_intersects_injected_gap=True, reference_value=.7, simulated_value=np.nan,
        error=np.nan)
    pd.DataFrame([row]).to_csv(pilot/'fixed_boundaries.csv', index=False)
    t, _, f, _ = load_pilot(pilot)
    t['placement_arm'] = 'episode_targeted:home_vector_alignment'
    result = report.fixed_table(f, t)
    result = result[(result.metric == 'home_vector_alignment') & (result.variant == '13/15mm')].iloc[0]
    assert result['Unavailable after loss'] == 1
    assert result['Available after loss'] == 0
    assert np.isnan(result['Mean signed error'])


def test_combined_report_handles_no_matches_or_evaluation_files(tmp_path):
    pilot, _ = write_fixture(tmp_path, unsupported=True)
    profile = tmp_path/'profile'; profile.mkdir()
    pd.DataFrame([dict(cohort='ctrl', video_path='/video.avi', pair=0,
        gap_start_frame=20, gap_stop_frame_exclusive=30, fps=10., primary_donor=True,
        duration_s=1., gap_frames=10, before_wall_mm=.5, after_wall_mm=2.)]).to_csv(profile/'gaps.csv', index=False)
    pd.DataFrame([dict(cohort='ctrl', scope='T2_SB2-5_donors', duration_bin='0.5-2s',
        lost_time_fraction=1.)]).to_csv(profile/'summary.csv', index=False)
    pd.DataFrame([dict(finite_values_at_lost_frames=0, nonfinite_values_at_tracked_frames=0)]).to_csv(profile/'missing_fields.csv', index=False)
    out = tmp_path/'combined'
    combined_report.main(['--profile-dir', str(profile), '--pilot-dir', str(pilot), '--output-dir', str(out)])
    assert (out/'report.html').exists()
    assert (out/'audited_pilot/report.html').exists()
    assert pd.read_csv(out/'pilot_effects.csv').empty
