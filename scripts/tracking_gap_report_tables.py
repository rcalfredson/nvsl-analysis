"""Validated, denominator-aware tables shared by tracking-gap pilot reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = {
    'tortuosity': ('full',), 'wall_contacts': ('full',),
    'wall_contact_length': ('full',),
    'turnback_ratio': ('3/5mm', '8/10mm', '13/15mm'),
    'home_vector_alignment': ('3/5mm', '8/10mm', '13/15mm'),
    'agarose_avoidance': ('physical_1mm',),
}
RATIOS = ('turnback_ratio', 'agarose_avoidance')
REDETECTION_COLUMNS = [
    'trial_id', 'cohort', 'duration_bin', 'metric', 'variant',
    'reference_mean_unfiltered', 'simulated_mean_unfiltered', 'mean_change',
    'reference_meets_five_episode_minimum', 'simulated_meets_five_episode_minimum',
    'reference_episodes', 'simulated_episodes', 'episode_count_change',
    'reference_without_overlap', 'simulated_without_overlap',
    'reference_overlapping_multiple', 'simulated_overlapping_multiple',
    'one_to_one_overlap_pairs', 'paired_value_mean_error',
    'paired_value_mean_absolute_error', 'paired_start_mean_absolute_shift_frames',
    'paired_stop_mean_absolute_shift_frames',
]
FIXED_COLUMNS = [
    'trial_id', 'cohort', 'duration_bin', 'support_intersects_injected_gap',
    'metric', 'variant', 'start_frame', 'stop_frame_exclusive',
    'support_start_frame', 'support_stop_frame_exclusive', 'reference_value',
    'simulated_value', 'available', 'error',
]
EPISODE_COLUMNS = ['trial_id', 'condition', 'metric', 'variant', 'start_frame',
                   'stop_frame_exclusive', 'value']


def read_output(path, columns):
    """The pilot writer omits an output file when it has no rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        result = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(result)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return result


def load_pilot(folder):
    folder = Path(folder)
    trials = pd.read_csv(folder/'trials.csv')
    if 'placement_arm' not in trials:
        trials['placement_arm'] = 'spatial_matched'
    if trials.trial_id.duplicated().any():
        raise ValueError('Duplicate trial IDs')
    if not trials.status.isin(['matched', 'unsupported']).all():
        raise ValueError('Unknown placement status')
    frames = [read_output(folder/name, columns) for name, columns in (
        ('redetection.csv', REDETECTION_COLUMNS), ('fixed_boundaries.csv', FIXED_COLUMNS),
        ('episodes.csv', EPISODE_COLUMNS))]
    if trials.status.eq('matched').any() and frames[0].empty:
        raise ValueError('Matched trials exist but redetection.csv has no evaluations')
    for data in frames:
        if not data.trial_id.isin(trials.loc[trials.status.eq('matched'), 'trial_id']).all():
            raise ValueError('Evaluation references an unknown or unmatched trial')
    r, f, episodes = frames
    for data, nonnumeric in (
        (r, {'trial_id', 'cohort', 'duration_bin', 'metric', 'variant',
             'reference_meets_five_episode_minimum', 'simulated_meets_five_episode_minimum'}),
        (f, {'trial_id', 'cohort', 'duration_bin', 'metric', 'variant',
             'available', 'support_intersects_injected_gap'}),
        (episodes, {'trial_id', 'condition', 'metric', 'variant'}),
    ):
        for column in set(data) - nonnumeric:
            data[column] = pd.to_numeric(data[column], errors='raise')
    if r.duplicated(['trial_id', 'metric', 'variant']).any():
        raise ValueError('Duplicate trial/metric/variant evaluation')
    for column in ('available', 'support_intersects_injected_gap'):
        if not f[column].isin([True, False]).all():
            raise ValueError(f'Invalid boolean {column}')
        f[column] = f[column].astype(bool)
    for condition in ('reference', 'simulated'):
        col = condition+'_meets_five_episode_minimum'
        if not r[col].isin([True, False]).all():
            raise ValueError(f'Invalid boolean {col}')
        r[col] = r[col].astype(bool)
        if not r[col].equals((r[condition+'_episodes'] >= 5).astype(bool)):
            raise ValueError(f'Inconsistent {condition} episode-minimum flag')
    meta = trials[['trial_id', 'placement_arm', 'video_path', 'pair']]
    return trials, r.merge(meta, on='trial_id', validate='many_to_one'), f.merge(
        meta, on='trial_id', validate='many_to_one'), episodes


def expected_cells(trials, *, by_cohort=False, fixed=False):
    cohorts = list(trials.cohort.drop_duplicates()) if by_cohort else [None]
    for cohort in cohorts:
        for metric, variants in VARIANTS.items():
            if fixed and metric in RATIOS:
                continue
            for variant in variants:
                yield ({'cohort': cohort} if by_cohort else {}) | {'metric': metric, 'variant': variant}


def cell_trials(trials, info, arm):
    name = 'episode_targeted:'+info['metric'] if arm == 'targeted' else 'spatial_matched'
    keep = trials.placement_arm.eq(name)
    if 'cohort' in info:
        keep &= trials.cohort.eq(info['cohort'])
    return trials[keep]


def select_cell(data, info):
    keep = pd.Series(True, index=data.index)
    for name, value in info.items():
        keep &= data[name].eq(value)
    return data[keep]


def support_status(attempts, evaluations, *, exposed=None):
    if attempts.empty:
        return 'arm not requested'
    if not attempts.status.eq('matched').any():
        return 'no matched placements'
    if evaluations.empty:
        return 'no evaluation rows'
    if exposed is not None:
        return 'evaluated' if len(exposed) else 'no exposed reference measurements'
    if not ((evaluations.reference_episodes > 0) | (evaluations.simulated_episodes > 0)).any():
        return 'no complete episodes'
    return 'evaluated'


def trial_audit(trials, redetect, episodes):
    """One row per attempt/metric/variant, including unsupported placements.

    Ratio numerators come from binary episode outcomes, not rounded means.
    Validate outcome counts/sums against the recorded means before reporting.
    """
    evaluations = redetect.set_index(['trial_id', 'metric', 'variant'])
    grouped = {key: g for key, g in episodes.groupby(['trial_id', 'metric', 'variant', 'condition'])}
    rows = []
    for t in trials.to_dict('records'):
        for metric, variants in VARIANTS.items():
            for variant in variants:
                key = (t['trial_id'], metric, variant)
                row = {name: t[name] for name in ('trial_id', 'cohort', 'video_path', 'pair', 'placement_arm',
                        'duration_bin', 'duration_s', 'gap_frames', 'status',
                        'evaluation_start_frame', 'evaluation_stop_frame_exclusive')}
                row.update(metric=metric, variant=variant)
                fps = t['gap_frames']/t['duration_s']
                row['fps'] = fps
                row['evaluation_duration_s'] = (t['evaluation_stop_frame_exclusive']-t['evaluation_start_frame'])/fps
                if key not in evaluations.index:
                    row['evaluation_status'] = 'unsupported placement' if t['status'] != 'matched' else 'no evaluation rows'
                    for condition in ('reference', 'simulated'):
                        for suffix in ('episodes', 'numerator', 'value', 'meets_five_episode_minimum'):
                            row[condition+'_'+suffix] = np.nan
                    row.update(eligibility_transition='not evaluated', both_means_defined=False,
                               both_meet_minimum=False, displayed_change=np.nan)
                else:
                    r = evaluations.loc[key]
                    row['evaluation_status'] = 'evaluated' if r.reference_episodes or r.simulated_episodes else 'no complete episodes'
                    eligible = []
                    for condition in ('reference', 'simulated'):
                        n = int(r[condition+'_episodes'])
                        mean = r[condition+'_mean_unfiltered']
                        ep = grouped.get((*key, condition), pd.DataFrame(columns=EPISODE_COLUMNS))
                        if len(ep) != n:
                            raise ValueError(f'Episode count mismatch for {key}, {condition}')
                        if n and not np.isfinite(ep.value).all():
                            raise ValueError(f'Nonfinite episode value for {key}, {condition}')
                        if n and not np.isclose(ep.value.mean(), mean, atol=1e-10, rtol=1e-10):
                            raise ValueError(f'Episode mean mismatch for {key}, {condition}')
                        if not n and np.isfinite(mean):
                            raise ValueError(f'Defined mean without episodes for {key}, {condition}')
                        if metric in RATIOS and not ep.value.isin([0, 1]).all():
                            raise ValueError(f'Nonbinary ratio outcome for {key}')
                        row[condition+'_episodes'] = n
                        row[condition+'_numerator'] = int(ep.value.sum()) if metric in RATIOS else np.nan
                        row[condition+'_value'] = mean
                        row[condition+'_meets_five_episode_minimum'] = n >= 5
                        eligible.append(n >= 5)
                    row['eligibility_transition'] = {
                        (True, True): 'eligible in both', (True, False): 'lost eligibility',
                        (False, True): 'gained eligibility', (False, False): 'below minimum in both',
                    }[tuple(eligible)]
                    row['both_meet_minimum'] = all(eligible)
                    row['both_means_defined'] = bool(np.isfinite(r.reference_mean_unfiltered) and np.isfinite(r.simulated_mean_unfiltered))
                    row['displayed_change'] = r.mean_change*(100 if metric in RATIOS else 1)
                row['change_units'] = 'percentage points' if metric in RATIOS else 'native metric units'
                # These identities allow reference reuse to be detected across trials.
                for condition in ('reference', 'simulated'):
                    ep = grouped.get((*key, condition), pd.DataFrame(columns=EPISODE_COLUMNS))
                    row[condition+'_episode_intervals'] = '|'.join(
                        f'{int(a)}:{int(b)}' for a, b in zip(ep.start_frame, ep.stop_frame_exclusive))
                rows.append(row)
    return pd.DataFrame(rows)
