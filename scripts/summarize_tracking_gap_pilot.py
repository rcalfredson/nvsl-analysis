#!/usr/bin/env python3
"""Build an explanatory companion report from existing single-gap pilot CSVs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tracking_gap_report_tables import (
    load_pilot, trial_audit, expected_cells, select_cell, cell_trials, support_status,
)

COHORTS = {'ctrl': 'Control', 'pfnd_kir': 'PFNd>Kir', 'antennae_removed': 'Antennae removed'}
METRICS = {
    'tortuosity': ('Between-reward tortuosity', 'dimensionless'),
    'wall_contacts': ('Wall contacts per reward interval', 'contacts'),
    'wall_contact_length': ('Path length of wall-positive intervals', 'mm'),
    'turnback_ratio': ('Turnback ratio', 'percentage points'),
    'home_vector_alignment': ('Home-vector alignment', 'alignment units'),
    'agarose_avoidance': ('Agarose avoidance ratio', 'percentage points'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_table(trials, keys):
    rows = []
    for key, g in trials.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = int(g.status.eq('matched').sum())
        rows.append(dict(zip(keys, key)) | {
            'Attempts': len(g), 'Matched': n, 'Unmatched': len(g)-n,
            'Matched (%)': 100*n/len(g),
            'Flies with matches': len(g[g.status.eq('matched')][['video_path', 'pair']].drop_duplicates()),
        })
    return pd.DataFrame(rows)


def detection_table(data, trials, by_cohort=False, *, arm="targeted", eligible_only=False):
    rows = []
    for info in expected_cells(trials, by_cohort=by_cohort):
        g = select_cell(data, info)
        attempts = cell_trials(trials, info, arm)
        metric = info['metric']
        e = g[(g.reference_episodes > 0) | (g.simulated_episodes > 0)]
        both = e[np.isfinite(e.reference_mean_unfiltered) & np.isfinite(e.simulated_mean_unfiltered)]
        factor = 100 if metric in ('turnback_ratio', 'agarose_avoidance') else 1
        if eligible_only:
            both = both[both.reference_meets_five_episode_minimum & both.simulated_meets_five_episode_minimum]
        change = both.mean_change * factor
        rows.append(info | {
            'Status': support_status(attempts, g),
            'Mean comparison policy': 'minimum five in both' if eligible_only else 'all defined local means',
            'Placement attempts': len(attempts),
            'Successful placements': int(attempts.status.eq('matched').sum()),
            'Below minimum in both': int((~e.reference_meets_five_episode_minimum & ~e.simulated_meets_five_episode_minimum).sum()),
            'Lost eligibility': int((e.reference_meets_five_episode_minimum & ~e.simulated_meets_five_episode_minimum).sum()),
            'Gained eligibility': int((~e.reference_meets_five_episode_minimum & e.simulated_meets_five_episode_minimum).sum()),
            'Eligible in both': int((e.reference_meets_five_episode_minimum & e.simulated_meets_five_episode_minimum).sum()),
            'Matched trials': len(g), 'Trials with episodes': len(e),
            'Flies with episodes': len(e[['video_path', 'pair']].drop_duplicates()),
            'Reference episodes': int(e.reference_episodes.sum()),
            'Simulated episodes': int(e.simulated_episodes.sum()),
            'Trials with count change': int(e.episode_count_change.ne(0).sum()),
            'Reference with no overlap': int(e.reference_without_overlap.sum()),
            'Simulated with no overlap': int(e.simulated_without_overlap.sum()),
            'Reference with multiple overlaps': int(e.reference_overlapping_multiple.sum()),
            'Simulated with multiple overlaps': int(e.simulated_overlapping_multiple.sum()),
            'Trials with paired boundary shift': int(((e.paired_start_mean_absolute_shift_frames > 0) |
                                                      (e.paired_stop_mean_absolute_shift_frames > 0)).sum()),
            'Trials changing five-episode status': int(e.reference_meets_five_episode_minimum.ne(e.simulated_meets_five_episode_minimum).sum()),
            'Trials losing defined mean': int((np.isfinite(e.reference_mean_unfiltered) & ~np.isfinite(e.simulated_mean_unfiltered)).sum()),
            'Trials gaining defined mean': int((~np.isfinite(e.reference_mean_unfiltered) & np.isfinite(e.simulated_mean_unfiltered)).sum()),
            'Trials with both means': len(both),
            'Trials with mean change': int((change.abs() > 1e-9).sum()),
            'Mean signed change': change.mean(),
            'Median absolute change': change.abs().median(),
            'Maximum absolute change': change.abs().max(),
            'Units': METRICS[metric][1],
        })
    return pd.DataFrame(rows)


def fixed_table(data, trials, by_cohort=False, *, arm="targeted"):
    rows = []
    for info in expected_cells(trials, by_cohort=by_cohort, fixed=True):
        g = select_cell(data, info)
        attempts = cell_trials(trials, info, arm)
        f = g[g.support_intersects_injected_gap]
        a = f[f.available & np.isfinite(f.error)]
        relative = a[a.reference_value.abs() > 1e-9]
        rows.append(info | {
            'Status': support_status(attempts, g, exposed=f),
            'Exposed trials': f.trial_id.nunique(),
            'Flies with exposure': len(f[['video_path', 'pair']].drop_duplicates()),
            'Exposed measurements': len(f), 'Available after loss': len(a),
            'Unavailable after loss': len(f)-len(a),
            'Available measurements changed': int((a.error.abs() > 1e-9).sum()),
            'Mean reference value (available)': a.reference_value.mean(),
            'Mean simulated value (available)': a.simulated_value.mean(),
            'Mean signed error': a.error.mean(),
            'Median absolute error': a.error.abs().median(),
            'Maximum absolute error': a.error.abs().max(),
            'Nonzero-reference measurements': len(relative),
            'Median absolute relative error (%)': (100*relative.error.abs()/relative.reference_value.abs()).median(),
            'Units': METRICS[info['metric']][1],
        })
    return pd.DataFrame(rows)


def denominator_table(audit):
    """Descriptive changes by original denominator; never pool placement arms."""
    data = audit[audit.evaluation_status.isin(['evaluated', 'no complete episodes'])].copy()
    data['Reference episode count'] = pd.cut(data.reference_episodes,
        [-1, 0, 4, 9, 19, np.inf], labels=['0', '1–4', '5–9', '10–19', '20+'])
    keys = ['placement_arm', 'cohort', 'metric', 'variant', 'Reference episode count']
    columns = keys + ['Evaluated trials', 'Trials with both means', 'Lost eligibility',
                      'Mean signed change', 'Maximum absolute change', 'Units']
    rows = []
    for key, g in data.groupby(keys, observed=True, sort=False):
        change = g.loc[g.both_means_defined, 'displayed_change']
        rows.append(dict(zip(keys, key)) | {
            'Evaluated trials': len(g), 'Trials with both means': len(change),
            'Lost eligibility': int(g.eligibility_transition.eq('lost eligibility').sum()),
            'Mean signed change': change.mean(), 'Maximum absolute change': change.abs().max(),
            'Units': METRICS[key[2]][1],
        })
    return pd.DataFrame(rows, columns=columns)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir', type=Path, required=True)
    p.add_argument('--existing-report-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error('Output must be a new or empty directory')
    paths = [args.pilot_dir/n for n in ('trials.csv', 'redetection.csv', 'fixed_boundaries.csv', 'episodes.csv')]
    paths.append(args.existing_report_dir/'real_gap_summary.csv')
    trials, redetect, fixed, episodes = load_pilot(args.pilot_dir)
    profile = pd.read_csv(paths[-1])
    audit = trial_audit(trials, redetect, episodes)
    targeted = redetect[redetect.placement_arm.eq('episode_targeted:'+redetect.metric)]
    targeted_fixed = fixed[fixed.placement_arm.eq('episode_targeted:'+fixed.metric)]
    spatial = redetect[redetect.placement_arm.eq('spatial_matched')]
    spatial_fixed = fixed[fixed.placement_arm.eq('spatial_matched')]
    tables = {
        'trial_details': audit,
        'changes_by_reference_denominator': denominator_table(audit),
        'targeted_detection_minimum_five': detection_table(targeted, trials, eligible_only=True),
        'targeted_detection_minimum_five_by_cohort': detection_table(targeted, trials, True, eligible_only=True),
        'spatial_detection_minimum_five': detection_table(spatial, trials, arm='spatial', eligible_only=True),
        'coverage_by_arm': coverage_table(trials, ['placement_arm']),
        'coverage_by_duration': coverage_table(trials, ['placement_arm', 'duration_bin']),
        'coverage_by_cohort': coverage_table(trials, ['cohort', 'placement_arm']),
        'targeted_detection': detection_table(targeted, trials),
        'targeted_fixed': fixed_table(targeted_fixed, trials),
        'targeted_detection_by_cohort': detection_table(targeted, trials, True),
        'targeted_fixed_by_cohort': fixed_table(targeted_fixed, trials, True),
        'spatial_detection': detection_table(spatial, trials, arm="spatial"),
        'spatial_fixed': fixed_table(spatial_fixed, trials, arm="spatial"),
    }
    highlights = []
    for metric in METRICS:
        variant = '13/15mm' if metric in ('turnback_ratio', 'home_vector_alignment') else ('physical_1mm' if metric == 'agarose_avoidance' else 'full')
        d = tables['targeted_detection'].query('metric == @metric and variant == @variant').iloc[0]
        if d['Status'] != 'evaluated':
            sample = f"{int(d['Successful placements'])} matched placements"
            result = d['Status'] + '; effect undefined'
        elif metric in ('turnback_ratio', 'agarose_avoidance'):
            sample = f"{int(d['Trials with both means'])} trials with both local means"
            result = (f"{int(d['Trials with mean change'])} changed; median absolute change "
                      f"{d['Median absolute change']:.3g}, maximum {d['Maximum absolute change']:.3g} percentage points "
                      f"(exploratory; {int(d['Eligible in both'])} trials meet the minimum in both)")
        else:
            f = tables['targeted_fixed'].query('metric == @metric and variant == @variant').iloc[0]
            sample = f"{int(f['Exposed measurements'])} exposed reference measurements; {int(f['Unavailable after loss'])} unavailable after loss"
            if f['Status'] != 'evaluated' or not f['Available after loss']:
                result = f['Status'] + '; no available exposed measurement; effect undefined'
            else:
                result = (f"{int(f['Available measurements changed'])} changed; median absolute error "
                          f"{f['Median absolute error']:.3g}, maximum {f['Maximum absolute error']:.3g} {f['Units']}")
        highlights.append({'Metric': METRICS[metric][0], 'Variant': variant, 'Comparison size': sample,
                           'Observed response': result,
                           'Trials changing episode count': (f"{int(d['Trials with count change'])} / {int(d['Trials with episodes'])}" if d['Trials with episodes'] else 'undefined (no evaluated episodes)')})
    tables['at_a_glance'] = pd.DataFrame(highlights)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out/(name+'.csv'), index=False)

    def render(name, columns=None):
        table = tables[name].copy()
        if name == 'trial_details':
            table = table[table.placement_arm.eq('episode_targeted:'+table.metric) & table.evaluation_status.eq('evaluated')]
        if columns:
            table = table[columns]
        table = table.rename(columns={'metric': 'Metric', 'variant': 'Variant', 'cohort': 'Cohort',
            'placement_arm': 'Placement arm', 'duration_bin': 'Gap duration',
            'trial_id': 'Trial', 'duration_s': 'Gap duration (s)',
            'evaluation_duration_s': 'Evaluated stretch (s)',
            'reference_numerator': 'Reference successes (ratios)',
            'simulated_numerator': 'Simulated successes (ratios)',
            'reference_episodes': 'Reference episodes', 'simulated_episodes': 'Simulated episodes',
            'reference_value': 'Reference mean / ratio', 'simulated_value': 'Simulated mean / ratio',
            'displayed_change': 'Change', 'change_units': 'Change units',
            'eligibility_transition': 'Minimum-count status'})
        if 'Metric' in table:
            table['Metric'] = table.Metric.map(lambda m: METRICS[m][0] if m in METRICS else m)
        if 'Cohort' in table:
            table['Cohort'] = table.Cohort.map(COHORTS)
        return '<div class="scroll">'+table.to_html(index=False, na_rep='undefined', float_format=lambda x: f'{x:,.4g}')+'</div>'

    def link(path):
        import os
        return html.escape(os.path.relpath(path, out), quote=True)

    nflies = len(trials[['video_path', 'pair']].drop_duplicates())
    ndonors = len(trials[['video_path', 'pair', 'donor_gap_id']].drop_duplicates())
    nmatched = int(trials.status.eq('matched').sum())
    detection_columns = ['metric', 'variant', 'Status', 'Matched trials', 'Trials with episodes', 'Flies with episodes',
                         'Reference episodes', 'Simulated episodes', 'Trials with count change',
                         'Reference with no overlap', 'Simulated with no overlap', 'Trials with paired boundary shift']
    value_columns = ['metric', 'variant', 'Status', 'Trials with both means', 'Trials with mean change',
                     'Mean signed change', 'Median absolute change', 'Maximum absolute change', 'Units']
    fixed_columns = ['metric', 'variant', 'Status', 'Exposed trials', 'Flies with exposure', 'Exposed measurements',
                     'Available after loss', 'Unavailable after loss', 'Available measurements changed',
                     'Mean signed error', 'Median absolute error', 'Maximum absolute error', 'Units']
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tracking-gap pilot: explained results</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#203047;max-width:1500px;margin:32px auto;padding:0 24px}
h1,h2,h3{line-height:1.25}h2{margin-top:40px}p{max-width:1050px}.note{background:#eef4fa;border-left:4px solid #3672aa;padding:14px 20px}
.scroll{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #ccd5df;padding:9px;text-align:right}th{background:#edf2f7}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f7f9fc}summary{cursor:pointer;font-weight:600;margin-top:22px}a{color:#185c99}</style>
<h1>What does an artificial tracking gap change?</h1>
<p>This companion report summarizes the existing pilot. It introduces no additional simulations and leaves the original report unchanged.</p>''',
        f'<p class="note"><strong>Scope:</strong> {nflies} flies, {ndonors} distinct real donor gaps, {len(trials)} donor/placement-arm attempts, and {nmatched} successful placements. These are conditional single-gap experiments, not estimates of manuscript cohort bias.</p>',
        '<h2>How the comparison works</h2><p>Sample real gap durations within duration bins; find a fully tracked stretch in the same fly with comparable observed endpoint geometry; hide its x/y, body dimensions and orientation; then apply the production interpolation and metric code. The reference is the original tracked result, not manually established biological ground truth. Actual reward times stay fixed.</p>',
        '<p><strong>Spatial-only placements</strong> do not require an episode at the injection site. <strong>Episode-targeted placements</strong> additionally intersect reference episode support for a particular metric. Duration bins are deliberately sampled, not weighted to their frequency in real recordings. Targeted results describe susceptibility when exposed; they are not the expected error from a randomly occurring real gap.</p>',
        '<h2>Results at a glance: targeted placements</h2><p>This table brings together two complementary comparisons: fixed-boundary measurements for numeric episode metrics, and redetected local means for ratios. They have different denominators and should not be compared as a ranking of robustness. The turnback/alignment rows show 13/15 mm; all radii appear below.</p>',
        render('at_a_glance'),
        '<p>The wall-contact count response can change without changing the number of reward intervals. In contrast, loss of wall-positive qualification can remove intervals from the path-length metric. The detailed tables separate those effects. A median of zero can coexist with a large maximum; inspect both and their sample sizes.</p>',
        '<h2>1. How much of the experiment has support?</h2><p>An attempt is one donor tested under one placement arm. The same donors are reused across arms; an unmatched attempt is an unsupported context, not a zero-effect test. “Flies with matches” counts distinct flies within that row.</p>',
        render('coverage_by_arm'),
        '<details><summary>Coverage by gap duration and by cohort</summary>'+render('coverage_by_duration')+render('coverage_by_cohort')+'</details>',
        f'<p>Matched placements longer than 10 seconds: {int((trials.duration_bin.eq(">10s") & trials.status.eq("matched")).sum())}. There are {int(profile.primary_donor_gaps.sum()):,} eligible donors in the full profile; this pilot uses only a small subset. Unmatched duration/context combinations remain untested.</p>',
        '<h2>2. Does rerunning detection change which episodes contribute?</h2>',
        '<p>The following tables use only the corresponding metric’s targeted arm. For each trial, both conditions use the same originally fully observed parent stretch and retain only episodes with complete support inside it. Radius variants remain separate. Turnback/alignment targeting uses the union of radius supports, so a matched trial can have no evaluable episode at a smaller radius.</p>',
        render('targeted_detection', detection_columns),
        '<p>Episode totals count episode appearances across trials; repeated trials can reuse the same original episode. A count change can reflect detection, qualification or measurement availability. “No overlap” means no temporal overlap with any episode in the other condition; it is not proof of a biological miss or false event. Boundary shifts refer only to unique one-to-one temporal-overlap pairs. Equal counts do not guarantee equal boundaries or values.</p>',
        '<details><summary>Split/merge diagnostics, availability and five-episode status</summary>'+render('targeted_detection', [
            'metric', 'variant', 'Trials with episodes', 'Reference with multiple overlaps', 'Simulated with multiple overlaps',
            'Trials losing defined mean', 'Trials gaining defined mean', 'Trials changing five-episode status'])+
        '<p>Multiple overlaps flag split/merge associations rather than proven event identities. Five-episode status is local to the evaluated stretch; it is not the manuscript’s recording-level inclusion decision.</p></details>',
        '<h2>3. How much do measurements change within the original episodes?</h2>',
        '<p>Hold original episode boundaries fixed and include only reference measurements whose support intersects the injected gap. Length comparisons retain the original wall-positive qualification. Alignment can become unavailable under its existing heading-quality rule. Turnback and avoidance ratios are assessed after redetection in the next table.</p>',
        render('targeted_fixed', fixed_columns),
        '<p>Error = simulated − reference. Signed errors show direction but can cancel; absolute errors show magnitude. Numeric errors use only measurements still available after loss, with unavailable measurements counted separately. “Changed” uses an absolute tolerance of 10<sup>−9</sup> in the displayed metric units. Undefined means no available observations, not zero error.</p>',
        '<details><summary>Reference values and relative measurement errors</summary>'+render('targeted_fixed', [
            'metric', 'variant', 'Available after loss', 'Mean reference value (available)', 'Mean simulated value (available)',
            'Nonzero-reference measurements', 'Median absolute relative error (%)', 'Units'])+
        '<p>Relative error is 100 × |simulated − reference| / |reference|, excluding reference magnitudes ≤10<sup>−9</sup>. It can be large near zero, particularly for alignment.</p></details>',
        '<h2>4. What happens to the local metric mean after redetection?</h2>',
        '<p>These statistics summarize changes in each trial’s mean over the evaluated parent stretch after rerunning the full episode calculation. They include changed membership as well as changed values. Both means must be defined; availability changes are reported above. Whole-stretch means can dilute effects confined to the injected gap.</p>',
        '<h3>All defined local means — exploratory</h3>',
        render('targeted_detection', value_columns),
        '<h3>At least five episodes in both conditions</h3>',
        render('targeted_detection_minimum_five', value_columns),
        '<p>This second table filters only the numeric mean-change comparison. The status/count columns still describe the full placement and evaluation pool. Passing the minimum does not guarantee a precise estimate. Cases that lose eligibility remain counted below rather than disappearing silently.</p>',
        render('targeted_detection', ['metric', 'variant', 'Status', 'Below minimum in both', 'Eligible in both', 'Lost eligibility', 'Gained eligibility']),
        '<h3>Trial-level denominators and eligibility</h3><p>Ratio values below are fractions (0–1), whereas their changes are percentage points. Numerators count successful binary outcomes. Unsupported placements have undefined counts and values; zero episodes in an evaluated stretch is an observed zero denominator, with an undefined mean.</p>',
        render('trial_details', ['trial_id', 'cohort', 'metric', 'variant', 'duration_s', 'evaluation_duration_s', 'reference_numerator', 'reference_episodes', 'reference_value', 'simulated_numerator', 'simulated_episodes', 'simulated_value', 'displayed_change', 'change_units', 'eligibility_transition']),
        '<p>Only evaluated corresponding targeted-arm rows are shown above. The trial_details CSV includes every attempt and metric/radius, including unsupported placements and other arms; numerators do not apply to non-ratio metrics. Saved episode intervals and video/fly identity allow reference reuse to be inspected.</p>',
        '<details><summary>Changes stratified by reference episode count</summary>'+render('changes_by_reference_denominator')+
        '<p>These bins are descriptive, not additional eligibility rules. Conditions can change denominator bin after masking; the table groups by the original count. Arms and cohorts remain separate. No rows means no evaluated denominators, not zero impact.</p></details>',
        '<p>Ratio changes are in percentage points: +1 means a ratio increased by 0.01. Other changes use the metric’s native units. These are diagnostic local means without applying the five-episode minimum, not manuscript per-fly estimates. Means of trial changes weight trials equally; they do not weight flies equally.</p>',
        '<h2>5. Check the spatial-only arm and cohort detail</h2><p>Keep these results separate from the targeted arms. They answer a different placement question and have different exposure counts. Cohort splits are descriptive and particularly sparse.</p>',
        '<details><summary>Spatial-only detection and local mean changes</summary>'+render('spatial_detection', detection_columns)+render('spatial_detection', value_columns)+'</details>',
        '<details><summary>Spatial-only local means meeting the minimum in both conditions</summary>'+render('spatial_detection_minimum_five', value_columns)+'</details>',
        '<details><summary>Spatial-only fixed-boundary measurements</summary>'+render('spatial_fixed', fixed_columns)+'</details>',
        '<details><summary>Targeted detection and local mean changes by cohort</summary>'+render('targeted_detection_by_cohort')+'</details>',
        '<details><summary>Targeted local mean changes by cohort: minimum five in both</summary>'+render('targeted_detection_minimum_five_by_cohort')+'</details>',
        '<details><summary>Targeted fixed-boundary measurements by cohort</summary>'+render('targeted_fixed_by_cohort')+'</details>',
        '<h2>What these results establish</h2><p>The pilot measures the response to one realistic artificial loss interval in a supported, originally observed context. The masking combines coordinate interpolation, unavailable body dimensions and production missing-data rules; detector effects cannot all be attributed to the interpolated x/y chord. Sparse or zero effects here do not establish robustness in unsupported contexts. Repeated trials and episodes are not independent replicates, so these tables do not present confidence intervals or significance tests.</p>',
        '<p>Frequent clustered loss, gaps longer than 10 seconds, reward-feedback effects, and cumulative changes in whole-recording metrics remain outside this pilot. Matching observed endpoints does not establish that the hidden behavior during real loss resembles the reference path.</p>',
        '<h2>Source reports and downloadable summary tables</h2>',
        f'<p><a href="{link(args.existing_report_dir/"report.html")}">Original report and real-gap characteristics</a> · <a href="{link(args.pilot_dir/"report.html")}">Pilot methods report</a></p>',
        '<ul>'+''.join(f'<li><a href="{name}.csv">{html.escape(name.replace("_", " "))}</a></li>' for name in tables)+'</ul></html>',
    ]
    (out/'report.html').write_text('\n'.join(parts))
    (out/'provenance.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'inputs_sha256': {str(path): digest(path) if path.exists() else None for path in paths},
        'report_code_sha256': digest(__file__),
        'table_code_sha256': digest(ROOT/'scripts/tracking_gap_report_tables.py'), 'numeric_change_tolerance': 1e-9,
    }, indent=2)+'\n')
    print(out/'report.html')


if __name__ == '__main__':
    main()
