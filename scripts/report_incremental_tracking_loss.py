#!/usr/bin/env python3
"""Make compact, readable tables from a completed additional-loss experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_episode_tracking_loss import Sources, COHORTS
from scripts.simulate_tracking_gaps import empty_folder

METRICS = dict(tortuosity='Tortuosity', wall_contacts='Wall contacts / interval',
               wall_contact_length='Wall-positive path length (mm)')


def number(value, signed=False):
    if pd.isna(value):
        return 'undefined'
    return f'{value:+.5g}' if signed else f'{value:.5g}'


def fraction(value):
    return f'{100*value:.2f}%' if pd.notna(value) else 'undefined'


def arm_name(arm):
    return 'Empirical spatial' if arm == 'empirical_spatial' else 'Episode targeted'


def contrast_name(value):
    return ' − '.join(COHORTS.get(v.strip(), v.strip()) for v in value.split(' - '))


def readout(run, out):
    out = empty_folder(out)
    sources = Sources(ROOT)
    sources.track(Path(__file__).resolve())
    manifest = json.loads(sources.track(run/'provenance.json').read_text())
    tables = {}
    for name in ('coverage', 'trials', 'cohort_summary', 'contrasts'):
        path = sources.track(run/f'{name}.csv')
        try:
            tables[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            if name != 'trials':
                raise
            tables[name] = pd.DataFrame(columns=['arm', 'metric', 'cohort', 'video_path', 'pair', 'repeat'])
    c, t, contrasts = tables['cohort_summary'], tables['trials'], tables['contrasts']
    coverage_rows, dose_rows, effect_rows, composition_rows = [], [], [], []
    for r in c.to_dict('records'):
        coverage_rows.append({'Arm': arm_name(r['arm']), 'Metric': METRICS[r['metric']],
            'Cohort': COHORTS[r['cohort']], 'Original eligible flies': r['manuscript_flies'],
            'Supported original eligible flies': r['supported_baseline_eligible_flies'],
            'All supported gate-passing flies': r['supported_flies'],
            'Supported videos': r['supported_videos'],
            'Supported donor gaps / all donors': f"{r['supported_donors']} / {r['donor_gaps']}",
            'Donor-gap coverage': fraction(r['donor_gap_coverage']),
            'Donor lost-time coverage': fraction(r['donor_time_coverage']),
            'Full baseline mean': number(r['manuscript_mean']),
            'Supported baseline mean': number(r['supported_baseline_mean'])})
        sub = t[t.arm.eq(r['arm']) & t.metric.eq(r['metric']) & t.cohort.eq(r['cohort'])]
        if not len(sub):
            continue
        dose_rows.append({'Arm': arm_name(r['arm']), 'Metric': METRICS[r['metric']],
            'Cohort': COHORTS[r['cohort']], 'Metric comparisons': len(sub),
            'Distinct artificial intervals': len(sub[['video_path', 'pair', 'placement_start', 'placement_stop_exclusive']].drop_duplicates()),
            'Median added gap (s)': number(sub.added_seconds.median()),
            'Maximum added gap (s)': number(sub.added_seconds.max()),
            'Median added window loss': fraction(sub.added_window_loss_fraction.median()),
            'Maximum added window loss': fraction(sub.added_window_loss_fraction.max()),
            'Flies losing eligibility in any repeat': r['flies_losing_eligibility'],
            'Flies gaining eligibility in any repeat': r['flies_gaining_eligibility']})
    for r in contrasts.to_dict('records'):
        identity = {'Arm': arm_name(r['arm']), 'Metric': METRICS[r['metric']],
                    'Contrast': contrast_name(r['contrast'])}
        direction = 'undefined' if pd.isna(r['direction_preserved']) else 'yes' if r['direction_preserved'] else 'no'
        effect_rows.append(dict(identity, **{'Supported baseline contrast': number(r['before_contrast']),
            'After added loss': number(r['after_contrast']), 'Change': number(r['total_change'], True),
            '95% video-bootstrap interval for change': f"[{number(r['total_change_ci_low'])}, {number(r['total_change_ci_high'])}]",
            'Change / |baseline contrast|': fraction(r['relative_total_change']),
            'Direction preserved': direction,
            'Range of change across repeats': f"[{number(r['repeat_change_min'])}, {number(r['repeat_change_max'])}]"}))
        composition_rows.append(dict(identity, **{'Full manuscript contrast': number(r['manuscript_contrast']),
            'Support-selection shift': number(r['support_selection_shift'], True),
            'Common-fly metric change': number(r['common_fly_change'], True),
            'Eligibility-loss contribution': number(r['eligibility_loss_shift'], True),
            'Eligibility-gain contribution': number(r['eligibility_gain_shift'], True),
            'Total composition change': number(r['composition_change'], True),
            'Total added-loss change': number(r['total_change'], True)}))
    display = dict(coverage=pd.DataFrame(coverage_rows), dose=pd.DataFrame(dose_rows),
                   contrasts=pd.DataFrame(effect_rows), composition=pd.DataFrame(composition_rows))
    for name, frame in display.items():
        frame.to_csv(out/f'{name}.csv', index=False)
    repeats = manifest['settings']['repeats']
    masks = t[['video_path', 'pair', 'arm', 'repeat']].drop_duplicates()
    doc = f'''<!doctype html><meta charset="utf-8"><title>Additional-loss sensitivity readout</title>
<style>body{{font:15px/1.6 system-ui;margin:32px;max-width:1650px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#edf2f5}}section{{overflow-x:auto;margin:18px 0}}p{{max-width:1100px}}.scope{{background:#fff3d9;padding:14px}}</style>
<h1>Additional-loss sensitivity: first expanded batch</h1>
<p class="scope"><b>Scope:</b> one additional gap per fly per repeat; complete T2 SB2–5 measurements; existing real gaps remain in both conditions. This measures incremental susceptibility of the current calculation. It does not measure the unknown error already caused by actual lost tracking, or the effects of reproducing the entire original loss burden.</p>
<p>{manifest['subjects']} source-gated flies; {repeats} repeats per supported arm; {len(masks)} artificial-gap trials and {len(t)} per-metric comparisons. The three pooled between-reward metrics are included. Turnback/alignment and agarose chamber comparisons are not included in this batch. Baseline values/counts and sampled placement support are validated against the frozen sources.</p>
<p><b>Empirical spatial</b> samples real donor gaps uniformly within each fly's supported pool. <b>Episode targeted</b> cycles supported duration bins and requires exposure of complete observed episode support for the named metric. The latter is a separate stress test, not an estimate weighted by real gap frequencies. Unsupported contexts are not assigned zero effects. Each repeat starts with the original tracking.</p>'''
    defined = contrasts[np.isfinite(contrasts.before_contrast) & np.isfinite(contrasts.after_contrast)]
    preserved = int(defined.direction_preserved.sum())
    doc += f'<p>Of {len(defined)} defined arm/metric/contrast averages, {preserved} retain the original contrast direction. This statement concerns this particular added-loss dose and supported population; it does not establish scientific equivalence.</p>'
    empirical = contrasts[contrasts.arm.eq('empirical_spatial')]
    targeted_length = contrasts[contrasts.arm.eq('episode_targeted:wall_contact_length')]
    if len(empirical) and empirical.relative_total_change.notna().any():
        doc += f'<p>In the empirical spatial arm, the largest absolute average relative contrast change is {fraction(empirical.relative_total_change.abs().max())}.</p>'
    if len(targeted_length) and targeted_length.relative_total_change.notna().any():
        doc += f'<p>In the targeted wall-positive path-length arm, the largest absolute average relative contrast change is {fraction(targeted_length.relative_total_change.abs().max())}. Check the composition table and eligibility cases below: preserving direction can coexist with changes in magnitude and contributing flies.</p>'
    descriptions = [
        ('coverage', '1. Who and which real gaps could be tested?',
         'Original eligible means the full manuscript pool for that metric. Supported original eligible counts the subset with a usable same-fly placement. All supported gate-passing flies additionally includes initially undefined measurements that could gain eligibility. Gap and lost-time coverage refer to bracketed within-window donors, not all original loss. Different baseline means reveal selection before adding a gap.'),
        ('dose', '2. How much loss was actually added, and did eligibility change?',
         'A per-metric comparison is one fly/repeat/metric, not a new independent fly. Repeats can select the same artificial interval; distinct-interval counts ignore donor identity when duration and placement are identical. The dose is the added duration and its fraction of the full window. A fly is counted once if it loses or gains eligibility in any repeat. Eligibility uses at least five episodes across the complete production window; there is no local five-episode requirement.'),
        ('contrasts', '3. Did cohort contrasts change in direction or magnitude?',
         'Contrasts are positive cohort minus negative cohort, in the metric’s native units. Changes are after minus before. Cohort means weight eligible flies equally within each repeat; contrast estimates average repeats. The interval resamples videos within cohort, keeping each video’s flies and repeats together. Repeat ranges show separate mask variability. Relative change is signed change divided by absolute baseline contrast and is unstable near zero.'),
        ('composition', '4. Was the change numerical, or did the contributing flies change?',
         'Support-selection shift happens before masking and compares the supported baseline with the full manuscript contrast. Common-fly change uses exactly the same flies eligible before and after. Eligibility-loss and eligibility-gain contributions sum to composition change. Common-fly change plus composition change equals total added-loss change. The support-selection shift is not included in that total.'),
    ]
    for name, title, explanation in descriptions:
        doc += f'<h2>{title}</h2><p>{explanation}</p>'
        doc += '<section>'+display[name].to_html(index=False, escape=True, na_rep='undefined')+'</section>'
    lost = t[t.before_eligible & ~t.after_eligible] if len(t) else t
    cases = []
    if len(lost):
        for (arm, metric, cohort, video, pair), g in lost.groupby(['arm', 'metric', 'cohort', 'video_path', 'pair']):
            total_repeats = len(t[t.arm.eq(arm) & t.metric.eq(metric) & t.video_path.eq(video) & t.pair.eq(pair)])
            cases.append({'Arm': arm_name(arm), 'Metric': METRICS[metric], 'Cohort': COHORTS[cohort],
                'Video': Path(video).name, 'Pair': pair, 'Affected repeats': f'{len(g)} / {total_repeats}',
                'Before episodes': g.before_n_episodes.iloc[0],
                'After episodes in affected repeats': ', '.join(str(n) for n in sorted(set(g.after_n_episodes))),
                'Before value': number(g.before_value.iloc[0]),
                'Added duration range (s)': f'[{number(g.added_seconds.min())}, {number(g.added_seconds.max())}]'})
    cases = pd.DataFrame(cases)
    cases.to_csv(out/'eligibility_cases.csv', index=False)
    doc += '<h2>5. Cases losing eligibility</h2><p>These are measurement-availability changes. They do not demonstrate a known error in the original hidden path. Repeat counts include reused placements.</p>'
    doc += cases.to_html(index=False, escape=True) if len(cases) else '<p>No initially eligible measurements lost eligibility.</p>'
    doc += '<p>Four repeats provide a bounded initial check, not a rare-event or precision guarantee. Long gaps and high-loss contexts can be poorly supported even when gap-count coverage is high; inspect lost-time coverage. This run does not test clustered multi-gap loss. No scientific equivalence tolerance has been selected.</p><h2>Source report and audit data</h2><ul>'
    doc = doc.replace('Four repeats provide', f'{repeats} repeats provide')
    for name in ('report.html', 'coverage.csv', 'trials.csv', 'cohort_summary.csv', 'contrasts.csv', 'provenance.json'):
        link = html.escape(os.path.relpath(run/name, out), quote=True)
        doc += f'<li><a href="{link}">{html.escape(name)}</a></li>'
    (out/'report.html').write_text(doc+'</ul>')
    sources.verify_unchanged()
    (out/'provenance.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(run), sources=sources.manifest), indent=2)+'\n')
    print(f'Wrote {out}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    readout(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
