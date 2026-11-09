#!/usr/bin/env python3
"""Readable census overview, including distinct matched reference episodes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path

import pandas as pd


def matched_reference_counts(donors):
    """Count reference episodes once per cell, regardless of donor/start reuse."""
    keys = ['genotype', 'chamber', 'metric', 'variant', 'period', 'baseline_eligible']
    identities = defaultdict(set)
    for row in donors.itertuples(index=False):
        if row.exposed_candidates:
            key = tuple(getattr(row, name) for name in keys)
            for index in json.loads(row.reference_episode_indices):
                identities[key].add((row.video_path, row.pair, index))
    return pd.DataFrame([dict(zip(keys, key)) | {'matched_reference_episodes': len(ids)}
                         for key, ids in identities.items()], columns=keys+['matched_reference_episodes'])


def membership_overlaps(subjects):
    rows = []
    for (video, pair), g in subjects.groupby(['video_path', 'pair']):
        if g.dataset.nunique() > 1:
            rows.append(dict(video_path=video, pair=pair,
                dataset_memberships=' | '.join(sorted(g.dataset.unique()))))
    return pd.DataFrame(rows, columns=['video_path', 'pair', 'dataset_memberships'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error('Output must be a new or empty directory')
    names = ['datasets', 'subjects', 'windows', 'donor_support', 'coverage', 'auc_coverage']
    inputs = {name: args.audit_dir/(name+'.csv') for name in names}
    inputs['provenance'] = args.audit_dir/'provenance.json'
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.values()}
    data = {name: pd.read_csv(inputs[name]) for name in names}
    out.mkdir(parents=True, exist_ok=True)
    keys = ['genotype', 'chamber', 'metric', 'variant', 'period', 'baseline_eligible']
    refs = matched_reference_counts(data['donor_support'])
    coverage = data['coverage'].merge(refs, on=keys, how='left', validate='one_to_one')
    coverage['matched_reference_episodes'] = coverage.matched_reference_episodes.fillna(0).astype(int)
    if not (coverage.matched_reference_episodes <= coverage.observed_reference_episodes).all():
        raise ValueError('Matched episode count exceeds observed reference pool')
    coverage.to_csv(out/'matched_reference_coverage.csv', index=False)
    subjects = data['subjects'][data['subjects'].admitted]
    overlaps = membership_overlaps(subjects)
    overlaps.to_csv(out/'overlapping_source_memberships.csv', index=False)
    loss = subjects.groupby(['genotype', 'chamber'], sort=False).agg(
        admitted_flies=('pair', 'size'), videos=('video_path', 'nunique'),
        median_recording_loss_pct=('recording_loss_pct', 'median'),
        p95_recording_loss_pct=('recording_loss_pct', lambda s: s.quantile(.95))).reset_index()
    loss.to_csv(out/'source_population.csv', index=False)
    eligible = coverage[coverage.baseline_eligible]
    pooled = eligible[eligible.metric.ne('agarose_avoidance')].copy()
    pooled.to_csv(out/'pooled_metric_coverage.csv', index=False)
    donor = data['donor_support']
    windows = data['windows']
    ag = donor[donor.metric.eq('agarose_avoidance') & donor.baseline_eligible]
    rows = []
    for (genotype, chamber), w in windows[windows.metric.eq('agarose_avoidance') & windows.baseline_eligible].groupby(['genotype', 'chamber'], sort=False):
        g = ag[ag.genotype.eq(genotype) & ag.chamber.eq(chamber)]
        total = g.duration_s.sum()
        supported, minimum = g[g.exposed_candidates > 0], g[g.minimum_five_candidates > 0]
        rows.append(dict(genotype=genotype, chamber=chamber,
            eligible_subjects=len(w[['video_path', 'pair']].drop_duplicates()), eligible_fly_cells=len(w),
            donor_gaps=len(g), donor_lost_seconds=total,
            excluded_lost_seconds=w.excluded_lost_seconds.sum(),
            episode_gap_fraction=len(supported)/len(g) if len(g) else float('nan'),
            episode_lost_time_fraction=supported.duration_s.sum()/total if total else float('nan'),
            supported_subjects=len(supported[['video_path', 'pair']].drop_duplicates()),
            minimum_five_supported_subjects=len(minimum[['video_path', 'pair']].drop_duplicates())))
    agarose = pd.DataFrame(rows)
    agarose.to_csv(out/'agarose_local_coverage.csv', index=False)
    duration = donor[donor.baseline_eligible].groupby(['genotype', 'chamber', 'metric', 'variant', 'duration_bin'], sort=False).agg(
        donors=('duration_s', 'size'), donor_seconds=('duration_s', 'sum'),
        supported_donors=('exposed_candidates', lambda s: int((s > 0).sum()))).reset_index()
    duration.to_csv(out/'duration_coverage.csv', index=False)

    def table(frame):
        frame = frame.copy()
        for col in frame:
            if col.endswith('_fraction'):
                frame[col] *= 100
        frame = frame.rename(columns={col: col.replace('_fraction', '_percent').replace('_', ' ') for col in frame})
        return '<div class="scroll">'+frame.to_html(index=False, na_rep='undefined', float_format=lambda n: f'{n:,.3g}')+'</div>'
    n = int(data['datasets'].admitted_subjects.sum())
    unique_flies = len(subjects[['video_path', 'pair']].drop_duplicates())
    unique_videos = subjects.video_path.nunique()
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Tracking-gap audit: support summary</title>
<style>body{font:16px/1.6 system-ui;color:#243449;max-width:1500px;margin:35px auto;padding:20px}p{max-width:1100px}.scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd;padding:8px}th{background:#edf3f9}h2{margin-top:40px}</style>
<h1>Where can we test tracking-loss effects?</h1>''',
        f'<p>The census covers <strong>{n} admitted fly–dataset memberships, representing {unique_flies} distinct video/fly identities in {unique_videos} recordings</strong>, across the five genotype/manipulation labels and both chamber types used for agarose. The original pooled metrics retain their 196-subject source roster. This report estimates reference support, not errors in metric values or group contrasts.</p>',
        '<h2>Source memberships requiring clarification</h2><p>The following admitted video/fly identities occur in more than one manuscript dataset. They are retained as specified by the notebook and archives, not reassigned. Check the intended cohort labels against experiment records before interpreting affected genotype/manipulation comparisons. Reuse does not create new independent subjects; it is not, by itself, proof that a source list is wrong.</p>'+table(overlaps),
        '<h2>1. Source populations</h2><p>Recording loss follows production handling of an initial missing run. Medians and percentiles weight flies equally; excluded source pairs are retained in the detailed census.</p>', table(loss),
        '<h2>2. Pooled metrics: eligible flies and local reference support</h2><p>A supported donor has at least one spatially matched same-fly placement overlapping a complete observed episode for this exact metric/radius. Coverage percentages use eligible donor gaps or their missing time as denominators; excluded boundary loss is separate. Supported subjects have both a donor and a usable placement. Zero indicates no support under the current rules, not zero error.</p>',
        table(pooled[['genotype', 'metric', 'variant', 'subjects', 'donors', 'episode_gap_fraction',
            'episode_lost_time_fraction', 'episode_subjects', 'episode_videos', 'minimum_five_subjects',
            'observed_reference_episodes', 'matched_reference_episodes']]),
        '<p>Observed reference episodes are the complete observed pool; matched reference episodes count only those reached by at least one donor placement, deduplicated within each fly/metric/radius/window. Repeated placements can reuse those same episodes. Minimum-five subjects have at least one episode-overlapping match in a parent stretch containing five complete reference episodes; this remains a local diagnostic.</p>',
        '<h2>3. Agarose: local support across eligible Pre/T1/T2 cells</h2><p>This overview pools donor gaps and lost time across baseline-eligible bucket cells, not per-fly error estimates. Subject counts are deduplicated across cells; fly-cell counts are shown separately. It does not imply that a fly is supported at every time point. The detailed census retains cell-specific chamber-contrast coverage.</p>', table(agarose),
        '<h2>4. Agarose: continuous reference windows for the plotted AUC</h2><p>Production AUC availability is reproduced from per-fly source values, including Pre only in T1 and production handling of an all-NaN trailing bucket. A trusted-defined window is a strictly observed candidate reference with a defined AUC and complete episode support; it is not independently validated biological ground truth. Local matching cannot substitute for this whole-window availability.</p>', table(data['auc_coverage']),
        '<details><summary>Support by gap duration (eligible cells pooled within endpoint)</summary>'+table(duration)+'</details>',
        '<h2>Interpretation and next decisions</h2><p>Assess gap-count and lost-time coverage together: good coverage of many short gaps can coexist with poor coverage of the lost-time burden. Support in selected observed contexts does not establish that actual unobserved behavior is comparable. These counts do not measure split/merge errors, joint placement of clustered gaps, uncertainty of cohort contrasts, or scientifically acceptable error.</p>',
        '<p>The next simulation design should use independent-recording and radius/window coverage, with targeted retracking where long-gap or continuous-window support is missing. Preserve the same-fly matching rules unless a separately labeled reference-library analysis is explicitly designed.</p>',
        f'<p><a href="{html.escape(os.path.relpath(args.audit_dir/"report.html", out), quote=True)}">Detailed census, contrast coverage and source validation</a></p>',
        '<ul>'+''.join(f'<li><a href="{name}.csv">{name.replace("_", " ")}</a></li>' for name in ['source_population', 'pooled_metric_coverage', 'agarose_local_coverage', 'duration_coverage', 'matched_reference_coverage', 'overlapping_source_memberships'])+'</ul></html>',
    ]
    (out/'report.html').write_text('\n'.join(parts))
    (out/'provenance.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=hashes, report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print(out/'report.html')


if __name__ == '__main__':
    main()
