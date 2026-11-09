#!/usr/bin/env python3
"""Census of same-fly reference support for manuscript tracking-loss contrasts."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import glob
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_episode_tracking_loss import (
    Sources, load_context, make_trajectory, agarose_episodes, agarose_signature,
)
from scripts.simulate_tracking_gaps import raw_fields, geometry, track_code, empty_folder
from src.analysis.trajectory import Trajectory
from src.analysis.episode_tracking_loss import GapIndex
from src.analysis.tracking_gap_support import audit_window
from src.plotting.time_series_auc import signed_auc_values
from src.analysis.agarose_preinvestigation import load_pickle

DATASETS = [
    ('17b', 'ctrl', 'flat', 'FIGURE_4B_CTRL_KIR_VIDEO_GLOBS'),
    ('17b', 'ctrl', 'agarose', 'FIGURE_4S_AGAROSE_CTRL_KIR_VIDEO_GLOBS'),
    ('17c', 'antennae_removed', 'flat', 'FIGURE_4D_ANTENNAE_REMOVED_VIDEO_GLOBS'),
    ('17c', 'antennae_removed', 'agarose', 'FIGURE_4T_ANTENNAE_REMOVED_VIDEO_GLOBS'),
    ('17d', 'mbkc1_kir', 'flat', 'EXTENDED_DATA_FIGURE_14H_MBKC1_VIDEO_GLOBS'),
    ('17d', 'mbkc1_kir', 'agarose', 'EXTENDED_DATA_FIGURE_16H_MBKC1_AGAROSE_VIDEO_GLOBS'),
    ('17e', 'pfnd_kir', 'flat', 'FIGURE_4I_PFND_KIR_VIDEO_GLOBS'),
    ('17e', 'pfnd_kir', 'agarose', 'FIGURE_4U_PFND_KIR_VIDEO_GLOBS'),
    ('17f', 'epg_kir', 'flat', 'EXTENDED_DATA_FIGURE_14I_EPG_VIDEO_GLOBS'),
    ('17f', 'epg_kir', 'agarose', 'EXTENDED_DATA_FIGURE_16I_EPG_AGAROSE_VIDEO_GLOBS'),
]


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')


def notebook_strings(path):
    """Read literal assignments only; never execute notebook code or recipes."""
    found = {}
    for cell in json.loads(Path(path).read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in found and found[target.id] != node.value.value:
                            raise ValueError(f'Ambiguous notebook assignment: {target.id}')
                        found[target.id] = node.value.value
    return found


def source_bundle(path):
    with np.load(path, allow_pickle=True) as z:
        bundle = {k: z[k] for k in z.files}
    for key in ('agarose_farthest_from_reward_only', 'agarose_wall_facing_entry_only',
                'agarose_exclude_reward_facing_arc_entries', 'exp_target_sync_bucket_filter_enabled',
                'exp_pi_threshold_filter_enabled'):
        if bool(bundle[key]):
            raise ValueError(f'Unexpected source selection/filter {key}: {path}')
    if (str(bundle['agarose_dual_circle_boundary_policy']) != 'hysteretic'
        or float(bundle['agarose_outer_radius_offset_mm']) != 1.
        or float(bundle['agarose_inner_radius_offset_mm']) != 0.
        or int(bundle['min_agarose_episodes']) != 5
        or float(bundle['bucket_len_min']) != 10.
        or float(bundle['agarose_pre_window_min']) != 10.):
        raise ValueError(f'Unsupported manuscript recipe settings: {path}')
    for prefix in ('agarose', 'agarose_pre'):
        n, numerator = bundle[prefix+'_total_exp'], bundle[prefix+'_avoid_exp']
        values = np.divide(numerator, n, out=np.full(n.shape, np.nan), where=n >= 5)
        np.testing.assert_allclose(values, bundle[prefix+'_ratio_exp'], equal_nan=True, atol=1e-12, rtol=1e-12)
    return bundle


def admitted_trajectory(context):
    """Replay only production admission checks, including physical-fly note overrides."""
    va = context['analysis']
    trj = Trajectory.__new__(Trajectory)
    trj.x, trj.y = context['x'].copy(), context['y'].copy()
    trj.theta, trj.f, trj.va, trj.opts = None, 0, va, va.opts
    if not trj._isEmpty():
        trj._setLostFrameThreshold()
        trj._interpolate()
        if not trj.bad():
            trj._calcDistances()
            trj._suspiciousJumps()
    before_note = bool(trj.bad())
    va.trx, va.trxf = [trj], (context['physical_id'],)
    va._readNoteFile(context['video_path'])
    return not bool(trj.bad()), before_note != bool(trj.bad())


def csv_records(frame):
    return frame.to_dict('records')


def pooled_records(frame):
    result = defaultdict(list)
    for r in csv_records(frame):
        result[r['metric'], r['variant']].append(dict(start=int(r['start_frame']),
            stop=int(r['stop_frame_exclusive']), support_start=int(r['support_start_frame']),
            support_stop=int(r['support_stop_frame_exclusive']), value=float(r['value']),
            heading_frames=json.loads(r['heading_sampled_frames'])))
    return dict(result)


def summarize(out, datasets, subjects, windows, donors, episodes, auc, availability):
    tables = {name: pd.DataFrame(rows) for name, rows in (
        ('datasets', datasets), ('subjects', subjects), ('windows', windows),
        ('donor_support', donors), ('reference_episodes', episodes), ('auc_windows', auc),
        ('source_availability', availability))}
    for name, frame in tables.items():
        frame.to_csv(out/(name+'.csv'), index=False)
    d, w = tables['donor_support'], tables['windows']
    keys = ['genotype', 'chamber', 'metric', 'variant', 'period', 'baseline_eligible']
    coverage = []
    for key, group in w.groupby(keys, sort=False):
        info = dict(zip(keys, key))
        select = pd.Series(True, index=d.index)
        for k, v in info.items():
            select &= d[k].eq(v)
        g = d[select]
        seconds = float(g.duration_s.sum())
        row = info | dict(subjects=len(group), videos=group.video_path.nunique(),
            lost_seconds=float(group.lost_seconds.sum()), donors=len(g), donor_lost_seconds=seconds,
            excluded_lost_seconds=float(group.excluded_lost_seconds.sum()),
            observed_reference_episodes=int(group.observed_reference_episodes.sum()),
            completely_observed_windows=int(group.fully_observed_window.sum()))
        for label, column in [('length', 'length_candidates'), ('spatial', 'spatial_candidates'),
                              ('episode', 'exposed_candidates'), ('minimum_five', 'minimum_five_candidates')]:
            supported = g[g[column] > 0]
            row[label+'_supported_donors'] = len(supported)
            row[label+'_gap_fraction'] = len(supported)/len(g) if len(g) else np.nan
            row[label+'_lost_time_fraction'] = supported.duration_s.sum()/seconds if seconds else np.nan
            row[label+'_subjects'] = len(supported[['video_path', 'pair']].drop_duplicates())
            row[label+'_videos'] = supported.video_path.nunique()
        coverage.append(row)
    summary = pd.DataFrame(coverage)
    summary.to_csv(out/'coverage.csv', index=False)
    if len(d):
        rows = []
        extra = ['duration_bin', 'wall_context', 'recording_loss_stratum']
        for key, g in d.groupby(keys+extra, sort=False):
            supported = g.exposed_candidates > 0
            rows.append(dict(zip(keys+extra, key)) | dict(
                donors=len(g), donor_seconds=float(g.duration_s.sum()), episode_supported_donors=int(supported.sum()),
                episode_supported_seconds=float(g.loc[supported, 'duration_s'].sum()),
                minimum_five_supported_donors=int((g.minimum_five_candidates > 0).sum())))
        pd.DataFrame(rows).to_csv(out/'coverage_by_duration_context.csv', index=False)
    def table(frame):
        return '<div class="scroll">'+frame.to_html(index=False, na_rep='undefined', float_format=lambda v: f'{v:,.4g}')+'</div>'
    eligible = summary[summary.baseline_eligible]
    # Coverage is a feasibility condition for contrasts, not an effect estimate.
    contrasts = []
    for genotype in ('ctrl', 'antennae_removed', 'mbkc1_kir', 'pfnd_kir', 'epg_kir'):
        for period in sorted(eligible.loc[eligible.metric.eq('agarose_avoidance'), 'period'].unique()):
            sides = eligible[(eligible.genotype == genotype) & (eligible.metric == 'agarose_avoidance') & (eligible.period == period)]
            row = dict(contrast='agarose minus flat', genotype=genotype, metric='agarose_avoidance', period=period)
            for chamber in ('flat', 'agarose'):
                side = sides[sides.chamber == chamber]
                row[chamber+'_eligible_subjects'] = int(side.subjects.sum())
                row[chamber+'_supported_subjects'] = int(side.episode_subjects.sum())
                row[chamber+'_minimum_five_supported_subjects'] = int(side.minimum_five_subjects.sum())
            contrasts.append(row)
    for metric, variant in eligible[eligible.metric.ne('agarose_avoidance')][['metric', 'variant']].drop_duplicates().itertuples(index=False, name=None):
        for a, b in [('pfnd_kir', 'ctrl'), ('antennae_removed', 'ctrl'), ('antennae_removed', 'pfnd_kir')]:
            row = dict(contrast=f'{a} minus {b}', genotype='flat cohorts', metric=metric, variant=variant, period='T2_SB2-5')
            for label, genotype in [('first', a), ('second', b)]:
                side = eligible[(eligible.metric == metric) & (eligible.variant == variant) & (eligible.genotype == genotype)]
                row[label+'_eligible_subjects'] = int(side.subjects.sum())
                row[label+'_supported_subjects'] = int(side.episode_subjects.sum())
                row[label+'_minimum_five_supported_subjects'] = int(side.minimum_five_subjects.sum())
            contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out/'contrast_coverage.csv', index=False)
    auc_summary = tables['auc_windows'].groupby(['genotype', 'chamber', 'training']).agg(
        subjects=('pair', 'size'), production_auc_defined=('production_auc_defined', 'sum'),
        completely_observed_auc_windows=('fully_observed_window', 'sum'),
        trusted_defined_auc_windows=('trusted_defined_auc_window', 'sum')).reset_index()
    auc_summary.to_csv(out/'auc_coverage.csv', index=False)
    display = eligible[['genotype', 'chamber', 'metric', 'variant', 'period', 'subjects', 'videos',
        'donors', 'episode_gap_fraction', 'episode_lost_time_fraction', 'episode_subjects', 'minimum_five_subjects']]
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Tracking-gap reference-support audit</title>
<style>body{font:15px/1.6 system-ui;margin:30px;padding:15px;color:#243449}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd;padding:7px}th{background:#eef3f8}p{max-width:1100px}</style>
<h1>Reference support for tracking-loss contrast experiments</h1>
<p>This is an exhaustive same-fly, same-window support census. No gaps are injected and no effects on manuscript contrasts are estimated. Every original x-NaN gap overlapping a window contributes to its loss inventory. Donors require both observed bracketing endpoints inside that window; boundary/unbracketed loss remains explicit.</p>
<p>Placements preserve exact duration, two seconds of observed context on both sides, both endpoint wall-distance categories, and the pilot's 2 mm matching tolerances. All five tracker fields must be finite throughout the parent reference run. Complete episode support must fit inside one such run. Minimum-five support additionally requires five complete reference episodes for that exact metric/radius in that parent run. Alignment heading overlap is a separate donor-level diagnostic.</p>
<p>Gap and lost-time coverage fractions use eligible donor gaps as their denominator, not all tracking loss. Consult excluded_lost_seconds in coverage.csv for loss outside the donor definition. Each row below is restricted to flies eligible for that baseline manuscript metric/cell; the CSV also retains ineligible subjects. Subjects with no donor gaps are included in subject totals but cannot support transplantation of their own gaps.</p>
<p>Local support does not establish a fully observed manuscript window, a representative hidden behavior distribution, independent biological replicates, or adequate precision. Episode identities and candidate counts permit reference reuse to be assessed. Unsupported contexts require additional reference data or a narrower claim.</p>'''
    document += '<h2>Source populations and validation</h2>'+table(tables['datasets'])
    document += '<h2>Eligible metric/window support</h2>'+table(display)
    document += '<h2>Whole-window agarose AUC reference availability</h2><p>AUC eligibility follows the production group-array calculation, including Pre for T1 and one globally all-NaN trailing-column removal. Trusted windows require a defined AUC, all five fields observed continuously across the actual evaluated span, and complete observed episode support. These are availability counts, not simulated AUC error.</p>'+table(auc_summary)
    document += '<h2>Contrasts: support on both sides</h2>'+table(pd.DataFrame(contrasts))
    document += '<p>Nonzero support on both sides is necessary but does not establish representative coverage or sufficient sample size. Candidate overlap with an episode can leave its outcome unchanged.</p>'
    names = list(tables)+['coverage', 'coverage_by_duration_context', 'contrast_coverage', 'auc_coverage', 'provenance']
    document += '<h2>Audit tables</h2><ul>'+''.join(f'<li><a href="{name}.{("json" if name == "provenance" else "csv")}">{html.escape(name)}</a></li>' for name in names if name == 'provenance' or (out/(name+'.csv')).exists())+'</ul></html>'
    (out/'report.html').write_text(document)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, default=ROOT/'results/episode_tracking_loss/flat_large_2026-11-08_reviewed')
    p.add_argument('--notebook', type=Path, default=ROOT/'notebooks/manuscript_figure_panels.ipynb')
    p.add_argument('--resume', action='store_true', help='Reuse validated per-video checkpoints in this output directory')
    args = p.parse_args(argv)
    if args.resume:
        out = args.output_dir
        if not (out/'checkpoints').is_dir():
            p.error('--resume requires an existing checkpoints directory')
        if (out/'provenance.json').exists():
            p.error('Completed audit outputs cannot be overwritten')
    else:
        out = empty_folder(args.output_dir)
        (out/'checkpoints').mkdir()
    sources = Sources(ROOT); track_code(sources)
    for path in (Path(__file__), ROOT/'src/analysis/tracking_gap_support.py', ROOT/'src/exporting/agarose_sli_bundle.py',
                 ROOT/'src/plotting/time_series_auc.py', ROOT/'src/plotting/com_sli_bundle_plotter.py',
                 ROOT/'src/plotting/wall_contacts_per_reward_interval_totals.py'):
        sources.track(path.resolve())
    literals = notebook_strings(sources.track(args.notebook))
    frozen = pd.read_csv(sources.track(args.source_dir/'subjects.csv'))
    frozen = frozen[frozen.policy.eq('baseline') & frozen.period.eq('T2_SB2-5') & frozen.metric.ne('agarose_avoidance')]
    original_eps = pd.read_csv(sources.track(args.source_dir/'episodes.csv'))
    original_eps = original_eps[original_eps.period.eq('T2_SB2-5') & original_eps.metric.ne('agarose_avoidance')]
    original_manifest = json.loads(sources.track(args.source_dir/'provenance.json').read_text())
    fixed_inputs = {k: v['sha256'] for k, v in sources.manifest.items()}
    frozen_groups = {key: g for key, g in frozen.groupby(['cohort', 'video_path', 'pair'])}
    episode_groups = {key: g for key, g in original_eps.groupby(['cohort', 'video_path', 'pair'])}
    datasets, subjects, windows, donors, reference_eps, auc, availability = [], [], [], [], [], [], []
    validated_original = set()
    for panel, genotype, chamber, variable in DATASETS:
        dataset = genotype+'_'+chamber
        bundle_path = sources.track((ROOT/literals[f'extended_data_figure_{panel}_{chamber}_bundle']).resolve())
        bundle = source_bundle(bundle_path)
        archive_videos = [str(Path(v).resolve()) for v in bundle['video_ids']]
        expected_videos = Counter(archive_videos)
        video_paths = set()
        for pattern in literals[variable].split(','):
            matched = sorted({str(Path(v).resolve()) for v in glob.glob(pattern) if Path(v).suffix.lower() == '.avi'})
            availability.append(dict(dataset=dataset, source='notebook_pattern', identifier=pattern, matched_videos=len(matched), status='matched' if matched else 'no matches'))
            video_paths.update(matched)
        if set(expected_videos)-video_paths:
            raise ValueError(f'Archived videos missing from notebook expansion: {dataset}: {set(expected_videos)-video_paths}')
        dataset_auc = []
        admitted_count = 0
        nb = bundle['agarose_total_exp'].shape[2]
        for vi, video in enumerate(sorted(video_paths), 1):
            print(f'{dataset}: video {vi}/{len(video_paths)} {Path(video).name}', flush=True)
            for suffix in ('.trx', '.data'):
                path = Path(video).with_suffix(suffix)
                if not path.exists():
                    raise FileNotFoundError(f'Missing manuscript source: {path}')
                sources.track(path)
                old = original_manifest['sources'].get(str(path))
                if old and old['sha256'] != sources.manifest[str(path)]['sha256']:
                    raise ValueError(f'Raw source changed since first audit: {path}')
            note = Path(video).with_suffix('').with_name(Path(video).stem+'__note.txt')
            if note.exists():
                sources.track(note)
            video_stat = Path(video).stat()
            video_info = dict(size=video_stat.st_size, mtime_ns=video_stat.st_mtime_ns)
            availability.append(dict(dataset=dataset, source='video_metadata_dependency', identifier=video,
                matched_videos=1, status='accessible; video size/mtime recorded (not content-hashed)', **video_info))
            fingerprint = hashlib.sha256(json.dumps(dict(fixed=fixed_inputs,
                bundle=sources.manifest[str(bundle_path)]['sha256'],
                raw={suffix: sources.manifest[str(Path(video).with_suffix(suffix))]['sha256'] for suffix in ('.trx', '.data')},
                note=sources.manifest[str(note)]['sha256'] if note.exists() else None,
                video_stat=video_info), sort_keys=True).encode()).hexdigest()
            checkpoint = out/'checkpoints'/f'{dataset}_{hashlib.sha256(video.encode()).hexdigest()[:16]}.json'
            containers = dict(subjects=subjects, windows=windows, donors=donors, references=reference_eps, auc=auc)
            if args.resume and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['fingerprint'] != fingerprint:
                    raise ValueError(f'Checkpoint sources/code changed: {checkpoint}')
                for name, target in containers.items():
                    target.extend(saved[name])
                availability.extend(saved['availability'])
                dataset_auc.extend(saved['dataset_auc'])
                admitted_count += sum(s['admitted'] for s in saved['subjects'])
                validated_original.update((genotype, s['video_path'], s['pair']) for s in saved['subjects']
                    if chamber == 'flat' and (genotype, s['video_path'], s['pair']) in frozen_groups and s['admitted'])
                continue
            offsets = {name: len(target) for name, target in containers.items()}
            old_auc, old_availability = len(dataset_auc), len(availability)
            raw_source = load_pickle(Path(video).with_suffix('.trx'))
            reconstructed = []
            for pair in (0, 1):
                # Some notebook sources are empty excluded recordings without
                # calibration metadata. No archive rows are assigned to them.
                if all(np.isnan(np.asarray(raw_source['x'][f], dtype=float)).mean() > .99 for f in (0, 1)) and not expected_videos[video]:
                    subjects.append(dict(dataset=dataset, genotype=genotype, chamber=chamber, video_path=video,
                        pair=pair, admitted=False, note_override=False, fps=None,
                        raw_loss_fraction=float(np.isnan(raw_source['x'][pair]).mean()),
                        recording_loss_pct=None, recording_loss_stratum='>60% / unavailable',
                        source_status='excluded empty recording; absent from manuscript archive'))
                    continue
                with redirect_stdout(io.StringIO()):
                    context = load_context(video, pair, 'exp', include_analysis=True)
                    admitted, note_override = admitted_trajectory(context)
                base = dict(dataset=dataset, genotype=genotype, chamber=chamber, video_path=video, pair=pair)
                missing = np.isnan(context['x'])
                good = np.flatnonzero(~missing)
                initial = int(good[0]) if len(good) else len(missing)
                loss = float(missing[initial:].mean()) if initial < len(missing) else np.nan
                base.update(recording_loss_pct=100*loss, recording_loss_stratum=(
                    '0-10%' if loss <= .1 else '10-30%' if loss <= .3 else '30-60%' if loss <= .6 else '>60% / unavailable'))
                subjects.append(base | dict(admitted=admitted, note_override=note_override, fps=context['fps'],
                    raw_loss_fraction=float(np.isnan(context['x']).mean())))
                if not admitted:
                    continue
                admitted_count += 1
                raw = raw_fields(context)
                with redirect_stdout(io.StringIO()):
                    va, trj = make_trajectory(context)
                    gap_index = GapIndex(trj.nan, context['fps'])
                    by_period, periods = agarose_episodes(va, trj, dict(px_per_mm=context['px_per_mm']), gap_index, bundle)
                # Include every canonical bucket; the production archive has six,
                # although the final bucket is often absent after synchronization.
                periods = periods[:1]+[(f'T{ti}_SB{bi}', *(context['geometries'][ti-1]['sync_buckets'][bi-1] if bi <= len(context['geometries'][ti-1]['sync_buckets']) else (0, 0))) for ti in (1, 2) for bi in range(1, nb+1)]
                signature = []
                ratios = {}
                period_support = {}
                for period, lo, hi in periods:
                    eps = by_period[period]
                    n, numerator = len(eps), sum(e['value'] for e in eps)
                    signature.extend((n, int(numerator)))
                    ratios[period] = numerator/n if n >= 5 else np.nan
                    ag_records = [dict(start=int(e['start_frame']), stop=int(e['stop_frame_exclusive']),
                        support_start=int(e['support_start_frame']), support_stop=int(e['support_stop_frame_exclusive']),
                        value=float(e['value']), heading_frames=[]) for e in eps]
                    period_support[period] = ag_records
                    if lo == hi:
                        windows.append(base | dict(metric='agarose_avoidance', variant='physical_1mm', period=period,
                            baseline_eligible=False, status='absent sync bucket', start_frame=lo, stop_frame_exclusive=hi,
                            lost_seconds=0., donor_gaps=0, excluded_lost_seconds=0., observed_reference_episodes=0,
                            fully_observed_window=False, baseline_episodes=0))
                        continue
                    geom = geometry(context)
                    geom['reward_center'] = va.trns[0 if period == 'Pre' else int(period[1])-1].circles(0)[0][:2]
                    stats, rows, refs = audit_window(raw, context['fps'], int(lo), int(hi), geom,
                        {('agarose_avoidance', 'physical_1mm'): ag_records})
                    cell = base | dict(period=period, baseline_eligible=n >= 5)
                    windows.append(cell | stats | dict(metric='agarose_avoidance', variant='physical_1mm', status='audited',
                        baseline_episodes=n, observed_reference_episodes=len(refs)))
                    donors.extend(cell | r for r in rows)
                    reference_eps.extend(cell | r for r in refs)
                reconstructed.append(tuple(signature))
                values = [[ratios['Pre']]+[ratios[f'T1_SB{i}'] for i in range(1, nb+1)],
                          [ratios[f'T2_SB{i}'] for i in range(1, nb+1)]]
                dataset_auc.append(values)
                observed = np.logical_and.reduce([np.isfinite(v) for v in raw.values()])
                for ti in (1, 2):
                    used = (['Pre'] if ti == 1 else [])+[f'T{ti}_SB{i}' for i in range(1, nb+1)]
                    valid = [(name, int(lo), int(hi)) for name, lo, hi in periods if name in used and hi > lo]
                    if not valid:
                        auc.append(base | dict(training=ti, fully_observed_window=False,
                            complete_episode_support=False, start_frame=None, stop_frame_exclusive=None))
                        continue
                    lo, hi = min(a for _, a, _ in valid), max(b for _, _, b in valid)
                    all_support = all(lo <= e['support_start'] < e['support_stop'] <= hi
                        and observed[e['support_start']:e['support_stop']].all() for name, _, _ in valid for e in period_support[name])
                    auc.append(base | dict(training=ti, fully_observed_window=bool(observed[lo:hi].all()),
                        complete_episode_support=bool(all_support), start_frame=lo, stop_frame_exclusive=hi))
                key = (genotype, video, pair)
                if chamber == 'flat' and key in frozen_groups:
                    validated_original.add(key)
                    cells = frozen_groups[key]
                    records = pooled_records(episode_groups.get(key, original_eps.iloc[:0]))
                    # Retain zero-episode cells explicitly.
                    records.update({(r.metric, r.variant): records.get((r.metric, r.variant), []) for r in cells.itertuples()})
                    rr = context['geometries'][1]['sync_buckets']
                    if len(rr) >= 5:
                        lo, hi = int(rr[1][0]), int(rr[4][1])
                        stats, rows, refs = audit_window(raw, context['fps'], lo, hi, geometry(context), records)
                        gates = {(r.metric, r.variant): bool(r.included) for r in cells.itertuples()}
                        for r in cells.itertuples():
                            windows.append(base | stats | dict(metric=r.metric, variant=r.variant, period='T2_SB2-5',
                                baseline_eligible=bool(r.included), status='audited', baseline_episodes=len(records[r.metric, r.variant]),
                                observed_reference_episodes=sum(e['metric'] == r.metric and e['variant'] == r.variant for e in refs)))
                        donors.extend(base | dict(period='T2_SB2-5', baseline_eligible=gates[r['metric'], r['variant']]) | r for r in rows)
                        reference_eps.extend(base | dict(period='T2_SB2-5', baseline_eligible=gates[r['metric'], r['variant']]) | r for r in refs)
                    else:
                        for r in cells.itertuples():
                            windows.append(base | dict(metric=r.metric, variant=r.variant, period='T2_SB2-5',
                                baseline_eligible=bool(r.included), status='incomplete pooled window',
                                start_frame=0, stop_frame_exclusive=0, lost_seconds=0., donor_gaps=0,
                                excluded_lost_seconds=0., observed_reference_episodes=0,
                                fully_observed_window=False, baseline_episodes=len(records[r.metric, r.variant])))
            expected = [agarose_signature(bundle, i, [p[0] for p in periods]) for i, v in enumerate(archive_videos) if v == video]
            if sorted(reconstructed) != sorted(expected):
                raise ValueError(f'Archive video-level multiset mismatch: {dataset} {video}\nreconstructed={reconstructed}\nexpected={expected}')
            saved = {name: target[offsets[name]:] for name, target in containers.items()}
            saved.update(fingerprint=fingerprint, dataset_auc=dataset_auc[old_auc:],
                availability=availability[old_availability:])
            temporary = checkpoint.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(saved, default=json_scalar))
            temporary.replace(checkpoint)
        # Preserve the production group-level treatment of a globally absent last bucket.
        for ti in (1, 2):
            matrix = np.array([v[ti-1] for v in dataset_auc], dtype=float)
            production = signed_auc_values(matrix)
            original = bundle['agarose_ratio_exp'][:, ti-1, :]
            if ti == 1:
                original = np.column_stack([bundle['agarose_pre_ratio_exp'], original])
            archived_auc = signed_auc_values(original)
            np.testing.assert_allclose(np.sort(production[np.isfinite(production)]), np.sort(archived_auc[np.isfinite(archived_auc)]), atol=1e-10)
            target = [r for r in auc if r['dataset'] == dataset and r['training'] == ti]
            for row, value in zip(target, production, strict=True):
                row['production_auc_defined'] = bool(np.isfinite(value))
                row['production_auc'] = float(value)
                row['trusted_defined_auc_window'] = bool(np.isfinite(value) and row['fully_observed_window'] and row['complete_episode_support'])
        datasets.append(dict(dataset=dataset, panel='ED'+panel, genotype=genotype, chamber=chamber,
            notebook_video_variable=variable, source_bundle=str(bundle_path), source_videos=len(video_paths),
            admitted_subjects=admitted_count, archive_subjects=len(archive_videos), archive_videos=len(expected_videos),
            validation='video-level unordered counts/numerators; production AUC samples'))
    if validated_original != set(frozen_groups):
        raise ValueError(f'Original source roster not fully audited: {set(frozen_groups)-validated_original}')
    sources.verify_unchanged()
    summarize(out, datasets, subjects, windows, donors, reference_eps, auc, availability)
    (out/'provenance.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        sources=sources.manifest, settings=dict(padding_seconds=2., tolerance_mm=2., search='exhaustive',
        scope='all manuscript source videos; same fly and metric window; no loss injection',
        notebook=str(args.notebook), source_dir=str(args.source_dir))), indent=2)+'\n')
    print(out/'report.html')


if __name__ == '__main__':
    main()
