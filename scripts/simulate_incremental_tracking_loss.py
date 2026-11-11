#!/usr/bin/env python3
"""Add one supported gap per fly/repeat; remeasure complete T2 SB2–5 windows.

This measures incremental sensitivity conditional on the current reconstruction,
not the unknown bias already caused by real missing tracking. First batch: the
three pooled between-reward metrics, with recorded rewards and source gates fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import html
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_episode_tracking_loss import Sources, COHORTS, close_value, load_context, sha256
from scripts.simulate_tracking_gaps import (
    DEFAULT_SOURCE, FIELDS, BINS, build_trajectory, native_records, raw_fields,
    primary_window, geometry, track_code, verify_wall_baseline,
)
from src.analysis.tracking_gap_simulation import (
    true_runs, candidate_starts, endpoint_features, matching_candidates, exposed_candidates,
)
from src.analysis.tracking_gap_support import observed_references
from src.analysis.tracking_gap_incremental import (
    METRICS, CONTRASTS, stable_rng, aggregate, episode_changes, contrast_components,
    video_bootstrap_weights,
)


def atomic_json(path, data):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def checked(sources, path, manifest):
    path = sources.track(path.resolve())
    if str(path) not in manifest or sources.manifest[str(path)]['sha256'] != manifest[str(path)]['sha256']:
        raise ValueError(f'Frozen source differs or is absent: {path}')
    return path


def arm_metrics(arm):
    return METRICS if arm == 'empirical_spatial' else (arm.split(':')[1],)


def simulate_subject(key, frozen, donors, args, sources):
    cohort, video, pair = key
    with redirect_stdout(io.StringIO()):
        context = load_context(video, pair, 'exp', include_analysis=True)
        va, trj = build_trajectory(context)
        records = native_records(va, trj, context, pooled_only=True)
        wall_verified = verify_wall_baseline(context, trj, sources, cohort, pair)
    raw = raw_fields(context)
    observed = np.logical_and.reduce([np.isfinite(raw[f]) for f in FIELDS])
    missing = np.isnan(raw['x'])
    lo, hi = primary_window(context)
    runs = true_runs(observed, lo, hi)
    baseline, masks = {}, {}
    for metric, row in frozen.set_index('metric').iterrows():
        eps = records[metric, 'full']
        reduced = aggregate(eps)
        close_value(reduced['n_episodes'], row.n_episodes, f'{key} {metric} episode count')
        close_value(reduced['episode_value_sum'], row.episode_value_sum, f'{key} {metric} sum')
        close_value(reduced['value'], row.measured_value, f'{key} {metric} baseline')
        if row.included:
            close_value(reduced['value'], row.reference_value, f'{key} {metric} manuscript')
        baseline[metric] = reduced
        _, retained = observed_references(observed, eps, lo, hi)
        mask = np.zeros(len(observed), dtype=bool)
        for ep in retained:
            mask[ep['support_start']:ep['support_stop']] = True
        masks[metric] = mask
    # All three audit cells enumerate exactly the same real donors.
    donor_keys = None
    for metric in METRICS:
        g = donors[donors.metric.eq(metric)]
        keys = list(zip(g.gap_start_frame, g.gap_stop_frame_exclusive))
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate census donors')
        if donor_keys is not None and set(keys) != donor_keys:
            raise ValueError('Metric donor inventories differ')
        donor_keys = set(keys)
    actual_donors = {(a, b) for a, b in true_runs(missing)
                     if lo <= a-1 and b < hi and
                     np.isfinite([raw['x'][a-1], raw['y'][a-1], raw['x'][b], raw['y'][b]]).all()}
    if donor_keys != actual_donors:
        raise ValueError('Real donor inventory differs from support census')
    base = dict(cohort=cohort, video_path=video, pair=pair,
                window_start=lo, window_stop_exclusive=hi, fps=context['fps'],
                original_window_lost_frames=int(missing[lo:hi].sum()), window_frames=hi-lo,
                original_recording_loss_pct=float(frozen.iloc[0].recording_loss_pct),
                wall_export_intervals_verified=wall_verified)
    eligible = set(frozen.loc[frozen.included, 'metric'])
    gate_passed = set(frozen.loc[frozen.source_gate_passed, 'metric'])
    arms = ['empirical_spatial']+[f'episode_targeted:{m}' for m in METRICS if m in eligible]
    coverage, trials = [], []
    for arm in arms:
        metric = next(m for m in METRICS if m in gate_passed) if arm == 'empirical_spatial' else arm.split(':')[1]
        pool = donors[donors.metric.eq(metric)].copy()
        field = 'spatial_candidates' if arm == 'empirical_spatial' else 'exposed_candidates'
        supported = pool[pool[field] > 0].reset_index(drop=True)
        frames = pool.gap_stop_frame_exclusive-pool.gap_start_frame
        supported_frames = supported.gap_stop_frame_exclusive-supported.gap_start_frame
        metrics = [m for m in arm_metrics(arm) if m in gate_passed]
        for m in metrics:
            coverage.append(dict(**base, arm=arm, metric=m, baseline_value=baseline[m]['value'],
                baseline_eligible=bool(m in eligible),
                baseline_n_episodes=baseline[m]['n_episodes'], donor_gaps=len(pool),
                donor_lost_frames=int(frames.sum()), supported_donors=len(supported),
                supported_donor_lost_frames=int(supported_frames.sum()),
                supported=bool(len(supported)),
                supported_duration_bins=';'.join(b for b in BINS if b in set(supported.duration_bin))))
        if not len(supported):
            continue
        bins = [b for b in BINS if b in set(supported.duration_bin)]
        order = stable_rng(args.seed, *key, arm, 'bins').permutation(len(bins))
        for repeat in range(args.repeats):
            rng = stable_rng(args.seed, *key, arm, repeat)
            if arm == 'empirical_spatial':
                choices = supported
                donor_probability = 1/len(supported)
            else:
                chosen_bin = bins[order[repeat % len(bins)]]
                choices = supported[supported.duration_bin.eq(chosen_bin)]
                # Random bin permutation, cycled across repeats: each repeat's
                # marginal bin probability is uniform over supported bins.
                donor_probability = 1/(len(bins)*len(choices))
            donor = choices.iloc[int(rng.integers(len(choices)))]
            length = int(donor.gap_stop_frame_exclusive-donor.gap_start_frame)
            starts, parents = candidate_starts(runs, length, int(np.ceil(2*context['fps'])))
            features = endpoint_features(raw['x'], raw['y'], starts, length, **geometry(context))
            donor_features = {n: v[0] for n, v in endpoint_features(raw['x'], raw['y'],
                [int(donor.gap_start_frame)], length, **geometry(context)).items()}
            matched = matching_candidates(features, donor_features, 2.)
            if len(matched) != int(donor.spatial_candidates):
                raise ValueError('Spatial support count differs from census')
            if arm != 'empirical_spatial':
                exposed = np.zeros(len(starts), dtype=bool)
                exposed[exposed_candidates(starts, length, masks[metric])] = True
                matched = matched[exposed[matched]]
            if len(matched) != int(donor[field]):
                raise ValueError('Targeted support count differs from census')
            selected = int(matched[int(rng.integers(len(matched)))])
            a, b = int(starts[selected]), int(starts[selected]+length)
            parent = runs[int(parents[selected])]
            with redirect_stdout(io.StringIO()):
                va, perturbed = build_trajectory(context, (a, b))
                after = native_records(va, perturbed, context, pooled_only=True)
            expected_missing = missing.copy(); expected_missing[a:b] = True
            np.testing.assert_array_equal(perturbed.nan, expected_missing)
            for m in metrics:
                reduced = aggregate(after[m, 'full'])
                touched = bool(masks[m][a:b].any())
                trials.append(dict(**base, arm=arm, repeat=repeat, metric=m,
                    donor_start=int(donor.gap_start_frame), donor_stop_exclusive=int(donor.gap_stop_frame_exclusive),
                    donor_duration_bin=donor.duration_bin, added_frames=length, added_seconds=length/context['fps'],
                    donor_selection_probability=donor_probability, placement_selection_probability=1/len(matched),
                    placement_start=a, placement_stop_exclusive=b, parent_start=parent[0], parent_stop_exclusive=parent[1],
                    matching_candidates=len(matched), complete_observed_episode_support_exposed=touched,
                    added_window_loss_fraction=length/(hi-lo),
                    before_n_episodes=baseline[m]['n_episodes'], before_value=baseline[m]['value'],
                    before_eligible=bool(m in eligible),
                    after_n_episodes=reduced['n_episodes'], after_episode_value_sum=reduced['episode_value_sum'],
                    after_eligible=reduced['eligible'], after_value=reduced['value'],
                    value_change=reduced['value']-baseline[m]['value'],
                    **episode_changes(records[m, 'full'], after[m, 'full'])))
    for f in FIELDS:
        np.testing.assert_array_equal(raw[f], raw_fields(context)[f])
    return clean(dict(coverage=coverage, trials=trials))


def subject_job(key, group, donors, args, signature, checkpoint, snapshot):
    sources = Sources(ROOT)
    sources.manifest = snapshot
    sources.verify_unchanged()
    result = dict(signature=signature, **simulate_subject(key, group, donors, args, sources))
    sources.verify_unchanged()
    atomic_json(checkpoint, result)
    return result


def summaries(coverage, trials, frozen, draws, seed):
    cohort_rows, contrasts = [], []
    for (arm, metric), cov in coverage.groupby(['arm', 'metric'], sort=True):
        data = trials[trials.arm.eq(arm) & trials.metric.eq(metric)] if len(trials) else trials
        for cohort in COHORTS:
            c = cov[cov.cohort.eq(cohort)]
            supported = c[c.supported]
            t = data[data.cohort.eq(cohort)] if len(data) else data
            defined = t[t.after_eligible] if len(t) else t
            all_baseline = frozen[frozen.metric.eq(metric) & frozen.cohort.eq(cohort) & frozen.included]
            cohort_rows.append(dict(arm=arm, metric=metric, cohort=cohort,
                manuscript_flies=len(all_baseline), gate_passed_flies=len(c),
                supported_flies=len(supported), supported_baseline_eligible_flies=int(supported.baseline_eligible.sum()),
                supported_videos=supported.video_path.nunique(), donor_gaps=int(c.donor_gaps.sum()),
                supported_donors=int(c.supported_donors.sum()),
                donor_gap_coverage=c.supported_donors.sum()/c.donor_gaps.sum() if c.donor_gaps.sum() else np.nan,
                donor_time_coverage=c.supported_donor_lost_frames.sum()/c.donor_lost_frames.sum() if c.donor_lost_frames.sum() else np.nan,
                manuscript_mean=all_baseline.value.mean(), supported_baseline_mean=supported.baseline_value.mean(),
                trials=len(t), undefined_after_trials=int((~t.after_eligible).sum()) if len(t) else 0,
                flies_losing_eligibility=len(t.loc[t.before_eligible & ~t.after_eligible, ['video_path', 'pair']].drop_duplicates()) if len(t) else 0,
                flies_gaining_eligibility=len(t.loc[~t.before_eligible & t.after_eligible, ['video_path', 'pair']].drop_duplicates()) if len(t) else 0,
                mean_added_seconds=t.added_seconds.mean() if len(t) else np.nan,
                mean_defined_change=defined.value_change.mean() if len(defined) else np.nan,
                max_absolute_defined_change=defined.value_change.abs().max() if len(defined) else np.nan))
        records = data.to_dict('records') if len(data) else []
        rng = stable_rng(seed, arm, metric, 'video-bootstrap')
        for positive, negative in CONTRASTS:
            point = contrast_components(records, positive, negative)
            bootstrap = []
            for _ in range(draws):
                bootstrap.append(contrast_components(records, positive, negative,
                                                      video_bootstrap_weights(records, rng)))
            row = dict(arm=arm, metric=metric, contrast=f'{positive} - {negative}', **point)
            full_means = frozen[frozen.metric.eq(metric) & frozen.included].groupby('cohort').value.mean()
            row['manuscript_contrast'] = full_means.get(positive, np.nan)-full_means.get(negative, np.nan)
            row['support_selection_shift'] = point['before_contrast']-row['manuscript_contrast']
            row['relative_total_change'] = point['total_change']/abs(point['before_contrast']) if abs(point['before_contrast']) > 1e-12 else np.nan
            row['direction_preserved'] = bool(np.sign(point['before_contrast']) == np.sign(point['after_contrast'])) if np.isfinite(point['after_contrast']) and np.isfinite(point['before_contrast']) else None
            for field in ('total_change', 'common_fly_change', 'composition_change'):
                values = np.asarray([b[field] for b in bootstrap], dtype=float)
                defined = values[np.isfinite(values)]
                row[field+'_bootstrap_defined'] = len(defined)
                row[field+'_ci_low'], row[field+'_ci_high'] = np.quantile(defined, [.025, .975]) if len(defined) else (np.nan, np.nan)
            repeat_changes = [contrast_components([r for r in records if r['repeat'] == rep], positive, negative)['total_change']
                              for rep in sorted({r['repeat'] for r in records})]
            row['repeat_change_min'] = min(repeat_changes, default=np.nan)
            row['repeat_change_max'] = max(repeat_changes, default=np.nan)
            contrasts.append(row)
    return pd.DataFrame(cohort_rows), pd.DataFrame(contrasts)


def report(out, cohorts, contrasts, args):
    def table(frame):
        return frame.to_html(index=False, na_rep='undefined', float_format=lambda x: f'{x:.5g}', escape=True)
    text = '''<!doctype html><meta charset="utf-8"><title>Additional tracking-loss sensitivity</title>
<style>body{font:15px/1.5 system-ui;margin:30px;max-width:1600px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}th{background:#eee}section{overflow-x:auto}p{max-width:1100px}</style>
<h1>Additional tracking-loss sensitivity: pooled between-reward metrics</h1>
<p>This compares current reconstructed full-window values with values after adding one artificial gap. Existing real gaps remain unchanged. It estimates incremental susceptibility conditional on the current reconstruction, not error already caused by actual tracking loss or a correction to manuscript values.</p>
<p>All flies passing the frozen admission/SLI gates are attempted, including those whose current metric is undefined because of too few episodes. Same-fly donors preserve exact duration; placements require originally finite x, y, width, height and orientation, observed endpoints, two seconds of padding, identical endpoint wall bins and other endpoint distances within 2 mm. No local five-episode minimum is imposed. Both measurements use complete T2 SB2–5 and the production five-episode minimum. Recorded reward times, admission and SLI gates are fixed. Detection is rerun from recording start; episodes are never clipped to the placement's observed parent.</p>
<p><b>Empirical spatial:</b> donors are sampled uniformly from each fly's spatially supported real gaps and placements uniformly from matching starts. This preserves within-fly observed gap frequencies conditional on support. Flies have equal weight; it is not a pooled-gap or whole-loss-burden estimate. <b>Episode targeted:</b> supported duration bins are randomly permuted and cycled, with uniform donors within each selected bin, and placements must touch complete observed episode support for the named metric. This is a separate stress test. Unsupported donors/flies are not assigned zero effects. Four repeats cannot characterize rare tail effects or establish equivalence.</p>
<h2>1. Cohort coverage, baseline selection and per-fly outcomes</h2><p>Donor gap/time coverage uses bracketed within-window donors as the denominator; terminal and boundary-crossing missingness is outside that distribution. Unsupported flies are excluded from simulation contrasts but retained here. Manuscript means describe the full original eligible pool; supported means show selection before adding loss. Trial counts include repeated masks of the same flies, not independent biological observations.</p>'''
    text = text.replace('Four repeats', f'{args.repeats} repeats')
    text += '<section>'+table(cohorts)+'</section>'
    cols = ['arm', 'metric', 'contrast', 'manuscript_contrast', 'before_contrast', 'after_contrast',
            'total_change', 'total_change_ci_low', 'total_change_ci_high', 'relative_total_change',
            'direction_preserved', 'repeat_change_min', 'repeat_change_max']
    text += '<h2>2. Contrast direction and magnitude</h2><p>Before/after contrasts use the arm-specific supported baseline pool, with after means following production eligibility. Positive minus negative is explicit. Change = after − before, in native units; relative change divides by the absolute before contrast and is unstable near zero. The repeat range describes mask variability. Percentile intervals resample whole videos within cohort, retaining all their flies and repeats together; they quantify sampling variability conditional on this simulation design and available support.</p><section>'+table(contrasts[cols])+'</section>'
    text += '<h2>3. Contrast composition</h2><p>Support selection shift = supported baseline contrast − full manuscript contrast. Common-fly change compares before and after among the exact same flies eligible in both conditions in each repeat. Composition change combines the shift from removing baseline values of flies losing eligibility and the shift from adding post-mask values of newly eligible flies. Total change = common-fly change + composition change. These are distinct from the pre-existing selection into the supported pool.</p><section>'+table(contrasts[['arm', 'metric', 'contrast', 'support_selection_shift', 'common_fly_change', 'eligibility_loss_shift', 'eligibility_gain_shift', 'composition_change', 'total_change']])+'</section>'
    text += '<h2>4. Audit data and limits</h2><p>This bounded batch tests a single additional gap per fly, not clustered multiple-gap masks or full observed loss burdens. Added durations and original loss are recorded per trial. Source cohorts for these three metrics are frozen from the reviewed audit and checked against current archives; the earlier support census is reused only for these unchanged flat-chamber cohorts. Agarose chamber comparisons and turnback/alignment remain follow-on work. No scientific equivalence threshold has been selected, and preservation of significance is not a target.</p><ul>'
    for name in ('coverage.csv', 'trials.csv', 'cohort_summary.csv', 'contrasts.csv', 'provenance.json'):
        text += f'<li><a href="{name}">{html.escape(name)}</a></li>'
    (out/'report.html').write_text(text+'</ul>')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--support-dir', type=Path, default=ROOT/'results/tracking_gap_simulation/support_audit_2026-11-09')
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--bootstrap-draws', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--workers', type=int, default=4, help='Independent subject processes; does not affect sampling')
    p.add_argument('--limit-per-cohort', type=int, help='Deterministic integration smoke run only')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if min(args.repeats, args.bootstrap_draws, args.workers) < 1 or (args.limit_per_cohort is not None and args.limit_per_cohort < 1):
        p.error('Counts must be positive')
    args.output_dir = args.output_dir.resolve()
    args.source_dir = args.source_dir.resolve(); args.support_dir = args.support_dir.resolve()
    out = args.output_dir
    if args.resume:
        if not (out/'checkpoints').is_dir() or (out/'provenance.json').exists():
            p.error('--resume needs an incomplete output with checkpoints')
    elif out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty')
    (out/'checkpoints').mkdir(parents=True, exist_ok=True)
    sources = Sources(ROOT); track_code(sources)
    for path in (Path(__file__), ROOT/'src/analysis/tracking_gap_incremental.py', ROOT/'src/analysis/tracking_gap_support.py',
                 ROOT/'src/plotting/wall_contacts_per_reward_interval_totals.py', ROOT/'src/analysis/episode_tracking_loss.py'):
        sources.track(path.resolve())
    original = json.loads(sources.track(args.source_dir/'provenance.json').read_text())['sources']
    audit = json.loads(sources.track(args.support_dir/'provenance.json').read_text())['sources']
    path = checked(sources, args.source_dir/'subjects.csv', audit)
    frozen = pd.read_csv(path)
    frozen = frozen[frozen.policy.eq('baseline') & frozen.period.eq('T2_SB2-5') & frozen.metric.isin(METRICS)].copy()
    donors = pd.read_csv(sources.track(args.support_dir/'donor_support.csv'), usecols=[
        'genotype', 'chamber', 'video_path', 'pair', 'period', 'metric',
        'gap_start_frame', 'gap_stop_frame_exclusive', 'duration_bin', 'spatial_candidates', 'exposed_candidates'])
    donors = donors[donors.chamber.eq('flat') & donors.period.eq('T2_SB2-5') & donors.metric.isin(METRICS)]
    # Validate current manuscript scalars, gates and interval data against the
    # reviewed source stage. The unrelated notebook/EPG changes do not invalidate
    # these cohorts, but this run does not refresh or endorse that old census.
    for cohort in COHORTS:
        for panel, suffix in [('j', 'tortuosity'), ('k', 'wall_contacts'), ('l', 'wall_contact_trajectory_length')]:
            checked(sources, ROOT/'exports/manuscript'/f'fig4{panel}_{cohort}_{suffix}_archive.npz', original)
            checked(sources, ROOT/'exports/manuscript'/f'fig4{panel}_{cohort}_sli_eligibility.npz', original)
        checked(sources, ROOT/'exports/manuscript'/f'fig4k_{cohort}_wall_contacts_episodes.csv', original)
    settings = dict(repeats=args.repeats, bootstrap_draws=args.bootstrap_draws, seed=args.seed,
                    limit_per_cohort=args.limit_per_cohort, metrics=list(METRICS),
                    padding_seconds=2., endpoint_tolerance_mm=2., additional_gaps_per_repeat=1,
                    evaluation='complete T2 SB2-5; current reconstructed baseline; source gates frozen',
                    source_dir=str(args.source_dir), support_dir=str(args.support_dir))
    fixed = {k: v['sha256'] for k, v in sources.manifest.items()}
    groups = [((c, v, int(p)), g) for (c, v, p), g in frozen.groupby(['cohort', 'video_path', 'pair'], sort=True)]
    groups = [(key, g) for key, g in groups if g.source_gate_passed.any()]
    if args.limit_per_cohort:
        groups = [(k, g) for c in COHORTS for k, g in [(k, g) for k, g in groups if k[0] == c][:args.limit_per_cohort]]
        selected = {k for k, _ in groups}
        frozen = frozen[[tuple(r) in selected for r in frozen[['cohort', 'video_path', 'pair']].itertuples(index=False, name=None)]]
    all_coverage, all_trials, video_metadata, jobs = [], [], {}, []
    for index, (key, group) in enumerate(groups, 1):
        cohort, video, pair = key
        for suffix in ('.trx', '.data'):
            checked(sources, Path(video).with_suffix(suffix), original)
            checked(sources, Path(video).with_suffix(suffix), audit)
        note = Path(video).with_name(Path(video).stem+'__note.txt')
        if note.exists() != (str(note) in original):
            raise ValueError(f'Note-file availability changed: {note}')
        if note.exists():
            checked(sources, note, original)
        stat = Path(video).stat()
        video_metadata[video] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        signature = dict(settings=settings, fixed=fixed, identity=list(key),
                         video_metadata=video_metadata[video], raw={str(Path(video).with_suffix(s)): sources.manifest[str(Path(video).with_suffix(s))]['sha256'] for s in ('.trx', '.data')})
        identifier = hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()[:20]
        checkpoint = out/'checkpoints'/f'{identifier}.json'
        if checkpoint.exists():
            result = json.loads(checkpoint.read_text())
            if result['signature'] != signature:
                raise ValueError(f'Checkpoint inputs changed: {checkpoint}')
            print(f'{index}/{len(groups)} reused {cohort} {Path(video).name} fly {pair}', flush=True)
        else:
            sub = donors[donors.genotype.eq(cohort) & donors.video_path.eq(video) & donors.pair.eq(pair)]
            jobs.append((key, group, sub, args, signature, checkpoint, dict(sources.manifest)))
            continue
        all_coverage.extend(result['coverage']); all_trials.extend(result['trials'])
    if jobs:
        print(f'Simulating {len(jobs)} subjects with {args.workers} worker processes', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(subject_job, *job): job[0] for job in jobs}
            for index, future in enumerate(as_completed(pending), 1):
                key = pending[future]
                result = future.result()
                all_coverage.extend(result['coverage']); all_trials.extend(result['trials'])
                print(f'{index}/{len(jobs)} completed {key[0]} {Path(key[1]).name} fly {key[2]}', flush=True)
    coverage, trials = pd.DataFrame(all_coverage), pd.DataFrame(all_trials)
    coverage = coverage.sort_values(['arm', 'metric', 'cohort', 'video_path', 'pair']).reset_index(drop=True)
    if len(trials):
        trials = trials.sort_values(['arm', 'metric', 'cohort', 'video_path', 'pair', 'repeat']).reset_index(drop=True)
    # Restore undefined scalars from JSON nulls for numeric calculations.
    if len(trials):
        for col in ('before_value', 'after_value', 'value_change'):
            trials[col] = pd.to_numeric(trials[col])
    print('Summarizing contrasts with recording-cluster bootstrap', flush=True)
    cohorts, contrasts = summaries(coverage, trials, frozen, args.bootstrap_draws, args.seed)
    sources.verify_unchanged()
    for video, metadata in video_metadata.items():
        stat = Path(video).stat()
        if metadata != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns):
            raise ValueError(f'Video metadata source changed: {video}')
    for name, data in [('coverage', coverage), ('trials', trials), ('cohort_summary', cohorts), ('contrasts', contrasts)]:
        data.to_csv(out/f'{name}.csv', index=False)
    report(out, cohorts, contrasts, args)
    atomic_json(out/'provenance.json', dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        settings=settings, workers=args.workers, sources=sources.manifest, video_metadata=video_metadata,
        subjects=len(groups), trials=len(trials), limitations='Conditional additional single-gap sensitivity; not existing interpolation bias'))
    print(f'Wrote {out}: {len(groups)} flies; {len(trials)} metric comparisons', flush=True)


if __name__ == '__main__':
    main()
