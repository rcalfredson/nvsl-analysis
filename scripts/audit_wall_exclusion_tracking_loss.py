#!/usr/bin/env python3
"""Compare whole-episode wall exclusion, actual loss exposure and cohort contrasts."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_episode_tracking_loss import (Sources,COHORTS,PAIRS,load_context,close_value,
    tortuosity_episodes,turnback_episodes,agarose_episodes,_selected_windows_for_va)
from scripts.simulate_tracking_gaps import (DEFAULT_SOURCE,build_trajectory,raw_fields,primary_window,
    track_code,verify_wall_baseline)
from scripts.simulate_incremental_tracking_loss import atomic_json,clean,checked
from src.analysis.episode_tracking_loss import GapIndex
from src.analysis.tracking_gap_incremental import (CONTRASTS,stable_rng,contrast_components,video_bootstrap_weights)
from src.analysis.wall_exclusion_tracking_loss import METRICS,POLICIES,classify,reduce_cell
from src.exporting.wall_contact_episode_filter import episode_overlaps_wall_contact
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window


def job(key,frozen,episodes,source_dir,snapshot,signature,checkpoint):
    cohort,video,pair=key
    sources=Sources(ROOT); sources.manifest=snapshot; sources.verify_unchanged()
    with redirect_stdout(io.StringIO()):
        context=load_context(video,pair,'exp',include_analysis=True)
        va,trj=build_trajectory(context)
        verified_wall=verify_wall_baseline(context,trj,sources,cohort,pair)
        source=sources.agarose(cohort)
        gaps=GapIndex(trj.nan,context['fps']); base=dict(px_per_mm=context['px_per_mm'])
        window=va.sync_bucket_ranges[1][1:5]
        native=tortuosity_episodes(va,trj,base,gaps,window)
        selected=_selected_windows_for_va(va,[1],skip_first=1,keep_first=4,last_sync_buckets=0)
        tbwindow=selected[0]['bucket_ranges'] if selected else []
        for inner,outer in PAIRS:
            ratio,alignment=turnback_episodes(va,trj,base,gaps,tbwindow,inner,outer)
            native.extend(ratio); native.extend(alignment)
        agarose,_=agarose_episodes(va,trj,base,gaps,source)
        for rows in agarose.values(): native.extend(rows)
    native=pd.DataFrame(native)
    keys=['metric','variant','period','start_frame','stop_frame_exclusive']
    if len(episodes):
        a=episodes.sort_values(keys).reset_index(drop=True)
        b=native.sort_values(keys).reset_index(drop=True)
        if len(a)!=len(b) or not a[keys].equals(b[keys]):
            raise ValueError(f'Native episode identities differ: {key}')
        np.testing.assert_allclose(a.value,b.value,rtol=1e-9,atol=1e-10)
        for field in ['support_start_frame','support_stop_frame_exclusive','episode_lost_frames','support_lost_frames','heading_missing_frames']:
            np.testing.assert_array_equal(a[field],b[field])
    regions=trj.boundary_event_stats['wall']['all']['edge']['boundary_contact_regions']
    contact=build_wall_contact_mask_for_window(va,0,fi=0,n_frames=len(trj.nan),enabled=True,
        warned_missing_wc=[False],log_tag='wall-exclusion-loss-audit')
    if contact is None: raise ValueError(f'Missing wall mask: {key}')
    raw=raw_fields(context)
    observed=np.isfinite(raw['x']) & np.isfinite(raw['y']) & ~trj.nan
    (left,top),(right,bottom)=context['floor']
    distance=np.minimum.reduce([raw['x']-left,right-raw['x'],raw['y']-top,bottom-raw['y']])/context['px_per_mm']
    near_wall=observed & (distance<=1.)
    near_gap=np.zeros(len(observed),dtype=bool)
    for start,stop in zip(gaps.starts,gaps.stops):
        if (start>0 and near_wall[start-1]) or (stop<len(observed) and near_wall[stop]):
            near_gap[start:stop]=True
    augmented=episodes.copy()
    measured_window=(window[0][0],window[-1][1]) if window else None
    for name,values in classify(augmented,contact,observed,near_wall,near_gap,measured_window).items():
        augmented[name]=values
    # Shared production overlap helper independently confirms primary bout rules.
    for r in augmented[augmented.metric.ne('tortuosity')].itertuples():
        if bool(r.wall_contact_frames) != episode_overlaps_wall_contact(dict(start=r.start_frame,stop=r.stop_frame_exclusive),regions):
            raise ValueError('Shared wall-filter mismatch')
    if window:
        lo,hi=window[0][0],window[-1][1]
        expected={(r.start_frame,r.stop_frame_exclusive) for r in augmented[
            augmented.metric.eq('tortuosity') & augmented.wall_contact_frames.eq(0)].itertuples()}
        finite={(r.start_frame,r.stop_frame_exclusive) for r in augmented[augmented.metric.eq('tortuosity')].itertuples()}
        with redirect_stdout(io.StringIO()):
            actual={(seg.s,seg.e) for seg in va._iter_between_reward_segment_com(
                va.trns[1],0,fi=lo,df=window[0][1]-lo,n_buckets=len(window),complete=[True]*len(window),
                relative_to_reward=True,per_segment_min_meddist_mm=0.,exclude_wall=True,wc=contact[lo:hi],
                exclude_nonwalk=False,min_walk_frames=2,dist_stats=(),exclude_reward_endpoints=False,
                debug=False,yield_skips=False) if (seg.s,seg.e) in finite}
        if actual!=expected: raise ValueError('Production tortuosity wall-option mismatch')
    groups={k:g for k,g in augmented.groupby(['metric','variant','period'])}
    subjects=[]
    for row in frozen.to_dict('records'):
        group=groups.get((row['metric'],row['variant'],row['period']),augmented.iloc[:0])
        before=reduce_cell(group,'wall_allowed',int(row['minimum_episodes']))
        close_value(before['n_episodes'],row['n_episodes'],f'{key} count')
        close_value(before['episode_value_sum'],row['episode_value_sum'],f'{key} sum')
        close_value(before['measured_value'],row['measured_value'],f'{key} measured baseline')
        for policy in POLICIES:
            result=reduce_cell(group,policy,int(row['minimum_episodes']))
            included=bool(row['source_gate_passed'] and np.isfinite(result['measured_value']))
            subjects.append(dict(cohort=cohort,video_path=video,pair=pair,metric=row['metric'],variant=row['variant'],
                period=row['period'],policy=policy,source_gate_passed=bool(row['source_gate_passed']),
                minimum_episodes=int(row['minimum_episodes']),baseline_eligible=bool(row['included']),
                baseline_n_episodes=int(row['n_episodes']),baseline_value=row['value'],
                baseline_exposed_episodes=before['exposed_episodes'],**result,included=included,
                value=result['measured_value'] if included else np.nan,wall_export_intervals_verified=verified_wall))
    sources.verify_unchanged()
    result=clean(dict(subjects=subjects,episodes=augmented.to_dict('records')))
    atomic_json(checkpoint,dict(signature=signature,**result))
    return result


def summarize(subjects,draws,seed):
    summaries=[]; contrasts=[]
    for key,g in subjects.groupby(['metric','variant','period'],sort=True):
        base=g[g.policy.eq('wall_allowed')]
        for policy in POLICIES:
            after=g[g.policy.eq(policy)]
            for cohort in COHORTS:
                b=base[base.cohort.eq(cohort) & base.included]
                a=after[after.cohort.eq(cohort) & after.included]
                joined=b.merge(a,on=['video_path','pair'],suffixes=('_before','_after'))
                # Before minimum-count attrition: original baseline-included flies.
                a_fixed=after.merge(b[['video_path','pair']],on=['video_path','pair'])
                def exposure(frame,num='exposed_episodes',den='n_episodes'):
                    return frame[num].sum()/frame[den].sum() if frame[den].sum() else np.nan
                summaries.append(dict(metric=key[0],variant=key[1],period=key[2],policy=policy,cohort=cohort,
                    baseline_flies=len(b),retained_flies=len(a),common_flies=len(joined),
                    baseline_videos=b.video_path.nunique(),retained_videos=a.video_path.nunique(),
                    baseline_episodes=int(b.n_episodes.sum()),episodes_after_wall_filter_before_attrition=int(a_fixed.n_episodes.sum()),
                    retained_episodes=int(a.n_episodes.sum()),baseline_exposure=exposure(b),retained_exposure=exposure(a),
                    exposure_before_fly_attrition=exposure(a_fixed),
                    common_fly_baseline_exposure=exposure(joined,'exposed_episodes_before','n_episodes_before'),
                    common_fly_after_exposure=exposure(joined,'exposed_episodes_after','n_episodes_after'),
                    baseline_mean=b.value.mean(),retained_mean=a.value.mean(),
                    common_fly_change=(joined.value_after-joined.value_before).mean(),
                    retained_episodes_with_observed_wall_proximity=int(a.episodes_with_observed_wall_proximity.sum()),
                    retained_exposed_episodes_with_near_wall_gap_endpoint=int(a.exposed_episodes_with_near_wall_gap_endpoint.sum())))
            if policy=='wall_allowed': continue
            pairs=base.merge(after,on=['cohort','video_path','pair','metric','variant','period'],suffixes=('_before','_after'))
            pairs=pairs[pairs.included_before]
            records=[dict(cohort=r.cohort,video_path=r.video_path,repeat=0,before_value=r.value_before,
                after_value=r.value_after,after_eligible=bool(r.included_after)) for r in pairs.itertuples()]
            rng=stable_rng(seed,*key,policy)
            weights=[video_bootstrap_weights(records,rng) for _ in range(draws)]
            for positive,negative in CONTRASTS:
                point=contrast_components(records,positive,negative)
                boot=np.asarray([contrast_components(records,positive,negative,w)['total_change'] for w in weights])
                valid=boot[np.isfinite(boot)]
                low,high=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
                contrasts.append(dict(metric=key[0],variant=key[1],period=key[2],policy=policy,
                    contrast=f'{positive} - {negative}',**point,
                    relative_change=point['total_change']/abs(point['before_contrast']) if abs(point['before_contrast'])>1e-12 else np.nan,
                    direction_preserved=(bool(np.sign(point['before_contrast'])==np.sign(point['after_contrast']))
                                         if np.isfinite(point['before_contrast']) and np.isfinite(point['after_contrast']) else None),
                    change_ci_low=low,change_ci_high=high,defined_bootstrap_draws=len(valid)))
    return pd.DataFrame(summaries),pd.DataFrame(contrasts)


def report(out,summary,contrasts):
    labels={'tortuosity':'Tortuosity','turnback_ratio':'Turnback ratio','home_vector_alignment':'Home-vector alignment','agarose_avoidance':'Agarose avoidance'}
    main=summary[summary.policy.eq('exclude_episode_contact') & (summary.period.eq('T2_SB2-5')|summary.period.eq('T2_SB5'))]
    rows=[]
    for r in main.to_dict('records'):
        rows.append({'Metric':labels[r['metric']],'Variant':r['variant'],'Window':r['period'],'Cohort':COHORTS[r['cohort']],
            'Flies: allowed → excluded':f"{r['baseline_flies']} → {r['retained_flies']}",
            'Episodes: allowed → excluded':f"{r['baseline_episodes']} → {r['retained_episodes']}",
            'Loss-exposed episodes: allowed':f"{100*r['baseline_exposure']:.2f}%",
            'Loss-exposed episodes: excluded':f"{100*r['retained_exposure']:.2f}%" if np.isfinite(r['retained_exposure']) else 'undefined',
            'Same flies: exposure before → after':f"{100*r['common_fly_baseline_exposure']:.2f}% → {100*r['common_fly_after_exposure']:.2f}%",
            'Mean: allowed → excluded':f"{r['baseline_mean']:.5g} → {r['retained_mean']:.5g}",
            'Same-fly mean change':r['common_fly_change']})
    compact=pd.DataFrame(rows); compact.to_csv(out/'comparison_table.csv',index=False)
    def table(frame): return frame.to_html(index=False,escape=True,na_rep='undefined',float_format=lambda x:f'{x:.5g}')
    primary=contrasts[contrasts.policy.eq('exclude_episode_contact') & (contrasts.period.eq('T2_SB2-5')|contrasts.period.eq('T2_SB5'))]
    doc='''<!doctype html><meta charset="utf-8"><title>Wall exclusion and tracking-loss exposure</title><style>body{font:15px/1.6 system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccc}th{background:#edf2f5}section{overflow-x:auto}p{max-width:1100px}</style>
<h1>Does excluding wall-contact episodes reduce tracking-loss exposure?</h1>
<p>This compares the current metric with a measurement using whole episodes without detected wall contact. It introduces no artificial gaps and does not delete or join individual trajectory frames. It covers the existing Ctrl, PFNd&gt;Kir and antennae-removed flat-large-chamber sources for tortuosity, turnback ratio, home-vector alignment and agarose avoidance. Wall-contact counts and wall-positive path length are excluded because removing wall contact makes them trivial or undefined. Agarose chamber comparisons and AUC effects are outside this run.</p>
<p>Recording-admission and metric-specific SLI gates remain fixed; agarose retains its separate original source pool. Five qualifying episodes are required over the original production window after filtering. Turnback and alignment retain each of 3/5, 8/10 and 13/15 mm. Agarose retains every original Pre/T1/T2 cell; the overview shows T2 SB5 and the full tables contain all cells.</p>
<h2>Definitions and validation</h2>
<p><b>Primary wall exclusion:</b> exclude an entire episode if a production wall/all/edge contact region overlaps its half-open measured interval. Tortuosity follows the existing option, testing contact only within the segment's overlap with the pooled window. Turnback and alignment match the shared production episode-wall helper. Agarose uses this same whole-episode rule as a new, explicitly defined audit filter; its existing wall-facing-entry options are different behavioral filters. Missing wall classifications stop the run rather than silently leaving a fly unfiltered.</p>
<p><b>Loss exposure:</b> an episode touches originally missing x positions anywhere in its full measurement support (episode plus adjacent boundary frames, and sampled heading frames for alignment). The wall-contact criterion and loss-support criterion intentionally have different bounds to preserve existing wall-option semantics. A stricter secondary policy excludes contact anywhere in this support interval. All original episodes, values and support-loss counts are independently recomputed and checked against the reviewed audit; production wall counts/lengths are checked against the frozen interval export.</p>
<h2>1. Cohort measurements and actual-loss exposure</h2>
<p>Exposure percentages pool qualifying episodes across flies; cohort means weight eligible flies equally. “Excluded” means after wall filtering and reapplying the episode minimum. The same-fly columns restrict both conditions to identical contributing flies. Complete CSVs also retain exposure before fly attrition, allowing the effect of episode selection to be separated from subject dropout. Smaller exposure alone is not proof of greater accuracy.</p>'''
    doc+='<section>'+table(compact)+'</section><h2>2. Cohort contrasts: direction, magnitude and composition</h2><p>Positive minus negative is explicit. Total change equals change among common flies plus composition change. Percentile intervals resample whole videos within cohort with all their flies together. Relative changes divide by absolute baseline contrast and become unstable near zero. These intervals describe this selection analysis, not hidden-path uncertainty.</p><section>'+table(primary)+'</section>'
    doc+='<h2>3. Residual wall uncertainty and stricter exclusion</h2><p>“No detected wall contact” does not establish wall-free behavior: lost tracking can hide contact, and detector state may depend on preceding history. Proximity is reported separately using observed original positions within 1 mm of a wall; missing positions are never treated as observed evidence. The audit also records retained loss-exposed episodes whose gaps have a near-wall endpoint. These are warning indicators, not reconstructed paths or an alternate primary threshold.</p><section>'+table(summary[summary.policy.eq('exclude_support_contact') & (summary.period.eq('T2_SB2-5')|summary.period.eq('T2_SB5'))])+'</section>'
    doc+='<h2>4. Full tables</h2><p>Wall exclusion defines a different behavioral population. Preserved ordering supports a difference away from detected wall contacts; it does not validate the original missing paths or justify choosing a policy solely because it preserves a preferred result.</p><ul>'
    for name in ['comparison_table.csv','summary.csv','contrasts.csv','subjects.csv','episodes.csv','provenance.json']:
        doc+=f'<li><a href="{name}">{html.escape(name)}</a></li>'
    (out/'report.html').write_text(doc+'</ul>')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bootstrap-draws',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--limit-per-cohort',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if min(args.workers,args.bootstrap_draws)<1 or (args.limit_per_cohort is not None and args.limit_per_cohort<1): p.error('Counts must be positive')
    out=args.output_dir.resolve(); source_dir=args.source_dir.resolve()
    if args.resume:
        if not (out/'checkpoints').is_dir() or (out/'provenance.json').exists(): p.error('Resume needs incomplete checkpoints')
    elif out.exists() and any(out.iterdir()): p.error('Output directory must be empty')
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    sources=Sources(ROOT); track_code(sources)
    for path in [Path(__file__),ROOT/'src/analysis/wall_exclusion_tracking_loss.py',ROOT/'scripts/simulate_incremental_tracking_loss.py',
                 ROOT/'src/analysis/tracking_gap_incremental.py',ROOT/'src/exporting/wall_contact_episode_filter.py',
                 ROOT/'src/plotting/wall_contact_utils.py',ROOT/'src/analysis/episode_tracking_loss.py',
                 ROOT/'src/plotting/wall_contacts_per_reward_interval_totals.py']:
        sources.track(path.resolve())
    for name in ['manuscript_figure_panels.ipynb','graphpad_exports.ipynb']:
        sources.track(ROOT/'notebooks'/name)
    original=json.loads(sources.track(source_dir/'provenance.json').read_text())['sources']
    frozen=pd.read_csv(sources.track(source_dir/'subjects.csv'))
    frozen=frozen[frozen.policy.eq('baseline') & frozen.metric.isin(METRICS)]
    episodes=pd.read_csv(sources.track(source_dir/'episodes.csv'))
    episodes=episodes[episodes.metric.isin(METRICS)]
    # Verify all frozen measurement/eligibility dependencies, excluding code
    # and the separately documented historical genotype-list correction.
    for name in original:
        path=Path(name)
        if path.parent==ROOT/'exports/manuscript': checked(sources,path,original)
    for cohort in COHORTS:
        sources.agarose(cohort)
    for name in list(sources.manifest):
        path=Path(name)
        if path.parent==ROOT/'exports': checked(sources,path,original)
    for name in original:
        path=Path(name)
        if path.is_relative_to(ROOT) and path.suffix in ('.py','.pyx','.so'):
            sources.track(path)
    settings=dict(source_dir=str(source_dir),draws=args.bootstrap_draws,seed=args.seed,
        limit_per_cohort=args.limit_per_cohort,policies=list(POLICIES),observed_proximity_mm=1.)
    fixed={k:v['sha256'] for k,v in sources.manifest.items()}
    groups=[((c,v,int(f)),g) for (c,v,f),g in frozen.groupby(['cohort','video_path','pair'])]
    if args.limit_per_cohort:
        groups=[(k,g) for c in COHORTS for k,g in [(k,g) for k,g in groups if k[0]==c][:args.limit_per_cohort]]
    jobs=[]; results=[]; metadata={}
    for key,g in groups:
        cohort,video,pair=key
        for suffix in ['.trx','.data']: checked(sources,Path(video).with_suffix(suffix),original)
        note=Path(video).with_name(Path(video).stem+'__note.txt')
        if note.exists()!=(str(note) in original): raise ValueError(f'Note availability changed: {note}')
        if note.exists(): checked(sources,note,original)
        stat=Path(video).stat(); metadata[video]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        sig=dict(identity=list(key),settings=settings,fixed=fixed,video_metadata=metadata[video],
            raw={str(Path(video).with_suffix(s)):sources.manifest[str(Path(video).with_suffix(s))]['sha256'] for s in ['.trx','.data']})
        name=hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()[:20]
        checkpoint=out/'checkpoints'/f'{name}.json'
        if checkpoint.exists():
            result=json.loads(checkpoint.read_text())
            if result['signature']!=sig: raise ValueError('Checkpoint signature changed')
            results.append(result); continue
        ep=episodes[episodes.cohort.eq(cohort)&episodes.video_path.eq(video)&episodes.pair.eq(pair)]
        jobs.append((key,g,ep,source_dir,dict(sources.manifest),sig,checkpoint))
    print(f'Auditing {len(groups)} flies; {len(jobs)} jobs with {args.workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(job,*j):j[0] for j in jobs}
        for index,future in enumerate(as_completed(pending),1):
            result=future.result(); results.append(result)
            print(f'{index}/{len(jobs)} completed {pending[future]}',flush=True)
    subjects=pd.DataFrame([r for result in results for r in result['subjects']])
    annotated=pd.DataFrame([r for result in results for r in result['episodes']])
    sort=['cohort','video_path','pair','metric','variant','period']
    subjects=subjects.sort_values(sort+['policy']).reset_index(drop=True)
    annotated=annotated.sort_values(sort+['start_frame','stop_frame_exclusive']).reset_index(drop=True)
    for field in ['baseline_value','measured_value','value']: subjects[field]=pd.to_numeric(subjects[field])
    print('Summarizing exposure, eligibility and video-bootstrap contrasts',flush=True)
    summary,contrasts=summarize(subjects,args.bootstrap_draws,args.seed)
    sources.verify_unchanged()
    for video,snapshot in metadata.items():
        stat=Path(video).stat()
        if snapshot!=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns): raise ValueError('Video metadata changed')
    for name,data in [('subjects',subjects),('episodes',annotated),('summary',summary),('contrasts',contrasts)]: data.to_csv(out/f'{name}.csv',index=False)
    report(out,summary,contrasts)
    atomic_json(out/'provenance.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),settings=settings,
        sources=sources.manifest,video_metadata=metadata,workers=args.workers,subjects=len(groups),
        scope='Four metric families; existing flat-chamber sources; actual tracking loss; no synthetic gaps'))
    print(f'Wrote {out}',flush=True)


if __name__=='__main__': main()
