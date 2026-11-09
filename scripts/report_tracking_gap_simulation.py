#!/usr/bin/env python3
"""Render empirical gap distributions and conditional pilot coverage/error summaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COHORTS = {"ctrl":"Control","pfnd_kir":"PFNd>Kir","antennae_removed":"Antennae removed"}
BINS = ["<=0.5s","0.5-2s","2-10s",">10s"]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir',type=Path,required=True)
    parser.add_argument('--pilot-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be a new or empty directory')
    out.mkdir(parents=True,exist_ok=True)
    paths=[args.profile_dir/n for n in ['gaps.csv','summary.csv','subjects.csv','missing_fields.csv','provenance.json']]
    paths += [args.pilot_dir/n for n in ['trials.csv','redetection.csv','fixed_boundaries.csv','baseline_checks.csv','provenance.json']]
    inputs={str(p):sha256(p) for p in paths}
    gaps=pd.read_csv(args.profile_dir/'gaps.csv')
    profile=pd.read_csv(args.profile_dir/'summary.csv')
    trials=pd.read_csv(args.pilot_dir/'trials.csv')
    redetect=pd.read_csv(args.pilot_dir/'redetection.csv')
    fixed=pd.read_csv(args.pilot_dir/'fixed_boundaries.csv')
    fields=pd.read_csv(args.profile_dir/'missing_fields.csv')
    if 'placement_arm' not in trials:
        trials['placement_arm']='spatial_matched'
    trial_columns=['trial_id','placement_arm','video_path','pair']
    redetect=redetect.merge(trials[trial_columns],on='trial_id',validate='many_to_one')
    fixed=fixed.merge(trials[trial_columns],on='trial_id',validate='many_to_one')
    profile_rows=[]
    for cohort in COHORTS:
        all_g=gaps[gaps.cohort==cohort].sort_values(['video_path','pair','gap_start_frame'])
        primary=all_g[all_g.primary_donor]
        spacing=[]
        for _,g in all_g.groupby(['video_path','pair']):
            spacing.extend((g.gap_start_frame.to_numpy()[1:]-g.gap_stop_frame_exclusive.to_numpy()[:-1])/g.fps.to_numpy()[1:])
        long=primary[primary.duration_s>10]
        per_fly=primary.groupby(['video_path','pair']).duration_s.median()
        profile_rows.append(dict(cohort=cohort,t2_gap_count=len(all_g),primary_donor_gaps=len(primary),
            pooled_gap_median_s=float(primary.duration_s.median()),pooled_gap_p95_s=float(primary.duration_s.quantile(.95)),
            median_of_fly_median_gap_s=float(per_fly.median()),
            gaps_over_10s_fraction=len(long)/len(primary),
            loss_time_over_10s_fraction=long.gap_frames.sum()/primary.gap_frames.sum(),
            endpoint_near_wall_1mm_fraction=float(((primary.before_wall_mm<=1)|(primary.after_wall_mm<=1)).mean()),
            next_gap_within_1s_fraction=float(np.mean(np.asarray(spacing)<=1)) if spacing else np.nan))
    pd.DataFrame(profile_rows).to_csv(out/'real_gap_summary.csv',index=False)
    coverage=trials.groupby(['cohort','placement_arm','duration_bin'],sort=False).agg(
        sampled_attempts=('trial_id','size'),matched_attempts=('status',lambda x:int((x=='matched').sum()))).reset_index()
    coverage.to_csv(out/'matching_coverage.csv',index=False)
    effect_rows=[]
    for (cohort,arm,metric,variant),r in redetect.groupby(['cohort','placement_arm','metric','variant']):
        evaluated=r[(r.reference_episodes>0)|(r.simulated_episodes>0)]
        f=fixed[(fixed.cohort==cohort)&(fixed.placement_arm==arm)&(fixed.metric==metric)&(fixed.variant==variant)&fixed.support_intersects_injected_gap]
        available=f[f.available]
        effect_rows.append(dict(cohort=cohort,placement_arm=arm,metric=metric,variant=variant,
            matched_trials=r.trial_id.nunique(),trials_with_evaluable_episodes=len(evaluated),
            trials_with_count_change=int((evaluated.episode_count_change!=0).sum()),
            reference_episodes=int(evaluated.reference_episodes.sum()),simulated_episodes=int(evaluated.simulated_episodes.sum()),
            reference_without_overlap=int(evaluated.reference_without_overlap.sum()),simulated_without_overlap=int(evaluated.simulated_without_overlap.sum()),
            one_to_one_pairs=int(evaluated.one_to_one_overlap_pairs.sum()),
            affected_fixed_episode_measurements=len(f),unavailable_fixed_measurements=len(f)-len(available),
            fixed_mean_error=float(available.error.mean()) if len(available) else np.nan,
            fixed_median_absolute_error=float(available.error.abs().median()) if len(available) else np.nan))
    effects=pd.DataFrame(effect_rows)
    effects.to_csv(out/'pilot_effects.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for cohort,label in COHORTS.items():
        g=gaps[(gaps.cohort==cohort)&gaps.primary_donor]
        xx=np.sort(g.duration_s)
        axes[0].step(xx,np.arange(1,len(xx)+1)/len(xx),where='post',label=label)
        s=profile[(profile.cohort==cohort)&(profile.scope=='T2_SB2-5_donors')].set_index('duration_bin').loc[BINS]
        axes[1].plot(BINS,s.lost_time_fraction,marker='o',label=label)
        attempted=trials[(trials.cohort==cohort)&(trials.placement_arm=='spatial_matched')]
        cs=attempted.groupby('duration_bin').status.agg(lambda x:float((x=='matched').mean())).reindex(BINS)
        axes[2].plot(BINS,cs,marker='o',label=label)
    axes[0].set_xscale('log');axes[0].set_xlabel('Real gap duration (s)');axes[0].set_ylabel('Cumulative fraction of gaps')
    axes[1].set_ylabel('Fraction of lost time');axes[2].set_ylabel('Fraction of sampled donors matched')
    for a,title in zip(axes,['Real gaps: T2 SB2–5','Long gaps contribute substantial lost time','Pilot support: spatial matching']):
        a.set_title(title);a.set_ylim(0,1.02);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'gap_profile.png',dpi=180);fig.savefig(out/'gap_profile.pdf');plt.close(fig)
    concise=effects[effects.placement_arm==('episode_targeted:'+effects.metric)]
    shown=concise[['cohort','metric','variant','matched_trials','trials_with_evaluable_episodes','trials_with_count_change',
                   'affected_fixed_episode_measurements','unavailable_fixed_measurements','fixed_median_absolute_error']]
    field_missing_ok=bool((fields.finite_values_at_lost_frames==0).all() and (fields.nonfinite_values_at_tracked_frames==0).all())
    text=["# Empirical tracking gaps and artificial-gap pilot", "",
        "This is a conditional pilot, not a population error-rate estimate or a correction to manuscript means. Subjects span recording-loss strata; donor gaps are sampled within duration bins. Repeated episodes and trials are not independent replicates.", "",
        f"Full profile: {len(gaps):,} T2-intersecting gaps, {int(gaps.primary_donor.sum()):,} bracketed T2 SB2–5 donors. Tracker field audit consistent across all subjects: {field_missing_ok} (x/y/width/height/orientation missing together).", "",
        f"Pilot: {trials[['video_path','pair']].drop_duplicates().shape[0]} subjects; {len(trials)} donor/placement-arm attempts; {int((trials.status=='matched').sum())} matched. A spatial match requires the same fly, exact duration, matching before/after wall bins, and reward distance, agarose distance, and endpoint chord within the declared tolerance. No shorter-gap or cross-cohort fallback.", "",
        "The episode_targeted arms add the requirement that the artificial gap intersect a fully observed reference episode's support. These arms test susceptibility conditional on exposure. Zero matches mean unsupported context, not zero error. Redetection uses the same observed run before and after masking, with incomplete boundary episodes excluded consistently.", "",
        "The main pilot table below shows targeted arms only. Fixed-boundary errors are restricted to reference episodes whose support intersects the artificial gap. Ratio outcomes are evaluated through redetection. Numeric errors are in original metric units. Undefined errors indicate no available affected measurement.", "",
        "Both short-term measurement/detection effects and missing measurement availability are reported. Single-gap trials do not reproduce clustered real gaps, feedback changes in reward delivery, or total recording loss. Fully tracked stretches may differ systematically from genuine loss periods; rare/long unmatched contexts require more data or retracking/manual review.", ""]
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    markup='<!doctype html><meta charset="utf-8"><title>Tracking-gap simulation pilot</title><style>body{font:15px/1.6 system-ui;margin:35px auto;max-width:1450px;padding:20px}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd;padding:7px;text-align:left}img{max-width:100%}</style>'
    markup+='<h1>Empirical tracking gaps and artificial-gap pilot</h1><pre>'+html.escape('\n\n'.join(text[2:]))+'</pre><img src="gap_profile.png">'
    markup+='<h2>Real gap characteristics</h2>'+pd.DataFrame(profile_rows).to_html(index=False,float_format=lambda v:f'{v:.4g}')
    markup+='<h2>Matching coverage by arm and duration</h2>'+coverage.to_html(index=False)
    markup+='<h2>Conditional targeted tests</h2>'+shown.to_html(index=False,float_format=lambda v:f'{v:.4g}',na_rep='undefined')
    markup+='<p><a href="pilot_effects.csv">Full effects table, including spatial-only placement</a> · <a href="real_gap_summary.csv">Gap summary</a> · <a href="matching_coverage.csv">Matching coverage</a></p>'
    (out/'report.html').write_text(markup)
    (out/'provenance.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=inputs,report_code_sha256=sha256(__file__)),indent=2)+'\n')
    print(out/'report.html')


if __name__=='__main__':
    main()
