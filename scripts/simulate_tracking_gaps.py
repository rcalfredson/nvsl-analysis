#!/usr/bin/env python3
"""Characterize real gaps and run a matched, single-gap counterfactual pilot.

The pilot measures conditional errors in fully observed stretches. It does not
estimate a correction to manuscript means or reproduce whole-recording loss.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import html
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.audit_episode_tracking_loss import (
    COHORTS, PAIRS, Sources, read_csv, write_csv, sha256, load_context,
    tortuosity_episodes, turnback_episodes, close_value,
)
from src.analysis.episode_tracking_loss import GapIndex
from src.analysis.tracking_gap_simulation import (
    true_runs, duration_bin, candidate_starts, endpoint_features,
    matching_candidates, compare_episodes, exposed_candidates,
)
from src.analysis.trajectory import Trajectory
from src.analysis.boundary_contact import EllipseToBoundaryDistCalculator
from src.plotting.between_reward_segment_metrics import tortuosity_metric_masked
from src.plotting.wall_contacts_per_reward_interval_totals import WallContactsPerRewardIntervalPerFlyCollector as WallCollector
from src.exporting.turnback_home_vector_alignment_sli_bundle import episode_home_vector_alignment
from src.utils.constants import AGAROSE_BOUNDARY_DIST, CONTACT_BUFFER_OFFSETS

FIELDS = ("x","y","w","h","theta")
BINS = ("<=0.5s","0.5-2s","2-10s",">10s")
DEFAULT_SOURCE = ROOT/"results/episode_tracking_loss/flat_large_2026-11-08_reviewed"


def empty_folder(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True,exist_ok=True)
    return path


def provenance(sources, **settings):
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
                python_version=sys.version,numpy_version=np.__version__,settings=settings,
                sources=sources.manifest)


def track_code(sources):
    from src.analysis import boundary_contact
    for path in (Path(__file__),ROOT/"src/analysis/tracking_gap_simulation.py",
                 ROOT/"scripts/audit_episode_tracking_loss.py",ROOT/"scripts/audit_tracking_gap_endpoints.py",
                 ROOT/"src/analysis/trajectory.py",ROOT/"src/analysis/video_analysis.py",
                 ROOT/"src/analysis/boundary_contact.pyx",Path(boundary_contact.__file__),
                 ROOT/"src/analysis/boundary_contact_frame_policy.py",ROOT/"src/utils/common.py",
                 ROOT/"src/utils/constants.py",ROOT/"src/plotting/between_reward_segment_metrics.py",
                 ROOT/"src/exporting/turnback_home_vector_alignment_sli_bundle.py"):
        sources.track(path.resolve())


def raw_fields(context):
    va = context["analysis"]
    return {f:np.asarray(va.trxRw[f][context["physical_id"]],dtype=float).copy() for f in FIELDS}


def geometry(context):
    va = context["analysis"]
    return dict(floor=context["floor"],scale=context["px_per_mm"],
                reward_center=va.trns[1].circles(0)[0][:2],
                wells=va.ct.arenaWells(va.xf,context["physical_id"]))


def primary_window(context):
    rr = context["geometries"][1]["sync_buckets"]
    return (int(rr[1][0]),int(rr[4][1])) if len(rr)>=5 else None


def subject_key(row):
    return row["cohort"],row["video_path"],int(row["pair"])


def profile(args):
    folder = empty_folder(args.output_dir)
    sources = Sources(ROOT)
    track_code(sources)
    old_manifest = json.loads(sources.track(args.source_dir/"provenance.json").read_text())
    baseline = read_csv(sources.track(args.source_dir/"subjects.csv"))
    roster = {}
    for row in baseline:
        if row["policy"]=="baseline":
            roster.setdefault(subject_key(row),row)
    all_gaps,subjects,runs_table,field_rows = [],[],[],[]
    for index,(key,row) in enumerate(roster.items(),1):
        cohort,video,pair = key
        print(f"Profile {index}/{len(roster)}: {cohort} {Path(video).name} fly {pair}",flush=True)
        for suffix in (".trx",".data"):
            path = sources.track(Path(video).with_suffix(suffix))
            if sources.manifest[str(path)]["sha256"] != old_manifest["sources"][str(path)]["sha256"]:
                raise ValueError(f"Raw source differs from component one: {path}")
        with redirect_stdout(io.StringIO()):
            context = load_context(video,pair,"exp",include_analysis=True)
        raw = raw_fields(context)
        missing = np.isnan(raw["x"])
        observed = np.logical_and.reduce([np.isfinite(raw[f]) for f in FIELDS])
        t2 = context["geometries"][1]
        lo,hi = t2["training_start_frame"],t2["training_stop_frame"]
        window = primary_window(context)
        runs = true_runs(observed,*window) if window else []
        base = dict(cohort=cohort,video_path=video,pair=pair,physical_tracking_id=context["physical_id"],
                    fps=context["fps"],px_per_mm=context["px_per_mm"])
        count = 0
        for gap_id,(start,stop) in enumerate(true_runs(missing)):
            if start>=hi or stop<=lo:
                continue
            bracketed = start>0 and stop<len(missing)
            within = bool(window and window[0]<=start-1 and stop<window[1])
            finite_endpoints = bool(bracketed and np.isfinite([raw["x"][start-1],raw["y"][start-1],raw["x"][stop],raw["y"][stop]]).all())
            duration = (stop-start)/context["fps"]
            rec = dict(**base,gap_id=int(gap_id),gap_start_frame=int(start),gap_stop_frame_exclusive=int(stop),
                       gap_frames=int(stop-start),duration_s=float(duration),duration_bin=duration_bin(duration),
                       t2_overlap_frames=int(min(stop,hi)-max(start,lo)),bracketed=bool(bracketed),
                       primary_donor=bool(within and finite_endpoints))
            names = ("before_wall_mm","before_wall_bin","before_reward_radius_mm","before_well_signed_mm",
                     "after_wall_mm","after_wall_bin","after_reward_radius_mm","after_well_signed_mm","endpoint_chord_mm")
            feats = endpoint_features(raw["x"],raw["y"],[start],stop-start,**geometry(context)) if finite_endpoints else {}
            rec.update({name:float(feats[name][0]) if feats else np.nan for name in names})
            rec["length_supported_with_context"] = bool(runs and max(b-a for a,b in runs)>=stop-start+2*int(np.ceil(args.padding_seconds*context["fps"]))+2)
            all_gaps.append(rec)
            count += int(rec["primary_donor"])
        for run_id,(a,b) in enumerate(runs):
            runs_table.append(dict(**base,run_id=run_id,start_frame=int(a),stop_frame_exclusive=int(b),duration_s=float((b-a)/context["fps"])))
        lost_t2 = missing[lo:hi]
        for name in FIELDS:
            vals = raw[name][lo:hi]
            field_rows.append(dict(**base,field=name,lost_frames=int(lost_t2.sum()),
                                   finite_values_at_lost_frames=int(np.isfinite(vals[lost_t2]).sum()),
                                   tracked_frames=int((~lost_t2).sum()),nonfinite_values_at_tracked_frames=int((~np.isfinite(vals[~lost_t2])).sum())))
        subjects.append(dict(**base,recording_loss_pct=float(row["recording_loss_pct"]),
                             t2_start=lo,t2_stop=hi,t2_lost_fraction=float(lost_t2.mean()),
                             primary_window_complete=bool(window),primary_start=window[0] if window else "",primary_stop=window[1] if window else "",
                             primary_donor_gaps=count,fully_observed_runs=len(runs),
                             longest_fully_observed_run_s=max(((b-a)/context["fps"] for a,b in runs),default=0.)))
    summary = []
    for cohort in COHORTS:
        for scope in ("T2_intersecting","T2_SB2-5_donors"):
            eps = [g for g in all_gaps if g["cohort"]==cohort and (scope=="T2_intersecting" or g["primary_donor"])]
            durations = np.array([g["duration_s"] for g in eps])
            for bin_name in BINS:
                sub = [g for g in eps if g["duration_bin"]==bin_name]
                summary.append(dict(cohort=cohort,scope=scope,duration_bin=bin_name,gaps=len(sub),
                    gap_fraction=len(sub)/len(eps) if eps else np.nan,
                    lost_time_fraction=sum(g["gap_frames"] for g in sub)/sum(g["gap_frames"] for g in eps) if eps else np.nan,
                    length_supported_fraction=float(np.mean([g["length_supported_with_context"] for g in sub])) if sub else np.nan,
                    median_gap_s=float(np.median(durations)) if durations.size else np.nan,
                    p95_gap_s=float(np.quantile(durations,.95)) if durations.size else np.nan,
                    max_gap_s=float(max(durations)) if durations.size else np.nan))
    sources.verify_unchanged()
    write_csv(folder/"gaps.csv",all_gaps)
    write_csv(folder/"subjects.csv",subjects)
    write_csv(folder/"fully_observed_runs.csv",runs_table)
    write_csv(folder/"missing_fields.csv",field_rows)
    write_csv(folder/"summary.csv",summary)
    (folder/"provenance.json").write_text(json.dumps(provenance(sources,padding_seconds=args.padding_seconds,
        scope="T2; primary simulation donors fully bracketed in T2 SB2-5",source_dir=str(args.source_dir)),indent=2)+"\n")
    paragraphs = ["Real tracking-gap profile", "",f"{len(subjects)} experimental subjects; original source files agree with component one.",
        "Gap counts weight each gap equally. Lost-time fractions weight by gap duration. Length support is an optimistic upper bound: geometry matching can further reduce support.",
        "Simulation donors must have observed endpoints fully within T2 SB2–5. Terminal/unbracketed gaps are not simulated as interpolated chords. Fully observed runs require finite x, y, width, height, and orientation.", ""]
    for c in COHORTS:
        ss = [r for r in summary if r["cohort"]==c and r["scope"]=="T2_SB2-5_donors"]
        paragraphs.append(f'{COHORTS[c]}: {sum(r["gaps"] for r in ss)} donors; median {ss[0]["median_gap_s"]:.3g}s; 95th percentile {ss[0]["p95_gap_s"]:.3g}s.')
    paragraphs.extend(["", "Pilot design: stratify donor sampling by duration within each selected fly; preserve exact donor duration; match observed endpoints by wall-distance bins, reward-center distance, agarose-boundary distance, and endpoint chord. Unsupported donors remain explicit, with no silent fallback to shorter gaps or other cohorts.",
        "This is observational support for a conditional simulation, not evidence that the unobserved paths inside real gaps are known."])
    (folder/"report.md").write_text("\n\n".join(paragraphs)+"\n")
    (folder/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Real tracking-gap profile</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape("\n\n".join(paragraphs))+'</pre><p><a href="summary.csv">Duration/support summary</a> · <a href="missing_fields.csv">Missing-field audit</a></p>')
    print(f"Wrote {folder}")


def build_trajectory(context, injection=None):
    """Mask every tracker field, then call production coordinate/theta interpolation."""
    raw = raw_fields(context)
    if injection:
        a,b = injection
        if not 0<a<b<len(raw["x"]):
            raise ValueError("A synthetic gap must have two recording-internal endpoints")
        if not np.logical_and.reduce([np.isfinite(raw[f][a-1:b+1]) for f in FIELDS]).all():
            raise ValueError("Synthetic gap and bracketing endpoints must be fully observed")
        for values in raw.values():
            values[a:b] = np.nan
    trj = Trajectory.__new__(Trajectory)
    for name,values in raw.items():
        setattr(trj,name,values)
    va = context["analysis"]
    trj.f,trj.va,trj.opts = 0,va,va.opts
    trj._max_proportion_lost_fms = 1.
    trj.pxPerMmFloor = va.ct.pxPerMmFloor()
    trj._interpolate()
    trj._calcDistances()
    va.trx,va.trxf = [trj],(context["physical_id"],)
    va.sync_bucket_ranges = [g["sync_buckets"] for g in context["geometries"]]
    # This is the production wall/all/edge path, with no reads/writes to caches.
    calc = EllipseToBoundaryDistCalculator(trj,va)
    thresholds = CONTACT_BUFFER_OFFSETS["wall"]
    calc.calc_dist_boundary_to_ellipse(boundary_type="wall",boundary_combo="all",offset=AGAROSE_BOUNDARY_DIST,
        event_thresholds=[thresholds["min"],thresholds["max"]],ellipse_ref_pt="edge",exclude_lost_frames=True)
    trj.boundary_event_stats = calc.return_data["boundary_event_stats"]
    return va,trj


def native_records(va,trj,context, *, pooled_only=False):
    """Recompute all six measurements, retaining production episode boundaries."""
    base = dict(px_per_mm=context["px_per_mm"])
    gaps = GapIndex(trj.nan,context["fps"])
    window = va.sync_bucket_ranges[1][1:5]
    records = defaultdict(list)
    def add(metric,variant,start,stop,value,*,support_start=None,support_stop=None,ep=None):
        records[metric,variant].append(dict(start=int(start),stop=int(stop),value=float(value),
            support_start=int(start)-1 if support_start is None else int(support_start),
            support_stop=int(stop)+1 if support_stop is None else int(support_stop),episode=ep))
    for row in tortuosity_episodes(va,trj,base,gaps,window):
        add("tortuosity","full",row["start_frame"],row["stop_frame_exclusive"],row["value"])
    lo,hi = window[0][0],window[-1][1]
    rf = WallCollector._sanitize_reward_frames(va._getOn(va.trns[1],False,f=0),start=lo,stop=hi)
    regions = trj.boundary_event_stats["wall"]["all"]["edge"]["boundary_contact_regions"]
    counts = WallCollector._count_regions_by_reward_interval_start(regions,rf)
    for a,b,n in zip(rf[:-1],rf[1:],counts):
        add("wall_contacts","full",a,b,n)
        if n:
            add("wall_contact_length","full",a,b,trj.distTrav(int(a),int(b))/context["px_per_mm"])
    if pooled_only:
        return records
    for inner,outer in PAIRS:
        ratio,alignment = turnback_episodes(va,trj,base,gaps,window,inner,outer)
        eps = {(ep["start"],ep["stop"]):ep for ep in trj.reward_turnback_dual_circle_episodes_for_training(
            trn=va.trns[1],inner_radius_mm=inner,outer_radius_mm=outer,border_width_mm=.1)}
        for metric,rows in (("turnback_ratio",ratio),("home_vector_alignment",alignment)):
            for row in rows:
                add(metric,f"{inner}/{outer}mm",row["start_frame"],row["stop_frame_exclusive"],row["value"],
                    support_start=row["support_start_frame"],support_stop=row["support_stop_frame_exclusive"],
                    ep=eps[row["start_frame"],row["stop_frame_exclusive"]])
    trj.calc_agarose_dual_circle_episodes(delta_mm=1.,boundary_policy="hysteretic")
    for ep in trj.agarose_dual_circle_episodes:
        if lo<=ep["start"]<hi:
            add("agarose_avoidance","physical_1mm",ep["start"],ep["stop"],ep["avoids_inner"])
    return records


def within_run(records,start,stop):
    return {k:[r for r in rows if start<=r["support_start"] and r["support_stop"]<=stop] for k,rows in records.items()}


def fixed_values(reference,simulated,context):
    """Isolate numeric changes on reference boundaries; no invented fixed turnback labels."""
    output = []
    va = context["analysis"]
    center = va.trns[1].circles(0)[0][:2]
    regions = simulated.boundary_event_stats["wall"]["all"]["edge"]["boundary_contact_regions"]
    for (metric,variant),rows in reference.items():
        if metric not in ("tortuosity","wall_contacts","wall_contact_length","home_vector_alignment"):
            continue
        for row in rows:
            a,b = row["start"],row["stop"]
            if metric=="tortuosity":
                value = tortuosity_metric_masked(traj=simulated,s=a,e=b,fi=a,nonwalk_mask=None,
                    exclude_nonwalk=False,px_per_mm=context["px_per_mm"],reward_center_xy=center)
            elif metric=="wall_contacts":
                value = sum(a<=reg.start<b for reg in regions)
            elif metric=="wall_contact_length":
                value = simulated.distTrav(a,b)/context["px_per_mm"]
            else:
                value = episode_home_vector_alignment(simulated,va.trns[1],row["episode"],window_radius_frames=2,
                    heading_estimator="mean",home_vector_anchor="intersection",max_interpolated_heading_frames=1)
            output.append(dict(metric=metric,variant=variant,start_frame=a,stop_frame_exclusive=b,
                               support_start_frame=row["support_start"],support_stop_frame_exclusive=row["support_stop"],
                               reference_value=row["value"],simulated_value=float(value),
                               available=bool(np.isfinite(value)),error=float(value-row["value"])))
    return output


def verify_wall_baseline(context,trj,sources,cohort,pair):
    path = sources.track(ROOT/"exports/manuscript"/f"fig4k_{cohort}_wall_contacts_episodes.csv")
    rows = [r for r in read_csv(path) if r["video_path"]==context["video_path"] and int(r["va_tag"])==pair]
    if not rows:
        return 0
    rf = np.array([int(r["start_reward_frame"]) for r in rows]+[int(rows[-1]["end_reward_frame"])])
    regions = trj.boundary_event_stats["wall"]["all"]["edge"]["boundary_contact_regions"]
    actual = WallCollector._count_regions_by_reward_interval_start(regions,rf)
    np.testing.assert_array_equal(actual,[int(r["wall_contact_event_count"]) for r in rows])
    for r in rows:
        close_value(trj.distTrav(int(r["start_reward_frame"]),int(r["end_reward_frame"]))/context["px_per_mm"],
                    float(r["trajectory_length_mm"]),"wall baseline interval length")
    return len(rows)


def pilot(args):
    folder = empty_folder(args.output_dir)
    sources = Sources(ROOT)
    track_code(sources)
    prof_manifest = json.loads(sources.track(args.profile_dir/"provenance.json").read_text())
    subjects = read_csv(sources.track(args.profile_dir/"subjects.csv"))
    donors = read_csv(sources.track(args.profile_dir/"gaps.csv"))
    source_dir = Path(prof_manifest["settings"]["source_dir"])
    baseline_subjects = read_csv(sources.track(source_dir/"subjects.csv"))
    by_subject = defaultdict(list)
    for row in donors:
        if row["primary_donor"]=="True":
            by_subject[subject_key(row)].append(row)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for cohort in COHORTS:
        pool = sorted([s for s in subjects if s["cohort"]==cohort and by_subject[subject_key(s)]],key=lambda s:float(s["recording_loss_pct"]))
        seen_videos = set()
        for stratum,indices in enumerate(np.array_split(np.arange(len(pool)),args.subjects_per_cohort)):
            candidates = [pool[int(i)] for i in indices if pool[int(i)]["video_path"] not in seen_videos]
            if not candidates:
                continue
            row = dict(candidates[int(rng.integers(len(candidates)))],selection_loss_stratum=stratum)
            chosen.append(row)
            seen_videos.add(row["video_path"])
    write_csv(folder/"selected_subjects.csv",chosen)
    donor_rng = np.random.default_rng(args.seed+1)
    trials,comparisons,frozen,episode_audit,checks = [],[],[],[],[]
    for si,subject in enumerate(chosen,1):
        cohort,video,pair = subject_key(subject)
        print(f"Pilot {si}/{len(chosen)}: {cohort} {Path(video).name} fly {pair}",flush=True)
        for suffix in (".trx",".data"):
            path = sources.track(Path(video).with_suffix(suffix))
            if sources.manifest[str(path)]["sha256"] != prof_manifest["sources"][str(path)]["sha256"]:
                raise ValueError(f"Raw source differs from gap profile: {path}")
        with redirect_stdout(io.StringIO()):
            context = load_context(video,pair,"exp",include_analysis=True)
            va,reference_trj = build_trajectory(context)
            baseline_records = native_records(va,reference_trj,context)
        wall_checked = verify_wall_baseline(context,reference_trj,sources,cohort,pair)
        original = [r for r in baseline_subjects if subject_key(r)==(cohort,video,pair) and r["policy"]=="baseline"]
        cells = 0
        for row in original:
            if row["period"]!="T2_SB2-5":
                continue
            eps = baseline_records[row["metric"],row["variant"]]
            close_value(len(eps),int(row["baseline_n_episodes"]),"component-one episode count")
            value = float(np.mean([e["value"] for e in eps])) if len(eps)>=5 else np.nan
            close_value(value,float(row["measured_value"]),"component-one measurement")
            cells += 1
        agarose_rows = [r for r in original if r["metric"]=="agarose_avoidance" and r["period"] in ("T2_SB2","T2_SB3","T2_SB4","T2_SB5")]
        ae = baseline_records["agarose_avoidance","physical_1mm"]
        close_value(len(ae),sum(int(r["baseline_n_episodes"]) for r in agarose_rows),"agarose pooled count")
        close_value(sum(e["value"] for e in ae),sum(float(r["episode_value_sum"]) for r in agarose_rows),"agarose pooled numerator")
        checks.append(dict(cohort=cohort,video_path=video,pair=pair,wall_intervals_verified=wall_checked,
                           component_one_measurement_cells_verified=cells,agarose_pooled_counts_verified=True))
        raw = raw_fields(context)
        observed = np.logical_and.reduce([np.isfinite(raw[f]) for f in FIELDS])
        window = primary_window(context)
        runs = true_runs(observed,*window)
        padding = int(np.ceil(args.padding_seconds*context["fps"]))
        supports = {}
        for (metric,_variant),records in baseline_records.items():
            mask = supports.setdefault(metric,np.zeros(len(observed),dtype=bool))
            for record in records:
                a,b = record["support_start"],record["support_stop"]
                if window[0]<=a<b<=window[1] and observed[a:b].all():
                    mask[a:b] = True
        for bin_name in BINS:
            pool = [r for r in by_subject[subject_key(subject)] if r["duration_bin"]==bin_name]
            if not pool:
                continue
            selected = donor_rng.choice(len(pool),size=min(args.gaps_per_bin,len(pool)),replace=False)
            for sample in selected:
                donor = pool[int(sample)]
                length = int(donor["gap_frames"])
                starts,parents = candidate_starts(runs,length,padding)
                target = {name:float(donor[name]) for name in ("before_wall_mm","before_wall_bin","before_reward_radius_mm","before_well_signed_mm",
                    "after_wall_mm","after_wall_bin","after_reward_radius_mm","after_well_signed_mm","endpoint_chord_mm")}
                features = endpoint_features(raw["x"],raw["y"],starts,length,**geometry(context)) if starts.size else {name:np.array([]) for name in target}
                matched = matching_candidates(features,target,args.tolerance_mm)
                arms = [("spatial_matched",matched)]
                if args.episode_targeted:
                    for metric,mask in sorted(supports.items()):
                        eligible = exposed_candidates(starts,length,mask)
                        arms.append(("episode_targeted:"+metric,np.intersect1d(matched,eligible)))
                for placement_arm,arm_candidates in arms:
                    trial_id = f"trial_{len(trials)+1:04d}"
                    trial = dict(trial_id=trial_id,cohort=cohort,video_path=video,pair=pair,duration_bin=bin_name,
                        donor_gap_id=int(donor["gap_id"]),donor_start_frame=int(donor["gap_start_frame"]),
                        gap_frames=length,duration_s=float(donor["duration_s"]),donor_bin_population=len(pool),donor_bin_sampled=len(selected),
                        length_candidates=len(starts),spatial_matched_candidates=len(matched),matched_candidates=len(arm_candidates),
                        placement_arm=placement_arm,status="matched" if arm_candidates.size else "unsupported",
                        synthetic_start_frame="",synthetic_stop_frame_exclusive="",evaluation_start_frame="",evaluation_stop_frame_exclusive="")
                    for name,value in target.items():
                        trial["donor_"+name] = value
                        trial["synthetic_"+name] = np.nan
                    trials.append(trial)
                    if not arm_candidates.size:
                        continue
                    mi = int(rng.choice(arm_candidates))
                    start,stop = int(starts[mi]),int(starts[mi]+length)
                    run_start,run_stop = (int(v) for v in runs[int(parents[mi])])
                    trial.update(synthetic_start_frame=start,synthetic_stop_frame_exclusive=stop,
                                 evaluation_start_frame=run_start,evaluation_stop_frame_exclusive=run_stop)
                    for name in target:
                        trial["synthetic_"+name] = float(features[name][mi])
                    with redirect_stdout(io.StringIO()):
                        va,synthetic_trj = build_trajectory(context,(start,stop))
                        after_records = native_records(va,synthetic_trj,context)
                    reference = within_run(baseline_records,run_start,run_stop)
                    after = within_run(after_records,run_start,run_stop)
                    for metric,variant in sorted(set(reference)|set(after)):
                        before_eps,after_eps = reference.get((metric,variant),[]),after.get((metric,variant),[])
                        stats = compare_episodes(before_eps,after_eps)
                        bmean = float(np.mean([e["value"] for e in before_eps])) if before_eps else np.nan
                        amean = float(np.mean([e["value"] for e in after_eps])) if after_eps else np.nan
                        comparisons.append(dict(trial_id=trial_id,cohort=cohort,duration_bin=bin_name,metric=metric,variant=variant,
                            reference_mean_unfiltered=bmean,simulated_mean_unfiltered=amean,mean_change=amean-bmean,
                            reference_meets_five_episode_minimum=len(before_eps)>=5,simulated_meets_five_episode_minimum=len(after_eps)>=5,**stats))
                        for label,eps in (("reference",before_eps),("simulated",after_eps)):
                            for ep in eps:
                                episode_audit.append(dict(trial_id=trial_id,condition=label,metric=metric,variant=variant,
                                    start_frame=ep["start"],stop_frame_exclusive=ep["stop"],value=ep["value"]))
                    for rec in fixed_values(reference,synthetic_trj,context):
                        frozen.append(dict(trial_id=trial_id,cohort=cohort,duration_bin=bin_name,
                            support_intersects_injected_gap=bool(rec["support_start_frame"]<stop and rec["support_stop_frame_exclusive"]>start),**rec))
    sources.verify_unchanged()
    write_csv(folder/"trials.csv",trials)
    write_csv(folder/"redetection.csv",comparisons)
    write_csv(folder/"fixed_boundaries.csv",frozen)
    write_csv(folder/"episodes.csv",episode_audit)
    write_csv(folder/"baseline_checks.csv",checks)
    (folder/"provenance.json").write_text(json.dumps(provenance(sources,seed=args.seed,padding_seconds=args.padding_seconds,
        tolerance_mm=args.tolerance_mm,subjects_per_cohort=args.subjects_per_cohort,gaps_per_bin=args.gaps_per_bin,
        episode_targeted=args.episode_targeted,
        scope="single empirical gap per trial; same-fly spatially matched; T2 SB2-5; fully observed reference run",
        profile_dir=str(args.profile_dir)),indent=2)+"\n")
    lines = ["Matched artificial-gap pilot", "",f"{len(chosen)} subjects; {len(trials)} donor/placement-arm attempts; {sum(t['status']=='matched' for t in trials)} matched simulations.",
        "Donors are sampled separately within each fly and duration bin, not in proportion to their population frequency. This pilot tests conditional sensitivity; pooled trial results are not cohort-level error rates or corrected manuscript means.",
        "Each trial hides x/y/width/height/orientation over one observed interval, then uses production interpolation and wall/all/edge detection. Rewards and SLI eligibility remain fixed. Reference and simulated metrics are evaluated within the same fully observed run; episode support must be entirely inside it.",
        "Optional episode_targeted arms additionally require overlap with a fully observed reference episode of the named metric. They are conditional stress tests, reported separately from spatial_matched placement. Unsupported attempts are kept in trials.csv.",
        "redetection.csv measures all six metrics and changes in episode counts, temporal overlap, labels/values, and timing. Means before the five-episode minimum are explicitly diagnostic, not manuscript per-fly estimates. Empty reference/simulated cells are undefined.",
        "fixed_boundaries.csv holds reference boundaries fixed for tortuosity, wall counts, full length of originally wall-contact-positive intervals, and alignment. Alignment can become unavailable under its existing missing-heading-frame rule. Turnback and avoidance ratios are evaluated through redetection, rather than assigning artificial outcomes to fixed boundaries.",
        "Overlap matches are diagnostic associations, not independently validated event identities. Lack of temporal overlap is not automatically a missed or false biological event. Fully tracked references are not manual ground truth; long/rare gap contexts without a match remain unsupported.", ""]
    for cohort in COHORTS:
        ts = [t for t in trials if t["cohort"]==cohort]
        lines.append(f"{COHORTS[cohort]}: {sum(t['status']=='matched' for t in ts)}/{len(ts)} sampled gaps matched.")
    (folder/"report.md").write_text("\n\n".join(lines)+"\n")
    (folder/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Artificial-gap pilot</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape("\n\n".join(lines))+'</pre><p><a href="trials.csv">Matching coverage</a> · <a href="redetection.csv">Metric and detection changes</a> · <a href="fixed_boundaries.csv">Fixed-boundary measurements</a></p>')
    print(f"Wrote {folder}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    for name in ("profile","pilot"):
        p = commands.add_parser(name)
        p.add_argument("--output-dir",type=Path,required=True)
        p.add_argument("--padding-seconds",type=float,default=2.)
        if name=="profile":
            p.add_argument("--source-dir",type=Path,default=DEFAULT_SOURCE)
        else:
            p.add_argument("--profile-dir",type=Path,required=True)
            p.add_argument("--subjects-per-cohort",type=int,default=3)
            p.add_argument("--gaps-per-bin",type=int,default=1)
            p.add_argument("--tolerance-mm",type=float,default=2.)
            p.add_argument("--seed",type=int,default=20261108)
            p.add_argument("--episode-targeted",action="store_true",help="Also test matched placements overlapping observed metric episodes; keep arms separate")
    args = parser.parse_args(argv)
    if not np.isfinite(args.padding_seconds) or args.padding_seconds<0:
        parser.error("Padding must be finite and nonnegative")
    if args.command=="pilot" and (args.subjects_per_cohort<1 or args.gaps_per_bin<1 or args.seed<0 or not np.isfinite(args.tolerance_mm) or args.tolerance_mm<=0):
        parser.error("Pilot sample sizes/tolerance must be positive; seed nonnegative")
    (profile if args.command=="profile" else pilot)(args)


if __name__=="__main__":
    main()
