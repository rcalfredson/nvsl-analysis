#!/usr/bin/env python3
"""Fixed-episode tracking-loss sensitivity for six large-chamber metrics.

Reads frozen manuscript sources; never runs analyze.py or writes its outputs.
Production detectors reconstruct episodes and must reproduce archived baselines.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_tracking_gap_endpoints import load_context
from scripts.survey_frame_loss import loss_metrics
from src.analysis.episode_filters import episode_within_windows
from src.analysis.episode_tracking_loss import GapIndex, reduce_episodes
from src.analysis.trajectory import Trajectory
from src.exporting.turnback_home_vector_alignment_sli_bundle import (
    episode_home_vector_alignment, episode_home_vector_alignment_components,
    _turnback_event_frame,
)
from src.exporting.turnback_outer_radius_sli_bundle import _selected_windows_for_va
from src.plotting.between_reward_segment_metrics import tortuosity_metric_masked

COHORTS = {"ctrl": "Control", "pfnd_kir": "PFNd>Kir", "antennae_removed": "Antennae removed"}
AGAROSE_SLUGS = {"ctrl": "intact_ctrlKir", "pfnd_kir": "intact_pfnKir", "antennae_removed": "ar_ctrlKir"}
POLICIES = ("baseline", "complete_episode", "complete_support")
PAIRS = ((3, 5), (8, 10), (13, 15))


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = {}

    def track(self, path):
        path = Path(path)
        if str(path) not in self.manifest:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise ValueError(f"Source changed while hashing: {path}")
            self.manifest[str(path)] = dict(sha256=digest, size=after.st_size,mtime_ns=after.st_mtime_ns)
        return path

    def verify_unchanged(self):
        for name,snapshot in self.manifest.items():
            stat = Path(name).stat()
            if (stat.st_size,stat.st_mtime_ns) != (snapshot["size"],snapshot["mtime_ns"]):
                raise ValueError(f"Source changed during analysis: {name}")

    def archive(self, name):
        path = self.track(self.root / "exports/manuscript" / name)
        provenance = path.with_suffix(".provenance.json")
        if provenance.exists():
            prov = json.loads(self.track(provenance).read_text())
            if prov["source_sha256"] != self.manifest[str(path)]["sha256"]:
                raise ValueError(f"Archive provenance mismatch: {path}")
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    def agarose(self, cohort):
        path = self.track(self.root / "exports" / f"agarose_avoid_1mm_{AGAROSE_SLUGS[cohort]}_flatLgc_2026-04-10.npz")
        with np.load(path, allow_pickle=True) as data:
            z = {k: data[k] for k in data.files}
        if str(z["agarose_dual_circle_boundary_policy"]) != "hysteretic":
            raise ValueError("Expected current hysteretic agarose boundary policy")
        for key in ("agarose_farthest_from_reward_only", "agarose_wall_facing_entry_only", "agarose_exclude_reward_facing_arc_entries"):
            if bool(z[key]):
                raise ValueError(f"Unsupported spatial filter: {key}")
        if bool(z["exp_target_sync_bucket_filter_enabled"]) or bool(z["exp_pi_threshold_filter_enabled"]):
            raise ValueError("Agarose reference must use its independent, unfiltered subject pool")
        minimum = int(z["min_agarose_episodes"])
        for prefix in ("agarose", "agarose_pre"):
            counts = np.asarray(z[f"{prefix}_total_exp"],dtype=int)
            numerator = np.asarray(z[f"{prefix}_avoid_exp"],dtype=float)
            ratio = np.divide(numerator,counts,out=np.full(counts.shape,np.nan),where=counts>=minimum)
            np.testing.assert_allclose(ratio,z[f"{prefix}_ratio_exp"],rtol=1e-12,atol=1e-12,equal_nan=True)
        return z


def identities(bundle):
    if "video_fns" in bundle:
        keys = [(str(v), int(f)) for v, f in zip(bundle["video_fns"], bundle["fly_ids"], strict=True)]
    else:
        keys = [(video, int(pair)) for video, pair in
                (str(uid).rsplit("::f",1) for uid in bundle["video_ids"])]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate video/fly identity in source bundle")
    return keys


def agarose_signature(bundle, index, periods):
    """Counts/numerators, without inferring fly identity from numerical values."""
    signature = []
    for period in periods:
        if period == "Pre":
            signature.extend((int(bundle["agarose_pre_total_exp"][index]),int(bundle["agarose_pre_avoid_exp"][index])))
        else:
            ti,bi = (int(x) for x in period.replace("T","").split("_SB"))
            signature.extend((int(bundle["agarose_total_exp"][index,ti-1,bi-1]),int(bundle["agarose_avoid_exp"][index,ti-1,bi-1])))
    return tuple(signature)


def scalar_values(bundle, keys):
    """Resolve each historical ID format against unique full video/pair identities."""
    aliases = {}
    for video, pair in keys:
        for alias in (
            f"{video}|fly_id={pair}|trx_idx=0",
            f"{Path(video).name}|va_tag={pair}|trx_idx=0",
            f"{Path(video).stem}:fly{pair}",
        ):
            if alias in aliases and aliases[alias] != (video, pair):
                raise ValueError(f"Ambiguous historical scalar identity: {alias}")
            aliases[alias] = (video, pair)
    result = {}
    for uid, value in zip(bundle["per_unit_ids_panel"][0], bundle["per_unit_values_panel"][0], strict=True):
        uid = str(uid)
        if not uid and not np.isfinite(value):
            continue
        key = aliases[uid]
        if key in result:
            raise ValueError(f"Duplicate scalar subject: {uid}")
        result[key] = float(value)
    return result


def index_wall_episodes(rows, keys):
    """Reject duplicate/truncated interval exports even if their means agree."""
    allowed = set(keys)
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        if int(row["training_index"]) != 2:
            continue
        key = (row["video_path"], int(row["va_tag"]))
        if key not in allowed or row["fly_role"] != "exp" or int(row["trajectory_index"]) != 0:
            raise ValueError(f"Unexpected wall-episode subject: {key}")
        start, stop = int(row["start_reward_frame"]), int(row["end_reward_frame"])
        bounds = int(row["window_start_frame"]), int(row["window_end_frame"])
        if not bounds[0] <= start < stop < bounds[1]:
            raise ValueError(f"Wall interval outside its reward window: {key}")
        episode_key = (*key, start, stop)
        if episode_key in seen:
            raise ValueError(f"Duplicate wall episode: {episode_key}")
        seen.add(episode_key)
        grouped[key].append(row)
    for key, episodes in grouped.items():
        ordered = sorted(episodes, key=lambda row: int(row["start_reward_frame"]))
        previous_stop = None
        for row in ordered:
            if int(row["pooled_episode_count"]) != len(episodes):
                raise ValueError(f"Wall episode count mismatch: {key}")
            if previous_stop is not None and int(row["start_reward_frame"]) != previous_stop:
                raise ValueError(f"Nonconsecutive wall reward intervals: {key}")
            previous_stop = int(row["end_reward_frame"])
    return grouped


def close_value(actual, expected, label):
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True):
        raise ValueError(f"Baseline mismatch ({label}): reconstructed={actual}, archived={expected}")


def make_trajectory(context):
    """Use production interpolation and distances, with admission frozen by archives."""
    va = context["analysis"]
    trj = Trajectory.__new__(Trajectory)
    trj.x, trj.y = context["x"].copy(), context["y"].copy()
    trj.theta = None
    trj.f, trj.va, trj.opts = 0, va, va.opts
    trj._max_proportion_lost_fms = 1.0
    trj._interpolate()
    trj._calcDistances()
    trj.pxPerMmFloor = va.ct.pxPerMmFloor()
    va.trx = [trj]
    va.trxf = (context["physical_id"],)
    va.sync_bucket_ranges = [g["sync_buckets"] for g in context["geometries"]]
    return va, trj


def episode_row(base, gaps, *, metric, variant, period, start, stop, value,
                support_start=None, support_stop=None, event=None, heading_frames=()):
    start, stop = int(start), int(stop)
    support_start = max(0, start - 1 if support_start is None else int(support_start))
    support_stop = min(len(gaps.missing), stop + 1 if support_stop is None else int(support_stop))
    if not support_start <= start < stop <= support_stop:
        raise ValueError("Decision support must contain the measurement interval")
    row = dict(**base, metric=metric, variant=variant, period=period,
               start_frame=start, stop_frame_exclusive=stop,
               support_start_frame=support_start, support_stop_frame_exclusive=support_stop,
               value=float(value), outcome_frame="" if event is None else int(event))
    for prefix, bounds in (("episode", (start, stop)), ("support", (support_start, support_stop))):
        row.update({f"{prefix}_{k}": v for k, v in gaps.exposure(*bounds).items()})
    row["outcome_frame_missing"] = bool(event is not None and gaps.missing[int(event)])
    row["heading_sampled_frames"] = json.dumps([int(f) for f in heading_frames])
    row["heading_missing_frames"] = sum(bool(gaps.missing[int(f)]) for f in heading_frames)
    return row


def tortuosity_episodes(va, trj, base, gaps, window):
    if not window:
        return []
    fi, end = window[0][0], window[-1][1]
    df = window[0][1] - fi
    trn = va.trns[1]
    center = trn.circles(0)[0][:2]
    rows = []
    for seg in va._iter_between_reward_segment_com(
        trn, 0, fi=fi, df=df, n_buckets=len(window), complete=[True]*len(window),
        relative_to_reward=True, per_segment_min_meddist_mm=0., exclude_wall=False,
        exclude_nonwalk=False, min_walk_frames=2, dist_stats=(),
        exclude_reward_endpoints=False, debug=False, yield_skips=False,
    ):
        value = tortuosity_metric_masked(
            traj=trj, s=seg.s, e=seg.e, fi=fi, nonwalk_mask=None,
            exclude_nonwalk=False, px_per_mm=base["px_per_mm"],
            mode="path_over_max_radius", reward_center_xy=center,
        )
        if np.isfinite(value):
            rows.append(episode_row(base, gaps, metric="tortuosity", variant="full",
                                    period="T2_SB2-5", start=seg.s, stop=seg.e, value=value))
    return rows


def turnback_episodes(va, trj, base, gaps, window, inner, outer):
    variant = f"{inner}/{outer}mm"
    windows = [dict(start=window[0][0], stop=window[-1][1])] if window else []
    ratios, alignments = [], []
    trn = va.trns[1]
    for ep in trj.reward_turnback_dual_circle_episodes_for_training(
        trn=trn, inner_radius_mm=inner, outer_radius_mm=outer, border_width_mm=.1,
    ):
        if not episode_within_windows(ep, windows):
            continue
        event = _turnback_event_frame(ep)
        ratios.append(episode_row(
            base, gaps, metric="turnback_ratio", variant=variant, period="T2_SB2-5",
            start=ep["start"], stop=ep["stop"], value=bool(ep["turns_back"]), event=event,
        ))
        if not ep["turns_back"] or not windows[0]["start"] <= event < windows[0]["stop"]:
            continue
        value = episode_home_vector_alignment(
            trj, trn, ep, window_radius_frames=2, heading_estimator="mean",
            home_vector_anchor="intersection", max_interpolated_heading_frames=1,
        )
        if not np.isfinite(value):
            continue
        comp = episode_home_vector_alignment_components(
            trj, trn, ep, window_radius_frames=2, home_vector_anchor="intersection",
            max_interpolated_heading_frames=1,
        )["mean"]
        frames = comp["sampled_frames"]
        alignments.append(episode_row(
            base, gaps, metric="home_vector_alignment", variant=variant, period="T2_SB2-5",
            start=ep["start"], stop=ep["stop"], value=value, event=event,
            support_start=min(ep["start"]-1, int(min(frames))),
            support_stop=max(ep["stop"]+1, int(max(frames))+1), heading_frames=frames,
        ))
    return ratios, alignments


def agarose_episodes(va, trj, base, gaps, source):
    trj.calc_agarose_dual_circle_episodes(
        delta_mm=float(source["agarose_outer_radius_offset_mm"]),
        inner_radius_offset_mm=float(source["agarose_inner_radius_offset_mm"]),
        boundary_policy=str(source["agarose_dual_circle_boundary_policy"]),
    )
    pre_end = int(va.trns[0].start)
    periods = [("Pre", max(int(va.startPre), pre_end-va._min2f(10)), pre_end)]
    for ti, rr in enumerate(va.sync_bucket_ranges[:2], 1):
        # Preserve absent buckets as explicit zero-count cells. Different flies
        # in one video can have different actual-reward synchronization coverage.
        for bi in range(1,6):
            a,b = rr[bi-1] if bi <= len(rr) else (0,0)
            periods.append((f"T{ti}_SB{bi}",a,b))
    rows = defaultdict(list)
    for ep in trj.agarose_dual_circle_episodes:
        for period, start, stop in periods:
            if start <= ep["start"] < stop:
                rows[period].append(episode_row(
                    base, gaps, metric="agarose_avoidance", variant="physical_1mm",
                    period=period, start=ep["start"], stop=ep["stop"],
                    value=ep["avoids_inner"], event=ep["start"],
                ))
                break
    return rows, periods


def subject_rows(base, episodes, *, metric, variant, period, expected, eligible, reason,
                 minimum=5, expected_count=None, expected_sum=None,
                 validation="individual_fly_archive",measurement_gate=True):
    baseline = reduce_episodes(episodes, policy="baseline", minimum=minimum)
    label = f'{base["cohort"]}/{base["video_path"]}/fly{base["pair"]}/{metric}/{variant}/{period}'
    # Baselines must agree before any sensitivity estimate is published.
    close_value(baseline["value"] if measurement_gate else np.nan, expected, label)
    if expected_count is not None:
        close_value(baseline["n_episodes"], expected_count, label+" count")
    if expected_sum is not None:
        close_value(baseline["episode_value_sum"], expected_sum, label+" numerator")
    result = []
    for policy in POLICIES:
        reduced = reduce_episodes(episodes, policy=policy, minimum=minimum)
        included = bool(eligible and np.isfinite(reduced["value"]))
        result.append(dict(
            **base, metric=metric, variant=variant, period=period, policy=policy,
            minimum_episodes=minimum, source_gate_passed=bool(eligible),
            baseline_eligible=bool(eligible and np.isfinite(baseline["value"])),
            baseline_exclusion_reason=reason or ("insufficient baseline episodes" if not np.isfinite(baseline["value"]) else ""),
            reference_value=float(expected), baseline_verified=True,baseline_validation=validation,
            baseline_n_episodes=baseline["n_episodes"],
            n_episodes=reduced["n_episodes"], excluded_episodes=baseline["n_episodes"]-reduced["n_episodes"],
            episode_value_sum=reduced["episode_value_sum"],
            measured_value=reduced["value"], included=included,
            value=reduced["value"] if included else float("nan"),
            episodes_with_loss=sum(e["episode_lost_frames"] > 0 for e in episodes),
            episodes_with_support_loss=sum(e["support_lost_frames"] > 0 for e in episodes),
            outcomes_missing=sum(e["outcome_frame_missing"] for e in episodes),
            heading_samples_missing=sum(e["heading_missing_frames"] > 0 for e in episodes),
            max_episode_gap_s=max((e["episode_max_intersecting_gap_s"] for e in episodes),default=0.),
            episode_frames=sum(e["episode_frames"] for e in episodes),
            episode_lost_frames=sum(e["episode_lost_frames"] for e in episodes),
            exclusion_reason=(reason if not eligible else "" if included else "insufficient retained episodes"),
        ))
    return result


def collect(root, sources, *, limit=None, cohorts=None):
    episodes, subjects, coverage, gap_records = [], [], [], []
    for cohort in (cohorts or COHORTS):
        master = sources.archive(f"fig4n_{cohort}_turnback_archive.npz")
        np.testing.assert_array_equal(master["turnback_excursion_bin_pair_inner_deltas_mm"],[3,8,13])
        np.testing.assert_array_equal(master["turnback_excursion_bin_pair_outer_deltas_mm"],[5,10,15])
        keys = identities(master)
        agarose = sources.agarose(cohort)
        agarose_videos = [str(v) for v in agarose["video_ids"]]
        if Counter(v for v,_ in keys) != Counter(agarose_videos):
            raise ValueError(f"Agarose/manuscript video multiplicities differ: {cohort}")
        # A pilot includes every fly from each selected video so the unkeyed
        # agarose reference can still be validated without guessing fly IDs.
        selected_videos = {v for v,_ in keys[:limit]} if limit else {v for v,_ in keys}
        reconstructed_agarose = defaultdict(list)
        scalars, gates = {}, {}
        for panel, metric, suffix in (("j", "tortuosity", "tortuosity"),
                                      ("k", "wall_contacts", "wall_contacts"),
                                      ("l", "wall_contact_length", "wall_contact_trajectory_length")):
            scalars[metric] = scalar_values(sources.archive(f"fig4{panel}_{cohort}_{suffix}_archive.npz"), keys)
            eligibility = sources.archive(f"fig4{panel}_{cohort}_sli_eligibility.npz")
            gates[metric] = {k:bool(np.isfinite(eligibility["sli_ts"][i,1,4]))
                             for i,k in enumerate(identities(eligibility))}
        alignment = {inner:scalar_values(sources.archive(f"fig4q_{cohort}_{inner}_{outer}_alignment_archive.npz"), keys)
                     for inner,outer in PAIRS}
        qelig = sources.archive(f"fig4q_{cohort}_sli_eligibility.npz")
        qgates = {k:bool(np.isfinite(qelig["sli_ts"][i,1,4])) for i,k in enumerate(identities(qelig))}
        path = sources.track(Path(root)/"exports/manuscript"/f"fig4k_{cohort}_wall_contacts_episodes.csv")
        wall = index_wall_episodes(read_csv(path), keys)
        selected_count = sum(v in selected_videos for v,_ in keys)
        processed = 0
        for i, key in enumerate(keys):
            video, pair = key
            if video not in selected_videos:
                continue
            processed += 1
            print(f"{cohort}: {processed}/{selected_count} {Path(video).name} fly {pair}", flush=True)
            for suffix in (".trx", ".data"):
                sources.track(Path(video).with_suffix(suffix))
            note = Path(video).with_name(Path(video).stem+"__note.txt")
            if note.exists():
                sources.track(note)
            with redirect_stdout(io.StringIO()):
                context = load_context(video,pair,"exp",include_analysis=True)
                va,trj = make_trajectory(context)
            gaps = GapIndex(trj.nan,context["fps"])
            yoked_id = int(va.ef2yc[pair] if va.ef2yc else pair+va.nef)
            base = dict(cohort=cohort, video_path=video, pair=pair, role="exp",
                        physical_tracking_id=context["physical_id"], fps=context["fps"],px_per_mm=context["px_per_mm"],
                        recording_loss_pct=loss_metrics(context["x"])["loss_pct"],
                        yoked_physical_tracking_id=yoked_id,
                        yoked_recording_loss_pct=loss_metrics(va.trxRw["x"][yoked_id])["loss_pct"])
            (left,top),(right,bottom) = context["floor"]
            for gs,ge in zip(gaps.starts,gaps.stops):
                rec = dict(**base,gap_start_frame=int(gs),gap_stop_frame_exclusive=int(ge),
                           gap_duration_s=float((ge-gs)/gaps.fps),initial_gap=bool(gs==0),trailing_gap=bool(ge==len(trj.x)))
                for prefix,frame in (("before",int(gs)-1),("after",int(ge))):
                    valid = 0 <= frame < len(trj.x)
                    x,y = (context["x"][frame],context["y"][frame]) if valid else (np.nan,np.nan)
                    rec.update({prefix+"_frame":frame if valid else "",prefix+"_x_px":float(x),prefix+"_y_px":float(y),
                                prefix+"_nearest_wall_mm":float(min(x-left,right-x,y-top,bottom-y)/context["px_per_mm"]) if np.isfinite([x,y]).all() else np.nan})
                gap_records.append(rec)
            window = va.sync_bucket_ranges[1][1:5]
            target = bool(master["exp_target_sync_bucket_filter_eligible"][i])
            # Check reconstructed synchronization against the archived SB5 gate.
            end = master["exp_target_sync_bucket_filter_target_bucket_stop"][i]
            if np.isfinite(end):
                close_value(va.sync_bucket_ranges[1][4][1],end,f"{key} SB5 stop")
            for ti,g in enumerate(context["geometries"][:2],1):
                coverage.append(dict(**base,period=f"T{ti}",start_frame=g["training_start_frame"],
                                     stop_frame_exclusive=g["training_stop_frame"],
                                     **gaps.exposure(g["training_start_frame"],g["training_stop_frame"]),
                                     initial_missing_frames=(int(gaps.stops[0]) if gaps.starts.size and gaps.starts[0]==0 else 0)))
            if window:
                coverage.append(dict(**base,period="T2_SB2-5",start_frame=window[0][0],stop_frame_exclusive=window[-1][1],
                                     **gaps.exposure(window[0][0],window[-1][1]),
                                     initial_missing_frames=(int(gaps.stops[0]) if gaps.starts.size and gaps.starts[0]==0 else 0)))
            eps = tortuosity_episodes(va,trj,base,gaps,window)
            episodes.extend(eps)
            subjects.extend(subject_rows(base,eps,metric="tortuosity",variant="full",period="T2_SB2-5",
                                         expected=scalars["tortuosity"].get(key,np.nan),
                                         eligible=gates["tortuosity"][key] and target,measurement_gate=target,
                                         reason="experimental SB5 absent" if not target else "" if gates["tortuosity"][key] else "undefined paired final SLI"))
            for metric in ("wall_contacts","wall_contact_length"):
                eps = []
                for row in wall[key]:
                    count = int(row["wall_contact_event_count"])
                    value = float(count) if metric=="wall_contacts" else float(row["trajectory_length_mm"])
                    if metric=="wall_contact_length" and (not count or not np.isfinite(value)):
                        continue
                    eps.append(episode_row(base,gaps,metric=metric,variant="full",period="T2_SB2-5",
                                           start=int(row["start_reward_frame"]),stop=int(row["end_reward_frame"]),value=value))
                episodes.extend(eps)
                subjects.extend(subject_rows(base,eps,metric=metric,variant="full",period="T2_SB2-5",
                                             expected=scalars[metric].get(key,np.nan),eligible=gates[metric][key] and target,measurement_gate=target,
                                             reason="experimental SB5 absent" if not target else "" if gates[metric][key] else "undefined paired final SLI"))
            tbwindows = _selected_windows_for_va(va,[1],skip_first=1,keep_first=4,last_sync_buckets=0)
            tbwindow = tbwindows[0]["bucket_ranges"] if tbwindows else []
            for pi,(inner,outer) in enumerate(PAIRS):
                ratio,align = turnback_episodes(va,trj,base,gaps,tbwindow,inner,outer)
                for metric,eps,expected,gate in (
                    ("turnback_ratio",ratio,float(master["turnback_excursion_bin_ratio_exp"][i,pi]),bool(np.isfinite(master["sli_ts"][i,1,4]))),
                    ("home_vector_alignment",align,alignment[inner].get(key,np.nan),qgates[key]),
                ):
                    episodes.extend(eps)
                    # Scalars are masked by SB5 presence in the source exporter.
                    subjects.extend(subject_rows(base,eps,metric=metric,variant=f"{inner}/{outer}mm",period="T2_SB2-5",
                        expected=expected,eligible=gate and target,measurement_gate=target,
                        reason="" if gate and target else "undefined paired final SLI" if not gate else "experimental SB5 absent",
                        expected_count=(int(master["turnback_excursion_bin_total_exp"][i,pi]) if metric=="turnback_ratio" else None),
                        expected_sum=(float(master["turnback_excursion_bin_turn_exp"][i,pi]) if metric=="turnback_ratio" else None)))
            by_period,periods = agarose_episodes(va,trj,base,gaps,agarose)
            signature = []
            for period,_,_ in periods:
                eps = by_period[period]
                episodes.extend(eps)
                minimum = int(agarose["min_agarose_episodes"])
                baseline = reduce_episodes(eps,policy="baseline",minimum=minimum)
                signature.extend((baseline["n_episodes"],int(baseline["episode_value_sum"])))
                subjects.extend(subject_rows(base,eps,metric="agarose_avoidance",variant="physical_1mm",period=period,
                    expected=baseline["value"],eligible=True,reason="",minimum=minimum,
                    validation="video_multiset_counts_and_numerators"))
            reconstructed_agarose[video].append(tuple(signature))
            # No numeric matching is used to assign archived rows to flies.
            # Validate the complete multiset only after all source flies for a
            # video are reconstructed. Fail the run before writing any reports.
            if len(reconstructed_agarose[video]) == agarose_videos.count(video):
                names = [p for p,_,_ in periods]
                expected = [agarose_signature(agarose,j,names) for j,v in enumerate(agarose_videos) if v==video]
                if sorted(reconstructed_agarose[video]) != sorted(expected):
                    raise ValueError(f"Agarose video-level baseline mismatch: {video}\nactual={reconstructed_agarose[video]}\narchived={expected}")
    return episodes,subjects,coverage,gap_records


def bootstrap_stats(rows, draws, seed, *, return_samples=False):
    """Resample whole videos, preserving the fly-weighted mean and paired policies."""
    groups = defaultdict(dict)
    for row in rows:
        key = (row["video_path"],row["pair"])
        if row["policy"] in groups[key]:
            raise ValueError(f"Duplicate subject/policy: {key}/{row['policy']}")
        groups[key][row["policy"]] = row
    videos = sorted({key[0] for key,g in groups.items() if g["baseline"]["included"]})
    rng = np.random.default_rng(seed)
    counts = (rng.multinomial(len(videos),np.full(len(videos),1/len(videos)),size=draws)
              if videos else np.zeros((draws,0),dtype=int))
    index = {v:i for i,v in enumerate(videos)}

    def estimate(selected,field="value"):
        if not selected:
            return np.nan, np.full(draws,np.nan)
        totals = np.zeros(len(videos))
        sizes = np.zeros(len(videos))
        for row in selected:
            vi = index[row["video_path"]]
            totals[vi] += row[field]
            sizes[vi] += 1
        den = counts@sizes
        boot = np.divide(counts@totals,den,out=np.full(draws,np.nan),where=den>0)
        return float(sum(r[field] for r in selected)/len(selected)),boot

    def ci(samples):
        good = samples[np.isfinite(samples)]
        return tuple(float(x) for x in np.quantile(good,[.025,.975])) if good.size else (np.nan,np.nan)

    baseline = [g["baseline"] for g in groups.values() if g["baseline"]["included"]]
    base_mean, base_boot = estimate(baseline)
    output = []
    for policy in POLICIES:
        retained = [g[policy] for g in groups.values() if g["baseline"]["included"] and g[policy]["included"]]
        matched_base = [g["baseline"] for g in groups.values() if g["baseline"]["included"] and g[policy]["included"]]
        mean,boot = estimate(retained)
        matched_mean,matched_boot = estimate(matched_base)
        low,high = ci(boot)
        dlo,dhi = ci(boot-matched_boot)
        baseline_eps = sum(r["baseline_n_episodes"] for r in baseline)
        output.append(dict(
            policy=policy,baseline_n_flies=len(baseline),n_flies=len(retained),
            n_videos=len({r["video_path"] for r in retained}),
            retention_fraction=len(retained)/len(baseline) if baseline else np.nan,
            baseline_mean=base_mean,mean=mean,ci_low=low,ci_high=high,
            sd=float(np.std([r["value"] for r in retained],ddof=1)) if len(retained)>1 else np.nan,
            median=float(np.median([r["value"] for r in retained])) if retained else np.nan,
            bootstrap_valid_draws=int(np.isfinite(boot).sum()),
            mean_shift_from_full_baseline=mean-base_mean,
            common_fly_baseline_mean=matched_mean,common_fly_shift=mean-matched_mean,
            common_fly_shift_ci_low=dlo,common_fly_shift_ci_high=dhi,
            baseline_episodes=baseline_eps,
            baseline_episodes_with_loss=sum(r["episodes_with_loss"] for r in baseline),
            baseline_episodes_with_support_loss=sum(r["episodes_with_support_loss"] for r in baseline),
            support_loss_episode_fraction=(sum(r["episodes_with_support_loss"] for r in baseline)/baseline_eps if baseline_eps else np.nan),
            retained_episodes=sum(r["n_episodes"] for r in retained),
            retained_episodes_before_fly_attrition=sum(g[policy]["n_episodes"] for g in groups.values() if g["baseline"]["included"]),
        ))
        if return_samples:
            output[-1].update(bootstrap_mean=boot,bootstrap_shift=boot-base_boot)
    return output


def summarize(subjects,draws=2000,seed=20261108):
    groups = defaultdict(list)
    for row in subjects:
        groups[(row["cohort"],row["metric"],row["variant"],row["period"])].append(row)
    summary = []
    for (cohort,metric,variant,period),rows in sorted(groups.items()):
        for stats in bootstrap_stats(rows,draws,seed):
            summary.append(dict(cohort=cohort,metric=metric,variant=variant,period=period,**stats))
    return summary


def episode_strata(episodes,subjects):
    """Describe behavioral selection; these are episode-weighted diagnostics."""
    fields = ("cohort","video_path","pair","metric","variant","period")
    included = {tuple(r[k] for k in fields) for r in subjects if r["policy"]=="baseline" and r["included"]}
    groups = defaultdict(list)
    for ep in episodes:
        if tuple(ep[k] for k in fields) in included:
            groups[(ep["cohort"],ep["metric"],ep["variant"],ep["period"],bool(ep["support_lost_frames"]))].append(ep)
    rows = []
    for (cohort,metric,variant,period,affected),eps in sorted(groups.items()):
        durations = [(e["stop_frame_exclusive"]-e["start_frame"])/e["fps"] for e in eps]
        rows.append(dict(cohort=cohort,metric=metric,variant=variant,period=period,
                         support_touches_gap=affected,n_episodes=len(eps),
                         n_flies=len({(e["video_path"],e["pair"]) for e in eps}),
                         duration_mean_s=float(np.mean(durations)),duration_median_s=float(np.median(durations)),
                         episode_value_mean=float(np.mean([e["value"] for e in eps])),
                         episode_value_median=float(np.median([e["value"] for e in eps])),
                         episode_lost_frame_fraction=sum(e["episode_lost_frames"] for e in eps)/sum(e["episode_frames"] for e in eps)))
    return rows


def write_report(folder,summary,subjects,coverage,*,pilot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [r for r in summary if r["policy"]=="complete_support" and
               (r["metric"]!="agarose_avoidance" or r["period"]=="T2_SB5")]
    lines = ["# Tracking-loss sensitivity of fixed episodes", "",
             "PILOT: limited subjects per cohort; not cohort estimates." if pilot else "Full three-cohort flat large-chamber analysis; experimental trajectories only.", "",
             "Tortuosity, wall metrics, turnback ratios, and alignment are checked against frozen individual-fly sources. Agarose archives omit fly IDs: reconstructed counts and numerators across all reported periods are verified as an unordered multiset within each video, without assigning archive rows to flies. Archived agarose ratios are checked against their counts. New episode and subject tables preserve explicit video/fly identity. Admission and paired-final-SLI eligibility are frozen to each panel's source. Agarose uses its own source pool without a final-SLI gate.", "",
             "The primary exclusion (`complete_support`) removes episodes touching original missing tracking in the episode or its immediate transition/heading support. `complete_episode` checks only the metric's half-open interval. Episodes are not redetected. Gap-free support does not validate tracking identity or rule out undetected episodes.", "",
             "## Primary comparison", "",
             "Agarose is shown here at T2 SB5; summary.csv includes Pre and every available bucket in Trainings 1–2. Changes are in original metric units (ratios are fractions, length is mm). Common-fly change compares the same retained flies. Confidence intervals use video-cluster resampling and are exploratory with few videos.", "",
             "| Cohort | Metric | Variant | Baseline n → retained n | Exposed episodes | Full baseline mean | Retained mean | Common-fly change |",
             "|---|---|---|---|---|---|---|---|"]
    for row in primary:
        lines.append(f'| {COHORTS[row["cohort"]]} | {row["metric"]} | {row["variant"]} | {row["baseline_n_flies"]} → {row["n_flies"]} | {row["support_loss_episode_fraction"]:.1%} | {row["baseline_mean"]:.4g} | {row["mean"]:.4g} | {row["common_fly_shift"]:+.4g} |')
    lines.extend(["", "## Interpretation limits", "",
        "Excluding gap-containing bouts selects behavior: long excursions and wall-associated behavior may be removed preferentially. A large change is a sensitivity result, not an estimate of tracking error or a corrected biological effect. Hidden bouts are absent from both analyses. Hysteresis and wall-contact states can depend on earlier history beyond the immediate support. Actual reward frames remain fixed; this analysis does not recreate missed circle entries or reward delivery.", "",
        "Tortuosity uses the production maximum-overlap bucket assignment and path/max-radius definition. Wall metrics reuse frozen contact-event assignments; wall-contact length retains the production forward-step endpoint convention. Turnback ratios count successes over all eligible detected bouts; alignment requires five successful bouts with finite values at each radius. Agarose assigns entire bouts by entry frame, so support can extend beyond its reported bucket.", "",
        "Source .trx/.data and archive hashes, analysis-code hashes, settings, and invocation are recorded in provenance.json. AVI pixels were not reviewed. Component two (artificial gaps) and manual review have not been run.", ""])
    (folder/"report.md").write_text("\n".join(lines))
    table = "<table><tr>"+"".join(f"<th>{html.escape(k)}</th>" for k in ("Cohort","Metric","Variant","Baseline n","Retained n","Exposed episodes","Baseline mean","Retained mean","Common-fly change"))+"</tr>"
    for r in primary:
        cells = (COHORTS[r["cohort"]],r["metric"],r["variant"],r["baseline_n_flies"],r["n_flies"],f'{r["support_loss_episode_fraction"]:.1%}',f'{r["baseline_mean"]:.4g}',f'{r["mean"]:.4g}',f'{r["common_fly_shift"]:+.4g}')
        table += "<tr>"+"".join(f"<td>{html.escape(str(c))}</td>" for c in cells)+"</tr>"
    table += "</table>"
    (folder/"report.html").write_text("<!doctype html><meta charset=utf-8><title>Episode tracking-loss sensitivity</title><style>body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Tracking-loss sensitivity of fixed episodes</h1>"+table+'<p><a href="summary.csv">Full summary</a> · <a href="subjects.csv">Subject audit</a> · <a href="episodes.csv">Episode audit</a> · <a href="contrasts.csv">Cohort contrasts</a></p><img src="sensitivity.png"><pre>'+html.escape("\n".join(lines[2:]))+"</pre>")
    labels = sorted({(r["metric"],r["variant"]) for r in primary})
    fig,axes = plt.subplots(len(labels),2,figsize=(10,3*len(labels)),squeeze=False)
    for (metric,variant),ax in zip(labels,axes):
        rows = sorted([r for r in primary if (r["metric"],r["variant"])==(metric,variant)],key=lambda r:list(COHORTS).index(r["cohort"]))
        xx = np.arange(len(rows))
        ax[0].bar(xx-.18,[r["baseline_mean"] for r in rows],.36,label="Full baseline")
        ax[0].bar(xx+.18,[r["mean"] for r in rows],.36,label="Complete support")
        ax[1].bar(xx,[r["retention_fraction"] for r in rows],color="#527c99")
        ax[1].set_ylim(0,1.05)
        ax[1].set_ylabel("Fraction of flies retained")
        for a in ax:
            a.set_xticks(xx,[COHORTS[r["cohort"]] for r in rows],rotation=15)
            a.set_title(f"{metric}: {variant}")
        ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/"sensitivity.png",dpi=160)
    fig.savefig(folder/"sensitivity.pdf")
    plt.close(fig)


def cohort_contrasts(summary,subjects,draws,seed):
    grouped = defaultdict(list)
    for r in subjects:
        grouped[(r["cohort"],r["metric"],r["variant"],r["period"])].append(r)
    samples = {}
    for key,rows in sorted(grouped.items()):
        # Independent cohort draws; identical video draws across policies.
        cohort_seed = seed + int(hashlib.sha256(key[0].encode()).hexdigest()[:8],16)
        for stats in bootstrap_stats(rows,draws,cohort_seed,return_samples=True):
            samples[(*key,stats["policy"])] = stats

    def interval(values):
        values = values[np.isfinite(values)]
        return (*[float(v) for v in np.quantile(values,[.025,.975])],len(values)) if values.size else (np.nan,np.nan,0)

    lookup = {(r["cohort"],r["metric"],r["variant"],r["period"],r["policy"]):r for r in summary}
    rows = []
    for r in summary:
        if r["cohort"]=="ctrl":
            continue
        ctrl = lookup.get(("ctrl",r["metric"],r["variant"],r["period"],r["policy"]))
        if ctrl is None:
            continue
        diff = r["mean"]-ctrl["mean"]
        baseline = r["baseline_mean"]-ctrl["baseline_mean"]
        a = samples[r["cohort"],r["metric"],r["variant"],r["period"],r["policy"]]
        b = samples["ctrl",r["metric"],r["variant"],r["period"],r["policy"]]
        low,high,valid = interval(a["bootstrap_mean"]-b["bootstrap_mean"])
        slow,shigh,svalid = interval(a["bootstrap_shift"]-b["bootstrap_shift"])
        rows.append(dict(cohort=r["cohort"],reference="ctrl",metric=r["metric"],variant=r["variant"],period=r["period"],
                         policy=r["policy"],difference=diff,baseline_difference=baseline,change_in_difference=diff-baseline,
                         ci_low=low,ci_high=high,bootstrap_valid_draws=valid,
                         change_ci_low=slow,change_ci_high=shigh,change_bootstrap_valid_draws=svalid,
                         cohort_n=r["n_flies"],reference_n=ctrl["n_flies"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True,help="New empty results directory")
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--limit-per-cohort",type=int,help="Pilot: first N archived subjects, expanded to include all source flies from their videos")
    parser.add_argument("--cohort",action="append",choices=list(COHORTS))
    parser.add_argument("--bootstrap-draws",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261108)
    args = parser.parse_args(argv)
    if (args.limit_per_cohort is not None and args.limit_per_cohort<1) or args.bootstrap_draws<1:
        parser.error("Limits and bootstrap draws must be positive")
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    if args.cohort and len(set(args.cohort)) != len(args.cohort):
        parser.error("Each cohort may be selected only once")
    folder = args.output_dir
    if folder.exists() and any(folder.iterdir()):
        parser.error("Output directory must be empty; existing results are never overwritten")
    folder.mkdir(parents=True,exist_ok=True)
    sources = Sources(args.root)
    code = [Path(__file__).resolve(),ROOT/"src/analysis/episode_tracking_loss.py",ROOT/"scripts/audit_tracking_gap_endpoints.py",
            ROOT/"scripts/survey_frame_loss.py",ROOT/"src/analysis/trajectory.py",ROOT/"src/analysis/video_analysis.py",
            ROOT/"src/analysis/episode_filters.py",ROOT/"src/analysis/training.py",
            ROOT/"src/utils/common.py",ROOT/"src/utils/constants.py",ROOT/"src/utils/util.py",
            ROOT/"src/plotting/between_reward_segment_metrics.py",
            ROOT/"src/exporting/turnback_outer_radius_sli_bundle.py",
            ROOT/"src/exporting/turnback_home_vector_alignment_sli_bundle.py"]
    for path in code:
        sources.track(path)
    code_hashes = {str(path):sources.manifest[str(path)]["sha256"] for path in code}
    eps,subjects,coverage,gaps = collect(args.root,sources,limit=args.limit_per_cohort,cohorts=args.cohort)
    sources.verify_unchanged()
    summary = summarize(subjects,args.bootstrap_draws,args.seed)
    write_csv(folder/"episodes.csv",eps)
    write_csv(folder/"subjects.csv",subjects)
    write_csv(folder/"window_coverage.csv",coverage)
    write_csv(folder/"gaps.csv",gaps)
    write_csv(folder/"summary.csv",summary)
    write_csv(folder/"contrasts.csv",cohort_contrasts(summary,subjects,args.bootstrap_draws,args.seed))
    write_csv(folder/"episode_strata.csv",episode_strata(eps,subjects))
    write_report(folder,summary,subjects,coverage,pilot=bool(args.limit_per_cohort))
    (folder/"provenance.json").write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        policies=POLICIES,minimum_episodes=5,bootstrap_draws=args.bootstrap_draws,seed=args.seed,
        pilot_limit_per_cohort=args.limit_per_cohort,cohorts=args.cohort or list(COHORTS),
        roles=["exp"],sources=sources.manifest,code_sha256=code_hashes,
        python_version=sys.version,numpy_version=np.__version__,
    ),indent=2)+"\n")
    print(f"Verified {len(subjects)//len(POLICIES)} subject/metric/window baselines; wrote {folder}")


if __name__=="__main__":
    main()
