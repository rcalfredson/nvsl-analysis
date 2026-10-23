"""Candidate selection and manual-event comparison for tracking-gap review.

Reconstructed events use production circle/entry primitives and missing-state
holding. No judgments about true behavior are made without manual annotations.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def reconstructed_entries(context):
    from src.analysis.trajectory import Trajectory

    events = {}
    for g in context["geometries"]:
        start, stop = g["active_start_frame"], g["active_stop_frame"]
        x, y = context["x"][start:stop], context["y"][start:stop]
        missing = np.isnan(x)
        changes = np.diff(np.r_[False, missing, False].astype(int))
        runs = zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        runs = list(runs)
        for circle in ("reward", "control"):
            states = Trajectory.calc_in_circle(None, x, y, *g[circle], border_width_px=context["border_px"])
            for a, b in runs:
                states[a:b] = states[a-1] if a else 0
            # Production counts first-training pre and training in separate slices.
            split = g["training_start_frame"] - start if g["training"] == 1 else 0
            parts = [(0, split), (split, len(states))] if split else [(0, len(states))]
            events[g["training"], circle] = np.concatenate([
                Trajectory._calcEnEx(states[a:b], start+a) for a, b in parts
            ]).astype(int)
    return events


def point_state(context, frame, geometry, circle):
    if frame is None or not geometry["training_start_frame"] <= frame < geometry["training_stop_frame"]:
        return "unavailable"
    x, y = context["x"][frame], context["y"][frame]
    if not np.isfinite([x, y]).all():
        return "unavailable"
    cx, cy, radius = geometry[circle]
    distance = np.hypot(x-cx, y-cy)
    return "inside" if distance < radius else "border" if distance < radius+context["border_px"] else "outside"


def bucket_labels(geometry, start, stop):
    return ";".join(str(i) for i, (a, b) in enumerate(geometry.get("sync_buckets", []), 1)
                    if max(start, a) < min(stop, b))


def gap_candidates(context, gaps, *, training=2, padding_seconds=1., near_mm=1., long_seconds=2.):
    geometry = next(g for g in context["geometries"] if g["training"] == training)
    lo, hi = geometry["training_start_frame"], geometry["training_stop_frame"]
    padding = int(round(padding_seconds*context["fps"]))
    candidates = []
    for gap in gaps:
        a, b = gap["gap_start_frame"], gap["gap_stop_frame_exclusive"]
        # Include a gap recovering exactly at training onset for boundary review.
        if a >= hi or b < lo:
            continue
        for circle in ("reward", "control"):
            before = point_state(context, a-1 if a else None, geometry, circle)
            after = point_state(context, b if b < len(context["x"]) else None, geometry, circle)
            distances = [abs(gap[p+"_"+circle+"_boundary_signed_mm"])
                         for p in ("before", "after") if gap[p+"_training"] == training
                         and gap[p+"_phase"] == "training" and gap[p+"_"+circle+"_boundary_signed_mm"] is not None]
            near = bool(distances and min(distances) <= near_mm)
            if "unavailable" in (before, after) or "border" in (before, after):
                stratum = "border_or_unavailable"
            elif before != after:
                stratum = "state_change"
            elif near:
                stratum = "near_boundary_same_state"
            elif gap["missing_duration_s"] >= long_seconds:
                stratum = "long_gap_same_state"
            else:
                stratum = "other_same_state"
            start, stop = max(lo, a-1-padding), min(hi, b+1+padding)
            if start >= stop:
                continue
            candidates.append(dict(video_path=context["video_path"], fly=context["pair"], role=context["role"],
                                   physical_tracking_id=context["physical_id"], training=training, circle=circle,
                                   gap_id=gap["gap_id"], gap_start_frame=a, gap_stop_frame_exclusive=b,
                                   missing_duration_s=gap["missing_duration_s"], before_state=before, after_state=after,
                                   near_boundary=near, stratum=stratum, review_start_frame=start,
                                   review_stop_frame_exclusive=stop, sync_buckets=bucket_labels(geometry, start, stop)))
    return candidates


def sample_candidates(candidates, per_stratum, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in candidates:
        key = (row["video_path"], row["fly"], row["role"], row["circle"], row["stratum"])
        groups.setdefault(key, []).append(row)
    selected = []
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda r:r["gap_id"])
        indices = rng.choice(len(pool), min(per_stratum, len(pool)), replace=False)
        selected += [dict(pool[i], stratum_population=len(pool), stratum_sample=min(per_stratum, len(pool))) for i in sorted(indices)]
    return selected


def exclude_previous_windows(candidates, previous):
    """Prevent rescoring previously exported intervals for the same fly/circle."""
    def overlaps(row, old):
        return ((row["video_path"], int(row["fly"]), row["role"], row["circle"]) ==
                (old["video_path"], int(old["fly"]), old["role"], old["circle"]) and
                max(int(row["review_start_frame"]), int(old["review_start_frame"])) <
                min(int(row["review_stop_frame_exclusive"]), int(old["review_stop_frame_exclusive"])))
    return [row for row in candidates if not any(overlaps(row, old) for old in previous)]


def merge_review_windows(selected):
    """Merge overlapping selected intervals per fly/circle; never double-score them."""
    groups = {}
    for row in selected:
        key = (row["video_path"], row["fly"], row["role"], row["training"], row["circle"])
        groups.setdefault(key, []).append(row)
    episodes = []
    for key in sorted(groups):
        current = None
        for row in sorted(groups[key], key=lambda r:r["review_start_frame"]):
            start, stop = row["review_start_frame"], row["review_stop_frame_exclusive"]
            if current is not None and start < current["review_stop_frame_exclusive"]:
                current["review_stop_frame_exclusive"] = max(stop, current["review_stop_frame_exclusive"])
                current["selected_gap_ids"].append(row["gap_id"])
                current["strata"].add(row["stratum"])
            else:
                if current is not None:
                    episodes.append(current)
                current = dict(video_path=key[0], fly=key[1], role=key[2], training=key[3], circle=key[4],
                               review_start_frame=start, review_stop_frame_exclusive=stop,
                               selected_gap_ids=[row["gap_id"]], strata={row["stratum"]}, source="gap_review")
        if current is not None:
            episodes.append(current)
    return episodes


def add_observed_controls(episodes, contexts, count, seed, seconds=2., *, excluded=()):
    rng = np.random.default_rng(seed)
    result = list(episodes)
    for context in contexts:
        geometry = next(g for g in context["geometries"] if g["training"] == 2)
        lo, hi = geometry["training_start_frame"], geometry["training_stop_frame"]
        length = max(2, int(round(seconds*context["fps"])))
        for circle in ("reward", "control"):
            valid = np.isfinite(context["x"]) & np.isfinite(context["y"])
            valid[:lo] = False
            valid[hi:] = False
            for ep in [*result, *excluded]:
                if (ep["video_path"], int(ep["fly"]), ep["role"], ep["circle"]) == (context["video_path"], context["pair"], context["role"], circle):
                    valid[int(ep["review_start_frame"]):int(ep["review_stop_frame_exclusive"])] = False
            for _ in range(count):
                # A scored interval needs a preceding observed frame for entry context.
                sums = np.r_[0, np.cumsum(valid.astype(int))]
                starts = np.flatnonzero(sums[length+1:]-sums[:-length-1] == length+1)+1
                starts = starts[(starts >= lo) & (starts+length <= hi)]
                if not len(starts):
                    break
                start = int(rng.choice(starts))
                stop = start+length
                result.append(dict(video_path=context["video_path"], fly=context["pair"], role=context["role"],
                                   training=2, circle=circle, review_start_frame=start, review_stop_frame_exclusive=stop,
                                   selected_gap_ids=[], strata={"observed_control"}, source="observed_control"))
                valid[start:stop] = False
    return result


def finalize_episodes(episodes, contexts, events_by_context):
    """Freeze reconstructed events inside scored bounds and assign stable IDs.

    Hash inputs are video/fly/role, training, circle, start, and exclusive stop.
    Display numbering and cohort metadata deliberately do not affect identity.
    """
    lookup = {(c["video_path"], c["pair"], c["role"]):c for c in contexts}
    rows = []
    for ep in episodes:
        key = (ep["video_path"], ep["fly"], ep["role"])
        context = lookup[key]
        geometry = next(g for g in context["geometries"] if g["training"] == ep["training"])
        start, stop = ep["review_start_frame"], ep["review_stop_frame_exclusive"]
        events = events_by_context[key][ep["training"], ep["circle"]]
        recorded = [int(f) for f in events if start <= f < stop]
        missing_frames = int(np.count_nonzero(np.isnan(context["x"][start:stop])))
        interval_type = ("fully_tracked_comparison" if ep["source"]=="observed_control" else
                         "gap_in_scored_interval" if missing_frames else "gap_in_context_only")
        identity = json.dumps([*key, ep["training"], ep["circle"], start, stop])
        rows.append(dict(review_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                         **{k:v for k,v in ep.items() if k not in ("strata", "selected_gap_ids")},
                         strata=";".join(sorted(ep["strata"])), selected_gap_ids=json.dumps(ep["selected_gap_ids"]),
                         fps=context["fps"], sync_buckets=bucket_labels(geometry, start, stop),
                         duration_s=(stop-start)/context["fps"], calculated_entry_count=len(recorded),
                         calculated_entry_frames=json.dumps(recorded), scored_missing_frames=missing_frames,
                         interval_type=interval_type, cohort=context.get("cohort", "")))
    return sorted(rows, key=lambda r:(r["video_path"], r["fly"], r["role"], r["review_start_frame"], r["circle"]))


def match_events(calculated, observed, tolerance):
    """Order-preserving temporal matching; count differences are reported separately."""
    a, b = sorted(calculated), sorted(observed)
    i = j = 0
    matches, unmatched_a, unmatched_b = [], [], []
    while i < len(a) and j < len(b):
        if abs(a[i]-b[j]) <= tolerance:
            matches.append((a[i], b[j]))
            i += 1
            j += 1
        elif a[i] < b[j]:
            unmatched_a.append(a[i])
            i += 1
        else:
            unmatched_b.append(b[j])
            j += 1
    return matches, unmatched_a+a[i:], unmatched_b+b[j:]


def compare_annotations(episodes, annotations, tolerance=2, *, require_confidence=True):
    """Validate a complete annotation set and compare counts separately from time.

    Pending/unscorable rows retain missing observed totals; an explicit empty
    list in a reviewed row means zero entries. Legacy ODS imports may opt out of
    requiring confidence, but invalid nonempty confidence values still fail.
    """
    index = {r["review_id"]:r for r in episodes}
    if len(index) != len(episodes):
        raise ValueError("Duplicate review IDs in episode file")
    seen, results = set(), []
    for row in annotations:
        ident = row["review_id"]
        if ident in seen or ident not in index:
            raise ValueError(f"Duplicate or unknown review ID: {ident}")
        seen.add(ident)
        status = row["review_status"].strip()
        if status not in ("pending", "reviewed", "unscorable"):
            raise ValueError(f"Invalid review status: {ident}")
        ep = index[ident]
        observed = []
        if status == "reviewed":
            try:
                observed = json.loads(row["observed_entry_frames"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Reviewed interval requires a JSON frame list (use [] for zero): {ident}") from exc
            if not isinstance(observed, list) or any(type(f) is not int for f in observed):
                raise ValueError(f"Entry frames must be a list of integers: {ident}")
            if len(set(observed)) != len(observed):
                raise ValueError(f"Duplicate observed frames: {ident}")
            if any(not int(ep["review_start_frame"]) <= f < int(ep["review_stop_frame_exclusive"]) for f in observed):
                raise ValueError(f"Observed frame is outside scored interval: {ident}")
            if row["confidence"] not in ("high", "medium", "low") and (require_confidence or row["confidence"]):
                raise ValueError(f"Reviewed interval requires high/medium/low confidence: {ident}")
        elif row["observed_entry_frames"].strip():
            raise ValueError(f"Only reviewed intervals may contain observed entries: {ident}")
        calculated = json.loads(ep["calculated_entry_frames"])
        matches, unmatched_calculated, unmatched_observed = match_events(calculated, observed, tolerance)
        results.append(dict(review_id=ident, video_path=ep["video_path"], fly=ep["fly"], role=ep["role"],
                            circle=ep["circle"], source=ep["source"], review_status=status,
                            reviewed_frames=int(ep["review_stop_frame_exclusive"])-int(ep["review_start_frame"]) if status=="reviewed" else 0,
                            confidence=row["confidence"], calculated_count=len(calculated),
                            observed_count=len(observed) if status=="reviewed" else None,
                            observed_minus_calculated=len(observed)-len(calculated) if status=="reviewed" else None,
                            matched_events=len(matches) if status=="reviewed" else None,
                            unmatched_calculated_frames=json.dumps(unmatched_calculated) if status=="reviewed" else "",
                            unmatched_observed_frames=json.dumps(unmatched_observed) if status=="reviewed" else "",
                            calculated_minus_observed_frame_offsets=json.dumps([a-b for a,b in matches]) if status=="reviewed" else "",
                            reviewer=row.get("reviewer", ""),
                            notes=row["notes"]))
    if seen != set(index):
        raise ValueError("Annotations must contain every review ID; retain pending rows")
    return results
