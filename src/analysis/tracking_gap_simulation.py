"""Pure sampling and episode-comparison helpers for tracking-gap experiments."""
from __future__ import annotations

import numpy as np


def true_runs(mask, start=0, stop=None):
    mask = np.asarray(mask, dtype=bool)
    stop = len(mask) if stop is None else int(stop)
    start = int(start)
    if mask.ndim != 1 or not 0 <= start <= stop <= len(mask):
        raise ValueError("Invalid mask interval")
    changes = np.diff(np.r_[False, mask[start:stop], False].astype(int))
    return list(zip(np.flatnonzero(changes == 1)+start,
                    np.flatnonzero(changes == -1)+start))


def duration_bin(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Gap duration must be positive")
    return "<=0.5s" if seconds <= .5 else "0.5-2s" if seconds <= 2 else "2-10s" if seconds <= 10 else ">10s"


def candidate_starts(runs, length, padding):
    """All placements with observed bracketing endpoints and context on both sides."""
    length, padding = int(length), int(padding)
    if length < 1 or padding < 0:
        raise ValueError("Invalid gap length or padding")
    starts, parents = [], []
    for index, (lo, hi) in enumerate(runs):
        # Frame before start and recovery at start+length must remain observed.
        first, last = int(lo)+padding+1, int(hi)-padding-length-1
        if last >= first:
            ss = np.arange(first, last+1, dtype=int)
            starts.append(ss)
            parents.append(np.full(len(ss), index, dtype=int))
    return (np.concatenate(starts), np.concatenate(parents)) if starts else (np.array([],dtype=int),np.array([],dtype=int))


def endpoint_features(x, y, starts, length, *, floor, scale, reward_center, wells):
    """Use only observed endpoints; never treat a donor's hidden path as known."""
    starts = np.asarray(starts, dtype=int)
    before, after = starts-1, starts+int(length)
    (left,top),(right,bottom) = floor
    out = {}
    for name, indices in (("before",before),("after",after)):
        xx, yy = np.asarray(x)[indices], np.asarray(y)[indices]
        wall = np.minimum.reduce([xx-left,right-xx,yy-top,bottom-yy])/scale
        out[name+"_wall_mm"] = wall
        out[name+"_wall_bin"] = np.digitize(wall,[1.,3.])
        out[name+"_reward_radius_mm"] = np.hypot(xx-reward_center[0],yy-reward_center[1])/scale
        radius,centers = wells
        out[name+"_well_signed_mm"] = np.min(
            np.stack([(np.hypot(xx-cx,yy-cy)-radius)/scale for cx,cy in centers]),axis=0)
    out["endpoint_chord_mm"] = np.hypot(np.asarray(x)[after]-np.asarray(x)[before],np.asarray(y)[after]-np.asarray(y)[before])/scale
    return out


def matching_candidates(features, donor, tolerance_mm=2.):
    """Exact before/after wall-distance bins plus bounded continuous differences."""
    if tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be positive")
    keep = np.ones(len(features["endpoint_chord_mm"]),dtype=bool)
    for name,values in features.items():
        if name.endswith("wall_bin"):
            keep &= values == donor[name]
        elif name.endswith("wall_mm"):
            continue  # wall bins are the prespecified wall-distance constraint
        else:
            keep &= np.abs(values-donor[name]) <= tolerance_mm
        keep &= np.isfinite(values) & np.isfinite(donor[name])
    return np.flatnonzero(keep)


def compare_episodes(reference, simulated):
    """Overlap graph diagnostics, without calling timing shifts missed detections."""
    overlap = np.zeros((len(reference),len(simulated)),dtype=bool)
    for i,a in enumerate(reference):
        for j,b in enumerate(simulated):
            overlap[i,j] = max(a["start"],b["start"]) < min(a["stop"],b["stop"])
    nr, ns = overlap.sum(axis=1), overlap.sum(axis=0)
    pairs = [(int(i),int(j)) for i,j in zip(*np.where(overlap)) if nr[i]==1 and ns[j]==1]
    errors = [float(simulated[j]["value"]-reference[i]["value"]) for i,j in pairs]
    return dict(
        reference_episodes=len(reference),simulated_episodes=len(simulated),
        episode_count_change=len(simulated)-len(reference),
        reference_without_overlap=int(np.sum(nr==0)),simulated_without_overlap=int(np.sum(ns==0)),
        reference_overlapping_multiple=int(np.sum(nr>1)),simulated_overlapping_multiple=int(np.sum(ns>1)),
        one_to_one_overlap_pairs=len(pairs),
        paired_value_mean_error=float(np.mean(errors)) if errors else np.nan,
        paired_value_mean_absolute_error=float(np.mean(np.abs(errors))) if errors else np.nan,
        paired_start_mean_absolute_shift_frames=(float(np.mean([abs(simulated[j]["start"]-reference[i]["start"]) for i,j in pairs])) if pairs else np.nan),
        paired_stop_mean_absolute_shift_frames=(float(np.mean([abs(simulated[j]["stop"]-reference[i]["stop"]) for i,j in pairs])) if pairs else np.nan),
    )


def exposed_candidates(starts, length, support_mask):
    """Return candidate indices whose missing interval intersects reference support."""
    prefix = np.r_[0,np.cumsum(np.asarray(support_mask,dtype=bool))]
    starts = np.asarray(starts,dtype=int)
    return np.flatnonzero(prefix[starts+int(length)]-prefix[starts]>0)
