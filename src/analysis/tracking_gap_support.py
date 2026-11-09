"""Exhaustive same-subject reference support, without injecting artificial loss."""
from __future__ import annotations

import numpy as np

from src.analysis.tracking_gap_simulation import (
    true_runs, candidate_starts, endpoint_features, matching_candidates,
    duration_bin, exposed_candidates,
)


def observed_references(observed, episodes, lo, hi):
    """Complete support in one original observed run; never clip an episode."""
    runs = true_runs(observed, lo, hi)
    run_starts = np.array([a for a, _ in runs])
    retained = []
    for episode in episodes:
        a, b = int(episode['support_start']), int(episode['support_stop'])
        parent = int(np.searchsorted(run_starts, a, side='right')-1)
        if parent >= 0 and runs[parent][0] <= a < b <= runs[parent][1]:
            retained.append(dict(episode, parent=parent))
    return runs, retained


def audit_window(raw, fps, lo, hi, geometry, metric_episodes, *, padding_seconds=2., tolerance_mm=2.):
    """Return per-gap/per-metric support and window/episode inventories.

    Geometry uses only observed donor endpoints. All donor gaps and candidate
    starts are enumerated. A candidate's parent must contain complete original
    episode support for an episode to be exposed. Minimum-five support is local
    to that same parent, not the whole manuscript measurement window.
    """
    if not (np.isfinite(fps) and fps > 0 and 0 <= lo < hi <= len(raw['x'])):
        raise ValueError('Invalid frame rate or window')
    observed = np.logical_and.reduce([np.isfinite(raw[name]) for name in ('x', 'y', 'w', 'h', 'theta')])
    missing = np.isnan(raw['x'])
    runs = true_runs(observed, lo, hi)
    padding = int(np.ceil(padding_seconds*fps))
    overlap = [(a, b) for a, b in true_runs(missing) if a < hi and b > lo]
    donors = [(a, b) for a, b in overlap if lo <= a-1 and b < hi
              and np.isfinite([raw['x'][a-1], raw['y'][a-1], raw['x'][b], raw['y'][b]]).all()]
    lost_frames = sum(min(b, hi)-max(a, lo) for a, b in overlap)
    donor_frames = sum(b-a for a, b in donors)
    spacing = [(overlap[i+1][0]-overlap[i][1])/fps for i in range(len(overlap)-1)]
    stats = dict(start_frame=lo, stop_frame_exclusive=hi, window_s=(hi-lo)/fps,
        gap_count=len(overlap), lost_frames=lost_frames, lost_seconds=lost_frames/fps,
        loss_fraction=lost_frames/(hi-lo), donor_gaps=len(donors), donor_lost_seconds=donor_frames/fps,
        excluded_gap_count=len(overlap)-len(donors), excluded_lost_seconds=(lost_frames-donor_frames)/fps,
        fully_observed_runs=len(runs), longest_observed_run_s=max(((b-a)/fps for a, b in runs), default=0.),
        fully_observed_window=bool(observed[lo:hi].all()),
        next_gap_within_1s_fraction=float(np.mean(np.array(spacing) <= 1)) if spacing else np.nan,
        invalid_other_field_frames=int((~observed[lo:hi] & ~missing[lo:hi]).sum()))
    masks, heading_masks, counts, references, episode_rows = {}, {}, {}, {}, []
    for key, episodes in metric_episodes.items():
        _, retained = observed_references(observed, episodes, lo, hi)
        references[key] = retained
        masks[key] = np.zeros(len(observed), dtype=bool)
        heading_masks[key] = np.zeros(len(observed), dtype=bool)
        counts[key] = np.zeros(len(runs), dtype=int)
        for index, ep in enumerate(retained):
            masks[key][ep['support_start']:ep['support_stop']] = True
            heading_masks[key][ep.get('heading_frames', [])] = True
            counts[key][ep['parent']] += 1
            episode_rows.append(dict(metric=key[0], variant=key[1], reference_episode_index=index, start_frame=ep['start'],
                stop_frame_exclusive=ep['stop'], support_start_frame=ep['support_start'],
                support_stop_frame_exclusive=ep['support_stop'], parent_run=ep['parent'],
                heading_frames=ep.get('heading_frames', [])))
    rows = []
    # Cache placements/features by duration only within this window, limiting memory.
    for length in sorted({b-a for a, b in donors}):
        starts, parents = candidate_starts(runs, length, padding)
        features = endpoint_features(raw['x'], raw['y'], starts, length, **geometry)
        exposure = {key: np.zeros(len(starts), dtype=bool) for key in metric_episodes}
        heading = {key: np.zeros(len(starts), dtype=bool) for key in metric_episodes}
        for key in metric_episodes:
            exposure[key][exposed_candidates(starts, length, masks[key])] = True
            heading[key][exposed_candidates(starts, length, heading_masks[key])] = True
        for a, b in [(a, b) for a, b in donors if b-a == length]:
            target = {name: float(values[0]) for name, values in endpoint_features(
                raw['x'], raw['y'], [a], length, **geometry).items()}
            matched = matching_candidates(features, target, tolerance_mm)
            for key in metric_episodes:
                exposed = matched[exposure[key][matched]]
                qualified = exposed[counts[key][parents[exposed]] >= 5]
                touched = set()
                # A reference episode is usable if any matched interval intersects support.
                matched_starts = starts[matched]
                matched_parents = parents[matched]
                for index, ep in enumerate(references[key]):
                    if np.any((matched_parents == ep['parent']) & (matched_starts < ep['support_stop'])
                              & (matched_starts+length > ep['support_start'])):
                        touched.add(index)
                row = dict(metric=key[0], variant=key[1], gap_start_frame=a,
                    gap_stop_frame_exclusive=b, duration_s=length/fps, duration_bin=duration_bin(length/fps),
                    wall_context='endpoint <=1mm' if min(target['before_wall_mm'], target['after_wall_mm']) <= 1 else 'other endpoints',
                    before_wall_bin=int(target['before_wall_bin']), after_wall_bin=int(target['after_wall_bin']),
                    length_candidates=len(starts), spatial_candidates=len(matched),
                    exposed_candidates=len(exposed), minimum_five_candidates=len(qualified),
                    heading_candidates=int(heading[key][matched].sum()) if key[0] == 'home_vector_alignment' else None,
                    source_gap_sampling_probability=1., candidate_search='exhaustive',
                    reference_episode_indices=sorted(touched))
                rows.append(row)
    return stats, rows, episode_rows
