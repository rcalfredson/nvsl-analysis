from __future__ import annotations

import os

import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)


TIMEFRAMES = ("pre_trn", "t1_start", "t2_end", "t3_end")


def _turn_prob_array_for_va(va, distances: list[float]) -> np.ndarray:
    values = np.full((len(distances), 2, len(TIMEFRAMES), 2), np.nan, dtype=float)
    by_dist = getattr(va, "turn_prob_by_distance", None)
    if not by_dist:
        return values

    for di, dist in enumerate(distances):
        if dist not in by_dist:
            continue
        dist_values = by_dist[dist]
        for fly_idx in range(min(2, len(dist_values))):
            fly_values = dist_values[fly_idx]
            for ti in range(min(len(TIMEFRAMES), len(fly_values))):
                pair = fly_values[ti]
                if pair is None or len(pair) < 2:
                    continue
                values[di, fly_idx, ti, 0] = pair[0]
                values[di, fly_idx, ti, 1] = pair[1]
    return values


def export_turn_prob_dist_sli_bundle(vas, opts, gls, out_fn):
    if not vas:
        raise ValueError(
            "No VideoAnalysis instances available for turn-probability export."
        )
    if not hasattr(vas[0], "turn_prob_by_distance"):
        raise ValueError(
            "turn_prob_by_distance is missing; run with --turn-prob-by-dist or "
            "--outside-circle-radii before exporting turn-probability bundles."
        )

    distances = list(getattr(vas[0], "turn_prob_by_distance").keys())
    if not distances:
        raise ValueError("turn_prob_by_distance has no distances to export.")

    n_videos = len(vas)
    turn_prob = np.full(
        (n_videos, len(distances), 2, len(TIMEFRAMES), 2), np.nan, dtype=float
    )
    for vi, va in enumerate(vas):
        turn_prob[vi] = _turn_prob_array_for_va(va, distances)

    sli_scalar, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas, opts)
    group_labels = np.asarray(gls or [_safe_group_label(opts, gls)], dtype=object)
    group_indices = np.asarray([getattr(va, "gidx", 0) for va in vas], dtype=int)
    video_ids = np.asarray(
        [getattr(va, "fn", f"va_{idx}") for idx, va in enumerate(vas)],
        dtype=object,
    )

    payload = dict(
        schema=np.array("turn_prob_dist_sli_bundle_v1"),
        group_label=np.array(_safe_group_label(opts, gls)),
        group_labels=group_labels,
        group_indices=group_indices,
        video_ids=video_ids,
        sli=np.asarray(sli_scalar, dtype=float),
        sli_timeseries=np.asarray(sli_ts, dtype=float),
        turn_prob_distances_mm=np.asarray(distances, dtype=float),
        turn_prob_values=turn_prob,
        turn_prob_timeframes=np.asarray(TIMEFRAMES, dtype=object),
        turn_prob_fly_roles=np.asarray(("exp", "ctrl"), dtype=object),
        turn_prob_directions=np.asarray(("toward", "away"), dtype=object),
        contact_geometry=np.array(getattr(opts, "contact_geometry", "")),
        use_union_filter=np.array(bool(getattr(opts, "use_union_filter", False))),
        turn_contact_thresh=np.array(int(getattr(opts, "turn_contact_thresh", 0))),
        min_vel_angle_delta=np.array(
            float(getattr(opts, "min_vel_angle_delta", np.nan))
        ),
    )

    out_dir = os.path.dirname(out_fn)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_fn, **payload)
    print(f"[export] wrote turn-probability-by-distance bundle: {out_fn}")
