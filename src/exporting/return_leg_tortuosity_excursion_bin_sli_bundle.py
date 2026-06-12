from __future__ import annotations

import os

import numpy as np

from src.analysis.between_reward_filters import (
    mask_metric_by_min_between_reward_trajectories,
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    mask_by_exp_target_sync_bucket_filter,
)
from src.analysis.sli_bundle_utils import (
    validate_return_leg_tortuosity_excursion_bin_bundle,
    validate_sli_bundle,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.plotting.between_reward_segment_binning import (
    sync_bucket_window,
    wall_contact_mask,
)
from src.plotting.between_reward_segment_metrics import tortuosity_metric_masked
from src.utils.parsers import parse_distances, parse_training_selector


def _selected_training_indices(vas, opts) -> list[int]:
    n_trn = len(getattr(vas[0], "trns", []) or []) if vas else 0
    raw = getattr(opts, "return_leg_tortuosity_excursion_bin_trainings", None)
    if not raw:
        return list(range(n_trn))

    selected = sorted(
        {
            int(idx1) - 1
            for idx1 in parse_training_selector(raw)
            if 0 <= int(idx1) - 1 < n_trn
        }
    )
    if selected:
        return selected
    print(
        "[export] WARNING: --return-leg-tortuosity-excursion-bin-trainings "
        "selected no valid trainings; falling back to all trainings."
    )
    return list(range(n_trn))


def _parse_edges(opts) -> tuple[np.ndarray, bool]:
    raw_new = getattr(opts, "return_leg_tortuosity_excursion_bin_radii_mm", None)
    raw_old = getattr(opts, "return_leg_tortuosity_excursion_bin_edges_mm", None)
    legacy = not (raw_new is not None and str(raw_new).strip())
    raw = raw_old if legacy else raw_new
    if raw is None or not str(raw).strip():
        raise ValueError(
            "Provide --return-leg-tortuosity-excursion-bin-radii-mm, "
            "--return-leg-tortuosity-excursion-bin-edges-mm, or a pair option."
        )

    edges = np.asarray(parse_distances(str(raw)), dtype=float)
    if edges.size < 2:
        raise ValueError("Return-leg tortuosity bin edges require at least two values.")
    if np.any(np.isnan(edges)) or np.any(np.isneginf(edges)):
        raise ValueError("Return-leg tortuosity bin edges must be numeric.")
    if np.any(np.isposinf(edges[:-1])):
        raise ValueError("'inf' may only be the final return-leg tortuosity bin edge.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Return-leg tortuosity bin edges must be strictly increasing.")
    return edges, legacy


def _parse_pairs(opts) -> tuple[np.ndarray, np.ndarray, bool] | None:
    raw_new = getattr(opts, "return_leg_tortuosity_excursion_bin_radius_pairs_mm", None)
    raw_old = getattr(opts, "return_leg_tortuosity_excursion_bin_pairs_mm", None)
    legacy = not (raw_new is not None and str(raw_new).strip())
    raw = raw_old if legacy else raw_new
    if raw is None or not str(raw).strip():
        return None

    lower: list[float] = []
    upper: list[float] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Return-leg tortuosity pair bins use lo:hi format, "
                "e.g. '3:5,8:10,13:15'."
            )
        lo_raw, hi_raw = item.split(":", 1)
        lo = float(parse_distances(lo_raw.strip())[0])
        hi = float(parse_distances(hi_raw.strip())[0])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("Return-leg tortuosity pair bounds must be finite.")
        if lo < 0 or hi <= lo:
            raise ValueError(
                "Each return-leg tortuosity pair requires 0 <= lower < upper."
            )
        lower.append(lo)
        upper.append(hi)

    if not lower:
        raise ValueError("At least one return-leg tortuosity pair bin is required.")
    return np.asarray(lower, float), np.asarray(upper, float), legacy


def _effective_windowing(opts) -> tuple[int, int]:
    raw_skip = getattr(
        opts, "return_leg_tortuosity_excursion_bin_skip_first_sync_buckets", None
    )
    raw_keep = getattr(
        opts, "return_leg_tortuosity_excursion_bin_keep_first_sync_buckets", None
    )
    skip = getattr(opts, "skip_first_sync_buckets", 0) if raw_skip is None else raw_skip
    keep = getattr(opts, "keep_first_sync_buckets", 0) if raw_keep is None else raw_keep
    return max(0, int(skip or 0)), max(0, int(keep or 0))


def _px_per_mm(va, traj) -> float | None:
    for get_value in (
        lambda: float(traj.pxPerMmFloor * va.xf.fctr),
        lambda: float(va.xf.fctr * va.ct.pxPerMmFloor()),
    ):
        try:
            value = get_value()
        except Exception:
            continue
        if np.isfinite(value) and value > 0:
            return value
    return None


def _nonwalk_mask(opts, traj, *, fi: int, n_frames: int):
    enabled = bool(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_exclude_nonwalking_frames",
            False,
        )
    )
    walking = getattr(traj, "walking", None)
    if not enabled or walking is None:
        return None

    s0 = max(0, min(int(fi), len(walking)))
    e0 = max(0, min(int(fi + n_frames), len(walking)))
    window = np.zeros((int(max(1, n_frames)),), dtype=bool)
    if e0 > s0:
        values = np.asarray(walking[s0:e0], dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        window[: len(values)] = values > 0
    return ~window


def _return_leg_start_mode(opts) -> str:
    mode = str(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_return_start_mode",
            "global_max",
        )
        or "global_max"
    ).strip().lower()
    if mode not in {"global_max", "post_last_wall_max"}:
        raise ValueError(
            "Unknown return-leg start mode "
            f"{mode!r}; expected 'global_max' or 'post_last_wall_max'."
        )
    return mode


def _post_last_wall_max_frame(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    wc,
    nonwalk_mask,
    exclude_nonwalk: bool,
    fallback_max_i: int,
    reward_center_xy: tuple[float, float],
) -> int | None:
    if wc is None:
        return None

    s = int(s)
    e = int(e)
    if e <= s:
        return None

    wc_start = max(0, min(s - int(fi), len(wc)))
    wc_stop = max(0, min(e - int(fi), len(wc)))
    wall_idx = np.flatnonzero(np.asarray(wc[wc_start:wc_stop], dtype=bool))
    if wall_idx.size == 0:
        return max(s, int(fallback_max_i))

    search_start = int(fi) + wc_start + int(wall_idx[-1]) + 1
    if search_start >= e:
        return None

    frames = np.arange(search_start, e, dtype=int)
    max_frame = min(len(traj.x), len(traj.y))
    frames = frames[(frames >= 0) & (frames < max_frame)]
    if frames.size == 0:
        return None

    xs = np.asarray(traj.x[frames], dtype=float)
    ys = np.asarray(traj.y[frames], dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    if exclude_nonwalk and nonwalk_mask is not None:
        offsets = frames - int(fi)
        in_mask = (offsets >= 0) & (offsets < len(nonwalk_mask))
        walking = np.zeros(frames.size, dtype=bool)
        walking[in_mask] = ~np.asarray(nonwalk_mask, dtype=bool)[offsets[in_mask]]
        keep &= walking
    if not np.any(keep):
        return None

    kept_frames = frames[keep]
    kept_x = xs[keep]
    kept_y = ys[keep]
    cx, cy = reward_center_xy
    radial_sq = (kept_x - float(cx)) ** 2 + (kept_y - float(cy)) ** 2
    return int(kept_frames[int(np.argmax(radial_sq))])


def _collect_records(
    vas,
    opts,
    *,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    legacy_distances: bool,
):
    exclude_wall = bool(
        getattr(opts, "return_leg_tortuosity_excursion_bin_exclude_wall_contact", False)
    )
    return_start_mode = _return_leg_start_mode(opts)
    if exclude_wall and return_start_mode == "post_last_wall_max":
        raise ValueError(
            "--return-leg-tortuosity-excursion-bin-exclude-wall-contact cannot be "
            "combined with --return-leg-tortuosity-excursion-bin-return-start-mode "
            "post_last_wall_max. Use one wall treatment at a time."
        )
    needs_wall_mask = exclude_wall or return_start_mode == "post_last_wall_max"
    exclude_nonwalk = bool(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_exclude_nonwalking_frames",
            False,
        )
    )
    min_walk_frames = max(
        2,
        int(
            getattr(opts, "return_leg_tortuosity_excursion_bin_min_walk_frames", 2) or 2
        ),
    )
    min_radius_mm = float(
        getattr(opts, "return_leg_tortuosity_excursion_bin_min_radius_mm", 0.0) or 0.0
    )
    min_displacement_mm = float(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_min_displacement_mm",
            0.0,
        )
        or 0.0
    )
    metric_mode = str(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_metric_mode",
            "path_over_max_radius",
        )
        or "path_over_max_radius"
    )
    debug = bool(getattr(opts, "return_leg_tortuosity_excursion_bin_debug", False))

    records = [[[] for _ in range(2)] for _ in vas]
    windows_meta = [[] for _ in vas]
    warned_missing_wc = [False]

    for vi, va in enumerate(vas):
        for role_idx, f in enumerate((0, 1)):
            if f >= len(getattr(va, "trx", [])):
                continue
            if role_idx == 1 and getattr(va, "noyc", False):
                continue
            traj = va.trx[f]
            if traj is None or traj.bad():
                continue

            for t_idx in selected_trainings:
                if t_idx >= len(getattr(va, "trns", [])):
                    continue
                trn = va.trns[t_idx]
                if trn is None or not trn.isCircle():
                    continue

                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=f,
                    skip_first=skip_first,
                    keep_first=keep_first,
                    use_exclusion_mask=False,
                )
                if n_buckets <= 0:
                    continue
                n_frames = int(max(1, n_buckets * df))
                windows_meta[vi].append(
                    {
                        "role": "exp" if role_idx == 0 else "ctrl",
                        "training_idx": int(t_idx),
                        "training_name": trn.name(),
                        "start": int(fi),
                        "stop": int(fi + n_frames),
                    }
                )

                wc = wall_contact_mask(
                    opts,
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    log_tag="return_leg_tortuosity_excursion_bin",
                    warned_missing_wc=warned_missing_wc,
                    enabled=needs_wall_mask,
                )
                nonwalk = _nonwalk_mask(opts, traj, fi=fi, n_frames=n_frames)
                pxmm = _px_per_mm(va, traj)
                if pxmm is None:
                    continue
                try:
                    cx, cy, radius_px = trn.circles(f)[0]
                except Exception:
                    continue
                reward_center_xy = (float(cx), float(cy))
                reward_radius_mm = float(radius_px) / pxmm

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    f,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=0.0,
                    exclude_wall=exclude_wall,
                    wc=wc,
                    exclude_nonwalk=exclude_nonwalk,
                    nonwalk_mask=nonwalk,
                    min_walk_frames=min_walk_frames,
                    dist_stats=("max",),
                    debug=False,
                    yield_skips=False,
                ):
                    s = int(getattr(seg, "s", -1))
                    e = int(getattr(seg, "e", -1))
                    max_i = getattr(seg, "max_d_i", None)
                    radial_mm = float(getattr(seg, "max_d_mm", np.nan))
                    if max_i is None or e <= s + 1 or not np.isfinite(radial_mm):
                        continue
                    if legacy_distances:
                        radial_mm = max(0.0, radial_mm - reward_radius_mm)

                    if return_start_mode == "post_last_wall_max":
                        metric_start = _post_last_wall_max_frame(
                            traj=traj,
                            s=s,
                            e=e,
                            fi=fi,
                            wc=wc,
                            nonwalk_mask=nonwalk,
                            exclude_nonwalk=exclude_nonwalk,
                            fallback_max_i=int(max_i),
                            reward_center_xy=reward_center_xy,
                        )
                        if metric_start is None:
                            continue
                    else:
                        metric_start = max(s, int(max_i))
                    if e <= metric_start:
                        continue
                    tortuosity = tortuosity_metric_masked(
                        traj=traj,
                        s=metric_start,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=pxmm,
                        mode=metric_mode,
                        reward_center_xy=reward_center_xy,
                        min_keep_frames=min_walk_frames,
                        min_displacement_mm=min_displacement_mm,
                        min_radius_mm=min_radius_mm,
                    )
                    if np.isfinite(tortuosity):
                        records[vi][role_idx].append(
                            (float(radial_mm), float(tortuosity))
                        )

        if debug:
            print(
                "[return_leg_tortuosity_excursion_bin] "
                f"{getattr(va, 'fn', f'va_{vi}')}: "
                f"exp_segments={len(records[vi][0])}, "
                f"ctrl_segments={len(records[vi][1])}"
            )
    return records, windows_meta


def _resolved_bins(
    opts, records
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    pair_spec = _parse_pairs(opts)
    if pair_spec is not None:
        lower, upper, legacy = pair_spec
        flattened = np.empty((2 * len(lower),), dtype=float)
        flattened[0::2] = lower
        flattened[1::2] = upper
        return lower, upper, flattened, True, legacy

    requested, legacy = _parse_edges(opts)
    resolved = requested.copy()
    open_ended = bool(np.isposinf(resolved[-1]))
    if open_ended:
        observed = [
            radial
            for video_roles in records
            for role_records in video_roles
            for radial, _tortuosity in role_records
            if np.isfinite(radial)
        ]
        max_observed = float(np.max(observed)) if observed else np.nan
        if not np.isfinite(max_observed) or max_observed <= float(resolved[-2]):
            raise ValueError(
                "The open-ended return-leg tortuosity bin has no observed value "
                "above its lower edge."
            )
        resolved[-1] = max_observed
    return resolved[:-1], resolved[1:], resolved, False, legacy


def _top_fraction(opts) -> float:
    raw = getattr(
        opts,
        "return_leg_tortuosity_excursion_bin_top_fraction",
        1.0,
    )
    fraction = 1.0 if raw is None else float(raw)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "--return-leg-tortuosity-excursion-bin-top-fraction must be in (0, 1]."
        )
    return fraction


def _aggregate_records(
    records,
    lower,
    upper,
    *,
    min_segments: int,
    top_fraction: float = 1.0,
):
    n_videos = len(records)
    n_bins = len(lower)
    means = [
        np.full((n_videos, n_bins), np.nan, dtype=float),
        np.full((n_videos, n_bins), np.nan, dtype=float),
    ]
    counts = [
        np.zeros((n_videos, n_bins), dtype=int),
        np.zeros((n_videos, n_bins), dtype=int),
    ]
    selected_counts = [
        np.zeros((n_videos, n_bins), dtype=int),
        np.zeros((n_videos, n_bins), dtype=int),
    ]

    for vi, video_roles in enumerate(records):
        for role_idx, role_records in enumerate(video_roles):
            for bin_idx, (lo, hi) in enumerate(zip(lower, upper)):
                values = np.asarray(
                    [
                        tort
                        for radial, tort in role_records
                        if float(lo) <= radial < float(hi)
                    ],
                    dtype=float,
                )
                counts[role_idx][vi, bin_idx] = int(values.size)
                if values.size:
                    n_selected = max(
                        1, int(np.ceil(float(top_fraction) * values.size))
                    )
                    selected = np.sort(values)[-n_selected:]
                    selected_counts[role_idx][vi, bin_idx] = int(selected.size)
                    means[role_idx][vi, bin_idx] = float(np.mean(selected))
            means[role_idx][vi] = mask_metric_by_min_between_reward_trajectories(
                means[role_idx][vi],
                counts[role_idx][vi],
                max(1, int(min_segments)),
            )
    return (
        means[0],
        means[1],
        counts[0],
        counts[1],
        selected_counts[0],
        selected_counts[1],
    )


def export_return_leg_tortuosity_excursion_bin_sli_bundle(vas, opts, gls, out_fn):
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    pair_spec = _parse_pairs(opts)
    return_start_mode = _return_leg_start_mode(opts)
    legacy_distances = pair_spec[2] if pair_spec is not None else _parse_edges(opts)[1]
    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first = _effective_windowing(opts)
    records, windows_meta = _collect_records(
        vas_ok,
        opts,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        legacy_distances=legacy_distances,
    )
    lower, upper, resolved_edges, pair_mode, legacy_distances = _resolved_bins(
        opts, records
    )
    min_segments = min_between_reward_sync_bucket_trajectories(opts)
    top_fraction = _top_fraction(opts)
    mean_exp, mean_ctrl, n_exp, n_ctrl, selected_n_exp, selected_n_ctrl = (
        _aggregate_records(
            records,
            lower,
            upper,
            min_segments=min_segments,
            top_fraction=top_fraction,
        )
    )

    n_videos = len(vas_ok)
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as exc:
        print(
            "[export] WARNING: failed to compute SLI for return-leg tortuosity "
            f"excursion-bin bundle: {exc}"
        )
        sli = np.full((n_videos,), np.nan, dtype=float)
        sli_ts = np.full(
            (n_videos, len(getattr(vas_ok[0], "trns", []) or []), 0),
            np.nan,
        )

    target_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    mean_exp = mask_by_exp_target_sync_bucket_filter(mean_exp, target_eligible)
    sli = mask_by_exp_target_sync_bucket_filter(sli, target_eligible)
    sli_ts = mask_by_exp_target_sync_bucket_filter(sli_ts, target_eligible)

    try:
        training_names = np.asarray([t.name() for t in vas_ok[0].trns], dtype=str)
    except Exception:
        training_names = np.asarray([], dtype=str)
    video_fns = np.asarray([getattr(va, "fn", "") for va in vas_ok], dtype=str)
    fly_ids = np.asarray([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
    video_ids = np.asarray(
        [f"{fn}::f{fly_id}" for fn, fly_id in zip(video_fns, fly_ids)],
        dtype=str,
    )
    window_summary = np.asarray(
        [
            "; ".join(
                f"{w['role']} T{w['training_idx'] + 1} {w['training_name']}"
                f"[{w['start']},{w['stop']})"
                for w in video_windows
            )
            for video_windows in windows_meta
        ],
        dtype=str,
    )

    requested = (
        np.asarray(resolved_edges, dtype=float) if pair_mode else _parse_edges(opts)[0]
    )
    payload = dict(
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        group_label=np.asarray(_safe_group_label(opts, gls)),
        bucket_len_min=np.array(np.nan, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        video_fns=video_fns,
        fly_ids=fly_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
        sli_select_skip_first_sync_buckets=np.array(
            max(0, int(getattr(opts, "sli_select_skip_first_sync_buckets", 0) or 0)),
            dtype=int,
        ),
        sli_select_keep_first_sync_buckets=np.array(
            max(0, int(getattr(opts, "sli_select_keep_first_sync_buckets", 0) or 0)),
            dtype=int,
        ),
        return_leg_tortuosity_excursion_bin_exp=mean_exp,
        return_leg_tortuosity_excursion_bin_ctrl=mean_ctrl,
        return_leg_tortuosity_excursion_binN_exp=n_exp,
        return_leg_tortuosity_excursion_binN_ctrl=n_ctrl,
        return_leg_tortuosity_excursion_bin_selectedN_exp=selected_n_exp,
        return_leg_tortuosity_excursion_bin_selectedN_ctrl=selected_n_ctrl,
        return_leg_tortuosity_excursion_bin_top_fraction=np.array(
            top_fraction, dtype=float
        ),
        return_leg_tortuosity_excursion_bin_aggregation=np.asarray(
            "mean" if top_fraction == 1.0 else "top_fraction_mean"
        ),
        return_leg_tortuosity_excursion_bin_edges_mm=np.asarray(
            resolved_edges, dtype=float
        ),
        return_leg_tortuosity_excursion_bin_requested_edges_mm=requested,
        return_leg_tortuosity_excursion_bin_pair_lower_mm=np.asarray(
            lower if pair_mode else [], dtype=float
        ),
        return_leg_tortuosity_excursion_bin_pair_upper_mm=np.asarray(
            upper if pair_mode else [], dtype=float
        ),
        return_leg_tortuosity_excursion_bin_pair_mode=np.array(pair_mode, dtype=bool),
        return_leg_tortuosity_excursion_bin_open_ended_upper_bin=np.array(
            (not pair_mode) and bool(np.isposinf(requested[-1])),
            dtype=bool,
        ),
        return_leg_tortuosity_excursion_bin_legacy_distance_from_circle=np.array(
            legacy_distances, dtype=bool
        ),
        return_leg_tortuosity_excursion_bin_trainings=np.asarray(
            selected_trainings, dtype=int
        ),
        return_leg_tortuosity_excursion_bin_skip_first_sync_buckets=np.array(
            skip_first, dtype=int
        ),
        return_leg_tortuosity_excursion_bin_keep_first_sync_buckets=np.array(
            keep_first, dtype=int
        ),
        return_leg_tortuosity_excursion_bin_exclude_wall_contact=np.array(
            bool(
                getattr(
                    opts,
                    "return_leg_tortuosity_excursion_bin_exclude_wall_contact",
                    False,
                )
            ),
            dtype=bool,
        ),
        return_leg_tortuosity_excursion_bin_exclude_nonwalking_frames=np.array(
            bool(
                getattr(
                    opts,
                    "return_leg_tortuosity_excursion_bin_exclude_nonwalking_frames",
                    False,
                )
            ),
            dtype=bool,
        ),
        return_leg_tortuosity_excursion_bin_min_walk_frames=np.array(
            max(
                2,
                int(
                    getattr(
                        opts,
                        "return_leg_tortuosity_excursion_bin_min_walk_frames",
                        2,
                    )
                    or 2
                ),
            ),
            dtype=int,
        ),
        return_leg_tortuosity_excursion_bin_min_radius_mm=np.array(
            float(
                getattr(
                    opts,
                    "return_leg_tortuosity_excursion_bin_min_radius_mm",
                    0.0,
                )
                or 0.0
            )
        ),
        return_leg_tortuosity_excursion_bin_min_displacement_mm=np.array(
            float(
                getattr(
                    opts,
                    "return_leg_tortuosity_excursion_bin_min_displacement_mm",
                    0.0,
                )
                or 0.0
            )
        ),
        return_leg_tortuosity_excursion_bin_metric_mode=np.asarray(
            str(
                getattr(
                    opts,
                    "return_leg_tortuosity_excursion_bin_metric_mode",
                    "path_over_max_radius",
                )
            )
        ),
        return_leg_tortuosity_excursion_bin_return_start_mode=np.asarray(
            return_start_mode
        ),
        return_leg_tortuosity_excursion_bin_window_summary=window_summary,
        return_leg_tortuosity_excursion_bin_description=np.asarray(
            (
                "Mean return-leg tortuosity of all between-reward trajectories "
                "binned by maximum radial distance"
                if top_fraction == 1.0
                else f"Mean of the top {100.0 * top_fraction:g}% return-leg "
                "tortuosity values per fly and maximum-distance bin"
            )
            + (
                "; return leg starts at the full-episode maximum distance"
                if return_start_mode == "global_max"
                else "; return leg starts at the maximum distance after the "
                "episode's final wall contact"
            )
        ),
        btw_rwd_sync_bucket_min_trajectories=np.array(min_segments, dtype=int),
        **exp_target_sync_bucket_filter_payload(
            vas_ok, opts, prefix="exp_target_sync_bucket_filter"
        ),
    )

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    validate_sli_bundle(payload, path=out_fn)
    validate_return_leg_tortuosity_excursion_bin_bundle(payload, path=out_fn)
    np.savez_compressed(out_fn, **payload)
    print(
        "[export] Wrote return-leg-tortuosity-excursion-bin+SLI bundle: "
        f"{out_fn} (n={n_videos}, bins={len(lower)})"
    )
