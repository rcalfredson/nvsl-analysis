from __future__ import annotations

import os

import numpy as np

from src.analysis.between_reward_filters import (
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sli_bundle_utils import (
    validate_post_wall_departure_tortuosity_bundle,
    validate_sli_bundle,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    mask_by_exp_target_sync_bucket_filter,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.plotting.between_reward_segment_binning import (
    sync_bucket_window,
    wall_contact_mask,
)
from src.plotting.between_reward_segment_metrics import dist_traveled_mm_masked
from src.utils.parsers import parse_training_selector


def selected_training_indices(vas, opts) -> list[int]:
    n_trn = len(getattr(vas[0], "trns", []) or []) if vas else 0
    raw = getattr(opts, "post_wall_departure_tortuosity_trainings", None)
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
        "[post-wall-departure-tortuosity] warning: training selector matched "
        "no trainings; using all trainings."
    )
    return list(range(n_trn))


def effective_windowing(opts) -> tuple[int, int]:
    skip = getattr(opts, "post_wall_departure_tortuosity_skip_first_sync_buckets", None)
    keep = getattr(opts, "post_wall_departure_tortuosity_keep_first_sync_buckets", None)
    if skip is None:
        skip = getattr(opts, "skip_first_sync_buckets", 0)
    if keep is None:
        keep = getattr(opts, "keep_first_sync_buckets", 0)
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
    if not bool(
        getattr(
            opts,
            "post_wall_departure_tortuosity_exclude_nonwalking_frames",
            False,
        )
    ):
        return None
    walking = getattr(traj, "walking", None)
    if walking is None:
        return None
    start = max(0, min(int(fi), len(walking)))
    stop = max(0, min(int(fi + n_frames), len(walking)))
    window = np.zeros((int(max(1, n_frames)),), dtype=bool)
    if stop > start:
        values = np.asarray(walking[start:stop], dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        window[: len(values)] = values > 0
    return ~window


def last_wall_departure_frame(
    wc,
    *,
    segment_start: int,
    segment_stop: int,
    window_start: int,
) -> tuple[int, int] | None:
    """Return (last wall-contact frame, first subsequent non-contact frame)."""
    if wc is None:
        return None
    s = int(segment_start)
    e = int(segment_stop)
    fi = int(window_start)
    start = max(0, min(s - fi, len(wc)))
    stop = max(0, min(e - fi, len(wc)))
    if stop <= start:
        return None
    contacts = np.flatnonzero(np.asarray(wc[start:stop], dtype=bool))
    if contacts.size == 0:
        return None
    last_contact = fi + start + int(contacts[-1])
    departure = last_contact + 1
    if departure >= e:
        return None
    return int(last_contact), int(departure)


def departure_distance_to_reward_circle_mm(
    traj,
    *,
    departure_frame: int,
    reward_circle,
    px_per_mm: float,
) -> tuple[float, tuple[float, float]] | None:
    """Shortest distance from departure position to the reward-circle perimeter."""
    frame = int(departure_frame)
    if frame < 0 or frame >= min(len(traj.x), len(traj.y)):
        return None
    x = float(traj.x[frame])
    y = float(traj.y[frame])
    cx, cy, radius_px = (float(value) for value in reward_circle)
    if not np.all(np.isfinite([x, y, cx, cy, radius_px, px_per_mm])):
        return None
    dx = x - cx
    dy = y - cy
    radial_px = float(np.hypot(dx, dy))
    if radial_px <= radius_px or radial_px <= 0 or px_per_mm <= 0:
        return None
    direct_mm = float((radial_px - radius_px) / px_per_mm)
    reward_edge_xy = (
        float(cx + radius_px * dx / radial_px),
        float(cy + radius_px * dy / radial_px),
    )
    return direct_mm, reward_edge_xy


def collect_post_wall_departure_tortuosity(vas, opts):
    selected_trainings = selected_training_indices(vas, opts)
    skip_first, keep_first = effective_windowing(opts)
    exclude_nonwalk = bool(
        getattr(
            opts,
            "post_wall_departure_tortuosity_exclude_nonwalking_frames",
            False,
        )
    )
    min_walk_frames = max(
        2,
        int(
            getattr(opts, "post_wall_departure_tortuosity_min_walk_frames", 2) or 2
        ),
    )
    min_direct_mm = max(
        0.0,
        float(
            getattr(
                opts,
                "post_wall_departure_tortuosity_min_direct_distance_mm",
                0.0,
            )
            or 0.0
        ),
    )

    records = [[[] for _ in range(2)] for _ in vas]
    details: list[dict] = []
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
            px_per_mm = _px_per_mm(va, traj)
            if px_per_mm is None:
                continue

            for training_idx in selected_trainings:
                if training_idx >= len(getattr(va, "trns", [])):
                    continue
                trn = va.trns[training_idx]
                if trn is None or not trn.isCircle():
                    continue
                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=training_idx,
                    f=f,
                    skip_first=skip_first,
                    keep_first=keep_first,
                    use_exclusion_mask=False,
                )
                if n_buckets <= 0:
                    continue
                metric_n_frames = int(max(1, n_buckets * df))
                wall_stop = min(
                    min(len(traj.x), len(traj.y)),
                    max(int(fi + metric_n_frames), int(trn.stop)),
                )
                wall_n_frames = int(max(1, wall_stop - fi))
                wc = wall_contact_mask(
                    opts,
                    va,
                    f,
                    fi=fi,
                    n_frames=wall_n_frames,
                    log_tag="post_wall_departure_tortuosity",
                    warned_missing_wc=warned_missing_wc,
                    enabled=True,
                )
                if wc is None:
                    continue
                nonwalk = _nonwalk_mask(
                    opts,
                    traj,
                    fi=fi,
                    n_frames=wall_n_frames,
                )
                try:
                    reward_circle = trn.circles(f)[0]
                except Exception:
                    continue

                windows_meta[vi].append(
                    {
                        "role": "exp" if role_idx == 0 else "ctrl",
                        "training_idx": int(training_idx),
                        "training_name": trn.name(),
                        "start": int(fi),
                        "stop": int(fi + metric_n_frames),
                    }
                )

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    f,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=0.0,
                    exclude_wall=False,
                    wc=None,
                    exclude_nonwalk=False,
                    nonwalk_mask=None,
                    min_walk_frames=2,
                    dist_stats=(),
                    debug=False,
                    yield_skips=False,
                ):
                    s = int(seg.s)
                    e = int(seg.e)
                    departure_info = last_wall_departure_frame(
                        wc,
                        segment_start=s,
                        segment_stop=e,
                        window_start=fi,
                    )
                    if departure_info is None:
                        continue
                    last_contact, departure = departure_info
                    direct_info = departure_distance_to_reward_circle_mm(
                        traj,
                        departure_frame=departure,
                        reward_circle=reward_circle,
                        px_per_mm=px_per_mm,
                    )
                    if direct_info is None:
                        continue
                    direct_mm, reward_edge_xy = direct_info
                    if direct_mm <= min_direct_mm:
                        continue

                    metric_stop = min(e + 1, len(traj.x), len(traj.y))
                    path_mm = dist_traveled_mm_masked(
                        traj=traj,
                        s=departure,
                        e=metric_stop,
                        fi=fi,
                        nonwalk_mask=nonwalk,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        min_keep_frames=min_walk_frames,
                    )
                    if not np.isfinite(path_mm) or path_mm <= 0:
                        continue
                    ratio = float(path_mm / direct_mm)
                    if not np.isfinite(ratio):
                        continue

                    records[vi][role_idx].append(ratio)
                    details.append(
                        {
                            "va": va,
                            "video_index": int(vi),
                            "role_idx": int(role_idx),
                            "trajectory_index": int(f),
                            "traj": traj,
                            "training_idx": int(training_idx),
                            "segment_start": s,
                            "segment_stop": e,
                            "last_wall_contact_frame": int(last_contact),
                            "departure_frame": int(departure),
                            "metric_stop": int(metric_stop),
                            "path_mm": float(path_mm),
                            "direct_mm": float(direct_mm),
                            "tortuosity": ratio,
                            "reward_edge_xy": reward_edge_xy,
                            "departure_xy": (
                                float(traj.x[departure]),
                                float(traj.y[departure]),
                            ),
                            "exclude_nonwalk": exclude_nonwalk,
                        }
                    )
    return records, details, windows_meta


def aggregate_post_wall_departure_tortuosity(records, *, min_episodes: int):
    n_videos = len(records)
    means = [
        np.full((n_videos, 1), np.nan, dtype=float),
        np.full((n_videos, 1), np.nan, dtype=float),
    ]
    counts = [
        np.zeros((n_videos, 1), dtype=int),
        np.zeros((n_videos, 1), dtype=int),
    ]
    threshold = max(1, int(min_episodes))
    for vi, video_roles in enumerate(records):
        for role_idx, values in enumerate(video_roles):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            counts[role_idx][vi, 0] = int(finite.size)
            if finite.size >= threshold:
                means[role_idx][vi, 0] = float(np.mean(finite))
    return means[0], means[1], counts[0], counts[1]


def build_post_wall_departure_tortuosity_sli_bundle(vas, opts, gls):
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        raise ValueError(
            "No non-skipped videos for post-wall departure tortuosity bundle."
        )
    records, details, windows_meta = collect_post_wall_departure_tortuosity(
        vas_ok, opts
    )
    min_episodes = min_between_reward_sync_bucket_trajectories(opts)
    mean_exp, mean_ctrl, n_exp, n_ctrl = (
        aggregate_post_wall_departure_tortuosity(
            records,
            min_episodes=min_episodes,
        )
    )
    n_videos = len(vas_ok)
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as exc:
        print(
            "[post-wall-departure-tortuosity] warning: failed to compute SLI: "
            f"{exc}"
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

    selected_trainings = selected_training_indices(vas_ok, opts)
    skip_first, keep_first = effective_windowing(opts)
    video_fns = np.asarray([getattr(va, "fn", "") for va in vas_ok], dtype=str)
    fly_ids = np.asarray([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
    payload = dict(
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        group_label=np.asarray(_safe_group_label(opts, gls)),
        bucket_len_min=np.asarray(np.nan),
        training_names=np.asarray(
            [trn.name() for trn in getattr(vas_ok[0], "trns", [])],
            dtype=str,
        ),
        video_ids=np.asarray(
            [f"{fn}::f{fly_id}" for fn, fly_id in zip(video_fns, fly_ids)],
            dtype=str,
        ),
        video_fns=video_fns,
        fly_ids=fly_ids,
        sli_training_idx=np.asarray(
            getattr(opts, "best_worst_trn", 1) - 1,
            dtype=int,
        ),
        sli_use_training_mean=np.asarray(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
        sli_select_skip_first_sync_buckets=np.asarray(
            max(
                0,
                int(
                    getattr(opts, "sli_select_skip_first_sync_buckets", 0)
                    or 0
                ),
            ),
            dtype=int,
        ),
        sli_select_keep_first_sync_buckets=np.asarray(
            max(
                0,
                int(
                    getattr(opts, "sli_select_keep_first_sync_buckets", 0)
                    or 0
                ),
            ),
            dtype=int,
        ),
        post_wall_departure_tortuosity_exp=mean_exp,
        post_wall_departure_tortuosity_ctrl=mean_ctrl,
        post_wall_departure_tortuosityN_exp=n_exp,
        post_wall_departure_tortuosityN_ctrl=n_ctrl,
        post_wall_departure_tortuosity_trainings=np.asarray(
            selected_trainings, dtype=int
        ),
        post_wall_departure_tortuosity_skip_first_sync_buckets=np.asarray(
            skip_first, dtype=int
        ),
        post_wall_departure_tortuosity_keep_first_sync_buckets=np.asarray(
            keep_first, dtype=int
        ),
        post_wall_departure_tortuosity_exclude_nonwalking_frames=np.asarray(
            bool(
                getattr(
                    opts,
                    "post_wall_departure_tortuosity_exclude_nonwalking_frames",
                    False,
                )
            )
        ),
        post_wall_departure_tortuosity_min_walk_frames=np.asarray(
            max(
                2,
                int(
                    getattr(
                        opts,
                        "post_wall_departure_tortuosity_min_walk_frames",
                        2,
                    )
                    or 2
                ),
            ),
            dtype=int,
        ),
        post_wall_departure_tortuosity_min_direct_distance_mm=np.asarray(
            max(
                0.0,
                float(
                    getattr(
                        opts,
                        "post_wall_departure_tortuosity_min_direct_distance_mm",
                        0.0,
                    )
                    or 0.0
                ),
            )
        ),
        post_wall_departure_tortuosity_denominator=np.asarray(
            "shortest distance from wall-departure point to reward-circle perimeter"
        ),
        post_wall_departure_tortuosity_window_summary=np.asarray(
            [
                "; ".join(
                    f"{window['role']} T{window['training_idx'] + 1} "
                    f"{window['training_name']}[{window['start']},{window['stop']})"
                    for window in video_windows
                )
                for video_windows in windows_meta
            ],
            dtype=str,
        ),
        btw_rwd_sync_bucket_min_trajectories=np.asarray(
            min_episodes, dtype=int
        ),
        **exp_target_sync_bucket_filter_payload(
            vas_ok,
            opts,
            prefix="exp_target_sync_bucket_filter",
        ),
    )
    validate_sli_bundle(payload)
    validate_post_wall_departure_tortuosity_bundle(payload)
    return payload, details


def export_post_wall_departure_tortuosity_sli_bundle(vas, opts, gls, out_fn):
    payload, _details = build_post_wall_departure_tortuosity_sli_bundle(
        vas, opts, gls
    )
    if not str(out_fn).lower().endswith(".npz"):
        out_fn = f"{out_fn}.npz"
    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(out_fn, **payload)
    print(
        "[export] Wrote post-wall-departure-tortuosity+SLI bundle: "
        f"{out_fn} (n={len(payload['sli'])})"
    )
