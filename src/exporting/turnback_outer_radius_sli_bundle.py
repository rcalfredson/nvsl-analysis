from __future__ import annotations

import os

import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.utils.parsers import parse_distances, parse_training_selector


def _dbg(opts, msg):
    if opts is not None and bool(
        getattr(opts, "turnback_outer_radius_debug", False)
    ):
        print(msg)


def _dbg_if(debug: bool, msg: str):
    if bool(debug):
        print(msg)


def _selected_training_indices(vas, opts) -> list[int]:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "turnback_outer_radius_trainings", None)
    if not raw:
        return list(range(n_trn))

    selected = []
    for idx1 in parse_training_selector(raw):
        idx0 = int(idx1) - 1
        if 0 <= idx0 < n_trn:
            selected.append(idx0)

    selected = sorted(set(selected))
    if selected:
        return selected

    print(
        "[export] WARNING: --turnback-outer-radius-trainings selected no valid "
        "trainings; falling back to all trainings."
    )
    return list(range(n_trn))


def _outer_deltas_mm(opts) -> np.ndarray:
    raw = getattr(opts, "turnback_outer_radius_outer_deltas_mm", None)
    if raw is None or not str(raw).strip():
        raise ValueError(
            "--turnback-outer-radius-outer-deltas-mm is required when exporting "
            "turnback-outer-radius bundles."
        )
    vals = np.asarray(parse_distances(str(raw)), dtype=float)
    if vals.size == 0:
        raise ValueError(
            "--turnback-outer-radius-outer-deltas-mm must contain at least one value."
        )
    return vals


def _effective_windowing(opts) -> tuple[int, int, int]:
    raw_skip = getattr(opts, "turnback_outer_radius_skip_first_sync_buckets", None)
    raw_keep = getattr(opts, "turnback_outer_radius_keep_first_sync_buckets", None)
    skip_first = (
        getattr(opts, "skip_first_sync_buckets", 0) if raw_skip is None else raw_skip
    )
    keep_first = (
        getattr(opts, "keep_first_sync_buckets", 0) if raw_keep is None else raw_keep
    )
    last_sync = int(getattr(opts, "turnback_outer_radius_last_sync_buckets", 0) or 0)
    return max(0, int(skip_first or 0)), max(0, int(keep_first or 0)), max(
        0, last_sync
    )


def _selected_windows_for_va(
    va,
    selected_trainings: list[int],
    *,
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
):
    windows = []
    sync_ranges = getattr(va, "sync_bucket_ranges", None)

    for t_idx in selected_trainings:
        if t_idx >= len(getattr(va, "trns", [])):
            continue
        trn = va.trns[t_idx]
        if trn is None or not trn.isCircle():
            continue

        if sync_ranges and t_idx < len(sync_ranges) and sync_ranges[t_idx]:
            rr = list(sync_ranges[t_idx])
            rr = rr[max(0, int(skip_first)) :]
            if keep_first > 0:
                rr = rr[: int(keep_first)]
            if last_sync_buckets > 0:
                rr = rr[-int(last_sync_buckets) :]
            if not rr:
                continue
            windows.append(
                {
                    "training_idx": int(t_idx),
                    "training_name": trn.name(),
                    "start": int(rr[0][0]),
                    "stop": int(rr[-1][1]),
                    "bucket_ranges": [(int(a), int(b)) for (a, b) in rr],
                }
            )
        else:
            windows.append(
                {
                    "training_idx": int(t_idx),
                    "training_name": trn.name(),
                    "start": int(trn.start),
                    "stop": int(trn.stop),
                    "bucket_ranges": [(int(trn.start), int(trn.stop))],
                }
            )
    return windows


def _frame_in_windows(frame: int, windows) -> bool:
    for win in windows:
        if int(win["start"]) <= frame < int(win["stop"]):
            return True
    return False


def _compute_outer_radius_curves(
    vas,
    *,
    outer_deltas_mm: np.ndarray,
    inner_delta_mm: float,
    border_width_mm: float,
    radius_offset_px: float,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    debug: bool,
):
    n_videos = len(vas)
    n_radii = int(outer_deltas_mm.size)
    ratio_exp = np.full((n_videos, n_radii), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_radii), np.nan, dtype=float)
    turn_exp = np.zeros((n_videos, n_radii), dtype=int)
    turn_ctrl = np.zeros((n_videos, n_radii), dtype=int)
    total_exp = np.zeros((n_videos, n_radii), dtype=int)
    total_ctrl = np.zeros((n_videos, n_radii), dtype=int)
    windows_meta = []

    for vi, va in enumerate(vas):
        vid = getattr(va, "fn", f"va_{vi}")
        windows = _selected_windows_for_va(
            va,
            selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
        )
        windows_meta.append(windows)
        if not windows:
            _dbg_if(
                debug,
                f"[turnback-outer-radius] {vid}: no eligible windows after selection",
            )
            continue

        windows_by_training = {
            int(win["training_idx"]): [win] for win in windows
        }

        for ri, outer_delta_mm in enumerate(outer_deltas_mm):
            for fly_idx, trj in enumerate(getattr(va, "trx", [])):
                if getattr(trj, "_bad", False):
                    continue

                turns = 0
                total = 0
                for t_idx in selected_trainings:
                    if t_idx >= len(getattr(va, "trns", [])):
                        continue
                    trn = va.trns[t_idx]
                    if trn is None or not trn.isCircle():
                        continue
                    if t_idx not in windows_by_training:
                        continue

                    episodes = trj.reward_turnback_dual_circle_episodes_for_training(
                        trn=trn,
                        inner_delta_mm=float(inner_delta_mm),
                        outer_delta_mm=float(outer_delta_mm),
                        border_width_mm=float(border_width_mm),
                        debug=False,
                        radius_offset_px=float(radius_offset_px),
                    )
                    if not episodes:
                        continue

                    for ep in episodes:
                        event_t = int(ep["stop"]) - 1
                        if not _frame_in_windows(event_t, windows_by_training[t_idx]):
                            continue
                        total += 1
                        if bool(ep.get("turns_back", False)):
                            turns += 1

                ratio = np.nan if total <= 0 else (float(turns) / float(total))
                if fly_idx == 0:
                    turn_exp[vi, ri] = int(turns)
                    total_exp[vi, ri] = int(total)
                    ratio_exp[vi, ri] = ratio
                elif fly_idx == 1:
                    turn_ctrl[vi, ri] = int(turns)
                    total_ctrl[vi, ri] = int(total)
                    ratio_ctrl[vi, ri] = ratio

            _dbg_if(
                debug,
                (
                    f"[turnback-outer-radius] {vid}: outer_delta={float(outer_delta_mm):g} "
                    f"exp={turn_exp[vi, ri]}/{total_exp[vi, ri]}"
                    + (
                        ""
                        if getattr(va, "noyc", False)
                        else f" ctrl={turn_ctrl[vi, ri]}/{total_ctrl[vi, ri]}"
                    )
                ),
            )

    return ratio_exp, ratio_ctrl, turn_exp, turn_ctrl, total_exp, total_ctrl, windows_meta


def export_turnback_outer_radius_sli_bundle(vas, opts, gls, out_fn):
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    outer_deltas_mm = _outer_deltas_mm(opts)
    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync_buckets = _effective_windowing(opts)
    inner_delta_mm = float(getattr(opts, "turnback_inner_delta_mm", 0.0) or 0.0)
    border_width_mm = float(getattr(opts, "turnback_border_width_mm", 0.1) or 0.1)
    radius_offset_px = float(
        getattr(opts, "turnback_inner_radius_offset_px", 0.0) or 0.0
    )
    debug = bool(getattr(opts, "turnback_outer_radius_debug", False))

    group_label = _safe_group_label(opts, gls)
    _dbg(opts, f"[turnback-outer-radius] out={out_fn}")
    _dbg(
        opts,
        (
            f"[turnback-outer-radius] group={group_label!r} "
            f"trainings={[t + 1 for t in selected_trainings]} "
            f"skip_first={skip_first} keep_first={keep_first} "
            f"last_sync_buckets={last_sync_buckets} "
            f"outer_deltas_mm={outer_deltas_mm.tolist()}"
        ),
    )

    (
        ratio_exp,
        ratio_ctrl,
        turn_exp,
        turn_ctrl,
        total_exp,
        total_ctrl,
        windows_meta,
    ) = _compute_outer_radius_curves(
        vas_ok,
        outer_deltas_mm=outer_deltas_mm,
        inner_delta_mm=inner_delta_mm,
        border_width_mm=border_width_mm,
        radius_offset_px=radius_offset_px,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync_buckets,
        debug=debug,
    )

    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(
            "[export] WARNING: failed to compute SLI for turnback-outer-radius "
            f"bundle: {e}"
        )
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full((len(vas_ok), 0, 0), np.nan, dtype=float)

    try:
        training_names = np.array(
            [vas_ok[0].trns[t_idx].name() for t_idx in selected_trainings],
            dtype=object,
        )
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_fns = np.array([getattr(va, "fn", "") for va in vas_ok], dtype=object)
        fly_ids = np.array([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
        video_ids = np.array(
            [f"{fn}::f{f}" for fn, f in zip(video_fns, fly_ids)],
            dtype=object,
        )
    except Exception:
        video_fns = np.array([""] * len(vas_ok), dtype=object)
        fly_ids = np.array([-1] * len(vas_ok), dtype=int)
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    window_strings = []
    for vi, wins in enumerate(windows_meta):
        if not wins:
            window_strings.append("")
            continue
        parts = []
        for win in wins:
            bucket_ranges = ",".join(
                f"[{int(a)},{int(b)})" for (a, b) in win["bucket_ranges"]
            )
            parts.append(
                f"T{int(win['training_idx']) + 1}:{win['training_name']}:{bucket_ranges}"
            )
        window_strings.append(" | ".join(parts))
        _dbg(
            opts,
            f"[turnback-outer-radius] {video_ids[vi]} windows={window_strings[-1]}",
        )

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        turnback_outer_radius_ratio_exp=np.asarray(ratio_exp, dtype=float),
        turnback_outer_radius_ratio_ctrl=np.asarray(ratio_ctrl, dtype=float),
        turnback_outer_radius_turn_exp=np.asarray(turn_exp, dtype=int),
        turnback_outer_radius_turn_ctrl=np.asarray(turn_ctrl, dtype=int),
        turnback_outer_radius_total_exp=np.asarray(total_exp, dtype=int),
        turnback_outer_radius_total_ctrl=np.asarray(total_ctrl, dtype=int),
        turnback_outer_radius_outer_deltas_mm=np.asarray(outer_deltas_mm, dtype=float),
        turnback_outer_radius_trainings=np.asarray(
            np.array(selected_trainings, dtype=int) + 1, dtype=int
        ),
        turnback_outer_radius_skip_first_sync_buckets=np.array(skip_first, dtype=int),
        turnback_outer_radius_keep_first_sync_buckets=np.array(keep_first, dtype=int),
        turnback_outer_radius_last_sync_buckets=np.array(
            last_sync_buckets, dtype=int
        ),
        turnback_outer_radius_inner_delta_mm=np.array(inner_delta_mm, dtype=float),
        turnback_outer_radius_inner_radius_offset_px=np.array(
            radius_offset_px, dtype=float
        ),
        turnback_outer_radius_border_width_mm=np.array(border_width_mm, dtype=float),
        turnback_outer_radius_window_summary=np.asarray(window_strings, dtype=object),
        group_label=np.array(group_label, dtype=object),
        training_names=training_names,
        video_fns=video_fns,
        fly_ids=fly_ids,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(
        "[export] Wrote turnback-outer-radius+SLI bundle: "
        f"{out_fn} (n={len(vas_ok)})"
    )
