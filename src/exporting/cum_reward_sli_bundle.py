from __future__ import annotations

import csv
import os

import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.utils.parsers import parse_training_selector
import src.utils.util as util


def _selected_training_indices(vas, opts) -> list[int]:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "cum_reward_sli_trainings", None)
    if not raw:
        return list(range(n_trn))

    selected = []
    for idx1 in parse_training_selector(raw):
        idx0 = int(idx1) - 1
        if 0 <= idx0 < n_trn:
            selected.append(idx0)

    selected = sorted(set(selected))
    if not selected:
        print(
            "[export] WARNING: --cum-reward-sli-trainings selected no valid trainings; "
            "falling back to all trainings."
        )
        return list(range(n_trn))
    return selected


def _selected_windows_for_va(
    va, selected_trainings, *, skip_first: int, keep_first: int
):
    windows = []
    for t_idx in selected_trainings:
        if t_idx >= len(getattr(va, "trns", [])):
            continue
        trn = va.trns[t_idx]
        fi, df, n_buckets, _complete = sync_bucket_window(
            va,
            trn,
            t_idx=t_idx,
            f=0,
            skip_first=skip_first,
            keep_first=keep_first,
            use_exclusion_mask=False,
        )
        if n_buckets <= 0:
            continue
        end = int(fi + n_buckets * df)
        windows.append((trn, int(fi), end))
    return windows


def _frames_in_windows(va, windows, *, calc=False, ctrl=False, f=None):
    xs = []
    for trn, fi, end in windows:
        on = va._getOn(trn, calc=calc, ctrl=ctrl, f=f)
        if on is None:
            continue
        on = np.asarray(on, dtype=int).reshape(-1)
        if on.size == 0:
            continue
        xs.append(on[(on >= fi) & (on < end)])
    if not xs:
        return np.zeros((0,), dtype=int)
    out = np.concatenate(xs)
    return np.sort(out.astype(int, copy=False))


def _pi_for_cutoffs(
    va,
    windows,
    *,
    f: int,
    cutoffs: np.ndarray,
    thresholds: np.ndarray,
    pi_threshold: int,
):
    if cutoffs.size == 0 or thresholds.size == 0:
        zeros_i = np.zeros((0,), dtype=int)
        zeros_f = np.zeros((0,), dtype=float)
        return zeros_i, zeros_f, zeros_i, zeros_i
    exp_entries = _frames_in_windows(va, windows, calc=True, ctrl=False, f=f)
    ctrl_entries = _frames_in_windows(va, windows, calc=True, ctrl=True, f=f)
    n_exp = np.searchsorted(exp_entries, cutoffs, side="right")
    n_ctrl = np.searchsorted(ctrl_entries, cutoffs, side="right")
    pi = np.asarray(util.prefIdx(n_exp, n_ctrl, n=pi_threshold), dtype=float)
    return thresholds, pi, n_exp.astype(int, copy=False), n_ctrl.astype(int, copy=False)


def _build_running_pi_curves(
    vas,
    selected_trainings,
    *,
    skip_first: int,
    keep_first: int,
    tick_spacing: int,
    pi_threshold: int,
    max_rewards: int | None,
):
    x_lists = []
    cutoff_lists = []
    pi_exp_lists = []
    pi_yok_lists = []
    n_exp0_lists = []
    n_ctrl0_lists = []
    n_exp1_lists = []
    n_ctrl1_lists = []
    sli_lists = []
    max_tick = 0

    for va in vas:
        windows = _selected_windows_for_va(
            va,
            selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
        )
        actual_rewards_exp = _frames_in_windows(
            va, windows, calc=False, ctrl=False, f=0
        )
        if max_rewards is not None and int(max_rewards) >= 0:
            actual_rewards_exp = actual_rewards_exp[: int(max_rewards)]
        if actual_rewards_exp.size == 0:
            common_x = np.zeros((0,), dtype=int)
            cutoffs = np.zeros((0,), dtype=int)
            pi_exp_common = np.zeros((0,), dtype=float)
            pi_yok_common = np.zeros((0,), dtype=float)
            n_exp0 = np.zeros((0,), dtype=int)
            n_ctrl0 = np.zeros((0,), dtype=int)
            n_exp1 = np.zeros((0,), dtype=int)
            n_ctrl1 = np.zeros((0,), dtype=int)
            sli = np.zeros((0,), dtype=float)
            x_lists.append(common_x)
            cutoff_lists.append(cutoffs)
            pi_exp_lists.append(pi_exp_common)
            pi_yok_lists.append(pi_yok_common)
            n_exp0_lists.append(n_exp0)
            n_ctrl0_lists.append(n_ctrl0)
            n_exp1_lists.append(n_exp1)
            n_ctrl1_lists.append(n_ctrl1)
            sli_lists.append(sli)
            continue

        common_x = np.arange(
            int(tick_spacing),
            int(actual_rewards_exp.size) + 1,
            int(tick_spacing),
            dtype=int,
        )
        if common_x.size == 0:
            cutoffs = np.zeros((0,), dtype=int)
            pi_exp_common = np.zeros((0,), dtype=float)
            pi_yok_common = np.zeros((0,), dtype=float)
            n_exp0 = np.zeros((0,), dtype=int)
            n_ctrl0 = np.zeros((0,), dtype=int)
            n_exp1 = np.zeros((0,), dtype=int)
            n_ctrl1 = np.zeros((0,), dtype=int)
            sli = np.zeros((0,), dtype=float)
            x_lists.append(common_x)
            cutoff_lists.append(cutoffs)
            pi_exp_lists.append(pi_exp_common)
            pi_yok_lists.append(pi_yok_common)
            n_exp0_lists.append(n_exp0)
            n_ctrl0_lists.append(n_ctrl0)
            n_exp1_lists.append(n_exp1)
            n_ctrl1_lists.append(n_ctrl1)
            sli_lists.append(sli)
            continue

        cutoffs = actual_rewards_exp[common_x - 1]
        _x_exp, pi_exp_common, n_exp0, n_ctrl0 = _pi_for_cutoffs(
            va,
            windows,
            f=0,
            cutoffs=cutoffs,
            thresholds=common_x,
            pi_threshold=pi_threshold,
        )
        if getattr(va, "noyc", False):
            pi_yok_common = np.full(pi_exp_common.shape, np.nan, dtype=float)
            n_exp1 = np.zeros(pi_exp_common.shape, dtype=int)
            n_ctrl1 = np.zeros(pi_exp_common.shape, dtype=int)
        else:
            _x_yok, pi_yok_common, n_exp1, n_ctrl1 = _pi_for_cutoffs(
                va,
                windows,
                f=1,
                cutoffs=cutoffs,
                thresholds=common_x,
                pi_threshold=pi_threshold,
            )
        sli = pi_exp_common - pi_yok_common
        if common_x.size:
            max_tick = max(max_tick, int(common_x[-1]))
        x_lists.append(common_x)
        cutoff_lists.append(cutoffs.astype(int, copy=False))
        pi_exp_lists.append(pi_exp_common)
        pi_yok_lists.append(pi_yok_common)
        n_exp0_lists.append(n_exp0)
        n_ctrl0_lists.append(n_ctrl0)
        n_exp1_lists.append(n_exp1)
        n_ctrl1_lists.append(n_ctrl1)
        sli_lists.append(sli)

    tick_spacing = max(1, int(tick_spacing or 1))
    common_ticks = (
        np.arange(tick_spacing, max_tick + 1, tick_spacing, dtype=float)
        if max_tick >= tick_spacing
        else np.zeros((0,), dtype=float)
    )
    n_videos = len(vas)
    n_ticks = common_ticks.size
    pi_exp_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    pi_yok_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    sli_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    total_actual_rewards = np.zeros((n_videos,), dtype=int)
    cutoff_frame_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    n_exp0_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    n_ctrl0_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    n_exp1_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)
    n_ctrl1_arr = np.full((n_videos, n_ticks), np.nan, dtype=float)

    for vi, xs in enumerate(x_lists):
        if xs.size == 0:
            continue
        total_actual_rewards[vi] = int(xs[-1])
        n = min(n_ticks, xs.size)
        cutoff_frame_arr[vi, :n] = cutoff_lists[vi][:n]
        pi_exp_arr[vi, :n] = pi_exp_lists[vi][:n]
        pi_yok_arr[vi, :n] = pi_yok_lists[vi][:n]
        n_exp0_arr[vi, :n] = n_exp0_lists[vi][:n]
        n_ctrl0_arr[vi, :n] = n_ctrl0_lists[vi][:n]
        n_exp1_arr[vi, :n] = n_exp1_lists[vi][:n]
        n_ctrl1_arr[vi, :n] = n_ctrl1_lists[vi][:n]
        sli_arr[vi, :n] = sli_lists[vi][:n]

    debug = dict(
        cutoff_frame=cutoff_frame_arr,
        n_exp0=n_exp0_arr,
        n_ctrl0=n_ctrl0_arr,
        n_exp1=n_exp1_arr,
        n_ctrl1=n_ctrl1_arr,
    )
    return common_ticks, sli_arr, pi_exp_arr, pi_yok_arr, total_actual_rewards, debug


def _write_cum_reward_sli_debug_tsv(
    path,
    *,
    vas,
    group_label,
    sli_scalar,
    ticks,
    total_actual_rewards,
    reward_pi_exp,
    reward_pi_yoked,
    sli_vs_cum_rewards,
    debug,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "video_index",
        "video_id",
        "group_label",
        "scalar_sli",
        "reward_tick",
        "cutoff_frame",
        "total_actual_rewards",
        "reached_reward_tick",
        "exp_reward_pi",
        "yoked_reward_pi",
        "sli",
        "sli_finite",
        "exp_calc_entries_exp_fly",
        "ctrl_calc_entries_exp_fly",
        "exp_calc_entries_yoked_fly",
        "ctrl_calc_entries_yoked_fly",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        ticks = np.asarray(ticks, dtype=float).reshape(-1)
        total_actual_rewards = np.asarray(total_actual_rewards, dtype=int).reshape(-1)
        cutoff_frame = np.asarray(debug["cutoff_frame"], dtype=float)
        n_exp0 = np.asarray(debug["n_exp0"], dtype=float)
        n_ctrl0 = np.asarray(debug["n_ctrl0"], dtype=float)
        n_exp1 = np.asarray(debug["n_exp1"], dtype=float)
        n_ctrl1 = np.asarray(debug["n_ctrl1"], dtype=float)
        reward_pi_exp = np.asarray(reward_pi_exp, dtype=float)
        reward_pi_yoked = np.asarray(reward_pi_yoked, dtype=float)
        sli_vs_cum_rewards = np.asarray(sli_vs_cum_rewards, dtype=float)
        sli_scalar = np.asarray(sli_scalar, dtype=float).reshape(-1)
        for vi, va in enumerate(vas):
            video_id = str(getattr(va, "fn", f"va_{vi}"))
            for tj, tick in enumerate(ticks):
                pi_exp = reward_pi_exp[vi, tj]
                pi_yok = reward_pi_yoked[vi, tj]
                sli = sli_vs_cum_rewards[vi, tj]
                writer.writerow(
                    {
                        "video_index": vi,
                        "video_id": video_id,
                        "group_label": group_label,
                        "scalar_sli": "" if not np.isfinite(sli_scalar[vi]) else float(sli_scalar[vi]),
                        "reward_tick": int(tick) if float(tick).is_integer() else float(tick),
                        "cutoff_frame": "" if not np.isfinite(cutoff_frame[vi, tj]) else int(cutoff_frame[vi, tj]),
                        "total_actual_rewards": int(total_actual_rewards[vi]),
                        "reached_reward_tick": int(total_actual_rewards[vi] >= int(round(float(tick)))),
                        "exp_reward_pi": "" if not np.isfinite(pi_exp) else float(pi_exp),
                        "yoked_reward_pi": "" if not np.isfinite(pi_yok) else float(pi_yok),
                        "sli": "" if not np.isfinite(sli) else float(sli),
                        "sli_finite": int(np.isfinite(sli)),
                        "exp_calc_entries_exp_fly": "" if not np.isfinite(n_exp0[vi, tj]) else int(n_exp0[vi, tj]),
                        "ctrl_calc_entries_exp_fly": "" if not np.isfinite(n_ctrl0[vi, tj]) else int(n_ctrl0[vi, tj]),
                        "exp_calc_entries_yoked_fly": "" if not np.isfinite(n_exp1[vi, tj]) else int(n_exp1[vi, tj]),
                        "ctrl_calc_entries_yoked_fly": "" if not np.isfinite(n_ctrl1[vi, tj]) else int(n_ctrl1[vi, tj]),
                    }
                )


def export_cum_reward_sli_bundle(vas, opts, gls, out_fn):
    from analyze import bucketLenForType

    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return
    n_videos = len(vas_ok)

    group_label = _safe_group_label(opts, gls)
    selected_trainings = _selected_training_indices(vas_ok, opts)
    tick_spacing = max(1, int(getattr(opts, "cum_reward_sli_tick_spacing", 5) or 5))
    skip_first = max(0, int(getattr(opts, "skip_first_sync_buckets", 0) or 0))
    keep_first = max(0, int(getattr(opts, "keep_first_sync_buckets", 0) or 0))
    pi_threshold = max(0, int(getattr(opts, "piTh", 10) or 0))
    min_fly_pct = float(getattr(opts, "cum_reward_sli_min_fly_pct", 95.0) or 0.0)
    raw_max_rewards = getattr(opts, "cum_reward_sli_max_rewards", None)
    max_rewards = None if raw_max_rewards is None else max(0, int(raw_max_rewards))

    try:
        sli, sli_ts_full = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for cum_reward_sli bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts_full = np.full((len(vas_ok), 0, 0), np.nan, dtype=float)

    (
        cum_reward_ticks,
        sli_vs_cum_rewards,
        reward_pi_exp,
        reward_pi_yoked,
        total_actual_rewards,
        debug,
    ) = _build_running_pi_curves(
        vas_ok,
        selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        tick_spacing=tick_spacing,
        pi_threshold=pi_threshold,
        max_rewards=max_rewards,
    )

    try:
        bl, _blf = bucketLenForType("rpid")
        bucket_len_min = float(bl)
    except Exception:
        bucket_len_min = np.nan

    va0 = vas_ok[0]
    try:
        training_names = np.array(
            [va0.trns[t_idx].name() for t_idx in selected_trainings],
            dtype=object,
        )
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_fns = np.array(
            [getattr(va, "fn", "") for va in vas_ok],
            dtype=object,
        )
        fly_ids = np.array([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
        video_ids = np.array(
            [f"{fn}::f{f}" for fn, f in zip(video_fns, fly_ids)],
            dtype=object,
        )
    except Exception:
        video_fns = np.array([""] * n_videos, dtype=object)
        fly_ids = np.array([-1] * n_videos, dtype=int)
        video_ids = np.array([f"va_{i}" for i in range(n_videos)], dtype=object)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts_full, dtype=float),
        cum_reward_sli_curve=np.asarray(sli_vs_cum_rewards, dtype=float),
        cum_reward_sli_ticks=np.asarray(cum_reward_ticks, dtype=float),
        cum_reward_sli_tick_spacing=np.array(tick_spacing, dtype=int),
        cum_reward_sli_pi_threshold=np.array(pi_threshold, dtype=int),
        cum_reward_sli_min_fly_pct=np.array(min_fly_pct, dtype=float),
        cum_reward_sli_max_rewards=(
            np.array(-1 if max_rewards is None else max_rewards, dtype=int)
        ),
        cum_reward_sli_reward_pi_exp=np.asarray(reward_pi_exp, dtype=float),
        cum_reward_sli_reward_pi_yoked=np.asarray(reward_pi_yoked, dtype=float),
        cum_reward_sli_total_actual_rewards=np.asarray(total_actual_rewards, dtype=int),
        cum_reward_sli_trainings=np.asarray(
            np.array(selected_trainings, dtype=int) + 1, dtype=int
        ),
        cum_reward_sli_skip_first_sync_buckets=np.array(skip_first, dtype=int),
        cum_reward_sli_keep_first_sync_buckets=np.array(keep_first, dtype=int),
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        fly_ids=fly_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(f"[export] Wrote cum_reward_sli+SLI bundle: {out_fn} (n={n_videos})")
    debug_tsv = getattr(opts, "cum_reward_sli_debug_tsv", None)
    if debug_tsv:
        _write_cum_reward_sli_debug_tsv(
            debug_tsv,
            vas=vas_ok,
            group_label=group_label,
            sli_scalar=sli,
            ticks=cum_reward_ticks,
            total_actual_rewards=total_actual_rewards,
            reward_pi_exp=reward_pi_exp,
            reward_pi_yoked=reward_pi_yoked,
            sli_vs_cum_rewards=sli_vs_cum_rewards,
            debug=debug,
        )
        print(f"[export] Wrote cum_reward_sli debug TSV: {debug_tsv}")
