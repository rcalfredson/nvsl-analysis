from __future__ import annotations

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
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
    exp_entries = _frames_in_windows(va, windows, calc=True, ctrl=False, f=f)
    ctrl_entries = _frames_in_windows(va, windows, calc=True, ctrl=True, f=f)
    n_exp = np.searchsorted(exp_entries, cutoffs, side="right")
    n_ctrl = np.searchsorted(ctrl_entries, cutoffs, side="right")
    pi = np.asarray(util.prefIdx(n_exp, n_ctrl, n=pi_threshold), dtype=float)
    return thresholds, pi


def _build_running_pi_curves(
    vas,
    selected_trainings,
    *,
    skip_first: int,
    keep_first: int,
    tick_spacing: int,
    pi_threshold: int,
):
    x_lists = []
    pi_exp_lists = []
    pi_yok_lists = []
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
        if actual_rewards_exp.size == 0:
            common_x = np.zeros((0,), dtype=int)
            pi_exp_common = np.zeros((0,), dtype=float)
            pi_yok_common = np.zeros((0,), dtype=float)
            sli = np.zeros((0,), dtype=float)
            x_lists.append(common_x)
            pi_exp_lists.append(pi_exp_common)
            pi_yok_lists.append(pi_yok_common)
            sli_lists.append(sli)
            continue

        common_x = np.arange(
            int(tick_spacing),
            int(actual_rewards_exp.size) + 1,
            int(tick_spacing),
            dtype=int,
        )
        if common_x.size == 0:
            pi_exp_common = np.zeros((0,), dtype=float)
            pi_yok_common = np.zeros((0,), dtype=float)
            sli = np.zeros((0,), dtype=float)
            x_lists.append(common_x)
            pi_exp_lists.append(pi_exp_common)
            pi_yok_lists.append(pi_yok_common)
            sli_lists.append(sli)
            continue

        cutoffs = actual_rewards_exp[common_x - 1]
        _x_exp, pi_exp_common = _pi_for_cutoffs(
            va,
            windows,
            f=0,
            cutoffs=cutoffs,
            thresholds=common_x,
            pi_threshold=pi_threshold,
        )
        if getattr(va, "noyc", False):
            pi_yok_common = np.full(pi_exp_common.shape, np.nan, dtype=float)
        else:
            _x_yok, pi_yok_common = _pi_for_cutoffs(
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
        pi_exp_lists.append(pi_exp_common)
        pi_yok_lists.append(pi_yok_common)
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

    for vi, xs in enumerate(x_lists):
        if xs.size == 0:
            continue
        total_actual_rewards[vi] = int(xs[-1])
        n = min(n_ticks, xs.size)
        pi_exp_arr[vi, :n] = pi_exp_lists[vi][:n]
        pi_yok_arr[vi, :n] = pi_yok_lists[vi][:n]
        sli_arr[vi, :n] = sli_lists[vi][:n]

    return common_ticks, sli_arr, pi_exp_arr, pi_yok_arr, total_actual_rewards


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
    ) = _build_running_pi_curves(
        vas_ok,
        selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        tick_spacing=tick_spacing,
        pi_threshold=pi_threshold,
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
        video_ids = np.array(
            [getattr(va, "fn", f"va_{i}") for i, va in enumerate(vas_ok)],
            dtype=object,
        )
    except Exception:
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
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(f"[export] Wrote cum_reward_sli+SLI bundle: {out_fn} (n={n_videos})")
