from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window


@dataclass(frozen=True)
class SelectedWindow:
    training_idx: int
    trn: object
    start_frame: int
    end_frame: int


def effective_sync_bucket_window(
    skip_first_sync_buckets: int = 0, keep_first_sync_buckets: int = 0
) -> tuple[int, int]:
    skip = int(skip_first_sync_buckets or 0)
    keep = int(keep_first_sync_buckets or 0)
    return max(0, skip), max(0, keep)


def selected_training_indices(
    ref_va,
    trainings: Sequence[int] | None,
    *,
    log_tag: str,
) -> list[int]:
    trns = getattr(ref_va, "trns", None) or []
    n_trn = len(trns)
    if n_trn <= 0:
        return []

    raw = list(trainings) if trainings else []
    if not raw:
        return list(range(n_trn))

    keep = sorted({int(x) - 1 for x in raw if 1 <= int(x) <= n_trn})
    if keep:
        return keep

    print(
        f"[{log_tag}] WARNING: selected trainings are out of range; "
        "falling back to all trainings."
    )
    return list(range(n_trn))


def training_window_label(selected_trainings: Sequence[int]) -> str:
    if not selected_trainings:
        return "all trainings"
    if len(selected_trainings) == 1:
        return f"T{selected_trainings[0] + 1}"
    runs = []
    start = prev = selected_trainings[0] + 1
    for idx0 in selected_trainings[1:]:
        cur = idx0 + 1
        if cur == prev + 1:
            prev = cur
            continue
        runs.append(f"T{start}" if start == prev else f"T{start}-T{prev}")
        start = prev = cur
    runs.append(f"T{start}" if start == prev else f"T{start}-T{prev}")
    return ", ".join(runs)


def selected_windows_for_va(
    va,
    selected_trainings: Sequence[int],
    *,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
    f: int = 0,
) -> list[SelectedWindow]:
    windows: list[SelectedWindow] = []
    trns = getattr(va, "trns", None) or []
    skip_first, keep_first = effective_sync_bucket_window(
        skip_first_sync_buckets, keep_first_sync_buckets
    )

    for t_idx in selected_trainings:
        if t_idx < 0 or t_idx >= len(trns):
            continue
        trn = trns[t_idx]
        fi, df, n_buckets, _complete = sync_bucket_window(
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
        end = int(fi + n_buckets * df)
        windows.append(
            SelectedWindow(
                training_idx=int(t_idx),
                trn=trn,
                start_frame=int(fi),
                end_frame=int(end),
            )
        )
    return windows


def frames_in_windows(va, windows: Sequence[SelectedWindow], *, calc=False, ctrl=False, f=0):
    xs = []
    for win in windows:
        on = va._getOn(win.trn, calc=calc, ctrl=ctrl, f=f)
        if on is None:
            continue
        on = np.asarray(on, dtype=int).reshape(-1)
        if on.size == 0:
            continue
        xs.append(on[(on >= win.start_frame) & (on < win.end_frame)])
    if not xs:
        return np.zeros((0,), dtype=int)
    out = np.concatenate(xs)
    return np.sort(out.astype(int, copy=False))


def window_duration_s(start_frame: int, end_frame: int, fps: float) -> float:
    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 1.0
    return max(0.0, float(end_frame - start_frame) / fps)


def cumulative_window_seconds_for_frame(
    windows: Sequence[SelectedWindow], frame: int, *, fps: float
) -> float:
    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 1.0
    total_s = 0.0
    target = int(frame)
    for win in windows:
        if target < win.start_frame:
            break
        if target < win.end_frame:
            total_s += max(0.0, float(target - win.start_frame) / fps)
            return total_s
        total_s += window_duration_s(win.start_frame, win.end_frame, fps)
    return np.nan


def locate_window_for_frame(
    windows: Sequence[SelectedWindow], frame: int
) -> SelectedWindow | None:
    target = int(frame)
    for win in windows:
        if win.start_frame <= target < win.end_frame:
            return win
    return None
