from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from src.analysis.behavior_states import (
    BehaviorState,
    analyze_trajectory_behavior_states,
    behavior_state_config_from_opts,
)
from src.exporting.com_sli_bundle import _safe_group_label
from src.exporting.turnback_excursion_bin_sli_bundle import _selected_windows_for_va
from src.exporting.turnback_home_vector_alignment_sli_bundle import (
    _mean_ci,
    _unit_id_for_va_fly,
    cosine_alignment,
)
from src.exporting.wall_contact_episode_filter import (
    episode_overlaps_wall_contact,
    wall_contact_regions_for_trj,
)
from src.utils.parsers import parse_training_selector

Y_LABEL = "Turn home-vector alignment improvement (deg)"
BASE_TITLE = "Turn home-vector alignment improvement"

TURN_FILTER_ALL = "all"
TURN_FILTER_EXCLUDE_WALL_CONTACT = "exclude_wall_contact"
TURN_FILTER_CHOICES = (TURN_FILTER_ALL, TURN_FILTER_EXCLUDE_WALL_CONTACT)

ANCHOR_FRAME = "frame"
ANCHOR_SEGMENT_MIDPOINT = "segment_midpoint"
ANCHOR_CHOICES = (ANCHOR_FRAME, ANCHOR_SEGMENT_MIDPOINT)


def _selected_training_indices(vas, opts) -> list[int]:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "turn_home_vector_alignment_trainings", None)
    if raw is None or not str(raw).strip():
        raw = "1,2"

    selected = []
    for idx1 in parse_training_selector(str(raw)):
        idx0 = int(idx1) - 1
        if 0 <= idx0 < n_trn:
            selected.append(idx0)

    selected = sorted(set(selected))
    if selected:
        return selected

    print(
        "[export] WARNING: --turn-home-vector-alignment-trainings selected no "
        "valid trainings; falling back to trainings 1 and 2 where available."
    )
    return list(range(min(2, n_trn)))


def _effective_windowing(opts) -> tuple[int, int, int]:
    raw_skip = getattr(opts, "turn_home_vector_alignment_skip_first_sync_buckets", None)
    raw_keep = getattr(opts, "turn_home_vector_alignment_keep_first_sync_buckets", None)
    raw_last = getattr(opts, "turn_home_vector_alignment_last_sync_buckets", None)

    if raw_skip is None:
        raw_skip = getattr(opts, "skip_first_sync_buckets", 0)
    if raw_keep is None:
        raw_keep = getattr(opts, "keep_first_sync_buckets", 0)
    if raw_last is None:
        raw_last = 0

    return (
        max(0, int(raw_skip or 0)),
        max(0, int(raw_keep or 0)),
        max(0, int(raw_last or 0)),
    )


def _panel_defs_for_va(va, selected_trainings, opts, skip_first, keep_first, last_sync):
    panels = []
    include_pre = bool(getattr(opts, "turn_home_vector_alignment_include_pre", True))
    if include_pre and getattr(va, "trns", None):
        ref_idx = selected_trainings[0] if selected_trainings else 0
        ref_idx = min(max(0, int(ref_idx)), len(va.trns) - 1)
        ref_trn = va.trns[ref_idx]
        if ref_trn is not None and ref_trn.isCircle():
            panels.append(
                {
                    "panel_key": "pre",
                    "label": "Pre-training",
                    "training_idx": int(ref_idx),
                    "trn": ref_trn,
                    "windows": [
                        {
                            "training_idx": int(ref_idx),
                            "training_name": "pre-training",
                            "start": 0,
                            "stop": int(ref_trn.start),
                            "bucket_ranges": [(0, int(ref_trn.start))],
                        }
                    ],
                }
            )

    train_windows = _selected_windows_for_va(
        va,
        selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync,
    )
    windows_by_training: dict[int, list[dict]] = {}
    for win in train_windows:
        windows_by_training.setdefault(int(win["training_idx"]), []).append(win)

    for t_idx in selected_trainings:
        if t_idx >= len(getattr(va, "trns", [])):
            continue
        trn = va.trns[t_idx]
        if trn is None or not trn.isCircle():
            continue
        windows = windows_by_training.get(int(t_idx), [])
        if not windows:
            continue
        panels.append(
            {
                "panel_key": f"training_{int(t_idx) + 1}",
                "label": f"Training {int(t_idx) + 1}",
                "training_idx": int(t_idx),
                "trn": trn,
                "windows": windows,
            }
        )
    return panels


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _finite_xy(trj, frame_idx: int) -> tuple[float, float] | None:
    if frame_idx < 0:
        return None
    n = min(len(getattr(trj, "x", [])), len(getattr(trj, "y", [])))
    if frame_idx >= n:
        return None
    x = float(trj.x[frame_idx])
    y = float(trj.y[frame_idx])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _vector_between_frames(trj, start_frame: int, stop_frame: int):
    a = _finite_xy(trj, int(start_frame))
    b = _finite_xy(trj, int(stop_frame))
    if a is None or b is None:
        return None
    ax, ay = a
    bx, by = b
    vx = float(bx - ax)
    vy = float(by - ay)
    if not (np.isfinite(vx) and np.isfinite(vy)):
        return None
    if np.hypot(vx, vy) <= 0.0:
        return None
    return vx, vy


def _anchor_xy(trj, frame_a: int, frame_b: int | None, *, anchor: str):
    a = _finite_xy(trj, int(frame_a))
    if a is None:
        return None
    if anchor == ANCHOR_FRAME:
        return a
    if anchor != ANCHOR_SEGMENT_MIDPOINT:
        raise ValueError(f"unknown turn home-vector anchor: {anchor!r}")
    if frame_b is None:
        return a
    b = _finite_xy(trj, int(frame_b))
    if b is None:
        return None
    return float((a[0] + b[0]) / 2.0), float((a[1] + b[1]) / 2.0)


def _home_vector_to_reward_center(anchor_xy, trn, trj):
    try:
        cx, cy, _r = trn.circles(getattr(trj, "f", 0))[0]
    except Exception:
        return None
    ax, ay = anchor_xy
    hx = float(cx) - float(ax)
    hy = float(cy) - float(ay)
    if not (np.isfinite(hx) and np.isfinite(hy)):
        return None
    if np.hypot(hx, hy) <= 0.0:
        return None
    return hx, hy


def _heading_home_angle_deg(heading, home) -> float:
    alignment = cosine_alignment(heading, home)
    if not np.isfinite(alignment):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0))))


def turn_home_vector_alignment_delta(
    trj,
    trn,
    turn_start: int,
    turn_stop: int,
    *,
    anchor: str = ANCHOR_FRAME,
) -> float:
    """
    Compute start-minus-end heading-to-home angular error for one turn island.

    A turn mask frame labels a plotted trajectory segment. For an island
    start..stop, start heading uses the last pre-turn segment plus first in-turn
    segment (frame start-1 to start+1), while end heading uses the last in-turn
    segment plus first post-turn segment (frame stop to stop+2).

    The returned value is in degrees. Positive values mean the heading became
    better aligned with the home vector over the turn.
    """
    start = int(turn_start)
    stop = int(turn_stop)
    if stop < start:
        return float("nan")

    start_heading = _vector_between_frames(trj, start - 1, start + 1)
    end_heading = _vector_between_frames(trj, stop, stop + 2)
    if start_heading is None or end_heading is None:
        return float("nan")

    start_anchor = _anchor_xy(trj, start, start + 1, anchor=anchor)
    end_anchor = _anchor_xy(trj, stop + 1, stop, anchor=anchor)
    if start_anchor is None or end_anchor is None:
        return float("nan")

    start_home = _home_vector_to_reward_center(start_anchor, trn, trj)
    end_home = _home_vector_to_reward_center(end_anchor, trn, trj)
    if start_home is None or end_home is None:
        return float("nan")

    start_angle = _heading_home_angle_deg(start_heading, start_home)
    end_angle = _heading_home_angle_deg(end_heading, end_home)
    if not (np.isfinite(start_angle) and np.isfinite(end_angle)):
        return float("nan")
    return float(start_angle - end_angle)


def _turn_within_windows(start: int, stop: int, windows) -> bool:
    for win in windows:
        if int(win["start"]) <= int(start) and int(stop) + 1 < int(win["stop"]):
            return True
    return False


def _iter_turn_alignment_deltas_for_panel(
    va,
    trj,
    panel: dict,
    *,
    turn_filter: str,
    anchor: str,
    wall_regions,
):
    states = getattr(trj, "behavior_state", None)
    if states is None:
        config = behavior_state_config_from_opts(getattr(va, "opts", None))
        states = analyze_trajectory_behavior_states(trj, config=config)
    if states is None:
        return

    turn_mask = np.asarray(states, dtype=np.int8) == int(BehaviorState.TURN)
    n = min(turn_mask.size, len(getattr(trj, "x", [])), len(getattr(trj, "y", [])))
    if n <= 3:
        return
    turn_mask = turn_mask[:n]

    trn = panel["trn"]
    windows = panel["windows"]
    for start, stop in _true_runs(turn_mask):
        if start <= 0 or stop + 2 >= n:
            continue
        if not _turn_within_windows(start, stop, windows):
            continue
        if turn_filter == TURN_FILTER_EXCLUDE_WALL_CONTACT:
            ep = {"start": int(start), "stop": int(stop) + 2}
            if episode_overlaps_wall_contact(ep, wall_regions):
                continue
        val = turn_home_vector_alignment_delta(
            trj, trn, start, stop, anchor=anchor
        )
        if np.isfinite(val):
            yield float(val)


def _collect_panel_values(vas, opts, *, selected_trainings, skip_first, keep_first, last_sync):
    turn_filter = str(
        getattr(opts, "turn_home_vector_alignment_turn_filter", TURN_FILTER_ALL)
        or TURN_FILTER_ALL
    )
    if turn_filter not in TURN_FILTER_CHOICES:
        raise ValueError(
            "turn_home_vector_alignment_turn_filter must be one of "
            f"{TURN_FILTER_CHOICES!r}; got {turn_filter!r}"
        )

    anchor = str(
        getattr(opts, "turn_home_vector_alignment_anchor", ANCHOR_FRAME) or ANCHOR_FRAME
    )
    if anchor not in ANCHOR_CHOICES:
        raise ValueError(
            f"turn_home_vector_alignment_anchor must be one of {ANCHOR_CHOICES!r}; "
            f"got {anchor!r}"
        )

    min_turns = max(1, int(getattr(opts, "turn_home_vector_alignment_min_turns", 1) or 1))

    panel_labels = []
    panel_values: list[list[float]] = []
    panel_ids: list[list[str]] = []
    panel_counts: list[list[int]] = []
    panel_keys = []

    panel_index_by_key: dict[str, int] = {}

    for vi, va in enumerate(vas):
        if getattr(va, "_skipped", False):
            continue

        panels = _panel_defs_for_va(
            va, selected_trainings, opts, skip_first, keep_first, last_sync
        )
        for panel in panels:
            key = str(panel["panel_key"])
            if key not in panel_index_by_key:
                panel_index_by_key[key] = len(panel_labels)
                panel_keys.append(key)
                panel_labels.append(str(panel["label"]))
                panel_values.append([])
                panel_ids.append([])
                panel_counts.append([])

        for fly_idx, trj in enumerate(getattr(va, "trx", [])):
            if fly_idx > 1:
                continue
            if fly_idx == 1 and getattr(va, "noyc", False):
                continue
            if fly_idx != 0:
                continue
            if getattr(trj, "_bad", False):
                continue

            wall_regions = wall_contact_regions_for_trj(
                trj,
                enabled=(turn_filter == TURN_FILTER_EXCLUDE_WALL_CONTACT),
                warned_missing=[False],
                log_tag="turn-home-vector-alignment",
            )
            unit_id = _unit_id_for_va_fly(va, vi, fly_idx)

            for panel in panels:
                vals = np.asarray(
                    list(
                        _iter_turn_alignment_deltas_for_panel(
                            va,
                            trj,
                            panel,
                            turn_filter=turn_filter,
                            anchor=anchor,
                            wall_regions=wall_regions,
                        )
                    ),
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                if vals.size < min_turns:
                    continue
                p_idx = panel_index_by_key[str(panel["panel_key"])]
                panel_values[p_idx].append(float(np.mean(vals)))
                panel_ids[p_idx].append(unit_id)
                panel_counts[p_idx].append(int(vals.size))

    per_unit_values_panel = [np.asarray(vals, dtype=float) for vals in panel_values]
    per_unit_ids_panel = [np.asarray(ids, dtype=object) for ids in panel_ids]
    turn_counts_panel = [np.asarray(counts, dtype=int) for counts in panel_counts]

    means = []
    ci_lo = []
    ci_hi = []
    n_units = []
    for vals in per_unit_values_panel:
        mean, lo, hi, n = _mean_ci(vals, ci_conf=0.95)
        means.append(mean)
        ci_lo.append(lo)
        ci_hi.append(hi)
        n_units.append(n)

    metric_meta = {
        "turn_filter": turn_filter,
        "anchor": anchor,
        "min_turns": int(min_turns),
        "turn_detector": "behavior_state",
        "turn_angular_source": str(
            getattr(opts, "behavior_state_turn_angular_source", "theta_or_path")
            or "theta_or_path"
        ),
    }
    return (
        panel_keys,
        panel_labels,
        per_unit_values_panel,
        per_unit_ids_panel,
        turn_counts_panel,
        np.asarray(means, dtype=float),
        np.asarray(ci_lo, dtype=float),
        np.asarray(ci_hi, dtype=float),
        np.asarray(n_units, dtype=int),
        metric_meta,
    )


def export_turn_home_vector_alignment_sli_bundle(vas, opts, gls, out_fn):
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync = _effective_windowing(opts)

    (
        panel_keys,
        panel_labels,
        per_unit_values_panel,
        per_unit_ids_panel,
        turn_counts_panel,
        mean,
        ci_lo,
        ci_hi,
        n_units_panel,
        metric_meta,
    ) = _collect_panel_values(
        vas_ok,
        opts,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync=last_sync,
    )

    meta = {
        "log_tag": "turn-home-vector-alignment",
        "y_label": Y_LABEL,
        "base_title": BASE_TITLE,
        "pool_trainings": False,
        "ci": True,
        "ci_conf": 0.95,
        "skip_first_sync_buckets": int(skip_first),
        "keep_first_sync_buckets": int(keep_first),
        "last_sync_buckets": int(last_sync),
        "training_selection": {
            "trainings_user": getattr(opts, "turn_home_vector_alignment_trainings", None),
            "trainings_effective": [int(i) + 1 for i in selected_trainings],
            "include_pre": bool(
                getattr(opts, "turn_home_vector_alignment_include_pre", True)
            ),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "group_label": _safe_group_label(opts, gls),
        "metric": "turn_home_vector_alignment_angle_improvement_deg",
        "heading_convention": (
            "path-derived heading; start uses frame start-1 to start+1, "
            "end uses frame stop to stop+2"
        ),
        **metric_meta,
    }

    payload = {
        "panel_labels": np.asarray(panel_labels, dtype=object),
        "per_unit_values_panel": np.asarray(per_unit_values_panel, dtype=object),
        "per_unit_ids_panel": np.asarray(per_unit_ids_panel, dtype=object),
        "mean": np.asarray(mean, dtype=float),
        "ci_lo": np.asarray(ci_lo, dtype=float),
        "ci_hi": np.asarray(ci_hi, dtype=float),
        "n_units_panel": np.asarray(n_units_panel, dtype=int),
        "meta_json": json.dumps(meta, sort_keys=True),
        "turn_home_vector_alignment_panel_keys": np.asarray(panel_keys, dtype=object),
        "turn_home_vector_alignment_turn_counts_panel": np.asarray(
            turn_counts_panel, dtype=object
        ),
        "turn_home_vector_alignment_trainings": np.asarray(
            selected_trainings, dtype=int
        ),
        "turn_home_vector_alignment_skip_first_sync_buckets": np.array(
            skip_first, dtype=int
        ),
        "turn_home_vector_alignment_keep_first_sync_buckets": np.array(
            keep_first, dtype=int
        ),
        "turn_home_vector_alignment_last_sync_buckets": np.array(
            last_sync, dtype=int
        ),
    }

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(out_fn, **payload)
    print(
        f"[export] Wrote turn home-vector alignment scalar bundle: {out_fn} "
        f"(panels={len(panel_labels)})"
    )
