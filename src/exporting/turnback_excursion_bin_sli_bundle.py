from __future__ import annotations

import os
import csv

import numpy as np

from src.analysis.episode_filters import (
    EPISODE_TYPE_INNER_EXIT_REENTRY,
    episode_filter_accounting_payload,
    min_episode_count_for_type,
)
from src.analysis.sli_bundle_utils import (
    validate_sli_bundle,
    validate_turnback_excursion_bin_bundle,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.utils.parsers import parse_distances, parse_training_selector


def _dbg(opts, msg):
    if opts is not None and bool(getattr(opts, "turnback_excursion_bin_debug", False)):
        print(msg)


def _dbg_if(debug: bool, msg: str):
    if bool(debug):
        print(msg)


def _selected_training_indices(vas, opts) -> list[int]:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "turnback_excursion_bin_trainings", None)
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
        "[export] WARNING: --turnback-excursion-bin-trainings selected no valid "
        "trainings; falling back to all trainings."
    )
    return list(range(n_trn))


def _bin_edges_mm(opts) -> np.ndarray:
    raw = getattr(opts, "turnback_excursion_bin_radii_mm", None)
    if raw is None or not str(raw).strip():
        raw = getattr(opts, "turnback_excursion_bin_edges_mm", None)
    if raw is None or not str(raw).strip():
        raise ValueError(
            "--turnback-excursion-bin-radii-mm is required when exporting "
            "turnback-excursion-bin bundles."
        )
    vals = np.asarray(parse_distances(str(raw)), dtype=float)
    if vals.size < 2:
        raise ValueError(
            "--turnback-excursion-bin-edges-mm must contain at least two values."
        )
    if np.any(np.isnan(vals)) or np.any(np.isneginf(vals)):
        raise ValueError(
            "--turnback-excursion-bin-edges-mm must contain valid numeric edges."
        )
    if np.any(np.isposinf(vals[:-1])):
        raise ValueError(
            "--turnback-excursion-bin-edges-mm may use 'inf' only as the last edge, not between finite edges."
        )
    if np.any(np.diff(vals) <= 0):
        raise ValueError(
            "--turnback-excursion-bin-edges-mm must be strictly increasing."
        )
    return vals


def _uses_legacy_bin_edges(opts) -> bool:
    raw_new = getattr(opts, "turnback_excursion_bin_radii_mm", None)
    raw_old = getattr(opts, "turnback_excursion_bin_edges_mm", None)
    return not (raw_new is not None and str(raw_new).strip()) and (
        raw_old is not None and str(raw_old).strip()
    )


def _pair_deltas_mm(opts) -> tuple[np.ndarray, np.ndarray] | None:
    raw = getattr(opts, "turnback_excursion_bin_radius_pairs_mm", None)
    if raw is None or not str(raw).strip():
        raw = getattr(opts, "turnback_excursion_bin_pairs_mm", None)
    if raw is None or not str(raw).strip():
        return None

    inner_vals: list[float] = []
    outer_vals: list[float] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--turnback-excursion-bin-pairs-mm entries must use inner:outer "
                "format, e.g. '2:4,6:8,14:16'."
            )
        inner_raw, outer_raw = item.split(":", 1)
        inner = float(parse_distances(inner_raw.strip())[0])
        outer = float(parse_distances(outer_raw.strip())[0])
        if not np.isfinite(inner) or not np.isfinite(outer):
            raise ValueError("--turnback-excursion-bin-pairs-mm values must be finite.")
        if inner < 0.0:
            raise ValueError(
                "--turnback-excursion-bin-pairs-mm inner deltas must be non-negative."
            )
        if outer <= inner:
            raise ValueError(
                "--turnback-excursion-bin-pairs-mm outer deltas must be greater "
                "than their matching inner deltas."
            )
        inner_vals.append(float(inner))
        outer_vals.append(float(outer))

    if not inner_vals:
        raise ValueError(
            "--turnback-excursion-bin-pairs-mm must contain at least one inner:outer pair."
        )
    return np.asarray(inner_vals, dtype=float), np.asarray(outer_vals, dtype=float)


def _effective_windowing(opts) -> tuple[int, int, int]:
    raw_skip = getattr(opts, "turnback_excursion_bin_skip_first_sync_buckets", None)
    raw_keep = getattr(opts, "turnback_excursion_bin_keep_first_sync_buckets", None)
    skip_first = (
        getattr(opts, "skip_first_sync_buckets", 0) if raw_skip is None else raw_skip
    )
    keep_first = (
        getattr(opts, "keep_first_sync_buckets", 0) if raw_keep is None else raw_keep
    )
    last_sync = int(getattr(opts, "turnback_excursion_bin_last_sync_buckets", 0) or 0)
    return max(0, int(skip_first or 0)), max(0, int(keep_first or 0)), max(0, last_sync)


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


def _integrated_bin_contribution(
    max_outer_delta_mm: float,
    a_mm: float,
    b_mm: float,
    *,
    min_valid_outer_delta_mm: float = 0.0,
) -> float:
    """
    Exact average over outer-radius thresholds r in [a_mm, b_mm) of
    1{max_outer_delta_mm < r}.
    """
    lo = max(float(a_mm), float(min_valid_outer_delta_mm))
    hi = float(b_mm)
    width = hi - lo
    if not np.isfinite(max_outer_delta_mm) or width <= 0:
        return np.nan
    if float(max_outer_delta_mm) < lo:
        return 1.0
    if float(max_outer_delta_mm) >= hi:
        return 0.0
    return float(hi - float(max_outer_delta_mm)) / width


def _resolve_open_ended_upper_edge(
    vas,
    *,
    bin_edges_mm: np.ndarray,
    inner_delta_mm: float,
    border_width_mm: float,
    radius_offset_px: float,
    inner_radius_mm: float | None = None,
    legacy_bin_edges: bool = True,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    debug: bool,
) -> tuple[np.ndarray, bool]:
    if not np.isposinf(float(bin_edges_mm[-1])):
        return np.asarray(bin_edges_mm, dtype=float), False

    lower = float(bin_edges_mm[-2])
    max_observed = -np.inf

    for va in vas:
        windows = _selected_windows_for_va(
            va,
            selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
        )
        if not windows:
            continue
        windows_by_training = {int(win["training_idx"]): [win] for win in windows}
        for fly_idx, trj in enumerate(getattr(va, "trx", [])):
            if getattr(trj, "_bad", False):
                continue
            if fly_idx > 1:
                continue
            if fly_idx == 1 and getattr(va, "noyc", False):
                continue
            for t_idx in selected_trainings:
                if t_idx >= len(getattr(va, "trns", [])):
                    continue
                trn = va.trns[t_idx]
                if trn is None or not trn.isCircle():
                    continue
                if t_idx not in windows_by_training:
                    continue
                episodes = trj.reward_turnback_excursion_episodes_for_training(
                    trn=trn,
                    inner_radius_mm=inner_radius_mm,
                    inner_delta_mm=inner_delta_mm,
                    border_width_mm=float(border_width_mm),
                    debug=False,
                    radius_offset_px=float(radius_offset_px),
                )
                for ep in episodes or []:
                    event_t = int(ep["stop"]) - 1
                    if not _frame_in_windows(event_t, windows_by_training[t_idx]):
                        continue
                    radial_mm = float(
                        ep.get("max_outer_delta_mm" if legacy_bin_edges else "max_outer_radius_mm", np.nan)
                    )
                    if np.isfinite(radial_mm):
                        max_observed = max(max_observed, radial_mm)

    if not np.isfinite(max_observed) or max_observed <= lower:
        raise ValueError(
            "--turnback-excursion-bin-edges-mm requested an open-ended upper "
            f"bin starting at {lower:g} mm, but no observed excursion exceeded it."
        )

    resolved = np.asarray(bin_edges_mm, dtype=float).copy()
    resolved[-1] = float(max_observed)
    _dbg_if(
        debug,
        "[turnback-excursion-bin] resolved final 'inf' edge to "
        f"{float(max_observed):g} mm (max observed outer-radius {'delta' if legacy_bin_edges else 'distance'})",
    )
    return resolved, True


def _compute_turnback_curves(
    vas,
    *,
    bin_edges_mm: np.ndarray,
    inner_delta_mm: float,
    border_width_mm: float,
    radius_offset_px: float,
    inner_radius_mm: float | None = None,
    legacy_bin_edges: bool = True,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    debug: bool,
    min_episodes: int = 0,
):
    bin_edges_mm, _open_ended_upper_bin = _resolve_open_ended_upper_edge(
        vas,
        bin_edges_mm=bin_edges_mm,
        inner_delta_mm=inner_delta_mm,
        border_width_mm=border_width_mm,
        inner_radius_mm=inner_radius_mm,
        legacy_bin_edges=legacy_bin_edges,
        radius_offset_px=radius_offset_px,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync_buckets,
        debug=debug,
    )

    n_videos = len(vas)
    n_bins = int(max(0, bin_edges_mm.size - 1))
    ratio_exp = np.full((n_videos, n_bins), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_bins), np.nan, dtype=float)
    turn_exp = np.zeros((n_videos, n_bins), dtype=float)
    turn_ctrl = np.zeros((n_videos, n_bins), dtype=float)
    total_exp = np.zeros((n_videos, n_bins), dtype=int)
    total_ctrl = np.zeros((n_videos, n_bins), dtype=int)
    windows_meta = []
    invalid_bin_hits = 0

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
                f"[turnback-excursion-bin] {vid}: no eligible windows after selection",
            )
            continue

        windows_by_training = {int(win["training_idx"]): [win] for win in windows}

        for fly_idx, trj in enumerate(getattr(va, "trx", [])):
            if getattr(trj, "_bad", False):
                continue
            if fly_idx > 1:
                continue
            if fly_idx == 1 and getattr(va, "noyc", False):
                continue

            turns = np.zeros(n_bins, dtype=float)
            total = np.zeros(n_bins, dtype=int)

            for t_idx in selected_trainings:
                if t_idx >= len(getattr(va, "trns", [])):
                    continue
                trn = va.trns[t_idx]
                if trn is None or not trn.isCircle():
                    continue
                if t_idx not in windows_by_training:
                    continue

                episodes = trj.reward_turnback_excursion_episodes_for_training(
                    trn=trn,
                    inner_radius_mm=inner_radius_mm,
                    inner_delta_mm=inner_delta_mm,
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
                    radial_mm = float(
                        ep.get("max_outer_delta_mm" if legacy_bin_edges else "max_outer_radius_mm", np.nan)
                    )
                    effective_inner_mm = float(
                        ep.get(
                            "effective_inner_delta_mm"
                            if legacy_bin_edges
                            else "effective_inner_radius_mm",
                            np.nan,
                        )
                    )
                    if not np.isfinite(radial_mm):
                        continue
                    if not np.isfinite(effective_inner_mm):
                        continue
                    if not bool(ep.get("turns_back", False)):
                        continue
                    for bin_idx in range(n_bins):
                        a_mm = float(bin_edges_mm[bin_idx])
                        b_mm = float(bin_edges_mm[bin_idx + 1])
                        contrib = _integrated_bin_contribution(
                            radial_mm,
                            a_mm,
                            b_mm,
                            min_valid_outer_delta_mm=effective_inner_mm,
                        )
                        if not np.isfinite(contrib):
                            if float(b_mm) <= float(effective_inner_mm):
                                invalid_bin_hits += 1
                            continue
                        total[bin_idx] += 1
                        turns[bin_idx] += contrib

            ratio = np.full(n_bins, np.nan, dtype=float)
            np.divide(turns, total, out=ratio, where=(total > 0))
            min_n = max(0, int(min_episodes or 0))
            if min_n > 0:
                ratio[total < min_n] = np.nan
            if fly_idx == 0:
                turn_exp[vi, :] = turns
                total_exp[vi, :] = total
                ratio_exp[vi, :] = ratio
            elif fly_idx == 1:
                turn_ctrl[vi, :] = turns
                total_ctrl[vi, :] = total
                ratio_ctrl[vi, :] = ratio

        _dbg_if(
            debug,
            (
                f"[turnback-excursion-bin] {vid}: "
                f"exp={turn_exp[vi].tolist()}/{total_exp[vi].tolist()}"
                + (
                    ""
                    if getattr(va, "noyc", False)
                    else f" ctrl={turn_ctrl[vi].tolist()}/{total_ctrl[vi].tolist()}"
                )
            ),
        )

    if debug and invalid_bin_hits > 0:
        _dbg_if(
            debug,
            "[turnback-excursion-bin] skipped fully invalid bin contributions "
            f"(outer bin at/below effective inner radius): n={int(invalid_bin_hits)}",
        )

    return (
        ratio_exp,
        ratio_ctrl,
        turn_exp,
        turn_ctrl,
        total_exp,
        total_ctrl,
        windows_meta,
    )


def _compute_pair_curves(
    vas,
    *,
    inner_deltas_mm: np.ndarray,
    outer_deltas_mm: np.ndarray,
    legacy_pair_deltas: bool = True,
    border_width_mm: float,
    radius_offset_px: float,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    debug: bool,
    min_episodes: int = 0,
):
    n_videos = len(vas)
    n_pairs = int(inner_deltas_mm.size)
    ratio_exp = np.full((n_videos, n_pairs), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_pairs), np.nan, dtype=float)
    turn_exp = np.zeros((n_videos, n_pairs), dtype=float)
    turn_ctrl = np.zeros((n_videos, n_pairs), dtype=float)
    total_exp = np.zeros((n_videos, n_pairs), dtype=int)
    total_ctrl = np.zeros((n_videos, n_pairs), dtype=int)
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
                f"[turnback-excursion-bin] {vid}: no eligible windows after selection",
            )
            continue
        windows_by_training = {int(win["training_idx"]): [win] for win in windows}

        for pair_idx, (inner_delta_mm, outer_delta_mm) in enumerate(
            zip(inner_deltas_mm, outer_deltas_mm)
        ):
            for fly_idx, trj in enumerate(getattr(va, "trx", [])):
                if getattr(trj, "_bad", False):
                    continue
                if fly_idx > 1:
                    continue
                if fly_idx == 1 and getattr(va, "noyc", False):
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
                        inner_radius_mm=(
                            None if legacy_pair_deltas else float(inner_delta_mm)
                        ),
                        inner_delta_mm=(
                            float(inner_delta_mm) if legacy_pair_deltas else None
                        ),
                        outer_radius_mm=(
                            None if legacy_pair_deltas else float(outer_delta_mm)
                        ),
                        outer_delta_mm=(
                            float(outer_delta_mm) if legacy_pair_deltas else None
                        ),
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

                min_n = max(0, int(min_episodes or 0))
                ratio = (
                    np.nan
                    if total <= 0 or (min_n > 0 and total < min_n)
                    else float(turns) / float(total)
                )
                if fly_idx == 0:
                    turn_exp[vi, pair_idx] = float(turns)
                    total_exp[vi, pair_idx] = int(total)
                    ratio_exp[vi, pair_idx] = ratio
                elif fly_idx == 1:
                    turn_ctrl[vi, pair_idx] = float(turns)
                    total_ctrl[vi, pair_idx] = int(total)
                    ratio_ctrl[vi, pair_idx] = ratio

            _dbg_if(
                debug,
                (
                    f"[turnback-excursion-bin] {vid}: "
                    f"pair={float(inner_delta_mm):g}:{float(outer_delta_mm):g} "
                    f"exp={turn_exp[vi, pair_idx]:g}/{total_exp[vi, pair_idx]}"
                    + (
                        ""
                        if getattr(va, "noyc", False)
                        else (
                            f" ctrl={turn_ctrl[vi, pair_idx]:g}/"
                            f"{total_ctrl[vi, pair_idx]}"
                        )
                    )
                ),
            )
    return (
        ratio_exp,
        ratio_ctrl,
        turn_exp,
        turn_ctrl,
        total_exp,
        total_ctrl,
        windows_meta,
    )


_DEBUG_EPISODE_FIELDS = [
    "video_index",
    "video_id",
    "fly_idx",
    "fly_role",
    "training_idx",
    "training_name",
    "pair_idx",
    "requested_inner_mm",
    "requested_outer_mm",
    "pair_mode",
    "window_start",
    "window_stop",
    "episode_idx",
    "start",
    "stop",
    "event_frame",
    "turns_back",
    "end_reason",
    "reward_cx_px",
    "reward_cy_px",
    "reward_radius_px",
    "reward_radius_mm",
    "px_per_mm",
    "inner_radius_px",
    "outer_radius_px",
    "inner_radius_mm",
    "outer_radius_mm",
    "inner_border_px",
    "outer_border_px",
    "border_width_mm",
    "start_x_px",
    "start_y_px",
    "event_x_px",
    "event_y_px",
    "start_distance_px",
    "event_distance_px",
    "start_distance_mm",
    "event_distance_mm",
]


def _write_turnback_pair_debug_episodes_csv(
    vas,
    *,
    out_csv: str,
    inner_deltas_mm: np.ndarray,
    outer_deltas_mm: np.ndarray,
    legacy_pair_deltas: bool,
    border_width_mm: float,
    radius_offset_px: float,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
) -> int:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n_rows = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DEBUG_EPISODE_FIELDS)
        writer.writeheader()

        for vi, va in enumerate(vas):
            vid = getattr(va, "fn", f"va_{vi}")
            windows = _selected_windows_for_va(
                va,
                selected_trainings,
                skip_first=skip_first,
                keep_first=keep_first,
                last_sync_buckets=last_sync_buckets,
            )
            if not windows:
                continue
            windows_by_training = {int(win["training_idx"]): [win] for win in windows}

            for pair_idx, (inner_delta_mm, outer_delta_mm) in enumerate(
                zip(inner_deltas_mm, outer_deltas_mm)
            ):
                for fly_idx, trj in enumerate(getattr(va, "trx", [])):
                    if getattr(trj, "_bad", False):
                        continue
                    if fly_idx > 1:
                        continue
                    if fly_idx == 1 and getattr(va, "noyc", False):
                        continue

                    fly_role = "exp" if fly_idx == 0 else "ctrl"
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
                            inner_radius_mm=(
                                None if legacy_pair_deltas else float(inner_delta_mm)
                            ),
                            inner_delta_mm=(
                                float(inner_delta_mm) if legacy_pair_deltas else None
                            ),
                            outer_radius_mm=(
                                None if legacy_pair_deltas else float(outer_delta_mm)
                            ),
                            outer_delta_mm=(
                                float(outer_delta_mm) if legacy_pair_deltas else None
                            ),
                            border_width_mm=float(border_width_mm),
                            debug=False,
                            radius_offset_px=float(radius_offset_px),
                        )
                        if not episodes:
                            continue

                        for ep_idx, ep in enumerate(episodes):
                            event_t = int(ep["stop"]) - 1
                            matched_windows = [
                                win
                                for win in windows_by_training[t_idx]
                                if _frame_in_windows(event_t, [win])
                            ]
                            if not matched_windows:
                                continue
                            for win in matched_windows:
                                row = {
                                    "video_index": int(vi),
                                    "video_id": vid,
                                    "fly_idx": int(fly_idx),
                                    "fly_role": fly_role,
                                    "training_idx": int(t_idx),
                                    "training_name": trn.name(),
                                    "pair_idx": int(pair_idx),
                                    "requested_inner_mm": float(inner_delta_mm),
                                    "requested_outer_mm": float(outer_delta_mm),
                                    "pair_mode": (
                                        "delta" if legacy_pair_deltas else "radius"
                                    ),
                                    "window_start": int(win["start"]),
                                    "window_stop": int(win["stop"]),
                                    "episode_idx": int(ep_idx),
                                }
                                for key in _DEBUG_EPISODE_FIELDS:
                                    if key not in row and key in ep:
                                        row[key] = ep[key]
                                writer.writerow(row)
                                n_rows += 1
    return n_rows


def export_turnback_excursion_bin_sli_bundle(vas, opts, gls, out_fn):
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    pair_deltas_mm = _pair_deltas_mm(opts)
    legacy_pair_deltas = not (
        getattr(opts, "turnback_excursion_bin_radius_pairs_mm", None) is not None
        and str(getattr(opts, "turnback_excursion_bin_radius_pairs_mm")).strip()
    )
    requested_bin_edges_mm = None if pair_deltas_mm is not None else _bin_edges_mm(opts)
    legacy_bin_edges = _uses_legacy_bin_edges(opts)
    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync_buckets = _effective_windowing(opts)
    inner_radius_mm = getattr(opts, "turnback_inner_radius_mm", None)
    inner_delta_mm = getattr(opts, "turnback_inner_delta_mm", None)
    if inner_radius_mm is not None:
        inner_radius_mm = float(inner_radius_mm)
    elif inner_delta_mm is None:
        inner_delta_mm = 0.0
    if inner_delta_mm is not None:
        inner_delta_mm = float(inner_delta_mm)
    border_width_mm = float(getattr(opts, "turnback_border_width_mm", 0.1) or 0.1)
    radius_offset_px = float(
        getattr(opts, "turnback_inner_radius_offset_px", 0.0) or 0.0
    )
    debug = bool(getattr(opts, "turnback_excursion_bin_debug", False))
    min_episodes = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)

    if pair_deltas_mm is None:
        bin_edges_mm, open_ended_upper_bin = _resolve_open_ended_upper_edge(
            vas_ok,
            bin_edges_mm=requested_bin_edges_mm,
            inner_delta_mm=inner_delta_mm,
            border_width_mm=border_width_mm,
            radius_offset_px=radius_offset_px,
            inner_radius_mm=inner_radius_mm,
            legacy_bin_edges=legacy_bin_edges,
            selected_trainings=selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
            debug=debug,
        )
        (
            ratio_exp,
            ratio_ctrl,
            turn_exp,
            turn_ctrl,
            total_exp,
            total_ctrl,
            windows_meta,
        ) = _compute_turnback_curves(
            vas_ok,
            bin_edges_mm=bin_edges_mm,
            inner_delta_mm=inner_delta_mm,
            border_width_mm=border_width_mm,
            radius_offset_px=radius_offset_px,
            inner_radius_mm=inner_radius_mm,
            legacy_bin_edges=legacy_bin_edges,
            selected_trainings=selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
            debug=debug,
            min_episodes=min_episodes,
        )
        pair_inner_deltas_mm = None
        pair_outer_deltas_mm = None
    else:
        pair_inner_deltas_mm, pair_outer_deltas_mm = pair_deltas_mm
        open_ended_upper_bin = False
        bin_edges_mm = np.asarray([], dtype=float)
        (
            ratio_exp,
            ratio_ctrl,
            turn_exp,
            turn_ctrl,
            total_exp,
            total_ctrl,
            windows_meta,
        ) = _compute_pair_curves(
            vas_ok,
            inner_deltas_mm=pair_inner_deltas_mm,
            outer_deltas_mm=pair_outer_deltas_mm,
            legacy_pair_deltas=legacy_pair_deltas,
            border_width_mm=border_width_mm,
            radius_offset_px=radius_offset_px,
            selected_trainings=selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
            debug=debug,
            min_episodes=min_episodes,
        )

    n_videos = len(vas_ok)
    n_bins = int(ratio_exp.shape[1])

    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as exc:
        print(
            "[export] WARNING: failed to compute SLI for turnback-excursion-bin "
            f"bundle: {exc}"
        )
        sli = np.full((n_videos,), np.nan, dtype=float)
        sli_ts = np.full(
            (n_videos, len(getattr(vas_ok[0], "trns", []) or []), 0), np.nan
        )

    group_label = _safe_group_label(opts, gls)
    try:
        training_names = np.array([t.name() for t in vas_ok[0].trns], dtype=object)
    except Exception:
        training_names = np.array([], dtype=object)

    video_ids = np.array(
        [getattr(va, "fn", f"va_{i}") for i, va in enumerate(vas_ok)],
        dtype=object,
    )
    fly_ids = np.array([0 for _ in vas_ok], dtype=int)
    window_strings = np.asarray(
        [
            "; ".join(
                f"T{int(w['training_idx']) + 1} {w['training_name']}[{int(w['start'])},{int(w['stop'])})"
                for w in vv
            )
            for vv in windows_meta
        ],
        dtype=object,
    )

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    payload = dict(
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        group_label=np.asarray(group_label),
        training_names=np.asarray(training_names, dtype=str),
        video_ids=np.asarray(video_ids, dtype=str),
        video_fns=np.asarray(video_ids, dtype=str),
        fly_ids=np.asarray(fly_ids, dtype=int),
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
        sli_select_skip_first_sync_buckets=np.array(
            (
                0
                if getattr(opts, "sli_select_skip_first_sync_buckets", None) is None
                else max(0, int(getattr(opts, "sli_select_skip_first_sync_buckets")))
            ),
            dtype=int,
        ),
        sli_select_keep_first_sync_buckets=np.array(
            (
                0
                if getattr(opts, "sli_select_keep_first_sync_buckets", None) is None
                else max(0, int(getattr(opts, "sli_select_keep_first_sync_buckets")))
            ),
            dtype=int,
        ),
        turnback_excursion_bin_ratio_exp=np.asarray(ratio_exp, dtype=float),
        turnback_excursion_bin_ratio_ctrl=np.asarray(ratio_ctrl, dtype=float),
        turnback_excursion_bin_turn_exp=np.asarray(turn_exp, dtype=float),
        turnback_excursion_bin_turn_ctrl=np.asarray(turn_ctrl, dtype=float),
        turnback_excursion_bin_total_exp=np.asarray(total_exp, dtype=int),
        turnback_excursion_bin_total_ctrl=np.asarray(total_ctrl, dtype=int),
        **episode_filter_accounting_payload(
            "episode_filter_turnback_excursion_bin_exp",
            total_exp,
            min_episodes,
        ),
        **episode_filter_accounting_payload(
            "episode_filter_turnback_excursion_bin_ctrl",
            total_ctrl,
            min_episodes,
        ),
        turnback_excursion_bin_edges_mm=np.asarray(bin_edges_mm, dtype=float),
        turnback_excursion_bin_radii_mm=np.asarray(
            [] if legacy_bin_edges else bin_edges_mm, dtype=float
        ),
        turnback_excursion_bin_requested_edges_mm=np.asarray(
            [] if requested_bin_edges_mm is None else requested_bin_edges_mm,
            dtype=float,
        ),
        turnback_excursion_bin_open_ended_upper_bin=np.array(
            open_ended_upper_bin, dtype=bool
        ),
        turnback_excursion_bin_trainings=np.asarray(selected_trainings, dtype=int),
        turnback_excursion_bin_skip_first_sync_buckets=np.array(skip_first, dtype=int),
        turnback_excursion_bin_keep_first_sync_buckets=np.array(keep_first, dtype=int),
        turnback_excursion_bin_last_sync_buckets=np.array(last_sync_buckets, dtype=int),
        turnback_excursion_bin_inner_radius_mm=np.array(
            np.nan if inner_radius_mm is None else inner_radius_mm, dtype=float
        ),
        turnback_excursion_bin_inner_delta_mm=np.array(
            np.nan if inner_delta_mm is None else inner_delta_mm, dtype=float
        ),
        turnback_excursion_bin_inner_radius_offset_px=np.array(
            radius_offset_px, dtype=float
        ),
        turnback_excursion_bin_border_width_mm=np.array(border_width_mm, dtype=float),
        min_turnback_episodes=np.array(min_episodes, dtype=int),
        turnback_excursion_bin_window_summary=window_strings,
        turnback_excursion_bin_description=np.asarray(
            (
                "Fixed dual-circle turnback probability for independent inner/outer radius pairs"
                if pair_deltas_mm is not None
                else "Exact bin-averaged turnback probability over radial distance bins from reward center"
            )
        ),
        bucket_len_min=np.array(np.nan, dtype=float),
    )
    if pair_deltas_mm is not None:
        payload.update(
            turnback_excursion_bin_pair_inner_deltas_mm=np.asarray(
                pair_inner_deltas_mm, dtype=float
            ),
            turnback_excursion_bin_pair_outer_deltas_mm=np.asarray(
                pair_outer_deltas_mm, dtype=float
            ),
            turnback_excursion_bin_pair_mode=np.array(True, dtype=bool),
        )
    validate_sli_bundle(payload, path=out_fn)
    validate_turnback_excursion_bin_bundle(payload, path=out_fn)
    np.savez_compressed(out_fn, **payload)

    debug_episodes_csv = getattr(
        opts, "turnback_excursion_bin_debug_episodes_csv", None
    )
    if debug_episodes_csv:
        if pair_deltas_mm is None:
            print(
                "[export] WARNING: --turnback-excursion-bin-debug-episodes-csv "
                "currently writes fixed pair-mode episodes only; skipping "
                "because no turnback excursion-bin pairs were requested."
            )
        else:
            n_debug_rows = _write_turnback_pair_debug_episodes_csv(
                vas_ok,
                out_csv=str(debug_episodes_csv),
                inner_deltas_mm=pair_inner_deltas_mm,
                outer_deltas_mm=pair_outer_deltas_mm,
                legacy_pair_deltas=legacy_pair_deltas,
                border_width_mm=border_width_mm,
                radius_offset_px=radius_offset_px,
                selected_trainings=selected_trainings,
                skip_first=skip_first,
                keep_first=keep_first,
                last_sync_buckets=last_sync_buckets,
            )
            print(
                "[export] Wrote turnback-excursion-bin debug episodes CSV: "
                f"{debug_episodes_csv} (rows={n_debug_rows})"
            )
    print(
        f"[export] Wrote turnback-excursion-bin+SLI bundle: {out_fn} "
        f"(n={n_videos}, bins={n_bins})"
    )
