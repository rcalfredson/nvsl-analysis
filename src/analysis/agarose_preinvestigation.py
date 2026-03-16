from __future__ import annotations

import csv
import itertools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import numpy.core as _numpy_core

import sys

sys.modules.setdefault("numpy._core", _numpy_core)


LARGE_CT_WIDTH = 244.0
LARGE_CT_HEIGHT = 244.0
LARGE_CT_NUM_COLS = 2
LARGE_CT_PX_PER_MM = 7.0
LARGE_CT_FLOOR_TL = (0.0, 0.0)
LARGE_CT_FLOOR_BR = (244.0, 244.0)
LARGE_CT_CHAMBER_OFFSETS = (284.0, 284.0)


class FlyDetector:
    pass


class FlyDetectorUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "FlyDetector":
            return FlyDetector
        return super().find_class(module, name)


@dataclass(frozen=True)
class BundleSelectionSpec:
    training_idx: int
    bucket_start_idx: int
    bucket_end_idx: int
    bucket_start_min: float
    bucket_end_min: float
    training_name: str


def round_half_away_from_zero(x: float) -> int:
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def true_regions(mask: np.ndarray) -> list[slice]:
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    stops = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1
    return [slice(int(s), int(e)) for s, e in zip(starts, stops)]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_nanmean(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def safe_nanmedian(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def safe_nanquantile(vals: np.ndarray, q: float) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) if vals.size else float("nan")


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return FlyDetectorUnpickler(f, encoding="latin1").load()


def replace_video_ext(video_path: str, suffix: str) -> str:
    p = Path(video_path)
    return str(p.with_suffix(suffix))


def infer_fps(ts: Iterable[float]) -> float:
    arr = np.asarray(list(ts), dtype=float)
    diffs = np.diff(arr[np.isfinite(arr)])
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return float(1.0 / np.median(diffs))


def experimental_fly_indices(frame_nums) -> list[int]:
    if isinstance(frame_nums, list):
        return [i for i, d in enumerate(frame_nums) if bool(d)]
    return [0]


class LargeChamberXformer:
    def __init__(self, tm: dict):
        self.fctr = float(tm["fctr"])
        self.x = float(tm["x"])
        self.y = float(tm["y"])

    def t2f(self, x: float, y: float, f: int, *, no_mirror: bool = False) -> tuple[int, int]:
        r, c = divmod(int(f), LARGE_CT_NUM_COLS)
        x_t = float(x)
        y_t = float(y)
        if c != 0 and not no_mirror:
            x_t = LARGE_CT_WIDTH - x_t
        x_t += LARGE_CT_CHAMBER_OFFSETS[0] * c + 4.0
        y_t += LARGE_CT_CHAMBER_OFFSETS[1] * r + 4.0
        return (
            round_half_away_from_zero(x_t * self.fctr + self.x),
            round_half_away_from_zero(y_t * self.fctr + self.y),
        )

    def floor(self, f: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            self.t2f(*LARGE_CT_FLOOR_TL, f, no_mirror=True),
            self.t2f(*LARGE_CT_FLOOR_BR, f, no_mirror=True),
        )

    def arena_wells(self, f: int) -> tuple[float, tuple[tuple[float, float], ...]]:
        well_radius = 4.0 * LARGE_CT_PX_PER_MM * self.fctr
        flr = np.array(self.floor(f), dtype=float)
        x_half, y_half = np.mean(flr, axis=0)
        wells = (
            (flr[0][0] + well_radius, y_half),
            (x_half, flr[0][1] + well_radius),
            (flr[1][0] - well_radius, y_half),
            (x_half, flr[1][1] - well_radius),
        )
        return float(well_radius), tuple((float(x), float(y)) for x, y in wells)


def preprocess_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lost_frac: float = 0.1,
    suspicious_jump_len_px: float = 30.0,
    suspicious_dist_px: float = 10.0,
    suspicious_frac_thresh: float = 0.03,
    suspicious_num_thresh: int = 3,
) -> dict:
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    nan_mask = np.isnan(x)
    nan_regions = true_regions(nan_mask)
    trx_start_idx = 0
    if nan_regions and nan_regions[0].start == 0:
        trx_start_idx = int(nan_regions[0].stop)
        nan_regions = nan_regions[1:]

    lost_lengths = np.array([r.stop - r.start for r in nan_regions], dtype=int)
    denom = max(1, len(x) - trx_start_idx)
    lost_frac = float(lost_lengths.sum() / denom)
    lost_bad = bool(lost_frac > max_lost_frac)

    for region in nan_regions:
        f0 = int(region.start)
        f1 = int(region.stop)
        xp = [f0 - 1, f1]
        for arr in (x, y):
            right = arr[f1] if f1 < len(arr) else arr[f0 - 1]
            arr[region] = np.interp(range(f0, f1), xp, [arr[f0 - 1], right])

    dist_fw = np.full_like(x, np.nan, dtype=float)
    if len(x) > 1:
        dist_fw[:-1] = np.linalg.norm(np.vstack((np.diff(x), np.diff(y))), axis=0)
    dist_fw[np.isnan(dist_fw)] = 0.0

    jump_starts = np.flatnonzero(dist_fw > suspicious_jump_len_px)

    def _dist(i: int, j: int) -> float:
        return float(np.linalg.norm((x[j] - x[i], y[j] - y[i])))

    suspicious_pairs = 0
    for prev, cur in zip(jump_starts[:-1], jump_starts[1:]):
        if _dist(prev + 1, cur) + _dist(prev, cur + 1) < suspicious_dist_px:
            suspicious_pairs += 1
    jump_frac = float(suspicious_pairs / len(jump_starts)) if len(jump_starts) else 0.0
    suspicious_bad = bool(
        jump_frac >= suspicious_frac_thresh and suspicious_pairs >= suspicious_num_thresh
    )

    return {
        "x": x,
        "y": y,
        "lost_frame_frac": lost_frac,
        "lost_frame_count": int(lost_lengths.sum()),
        "lost_sequence_count": int(len(lost_lengths)),
        "max_lost_sequence": int(lost_lengths.max()) if len(lost_lengths) else 0,
        "long_jump_count": int(len(jump_starts)),
        "suspicious_jump_pairs": int(suspicious_pairs),
        "is_bad_reconstruction": bool(lost_bad or suspicious_bad),
    }


def calc_in_circle(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    border_width_px: float,
) -> np.ndarray:
    dc = np.linalg.norm(np.vstack((x - cx, y - cy)), axis=0)
    return ((dc < radius_px).astype(int) + (dc < radius_px + border_width_px)) > 0


def reconstruct_pretraining_diagnostics(
    video_path: str,
    *,
    delta_mm: float = 1.0,
    pre_window_min: float = 10.0,
) -> list[dict]:
    data = load_pickle(replace_video_ext(video_path, ".data"))
    trx = load_pickle(replace_video_ext(video_path, ".trx"))
    proto = data["protocol"]
    if proto.get("ct") != "large":
        raise ValueError(f"{video_path}: expected large chamber, got {proto.get('ct')!r}")

    frame_nums = proto["frameNums"]
    info = proto["info"]
    exp_flies = experimental_fly_indices(frame_nums)
    fps = infer_fps(trx["ts"])
    xf = LargeChamberXformer(proto["tm"])
    rows: list[dict] = []
    border_width_px = 0.1 * LARGE_CT_PX_PER_MM * xf.fctr

    for exp_fly in exp_flies:
        x_raw = np.asarray(trx["x"][exp_fly], dtype=float)
        y_raw = np.asarray(trx["y"][exp_fly], dtype=float)
        pp = preprocess_trajectory(x_raw, y_raw, max_lost_frac=0.1)
        x = pp["x"]
        y = pp["y"]

        fns = frame_nums[exp_fly] if isinstance(frame_nums, list) else frame_nums
        pre_stop = int(fns["startTrain"][0])
        pre_start = max(int(fns["startPre"][0]), pre_stop - round_half_away_from_zero(pre_window_min * 60.0 * fps))
        if pre_start >= pre_stop:
            pre_slice = slice(pre_stop, pre_stop)
        else:
            pre_slice = slice(pre_start, pre_stop)

        inner_radius_px, centers = xf.arena_wells(exp_fly)
        outer_radius_px = inner_radius_px + delta_mm * LARGE_CT_PX_PER_MM * xf.fctr
        in_outer = np.zeros(len(x), dtype=bool)
        in_inner = np.zeros(len(x), dtype=bool)
        perwell_outer = []
        perwell_inner = []

        for cx, cy in centers:
            outer = calc_in_circle(x, y, cx, cy, outer_radius_px, border_width_px)
            inner = calc_in_circle(x, y, cx, cy, inner_radius_px, border_width_px)
            in_outer |= outer
            in_inner |= inner
            perwell_outer.append(outer)
            perwell_inner.append(inner)

        pre_outer = in_outer[pre_slice]
        pre_inner = in_inner[pre_slice]
        pre_outer_regions = [
            region for region in true_regions(in_outer) if pre_start <= region.start < pre_stop
        ]
        avoid_count = 0
        episode_lengths = []
        for region in pre_outer_regions:
            episode_lengths.append(int(region.stop - region.start))
            if not np.any(in_inner[region]):
                avoid_count += 1
        total_count = len(pre_outer_regions)
        contact_count = total_count - avoid_count

        perwell_visits = []
        perwell_frames = []
        perwell_inner_frames = []
        wells_outer_visited = 0
        wells_inner_visited = 0
        for outer, inner in zip(perwell_outer, perwell_inner):
            o_pre = outer[pre_slice]
            i_pre = inner[pre_slice]
            perwell_frames.append(int(np.count_nonzero(o_pre)))
            perwell_inner_frames.append(int(np.count_nonzero(i_pre)))
            visits = sum(1 for region in true_regions(outer) if pre_start <= region.start < pre_stop)
            perwell_visits.append(int(visits))
            wells_outer_visited += int(np.any(o_pre))
            wells_inner_visited += int(np.any(i_pre))

        n_outer_overlap_ge2 = 0
        if perwell_outer:
            stacked_outer = np.vstack([outer[pre_slice] for outer in perwell_outer])
            n_outer_overlap_ge2 = int(np.count_nonzero(np.sum(stacked_outer, axis=0) >= 2))

        rows.append(
            {
                "video_id": str(video_path),
                "row_uid": f"{video_path}__expfly{exp_fly}",
                "exp_fly_index": int(exp_fly),
                "ctrl_fly_index": int(exp_fly + len(exp_flies)),
                "fps_reconstructed": float(fps),
                "pre_start_frame_reconstructed": int(pre_start),
                "pre_stop_frame_reconstructed": int(pre_stop),
                "pre_window_min_reconstructed": float((pre_stop - pre_start) / (60.0 * fps)) if pre_stop > pre_start else 0.0,
                "is_bad_reconstruction": bool(pp["is_bad_reconstruction"]),
                "lost_frame_frac": float(pp["lost_frame_frac"]),
                "lost_frame_count": int(pp["lost_frame_count"]),
                "lost_sequence_count": int(pp["lost_sequence_count"]),
                "max_lost_sequence": int(pp["max_lost_sequence"]),
                "long_jump_count": int(pp["long_jump_count"]),
                "suspicious_jump_pairs": int(pp["suspicious_jump_pairs"]),
                "recon_pre_avoid_count": int(avoid_count),
                "recon_pre_contact_count": int(contact_count),
                "recon_pre_total_count": int(total_count),
                "recon_pre_ratio": float(avoid_count / total_count) if total_count > 0 else float("nan"),
                "recon_pre_outer_frame_frac": float(np.mean(pre_outer)) if pre_outer.size else float("nan"),
                "recon_pre_inner_frame_frac": float(np.mean(pre_inner)) if pre_inner.size else float("nan"),
                "recon_pre_outer_not_inner_frac": float(np.mean(pre_outer & ~pre_inner)) if pre_outer.size else float("nan"),
                "recon_pre_outer_episode_count": int(total_count),
                "recon_pre_perwell_outer_visit_count": int(sum(perwell_visits)),
                "recon_pre_wells_outer_visited": int(wells_outer_visited),
                "recon_pre_wells_inner_visited": int(wells_inner_visited),
                "recon_pre_outer_episode_mean_frames": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
                "recon_pre_outer_episode_median_frames": float(np.median(episode_lengths)) if episode_lengths else float("nan"),
                "recon_pre_outer_overlap_ge2_frac": float(n_outer_overlap_ge2 / pre_outer.size) if pre_outer.size else float("nan"),
                "recon_pre_perwell_outer_frames_sum": int(sum(perwell_frames)),
                "recon_pre_perwell_inner_frames_sum": int(sum(perwell_inner_frames)),
                "recon_inner_radius_px": float(inner_radius_px),
                "recon_outer_radius_px": float(outer_radius_px),
            }
        )

    return rows


def load_bundle_rows(
    bundle_path: str,
    *,
    label: str | None = None,
    training_index_1based: int = 2,
    bucket_index: int = -1,
    bucket_start_index: int | None = None,
    bucket_end_index: int | None = None,
) -> tuple[list[dict], BundleSelectionSpec]:
    bundle = np.load(bundle_path, allow_pickle=True)
    exp = np.asarray(bundle["agarose_ratio_exp"], dtype=float)
    pre = np.asarray(bundle["agarose_pre_ratio_exp"], dtype=float)
    pre_total = np.asarray(bundle["agarose_pre_total_exp"], dtype=int)
    pre_avoid = np.asarray(bundle["agarose_pre_avoid_exp"], dtype=int)
    pre_contact = pre_total - pre_avoid
    video_ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)
    n_rows, n_trainings, n_buckets = exp.shape

    training_idx = int(training_index_1based) - 1
    if training_idx < 0 or training_idx >= n_trainings:
        raise IndexError(f"training index {training_index_1based} out of range for {n_trainings} trainings")
    if bucket_start_index is None and bucket_end_index is None:
        bucket_start_idx = bucket_index if bucket_index >= 0 else n_buckets + int(bucket_index)
        bucket_end_idx = bucket_start_idx
        if (
            bucket_index == -1
            and bucket_start_idx >= 0
            and not np.any(np.isfinite(exp[:, training_idx, bucket_start_idx]))
        ):
            finite_cols = np.flatnonzero(np.any(np.isfinite(exp[:, training_idx, :]), axis=0))
            if finite_cols.size:
                bucket_start_idx = int(finite_cols[-1])
                bucket_end_idx = bucket_start_idx
    else:
        if bucket_start_index is None or bucket_end_index is None:
            raise ValueError("bucket_start_index and bucket_end_index must be provided together")
        bucket_start_idx = bucket_start_index if bucket_start_index >= 0 else n_buckets + int(bucket_start_index)
        bucket_end_idx = bucket_end_index if bucket_end_index >= 0 else n_buckets + int(bucket_end_index)
    if bucket_start_idx < 0 or bucket_end_idx >= n_buckets or bucket_end_idx < bucket_start_idx:
        raise IndexError(f"invalid bucket selection {bucket_start_idx}..{bucket_end_idx} for {n_buckets} buckets")

    post_window = exp[:, training_idx, bucket_start_idx : bucket_end_idx + 1]
    post = np.full(post_window.shape[0], np.nan, dtype=float)
    for i in range(post_window.shape[0]):
        post[i] = safe_nanmean(post_window[i])
    training_name = f"training {training_idx + 1}"
    bucket_len_min = float(np.asarray(bundle["bucket_len_min"]).reshape(()))
    spec = BundleSelectionSpec(
        training_idx=int(training_idx),
        bucket_start_idx=int(bucket_start_idx),
        bucket_end_idx=int(bucket_end_idx),
        bucket_start_min=float((bucket_start_idx + 1) * bucket_len_min),
        bucket_end_min=float((bucket_end_idx + 1) * bucket_len_min),
        training_name=training_name,
    )

    rows = []
    duplicate_counts: dict[str, int] = {}
    for video_id in video_ids:
        key = str(video_id)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1

    for i in range(n_rows):
        video_id = str(video_ids[i])
        rows.append(
            {
                "bundle_label": label or Path(bundle_path).stem,
                "bundle_path": str(bundle_path),
                "bundle_row_index": int(i),
                "video_id": video_id,
                "video_duplicate_count_in_bundle": int(duplicate_counts[video_id]),
                "bundle_pre_ratio": float(pre[i]),
                "bundle_pre_avoid_count": int(pre_avoid[i]),
                "bundle_pre_contact_count": int(pre_contact[i]),
                "bundle_pre_total_count": int(pre_total[i]),
                "bundle_post_ratio": float(post[i]),
                "bundle_delta": float(post[i] - pre[i]) if np.isfinite(pre[i]) and np.isfinite(post[i]) else float("nan"),
                "bundle_post_is_finite": bool(np.isfinite(post[i])),
                "bundle_pre_is_finite": bool(np.isfinite(pre[i])),
            }
        )
    return rows, spec


def _assignment_cost(bundle_row: dict, recon_row: dict) -> float:
    ratio_cost = abs(float(bundle_row["bundle_pre_ratio"]) - float(recon_row["recon_pre_ratio"]))
    total_cost = abs(int(bundle_row["bundle_pre_total_count"]) - int(recon_row["recon_pre_total_count"])) * 0.01
    avoid_cost = abs(int(bundle_row["bundle_pre_avoid_count"]) - int(recon_row["recon_pre_avoid_count"])) * 0.01
    bad_penalty = 0.1 if recon_row.get("is_bad_reconstruction", False) else 0.0
    return float(ratio_cost + total_cost + avoid_cost + bad_penalty)


def attach_reconstruction(bundle_rows: list[dict], recon_rows: list[dict]) -> list[dict]:
    by_video_bundle: dict[str, list[dict]] = {}
    by_video_recon: dict[str, list[dict]] = {}
    for row in bundle_rows:
        by_video_bundle.setdefault(str(row["video_id"]), []).append(row)
    for row in recon_rows:
        by_video_recon.setdefault(str(row["video_id"]), []).append(row)

    merged: list[dict] = []
    for video_id, b_rows in by_video_bundle.items():
        r_rows = by_video_recon.get(video_id, [])
        if not r_rows:
            for row in b_rows:
                merged.append({**row, "recon_match_status": "missing_raw"})
            continue

        use_rows = r_rows[:]
        if len(use_rows) < len(b_rows):
            padded = use_rows + [None] * (len(b_rows) - len(use_rows))
            candidate_perms = [tuple(padded)]
        else:
            candidate_perms = itertools.permutations(use_rows, len(b_rows))

        best_perm = None
        best_cost = float("inf")
        for perm in candidate_perms:
            cost = 0.0
            for b_row, r_row in zip(b_rows, perm):
                if r_row is None:
                    cost += 10.0
                else:
                    cost += _assignment_cost(b_row, r_row)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        if best_perm is None:
            best_perm = tuple(use_rows[: len(b_rows)]) + (None,) * max(0, len(b_rows) - len(use_rows))
        for b_row, r_row in zip(b_rows, best_perm):
            if r_row is None:
                merged.append({**b_row, "recon_match_status": "missing_fly"})
                continue
            ratio_diff = float(b_row["bundle_pre_ratio"] - r_row["recon_pre_ratio"]) if (
                np.isfinite(b_row["bundle_pre_ratio"]) and np.isfinite(r_row["recon_pre_ratio"])
            ) else float("nan")
            total_diff = int(b_row["bundle_pre_total_count"]) - int(r_row["recon_pre_total_count"])
            avoid_diff = int(b_row["bundle_pre_avoid_count"]) - int(r_row["recon_pre_avoid_count"])
            match_status = "heuristic"
            if total_diff == 0 and avoid_diff == 0 and (np.isnan(ratio_diff) or abs(ratio_diff) < 1e-12):
                match_status = "exact"
            elif np.isfinite(ratio_diff) and abs(ratio_diff) < 1e-12:
                match_status = "ratio_exact"
            merged.append(
                {
                    **b_row,
                    **r_row,
                    "recon_match_status": match_status,
                    "recon_ratio_diff": ratio_diff,
                    "recon_total_diff": int(total_diff),
                    "recon_avoid_diff": int(avoid_diff),
                    "recon_assignment_cost": float(best_cost),
                }
            )
    return merged


def summarize_rows(rows: list[dict], *, chamber_label: str, group_key: str | None = None) -> list[dict]:
    if not rows:
        return []
    groups: dict[str, list[dict]] = {}
    if group_key is None:
        groups["all"] = rows
    else:
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)

    out = []
    for group_name, g_rows in groups.items():
        pre_vals = np.array([row["bundle_pre_ratio"] for row in g_rows], dtype=float)
        post_vals = np.array([row["bundle_post_ratio"] for row in g_rows], dtype=float)
        delta_vals = np.array([row["bundle_delta"] for row in g_rows], dtype=float)
        pre_total = np.array([row["bundle_pre_total_count"] for row in g_rows], dtype=float)
        pre_avoid = np.array([row["bundle_pre_avoid_count"] for row in g_rows], dtype=float)
        pre_contact = np.array([row["bundle_pre_contact_count"] for row in g_rows], dtype=float)
        recon_outer = np.array([row.get("recon_pre_outer_frame_frac", np.nan) for row in g_rows], dtype=float)
        recon_inner = np.array([row.get("recon_pre_inner_frame_frac", np.nan) for row in g_rows], dtype=float)
        recon_bad = np.array([bool(row.get("is_bad_reconstruction", False)) for row in g_rows], dtype=bool)
        finite_pre = np.isfinite(pre_vals)
        finite_post = np.isfinite(post_vals)
        finite_delta = np.isfinite(delta_vals)
        weighted = float(np.sum(pre_avoid[pre_total > 0]) / np.sum(pre_total[pre_total > 0])) if np.any(pre_total > 0) else float("nan")
        out.append(
            {
                "chamber": chamber_label,
                "group_key": group_key or "all",
                "group_value": group_name,
                "n_rows": int(len(g_rows)),
                "n_unique_videos": int(len({str(row['video_id']) for row in g_rows})),
                "pre_ratio_mean": safe_nanmean(pre_vals),
                "pre_ratio_median": safe_nanmedian(pre_vals),
                "pre_ratio_weighted": weighted,
                "pre_ratio_p75": safe_nanquantile(pre_vals, 0.75),
                "post_ratio_mean": safe_nanmean(post_vals),
                "delta_mean": safe_nanmean(delta_vals),
                "pre_total_mean": safe_nanmean(pre_total),
                "pre_avoid_mean": safe_nanmean(pre_avoid),
                "pre_contact_mean": safe_nanmean(pre_contact),
                "pre_total_median": safe_nanmedian(pre_total),
                "n_pre_ratio_ge_0_75": int(np.count_nonzero(finite_pre & (pre_vals >= 0.75))),
                "n_pre_ratio_le_0_25": int(np.count_nonzero(finite_pre & (pre_vals <= 0.25))),
                "n_pre_total_le_3": int(np.count_nonzero(pre_total <= 3)),
                "n_pre_total_ge_30": int(np.count_nonzero(pre_total >= 30)),
                "recon_outer_frame_frac_mean": safe_nanmean(recon_outer),
                "recon_inner_frame_frac_mean": safe_nanmean(recon_inner),
                "recon_bad_frac": float(np.mean(recon_bad)) if recon_bad.size else float("nan"),
                "n_finite_post": int(np.count_nonzero(finite_post)),
                "n_finite_delta": int(np.count_nonzero(finite_delta)),
            }
        )
    return out


def add_leave_one_out_shifts(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    pre_vals = np.array([row["bundle_pre_ratio"] for row in rows], dtype=float)
    overall = safe_nanmean(pre_vals)
    by_video: dict[str, np.ndarray] = {}
    for idx, row in enumerate(rows):
        by_video.setdefault(str(row["video_id"]), []).append(idx)
    for video_id, idxs in by_video.items():
        keep = np.ones(len(rows), dtype=bool)
        keep[np.asarray(idxs, dtype=int)] = False
        loo = safe_nanmean(pre_vals[keep]) if np.any(keep) else float("nan")
        for idx in idxs:
            rows[idx]["video_leave_one_out_pre_ratio"] = loo
            rows[idx]["video_leave_one_out_shift"] = float(loo - overall) if np.isfinite(loo) else float("nan")
            rows[idx]["video_mean_pre_ratio"] = safe_nanmean(pre_vals[np.asarray(idxs, dtype=int)])
    return rows
