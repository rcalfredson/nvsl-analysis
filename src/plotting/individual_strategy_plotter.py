# src/plotting/individual_strategy_plotter.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage

# ---- constants -------------------------------------------------------------

TARGET_DIST_MM = 5.0
TIMEFRAMES = ["pre_trn", "t1_start", "t2_end", "t3_end"]
TIMEFRAME_TITLES = {
    "pre_trn": "Pre-training",
    "t1_start": "Training 1 start",
    "t2_end": "Training 2 end",
    "t3_end": "Training 3 end",
}

# --- helpers for logging ---------------------------------------------------


def _warn(msg: str):
    print(f"[individual_strategy] WARNING: {msg}")


def _get_video_label(va) -> str:
    for attr in ("fn",):
        if hasattr(va, attr):
            val = getattr(va, attr)
            if val:
                return str(val)

    # Fallback: something stable-ish even if we don't know the exact attribute
    return getattr(va, "name", f"VideoAnalysis_{id(va)}")


# ---- helpers for data access ----------------------------------------------


def select_distance_bin_near_5mm(vas, target_mm: float = TARGET_DIST_MM) -> float:
    """
    Choose the distance bin (in mm) that is closest to `target_mm`, based on
    the keys in va.turn_prob_by_distance. We assume all vas share the same bins.
    """
    distances = np.array(sorted(vas[0].turn_prob_by_distance.keys()), dtype=float)
    idx = int(np.argmin(np.abs(distances - target_mm)))
    chosen = float(distances[idx])
    return chosen


def ensure_sync_med_dist(va, min_no_contact_s=None):
    """
    Ensure va.syncMedDist exists and is populated. Call this once per va
    before we try to read from it.
    """
    if not hasattr(va, "syncMedDist") or va.syncMedDist is None:
        va.bySyncBucketMedDist(min_no_contact_s=min_no_contact_s)


def extract_sharp_turn_away_timecourse(va, chosen_dist, direction_idx: int = 0):
    """
    Return array of shape (n_timeframes,) with:
        (exp - yoked) sharp-turn probability corresponding to
        “away from agarose” at the specific distance bin `chosen_dist`.

    Convention in turn_prob_by_distance:
        [timeframe][0] -> toward chamber center
        [timeframe][1] -> away from chamber center

    Because “away from agarose” = “toward chamber center”, we use
    direction_idx = 0 by default.
    """
    n_t = len(TIMEFRAMES)
    vals = np.full(n_t, np.nan, dtype=float)

    # turn_prob_by_distance[dist][role][timeframe][direction]
    per_dist = va.turn_prob_by_distance[chosen_dist]

    for ti in range(n_t):
        # role index: 0 = exp, 1 = yoked
        exp_val = per_dist[0][ti][direction_idx]
        ctrl_val = per_dist[1][ti][direction_idx]
        vals[ti] = exp_val - ctrl_val

    return vals


def gather_sharp_turn_away_matrix(vas, chosen_dist):
    """
    Stack per-video sharp-turn-away timecourses into a matrix of shape
    (n_videos, n_timeframes).
    """
    series = [extract_sharp_turn_away_timecourse(va, chosen_dist) for va in vas]
    return np.vstack(series) if series else np.empty((0, len(TIMEFRAMES)))


def extract_median_distance_bucket_series(
    va,
    training_idx: int,
    min_no_contact_s=None,
) -> np.ndarray:
    """
    Return array of shape (n_buckets,) with:
        (exp - yoked) median distance to reward-circle center,
    for the given training index.

    Uses va.syncMedDist structure created by bySyncBucketMedDist().
    """
    ensure_sync_med_dist(va, min_no_contact_s=min_no_contact_s)

    if training_idx >= len(va.syncMedDist):
        return np.array([])

    med_dict = va.syncMedDist[training_idx]
    exp_vals = np.asarray(med_dict.get("exp", []), dtype=float)

    ctrl_vals = med_dict.get("ctrl")
    if ctrl_vals is not None:
        ctrl_vals = np.asarray(ctrl_vals, dtype=float)
        # exp - yoked, bucket-wise
        return exp_vals - ctrl_vals

    # if no yoked fly, just return exp medians
    return exp_vals


def gather_median_distance_matrix(
    vas,
    training_idx: int,
    min_no_contact_s=None,
) -> np.ndarray:
    """
    Build a matrix of shape (n_videos, n_buckets) with exp−yoked median distance
    per sync bucket for the given training index. Pads with NaNs so all rows
    have the same length.
    """
    series = []
    max_len = 0

    for va in vas:
        vals = extract_median_distance_bucket_series(
            va,
            training_idx=training_idx,
            min_no_contact_s=min_no_contact_s,
        )
        series.append(vals)
        max_len = max(max_len, vals.size)

    if not series:
        return np.empty((0, 0))

    padded = []
    for vals in series:
        if vals.size < max_len:
            pad = np.full(max_len - vals.size, np.nan)
            vals = np.concatenate([vals, pad])
        padded.append(vals)

    return np.vstack(padded)


def extract_large_turn_ratio_timecourse(va) -> np.ndarray:
    """
    Return array of shape (n_trainings,) with large-turn probability per exit
    for the *experimental* fly only.

    Yoked flies are being omitted for the time being because they rarely
    produced reward-anchored exits, leading to unstable ratios.


    We assume va.large_turn_stats has shape:
        (n_trainings, n_flies, 3)
    where:
        [..., :, 0] -> sum of histogram counts (total turns)
        [..., :, 1] -> median distance of turn endpoints (mm)
        [..., :, 2] -> turn-to-exit ratio
    """
    if not hasattr(va, "large_turn_stats") or va.large_turn_stats is None:
        return np.array([])

    stats = np.asarray(va.large_turn_stats, dtype=float)

    if stats.ndim != 3 or stats.shape[0] == 0 or stats.shape[2] < 3:
        return np.array([])

    # axis 0: training index
    # axis 1: fly index (0 = exp, 1 = yoked)
    # axis 2: metric index (2 = turn-to-exit ratio)
    exp_vals = stats[:, 0, 2]

    return exp_vals


def gather_large_turn_ratio_matrix(vas) -> np.ndarray:
    """
    Stack per-video large-turn ratio timecourses into a matrix of shape
    (n_videos, n_trainings). Rows correspond to VideoAnalysis objects (pairs).
    """
    series = []
    max_len = 0

    for va in vas:
        vals = extract_large_turn_ratio_timecourse(va)
        series.append(vals)
        max_len = max(max_len, vals.size)

    if not series:
        return np.empty((0, 0))

    # Pad with NaNs so all rows have the same length (just in case)
    padded = []
    for vals in series:
        if vals.size < max_len:
            pad = np.full(max_len - vals.size, np.nan)
            vals = np.concatenate([vals, pad])
        padded.append(vals)

    return np.vstack(padded)


def extract_weaving_ratio_timecourse(va) -> np.ndarray:
    """
    Return array of shape (n_trainings,) with the fraction of reward-circle
    exits that are classified as weaving (non-large-turn) for the
    *experimental* fly only.

    Source:
        va.weaving_exit_stats[fly_idx][training_idx] = (weaving_count, total_exits)
    """
    if not hasattr(va, "weaving_exit_stats") or va.weaving_exit_stats is None:
        return np.array([])

    per_fly = va.weaving_exit_stats
    if not isinstance(per_fly, list) or len(per_fly) == 0:
        return np.array([])

    # By convention, fly index 0 = experimental
    exp_idx = 0
    if exp_idx >= len(per_fly):
        return np.array([])

    per_trn = per_fly[exp_idx]
    if not isinstance(per_trn, list) or len(per_trn) == 0:
        return np.array([])

    n_trn = len(per_trn)
    ratios = np.full(n_trn, np.nan, dtype=float)

    for t_idx in range(n_trn):
        weaving_count, total_exits = per_trn[t_idx]
        total_exits = float(total_exits)
        if total_exits > 0:
            ratios[t_idx] = float(weaving_count) / total_exits

    return ratios


def gather_weaving_ratio_matrix(vas) -> np.ndarray:
    """
    Stack per-video weaving-exit ratios into a matrix of shape
    (n_videos, n_trainings). Rows correspond to VideoAnalysis objects (pairs).
    """
    series = []
    max_len = 0

    for va in vas:
        vals = extract_weaving_ratio_timecourse(va)
        series.append(vals)
        max_len = max(max_len, vals.size)

    if not series or max_len == 0:
        return np.empty((0, 0))

    padded = []
    for vals in series:
        if vals.size < max_len:
            pad = np.full(max_len - vals.size, np.nan)
            vals = np.concatenate([vals, pad])
        padded.append(vals)

    return np.vstack(padded)


def _debug_dump_large_turn_summary(
    vas,
    selected_bottom: Sequence[int],
    selected_top: Sequence[int],
    cfg: IndividualStrategyConfig,
    opts,
):
    """
    Optional debug helper: write out a TSV summarizing large-turn stats for the
    flies in the top/bottom SLI groups.

    Controlled by opts.debug_large_turns (boolean). If that attribute is absent
    or False, this is a no-op.

    Output file:
        <out_dir>/debug_large_turns/large_turn_summary.tsv
    """
    debug_flag = getattr(opts, "debug_large_turns", False)
    if not debug_flag:
        return

    out_dir = cfg.out_dir / "debug_large_turns"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "large_turn_summary.tsv"

    with open(out_path, "w") as f:
        f.write(
            "group\trow_idx\tvideo\tfly\ttraining_idx\trole\tturns\tturn_to_exit_ratio\tapprox_exits\n"
        )

        # Combine bottom + top groups, but keep label of which group each belongs to
        for group_name, indices in (("bottom", selected_bottom), ("top", selected_top)):
            for row_idx in indices:
                if not (0 <= row_idx < len(vas)):
                    continue

                va = vas[row_idx]
                video_label = _get_video_label(va)
                fly_index = getattr(va, "f", None)

                if not hasattr(va, "large_turn_stats") or va.large_turn_stats is None:
                    continue

                stats = np.asarray(va.large_turn_stats, dtype=float)
                if stats.ndim != 3 or stats.shape[0] == 0 or stats.shape[2] < 3:
                    continue

                n_trns, n_flies, _ = stats.shape

                for trn_idx in range(n_trns):
                    for fly_idx in range(n_flies):
                        turns = stats[trn_idx, fly_idx, 0]
                        ratio = stats[trn_idx, fly_idx, 2]

                        # Infer approximate number of exits when possible
                        if np.isfinite(turns) and np.isfinite(ratio) and ratio > 0:
                            approx_exits = turns / ratio
                        else:
                            approx_exits = np.nan

                        role = "exp" if fly_idx == 0 else "yoked"

                        f.write(
                            f"{group_name}\t"
                            f"{row_idx}\t"
                            f"{video_label}\t"
                            f"{fly_index}\t"
                            f"{trn_idx + 1}\t"
                            f"{role}\t"
                            f"{turns:.6g}\t"
                            f"{ratio:.6g}\t"
                            f"{approx_exits:.6g}\n"
                        )


# ---- plotting helpers ------------------------------------------------------


@dataclass
class IndividualStrategyConfig:
    out_dir: Path
    image_format: str = "png"
    top_color: str = "#1f4da1"  # bluish for top learners
    bottom_color: str = "#a00000"  # reddish for bottom learners
    alpha_top: float = 0.9
    alpha_bottom: float = 0.9


def _plot_with_gapped_markers(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    alpha: float,
    linewidth: float = 1.5,
    marker_size_line: float = 3.0,
    marker_size_single: float = 4.0,
):
    """
    Plot y(x) with:
      - line segments only over contiguous finite regions
      - markers at all finite points
      - isolated points (singletons) as marker-only (no line).

    This preserves information when NaNs break the timecourse.
    """
    x = np.asarray(x)
    y = np.asarray(y, float)

    finite = np.isfinite(y)
    if not finite.any():
        return

    # Find contiguous True-runs in `finite`
    # We pad with False on both ends so diff() gives boundaries:
    # e.g. finite = [F, T, T, F, T] -> transitions at indices [1, 3, 4, 5]
    # segments: [1:3), [4:5)
    boundaries = np.where(np.diff(np.concatenate(([False], finite, [False]))))[0]

    for start, end in zip(boundaries[0::2], boundaries[1::2]):
        seg_len = end - start
        seg_x = x[start:end]
        seg_y = y[start:end]

        if seg_len == 1:
            # Single point -> show as marker only
            ax.plot(
                seg_x,
                seg_y,
                linestyle="none",
                marker="o",
                color=color,
                alpha=alpha,
                markersize=marker_size_single,
            )
        else:
            # Contiguous run >= 2 points -> line + markers
            ax.plot(
                seg_x,
                seg_y,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                marker="o",
                markersize=marker_size_line,
            )


def _plot_overlays(
    *,
    title: str,
    y_label: str,
    x: np.ndarray,
    x_labels: list[str],
    matrix: np.ndarray,
    selected_bottom: Sequence[int],
    selected_top: Sequence[int],
    cfg: IndividualStrategyConfig,
    customizer: PlotCustomizer,
    filename_stub: str,
):
    """
    Generic overlay plotting function.

    matrix: shape (n_items, n_timepoints)
    selected_bottom/top: indices into the first dimension of `matrix`
                         (as returned by select_extremes etc.).
    """
    if matrix.size == 0:
        return

    # Slightly larger default figure size
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Plot top learners
    for idx in selected_top:
        if 0 <= idx < matrix.shape[0]:
            _plot_with_gapped_markers(
                ax,
                x,
                matrix[idx, :],
                color=cfg.top_color,
                alpha=cfg.alpha_top,
                linewidth=1.5,
            )

    # Plot bottom learners
    for idx in selected_bottom:
        if 0 <= idx < matrix.shape[0]:
            _plot_with_gapped_markers(
                ax,
                x,
                matrix[idx, :],
                color=cfg.bottom_color,
                alpha=cfg.alpha_bottom,
                linewidth=1.5,
            )

    ax.axhline(0, color="0.7", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)

    # Minimal legend
    ax.plot([], [], color=cfg.top_color, label="Top learners")
    ax.plot([], [], color=cfg.bottom_color, label="Bottom learners")
    ax.legend(loc="best", frameon=False)

    customizer.adjust_padding_proportionally()

    # Use user-selected image format (opts.imageFormat)
    ext = cfg.image_format or "png"
    out_path = cfg.out_dir / f"{filename_stub}.{ext}"
    writeImage(str(out_path), format=ext)
    plt.close(fig)


# ---- main entry point from postAnalyze ------------------------------------


def plot_individual_strategy_overlays(
    vas,
    gls,
    opts,
    sli_selected: tuple[Sequence[int], Sequence[int]],
    out_dir: str | Path = "imgs",
    plot_customizer: PlotCustomizer | None = None,
):
    """
    Main entry point to be called from postAnalyze once top/bottom SLI flies
    have been selected during the rpid branch.

    Currently plots:
      1) Sharp-turn probability “away from agarose” (operationally:
         turns toward chamber center) at the distance bin closest to 5 mm
         across the four session-level timeframes (pre, T1 start, T2 end, T3 end).
      2) Median distance to reward-circle center by sync bucket for the training
         used to define best/worst SLI (opts.best_worst_trn).
    """
    selected_bottom, selected_top = sli_selected
    if not selected_bottom or not selected_top:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_format = getattr(opts, "imageFormat", "png") or "png"
    cfg = IndividualStrategyConfig(out_dir=out_dir, image_format=image_format)
    customizer = plot_customizer or PlotCustomizer()

    # ------------------------------------------------------------------ #
    # 1) Sharp turns “away from agarose” at ~5 mm                        #
    # ------------------------------------------------------------------ #

    if not all(hasattr(va, "turn_prob_by_distance") for va in vas):
        _warn("turn_prob_by_distance missing — skipping sharp-turn overlays.")
    else:
        try:
            chosen_dist = select_distance_bin_near_5mm(vas, target_mm=TARGET_DIST_MM)
            st_away_matrix = gather_sharp_turn_away_matrix(vas, chosen_dist)

            x_sessions = np.arange(len(TIMEFRAMES))
            x_session_labels = [TIMEFRAME_TITLES[tf] for tf in TIMEFRAMES]

            # Build nicer subtitle depending on whether we hit 5.0 mm exactly
            if abs(chosen_dist - TARGET_DIST_MM) < 1e-6:
                dist_label = f"(distance bin = {chosen_dist:.2f} mm)"
            else:
                dist_label = f"(nearest bin to {TARGET_DIST_MM:.1f} mm: using {chosen_dist:.2f} mm)"

            title_sharp = "Sharp-turn probability away from agarose\n" f"{dist_label}"
            y_label_sharp = "Sharp-turn probability (exp − yoked)"

            _plot_overlays(
                title=title_sharp,
                y_label=y_label_sharp,
                x=x_sessions,
                x_labels=x_session_labels,
                matrix=st_away_matrix,
                selected_bottom=selected_bottom,
                selected_top=selected_top,
                cfg=cfg,
                customizer=customizer,
                filename_stub="individual_sharp_turn_away_5mm_overlays",
            )
        except Exception as e:
            _warn(f"Exception during sharp-turn plotting — skipping. Details: {e}")

    # ------------------------------------------------------------------ #
    # 2) Median distance to reward vs sync bucket                        #
    # ------------------------------------------------------------------ #

    training_idx = getattr(opts, "best_worst_trn", 1) - 1
    min_no_contact_s = getattr(opts, "min_no_contact_s", None)

    if not all(hasattr(va, "bySyncBucketMedDist") for va in vas):
        _warn("bySyncBucketMedDist unavailable — skipping median-distance overlays.")
    else:
        try:
            mdist_matrix = gather_median_distance_matrix(
                vas,
                training_idx=training_idx,
                min_no_contact_s=min_no_contact_s,
            )

            if mdist_matrix.size == 0:
                _warn("Median-distance data empty — skipping median-distance overlays.")
            else:
                n_buckets = mdist_matrix.shape[1]
                x_buckets = np.arange(n_buckets)
                x_bucket_labels = [f"Bucket {i+1}" for i in range(n_buckets)]

                title_mdist = (
                    "Median distance to reward-circle center\n"
                    f"(training {training_idx + 1}, exp − yoked)"
                )
                y_label_mdist = "Median distance to reward (mm, exp − yoked)"

                _plot_overlays(
                    title=title_mdist,
                    y_label=y_label_mdist,
                    x=x_buckets,
                    x_labels=x_bucket_labels,
                    matrix=mdist_matrix,
                    selected_bottom=selected_bottom,
                    selected_top=selected_top,
                    cfg=cfg,
                    customizer=customizer,
                    filename_stub="individual_median_distance_overlays",
                )
        except Exception as e:
            _warn(f"Exception during median-distance plotting — skipping. Details: {e}")

    # ------------------------------------------------------------------ #
    # 3) Large-turn probability per exit (reward-circle anchored)       #
    # ------------------------------------------------------------------ #

    if not all(hasattr(va, "large_turn_stats") for va in vas):
        _warn("large_turn_stats missing — skipping large-turn overlays.")
    else:
        try:
            lt_matrix = gather_large_turn_ratio_matrix(vas)

            if lt_matrix.size == 0:
                _warn("Large-turn ratio data empty — skipping large-turn overlays.")
            else:
                n_trns = lt_matrix.shape[1]
                x_trn = np.arange(n_trns)
                x_trn_labels = [f"Training {i+1}" for i in range(n_trns)]

                title_lt = "Large-turn probability after reward-circle exit\n(exp only)"
                y_label_lt = "Large turns / exits (experimental fly)"

                _plot_overlays(
                    title=title_lt,
                    y_label=y_label_lt,
                    x=x_trn,
                    x_labels=x_trn_labels,
                    matrix=lt_matrix,
                    selected_bottom=selected_bottom,
                    selected_top=selected_top,
                    cfg=cfg,
                    customizer=customizer,
                    filename_stub="individual_large_turn_ratio_overlays",
                )

                _debug_dump_large_turn_summary(
                    vas,
                    selected_bottom=selected_bottom,
                    selected_top=selected_top,
                    cfg=cfg,
                    opts=opts,
                )
        except Exception as e:
            _warn(f"Exception during large-turn plotting — skipping. Details: {e}")

    # ------------------------------------------------------------------ #
    # 4) Weaving probability per reward-circle exit (experimental only)  #
    # ------------------------------------------------------------------ #

    if not all(hasattr(va, "weaving_exit_stats") for va in vas):
        _warn("weaving_exit_stats missing — skipping weaving overlays.")
    else:
        try:
            weaving_matrix = gather_weaving_ratio_matrix(vas)

            if weaving_matrix.size == 0:
                _warn("Weaving ratio data empty — skipping weaving overlays.")
            else:
                n_trns_weaving = weaving_matrix.shape[1]
                x_weaving = np.arange(n_trns_weaving)
                x_weaving_labels = [f"Training {i+1}" for i in range(n_trns_weaving)]

                title_weaving = (
                    "Weaving-type exits after reward-circle exit\n"
                    "(experimental fly only)"
                )
                y_label_weaving = "Weaving exits / total exits (experimental fly)"

                _plot_overlays(
                    title=title_weaving,
                    y_label=y_label_weaving,
                    x=x_weaving,
                    x_labels=x_weaving_labels,
                    matrix=weaving_matrix,
                    selected_bottom=selected_bottom,
                    selected_top=selected_top,
                    cfg=cfg,
                    customizer=customizer,
                    filename_stub="individual_weaving_ratio_overlays",
                )
        except Exception as e:
            _warn(f"Exception during weaving plotting — skipping. Details: {e}")
