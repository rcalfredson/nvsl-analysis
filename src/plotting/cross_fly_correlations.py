# src/plotting/cross_fly_correlations.py

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.reward_window_utils import (
    cumulative_window_seconds_for_frame,
    frames_in_windows,
    selected_windows_for_va,
)
from src.utils.common import writeImage
from src.utils.debug_fly_groups import log_fly_group


BBOX_STYLE = dict(
    facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
)


_layout_logger = logging.getLogger("cross_fly_corr_layout")
_layout_logger.setLevel(logging.INFO)
_layout_logger_initialized = False


def init_correlation_layout_logging(log_path="debug_correlation_layout.log"):
    """
    Initialize file logging for stats-box/legend layout debugging.
    """
    global _layout_logger_initialized
    if _layout_logger_initialized:
        return

    handler = logging.FileHandler(str(log_path))
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    _layout_logger.addHandler(handler)
    _layout_logger_initialized = True


def _maybe_init_correlation_layout_logging_from_env():
    log_path = os.environ.get("CROSS_FLY_LAYOUT_DEBUG_LOG", "").strip()
    if not log_path:
        return
    init_correlation_layout_logging(
        "debug_correlation_layout.log" if log_path == "1" else log_path
    )


def _log_correlation_layout(message: str):
    if _layout_logger_initialized:
        _layout_logger.info(message)


_maybe_init_correlation_layout_logging_from_env()


@dataclass
class CorrelationPlotConfig:
    out_dir: Path
    image_format: str = "png"
    dot_color: str = "#005bbb"
    alpha: float = 0.85
    figsize: tuple = (5.5, 4.5)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


def _correlation_out_path(out_dir: Path, filename: str, image_format: str) -> Path:
    ext = image_format or "png"
    return out_dir / f"{filename}.{ext}"


@dataclass(frozen=True)
class SLIContext:
    """
    Describes what sli_values represent.
    - training_idx: 0-based index of the training whose SLI is being used
    - average_over_buckets: True => mean over sync buckets in that training
                            False => last sync bucket in that training
    """

    training_idx: int
    average_over_buckets: bool = False
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    explicit_bucket_idx: int | None = None

    def _window_bounds(self) -> tuple[int, int | None]:
        if self.explicit_bucket_idx is not None:
            sb = int(self.explicit_bucket_idx) + 1
            return sb, sb
        start_sb = int(self.skip_first_sync_buckets or 0) + 1  # 1-based
        keep = int(self.keep_first_sync_buckets or 0)
        end_sb = None if keep <= 0 else start_sb + keep - 1
        return start_sb, end_sb

    def _window_text(self, *, abbrev_sb: bool = True) -> str:
        start_sb, end_sb = self._window_bounds()
        if abbrev_sb:
            if end_sb is None:
                return f"SB{start_sb}-end"
            if end_sb == start_sb:
                return f"SB{start_sb}"
            return f"SB{start_sb}-SB{end_sb}"

        if end_sb is None:
            return f"sync bucket {start_sb}-end"
        if end_sb == start_sb:
            return f"sync bucket {start_sb}"
        return f"sync buckets {start_sb}-{end_sb}"

    def label_long(self) -> str:
        trn = self.training_idx + 1
        start_sb, end_sb = self._window_bounds()
        if start_sb == end_sb:
            return f"SLI (sync bucket {start_sb}, training {trn})"
        window_txt = (
            f", sync buckets {start_sb}-end"
            if end_sb is None
            else f", sync buckets {start_sb}-{end_sb}"
        )
        if self.average_over_buckets:
            return f"SLI (mean over{window_txt}, training {trn})"
        return f"SLI (last sync bucket within{window_txt}, training {trn})"

    def label_short(self, abbrev_sb=True) -> str:
        trn = self.training_idx + 1
        window_txt = self._window_text(abbrev_sb=abbrev_sb)
        if self._window_bounds()[0] == self._window_bounds()[1]:
            return f"SLI (T{trn}, {window_txt})"
        if self.average_over_buckets:
            return f"SLI (T{trn}, mean, {window_txt})"
        return f"SLI (T{trn}, last, {window_txt})"


def _window_context_suffix(ctx: SLIContext, *, prefix: str) -> str:
    mode = "mean" if ctx.average_over_buckets else "last"
    parts = [f"{prefix}T{ctx.training_idx + 1}", mode]
    if ctx.explicit_bucket_idx is not None:
        parts.append(f"sb{int(ctx.explicit_bucket_idx) + 1}")
        return "_".join(parts)
    skip_k = max(0, int(ctx.skip_first_sync_buckets or 0))
    keep_k = max(0, int(ctx.keep_first_sync_buckets or 0))
    if skip_k:
        parts.append(f"skip{skip_k}")
    if keep_k:
        parts.append(f"keep{keep_k}")
    return "_".join(parts)


def _windowed_metric_label(metric_name: str, ctx: SLIContext) -> str:
    window_txt = ctx._window_text(abbrev_sb=True)
    if ctx.average_over_buckets:
        return f"{metric_name}\n(mean {window_txt}, T{ctx.training_idx + 1})"
    if ctx._window_bounds()[0] == ctx._window_bounds()[1]:
        return f"{metric_name}\n(T{ctx.training_idx + 1}, {window_txt})"
    if ctx.skip_first_sync_buckets or ctx.keep_first_sync_buckets:
        return f"{metric_name}\n(last valid, {window_txt}, T{ctx.training_idx + 1})"
    if ctx.training_idx != 0:
        return f"{metric_name}\n(last valid, T{ctx.training_idx + 1})"
    return metric_name


def _first_n_reward_rate_label(
    *,
    first_n_rewards: int,
    ctx: SLIContext,
    max_time_to_nth_s: float | None = None,
    time_basis: str = "window_start",
) -> str:
    window_txt = ctx._window_text(abbrev_sb=True)
    cutoff_txt = ""
    if max_time_to_nth_s is not None and np.isfinite(float(max_time_to_nth_s)):
        cutoff_txt = f", <= {float(max_time_to_nth_s):g}s"
    basis_txt = (
        "first-to-nth span" if str(time_basis) == "first_to_nth" else "window start to nth"
    )
    return (
        f"rewards per minute\n"
        f"(first {int(first_n_rewards)} calc rewards, {basis_txt}{cutoff_txt}, {window_txt}, T{ctx.training_idx + 1})"
    )


def early_sli_label(*, training_idx: int, skip_first_sync_buckets: int) -> str:
    # training_idx is 0-based
    trn = training_idx + 1
    k = int(skip_first_sync_buckets or 0)
    sb = k + 1  # 1-based
    sb_txt = "first sync bucket" if sb == 1 else f"SB{sb}"
    return f"SLI (T{trn}, {sb_txt})"


def _compute_group_corr(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> tuple[float, float] | None:
    """
    Compute Pearson correlation for a given index set, handling NaNs and
    small sample sizes. Returns (r, p) or None if not enough valid data.
    """
    if idx is None or idx.size == 0:
        return None

    idx = np.asarray(idx, dtype=int)
    x_g = np.asarray(x, float)[idx]
    y_g = np.asarray(y, float)[idx]

    mask = np.isfinite(x_g) & np.isfinite(y_g)
    if np.sum(mask) < 3:
        return None

    return pearsonr(x_g[mask], y_g[mask])


def _place_legend_without_point_overlap(
    ax,
    handles,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scatter_artist=None,
    frameon: bool = True,
):
    """
    Place a legend so that its frame does not overlap any plotted markers.

    The helper first searches a set of standard in-axes locations. If none of
    them are clean, it adds a modest upper y headroom band and places the
    legend there.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite]
    y_f = y[finite]

    legend = ax.legend(handles=handles, loc="best", frameon=frameon)
    legend_fontsize = None
    if x_f.size == 0:
        return legend

    n_entries = len(handles)
    if n_entries > 3:
        base_fontsize = legend.get_texts()[0].get_fontsize() if legend.get_texts() else 10
        if n_entries == 4:
            scale = 0.75
        else:
            scale = max(0.75 - 0.05 * (n_entries - 4), 0.5)
        legend_fontsize = max(base_fontsize * scale, 6)
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_fontsize)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts_display = ax.transData.transform(np.column_stack([x_f, y_f]))

    marker_pad_px = 2.0
    if scatter_artist is not None:
        try:
            sizes = np.asarray(scatter_artist.get_sizes(), float)
            if sizes.size:
                marker_pad_px = max(
                    marker_pad_px,
                    float(np.sqrt(np.nanmax(sizes) / np.pi) * fig.dpi / 72.0 + 1.5),
                )
        except Exception:
            pass

    candidates = [
        "upper left",
        "upper right",
        "lower left",
        "lower right",
        "center left",
        "center right",
        "upper center",
        "lower center",
    ]

    best_loc = None
    best_overlap = None
    best_height_frac = 0.0

    for loc in candidates:
        legend.remove()
        legend = ax.legend(
            handles=handles,
            loc=loc,
            frameon=frameon,
            fontsize=legend_fontsize,
        )
        fig.canvas.draw()
        legend_bbox_raw = legend.get_window_extent(renderer=renderer)
        legend_bbox = legend_bbox_raw.expanded(
            (legend_bbox_raw.width + 2 * marker_pad_px) / max(legend_bbox_raw.width, 1.0),
            (legend_bbox_raw.height + 2 * marker_pad_px)
            / max(legend_bbox_raw.height, 1.0),
        )
        inside = (
            (pts_display[:, 0] >= legend_bbox.x0)
            & (pts_display[:, 0] <= legend_bbox.x1)
            & (pts_display[:, 1] >= legend_bbox.y0)
            & (pts_display[:, 1] <= legend_bbox.y1)
        )
        overlap = int(np.sum(inside))
        best_height_frac = max(
            best_height_frac,
            legend_bbox_raw.height / max(ax.bbox.height, 1.0),
        )
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_loc = loc
        if overlap == 0:
            _log_correlation_layout(
                f"title={ax.get_title()!r} legend_mode=in_axes loc={loc!r} "
                f"overlap_points=0 marker_pad_px={marker_pad_px:.2f}"
            )
            return legend

    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = max(float(np.nanmax(y_f) - np.nanmin(y_f)), 1.0)

    extra_top = max((best_height_frac + 0.06) * y_span, 0.16 * y_span)
    original_top = y1

    fallback_loc = "upper right"
    if best_loc in ("upper left", "lower left", "center left"):
        fallback_loc = "upper left"

    overlap = None
    for _ in range(8):
        ax.set_ylim(y0, original_top + extra_top)
        fig.canvas.draw()
        pts_display = ax.transData.transform(np.column_stack([x_f, y_f]))

        legend.remove()
        legend = ax.legend(
            handles=handles,
            loc=fallback_loc,
            frameon=frameon,
            fontsize=legend_fontsize,
        )
        fig.canvas.draw()
        legend_bbox_raw = legend.get_window_extent(renderer=renderer)
        legend_bbox = legend_bbox_raw.expanded(
            (legend_bbox_raw.width + 2 * marker_pad_px)
            / max(legend_bbox_raw.width, 1.0),
            (legend_bbox_raw.height + 2 * marker_pad_px)
            / max(legend_bbox_raw.height, 1.0),
        )
        inside = (
            (pts_display[:, 0] >= legend_bbox.x0)
            & (pts_display[:, 0] <= legend_bbox.x1)
            & (pts_display[:, 1] >= legend_bbox.y0)
            & (pts_display[:, 1] <= legend_bbox.y1)
        )
        overlap = int(np.sum(inside))
        if overlap == 0:
            break
        extra_top *= 1.35
    _log_correlation_layout(
        f"title={ax.get_title()!r} legend_mode=headroom loc={fallback_loc!r} "
        f"best_in_axes_loc={best_loc!r} overlap_points={overlap} "
        f"extra_top={extra_top:.4f} original_top={original_top:.4f}"
    )
    return legend


def _add_smart_stats_box(
    ax,
    text: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fontsize: int = 10,
    max_overlap_frac: float = 0.08,
):
    """
    Place a stats textbox where it obscures as few points as possible.

    The function first tries the four plot corners. If each candidate would
    still cover a substantial fraction of points, it adds upper y headroom and
    moves the textbox into that empty band above the scatter cloud.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite]
    y_f = y[finite]

    if x_f.size == 0:
        return ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            zorder=5,
            bbox=BBOX_STYLE,
        )

    fig = ax.figure
    probe = ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        zorder=5,
        alpha=0.0,
        bbox=BBOX_STYLE,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts_display = ax.transData.transform(np.column_stack([x_f, y_f]))
    legend = ax.get_legend()

    def _legend_metrics():
        if legend is None:
            return None, None, None, None
        legend_bbox_local = legend.get_window_extent(renderer=renderer)
        legend_pts_axes = ax.transAxes.inverted().transform(
            np.array(
                [
                    [legend_bbox_local.x0, legend_bbox_local.y0],
                    [legend_bbox_local.x1, legend_bbox_local.y1],
                ]
            )
        )
        legend_axes_bbox_local = (
            float(np.min(legend_pts_axes[:, 0])),
            float(np.min(legend_pts_axes[:, 1])),
            float(np.max(legend_pts_axes[:, 0])),
            float(np.max(legend_pts_axes[:, 1])),
        )
        legend_top_frac_local = float(np.max(legend_pts_axes[:, 1]))
        legend_center_x_frac_local = float(np.mean(legend_pts_axes[:, 0]))
        return (
            legend_bbox_local,
            legend_top_frac_local,
            legend_center_x_frac_local,
            legend_axes_bbox_local,
        )

    legend_bbox, legend_top_frac, legend_center_x_frac, legend_axes_bbox = (
        _legend_metrics()
    )

    candidates = [
        dict(x=0.05, y=0.95, ha="left", va="top"),
        dict(x=0.95, y=0.95, ha="right", va="top"),
        dict(x=0.05, y=0.05, ha="left", va="bottom"),
        dict(x=0.95, y=0.05, ha="right", va="bottom"),
    ]

    best_candidate = None
    best_overlap = None
    best_patch_bbox = None
    best_raw_overlap = None

    for candidate in candidates:
        probe.set_position((candidate["x"], candidate["y"]))
        probe.set_ha(candidate["ha"])
        probe.set_va(candidate["va"])
        fig.canvas.draw()
        patch_bbox = probe.get_bbox_patch().get_window_extent(renderer=renderer)
        inside = (
            (pts_display[:, 0] >= patch_bbox.x0)
            & (pts_display[:, 0] <= patch_bbox.x1)
            & (pts_display[:, 1] >= patch_bbox.y0)
            & (pts_display[:, 1] <= patch_bbox.y1)
        )
        raw_overlap = float(np.mean(inside))
        overlap = raw_overlap
        if legend_bbox is not None:
            overlaps_legend = not (
                patch_bbox.x1 < legend_bbox.x0
                or patch_bbox.x0 > legend_bbox.x1
                or patch_bbox.y1 < legend_bbox.y0
                or patch_bbox.y0 > legend_bbox.y1
            )
            if overlaps_legend:
                overlap = 1.0 + overlap
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_candidate = candidate
            best_patch_bbox = patch_bbox
            best_raw_overlap = raw_overlap

    probe.remove()

    if best_candidate is not None and best_overlap is not None:
        if best_overlap <= max_overlap_frac:
            text_artist = ax.text(
                best_candidate["x"],
                best_candidate["y"],
                text,
                transform=ax.transAxes,
                va=best_candidate["va"],
                ha=best_candidate["ha"],
                fontsize=fontsize,
                zorder=5,
                bbox=BBOX_STYLE,
            )
            fig.canvas.draw()
            stats_bbox = text_artist.get_bbox_patch().get_window_extent(renderer=renderer)
            intersects_legend = (
                legend_bbox is not None
                and not (
                    stats_bbox.x1 < legend_bbox.x0
                    or stats_bbox.x0 > legend_bbox.x1
                    or stats_bbox.y1 < legend_bbox.y0
                    or stats_bbox.y0 > legend_bbox.y1
                )
            )
            stats_pts_axes = ax.transAxes.inverted().transform(
                np.array([[stats_bbox.x0, stats_bbox.y0], [stats_bbox.x1, stats_bbox.y1]])
            )
            stats_axes_bbox = (
                float(np.min(stats_pts_axes[:, 0])),
                float(np.min(stats_pts_axes[:, 1])),
                float(np.max(stats_pts_axes[:, 0])),
                float(np.max(stats_pts_axes[:, 1])),
            )
            _log_correlation_layout(
                f"title={ax.get_title()!r} mode=corner candidate={best_candidate} "
                f"raw_overlap={best_raw_overlap:.4f} score={best_overlap:.4f} "
                f"legend_axes_bbox={legend_axes_bbox} stats_axes_bbox={stats_axes_bbox} "
                f"intersects_legend={intersects_legend}"
            )
            return text_artist

    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = max(float(np.nanmax(y_f) - np.nanmin(y_f)), 1.0)

    box_height_frac = 0.18
    if best_patch_bbox is not None and ax.bbox.height > 0:
        box_height_frac = best_patch_bbox.height / ax.bbox.height

    legend_nudge_down = 0.0
    if legend is not None and legend_top_frac is not None:
        desired_legend_top_frac = max(0.55, 0.97 - box_height_frac - 0.02)
        if legend_top_frac > desired_legend_top_frac:
            legend_nudge_down = legend_top_frac - desired_legend_top_frac
            anchor_bbox = legend.get_bbox_to_anchor().transformed(
                ax.transAxes.inverted()
            )
            legend.set_bbox_to_anchor(
                (
                    float(anchor_bbox.x0),
                    float(anchor_bbox.y0),
                    float(anchor_bbox.width),
                    max(float(anchor_bbox.height) - legend_nudge_down, 0.20),
                ),
                transform=ax.transAxes,
            )
            fig.canvas.draw()
            legend_bbox, legend_top_frac, legend_center_x_frac, legend_axes_bbox = (
                _legend_metrics()
            )

    extra_top = max((box_height_frac + 0.08) * y_span, 0.18 * y_span)
    legend_clear_frac = None
    if legend_top_frac is not None:
        legend_clear_frac = np.clip(legend_top_frac - 0.03, 0.10, 0.95)
        extra_top = max(extra_top, y_span * (1.0 / legend_clear_frac - 1.0))
    original_top = y1
    ax.set_ylim(y0, y1 + extra_top)

    x0, x1 = ax.get_xlim()
    x_span = x1 - x0
    if not np.isfinite(x_span) or x_span <= 0:
        x_span = max(float(np.nanmax(x_f) - np.nanmin(x_f)), 1.0)

    stats_x = x0 + 0.02 * x_span
    stats_ha = "left"
    if legend_center_x_frac is not None and legend_center_x_frac < 0.5:
        stats_x = x1 - 0.02 * x_span
        stats_ha = "right"

    top_margin_frac = 0.03
    text_artist = ax.text(
        0.98 if stats_ha == "right" else 0.02,
        1.0 - top_margin_frac,
        text,
        transform=ax.transAxes,
        va="top",
        ha=stats_ha,
        fontsize=fontsize,
        zorder=5,
        bbox=BBOX_STYLE,
    )
    fig.canvas.draw()
    stats_bbox = text_artist.get_bbox_patch().get_window_extent(renderer=renderer)
    intersects_legend = (
        legend_bbox is not None
        and not (
            stats_bbox.x1 < legend_bbox.x0
            or stats_bbox.x0 > legend_bbox.x1
            or stats_bbox.y1 < legend_bbox.y0
            or stats_bbox.y0 > legend_bbox.y1
        )
    )
    vertical_gap_px = None
    if legend_bbox is not None:
        vertical_gap_px = max(stats_bbox.y0 - legend_bbox.y1, legend_bbox.y0 - stats_bbox.y1)
    stats_pts_axes = ax.transAxes.inverted().transform(
        np.array([[stats_bbox.x0, stats_bbox.y0], [stats_bbox.x1, stats_bbox.y1]])
    )
    stats_axes_bbox = (
        float(np.min(stats_pts_axes[:, 0])),
        float(np.min(stats_pts_axes[:, 1])),
        float(np.max(stats_pts_axes[:, 0])),
        float(np.max(stats_pts_axes[:, 1])),
    )
    _log_correlation_layout(
        f"title={ax.get_title()!r} mode=headroom candidate={best_candidate} "
        f"raw_overlap={best_raw_overlap:.4f} score={best_overlap:.4f} "
        f"box_height_frac={box_height_frac:.4f} legend_top_frac={legend_top_frac} "
        f"legend_nudge_down={legend_nudge_down:.4f} "
        f"legend_clear_frac={legend_clear_frac} extra_top={extra_top:.4f} "
        f"legend_axes_bbox={legend_axes_bbox} stats_axes_bbox={stats_axes_bbox} "
        f"vertical_gap_px={vertical_gap_px} intersects_legend={intersects_legend}"
    )
    return text_artist


def _normalize_selected_groups(
    sli_selected: tuple[Sequence[int], Sequence[int]] | None,
    sli_extremes: str | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Normalize caller-provided selected groups.

    Returns
    -------
    bottom_idx, top_idx, mode
        mode is one of {"top", "bottom", "both"} or None if nothing to plot.
    """
    if sli_selected is None:
        return np.array([], dtype=int), np.array([], dtype=int), None

    bottom_raw, top_raw = sli_selected
    bottom_idx = np.asarray(bottom_raw if bottom_raw is not None else [], dtype=int)
    top_idx = np.asarray(top_raw if top_raw is not None else [], dtype=int)

    mode = sli_extremes or "both"

    if mode == "top":
        if top_idx.size == 0:
            return bottom_idx, top_idx, None
        return np.array([], dtype=int), top_idx, "top"

    if mode == "bottom":
        if bottom_idx.size == 0:
            return bottom_idx, top_idx, None
        return bottom_idx, np.array([], dtype=int), "bottom"

    # default: both
    if bottom_idx.size == 0 and top_idx.size == 0:
        return bottom_idx, top_idx, None

    return bottom_idx, top_idx, "both"


def plot_selected_group_scatter(
    *,
    x: np.ndarray,
    y: np.ndarray,
    bottom_idx: np.ndarray,
    top_idx: np.ndarray,
    mode: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
    out_dir: Path,
    customizer: PlotCustomizer,
    top_label: str = "Top SLI-selected",
    bottom_label: str = "Bottom SLI-selected",
    other_label: str = "Other",
    top_color: str = "#1f77b4",
    bottom_color: str = "#cc0000",
    other_color: str = "#aaaaaa",
    alpha: float = 0.85,
    figsize: tuple = (5.5, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    include_all_corr: bool = False,
    image_format: str = "png",
):
    """
    Plot all points, highlighting selected top/bottom SLI groups and reporting
    correlations for the highlighted group(s) only.

    mode:
        "top"     -> highlight top group only
        "bottom"  -> highlight bottom group only
        "both"    -> highlight both groups
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        print(f"[correlations] WARNING: not enough valid data for {filename}")
        return

    x_f = x[mask]
    y_f = y[mask]
    valid_global_idx = np.arange(x.shape[0])[mask]

    bottom_set = set(np.asarray(bottom_idx, dtype=int).tolist())
    top_set = set(np.asarray(top_idx, dtype=int).tolist())

    if mode == "both":
        overlap = top_set & bottom_set
        if overlap:
            print()

    classes = []
    point_colors = []

    for idx in valid_global_idx:
        if mode in ("both", "top") and idx in top_set:
            cls = "top"
            color = top_color
        elif mode in ("both", "bottom") and idx in bottom_set:
            cls = "bottom"
            color = bottom_color
        else:
            cls = "other"
            color = other_color
        classes.append(cls)
        point_colors.append(color)

    classes_arr = np.asarray(classes, dtype=object)

    fig, ax = plt.subplots(figsize=figsize)
    scatter_artist = ax.scatter(x_f, y_f, c=point_colors, alpha=alpha)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)
    ax.grid(False)

    corr_all = None
    corr_top = None
    corr_bottom = None
    if include_all_corr and x_f.size >= 3:
        r_a, p_a = pearsonr(x_f, y_f)
        corr_all = (float(r_a), float(p_a), int(x_f.size))

    if mode in ("both", "top"):
        plotted_top_mask = classes_arr == "top"
        if np.sum(plotted_top_mask) >= 3:
            r_t, p_t = pearsonr(x_f[plotted_top_mask], y_f[plotted_top_mask])
            corr_top = (float(r_t), float(p_t), int(np.sum(plotted_top_mask)))

    if mode in ("both", "bottom"):
        plotted_bottom_mask = classes_arr == "bottom"
        if np.sum(plotted_bottom_mask) >= 3:
            r_b, p_b = pearsonr(x_f[plotted_bottom_mask], y_f[plotted_bottom_mask])
            corr_bottom = (float(r_b), float(p_b), int(np.sum(plotted_bottom_mask)))

    lines = []

    if include_all_corr:
        if corr_all is not None:
            r_a, p_a, n_a = corr_all
            lines.append(f"All (finite): r = {r_a:.3f}, p = {p_a:.3g} (n={n_a})")
        else:
            lines.append("All (finite): r = n/a")

    if mode in ("both", "top"):
        if corr_top is not None:
            r_t, p_t, n_t = corr_top
            lines.append(f"{top_label}: r = {r_t:.3f}, p = {p_t:.3g} (n={n_t})")
        else:
            lines.append(f"{top_label}: r = n/a")

    if mode in ("both", "bottom"):
        if corr_bottom is not None:
            r_b, p_b, n_b = corr_bottom
            lines.append(f"{bottom_label}: r = {r_b:.3f}, p = {p_b:.3g} (n={n_b})")
        else:
            lines.append(f"{bottom_label}: r = n/a")

    handles = []
    if mode in ("both", "top"):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=top_color,
                markersize=8,
                label=top_label,
            )
        )

    if mode in ("both", "bottom"):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=bottom_color,
                markersize=8,
                label=bottom_label,
            )
        )

    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=other_color,
            markersize=8,
            label=other_label,
        )
    )
    _place_legend_without_point_overlap(
        ax, handles, x_f, y_f, scatter_artist=scatter_artist, frameon=True
    )
    _add_smart_stats_box(ax, "\n".join(lines), x_f, y_f)

    customizer.adjust_padding_proportionally()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = _correlation_out_path(out_dir, filename, image_format)
    writeImage(str(out_path), format=image_format)
    plt.close(fig)


def _scatter_with_corr(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cfg: CorrelationPlotConfig,
    filename: str,
    customizer: PlotCustomizer,
):
    # Filter out NaN pairs
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        print(f"[correlations] WARNING: not enough valid data for {filename}")
        return

    x_f = x[mask]
    y_f = y[mask]

    r, p = pearsonr(x_f, y_f)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.scatter(x_f, y_f, color=cfg.dot_color, alpha=cfg.alpha)

    # --- apply shared axis limits if provided
    if cfg.xlim is not None:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim is not None:
        ax.set_ylim(cfg.ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)
    ax.grid(False)

    _add_smart_stats_box(ax, f"r = {r:.3f}\np = {p:.3g}", x_f, y_f)

    customizer.adjust_padding_proportionally()
    fig.tight_layout()
    out_path = _correlation_out_path(cfg.out_dir, filename, cfg.image_format)
    writeImage(str(out_path), format=cfg.image_format)
    plt.close(fig)


def _ensure_rewards_per_distance(va) -> bool:
    """
    Make sure va.rwdsPerDist exists.
    """
    if getattr(va, "rwdsPerDist", None) is None:
        if hasattr(va, "_rewards_per_distance"):
            va._rewards_per_distance(silent=True)
        else:
            print(
                "[correlations] WARNING: no rwdsPerDist and no _rewards_per_distance()"
            )
            return False
    return True


def _ensure_rewards_per_minute_by_sync_bucket(va) -> bool:
    """
    Make sure va.rwdsPerMinBySyncBucket exists.
    """
    if getattr(va, "rwdsPerMinBySyncBucket", None) is None:
        if hasattr(va, "_rewards_per_minute_by_sync_bucket"):
            va._rewards_per_minute_by_sync_bucket(silent=True)
        else:
            print(
                "[correlations] WARNING: no rwdsPerMinBySyncBucket and no "
                "_rewards_per_minute_by_sync_bucket()"
            )
            return False
    return True


def _ensure_reward_pi_pre(va) -> bool:
    """
    Make sure va.rewardPIPre exists (pre-training reward PI).
    """
    if getattr(va, "rewardPIPre", None) is None:
        if hasattr(va, "calcRewardsPre"):
            va.calcRewardsPre()
        else:
            print("[correlations] WARNING: no rewardPIPre and no calcRewardsPre()")
            return False
    return True


def _rewards_per_minute_for_first_n_calc_rewards(
    va,
    *,
    training_idx: int,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
    first_n_rewards: int,
    max_time_to_nth_s: float | None = None,
    time_basis: str = "window_start",
) -> float:
    n_target = max(1, int(first_n_rewards or 1))
    windows = selected_windows_for_va(
        va,
        [int(training_idx)],
        skip_first_sync_buckets=int(skip_first_sync_buckets or 0),
        keep_first_sync_buckets=int(keep_first_sync_buckets or 0),
        f=0,
    )
    if not windows:
        return np.nan

    fps = float(getattr(va, "fps", 1.0) or 1.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 1.0

    calc_rewards = frames_in_windows(va, windows, calc=True, ctrl=False, f=0)
    if calc_rewards.size < n_target:
        return np.nan

    cutoff_frame = int(calc_rewards[n_target - 1])
    elapsed_s = cumulative_window_seconds_for_frame(windows, cutoff_frame, fps=fps)
    if not np.isfinite(elapsed_s) or elapsed_s <= 0:
        return np.nan
    if max_time_to_nth_s is not None:
        try:
            max_time_to_nth_s = float(max_time_to_nth_s)
        except Exception:
            max_time_to_nth_s = None
        if (
            max_time_to_nth_s is not None
            and np.isfinite(max_time_to_nth_s)
            and elapsed_s > max_time_to_nth_s
        ):
            return np.nan
    if str(time_basis) == "first_to_nth":
        if n_target < 2:
            return np.nan
        first_s = cumulative_window_seconds_for_frame(
            windows, int(calc_rewards[0]), fps=fps
        )
        if not np.isfinite(first_s):
            return np.nan
        span_s = elapsed_s - first_s
        if not np.isfinite(span_s) or span_s <= 0:
            return np.nan
        return float((n_target - 1) * 60.0 / span_s)

    return float(n_target * 60.0 / elapsed_s)


def _reduce_sync_bucket_series(
    vec,
    *,
    bucket_idx: int | None = None,
    average_over_buckets: bool = False,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
    reduce: str = "mean",
) -> float:
    arr = np.asarray(vec, float)
    if arr.size == 0:
        return np.nan

    k = max(0, int(skip_first_sync_buckets or 0))

    if bucket_idx is not None:
        b = int(bucket_idx)
        if 0 <= b < arr.size and np.isfinite(arr[b]):
            return float(arr[b])
        return np.nan

    end = arr.size
    keep = int(keep_first_sync_buckets or 0)
    if keep > 0:
        end = min(arr.size, k + keep)

    sub = arr[k:end] if k < arr.size else np.array([], float)
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return np.nan

    if average_over_buckets:
        if reduce == "median":
            return float(np.median(sub))
        return float(np.mean(sub))

    return _last_valid_scalar(sub)


def _last_valid_scalar(row) -> float:
    arr = np.asarray(row, float)
    for v in arr[::-1]:
        if np.isfinite(v):
            return float(v)
    return np.nan


def _fast_slow_indices_from_sli_T1_first(
    sli_T1_first: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute disjoint fast and slow index sets based on SLI in the first
    sync bucket of T1 (reward PI, exp − yoked).

    - fast = top `frac` of finite values
    - slow = bottom `frac` of finite values

    If 2*frac == 1, the finite flies are partitioned exhaustively; any
    rounding remainder is assigned to the slow group.
    """
    arr = np.asarray(sli_T1_first, float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=int)

    finite_vals = arr[mask]
    finite_idx = np.arange(arr.shape[0])[mask]
    n_finite = finite_vals.size

    k_slow = max(1, int(frac * n_finite))
    k_fast = max(1, int(frac * n_finite))

    if np.isclose(2.0 * float(frac), 1.0, atol=1e-12):
        assigned = k_slow + k_fast
        if assigned < n_finite:
            k_slow += n_finite - assigned

    if k_slow + k_fast > n_finite:
        k_slow = min(k_slow, max(0, n_finite - k_fast))

    if k_slow == 0 or k_fast == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(finite_vals)  # ascending
    slow_idx = finite_idx[order[:k_slow]]
    fast_idx = finite_idx[order[-k_fast:]]

    return fast_idx, slow_idx


def _ensure_sync_med_dist(va):
    if not hasattr(va, "syncMedDist") or va.syncMedDist is None:
        if hasattr(va, "bySyncBucketMedDist"):
            va.bySyncBucketMedDist()
        else:
            print("[correlations] WARNING: no syncMedDist and no bySyncBucketMedDist()")


def summarize_fast_vs_strong(
    sli_T1_first: np.ndarray,
    sli_strong: np.ndarray,
    vas,
    opts,
    frac: float = 0.2,
    *,
    strong_label: str = "Strong learners",
):
    """
    Summarize proportions of fast vs strong learners.
    - fast = top percentile of SLI at first sync bucket of T1
    - strong = top percentile of SLI according to `sli_strong`
      (definition controlled upstream; label passed in for logging/printing)

    This version uses *separate* validity masks, so a fly with NaN in one
    bucket can still be classified in the other.
    """
    sli_T1_first = np.asarray(sli_T1_first, float)
    sli_strong = np.asarray(sli_strong, float)

    N_total = len(vas)
    if N_total == 0:
        return

    # Global percentile count (same rule as select_extremes)
    k_global = max(1, int(frac * N_total))

    # FAST LEARNERS (T1 first bucket)
    mask1 = np.isfinite(sli_T1_first)
    sli1 = sli_T1_first[mask1]

    # --- STRONG LEARNERS (definition controlled upstream) ---
    mask2 = np.isfinite(sli_strong)
    sli2 = sli_strong[mask2]

    if len(sli1) == 0 or len(sli2) == 0:
        print("[correlations] WARNING: no finite SLI values for fast/strong summary")
        return

    # clamp k to finite size
    k1 = min(k_global, len(sli1))
    k2 = min(k_global, len(sli2))

    # argpartition selection on the finite values
    idx1 = np.argpartition(sli1, -k1)[-k1:]
    orig_idx1 = np.arange(N_total)[mask1]
    fast_global = set(orig_idx1[idx1])
    idx2 = np.argpartition(sli2, -k2)[-k2:]
    orig_idx2 = np.arange(N_total)[mask2]
    strong_global = set(orig_idx2[idx2])

    # Overlap
    overlap = fast_global & strong_global

    print("\n=== Fast vs Strong learner summary ===")
    print(f"Fast learners:   {len(fast_global)} (k={k1}, from N={N_total})")
    print(f"{strong_label}: {len(strong_global)} (k={k2}, from N={N_total})")
    print(f"Overlap:         {len(overlap)}")

    summary = {
        "fast": np.array(sorted(fast_global)),
        "strong": np.array(sorted(strong_global)),
        "overlap": np.array(sorted(overlap)),
    }

    if getattr(opts, "log_fly_grps", False):
        log_fly_group("FAST_LEARNERS", summary["fast"], vas)
        log_fly_group("STRONG_LEARNERS", summary["strong"], vas)
        log_fly_group("FAST_STRONG_OVERLAP", summary["overlap"], vas)

    return summary


def plot_fast_vs_strong_scatter(
    sli_T1_first: np.ndarray,
    sli_strong: np.ndarray,
    vas,
    fast_idx: np.ndarray,
    strong_idx: np.ndarray,
    out_dir: Path,
    frac: float,
    customizer: PlotCustomizer,
    *,
    strong_y_label: str,
    strong_title_suffix: str,
    x_label: str,
    image_format: str = "png",
):
    """
    Scatter plot of:
        X = SLI at T1 first sync bucket (fast learners)
        Y = SLI along timeframe used for strong learners (defined upstream)

    Points are colored by group:
        - Fast-only (fast & not strong)
        - Strong-only (strong & not fast)
        - Overlap (fast & strong)
        - Unclassified (neither)

    Also computes (descriptive) Pearson correlations for:
        - Fast group, including overlap points
        - Strong group, including overlap points
    """
    x = np.asarray(sli_T1_first, float)
    y = np.asarray(sli_strong, float)

    # Masks
    mask_x = np.isfinite(x)
    mask_y = np.isfinite(y)
    mask = mask_x & mask_y  # only for plotting (not classification)

    x_f = x[mask]
    y_f = y[mask]

    # Build global index arrays
    valid_global_idx = np.arange(len(vas))[mask]

    fast_set = set(fast_idx.tolist())
    strong_set = set(strong_idx.tolist())
    overlap_set = fast_set & strong_set

    # Classification per plotted point
    classes = []
    for idx in valid_global_idx:
        if idx in overlap_set:
            classes.append("overlap")
        elif idx in fast_set:
            classes.append("fast")
        elif idx in strong_set:
            classes.append("strong")
        else:
            classes.append("other")

    classes_arr = np.asarray(classes, dtype=object)

    def _corr_from_class_mask(m: np.ndarray) -> tuple[float, float, int] | None:
        """
        Compute Pearson (r, p) on the *plotted* points selected by mask `m`.
        Returns (r, p, n) or None if fewer than 3 points.
        """
        m = np.asarray(m, dtype=bool)
        n = int(np.sum(m))
        if n < 3:
            return None
        r, p = pearsonr(x_f[m], y_f[m])
        return float(r), float(p), n

    # Correlations: include overlap in both fast and strong groups
    # NOTE: correlations are plotted on plotted points (finite x/y) only
    corr_fast_incl_overlap = _corr_from_class_mask(
        (classes_arr == "fast") | (classes_arr == "overlap")
    )
    corr_strong_incl_overlap = _corr_from_class_mask(
        (classes_arr == "strong") | (classes_arr == "overlap")
    )

    # Correlation across *all* plotted points (finite x/y only)
    corr_all = None
    n_all = int(x_f.size)
    if n_all >= 3:
        r_a, p_a = pearsonr(x_f, y_f)
        corr_all = (float(r_a), float(p_a), n_all)

    # Colors (simple, can be refined)
    color_map = {
        "overlap": "#cc0000",  # red
        "fast": "#1f77b4",  # blue
        "strong": "#2ca02c",  # green
        "other": "#aaaaaa",  # gray
    }

    point_colors = [color_map[c] for c in classes]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scatter_artist = ax.scatter(x_f, y_f, c=point_colors, alpha=0.85)

    ax.set_xlabel(x_label)
    ax.set_ylabel(strong_y_label)
    ax.set_title(
        f"Fast vs Strong Learners ({strong_title_suffix}, top {frac*100:.0f}% each)",
        pad=10,
    )

    # Display descriptive correlations (fast/strong each including overlap)
    lines = []
    if corr_all is not None:
        r_a, p_a, n_a = corr_all
        lines.append(f"All (finite):           r = {r_a:.3f}, p = {p_a:.3g} (n={n_a})")
    else:
        lines.append("All (finite):           r = n/a")
    if corr_fast_incl_overlap is not None:
        r_f, p_f, n_f = corr_fast_incl_overlap
        lines.append(f"Fast (incl overlap):  r = {r_f:.3f}, p = {p_f:.3g} (n={n_f})")
    else:
        lines.append("Fast (incl overlap):  r = n/a")

    if corr_strong_incl_overlap is not None:
        r_s, p_s, n_s = corr_strong_incl_overlap
        lines.append(f"Strong (incl overlap): r = {r_s:.3f}, p = {p_s:.3g} (n={n_s})")
    else:
        lines.append("Strong (incl overlap): r = n/a")

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["fast"],
            markersize=8,
            label="Fast only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["strong"],
            markersize=8,
            label="Strong only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["overlap"],
            markersize=8,
            label="Overlap",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["other"],
            markersize=8,
            label="Other",
        ),
    ]
    _place_legend_without_point_overlap(
        ax, handles, x_f, y_f, scatter_artist=scatter_artist, frameon=True
    )
    _add_smart_stats_box(ax, "\n".join(lines), x_f, y_f)

    # Optional proportional padding
    customizer.adjust_padding_proportionally()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = _correlation_out_path(out_dir, "scatter_fast_vs_strong", image_format)
    writeImage(str(out_path), format=image_format)
    plt.close(fig)


def plot_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow(
    pre_pi_diff_vals: np.ndarray,
    reward_pi_first_bucket: np.ndarray,
    fast_idx: np.ndarray,
    slow_idx: np.ndarray,
    out_dir: Path,
    frac: float,
    customizer: PlotCustomizer,
    early_label: str,
    image_format: str = "png",
):
    """
    Correlation plot:

        X = pre-training reward PI (exp − yoked)
        Y = reward PI at T1, first sync bucket (exp − yoked)

    All flies are shown, color-coded by membership:

        - Fast learners (top `frac` of SLI in T1 first bucket)
        - Slow learners (bottom `frac` of SLI in T1 first bucket)
        - Other (middle SLI values)

    Correlations are computed separately for:
        - Fast group
        - Slow group
    """
    x = np.asarray(pre_pi_diff_vals, float)
    y = np.asarray(reward_pi_first_bucket, float)

    # Global finite mask for plotting
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        print(
            "[correlations] WARNING: not enough valid data for "
            "pre-PI vs early-PI fast/slow plot"
        )
        return

    x_f = x[mask]
    y_f = y[mask]
    valid_global_idx = np.arange(x.shape[0])[mask]

    fast_set = set(np.asarray(fast_idx, dtype=int).tolist())
    slow_set = set(np.asarray(slow_idx, dtype=int).tolist())

    color_map = {
        "fast": "#1f77b4",  # blue
        "slow": "#cc0000",  # red
        "other": "#aaaaaa",  # gray
    }

    classes = []
    point_colors = []

    for idx in valid_global_idx:
        if idx in fast_set:
            cls = "fast"
        elif idx in slow_set:
            cls = "slow"
        else:
            cls = "other"
        classes.append(cls)
        point_colors.append(color_map[cls])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scatter_artist = ax.scatter(x_f, y_f, c=point_colors, alpha=0.85)

    ax.set_xlabel("\nBPI\n(exp - yok, pre-training)")
    ax.set_ylabel(early_label.replace("SLI", "SLI\n"))
    ax.set_title(
        f"Pre-training vs early reward preference\n"
        f"(fast vs slow learners, top/bottom {frac * 100:.0f}% SLI)"
    )

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["fast"],
            markersize=8,
            label="Fast",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["slow"],
            markersize=8,
            label="Slow",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["other"],
            markersize=8,
            label="Other",
        ),
    ]
    _place_legend_without_point_overlap(
        ax, handles, x_f, y_f, scatter_artist=scatter_artist, frameon=True
    )

    # Correlations for each group
    corr_fast = _compute_group_corr(x, y, fast_idx)
    corr_slow = _compute_group_corr(x, y, slow_idx)

    lines = []
    if corr_fast is not None:
        r_f, p_f = corr_fast
        lines.append(f"Fast:  r = {r_f:.3f}, p = {p_f:.3g}")
    else:
        lines.append("Fast:  r = n/a")

    if corr_slow is not None:
        r_s, p_s = corr_slow
        lines.append(f"Slow:  r = {r_s:.3f}, p = {p_s:.3g}")
    else:
        lines.append("Slow:  r = n/a")

    _add_smart_stats_box(ax, "\n".join(lines), x_f, y_f)

    customizer.adjust_padding_proportionally()

    out_path = _correlation_out_path(
        out_dir,
        "corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow",
        image_format,
    )
    writeImage(str(out_path), format=image_format)
    plt.close(fig)


def plot_cross_fly_correlations(
    sli_values: Sequence[float],
    vas: Sequence,
    training_idx: int,
    opts,
    reward_pi_first_bucket: Sequence[float] | None = None,
    out_dir: str | Path = "imgs/correlations",
    plot_customizer: PlotCustomizer | None = None,
    *,
    sli_ctx: SLIContext | None = None,
    reward_rate_ctx: SLIContext | None = None,
    sli_selected: tuple[Sequence[int], Sequence[int]] | None = None,
    sli_extremes: str | None = None,
):
    """
    Cross-fly correlations:

      1) SLI_final vs reward-per-distance (final bucket of chosen training)
      2) SLI_final vs median distance to reward during chosen training
      3) Pre-training reward PI (exp − yoked) vs SLI_final
      3b) Pre-training floor exploration vs SLI at T1, first sync bucket
      3c) Pre-training floor exploration vs SLI_final
      4) Reward PI (T1, first sync bucket, exp − yoked) vs total rewards
         in that same bucket (experimental fly)
      5) Pre-training reward PI (exp − yoked) vs T1 first-bucket reward PI:
           a) all learners
           b) fast learners only
           c) fast vs slow learners (top and bottom percentile of early SLI)
      6) SLI at T1 first sync bucket vs SLI at T2 final sync bucket,
         color-coded by fast / strong / overlap / other.

    `sli_values` should be a 1D sequence aligned with `vas`
    (one SLI per VideoAnalysis / learner).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CorrelationPlotConfig(
        out_dir=out_dir,
        image_format=getattr(opts, "imageFormat", "png"),
        xlim=getattr(opts, "corr_xlim", None),
        ylim=getattr(opts, "corr_ylim", None),
    )
    frac = getattr(opts, "best_worst_fraction", 0.2)
    customizer = plot_customizer or PlotCustomizer()

    selected_bottom_idx, selected_top_idx, selected_mode = _normalize_selected_groups(
        sli_selected=sli_selected,
        sli_extremes=sli_extremes,
    )

    top_frac = getattr(opts, "top_sli_fraction", None)
    bottom_frac = getattr(opts, "bottom_sli_fraction", None)

    top_pct_txt = (
        f"{top_frac * 100:.0f}%" if top_frac is not None else f"{frac * 100:.0f}%"
    )
    bottom_pct_txt = (
        f"{bottom_frac * 100:.0f}%" if bottom_frac is not None else f"{frac * 100:.0f}%"
    )

    top_sel_label = f"Top SLI-selected ({top_pct_txt})"
    bottom_sel_label = f"Bottom SLI-selected ({bottom_pct_txt})"

    sli_vals = np.asarray(sli_values, float)
    if sli_vals.shape[0] != len(vas):
        print(
            "[correlations] WARNING: len(sli_values) != len(vas) "
            f"({sli_vals.shape[0]} vs {len(vas)})"
        )

    reward_pi_training_vals = None
    if reward_pi_first_bucket is not None:
        reward_pi_training_vals = np.asarray(reward_pi_first_bucket, float)
        if reward_pi_training_vals.shape[0] != len(vas):
            print(
                "[correlations] WARNING: len(reward_pi_first_bucket) != len(vas) "
                f"({reward_pi_training_vals.shape[0]} vs {len(vas)})"
            )

    if sli_ctx is None:
        sli_ctx = SLIContext(training_idx=training_idx, average_over_buckets=False)
    if reward_rate_ctx is None:
        reward_rate_ctx = sli_ctx

    x_label_sli = sli_ctx.label_short(abbrev_sb=False)
    y_label_sli = sli_ctx.label_short(abbrev_sb=True)

    skip_k = int(getattr(sli_ctx, "skip_first_sync_buckets", 0) or 0)
    skip_k = max(0, skip_k)
    keep_k = int(getattr(sli_ctx, "keep_first_sync_buckets", 0) or 0)
    keep_k = max(0, keep_k)
    sli_bucket_idx = getattr(sli_ctx, "explicit_bucket_idx", None)
    reward_training_idx = int(getattr(reward_rate_ctx, "training_idx", training_idx) or 0)
    reward_avg = bool(getattr(reward_rate_ctx, "average_over_buckets", False))
    reward_skip_k = int(getattr(reward_rate_ctx, "skip_first_sync_buckets", 0) or 0)
    reward_skip_k = max(0, reward_skip_k)
    reward_keep_k = int(getattr(reward_rate_ctx, "keep_first_sync_buckets", 0) or 0)
    reward_keep_k = max(0, reward_keep_k)
    reward_bucket_idx = getattr(reward_rate_ctx, "explicit_bucket_idx", None)
    reward_first_n = int(getattr(opts, "corr_reward_rate_first_n_rewards", 0) or 0)
    reward_first_n = max(0, reward_first_n)
    reward_max_time_to_nth_s = getattr(opts, "corr_reward_rate_max_time_to_nth_s", None)
    reward_first_n_time_basis = str(
        getattr(opts, "corr_reward_rate_first_n_time_basis", "window_start")
        or "window_start"
    )
    early_lbl = early_sli_label(training_idx=0, skip_first_sync_buckets=skip_k)  # T1
    early_sb_txt = f"SB{skip_k + 1}"
    if early_sb_txt == "SB1":
        early_sb_txt = "first sync bucket"
    t1_sb1_lbl = early_sli_label(
        training_idx=0, skip_first_sync_buckets=0
    )  # always SB1

    rpd_vals = []
    rpt_vals = []
    med_train_vals = []
    pre_pi_diff_vals = []
    total_reward_vals = []
    pre_coverage_vals = []

    for va in vas:
        # --- Reward per distance (final bucket of training_idx) ---
        if _ensure_rewards_per_distance(va):
            row_idx = 2 * training_idx  # exp row
            if 0 <= row_idx < len(va.rwdsPerDist):
                exp_row = va.rwdsPerDist[row_idx]
                rpd_val = _reduce_sync_bucket_series(
                    exp_row,
                    bucket_idx=sli_bucket_idx,
                    average_over_buckets=bool(sli_ctx.average_over_buckets),
                    skip_first_sync_buckets=skip_k,
                    keep_first_sync_buckets=keep_k,
                )
            else:
                rpd_val = np.nan
        else:
            rpd_val = np.nan

        # --- Reward per time (may use a different training/window from SLI) ---
        if reward_first_n > 0:
            rpt_val = _rewards_per_minute_for_first_n_calc_rewards(
                va,
                training_idx=reward_training_idx,
                skip_first_sync_buckets=reward_skip_k,
                keep_first_sync_buckets=reward_keep_k,
                first_n_rewards=reward_first_n,
                max_time_to_nth_s=reward_max_time_to_nth_s,
                time_basis=reward_first_n_time_basis,
            )
        elif _ensure_rewards_per_minute_by_sync_bucket(va):
            row_idx = 2 * reward_training_idx  # exp row
            if 0 <= row_idx < len(va.rwdsPerMinBySyncBucket):
                exp_row = va.rwdsPerMinBySyncBucket[row_idx]
                rpt_val = _reduce_sync_bucket_series(
                    exp_row,
                    bucket_idx=reward_bucket_idx,
                    average_over_buckets=reward_avg,
                    skip_first_sync_buckets=reward_skip_k,
                    keep_first_sync_buckets=reward_keep_k,
                )
            else:
                rpt_val = np.nan
        else:
            rpt_val = np.nan

        # --- Median distance to reward during training ---
        _ensure_sync_med_dist(va)
        if hasattr(va, "syncMedDist") and training_idx < len(va.syncMedDist):
            med_vec = np.asarray(va.syncMedDist[training_idx].get("exp", []), float)
            end_k = med_vec.size if keep_k <= 0 else min(med_vec.size, skip_k + keep_k)
            if med_vec.size and skip_k < end_k:
                med_train = np.nanmedian(med_vec[skip_k:end_k])
            else:
                med_train = np.nan
        else:
            med_train = np.nan

        # --- Pre-training reward preference index (exp − yoked) ---
        if _ensure_reward_pi_pre(va):
            pre_arr = np.asarray(getattr(va, "rewardPIPre", []), float)
            if pre_arr.size == 0:
                pre_diff = np.nan
            elif pre_arr.size == 1:
                # No yoked partner; just use the single value
                pre_diff = float(pre_arr[0])
            else:
                # Assume index 0 = experimental, 1 = yoked
                pre_diff = float(pre_arr[0] - pre_arr[1])
        else:
            pre_diff = np.nan

        # --- Pre-training floor exploration (experimental fly only) ---
        coverage = np.nan
        try:
            if not hasattr(va, "preFloorExploredFrac"):
                if hasattr(va, "calcPreFloorExploration"):
                    va.calcPreFloorExploration()
            if hasattr(va, "preFloorExploredFrac") and len(va.preFloorExploredFrac) > 0:
                coverage = float(va.preFloorExploredFrac[0])
        except Exception:
            coverage = np.nan

        # --- Total rewards in the same sync bucket used for the reward-PI X variable ---
        try:
            calc_idx = 1
            training_idx_T1 = 0
            bucket_idx = skip_k  # first included bucket of T1 (aligned with X variable)

            tot = getattr(va, "numRewardsTot", None)

            if (
                isinstance(tot, (list, tuple))
                and len(tot) >= calc_idx
                and isinstance(tot[calc_idx], (list, tuple))
            ):
                flat_list = tot[calc_idx][
                    0
                ]  # 0 = reward circle; entries: (exp T1, yok T1, exp T2, yok T2, ...)

                # compute flat index into alternating exp/yok structure
                flat_idx_exp = 2 * training_idx_T1

                if flat_idx_exp < len(flat_list):
                    bucket_vals = flat_list[flat_idx_exp]
                    if isinstance(
                        bucket_vals, (list, tuple, np.ndarray)
                    ) and bucket_idx < len(bucket_vals):
                        total_rewards = float(bucket_vals[bucket_idx])
                    else:
                        total_rewards = np.nan
                else:
                    total_rewards = np.nan
            else:
                total_rewards = np.nan

        except Exception:
            total_rewards = np.nan

        rpd_vals.append(rpd_val)
        rpt_vals.append(rpt_val)
        med_train_vals.append(med_train)
        pre_pi_diff_vals.append(pre_diff)
        total_reward_vals.append(total_rewards)
        pre_coverage_vals.append(coverage)

    rpd_vals = np.asarray(rpd_vals, float)
    rpt_vals = np.asarray(rpt_vals, float)
    med_train_vals = np.asarray(med_train_vals, float)
    pre_pi_diff_vals = np.asarray(pre_pi_diff_vals, float)
    total_reward_vals = np.asarray(total_reward_vals, float)
    pre_coverage_vals = np.asarray(pre_coverage_vals, float)

    # --- Fast/strong learner summary (for plots 5b/5c and fast/strong scatter) ---
    summary = None
    if reward_pi_training_vals is not None:
        try:
            strong_label = (
                f"Strong learners (top {frac*100:.1f}%, {sli_ctx.label_short()})"
            )
            summary = summarize_fast_vs_strong(
                sli_T1_first=reward_pi_training_vals,
                sli_strong=sli_vals,
                vas=vas,
                opts=opts,
                frac=frac,
                strong_label=strong_label,
            )
        except Exception as e:
            print(f"[correlations] WARNING: failed fast/strong summary: {e}")

    rpd_suffix = _window_context_suffix(sli_ctx, prefix="sli")
    rpt_suffix = (
        f"{_window_context_suffix(sli_ctx, prefix='sli')}__"
        f"{_window_context_suffix(reward_rate_ctx, prefix='rpt')}"
    )
    if reward_first_n > 0:
        rpt_suffix = f"{rpt_suffix}__first{reward_first_n}calc"
        if reward_first_n_time_basis == "first_to_nth":
            rpt_suffix = f"{rpt_suffix}__first_to_nth"
        if reward_max_time_to_nth_s is not None:
            try:
                cutoff_suffix = float(reward_max_time_to_nth_s)
            except Exception:
                cutoff_suffix = None
            if cutoff_suffix is not None and np.isfinite(cutoff_suffix):
                rpt_suffix = f"{rpt_suffix}__maxtime{cutoff_suffix:g}s"

    rpd_y_label = "rewards per distance $[m^{{-1}}]$"
    if reward_first_n > 0:
        rpt_y_label = _first_n_reward_rate_label(
            first_n_rewards=reward_first_n,
            ctx=reward_rate_ctx,
            max_time_to_nth_s=reward_max_time_to_nth_s,
            time_basis=reward_first_n_time_basis,
        )
    else:
        rpt_y_label = _windowed_metric_label("rewards per minute", reward_rate_ctx)

    if sli_ctx.average_over_buckets:
        window_txt = sli_ctx._window_text(abbrev_sb=True)
        rpd_y_label = f"rewards per distance $[m^{{-1}}]$\n(mean {window_txt})"
    else:
        if sli_bucket_idx is not None:
            window_txt = sli_ctx._window_text(abbrev_sb=True)
            rpd_y_label = f"rewards per distance $[m^{{-1}}]$\n(T{sli_ctx.training_idx + 1}, {window_txt})"
        elif skip_k or keep_k:
            window_txt = sli_ctx._window_text(abbrev_sb=True)
            rpd_y_label = f"rewards per distance $[m^{{-1}}]$\n(last valid, {window_txt})"

    # --- Plot 1: SLI_final vs reward-per-distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpd_vals,
        title="Rewards per distance vs SLI",
        x_label=x_label_sli,
        y_label=rpd_y_label,
        cfg=cfg,
        filename=f"corr_rpd_vs_sli_{rpd_suffix}",
        customizer=customizer,
    )
    if selected_mode is not None:
        if selected_mode == "top":
            title_1_sel = "Rewards per distance vs SLI (top SLI-selected learners)"
            filename_1_sel = f"corr_rpd_vs_sli_{rpd_suffix}_top_selected"
        elif selected_mode == "bottom":
            title_1_sel = "Rewards per distance vs SLI (bottom SLI-selected learners)"
            filename_1_sel = f"corr_rpd_vs_sli_{rpd_suffix}_bottom_selected"
        else:
            title_1_sel = (
                "Rewards per distance vs SLI (top vs bottom SLI-selected learners)"
            )
            filename_1_sel = f"corr_rpd_vs_sli_{rpd_suffix}_selected_extremes"

        plot_selected_group_scatter(
            x=sli_vals,
            y=rpd_vals,
            bottom_idx=selected_bottom_idx,
            top_idx=selected_top_idx,
            mode=selected_mode,
            title=title_1_sel,
            x_label=x_label_sli,
            y_label=rpd_y_label,
            filename=filename_1_sel,
            out_dir=out_dir,
            customizer=customizer,
            top_label=top_sel_label,
            bottom_label=bottom_sel_label,
            xlim=cfg.xlim,
            ylim=cfg.ylim,
            image_format=cfg.image_format,
        )

    # --- Plot 1b: SLI_final vs reward-per-time (SLI on Y axis) ---
    _scatter_with_corr(
        x=rpt_vals,
        y=sli_vals,
        title="SLI vs rewards per minute",
        x_label=rpt_y_label,
        y_label=y_label_sli,
        cfg=cfg,
        filename=f"corr_sli_vs_rpt_{rpt_suffix}",
        customizer=customizer,
    )
    if selected_mode is not None:
        if selected_mode == "top":
            title_1b_sel = "SLI vs rewards per minute (top SLI-selected learners)"
            filename_1b_sel = f"corr_sli_vs_rpt_{rpt_suffix}_top_selected"
        elif selected_mode == "bottom":
            title_1b_sel = "SLI vs rewards per minute (bottom SLI-selected learners)"
            filename_1b_sel = f"corr_sli_vs_rpt_{rpt_suffix}_bottom_selected"
        else:
            title_1b_sel = "SLI vs rewards per minute (top vs bottom SLI-selected learners)"
            filename_1b_sel = f"corr_sli_vs_rpt_{rpt_suffix}_selected_extremes"

        plot_selected_group_scatter(
            x=rpt_vals,
            y=sli_vals,
            bottom_idx=selected_bottom_idx,
            top_idx=selected_top_idx,
            mode=selected_mode,
            title=title_1b_sel,
            x_label=rpt_y_label,
            y_label=y_label_sli,
            filename=filename_1b_sel,
            out_dir=out_dir,
            customizer=customizer,
            top_label=top_sel_label,
            bottom_label=bottom_sel_label,
            figsize=(6.8, 5.6),
            xlim=cfg.xlim,
            ylim=cfg.ylim,
            include_all_corr=True,
            image_format=cfg.image_format,
        )

    # --- Plot 2: SLI_final vs median training distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=med_train_vals,
        title="SLI vs median distance to reward",
        x_label=x_label_sli,
        y_label="Median distance during training (mm)",
        cfg=cfg,
        filename="corr_sli_vs_median_training",
        customizer=customizer,
    )

    # --- Plot 3: Pre-training reward PI (exp − yoked) vs SLI_final ---
    _scatter_with_corr(
        x=pre_pi_diff_vals,
        y=sli_vals,
        title="Baseline PI vs SLI",
        x_label="Baseline PI\n(exp − yok, pre-training)",
        y_label=y_label_sli,
        cfg=cfg,
        filename="corr_pre_reward_pi_vs_sli",
        customizer=customizer,
    )

    # --- Plot 3b: Pre-training exploration vs SLI at T1, first sync bucket ---
    if reward_pi_training_vals is not None:
        _scatter_with_corr(
            x=pre_coverage_vals,
            y=reward_pi_training_vals,
            title="Pre-training exploration vs early SLI",
            x_label="Fraction of floor explored during pre-training\n(exp fly)",
            y_label=early_lbl,
            cfg=cfg,
            filename="corr_pre_floor_exploration_vs_sli_T1_first",
            customizer=customizer,
        )
        if selected_mode is not None:
            if selected_mode == "top":
                title_3b_sel = (
                    "Pre-training exploration vs early SLI (top SLI-selected learners)"
                )
                filename_3b_sel = (
                    "corr_pre_floor_exploration_vs_sli_T1_first_top_selected"
                )
            elif selected_mode == "bottom":
                title_3b_sel = "Pre-training exploration vs early SLI (bottom SLI-selected learners)"
                filename_3b_sel = (
                    "corr_pre_floor_exploration_vs_sli_T1_first_bottom_selected"
                )
            else:
                title_3b_sel = "Pre-training exploration vs early SLI (top vs bottom SLI-selected learners)"
                filename_3b_sel = (
                    "corr_pre_floor_exploration_vs_sli_T1_first_selected_extremes"
                )

            plot_selected_group_scatter(
                x=pre_coverage_vals,
                y=reward_pi_training_vals,
                bottom_idx=selected_bottom_idx,
                top_idx=selected_top_idx,
                mode=selected_mode,
                title=title_3b_sel,
                x_label="Fraction of floor explored during pre-training\n(exp fly)",
                y_label=early_lbl,
                filename=filename_3b_sel,
                out_dir=out_dir,
                customizer=customizer,
                top_label=top_sel_label,
                bottom_label=bottom_sel_label,
                xlim=cfg.xlim,
                ylim=cfg.ylim,
                image_format=cfg.image_format,
            )
    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; "
            "skipping pre-training exploration vs early SLI plot"
        )

    # --- Plot 3c: Pre-training exploration vs SLI_final (training {trn_label_idx}) ---
    _scatter_with_corr(
        x=pre_coverage_vals,
        y=sli_vals,
        title="Pre-training exploration vs SLI",
        x_label="Fraction of floor explored during pre-training\n(exp fly)",
        y_label=y_label_sli,
        cfg=cfg,
        filename="corr_pre_floor_exploration_vs_sli_final",
        customizer=customizer,
    )

    if selected_mode is not None:
        if selected_mode == "top":
            title_3c_sel = "Pre-training exploration vs SLI (top SLI-selected learners)"
            filename_3c_sel = "corr_pre_floor_exploration_vs_sli_final_top_selected"
        elif selected_mode == "bottom":
            title_3c_sel = (
                "Pre-training exploration vs SLI (bottom SLI-selected learners)"
            )
            filename_3c_sel = "corr_pre_floor_exploration_vs_sli_final_bottom_selected"
        else:
            title_3c_sel = (
                "Pre-training exploration vs SLI (top vs bottom SLI-selected learners)"
            )
            filename_3c_sel = (
                "corr_pre_floor_exploration_vs_sli_final_selected_extremes"
            )

        plot_selected_group_scatter(
            x=pre_coverage_vals,
            y=sli_vals,
            bottom_idx=selected_bottom_idx,
            top_idx=selected_top_idx,
            mode=selected_mode,
            title=title_3c_sel,
            x_label="Fraction of floor explored during pre-training\n(exp fly)",
            y_label=y_label_sli,
            filename=filename_3c_sel,
            out_dir=out_dir,
            customizer=customizer,
            top_label=top_sel_label,
            bottom_label=bottom_sel_label,
            xlim=cfg.xlim,
            ylim=cfg.ylim,
            image_format=cfg.image_format,
        )

    if reward_pi_training_vals is not None:
        # --- Plot 4: Reward PI (T1, first bucket) vs total rewards in that bucket ---
        _scatter_with_corr(
            x=reward_pi_training_vals,
            y=total_reward_vals,
            title="Early SLI vs total rewards",
            x_label=early_lbl,
            y_label=f"Total rewards\n(exp, T1, {early_sb_txt})",
            cfg=cfg,
            filename="corr_reward_pi_first_bucket_vs_total_rewards",
            customizer=customizer,
        )

        # --- Plot 5a: Pre-training PI vs T1 first-bucket PI (all learners) ---
        _scatter_with_corr(
            x=pre_pi_diff_vals,
            y=reward_pi_training_vals,
            title="Baseline PI vs early SLI",
            x_label="Baseline PI\n(exp - yok, pre-training)",
            y_label=early_lbl.replace("SLI", "SLI\n"),
            cfg=cfg,
            filename="corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_all",
            customizer=customizer,
        )

        # --- Plot 5b: Pre-training PI vs T1 first-bucket PI (fast learners only) ---
        if summary is not None and "fast" in summary:
            fast_idx = np.asarray(summary["fast"], dtype=int)
            if fast_idx.size == 0:
                print(
                    "[correlations] WARNING: no fast learners; "
                    "skipping fast-only pre-vs-early PI correlation"
                )
            else:
                _scatter_with_corr(
                    x=pre_pi_diff_vals[fast_idx],
                    y=reward_pi_training_vals[fast_idx],
                    title="Baseline PI vs early SLI (fast learners)",
                    x_label="Baseline PI\n(exp - yok, pre-training)",
                    y_label=early_lbl.replace("SLI", "SLI\n"),
                    cfg=cfg,
                    filename="corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast",
                    customizer=customizer,
                )
        else:
            print(
                "[correlations] WARNING: missing fast-learner summary; "
                "skipping fast-only pre-vs-early PI correlation"
            )

        # --- Plot 5c: Pre-training vs T1 first-bucket PI (fast vs slow) ---
        fast_idx_fs, slow_idx_fs = _fast_slow_indices_from_sli_T1_first(
            reward_pi_training_vals, frac
        )

        if fast_idx_fs.size == 0 or slow_idx_fs.size == 0:
            print(
                "[correlations] WARNING: empty fast/slow groups; "
                "skipping fast/slow pre-vs-early PI correlation"
            )
        else:
            plot_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow(
                pre_pi_diff_vals=pre_pi_diff_vals,
                reward_pi_first_bucket=reward_pi_training_vals,
                fast_idx=fast_idx_fs,
                slow_idx=slow_idx_fs,
                out_dir=out_dir,
                frac=frac,
                customizer=customizer,
                early_label=early_lbl,
                image_format=cfg.image_format,
            )

        # --- Plot 5d: Baseline PI vs early SLI for selected SLI groups ---
        if selected_mode is not None:
            if selected_mode == "top":
                title_5_sel = "Baseline PI vs early SLI (top SLI-selected learners)"
                filename_5_sel = (
                    "corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_top_selected"
                )
            elif selected_mode == "bottom":
                title_5_sel = "Baseline PI vs early SLI (bottom SLI-selected learners)"
                filename_5_sel = (
                    "corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_bottom_selected"
                )
            else:
                title_5_sel = (
                    "Baseline PI vs early SLI (top vs bottom SLI-selected learners)"
                )
                filename_5_sel = (
                    "corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_selected_extremes"
                )

            plot_selected_group_scatter(
                x=pre_pi_diff_vals,
                y=reward_pi_training_vals,
                bottom_idx=selected_bottom_idx,
                top_idx=selected_top_idx,
                mode=selected_mode,
                title=title_5_sel,
                x_label="Baseline PI\n(exp - yok, pre-training)",
                y_label=early_lbl.replace("SLI", "SLI\n"),
                filename=filename_5_sel,
                out_dir=out_dir,
                customizer=customizer,
                top_label=top_sel_label,
                bottom_label=bottom_sel_label,
                xlim=cfg.xlim,
                ylim=cfg.ylim,
                image_format=cfg.image_format,
            )

    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; "
            "skipping plots 4–5"
        )

    if summary is not None:
        plot_fast_vs_strong_scatter(
            sli_T1_first=reward_pi_first_bucket,
            sli_strong=sli_vals,
            vas=vas,
            fast_idx=summary["fast"],
            strong_idx=summary["strong"],
            out_dir=out_dir,
            frac=frac,
            customizer=customizer,
            strong_y_label=x_label_sli,
            strong_title_suffix=sli_ctx.label_short(),
            x_label=t1_sb1_lbl,
            image_format=cfg.image_format,
        )
