# src/plotting/cross_fly_correlations.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.analysis.correlation_stats import pearson_correlation_summary
from src.analysis.sli_tools import default_single_bucket_idx
from src.exporting.speed_sli_bundle import _extract_speed_arrays
from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.palettes import (
    MUTED_CATEGORICAL,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    correlation_plot_color,
)
from src.plotting.p_value_format import format_plot_p_value
from src.plotting.rewards_per_distance_totals import pooled_rewards_per_distance_window
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.axis_size import DEFAULT_PLOT_AXIS_SIZE_INCHES, set_axis_size_inches
from src.plotting.reward_window_utils import (
    cumulative_window_seconds_for_frame,
    frames_in_windows,
    selected_windows_for_va,
)
from src.utils.common import writeImage
from src.utils.debug_fly_groups import log_fly_group, write_sorted_fly_list

BBOX_STYLE = dict(
    facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
)
STATS_BOX_MIN_FONTSIZE = 12.0
TREND_LINE_P_THRESHOLD = 0.05
CORRELATION_REFERENCE_FONT_SIZE = 16.0


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
    dot_color: str = MUTED_CATEGORICAL[0]
    alpha: float = 0.85
    figsize: tuple = (5.5, 4.5)
    axis_size_inches: tuple[float, float] = DEFAULT_PLOT_AXIS_SIZE_INCHES
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    export_npz_dir: Optional[Path] = None
    export_group_label: Optional[str] = None
    window_metric_aggregation: str = "pooled"
    rpd_pooled_validity: str = "window"
    rpd_pooled_min_rewards: int = 5


def _correlation_out_path(out_dir: Path, filename: str, image_format: str) -> Path:
    ext = image_format or "png"
    return out_dir / f"{filename}.{ext}"


def _cfg_with_plot_color(
    cfg: CorrelationPlotConfig, plot_key: str
) -> CorrelationPlotConfig:
    return replace(
        cfg,
        dot_color=correlation_plot_color(plot_key, fallback=cfg.dot_color),
    )


def _corr_export_group_label(opts) -> str | None:
    label = getattr(opts, "corr_export_group_label", None)
    if label:
        return str(label)
    label = getattr(opts, "export_group_label", None)
    if label:
        return str(label)
    labels = getattr(opts, "groupLabels", None)
    if labels:
        return str(labels).split("|")[0]
    return None


def _export_scatter_npz(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
    cfg: CorrelationPlotConfig,
    r: float,
    p: float,
) -> None:
    if cfg.export_npz_dir is None:
        return
    out_dir = Path(cfg.export_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.npz"
    summary = pearson_correlation_summary(x, y)
    meta = {
        "metric": "cross_fly_correlation_scatter",
        "filename": filename,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "group": cfg.export_group_label,
        "window_metric_aggregation": cfg.window_metric_aggregation,
        "rpd_pooled_validity": cfg.rpd_pooled_validity,
        "rpd_pooled_min_rewards": cfg.rpd_pooled_min_rewards,
        "r": float(r),
        "p": float(p),
        "n": int(summary.n),
        "corr_method": "pearson",
    }
    np.savez_compressed(
        out_path,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        r=np.asarray(float(r), dtype=float),
        p=np.asarray(float(p), dtype=float),
        n=np.asarray(int(summary.n), dtype=int),
        x_label=np.asarray(str(x_label), dtype=object),
        y_label=np.asarray(str(y_label), dtype=object),
        title=np.asarray(str(title), dtype=object),
        group=np.asarray(
            "" if cfg.export_group_label is None else cfg.export_group_label,
            dtype=object,
        ),
        meta_json=json.dumps(meta, sort_keys=True),
    )
    print(f"[correlations] wrote scatter export {out_path}")


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
    total_sync_buckets: int | None = None

    def _window_bounds(self) -> tuple[int, int | None]:
        if self.explicit_bucket_idx is not None:
            sb = int(self.explicit_bucket_idx) + 1
            return sb, sb
        start_sb = int(self.skip_first_sync_buckets or 0) + 1  # 1-based
        keep = int(self.keep_first_sync_buckets or 0)
        if keep > 0:
            end_sb = start_sb + keep - 1
        elif self.total_sync_buckets is not None:
            start_idx = start_sb - 1
            total = int(self.total_sync_buckets)
            end_sb = (
                None
                if total <= start_idx
                else default_single_bucket_idx(start_idx, total) + 1
            )
        else:
            end_sb = None
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

    def _axis_window_text(self) -> str:
        trn = self.training_idx + 1
        start_sb, end_sb = self._window_bounds()
        if end_sb is None:
            return f"T{trn} SB{start_sb}–end"
        if end_sb == start_sb:
            return f"T{trn} SB{start_sb}"
        return f"T{trn} SB{start_sb}–{end_sb}"

    def axis_label(self) -> str:
        window_txt = self._axis_window_text()
        start_sb, end_sb = self._window_bounds()
        if self.average_over_buckets:
            return f"Mean SLI over {window_txt}"
        if start_sb == end_sb:
            return f"SLI at {window_txt}"
        if end_sb is None and start_sb == 1:
            return f"SLI at final SB of T{self.training_idx + 1}"
        return f"SLI at final SB in {window_txt}"

    def metric_axis_label(self, metric_name: str, *, unit: str | None = None) -> str:
        window_txt = self._axis_window_text()
        start_sb, end_sb = self._window_bounds()
        if self.average_over_buckets:
            window_phrase = f"mean over {window_txt}"
        elif start_sb == end_sb:
            window_phrase = f"at {window_txt}"
        elif end_sb is None and start_sb == 1:
            window_phrase = f"at final SB of T{self.training_idx + 1}"
        else:
            window_phrase = f"at final SB in {window_txt}"
        label = f"{metric_name}, {window_phrase}"
        if unit:
            label = f"{label} ({unit})"
        return label

    def label_long(self) -> str:
        trn = self.training_idx + 1
        start_sb, end_sb = self._window_bounds()
        if start_sb == end_sb:
            return f"SLI (sync bucket {start_sb}, training {trn})"
        if not self.average_over_buckets:
            if end_sb is None:
                return f"SLI (sync buckets {start_sb}-end, training {trn})"
            return f"SLI (sync buckets {start_sb}-{end_sb}, training {trn})"
        window_txt = (
            f", sync buckets {start_sb}-end"
            if end_sb is None
            else f", sync buckets {start_sb}-{end_sb}"
        )
        return f"SLI (mean over{window_txt}, training {trn})"

    def label_short(self, abbrev_sb=True) -> str:
        trn = self.training_idx + 1
        window_txt = self._window_text(abbrev_sb=abbrev_sb)
        if self._window_bounds()[0] == self._window_bounds()[1]:
            return f"SLI (T{trn}, {window_txt})"
        if self.average_over_buckets:
            return f"SLI (T{trn}, mean, {window_txt})"
        return f"SLI (T{trn}, {window_txt})"


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


def _default_t2_speed_vs_final_sli_contexts() -> (
    tuple[SLIContext, tuple[tuple[SLIContext, str], ...]]
):
    """Fixed contexts for the default speed comparisons against T2 SB5 SLI."""
    final_sli_ctx = SLIContext(
        training_idx=1,
        average_over_buckets=False,
        explicit_bucket_idx=4,
        total_sync_buckets=5,
    )
    return final_sli_ctx, (
        (
            SLIContext(
                training_idx=1,
                average_over_buckets=False,
                explicit_bucket_idx=4,
                total_sync_buckets=5,
            ),
            "Speed at T2 SB5 and final SLI",
        ),
        (
            SLIContext(
                training_idx=1,
                average_over_buckets=True,
                keep_first_sync_buckets=5,
                total_sync_buckets=5,
            ),
            "Mean T2 speed (SB1-SB5) and final SLI",
        ),
    )


def _default_pre_training_speed_vs_mean_t2_sli_context() -> SLIContext:
    """Return the fixed SLI window used with final-10-min pre-training speed."""
    return SLIContext(
        training_idx=1,
        average_over_buckets=True,
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=4,
        total_sync_buckets=5,
    )


def _default_t1_vs_t2_mean_sli_contexts() -> tuple[SLIContext, SLIContext]:
    """Return the fixed T1/T2 SLI windows used for the across-training comparison."""
    return (
        SLIContext(
            training_idx=0,
            average_over_buckets=True,
            skip_first_sync_buckets=1,
            keep_first_sync_buckets=4,
            total_sync_buckets=5,
        ),
        SLIContext(
            training_idx=1,
            average_over_buckets=True,
            skip_first_sync_buckets=1,
            keep_first_sync_buckets=4,
            total_sync_buckets=5,
        ),
    )


def _speed_selection_group_labels(
    sli_ctx: SLIContext,
    *,
    top_pct_txt: str,
    bottom_pct_txt: str,
) -> tuple[str, str]:
    """Describe both the selected fraction and the SLI window defining it."""
    selection_ctx_label = sli_ctx.axis_label()
    return (
        f"Top {top_pct_txt} by {selection_ctx_label}",
        f"Bottom {bottom_pct_txt} by {selection_ctx_label}",
    )


def _windowed_metric_label(metric_name: str, ctx: SLIContext) -> str:
    window_txt = ctx._window_text(abbrev_sb=True)
    if ctx.average_over_buckets:
        return f"{metric_name}\n(mean {window_txt}, T{ctx.training_idx + 1})"
    if ctx._window_bounds()[0] == ctx._window_bounds()[1]:
        return f"{metric_name}\n(T{ctx.training_idx + 1}, {window_txt})"
    if ctx.skip_first_sync_buckets or ctx.keep_first_sync_buckets:
        return f"{metric_name}\n({window_txt}, T{ctx.training_idx + 1})"
    if ctx.training_idx != 0:
        return f"{metric_name}\n(T{ctx.training_idx + 1})"
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
        "first-to-nth span"
        if str(time_basis) == "first_to_nth"
        else "window start to nth"
    )
    return (
        f"rewards per minute\n"
        f"(first {int(first_n_rewards)} calc rewards, {basis_txt}{cutoff_txt}, {window_txt}, T{ctx.training_idx + 1})"
    )


def early_sli_label(*, training_idx: int, skip_first_sync_buckets: int) -> str:
    k = int(skip_first_sync_buckets or 0)
    return SLIContext(training_idx=training_idx, explicit_bucket_idx=k).axis_label()


def _format_corr_annotation(
    r: float, p: float, n: int, *, label: str | None = None
) -> str:
    stats = f"n = {int(n)}, r = {r:.3f}, p = {format_plot_p_value(p)}"
    return f"{label}: {stats}" if label else stats


def _format_labeled_corr_with_n(
    r: float, p: float, n: int, *, label: str | None = None
) -> str:
    stats = f"r = {r:.3f}, p = {format_plot_p_value(p)}"
    return f"{label} (n = {int(n)}): {stats}" if label else f"n = {int(n)}, {stats}"


def _format_labeled_corr_na_with_n(n: int, *, label: str) -> str:
    return f"{label} (n = {int(n)}): r = n/a, p = n/a"


def _format_compact_labeled_corr_with_n(
    r: float,
    p: float,
    n: int,
    *,
    label: str,
) -> str:
    """Format a narrow correlation row for large-font in-axes annotations."""
    return f"{label}: n={int(n)}, r={r:.3f}, p={format_plot_p_value(p)}"


def _format_compact_labeled_corr_na_with_n(n: int, *, label: str) -> str:
    return f"{label}: n={int(n)}, r=n/a, p=n/a"


def _compute_group_corr(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> tuple[float, float, int] | None:
    """
    Compute Pearson correlation for a given index set, handling NaNs and
    small sample sizes. Returns (r, p, n) or None if not enough valid data.
    """
    if idx is None or idx.size == 0:
        return None

    idx = np.asarray(idx, dtype=int)
    x_g = np.asarray(x, float)[idx]
    y_g = np.asarray(y, float)[idx]

    mask = np.isfinite(x_g) & np.isfinite(y_g)
    n = int(np.sum(mask))
    if n < 3:
        return None

    r, p = pearsonr(x_g[mask], y_g[mask])
    return float(r), float(p), n


def _add_significant_trend_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    *,
    color: str,
    linestyle: str = "--",
    linewidth: float = 1.6,
    alpha: float = 0.85,
) -> bool:
    """
    Draw a linear trend line when the reported Pearson p-value is significant.
    """
    try:
        p_val = float(p)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(p_val) or p_val > TREND_LINE_P_THRESHOLD:
        return False

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite]
    y_f = y[finite]
    if x_f.size < 3 or np.unique(x_f).size < 2:
        return False

    try:
        slope, intercept = np.polyfit(x_f, y_f, 1)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return False
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return False

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_line = np.asarray(xlim, float)
    if x_line.size != 2 or not np.all(np.isfinite(x_line)):
        return False
    y_line = slope * x_line + intercept
    ax.plot(
        x_line,
        y_line,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return True


def _shrink_clipped_ylabels(
    fig, *, min_scale: float = 0.72, pad_px: float = 2.0
) -> bool:
    """
    Reduce oversized Y-axis labels only when their rendered bbox is clipped.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
    except Exception:
        return False

    changed = False
    for ax in fig.get_axes():
        label = ax.yaxis.get_label()
        if not label.get_visible() or not label.get_text():
            continue

        original_size = float(label.get_fontsize())
        min_size = original_size * float(min_scale)
        for _ in range(10):
            bbox = label.get_window_extent(renderer=renderer)
            clipped = (
                float(bbox.x0) < float(fig_bbox.x0) + pad_px
                or float(bbox.y0) < float(fig_bbox.y0) + pad_px
                or float(bbox.y1) > float(fig_bbox.y1) - pad_px
            )
            if not clipped:
                break

            current_size = float(label.get_fontsize())
            next_size = max(min_size, current_size * 0.94)
            if next_size >= current_size:
                break
            label.set_fontsize(next_size)
            changed = True
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

    return changed


def _split_axis_label_evenly(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    mid = len(words) // 2
    left, right = words[:mid], words[mid:]
    if len(left) < 2 or len(right) < 2:
        return text
    return " ".join(left) + "\n" + " ".join(right)


def _wrap_clipped_axis_labels(fig, *, pad_px: float = 2.0) -> bool:
    """
    Wrap axis labels only when their rendered bbox exceeds the figure.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
    except Exception:
        return False

    changed = False
    for ax in fig.get_axes():
        x_label = ax.xaxis.get_label()
        x_text = x_label.get_text()
        if x_label.get_visible() and x_text and "\n" not in x_text:
            bbox = x_label.get_window_extent(renderer=renderer)
            clipped = (
                float(bbox.x0) < float(fig_bbox.x0) + pad_px
                or float(bbox.x1) > float(fig_bbox.x1) - pad_px
            )
            if clipped:
                wrapped = _split_axis_label_evenly(x_text)
                if wrapped != x_text:
                    x_label.set_text(wrapped)
                    changed = True

        y_label = ax.yaxis.get_label()
        y_text = y_label.get_text()
        if y_label.get_visible() and y_text and "\n" not in y_text:
            bbox = y_label.get_window_extent(renderer=renderer)
            clipped = (
                float(bbox.y0) < float(fig_bbox.y0) + pad_px
                or float(bbox.y1) > float(fig_bbox.y1) - pad_px
            )
            if clipped:
                wrapped = _split_axis_label_evenly(y_text)
                if wrapped != y_text:
                    y_label.set_text(wrapped)
                    changed = True

    return changed


def _finalize_correlation_layout(
    fig,
    customizer: PlotCustomizer,
    *,
    rect=None,
    axis_size_inches=None,
) -> None:
    customizer.adjust_padding_proportionally(wrap_axis_labels=False)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    if _wrap_clipped_axis_labels(fig):
        if rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=rect)
    if _shrink_clipped_ylabels(fig):
        if rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=rect)
    if axis_size_inches is None:
        axis_size_inches = getattr(
            customizer, "standard_plot_axis_size", DEFAULT_PLOT_AXIS_SIZE_INCHES
        )
    set_axis_size_inches(fig.axes[0], axis_size_inches)


def _correlation_axis_size_for_font(
    customizer: PlotCustomizer,
    *,
    base_size=DEFAULT_PLOT_AXIS_SIZE_INCHES,
) -> tuple[float, float]:
    """Grow correlation axes sublinearly from the font-16 baseline."""
    font_size = float(
        getattr(customizer, "font_size", CORRELATION_REFERENCE_FONT_SIZE)
    )
    font_scale = max(1.0, font_size / CORRELATION_REFERENCE_FONT_SIZE)
    scale = min(font_scale**0.35, 1.15)
    return float(base_size[0]) * scale, float(base_size[1]) * scale


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
        base_fontsize = (
            legend.get_texts()[0].get_fontsize() if legend.get_texts() else 10
        )
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
    fontsize: float | None = None,
    max_overlap_frac: float = 0.08,
    max_headroom_frac: float = 0.25,
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
    if fontsize is None:
        reference_sizes = [
            ax.xaxis.label.get_size(),
            ax.yaxis.label.get_size(),
            *(tick.get_size() for tick in ax.get_xticklabels()),
            *(tick.get_size() for tick in ax.get_yticklabels()),
        ]
        reference_size = max(
            float(size) for size in reference_sizes if size is not None
        )
        fontsize = max(STATS_BOX_MIN_FONTSIZE, 0.90 * reference_size)

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
            stats_bbox = text_artist.get_bbox_patch().get_window_extent(
                renderer=renderer
            )
            intersects_legend = legend_bbox is not None and not (
                stats_bbox.x1 < legend_bbox.x0
                or stats_bbox.x0 > legend_bbox.x1
                or stats_bbox.y1 < legend_bbox.y0
                or stats_bbox.y0 > legend_bbox.y1
            )
            stats_pts_axes = ax.transAxes.inverted().transform(
                np.array(
                    [[stats_bbox.x0, stats_bbox.y0], [stats_bbox.x1, stats_bbox.y1]]
                )
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

    if best_candidate is not None and extra_top > max_headroom_frac * y_span:
        capped_extra_top = max_headroom_frac * y_span
        ax.set_ylim(y0, y1 + capped_extra_top)
        fig.canvas.draw()
        pts_display = ax.transData.transform(np.column_stack([x_f, y_f]))

        fallback_fontsize = float(fontsize)
        fallback_candidate = best_candidate
        fallback_score = None
        fallback_raw_overlap = None
        min_fontsize = STATS_BOX_MIN_FONTSIZE

        probe = ax.text(
            fallback_candidate["x"],
            fallback_candidate["y"],
            text,
            transform=ax.transAxes,
            va=fallback_candidate["va"],
            ha=fallback_candidate["ha"],
            fontsize=fallback_fontsize,
            zorder=5,
            alpha=0.0,
            bbox=BBOX_STYLE,
        )
        for scale in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60):
            candidate_fontsize = max(min_fontsize, float(fontsize) * scale)
            probe.set_fontsize(candidate_fontsize)
            fig.canvas.draw()
            patch_bbox = probe.get_bbox_patch().get_window_extent(renderer=renderer)
            patch_pts_axes = ax.transAxes.inverted().transform(
                np.array(
                    [[patch_bbox.x0, patch_bbox.y0], [patch_bbox.x1, patch_bbox.y1]]
                )
            )
            patch_axes_bbox = (
                float(np.min(patch_pts_axes[:, 0])),
                float(np.min(patch_pts_axes[:, 1])),
                float(np.max(patch_pts_axes[:, 0])),
                float(np.max(patch_pts_axes[:, 1])),
            )
            inside = (
                (pts_display[:, 0] >= patch_bbox.x0)
                & (pts_display[:, 0] <= patch_bbox.x1)
                & (pts_display[:, 1] >= patch_bbox.y0)
                & (pts_display[:, 1] <= patch_bbox.y1)
            )
            raw_overlap = float(np.mean(inside))
            score = raw_overlap
            overflows_axes = (
                patch_axes_bbox[0] < 0.0
                or patch_axes_bbox[2] > 1.0
                or patch_axes_bbox[1] < 0.0
                or patch_axes_bbox[3] > 1.0
            )
            if overflows_axes:
                score = 1.0 + score
            if legend_bbox is not None:
                overlaps_legend = not (
                    patch_bbox.x1 < legend_bbox.x0
                    or patch_bbox.x0 > legend_bbox.x1
                    or patch_bbox.y1 < legend_bbox.y0
                    or patch_bbox.y0 > legend_bbox.y1
                )
                if overlaps_legend:
                    score = 1.0 + score
            if fallback_score is None or score < fallback_score:
                fallback_fontsize = candidate_fontsize
                fallback_score = score
                fallback_raw_overlap = raw_overlap
            if score <= max_overlap_frac and not overflows_axes:
                break
        probe.remove()

        text_artist = ax.text(
            fallback_candidate["x"],
            fallback_candidate["y"],
            text,
            transform=ax.transAxes,
            va=fallback_candidate["va"],
            ha=fallback_candidate["ha"],
            fontsize=fallback_fontsize,
            zorder=5,
            bbox=BBOX_STYLE,
        )
        fig.canvas.draw()
        stats_bbox = text_artist.get_bbox_patch().get_window_extent(renderer=renderer)
        intersects_legend = legend_bbox is not None and not (
            stats_bbox.x1 < legend_bbox.x0
            or stats_bbox.x0 > legend_bbox.x1
            or stats_bbox.y1 < legend_bbox.y0
            or stats_bbox.y0 > legend_bbox.y1
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
            f"title={ax.get_title()!r} mode=corner_capped_headroom "
            f"candidate={fallback_candidate} raw_overlap={fallback_raw_overlap:.4f} "
            f"score={fallback_score:.4f} requested_extra_top={extra_top:.4f} "
            f"capped_extra_top={capped_extra_top:.4f} "
            f"fontsize={fallback_fontsize:.2f} "
            f"legend_axes_bbox={legend_axes_bbox} stats_axes_bbox={stats_axes_bbox} "
            f"intersects_legend={intersects_legend}"
        )
        return text_artist

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
    intersects_legend = legend_bbox is not None and not (
        stats_bbox.x1 < legend_bbox.x0
        or stats_bbox.x0 > legend_bbox.x1
        or stats_bbox.y1 < legend_bbox.y0
        or stats_bbox.y0 > legend_bbox.y1
    )
    vertical_gap_px = None
    if legend_bbox is not None:
        vertical_gap_px = max(
            stats_bbox.y0 - legend_bbox.y1, legend_bbox.y0 - stats_bbox.y1
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
        f"title={ax.get_title()!r} mode=headroom candidate={best_candidate} "
        f"raw_overlap={best_raw_overlap:.4f} score={best_overlap:.4f} "
        f"box_height_frac={box_height_frac:.4f} legend_top_frac={legend_top_frac} "
        f"legend_nudge_down={legend_nudge_down:.4f} "
        f"legend_clear_frac={legend_clear_frac} extra_top={extra_top:.4f} "
        f"legend_axes_bbox={legend_axes_bbox} stats_axes_bbox={stats_axes_bbox} "
        f"vertical_gap_px={vertical_gap_px} intersects_legend={intersects_legend}"
    )
    return text_artist


def _place_correlation_overlays(
    ax,
    legend_handles,
    stats_text: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scatter_artist=None,
    compact_stats_text: str | None = None,
    compact_legend_labels: Sequence[str] | None = None,
    compact_labels_min_font_size: float = 24.0,
    configured_font_size: float | None = None,
    axis_scale: float | None = None,
    max_headroom_frac: float = 0.50,
    split_corner_max_right_frac: float = 0.20,
    split_corner_max_lower_frac: float = 0.20,
    annotation_band_max_headroom_frac: float = 0.90,
):
    """
    Jointly place a correlation legend and stats box.

    Placement is evaluated after the axes have reached their final physical
    size. Internal candidates must avoid scatter markers, plotted lines, each
    other, and the axes boundary. Added y headroom is always measured from the
    original data range. General placements are capped by max_headroom_frac;
    the split-corner placement may add modest right and lower padding, and a
    stacked annotation-band placement may use up to
    annotation_band_max_headroom_frac. Compact wording is available only at
    configured font sizes greater than or equal to
    compact_labels_min_font_size.

    If no internal layout is collision-free, both overlays are placed outside
    the right side of the axes.
    """
    fig = ax.figure

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite]
    y_f = y[finite]

    # This is the immutable baseline. Every trial is derived from it rather
    # than from the y-limit left behind by the previous trial.
    base_x0, base_x1 = ax.get_xlim()
    base_x_span = float(base_x1 - base_x0)
    if not np.isfinite(base_x_span) or base_x_span <= 0:
        base_x_span = max(float(np.ptp(x_f)) if x_f.size else 0.0, 1.0)
        base_x1 = base_x0 + base_x_span

    base_y0, base_y1 = ax.get_ylim()
    base_y_span = float(base_y1 - base_y0)
    if not np.isfinite(base_y_span) or base_y_span <= 0:
        base_y_span = max(float(np.ptp(y_f)) if y_f.size else 0.0, 1.0)
        base_y1 = base_y0 + base_y_span

    reference_sizes = [
        ax.xaxis.label.get_size(),
        ax.yaxis.label.get_size(),
        *(tick.get_size() for tick in ax.get_xticklabels()),
        *(tick.get_size() for tick in ax.get_yticklabels()),
    ]
    reference_size = max(float(size) for size in reference_sizes if size is not None)
    if configured_font_size is None:
        configured_font_size = reference_size
    configured_font_size = float(configured_font_size)

    def _unique_font_sizes(*values):
        sizes = []
        for value in values:
            value = float(value)
            if not any(np.isclose(value, previous) for previous in sizes):
                sizes.append(value)
        return tuple(sizes)

    # Grow overlays faster than the axes but more slowly than the configured
    # font. This makes their proportional footprint increase while keeping
    # long multi-line annotations physically capable of fitting in the axes.
    # The second tier permits only a modest reduction; there is deliberately
    # no absolute fallback that could make a larger configured font smaller
    # than the same overlay at a lower configured size.
    configured_scale = max(
        1.0, configured_font_size / CORRELATION_REFERENCE_FONT_SIZE
    )
    overlay_scale = configured_scale**0.70
    preferred_legend_fontsize = (
        CORRELATION_REFERENCE_FONT_SIZE - 3.0
    ) * overlay_scale
    preferred_stats_fontsize = CORRELATION_REFERENCE_FONT_SIZE * overlay_scale
    legend_font_sizes = _unique_font_sizes(
        max(preferred_legend_fontsize, 6.0),
        max(0.90 * preferred_legend_fontsize, 6.0),
    )
    stats_font_sizes = _unique_font_sizes(
        max(preferred_stats_fontsize, 6.0),
        max(0.90 * preferred_stats_fontsize, 6.0),
    )

    use_compact_labels = configured_font_size >= float(
        compact_labels_min_font_size
    )

    stats_text_variants = [("full", stats_text)]
    if (
        use_compact_labels
        and compact_stats_text
        and compact_stats_text != stats_text
    ):
        stats_text_variants.append(("compact", compact_stats_text))

    if use_compact_labels and compact_legend_labels is not None:
        compact_legend_labels = tuple(compact_legend_labels)
        if len(compact_legend_labels) != len(legend_handles):
            raise ValueError(
                "compact_legend_labels must match the number of legend handles"
            )
    else:
        compact_legend_labels = tuple(
            handle.get_label() for handle in legend_handles
        )

    # Search the smallest headroom first.
    headroom_fracs = tuple(np.linspace(0.0, float(max_headroom_frac), 6))

    legend_locations = (
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "center right",
        "center left",
        "upper center",
        "lower center",
    )

    stats_candidates = (
        dict(x=0.03, y=0.97, ha="left", va="top"),
        dict(x=0.97, y=0.97, ha="right", va="top"),
        dict(x=0.03, y=0.03, ha="left", va="bottom"),
        dict(x=0.97, y=0.03, ha="right", va="bottom"),
    )

    marker_pad_px = 2.0
    if scatter_artist is not None:
        try:
            sizes = np.asarray(scatter_artist.get_sizes(), dtype=float)
            finite_sizes = sizes[np.isfinite(sizes) & (sizes > 0)]

            if finite_sizes.size:
                # Scatter sizes are specified in points squared. Use a
                # conservative marker radius plus a small visual margin.
                marker_radius_px = (
                    0.5 * np.sqrt(float(np.max(finite_sizes))) * fig.dpi / 72.0
                )
                marker_pad_px = max(marker_pad_px, marker_radius_px + 2.0)
        except (AttributeError, TypeError, ValueError):
            pass

    def _expanded_bbox(bbox, pad_px):
        width = max(float(bbox.width), 1.0)
        height = max(float(bbox.height), 1.0)
        return bbox.expanded(
            (width + 2.0 * pad_px) / width,
            (height + 2.0 * pad_px) / height,
        )

    def _bbox_overlap(first, second):
        return not (
            first.x1 < second.x0
            or first.x0 > second.x1
            or first.y1 < second.y0
            or first.y0 > second.y1
        )

    def _bbox_inside(inner, outer, pad_px=1.0):
        return (
            inner.x0 >= outer.x0 + pad_px
            and inner.x1 <= outer.x1 - pad_px
            and inner.y0 >= outer.y0 + pad_px
            and inner.y1 <= outer.y1 - pad_px
        )

    def _count_point_overlap(bbox, points_display, pad_px):
        if points_display.size == 0:
            return 0

        padded_bbox = _expanded_bbox(bbox, pad_px)
        inside = (
            (points_display[:, 0] >= padded_bbox.x0)
            & (points_display[:, 0] <= padded_bbox.x1)
            & (points_display[:, 1] >= padded_bbox.y0)
            & (points_display[:, 1] <= padded_bbox.y1)
        )
        return int(np.sum(inside))

    def _line_samples_display(max_step_px=2.0):
        """
        Return densely sampled display-coordinate points for visible lines.

        The interpolation keeps the maximum distance between samples small,
        making bbox/line intersection checks reliable for straight trend
        lines without requiring a full segment-rectangle intersection
        implementation.
        """
        sampled = []

        for line in ax.lines:
            if not line.get_visible():
                continue

            try:
                vertices = line.get_path().transformed(line.get_transform()).vertices
            except (AttributeError, TypeError, ValueError):
                continue
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2:
                continue

            finite_vertices = np.all(np.isfinite(vertices), axis=1)
            vertices = vertices[finite_vertices]
            if vertices.size == 0:
                continue

            if len(vertices) == 1:
                sampled.append(vertices)
                continue

            for start, end in zip(vertices[:-1], vertices[1:]):
                distance = float(np.linalg.norm(end - start))
                n_samples = max(2, int(np.ceil(distance / max_step_px)) + 1)
                sampled.append(np.linspace(start, end, n_samples, endpoint=True))

        if not sampled:
            return np.empty((0, 2), dtype=float)

        return np.vstack(sampled)

    def _overlay_is_clear(
        bbox,
        *,
        points_display,
        line_points_display,
        axes_bbox,
        point_pad_px,
    ):
        if not _bbox_inside(bbox, axes_bbox):
            return False

        if _count_point_overlap(bbox, points_display, point_pad_px):
            return False

        if _count_point_overlap(bbox, line_points_display, 2.5):
            return False

        return True

    rejection_counts = Counter()
    rejection_hits = Counter()
    closest_candidate = None
    closest_candidates_by_layout = {}

    def _overlay_rejections(
        bbox,
        *,
        points_display,
        line_points_display,
        axes_bbox,
        point_pad_px,
    ):
        reasons = {}
        if not _bbox_inside(bbox, axes_bbox):
            reasons["outside_axes"] = 1

        point_hits = _count_point_overlap(bbox, points_display, point_pad_px)
        if point_hits:
            reasons["points"] = point_hits

        line_hits = _count_point_overlap(bbox, line_points_display, 2.5)
        if line_hits:
            reasons["lines"] = line_hits

        return reasons

    def _outside_distance_px(bbox, axes_bbox):
        return sum(
            (
                max(0.0, axes_bbox.x0 + 1.0 - bbox.x0),
                max(0.0, bbox.x1 - (axes_bbox.x1 - 1.0)),
                max(0.0, axes_bbox.y0 + 1.0 - bbox.y0),
                max(0.0, bbox.y1 - (axes_bbox.y1 - 1.0)),
            )
        )

    def _overlap_area(first, second):
        width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
        return width * height

    def _record_rejections(prefix, reasons):
        for reason, hits in reasons.items():
            rejection_counts[f"{prefix}_{reason}"] += 1
            rejection_hits[f"{prefix}_{reason}"] += hits

    def _consider_candidate(
        *,
        description,
        legend_bbox,
        stats_bbox,
        legend_reasons,
        stats_reasons,
        axes_bbox,
    ):
        nonlocal closest_candidate

        overlays_overlap = _bbox_overlap(legend_bbox, stats_bbox)
        _record_rejections("legend", legend_reasons)
        _record_rejections("stats", stats_reasons)
        if overlays_overlap:
            rejection_counts["overlay_overlap"] += 1

        # This score is diagnostic only. It favors candidates with fewer data
        # collisions, followed by less boundary overflow and box overlap.
        data_hits = sum(
            hits
            for reason, hits in (*legend_reasons.items(), *stats_reasons.items())
            if reason in {"points", "lines"}
        )
        outside_px = float(
            _outside_distance_px(legend_bbox, axes_bbox)
            + _outside_distance_px(stats_bbox, axes_bbox)
        )
        overlap_fraction = float(
            _overlap_area(legend_bbox, stats_bbox)
            / max(float(axes_bbox.width * axes_bbox.height), 1.0)
        )

        def axes_fraction_bounds(bbox):
            return tuple(
                round(float(value), 3)
                for value in (
                    (bbox.x0 - axes_bbox.x0) / axes_bbox.width,
                    (bbox.y0 - axes_bbox.y0) / axes_bbox.height,
                    (bbox.x1 - axes_bbox.x0) / axes_bbox.width,
                    (bbox.y1 - axes_bbox.y0) / axes_bbox.height,
                )
            )

        score = (
            len(legend_reasons) + len(stats_reasons) + int(overlays_overlap),
            data_hits,
            outside_px,
            overlap_fraction,
        )
        candidate = dict(
            score=score,
            description=description,
            legend_reasons=legend_reasons,
            stats_reasons=stats_reasons,
            overlays_overlap=overlays_overlap,
            legend_bounds=axes_fraction_bounds(legend_bbox),
            stats_bounds=axes_fraction_bounds(stats_bbox),
        )
        if closest_candidate is None or score < closest_candidate["score"]:
            closest_candidate = candidate
        layout = description.split(maxsplit=1)[0]
        previous_for_layout = closest_candidates_by_layout.get(layout)
        if previous_for_layout is None or score < previous_for_layout["score"]:
            closest_candidates_by_layout[layout] = candidate

        return not legend_reasons and not stats_reasons and not overlays_overlap

    def _format_rejection_summary():
        if not rejection_counts:
            return "none"
        entries = []
        for reason in sorted(rejection_counts):
            entry = f"{reason}:{rejection_counts[reason]}"
            if rejection_hits[reason]:
                entry += f"(hits={rejection_hits[reason]})"
            entries.append(entry)
        return ",".join(entries)

    def _format_closest_candidate():
        if closest_candidate is None:
            return "none"

        entries = []
        for layout in sorted(closest_candidates_by_layout):
            candidate = closest_candidates_by_layout[layout]
            entries.append(
                f"[{candidate['description']} "
                f"legend_reasons={candidate['legend_reasons']} "
                f"stats_reasons={candidate['stats_reasons']} "
                f"overlay_overlap={candidate['overlays_overlap']} "
                f"legend_bounds={candidate['legend_bounds']} "
                f"stats_bounds={candidate['stats_bounds']} "
                f"score={candidate['score']}]"
            )
        return " ".join(entries)

    # Exhaust every placement/headroom option at the preferred font sizes
    # before trying the slightly reduced tier. Within each tier, retain the
    # full annotation wording when possible, then try its compact equivalent.
    font_tiers = tuple(zip(legend_font_sizes, stats_font_sizes))
    for font_tier, (legend_fontsize, stats_fontsize) in enumerate(font_tiers):
        for stats_format, candidate_stats_text in stats_text_variants:
            for headroom_frac in headroom_fracs:
                candidate_top = base_y1 + headroom_frac * base_y_span
                ax.set_ylim(base_y0, candidate_top)
                fig.canvas.draw()

                renderer = fig.canvas.get_renderer()
                axes_bbox = ax.get_window_extent(renderer=renderer)
                if x_f.size:
                    points_display = ax.transData.transform(
                        np.column_stack([x_f, y_f])
                    )
                else:
                    points_display = np.empty((0, 2), dtype=float)

                line_points_display = _line_samples_display()

                # Preserve the current single-column appearance when possible,
                # but allow two columns to reduce vertical height.
                for legend_ncol in (1, 2):
                    for legend_loc in legend_locations:
                        legend = ax.legend(
                            handles=legend_handles,
                            loc=legend_loc,
                            ncol=legend_ncol,
                            frameon=True,
                            fontsize=legend_fontsize,
                        )
                        fig.canvas.draw()

                        renderer = fig.canvas.get_renderer()
                        legend_bbox = legend.get_window_extent(renderer=renderer)

                        legend_reasons = _overlay_rejections(
                            legend_bbox,
                            points_display=points_display,
                            line_points_display=line_points_display,
                            axes_bbox=axes_bbox,
                            point_pad_px=marker_pad_px,
                        )
                        if legend_reasons:
                            _record_rejections("legend", legend_reasons)
                            legend.remove()
                            continue

                        for candidate in stats_candidates:
                            stats_artist = ax.text(
                                candidate["x"],
                                candidate["y"],
                                candidate_stats_text,
                                transform=ax.transAxes,
                                ha=candidate["ha"],
                                va=candidate["va"],
                                fontsize=stats_fontsize,
                                zorder=5,
                                bbox=BBOX_STYLE,
                            )
                            fig.canvas.draw()

                            renderer = fig.canvas.get_renderer()
                            stats_bbox = (
                                stats_artist.get_bbox_patch().get_window_extent(
                                    renderer=renderer
                                )
                            )

                            stats_reasons = _overlay_rejections(
                                stats_bbox,
                                points_display=points_display,
                                line_points_display=line_points_display,
                                axes_bbox=axes_bbox,
                                point_pad_px=marker_pad_px,
                            )
                            candidate_valid = _consider_candidate(
                                description=(
                                    f"layout=standard font_tier={font_tier} "
                                    f"stats_format={stats_format} "
                                    f"headroom_frac={headroom_frac:.3f} "
                                    f"legend_loc={legend_loc!r} "
                                    f"legend_ncol={legend_ncol} "
                                    f"stats_candidate={candidate}"
                                ),
                                legend_bbox=legend_bbox,
                                stats_bbox=stats_bbox,
                                legend_reasons={},
                                stats_reasons=stats_reasons,
                                axes_bbox=axes_bbox,
                            )

                            if candidate_valid:
                                # One final draw and bbox check guards against
                                # any renderer-dependent geometry changes.
                                fig.canvas.draw()
                                renderer = fig.canvas.get_renderer()
                                final_legend_bbox = legend.get_window_extent(
                                    renderer=renderer
                                )
                                final_stats_bbox = (
                                    stats_artist.get_bbox_patch().get_window_extent(
                                        renderer=renderer
                                    )
                                )

                                final_valid = (
                                    _overlay_is_clear(
                                        final_legend_bbox,
                                        points_display=points_display,
                                        line_points_display=line_points_display,
                                        axes_bbox=axes_bbox,
                                        point_pad_px=marker_pad_px,
                                    )
                                    and _overlay_is_clear(
                                        final_stats_bbox,
                                        points_display=points_display,
                                        line_points_display=line_points_display,
                                        axes_bbox=axes_bbox,
                                        point_pad_px=marker_pad_px,
                                    )
                                    and not _bbox_overlap(
                                        final_legend_bbox, final_stats_bbox
                                    )
                                )

                                if final_valid:
                                    _log_correlation_layout(
                                        f"title={ax.get_title()!r} "
                                        f"mode=joint_internal "
                                        f"layout=standard "
                                        f"configured_fontsize="
                                        f"{configured_font_size:.2f} "
                                        f"axis_scale={axis_scale} "
                                        f"font_tier={font_tier} "
                                        f"stats_format={stats_format} "
                                        f"headroom_frac={headroom_frac:.3f} "
                                        f"legend_loc={legend_loc!r} "
                                        f"legend_ncol={legend_ncol} "
                                        f"legend_fontsize="
                                        f"{legend_fontsize:.2f} "
                                        f"stats_candidate={candidate} "
                                        f"stats_fontsize="
                                        f"{stats_fontsize:.2f} "
                                        f"rejections="
                                        f"{_format_rejection_summary()}"
                                    )
                                    return legend, stats_artist
                            stats_artist.remove()
                        legend.remove()

    # A compact legend can often occupy the lower-right region if the data are
    # shifted slightly up and left. Search modest lower-y and right-x padding
    # before reserving a large shared annotation band at the top.
    def _padding_fraction_grid(maximum, step=0.05):
        maximum = max(0.0, float(maximum))
        if np.isclose(maximum, 0.0):
            return (0.0,)
        n_steps = max(1, int(np.ceil(maximum / float(step))))
        return tuple(np.linspace(0.0, maximum, n_steps + 1))

    split_right_fracs = _padding_fraction_grid(split_corner_max_right_frac)
    split_lower_fracs = _padding_fraction_grid(split_corner_max_lower_frac)
    split_padding_candidates = sorted(
        (
            (headroom_frac, right_frac, lower_frac)
            for headroom_frac in headroom_fracs
            for right_frac in split_right_fracs
            for lower_frac in split_lower_fracs
        ),
        key=lambda values: (
            values[0],
            values[1] + values[2],
            max(values[1], values[2]),
            values[2],
            values[1],
        ),
    )
    split_stats_variants = list(reversed(stats_text_variants))
    split_font_tiers = list(font_tiers)
    if not use_compact_labels:
        # Full correlation wording can be only a few pixels wider than the
        # axes at intermediate font sizes. Permit a narrowly scoped third
        # stats tier while retaining the existing 90% legend tier.
        full_label_tier = (
            legend_font_sizes[-1],
            max(0.88 * preferred_stats_fontsize, 6.0),
        )
        if not any(
            np.isclose(full_label_tier[0], legend_size)
            and np.isclose(full_label_tier[1], stats_size)
            for legend_size, stats_size in split_font_tiers
        ):
            split_font_tiers.append(full_label_tier)

    for font_tier, (legend_fontsize, stats_fontsize) in enumerate(
        split_font_tiers
    ):
        for stats_format, candidate_stats_text in split_stats_variants:
            for headroom_frac, right_frac, lower_frac in split_padding_candidates:
                candidate_right = base_x1 + right_frac * base_x_span
                candidate_bottom = base_y0 - lower_frac * base_y_span
                candidate_top = base_y1 + headroom_frac * base_y_span
                ax.set_xlim(base_x0, candidate_right)
                ax.set_ylim(candidate_bottom, candidate_top)
                fig.canvas.draw()

                renderer = fig.canvas.get_renderer()
                axes_bbox = ax.get_window_extent(renderer=renderer)
                if x_f.size:
                    points_display = ax.transData.transform(
                        np.column_stack([x_f, y_f])
                    )
                else:
                    points_display = np.empty((0, 2), dtype=float)
                line_points_display = _line_samples_display()

                stats_artist = ax.text(
                    0.5,
                    0.97,
                    candidate_stats_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=stats_fontsize,
                    linespacing=1.0,
                    zorder=5,
                    bbox={
                        **BBOX_STYLE,
                        "boxstyle": "round,pad=0.15",
                    },
                )

                for legend_ncol in (1, 2):
                    legend = ax.legend(
                        handles=legend_handles,
                        labels=compact_legend_labels,
                        loc="lower right",
                        bbox_to_anchor=(0.97, 0.03),
                        borderaxespad=0.0,
                        borderpad=0.25,
                        labelspacing=0.20,
                        handletextpad=0.40,
                        columnspacing=0.80,
                        ncol=legend_ncol,
                        frameon=True,
                        fontsize=legend_fontsize,
                    )
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    legend_bbox = legend.get_window_extent(renderer=renderer)
                    stats_bbox = stats_artist.get_bbox_patch().get_window_extent(
                        renderer=renderer
                    )

                    legend_reasons = _overlay_rejections(
                        legend_bbox,
                        points_display=points_display,
                        line_points_display=line_points_display,
                        axes_bbox=axes_bbox,
                        point_pad_px=marker_pad_px,
                    )
                    stats_reasons = _overlay_rejections(
                        stats_bbox,
                        points_display=points_display,
                        line_points_display=line_points_display,
                        axes_bbox=axes_bbox,
                        point_pad_px=marker_pad_px,
                    )
                    candidate_valid = _consider_candidate(
                        description=(
                            f"layout=split_corner font_tier={font_tier} "
                            f"stats_format={stats_format} "
                            f"headroom_frac={headroom_frac:.3f} "
                            f"right_padding_frac={right_frac:.3f} "
                            f"lower_padding_frac={lower_frac:.3f} "
                            f"legend_ncol={legend_ncol}"
                        ),
                        legend_bbox=legend_bbox,
                        stats_bbox=stats_bbox,
                        legend_reasons=legend_reasons,
                        stats_reasons=stats_reasons,
                        axes_bbox=axes_bbox,
                    )

                    if candidate_valid:
                        fig.canvas.draw()
                        renderer = fig.canvas.get_renderer()
                        final_legend_bbox = legend.get_window_extent(
                            renderer=renderer
                        )
                        final_stats_bbox = (
                            stats_artist.get_bbox_patch().get_window_extent(
                                renderer=renderer
                            )
                        )
                        final_valid = (
                            _overlay_is_clear(
                                final_legend_bbox,
                                points_display=points_display,
                                line_points_display=line_points_display,
                                axes_bbox=axes_bbox,
                                point_pad_px=marker_pad_px,
                            )
                            and _overlay_is_clear(
                                final_stats_bbox,
                                points_display=points_display,
                                line_points_display=line_points_display,
                                axes_bbox=axes_bbox,
                                point_pad_px=marker_pad_px,
                            )
                            and not _bbox_overlap(
                                final_legend_bbox, final_stats_bbox
                            )
                        )
                        if final_valid:
                            _log_correlation_layout(
                                f"title={ax.get_title()!r} "
                                f"mode=joint_internal "
                                f"layout=split_corner "
                                f"configured_fontsize="
                                f"{configured_font_size:.2f} "
                                f"axis_scale={axis_scale} "
                                f"font_tier={font_tier} "
                                f"stats_format={stats_format} "
                                f"headroom_frac={headroom_frac:.3f} "
                                f"right_padding_frac={right_frac:.3f} "
                                f"lower_padding_frac={lower_frac:.3f} "
                                f"legend_ncol={legend_ncol} "
                                f"legend_fontsize={legend_fontsize:.2f} "
                                f"stats_fontsize={stats_fontsize:.2f} "
                                f"rejections="
                                f"{_format_rejection_summary()}"
                            )
                            return legend, stats_artist

                    legend.remove()

                stats_artist.remove()

    # When independent corner placements cannot coexist, reserve a top band
    # and stack a compact stats box above a two-column legend. The band can use
    # more headroom than the general search because that space is occupied by
    # annotations rather than left as unexplained empty padding.
    band_max_headroom_frac = max(
        float(max_headroom_frac), float(annotation_band_max_headroom_frac)
    )
    band_headroom_fracs = tuple(np.linspace(0.0, band_max_headroom_frac, 16))
    band_stats_variants = list(reversed(stats_text_variants))
    band_gap_points = 6.0

    for font_tier, (legend_fontsize, stats_fontsize) in enumerate(font_tiers):
        for stats_format, candidate_stats_text in band_stats_variants:
            for headroom_frac in band_headroom_fracs:
                candidate_top = base_y1 + headroom_frac * base_y_span
                ax.set_xlim(base_x0, base_x1)
                ax.set_ylim(base_y0, candidate_top)
                fig.canvas.draw()

                renderer = fig.canvas.get_renderer()
                axes_bbox = ax.get_window_extent(renderer=renderer)
                if x_f.size:
                    points_display = ax.transData.transform(
                        np.column_stack([x_f, y_f])
                    )
                else:
                    points_display = np.empty((0, 2), dtype=float)
                line_points_display = _line_samples_display()

                stats_artist = ax.text(
                    0.5,
                    0.97,
                    candidate_stats_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=stats_fontsize,
                    linespacing=1.0,
                    zorder=5,
                    bbox={
                        **BBOX_STYLE,
                        "boxstyle": "round,pad=0.15",
                    },
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                stats_bbox = stats_artist.get_bbox_patch().get_window_extent(
                    renderer=renderer
                )
                stats_axes_points = ax.transAxes.inverted().transform(
                    np.array(
                        [
                            [stats_bbox.x0, stats_bbox.y0],
                            [stats_bbox.x1, stats_bbox.y1],
                        ]
                    )
                )
                stats_bottom_axes = float(np.min(stats_axes_points[:, 1]))
                gap_axes = (
                    band_gap_points * fig.dpi / 72.0 / max(axes_bbox.height, 1.0)
                )

                legend = ax.legend(
                    handles=legend_handles,
                    labels=compact_legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, stats_bottom_axes - gap_axes),
                    borderaxespad=0.0,
                    borderpad=0.25,
                    labelspacing=0.20,
                    handletextpad=0.40,
                    columnspacing=0.80,
                    ncol=2,
                    frameon=True,
                    fontsize=legend_fontsize,
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                stats_bbox = stats_artist.get_bbox_patch().get_window_extent(
                    renderer=renderer
                )

                legend_reasons = _overlay_rejections(
                    legend_bbox,
                    points_display=points_display,
                    line_points_display=line_points_display,
                    axes_bbox=axes_bbox,
                    point_pad_px=marker_pad_px,
                )
                stats_reasons = _overlay_rejections(
                    stats_bbox,
                    points_display=points_display,
                    line_points_display=line_points_display,
                    axes_bbox=axes_bbox,
                    point_pad_px=marker_pad_px,
                )
                candidate_valid = _consider_candidate(
                    description=(
                        f"layout=annotation_band font_tier={font_tier} "
                        f"stats_format={stats_format} "
                        f"headroom_frac={headroom_frac:.3f} legend_ncol=2"
                    ),
                    legend_bbox=legend_bbox,
                    stats_bbox=stats_bbox,
                    legend_reasons=legend_reasons,
                    stats_reasons=stats_reasons,
                    axes_bbox=axes_bbox,
                )

                if candidate_valid:
                    # Repeat the measurements once after the final draw, just
                    # as in the standard-placement path.
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    final_legend_bbox = legend.get_window_extent(renderer=renderer)
                    final_stats_bbox = (
                        stats_artist.get_bbox_patch().get_window_extent(
                            renderer=renderer
                        )
                    )
                    final_valid = (
                        _overlay_is_clear(
                            final_legend_bbox,
                            points_display=points_display,
                            line_points_display=line_points_display,
                            axes_bbox=axes_bbox,
                            point_pad_px=marker_pad_px,
                        )
                        and _overlay_is_clear(
                            final_stats_bbox,
                            points_display=points_display,
                            line_points_display=line_points_display,
                            axes_bbox=axes_bbox,
                            point_pad_px=marker_pad_px,
                        )
                        and not _bbox_overlap(
                            final_legend_bbox, final_stats_bbox
                        )
                    )
                    if final_valid:
                        _log_correlation_layout(
                            f"title={ax.get_title()!r} "
                            f"mode=joint_internal "
                            f"layout=annotation_band "
                            f"configured_fontsize={configured_font_size:.2f} "
                            f"axis_scale={axis_scale} "
                            f"font_tier={font_tier} "
                            f"stats_format={stats_format} "
                            f"headroom_frac={headroom_frac:.3f} "
                            f"legend_ncol=2 "
                            f"legend_fontsize={legend_fontsize:.2f} "
                            f"stats_fontsize={stats_fontsize:.2f} "
                            f"rejections={_format_rejection_summary()}"
                        )
                        return legend, stats_artist

                legend.remove()
                stats_artist.remove()

    # No collision-free internal layout exists within the allowed headroom.
    # Restore the original data range and place both overlays outside the
    # right side of the axes. bbox_inches="tight" will preserve them.
    ax.set_xlim(base_x0, base_x1)
    ax.set_ylim(base_y0, base_y1)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fontsize=legend_font_sizes[0],
    )
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    legend_axes_points = ax.transAxes.inverted().transform(
        np.array([[legend_bbox.x0, legend_bbox.y0], [legend_bbox.x1, legend_bbox.y1]])
    )
    legend_bottom_axes = float(np.min(legend_axes_points[:, 1]))

    stats_artist = ax.text(
        1.02,
        legend_bottom_axes - 0.04,
        stats_text_variants[-1][1],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=stats_font_sizes[0],
        zorder=5,
        clip_on=False,
        bbox=BBOX_STYLE,
    )
    fig.canvas.draw()

    _log_correlation_layout(
        f"title={ax.get_title()!r} "
        f"mode=joint_external "
        f"configured_fontsize={configured_font_size:.2f} "
        f"axis_scale={axis_scale} "
        f"stats_format={stats_text_variants[-1][0]} "
        f"headroom_frac=0 "
        f"legend_fontsize={legend_font_sizes[0]:.2f} "
        f"stats_fontsize={stats_font_sizes[0]:.2f} "
        f"rejections={_format_rejection_summary()} "
        f"closest_candidate={_format_closest_candidate()}"
    )

    return legend, stats_artist


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
    top_color: str = correlation_plot_color("selected_top"),
    bottom_color: str = correlation_plot_color("selected_bottom"),
    other_color: str = correlation_plot_color("selected_other", fallback=NEUTRAL_MID),
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
            lines.append(_format_corr_annotation(r_a, p_a, n_a, label="All (finite)"))
        else:
            lines.append("All (finite): r = n/a")

    if mode in ("both", "top"):
        if corr_top is not None:
            r_t, p_t, n_t = corr_top
            lines.append(_format_corr_annotation(r_t, p_t, n_t, label=top_label))
        else:
            lines.append(f"{top_label}: r = n/a")

    if mode in ("both", "bottom"):
        if corr_bottom is not None:
            r_b, p_b, n_b = corr_bottom
            lines.append(_format_corr_annotation(r_b, p_b, n_b, label=bottom_label))
        else:
            lines.append(f"{bottom_label}: r = n/a")

    if corr_all is not None:
        _r_a, p_a, _n_a = corr_all
        _add_significant_trend_line(ax, x_f, y_f, p_a, color=NEUTRAL_MID)
    if corr_top is not None:
        _r_t, p_t, _n_t = corr_top
        top_mask = classes_arr == "top"
        _add_significant_trend_line(
            ax,
            x_f[top_mask],
            y_f[top_mask],
            p_t,
            color=top_color,
        )
    if corr_bottom is not None:
        _r_b, p_b, _n_b = corr_bottom
        bottom_mask = classes_arr == "bottom"
        _add_significant_trend_line(
            ax,
            x_f[bottom_mask],
            y_f[bottom_mask],
            p_b,
            color=bottom_color,
        )

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

    _finalize_correlation_layout(fig, customizer, rect=(0, 0, 1, 0.98))
    out_path = _correlation_out_path(out_dir, filename, image_format)
    writeImage(str(out_path), format=image_format)
    plt.close(fig)


def plot_correlation_scatter(
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
    """Plot and export one generic Pearson correlation scatter.

    The inputs need only be aligned one-dimensional numeric vectors; unlike
    :func:`plot_cross_fly_correlations`, no ``VideoAnalysis`` objects are needed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation x and y must be aligned one-dimensional arrays")
    summary = pearson_correlation_summary(x, y)
    mask = np.isfinite(x) & np.isfinite(y)
    if summary.n < 3 or not np.isfinite(summary.r) or not np.isfinite(summary.p):
        print(f"[correlations] WARNING: not enough valid data for {filename}")
        return summary

    x_f = x[mask]
    y_f = y[mask]

    r, p = summary.r, summary.p

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

    _add_significant_trend_line(ax, x_f, y_f, p, color=cfg.dot_color)
    _add_smart_stats_box(ax, _format_corr_annotation(r, p, x_f.size), x_f, y_f)

    _finalize_correlation_layout(fig, customizer, axis_size_inches=cfg.axis_size_inches)
    out_path = _correlation_out_path(cfg.out_dir, filename, cfg.image_format)
    writeImage(str(out_path), format=cfg.image_format)
    _export_scatter_npz(
        x=x_f,
        y=y_f,
        title=title,
        x_label=x_label,
        y_label=y_label,
        filename=filename,
        cfg=cfg,
        r=float(r),
        p=float(p),
    )
    plt.close(fig)
    return summary


def _scatter_with_corr(**kwargs):
    """Backward-compatible private alias for the public scatter helper."""
    return plot_correlation_scatter(**kwargs)


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


def _window_aggregation_mode(opts) -> str:
    mode = str(
        getattr(opts, "corr_window_metric_aggregation", "pooled") or "pooled"
    ).lower()
    if mode not in ("pooled", "bucketwise"):
        raise ValueError(f"Unsupported correlation window aggregation: {mode!r}")
    return mode


def _context_bucket_window(ctx: SLIContext) -> tuple[int, int]:
    if ctx.explicit_bucket_idx is not None:
        return max(0, int(ctx.explicit_bucket_idx)), 1
    return (
        max(0, int(ctx.skip_first_sync_buckets or 0)),
        max(0, int(ctx.keep_first_sync_buckets or 0)),
    )


def _pooled_rewards_per_distance_for_context(
    va,
    *,
    ctx: SLIContext,
    f: int,
    validity_policy: str = "window",
    min_rewards: int = 5,
) -> float:
    training_idx = int(ctx.training_idx)
    trns = getattr(va, "trns", None) or []
    if training_idx < 0 or training_idx >= len(trns):
        return np.nan

    skip_first, keep_first = _context_bucket_window(ctx)
    result = pooled_rewards_per_distance_window(
        va,
        trns[training_idx],
        t_idx=training_idx,
        f=f,
        skip_first=skip_first,
        keep_first=keep_first,
        validity_policy=validity_policy,
        min_rewards=min_rewards,
    )
    return np.nan if result is None else float(result.value)


def _extract_exp_speed_for_context(
    vas: Sequence,
    opts,
    ctx: SLIContext,
    *,
    aggregation: str = "pooled",
) -> np.ndarray:
    """
    Return one experimental-fly speed scalar per VideoAnalysis for ctx's
    training/window, in mm/s.
    """
    try:
        speed_arrays = _extract_speed_arrays(vas, opts)
    except Exception as e:
        print(f"[correlations] WARNING: failed to compute speed arrays: {e}")
        return np.full(len(vas), np.nan, dtype=float)

    return _reduce_exp_speed_for_context(
        speed_arrays,
        n_vas=len(vas),
        ctx=ctx,
        aggregation=aggregation,
    )


def _reduce_exp_speed_for_context(
    speed_arrays: dict,
    *,
    n_vas: int,
    ctx: SLIContext,
    aggregation: str = "pooled",
) -> np.ndarray:
    """Reduce precomputed experimental-fly speed arrays for one context."""

    speed_exp = np.asarray(speed_arrays.get("speed_exp", []), dtype=float)
    if speed_exp.ndim != 3 or speed_exp.shape[0] != n_vas:
        print(
            "[correlations] WARNING: speed_exp array has unexpected shape; "
            "skipping speed vs SLI correlation"
        )
        return np.full(n_vas, np.nan, dtype=float)

    training_idx = int(getattr(ctx, "training_idx", 0) or 0)
    if training_idx < 0 or training_idx >= speed_exp.shape[1]:
        return np.full(n_vas, np.nan, dtype=float)

    if aggregation == "bucketwise":
        return np.asarray(
            [
                _reduce_sync_bucket_series(
                    speed_exp[vi, training_idx, :],
                    bucket_idx=getattr(ctx, "explicit_bucket_idx", None),
                    average_over_buckets=bool(ctx.average_over_buckets),
                    skip_first_sync_buckets=int(ctx.skip_first_sync_buckets or 0),
                    keep_first_sync_buckets=int(ctx.keep_first_sync_buckets or 0),
                )
                for vi in range(speed_exp.shape[0])
            ],
            dtype=float,
        )

    speed_n = np.asarray(speed_arrays.get("speedN_exp", []), dtype=float)
    if speed_n.shape != speed_exp.shape:
        return np.full(n_vas, np.nan, dtype=float)
    skip_first, keep_first = _context_bucket_window(ctx)
    end = speed_exp.shape[2]
    if keep_first > 0:
        end = min(end, skip_first + keep_first)
    if skip_first >= end:
        return np.full(n_vas, np.nan, dtype=float)

    values = speed_exp[:, training_idx, skip_first:end]
    weights = speed_n[:, training_idx, skip_first:end]
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    weighted_sum = np.sum(np.where(valid, values * weights, 0.0), axis=1)
    total_weight = np.sum(np.where(valid, weights, 0.0), axis=1)
    return np.divide(
        weighted_sum,
        total_weight,
        out=np.full(weighted_sum.shape, np.nan, dtype=float),
        where=total_weight > 0,
    )


def _extract_exp_pre_training_speed(vas: Sequence) -> np.ndarray:
    """Extract experimental-fly speed from the final 10 minutes before T1.

    ``VideoAnalysis.speed()`` stores one ``(pre, training)`` pair for each fly
    in each training.  The first training's experimental-fly row therefore
    contains the experiment-wide pre-training speed at index 0.
    """
    values = []
    for va in vas:
        value = np.nan
        try:
            flies = list(getattr(va, "flies", (0,)))
            exp_row_idx = flies.index(0)
            speed_rows = getattr(va, "speed", ())
            exp_t1 = np.asarray(speed_rows[exp_row_idx], dtype=float).reshape(-1)
            if exp_t1.size:
                value = float(exp_t1[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def _extract_exp_full_pre_training_speed(vas: Sequence) -> np.ndarray:
    """Calculate experimental-fly mean bottom speed over the entire T1 pre-period.

    This uses the same speed samples and bottom-of-chamber mask as
    ``VideoAnalysis.speed()``, but starts at ``startPre`` instead of retaining
    only its final 10-minute summary.  As in that method, a pre-period pulse (if
    present) marks the end of the speed window; otherwise T1 start is used.
    """
    values = []
    for va in vas:
        value = np.nan
        try:
            if 0 not in tuple(getattr(va, "flies", ())):
                values.append(value)
                continue
            start = int(va.startPre)
            stop = int(va.trns[0].start)
            pre_pulses = np.asarray(va.on, dtype=int)
            pre_pulses = pre_pulses[(pre_pulses >= start) & (pre_pulses < stop)]
            if pre_pulses.size:
                stop = int(pre_pulses[-1])

            traj = va.trx[0]
            speed = np.asarray(traj.sp[start:stop], dtype=float)
            on_bottom = np.asarray(traj.onBottomPre[start:stop], dtype=bool)
            speed = speed[on_bottom] / float(traj.pxPerMmFloor)
            if speed.size >= 100:
                value = float(np.mean(speed))
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def _pooled_median_distance_for_context(va, ctx: SLIContext) -> float:
    training_idx = int(ctx.training_idx)
    trns = getattr(va, "trns", None) or []
    trx = getattr(va, "trx", None) or []
    if training_idx < 0 or training_idx >= len(trns) or not trx:
        return np.nan
    try:
        if trx[0].bad():
            return np.nan
    except Exception:
        if getattr(trx[0], "_bad", True):
            return np.nan

    skip_first, keep_first = _context_bucket_window(ctx)
    trn = trns[training_idx]
    fi, df, n_buckets, _complete = sync_bucket_window(
        va,
        trn,
        t_idx=training_idx,
        f=0,
        skip_first=skip_first,
        keep_first=keep_first,
        use_exclusion_mask=False,
    )
    if n_buckets <= 0:
        return np.nan

    try:
        cx, cy, _ = trn.circles(0)[0]
        px_per_mm = float(va.xf.fctr) * float(va.ct.pxPerMmFloor())
    except Exception:
        return np.nan
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        return np.nan

    traj = trx[0]
    distances = []
    for j in range(int(n_buckets)):
        bucket_idx = skip_first + j
        try:
            if va.is_excluded_pair(0, training_idx, bucket_idx):
                continue
        except Exception:
            return np.nan
        start = int(fi + j * df)
        stop = int(start + df)
        xs = np.asarray(traj.x[start:stop], dtype=float)
        ys = np.asarray(traj.y[start:stop], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.any(valid):
            distances.append(np.hypot(xs[valid] - cx, ys[valid] - cy))
    if not distances:
        return np.nan
    return float(np.median(np.concatenate(distances)) / px_per_mm)


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
    fast_corr_mask = (classes_arr == "fast") | (classes_arr == "overlap")
    strong_corr_mask = (classes_arr == "strong") | (classes_arr == "overlap")

    # Correlation across *all* plotted points (finite x/y only)
    corr_all = None
    n_all = int(x_f.size)
    if n_all >= 3:
        r_a, p_a = pearsonr(x_f, y_f)
        corr_all = (float(r_a), float(p_a), n_all)

    # Colors (simple, can be refined)
    color_map = {
        "overlap": correlation_plot_color("fast_vs_strong_overlap"),
        "fast": correlation_plot_color("fast_vs_strong_fast"),
        "strong": correlation_plot_color("fast_vs_strong_strong"),
        "other": correlation_plot_color("fast_vs_strong_other", fallback=NEUTRAL_MID),
    }

    point_colors = [color_map[c] for c in classes]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scatter_artist = ax.scatter(x_f, y_f, c=point_colors, alpha=0.85)

    ax.set_xlabel(x_label)
    ax.set_ylabel(strong_y_label)
    ax.set_title("Initial SLI and later SLI", pad=10)

    # Display descriptive correlations (fast/strong each including overlap)
    lines = []
    compact_lines = []
    if corr_all is not None:
        r_a, p_a, n_a = corr_all
        lines.append(_format_labeled_corr_with_n(r_a, p_a, n_a, label="All flies"))
        compact_lines.append(
            _format_compact_labeled_corr_with_n(r_a, p_a, n_a, label="All")
        )
    else:
        lines.append(_format_labeled_corr_na_with_n(n_all, label="All flies"))
        compact_lines.append(
            _format_compact_labeled_corr_na_with_n(n_all, label="All")
        )
    if corr_fast_incl_overlap is not None:
        r_f, p_f, n_f = corr_fast_incl_overlap
        lines.append(_format_labeled_corr_with_n(r_f, p_f, n_f, label="Fast learners"))
        compact_lines.append(
            _format_compact_labeled_corr_with_n(r_f, p_f, n_f, label="Fast")
        )
    else:
        lines.append(
            _format_labeled_corr_na_with_n(
                int(np.sum(fast_corr_mask)), label="Fast learners"
            )
        )
        compact_lines.append(
            _format_compact_labeled_corr_na_with_n(
                int(np.sum(fast_corr_mask)), label="Fast"
            )
        )

    if corr_strong_incl_overlap is not None:
        r_s, p_s, n_s = corr_strong_incl_overlap
        lines.append(
            _format_labeled_corr_with_n(r_s, p_s, n_s, label="Strong learners")
        )
        compact_lines.append(
            _format_compact_labeled_corr_with_n(r_s, p_s, n_s, label="Strong")
        )
    else:
        lines.append(
            _format_labeled_corr_na_with_n(
                int(np.sum(strong_corr_mask)), label="Strong learners"
            )
        )
        compact_lines.append(
            _format_compact_labeled_corr_na_with_n(
                int(np.sum(strong_corr_mask)), label="Strong"
            )
        )

    if corr_all is not None:
        _r_a, p_a, _n_a = corr_all
        _add_significant_trend_line(ax, x_f, y_f, p_a, color=NEUTRAL_MID)
    if corr_fast_incl_overlap is not None:
        _r_f, p_f, _n_f = corr_fast_incl_overlap
        _add_significant_trend_line(
            ax,
            x_f[fast_corr_mask],
            y_f[fast_corr_mask],
            p_f,
            color=color_map["fast"],
        )
    if corr_strong_incl_overlap is not None:
        _r_s, p_s, _n_s = corr_strong_incl_overlap
        _add_significant_trend_line(
            ax,
            x_f[strong_corr_mask],
            y_f[strong_corr_mask],
            p_s,
            color=color_map["strong"],
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
            label="Fast learners only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["strong"],
            markersize=8,
            label="Strong learners only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["overlap"],
            markersize=8,
            label="Fast + strong",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["other"],
            markersize=8,
            label="Other flies",
        ),
    ]
    # Finish all figure and axes resizing before measuring overlay geometry.
    base_axis_size = getattr(
        customizer,
        "standard_plot_axis_size",
        DEFAULT_PLOT_AXIS_SIZE_INCHES,
    )
    scaled_axis_size = _correlation_axis_size_for_font(
        customizer,
        base_size=base_axis_size,
    )
    axis_scale = float(scaled_axis_size[0]) / float(base_axis_size[0])
    _finalize_correlation_layout(
        fig,
        customizer,
        rect=(0, 0, 1, 0.96),
        axis_size_inches=scaled_axis_size,
    )

    _place_correlation_overlays(
        ax,
        handles,
        "\n".join(lines),
        x_f,
        y_f,
        scatter_artist=scatter_artist,
        compact_stats_text="\n".join(compact_lines),
        compact_legend_labels=(
            "Fast only",
            "Strong only",
            "Fast + strong",
            "Other",
        ),
        compact_labels_min_font_size=24.0,
        configured_font_size=float(customizer.font_size),
        axis_scale=axis_scale,
        max_headroom_frac=0.50,
        split_corner_max_right_frac=0.20,
        split_corner_max_lower_frac=0.20,
        annotation_band_max_headroom_frac=0.90,
    )
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
        "fast": correlation_plot_color("selected_top"),
        "slow": correlation_plot_color("selected_bottom"),
        "other": correlation_plot_color("selected_other", fallback=NEUTRAL_LIGHT),
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

    classes_arr = np.asarray(classes, dtype=object)

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
        r_f, p_f, n_f = corr_fast
        lines.append(_format_corr_annotation(r_f, p_f, n_f, label="Fast"))
    else:
        lines.append("Fast:  r = n/a")

    if corr_slow is not None:
        r_s, p_s, n_s = corr_slow
        lines.append(_format_corr_annotation(r_s, p_s, n_s, label="Slow"))
    else:
        lines.append("Slow:  r = n/a")

    if corr_fast is not None:
        _r_f, p_f, _n_f = corr_fast
        fast_mask = classes_arr == "fast"
        _add_significant_trend_line(
            ax,
            x_f[fast_mask],
            y_f[fast_mask],
            p_f,
            color=color_map["fast"],
        )
    if corr_slow is not None:
        _r_s, p_s, _n_s = corr_slow
        slow_mask = classes_arr == "slow"
        _add_significant_trend_line(
            ax,
            x_f[slow_mask],
            y_f[slow_mask],
            p_s,
            color=color_map["slow"],
        )

    _add_smart_stats_box(ax, "\n".join(lines), x_f, y_f)

    _finalize_correlation_layout(fig, customizer)

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
    sli_t2_sb5_values: Sequence[float] | None = None,
    sli_t1_sb2_sb5_mean_values: Sequence[float] | None = None,
    sli_t2_sb2_sb5_mean_values: Sequence[float] | None = None,
    sli_selected: tuple[Sequence[int], Sequence[int]] | None = None,
    sli_extremes: str | None = None,
):
    """
    Cross-fly correlations:

      1) SLI vs reward-per-distance over the selected training/window
      1b) SLI vs experimental-minus-yoked reward-per-distance
      1c) SLI vs reward rate
      1d) Speed at T2 SB5 vs SLI at T2 SB5
      1e) Mean speed over T2 SB1-SB5 vs SLI at T2 SB5
      1f) Final-10-min pre-training speed vs mean SLI over T2 SB2-SB5
      1g) Full-pre-period speed vs SLI at T2 SB5
      1h) Mean SLI over T1 SB2-SB5 vs mean SLI over T2 SB2-SB5
      2) SLI vs median distance to reward over the selected training/window
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

    window_metric_aggregation = _window_aggregation_mode(opts)
    rpd_pooled_validity = str(
        getattr(opts, "rpd_pooled_validity", "window") or "window"
    )
    rpd_pooled_min_rewards = max(
        0,
        int(getattr(opts, "rpd_pooled_min_rewards", 5) or 0),
    )
    cfg = CorrelationPlotConfig(
        out_dir=out_dir,
        image_format=getattr(opts, "imageFormat", "png"),
        xlim=getattr(opts, "corr_xlim", None),
        ylim=getattr(opts, "corr_ylim", None),
        export_npz_dir=(
            None
            if getattr(opts, "corr_export_npz_dir", None) is None
            else Path(getattr(opts, "corr_export_npz_dir"))
        ),
        export_group_label=_corr_export_group_label(opts),
        window_metric_aggregation=window_metric_aggregation,
        rpd_pooled_validity=rpd_pooled_validity,
        rpd_pooled_min_rewards=rpd_pooled_min_rewards,
        axis_size_inches=getattr(
            opts, "standard_plot_axis_size", DEFAULT_PLOT_AXIS_SIZE_INCHES
        ),
    )
    frac = getattr(opts, "best_worst_fraction", 0.2)
    customizer = plot_customizer or PlotCustomizer()
    customizer.standard_plot_axis_size = cfg.axis_size_inches

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

    sli_t2_sb5_vals = None
    if sli_t2_sb5_values is not None:
        sli_t2_sb5_vals = np.asarray(sli_t2_sb5_values, float)
        if sli_t2_sb5_vals.shape[0] != len(vas):
            print(
                "[correlations] WARNING: len(sli_t2_sb5_values) != len(vas) "
                f"({sli_t2_sb5_vals.shape[0]} vs {len(vas)}); "
                "skipping fixed T2 speed vs final-SLI plots"
            )
            sli_t2_sb5_vals = None

    sli_t1_sb2_sb5_mean_vals = None
    if sli_t1_sb2_sb5_mean_values is not None:
        sli_t1_sb2_sb5_mean_vals = np.asarray(
            sli_t1_sb2_sb5_mean_values, dtype=float
        )
        if sli_t1_sb2_sb5_mean_vals.shape != (len(vas),):
            print(
                "[correlations] WARNING: sli_t1_sb2_sb5_mean_values must contain "
                f"one value per VideoAnalysis ({len(vas)} expected); skipping "
                "fixed T1-vs-T2 mean-SLI plot"
            )
            sli_t1_sb2_sb5_mean_vals = None

    sli_t2_sb2_sb5_mean_vals = None
    if sli_t2_sb2_sb5_mean_values is not None:
        sli_t2_sb2_sb5_mean_vals = np.asarray(sli_t2_sb2_sb5_mean_values, dtype=float)
        if sli_t2_sb2_sb5_mean_vals.shape != (len(vas),):
            print(
                "[correlations] WARNING: sli_t2_sb2_sb5_mean_values must contain "
                f"one value per VideoAnalysis ({len(vas)} expected); skipping "
                "fixed plots requiring mean T2 SB2-SB5 SLI"
            )
            sli_t2_sb2_sb5_mean_vals = None

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

    x_label_sli = sli_ctx.axis_label()
    y_label_sli = sli_ctx.axis_label()
    corr_sli_vs_rpt_xlabel_override = getattr(opts, "corr_sli_vs_rpt_xlabel", None)
    corr_sli_vs_rpt_ylabel_override = getattr(opts, "corr_sli_vs_rpt_ylabel", None)
    corr_pre_reward_pi_vs_sli_xlabel_override = getattr(
        opts, "corr_pre_reward_pi_vs_sli_xlabel", None
    )
    corr_pre_reward_pi_vs_sli_ylabel_override = getattr(
        opts, "corr_pre_reward_pi_vs_sli_ylabel", None
    )
    corr_pre_floor_exploration_vs_sli_xlabel_override = getattr(
        opts, "corr_pre_floor_exploration_vs_sli_xlabel", None
    )
    corr_pre_floor_exploration_vs_sli_ylabel_override = getattr(
        opts, "corr_pre_floor_exploration_vs_sli_ylabel", None
    )
    corr_fast_vs_strong_xlabel_override = getattr(
        opts, "corr_fast_vs_strong_xlabel", None
    )
    corr_fast_vs_strong_ylabel_override = getattr(
        opts, "corr_fast_vs_strong_ylabel", None
    )

    skip_k = int(getattr(sli_ctx, "skip_first_sync_buckets", 0) or 0)
    skip_k = max(0, skip_k)
    keep_k = int(getattr(sli_ctx, "keep_first_sync_buckets", 0) or 0)
    keep_k = max(0, keep_k)
    sli_bucket_idx = getattr(sli_ctx, "explicit_bucket_idx", None)
    reward_training_idx = int(
        getattr(reward_rate_ctx, "training_idx", training_idx) or 0
    )
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
    rpd_exp_minus_yoked_vals = []
    rpt_vals = []
    med_train_vals = []
    pre_pi_diff_vals = []
    total_reward_vals = []
    pre_coverage_vals = []

    for va in vas:
        # --- Reward per distance over the selected training/window ---
        if window_metric_aggregation == "pooled":
            rpd_val = _pooled_rewards_per_distance_for_context(
                va,
                ctx=sli_ctx,
                f=0,
                validity_policy=rpd_pooled_validity,
                min_rewards=rpd_pooled_min_rewards,
            )
            rpd_yoked_val = _pooled_rewards_per_distance_for_context(
                va,
                ctx=sli_ctx,
                f=1,
                validity_policy=rpd_pooled_validity,
                min_rewards=rpd_pooled_min_rewards,
            )
        elif _ensure_rewards_per_distance(va):
            exp_row_idx = 2 * training_idx
            yoked_row_idx = exp_row_idx + 1
            if 0 <= exp_row_idx < len(va.rwdsPerDist):
                exp_row = va.rwdsPerDist[exp_row_idx]
                rpd_val = _reduce_sync_bucket_series(
                    exp_row,
                    bucket_idx=sli_bucket_idx,
                    average_over_buckets=bool(sli_ctx.average_over_buckets),
                    skip_first_sync_buckets=skip_k,
                    keep_first_sync_buckets=keep_k,
                )
            else:
                rpd_val = np.nan
            if 0 <= yoked_row_idx < len(va.rwdsPerDist):
                yoked_row = va.rwdsPerDist[yoked_row_idx]
                rpd_yoked_val = _reduce_sync_bucket_series(
                    yoked_row,
                    bucket_idx=sli_bucket_idx,
                    average_over_buckets=bool(sli_ctx.average_over_buckets),
                    skip_first_sync_buckets=skip_k,
                    keep_first_sync_buckets=keep_k,
                )
            else:
                rpd_yoked_val = np.nan
        else:
            rpd_val = np.nan
            rpd_yoked_val = np.nan
        rpd_exp_minus_yoked_val = rpd_val - rpd_yoked_val

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

        # --- Median distance to reward during the selected training/window ---
        if window_metric_aggregation == "pooled":
            med_train = _pooled_median_distance_for_context(va, sli_ctx)
        else:
            _ensure_sync_med_dist(va)
            if hasattr(va, "syncMedDist") and training_idx < len(va.syncMedDist):
                med_vec = np.asarray(
                    va.syncMedDist[training_idx].get("exp", []),
                    float,
                )
                end_k = (
                    med_vec.size if keep_k <= 0 else min(med_vec.size, skip_k + keep_k)
                )
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
        rpd_exp_minus_yoked_vals.append(rpd_exp_minus_yoked_val)
        rpt_vals.append(rpt_val)
        med_train_vals.append(med_train)
        pre_pi_diff_vals.append(pre_diff)
        total_reward_vals.append(total_rewards)
        pre_coverage_vals.append(coverage)

    rpd_vals = np.asarray(rpd_vals, float)
    rpd_exp_minus_yoked_vals = np.asarray(rpd_exp_minus_yoked_vals, float)
    rpt_vals = np.asarray(rpt_vals, float)
    try:
        speed_arrays = _extract_speed_arrays(vas, opts)
    except Exception as e:
        print(f"[correlations] WARNING: failed to compute speed arrays: {e}")
        speed_arrays = {}
    med_train_vals = np.asarray(med_train_vals, float)
    pre_pi_diff_vals = np.asarray(pre_pi_diff_vals, float)
    total_reward_vals = np.asarray(total_reward_vals, float)
    pre_coverage_vals = np.asarray(pre_coverage_vals, float)

    cohort_debug_dir = getattr(opts, "dump_sli_cohorts", None)
    if cohort_debug_dir:
        # This is the same pairwise finite mask used by
        # pearson_correlation_summary() for plot 1 below.
        write_sorted_fly_list(
            Path(cohort_debug_dir) / "rpd_vs_sli_correlation_flies.txt",
            np.isfinite(sli_vals) & np.isfinite(rpd_vals),
            vas,
        )

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

    rpd_y_label = sli_ctx.metric_axis_label(
        "Rewards per distance", unit="$\\mathrm{m}^{-1}$"
    )
    rpd_diff_y_label = (
        sli_ctx.metric_axis_label(
            "Rewards per distance, exp - yok",
            unit="$\\mathrm{m}^{-1}$",
        )
        .replace(
            ", mean over ",
            ",\nmean over ",
        )
        .replace(
            ", at ",
            ",\nat ",
        )
    )
    if reward_first_n > 0:
        rpt_y_label = _first_n_reward_rate_label(
            first_n_rewards=reward_first_n,
            ctx=reward_rate_ctx,
            max_time_to_nth_s=reward_max_time_to_nth_s,
            time_basis=reward_first_n_time_basis,
        )
    else:
        rpt_y_label = reward_rate_ctx.metric_axis_label(
            "Reward rate",
            unit="$\\mathrm{min}^{-1}$",
        )
    pre_period_exploration_title = "Pre-period exploration and SLI"
    pre_period_exploration_xlabel = (
        "Fraction of floor explored during pre period (exp fly)"
    )

    # --- Plot 1: SLI_final vs reward-per-distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpd_vals,
        title="Rewards per distance vs SLI",
        x_label=x_label_sli,
        y_label=rpd_y_label,
        cfg=_cfg_with_plot_color(cfg, "rewards_per_distance_vs_sli"),
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

    # --- Plot 1b: SLI_final vs exp-minus-yoked reward-per-distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpd_exp_minus_yoked_vals,
        title="Δ rewards per distance vs SLI",
        x_label=x_label_sli,
        y_label=rpd_diff_y_label,
        cfg=_cfg_with_plot_color(
            cfg,
            "rewards_per_distance_exp_minus_yoked_vs_sli",
        ),
        filename=f"corr_rpd_exp_minus_yoked_vs_sli_{rpd_suffix}",
        customizer=customizer,
    )
    if selected_mode is not None:
        if selected_mode == "top":
            title_1b_sel = (
                "Exp-minus-yoked rewards per distance vs SLI "
                "(top SLI-selected learners)"
            )
            filename_1b_sel = (
                f"corr_rpd_exp_minus_yoked_vs_sli_{rpd_suffix}_top_selected"
            )
        elif selected_mode == "bottom":
            title_1b_sel = (
                "Exp-minus-yoked rewards per distance vs SLI "
                "(bottom SLI-selected learners)"
            )
            filename_1b_sel = (
                f"corr_rpd_exp_minus_yoked_vs_sli_{rpd_suffix}_bottom_selected"
            )
        else:
            title_1b_sel = (
                "Exp-minus-yoked rewards per distance vs SLI "
                "(top vs bottom SLI-selected learners)"
            )
            filename_1b_sel = (
                f"corr_rpd_exp_minus_yoked_vs_sli_{rpd_suffix}_selected_extremes"
            )

        plot_selected_group_scatter(
            x=sli_vals,
            y=rpd_exp_minus_yoked_vals,
            bottom_idx=selected_bottom_idx,
            top_idx=selected_top_idx,
            mode=selected_mode,
            title=title_1b_sel,
            x_label=x_label_sli,
            y_label=rpd_diff_y_label,
            filename=filename_1b_sel,
            out_dir=out_dir,
            customizer=customizer,
            top_label=top_sel_label,
            bottom_label=bottom_sel_label,
            xlim=cfg.xlim,
            ylim=cfg.ylim,
            image_format=cfg.image_format,
        )

    # --- Plot 1c: SLI_final vs reward-per-time ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpt_vals,
        title="Reward rate vs SLI",
        x_label=str(corr_sli_vs_rpt_xlabel_override or x_label_sli),
        y_label=str(corr_sli_vs_rpt_ylabel_override or rpt_y_label),
        cfg=_cfg_with_plot_color(cfg, "rewards_per_minute_vs_sli"),
        filename=f"corr_rpt_vs_sli_{rpt_suffix}",
        customizer=customizer,
    )
    if selected_mode is not None:
        if selected_mode == "top":
            title_1c_sel = "Reward rate vs SLI (top SLI-selected learners)"
            filename_1c_sel = f"corr_rpt_vs_sli_{rpt_suffix}_top_selected"
        elif selected_mode == "bottom":
            title_1c_sel = "Reward rate vs SLI (bottom SLI-selected learners)"
            filename_1c_sel = f"corr_rpt_vs_sli_{rpt_suffix}_bottom_selected"
        else:
            title_1c_sel = "Reward rate vs SLI (top vs bottom SLI-selected learners)"
            filename_1c_sel = f"corr_rpt_vs_sli_{rpt_suffix}_selected_extremes"

        plot_selected_group_scatter(
            x=sli_vals,
            y=rpt_vals,
            bottom_idx=selected_bottom_idx,
            top_idx=selected_top_idx,
            mode=selected_mode,
            title=title_1c_sel,
            x_label=str(corr_sli_vs_rpt_xlabel_override or x_label_sli),
            y_label=str(corr_sli_vs_rpt_ylabel_override or rpt_y_label),
            filename=filename_1c_sel,
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

    # --- Plots 1d/1e: fixed T2 speed windows vs final SLI (T2 SB5) ---
    if sli_t2_sb5_vals is not None:
        final_sli_ctx, fixed_speed_plots = _default_t2_speed_vs_final_sli_contexts()
        for speed_ctx, title in fixed_speed_plots:
            fixed_speed_vals = _reduce_exp_speed_for_context(
                speed_arrays,
                n_vas=len(vas),
                ctx=speed_ctx,
                aggregation=window_metric_aggregation,
            )
            fixed_suffix = (
                f"{_window_context_suffix(final_sli_ctx, prefix='sli')}__"
                f"{_window_context_suffix(speed_ctx, prefix='speed')}"
            )
            fixed_speed_x_label = speed_ctx.metric_axis_label("Mean speed", unit="mm/s")
            final_sli_y_label = final_sli_ctx.axis_label()
            _scatter_with_corr(
                x=fixed_speed_vals,
                y=sli_t2_sb5_vals,
                title=title,
                x_label=fixed_speed_x_label,
                y_label=final_sli_y_label,
                cfg=_cfg_with_plot_color(cfg, "speed_vs_sli"),
                filename=f"corr_speed_vs_sli_{fixed_suffix}",
                customizer=customizer,
            )
            if selected_mode is not None:
                if selected_mode == "top":
                    selection_title = "top SLI-selected learners"
                    selection_suffix = "top_selected"
                elif selected_mode == "bottom":
                    selection_title = "bottom SLI-selected learners"
                    selection_suffix = "bottom_selected"
                else:
                    selection_title = "top vs bottom SLI-selected learners"
                    selection_suffix = "selected_extremes"

                speed_top_sel_label, speed_bottom_sel_label = (
                    _speed_selection_group_labels(
                        sli_ctx,
                        top_pct_txt=top_pct_txt,
                        bottom_pct_txt=bottom_pct_txt,
                    )
                )
                plot_selected_group_scatter(
                    x=fixed_speed_vals,
                    y=sli_t2_sb5_vals,
                    bottom_idx=selected_bottom_idx,
                    top_idx=selected_top_idx,
                    mode=selected_mode,
                    title=f"{title} ({selection_title})",
                    x_label=fixed_speed_x_label,
                    y_label=final_sli_y_label,
                    filename=(f"corr_speed_vs_sli_{fixed_suffix}_{selection_suffix}"),
                    out_dir=out_dir,
                    customizer=customizer,
                    top_label=speed_top_sel_label,
                    bottom_label=speed_bottom_sel_label,
                    xlim=cfg.xlim,
                    ylim=cfg.ylim,
                    include_all_corr=True,
                    image_format=cfg.image_format,
                )

    # --- Plot 1f: final-10-min pre-training speed vs mean T2 SB2-SB5 SLI ---
    if sli_t2_sb2_sb5_mean_vals is not None:
        mean_sli_ctx = _default_pre_training_speed_vs_mean_t2_sli_context()
        pre_training_speed_vals = _extract_exp_pre_training_speed(vas)
        mean_sli_suffix = _window_context_suffix(mean_sli_ctx, prefix="sli")
        _scatter_with_corr(
            x=pre_training_speed_vals,
            y=sli_t2_sb2_sb5_mean_vals,
            title="Pre-training speed and mean T2 SLI",
            x_label="Locomotion speed (mm/s)",
            y_label=mean_sli_ctx.axis_label(),
            cfg=_cfg_with_plot_color(cfg, "pre_training_speed_vs_mean_t2_sli"),
            filename=(
                "corr_pre_training_speed_vs_sli_"
                f"{mean_sli_suffix}__speed_preT1_last10min"
            ),
            customizer=customizer,
        )

    # --- Plot 1g: entire pre-training speed vs final T2 SB5 SLI ---
    if sli_t2_sb5_vals is not None:
        final_sli_ctx, _ = _default_t2_speed_vs_final_sli_contexts()
        full_pre_training_speed_vals = _extract_exp_full_pre_training_speed(vas)
        final_sli_suffix = _window_context_suffix(final_sli_ctx, prefix="sli")
        _scatter_with_corr(
            x=full_pre_training_speed_vals,
            y=sli_t2_sb5_vals,
            title="Full pre-training speed and final T2 SLI",
            x_label="Mean speed during entire pre-training period (mm/s)",
            y_label=final_sli_ctx.axis_label(),
            cfg=_cfg_with_plot_color(cfg, "speed_vs_sli"),
            filename=(
                "corr_pre_training_speed_vs_sli_"
                f"{final_sli_suffix}__speed_preT1_full"
            ),
            customizer=customizer,
        )

    # --- Plot 1h: mean T1 SB2-SB5 SLI vs mean T2 SB2-SB5 SLI ---
    if (
        sli_t1_sb2_sb5_mean_vals is not None
        and sli_t2_sb2_sb5_mean_vals is not None
    ):
        t1_mean_sli_ctx, t2_mean_sli_ctx = _default_t1_vs_t2_mean_sli_contexts()
        fixed_suffix = (
            f"{_window_context_suffix(t1_mean_sli_ctx, prefix='sli')}__"
            f"{_window_context_suffix(t2_mean_sli_ctx, prefix='sli')}"
        )
        _scatter_with_corr(
            x=sli_t1_sb2_sb5_mean_vals,
            y=sli_t2_sb2_sb5_mean_vals,
            title="Mean T1 SLI vs mean T2 SLI",
            x_label=t1_mean_sli_ctx.axis_label(),
            y_label=t2_mean_sli_ctx.axis_label(),
            cfg=_cfg_with_plot_color(cfg, "sli_vs_sli"),
            filename=f"corr_sli_vs_sli_{fixed_suffix}",
            customizer=customizer,
        )

    # --- Plot 2: SLI_final vs median training distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=med_train_vals,
        title="SLI vs median distance to reward",
        x_label=x_label_sli,
        y_label="Median distance during training (mm)",
        cfg=_cfg_with_plot_color(cfg, "median_distance_vs_sli"),
        filename="corr_sli_vs_median_training",
        customizer=customizer,
    )

    # --- Plot 3: Pre-period SLI vs SLI_final ---
    _scatter_with_corr(
        x=pre_pi_diff_vals,
        y=sli_vals,
        title="Pre-period SLI and SLI",
        x_label=str(corr_pre_reward_pi_vs_sli_xlabel_override or "Pre-period SLI"),
        y_label=str(corr_pre_reward_pi_vs_sli_ylabel_override or y_label_sli),
        cfg=_cfg_with_plot_color(cfg, "baseline_pi_vs_sli"),
        filename="corr_pre_reward_pi_vs_sli",
        customizer=customizer,
    )

    # --- Plot 3b: Pre-period exploration vs SLI at T1, first sync bucket ---
    if reward_pi_training_vals is not None:
        _scatter_with_corr(
            x=pre_coverage_vals,
            y=reward_pi_training_vals,
            title=pre_period_exploration_title,
            x_label=str(
                corr_pre_floor_exploration_vs_sli_xlabel_override
                or pre_period_exploration_xlabel
            ),
            y_label=str(corr_pre_floor_exploration_vs_sli_ylabel_override or early_lbl),
            cfg=_cfg_with_plot_color(cfg, "pre_training_exploration_vs_sli"),
            filename="corr_pre_floor_exploration_vs_sli_T1_first",
            customizer=customizer,
        )
        if selected_mode is not None:
            if selected_mode == "top":
                title_3b_sel = (
                    f"{pre_period_exploration_title} (top SLI-selected learners)"
                )
                filename_3b_sel = (
                    "corr_pre_floor_exploration_vs_sli_T1_first_top_selected"
                )
            elif selected_mode == "bottom":
                title_3b_sel = (
                    f"{pre_period_exploration_title} (bottom SLI-selected learners)"
                )
                filename_3b_sel = (
                    "corr_pre_floor_exploration_vs_sli_T1_first_bottom_selected"
                )
            else:
                title_3b_sel = (
                    f"{pre_period_exploration_title} "
                    "(top vs bottom SLI-selected learners)"
                )
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
                x_label=str(
                    corr_pre_floor_exploration_vs_sli_xlabel_override
                    or pre_period_exploration_xlabel
                ),
                y_label=str(
                    corr_pre_floor_exploration_vs_sli_ylabel_override or early_lbl
                ),
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
            "skipping pre-period exploration vs early SLI plot"
        )

    # --- Plot 3c: Pre-period exploration vs SLI_final (training {trn_label_idx}) ---
    _scatter_with_corr(
        x=pre_coverage_vals,
        y=sli_vals,
        title=pre_period_exploration_title,
        x_label=str(
            corr_pre_floor_exploration_vs_sli_xlabel_override
            or pre_period_exploration_xlabel
        ),
        y_label=str(corr_pre_floor_exploration_vs_sli_ylabel_override or y_label_sli),
        cfg=_cfg_with_plot_color(cfg, "pre_training_exploration_vs_sli"),
        filename="corr_pre_floor_exploration_vs_sli_final",
        customizer=customizer,
    )

    if selected_mode is not None:
        if selected_mode == "top":
            title_3c_sel = f"{pre_period_exploration_title} (top SLI-selected learners)"
            filename_3c_sel = "corr_pre_floor_exploration_vs_sli_final_top_selected"
        elif selected_mode == "bottom":
            title_3c_sel = (
                f"{pre_period_exploration_title} (bottom SLI-selected learners)"
            )
            filename_3c_sel = "corr_pre_floor_exploration_vs_sli_final_bottom_selected"
        else:
            title_3c_sel = (
                f"{pre_period_exploration_title} "
                "(top vs bottom SLI-selected learners)"
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
            x_label=str(
                corr_pre_floor_exploration_vs_sli_xlabel_override
                or pre_period_exploration_xlabel
            ),
            y_label=str(
                corr_pre_floor_exploration_vs_sli_ylabel_override or y_label_sli
            ),
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
            cfg=_cfg_with_plot_color(cfg, "early_sli_vs_total_rewards"),
            filename="corr_reward_pi_first_bucket_vs_total_rewards",
            customizer=customizer,
        )

        # --- Plot 5a: Pre-training PI vs T1 first-bucket PI (all learners) ---
        _scatter_with_corr(
            x=pre_pi_diff_vals,
            y=reward_pi_training_vals,
            title="Baseline PI vs early SLI",
            x_label="Baseline PI\n(exp - yok, pre-training)",
            y_label=early_lbl,
            cfg=_cfg_with_plot_color(cfg, "baseline_pi_vs_sli"),
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
                    y_label=early_lbl,
                    cfg=_cfg_with_plot_color(cfg, "baseline_pi_vs_sli"),
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
                y_label=early_lbl,
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
            strong_y_label=str(corr_fast_vs_strong_ylabel_override or x_label_sli),
            strong_title_suffix=sli_ctx.label_short(),
            x_label=str(corr_fast_vs_strong_xlabel_override or "SLI for T1 SB1"),
            image_format=cfg.image_format,
        )
