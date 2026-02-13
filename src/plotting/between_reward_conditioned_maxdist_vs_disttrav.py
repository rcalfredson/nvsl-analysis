# src/plotting/between_reward_conditioned_maxdist_vs_disttrav.py
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.utils.common import maybe_sentence_case, writeImage

from src.plotting.between_reward_hexbin_density import (
    BetweenRewardHexbinConfig,
    HexBinGridSpec,
    collect_per_fly_segment_points,
    _x_mode_norm,
    _pretty_x_mode,
)

# ----------------------------
# Axis helpers
# ----------------------------


def _format_mm(v: float) -> str:
    """Pretty-print mm bin edges as integers when reasonable."""
    if not np.isfinite(v):
        return "nan"
    # Round to nearest int for compact bin labels (matches your ask).
    return str(int(np.rint(v)))


def _mm_labels_from_edges_plotspace(edges_plot: np.ndarray) -> list[str]:
    """
    Build bin labels in mm for x-edges that are currently in plot-space (log1p-space).
    """
    edges_plot = np.asarray(edges_plot, float)
    mm = np.expm1(edges_plot)
    mm = np.maximum(0.0, mm)

    labs: list[str] = []
    for a, b in zip(mm[:-1], mm[1:]):
        labs.append(f"{_format_mm(a)}-{_format_mm(b)}")
    return labs


def _apply_categorical_mm_xaxis(
    ax: plt.Axes, *, edges_plot: np.ndarray, customizer: PlotCustomizer
) -> np.ndarray:
    """
    When x is binned in log1p-space, switch to a categorical x-axis:
        positions: 0..B-1
        tick labels: "mm0-mm1" per bin
    Returns:
        pos: (B,) array of categorical x positions.
    """
    edges_plot = np.asarray(edges_plot, float)
    B = int(max(1, edges_plot.size - 1))
    pos = np.arange(B, dtype=float)
    labs = _mm_labels_from_edges_plotspace(edges_plot)

    ax.set_xticks(pos)
    ax.set_xticklabels(
        labs,
        rotation=45,
        ha="right",
        fontsize=customizer.in_plot_font_size,
    )
    ax.set_xlim(-0.5, float(B) - 0.5)
    return pos


# ----------------------------
# Config + Result
# ----------------------------


@dataclass
class BetweenRewardConditionedMaxDistVsDistTravConfig:
    """
    Distance-traveled-binned max-distance analysis for between-reward segments.

    Binning:
        - x-axis: distance traveled during segment (Ltotal) OR return-leg (Lreturn)

    Metric per segment:
        - y-axis: max distance from reward center during segment (Dmax)

    Aggregation:
        - per-fly mean Dmax within each x-bin
        - then mean + CI across flies per x-bin
    """

    out_file: str
    out_npz: str | None = None

    training_index: int = 1  # Training 2 default (0-based)
    skip_first_sync_buckets: int = 0
    use_reward_exclusion_mask: bool = False

    # segment filters
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    min_walk_frames: int = 2
    per_segment_min_meddist_mm: float = 0.0

    # which x metric
    x_mode: str = "Ltotal"  # "Ltotal" or "Lreturn"

    # binning (in the same space we compute points in)
    x_bin_width_mm: float = 50.0
    x_bin_width_log: float = 1.0
    x_min_mm: float = 0.0
    x_max_mm: float = 1600.0
    y_max_mm: float | None = (
        None  # optional y cap in linear mm (transformed if log1p_y)
    )

    # transforms (if True, bins are in log1p-space)
    log1p_x: bool = False
    log1p_y: bool = False

    # Plot options
    ci_conf: float = 0.95
    ymax: float | None = None
    subset_label: str | None = None


@dataclass(frozen=True)
class BetweenRewardConditionedMaxDistVsDistTravResult:
    x_edges: np.ndarray
    x_centers: np.ndarray

    mean_y: np.ndarray
    ci_lo_y: np.ndarray
    ci_hi_y: np.ndarray

    n_units: np.ndarray
    meta: dict

    per_unit_y: np.ndarray | None = None  # (N_units, B)

    def validate(self) -> None:
        edges = np.asarray(self.x_edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("x_edges must be 1D with >= 2 entries")
        B = int(edges.size - 1)

        def _chk(name: str) -> None:
            a = np.asarray(getattr(self, name))
            if a.ndim != 1 or a.size != B:
                raise ValueError(f"{name} must be 1D length {B}")

        for nm in ("x_centers", "mean_y", "ci_lo_y", "ci_hi_y", "n_units"):
            _chk(nm)

        if self.per_unit_y is not None:
            pu = np.asarray(self.per_unit_y, float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit_y must be shape (N, {B})")

    def save_npz(self, path: str) -> None:
        self.validate()
        kwargs = dict(
            x_edges=np.asarray(self.x_edges, float),
            x_centers=np.asarray(self.x_centers, float),
            mean_y=np.asarray(self.mean_y, float),
            ci_lo_y=np.asarray(self.ci_lo_y, float),
            ci_hi_y=np.asarray(self.ci_hi_y, float),
            n_units=np.asarray(self.n_units, int),
            meta=np.asarray([self.meta], dtype=object),
        )
        if self.per_unit_y is not None:
            kwargs["per_unit_y"] = np.asarray(self.per_unit_y, float)
        np.savez_compressed(path, **kwargs)

    @staticmethod
    def load_npz(path: str) -> "BetweenRewardConditionedMaxDistVsDistTravResult":
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta_obj = z["meta"]
                meta = meta_obj.item() if hasattr(meta_obj, "item") else {}
            except Exception:
                meta = {}

        res = BetweenRewardConditionedMaxDistVsDistTravResult(
            x_edges=np.asarray(z["x_edges"], float),
            x_centers=np.asarray(z["x_centers"], float),
            mean_y=np.asarray(z["mean_y"], float),
            ci_lo_y=np.asarray(z["ci_lo_y"], float),
            ci_hi_y=np.asarray(z["ci_hi_y"], float),
            n_units=np.asarray(z["n_units"], int),
            meta=dict(meta) if isinstance(meta, dict) else {},
            per_unit_y=(
                np.asarray(z["per_unit_y"], float) if "per_unit_y" in z else None
            ),
        )
        res.validate()
        return res


# ----------------------------
# Plotter
# ----------------------------


class BetweenRewardConditionedMaxDistVsDistTravPlotter:
    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardConditionedMaxDistVsDistTravConfig,
    ):
        self.vas = list(vas)
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = "btw_rwd_disttrav_binned_dmax"

    def _x_edges(self) -> np.ndarray:
        if self.cfg.log1p_x:
            x0 = np.log1p(max(0.0, float(self.cfg.x_min_mm)))
            x1 = np.log1p(max(0.0, float(self.cfg.x_max_mm)))
            w = float(self.cfg.x_bin_width_log)  # new cfg field
            n = int(np.ceil((x1 - x0) / w))
            edges = x0 + w * np.arange(n + 1)
            edges[-1] = x1
            return edges
        else:
            w = float(self.cfg.x_bin_width_mm)
            x0 = float(self.cfg.x_min_mm)
            x1 = float(self.cfg.x_max_mm)
            if not (
                np.isfinite(w)
                and w > 0
                and np.isfinite(x0)
                and np.isfinite(x1)
                and x1 > x0
            ):
                edges = np.asarray([0.0, 1.0], float)
            else:
                n = int(np.ceil((x1 - x0) / w))
                edges = x0 + w * np.arange(n + 1, dtype=float)
                edges[-1] = x1

        return edges

    def _collect_per_fly_binned_means(self) -> tuple[np.ndarray, dict]:
        """
        Returns:
            Y: (N_units, B) per-unit mean(Dmax) per x-bin
            meta: dict
        """
        edges = self._x_edges()
        B = int(max(1, edges.size - 1))

        # Build a tiny "hex cfg" to reuse collect_per_fly_segment_points.
        # Key: extent is in *plot-space*; we choose it to match our bin range.
        y_min_lin = 0.0

        # default: no cap
        y_max_plot = 1e9

        # optional: clamp at cfg.y_max_mm (specified in linear mm)
        if self.cfg.y_max_mm is not None:
            try:
                y_max_lin = float(self.cfg.y_max_mm)
                if np.isfinite(y_max_lin) and y_max_lin > y_min_lin:
                    # points are in plot-space, so transform if needed
                    y_max_plot = (
                        float(np.log1p(y_max_lin)) if self.cfg.log1p_y else y_max_lin
                    )
            except Exception:
                pass
        y_min_plot = float(np.log1p(y_min_lin)) if self.cfg.log1p_y else y_min_lin

        hb_cfg = BetweenRewardHexbinConfig(
            training_index=int(self.cfg.training_index),
            skip_first_sync_buckets=int(self.cfg.skip_first_sync_buckets),
            use_reward_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
            exclude_wall_contact=bool(self.cfg.exclude_wall_contact),
            exclude_nonwalking_frames=bool(self.cfg.exclude_nonwalking_frames),
            min_walk_frames=int(self.cfg.min_walk_frames),
            per_segment_min_meddist_mm=float(self.cfg.per_segment_min_meddist_mm),
            x_mode=str(self.cfg.x_mode),
            log1p_x=bool(self.cfg.log1p_x),
            log1p_y=bool(self.cfg.log1p_y),
            hex=HexBinGridSpec(
                gridside=1,
                extent=(
                    float(edges[0]),
                    float(edges[-1]),
                    float(y_min_plot),
                    float(y_max_plot),
                ),
                mincnt=1,
            ),
        )

        per_fly_points, unit_info, meta0 = collect_per_fly_segment_points(
            self.vas, cfg=hb_cfg, opts=self.opts, log_tag=self.log_tag
        )

        per_unit: list[np.ndarray] = []

        # Bin within each fly
        for pts in per_fly_points:
            pts = np.asarray(pts, float)
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
                continue

            x = pts[:, 0]
            y = pts[:, 1]
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if x.size == 0:
                continue

            bin_vals: list[list[float]] = [[] for _ in range(B)]
            jj = np.searchsorted(edges, x, side="right") - 1
            for j, yi in zip(jj, y):
                j = int(j)
                if 0 <= j < B and np.isfinite(yi):
                    bin_vals[j].append(float(yi))

            vec = np.full((B,), np.nan, float)
            for j in range(B):
                if bin_vals[j]:
                    vv = np.asarray(bin_vals[j], float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        vec[j] = float(np.mean(vv))

            if np.any(np.isfinite(vec)):
                per_unit.append(vec)

        if not per_unit:
            meta = dict(meta0)
            meta.update(dict(n_fly_units=0, x_edges=edges.tolist()))
            return np.empty((0, B), float), meta

        Y = np.stack(per_unit, axis=0)

        meta = dict(meta0)
        meta.update(
            dict(
                n_fly_units=int(Y.shape[0]),
                x_mode=str(self.cfg.x_mode),
                log1p_x=bool(self.cfg.log1p_x),
                log1p_y=bool(self.cfg.log1p_y),
                x_edges=edges.tolist(),
                x_bin_width_mm=float(self.cfg.x_bin_width_mm),
                x_bin_width_log=float(self.cfg.x_bin_width_log),
                x_min_mm=float(self.cfg.x_min_mm),
                x_max_mm=float(self.cfg.x_max_mm),
                y_max_mm=(
                    float(self.cfg.y_max_mm) if self.cfg.y_max_mm is not None else None
                ),
                y_cap_applied=bool(self.cfg.y_max_mm is not None),
                ci_conf=float(self.cfg.ci_conf),
                unit_info=unit_info,
            )
        )
        return Y, meta

    def compute_result(self) -> BetweenRewardConditionedMaxDistVsDistTravResult:
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        Y, meta = self._collect_per_fly_binned_means()

        mean_y = np.full((B,), np.nan, float)
        lo_y = np.full((B,), np.nan, float)
        hi_y = np.full((B,), np.nan, float)
        n_units = np.zeros((B,), int)

        if Y.size:
            for j in range(B):
                m, lo, hi, n = util.meanConfInt(Y[:, j], conf=float(self.cfg.ci_conf))
                mean_y[j], lo_y[j], hi_y[j] = float(m), float(lo), float(hi)
                n_units[j] = int(n)

        # If we ended up with a final "bin" that has zero width and no entries,
        # drop it so it doesn't show up as an empty tick/bar.
        # (Common when edges[-2] == edges[-1] due to clamping/rounding.)
        if edges.size >= 3:
            w_last = float(edges[-1] - edges[-2])
            j_last = int(edges.size - 2)  # last bin index
            if (
                np.isfinite(w_last)
                and w_last <= max(1e-12, 1e-5 * float(edges[-1] - edges[0]))
                and int(n_units[j_last]) <= 0
                and (not np.isfinite(mean_y[j_last]))
            ):
                edges = edges[:-1]
                centers = centers[:-1]
                mean_y = mean_y[:-1]
                lo_y = lo_y[:-1]
                hi_y = hi_y[:-1]
                n_units = n_units[:-1]
                if Y.size:
                    Y = Y[:, :-1]

        # Keep meta x_edges consistent with any trimming above
        try:
            meta = dict(meta) if isinstance(meta, dict) else {}
            meta["x_edges"] = np.asarray(edges, float).tolist()
        except Exception:
            pass

        return BetweenRewardConditionedMaxDistVsDistTravResult(
            x_edges=np.asarray(edges, float),
            x_centers=np.asarray(centers, float),
            mean_y=mean_y,
            ci_lo_y=lo_y,
            ci_hi_y=hi_y,
            n_units=n_units,
            meta=meta,
            per_unit_y=(Y if Y.size else None),
        )

    def plot(self) -> None:
        res = self.compute_result()
        res.validate()

        x = np.asarray(res.x_centers, float)
        edges = np.asarray(res.x_edges, float)
        widths = edges[1:] - edges[:-1]
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

        # x positions: plot-space centers, OR categorical bins if log1p_x
        if bool(self.cfg.log1p_x):
            x = _apply_categorical_mm_xaxis(
                ax, edges_plot=edges, customizer=self.customizer
            )
            widths_plot = np.full_like(x, 0.82, dtype=float)
        else:
            x = np.asarray(res.x_centers, float)
            widths_plot = widths

        out0 = str(self.cfg.out_file)
        root, ext = os.path.splitext(out0)
        if not ext:
            ext = "." + str(getattr(self.opts, "imageFormat", "png")).lstrip(".")
        out_path = f"{root}{ext}"

        y = np.asarray(res.mean_y, float)
        lo = np.asarray(res.ci_lo_y, float)
        hi = np.asarray(res.ci_hi_y, float)

        if not np.any(np.isfinite(y)):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(widths_plot)

            ax.bar(
                x[fin],
                y[fin],
                width=0.92 * widths_plot[fin],
                align="center",
                alpha=0.75,
                linewidth=0.8,
            )

            fin_ci = fin & np.isfinite(lo) & np.isfinite(hi)
            if fin_ci.any():
                yerr = np.vstack([y[fin_ci] - lo[fin_ci], hi[fin_ci] - y[fin_ci]])
                ax.errorbar(
                    x[fin_ci],
                    y[fin_ci],
                    yerr=yerr,
                    fmt="none",
                    elinewidth=1.1,
                    capsize=2.0,
                    alpha=0.9,
                )

            # Labels (respect x_mode + transforms)
            xdesc = _pretty_x_mode(self.cfg.x_mode)
            xlab = "Distance traveled [mm]"
            if _x_mode_norm(self.cfg.x_mode) == "lreturn":
                xlab = "Distance traveled after farthest point [mm]"
            ylab = (
                "Mean max distance from reward [mm]"
                if not self.cfg.log1p_y
                else "Mean max distance (log1p)"
            )

            ax.set_xlabel(maybe_sentence_case(xlab))
            ax.set_ylabel(maybe_sentence_case(ylab))

            if not bool(self.cfg.log1p_x):
                ax.set_xlim(float(edges[0]), float(edges[-1]))
            ax.set_ylim(bottom=0)

            if self.cfg.ymax is not None:
                ax.set_ylim(top=float(self.cfg.ymax))
            else:
                y_top = np.nanmax(hi) if np.isfinite(np.nanmax(hi)) else np.nanmax(y)
                if np.isfinite(y_top):
                    ax.set_ylim(top=float(y_top) * 1.12)

            # n labels
            ylim0, ylim1 = ax.get_ylim()
            y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
            for xi, hi_i, ni in zip(x, hi, res.n_units):
                if np.isfinite(xi) and np.isfinite(hi_i) and int(ni) > 0:
                    util.pltText(
                        float(xi),
                        float(hi_i + y_off),
                        f"{int(ni)}",
                        ha="center",
                        size=self.customizer.in_plot_font_size,
                        color=".2",
                    )

            tnum = int(self.cfg.training_index) + 1
            title = f"between-reward max distance vs distance traveled ({xdesc})"
            ax.set_title(maybe_sentence_case(title))

            parts = [f"T{tnum}"]
            if int(self.cfg.skip_first_sync_buckets) > 0:
                parts.append(
                    f"skip first {int(self.cfg.skip_first_sync_buckets)} bucket(s)"
                )
            if self.cfg.subset_label:
                parts.append(str(self.cfg.subset_label))
            fig.text(
                0.02,
                0.98,
                " | ".join(parts),
                ha="left",
                va="top",
                fontsize=self.customizer.in_plot_font_size,
                color="0",
            )

        if self.customizer.font_size_customized:
            self.customizer.adjust_padding_proportionally()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        writeImage(out_path, format=self.opts.imageFormat)
        plt.close(fig)

        if self.cfg.out_npz:
            try:
                res.save_npz(str(self.cfg.out_npz))
                print(f"[{self.log_tag}] wrote {self.cfg.out_npz}")
            except Exception as e:
                print(f"[{self.log_tag}] WARNING: failed to write NPZ: {e}")

        print(f"[{self.log_tag}] wrote {out_path}")


def plot_btw_rwd_conditioned_dmax_vs_disttrav_overlay(
    *,
    results: Sequence["BetweenRewardConditionedMaxDistVsDistTravResult"],
    labels: Sequence[str],
    out_file: str,
    opts,
    customizer: PlotCustomizer,
    log_tag: str = "btw_rwd_dmax_vs_disttrav",
    do_stats: bool = False,
    stats_alpha: float = 0.05,
) -> None:
    """
    Plot multiple cached BetweenRewardConditionedMaxDistVsDistTravResult objects as grouped bars.

    One image: <out_file>.<ext> (or preserves ext if given).
    """
    if not results:
        raise ValueError("No results provided")
    if len(results) != len(labels):
        m = min(len(results), len(labels))
        results = list(results)[:m]
        labels = list(labels)[:m]

    # Derive output path
    base = str(out_file)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = "." + str(getattr(opts, "imageFormat", "png")).lstrip(".")
        root = base
    out_path = f"{root}{ext}"

    # Reference x-axis from first result
    ref = results[0]
    ref.validate()
    edges = np.asarray(ref.x_edges, dtype=float)
    widths = edges[1:] - edges[:-1]

    # Validate x_edges match across results
    for r, lab in zip(results, labels):
        r.validate()
        e2 = np.asarray(r.x_edges, dtype=float)
        if e2.shape != edges.shape or not np.allclose(e2, edges, equal_nan=True):
            raise ValueError(
                f"[{log_tag}] x_edges mismatch for {lab!r}; cannot overlay safely."
            )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4))

    # Determine if we should switch to categorical bins (mm-labeled) for log1p-x
    meta_ref = getattr(ref, "meta", {}) or {}
    log1p_x_ref = bool(meta_ref.get("log1p_x", False))
    if log1p_x_ref:
        x_centers_for_stats = _apply_categorical_mm_xaxis(
            ax, edges_plot=edges, customizer=customizer
        )
        widths_plot = np.full_like(x_centers_for_stats, 0.82, dtype=float)
    else:
        x_centers_for_stats = np.asarray(ref.x_centers, dtype=float)
        widths_plot = widths

    n_groups = len(results)
    frac = 0.86
    bar_w = frac * widths_plot / max(1, n_groups)

    any_data = False
    pending_labels: list[tuple[float, float, int]] = []  # (xpos, y_top, n)
    xpos_by_group: list[np.ndarray] = []
    per_unit_by_group: list[np.ndarray] = []
    hi_by_group: list[np.ndarray] = []

    for gi, (r, lab) in enumerate(zip(results, labels)):
        y = np.asarray(r.mean_y, dtype=float)
        lo_i = np.asarray(r.ci_lo_y, dtype=float)
        hi_i = np.asarray(r.ci_hi_y, dtype=float)
        n_i = np.asarray(r.n_units, dtype=int)

        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        xb = x_centers_for_stats + offset
        xpos_by_group.append(np.asarray(xb, float))
        hi_by_group.append(hi_i)

        fin = np.isfinite(xb) & np.isfinite(y) & np.isfinite(widths_plot)
        if not fin.any():
            continue
        any_data = True

        ax.bar(
            xb[fin],
            y[fin],
            width=bar_w[fin],
            align="center",
            alpha=0.75,
            linewidth=0.8,
            label=str(lab),
        )

        fin_ci = fin & np.isfinite(lo_i) & np.isfinite(hi_i)
        if fin_ci.any():
            yerr = np.vstack([y[fin_ci] - lo_i[fin_ci], hi_i[fin_ci] - y[fin_ci]])
            ax.errorbar(
                xb[fin_ci],
                y[fin_ci],
                yerr=yerr,
                fmt="none",
                elinewidth=1.1,
                capsize=2.0,
                alpha=0.9,
            )

        # stash per-unit (if present) for stats
        if r.per_unit_y is not None:
            per_unit_by_group.append(np.asarray(r.per_unit_y, float))

        # queue n-labels
        idxs = np.where(fin)[0]
        for j in idxs:
            nn = int(n_i[j])
            if nn <= 0:
                continue
            y_top = float(hi_i[j]) if np.isfinite(hi_i[j]) else float(y[j])
            if np.isfinite(y_top):
                pending_labels.append((float(xb[j]), y_top, nn))

    if not any_data:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        # Labels: derive from meta of first result if present
        meta = getattr(ref, "meta", {}) or {}
        x_mode = str(meta.get("x_mode", "")).strip()
        log1p_x = bool(meta.get("log1p_x", False))
        log1p_y = bool(meta.get("log1p_y", False))

        if x_mode.lower() == "lreturn":
            xlab = "Distance traveled after farthest point [mm]"
        else:
            xlab = "Distance traveled [mm]"
        ylab = (
            "Mean max distance from reward [mm]"
            if not log1p_y
            else "Mean max distance (log1p)"
        )

        ax.set_xlabel(maybe_sentence_case(xlab))
        ax.set_ylabel(maybe_sentence_case(ylab))

        if (not log1p_x) and edges.size >= 2 and np.all(np.isfinite(edges[[0, -1]])):
            ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_ylim(bottom=0)

        # Optional y max from CLI
        ymax = getattr(opts, "btw_rwd_conditioned_dmax_vs_disttrav_ymax", None)
        if ymax is not None:
            try:
                ax.set_ylim(top=float(ymax))
            except Exception:
                pass

        # Optional stats (only if every group has per_unit_y)
        if do_stats and len(per_unit_by_group) != n_groups:
            print(
                f"[{log_tag}] WARNING: stats requested but some results lack per_unit_y; skipping stats."
            )
        if do_stats and len(per_unit_by_group) == n_groups:
            cfg_stats = StatAnnotConfig(alpha=float(stats_alpha), nlabel_off_frac=0.04)
            annotate_grouped_bars_per_bin(
                ax,
                x_centers=x_centers_for_stats,
                xpos_by_group=xpos_by_group,
                per_unit_by_group=per_unit_by_group,
                hi_by_group=hi_by_group,
                group_names=[str(l) for l in labels],
                cfg=cfg_stats,
            )

        ax.legend(loc="best", fontsize=customizer.in_plot_font_size)
        ax.set_title(
            maybe_sentence_case("between-reward max distance vs distance traveled")
        )

        # n labels
        if pending_labels:
            ylim0, ylim1 = ax.get_ylim()
            y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
            for xb, y_top, nn in pending_labels:
                util.pltText(
                    xb,
                    y_top + y_off,
                    f"{nn}",
                    ha="center",
                    size=customizer.in_plot_font_size,
                    color=".2",
                )

    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally()
    fig.tight_layout(rect=(0, 0, 1, 1))
    writeImage(out_path, format=opts.imageFormat)
    plt.close(fig)
    print(f"[{log_tag}] wrote {out_path}")
