# src/plotting/overlay_training_metric_hist.py

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.bin_edges import (
    is_grouped_edges,
    normalize_panel_edges,
    geom_from_edges,
)
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin


@dataclass(frozen=True)
class ExportedTrainingHistogram:
    group: str
    panel_labels: list[str]
    # pooled-mode payload (may be empty/unused when per_fly=True)
    counts: np.ndarray  # (n_panels, bins)
    # per-fly-mode payload (None if unavailable)
    mean: np.ndarray | None  # (n_panels, bins)
    ci_lo: np.ndarray | None  # (n_panels, bins)
    ci_hi: np.ndarray | None  # (n_panels, bins)
    n_units: np.ndarray | None  # (n_panels, bins) int
    n_units_panel: np.ndarray | None  # (n_panels,) int
    per_unit_panel: (
        np.ndarray | None
    )  # object array length n_panels; each entry (N_panel, bins)
    per_unit_ids_panel: (
        np.ndarray | None
    )  # object array length n_panels; each entry (N_panel,)
    # bin_edges:
    #   - flat: (n_panels, bins+1) float array
    #   - grouped: object array length n_panels; each entry is list[np.ndarray] of 1D edges
    bin_edges: np.ndarray
    n_used: np.ndarray  # (n_panels,)
    meta: dict

    @property
    def bins(self) -> int:
        return int(self.meta.get("bins"))

    @property
    def xmax_effective(self) -> float | None:
        return self.meta.get("xmax_effective", None)

    @property
    def pool_trainings(self) -> bool:
        return bool(self.meta.get("pool_trainings", False))

    @property
    def per_fly(self) -> bool:
        return bool(self.meta.get("per_fly", False))

    @property
    def normalize(self) -> bool:
        # present in histogram config, not always in meta historically
        return bool(self.meta.get("normalize", False))


def _maybe_none_array(x) -> np.ndarray | None:
    """
    np.savez will sometimes store None as a 0-d object array.
    Normalize those cases back to Python None.
    """
    if x is None:
        return None
    arr = np.asarray(x)
    # Common patterns: array(None, dtype=object) or array([None], dtype=object)
    if arr.dtype == object:
        try:
            if arr.shape == () and arr.item() is None:
                return None
            if arr.size == 1 and arr.ravel()[0] is None:
                return None
        except Exception:
            pass
    return arr


def _edges_equal(a_item, b_item) -> bool:
    a = normalize_panel_edges(a_item)
    b = normalize_panel_edges(b_item)
    if isinstance(a, list) != isinstance(b, list):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(
            aa.shape == bb.shape and np.allclose(aa, bb, rtol=0, atol=1e-12)
            for aa, bb in zip(a, b)
        )
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-12)


def _panel_n_label(h: ExportedTrainingHistogram, p_idx: int) -> str:
    if h.per_fly:
        if h.n_units_panel is not None and h.n_units_panel.shape[0] > p_idx:
            return str(int(h.n_units_panel[p_idx]))
        # fallback if older exports: try max n_units across bins
        if h.n_units is not None and h.n_units.shape[0] > p_idx:
            return str(int(np.nanmax(h.n_units[p_idx])))
        return "?"
    else:
        return str(int(h.n_used[p_idx]))


def load_export_npz(group: str, path: str) -> ExportedTrainingHistogram:
    d = np.load(path, allow_pickle=True)
    panel_labels = [str(x) for x in list(d["panel_labels"])]
    counts = np.asarray(d["counts"])
    mean = _maybe_none_array(d["mean"] if "mean" in d.files else None)
    ci_lo = _maybe_none_array(d["ci_lo"] if "ci_lo" in d.files else None)
    ci_hi = _maybe_none_array(d["ci_hi"] if "ci_hi" in d.files else None)
    n_units = _maybe_none_array(d["n_units"] if "n_units" in d.files else None)
    n_units_panel = _maybe_none_array(
        d["n_units_panel"] if "n_units_panel" in d.files else None
    )
    per_unit_panel = _maybe_none_array(
        d["per_unit_panel"] if "per_unit_panel" in d.files else None
    )
    per_unit_ids_panel = _maybe_none_array(
        d["per_unit_ids_panel"] if "per_unit_ids_panel" in d.files else None
    )
    bin_edges = np.asarray(d["bin_edges"])
    n_used = np.asarray(d["n_used"])
    meta_json = d["meta_json"].item() if "meta_json" in d.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(meta_json)
    return ExportedTrainingHistogram(
        group=group,
        panel_labels=panel_labels,
        counts=counts,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_units=n_units,
        n_units_panel=n_units_panel,
        per_unit_panel=per_unit_panel,
        per_unit_ids_panel=per_unit_ids_panel,
        bin_edges=bin_edges,
        n_used=n_used,
        meta=meta,
    )


def _fmt_mismatch(name: str, vals: list) -> str:
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return f"{name} differs across inputs: {uniq}"


def validate_alignment(hists: list[ExportedTrainingHistogram]) -> None:
    if len(hists) < 2:
        return

    bins = [h.bins for h in hists]
    if len(set(bins)) != 1:
        raise ValueError(_fmt_mismatch("bins", bins))

    per_fly = [h.per_fly for h in hists]
    if len(set(per_fly)) != 1:
        raise ValueError(_fmt_mismatch("per_fly", per_fly))

    pools = [h.pool_trainings for h in hists]
    if len(set(pools)) != 1:
        raise ValueError(_fmt_mismatch("pool_trainings", pools))

    xmax_eff = [h.xmax_effective for h in hists]
    # For numeric xmax, require close equality; for None, all must be None.
    if any(x is None for x in xmax_eff):
        if not all(x is None for x in xmax_eff):
            raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))
    else:
        # All numeric
        x0 = float(xmax_eff[0])
        for x in xmax_eff[1:]:
            if not np.isclose(float(x), x0, rtol=0, atol=1e-9):
                raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))

    # Panel labels must match exactly
    labels0 = hists[0].panel_labels
    for h in hists[1:]:
        if h.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [hh.panel_labels for hh in hists])
            )

    # Bin edges must match per panel
    edges0 = hists[0].bin_edges
    for h in hists[1:]:
        if len(h.bin_edges) != len(edges0):
            raise ValueError("bin_edges shape differs across inputs")
        for p_idx in range(len(edges0)):
            if not _edges_equal(h.bin_edges[p_idx], edges0[p_idx]):
                raise ValueError(
                    "bin_edges differ across inputs. Re-export with the same bin_edges/bin_edges_groups."
                )


def _panel_y(h: ExportedTrainingHistogram, p_idx: int) -> np.ndarray:
    """
    Return y values of length (bins,) for this panel.
    - per_fly=True: use mean (already PDF if normalize=True)
    - per_fly=False: use pooled counts
    """
    if h.per_fly:
        if h.mean is None:
            return np.full((h.bins,), np.nan, dtype=float)
        if h.mean.shape[0] <= p_idx:
            return np.full((h.bins,), np.nan, dtype=float)
        y = np.asarray(h.mean[p_idx], dtype=float)
        return y
    else:
        if h.counts.shape[0] <= p_idx:
            return np.zeros((h.bins,), dtype=float)
        return np.asarray(h.counts[p_idx], dtype=float)


def plot_overlays(
    hists: list[ExportedTrainingHistogram],
    *,
    mode: str,  # "pdf" or "cdf"
    layout: str = "overlay",  # "overlay" or "grouped"
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    xmax_plot: float | None = None,
    categorical_bin_ratio_max: float = 4.0,
    debug: bool = False,
) -> plt.Figure:
    if mode not in ("pdf", "cdf"):
        raise ValueError("mode must be 'pdf' or 'cdf'")
    if layout not in ("overlay", "grouped"):
        raise ValueError("layout must be 'overlay' or 'grouped'")
    if layout == "grouped" and mode != "pdf":
        raise ValueError("layout='grouped' is implemented for mode='pdf' only")

    validate_alignment(hists)
    panel_labels = hists[0].panel_labels
    n_panels = len(panel_labels)

    # ---- figure sizing ----
    G = len(hists)
    bins = hists[0].bins

    if layout == "grouped" and mode == "pdf":
        # px heuristics
        px_per_group = 12  # tweakable: 10-14
        dpi = 100
        panel_width = (bins * G * px_per_group) / dpi
        panel_width = max(panel_width, 7.5)  # floor so small histograms aren't tiny
    else:
        panel_width = 4.0

    fig_width = panel_width * n_panels
    fig_height = 4.5
    capsize = 1.25 if layout == "grouped" else 2.0

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    edges = hists[0].bin_edges
    # For a step plot, we’ll use edges and a y of length bins+1
    for p_idx, (ax, plabel) in enumerate(zip(axes, panel_labels)):
        any_data = False
        keep_bins = None

        e_item0 = normalize_panel_edges(hists[0].bin_edges[p_idx])
        e_item = e_item0

        # --- optional plot-time truncation ---
        if xmax_plot is not None and np.isfinite(xmax_plot):
            xcut = float(xmax_plot)

            if is_grouped_edges(e_item0):
                new_groups = []
                kept = 0
                for g in e_item0:
                    g = np.asarray(g, float)
                    # keep bins whose RIGHT edge <= xcut
                    right = g[1:]
                    k = int(np.sum(right <= xcut))
                    if k <= 0:
                        continue
                    new_groups.append(g[: k + 1])
                    kept += k
                if kept <= 0:
                    kept = 1
                    # keep first bin of the first group so geometry isn't empty
                    g0 = np.asarray(e_item0[0], float)
                    new_groups = [g0[:2]]
                e_item = new_groups
                keep_bins = kept
            else:
                e_flat = np.asarray(e_item0, float)
                keep_bins = int(
                    np.searchsorted(e_flat[1:], xcut, side="right")
                )  # count of bins
                keep_bins = max(1, min(keep_bins, len(e_flat) - 1))
                e_item = e_flat[: keep_bins + 1]

        # Precompute geometry for grouped bars (PDF only)
        if mode == "pdf":
            widths, centers, bin_ranges, B, x0, x1 = geom_from_edges(e_item)

            # Decide whether to switch to categorical spacing
            wpos = widths[np.isfinite(widths) & (widths > 0)]
            width_ratio = float(np.nanmax(wpos) / np.nanmin(wpos)) if wpos.size else 1.0
            categorical_x = (layout == "grouped") and (
                width_ratio > float(categorical_bin_ratio_max)
            )

        if mode == "pdf" and layout == "grouped":
            G = max(1, len(hists))

            if categorical_x:
                # Categorical x positions: 0,1,2... so bins have equal spacing
                centers_x = np.arange(B, dtype=float)

                group_band = 0.80
                bar_w = np.full((B,), group_band / G, dtype=float)
            else:
                # Proportional x positions: respect bin widths
                centers_x = centers

                group_band = 0.95 * widths
                bar_w = group_band / G

            gpos = (np.arange(G) - (G - 1) / 2.0)[:, None]  # (G,1)
            offsets = gpos * bar_w[None, :]

        xpos_by_group: list[np.ndarray] = []
        per_unit_by_group: list[np.ndarray | None] = []
        per_unit_ids_by_group: list[np.ndarray | None] = []
        hi_by_group: list[np.ndarray] = []
        group_names = [h.group for h in hists]

        for g_idx, h in enumerate(hists):
            y_raw = _panel_y(h, p_idx)

            # Determine total / skip logic depending on payload type
            if h.per_fly:
                # mean may contain NaNs for bins with no contributing units
                y_bins = np.asarray(y_raw, dtype=float)

                if keep_bins is not None:
                    y_bins = y_bins[:keep_bins]

                if not np.any(np.isfinite(y_bins)):
                    continue
                any_data = True
                # In per-fly mode, the exported mean is already a PDF if normalize=True.
                # If normalize was False, mean is "mean #segments per fly", which is not a PDF.
                # Overlay script is intended for PDF/CDF, so require normalized per-fly exports.
                if not h.normalize:
                    raise ValueError(
                        "Overlay plotting with per_fly=True requires normalize=True in the export "
                        "(so mean represents a probability distribution)."
                    )
            else:
                counts = np.asarray(y_raw, dtype=float)
                total = float(np.sum(counts))
                if total <= 0:
                    continue
                any_data = True
                if mode == "pdf":
                    y_bins = counts / total

                    if keep_bins is not None:
                        y_bins = y_bins[:keep_bins]

            # ---- plotting ----
            if mode == "pdf":
                if layout == "overlay":
                    if is_grouped_edges(e_item):
                        # plot each group as a separate step segment (no bridging across gaps)
                        pos = 0
                        for gi, g in enumerate(e_item):
                            nb = int(len(g) - 1)
                            if nb <= 0:
                                continue
                            y_seg = y_bins[pos : pos + nb]
                            if y_seg.size != nb:
                                break
                            y_step = np.concatenate([y_seg, [y_seg[-1]]])
                            ax.step(
                                g,
                                y_step,
                                where="post",
                                label=(
                                    f"{h.group} (n={_panel_n_label(h, p_idx)})"
                                    if gi == 0
                                    else None
                                ),
                            )
                            pos += nb
                    else:
                        y_step = np.concatenate([y_bins, [y_bins[-1]]])
                        ax.step(
                            e_item,
                            y_step,
                            where="post",
                            label=f"{h.group} (n={_panel_n_label(h, p_idx)})",
                        )
                else:
                    # grouped / dodged bars
                    x = centers_x + offsets[g_idx]

                    # record x positions for bracket drawing
                    xpos_by_group.append(np.asarray(x, float))

                    # baseline for brackets (top of CI if available; else bar height)
                    if h.ci_hi is not None and h.ci_hi.shape[0] > p_idx:
                        tmp_hi = np.asarray(h.ci_hi[p_idx], float)
                        if keep_bins is not None:
                            tmp_hi = tmp_hi[:keep_bins]
                        hi_by_group.append(tmp_hi)
                    else:
                        hi_by_group.append(np.asarray(y_bins, float))

                    # record per-fly per-bin PDF values for stats
                    if h.per_fly:
                        pu_obj = getattr(h, "per_unit_panel", None)
                        pu = None
                        if pu_obj is not None and np.asarray(pu_obj).shape[0] > p_idx:
                            pu = pu_obj[p_idx]

                        ids_obj = getattr(h, "per_unit_ids_panel", None)
                        ids = None
                        if ids_obj is not None and np.asarray(ids_obj).shape[0] > p_idx:
                            ids = ids_obj[p_idx]

                        if pu is None:
                            # keep placeholder to error cleanly later if stats requested
                            per_unit_by_group.append(None)  # type: ignore[arg-type]
                            per_unit_ids_by_group.append(None)
                        else:
                            pu = np.asarray(pu, float)
                            if keep_bins is not None:
                                pu = pu[:, :keep_bins]
                            per_unit_by_group.append(pu)  # (N_panel, bins)

                            # ids must align with pu rows
                            if ids is None:
                                per_unit_ids_by_group.append(None)
                            else:
                                ids = np.asarray(ids, dtype=object).ravel()
                                if ids.shape[0] != pu.shape[0]:
                                    raise ValueError(
                                        f"per_unit_ids_panel size mismatch for group={h.group}, panel={plabel}: "
                                        f"ids={ids.shape[0]} vs per_unit_panel rows={pu.shape[0]}"
                                    )
                                per_unit_ids_by_group.append(ids)

                    # bar() ignores NaNs poorly; replace NaNs with 0-height bars
                    y_plot = np.where(np.isfinite(y_bins), y_bins, 0.0)
                    ax.bar(
                        x,
                        y_plot,
                        width=bar_w,
                        align="center",
                        label=f"{h.group} (n={_panel_n_label(h, p_idx)})",
                    )

                # CI whiskers: only meaningful for per-fly PDF overlays
                if (
                    h.per_fly
                    and getattr(h, "ci_lo", None) is not None
                    and getattr(h, "ci_hi", None) is not None
                ):
                    if h.ci_lo.shape[0] <= p_idx or h.ci_hi.shape[0] <= p_idx:
                        continue
                    lo = np.asarray(h.ci_lo[p_idx], dtype=float)
                    hi = np.asarray(h.ci_hi[p_idx], dtype=float)
                    y = np.asarray(y_bins, dtype=float)

                    if keep_bins is not None:
                        lo = lo[:keep_bins]
                        hi = hi[:keep_bins]

                    # yerr expects non-negative deltas; guard NaNs
                    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                    # whisker x positions depend on layout
                    if layout == "overlay":
                        xerr = centers
                    else:
                        xerr = centers_x + offsets[g_idx]
                    ax.errorbar(
                        xerr[mask],
                        y[mask],
                        yerr=np.vstack([y[mask] - lo[mask], hi[mask] - y[mask]]),
                        fmt="none",
                        ecolor="0.15",
                        capsize=capsize,
                        capthick=1.0,
                        elinewidth=1.0,
                        alpha=0.9,
                        zorder=3,
                    )
            else:
                # mode == "cdf"
                if h.per_fly:
                    # y_bins is already a PDF across bins (sum ~ 1, ignoring NaNs)
                    yy = np.where(np.isfinite(y_bins), y_bins, 0.0)
                    cdf = np.cumsum(yy)
                    # normalize in case of tiny numerical drift
                    if cdf.size and cdf[-1] > 0:
                        cdf = cdf / cdf[-1]
                else:
                    # pooled counts
                    cdf = np.cumsum(counts) / total
                    y_bins = counts / total

                if is_grouped_edges(e_item):
                    # Draw per-group, carrying forward cumulative value
                    pos = 0
                    cprev = 0.0
                    for gi, g in enumerate(e_item):
                        nb = int(len(g) - 1)
                        if nb <= 0:
                            continue
                        yy = np.where(
                            np.isfinite(y_bins[pos : pos + nb]),
                            y_bins[pos : pos + nb],
                            0.0,
                        )
                        cdf_seg = cprev + np.cumsum(yy)
                        y_step = np.concatenate([[cprev], cdf_seg])
                        ax.step(
                            g,
                            y_step,
                            where="post",
                            label=(
                                f"{h.group} (n={_panel_n_label(h, p_idx)})"
                                if gi == 0
                                else None
                            ),
                        )
                        cprev = float(cdf_seg[-1]) if cdf_seg.size else cprev
                        pos += nb
                else:

                    y_step = np.concatenate([[0.0], cdf])
                    ax.step(
                        e_item,
                        y_step,
                        where="post",
                        label=f"{h.group} (n={_panel_n_label(h, p_idx)})",
                    )

        if not any_data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        ax.set_title(plabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if p_idx == 0:
            if ylabel:
                ax.set_ylabel(ylabel)

        if ymax is not None:
            ax.set_ylim(bottom=0, top=float(ymax))

        # ---- bin-range x tick labels for grouped PDF bars ----
        if mode == "pdf" and layout == "grouped":
            # centers/edges may have been truncated above
            ax.set_xticks(centers_x)

            # label bins as ranges: "0-10", "10-20", ...
            labels_xt = []
            for a, b in bin_ranges:
                # choose formatting: ints if close, else compact float
                if np.isclose(a, round(a)) and np.isclose(b, round(b)):
                    labels_xt.append(f"{int(round(a))}-{int(round(b))}")
                else:
                    labels_xt.append(f"{a:0.2f}-{b:0.2f}")

            ax.set_xticklabels(labels_xt, rotation=0, fontsize=8)

        if mode == "pdf" and layout == "grouped":
            if categorical_x:
                ax.set_xlim(-0.5, B - 0.5)
            else:
                ax.set_xlim(x0, x1)

        if stats and mode == "pdf" and layout == "grouped":
            # require per-fly PDF inputs
            if not all(h.per_fly for h in hists):
                raise ValueError(
                    "Stats require per_fly=True exports (one PDF per fly)."
                )
            if any(pu is None for pu in per_unit_by_group):
                raise ValueError(
                    "Stats requested but per_unit_panel missing in one or more inputs. "
                    "Re-export with per_unit_panel enabled."
                )

            if stats_paired and any(ids is None for ids in per_unit_ids_by_group):
                raise ValueError(
                    "Paired stats requested but per_unit_ids_panel missing in one or more inputs. "
                    "Re-export with per_unit_ids_panel enabled."
                )

            cfg_stats = StatAnnotConfig(
                alpha=float(stats_alpha),
                min_n_per_group=3,
                nlabel_off_frac=0.0,  # IMPORTANT: n is in legend, not above bars
            )

            annotate_grouped_bars_per_bin(
                ax,
                x_centers=centers_x,
                xpos_by_group=xpos_by_group,
                per_unit_by_group=per_unit_by_group,
                per_unit_ids_by_group=per_unit_ids_by_group,
                hi_by_group=hi_by_group,
                group_names=group_names,
                cfg=cfg_stats,
                paired=bool(stats_paired),
                panel_label=plabel,
                debug=debug,
            )

        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
