# src/plotting/overlay_training_metric_scalar_bars.py

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.palettes import NEUTRAL_DARK, group_accent_color, group_fill_color
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.utils.util import meanConfInt


@dataclass(frozen=True)
class ExportedTrainingScalarBars:
    group: str
    panel_labels: list[str]

    # object arrays length n_panels:
    #   per_unit_values_panel[p] -> (N_panel,) float
    #   per_unit_ids_panel[p] -> (N_panel,) object
    per_unit_values_panel: np.ndarray
    per_unit_ids_panel: np.ndarray

    # (n_panels,)
    mean: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    n_units_panel: np.ndarray

    meta: dict

    @property
    def pool_trainings(self) -> bool:
        return bool(self.meta.get("pool_trainings", False))

    @property
    def ci_conf(self) -> float:
        return float(self.meta.get("ci_conf", 0.95))


def _maybe_none_array(x) -> np.ndarray | None:
    """
    np.savez sometimes stores None as a 0-d object array.
    Normalize those cases back to Python None.
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            if arr.shape == () and arr.item() is None:
                return None
            if arr.size == 1 and arr.ravel()[0] is None:
                return None
        except Exception:
            pass
    return arr


def _wrapped_xlabel_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if " from " in text:
        return text.replace(" from ", "\nfrom ", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    return text


def _ensure_xlabel_visible(fig: plt.Figure, ax: plt.Axes) -> None:
    label = ax.xaxis.get_label()
    if not label.get_text():
        return

    pad_y_px = 6.0
    pad_x_px = max(18.0, 0.9 * float(label.get_fontsize()) + 8.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = label.get_window_extent(renderer=renderer)
    fig_bbox = fig.bbox
    x_ok = (
        bbox.x0 >= fig_bbox.x0 + pad_x_px and bbox.x1 <= fig_bbox.x1 - pad_x_px
    )
    y_ok = bbox.y0 >= fig_bbox.y0 + pad_y_px

    if x_ok and y_ok:
        return

    wrapped = _wrapped_xlabel_text(label.get_text())
    if wrapped != label.get_text():
        label.set_text(wrapped)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = label.get_window_extent(renderer=renderer)
        x_ok = (
            bbox.x0 >= fig_bbox.x0 + pad_x_px and bbox.x1 <= fig_bbox.x1 - pad_x_px
        )
        y_ok = bbox.y0 >= fig_bbox.y0 + pad_y_px

    if x_ok and y_ok:
        return

    overflow_bottom_px = max((fig_bbox.y0 + pad_y_px) - bbox.y0, 0.0)
    overflow_left_px = max((fig_bbox.x0 + pad_x_px) - bbox.x0, 0.0)
    overflow_right_px = max(bbox.x1 - (fig_bbox.x1 - pad_x_px), 0.0)

    fig_h_px = max(fig.get_size_inches()[1] * fig.dpi, 1.0)
    fig_w_px = max(fig.get_size_inches()[0] * fig.dpi, 1.0)

    extra_bottom = float(overflow_bottom_px / fig_h_px) + 0.01
    extra_left = float(overflow_left_px / fig_w_px) + 0.005
    extra_right = float(overflow_right_px / fig_w_px) + 0.005

    new_bottom = min(fig.subplotpars.bottom + extra_bottom, 0.38)
    new_left = min(fig.subplotpars.left + extra_left, 0.20)
    new_right = max(fig.subplotpars.right - extra_right, 0.82)
    if new_right <= new_left:
        new_left = fig.subplotpars.left
        new_right = fig.subplotpars.right

    fig.subplots_adjust(bottom=new_bottom, left=new_left, right=new_right)
    fig.canvas.draw()


def _paired_filter_mats_all_panels(
    mats: list[np.ndarray],
    ids: list[np.ndarray],
    *,
    P: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Build paired-only matrices across ALL panels:
      - For each panel p, find IDs present in ALL groups (finite values in that column).
      - Build a stable union of IDs that are paired in at least one panel.
      - Output matrices are (N_union, P), with entries present only where the ID is paired for that panel.
    Returns:
      (mats_filt_per_group, union_ids, paired_n_per_panel)
    """
    # per-panel common ids
    common_ids_by_panel: list[list[str]] = []
    paired_n = np.zeros((P,), dtype=int)

    for p in range(P):
        sets = []
        for M, I in zip(mats, ids):
            col = np.asarray(M[:, p], float)
            mask = np.isfinite(col)
            keys = {str(i) for i in I[mask] if i is not None}
            sets.append(keys)
        common = set.intersection(*sets) if sets else set()
        keys_sorted = sorted(common)
        common_ids_by_panel.append(keys_sorted)
        paired_n[p] = int(len(keys_sorted))

    # stable union of ids paired in at least one panel
    seen = set()
    union_ids: list[str] = []
    for keys in common_ids_by_panel:
        for k in keys:
            if k not in seen:
                seen.add(k)
                union_ids.append(k)
    union_ids = sorted(union_ids)

    # build filtered matrices
    mats_filt: list[np.ndarray] = []
    for M, I in zip(mats, ids):
        idx_map = {str(i): ii for ii, i in enumerate(I) if i is not None}
        out = np.full((len(union_ids), P), np.nan, dtype=float)
        for p in range(P):
            keys = set(common_ids_by_panel[p])
            if not keys:
                continue
            for r, k in enumerate(union_ids):
                if k not in keys:
                    continue
                ii = idx_map.get(k, None)
                if ii is None:
                    continue
                v = float(M[ii, p])
                if np.isfinite(v):
                    out[r, p] = v
        mats_filt.append(out)

    return mats_filt, np.asarray(union_ids, dtype=object), paired_n


def _mean_ci_from_util(x: np.ndarray, conf: float) -> tuple[float, float, float, int]:
    """
    Mean and t CI across finite x, via src.utils.util.meanConfInt.
    Returns (mean, lo, hi, n).
    """
    m, lo, hi, n = meanConfInt(np.asarray(x, float), conf=float(conf), asDelta=False)
    return float(m), float(lo), float(hi), int(n)


def load_export_npz(group: str, path: str) -> ExportedTrainingScalarBars:
    d = np.load(path, allow_pickle=True)

    panel_labels = [str(x) for x in list(d["panel_labels"])]

    per_unit_values_panel = np.asarray(d["per_unit_values_panel"], dtype=object)
    per_unit_ids_panel = np.asarray(d["per_unit_ids_panel"], dtype=object)

    mean = np.asarray(d["mean"], dtype=float)
    ci_lo = np.asarray(d["ci_lo"], dtype=float)
    ci_hi = np.asarray(d["ci_hi"], dtype=float)
    n_units_panel = np.asarray(d["n_units_panel"], dtype=int)

    meta_json = d["meta_json"].item() if "meta_json" in d.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(meta_json)

    return ExportedTrainingScalarBars(
        group=group,
        panel_labels=panel_labels,
        per_unit_values_panel=per_unit_values_panel,
        per_unit_ids_panel=per_unit_ids_panel,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_units_panel=n_units_panel,
        meta=meta,
    )


def _fmt_mismatch(name: str, vals: list) -> str:
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return f"{name} differs across inputs: {uniq}"


def validate_alignment(xs: list[ExportedTrainingScalarBars]) -> None:
    if len(xs) < 2:
        return

    pools = [x.pool_trainings for x in xs]
    if len(set(pools)) != 1:
        raise ValueError(_fmt_mismatch("pool_trainings", pools))

    labels0 = xs[0].panel_labels
    for x in xs[1:]:
        if x.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [y.panel_labels for y in xs])
            )


def _panel_n(x: ExportedTrainingScalarBars, p_idx: int) -> int | None:
    if x.n_units_panel is None:
        return None
    if p_idx < 0 or p_idx >= int(x.n_units_panel.shape[0]):
        return None
    try:
        return int(x.n_units_panel[p_idx])
    except Exception:
        return None


def _legend_n_for_group(x: ExportedTrainingScalarBars) -> int | None:
    """
    Return a single n to show in the legend if it's unambiguous.
    - If only one panel, use that panel's n.
    - If multiple panels, use n only if it's constant (ignoring zeros/None).
    Otherwise return None (legend should omit n).
    """
    P = len(x.panel_labels)
    if P <= 0:
        return None
    if P == 1:
        return _panel_n(x, 0)

    if x.n_units_panel is None:
        return None
    n = np.asarray(x.n_units_panel, int).ravel()
    n = n[np.isfinite(n) & (n > 0)]
    if n.size == 0:
        return None
    uniq = np.unique(n)
    if uniq.size == 1:
        return int(uniq[0])
    return None


def _constant_group_n(x: ExportedTrainingScalarBars) -> int | None:
    """
    Return a group's single constant positive n across all panels, or None if the
    sample size varies by panel or is unavailable.
    """
    if x.n_units_panel is None:
        return None
    n = np.asarray(x.n_units_panel, int).ravel()
    n = n[np.isfinite(n) & (n > 0)]
    if n.size == 0:
        return None
    uniq = np.unique(n)
    if uniq.size != 1:
        return None
    return int(uniq[0])


def _global_constant_legend_n(xs: list[ExportedTrainingScalarBars]) -> int | None:
    """
    Return a single legend n only when every plotted group has the same constant
    positive n across all panels. Otherwise return None and prefer per-panel labels.
    """
    if not xs:
        return None
    ns = []
    for x in xs:
        n = _constant_group_n(x)
        if n is None:
            return None
        ns.append(int(n))
    uniq = np.unique(np.asarray(ns, dtype=int))
    if uniq.size != 1:
        return None
    return int(uniq[0])


def _all_groups_have_constant_legend_n(
    xs: list[ExportedTrainingScalarBars],
) -> list[int] | None:
    if not xs:
        return None
    ns = []
    for x in xs:
        n = _legend_n_for_group(x)
        if n is None:
            return None
        ns.append(int(n))
    return ns


def _legend_n_from_paired(paired_n_per_panel: np.ndarray, P: int) -> int | None:
    if paired_n_per_panel is None or P <= 0:
        return None
    if P == 1:
        return int(paired_n_per_panel[0])

    nz = paired_n_per_panel[(paired_n_per_panel > 0) & np.isfinite(paired_n_per_panel)]
    if nz.size == 0:
        return None
    uniq = np.unique(nz)
    return int(uniq[0]) if uniq.size == 1 else None


def _panel_n_label(x: ExportedTrainingScalarBars, p_idx: int) -> str:
    if x.n_units_panel is not None and x.n_units_panel.shape[0] > p_idx:
        return str(int(x.n_units_panel[p_idx]))
    return "?"


def _per_panel_n_text(
    ns: list[int],
    *,
    compact_n_labels: bool,
) -> str | None:
    ns = [int(n) for n in ns if n is not None and int(n) > 0]
    if not ns:
        return None
    uniq = sorted(set(ns))
    if len(uniq) == 1:
        return f"n={uniq[0]}"
    joined = "/".join(map(str, ns))
    return joined if compact_n_labels else f"n={joined}"


def _draw_panel_n_labels(
    ax: plt.Axes,
    *,
    x_centers: np.ndarray,
    hi_by_group: list[np.ndarray],
    panel_n_texts: list[str | None],
    bracket_tops: np.ndarray | None,
    annotation_font_size: float,
    compact_n_labels: bool,
    dense_bar_layout: bool,
    font_scale: float,
    n_label_rotation: float,
) -> None:
    if not panel_n_texts:
        return

    ylim0, ylim1 = ax.get_ylim()
    y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    ax_h_px = max(float(ax_bbox.height), 1.0)
    data_per_px = y_rng / ax_h_px
    font_px = float(annotation_font_size) * float(fig.dpi) / 72.0
    text_clearance = (0.45 * font_px + 2.0) * data_per_px
    y_pad = (
        (0.024 + 0.016 * max(font_scale - 1.0, 0.0)) * y_rng + text_clearance
    )

    for p, n_text in enumerate(panel_n_texts):
        if not n_text:
            continue

        y_top = np.nan
        for hi in hi_by_group:
            if p < hi.shape[0] and np.isfinite(hi[p]):
                y_top = float(hi[p]) if not np.isfinite(y_top) else max(float(y_top), float(hi[p]))
        if bracket_tops is not None and p < bracket_tops.shape[0] and np.isfinite(bracket_tops[p]):
            y_top = (
                float(bracket_tops[p])
                if not np.isfinite(y_top)
                else max(float(y_top), float(bracket_tops[p]))
            )
        if not np.isfinite(y_top):
            continue

        ax.text(
            float(x_centers[p]),
            float(
                y_top
                + y_pad
                + ((0.018 * y_rng) if dense_bar_layout and (p % 2 == 1) else 0.0)
            ),
            n_text,
            ha="center",
            va="bottom",
            fontsize=annotation_font_size,
            rotation=n_label_rotation,
            color="0.2",
            clip_on=False,
            zorder=9,
        )


def _ensure_top_text_visible(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    texts: list,
    top_pad_px: float = 4.0,
    max_iter: int = 3,
) -> None:
    if not texts:
        return

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        ax_h_px = max(float(ax_bbox.height), 1.0)
        ylim0, ylim1 = ax.get_ylim()
        y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0

        overflow_px = 0.0
        for txt in texts:
            if not txt.get_visible():
                continue
            bbox = txt.get_window_extent(renderer=renderer)
            overflow_px = max(overflow_px, float(bbox.y1 - (ax_bbox.y1 - top_pad_px)))

        if overflow_px <= 0.0:
            break

        extra_data = (overflow_px / ax_h_px) * y_rng + 0.01 * y_rng
        ax.set_ylim(ylim0, ylim1 + extra_data)


def _expand_bbox(bbox, pad_px: float):
    return bbox.expanded(
        (float(bbox.width) + 2.0 * pad_px) / max(float(bbox.width), 1.0),
        (float(bbox.height) + 2.0 * pad_px) / max(float(bbox.height), 1.0),
    )


def _ensure_legend_clear_of_annotations(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    legend,
    texts: list,
    lines: list,
    pad_px: float = 6.0,
    max_iter: int = 4,
) -> None:
    if legend is None:
        return
    if not texts and not lines:
        return

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_bbox = _expand_bbox(legend_bbox, pad_px=float(pad_px))
        ax_bbox = ax.get_window_extent(renderer=renderer)
        ax_h_px = max(float(ax_bbox.height), 1.0)
        ylim0, ylim1 = ax.get_ylim()
        y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0

        overlap_px = 0.0
        for artist in [*texts, *lines]:
            if hasattr(artist, "get_visible") and not artist.get_visible():
                continue
            try:
                bbox = artist.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bbox is None:
                continue
            if legend_bbox.overlaps(bbox):
                overlap_px = max(
                    overlap_px,
                    float(min(legend_bbox.y1, bbox.y1) - max(legend_bbox.y0, bbox.y0)),
                )

        if overlap_px <= 0.0:
            break

        extra_data = (overlap_px / ax_h_px) * y_rng + 0.02 * y_rng
        ax.set_ylim(ylim0, ylim1 + extra_data)


def _group_union_matrix(
    x: ExportedTrainingScalarBars,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (N_union, P) matrix of per-unit values and a parallel (N_union,) id array.
    Missing values are NaN.
    """
    P = len(x.panel_labels)

    # union of IDs across panels (finite values only)
    union = set()
    for p in range(P):
        ids = np.asarray(x.per_unit_ids_panel[p], dtype=object).ravel()
        vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
        mask = np.isfinite(vals) & (ids != None)
        union |= {str(i) for i in ids[mask]}

    union_ids = sorted(union)
    idx = {k: i for i, k in enumerate(union_ids)}
    M = np.full((len(union_ids), P), np.nan, dtype=float)

    for p in range(P):
        ids = np.asarray(x.per_unit_ids_panel[p], dtype=object).ravel()
        vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
        for i0, v0 in zip(ids, vals):
            if i0 is None:
                continue
            v = float(v0)
            if not np.isfinite(v):
                continue
            r = idx.get(str(i0), None)
            if r is not None:
                M[r, p] = v

    return M, np.asarray(union_ids, dtype=object)


def plot_overlays(
    xs: list[ExportedTrainingScalarBars],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
    bar_alpha: float = 0.90,
    opts=None,
) -> plt.Figure:
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)

    validate_alignment(xs)

    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))
    font_scale = max(float(customizer.increase_factor), 1.0)
    annotation_font_size = max(
        7,
        min(float(customizer.in_plot_font_size) - 5.0, 10.0),
    )
    legend_font_size = max(
        8,
        min(float(customizer.in_plot_font_size), 14.0),
    )

    panel_labels = xs[0].panel_labels
    P = len(panel_labels)
    G = len(xs)
    if P >= 10:
        width_scale = min(1.0 + 0.32 * (font_scale - 1.0), 1.55)
    elif P >= 8:
        width_scale = min(1.0 + 0.24 * (font_scale - 1.0), 1.40)
    else:
        width_scale = min(1.0 + 0.10 * (font_scale - 1.0), 1.20)
    fig_w = max(6.0, 1.2 * P * width_scale)
    fig_h = 4.5 * min(1.0 + 0.12 * (font_scale - 1.0), 1.24)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    x_centers = np.arange(P, dtype=float)
    dense_bar_layout = P >= 10
    compact_n_labels = dense_bar_layout and font_scale >= 1.15
    n_label_rotation = 28 if compact_n_labels else 0

    # grouped/dodged bars
    group_band = 0.80
    bar_w = group_band / max(1, G)
    gpos = np.arange(G) - (G - 1) / 2.0
    offsets = gpos * bar_w  # (G,)

    # Per-group union matrices for stats (and paired recompute)
    mats = []
    ids_union = []
    for x in xs:
        M, I = _group_union_matrix(x)
        mats.append(M)
        ids_union.append(I)

    # If paired plotting, recompute displayed mean/CI per panel per group from paired-only
    paired_n_per_panel = None
    means_plot = []
    lo_plot = []
    hi_plot = []

    xpos_by_group = []
    hi_by_group = []
    per_unit_by_group = []
    per_unit_ids_by_group = []

    if stats and stats_paired:
        # Build paired-only matrices/IDs across all panels (so stats and plotting align)
        mats_paired, ids_paired, paired_n_per_panel = _paired_filter_mats_all_panels(
            mats, ids_union, P=P
        )

        # Overwrite what stats sees (paired-only)
        per_unit_by_group = [m for m in mats_paired]
        per_unit_ids_by_group = [ids_paired for _ in mats_paired]

        # Recompute displayed mean/CI from the paired-only matrices
        means_plot = []
        lo_plot = []
        hi_plot = []
        for out in mats_paired:
            m = np.full((P,), np.nan, float)
            lo = np.full((P,), np.nan, float)
            hi = np.full((P,), np.nan, float)
            for p in range(P):
                mm, l0, h0, _n = _mean_ci_from_util(out[:, p], conf=xs[0].ci_conf)
                m[p] = mm
                lo[p] = l0
                hi[p] = h0
            means_plot.append(m)
            lo_plot.append(lo)
            hi_plot.append(hi)
    else:
        for x in xs:
            means_plot.append(np.asarray(x.mean, float))
            lo_plot.append(np.asarray(x.ci_lo, float))
            hi_plot.append(np.asarray(x.ci_hi, float))

    per_group_legend_ns = None
    global_legend_n = None
    if stats and stats_paired and paired_n_per_panel is not None:
        global_legend_n = _legend_n_from_paired(paired_n_per_panel, P)
    else:
        per_group_legend_ns = _all_groups_have_constant_legend_n(xs)

    for gi, x in enumerate(xs):
        xg = x_centers + offsets[gi]
        xpos_by_group.append(np.asarray(xg, float))

        y = np.asarray(means_plot[gi], float)
        y_plot = np.where(np.isfinite(y), y, 0.0)
        n_leg = global_legend_n
        if per_group_legend_ns is not None and gi < len(per_group_legend_ns):
            n_leg = per_group_legend_ns[gi]
        label = f"{x.group} (n={n_leg})" if n_leg is not None else f"{x.group}"
        bar_color = group_fill_color(gi)
        edge_color = group_accent_color(gi)
        ax.bar(
            xg,
            y_plot,
            width=bar_w,
            align="center",
            label=label,
            color=bar_color,
            edgecolor=edge_color,
            alpha=bar_alpha,
            linewidth=0.9,
        )

        # CI whiskers (only if export had CI; in paired-mode we recomputed them anyway)
        lo = np.asarray(lo_plot[gi], float)
        hi = np.asarray(hi_plot[gi], float)
        mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if np.any(mask):
            ax.errorbar(
                xg[mask],
                y[mask],
                yerr=np.vstack([y[mask] - lo[mask], hi[mask] - y[mask]]),
                fmt="none",
                ecolor=NEUTRAL_DARK,
                capsize=3,
                capthick=1.0,
                elinewidth=1.0,
                alpha=0.9,
                zorder=3,
            )

        # baseline for brackets
        hi_by_group.append(np.where(np.isfinite(hi), hi, y))

        # stats payload
        if not (stats and stats_paired):
            per_unit_by_group.append(mats[gi])  # (N_union, P)
            per_unit_ids_by_group.append(ids_union[gi])  # (N_union,)

    panel_n_texts: list[str | None] = []
    bracket_tops = None
    base_text_count = len(ax.texts)
    base_line_count = len(ax.lines)

    tick_rotation = 30
    if P >= 8:
        tick_rotation = 40
    if P >= 10 and font_scale >= 1.15:
        tick_rotation = 55
    if P >= 10 and font_scale >= 1.35:
        tick_rotation = 60
    ax.set_xticks(x_centers)
    ax.set_xticklabels(panel_labels, rotation=tick_rotation, ha="right")

    if xlabel:
        xlabel_use = str(xlabel)
        if (
            dense_bar_layout
            and font_scale >= 1.15
            and "\n" not in xlabel_use
            and len(xlabel_use) >= 36
        ):
            if " from " in xlabel_use:
                xlabel_use = xlabel_use.replace(" from ", "\nfrom ", 1)
            elif " (" in xlabel_use:
                xlabel_use = xlabel_use.replace(" (", "\n(", 1)
        ax.set_xlabel(xlabel_use)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))

    if stats:
        cfg_stats = StatAnnotConfig(
            alpha=float(stats_alpha),
            min_n_per_group=3,
            headroom_frac=0.30 + 0.10 * max(font_scale - 1.0, 0.0),
            stack_gap_frac=0.050 + 0.014 * max(font_scale - 1.0, 0.0),
            gap_above_bars_frac=0.045 + 0.012 * max(font_scale - 1.0, 0.0),
            nlabel_off_frac=0.0,
            bracket_fontsize=annotation_font_size,
        )
        bracket_tops = annotate_grouped_bars_per_bin(
            ax,
            x_centers=x_centers,
            xpos_by_group=xpos_by_group,
            per_unit_by_group=per_unit_by_group,
            per_unit_ids_by_group=per_unit_ids_by_group,
            hi_by_group=hi_by_group,
            group_names=[x.group for x in xs],
            cfg=cfg_stats,
            paired=bool(stats_paired),
            panel_label=None,
            debug=debug,
        )

        # Optional: if paired and n varies by panel, annotate n above bars
        if stats_paired and paired_n_per_panel is not None:
            nz = paired_n_per_panel[paired_n_per_panel > 0]
            uniq = np.unique(nz) if nz.size else np.asarray([], int)
            n_constant = int(uniq[0]) if uniq.size == 1 else None

            if n_constant is None:
                panel_n_texts = [None] * P
                for p in range(P):
                    npp = int(paired_n_per_panel[p])
                    if npp <= 0:
                        continue
                    panel_n_texts[p] = f"n={npp}"
        elif per_group_legend_ns is None:
            for p in range(P):
                ns = []
                for x in xs:
                    npp = _panel_n(x, p)
                    if npp is not None and npp > 0:
                        ns.append(int(npp))
                panel_n_texts.append(
                    _per_panel_n_text(ns, compact_n_labels=compact_n_labels)
                )
    elif per_group_legend_ns is None:
        for p in range(P):
            ns = []
            for x in xs:
                npp = _panel_n(x, p)
                if npp is not None and npp > 0:
                    ns.append(int(npp))
            panel_n_texts.append(
                _per_panel_n_text(ns, compact_n_labels=compact_n_labels)
            )

    if panel_n_texts:
        _draw_panel_n_labels(
            ax,
            x_centers=x_centers,
            hi_by_group=hi_by_group,
            panel_n_texts=panel_n_texts,
            bracket_tops=bracket_tops,
            annotation_font_size=annotation_font_size,
            compact_n_labels=compact_n_labels,
            dense_bar_layout=dense_bar_layout,
            font_scale=font_scale,
            n_label_rotation=n_label_rotation,
        )

    if title:
        fig.suptitle(title)

    legend = ax.legend(fontsize=legend_font_size)
    if customizer.customized:
        customizer.adjust_padding_proportionally()
        xlabel_text = ax.xaxis.get_label().get_text()
        xlabel_lines = max(1, str(xlabel_text).count("\n") + 1)
        long_xlabel = len(str(xlabel_text).replace("\n", " ")) >= 36
        if dense_bar_layout or (long_xlabel and font_scale >= 1.15):
            bottom = 0.13 + 0.05 * max(font_scale - 1.0, 0.0)
            if xlabel_lines == 1 and long_xlabel and font_scale >= 1.15:
                bottom += 0.03
            if xlabel_lines >= 2:
                bottom += 0.05
            right = 0.97
            top = 0.96
            fig.subplots_adjust(
                bottom=min(bottom, 0.28),
                right=right,
                top=top,
            )
    else:
        fig.tight_layout()
    _ensure_xlabel_visible(fig, ax)
    _ensure_legend_clear_of_annotations(
        fig,
        ax,
        legend=legend,
        texts=ax.texts[base_text_count:],
        lines=ax.lines[base_line_count:],
    )
    _ensure_top_text_visible(fig, ax, texts=ax.texts[base_text_count:])
    return fig
