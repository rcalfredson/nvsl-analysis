# src/plotting/overlay_training_metric_scalar_bars.py

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.stats import ttest_ind

from src.plotting.palettes import (
    NEUTRAL_DARK,
    NEUTRAL_MID,
    group_metric_edge_color,
    group_metric_edge_color_for_label,
    group_metric_fill_color,
    group_metric_fill_color_for_label,
    normalize_metric_palette_family,
)
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import (
    StatAnnotConfig,
    annotate_grouped_bars_per_bin,
    draw_sig_bracket,
    format_sig_label,
    holm_adjust,
)
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


@dataclass(frozen=True)
class OmnibusLearnerEntry:
    learner: str
    genotype: str
    export: ExportedTrainingScalarBars


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


def _italicize_xtick_labels(ax: plt.Axes) -> None:
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")


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
    x_ok = bbox.x0 >= fig_bbox.x0 + pad_x_px and bbox.x1 <= fig_bbox.x1 - pad_x_px
    y_ok = bbox.y0 >= fig_bbox.y0 + pad_y_px

    if x_ok and y_ok:
        return

    wrapped = _wrapped_xlabel_text(label.get_text())
    if wrapped != label.get_text():
        label.set_text(wrapped)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = label.get_window_extent(renderer=renderer)
        x_ok = bbox.x0 >= fig_bbox.x0 + pad_x_px and bbox.x1 <= fig_bbox.x1 - pad_x_px
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


def _wrapped_ylabel_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if "Rewards per distance traveled" in text:
        return text.replace(
            "Rewards per distance traveled",
            "Rewards per\ndistance traveled",
            1,
        )
    for phrase in (
        " without wall contact",
        " between-reward ",
        " return-leg ",
    ):
        if phrase in text:
            before, after = text.split(phrase, 1)
            return f"{before}\n{phrase.strip()} {after}".rstrip()
    if ", " in text:
        return text.replace(", ", ",\n", 1)
    if " per " in text:
        return text.replace(" per ", "\nper ", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    return text


def _ensure_ylabel_visible(fig: plt.Figure, ax: plt.Axes) -> None:
    label = ax.yaxis.get_label()
    if not label.get_text():
        return

    pad_x_px = max(8.0, 0.55 * float(label.get_fontsize()) + 4.0)
    pad_y_px = max(10.0, 0.45 * float(label.get_fontsize()) + 4.0)
    wrapped = _wrapped_ylabel_text(label.get_text())
    if wrapped != label.get_text():
        label.set_text(wrapped)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    def _within_bounds() -> bool:
        bbox = label.get_window_extent(renderer=renderer)
        x_ok = bbox.x0 >= fig_bbox.x0 + pad_x_px
        y_ok = bbox.y0 >= fig_bbox.y0 + pad_y_px and bbox.y1 <= fig_bbox.y1 - pad_y_px
        return x_ok and y_ok

    if _within_bounds():
        return

    if _within_bounds():
        return

    bbox = label.get_window_extent(renderer=renderer)
    overflow_left_px = max((fig_bbox.x0 + pad_x_px) - bbox.x0, 0.0)
    overflow_bottom_px = max((fig_bbox.y0 + pad_y_px) - bbox.y0, 0.0)
    overflow_top_px = max(bbox.y1 - (fig_bbox.y1 - pad_y_px), 0.0)

    fig_h_px = max(fig.get_size_inches()[1] * fig.dpi, 1.0)
    fig_w_px = max(fig.get_size_inches()[0] * fig.dpi, 1.0)

    extra_left = float(overflow_left_px / fig_w_px) + 0.01
    extra_bottom = float(overflow_bottom_px / fig_h_px) + 0.005
    extra_top = float(overflow_top_px / fig_h_px) + 0.005

    new_left = min(fig.subplotpars.left + extra_left, 0.28)
    new_bottom = min(fig.subplotpars.bottom + extra_bottom, 0.20)
    new_top = max(fig.subplotpars.top - extra_top, 0.84)
    if new_top <= new_bottom:
        new_bottom = fig.subplotpars.bottom
        new_top = fig.subplotpars.top

    fig.subplots_adjust(left=new_left, bottom=new_bottom, top=new_top)
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


def _scalar_bar_y_limits(
    *,
    means,
    lows,
    highs,
    samples=None,
    ymax: float | None = None,
) -> tuple[float, float]:
    vals = []
    for seq in (means, lows, highs):
        vals.extend(np.asarray(seq, dtype=float).ravel().tolist())
    if samples is not None:
        for sample in samples:
            vals.extend(np.asarray(sample, dtype=float).ravel().tolist())

    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, float(ymax) if ymax is not None else 1.0

    data_min = min(float(np.min(finite)), 0.0)
    data_max = max(float(np.max(finite)), 0.0)
    span = data_max - data_min
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(data_max), abs(data_min), 1.0)

    bottom = data_min - (0.10 * span if data_min < 0.0 else 0.0)
    top = float(ymax) if ymax is not None else data_max + 0.16 * span
    if top <= bottom:
        top = bottom + max(span, 1.0)
    return float(bottom), float(top)


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


def _metric_palette_family_from_exports(
    xs: list[ExportedTrainingScalarBars],
) -> str | None:
    families = []
    for x in xs:
        meta = dict(x.meta or {})
        family = normalize_metric_palette_family(
            meta.get("metric_palette_family") or meta.get("metric")
        )
        if family is not None:
            families.append(family)
    if not families:
        return None
    uniq = []
    for family in families:
        if family not in uniq:
            uniq.append(family)
    if len(uniq) == 1:
        return uniq[0]
    return None


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


def baseline_delta_exports(
    xs: list[ExportedTrainingScalarBars],
    *,
    baseline_panel: str,
    target_panels: list[str] | tuple[str, ...] | None = None,
) -> list[ExportedTrainingScalarBars]:
    if not xs:
        return []

    labels0 = xs[0].panel_labels
    if baseline_panel not in labels0:
        raise ValueError(
            f"baseline panel {baseline_panel!r} not found; available: {labels0!r}"
        )
    baseline_idx = labels0.index(baseline_panel)

    if target_panels is None:
        target_panels = [label for label in labels0 if label != baseline_panel]
    target_panels = [str(label) for label in target_panels]
    missing = [label for label in target_panels if label not in labels0]
    if missing:
        raise ValueError(
            f"target panel(s) {missing!r} not found; available: {labels0!r}"
        )
    target_idxs = [labels0.index(label) for label in target_panels]
    delta_labels = [f"{label} - {baseline_panel}" for label in target_panels]

    out: list[ExportedTrainingScalarBars] = []
    for x in xs:
        if x.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [y.panel_labels for y in xs])
            )
        matrix, unit_ids = _group_union_matrix(x)
        per_panel_values = []
        per_panel_ids = []
        means = []
        lows = []
        highs = []
        ns = []
        for target_idx in target_idxs:
            vals = matrix[:, target_idx] - matrix[:, baseline_idx]
            mask = np.isfinite(vals)
            vals_finite = vals[mask].astype(float, copy=False)
            ids_finite = unit_ids[mask].astype(object, copy=False)
            mean, lo, hi, n = _mean_ci_from_util(vals_finite, conf=x.ci_conf)
            per_panel_values.append(vals_finite)
            per_panel_ids.append(ids_finite)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
            ns.append(n)

        meta = dict(x.meta or {})
        meta["baseline_delta_panel"] = baseline_panel
        meta["baseline_delta_target_panels"] = list(target_panels)
        meta["y_label"] = f"Change in {meta.get('y_label', 'value')}"
        meta["base_title"] = f"Change in {meta.get('base_title', 'overlay bars')}"

        out.append(
            ExportedTrainingScalarBars(
                group=x.group,
                panel_labels=delta_labels,
                per_unit_values_panel=np.asarray(per_panel_values, dtype=object),
                per_unit_ids_panel=np.asarray(per_panel_ids, dtype=object),
                mean=np.asarray(means, dtype=float),
                ci_lo=np.asarray(lows, dtype=float),
                ci_hi=np.asarray(highs, dtype=float),
                n_units_panel=np.asarray(ns, dtype=int),
                meta=meta,
            )
        )
    return out


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
    y_pad = (0.024 + 0.016 * max(font_scale - 1.0, 0.0)) * y_rng + text_clearance

    for p, n_text in enumerate(panel_n_texts):
        if not n_text:
            continue

        y_top = np.nan
        for hi in hi_by_group:
            if p < hi.shape[0] and np.isfinite(hi[p]):
                y_top = (
                    float(hi[p])
                    if not np.isfinite(y_top)
                    else max(float(y_top), float(hi[p]))
                )
        if (
            bracket_tops is not None
            and p < bracket_tops.shape[0]
            and np.isfinite(bracket_tops[p])
        ):
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


def _artist_bbox(artist, renderer):
    if hasattr(artist, "get_visible") and not artist.get_visible():
        return None
    try:
        return artist.get_window_extent(renderer=renderer)
    except Exception:
        return None


def _requested_ymax_is_clear(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    requested_ymax: float,
    texts: list,
    lines: list,
    legend,
    top_pad_px: float = 4.0,
    legend_pad_px: float = 6.0,
) -> bool:
    ylim0, ylim1 = ax.get_ylim()
    if not np.isfinite(requested_ymax) or requested_ymax <= ylim0:
        return False

    try:
        ax.set_ylim(ylim0, requested_ymax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)

        for txt in texts:
            bbox = _artist_bbox(txt, renderer)
            if bbox is None:
                continue
            if float(bbox.y1) > float(ax_bbox.y1) - float(top_pad_px):
                return False

        if legend is not None:
            legend_bbox = legend.get_window_extent(renderer=renderer)
            legend_bbox = _expand_bbox(legend_bbox, pad_px=float(legend_pad_px))
            for artist in [*texts, *lines]:
                bbox = _artist_bbox(artist, renderer)
                if bbox is None:
                    continue
                if legend_bbox.overlaps(bbox):
                    return False

        return True
    finally:
        ax.set_ylim(ylim0, ylim1)


def _restore_requested_ymax_if_clear(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ymax: float | None,
    texts: list,
    lines: list,
    legend,
) -> None:
    if ymax is None:
        return

    requested_ymax = float(ymax)
    ylim0, ylim1 = ax.get_ylim()
    if not np.isfinite(ylim1) or ylim1 <= requested_ymax + 1e-10:
        return

    if _requested_ymax_is_clear(
        fig,
        ax,
        requested_ymax=requested_ymax,
        texts=texts,
        lines=lines,
        legend=legend,
    ):
        ax.set_ylim(ylim0, requested_ymax)


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


def _first_panel_samples(x: ExportedTrainingScalarBars) -> np.ndarray:
    if len(x.per_unit_values_panel) == 0:
        return np.asarray([], dtype=float)
    vals = np.asarray(x.per_unit_values_panel[0], dtype=float).ravel()
    return vals[np.isfinite(vals)]


def _first_panel_mean_ci(
    x: ExportedTrainingScalarBars,
) -> tuple[float, float, float, int]:
    y = float(np.asarray(x.mean, dtype=float).ravel()[0])
    lo = float(np.asarray(x.ci_lo, dtype=float).ravel()[0])
    hi = float(np.asarray(x.ci_hi, dtype=float).ravel()[0])
    n = _panel_n(x, 0)
    return y, lo, hi, int(n) if n is not None else 0


def _default_learner_order(entries: list[OmnibusLearnerEntry]) -> list[str]:
    labels = []
    for entry in entries:
        if entry.learner not in labels:
            labels.append(entry.learner)

    def key(label: str) -> tuple[int, str]:
        lower = label.lower()
        if lower.startswith("top"):
            return (0, lower)
        if lower.startswith("bottom"):
            return (1, lower)
        return (2, lower)

    return sorted(labels, key=key)


def _default_genotype_order(entries: list[OmnibusLearnerEntry]) -> list[str]:
    labels = []
    for entry in entries:
        if entry.genotype not in labels:
            labels.append(entry.genotype)
    return labels


def _draw_omnibus_stats(
    ax: plt.Axes,
    *,
    samples_by_key: dict[tuple[str, str], np.ndarray],
    x_by_key: dict[tuple[str, str], float],
    hi_by_key: dict[tuple[str, str], float],
    learner_order: list[str],
    genotype_order: list[str],
    alpha: float,
    fontsize: float,
    debug: bool,
) -> np.ndarray:
    comparisons: list[tuple[tuple[str, str], tuple[str, str], str]] = []
    for learner in learner_order:
        for i, genotype_a in enumerate(genotype_order):
            for genotype_b in genotype_order[i + 1 :]:
                comparisons.append(
                    ((learner, genotype_a), (learner, genotype_b), "across genotype")
                )
    for genotype in genotype_order:
        for i, learner_a in enumerate(learner_order):
            for learner_b in learner_order[i + 1 :]:
                comparisons.append(
                    ((learner_a, genotype), (learner_b, genotype), "within genotype")
                )

    valid: list[tuple[tuple[str, str], tuple[str, str], str, float]] = []
    for key_a, key_b, family in comparisons:
        xa = samples_by_key.get(key_a, np.asarray([], dtype=float))
        xb = samples_by_key.get(key_b, np.asarray([], dtype=float))
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]
        p = np.nan
        if xa.size >= 3 and xb.size >= 3:
            try:
                _, p = ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
            except Exception:
                p = np.nan
        valid.append((key_a, key_b, family, float(p)))

    p_adj = holm_adjust([item[3] for item in valid])
    if debug:
        for (key_a, key_b, family, p_raw), p_holm in zip(valid, p_adj):
            xa = samples_by_key.get(key_a, np.asarray([], dtype=float))
            xb = samples_by_key.get(key_b, np.asarray([], dtype=float))
            print(
                "[omnibus_stats] "
                f"{key_a[0]} {key_a[1]} vs {key_b[0]} {key_b[1]} "
                f"({family}, Welch, n={np.isfinite(xa).sum()}/{np.isfinite(xb).sum()}): "
                f"raw p={p_raw:.6g}, Holm p={float(p_holm):.6g}"
            )

    ylim0, ylim1 = ax.get_ylim()
    y_rng0 = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    ax.set_ylim(ylim0, ylim1 + 0.46 * y_rng0)
    ylim0, ylim1 = ax.get_ylim()
    y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    bracket_h = 0.012 * y_rng
    step = 0.075 * y_rng
    gap = 0.050 * y_rng
    tops: list[tuple[float, float, float]] = []
    top_by_bar = np.full((len(x_by_key),), np.nan, dtype=float)
    bar_keys = list(x_by_key)

    sig_items = [
        (key_a, key_b, p_holm)
        for (key_a, key_b, _family, _p_raw), p_holm in zip(valid, p_adj)
        if np.isfinite(p_holm) and float(p_holm) < float(alpha)
    ]
    sig_items = sorted(
        sig_items,
        key=lambda item: abs(float(x_by_key[item[0]]) - float(x_by_key[item[1]])),
    )

    for key_a, key_b, p_holm in sig_items:
        if key_a not in x_by_key or key_b not in x_by_key:
            continue
        x1 = float(x_by_key[key_a])
        x2 = float(x_by_key[key_b])
        if x2 < x1:
            x1, x2 = x2, x1
        y_base = max(
            float(hi_by_key.get(key_a, np.nan)),
            float(hi_by_key.get(key_b, np.nan)),
        )
        if not np.isfinite(y_base):
            continue
        y = y_base + gap
        for existing_y, existing_x1, existing_x2 in tops:
            overlaps = not (x2 < existing_x1 or x1 > existing_x2)
            if overlaps and y <= existing_y + step:
                y = existing_y + step
        label = format_sig_label(float(p_holm))
        if not label:
            continue
        draw_sig_bracket(
            ax,
            x1=x1,
            x2=x2,
            y=y,
            h=bracket_h,
            text=label,
            fontsize=fontsize,
        )
        tops.append((y, x1, x2))
        y_top = y + bracket_h + 0.035 * y_rng
        for key in (key_a, key_b):
            try:
                idx = bar_keys.index(key)
            except ValueError:
                continue
            top_by_bar[idx] = (
                y_top
                if not np.isfinite(top_by_bar[idx])
                else max(top_by_bar[idx], y_top)
            )

    return top_by_bar


def plot_omnibus_learner_overlays(
    entries: list[OmnibusLearnerEntry],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    ytick_step: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    debug: bool = False,
    bar_alpha: float = 0.90,
    opts=None,
) -> plt.Figure:
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)
    if not entries:
        raise ValueError("At least one omnibus learner entry is required.")

    panel_labels = {tuple(entry.export.panel_labels) for entry in entries}
    if len(panel_labels) != 1:
        raise ValueError(
            _fmt_mismatch(
                "panel_labels", [entry.export.panel_labels for entry in entries]
            )
        )

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
    xtick_font_size = max(
        8,
        min(float(customizer.in_plot_font_size) - 2.0, 12.0),
    )
    cluster_label_font_size = max(
        8,
        min(float(customizer.in_plot_font_size) - 3.0, 11.0),
    )

    learner_order = _default_learner_order(entries)
    genotype_order = _default_genotype_order(entries)
    by_key = {(entry.learner, entry.genotype): entry.export for entry in entries}
    missing = [
        (learner, genotype)
        for learner in learner_order
        for genotype in genotype_order
        if (learner, genotype) not in by_key
    ]
    if missing:
        raise ValueError(f"Missing omnibus learner/genotype entries: {missing}")

    cluster_gap = 1.35
    within_gap = 0.88
    x_by_key: dict[tuple[str, str], float] = {}
    keys_in_plot_order: list[tuple[str, str]] = []
    for li, learner in enumerate(learner_order):
        base = li * ((len(genotype_order) - 1) * within_gap + cluster_gap)
        for gi, genotype in enumerate(genotype_order):
            key = (learner, genotype)
            x_by_key[key] = float(base + gi * within_gap)
            keys_in_plot_order.append(key)

    fig_w = max(
        7.2,
        1.05 * len(keys_in_plot_order) * min(1.0 + 0.12 * (font_scale - 1.0), 1.25),
    )
    fig_h = 4.8 * min(1.0 + 0.12 * (font_scale - 1.0), 1.24)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    xs_plot = []
    means = []
    lows = []
    highs = []
    tick_labels = []
    samples_by_key: dict[tuple[str, str], np.ndarray] = {}
    hi_by_key: dict[tuple[str, str], float] = {}
    for key in keys_in_plot_order:
        x = by_key[key]
        y, lo, hi, n = _first_panel_mean_ci(x)
        xs_plot.append(x_by_key[key])
        means.append(y)
        lows.append(lo)
        highs.append(hi)
        hi_by_key[key] = hi if np.isfinite(hi) else y
        samples_by_key[key] = _first_panel_samples(x)
        tick_labels.append(f"{key[1]}\n(n={n})" if n > 0 else key[1])

    genotype_color = {genotype: gi for gi, genotype in enumerate(genotype_order)}
    for key, xpos, y, lo, hi in zip(keys_in_plot_order, xs_plot, means, lows, highs):
        gi = genotype_color[key[1]]
        learner_alpha = (
            bar_alpha if key[0] == learner_order[0] else min(bar_alpha, 0.68)
        )
        ax.bar(
            xpos,
            0.0 if not np.isfinite(y) else y,
            width=0.58,
            color=group_metric_fill_color(gi, "wall_contacts"),
            edgecolor=group_metric_edge_color(gi, "wall_contacts"),
            linewidth=0.9,
            alpha=learner_alpha,
            label=key[1] if key[0] == learner_order[0] else None,
        )
        if np.isfinite(y) and np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(
                [xpos],
                [y],
                yerr=np.asarray([[y - lo], [hi - y]], dtype=float),
                fmt="none",
                ecolor=NEUTRAL_DARK,
                capsize=3,
                capthick=1.0,
                elinewidth=1.0,
                alpha=0.9,
                zorder=3,
            )

    ax.set_xticks(np.asarray(xs_plot, dtype=float))
    ax.set_xticklabels(
        tick_labels,
        rotation=34,
        ha="right",
        rotation_mode="anchor",
        fontsize=xtick_font_size,
    )
    _italicize_xtick_labels(ax)

    ylim_bottom, ylim_top = _scalar_bar_y_limits(
        means=means,
        lows=lows,
        highs=highs,
        samples=samples_by_key.values(),
        ymax=ymax,
    )
    ax.set_ylim(ylim_bottom, ylim_top)
    if ylim_bottom < 0.0 < ylim_top:
        ax.axhline(0.0, color=NEUTRAL_MID, linewidth=0.8, alpha=0.65, zorder=0)
    bracket_tops = None
    base_text_count = len(ax.texts)
    if stats:
        bracket_tops = _draw_omnibus_stats(
            ax,
            samples_by_key=samples_by_key,
            x_by_key=x_by_key,
            hi_by_key=hi_by_key,
            learner_order=learner_order,
            genotype_order=genotype_order,
            alpha=float(stats_alpha),
            fontsize=annotation_font_size,
            debug=debug,
        )

    cluster_centers = []
    for learner in learner_order:
        cluster_xs = [x_by_key[(learner, genotype)] for genotype in genotype_order]
        cluster_centers.append(float(np.mean(cluster_xs)))
    for center, learner in zip(cluster_centers, learner_order):
        ax.text(
            center,
            -0.30,
            learner,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=cluster_label_font_size,
            color=NEUTRAL_DARK,
            clip_on=False,
        )
    if len(cluster_centers) > 1:
        for left, right in zip(cluster_centers[:-1], cluster_centers[1:]):
            ax.axvline(
                (left + right) / 2.0,
                color=NEUTRAL_MID,
                linewidth=0.8,
                alpha=0.35,
            )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    if ytick_step is not None and float(ytick_step) > 0:
        step = float(ytick_step)
        decimals = next(
            (
                precision
                for precision in range(7)
                if np.isclose(step, round(step, precision), rtol=0.0, atol=1e-10)
            ),
            6,
        )
        if ymax is not None:
            ticks = np.arange(0.0, float(ymax) + 0.5 * step, step)
            ticks = ticks[ticks <= float(ymax) + 1e-10]
            ax.set_yticks(ticks)
        else:
            ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        prop={"style": "italic", "size": legend_font_size},
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.34), right=0.78)
    _ensure_xlabel_visible(fig, ax)
    _ensure_ylabel_visible(fig, ax)
    _ensure_legend_clear_of_annotations(
        fig,
        ax,
        legend=legend,
        texts=ax.texts[base_text_count:],
        lines=[],
    )
    _ensure_top_text_visible(fig, ax, texts=ax.texts[base_text_count:])
    _restore_requested_ymax_if_clear(
        fig,
        ax,
        ymax=ymax,
        texts=ax.texts[base_text_count:],
        lines=[],
        legend=legend,
    )
    return fig


def plot_overlays(
    xs: list[ExportedTrainingScalarBars],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    ytick_step: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
    bar_alpha: float = 0.90,
    palette: str | list[str] | tuple[str, ...] | None = None,
    single_panel_group_ticks: bool = True,
    show_points: bool = False,
    point_alpha: float = 0.68,
    point_size: float = 16.0,
    opts=None,
) -> plt.Figure:
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)

    validate_alignment(xs)
    metric_palette_family = None
    if palette is None:
        metric_palette_family = _metric_palette_family_from_exports(xs)

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
    single_panel_group_layout = bool(single_panel_group_ticks) and P == 1 and G > 0
    if P >= 10:
        width_scale = min(1.0 + 0.32 * (font_scale - 1.0), 1.55)
    elif P >= 8:
        width_scale = min(1.0 + 0.24 * (font_scale - 1.0), 1.40)
    else:
        width_scale = min(1.0 + 0.10 * (font_scale - 1.0), 1.20)
    fig_w = max(6.0, 1.2 * (G if single_panel_group_layout else P) * width_scale)
    fig_h = 4.5 * min(1.0 + 0.12 * (font_scale - 1.0), 1.24)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    x_centers = (
        np.asarray([0.5 * max(G - 1, 0)], dtype=float)
        if single_panel_group_layout
        else np.arange(P, dtype=float)
    )
    dense_bar_layout = P >= 10
    compact_n_labels = dense_bar_layout and font_scale >= 1.15
    n_label_rotation = 28 if compact_n_labels else 0

    # grouped/dodged bars
    if single_panel_group_layout:
        bar_w = 0.68
        offsets = np.zeros((G,), dtype=float)
    else:
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
        if single_panel_group_layout:
            xg = np.asarray([float(gi)], dtype=float)
        else:
            xg = x_centers + offsets[gi]
        xpos_by_group.append(np.asarray(xg, float))

        y = np.asarray(means_plot[gi], float)
        y_plot = np.where(np.isfinite(y), y, 0.0)
        n_leg = global_legend_n
        if per_group_legend_ns is not None and gi < len(per_group_legend_ns):
            n_leg = per_group_legend_ns[gi]
        label = f"{x.group} (n={n_leg})" if n_leg is not None else f"{x.group}"
        bar_color = group_metric_fill_color_for_label(
            x.group, gi, metric_palette_family, palette=palette
        )
        edge_color = group_metric_edge_color_for_label(
            x.group, gi, metric_palette_family, palette=palette
        )
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

        if show_points:
            panel_values = x.per_unit_values_panel
            jitter_span = min(float(bar_w) * 0.56, 0.22)
            for p in range(P):
                if p >= len(panel_values):
                    continue
                vals = np.asarray(panel_values[p], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                if vals.size == 1:
                    jitter = np.asarray([0.0], dtype=float)
                else:
                    jitter = np.linspace(
                        -0.5 * jitter_span,
                        0.5 * jitter_span,
                        vals.size,
                        dtype=float,
                    )
                ax.scatter(
                    np.full(vals.size, float(xg[p])) + jitter,
                    vals,
                    s=float(point_size),
                    facecolors="white",
                    edgecolors=edge_color,
                    linewidths=0.8,
                    alpha=float(point_alpha),
                    zorder=4,
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

    tick_rotation = 30
    if P >= 8:
        tick_rotation = 40
    if P >= 10 and font_scale >= 1.15:
        tick_rotation = 55
    if P >= 10 and font_scale >= 1.35:
        tick_rotation = 60
    if single_panel_group_layout:
        tick_labels = []
        for x in xs:
            npp = _panel_n(x, 0)
            if npp is not None and npp > 0:
                tick_labels.append(f"{x.group}\n(n={npp})")
            else:
                tick_labels.append(str(x.group))
        ax.set_xticks(np.arange(G, dtype=float))
        single_panel_tick_rotation = 18
        if font_scale >= 1.15:
            single_panel_tick_rotation = 24
        if font_scale >= 1.35:
            single_panel_tick_rotation = 30
        ax.set_xticklabels(
            tick_labels,
            rotation=single_panel_tick_rotation,
            ha="right",
            rotation_mode="anchor",
        )
        _italicize_xtick_labels(ax)
        ax.set_xlim(-0.6, float(max(G - 1, 0)) + 0.6)
    else:
        ax.set_xticks(x_centers)
        ax.set_xticklabels(panel_labels, rotation=tick_rotation, ha="right")

    if xlabel:
        xlabel_use = str(xlabel)
        auto_single_panel_xlabel = (
            single_panel_group_layout and xlabel_use.strip().lower() == "training"
        )
        if not auto_single_panel_xlabel:
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

    ylim_bottom, ylim_top = _scalar_bar_y_limits(
        means=means_plot,
        lows=lo_plot,
        highs=hi_plot,
        samples=per_unit_by_group,
        ymax=ymax,
    )
    ax.set_ylim(ylim_bottom, ylim_top)
    if ylim_bottom < 0.0 < ylim_top:
        ax.axhline(0.0, color=NEUTRAL_MID, linewidth=0.8, alpha=0.65, zorder=0)
    base_line_count = len(ax.lines)

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
        elif per_group_legend_ns is None and not single_panel_group_layout:
            for p in range(P):
                ns = []
                for x in xs:
                    npp = _panel_n(x, p)
                    if npp is not None and npp > 0:
                        ns.append(int(npp))
                panel_n_texts.append(
                    _per_panel_n_text(ns, compact_n_labels=compact_n_labels)
                )
    elif per_group_legend_ns is None and not single_panel_group_layout:
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

    show_legend = not single_panel_group_layout
    legend = (
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            prop={"style": "italic", "size": legend_font_size},
        )
        if show_legend
        else None
    )
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
    if ytick_step is not None and float(ytick_step) > 0:
        step = float(ytick_step)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        decimals = next(
            (
                precision
                for precision in range(7)
                if np.isclose(step, round(step, precision), rtol=0.0, atol=1e-10)
            ),
            6,
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    if legend is not None:
        # Reserve canvas width for the outside-right legend instead of letting it
        # compete with bars, error bars, or significance annotations inside axes.
        fig.subplots_adjust(right=min(fig.subplotpars.right, 0.74))
    _ensure_xlabel_visible(fig, ax)
    _ensure_ylabel_visible(fig, ax)
    _ensure_legend_clear_of_annotations(
        fig,
        ax,
        legend=legend,
        texts=ax.texts[base_text_count:],
        lines=ax.lines[base_line_count:],
    )
    _ensure_top_text_visible(fig, ax, texts=ax.texts[base_text_count:])
    _restore_requested_ymax_if_clear(
        fig,
        ax,
        ymax=ymax,
        texts=ax.texts[base_text_count:],
        lines=ax.lines[base_line_count:],
        legend=legend,
    )
    return fig
