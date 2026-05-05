from __future__ import annotations

import os
import warnings
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

import src.utils.util as util
from src.utils.common import (
    maybe_sentence_case,
    pch,
    writeImage,
    ttest_ind,
    pick_non_overlapping_y,
)
from src.plotting.plot_customizer import PlotCustomizer


def _as_scalar(x):
    # handle np arrays holding single objects/scalars
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _q_to_pct_str(q: float) -> str:
    # "5%" for 0.05, "2.5%" for 0.025, etc.
    pct = 100.0 * float(q)
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:g}%"


def _load_bundle(path: str) -> dict:
    d = np.load(path, allow_pickle=True)

    req = [
        "btw_rwd_shortest_tail_exp",
        "btw_rwd_shortest_tailN_exp",
        "group_label",
        "training_names",
        "video_ids",
        "q",
        "n_min",
        "k_floor",
        "calc",
        "ctrl",
    ]
    missing = [k for k in req if k not in d.files]
    if missing:
        raise ValueError(f"Bundle {path} is missing keys: {missing}")

    out = {k: d[k] for k in req}
    out["path"] = path

    out["group_label"] = str(
        _as_scalar(
            out["group_label"][0]
            if isinstance(out["group_label"], np.ndarray)
            else out["group_label"]
        )
    )
    out["q"] = float(_as_scalar(out["q"]))
    out["n_min"] = int(_as_scalar(out["n_min"]))
    out["k_floor"] = int(_as_scalar(out["k_floor"]))
    out["calc"] = bool(_as_scalar(out["calc"]))
    out["ctrl"] = bool(_as_scalar(out["ctrl"]))

    # normalize training names + video ids to 1D object arrays
    out["training_names"] = np.asarray(out["training_names"], dtype=object)
    out["video_ids"] = np.asarray(out["video_ids"], dtype=object)

    # normalize arrays
    out["btw_rwd_shortest_tail_exp"] = np.asarray(
        out["btw_rwd_shortest_tail_exp"], dtype=float
    )
    out["btw_rwd_shortest_tailN_exp"] = np.asarray(
        out["btw_rwd_shortest_tailN_exp"], dtype=int
    )

    return out


def _mean_ci_over_videos_1d(vals_2d: np.ndarray) -> np.ndarray:
    """
    vals_2d: (n_videos, n_trn), NaNs allowed
    returns mci: (4, n_trn) [mean, lo, hi, n]
    """
    n_trn = vals_2d.shape[1]
    mci = np.array([util.meanConfInt(vals_2d[:, ti]) for ti in range(n_trn)]).T
    return mci


def _anova_p(groups, *, min_n_per_group=3) -> float:
    """
    groups: list[np.ndarray], each 1D; NaNs removed beforehand by caller
    """
    try:
        clean = []
        for g in groups:
            g = np.asarray(g, dtype=float)
            g = g[np.isfinite(g)]
            if g.size >= int(min_n_per_group):
                clean.append(g)
        if len(clean) < 2:
            return np.nan
        _F, p = f_oneway(*clean)
        return float(p)
    except Exception:
        return np.nan


def _wrapped_ylabel_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if ", " in text:
        return text.replace(", ", ",\n", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    return text


def _ensure_ylabel_visible(fig: plt.Figure, ax: plt.Axes) -> None:
    label = ax.yaxis.get_label()
    if not label.get_text():
        return

    pad_x_px = max(8.0, 0.55 * float(label.get_fontsize()) + 4.0)
    pad_y_px = max(10.0, 0.45 * float(label.get_fontsize()) + 4.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    def _within_bounds() -> bool:
        bbox = label.get_window_extent(renderer=renderer)
        x_ok = bbox.x0 >= fig_bbox.x0 + pad_x_px
        y_ok = (
            bbox.y0 >= fig_bbox.y0 + pad_y_px
            and bbox.y1 <= fig_bbox.y1 - pad_y_px
        )
        return x_ok and y_ok

    if _within_bounds():
        return

    wrapped = _wrapped_ylabel_text(label.get_text())
    if wrapped != label.get_text():
        label.set_text(wrapped)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
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

    new_left = min(fig.subplotpars.left + extra_left, 0.22)
    new_bottom = min(fig.subplotpars.bottom + extra_bottom, 0.16)
    new_top = max(fig.subplotpars.top - extra_top, 0.82)
    if new_top <= new_bottom:
        new_bottom = fig.subplotpars.bottom
        new_top = fig.subplotpars.top

    fig.subplots_adjust(left=new_left, bottom=new_bottom, top=new_top)
    fig.canvas.draw()


def plot_btw_rwd_shortest_tail_bundles(
    bundle_paths,
    out_fn,
    *,
    labels=None,
    opts=None,
    show_title=False,
    y_label=None,
    stats=True,
    min_n_per_group_anova=3,
):
    """
    Plot per-training shortest-tail between-reward distance from one or more bundles.

    - Each bundle is a group.
    - Points: mean over videos (VAs); error: CI over videos.
    - Optional stats:
        * 2 bundles: t-test per training (stars)
        * 3+ bundles: one-way ANOVA per training (stars)
    """
    if opts is None:
        opts = SimpleNamespace(
            wspace=0.35,
            imageFormat="png",
            fontSize=None,
            fontFamily=None,
        )

    bundles = [_load_bundle(p) for p in bundle_paths]
    ng = len(bundles)
    if ng == 0:
        raise ValueError("No bundles provided")

    # labels
    if labels is not None:
        if len(labels) != ng:
            raise ValueError("labels length must match number of bundles")
        group_labels = list(labels)
    else:
        group_labels = [b["group_label"] for b in bundles]

    # y label default: bake q into label, unless caller provides one explicitly
    if y_label is None:
        q0 = bundles[0]["q"]
        y_label = f"Mean between-reward distance, shortest {_q_to_pct_str(q0)} (mm)"

    # consistency checks on training dimension
    n_trn = bundles[0]["btw_rwd_shortest_tail_exp"].shape[1]
    for b in bundles:
        arr = b["btw_rwd_shortest_tail_exp"]
        if arr.ndim != 2:
            raise ValueError(
                f"{b['path']}: expected 2D exp array, got shape {arr.shape}"
            )
        if arr.shape[1] != n_trn:
            raise ValueError("Bundles disagree on number of trainings (exp.shape[1]).")

    # training labels
    tnames = bundles[0]["training_names"]
    if tnames.size < n_trn:
        tnames = np.array([f"T{i+1}" for i in range(n_trn)], dtype=object)
    else:
        tnames = tnames[:n_trn]

    # x positions
    xs = np.arange(1, n_trn + 1, dtype=float)

    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))
    figsize = pch((7.0, 4.2), (10, 5))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.sca(ax)

    # styling: “same color, different linestyles” like your overlay plots
    exp_color = "C0"
    linestyles = ["-", "--", ":", "-."]

    # compute + plot each group
    mci_by_group = []
    raw_by_group = []  # list of list-of-arrays per training (for stats)
    pending_n_labels = []
    lbls = defaultdict(list)
    ylim = [0.0, 20.0]

    for gi, b in enumerate(bundles):
        exp = b["btw_rwd_shortest_tail_exp"][:, :n_trn]
        mci = _mean_ci_over_videos_1d(exp)  # (4, n_trn)

        mci_by_group.append(mci)

        # samples per training for tests
        raw = []
        for ti in range(n_trn):
            v = np.asarray(exp[:, ti], dtype=float)
            v = v[np.isfinite(v)]
            raw.append(v)
        raw_by_group.append(raw)

        ys = mci[0, :]
        lo = mci[1, :]
        hi = mci[2, :]
        ns = mci[3, :]

        fin = np.isfinite(ys)
        ls = linestyles[gi % len(linestyles)]

        (line,) = plt.plot(
            xs[fin],
            ys[fin],
            color=exp_color,
            marker="o",
            ms=4,
            mec=exp_color,
            linewidth=2,
            linestyle=ls,
        )
        line.set_label(group_labels[gi])

        if np.isfinite(lo).any() and np.isfinite(hi).any():
            plt.fill_between(xs[fin], lo[fin], hi[fin], color=exp_color, alpha=0.15)

        for ti in range(n_trn):
            if not np.isfinite(ys[ti]) or ns[ti] <= 0:
                continue
            anchor_candidates = [ys[ti]]
            if np.isfinite(hi[ti]):
                anchor_candidates.append(hi[ti])
            anchor_y = float(np.nanmax(anchor_candidates))
            pending_n_labels.append(
                (
                    ti,
                    float(xs[ti]),
                    anchor_y,
                    f"{int(ns[ti])}",
                )
            )

    # axes cosmetics
    plt.xticks(xs, [maybe_sentence_case(str(t)) for t in tnames])
    plt.xlabel("Training")
    plt.ylabel(maybe_sentence_case(y_label))
    ax.legend(
        frameon=False,
        handlelength=3.5,  # default ~2.0; bump it
        handletextpad=0.6,
        markerscale=1.0,
        prop={"style": "italic"},
    )

    # opt-in title
    if show_title:
        meta0 = bundles[0]
        title = f"Between-reward shortest-tail (q={meta0['q']}, n_min={meta0['n_min']}, k_floor={meta0['k_floor']})"
        plt.title(maybe_sentence_case(title))

    # y-lims: pad a bit
    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]) and ylim[1] > ylim[0]:
        pad = 0.12 * (ylim[1] - ylim[0])
        ax.set_ylim(ylim[0] - pad, ylim[1] + pad)

    span = (ylim[1] - ylim[0]) if np.isfinite(ylim[1] - ylim[0]) else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    # n labels (placed after y-lims are known so collision avoidance is meaningful)
    for ti, x_pos, anchor_y, n_text in sorted(
        pending_n_labels, key=lambda item: (item[0], item[2])
    ):
        base_y = anchor_y + 0.05 * span
        avoid_ys = [anchor_y] + lbls[ti]
        y_n, va_align = pick_non_overlapping_y(
            base_y,
            avoid_ys,
            ylim,
            prefer="above",
            gap=0.065,
            step=0.024,
        )
        util.pltText(
            x_pos,
            y_n,
            n_text,
            ha="center",
            va=va_align,
            size=customizer.in_plot_font_size,
            color=".2",
        )
        lbls[ti].append(float(y_n))

    # ---- stats annotations ----
    if stats and ng >= 2:
        if ng == 2:
            for ti in range(n_trn):
                x0 = raw_by_group[0][ti]
                x1 = raw_by_group[1][ti]
                if x0.size < 2 or x1.size < 2:
                    continue
                try:
                    _t, p = ttest_ind(x0, x1)[:2]
                except Exception:
                    continue
                stars = util.p2stars(p, nanR="")
                if not stars:
                    continue

                mu0 = np.nanmean(x0) if x0.size else np.nan
                mu1 = np.nanmean(x1) if x1.size else np.nan
                if not (np.isfinite(mu0) or np.isfinite(mu1)):
                    continue
                anchor_y = float(np.nanmax([mu0, mu1]))

                base_y = anchor_y + 0.10 * span
                prefer = "above"
                margin = 0.06 * span
                if np.isfinite(ylim[1]) and base_y + 0.20 * span > (ylim[1] - margin):
                    prefer = "below"

                avoid_ys = [anchor_y] + lbls[ti]
                ys_star, va_align = pick_non_overlapping_y(
                    base_y, avoid_ys, ylim, prefer=prefer
                )

                util.pltText(
                    xs[ti],
                    ys_star,
                    stars,
                    ha="center",
                    va=va_align,
                    size=customizer.in_plot_font_size,
                    color="0",
                    weight="bold",
                )
                lbls[ti].append(float(ys_star))
        else:
            for ti in range(n_trn):
                groups_here = [raw_by_group[gi][ti] for gi in range(ng)]
                p = _anova_p(groups_here, min_n_per_group=min_n_per_group_anova)
                stars = util.p2stars(p, nanR="")
                if not stars:
                    continue

                mus = [mci_by_group[gi][0, ti] for gi in range(ng)]
                if not np.any(np.isfinite(mus)):
                    continue
                anchor_y = float(np.nanmax(mus))

                base_y = anchor_y + 0.10 * span
                prefer = "above"
                margin = 0.06 * span
                if np.isfinite(ylim[1]) and base_y + 0.20 * span > (ylim[1] - margin):
                    prefer = "below"

                avoid_ys = [anchor_y] + lbls[ti]
                ys_star, va_align = pick_non_overlapping_y(
                    base_y, avoid_ys, ylim, prefer=prefer
                )

                util.pltText(
                    xs[ti],
                    ys_star,
                    stars,
                    ha="center",
                    va=va_align,
                    size=customizer.in_plot_font_size,
                    color="0",
                    weight="bold",
                )
                lbls[ti].append(float(ys_star))

    if customizer.customized:
        customizer.adjust_padding_proportionally(wspace=getattr(opts, "wspace", 0.35))
    else:
        fig.tight_layout()

    ax.set_ylim(*ylim)
    _ensure_ylabel_visible(fig, ax)
    writeImage(out_fn, format=getattr(opts, "imageFormat", "png"))
    plt.close()
