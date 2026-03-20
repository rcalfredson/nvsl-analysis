from __future__ import annotations

import os
from types import SimpleNamespace
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd

from src.analysis.sli_bundle_utils import (
    bundle_fly_ids,
    bundle_video_ids,
    load_sli_bundle,
)
from src.analysis.sli_tools import select_fractional_groups
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import pick_above_or_expand, ttest_ind, writeImage
import src.utils.util as util


def _pct_label(which, frac):
    if frac is None:
        return which
    return f"{which} {int(round(100 * frac))}%"


def _selected_indices(
    bundle, *, sli_extremes=None, top_fraction=None, bottom_fraction=None
):
    n = len(bundle["sli"])
    if sli_extremes is None:
        return np.arange(n, dtype=int)

    sli = np.asarray(bundle["sli"], dtype=float)
    bottom, top = select_fractional_groups(
        pd.Series(sli),
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    bottom = np.asarray([] if bottom is None else bottom, dtype=int)
    top = np.asarray([] if top is None else top, dtype=int)

    if sli_extremes == "top":
        return top
    if sli_extremes == "bottom":
        return bottom
    if sli_extremes == "both":
        return {"bottom": bottom, "top": top}
    raise ValueError(f"Unsupported sli_extremes={sli_extremes!r}")


def _mean_ci_over_videos(vals_2d):
    if vals_2d.size == 0:
        return np.full((4, 0), np.nan, dtype=float)
    nb = vals_2d.shape[1]
    return np.array([util.meanConfInt(vals_2d[:, b]) for b in range(nb)]).T


def _sparse_ticks(ticks: np.ndarray, *, max_labels: int = 12) -> np.ndarray:
    ticks = np.asarray(ticks, dtype=float).reshape(-1)
    if ticks.size <= max_labels:
        return ticks
    step = int(np.ceil(ticks.size / float(max_labels)))
    out = ticks[::step]
    if out.size == 0 or out[-1] != ticks[-1]:
        out = np.concatenate((out, ticks[-1:]))
    if out.size >= 2:
        last_gap = abs(out[-1] - out[-2])
        target_gap = (
            abs(ticks[min(step, ticks.size - 1)] - ticks[0])
            if ticks.size > 1
            else np.inf
        )
        if np.isfinite(target_gap) and target_gap > 0 and last_gap < 0.75 * target_gap:
            out = np.concatenate((out[:-2], out[-1:]))
    return out


def _infer_total_rewards(bundle):
    totals = bundle.get("cum_reward_sli_total_actual_rewards")
    if totals is not None:
        return np.asarray(totals, dtype=float).reshape(-1)

    curve = np.asarray(bundle["cum_reward_sli_curve"], dtype=float)
    ticks = np.asarray(bundle["cum_reward_sli_ticks"], dtype=float).reshape(-1)
    out = np.zeros((curve.shape[0],), dtype=float)
    for i, row in enumerate(curve):
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size:
            out[i] = float(ticks[finite[-1]])
    return out


def _subset_max_supported_tick(bundle, sub_idx, min_fly_pct):
    ticks = np.asarray(bundle["cum_reward_sli_ticks"], dtype=float).reshape(-1)
    if ticks.size == 0 or sub_idx.size == 0 or float(min_fly_pct) <= 0:
        return None

    totals = _infer_total_rewards(bundle)
    subset_totals = np.asarray(totals[sub_idx], dtype=float)
    subset_totals = subset_totals[np.isfinite(subset_totals)]
    if subset_totals.size == 0:
        return None

    req = int(np.ceil(subset_totals.size * (float(min_fly_pct) / 100.0)))
    req = min(max(req, 1), subset_totals.size)
    sorted_desc = np.sort(subset_totals)[::-1]
    return float(sorted_desc[req - 1])


def _shared_tick_positions(bundles, max_supported_tick):
    shared = None
    tick_maps = []
    for b in bundles:
        ticks = np.asarray(b["cum_reward_sli_ticks"], dtype=float).reshape(-1)
        if max_supported_tick is not None:
            ticks = ticks[ticks <= float(max_supported_tick)]
        tick_maps.append({float(x): i for i, x in enumerate(ticks)})
        shared = ticks if shared is None else np.intersect1d(shared, ticks)

    shared = np.asarray([] if shared is None else shared, dtype=float)
    if shared.size == 0:
        return shared, []

    positions = []
    for tm in tick_maps:
        positions.append(np.array([tm[float(x)] for x in shared], dtype=int))
    return shared, positions


def _resolve_min_fly_pct(bundles, min_fly_pct):
    if min_fly_pct is not None:
        return float(min_fly_pct)

    vals = []
    for b in bundles:
        v = b.get("cum_reward_sli_min_fly_pct")
        if v is None:
            vals.append(95.0)
        else:
            vals.append(float(np.asarray(v).reshape(())))
    if not vals:
        return 95.0
    return min(vals)


def _tick_dodge_step(ticks: np.ndarray) -> float:
    ticks = np.asarray(ticks, dtype=float).reshape(-1)
    if ticks.size >= 2:
        diffs = np.diff(ticks)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs)) * 0.15
    return 0.15


def _centered_x_offsets(count: int, dodge_step: float) -> np.ndarray:
    if count <= 1 or not np.isfinite(dodge_step) or dodge_step <= 0:
        return np.zeros((max(count, 1),), dtype=float)
    idx = np.arange(count, dtype=float) - 0.5 * (count - 1)
    return idx * (2.0 * float(dodge_step))


def _metric_spec(metric: str | None) -> tuple[str, str, str]:
    metric_name = str(metric or "sli").strip().lower()
    specs = {
        "sli": (
            "cum_reward_sli_curve",
            "SLI",
            "SLI vs cumulative rewards",
        ),
        "reward_pi": (
            "cum_reward_sli_reward_pi_exp",
            "Reward PI (experimental fly)",
            "Reward PI vs cumulative rewards",
        ),
        "reward_pi_exp": (
            "cum_reward_sli_reward_pi_exp",
            "Reward PI (experimental fly)",
            "Reward PI vs cumulative rewards",
        ),
        "reward_pi_yoked": (
            "cum_reward_sli_reward_pi_yoked",
            "Reward PI (yoked fly)",
            "Reward PI vs cumulative rewards",
        ),
    }
    if metric_name not in specs:
        raise ValueError(
            "Unsupported metric={!r}; expected one of: {}".format(
                metric,
                ", ".join(sorted(specs)),
            )
        )
    return specs[metric_name]


def _sample_indices(
    sub_idx: np.ndarray,
    *,
    sample_n: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    sub_idx = np.asarray(sub_idx, dtype=int)
    if sub_idx.size == 0:
        return sub_idx
    if sample_n is None:
        return sub_idx
    n = max(0, int(sample_n))
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n >= sub_idx.size or rng is None:
        return sub_idx.copy()
    chosen = np.sort(rng.choice(sub_idx, size=n, replace=False))
    return np.asarray(chosen, dtype=int)


def _subset_plot_label(
    group_label: str,
    *,
    which: str | None,
    sli_extremes,
    sli_top_fraction,
    sli_bottom_fraction,
    standalone_extreme_labels: bool,
) -> str:
    label = str(group_label)
    if which == "top":
        extreme_label = _pct_label("Top", sli_top_fraction)
        return (
            extreme_label
            if standalone_extreme_labels
            else f"{label} ({extreme_label.lower()})"
        )
    if which == "bottom":
        extreme_label = _pct_label("Bottom", sli_bottom_fraction)
        return (
            extreme_label
            if standalone_extreme_labels
            else f"{label} ({extreme_label.lower()})"
        )
    if sli_extremes == "top":
        return f"{label} ({_pct_label('top', sli_top_fraction)})"
    if sli_extremes == "bottom":
        return f"{label} ({_pct_label('bottom', sli_bottom_fraction)})"
    return label


def _lighten_color(color, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    amt = min(max(float(amount), 0.0), 1.0)
    return tuple(rgb + (1.0 - rgb) * amt)


def _sampled_line_color(
    *,
    color_mode: str,
    base_color,
    order: int,
    n_sampled: int,
    rng: np.random.Generator | None,
):
    if str(color_mode) == "random":
        cmap = cm.get_cmap("tab20")
        if rng is None:
            idx = order % cmap.N
        else:
            idx = int(rng.integers(0, cmap.N))
        return tuple(cmap(idx)[:3])
    return _lighten_color(
        base_color,
        0.15 + 0.55 * (float(order) / max(n_sampled - 1, 1)),
    )


def _short_video_label(raw) -> str:
    text = str(raw)
    if "::f" in text:
        text = text.split("::f", 1)[0]
    text = os.path.basename(text.rstrip("/")) or text
    if text.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
        text = os.path.splitext(text)[0]
    return text


def _fly_label(bundle: dict, abs_idx: int) -> str:
    video_ids = bundle_video_ids(bundle)
    fly_ids = bundle_fly_ids(bundle)
    base = (
        _short_video_label(video_ids[abs_idx])
        if video_ids is not None and 0 <= int(abs_idx) < len(video_ids)
        else f"fly {int(abs_idx) + 1}"
    )
    if fly_ids is not None and 0 <= int(abs_idx) < len(fly_ids):
        fly_id = int(fly_ids[abs_idx])
        if fly_id >= 0:
            return f"{base} [f={fly_id}]"
    return base


def plot_cum_reward_sli_bundles(
    bundle_paths,
    out_fn,
    *,
    labels=None,
    sli_extremes=None,
    sli_fraction=None,
    sli_top_fraction=None,
    sli_bottom_fraction=None,
    standalone_extreme_labels=False,
    ci_min_n=3,
    show_n=False,
    min_fly_pct=None,
    metric="sli",
    individual_sample_n=None,
    individual_seed=0,
    individual_color_mode="gradient",
    show_legend=True,
    opts=None,
):
    if opts is None:
        opts = SimpleNamespace(imageFormat="png")

    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction
    if (
        sli_extremes is not None
        and sli_top_fraction is None
        and sli_bottom_fraction is None
    ):
        sli_top_fraction = 0.2
        sli_bottom_fraction = 0.2

    bundles = [load_sli_bundle(p) for p in bundle_paths]
    if not bundles:
        raise ValueError("No bundles provided")
    min_fly_pct = _resolve_min_fly_pct(bundles, min_fly_pct)
    metric_key, y_label, plot_title = _metric_spec(metric)

    for b in bundles:
        for key in (metric_key, "cum_reward_sli_ticks"):
            if key not in b:
                raise ValueError(
                    f"Bundle {b.get('path', '<unknown>')} is missing key {key!r}"
                )

    group_labels = (
        list(labels) if labels is not None else [b["group_label"] for b in bundles]
    )
    if len(group_labels) != len(bundles):
        raise ValueError("labels length must match number of bundles")
    if standalone_extreme_labels and sli_extremes != "both":
        raise ValueError(
            "standalone_extreme_labels is supported only with sli_extremes='both'."
        )
    if sli_extremes == "both" and len(bundles) != 1:
        raise ValueError("sli_extremes='both' is supported only for a single bundle.")

    customizer = PlotCustomizer()
    fig = plt.figure(figsize=(7.5, 4.8))
    ax = plt.gca()
    individual_mode = (
        individual_sample_n is not None and int(individual_sample_n) > 0
    )
    rng = np.random.default_rng(individual_seed) if individual_mode else None
    n_labels = []
    lbls = defaultdict(list)
    y_min = None
    y_max = None
    means_by_group = []
    vals_by_group = []
    ns_by_group = []
    subset_specs_by_bundle = []
    subset_max_ticks = []

    for b in bundles:
        idx = _selected_indices(
            b,
            sli_extremes=sli_extremes,
            top_fraction=sli_top_fraction,
            bottom_fraction=sli_bottom_fraction,
        )
        if sli_extremes == "both":
            subset_specs = [
                ("bottom", np.asarray(idx["bottom"], dtype=int), "--"),
                ("top", np.asarray(idx["top"], dtype=int), "-"),
            ]
        else:
            subset_specs = [(None, np.asarray(idx, dtype=int), "-")]
        subset_specs_by_bundle.append(subset_specs)

        for _which, sub_idx, _linestyle in subset_specs:
            if sub_idx.size == 0:
                continue
            subset_max = _subset_max_supported_tick(b, sub_idx, min_fly_pct)
            if subset_max is not None:
                subset_max_ticks.append(float(subset_max))

    max_supported_tick = (
        min(subset_max_ticks) if subset_max_ticks else None
    )
    ticks0, tick_positions = _shared_tick_positions(bundles, max_supported_tick)
    if ticks0.size == 0:
        raise ValueError(
            "No shared cumulative-reward ticks remain after applying the min-fly support cutoff."
        )
    dodge_step = _tick_dodge_step(ticks0)

    for gi, b in enumerate(bundles):
        subset_specs = subset_specs_by_bundle[gi]
        tick_idx = tick_positions[gi]

        for which, sub_idx, linestyle in subset_specs:
            if sub_idx.size == 0:
                continue

            vals_full = np.asarray(b[metric_key], dtype=float)[sub_idx, :]
            vals = vals_full[:, tick_idx]
            label = _subset_plot_label(
                group_labels[gi],
                which=which,
                sli_extremes=sli_extremes,
                sli_top_fraction=sli_top_fraction,
                sli_bottom_fraction=sli_bottom_fraction,
                standalone_extreme_labels=standalone_extreme_labels,
            )

            if individual_mode:
                sampled_rel = _sample_indices(
                    np.arange(sub_idx.size, dtype=int),
                    sample_n=individual_sample_n,
                    rng=rng,
                )
                if sampled_rel.size == 0:
                    continue
                sampled_abs = sub_idx[sampled_rel]
                (anchor_line,) = plt.plot(
                    [], [], label=label, linewidth=2, linestyle=linestyle
                )
                base_color = anchor_line.get_color()
                n_sampled = sampled_rel.size
                for order, (rel_idx, abs_idx) in enumerate(
                    zip(sampled_rel, sampled_abs), start=1
                ):
                    row = np.asarray(vals[rel_idx], dtype=float)
                    if not np.isfinite(row).any():
                        continue
                    color = _sampled_line_color(
                        color_mode=individual_color_mode,
                        base_color=base_color,
                        order=order - 1,
                        n_sampled=n_sampled,
                        rng=rng,
                    )
                    fly_label = _fly_label(b, int(abs_idx))
                    if len(subset_specs) > 1 or len(bundles) > 1:
                        fly_label = f"{label}: {fly_label}"
                    plt.plot(
                        ticks0,
                        row,
                        label=fly_label,
                        linewidth=1.4,
                        linestyle=linestyle,
                        alpha=0.95,
                        color=color,
                    )
                    ys = row[np.isfinite(row)]
                    if ys.size:
                        ys_lo = float(np.nanmin(ys))
                        ys_hi = float(np.nanmax(ys))
                        y_min = ys_lo if y_min is None else min(y_min, ys_lo)
                        y_max = ys_hi if y_max is None else max(y_max, ys_hi)
                anchor_line.remove()
                continue

            mci = _mean_ci_over_videos(vals)
            if mci.shape[1] == 0:
                continue

            (line,) = plt.plot(
                ticks0, mci[0], label=label, linewidth=2, linestyle=linestyle
            )
            ci_mask = np.asarray(mci[3], dtype=float) >= float(ci_min_n)
            lo = np.where(ci_mask, mci[1], np.nan)
            hi = np.where(ci_mask, mci[2], np.nan)
            plt.fill_between(ticks0, lo, hi, alpha=0.2, color=line.get_color())
            if np.isfinite(lo).any():
                lo_f = np.nanmin(lo)
                y_min = lo_f if y_min is None else min(y_min, lo_f)
            if np.isfinite(hi).any():
                hi_f = np.nanmax(hi)
                y_max = hi_f if y_max is None else max(y_max, hi_f)
            ys = np.asarray(mci[0], dtype=float)
            if np.isfinite(ys).any():
                ys_lo = np.nanmin(ys)
                ys_hi = np.nanmax(ys)
                y_min = ys_lo if y_min is None else min(y_min, ys_lo)
                y_max = ys_hi if y_max is None else max(y_max, ys_hi)
            means_by_group.append(np.asarray(mci[0], dtype=float))
            ns_by_group.append(np.asarray(mci[3], dtype=int))
            vals_by_group.append(
                [vals[np.isfinite(vals[:, bj]), bj] for bj in range(vals.shape[1])]
            )
            n_labels.append(
                {
                    "label": label,
                    "means": np.asarray(mci[0], dtype=float),
                    "ns": np.asarray(mci[3], dtype=int),
                    "color": "0.2",
                }
            )

    plt.xlabel("Cumulative rewards")
    plt.ylabel(y_label)
    plt.title(
        f"{plot_title} (sampled individual flies)" if individual_mode else plot_title
    )
    plt.axhline(0, color="0.5", linewidth=1, linestyle="--")
    plt.xlim(left=0)
    if ticks0.size:
        tick_labels = _sparse_ticks(ticks0, max_labels=12)
        ax.set_xticks(tick_labels)
        ax.set_xticklabels(
            [f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in tick_labels]
        )
    if show_legend:
        if individual_mode:
            fig.subplots_adjust(right=0.70)
            ax.legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=max(7, customizer.in_plot_font_size - 4),
                title="Sampled flies",
                title_fontsize=max(8, customizer.in_plot_font_size - 3),
                handlelength=2.2,
                borderaxespad=0.0,
            )
        else:
            plt.legend(frameon=False)
    if (
        y_min is not None
        and y_max is not None
        and np.isfinite(y_min)
        and np.isfinite(y_max)
    ):
        ylim = [float(y_min), float(y_max)]
        span = ylim[1] - ylim[0]
        if span <= 0:
            span = 1.0
            ylim = [ylim[0] - 0.5, ylim[1] + 0.5]
        ylim[0] = min(ylim[0], 0.0) - 0.03 * span
        ylim[1] = ylim[1] + 0.10 * span
        ax.set_ylim(ylim[0], ylim[1])
    ylim = list(ax.get_ylim())
    if show_n and n_labels and not individual_mode:
        span = ylim[1] - ylim[0]
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        for info in n_labels:
            means = np.asarray(info["means"], dtype=float)
            ns = np.asarray(info["ns"], dtype=int)
            for bj, (x, y, n) in enumerate(zip(ticks0, means, ns)):
                if int(n) <= 0 or not np.isfinite(y):
                    continue
                txts_here = lbls.get(bj, [])
                bucket_label_ys = [
                    (t._final_y_ if hasattr(t, "_final_y_") else getattr(t, "_y_", None))
                    for t in txts_here
                ]
                bucket_label_ys = [
                    y0 for y0 in bucket_label_ys if y0 is not None and np.isfinite(y0)
                ]
                base_y = float(y) + 0.018 * span
                ys_txt, va_align = pick_above_or_expand(
                    base_y,
                    [float(y)],
                    ylim,
                    span_override=span,
                )
                if ys_txt is None:
                    continue
                need_dodge = False
                if bucket_label_ys:
                    gap_thresh = 0.06 * span
                    need_dodge = any(abs(float(ys_txt) - y0) < gap_thresh for y0 in bucket_label_ys)
                if need_dodge:
                    offsets = _centered_x_offsets(len(txts_here) + 1, dodge_step)
                    for txt, offset in zip(txts_here, offsets[:-1]):
                        txt.set_x(float(x) + float(offset))
                    x_pos = float(x) + float(offsets[-1])
                else:
                    x_pos = float(x)
                txt = util.pltText(
                    x_pos,
                    ys_txt,
                    f"{int(n)}",
                    ha="center",
                    va=va_align,
                    size=max(7, customizer.in_plot_font_size - 2),
                    color=info["color"],
                )
                txt._y_ = float(y)
                txt._final_y_ = float(ys_txt)
                lbls[bj].append(txt)
    span = ylim[1] - ylim[0]
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    if (
        not individual_mode
        and len(vals_by_group) == 2
        and all(v is not None for v in vals_by_group)
    ):
        m0 = means_by_group[0]
        m1 = means_by_group[1]
        n0 = ns_by_group[0]
        n1 = ns_by_group[1]
        for bj in range(len(vals_by_group[0])):
            x0 = vals_by_group[0][bj]
            x1 = vals_by_group[1][bj]
            if int(n0[bj]) < int(ci_min_n) or int(n1[bj]) < int(ci_min_n):
                continue
            if x0.size < 2 or x1.size < 2:
                continue
            try:
                _t, p = ttest_ind(x0, x1)[:2]
            except Exception:
                continue
            stars = util.p2stars(p, nanR="")
            if not stars:
                continue
            if not (np.isfinite(m0[bj]) or np.isfinite(m1[bj])):
                continue
            anchor_y = float(np.nanmax([m0[bj], m1[bj]]))
            txts_here = lbls.get(bj, [])
            avoid_ys = [
                (t._final_y_ if hasattr(t, "_final_y_") else t._y_)
                for t in txts_here
                if hasattr(t, "_final_y_") or hasattr(t, "_y_")
            ]
            avoid_ys.append(anchor_y)
            base_y = anchor_y + 0.04 * span
            ys_txt, va_align = pick_above_or_expand(base_y, avoid_ys, ylim)
            if ys_txt is None:
                continue
            txt = util.pltText(
                ticks0[bj],
                ys_txt,
                stars,
                ha="center",
                va=va_align,
                size=customizer.in_plot_font_size,
                color="0",
                weight="bold",
            )
            txt._y_ = anchor_y
            txt._final_y_ = float(ys_txt)
            lbls[bj].append(txt)
    ax.set_ylim(ylim[0], ylim[1])
    customizer.adjust_padding_proportionally()
    writeImage(out_fn, format=getattr(opts, "imageFormat", "png"))
    plt.close()
