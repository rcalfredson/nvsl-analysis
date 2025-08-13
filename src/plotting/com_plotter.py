import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import ttest_ind, writeImage
from src.utils.util import meanConfInt, p2stars, slugify


def _draw_bucketwise_sig_stars(
    ax, xs, ymins, ymaxs, top_envelope, arr_a, arr_b, in_plot_font_size
):
    """
    Draw stars comparing two groups at each bucket.
    arr_a, arr_b: shape (n_videos_in_group, n_buckets)
    Returns max_star_top_y_data: the top of the tallest star text bbox (data units).
    """
    stars_to_draw = []

    # 1) figure out data range to compute anchor pad
    if ymins and ymaxs:
        data_min = float(np.min(ymins))
        data_max = float(np.max(ymaxs))
    else:
        finite_env = np.isfinite(top_envelope)
        data_min = 0.0
        data_max = (
            float(np.nanmax(top_envelope[finite_env])) if finite_env.any() else 1.0
        )

    data_range = max(1e-9, data_max - data_min)
    anchor_pad = 0.05 * data_range  # pad between CI cap and *text anchor*

    # 2) compute anchors & store to draw
    for j in range(arr_a.shape[1]):  # over buckets
        t, p, na, nb, _ = ttest_ind(arr_a[:, j], arr_b[:, j])  # Welch handled upstream
        s = p2stars(p)
        if not s:
            continue
        y_cap = top_envelope[j]
        if not np.isfinite(y_cap):
            continue
        y_anchor = y_cap + anchor_pad  # va="bottom" → anchor is bottom of glyph
        stars_to_draw.append((xs[j], y_anchor, s))

    if not stars_to_draw:
        return np.nan  # nothing to draw / no ylim change

    # 3) draw the stars (visible), then measure their bboxes with renderer
    text_artists = []
    for xj, yj, s in stars_to_draw:
        txt = ax.text(
            xj,
            yj,
            s,
            ha="center",
            va="bottom",
            fontsize=in_plot_font_size,
            color="0",
            weight="bold",
        )
        text_artists.append(txt)

    # Important: force a draw so extents are computed
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # 4) find the tallest star TOP in data units
    max_star_top_y_data = -np.inf
    tallest_h_data = 0.0
    for t in text_artists:
        bbox_disp = t.get_window_extent(renderer=renderer)
        # convert bbox height to data units (use any x; y is what matters)
        y0_data = inv.transform((bbox_disp.x0, bbox_disp.y0))[1]
        y1_data = inv.transform((bbox_disp.x0, bbox_disp.y1))[1]
        h_data = abs(y1_data - y0_data)
        tallest_h_data = max(tallest_h_data, h_data)
        max_star_top_y_data = max(max_star_top_y_data, y1_data)

    # 5) expand ylim so the *top of the tallest text box* is inside, sign-agnostic
    if np.isfinite(max_star_top_y_data):
        cur_lo, cur_hi = ax.get_ylim()
        data_span = max(1e-9, data_max - data_min)
        # small cushion above the tallest text box; also guard against zero-height fonts
        top_head = max(0.04 * data_span, 0.25 * tallest_h_data)
        bot_head = 0.02 * data_span

        desired_top = max_star_top_y_data + top_head
        desired_bottom = data_min - bot_head

        ax.set_ylim(
            bottom=min(cur_lo, desired_bottom),
            top=max(cur_hi, desired_top),
        )

    return max_star_top_y_data


def plot_com_distance(
    vas,  # list of VideoAnalysis instances
    trns,  # list of Training instances (same for all videos)
    bucket_len_minutes,  # minutes per sync bucket
    customizer: PlotCustomizer,
    gis=None,  # array-like of group indices per video (len == len(vas))
    gls=None,  # list of group labels (len == n_groups)
    outdir: str = None,
    title: str = "COM → reward distance",
    format: str = "png",  # file format to use when saving plots
):
    """
    vas.shape → (n_videos)
    trns: length = n_trns
    Each va.syncMedDist[t_idx]['exp'] is a length-n_buckets list of distances
    If control present, va.syncMedDist[t_idx]['ctrl'] likewise.
    """
    # ----- 1) pack into a 4D array: (vid, trn, fly, bucket) -----
    n_videos = len(vas)
    n_trns = len(trns)
    n_buckets = len(vas[0].syncMedDist[0]["exp"])

    # auto-detect if any control data is present
    has_ctrl = any(
        ("ctrl" in dist_dict and len(dist_dict["ctrl"]) > 0)
        for va in vas
        for dist_dict in va.syncMedDist
    )
    n_flies = 2 if has_ctrl else 1

    com4 = np.full((n_videos, n_trns, n_flies, n_buckets), np.nan)
    for vid, va in enumerate(vas):
        for t_idx, dist_dict in enumerate(va.syncMedDist):
            com4[vid, t_idx, 0, : len(dist_dict["exp"])] = dist_dict["exp"]
            if has_ctrl and "ctrl" in dist_dict and dist_dict["ctrl"]:
                com4[vid, t_idx, 1, : len(dist_dict["ctrl"])] = dist_dict["ctrl"]

    # ----- 2) grouping logic -----
    multi_group = gis is not None and gls is not None and len(set(gis)) > 1
    if multi_group:
        gis = np.asarray(gis)
        groups = sorted(set(gis))
        # validate gls covers all groups by index
        assert max(groups) < len(gls), "gls must include a label for each group index"
    else:
        groups = [None]  # single 'virtual' group

    # helper to compute mean ± CI over videos
    def mean_ci_over_vids(data_vid_trn_bucket: np.ndarray):
        # data shape: (n_videos_subset, n_trns, n_buckets)
        means = np.empty((n_trns, n_buckets))
        deltas = np.empty((n_trns, n_buckets))
        counts = np.empty((n_trns, n_buckets), int)
        for i in range(n_trns):
            for j in range(n_buckets):
                m, d, n = meanConfInt(
                    data_vid_trn_bucket[:, i, j], conf=0.95, asDelta=True
                )
                means[i, j] = m
                deltas[i, j] = d
                counts[i, j] = n
        return means, deltas, counts

    # ----- 3) x-axis -----
    xs = (np.arange(n_buckets) + 1) * bucket_len_minutes

    # ----- 4) plot -----
    fig, axes = plt.subplots(1, n_trns, figsize=(4 * n_trns, 4), sharey=True)
    if n_trns == 1:
        axes = [axes]

    # consistent colors for fly types; styles vary per group
    exp_color = "#1f4da1"
    ctrl_color = "#a00000"
    group_linestyles = ["-", "--", ":", "-."]  # cycles if >4 groups
    alpha = 0.18
    ms = 2

    style_idx_for_group = {}

    present_groups = set()

    max_star_y_per_ax = [np.nan] * n_trns

    for i, ax in enumerate(axes):
        # track y-lims while drawing
        ymins, ymaxs = [], []
        top_envelope = np.full(n_buckets, -np.inf)
        present_in_ax = []

        for g_pos, g in enumerate(groups):
            if multi_group:
                vid_idxs = np.flatnonzero(gis == g)
            else:
                vid_idxs = np.arange(n_videos)

            if vid_idxs.size == 0:
                continue

            exp_arr = com4[vid_idxs, :, 0, :]  # (n_vids_in_group, n_trns, n_buckets)
            m_exp, ci_exp, n_exp = mean_ci_over_vids(exp_arr)
            if has_ctrl:
                ctrl_arr = com4[vid_idxs, :, 1, :]
                m_ctrl, ci_ctrl, n_ctrl = mean_ci_over_vids(ctrl_arr)

            # limit to available data across exp/ctrl
            valid = ~np.isnan(m_exp[i])
            if has_ctrl:
                valid |= ~np.isnan(m_ctrl[i])
            last = np.where(valid)[0].max() if valid.any() else -1

            if last < 0:
                continue

            present_groups.add(g if multi_group else 0)
            present_in_ax.append(g if multi_group else 0)

            xs_i = xs[: last + 1].copy()

            m_exp_i = m_exp[i, : last + 1]
            ci_exp_i = ci_exp[i, : last + 1]
            n_exp_i = n_exp[i, : last + 1]
            if has_ctrl:
                m_ctrl_i = m_ctrl[i, : last + 1]
                ci_ctrl_i = ci_ctrl[i, : last + 1]
                n_ctrl_i = n_ctrl[i, : last + 1]

            for jj in range(last + 1):
                cand = m_exp_i[jj] + ci_exp_i[jj]
                if has_ctrl:
                    cand = np.nanmax([cand, m_ctrl_i[jj] + ci_ctrl_i[jj]])
                if np.isfinite(cand):
                    top_envelope[jj] = max(top_envelope[jj], cand)

            ls_idx = g_pos % len(group_linestyles)
            ls = group_linestyles[ls_idx]
            style_idx_for_group[g if multi_group else 0] = ls_idx

            # Experimental
            ax.fill_between(
                xs_i,
                m_exp_i - ci_exp_i,
                m_exp_i + ci_exp_i,
                color=exp_color,
                alpha=alpha,
                linewidth=0,
            )
            ax.plot(
                xs_i,
                m_exp_i,
                marker="o",
                markersize=ms,
                color=exp_color,
                linestyle=ls,
                label=None,
            )
            # Control
            if has_ctrl:
                ax.fill_between(
                    xs_i,
                    m_ctrl_i - ci_ctrl_i,
                    m_ctrl_i + ci_ctrl_i,
                    color=ctrl_color,
                    alpha=alpha,
                    linewidth=0,
                )
                ax.plot(
                    xs_i,
                    m_ctrl_i,
                    marker="o",
                    markersize=ms,
                    color=ctrl_color,
                    linestyle=ls,
                    label=None,
                )

            # annotate counts (slightly offset vertically)
            y_min, y_max = ax.get_ylim()
            dy = 0.02 * (y_max - y_min) if y_max > y_min else 0.5
            for j, x in enumerate(xs_i):
                ax.text(
                    x,
                    m_exp_i[j] - dy,
                    str(n_exp_i[j]),
                    ha="center",
                    va="top",
                    fontsize=customizer.in_plot_font_size,
                    color=exp_color,
                )
                if has_ctrl:
                    ax.text(
                        x,
                        m_ctrl_i[j] + dy,
                        str(n_ctrl_i[j]),
                        ha="center",
                        va="bottom",
                        fontsize=customizer.in_plot_font_size,
                        color=ctrl_color,
                    )

            ymins.append(
                min(
                    np.nanmin(m_exp_i - ci_exp_i),
                    (
                        np.nanmin(m_ctrl_i - ci_ctrl_i)
                        if has_ctrl
                        else np.nanmin(m_exp_i - ci_exp_i)
                    ),
                )
            )
            ymaxs.append(
                max(
                    np.nanmax(m_exp_i + ci_exp_i),
                    (
                        np.nanmax(m_ctrl_i + ci_ctrl_i)
                        if has_ctrl
                        else np.nanmax(m_exp_i + ci_exp_i)
                    ),
                )
            )

        if multi_group and len(present_in_ax) == 2:
            g0, g1 = present_in_ax[0], present_in_ax[1]
            idx0 = np.flatnonzero(gis == g0)
            idx1 = np.flatnonzero(gis == g1)

            # per-video EXP values for training i (shape: n_vids_in_group × n_buckets)
            a = com4[idx0, i, 0, :]
            b = com4[idx1, i, 0, :]

            max_star_y = _draw_bucketwise_sig_stars(
                ax, xs, ymins, ymaxs, top_envelope, a, b, customizer.in_plot_font_size
            )
            if np.isfinite(max_star_y):
                max_star_y_per_ax[i] = max_star_y

        ax.set_title(f"training {trns[i].n}")
        ax.set_xlabel(f"end points [min] of {bucket_len_minutes} min sync buckets")
        if i == 0:
            ax.set_ylabel("median dist. to\nreward circle center [mm]")

    # legend & layout
    if multi_group:
        ordered_present = [g for g in sorted(set(gis)) if g in present_groups]
        proxy_handles = [
            Line2D(
                [0],
                [0],
                color=exp_color,
                linestyle=group_linestyles[style_idx_for_group[g]],
                linewidth=2,
                marker=None,
            )
            for g in ordered_present
        ]
        proxy_labels = [gls[g] for g in ordered_present]

        # make the legend on the first subplot
        leg = axes[0].legend(
            handles=proxy_handles,
            labels=proxy_labels,
            loc="upper left",  # we'll reposition right after measuring
            bbox_to_anchor=(0.02, 0.98),
            prop={"style": "italic"},
        )

        # ---- smart anchor: ABOVE stars if it fits, else bottom-left ----
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # legend height in axes fraction
        leg_h_axes = (
            leg.get_window_extent(renderer=renderer).height / axes[0].bbox.height
        )
        pad_axes = 0.02  # small gap from edges/stars

        top_star_y = max_star_y_per_ax[0]  # in DATA units
        if np.isfinite(top_star_y):
            # convert top star y from data -> axes fraction
            y_star_axes = (
                axes[0]
                .transAxes.inverted()
                .transform(axes[0].transData.transform((0.0, top_star_y)))[1]
            )

            # space available *above* the stars (between axes top and star top)
            # using top-anchored legend: its bottom will be at y_top - leg_h_axes
            y_top_anchor = 1.0 - pad_axes
            legend_bottom_if_top = y_top_anchor - leg_h_axes

            # require legend bottom >= star top + pad to count as "above the stars"
            if legend_bottom_if_top >= (y_star_axes + pad_axes):
                leg.set_loc("upper left")
                leg.set_bbox_to_anchor(
                    (0.02, y_top_anchor), transform=axes[0].transAxes
                )
            else:
                # not enough room above: move to bottom-left
                leg.set_loc("lower left")
                leg.set_bbox_to_anchor((0.02, pad_axes), transform=axes[0].transAxes)
        else:
            # no stars — keep high
            leg.set_loc("upper left")
            leg.set_bbox_to_anchor((0.02, 1.0 - pad_axes), transform=axes[0].transAxes)
    else:
        for ax in axes:
            ax.legend_.remove() if ax.get_legend() else None
    fig.suptitle(title)
    plt.tight_layout()

    # ----- 5) save/show -----
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fn = f"{slugify(title)}.png"
        save_path = os.path.join(outdir, fn)
        writeImage(save_path, format=format)
        print(f"Saved COM-distance plot to {save_path}")
    else:
        plt.show()

    # ===== EXTRA FIGURE: experimental − yoked (only if controls exist) =====
    if has_ctrl:
        # Compute (exp - ctrl) per video/training/bucket
        # Shape for exp/ctrl: (n_videos, n_trns, n_buckets)
        exp_all = com4[:, :, 0, :]
        ctrl_all = com4[:, :, 1, :]
        diff_all = exp_all - ctrl_all  # nan-propagates as desired

        fig2, axes2 = plt.subplots(1, n_trns, figsize=(4 * n_trns, 4), sharey=True)
        if n_trns == 1:
            axes2 = [axes2]

        # Styling
        diff_color = "#1f4da1"  # reuse exp color for continuity
        group_linestyles = ["-", "--", ":", "-."]  # same as above
        alpha = 0.18
        ms = 2

        # Keep legend styles consistent with the first figure
        # (re-using style_idx_for_group and present_groups computed earlier)
        for i, ax in enumerate(axes2):
            ymins, ymaxs = [], []
            top_envelope = np.full(n_buckets, -np.inf)
            present_in_ax = []

            for g_pos, g in enumerate(groups):
                if multi_group:
                    vid_idxs = np.flatnonzero(gis == g)
                else:
                    vid_idxs = np.arange(n_videos)

                if vid_idxs.size == 0:
                    continue

                # take diffs for this group and training i: (n_vids_in_group, n_buckets)
                diff_arr_g_full = diff_all[vid_idxs, :, :]
                m_diff, ci_diff, n_diff = mean_ci_over_vids(diff_arr_g_full)

                # select current training i
                m_i = m_diff[i]
                ci_i = ci_diff[i]
                n_i = n_diff[i]

                # trim to common available range
                valid = ~np.isnan(m_i)
                last = np.where(valid)[0].max() if valid.any() else -1
                if last < 0:
                    continue

                xs_i = xs[: last + 1]
                m_i = m_i[: last + 1]
                ci_i = ci_i[: last + 1]
                n_i = n_i[: last + 1]

                for jj in range(len(xs_i)):
                    cand = m_i[jj] + ci_i[jj]
                    if np.isfinite(cand):
                        top_envelope[jj] = max(top_envelope[jj], cand)

                present_in_ax.append(g if multi_group else 0)

                ls = group_linestyles[style_idx_for_group[g if multi_group else 0]]

                # band + line
                ax.fill_between(
                    xs_i,
                    m_i - ci_i,
                    m_i + ci_i,
                    color=diff_color,
                    alpha=alpha,
                    linewidth=0,
                )
                ax.plot(
                    xs_i, m_i, marker="o", markersize=ms, color=diff_color, linestyle=ls
                )

                # counts (put them slightly below the curve for readability)
                y_min, y_max = ax.get_ylim()
                dy = 0.02 * (y_max - y_min) if y_max > y_min else 0.5
                for xj, yj, nj in zip(xs_i, m_i, n_i):
                    ax.text(
                        xj,
                        yj - dy,
                        str(nj),
                        ha="center",
                        va="top",
                        fontsize=customizer.in_plot_font_size,
                        color=diff_color,
                    )

                ymins.append(np.nanmin(m_i - ci_i))
                ymaxs.append(np.nanmax(m_i + ci_i))

                if multi_group and len(present_in_ax) == 2:
                    g0, g1 = present_in_ax[0], present_in_ax[1]
                    idx0 = np.flatnonzero(gis == g0)
                    idx1 = np.flatnonzero(gis == g1)

                    a = diff_all[idx0, i, :]
                    b = diff_all[idx1, i, :]

                    max_star_y = _draw_bucketwise_sig_stars(
                        ax,
                        xs,
                        ymins,
                        ymaxs,
                        top_envelope,
                        a,
                        b,
                        customizer.in_plot_font_size,
                    )
                    if np.isfinite(max_star_y):
                        max_star_y_per_ax[i] = max_star_y

            # zero reference line
            ax.axhline(0, linewidth=1, linestyle="--", color="0.35", zorder=0)

            # titles/labels
            ax.set_title(f"training {trns[i].n}")
            ax.set_xlabel(f"end points [min] of {bucket_len_minutes} min sync buckets")
            if i == 0:
                ax.set_ylabel(
                    "median dist. to reward\ncircle center (exp. - yok.) [mm]"
                )

            # tighten y based on plotted data with small headroom
            # if ymins and ymaxs:
            #     lo, hi = min(ymins), max(ymaxs)
            #     if np.isfinite(lo) and np.isfinite(hi):
            #         pad = 0.06 * max(1e-9, hi - lo)
            #         ax.set_ylim(lo - pad, hi + pad)

        # legend (reuse group labels/styles, identical placement heuristic optional)
        if multi_group:
            ordered_present = [g for g in sorted(set(gis)) if g in present_groups]
            proxy_handles = [
                Line2D(
                    [0],
                    [0],
                    color=diff_color,
                    linestyle=group_linestyles[style_idx_for_group[g]],
                    linewidth=2,
                    marker=None,
                )
                for g in ordered_present
            ]
            proxy_labels = [gls[g] for g in ordered_present]
            axes2[0].legend(
                handles=proxy_handles,
                labels=proxy_labels,
                loc="best",
                prop={"style": "italic"},
            )
        else:
            for ax in axes2:
                ax.legend_.remove() if ax.get_legend() else None

        fig2.suptitle(f"{title} — experimental minus yoked")
        plt.tight_layout()

        # save/show
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            fn = f"{slugify(title)}__exp_minus_yoked.png"
            save_path = os.path.join(outdir, fn)
            writeImage(save_path, format=format)
            print(f"Saved COM-distance (exp − yoked) plot to {save_path}")
        else:
            plt.show()
