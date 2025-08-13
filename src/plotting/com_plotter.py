import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import ttest_ind, writeImage
from src.utils.util import meanConfInt, p2stars, slugify


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
    has_ctrl = False
    n_flies = 2 if has_ctrl else 1

    com4 = np.full((n_videos, n_trns, n_flies, n_buckets), np.nan)
    for vid, va in enumerate(vas):
        for t_idx, dist_dict in enumerate(va.syncMedDist):
            com4[vid, t_idx, 0, : len(dist_dict["exp"])] = dist_dict["exp"]
            if has_ctrl:
                com4[vid, t_idx, 1, : len(dist_dict.get("ctrl", []))] = dist_dict.get(
                    "ctrl", []
                )

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

        stars_to_draw = []
        max_star_y = -np.inf

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

            # pull raw EXP values for both groups: shape (n_videos_in_group, n_buckets)
            idx0 = np.flatnonzero(gis == g0)
            idx1 = np.flatnonzero(gis == g1)
            a = com4[idx0, i, 0, :]  # EXP of group 0 for training i
            b = com4[idx1, i, 0, :]

            y_min, y_max = ax.get_ylim()

            # compute a pad in data units from the *data* range (not the current ylim)
            if ymins and ymaxs:
                data_min = min(ymins)
                data_max = max(ymaxs)
            else:
                data_min, data_max = 0.0, np.nanmax(top_envelope)
            data_range = max(data_max - data_min, 1e-9)
            y_pad = 0.05 * data_range  # 5% of data range; tweak as desired

            for j in range(n_buckets):
                # two-sample (Welch controlled by your global WELCH flag in common.ttest)
                t, p, na, nb, _ = ttest_ind(a[:, j], b[:, j])
                stars = p2stars(p)
                if not stars:
                    continue
                y_star = top_envelope[j]
                if not np.isfinite(y_star):
                    continue  # nothing plotted at this bucket
                y_draw = y_star + y_pad
                stars_to_draw.append((xs[j], y_draw, stars))
                if np.isfinite(y_draw):
                    max_star_y = max(max_star_y, y_draw)

                # Now set ylim high enough for stars *and* curves
                top_curves = max(ymaxs) if ymaxs else None
                top_needed = max_star_y if np.isfinite(max_star_y) else -np.inf
                if top_curves is None and not np.isfinite(top_needed):
                    pass
                else:
                    # add a small headroom
                    top_final = max(
                        (top_curves * 1.10) if top_curves is not None else -np.inf,
                        (top_needed * 1.05) if np.isfinite(top_needed) else -np.inf,
                    )
                    ax.set_ylim(bottom=0, top=top_final)

                # Finally draw the stars at the computed positions
                for xj, yj, s in stars_to_draw:
                    ax.text(
                        xj,
                        yj,
                        s,
                        ha="center",
                        va="bottom",
                        fontsize=customizer.in_plot_font_size,
                        color="0",
                        weight="bold",
                    )

                ax.set_title(f"training {trns[i].n}")
                ax.set_xlabel(
                    f"end points [min] of {bucket_len_minutes} min sync buckets"
                )
                if i == 0:
                    ax.set_ylabel("median dist. to\nreward circle center [mm]")
            if np.isfinite(max_star_y):
                max_star_y_per_ax[i] = max_star_y

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
