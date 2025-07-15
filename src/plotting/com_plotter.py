import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.util import meanConfInt, slugify


def plot_com_distance(
    vas,  # list of VideoAnalysis instances
    trns,  # list of Training instances (same for all videos)
    bucket_len_minutes,  # minutes per sync bucket (same df you used)
    gis=None,  # array of group-indices per video
    gls=None,  # list of group labels
    outdir: str = None,
    title: str = "COM → reward distance",
):
    """
    vas.shape → (n_videos)
    trns: length = n_trns
    Each va.syncCOMDist[t_idx]['exp'] is a length-n_buckets list of distances
    If control present, va.syncCOMDist[t_idx]['ctrl'] likewise.
    """
    # ----- 1) pack into a 4D array: (vid, trn, fly, bucket) -----
    n_videos = len(vas)
    n_trns = len(trns)
    # assume all videos have same bucket count for training 0:
    n_buckets = len(vas[0].syncCOMDist[0]["exp"])
    has_ctrl = "ctrl" in vas[0].syncCOMDist[0]
    n_flies = 2 if has_ctrl else 1

    com4 = np.full((n_videos, n_trns, n_flies, n_buckets), np.nan)
    for vid, va in enumerate(vas):
        for t_idx, dist_dict in enumerate(va.syncCOMDist):
            # exp
            com4[vid, t_idx, 0, : len(dist_dict["exp"])] = dist_dict["exp"]
            # ctrl (if present)
            if has_ctrl:
                com4[vid, t_idx, 1, : len(dist_dict.get("ctrl", []))] = dist_dict.get(
                    "ctrl", []
                )

    # Delegate multi-group case to recursion
    if gis is not None and gls is not None and len(set(gis)) > 1:
        base_dir = outdir or "."
        os.makedirs(base_dir, exist_ok=True)
        for g, label in enumerate(gls):
            idxs = np.flatnonzero(np.array(gis) == g)
            if idxs.size == 0:
                continue
            # create a subdirectory per group
            group_dir = os.path.join(base_dir, slugify(label))
            os.makedirs(group_dir, exist_ok=True)
            plot_com_distance(
                [vas[i] for i in idxs],
                trns,
                bucket_len_minutes,
                gis=None,
                gls=None,
                outdir=group_dir,
                title=f"{title} — {label}",
            )
        return

    # ----- 2) compute mean ± CI for each fly type -----
    # slice out experimental (fly=0) and control (fly=1) arrays:
    exp_arr = com4[:, :, 0, :]  # shape (n_videos, n_trns, n_buckets)
    if has_ctrl:
        ctrl_arr = com4[:, :, 1, :]
    else:
        ctrl_arr = None

    def mean_ci(data: np.ndarray):
        # data: (n_videos, n_trns, n_buckets)
        means = np.empty((n_trns, n_buckets))
        deltas = np.empty((n_trns, n_buckets))
        counts = np.empty((n_trns, n_buckets), int)
        for i in range(n_trns):
            for j in range(n_buckets):
                m, d, n = meanConfInt(data[:, i, j], conf=0.95, asDelta=True)
                means[i, j] = m
                deltas[i, j] = d
                counts[i, j] = n
        return means, deltas, counts

    m_exp, ci_exp, n_exp = mean_ci(exp_arr)
    if has_ctrl:
        m_ctrl, ci_ctrl, n_ctrl = mean_ci(ctrl_arr)

    # ----- 3) build x-axis -----
    xs = (np.arange(n_buckets) + 1) * bucket_len_minutes
    xlim = (0, xs[-1] + bucket_len_minutes)

    # ----- 4) plot -----
    fig, axes = plt.subplots(1, n_trns, figsize=(4 * n_trns, 4), sharey=True)
    if n_trns == 1:
        axes = [axes]

    exp_color = "#1f4da1"
    ctrl_color = "#a00000"
    alpha = 0.2
    ms = 4

    for i, ax in enumerate(axes):
        # CI bands
        ax.fill_between(
            xs, m_exp[i] - ci_exp[i], m_exp[i] + ci_exp[i], color=exp_color, alpha=alpha
        )
        ax.plot(
            xs,
            m_exp[i],
            marker="o",
            markersize=ms,
            color=exp_color,
            label="Experimental",
        )

        if has_ctrl:
            ax.fill_between(
                xs,
                m_ctrl[i] - ci_ctrl[i],
                m_ctrl[i] + ci_ctrl[i],
                color=ctrl_color,
                alpha=alpha,
            )
            ax.plot(
                xs,
                m_ctrl[i],
                marker="o",
                markersize=ms,
                color=ctrl_color,
                label="Control",
            )

        # annotate counts just below each CI
        y_min, y_max = ax.get_ylim()
        dy = 0.02 * (y_max - y_min)
        for j, x in enumerate(xs):
            ax.text(
                x,
                m_exp[i, j] - dy,
                str(n_exp[i, j]),
                ha="center",
                va="top",
                fontsize=8,
                color=exp_color,
            )
            if has_ctrl:
                ax.text(
                    x,
                    m_ctrl[i, j] + dy,
                    str(n_ctrl[i, j]),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=ctrl_color,
                )

        ax.set_title(f"Training {trns[i].n}")
        ax.set_xlabel("Sync-bucket end [min]")
        ax.set_xlim(*xlim)
        if i == 0:
            ax.set_ylabel("Distance to reward center [px]")

    axes[-1].legend(loc="best")
    fig.suptitle(title)
    plt.tight_layout()
    ax.set_ylim(bottom=0)

    # ----- 5) save/show -----
    if outdir:
        save_dir = outdir
        os.makedirs(save_dir, exist_ok=True)
        fn = f"{slugify(title)}.png"
        save_path = os.path.join(save_dir, fn)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved COM-distance plot to {save_path}")
    else:
        plt.show()
