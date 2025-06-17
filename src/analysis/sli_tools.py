import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from src.utils.util import meanConfInt, slugify


def plot_sli_extremes(
    perf4: np.ndarray,  # shape: (n_videos, n_trns, 2, n_buckets)
    trns: List,  # list of Training instances
    bucket_len_minutes: float,  # minutes per sync bucket
    selected_bottom: Optional[List[int]] = None,
    selected_top: Optional[List[int]] = None,
    gis: Optional[np.ndarray] = None,
    gls: Optional[List[str]] = None,
    training_idx: Optional[int] = None,
    fraction: float = 0.1,
    outdir: Optional[str] = None,
    title: str = "SLI over time",
    tp: Optional[str] = None,
    n_nonpost_buckets: Optional[int] = None,
) -> None:
    """
    Plot or save the SLI time series for bottom/top flies, handling both training and post-training.

    If `gis` and `gls` specify multiple groups, generates one plot per group.

    Parameters:
    - perf4: np.ndarray of shape (n_videos, n_trns, 2, n_buckets)
    - trns: list of Training objects
    - bucket_len_minutes: length of each sync bucket in minutes
    - selected_bottom: indices of bottom flies for single-group plot
    - selected_top: indices of top flies for single-group plot
    - gis: optional array of group indices per video
    - gls: optional list of group labels
    - training_idx: index for computing extremes if not precomputed
    - fraction: fraction of flies to select when computing extremes
    - outdir: directory or filepath to save PNG; if None, will plt.show()
    - title: base title for plot
    - tp: optional string identifying type (e.g. 'rpid' or 'rpipd') for filenames
    - n_nonpost_buckets: number of nonpost sync buckets (required for post-training x-axis)
    """
    # Multi-group case: one plot per group
    if gis is not None and gls is not None and len(set(gis)) > 1:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        for g, label in enumerate(gls):
            idxs = np.flatnonzero(gis == g)
            if idxs.size == 0:
                continue
            perf4_g = perf4[idxs]
            # compute new extremes for this group
            if training_idx is None:
                raise ValueError("training_idx required to compute extremes.")

            sli_ser_g = compute_sli_per_fly(perf4_g, training_idx)
            bottom_g, top_g = select_extremes(sli_ser_g, fraction)
            # filename and title
            slug = slugify(label)
            suffix = slugify(tp) if tp else "train"
            fname = f"sli_extremes_{suffix}_train{training_idx+1}_{slug}.png"
            fig_title = f"{title} — {label}"
            # recurse for single-group plotting and saving
            plot_sli_extremes(
                perf4=perf4_g,
                trns=trns,
                bucket_len_minutes=bucket_len_minutes,
                selected_bottom=bottom_g,
                selected_top=top_g,
                gis=None,
                gls=None,
                training_idx=None,
                fraction=fraction,
                outdir=os.path.join(outdir, fname) if outdir else None,
                title=fig_title,
                tp=tp,
                n_nonpost_buckets=n_nonpost_buckets,
            )
        return

    # Single-group base plotting
    sli = perf4[:, :, 0, :] - perf4[:, :, 1, :]
    if selected_bottom is None or selected_top is None:
        raise ValueError(
            "selected_bottom and selected_top required for single-group plot."
        )
    sli_bot = sli[selected_bottom]
    sli_top = sli[selected_top]

    
    def mean_ci(group: np.ndarray):
        """
        group shape: (n_flies, n_trns, n_buckets)
        returns:
            means, cis, counts: each shape (n_trns, n_buckets)
        """
        n_trns = group.shape[1]
        n_buckets = group.shape[2]
        means = np.empty((n_trns, n_buckets))
        ci_deltas = np.empty((n_trns, n_buckets))
        counts = np.empty((n_trns, n_buckets), dtype=int)

        for i in range(n_trns):
            for j in range(n_buckets):
                vals = group[:, i, j]
                m, d, n = meanConfInt(vals, conf=0.95, asDelta=True)
                means[i, j] = m
                ci_deltas[i, j] = d
                counts[i, j] = n

        return means, ci_deltas, counts

    m_bot, ci_bot, n_bot = mean_ci(sli_bot)
    m_top, ci_top, n_top = mean_ci(sli_top)

    # determine if post-training and compute x-axis
    n_buckets = sli.shape[-1]
    is_post = tp is not None and tp.startswith("rpip")
    if is_post:
        if n_nonpost_buckets is None:
            raise ValueError("n_nonpost_buckets required for post-training plots.")
        xs = (np.arange(n_buckets) - (n_nonpost_buckets - 1)) * bucket_len_minutes
        xlim = (xs[0] - bucket_len_minutes, xs[-1] + bucket_len_minutes)
    else:
        xs = (np.arange(n_buckets) + 1) * bucket_len_minutes
        xlim = (0, xs[-1])

    # create subplots
    n_trns = sli.shape[1]
    fig, axes = plt.subplots(1, n_trns, figsize=(4 * n_trns, 4), sharey=True)
    if n_trns == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        bottom_color = "#a00000"
        top_color = "#1f4da1"
        ci_alpha = 0.2
        ms = 4
        # draw CI bands
        ax.fill_between(
            xs,
            m_bot[i] - ci_bot[i],
            m_bot[i] + ci_bot[i],
            color=bottom_color,
            alpha=ci_alpha,
        )
        ax.fill_between(
            xs,
            m_top[i] - ci_top[i],
            m_top[i] + ci_top[i],
            color=top_color,
            alpha=ci_alpha,
        )
        pct = int(fraction * 100)
        bottom_label = f"Bottom {pct}%"
        top_label = f"Top {pct}%"

        # draw means
        ax.plot(
            xs,
            m_bot[i],
            marker="o",
            markersize=ms,
            color=bottom_color,
            label=bottom_label,
        )
        ax.plot(
            xs, m_top[i], marker="o", markersize=ms, color=top_color, label=top_label
        )
        ax.axhline(0, color='black', linewidth=1)
        # annotate counts
        y_min, y_max = ax.get_ylim()
        offset = 0.02 * (y_max - y_min)
        for j, x in enumerate(xs):
            ax.text(
                x,
                m_bot[i, j] - offset,
                str(int(n_bot[i, j])),
                ha="center",
                va="top",
                fontsize=8,
                color=bottom_color,
            )
            ax.text(
                x,
                m_top[i, j] + offset,
                str(int(n_top[i, j])),
                ha="center",
                va="bottom",
                fontsize=8,
                color=top_color,
            )
        # labels and limits
        ax.set_title(f"Training {trns[i].n}")
        ax.set_xlabel("Sync-bucket end [min]")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, 1.50)
        if i == 0:
            ax.set_ylabel("RPI (exp - yok)")

    axes[-1].legend(loc="best")
    fig.suptitle(title)
    plt.tight_layout()

    # save or show
    if outdir:
        if outdir.lower().endswith((".png", ".pdf")):
            save_path = outdir
        else:
            os.makedirs(outdir, exist_ok=True)
            # pick whether this is the training-period or post-training plot
            suffix = "train" if tp == "rpid" else "post"
            idx = f"_t{training_idx+ 1}" if training_idx is not None else ""
            fname = f"sli_extremes_{suffix}{idx}.png"
            save_path = os.path.join(outdir, fname)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved SLI extremes plot to {save_path}")
    else:
        plt.show()


def compute_sli_per_fly(
    perf4: np.ndarray, training_idx: int, bucket_idx: Optional[int] = None
) -> pd.Series:
    if bucket_idx is None:
        bucket_idx = perf4.shape[3] - 2
    sli = {
        vid: perf4[vid, training_idx, 0, bucket_idx]
        - perf4[vid, training_idx, 1, bucket_idx]
        for vid in range(perf4.shape[0])
    }
    return pd.Series(sli, name="SLI").astype(float)


def select_extremes(
    sli_series: pd.Series, fraction: float = 0.1
) -> Tuple[List[int], List[int]]:
    n = len(sli_series)
    k = max(1, int(n * fraction))
    bottom = sli_series.nsmallest(k).index.tolist()
    top = sli_series.nlargest(k).index.tolist()
    return bottom, top
