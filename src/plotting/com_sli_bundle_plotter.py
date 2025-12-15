import os
import warnings
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.utils.common import maybe_sentence_case, pch, writeImage
from src.plotting.plot_customizer import PlotCustomizer


def _as_scalar(x):
    # handle np arrays holding single objects/scalars
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _load_bundle(path):
    d = np.load(path, allow_pickle=True)
    # required keys
    req = [
        "sli",
        "commag_exp",
        "commag_ctrl",
        "group_label",
        "bucket_len_min",
        "training_names",
        "video_ids",
        "sli_training_idx",
        "sli_use_training_mean",
    ]
    missing = [k for k in req if k not in d.files]
    if missing:
        raise ValueError(f"Bundle {path} is missing keys: {missing}")

    out = {k: d[k] for k in req}
    out["path"] = path

    # normalize a few things
    out["group_label"] = str(_as_scalar(out["group_label"]))
    out["bucket_len_min"] = float(_as_scalar(out["bucket_len_min"]))
    out["sli_training_idx"] = int(_as_scalar(out["sli_training_idx"]))
    out["sli_use_training_mean"] = bool(_as_scalar(out["sli_use_training_mean"]))
    return out


def _fmt_bucket_len(bl):
    # mimic “blf” vibe (integer when possible)
    if np.isfinite(bl) and abs(bl - round(bl)) < 1e-9:
        return str(int(round(bl)))
    return f"{bl:g}"


def _select_sli_extremes(sli, fraction, which):
    """
    sli: (n_videos,) float
    which: None | "top" | "bottom" | "both"
    returns:
      - indices (np.ndarray of ints) into videos
      - group_labels (list[str]) if which=="both" else None
      - group_ids (np.ndarray) same length as indices, for legend grouping (0/1)
    """
    n = len(sli)
    if which is None:
        idx = np.arange(n, dtype=int)
        return idx, None, np.zeros(n, dtype=int)

    sli = np.asarray(sli, dtype=float)
    finite = np.isfinite(sli)
    if not finite.any():
        return (
            np.array([], dtype=int),
            (["Bottom", "Top"] if which == "both" else None),
            np.array([], dtype=int),
        )

    k = max(1, int(round(fraction * finite.sum())))
    finite_idx = np.flatnonzero(finite)
    order = finite_idx[np.argsort(sli[finite_idx])]

    bottom = order[:k].tolist()
    top = order[-k:].tolist()

    if which == "bottom":
        idx = np.array(bottom, dtype=int)
        return idx, None, np.zeros(len(idx), dtype=int)
    if which == "top":
        idx = np.array(top, dtype=int)
        return idx, None, np.zeros(len(idx), dtype=int)
    if which == "both":
        idx = np.array(bottom + top, dtype=int)
        # 0 for bottom, 1 for top
        gid = np.array([0] * len(bottom) + [1] * len(top), dtype=int)
        return idx, [f"Bottom {int(fraction*100)}%", f"Top {int(fraction*100)}%"], gid

    raise ValueError(f"Unknown which={which!r}")


def _mean_ci_over_videos(vals_2d):
    """
    vals_2d: (n_videos, nb) with NaNs allowed
    returns mci: (4, nb) [mean, lo, hi, n]
    """
    nb = vals_2d.shape[1]
    mci = np.array([util.meanConfInt(vals_2d[:, b]) for b in range(nb)]).T
    return mci


def plot_com_sli_bundles(
    bundle_paths,
    out_fn,
    *,
    labels=None,
    num_trainings=None,
    include_ctrl=False,
    sli_extremes=None,  # None | "top" | "bottom" | "both"
    sli_fraction=0.2,
    opts=None,
):
    """
    Plot COM magnitude vs sync bucket from one or more exported bundles.

    - Each bundle is a “group” (regular / antennae-removed / PFN-silenced, etc).
    - Lines are mean over videos; shaded region is CI.
    - Optionally filter within each bundle by SLI percentile.
    """
    if opts is None:
        # minimal opts object for PlotCustomizer + writeImage usage
        opts = SimpleNamespace(
            wspace=0.35,
            imageFormat="png",
        )

    bundles = [_load_bundle(p) for p in bundle_paths]
    ng = len(bundles)
    if ng == 0:
        raise ValueError("No bundles provided")

    if labels is not None:
        if len(labels) != ng:
            raise ValueError("labels length must match number of bundles")
        group_labels = list(labels)
    else:
        group_labels = [b["group_label"] for b in bundles]

    # Consistency checks
    bls = np.array([b["bucket_len_min"] for b in bundles], dtype=float)
    if not np.all(np.isfinite(bls)):
        warnings.warn(
            "One or more bundles have non-finite bucket_len_min; x-axis may be wrong."
        )
    else:
        if np.nanmax(bls) - np.nanmin(bls) > 1e-6:
            raise ValueError(f"bucket_len_min differs across bundles: {bls}")
    bl = float(bls[0])
    blf = _fmt_bucket_len(bl)

    # training names: require consistent length; content may vary slightly
    n_trains = bundles[0]["commag_exp"].shape[1]
    if any(b["commag_exp"].shape[1] != n_trains for b in bundles):
        raise ValueError(
            "Bundles disagree on number of trainings (commag_exp.shape[1])."
        )

    # optional training limit
    if num_trainings is not None:
        n_trains = min(n_trains, int(num_trainings))

    # nb
    nb = bundles[0]["commag_exp"].shape[2]
    if any(b["commag_exp"].shape[2] != nb for b in bundles):
        raise ValueError(
            "Bundles disagree on number of sync buckets (commag_exp.shape[2])."
        )

    # x positions: end points in minutes, matching plotRewards() for non-post
    xs = (np.arange(nb) + 1) * bl

    # Styling: mimic plotRewards commag behavior (group via linestyle)
    # Use matplotlib default C0/C1 to stay close to existing look.
    exp_color = "C0"
    ctrl_color = "C1"
    linestyles = ["-", "--", ":", "-."]  # extend if needed

    customizer = PlotCustomizer()

    nc = n_trains
    figsize = pch(
        ([5.33, 11.74, 18.18][nc - 1] if nc <= 3 else 18.18 + (nc - 3) * 6.0, 4.68),
        ((10, 15, 20)[nc - 1] if nc <= 3 else 20 + (nc - 3) * 6.0, 5),
    )
    fig, axs = plt.subplots(1, nc, figsize=figsize)
    if nc == 1:
        axs = np.array([axs])

    # Track global y-lims (like dynamic behavior in plotRewards)
    ylim = [-1.0, 1.0]
    mci_min, mci_max = None, None

    # If "both" mode, we effectively double “groups” per bundle.
    # But per your stated goal, we’ll keep it simple:
    # - "both" draws bottom/top as two linestyles PER bundle *would* get messy,
    #   so we treat "both" as two pseudo-groups overall with labels.
    # For now: support both, but only when a single bundle is provided.
    if sli_extremes == "both" and ng != 1:
        raise ValueError(
            'sli_extremes="both" is supported only when plotting a single bundle (to keep the plot readable).'
        )

    # Build per-bundle selections
    selections = []
    for b in bundles:
        idx, both_labels, gid = _select_sli_extremes(
            b["sli"], sli_fraction, sli_extremes
        )
        selections.append((idx, both_labels, gid))

    # annotate SLI selection if used
    if sli_extremes is not None:
        # Assume consistent SLI settings; if not, still show the first and warn.
        stis = [b["sli_training_idx"] for b in bundles]
        means = [b["sli_use_training_mean"] for b in bundles]
        if len(set(stis)) > 1 or len(set(means)) > 1:
            warnings.warn(
                "Bundles disagree on sli_training_idx and/or sli_use_training_mean; annotation may be misleading."
            )
        sli_mode = (
            "mean over buckets"
            if bool(means[0])
            else "single bucket (per exporter settings)"
        )
        fig.text(
            0.02,
            0.98,
            f"SLI filter: {sli_extremes} {int(sli_fraction*100)}% within group; T{stis[0]+1}; {sli_mode}",
            ha="left",
            va="top",
            fontsize=customizer.in_plot_font_size,
            color="0",
        )

    # plotting
    for ti in range(n_trains):
        ax = axs[ti]
        plt.sca(ax)

        # choose a training title from first bundle (best effort)
        try:
            tnames0 = bundles[0]["training_names"]
            title = str(tnames0[ti]) if len(tnames0) > ti else f"training {ti+1}"
        except Exception:
            title = f"training {ti+1}"

        # Each bundle is a “group”
        for gi, b in enumerate(bundles):
            sel_idx, both_labels, gid = selections[gi]
            if sel_idx.size == 0:
                continue

            exp = np.asarray(b["commag_exp"], dtype=float)[sel_idx, ti, :]
            mci = _mean_ci_over_videos(exp)

            # update global min/max for dynamic y-lims
            if not np.all(np.isnan(mci[1, :])):
                mci_min = (
                    np.nanmin(mci[1, :])
                    if mci_min is None
                    else min(mci_min, np.nanmin(mci[1, :]))
                )
            if not np.all(np.isnan(mci[2, :])):
                mci_max = (
                    np.nanmax(mci[2, :])
                    if mci_max is None
                    else max(mci_max, np.nanmax(mci[2, :]))
                )

            ys = mci[0, :]
            fin = np.isfinite(ys)
            ls = linestyles[gi % len(linestyles)]
            (line,) = plt.plot(
                xs[fin],
                ys[fin],
                color=exp_color,
                marker="o",
                ms=3,
                mec=exp_color,
                linewidth=2,
                linestyle=ls,
            )
            if ti == 0:
                line.set_label(group_labels[gi])

            # CI shading
            if np.isfinite(mci[1, :]).any() and np.isfinite(mci[2, :]).any():
                plt.fill_between(
                    xs[fin], mci[1, :][fin], mci[2, :][fin], color=exp_color, alpha=0.15
                )

            # optionally show n per bucket (like plotRewards)
            for bj, n in enumerate(mci[3, :]):
                if n > 0 and np.isfinite(ys[bj]):
                    util.pltText(
                        xs[bj],
                        ys[bj] + 0.04 * (ylim[1] - ylim[0]),
                        f"{int(n)}",
                        ha="center",
                        size=customizer.in_plot_font_size,
                        color=".2",
                    )

            # ctrl overlay (optional)
            if include_ctrl:
                ctrl = np.asarray(b["commag_ctrl"], dtype=float)[sel_idx, ti, :]
                mci_c = _mean_ci_over_videos(ctrl)
                ys_c = mci_c[0, :]
                fin_c = np.isfinite(ys_c)
                if fin_c.any():
                    plt.plot(
                        xs[fin_c],
                        ys_c[fin_c],
                        color=ctrl_color,
                        marker="o",
                        ms=3,
                        mec=ctrl_color,
                        linewidth=2,
                        linestyle=ls,
                        alpha=0.95,
                    )
                    if (
                        np.isfinite(mci_c[1, :]).any()
                        and np.isfinite(mci_c[2, :]).any()
                    ):
                        plt.fill_between(
                            xs[fin_c],
                            mci_c[1, :][fin_c],
                            mci_c[2, :][fin_c],
                            color=ctrl_color,
                            alpha=0.12,
                        )

        plt.title(maybe_sentence_case(title))
        plt.xlabel(maybe_sentence_case(f"end points [min] of {blf} min sync buckets"))
        if ti == 0:
            plt.ylabel(maybe_sentence_case("COM dist. to circle center [mm]"))
        plt.axhline(color="k")
        plt.xlim(0, xs[-1])

    # Dynamic y-lims similar to plotRewards behavior
    if mci_max is not None and np.isfinite(mci_max):
        base_pad = 1.1
        if mci_max > ylim[1]:
            ylim[1] = mci_max * base_pad
    if mci_min is not None and np.isfinite(mci_min):
        if mci_min < ylim[0]:
            ylim[0] = mci_min * 1.2

    for ax in fig.get_axes():
        ax.set_ylim(ylim[0], ylim[1])

    # legend
    axs[0].legend(frameon=False)
    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally(wspace=getattr(opts, "wspace", 0.35))

    # save
    base, ext = os.path.splitext(out_fn)
    if ext == "":
        out_fn = base + ".png"
    writeImage(out_fn, format=getattr(opts, "imageFormat", "png"))
    plt.close()
