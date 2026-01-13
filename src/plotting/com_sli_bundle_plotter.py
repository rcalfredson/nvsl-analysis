import os
import warnings
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

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


def _load_bundle(path):
    d = np.load(path, allow_pickle=True)
    # required keys
    req = [
        "sli",
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

    # Look for optional keys by prefix
    for k in d.files:
        if k in out:
            continue
        if k.startswith(
            ("commag_", "wallpct_", "turnback_", "agarose_", "sli_")
        ) or k in ("sli_ts",):
            out[k] = d[k]
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

    # Match pandas select_extremes(): k based on total n (incl NaNs)
    k = max(1, int(n * fraction))
    finite_idx = np.flatnonzero(finite)
    k_eff = min(k, finite_idx.size)
    order = finite_idx[np.argsort(sli[finite_idx])]

    bottom = order[:k_eff].tolist()
    top = order[-k_eff:].tolist()

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
    metric="commag",
    turnback_mode="exp",  # exp | ctrl | exp_minus_ctrl
):
    """
    Plot COM magnitude or SLI vs sync bucket from one or more exported bundles.

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

    if metric == "commag":
        series_key = "commag_exp"
        need_keys = ["commag_exp"]
    elif metric == "sli":
        if any("sli_ts" not in b for b in bundles):
            raise ValueError("One or more bundles are missing sli_ts; re-export them.")
        series_key = "sli_ts"
        need_keys = ["sli_ts"]
        include_ctrl = False
    elif metric == "turnback":
        if turnback_mode == "exp":
            series_key = "turnback_ratio_exp"
            need_keys = ["turnback_ratio_exp"]
        elif turnback_mode == "ctrl":
            series_key = "turnback_ratio_ctrl"
            need_keys = ["turnback_ratio_ctrl"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "turnback_ratio_exp"
            need_keys = ["turnback_ratio_exp", "turnback_ratio_ctrl"]
            include_ctrl = False
        else:
            raise ValueError(f"Unknown turnback_mode={turnback_mode!r}")
    elif metric == "agarose":
        if turnback_mode == "exp":
            series_key = "agarose_ratio_exp"
            need_keys = ["agarose_ratio_exp"]
        elif turnback_mode == "ctrl":
            series_key = "agarose_ratio_ctrl"
            need_keys = ["agarose_ratio_ctrl"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "agarose_ratio_exp"
            need_keys = ["agarose_ratio_exp", "agarose_ratio_ctrl"]
            include_ctrl = False
        else:
            raise ValueError(f"Unknown mode={turnback_mode!r} for metric=agarose")
    elif metric == "wallpct":
        series_key = "wallpct_exp"
        need_keys = ["wallpct_exp"]
    else:
        raise ValueError(
            "Invalid metric specified; supported: 'commag', 'sli', 'wallpct'."
        )

    def _series_for_bundle(b):
        """
        Return array shaped (n_videos, n_trains, nb) for the requested plot.
        """
        if metric == "turnback" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["turnback_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["turnback_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "agarose" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["agarose_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["agarose_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        return np.asarray(b[series_key], dtype=float)

    for b in bundles:
        missing = [k for k in need_keys if k not in b]
        if missing:
            raise ValueError(
                f"Bundle {b.get('path', '<unknown>')} is missing keys for metric={metric}: {missing}"
            )

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
    s0 = _series_for_bundle(bundles[0])
    n_trains = s0.shape[1]
    if any(_series_for_bundle(b).shape[1] != n_trains for b in bundles):
        raise ValueError(
            f"Bundles disagree on number of trainings ({series_key}.shape[1])."
        )

    # optional training limit
    if num_trainings is not None:
        n_trains = min(n_trains, int(num_trainings))

    # nb
    nb = s0.shape[2]
    if any(_series_for_bundle(b).shape[2] != nb for b in bundles):
        raise ValueError(
            f"Bundles disagree on number of sync buckets ({series_key}.shape[2])."
        )
    nb = s0.shape[2]
    if nb == 0:
        raise ValueError(
            f"{series_key} has 0 buckets; check bundle export / data presence."
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
    if metric == "wallpct":
        ylim = [0.0, 100.0]
    elif metric == "turnback":
        ylim = [-0.5, 0.5] if turnback_mode == "exp_minus_ctrl" else [0.0, 0.5]
    elif metric == "agarose":
        # ratio is 0..1; exp-minus-ctrl can go negative
        ylim = [-0.5, 0.5] if turnback_mode == "exp_minus_ctrl" else [0.0, 1.0]
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

        # Per(training,bucket) label registry so stars can avoid overlapping
        # with existing text (n labels and other stars).
        lbls = defaultdict(list)  # key: bucket index -> list of text-ish objs

        # choose a training title from first bundle (best effort)
        try:
            tnames0 = bundles[0]["training_names"]
            title = str(tnames0[ti]) if len(tnames0) > ti else f"training {ti+1}"
        except Exception:
            title = f"training {ti+1}"

        # For 2-group t-tests, collect per-bucket mean and raw values
        do_ttests = ng == 2
        means_by_group = [None] * ng  # each: (nb,) float
        vals_by_group = [
            None
        ] * ng  # each: (nb,) list[np.ndarray] (finite per-bucket samples)

        # Each bundle is a "group"
        for gi, b in enumerate(bundles):
            sel_idx, both_labels, gid = selections[gi]
            if sel_idx.size == 0:
                continue

            series = _series_for_bundle(b)
            exp = series[sel_idx, ti, :]
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

            if metric == "wallpct":
                mci = mci.copy()
                mci[0, :] *= 100.0
                mci[1, :] *= 100.0
                mci[2, :] *= 100.0

            # For t-tests we compare what we're plotting (wallpct is scaled by 100)
            if do_ttests:
                exp_for_test = np.asarray(exp, dtype=float)
                if metric == "wallpct":
                    exp_for_test = exp_for_test * 100.0
                # Store finite samples per bucket
                vals_by_group[gi] = [
                    exp_for_test[np.isfinite(exp_for_test[:, bj]), bj]
                    for bj in range(exp_for_test.shape[1])
                ]
                means_by_group[gi] = np.asarray(mci[0, :], dtype=float)
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
                    txt = util.pltText(
                        xs[bj],
                        ys[bj] + 0.04 * (ylim[1] - ylim[0]),
                        f"{int(n)}",
                        ha="center",
                        size=customizer.in_plot_font_size,
                        color=".2",
                    )
                    # register for overlap avoidance
                    txt._y_ = float(ys[bj])
                    txt._final_y_ = float(ys[bj] + 0.04 * (ylim[1] - ylim[0]))
                    lbls[bj].append(txt)

            # ctrl overlay (optional)
            if include_ctrl:
                if metric == "commag":
                    ctrl_key = "commag_ctrl"
                elif metric == "wallpct":
                    ctrl_key = "wallpct_ctrl"
                elif metric == "turnback":
                    ctrl_key = "turnback_ratio_ctrl"
                elif metric == "agarose":
                    ctrl_key = "agarose_ratio_ctrl"
                else:
                    ctrl_key = None
                if ctrl_key is None:
                    continue
                ctrl_arr = np.asarray(b[ctrl_key], dtype=float)
                if ctrl_arr.shape[0] != len(b["sli"]):
                    print(
                        f"[plot] WARNING: {b['path']} {ctrl_key} shape mismatch; skipping ctrl overlay"
                    )
                    continue
                ctrl = ctrl_arr[sel_idx, ti, :]
                mci_c = _mean_ci_over_videos(ctrl)
                if metric == "wallpct":
                    mci_c = mci_c.copy()
                    mci_c[0, :] *= 100.0
                    mci_c[1, :] *= 100.0
                    mci_c[2, :] *= 100.0
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

        # ---- Two-group t-tests + star annotations (plotRewards-style) ----
        if (
            do_ttests
            and all(m is not None for m in means_by_group)
            and all(v is not None for v in vals_by_group)
        ):
            m0 = means_by_group[0]
            m1 = means_by_group[1]
            for bj in range(nb):
                x0 = vals_by_group[0][bj]
                x1 = vals_by_group[1][bj]

                # Require some data on both sides
                if x0.size < 2 or x1.size < 2:
                    continue

                try:
                    _t, p = ttest_ind(x0, x1)[:2]
                except Exception:
                    continue

                stars = util.p2stars(p, nanR="")

                # Choose anchor near the higher mean of the two groups at this bucket
                if not (np.isfinite(m0[bj]) or np.isfinite(m1[bj])):
                    continue
                anchor_y = np.nanmax([m0[bj], m1[bj]])

                # Avoid y-collisions with existing labels at this bucket
                txts_here = lbls.get(bj, [])
                avoid_ys = [
                    (t._final_y_ if hasattr(t, "_final_y_") else t._y_)
                    for t in txts_here
                    if hasattr(t, "_final_y_") or hasattr(t, "_y_")
                ]
                if np.isfinite(anchor_y):
                    avoid_ys.append(float(anchor_y))

                # plotRewards-style base y: above the anchor, but choose below if near top margin
                span = ylim[1] - ylim[0]
                base_y_for_star = float(anchor_y) + 0.04 * span
                prefer = "above"
                margin = 0.05 * (ylim[1] - ylim[0])
                if base_y_for_star + 0.15 * (ylim[1] - ylim[0]) > ylim[1] - margin:
                    prefer = "below"

                ys_star, va_align = pick_non_overlapping_y(
                    base_y_for_star, avoid_ys, ylim, prefer=prefer
                )

                txt = util.pltText(
                    xs[bj],
                    ys_star,
                    stars,
                    ha="center",
                    va=va_align,
                    size=customizer.in_plot_font_size,
                    color="0",
                    weight="bold",
                )
                txt._y_ = float(anchor_y)
                txt._final_y_ = float(ys_star)
                lbls[bj].append(txt)

        plt.title(maybe_sentence_case(title))
        plt.xlabel(maybe_sentence_case(f"end points [min] of {blf} min sync buckets"))

        if metric == "commag":
            y_label = "COM dist. to circle center [mm]"
        elif metric == "sli":
            y_label = "SLI"
        elif metric == "turnback":
            if turnback_mode == "exp_minus_ctrl":
                y_label = "Dual-circle turnback (exp - yoked)"
            elif turnback_mode == "ctrl":
                y_label = "Dual-circle turnback (yoked)"
            else:
                y_label = "Dual-circle turnback ratio"
        elif metric == "agarose":
            if turnback_mode == "exp_minus_ctrl":
                y_label = "Agarose avoidance (exp - yoked)"
            elif turnback_mode == "ctrl":
                y_label = "Agarose avoidance (yoked)"
            else:
                y_label = "Agarose avoidance ratio"
        elif metric == "wallpct":
            y_label = "% time on wall"
        if ti == 0:
            plt.ylabel(maybe_sentence_case(y_label))
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
