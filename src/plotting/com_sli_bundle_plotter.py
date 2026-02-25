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
            (
                "commag_",
                "weaving_",
                "wallpct_",
                "turnback_",
                "agarose_",
                "lgturn_",
                "reward_lgturn_",
                "reward_lv_",
                "sli_",
            )
        ) or k in ("sli_ts",):
            out[k] = d[k]
    return out


def _as_str_array(x):
    # video_ids often come out as dtype=object arrays
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([str(v) for v in arr], dtype=object)


def _bundle_video_ids(bundle):
    key = "video_uid" if "video_uid" in bundle else "video_ids"
    if key not in bundle:
        return None
    return _as_str_array(bundle[key])


def _align_by_video_ids(base_bundle, comp_bundle):
    """
    Returns:
      base_idx, comp_idx, n_match
    Where base_idx and comp_idx are index arrays selecting matched videos in the SAME ID order.
    """
    if "video_ids" not in base_bundle or "video_ids" not in comp_bundle:
        return None, None, 0
    base_ids = _bundle_video_ids(base_bundle)
    comp_ids = _bundle_video_ids(comp_bundle)
    if base_ids is None or comp_ids is None:
        return None, None, 0

    base_map = {vid: i for i, vid in enumerate(base_ids)}
    comp_map = {vid: i for i, vid in enumerate(comp_ids)}

    common = [vid for vid in comp_ids if vid in base_map]
    if not common:
        return None, None, 0

    base_idx = np.array([base_map[vid] for vid in common], dtype=int)
    comp_idx = np.array([comp_map[vid] for vid in common], dtype=int)
    return base_idx, comp_idx, int(len(common))


def _check_delta_compat(base, comp, *, keys, metric_label="metric"):
    """
    Basic sanity checks so we don't silently subtract apples from wheelbarrows.
    """
    # bucket length consistency
    bl0 = float(_as_scalar(base.get("bucket_len_min", np.nan)))
    bl1 = float(_as_scalar(comp.get("bucket_len_min", np.nan)))
    if np.isfinite(bl0) and np.isfinite(bl1) and abs(bl0 - bl1) > 1e-6:
        raise ValueError(
            f"Cannot delta: bucket_len_min differs (base={bl0}, comp={bl1})."
        )

    # require series arrays exist and shapes compatible
    for k in keys:
        if k not in base:
            raise ValueError(f"Baseline bundle missing key {k!r} for {metric_label}.")
        if k not in comp:
            raise ValueError(f"Bundle missing key {k!r} for {metric_label}.")

    # optional: turnback-specific metadata checks (safe to ignore if absent)
    # If both present, require equality for inner_delta_mm since you're testing pixel rounding.
    if "turnback_inner_delta_mm" in base and "turnback_inner_delta_mm" in comp:
        d0 = float(_as_scalar(base["turnback_inner_delta_mm"]))
        d1 = float(_as_scalar(comp["turnback_inner_delta_mm"]))
        if abs(d0 - d1) > 1e-9:
            raise ValueError(
                f"Cannot delta turnback: turnback_inner_delta_mm differs (base={d0}, comp={d1})."
            )


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


def _anova_p_per_bucket(group_samples, *, min_n_per_group=3):
    """
    group_samples: list[np.ndarray], each 1D
    Returns: p-value (float) or np.nan
    Notes:
      - NaNs/infs are removed.
      - Requires >=2 groups with >=min_n_per_group samples each.
    """
    try:
        clean = []
        for g in group_samples:
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
    delta_vs_path=None,  # baseline bundle path; if set, plot (bundle - baseline)
    delta_label=None,  # label prefix in delta mode
    delta_ylabel=None,
    delta_allow_unpaired=False,
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

    base_bundle = None
    if delta_vs_path:
        base_bundle = _load_bundle(delta_vs_path)

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
            include_ctrl = False
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
    elif metric == "weaving":
        if turnback_mode == "exp":
            series_key = "weaving_ratio_exp"
            need_keys = ["weaving_ratio_exp"]
            include_ctrl = False
        elif turnback_mode == "ctrl":
            series_key = "weaving_ratio_ctrl"
            need_keys = ["weaving_ratio_ctrl"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "weaving_ratio_exp"
            need_keys = ["weaving_ratio_exp", "weaving_ratio_ctrl"]
            include_ctrl = False
        else:
            raise ValueError(
                f"Unknown turnback_mode={turnback_mode!r} for metric=weaving"
            )
    elif metric == "agarose":
        if turnback_mode == "exp":
            series_key = "agarose_ratio_exp"
            need_keys = ["agarose_ratio_exp"]
            include_ctrl = False
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
    elif metric == "lgturn_startdist":
        series_key = "lgturn_startdist_exp"
        need_keys = ["lgturn_startdist_exp"]
    elif metric == "reward_lgturn_pathlen":
        if turnback_mode == "exp":
            series_key = "reward_lgturn_pathlen_exp"
            need_keys = ["reward_lgturn_pathlen_exp"]
            include_ctrl = False
        elif turnback_mode == "ctrl":
            series_key = "reward_lgturn_pathlen_ctrl"
            need_keys = ["reward_lgturn_pathlen_ctrl"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lgturn_pathlen_exp"
            need_keys = ["reward_lgturn_pathlen_exp", "reward_lgturn_pathlen_ctrl"]
            include_ctrl = False
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=reward_lgturn_pathlen"
            )
    elif metric == "reward_lv":
        if turnback_mode == "exp":
            series_key = "reward_lv_exp"
            need_keys = ["reward_lv_exp"]
            include_ctrl = False
        elif turnback_mode == "ctrl":
            series_key = "reward_lv_ctrl"
            need_keys = ["reward_lv_ctrl"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lv_exp"
            need_keys = ["reward_lv_exp", "reward_lv_ctrl"]
            include_ctrl = False
        else:
            raise ValueError(f"Unknown mode={turnback_mode!r} for metric=reward_lv")
    elif metric == "reward_lgturn_prevalence":
        # Prevalence = (# rewards with detect post-reward turn) / (# rewards)
        # The underlying arrays are per-video, per-training, per-bucket.
        if turnback_mode == "exp":
            series_key = "reward_lgturn_pathlenN_exp"
            need_keys = ["reward_lgturn_pathlenN_exp", "reward_lgturn_rewards"]
            include_ctrl = False
        elif turnback_mode == "ctrl":
            series_key = "reward_lgturn_pathlenN_ctrl"
            need_keys = ["reward_lgturn_pathlenN_ctrl", "reward_lgturn_rewards"]
            include_ctrl = False
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lgturn_pathlenN_exp"
            need_keys = [
                "reward_lgturn_pathlenN_exp",
                "reward_lgturn_pathlenN_ctrl",
                "reward_lgturn_rewards",
            ]
            include_ctrl = False
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=reward_lgturn_prevalence"
            )
    else:
        raise ValueError(
            "Invalid metric specified; supported: 'commag', 'sli', 'turnback', "
            "'agarose', 'wallpct', 'lgturn_startdist', 'reward_lgturn_pathlen', "
            "'reward_lv', 'reward_lgturn_prevalence', 'weaving'."
        )

    def _series_for_bundle(b):
        """
        Return array shaped (n_videos, n_trains, nb) for the requested plot.
        """
        if metric == "turnback" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["turnback_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["turnback_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "weaving" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["weaving_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["weaving_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "agarose" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["agarose_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["agarose_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "reward_lgturn_pathlen" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["reward_lgturn_pathlen_exp"], dtype=float)
            ctrl_arr = np.asarray(b["reward_lgturn_pathlen_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "reward_lv" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["reward_lv_exp"], dtype=float)
            ctrl_arr = np.asarray(b["reward_lv_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "reward_lgturn_prevalence":
            rewards = np.asarray(b["reward_lgturn_rewards"], dtype=float)
            rewards_safe = np.where(rewards > 0, rewards, np.nan)

            if turnback_mode == "exp":
                turns = np.asarray(b["reward_lgturn_pathlenN_exp"], dtype=float)
                return turns / rewards_safe

            if turnback_mode == "ctrl":
                turns = np.asarray(b["reward_lgturn_pathlenN_ctrl"], dtype=float)
                return turns / rewards_safe

            if turnback_mode == "exp_minus_ctrl":
                turns_exp = np.asarray(b["reward_lgturn_pathlenN_exp"], dtype=float)
                turns_ctrl = np.asarray(b["reward_lgturn_pathlenN_ctrl"], dtype=float)
                return (turns_exp - turns_ctrl) / rewards_safe
        return np.asarray(b[series_key], dtype=float)

    def _series_for_bundle_delta(b):
        """
        Return series for bundle b. If delta_vs_path is set, return (b - base_bundle),
        aligned by video_ids when possible. Prefers paired deltas.
        """
        s = _series_for_bundle(b)
        if base_bundle is None:
            return s

        # series from baseline
        sb = _series_for_bundle(base_bundle)

        # shape check early
        if s.ndim != 3 or sb.ndim != 3:
            raise ValueError("Series arrays must be 3D (n_videos, n_trains, nb).")
        if s.shape[1:] != sb.shape[1:]:
            raise ValueError(
                f"Cannot delta: series shapes differ (comp={s.shape}, base={sb.shape})."
            )

        # Paired alignment by video_ids
        bidx, cidx, n_match = _align_by_video_ids(base_bundle, b)
        if n_match and n_match >= 2:
            return s[cidx, :, :] - sb[bidx, :, :]

        # Fallback: unpaired delta (mean difference) by repeating mean delta per "pseudo video"
        if not delta_allow_unpaired:
            raise ValueError(
                "Cannot compute paired delta: insufficient overlapping video_ids "
                f"(matched={n_match}). Re-export with consistent video sets or use --delta-allow-unpaired."
            )

        # Compute mean(base) and mean(comp) per train/bucket, then treat as 1-sample series
        # so downstream mean+CI yields n=1.
        m_comp = np.nanmean(s, axis=0, keepdims=True)
        m_base = np.nanmean(sb, axis=0, keepdims=True)
        return m_comp - m_base

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

    if base_bundle is not None:
        prefix = delta_label or "Δ vs baseline"
        group_labels = [f"{prefix}: {gl}" for gl in group_labels]

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
    # s0 = _series_for_bundle(bundles[0])
    if base_bundle is not None:
        _check_delta_compat(
            base_bundle, bundles[0], keys=need_keys, metric_label=metric
        )
    s0 = _series_for_bundle_delta(bundles[0])
    n_trains = s0.shape[1]
    if any(_series_for_bundle_delta(b).shape[1] != n_trains for b in bundles):
        raise ValueError(
            f"Bundles disagree on number of trainings ({series_key}.shape[1])."
        )

    # optional training limit
    if num_trainings is not None:
        n_trains = min(n_trains, int(num_trainings))

    # nb
    nb = s0.shape[2]
    if any(_series_for_bundle_delta(b).shape[2] != nb for b in bundles):
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
    elif metric == "lgturn_startdist":
        ylim = [0.0, 6.0]
    elif metric == "reward_lv":
        ylim = [-1.0, 3.0] if turnback_mode == "exp_minus_ctrl" else [0.0, 3.0]
    elif metric == "reward_lgturn_pathlen":
        ylim = [-3.0, 4.0] if turnback_mode == "exp_minus_ctrl" else [0.0, 8.0]
    elif metric == "reward_lgturn_prevalence":
        ylim = [-0.5, 0.5] if turnback_mode == "exp_minus_ctrl" else [0.0, 1.0]
    elif metric == "weaving":
        ylim = [-0.5, 0.5] if turnback_mode == "exp_minus_ctrl" else [0.0, 0.5]

    if base_bundle is not None:
        ylim = [
            -0.05,
            0.05,
        ]  # initial fallback; refined after computing mci_min/mci_max
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
        do_anova = ng >= 3
        means_by_group = [None] * ng  # each: (nb,) float
        vals_by_group = [
            None
        ] * ng  # each: (nb,) list[np.ndarray] (finite per-bucket samples)
        min_n_per_group_anova = 3

        # Each bundle is a "group"
        for gi, b in enumerate(bundles):
            sel_idx, both_labels, gid = selections[gi]
            if sel_idx.size == 0:
                continue

            # For delta mode, series is (bundle - baseline)
            if base_bundle is not None:
                _check_delta_compat(base_bundle, b, keys=need_keys, metric_label=metric)
            series = _series_for_bundle_delta(b)
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
            if do_ttests or do_anova:
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
                elif metric == "lgturn_startdist":
                    ctrl_key = "lgturn_startdist_ctrl"
                elif metric == "reward_lgturn_pathlen":
                    ctrl_key = "reward_lgturn_pathlen_ctrl"
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

        # ---- 3+ group one-way ANOVA + star annotations (plotRewards-style) ----
        if (
            do_anova
            and all(m is not None for m in means_by_group)
            and all(v is not None for v in vals_by_group)
        ):
            # For each bucket, run an omnibus ANOVA across groups.
            # (No post-hoc here; this mirrors your "one symbol per bucket" style.)
            for bj in range(nb):
                groups_here = []
                ok = True
                for gi in range(ng):
                    x = vals_by_group[gi][bj]
                    if x.size < min_n_per_group_anova:
                        ok = False
                        break
                    groups_here.append(x)
                if not ok:
                    continue

                p = _anova_p_per_bucket(
                    groups_here, min_n_per_group=min_n_per_group_anova
                )
                stars = util.p2stars(p, nanR="")
                if not stars:
                    continue

                # Anchor at the max mean among groups at this bucket
                mus = [means_by_group[gi][bj] for gi in range(ng)]
                if not np.any(np.isfinite(mus)):
                    continue
                anchor_y = float(np.nanmax(mus))

                # Avoid y-collisions with existing labels at this bucket
                txts_here = lbls.get(bj, [])
                avoid_ys = [
                    (t._final_y_ if hasattr(t, "_final_y_") else t._y_)
                    for t in txts_here
                    if hasattr(t, "_final_y_") or hasattr(t, "_y_")
                ]
                if np.isfinite(anchor_y):
                    avoid_ys.append(float(anchor_y))

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
                y_label = "Turnback ratio (exp - yok)"
            elif turnback_mode == "ctrl":
                y_label = "Turnback ratio (yok)"
            else:
                y_label = "Turnback ratio"
        elif metric == "agarose":
            if turnback_mode == "exp_minus_ctrl":
                y_label = "Agarose avoidance (exp - yok)"
            elif turnback_mode == "ctrl":
                y_label = "Agarose avoidance (yok)"
            else:
                y_label = "Agarose avoidance ratio"
        elif metric == "wallpct":
            y_label = "% time on wall"
        elif metric == "lgturn_startdist":
            y_label = "Large-turn start dist. to circle center [mm]"
        elif metric == "reward_lv":
            y_label = "Reward timing local variation (LV)"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
        elif metric == "reward_lgturn_pathlen":
            y_label = "Path length from reward to large-turn start [mm]"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
        elif metric == "reward_lgturn_prevalence":
            y_label = "Post-reward large-turn prevalence"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
        elif metric == "weaving":
            if turnback_mode == "exp_minus_ctrl":
                y_label = "Weaving-per-exit ratio\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label = "Weaving-per-exit ratio\n(yok)"
            else:
                y_label = "Weaving-per-exit ratio"

        if base_bundle is not None:
            if delta_ylabel:
                y_label = str(delta_ylabel)
            else:
                y_label = f"Δ {y_label}"
        if ti == 0:
            plt.ylabel(maybe_sentence_case(y_label))
        if base_bundle is not None:
            plt.axhline(0.0, color="k")
        else:
            plt.axhline(color="k")
        plt.xlim(0, xs[-1])

    # --- Delta-mode y-lims: center around 0 and keep it tight unless the data demand otherwise ---
    if base_bundle is not None:
        if (
            mci_min is None
            or mci_max is None
            or not (np.isfinite(mci_min) and np.isfinite(mci_max))
        ):
            # Fall back to a small symmetric range
            ylim = [-0.05, 0.05]
        else:
            # Symmetric about 0, based on observed extrema (including CI bounds)
            span = float(max(abs(mci_min), abs(mci_max)))

            # Add a little breathing room so CI shading and stars don't clip.
            pad = 1.25
            span *= pad

            # Guard against a degenerate case where everything is exactly zero/nan
            span = max(span, 0.01)

            ylim = [-span, span]

    # Dynamic y-lims similar to plotRewards behavior
    if base_bundle is None:
        if mci_max is not None and np.isfinite(mci_max):
            base_pad = 1.1
            if mci_max > ylim[1]:
                ylim[1] = mci_max * base_pad
        if mci_min is not None and np.isfinite(mci_min):
            if mci_min < ylim[0]:
                ylim[0] = mci_min * 1.2

    for ax in fig.get_axes():
        ax.set_ylim(ylim[0], ylim[1])

    # legend (or suptitle if only one entry)
    handles, leg_labels = axs[0].get_legend_handles_labels()
    # Only promote to suptitle when there is exactly one legend entry
    if len(leg_labels) == 1:
        fig.suptitle(leg_labels[0], y=0.995)
    else:
        axs[0].legend(frameon=False)
    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally(wspace=getattr(opts, "wspace", 0.35))

    # save
    base, ext = os.path.splitext(out_fn)
    if ext == "":
        out_fn = base + ".png"
    writeImage(out_fn, format=getattr(opts, "imageFormat", "png"))
    plt.close()
