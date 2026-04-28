import os
import warnings
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import f_oneway

from src.plotting.palettes import get_palette, FLY_COLS
from src.analysis.sli_bundle_utils import (
    align_by_video_ids,
    as_scalar,
    load_sli_bundle,
    normalize_sli_bundle,
)
import src.utils.util as util
from src.utils.common import (
    maybe_sentence_case,
    pch,
    pick_above_or_expand,
    writeImage,
    ttest_ind,
    pick_non_overlapping_y,
)
from src.plotting.plot_customizer import PlotCustomizer


def _pct_label(prefix, frac):
    if frac is None:
        return prefix
    return f"{prefix} {int(round(frac * 100))}%"


def _format_sli_training_mean_window(skip_first_sync_buckets, keep_first_sync_buckets):
    skip_k = max(0, int(skip_first_sync_buckets))
    keep_k = max(0, int(keep_first_sync_buckets))
    if skip_k == 0 and keep_k == 0:
        return "full training"
    if skip_k > 0 and keep_k > 0:
        return f"skip first {skip_k} bucket(s), then keep first {keep_k}"
    if skip_k > 0:
        return f"skip first {skip_k} bucket(s)"
    return f"keep first {keep_k} bucket(s)"


def _legend_handle_for_group(label, color, linestyle):
    handle = Line2D(
        [0, 1, 2, 3],
        [0, 0, 0, 0],
        color=color,
        marker="o",
        markersize=4,
        markerfacecolor=color,
        markeredgecolor=color,
        linewidth=2,
        linestyle=linestyle,
        label=label,
    )
    if linestyle == "--":
        # Make the dashed legend entry more exaggerated than the plotted line so it
        # remains visibly dashed even with a marker present.
        handle.set_linestyle((4, (8, 4)))
        handle.set_markevery([3])
    else:
        handle.set_markevery([2])
    return handle


def _bundle_metric_palette(metric):
    if metric == "commag":
        return get_palette("commag")
    elif metric == "agarose":
        return get_palette("agarose")
    elif metric == "sli":
        return get_palette("rpid")
    elif metric == "turnback":
        return get_palette("turnback")
    elif metric == "between_reward_maxdist":
        return get_palette("between_reward_maxdist")
    elif metric == "between_reward_return_leg_dist":
        return get_palette("between_reward_return_leg_dist")
    else:
        return FLY_COLS


def _ctrl_overlay_key(metric):
    if metric == "commag":
        return "commag_ctrl"
    if metric == "wallpct":
        return "wallpct_ctrl"
    if metric == "between_reward_maxdist":
        return "between_reward_maxdist_ctrl"
    if metric == "between_reward_return_leg_dist":
        return "between_reward_return_leg_dist_ctrl"
    if metric == "turnback":
        return "turnback_ratio_ctrl"
    if metric == "agarose":
        return "agarose_ratio_ctrl"
    if metric == "lgturn_startdist":
        return "lgturn_startdist_ctrl"
    if metric == "reward_lgturn_pathlen":
        return "reward_lgturn_pathlen_ctrl"
    return None


def _check_delta_compat(base, comp, *, keys, metric_label="metric"):
    """
    Basic sanity checks so we don't silently subtract apples from wheelbarrows.
    """
    # bucket length consistency
    bl0 = float(as_scalar(base.get("bucket_len_min", np.nan)))
    bl1 = float(as_scalar(comp.get("bucket_len_min", np.nan)))
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
        d0 = float(as_scalar(base["turnback_inner_delta_mm"]))
        d1 = float(as_scalar(comp["turnback_inner_delta_mm"]))
        if abs(d0 - d1) > 1e-9:
            raise ValueError(
                f"Cannot delta turnback: turnback_inner_delta_mm differs (base={d0}, comp={d1})."
            )


def _fmt_bucket_len(bl):
    # mimic “blf” vibe (integer when possible)
    if np.isfinite(bl) and abs(bl - round(bl)) < 1e-9:
        return str(int(round(bl)))
    return f"{bl:g}"


def _select_sli_extremes(
    sli,
    *,
    top_fraction=None,
    bottom_fraction=None,
    which=None,
):
    """
    sli: (n_videos,) float
    which: None | "top" | "bottom" | "both"

    returns:
      - indices (np.ndarray of ints) into videos
      - group_labels (list[str]) if which=="both" else None
      - group_ids (np.ndarray) same length as indices, for legend grouping (0/1)
    """
    if which is None:
        n = len(sli)
        idx = np.arange(n, dtype=int)
        return idx, None, np.zeros(n, dtype=int)

    sli = np.asarray(sli, dtype=float)
    finite = np.isfinite(sli)
    if not finite.any():
        if which == "both":
            return (
                np.array([], dtype=int),
                [
                    _pct_label("Bottom", bottom_fraction),
                    _pct_label("Top", top_fraction),
                ],
                np.array([], dtype=int),
            )
        return np.array([], dtype=int), None, np.array([], dtype=int)

    finite_idx = np.flatnonzero(finite)
    order = finite_idx[np.argsort(sli[finite_idx])]
    n_finite = finite_idx.size

    if (
        top_fraction is not None
        and bottom_fraction is not None
        and float(top_fraction) + float(bottom_fraction) > 1.0 + 1e-12
    ):
        raise ValueError(
            "top_fraction + bottom_fraction must be <= 1 for disjoint selections "
            f"(got top_fraction={top_fraction!r}, "
            f"bottom_fraction={bottom_fraction!r})"
        )

    def _k(frac):
        if frac is None:
            return 0
        return min(max(1, int(n_finite * frac)), n_finite)

    k_bottom = _k(bottom_fraction)
    k_top = _k(top_fraction)
    if top_fraction is not None and bottom_fraction is not None:
        k_bottom = min(k_bottom, max(0, n_finite - k_top))

    bottom = order[:k_bottom].tolist() if k_bottom > 0 else []
    top_pool = order[k_bottom:] if k_bottom > 0 else order
    top = top_pool[-k_top:].tolist() if k_top > 0 else []

    if which == "bottom":
        idx = np.array(bottom, dtype=int)
        return idx, None, np.zeros(len(idx), dtype=int)

    if which == "top":
        idx = np.array(top, dtype=int)
        return idx, None, np.zeros(len(idx), dtype=int)

    if which == "both":
        idx = np.array(bottom + top, dtype=int)
        gid = np.array([0] * len(bottom) + [1] * len(top), dtype=int)
        labels = [
            _pct_label("Bottom", bottom_fraction),
            _pct_label("Top", top_fraction),
        ]
        return idx, labels, gid

    raise ValueError(f"Unknown which={which!r}")


def _mean_ci_over_videos(vals_2d):
    """
    vals_2d: (n_videos, nb) with NaNs allowed
    returns mci: (4, nb) [mean, lo, hi, n]
    """
    nb = vals_2d.shape[1]
    mci = np.array([util.meanConfInt(vals_2d[:, b]) for b in range(nb)]).T
    return mci


def _update_y_bounds(plot_mci, y_bounds):
    ymin, ymax = y_bounds
    lo = np.asarray(plot_mci[1, :], dtype=float)
    hi = np.asarray(plot_mci[2, :], dtype=float)

    if np.isfinite(lo).any():
        lo_min = float(np.nanmin(lo))
        ymin = lo_min if ymin is None else min(ymin, lo_min)
    if np.isfinite(hi).any():
        hi_max = float(np.nanmax(hi))
        ymax = hi_max if ymax is None else max(ymax, hi_max)

    return ymin, ymax


def _expanded_bbox(bbox, pad_px):
    return bbox.expanded(
        (float(bbox.width) + 2.0 * pad_px) / max(float(bbox.width), 1.0),
        (float(bbox.height) + 2.0 * pad_px) / max(float(bbox.height), 1.0),
    )


def _artist_overlap_score(ax, legend, *, pad_px=5.0):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = _expanded_bbox(
        legend.get_window_extent(renderer=renderer), float(pad_px)
    )
    score = 0.0

    for artist in [*ax.lines, *ax.collections, *ax.texts]:
        if artist is legend:
            continue
        if hasattr(artist, "get_visible") and not artist.get_visible():
            continue
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bbox is None:
            continue
        bbox = _expanded_bbox(bbox, float(pad_px) * 0.35)
        if not legend_bbox.overlaps(bbox):
            continue
        overlap_w = max(0.0, min(legend_bbox.x1, bbox.x1) - max(legend_bbox.x0, bbox.x0))
        overlap_h = max(0.0, min(legend_bbox.y1, bbox.y1) - max(legend_bbox.y0, bbox.y0))
        if overlap_w <= 0.0 or overlap_h <= 0.0:
            continue
        overlap_score = overlap_w * overlap_h

        # Penalize text overlaps more heavily than line/band overlaps because they
        # tend to become unreadable first.
        if artist in ax.texts:
            overlap_score *= 1.15
        score += overlap_score

    # Mild preference ordering for ties so we keep historical placements when safe.
    loc = getattr(legend, "_loc", None)
    preference = {
        1: 0.0,   # upper right
        2: 15.0,  # upper left
        4: 30.0,  # lower right
        3: 45.0,  # lower left
    }
    score += preference.get(loc, 60.0)
    return float(score)


def _place_interior_legend_min_overlap(
    ax,
    *,
    handles,
    labels,
    legend_kwargs=None,
    candidate_locs=None,
):
    if legend_kwargs is None:
        legend_kwargs = {}
    if candidate_locs is None:
        candidate_locs = ["upper right", "upper left", "lower right", "lower left"]

    best_loc = None
    best_score = None
    for loc in candidate_locs:
        leg = ax.legend(handles=handles, labels=labels, loc=loc, **legend_kwargs)
        score = _artist_overlap_score(ax, leg)
        leg.remove()
        if best_score is None or score < best_score:
            best_loc = loc
            best_score = score

    if best_loc is None:
        return None
    return ax.legend(
        handles=handles,
        labels=labels,
        loc=best_loc,
        **legend_kwargs,
    )


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


def plot_com_sli_bundle_data(
    bundles,
    out_fn,
    *,
    labels=None,
    num_trainings=None,
    include_ctrl=False,
    sli_extremes=None,  # None | "top" | "bottom" | "both"
    sli_fraction=None,  # legacy shared fraction
    sli_top_fraction=None,  # Optional[float]
    sli_bottom_fraction=None,
    opts=None,
    metric="commag",
    turnback_mode="exp",  # exp | ctrl | exp_minus_ctrl
    delta_vs_bundle=None,  # baseline bundle dict; if set, plot (bundle - baseline)
    delta_label=None,  # label prefix in delta mode
    xlabel=None,
    ylabel=None,
    delta_ylabel=None,
    ymax=None,
    delta_allow_unpaired=False,
    include_pre=False,
    show_legend=False,
    show_description_labels=False,
):
    """
    Plot COM magnitude or SLI vs sync bucket from one or more normalized bundles.

    - Each bundle is a “group” (regular / antennae-removed / PFN-silenced, etc).
    - Lines are mean over videos; shaded region is CI.
    - Optionally filter within each bundle by SLI percentile.
    - Optional top-of-figure description labels are hidden by default.
    """
    if opts is None:
        # minimal opts object for PlotCustomizer + writeImage usage
        opts = SimpleNamespace(
            wspace=0.35,
            imageFormat="png",
            fontSize=None,
            fontFamily=None,
        )

    def _parse_num_trainings_limit(num_trainings_value):
        if num_trainings_value is None:
            return None
        raw = str(num_trainings_value).strip()
        if not raw:
            return None
        if "," in raw or "-" in raw:
            raise ValueError(
                "--num-trainings only supports a single integer for this plot type; "
                "selector syntax like '1,3' or '2-4' is currently supported for heatmaps."
            )
        limit = int(raw)
        if limit < 1:
            raise ValueError("--num-trainings must be >= 1")
        return limit

    requested_include_ctrl = bool(include_ctrl)
    include_ctrl = False

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

    bundles = [normalize_sli_bundle(b) for b in bundles]
    ng = len(bundles)
    if ng == 0:
        raise ValueError("No bundles provided")

    base_bundle = normalize_sli_bundle(delta_vs_bundle) if delta_vs_bundle else None

    if metric == "commag":
        if turnback_mode == "exp":
            series_key = "commag_exp"
            need_keys = ["commag_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "commag_ctrl"
            need_keys = ["commag_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "commag_exp"
            need_keys = ["commag_exp", "commag_ctrl"]
        else:
            raise ValueError(f"Unknown mode={turnback_mode!r} for metric=commag")
    elif metric == "sli":
        if any("sli_ts" not in b for b in bundles):
            raise ValueError("One or more bundles are missing sli_ts; re-export them.")
        series_key = "sli_ts"
        need_keys = ["sli_ts"]
    elif metric == "between_reward_maxdist":
        if turnback_mode == "exp":
            series_key = "between_reward_maxdist_exp"
            need_keys = ["between_reward_maxdist_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "between_reward_maxdist_ctrl"
            need_keys = ["between_reward_maxdist_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "between_reward_maxdist_exp"
            need_keys = [
                "between_reward_maxdist_exp",
                "between_reward_maxdist_ctrl",
            ]
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=between_reward_maxdist"
            )
    elif metric == "between_reward_return_leg_dist":
        if turnback_mode == "exp":
            series_key = "between_reward_return_leg_dist_exp"
            need_keys = ["between_reward_return_leg_dist_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "between_reward_return_leg_dist_ctrl"
            need_keys = ["between_reward_return_leg_dist_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "between_reward_return_leg_dist_exp"
            need_keys = [
                "between_reward_return_leg_dist_exp",
                "between_reward_return_leg_dist_ctrl",
            ]
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=between_reward_return_leg_dist"
            )
    elif metric == "turnback":
        if turnback_mode == "exp":
            series_key = "turnback_ratio_exp"
            need_keys = ["turnback_ratio_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "turnback_ratio_ctrl"
            need_keys = ["turnback_ratio_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "turnback_ratio_exp"
            need_keys = ["turnback_ratio_exp", "turnback_ratio_ctrl"]
        else:
            raise ValueError(f"Unknown turnback_mode={turnback_mode!r}")
    elif metric == "weaving":
        if turnback_mode == "exp":
            series_key = "weaving_ratio_exp"
            need_keys = ["weaving_ratio_exp"]
        elif turnback_mode == "ctrl":
            series_key = "weaving_ratio_ctrl"
            need_keys = ["weaving_ratio_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "weaving_ratio_exp"
            need_keys = ["weaving_ratio_exp", "weaving_ratio_ctrl"]
        else:
            raise ValueError(
                f"Unknown turnback_mode={turnback_mode!r} for metric=weaving"
            )
    elif metric == "agarose":
        if turnback_mode == "exp":
            series_key = "agarose_ratio_exp"
            need_keys = ["agarose_ratio_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "agarose_ratio_ctrl"
            need_keys = ["agarose_ratio_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "agarose_ratio_exp"
            need_keys = ["agarose_ratio_exp", "agarose_ratio_ctrl"]
        else:
            raise ValueError(f"Unknown mode={turnback_mode!r} for metric=agarose")
        if include_pre:
            if turnback_mode == "exp":
                need_keys = need_keys + ["agarose_pre_ratio_exp"]
            elif turnback_mode == "ctrl":
                need_keys = need_keys + ["agarose_pre_ratio_ctrl"]
            else:
                need_keys = need_keys + [
                    "agarose_pre_ratio_exp",
                    "agarose_pre_ratio_ctrl",
                ]
    elif metric == "wallpct":
        series_key = "wallpct_exp"
        need_keys = ["wallpct_exp"]
        include_ctrl = requested_include_ctrl
    elif metric == "lgturn_startdist":
        series_key = "lgturn_startdist_exp"
        need_keys = ["lgturn_startdist_exp"]
        include_ctrl = requested_include_ctrl
    elif metric == "reward_lgturn_pathlen":
        if turnback_mode == "exp":
            series_key = "reward_lgturn_pathlen_exp"
            need_keys = ["reward_lgturn_pathlen_exp"]
            include_ctrl = requested_include_ctrl
        elif turnback_mode == "ctrl":
            series_key = "reward_lgturn_pathlen_ctrl"
            need_keys = ["reward_lgturn_pathlen_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lgturn_pathlen_exp"
            need_keys = ["reward_lgturn_pathlen_exp", "reward_lgturn_pathlen_ctrl"]
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=reward_lgturn_pathlen"
            )
    elif metric == "reward_lv":
        if turnback_mode == "exp":
            series_key = "reward_lv_exp"
            need_keys = ["reward_lv_exp"]
        elif turnback_mode == "ctrl":
            series_key = "reward_lv_ctrl"
            need_keys = ["reward_lv_ctrl"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lv_exp"
            need_keys = ["reward_lv_exp", "reward_lv_ctrl"]
        else:
            raise ValueError(f"Unknown mode={turnback_mode!r} for metric=reward_lv")
    elif metric == "reward_lgturn_prevalence":
        # Prevalence = (# rewards with detect post-reward turn) / (# rewards)
        # The underlying arrays are per-video, per-training, per-bucket.
        if turnback_mode == "exp":
            series_key = "reward_lgturn_pathlenN_exp"
            need_keys = ["reward_lgturn_pathlenN_exp", "reward_lgturn_rewards"]
        elif turnback_mode == "ctrl":
            series_key = "reward_lgturn_pathlenN_ctrl"
            need_keys = ["reward_lgturn_pathlenN_ctrl", "reward_lgturn_rewards"]
        elif turnback_mode == "exp_minus_ctrl":
            series_key = "reward_lgturn_pathlenN_exp"
            need_keys = [
                "reward_lgturn_pathlenN_exp",
                "reward_lgturn_pathlenN_ctrl",
                "reward_lgturn_rewards",
            ]
        else:
            raise ValueError(
                f"Unknown mode={turnback_mode!r} for metric=reward_lgturn_prevalence"
            )
    else:
        raise ValueError(
            "Invalid metric specified; supported: 'commag', 'sli', "
            "'between_reward_maxdist', 'between_reward_return_leg_dist', "
            "'turnback', 'agarose', 'wallpct', "
            "'lgturn_startdist', 'reward_lgturn_pathlen', 'reward_lv', "
            "'reward_lgturn_prevalence', 'weaving'."
        )

    if include_pre and metric != "agarose":
        raise ValueError("--include-pre is currently supported only for metric='agarose'.")

    ctrl_key = _ctrl_overlay_key(metric) if include_ctrl else None
    if include_ctrl and ctrl_key is None:
        warnings.warn(
            f"--include-ctrl is not supported for metric={metric!r}; ignoring it."
        )
        include_ctrl = False
    elif include_ctrl and turnback_mode != "exp":
        warnings.warn(
            f"--include-ctrl is redundant for metric={metric!r} with mode={turnback_mode!r}; ignoring it."
        )
        include_ctrl = False
    elif include_ctrl:
        need_keys = need_keys + [ctrl_key]
        if metric == "agarose" and include_pre:
            need_keys = need_keys + ["agarose_pre_ratio_ctrl"]

    def _series_for_bundle(b):
        """
        Return array shaped (n_videos, n_trains, nb) for the requested plot.
        """
        if metric == "commag" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["commag_exp"], dtype=float)
            ctrl_arr = np.asarray(b["commag_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "turnback" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["turnback_ratio_exp"], dtype=float)
            ctrl_arr = np.asarray(b["turnback_ratio_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if metric == "between_reward_maxdist" and turnback_mode == "exp_minus_ctrl":
            exp_arr = np.asarray(b["between_reward_maxdist_exp"], dtype=float)
            ctrl_arr = np.asarray(b["between_reward_maxdist_ctrl"], dtype=float)
            return exp_arr - ctrl_arr
        if (
            metric == "between_reward_return_leg_dist"
            and turnback_mode == "exp_minus_ctrl"
        ):
            exp_arr = np.asarray(b["between_reward_return_leg_dist_exp"], dtype=float)
            ctrl_arr = np.asarray(b["between_reward_return_leg_dist_ctrl"], dtype=float)
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

    def _pre_series_for_bundle(b):
        if metric != "agarose" or not include_pre:
            return None
        if turnback_mode == "exp":
            return np.asarray(b["agarose_pre_ratio_exp"], dtype=float)
        if turnback_mode == "ctrl":
            return np.asarray(b["agarose_pre_ratio_ctrl"], dtype=float)
        exp_arr = np.asarray(b["agarose_pre_ratio_exp"], dtype=float)
        ctrl_arr = np.asarray(b["agarose_pre_ratio_ctrl"], dtype=float)
        return exp_arr - ctrl_arr

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
        bidx, cidx, n_match = align_by_video_ids(base_bundle, b)
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

    def _pre_series_for_bundle_delta(b):
        s = _pre_series_for_bundle(b)
        if s is None or base_bundle is None:
            return s

        sb = _pre_series_for_bundle(base_bundle)
        if sb is None:
            raise ValueError(
                "Baseline bundle is missing pre-training agarose keys required by --include-pre."
            )

        if s.ndim != 1 or sb.ndim != 1:
            raise ValueError("Pre-training agarose arrays must be 1D (n_videos,).")

        bidx, cidx, n_match = align_by_video_ids(base_bundle, b)
        if n_match and n_match >= 2:
            return s[cidx] - sb[bidx]

        if not delta_allow_unpaired:
            raise ValueError(
                "Cannot compute paired delta for pre-training agarose: insufficient "
                f"overlapping video_ids (matched={n_match})."
            )

        return np.array([np.nanmean(s) - np.nanmean(sb)], dtype=float)

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
    limit = _parse_num_trainings_limit(num_trainings)
    if limit is not None:
        n_trains = min(n_trains, limit)

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
    pre_x = 0.0

    # Styling: mimic plotRewards commag behavior (group via linestyle)
    # Use matplotlib default C0/C1 to stay close to existing look.
    exp_color, ctrl_color = _bundle_metric_palette(metric)
    linestyles = ["-", "--", ":", "-."]  # extend if needed

    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))

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
    elif metric == "between_reward_maxdist":
        ylim = [-2.0, 2.0] if turnback_mode == "exp_minus_ctrl" else [0.0, 20.0]
    elif metric == "between_reward_return_leg_dist":
        ylim = [-5.0, 5.0] if turnback_mode == "exp_minus_ctrl" else [0.0, 20.0]
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
            b["sli"],
            top_fraction=sli_top_fraction,
            bottom_fraction=sli_bottom_fraction,
            which=sli_extremes,
        )
        selections.append((idx, both_labels, gid))

    # annotate SLI selection if explicitly requested
    if show_description_labels and sli_extremes is not None:
        # Assume consistent SLI settings; if not, still show the first and warn.
        stis = [b["sli_training_idx"] for b in bundles]
        means = [b["sli_use_training_mean"] for b in bundles]
        skips = [b.get("sli_select_skip_first_sync_buckets", 0) for b in bundles]
        keeps = [b.get("sli_select_keep_first_sync_buckets", 0) for b in bundles]
        if (
            len(set(stis)) > 1
            or len(set(means)) > 1
            or len(set(skips)) > 1
            or len(set(keeps)) > 1
        ):
            warnings.warn(
                "Bundles disagree on SLI selection metadata; annotation may be misleading."
            )
        sli_mode = "mean over buckets" if bool(means[0]) else "single bucket"
        sli_mode_window = ""
        if bool(means[0]):
            sli_mode_window = "; " + _format_sli_training_mean_window(
                skips[0], keeps[0]
            )
        if sli_extremes == "top":
            sel_txt = _pct_label("Top", sli_top_fraction)
        elif sli_extremes == "bottom":
            sel_txt = _pct_label("Bottom", sli_bottom_fraction)
        elif sli_extremes == "both":
            sel_txt = (
                f"{_pct_label('Bottom', sli_bottom_fraction)} vs "
                f"{_pct_label('Top', sli_top_fraction)}"
            )
        else:
            sel_txt = None

        if sel_txt is not None:
            fig.text(
                0.1,
                0.98,
                f"SLI filter: {sel_txt} within group; T{stis[0]+1}; {sli_mode}{sli_mode_window}",
                ha="left",
                va="top",
                fontsize=customizer.in_plot_font_size,
                color="0",
            )

    panel_annotation_contexts = []

    # plotting
    for ti in range(n_trains):
        ax = axs[ti]
        plt.sca(ax)
        panel_has_pre = bool(include_pre and metric == "agarose" and ti == 0)
        panel_xs = np.concatenate(([pre_x], xs)) if panel_has_pre else xs

        pending_n_labels = []

        # choose a training title from first bundle (best effort)
        try:
            tnames0 = bundles[0]["training_names"]
            title = str(tnames0[ti]) if len(tnames0) > ti else f"training {ti+1}"
        except Exception:
            title = f"training {ti+1}"

        min_n_per_group_anova = 3

        plot_groups = []
        for gi, b in enumerate(bundles):
            sel_idx, both_labels, gid = selections[gi]

            if sel_idx.size == 0:
                continue

            if sli_extremes == "both":
                for sub_gid, sub_label in enumerate(both_labels):
                    sub_idx = sel_idx[gid == sub_gid]
                    if sub_idx.size == 0:
                        continue
                    plot_groups.append(
                        {
                            "bundle": b,
                            "sel_idx": sub_idx,
                            "label": sub_label,
                            "style_idx": sub_gid,
                        }
                    )
            else:
                plot_groups.append(
                    {
                        "bundle": b,
                        "sel_idx": sel_idx,
                        "label": group_labels[gi],
                        "style_idx": gi,
                    }
                )

        n_plot_groups = len(plot_groups)

        # Stats should follow the number of plotted groups, not the number of bundles.
        do_ttests = n_plot_groups == 2
        do_anova = n_plot_groups >= 3
        means_by_group = [None] * n_plot_groups  # each: (nb,) float
        vals_by_group = [
            None
        ] * n_plot_groups  # each: list[np.ndarray] (finite per-bucket samples)

        for gi, pg in enumerate(plot_groups):
            b = pg["bundle"]
            sel_idx = pg["sel_idx"]
            label = pg["label"]
            style_idx = pg["style_idx"]

            if sel_idx.size == 0:
                continue

            # For delta mode, series is (bundle - baseline)
            if base_bundle is not None:
                _check_delta_compat(base_bundle, b, keys=need_keys, metric_label=metric)
            series = _series_for_bundle_delta(b)
            exp = series[sel_idx, ti, :]
            mci = _mean_ci_over_videos(exp)
            plot_mci = mci
            exp_for_test = np.asarray(exp, dtype=float)

            if panel_has_pre:
                pre_series = _pre_series_for_bundle_delta(b)
                pre_vals = np.asarray(pre_series[sel_idx], dtype=float)
                pre_mci = np.asarray(util.meanConfInt(pre_vals), dtype=float).reshape(4, 1)
                plot_mci = np.concatenate((pre_mci, plot_mci), axis=1)
                exp_for_test = np.concatenate((pre_vals[:, None], exp_for_test), axis=1)

            if metric == "wallpct":
                plot_mci = plot_mci.copy()
                plot_mci[0, :] *= 100.0
                plot_mci[1, :] *= 100.0
                plot_mci[2, :] *= 100.0

            mci_min, mci_max = _update_y_bounds(plot_mci, (mci_min, mci_max))

            # For t-tests we compare what we're plotting (wallpct is scaled by 100)
            if do_ttests or do_anova:
                if metric == "wallpct":
                    exp_for_test = exp_for_test * 100.0
                # Store finite samples per bucket
                vals_by_group[gi] = [
                    exp_for_test[np.isfinite(exp_for_test[:, bj]), bj]
                    for bj in range(exp_for_test.shape[1])
                ]
                means_by_group[gi] = np.asarray(plot_mci[0, :], dtype=float)
            ys = plot_mci[0, :]
            fin = np.isfinite(ys)
            ls = linestyles[style_idx % len(linestyles)]
            (line,) = plt.plot(
                panel_xs[fin],
                ys[fin],
                color=exp_color,
                marker="o",
                ms=3,
                mec=exp_color,
                linewidth=2,
                linestyle=ls,
            )
            if ti == 0:
                line.set_label(label)

            # CI shading
            if np.isfinite(plot_mci[1, :]).any() and np.isfinite(plot_mci[2, :]).any():
                plt.fill_between(
                    panel_xs[fin],
                    plot_mci[1, :][fin],
                    plot_mci[2, :][fin],
                    color=exp_color,
                    alpha=0.15,
                )

            # Collect n-per-bucket labels and place them after final y-lims are known.
            for bj, n in enumerate(plot_mci[3, :]):
                if n > 0 and np.isfinite(ys[bj]):
                    pending_n_labels.append(
                        {
                            "bucket_idx": int(bj),
                            "x": float(panel_xs[bj]),
                            "anchor_y": float(ys[bj]),
                            "n": int(n),
                        }
                    )

            # ctrl overlay (optional)
            if include_ctrl:
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
                plot_mci_c = mci_c
                if panel_has_pre:
                    pre_ctrl_vals = np.asarray(
                        b["agarose_pre_ratio_ctrl"], dtype=float
                    )[sel_idx]
                    pre_mci_c = np.asarray(
                        util.meanConfInt(pre_ctrl_vals), dtype=float
                    ).reshape(4, 1)
                    plot_mci_c = np.concatenate((pre_mci_c, plot_mci_c), axis=1)
                if metric == "wallpct":
                    plot_mci_c = plot_mci_c.copy()
                    plot_mci_c[0, :] *= 100.0
                    plot_mci_c[1, :] *= 100.0
                    plot_mci_c[2, :] *= 100.0
                mci_min, mci_max = _update_y_bounds(plot_mci_c, (mci_min, mci_max))
                ys_c = plot_mci_c[0, :]
                fin_c = np.isfinite(ys_c)
                if fin_c.any():
                    plt.plot(
                        panel_xs[fin_c],
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
                        np.isfinite(plot_mci_c[1, :]).any()
                        and np.isfinite(plot_mci_c[2, :]).any()
                    ):
                        plt.fill_between(
                            panel_xs[fin_c],
                            plot_mci_c[1, :][fin_c],
                            plot_mci_c[2, :][fin_c],
                            color=ctrl_color,
                            alpha=0.12,
                        )

        plt.title(maybe_sentence_case(title))
        default_xlabel = "time elapsed [min]"
        plt.xlabel(maybe_sentence_case(str(xlabel or default_xlabel)))

        if metric == "commag":
            y_label = "COM dist. to circle center [mm]"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
        elif metric == "sli":
            y_label = "SLI"
        elif metric == "turnback":
            if turnback_mode == "exp_minus_ctrl":
                y_label = "Turnback ratio (exp - yok)"
            elif turnback_mode == "ctrl":
                y_label = "Turnback ratio (yok)"
            else:
                y_label = "Turnback ratio"
        elif metric == "between_reward_maxdist":
            y_label = "Mean between-reward max dist. [mm]"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
        elif metric == "between_reward_return_leg_dist":
            y_label = "Mean between-reward return-leg dist. [mm]"
            if turnback_mode == "exp_minus_ctrl":
                y_label += "\n(exp - yok)"
            elif turnback_mode == "ctrl":
                y_label += "\n(yok)"
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
        if ylabel:
            y_label = str(ylabel)
        if ti == 0:
            plt.ylabel(maybe_sentence_case(y_label))
        if base_bundle is not None:
            plt.axhline(0.0, color="k")
        else:
            plt.axhline(color="k")
        if panel_has_pre:
            ax.axvline(bl / 2.0, color="0.5", linestyle=":", linewidth=1)
            ax.set_xticks(panel_xs)
            ax.set_xticklabels(["pre"] + [f"{x:g}" for x in xs])
            plt.xlim(-0.5 * bl, xs[-1])
        else:
            plt.xlim(0, xs[-1])

        panel_annotation_contexts.append(
            {
                "ax": ax,
                "panel_xs": np.asarray(panel_xs, dtype=float),
                "pending_n_labels": pending_n_labels,
                "do_anova": bool(do_anova),
                "do_ttests": bool(do_ttests),
                "means_by_group": means_by_group,
                "vals_by_group": vals_by_group,
                "n_plot_groups": int(n_plot_groups),
                "min_n_per_group_anova": int(min_n_per_group_anova),
            }
        )

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

    if ymax is not None:
        ymax = float(ymax)
        if not np.isfinite(ymax):
            raise ValueError("--ymax must be finite when provided.")
        if ymax <= ylim[0]:
            raise ValueError(
                f"--ymax ({ymax:g}) must be greater than the lower y-limit ({ylim[0]:g})."
            )
        ylim[1] = ymax

    for ax in fig.get_axes():
        ax.set_ylim(ylim[0], ylim[1])

    span = ylim[1] - ylim[0]
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    for ctx in panel_annotation_contexts:
        ax = ctx["ax"]
        plt.sca(ax)
        lbls = defaultdict(list)

        for info in ctx["pending_n_labels"]:
            base_y = float(info["anchor_y"]) + 0.04 * span
            y_n, va_align = pick_above_or_expand(
                base_y,
                [float(info["anchor_y"])],
                ylim,
                span_override=span,
            )
            if y_n is None:
                continue
            txt = util.pltText(
                float(info["x"]),
                y_n,
                f"{int(info['n'])}",
                ha="center",
                va=va_align,
                size=customizer.in_plot_font_size,
                color=".2",
            )
            txt._y_ = float(info["anchor_y"])
            txt._final_y_ = float(y_n)
            lbls[int(info["bucket_idx"])].append(txt)

        if (
            ctx["do_anova"]
            and all(m is not None for m in ctx["means_by_group"])
            and all(v is not None for v in ctx["vals_by_group"])
        ):
            for bj in range(len(ctx["vals_by_group"][0])):
                groups_here = []
                ok = True
                for gi in range(ctx["n_plot_groups"]):
                    x = ctx["vals_by_group"][gi][bj]
                    if x.size < ctx["min_n_per_group_anova"]:
                        ok = False
                        break
                    groups_here.append(x)
                if not ok:
                    continue

                p = _anova_p_per_bucket(
                    groups_here,
                    min_n_per_group=ctx["min_n_per_group_anova"],
                )
                stars = util.p2stars(p, nanR="")
                if not stars:
                    continue

                mus = [ctx["means_by_group"][gi][bj] for gi in range(ctx["n_plot_groups"])]
                if not np.any(np.isfinite(mus)):
                    continue
                anchor_y = float(np.nanmax(mus))

                txts_here = lbls.get(bj, [])
                avoid_ys = [
                    (t._final_y_ if hasattr(t, "_final_y_") else t._y_)
                    for t in txts_here
                    if hasattr(t, "_final_y_") or hasattr(t, "_y_")
                ]
                if np.isfinite(anchor_y):
                    avoid_ys.append(float(anchor_y))

                stacked_top = max(avoid_ys) if avoid_ys else float(anchor_y)
                base_y_for_star = max(
                    float(anchor_y) + 0.10 * span,
                    float(stacked_top) + 0.06 * span,
                )

                ys_star, va_align = pick_above_or_expand(
                    base_y_for_star,
                    avoid_ys,
                    ylim,
                    span_override=span,
                )
                if ys_star is None:
                    continue

                txt = util.pltText(
                    float(ctx["panel_xs"][bj]),
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

        if (
            ctx["do_ttests"]
            and all(m is not None for m in ctx["means_by_group"])
            and all(v is not None for v in ctx["vals_by_group"])
        ):
            m0 = ctx["means_by_group"][0]
            m1 = ctx["means_by_group"][1]
            for bj in range(len(ctx["vals_by_group"][0])):
                x0 = ctx["vals_by_group"][0][bj]
                x1 = ctx["vals_by_group"][1][bj]
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
                if np.isfinite(anchor_y):
                    avoid_ys.append(float(anchor_y))

                stacked_top = max(avoid_ys) if avoid_ys else float(anchor_y)
                base_y_for_star = max(
                    float(anchor_y) + 0.10 * span,
                    float(stacked_top) + 0.06 * span,
                )

                ys_star, va_align = pick_above_or_expand(
                    base_y_for_star,
                    avoid_ys,
                    ylim,
                    span_override=span,
                )
                if ys_star is None:
                    continue

                txt = util.pltText(
                    float(ctx["panel_xs"][bj]),
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

    # legend (or suptitle if only one entry)
    handles, leg_labels = axs[0].get_legend_handles_labels()
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, leg_labels):
        legend_handles.append(
            _legend_handle_for_group(
                label=label,
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
            )
        )
        legend_labels.append(label)

    if show_legend and include_ctrl:
        ctrl_condition_handles = [
            _legend_handle_for_group(
                label="Experimental",
                color=exp_color,
                linestyle="-",
            ),
            _legend_handle_for_group(
                label="Yoked",
                color=ctrl_color,
                linestyle="-",
            ),
        ]
        ctrl_condition_labels = ["Experimental", "Yoked"]
        if len(legend_labels) == 1:
            legend_handles = ctrl_condition_handles
            legend_labels = ctrl_condition_labels
        else:
            legend_handles.extend(ctrl_condition_handles)
            legend_labels.extend(ctrl_condition_labels)

    # Annotation placement can expand the shared y-limits in-place. Re-apply the
    # final limits after all labels and stars are placed so they stay inside axes.
    for ax in fig.get_axes():
        ax.set_ylim(ylim[0], ylim[1])

    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally(wspace=getattr(opts, "wspace", 0.35))

    # Only promote to suptitle when there is exactly one legend entry and no explicit legend was requested.
    if not show_legend and len(legend_labels) == 1:
        fig.suptitle(leg_labels[0], y=0.995)
    elif legend_labels:
        _place_interior_legend_min_overlap(
            axs[0],
            handles=legend_handles,
            labels=legend_labels,
            legend_kwargs=dict(
                frameon=True,
                facecolor="white",
                framealpha=0.92,
                edgecolor="0.8",
                handlelength=3.2,
            ),
        )

    # save
    base, ext = os.path.splitext(out_fn)
    if ext == "":
        out_fn = base + ".png"
    writeImage(out_fn, format=getattr(opts, "imageFormat", "png"))
    plt.close()


def plot_com_sli_bundles(
    bundle_paths,
    out_fn,
    *,
    labels=None,
    num_trainings=None,
    include_ctrl=False,
    sli_extremes=None,
    sli_fraction=None,
    sli_top_fraction=None,
    sli_bottom_fraction=None,
    opts=None,
    metric="commag",
    turnback_mode="exp",
    delta_vs_path=None,
    delta_label=None,
    xlabel=None,
    ylabel=None,
    delta_ylabel=None,
    ymax=None,
    delta_allow_unpaired=False,
    include_pre=False,
    show_legend=False,
    show_description_labels=False,
):
    bundles = [load_sli_bundle(p) for p in bundle_paths]
    delta_vs_bundle = load_sli_bundle(delta_vs_path) if delta_vs_path else None
    return plot_com_sli_bundle_data(
        bundles,
        out_fn,
        labels=labels,
        num_trainings=num_trainings,
        include_ctrl=include_ctrl,
        sli_extremes=sli_extremes,
        sli_fraction=sli_fraction,
        sli_top_fraction=sli_top_fraction,
        sli_bottom_fraction=sli_bottom_fraction,
        opts=opts,
        metric=metric,
        turnback_mode=turnback_mode,
        delta_vs_bundle=delta_vs_bundle,
        delta_label=delta_label,
        xlabel=xlabel,
        ylabel=ylabel,
        delta_ylabel=delta_ylabel,
        ymax=ymax,
        delta_allow_unpaired=delta_allow_unpaired,
        include_pre=include_pre,
        show_legend=show_legend,
        show_description_labels=show_description_labels,
    )
