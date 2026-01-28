# src/plotting/cross_fly_correlations.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage
from src.utils.debug_fly_groups import log_fly_group


@dataclass
class CorrelationPlotConfig:
    out_dir: Path
    dot_color: str = "#005bbb"
    alpha: float = 0.85
    figsize: tuple = (5.5, 4.5)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SLIContext:
    """
    Describes what sli_values represent.
    - training_idx: 0-based index of the training whose SLI is being used
    - average_over_buckets: True => mean over sync buckets in that training
                            False => last sync bucket in that training
    """

    training_idx: int
    average_over_buckets: bool = False

    def label_long(self) -> str:
        trn = self.training_idx + 1
        if self.average_over_buckets:
            return f"SLI (mean over sync buckets, training {trn})"
        return f"SLI (last sync bucket, training {trn})"

    def label_short(self) -> str:
        trn = self.training_idx + 1
        if self.average_over_buckets:
            return f"SLI (mean, T{trn})"
        return f"SLI (last SB, T{trn})"


def _compute_group_corr(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> tuple[float, float] | None:
    """
    Compute Pearson correlation for a given index set, handling NaNs and
    small sample sizes. Returns (r, p) or None if not enough valid data.
    """
    if idx is None or idx.size == 0:
        return None

    idx = np.asarray(idx, dtype=int)
    x_g = np.asarray(x, float)[idx]
    y_g = np.asarray(y, float)[idx]

    mask = np.isfinite(x_g) & np.isfinite(y_g)
    if np.sum(mask) < 3:
        return None

    return pearsonr(x_g[mask], y_g[mask])


def _scatter_with_corr(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cfg: CorrelationPlotConfig,
    filename: str,
    customizer: PlotCustomizer,
):
    # Filter out NaN pairs
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        print(f"[correlations] WARNING: not enough valid data for {filename}")
        return

    x_f = x[mask]
    y_f = y[mask]

    r, p = pearsonr(x_f, y_f)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.scatter(x_f, y_f, color=cfg.dot_color, alpha=cfg.alpha)

    # --- apply shared axis limits if provided
    if cfg.xlim is not None:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim is not None:
        ax.set_ylim(cfg.ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)
    ax.grid(False)

    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        zorder=5,
        bbox=dict(
            facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
        ),
    )

    customizer.adjust_padding_proportionally()
    writeImage(str(cfg.out_dir / f"{filename}.png"), format="png")
    plt.close(fig)


def _ensure_rewards_per_distance(va) -> bool:
    """
    Make sure va.rwdsPerDist exists.
    """
    if getattr(va, "rwdsPerDist", None) is None:
        if hasattr(va, "_rewards_per_distance"):
            va._rewards_per_distance()
        else:
            print(
                "[correlations] WARNING: no rwdsPerDist and no _rewards_per_distance()"
            )
            return False
    return True


def _ensure_reward_pi_pre(va) -> bool:
    """
    Make sure va.rewardPIPre exists (pre-training reward PI).
    """
    if getattr(va, "rewardPIPre", None) is None:
        if hasattr(va, "calcRewardsPre"):
            va.calcRewardsPre()
        else:
            print("[correlations] WARNING: no rewardPIPre and no calcRewardsPre()")
            return False
    return True


def _last_valid_scalar(row) -> float:
    arr = np.asarray(row, float)
    for v in arr[::-1]:
        if np.isfinite(v):
            return float(v)
    return np.nan


def _fast_slow_indices_from_sli_T1_first(
    sli_T1_first: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute disjoint fast and slow index sets based on SLI in the first
    sync bucket of T1 (reward PI, exp − yoked).

    - fast = top `frac` of finite values
    - slow = bottom `frac` of finite values

    If 2*k would exceed the number of finite flies, k is clamped so that
    fast and slow remain disjoint.
    """
    arr = np.asarray(sli_T1_first, float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=int)

    finite_vals = arr[mask]
    finite_idx = np.arange(arr.shape[0])[mask]
    n_finite = finite_vals.size

    k = max(1, int(frac * n_finite))
    if 2 * k > n_finite:
        k = n_finite // 2

    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(finite_vals)  # ascending
    slow_idx = finite_idx[order[:k]]
    fast_idx = finite_idx[order[-k:]]

    return fast_idx, slow_idx


def _ensure_sync_med_dist(va):
    if not hasattr(va, "syncMedDist") or va.syncMedDist is None:
        if hasattr(va, "bySyncBucketMedDist"):
            va.bySyncBucketMedDist()
        else:
            print("[correlations] WARNING: no syncMedDist and no bySyncBucketMedDist()")


def summarize_fast_vs_strong(
    sli_T1_first: np.ndarray,
    sli_strong: np.ndarray,
    vas,
    opts,
    frac: float = 0.2,
    *,
    strong_label: str = "Strong learners",
):
    """
    Summarize proportions of fast vs strong learners.
    - fast = top percentile of SLI at first sync bucket of T1
    - strong = top percentile of SLI according to `sli_strong`
      (definition controlled upstream; label passed in for logging/printing)

    This version uses *separate* validity masks, so a fly with NaN in one
    bucket can still be classified in the other.
    """
    sli_T1_first = np.asarray(sli_T1_first, float)
    sli_strong = np.asarray(sli_strong, float)

    N_total = len(vas)
    if N_total == 0:
        return

    # Global percentile count (same rule as select_extremes)
    k_global = max(1, int(frac * N_total))

    # FAST LEARNERS (T1 first bucket)
    mask1 = np.isfinite(sli_T1_first)
    sli1 = sli_T1_first[mask1]

    # --- STRONG LEARNERS (definition controlled upstream) ---
    mask2 = np.isfinite(sli_strong)
    sli2 = sli_strong[mask2]

    if len(sli1) == 0 or len(sli2) == 0:
        print("[correlations] WARNING: no finite SLI values for fast/strong summary")
        return

    # clamp k to finite size
    k1 = min(k_global, len(sli1))
    k2 = min(k_global, len(sli2))

    # argpartition selection on the finite values
    idx1 = np.argpartition(sli1, -k1)[-k1:]
    orig_idx1 = np.arange(N_total)[mask1]
    fast_global = set(orig_idx1[idx1])
    idx2 = np.argpartition(sli2, -k2)[-k2:]
    orig_idx2 = np.arange(N_total)[mask2]
    strong_global = set(orig_idx2[idx2])

    # Overlap
    overlap = fast_global & strong_global

    print("\n=== Fast vs Strong learner summary ===")
    print(f"Fast learners:   {len(fast_global)} (k={k1}, from N={N_total})")
    print(f"{strong_label}: {len(strong_global)} (k={k2}, from N={N_total})")
    print(f"Overlap:         {len(overlap)}")

    summary = {
        "fast": np.array(sorted(fast_global)),
        "strong": np.array(sorted(strong_global)),
        "overlap": np.array(sorted(overlap)),
    }

    if getattr(opts, "log_fly_grps", False):
        log_fly_group("FAST_LEARNERS", summary["fast"], vas)
        log_fly_group("STRONG_LEARNERS", summary["strong"], vas)
        log_fly_group("FAST_STRONG_OVERLAP", summary["overlap"], vas)

    return summary


def plot_fast_vs_strong_scatter(
    sli_T1_first: np.ndarray,
    sli_strong: np.ndarray,
    vas,
    fast_idx: np.ndarray,
    strong_idx: np.ndarray,
    out_dir: Path,
    frac: float,
    customizer: PlotCustomizer,
    *,
    strong_y_label: str,
    strong_title_suffix: str,
):
    """
    Scatter plot of:
        X = SLI at T1 first sync bucket (fast learners)
        Y = SLI along timeframe used for strong learners (defined upstream)

    Points are colored by group:
        - Fast-only (fast & not strong)
        - Strong-only (strong & not fast)
        - Overlap (fast & strong)
        - Unclassified (neither)

    Also computes (descriptive) Pearson correlations for:
        - Fast group, including overlap points
        - Strong group, including overlap points
    """
    x = np.asarray(sli_T1_first, float)
    y = np.asarray(sli_strong, float)

    # Masks
    mask_x = np.isfinite(x)
    mask_y = np.isfinite(y)
    mask = mask_x & mask_y  # only for plotting (not classification)

    x_f = x[mask]
    y_f = y[mask]

    # Build global index arrays
    valid_global_idx = np.arange(len(vas))[mask]

    fast_set = set(fast_idx.tolist())
    strong_set = set(strong_idx.tolist())
    overlap_set = fast_set & strong_set

    # Classification per plotted point
    classes = []
    for idx in valid_global_idx:
        if idx in overlap_set:
            classes.append("overlap")
        elif idx in fast_set:
            classes.append("fast")
        elif idx in strong_set:
            classes.append("strong")
        else:
            classes.append("other")

    classes_arr = np.asarray(classes, dtype=object)

    def _corr_from_class_mask(m: np.ndarray) -> tuple[float, float, int] | None:
        """
        Compute Pearson (r, p) on the *plotted* points selected by mask `m`.
        Returns (r, p, n) or None if fewer than 3 points.
        """
        m = np.asarray(m, dtype=bool)
        n = int(np.sum(m))
        if n < 3:
            return None
        r, p = pearsonr(x_f[m], y_f[m])
        return float(r), float(p), n

    # Correlations: include overlap in both fast and strong groups
    # NOTE: correlations are plotted on plotted points (finite x/y) only
    corr_fast_incl_overlap = _corr_from_class_mask(
        (classes_arr == "fast") | (classes_arr == "overlap")
    )
    corr_strong_incl_overlap = _corr_from_class_mask(
        (classes_arr == "strong") | (classes_arr == "overlap")
    )

    # Correlation across *all* plotted points (finite x/y only)
    corr_all = None
    n_all = int(x_f.size)
    if n_all >= 3:
        r_a, p_a = pearsonr(x_f, y_f)
        corr_all = (float(r_a), float(p_a), n_all)

    # Colors (simple, can be refined)
    color_map = {
        "overlap": "#cc0000",  # red
        "fast": "#1f77b4",  # blue
        "strong": "#2ca02c",  # green
        "other": "#aaaaaa",  # gray
    }

    point_colors = [color_map[c] for c in classes]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_f, y_f, c=point_colors, alpha=0.85)

    ax.set_xlabel("SLI (T1, first sync bucket)")
    ax.set_ylabel(strong_y_label)
    ax.set_title(
        f"Fast vs Strong Learners ({strong_title_suffix}, top {frac*100:.0f}% each)",
        pad=10,
    )

    # Create extra vertical headroom for the text block
    # (keeps annotation from overlapping datapoints)
    x_min, x_max = float(np.nanmin(x_f)), float(np.nanmax(x_f))
    y_min, y_max = float(np.nanmin(y_f)), float(np.nanmax(y_f))
    x_rng = x_max - x_min
    y_rng = y_max - y_min
    if not np.isfinite(x_rng) or x_rng <= 0:
        x_rng = 1.0
    if not np.isfinite(y_rng) or y_rng <= 0:
        y_rng = 1.0

    # Add headroom proportional to the number of stat lines we print.
    top_pad = 0.25 * y_rng  # temporary; overwritten after we build `lines`
    ax.set_ylim(y_min, y_max + top_pad)

    # Display descriptive correlations (fast/strong each including overlap)
    lines = []
    if corr_all is not None:
        r_a, p_a, n_a = corr_all
        lines.append(f"All (finite):           r = {r_a:.3f}, p = {p_a:.3g} (n={n_a})")
    else:
        lines.append("All (finite):           r = n/a")
    if corr_fast_incl_overlap is not None:
        r_f, p_f, n_f = corr_fast_incl_overlap
        lines.append(f"Fast (incl overlap):  r = {r_f:.3f}, p = {p_f:.3g} (n={n_f})")
    else:
        lines.append("Fast (incl overlap):  r = n/a")

    if corr_strong_incl_overlap is not None:
        r_s, p_s, n_s = corr_strong_incl_overlap
        lines.append(f"Strong (incl overlap): r = {r_s:.3f}, p = {p_s:.3g} (n={n_s})")
    else:
        lines.append("Strong (incl overlap): r = n/a")

    # Now that we know how many lines we're printing, increase headroom if needed.
    # Keeps the textbox from crowding the point cloud near the top.
    top_pad = max(0.25 * y_rng, (0.10 * len(lines) + 0.05) * y_rng)
    ax.set_ylim(y_min, y_max + top_pad)

    ax.text(
        x_min + 0.02 * x_rng,
        y_max + 0.90 * top_pad,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        zorder=5,
        bbox=dict(
            facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
        ),
    )

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["fast"],
            markersize=8,
            label="Fast only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["strong"],
            markersize=8,
            label="Strong only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["overlap"],
            markersize=8,
            label="Overlap",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["other"],
            markersize=8,
            label="Other",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    # Optional proportional padding
    customizer.adjust_padding_proportionally()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "scatter_fast_vs_strong.png"
    writeImage(str(out_path), format="png")
    plt.close(fig)


def plot_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow(
    pre_pi_diff_vals: np.ndarray,
    reward_pi_first_bucket: np.ndarray,
    fast_idx: np.ndarray,
    slow_idx: np.ndarray,
    out_dir: Path,
    frac: float,
    customizer: PlotCustomizer,
):
    """
    Correlation plot:

        X = pre-training reward PI (exp − yoked)
        Y = reward PI at T1, first sync bucket (exp − yoked)

    All flies are shown, color-coded by membership:

        - Fast learners (top `frac` of SLI in T1 first bucket)
        - Slow learners (bottom `frac` of SLI in T1 first bucket)
        - Other (middle SLI values)

    Correlations are computed separately for:
        - Fast group
        - Slow group
    """
    x = np.asarray(pre_pi_diff_vals, float)
    y = np.asarray(reward_pi_first_bucket, float)

    # Global finite mask for plotting
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        print(
            "[correlations] WARNING: not enough valid data for "
            "pre-PI vs early-PI fast/slow plot"
        )
        return

    x_f = x[mask]
    y_f = y[mask]
    valid_global_idx = np.arange(x.shape[0])[mask]

    fast_set = set(np.asarray(fast_idx, dtype=int).tolist())
    slow_set = set(np.asarray(slow_idx, dtype=int).tolist())

    color_map = {
        "fast": "#1f77b4",  # blue
        "slow": "#cc0000",  # red
        "other": "#aaaaaa",  # gray
    }

    classes = []
    point_colors = []

    for idx in valid_global_idx:
        if idx in fast_set:
            cls = "fast"
        elif idx in slow_set:
            cls = "slow"
        else:
            cls = "other"
        classes.append(cls)
        point_colors.append(color_map[cls])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_f, y_f, c=point_colors, alpha=0.85)

    ax.set_xlabel("\nBPI\n(exp - yok, pre-training)")
    ax.set_ylabel("SLI\n(T1, first sync bucket)")
    ax.set_title(
        f"Pre-training vs early reward preference\n"
        f"(fast vs slow learners, top/bottom {frac * 100:.0f}% SLI)"
    )

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["fast"],
            markersize=8,
            label="Fast",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["slow"],
            markersize=8,
            label="Slow",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map["other"],
            markersize=8,
            label="Other",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    # Correlations for each group
    corr_fast = _compute_group_corr(x, y, fast_idx)
    corr_slow = _compute_group_corr(x, y, slow_idx)

    lines = []
    if corr_fast is not None:
        r_f, p_f = corr_fast
        lines.append(f"Fast:  r = {r_f:.3f}, p = {p_f:.3g}")
    else:
        lines.append("Fast:  r = n/a")

    if corr_slow is not None:
        r_s, p_s = corr_slow
        lines.append(f"Slow:  r = {r_s:.3f}, p = {p_s:.3g}")
    else:
        lines.append("Slow:  r = n/a")

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        zorder=5,
        bbox=dict(
            facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
        ),
    )

    customizer.adjust_padding_proportionally()

    out_path = out_dir / "corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow.png"
    writeImage(str(out_path), format="png")
    plt.close(fig)


def plot_cross_fly_correlations(
    sli_values: Sequence[float],
    vas: Sequence,
    training_idx: int,
    opts,
    reward_pi_first_bucket: Sequence[float] | None = None,
    out_dir: str | Path = "imgs/correlations",
    plot_customizer: PlotCustomizer | None = None,
    *,
    sli_ctx: SLIContext | None = None,
):
    """
    Cross-fly correlations:

      1) SLI_final vs reward-per-distance (final bucket of chosen training)
      2) SLI_final vs median distance to reward during chosen training
      3) Pre-training reward PI (exp − yoked) vs SLI_final
      3b) Pre-training floor exploration vs SLI at T1, first sync bucket
      3c) Pre-training floor exploration vs SLI_final
      4) Reward PI (T1, first sync bucket, exp − yoked) vs total rewards
         in that same bucket (experimental fly)
      5) Pre-training reward PI (exp − yoked) vs T1 first-bucket reward PI:
           a) all learners
           b) fast learners only
           c) fast vs slow learners (top and bottom percentile of early SLI)
      6) SLI at T1 first sync bucket vs SLI at T2 final sync bucket,
         color-coded by fast / strong / overlap / other.

    `sli_values` should be a 1D sequence aligned with `vas`
    (one SLI per VideoAnalysis / learner).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CorrelationPlotConfig(
        out_dir=out_dir,
        xlim=getattr(opts, "corr_xlim", None),
        ylim=getattr(opts, "corr_ylim", None),
    )
    customizer = plot_customizer or PlotCustomizer()

    sli_vals = np.asarray(sli_values, float)
    if sli_vals.shape[0] != len(vas):
        print(
            "[correlations] WARNING: len(sli_values) != len(vas) "
            f"({sli_vals.shape[0]} vs {len(vas)})"
        )

    reward_pi_training_vals = None
    if reward_pi_first_bucket is not None:
        reward_pi_training_vals = np.asarray(reward_pi_first_bucket, float)
        if reward_pi_training_vals.shape[0] != len(vas):
            print(
                "[correlations] WARNING: len(reward_pi_first_bucket) != len(vas) "
                f"({reward_pi_training_vals.shape[0]} vs {len(vas)})"
            )

    if sli_ctx is None:
        sli_ctx = SLIContext(training_idx=training_idx, average_over_buckets=False)

    x_label_sli = sli_ctx.label_long()

    rpd_vals = []
    med_train_vals = []
    pre_pi_diff_vals = []
    total_reward_vals = []
    pre_coverage_vals = []

    for va in vas:
        # --- Reward per distance (final bucket of training_idx) ---
        if _ensure_rewards_per_distance(va):
            row_idx = 2 * training_idx  # exp row
            if 0 <= row_idx < len(va.rwdsPerDist):
                exp_row = va.rwdsPerDist[row_idx]
                rpd_val = _last_valid_scalar(exp_row)
            else:
                rpd_val = np.nan
        else:
            rpd_val = np.nan

        # --- Median distance to reward during training ---
        _ensure_sync_med_dist(va)
        if hasattr(va, "syncMedDist") and training_idx < len(va.syncMedDist):
            med_vec = np.asarray(va.syncMedDist[training_idx].get("exp", []), float)
            med_train = np.nanmedian(med_vec) if med_vec.size else np.nan
        else:
            med_train = np.nan

        # --- Pre-training reward preference index (exp − yoked) ---
        if _ensure_reward_pi_pre(va):
            pre_arr = np.asarray(getattr(va, "rewardPIPre", []), float)
            if pre_arr.size == 0:
                pre_diff = np.nan
            elif pre_arr.size == 1:
                # No yoked partner; just use the single value
                pre_diff = float(pre_arr[0])
            else:
                # Assume index 0 = experimental, 1 = yoked
                pre_diff = float(pre_arr[0] - pre_arr[1])
        else:
            pre_diff = np.nan

        # --- Pre-training floor exploration (experimental fly only) ---
        coverage = np.nan
        try:
            if not hasattr(va, "preFloorExploredFrac"):
                if hasattr(va, "calcPreFloorExploration"):
                    va.calcPreFloorExploration()
            if hasattr(va, "preFloorExploredFrac") and len(va.preFloorExploredFrac) > 0:
                coverage = float(va.preFloorExploredFrac[0])
        except Exception:
            coverage = np.nan

        # --- Total rewards in the same sync bucket used for the reward-PI X variable ---
        try:
            calc_idx = 1
            training_idx_T1 = 0
            bucket_idx = 0  # first sync bucket of T1

            tot = getattr(va, "numRewardsTot", None)

            if (
                isinstance(tot, (list, tuple))
                and len(tot) >= calc_idx
                and isinstance(tot[calc_idx], (list, tuple))
            ):
                flat_list = tot[calc_idx][
                    0
                ]  # 0 = reward circle; entries: (exp T1, yok T1, exp T2, yok T2, ...)

                # compute flat index into alternating exp/yok structure
                flat_idx_exp = 2 * training_idx_T1

                if flat_idx_exp < len(flat_list):
                    bucket_vals = flat_list[flat_idx_exp]
                    if isinstance(
                        bucket_vals, (list, tuple, np.ndarray)
                    ) and bucket_idx < len(bucket_vals):
                        total_rewards = float(bucket_vals[bucket_idx])
                    else:
                        total_rewards = np.nan
                else:
                    total_rewards = np.nan
            else:
                total_rewards = np.nan

        except Exception:
            total_rewards = np.nan

        rpd_vals.append(rpd_val)
        med_train_vals.append(med_train)
        pre_pi_diff_vals.append(pre_diff)
        total_reward_vals.append(total_rewards)
        pre_coverage_vals.append(coverage)

    rpd_vals = np.asarray(rpd_vals, float)
    med_train_vals = np.asarray(med_train_vals, float)
    pre_pi_diff_vals = np.asarray(pre_pi_diff_vals, float)
    total_reward_vals = np.asarray(total_reward_vals, float)
    pre_coverage_vals = np.asarray(pre_coverage_vals, float)

    # --- Fast/strong learner summary (for plots 5b/5c and fast/strong scatter) ---
    summary = None
    if reward_pi_training_vals is not None:
        try:
            frac = getattr(opts, "best_worst_fraction", 0.2)
            strong_label = (
                f"Strong learners (top {frac*100:.1f}%, {sli_ctx.label_short()})"
            )
            summary = summarize_fast_vs_strong(
                sli_T1_first=reward_pi_training_vals,
                sli_strong=sli_vals,
                vas=vas,
                opts=opts,
                frac=frac,
                strong_label=strong_label,
            )
        except Exception as e:
            print(f"[correlations] WARNING: failed fast/strong summary: {e}")

    # --- Plot 1: SLI_final vs reward-per-distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpd_vals,
        title="Rewards per distance vs SLI",
        x_label=x_label_sli,
        y_label="rewards per distance $[m^{-1}]$\n$(\\text{exp} - \\text{yok})$",
        cfg=cfg,
        filename="corr_rpd_vs_sli",
        customizer=customizer,
    )

    # --- Plot 2: SLI_final vs median training distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=med_train_vals,
        title="SLI vs median distance to reward",
        x_label=x_label_sli,
        y_label="Median distance during training (mm)",
        cfg=cfg,
        filename="corr_sli_vs_median_training",
        customizer=customizer,
    )

    # --- Plot 3: Pre-training reward PI (exp − yoked) vs SLI_final ---
    _scatter_with_corr(
        x=pre_pi_diff_vals,
        y=sli_vals,
        title="Baseline PI vs SLI",
        x_label="Baseline PI\n(exp − yok, pre-training)",
        y_label=x_label_sli,
        cfg=cfg,
        filename="corr_pre_reward_pi_vs_sli",
        customizer=customizer,
    )

    # --- Plot 3b: Pre-training exploration vs SLI at T1, first sync bucket ---
    if reward_pi_training_vals is not None:
        _scatter_with_corr(
            x=pre_coverage_vals,
            y=reward_pi_training_vals,
            title="Pre-training exploration vs early SLI",
            x_label="Fraction of floor explored during pre-training\n(exp fly)",
            y_label="SLI (T1, first sync bucket)",
            cfg=cfg,
            filename="corr_pre_floor_exploration_vs_sli_T1_first",
            customizer=customizer,
        )
    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; "
            "skipping pre-training exploration vs early SLI plot"
        )

    # --- Plot 3c: Pre-training exploration vs SLI_final (training {trn_label_idx}) ---
    _scatter_with_corr(
        x=pre_coverage_vals,
        y=sli_vals,
        title="Pre-training exploration vs SLI",
        x_label="Fraction of floor explored during pre-training\n(exp fly)",
        y_label=x_label_sli,
        cfg=cfg,
        filename="corr_pre_floor_exploration_vs_sli_final",
        customizer=customizer,
    )

    if reward_pi_training_vals is not None:
        # --- Plot 4: Reward PI (T1, first bucket) vs total rewards in that bucket ---
        _scatter_with_corr(
            x=reward_pi_training_vals,
            y=total_reward_vals,
            title="Early SLI vs total rewards",
            x_label="SLI\n(T1, first sync bucket)",
            y_label="Total rewards\n(exp, T1, first sync bucket)",
            cfg=cfg,
            filename="corr_reward_pi_first_bucket_vs_total_rewards",
            customizer=customizer,
        )

        # --- Plot 5a: Pre-training PI vs T1 first-bucket PI (all learners) ---
        _scatter_with_corr(
            x=pre_pi_diff_vals,
            y=reward_pi_training_vals,
            title="Baseline PI vs early SLI",
            x_label="Baseline PI\n(exp - yok, pre-training)",
            y_label="SLI\n(T1, first sync bucket)",
            cfg=cfg,
            filename="corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_all",
            customizer=customizer,
        )

        # --- Plot 5b: Pre-training PI vs T1 first-bucket PI (fast learners only) ---
        if summary is not None and "fast" in summary:
            fast_idx = np.asarray(summary["fast"], dtype=int)
            if fast_idx.size == 0:
                print(
                    "[correlations] WARNING: no fast learners; "
                    "skipping fast-only pre-vs-early PI correlation"
                )
            else:
                _scatter_with_corr(
                    x=pre_pi_diff_vals[fast_idx],
                    y=reward_pi_training_vals[fast_idx],
                    title="Baseline PI vs early SLI (fast learners)",
                    x_label="Baseline PI\n(exp - yok, pre-training)",
                    y_label="SLI\n(T1, first sync bucket)",
                    cfg=cfg,
                    filename="corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast",
                    customizer=customizer,
                )
        else:
            print(
                "[correlations] WARNING: missing fast-learner summary; "
                "skipping fast-only pre-vs-early PI correlation"
            )

        # --- Plot 5c: Pre-training vs T1 first-bucket PI (fast vs slow) ---
        frac = getattr(opts, "best_worst_fraction", 0.2)
        fast_idx_fs, slow_idx_fs = _fast_slow_indices_from_sli_T1_first(
            reward_pi_training_vals, frac
        )

        if fast_idx_fs.size == 0 or slow_idx_fs.size == 0:
            print(
                "[correlations] WARNING: empty fast/slow groups; "
                "skipping fast/slow pre-vs-early PI correlation"
            )
        else:
            plot_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast_slow(
                pre_pi_diff_vals=pre_pi_diff_vals,
                reward_pi_first_bucket=reward_pi_training_vals,
                fast_idx=fast_idx_fs,
                slow_idx=slow_idx_fs,
                out_dir=out_dir,
                frac=frac,
                customizer=customizer,
            )

    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; "
            "skipping plots 4–5"
        )

    if summary is not None:
        plot_fast_vs_strong_scatter(
            sli_T1_first=reward_pi_training_vals,
            sli_strong=sli_vals,
            vas=vas,
            fast_idx=summary["fast"],
            strong_idx=summary["strong"],
            out_dir=out_dir,
            frac=frac,
            customizer=customizer,
            strong_y_label=x_label_sli,
            strong_title_suffix=sli_ctx.label_short(),
        )
