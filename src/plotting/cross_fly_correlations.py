# src/plotting/cross_fly_correlations.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)

    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
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


def _ensure_sync_med_dist(va, min_no_contact_s=None):
    if not hasattr(va, "syncMedDist") or va.syncMedDist is None:
        if hasattr(va, "bySyncBucketMedDist"):
            va.bySyncBucketMedDist(min_no_contact_s=min_no_contact_s)
        else:
            print("[correlations] WARNING: no syncMedDist and no bySyncBucketMedDist()")


def summarize_fast_vs_strong(
    sli_T1_first: np.ndarray,
    sli_T2_last: np.ndarray,
    vas,
    opts,
    frac: float = 0.2,
):
    """
    Summarize proportions of fast vs strong learners.
    - fast = top percentile of SLI at first sync bucket of T1
    - strong = top percentile of SLI at last sync bucket of T2

    This version uses *separate* validity masks, so a fly with NaN in one
    bucket can still be classified in the other.
    """
    sli_T1_first = np.asarray(sli_T1_first, float)
    sli_T2_last = np.asarray(sli_T2_last, float)

    N_total = len(vas)
    if N_total == 0:
        return

    # Global percentile count (same rule as select_extremes)
    k_global = max(1, int(frac * N_total))

    # FAST LEARNERS (T1 first bucket)
    mask1 = np.isfinite(sli_T1_first)
    sli1 = sli_T1_first[mask1]

    # --- STRONG LEARNERS (T2 last bucket) ---
    mask2 = np.isfinite(sli_T2_last)
    sli2 = sli_T2_last[mask2]

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
    print(f"Strong learners: {len(strong_global)} (k={k2}, from N={N_total})")
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
    sli_T2_last: np.ndarray,
    vas,
    fast_idx: np.ndarray,
    strong_idx: np.ndarray,
    out_dir: Path,
    frac: float,
    customizer: PlotCustomizer,
):
    """
    Scatter plot of:
        X = SLI at T1 first sync bucket (fast learners)
        Y = SLI at T2 final sync bucket (strong learners)

    Points are colored by group:
        - Fast-only (fast & not strong)
        - Strong-only (strong & not fast)
        - Overlap (fast & strong)
        - Unclassified (neither)
    """
    x = np.asarray(sli_T1_first, float)
    y = np.asarray(sli_T2_last, float)

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
    ax.set_ylabel("SLI (T2, last sync bucket)")
    ax.set_title(f"Fast vs Strong Learners (top {frac*100:.0f}% each)")

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

    out_path = out_dir / "scatter_fast_vs_strong.png"
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
):
    """
    Cross-fly correlations:

      1) SLI_final vs reward-per-distance (final bucket of chosen training)
      2) SLI_final vs median distance to reward during chosen training
      3) Pre-training reward PI (exp − yoked) vs SLI_final
      4) Reward PI (T1, first sync bucket, exp - yoked) vs total rewards
         in that same bucket (experimental fly)
      5) Pre-training reward PI (exp − yoked) vs first-bucket reward PI:
            a) all learners
            b) fast learners only

    `sli_values` should be a 1D sequence aligned with `vas`
    (one SLI per VideoAnalysis / learner).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CorrelationPlotConfig(out_dir=out_dir)
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

    # Use 1-based training index in axis label
    trn_label_idx = training_idx + 1
    x_label_sli = f"SLI (last sync bucket, training {trn_label_idx})"

    rpd_vals = []
    med_train_vals = []
    pre_pi_diff_vals = []
    total_reward_vals = []

    min_no_contact_s = getattr(opts, "min_no_contact_s", None)

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
        _ensure_sync_med_dist(va, min_no_contact_s=min_no_contact_s)
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

    rpd_vals = np.asarray(rpd_vals, float)
    med_train_vals = np.asarray(med_train_vals, float)
    pre_pi_diff_vals = np.asarray(pre_pi_diff_vals, float)
    total_reward_vals = np.asarray(total_reward_vals, float)

    # --- Fast/strong learner summary (needed for fast-only Plot 5) ---
    summary = None
    if reward_pi_training_vals is not None:
        try:
            summary = summarize_fast_vs_strong(
                sli_T1_first=reward_pi_training_vals,
                sli_T2_last=sli_vals,
                vas=vas,
                opts=opts,
                frac=getattr(opts, "best_worst_fraction", 0.2),
            )
        except Exception as e:
            print(f"[correlations] WARNING: failed fast/strong summary: {e}")

    # --- Plot 1: SLI_final vs reward-per-distance ---
    _scatter_with_corr(
        x=sli_vals,
        y=rpd_vals,
        title="SLI vs rewards per distance",
        x_label=x_label_sli,
        y_label="rewards per distance $[m^{-1}]$\n$(\\text{exp} - \\text{yok})$",
        cfg=cfg,
        filename="corr_sli_vs_rpd",
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
        title="Pre-training reward preference vs SLI",
        x_label="Reward PI (pre-training, exp − yoked)",
        y_label=x_label_sli,
        cfg=cfg,
        filename="corr_pre_reward_pi_vs_sli",
        customizer=customizer,
    )

    if reward_pi_training_vals is not None:
        # --- Plot 4: Reward PI (T1, first bucket) vs total rewards in that bucket ---
        _scatter_with_corr(
            x=reward_pi_training_vals,
            y=total_reward_vals,
            title="Reward PI vs total rewards",
            x_label="Reward PI\n(exp - yok, T1, first sync bucket)",
            y_label="Total rewards\n(exp, T1, first sync bucket)",
            cfg=cfg,
            filename="corr_reward_pi_first_bucket_vs_total_rewards",
            customizer=customizer,
        )

        # --- Plot 5a: Pre-training PI vs T1 first-bucket PI (all learners) ---
        _scatter_with_corr(
            x=pre_pi_diff_vals,
            y=reward_pi_training_vals,
            title="Pre-training vs early reward preference (all learners)",
            x_label="Reward PI\n(exp - yok, pre-training)",
            y_label="Reward PI\n(exp - yok, T1, first sync bucket)",
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
                    title="Pre-training vs early reward preference (fast learners)",
                    x_label="Reward PI\n(exp - yok, pre-training)",
                    y_label="Reward PI\n(exp - yok, T1, first sync bucket)",
                    cfg=cfg,
                    filename="corr_pre_reward_pi_vs_T1_first_bucket_reward_pi_fast",
                    customizer=customizer,
                )
        else:
            print(
                "[correlations] WARNING: missing fast-learner summary; "
                "skipping fast-only pre-vs-early PI correlation"
            )
    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; "
            "skipping plots 4–5"
        )

    if summary is not None:
        plot_fast_vs_strong_scatter(
            sli_T1_first=reward_pi_training_vals,
            sli_T2_last=sli_vals,
            vas=vas,
            fast_idx=summary["fast"],
            strong_idx=summary["strong"],
            out_dir=Path(out_dir),
            frac=getattr(opts, "best_worst_fraction", 0.2),
            customizer=customizer,
        )
