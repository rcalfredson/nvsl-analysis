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
                and len(tot) >= 2
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

    # --- Plot 4: SLI_final vs total rewards ---
    if reward_pi_training_vals is not None:
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
    else:
        print(
            "[correlations] WARNING: missing reward_pi_training_vals; skipping plot 4"
        )
