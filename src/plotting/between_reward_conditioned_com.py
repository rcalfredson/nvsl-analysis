from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils.common import writeImage
from src.utils.util import meanConfInt


@dataclass
class BetweenRewardConditionedCOMConfig:
    """
    Distance-binned COM analysis for between-reward trajectories.

    Output:
        x-axis: distance-from-reward bins (mm)
        y-axis: mean COM magnitude (mm) per fly, then mean+CI across flies

    Notes:
        - This plotter treats each "fly unit" as one row in the aggregation.
          In exp+yoked recordings, this is the experimental fly only (f==0).
        - Wall-contact filtering reuses boundary-contact event data (if present).
        - Training window can be restricted to a subset of sync buckets.
    """

    out_file: str

    # Which training to analyze (0-based index).
    training_index: int = 1  # Training 2 by default (0-based)

    # Exclude first K sync buckets from the training window (e.g. skip bucket 0).
    skip_first_sync_buckets: int = 0

    # If True: use VideoAnalysis.reward_exclusion_mask to mark some buckets incomplete.
    # This is optional; can be disabled if you want "all buckets are complete" behavior.
    use_reward_exclusion_mask: bool = False

    # Conditioning variable (x-axis).
    # - "median": uses seg.med_d_mm
    # - "max": uses seg.max_d_mm
    cond_stat: str = "median"

    # Distance binning (mm)
    x_bin_width_mm: float = 2.0
    x_min_mm: float = 0.0
    x_max_mm: float = 20.0

    # Plot options
    ci_conf: float = 0.95
    ymax: float | None = None
    subset_label: str | None = None


class BetweenRewardConditionedCOMPlotter:
    """
    Plot conditioned between-reward COM magnitude vs. a trajectory-level distance statistic.

    Current support:
        - cond_stat == "median" (uses seg.med_d_mm)
        - cond_stat == "max"    (uses seg.max_d_mm)
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardConditionedCOMConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = "btw_rwd_dist_binned_com"

    # ---------------------------- bucket windowing ----------------------------

    def _sync_bucket_window(
        self,
        va: "VideoAnalysis",
        trn,
        *,
        t_idx: int,
        f: int,
        skip_first: int,
        use_exclusion_mask: bool,
    ) -> tuple[int, int, int, list[bool]]:
        """
        Return (fi, df, n_buckets, complete) describing the included sync buckets.

        - fi: absolute start frame of first included bucket
        - df: bucket length in frames (validated for uniformity)
        - n_buckets: number of included buckets
        - complete: list[bool], one per included bucket

        Fallback behavior if sync buckets aren't available:
            - single bucket spanning the whole training [trn.start, trn.stop)
        """
        ranges = getattr(va, "sync_bucket_ranges", None)
        if not ranges or t_idx >= len(ranges):
            fi0 = int(trn.start)
            df0 = int(max(1, trn.stop - trn.start))
            return (fi0, df0, 1, [True])

        rr = ranges[t_idx]
        if not rr:
            fi0 = int(trn.start)
            df0 = int(max(1, trn.stop - trn.start))
            return (fi0, df0, 1, [True])

        if skip_first < 0:
            skip_first = 0
        if skip_first >= len(rr):
            return (0, 1, 0, [])

        rr2 = rr[skip_first:]
        fi = int(rr2[0][0])

        df_list = [int(b - a) for (a, b) in rr2 if b > a]
        if not df_list:
            return (0, 1, 0, [])

        df = int(df_list[0])
        if any(int(d) != df for d in df_list):
            # Non-uniform bucket widths; fall back to single wide bucket.
            la = int(rr2[-1][1])
            df_fallback = int(max(1, la - fi))
            return (fi, df_fallback, 1, [True])

        n_buckets = int(len(rr2))

        if use_exclusion_mask and hasattr(va, "reward_exclusion_mask"):
            # reward_exclusion_mask indexed by [t.n-1][f][b_idx]
            try:
                mask = va.reward_exclusion_mask[trn.n - 1][f]
            except Exception:
                mask = []
            complete = []
            for j in range(n_buckets):
                b_idx_orig = j + skip_first
                is_excl = bool(mask[b_idx_orig]) if b_idx_orig < len(mask) else False
                complete.append(not is_excl)
        else:
            complete = [True] * n_buckets

        return (fi, df, n_buckets, complete)

    # ---------------------------- binning by distance ----------------------------

    def _x_edges(self) -> np.ndarray:
        w = float(self.cfg.x_bin_width_mm)
        if not np.isfinite(w) or w <= 0:
            w = 2.0
        x0 = float(self.cfg.x_min_mm)
        x1 = float(self.cfg.x_max_mm)
        if not np.isfinite(x0):
            x0 = 0.0
        if not np.isfinite(x1) or x1 <= x0:
            x1 = x0 + 10.0

        # Ensure last edge reaches at least x_max_mm.
        edges = np.arange(x0, x1 + 0.5 * w, w, dtype=float)
        if edges.size < 2:
            edges = np.array([x0, x0 + w], dtype=float)
        return edges

    def _cond_value_for_segment(self, seg) -> float:
        """
        Return the conditioning x-value (mm) for a segment.

        Options:
            - median distance (seg.med_d_mm)
            - max distance (seg.max_d_mm)
        """
        cs = str(self.cfg.cond_stat or "median").lower().strip()
        if cs in ("median", "med", "meddist"):
            return float(getattr(seg, "med_d_mm", np.nan))
        if cs in ("max", "maxdist", "max_d"):
            return float(getattr(seg, "max_d_mm", np.nan))
        raise ValueError(f"Unknown cond_stat={self.cfg.cond_stat!r}")

    # ---------------------------- core computation ----------------------------

    def _collect_per_fly_binned_means(self) -> list[np.ndarray]:
        """
        For each fly unit, compute a (B,) vector where each entry is:

            mean(COM_mag_mm) over segments whose cond_stat falls into x-bin j,

        with NaN for bins with no segments.
        """
        edges = self._x_edges()
        B = int(max(1, edges.size - 1))

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        per_fly_vectors: list[np.ndarray] = []

        t_idx = int(self.cfg.training_index)
        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue
            trns = getattr(va, "trns", [])
            if t_idx < 0 or t_idx >= len(trns):
                continue
            trn = trns[t_idx]

            for f in va.flies:
                # experimental-only in exp+yoked
                if not va.noyc and f != 0:
                    continue

                fi, df, n_buckets, complete = self._sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=f,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = build_wall_contact_mask_for_window(
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=exclude_wall,
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )

                # Collect segment COM magnitudes into bins by conditioning stat
                bin_vals: list[list[float]] = [[] for _ in range(B)]

                cs = str(self.cfg.cond_stat or "median").lower().strip()
                if cs in ("max", "maxdist", "max_d"):
                    dist_stats = ("median", "max")
                else:
                    dist_stats = ("median",)

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    f,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=min_med_mm,
                    exclude_wall=exclude_wall,
                    wc=wc,
                    dist_stats=dist_stats,
                    debug=False,
                    yield_skips=False,
                ):
                    x = self._cond_value_for_segment(seg)
                    y = float(getattr(seg, "mag_mm", np.nan))
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    # Bin index
                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue
                    bin_vals[j].append(y)

                # Reduce per bin to a per-fly mean
                vec = np.full((B,), np.nan, dtype=float)
                for j in range(B):
                    if not bin_vals[j]:
                        continue
                    vv = np.asarray(bin_vals[j], dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size == 0:
                        continue
                    vec[j] = float(np.nanmean(vv))

                # Keep fly if it contributed at least one bin
                if np.any(np.isfinite(vec)):
                    per_fly_vectors.append(vec)

        return per_fly_vectors

    def compute_summary(self) -> dict:
        """
        Returns dict with:
            - x_edges, x_centers
            - mean, ci_lo, ci_hi, n_units
            - meta
        """
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        per_fly = self._collect_per_fly_binned_means()
        if not per_fly:
            return {
                "x_edges": edges,
                "x_centers": centers,
                "mean": np.full((B,), np.nan, dtype=float),
                "ci_lo": np.full((B,), np.nan, dtype=float),
                "ci_hi": np.full((B,), np.nan, dtype=float),
                "n_units": np.zeros((B,), dtype=int),
                "meta": {
                    "log_tag": self.log_tag,
                    "training_index": int(self.cfg.training_index),
                    "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
                    "use_reward_exclusion_mask": bool(
                        self.cfg.use_reward_exclusion_mask
                    ),
                    "dist_stat": str(self.cfg.cond_stat),
                    "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
                    "x_min_mm": float(self.cfg.x_min_mm),
                    "x_max_mm": float(self.cfg.x_max_mm),
                    "ci_conf": float(self.cfg.ci_conf),
                    "n_fly_units": 0,
                },
            }

        Y = np.stack(per_fly, axis=0)  # (N, B)

        mean = np.full((B,), np.nan, dtype=float)
        lo = np.full((B,), np.nan, dtype=float)
        hi = np.full((B,), np.nan, dtype=float)
        n_units = np.zeros((B,), dtype=int)

        for j in range(B):
            m, lo_j, hi_j, n_j = meanConfInt(Y[:, j], conf=float(self.cfg.ci_conf))
            mean[j] = float(m)
            lo[j] = float(lo_j)
            hi[j] = float(hi_j)
            n_units[j] = int(n_j)

        return {
            "x_edges": edges,
            "x_centers": centers,
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_units": n_units,
            "meta": {
                "log_tag": self.log_tag,
                "training_index": int(self.cfg.training_index),
                "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
                "use_reward_exclusion_mask": bool(self.cfg.use_reward_exclusion_mask),
                "dist_stat": str(self.cfg.cond_stat),
                "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
                "x_min_mm": float(self.cfg.x_min_mm),
                "x_max_mm": float(self.cfg.x_max_mm),
                "ci_conf": float(self.cfg.ci_conf),
                "n_fly_units": int(Y.shape[0]),
            },
        }

    def plot(self) -> None:
        """
        Plot mean ± CI across flies as a function of distance bins.
        """
        data = self.compute_summary()
        x = np.asarray(data["x_centers"], dtype=float)
        y = np.asarray(data["mean"], dtype=float)
        lo = np.asarray(data["ci_lo"], dtype=float)
        hi = np.asarray(data["ci_hi"], dtype=float)
        n_units = np.asarray(data["n_units"], dtype=int)

        fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

        if not np.any(np.isfinite(y)):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            ax.plot(x, y, marker="o", linewidth=1.5)

            # error bars (non-negative deltas)
            yerr = np.vstack([y - lo, hi - y])
            yerr = np.where(np.isfinite(yerr), yerr, 0)
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=2, ecolor="0.2")

            ax.set_xlabel(
                "Distance from reward center [mm]\n(binned by trajectory statistic)"
            )
            ax.set_ylabel("Mean COM magnitude per fly [mm]")
            ax.set_ylim(bottom=0)
            if self.cfg.ymax is not None:
                ax.set_ylim(top=float(self.cfg.ymax))

            # light annotation: how many flies contributed per bin
            # (kept minimal so it doesn't clutter; can be upgraded later)
            for xi, yi, ni in zip(x, y, n_units):
                if np.isfinite(yi) and ni > 0:
                    ax.text(xi, yi, f" {ni}", fontsize=7, va="center")

        title = "Between-reward COM magnitude vs distance-from-reward"
        subtitle_parts = []
        subtitle_parts.append(f"Training {int(self.cfg.training_index) + 1}")
        if int(self.cfg.skip_first_sync_buckets) > 0:
            subtitle_parts.append(
                f"skip first {int(self.cfg.skip_first_sync_buckets)} bucket(s)"
            )
        subtitle_parts.append(f"binned by: {self.cfg.cond_stat}")
        if self.cfg.subset_label:
            subtitle_parts.append(self.cfg.subset_label)
        fig.suptitle(title + "\n" + " | ".join(subtitle_parts))

        fig.tight_layout()
        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
