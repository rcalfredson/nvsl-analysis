from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils.common import maybe_sentence_case, ttest_ind, writeImage


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


@dataclass(frozen=True)
class BetweenRewardConditionedCOMResult:
    """
    Portable result object for distance-binned between-reward COM analysis.

    Intended use:
        - Stage A: compute once per group (slow), save_npz(...)
        - Stage B: load multiple cached results, plot side-by-side (fast)
    """

    x_edges: np.ndarray
    x_centers: np.ndarray
    mean: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    n_units: np.ndarray
    meta: dict
    per_unit: np.ndarray | None = None  # (N_units, B) NaN where no data

    def validate(self) -> None:
        x_edges = np.asarray(self.x_edges, dtype=float)
        x_centers = np.asarray(self.x_centers, dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        ci_lo = np.asarray(self.ci_lo, dtype=float)
        ci_hi = np.asarray(self.ci_hi, dtype=float)
        n_units = np.asarray(self.n_units, dtype=int)

        if x_edges.ndim != 1 or x_edges.size < 2:
            raise ValueError("x_edges must be 1D with at least 2 entries")
        B = int(x_edges.size - 1)
        if x_centers.ndim != 1 or x_centers.size != B:
            raise ValueError(f"x_centers must be 1D with length {B}")
        for name, arr in (
            ("mean", mean),
            ("ci_lo", ci_lo),
            ("ci_hi", ci_hi),
        ):
            if arr.ndim != 1 or arr.size != B:
                raise ValueError(f"{name} must be 1D with length {B}")
        if n_units.ndim != 1 or n_units.size != B:
            raise ValueError(f"n_units must be 1D with length {B}")
        if self.per_unit is not None:
            pu = np.asarray(self.per_unit, dtype=float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit must be 2D with shape (N, {B})")

    def save_npz(self, path: str) -> None:
        """
        Save the result to a compressed NPZ. `meta` is stored as an object array.
        """
        self.validate()
        kwargs = dict(
            x_edges=np.asarray(self.x_edges, dtype=float),
            x_centers=np.asarray(self.x_centers, dtype=float),
            mean=np.asarray(self.mean, dtype=float),
            ci_lo=np.asarray(self.ci_lo, dtype=float),
            ci_hi=np.asarray(self.ci_hi, dtype=float),
            n_units=np.asarray(self.n_units, dtype=int),
            meta=np.array([self.meta], dtype=object),
        )
        if self.per_unit is not None:
            kwargs["per_unit"] = np.asarray(self.per_unit, dtype=float)
        np.savez_compressed(path, **kwargs)

    @staticmethod
    def load_npz(path: str) -> BetweenRewardConditionedCOMResult:
        """
        Load a result saved via save_npz(...).
        """
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta_obj = z["meta"]
                meta = meta_obj.item() if hasattr(meta_obj, "item") else {}
            except Exception:
                meta = {}
        per_unit = None
        if "per_unit" in z:
            per_unit = np.asarray(z["per_unit"], dtype=float)
        res = BetweenRewardConditionedCOMResult(
            x_edges=np.asarray(z["x_edges"], dtype=float),
            x_centers=np.asarray(z["x_centers"], dtype=float),
            mean=np.asarray(z["mean"], dtype=float),
            ci_lo=np.asarray(z["ci_lo"], dtype=float),
            ci_hi=np.asarray(z["ci_hi"], dtype=float),
            n_units=np.asarray(z["n_units"], dtype=int),
            meta=dict(meta) if isinstance(meta, dict) else {},
            per_unit=per_unit,
        )
        res.validate()
        return res


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
            m, lo_j, hi_j, n_j = util.meanConfInt(Y[:, j], conf=float(self.cfg.ci_conf))
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

    def compute_result(self) -> BetweenRewardConditionedCOMResult:
        """
        Return a portable result object suitable for caching/export.
        """
        d = self.compute_summary()
        per_unit = None
        try:
            per_fly = self._collect_per_fly_binned_means()
            if per_fly:
                per_unit = np.stack(per_fly, axis=0)
        except Exception:
            per_unit = None

        return BetweenRewardConditionedCOMResult(
            x_edges=np.asarray(d["x_edges"], dtype=float),
            x_centers=np.asarray(d["x_centers"], dtype=float),
            mean=np.asarray(d["mean"], dtype=float),
            ci_lo=np.asarray(d["ci_lo"], dtype=float),
            ci_hi=np.asarray(d["ci_hi"], dtype=float),
            n_units=np.asarray(d["n_units"], dtype=int),
            meta=dict(d.get("meta", {})),
            per_unit=per_unit,
        )

    def plot(self) -> None:
        """
        Plot mean ± CI across flies as a function of distance bins (histogram-style bars)
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
            # Histogram-like bars: one bar per distance bin (height = mean over flies)
            color = "C0"
            edges = np.asarray(data["x_edges"], dtype=float)
            widths = edges[1:] - edges[:-1]

            fin = np.isfinite(y) & np.isfinite(x) & np.isfinite(widths)

            ax.bar(
                x[fin],
                y[fin],
                width=0.92 * widths[fin],
                align="center",
                color=color,
                alpha=0.75,
                edgecolor=color,
                linewidth=0.8,
            )

            # CI as asymmetric error bars derived from (lo, hi)
            fin_ci = fin & np.isfinite(lo) & np.isfinite(hi)
            if fin_ci.any():
                yerr = np.vstack([y[fin_ci] - lo[fin_ci], hi[fin_ci] - y[fin_ci]])
                ax.errorbar(
                    x[fin_ci],
                    y[fin_ci],
                    yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=2.5,
                    alpha=0.9,
                )

            # Labels in the same “sentence-case” convention
            ax.set_xlabel(
                maybe_sentence_case("distance from reward center [mm] (binned)")
            )
            ax.set_ylabel(
                maybe_sentence_case("mean COM dist. to circle center per fly [mm]")
            )

            # x-lims align to bin span (centers are inside, but this keeps it tidy)
            if edges.size >= 2 and np.all(np.isfinite(edges[[0, -1]])):
                ax.set_xlim(float(edges[0]), float(edges[-1]))

            # y-lims: 0-based; dynamic top unless user fixed it
            ax.set_ylim(bottom=0)
            if self.cfg.ymax is not None:
                ax.set_ylim(top=float(self.cfg.ymax))
            else:
                y_top = np.nanmax(hi) if np.isfinite(np.nanmax(hi)) else np.nanmax(y)
                if np.isfinite(y_top):
                    ax.set_ylim(top=float(y_top) * 1.10)

            # n labels per bin (small, lightly offset)
            ylim0, ylim1 = ax.get_ylim()
            y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
            for xi, yi, ni in zip(x, y, n_units):
                if np.isfinite(xi) and np.isfinite(yi) and int(ni) > 0:
                    util.pltText(
                        xi,
                        yi + y_off,
                        f"{int(ni)}",
                        ha="center",
                        size=self.customizer.in_plot_font_size,
                        color=".2",
                    )

            # Title + small annotation (closer to bundle plotter style)
            ax.set_title(
                maybe_sentence_case(
                    "between-reward COM magnitude vs distance-from-reward"
                )
            )
            parts = [f"T{int(self.cfg.training_index) + 1}"]
            if int(self.cfg.skip_first_sync_buckets) > 0:
                parts.append(
                    f"skip first {int(self.cfg.skip_first_sync_buckets)} bucket(s)"
                )
            parts.append(f"binned by {str(self.cfg.cond_stat)} dist")
            if self.cfg.subset_label:
                parts.append(str(self.cfg.subset_label))
            fig.text(
                0.02,
                0.98,
                " | ".join(parts),
                ha="left",
                va="top",
                fontsize=self.customizer.in_plot_font_size,
                color="0",
            )
        if self.customizer.font_size_customized:
            self.customizer.adjust_padding_proportionally()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")


def plot_btw_rwd_conditioned_com_overlay(
    *,
    results: Sequence[BetweenRewardConditionedCOMResult],
    labels: Sequence[str],
    out_file: str,
    opts,
    customizer: PlotCustomizer,
    log_tag: str = "btw_rwd_dist_binned_com",
) -> None:
    """
    Plot multiple cached BetweenRewardConditionedCOMResult objects as grouped bars.

    For each distance bin, draw one bar per group side-by-side (plus CI).
    Intended for fast 'Stage B' plotting from NPZ caches.
    """
    if not results:
        raise ValueError("No results provided")

    # Use first result's x-axis as reference
    ref = results[0]
    ref.validate()
    x = np.asarray(ref.x_centers, dtype=float)
    edges = np.asarray(ref.x_edges, dtype=float)
    widths = edges[1:] - edges[:-1]
    B = int(x.size)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

    # --- grouped bars geometry ---
    pairs = list(zip(results, labels))
    n_groups = len(pairs)
    if n_groups == 0:
        raise ValueError("No results provided")
    # Keep bars within each bin: total footprint is frac * bin_width
    frac = 0.86
    bar_w = frac * widths / max(1, n_groups)

    # ---------- t-test config / parsing ----------
    # Map labels to indices
    label_to_idx = {str(lab): i for i, (_, lab) in enumerate(pairs)}

    # Parse requested pairs for independent t-tests to show on plot
    pair_specs = getattr(opts, "btw_rwd_conditioned_com_ttest_ind", None) or []
    pairs_req: list[tuple[str, str]] = []
    for spec in pair_specs:
        spec = str(spec)
        if ":" not in spec:
            print(f"[{log_tag}] WARNING: bad ttest spec {spec!r} (expected A:B)")
            continue
        a, b = [s.strip() for s in spec.split(":", 1)]
        if a in label_to_idx and b in label_to_idx and a != b:
            pairs_req.append((a, b))
        else:
            print(f"[{log_tag}] WARNING: unknown/invalid ttest labels in {spec!r}")

    def _maybe_correct_pvals(pvals: np.ndarray) -> np.ndarray:
        """
        Apply a correction across bins for a single pair.
        """
        corr = str(getattr(opts, "btw_rwd_conditioned_com_ttest_correct", "none"))
        p = np.asarray(pvals, dtype=float).copy()
        fin = np.isfinite(p)
        m = int(fin.sum())
        if m <= 0 or corr == "none":
            return p
        if corr == "bonferroni":
            p[fin] = np.minimum(1.0, p[fin] * float(m))
            return p
        if corr == "fdr_bh":
            # Benjamini-Hochberg (simple implementation)
            idx = np.where(fin)[0]
            pv = p[idx]
            order = np.argsort(pv)
            pv_sorted = pv[order]
            q = pv_sorted * float(m) / (np.arange(1, m + 1))
            # enforce monotonicity
            q = np.minimum.accumulate(q[::-1])[::-1]
            out = p.copy()
            out[idx[order]] = np.minimum(1.0, q)
            return out
        return p

    def _draw_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.0, clip_on=False)
        ax.text(
            (x1 + x2) / 2.0,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=customizer.in_plot_font_size,
        )

    # y-offset for n-labels are computed after y-lims are known.
    pending_nlabels: list[tuple[int, int, float, float, int]] = []  # (gi, j, x, y_top, n)

    any_data = False
    if len(labels) != len(results):
        print(
            f"[{log_tag}] WARNING: labels/results length mismatch; truncating to min."
        )
    drawn = np.zeros((n_groups, B), dtype=bool)
    occupied_top = np.full((n_groups, B), -np.inf, dtype=float)
    for gi, (res, lab) in enumerate(pairs):
        res.validate()

        x2 = np.asarray(res.x_centers, dtype=float)
        edges2 = np.asarray(res.x_edges, dtype=float)
        if x2.shape != x.shape or not np.allclose(x2, x, equal_nan=True):
            print(
                f"[{log_tag}] WARNING: x_centers mismatch for {lab!r}; grouped bars may be misaligned."
            )
        if edges2.shape != edges.shape or not np.allclose(
            edges2, edges, equal_nan=True
        ):
            raise ValueError(
                f"[{log_tag}] x_edges mismatch for {lab!r}; cannot group bars safely."
            )

        y = np.asarray(res.mean, dtype=float)
        lo = np.asarray(res.ci_lo, dtype=float)
        hi = np.asarray(res.ci_hi, dtype=float)
        n_units = np.asarray(res.n_units, dtype=int)

        # Per-group bar centers: shift within each bin
        # Example: for 3 groups, offsets are [-1, 0, +1] * bar_w, centered
        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        x_bar = x + offset

        fin = np.isfinite(x_bar) & np.isfinite(y) & np.isfinite(widths)
        if not fin.any():
            continue
        any_data = True

        drawn[gi, fin] = True

        ax.bar(
            x_bar[fin],
            y[fin],
            width=bar_w[fin],
            align="center",
            alpha=0.75,
            linewidth=0.8,
            label=str(lab),
        )

        # Queue n-labels per bar (place after we know y-lims)
        # Store tuples: (x_pos, y_val, n)
        idxs = np.where(fin)[0]
        for j in idxs:
            xb = float(x_bar[j])
            yb = float(y[j])
            nb = int(n_units[j])
            if nb <= 0 or not (np.isfinite(xb) and np.isfinite(yb)):
                continue
            hib = float(hi[j]) if np.isfinite(hi[j]) else yb
            pending_nlabels.append((gi, int(j), xb, hib, nb))

        fin_ci = fin & np.isfinite(lo) & np.isfinite(hi)
        if fin_ci.any():
            yerr = np.vstack([y[fin_ci] - lo[fin_ci], hi[fin_ci] - y[fin_ci]])
            ax.errorbar(
                x_bar[fin_ci],
                y[fin_ci],
                yerr=yerr,
                fmt="none",
                elinewidth=1.1,
                capsize=2.0,
                alpha=0.9,
            )
    if not any_data:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        ax.set_xlabel(maybe_sentence_case("distance from reward circle [mm] (binned)"))
        ax.set_ylabel(
            maybe_sentence_case("mean COM dist. to circle center per fly [mm]")
        )

        # Match "bin span" x-lims like in base plot
        if edges.size >= 2 and np.all(np.isfinite(edges[[0, -1]])):
            ax.set_xlim(float(edges[0]), float(edges[-1]))

        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=customizer.in_plot_font_size)
        ax.set_title(
            maybe_sentence_case("between-reward COM magnitude vs distance-from-reward")
        )

        # Optional fixed ymax
        ymax = getattr(opts, "btw_rwd_conditioned_com_ymax", None)
        if ymax is not None:
            try:
                ax.set_ylim(top=float(ymax))
            except Exception:
                pass

        # Place queued n-labels
        if pending_nlabels:
            ylim0, ylim1 = ax.get_ylim()
            y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
            for gi, j, xb, y_top, nb in pending_nlabels:
                y_lab = float(y_top + y_off)
                util.pltText(
                    xb,
                    y_lab,
                    f"{nb}",
                    ha="center",
                    size=customizer.in_plot_font_size,
                    color=".2",
                )
                occupied_top[gi, j] = max(occupied_top[gi, j], y_lab)

        # ---------- t-tests + bracket+stars annotations ----------
        # Only attempt if requested.
        if pairs_req:
            min_n = int(getattr(opts, "btw_rwd_conditioned_com_ttest_min_n", 2))

            # Place brackets above the larger CI bound of the two compared bars.
            # Track the highest annotation so we can expand y-lims if needed.
            ylim0, ylim1 = ax.get_ylim()
            y_span = (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
            h = 0.02 * y_span
            pad = 0.03 * y_span
            max_annot_y = float("-inf")

            # Precompute per-group x-offset per bin for quick lookup
            # offset_i is a vector (B,) because bar_w is vector (B,)
            offsets = []
            for gi in range(n_groups):
                offsets.append((gi - (n_groups - 1) / 2.0) * bar_w)

            for a, b in pairs_req:
                ia, ib = label_to_idx[a], label_to_idx[b]
                ra, rb = pairs[ia][0], pairs[ib][0]
                if ra.per_unit is None or rb.per_unit is None:
                    print(
                        f"[{log_tag}] WARNING: missing per_unit in cached NPZ for {a!r} or {b!r}; cannot t-test."
                    )
                    continue

                A = np.asarray(ra.per_unit, dtype=float)
                Bv = np.asarray(rb.per_unit, dtype=float)
                if A.ndim != 2 or Bv.ndim != 2 or A.shape[1] != B or Bv.shape[1] != B:
                    print(
                        f"[{log_tag}] WARNING: per_unit shape mismatch for {a!r} or {b!r}; cannot t-test."
                    )
                    continue

                pvals = np.full((B,), np.nan, dtype=float)
                for j in range(B):
                    _, p, na, nb, _ = ttest_ind(
                        A[:, j], Bv[:, j], min_n=min_n, silent=True
                    )
                    pvals[j] = p
                p_use = _maybe_correct_pvals(pvals)

                # draw per-bin brackets if we have finite y and p
                for j in range(B):
                    if not (drawn[ia, j] and drawn[ib, j]):
                        continue
                    if not np.isfinite(p_use[j]):
                        continue

                    stars = util.p2stars(p_use[j], nanR=None)
                    if stars is None:
                        continue
                    if stars.startswith("ns"):
                        continue  # only annotate significant by default

                    # bar x-positions in this bin
                    xa = float(x[j] + offsets[ia][j])
                    xb = float(x[j] + offsets[ib][j])

                    # y baseline: above the higher CI bound (or mean if CI missing)
                    ya = (
                        float(ra.ci_hi[j])
                        if np.isfinite(ra.ci_hi[j])
                        else float(ra.mean[j])
                    )
                    yb = (
                        float(rb.ci_hi[j])
                        if np.isfinite(rb.ci_hi[j])
                        else float(rb.mean[j])
                    )
                    # y0 = np.nanmax([ya, yb])
                    y_occ_a = occupied_top[ia, j]
                    y_occ_b = occupied_top[ib, j]
                    y0 = np.nanmax([ya, yb, y_occ_a, y_occ_b])

                    y_br = float(y0 + pad)
                    _draw_bracket(ax, xa, xb, y_br, h, stars)
                    max_annot_y = max(max_annot_y, y_br + h)

            # expand ylim if needed
            if np.isfinite(max_annot_y):
                ylim0, ylim1 = ax.get_ylim()
                if max_annot_y > ylim1:
                    ax.set_ylim(top=max_annot_y * 1.05)

    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally()
    fig.tight_layout(rect=(0, 0, 1, 1))
    writeImage(out_file, format=opts.imageFormat)
    plt.close(fig)
    print(f"[{log_tag}] wrote {out_file}")
