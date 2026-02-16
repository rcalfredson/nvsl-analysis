from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils.common import maybe_sentence_case, writeImage


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

    # ---------------------------- utility ----------------------------
    def _video_base(self, va: "VideoAnalysis") -> str:
        fn = getattr(va, "fn", None)
        if fn:
            try:
                return os.path.splitext(os.path.basename(str(fn)))[0]
            except Exception:
                pass
        return f"va_{id(va)}"

    def _fly_role_name(self, role_idx: int) -> str:
        return "exp" if int(role_idx) == 0 else "yok"

    # ---------------------------- behavior masking ----------------------------

    def _build_nonwalk_mask(self, va, trx_idx, fi, n_frames):
        exclude_nonwalk = bool(
            getattr(
                self.opts, "btw_rwd_conditioned_com_exclude_nonwalking_frames", False
            )
        )

        nonwalk_mask = None
        if exclude_nonwalk:
            traj = va.trx[trx_idx]
            walking = getattr(traj, "walking", None)
            if walking is None:
                # No walking info; safest is to behave like "can't apply mask"
                # (either warn once, or just treat as disabled for this fly).
                nonwalk_mask = None
            else:
                # window is [fi, fi+n_frames)
                s0 = max(0, min(int(fi), len(walking)))
                e0 = max(0, min(int(fi + n_frames), len(walking)))
                # If window exceeds array, pad missing with "exclude" to be conservative.
                wwin = np.zeros((n_frames,), dtype=bool)
                if e0 > s0:
                    wseg = np.asarray(walking[s0:e0], dtype=float)
                    wseg = np.where(np.isfinite(wseg), wseg, 0.0)
                    wwin[: len(wseg)] = wseg > 0
                nonwalk_mask = ~wwin
        return nonwalk_mask

    # ---------------------------- bucket windowing ----------------------------

    def _sync_bucket_window(
        self,
        va: "VideoAnalysis",
        trn,
        *,
        t_idx: int,
        role_idx: int,
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
            # reward_exclusion_mask indexed by [t.n-1][role_idx][b_idx]
            try:
                mask = va.reward_exclusion_mask[trn.n - 1][role_idx]
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

    def _collect_segments_by_bin(self) -> tuple[np.ndarray, list[list[dict]]]:
        """
        Returns (edges, segs_by_bin) where segs_by_bin[j] is a list of dict rows, each describing
        one segment that landed in distance bin j, included the fly-unit identity fields:
            video_id, fly_idx, role_idx, fly_role
        plus segment fields:
            s, e, mag_mm, cond_x_mm, med_d_mm, max_d_mm
        """
        edges = self._x_edges()
        B = int(max(1, edges.size - 1))

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        exclude_nonwalk = bool(
            getattr(
                self.opts, "btw_rwd_conditioned_com_exclude_nonwalking_frames", False
            )
        )
        min_walk_frames = int(
            getattr(self.opts, "btw_rwd_conditioned_com_min_walk_frames", 2) or 2
        )

        segs_by_bin: list[list[dict]] = [[] for _ in range(B)]

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

            video_id = self._video_base(va)

            cell_idx = int(getattr(va, "f", -1))

            for role_idx, trx_idx in enumerate(va.flies):
                if not va.noyc and role_idx != 0:
                    continue

                fly_role = self._fly_role_name(role_idx)

                fi, df, n_buckets, complete = self._sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    role_idx=role_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = build_wall_contact_mask_for_window(
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=exclude_wall,
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )

                nonwalk_mask = self._build_nonwalk_mask(va, trx_idx, fi, n_frames)

                cs = str(self.cfg.cond_stat or "median").lower().strip()
                if cs in ("max", "maxdist", "max_d"):
                    dist_stats = ("median", "max")
                else:
                    dist_stats = ("median",)

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    trx_idx,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=min_med_mm,
                    exclude_wall=exclude_wall,
                    exclude_nonwalk=exclude_nonwalk,
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=min_walk_frames,
                    wc=wc,
                    exclude_reward_endpoints=bool(
                        getattr(
                            self.opts, "btw_rwd_com_exclude_reward_endpoints", False
                        )
                    ),
                    dist_stats=dist_stats,
                    debug=False,
                    yield_skips=False,
                ):
                    x = self._cond_value_for_segment(seg)
                    y = float(getattr(seg, "mag_mm", np.nan))
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue

                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue

                    row = dict(
                        video_id=str(video_id),
                        fly_idx=cell_idx,
                        role_idx=int(role_idx),
                        fly_role=str(fly_role),
                        training_index=int(self.cfg.training_index),
                        bin_lo_mm=float(edges[j]),
                        bin_hi_mm=float(edges[j + 1]),
                        cond_x_mm=float(x),
                        mag_mm=float(y),
                        med_d_mm=float(getattr(seg, "med_d_mm", np.nan)),
                        max_d_mm=float(getattr(seg, "max_d_mm", np.nan)),
                        s=int(getattr(seg, "s", -1)),
                        e=int(getattr(seg, "e", -1)),
                    )
                    segs_by_bin[j].append(row)
        return edges, segs_by_bin

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

            for role_idx, trx_idx in enumerate(va.flies):
                # experimental-only in exp+yoked
                if not va.noyc and role_idx != 0:
                    continue

                fi, df, n_buckets, complete = self._sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    role_idx=role_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = build_wall_contact_mask_for_window(
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=exclude_wall,
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )

                nonwalk_mask = self._build_nonwalk_mask(va, trx_idx, fi, n_frames)

                exclude_nonwalk = bool(
                    getattr(
                        self.opts,
                        "btw_rwd_conditioned_com_exclude_nonwalking_frames",
                        False,
                    )
                )
                min_walk_frames = int(
                    getattr(self.opts, "btw_rwd_conditioned_com_min_walk_frames", 2)
                    or 2
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
                    trx_idx,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=min_med_mm,
                    exclude_wall=exclude_wall,
                    wc=wc,
                    exclude_reward_endpoints=bool(
                        getattr(
                            self.opts, "btw_rwd_com_exclude_reward_endpoints", False
                        )
                    ),
                    exclude_nonwalk=exclude_nonwalk,
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=min_walk_frames,
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

    def _collect_per_fly_binned_walk_fracs(self) -> list[np.ndarray]:
        """
        For each fly unit, compute a (B,) vector where each entry is:

            mean(walking_fraction) over segments whose cond_stat falls into x-bin j,

        where walking_fraction for a segment is mean(traj.walking[s:e]).
        NaN for bins with no segments.
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

            for role_idx, trx_idx in enumerate(va.flies):
                if not va.noyc and role_idx != 0:
                    continue
                fi, df, n_buckets, complete = self._sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    role_idx=role_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = build_wall_contact_mask_for_window(
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=exclude_wall,
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )

                traj = va.trx[trx_idx]
                walking = getattr(traj, "walking", None)
                if walking is None:
                    # No walking array; skip this fly for this debug export.
                    continue

                bin_vals: list[list[float]] = [[] for _ in range(B)]

                cs = str(self.cfg.cond_stat or "median").lower().strip()
                if cs in ("max", "maxdist", "max_d"):
                    dist_stats = ("median", "max")
                else:
                    dist_stats = ("median",)

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    trx_idx,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=min_med_mm,
                    exclude_wall=exclude_wall,
                    wc=wc,
                    exclude_reward_endpoints=bool(
                        getattr(
                            self.opts, "btw_rwd_com_exclude_reward_endpoints", False
                        )
                    ),
                    dist_stats=dist_stats,
                    debug=False,
                    yield_skips=False,
                ):
                    x = self._cond_value_for_segment(seg)
                    if not np.isfinite(x):
                        continue

                    s = int(getattr(seg, "s", -1))
                    e = int(getattr(seg, "e", -1))
                    if e <= s + 1:
                        continue

                    # Clamp to walking array bounds
                    s2 = max(0, min(s, len(walking)))
                    e2 = max(0, min(e, len(walking)))
                    if e2 <= s2:
                        continue

                    wseg = np.asarray(walking[s2:e2], dtype=float)
                    wseg = wseg[np.isfinite(wseg)]
                    if wseg.size == 0:
                        continue

                    walk_frac = float(
                        np.mean(wseg > 0)
                    )  # robust if walking is bool or 0/1

                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue
                    bin_vals[j].append(walk_frac)

                vec = np.full((B,), np.nan, dtype=float)
                for j in range(B):
                    if not bin_vals[j]:
                        continue
                    vv = np.asarray(bin_vals[j], dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size == 0:
                        continue
                    vec[j] = float(np.nanmean(vv))

                if np.any(np.isfinite(vec)):
                    per_fly_vectors.append(vec)

        return per_fly_vectors

    def _write_walk_tsv(self, path: str) -> None:
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        per_fly = self._collect_per_fly_binned_walk_fracs()
        if not per_fly:
            print(f"[{self.log_tag}] walk TSV: no data (no flies with walking[])")
            return

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

        util.ensureDir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "bin_lo_mm",
                        "bin_hi_mm",
                        "x_center_mm",
                        "mean_walk_frac",
                        "ci_lo",
                        "ci_hi",
                        "n_units",
                        "training_index",
                        "cond_stat",
                        "skip_first_sync_buckets",
                        "exclude_wall_contact",
                        "min_meddist_mm",
                    ]
                )
                + "\n"
            )
            for j in range(B):
                f.write(
                    "\t".join(
                        map(
                            str,
                            [
                                float(edges[j]),
                                float(edges[j + 1]),
                                float(centers[j]),
                                float(mean[j]) if np.isfinite(mean[j]) else "nan",
                                float(lo[j]) if np.isfinite(lo[j]) else "nan",
                                float(hi[j]) if np.isfinite(hi[j]) else "nan",
                                int(n_units[j]),
                                int(self.cfg.training_index),
                                str(self.cfg.cond_stat),
                                int(self.cfg.skip_first_sync_buckets),
                                int(
                                    bool(
                                        getattr(
                                            self.opts, "com_exclude_wall_contact", False
                                        )
                                    )
                                ),
                                float(
                                    getattr(
                                        self.opts, "com_per_segment_min_meddist_mm", 0.0
                                    )
                                    or 0.0
                                ),
                            ],
                        )
                    )
                    + "\n"
                )
        print(f"[{self.log_tag}] wrote walk TSV: {path}")

    def _write_top_fly_units_tsv(self, path: str, *, k: int = 5) -> None:
        edges, segs_by_bin = self._collect_segments_by_bin()
        B = int(max(1, edges.size - 1))
        k = int(max(1, k))

        util.ensureDir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "bin_lo_mm",
                        "bin_hi_mm",
                        "x_center_mm",
                        "rank_in_bin",
                        "mean_com_mm_per_fly",
                        "n_segments_in_bin",
                        "video_id",
                        "fly_idx",
                        "role_idx",
                        "fly_role",
                        "training_index",
                        "cond_stat",
                        "skip_first_sync_buckets",
                        "exclude_wall_contact",
                        "min_meddist_mm",
                    ]
                )
                + "\n"
            )

            for j in range(B):
                rows = segs_by_bin[j]
                if not rows:
                    continue

                # group by fly-unit identity
                groups: dict[tuple[str, int, int], list[float]] = {}
                for r in rows:
                    key = (str(r["video_id"]), int(r["fly_idx"]), int(r["role_idx"]))
                    groups.setdefault(key, []).append(float(r["mag_mm"]))

                scored: list[tuple[float, int, str, int, int, str]] = []
                for (vid, fly_idx, role_idx), mags in groups.items():
                    mags = [m for m in mags if np.isfinite(m)]
                    if not mags:
                        continue
                    mean_mag = float(np.mean(mags))
                    scored.append(
                        (
                            mean_mag,
                            int(len(mags)),
                            str(vid),
                            int(fly_idx),
                            int(role_idx),
                            "exp" if int(role_idx) == 0 else "yok",
                        )
                    )
                if not scored:
                    continue

                scored.sort(key=lambda t: t[0], reverse=True)
                top = scored[:k]
                x_center = float(0.5 * (edges[j] + edges[j + 1]))

                for rank, (
                    mean_mag,
                    n_seg,
                    vid,
                    fly_idx,
                    role_idx,
                    fly_role,
                ) in enumerate(top, start=1):
                    f.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    float(edges[j]),
                                    float(edges[j + 1]),
                                    x_center,
                                    int(rank),
                                    float(mean_mag),
                                    int(n_seg),
                                    str(vid),
                                    int(fly_idx),
                                    int(role_idx),
                                    str(fly_role),
                                    int(self.cfg.training_index),
                                    str(self.cfg.cond_stat),
                                    int(self.cfg.skip_first_sync_buckets),
                                    int(
                                        bool(
                                            getattr(
                                                self.opts,
                                                "com_exclude_wall_contact",
                                                False,
                                            )
                                        )
                                    ),
                                    float(
                                        getattr(
                                            self.opts,
                                            "com_per_segment_min_meddist_mm",
                                            0.0,
                                        )
                                        or 0.0
                                    ),
                                ],
                            )
                        )
                        + "\n"
                    )
        print(f"[{self.log_tag}] wrote top fly-units TSV: {path}")

    def _write_top_segments_tsv(self, path: str, *, k: int = 25) -> None:
        edges, segs_by_bin = self._collect_segments_by_bin()
        B = int(max(1, edges.size - 1))
        k = int(max(1, k))

        util.ensureDir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "bin_lo_mm",
                        "bin_hi_mm",
                        "x_center_mm",
                        "rank_in_bin",
                        "mag_mm",
                        "cond_x_mm",
                        "med_d_mm",
                        "max_d_mm",
                        "s",
                        "e",
                        "video_id",
                        "fly_idx",
                        "role_idx",
                        "fly_role",
                        "training_index",
                        "cond_stat",
                        "skip_first_sync_buckets",
                        "exclude_wall_contact",
                        "min_meddist_mm",
                    ]
                )
                + "\n"
            )

            for j in range(B):
                rows = segs_by_bin[j]
                if not rows:
                    continue

                rows2 = []
                for r in rows:
                    m = float(r.get("mag_mm", np.nan))
                    if np.isfinite(m):
                        rows2.append(r)
                if not rows2:
                    continue

                rows2.sort(key=lambda r: float(r["mag_mm"]), reverse=True)
                top = rows2[:k]
                x_center = float(0.5 * (edges[j] + edges[j + 1]))

                for rank, r in enumerate(top, start=1):
                    med = float(r.get("med_d_mm", np.nan))
                    med_out = med if np.isfinite(med) else "nan"
                    max_d = float(r.get("max_d_mm", np.nan))
                    max_out = max_d if np.isfinite(max_d) else "nan"
                    f.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    float(edges[j]),
                                    float(edges[j + 1]),
                                    x_center,
                                    int(rank),
                                    float(r["mag_mm"]),
                                    float(r["cond_x_mm"]),
                                    med_out,
                                    max_out,
                                    int(r["s"]),
                                    int(r["e"]),
                                    str(r["video_id"]),
                                    int(r["fly_idx"]),
                                    int(r["role_idx"]),
                                    str(r["fly_role"]),
                                    int(r["training_index"]),
                                    str(self.cfg.cond_stat),
                                    int(self.cfg.skip_first_sync_buckets),
                                    int(
                                        bool(
                                            getattr(
                                                self.opts,
                                                "com_exclude_wall_contact",
                                                False,
                                            )
                                        )
                                    ),
                                    float(
                                        getattr(
                                            self.opts,
                                            "com_per_segment_min_meddist_mm",
                                            0.0,
                                        )
                                        or 0.0
                                    ),
                                ],
                            )
                        )
                        + "\n"
                    )
        print(f"[{self.log_tag}] wrote top segments TSV: {path}")

    def _compute_from_per_fly(self) -> tuple[dict, np.ndarray | None]:
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        per_fly = self._collect_per_fly_binned_means()
        if not per_fly:
            summary = {
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
            return summary, None

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

        summary = {
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
        return summary, Y

    def compute_summary(self) -> dict:
        """
        Returns dict with:
            - x_edges, x_centers
            - mean, ci_lo, ci_hi, n_units
            - meta
        """
        d, _Y = self._compute_from_per_fly()
        return d

    def compute_result(self) -> BetweenRewardConditionedCOMResult:
        """
        Return a portable result object suitable for caching/export.
        """
        d, Y = self._compute_from_per_fly()

        return BetweenRewardConditionedCOMResult(
            x_edges=np.asarray(d["x_edges"], dtype=float),
            x_centers=np.asarray(d["x_centers"], dtype=float),
            mean=np.asarray(d["mean"], dtype=float),
            ci_lo=np.asarray(d["ci_lo"], dtype=float),
            ci_hi=np.asarray(d["ci_hi"], dtype=float),
            n_units=np.asarray(d["n_units"], dtype=int),
            meta=dict(d.get("meta", {})),
            per_unit=(np.asarray(Y, dtype=float) if Y is not None else None),
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
                maybe_sentence_case("Between-reward COM vs distance from reward")
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

        walk_path = getattr(self.opts, "btw_rwd_conditioned_com_walk_stats_out", None)
        if walk_path:
            try:
                self._write_walk_tsv(str(walk_path))
            except Exception as e:
                print(f"[{self.log_tag}] WARNING: failed to write walk TSV: {e}")

        # --- optional debug exports: who/what is driving large COM bins? ---
        top_fly_path = getattr(
            self.opts, "btw_rwd_conditioned_com_top_fly_units_out", None
        )
        if top_fly_path:
            k = int(
                getattr(self.opts, "btw_rwd_conditioned_com_top_fly_units_k", 5) or 5
            )
            try:
                self._write_top_fly_units_tsv(str(top_fly_path), k=k)
            except Exception as e:
                print(
                    f"[{self.log_tag}] WARNING: failed to write top-fly-units TSV: {e}"
                )

        top_seg_path = getattr(self.opts, "btw_rwd_conditioned_com_top_segs_out", None)
        if top_seg_path:
            k = int(getattr(self.opts, "btw_rwd_conditioned_com_top_segs_k", 25) or 25)
            try:
                self._write_top_segments_tsv(str(top_seg_path), k=k)
            except Exception as e:
                print(
                    f"[{self.log_tag}] WARNING: failed to write top-segments TSV: {e}"
                )


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

    # --- layout: optionally increase spacing between bins for overlays ---
    # 1.0 = unchanged. >1 spreads bins apart (more whitespace).
    bin_spacing = float(
        getattr(opts, "btw_rwd_conditioned_com_overlay_bin_spacing", 1.12) or 1.12
    )
    if not np.isfinite(bin_spacing) or bin_spacing <= 0:
        bin_spacing = 1.0

    origin = float(x[0]) if x.size else 0.0
    x_plot = origin + (x - origin) * bin_spacing

    # --- grouped bars geometry ---
    pairs = list(zip(results, labels))
    n_groups = len(pairs)
    if n_groups == 0:
        raise ValueError("No results provided")

    base_w, base_h = 6.8, 4.2
    extra_w = 0.0
    if n_groups >= 3:
        extra_w = 1.0 + 0.7 * (n_groups - 3)

    # How much did we stretch the x-span?
    x0 = float(edges[0])
    x1 = float(edges[-1])
    x0_layout = origin + (x0 - origin) * bin_spacing
    x1_layout = origin + (x1 - origin) * bin_spacing

    span_raw = max(1e-9, x1 - x0)
    span_layout = max(1e-9, x1_layout - x0_layout)
    span_scale = span_layout / span_raw  # ~ bin_spacing (when origin ~ x0)

    # Scale the figure width so spacing doesn't get "crammed" into same pixels.
    # Optional clamp to avoid absurdly huge figures if someone sets bin_spacing=10.
    max_scale = float(
        getattr(opts, "btw_rwd_conditioned_com_overlay_max_width_scale", 2.5) or 2.5
    )
    span_scale = min(span_scale, max_scale)

    fig, ax = plt.subplots(1, 1, figsize=((base_w + extra_w) * span_scale, base_h))

    # Keep bars within each bin: total footprint is frac * bin_width
    frac = 0.86
    bar_w = frac * widths / max(1, n_groups)

    # y-offset for n-labels are computed after y-lims are known.
    pending_nlabels: list[tuple[int, int, float, float, int]] = (
        []
    )  # (gi, j, x, y_top, n)
    xpos_by_group: list[np.ndarray] = []
    active = []

    any_data = False
    if len(labels) != len(results):
        print(
            f"[{log_tag}] WARNING: labels/results length mismatch; truncating to min."
        )
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
        x_bar = x_plot + offset
        xpos_by_group.append(np.asarray(x_bar, float))

        fin = np.isfinite(x_bar) & np.isfinite(y) & np.isfinite(widths)
        if not fin.any():
            continue
        any_data = True
        active.append((gi, res, lab, x_bar, fin))

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
        # --- pick x-label from cached result meta (median vs max) ---
        dist_stat = ""
        try:
            dist_stat = (
                str(getattr(ref, "meta", {}).get("dist_stat", "")).strip().lower()
            )
        except Exception:
            dist_stat = ""

        if dist_stat in ("max", "maxdist", "max_d"):
            xlab = "Max distance to reward (mm)"
        elif dist_stat in ("median", "med", "meddist"):
            xlab = "Median distance to reward (mm)"
        else:
            # fallback if meta missing/unknown
            xlab = "Distance to reward (mm)"

        ax.set_xlabel(maybe_sentence_case(xlab))
        ax.set_ylabel(maybe_sentence_case("COM distance to reward (mm)"))

        # Match "bin span" x-lims like in base plot, but allow override via opts
        x0 = float(edges[0])
        x1 = float(edges[-1])

        xmax_opt = getattr(opts, "btw_rwd_conditioned_com_xmax", None)
        if xmax_opt is not None:
            try:
                xmax = float(xmax_opt)
                if np.isfinite(xmax) and xmax > x0:
                    # Clamp to not exceed cached bin span
                    x1 = min(x1, xmax)
            except Exception:
                pass

        x0_layout = origin + (x0 - origin) * bin_spacing
        x1_layout = origin + (x1 - origin) * bin_spacing
        ax.set_xlim(x0_layout, x1_layout)

        # --- lock x ticks to bins (so bin_spacing never changes tick labels) ---
        # One tick per bin, label = range (e.g., "0–2", "2–4", ...)
        tick_pos = np.asarray(x_plot, dtype=float)

        # If --btw_rwd_conditioned_com_xmax trims the plot, drop ticks beyond it
        # so you don't get labels for bins that are off-screen.
        keep = np.ones_like(tick_pos, dtype=bool)
        if "x1_layout" in locals():
            keep &= tick_pos <= float(x1_layout) + 1e-9

        tick_pos = tick_pos[keep]
        edges_show = edges[: (tick_pos.size + 1)]

        def _fmt_edge(v: float) -> str:
            # int-like -> no decimals; otherwise 1 decimal
            if np.isfinite(v) and abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.1f}"

        tick_labels = [
            f"{_fmt_edge(float(a))}–{_fmt_edge(float(b))}"
            for a, b in zip(edges_show[:-1], edges_show[1:])
        ]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            tick_labels,
            rotation=0,
            ha="center",
            fontsize=customizer.in_plot_font_size,
        )

        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=customizer.in_plot_font_size)
        if bool(getattr(opts, "btw_rwd_conditioned_com_overlay_title", False)):
            ax.set_title(
                maybe_sentence_case("between-reward COM vs distance-from-reward")
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

        do_stats = bool(getattr(opts, "btw_rwd_conditioned_com_stats", False))
        if do_stats and not any(r.per_unit is None for r in results):
            cfg_stats = StatAnnotConfig(
                alpha=float(
                    getattr(opts, "btw_rwd_conditioned_com_stats_alpha", 0.05) or 0.05
                ),
                nlabel_off_frac=0.04,
            )
            annotate_grouped_bars_per_bin(
                ax,
                x_centers=x_plot,
                xpos_by_group=[a[3] for a in active],
                per_unit_by_group=[np.asarray(a[1].per_unit, float) for a in active],  # type: ignore[arg-type]
                hi_by_group=[np.asarray(a[1].ci_hi, float) for a in active],
                group_names=[str(a[2]) for a in active],
                cfg=cfg_stats,
            )

    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally()
    fig.tight_layout()
    writeImage(out_file, format=opts.imageFormat)
    plt.close(fig)
    print(f"[{log_tag}] wrote {out_file}")
