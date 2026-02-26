from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.plotting.between_reward_segment_binning import (
    x_edges as make_x_edges,
    sync_bucket_window,
    build_nonwalk_mask,
    wall_contact_mask,
)
from src.plotting.between_reward_segment_metrics import dist_traveled_mm_masked
from src.plotting.grouped_bar_layout import grouped_bar_layout_from_edges
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.utils.common import maybe_sentence_case, writeImage


@dataclass
class BetweenRewardConditionedDistTravConfig:
    """
    Distance-binned distance-traveled analysis for between-reward segments.

    Binning:
        - x-axis: max distance from reward center during segment (mm)

    Metrics per segment:
        - total distance traveled over [s, e) frames (step indices [s, e-1))
        - tail distance traveled from max-distance frame to end (same step semantics)

    Aggregation:
        - per-fly mean within each x-bin
        - then mean + CI across flies per x-bin
    """

    out_file: str

    # Which training to analyze (0-based index).
    training_index: int = 1  # Training 2 default (0-based)

    skip_first_sync_buckets: int = 0
    use_reward_exclusion_mask: bool = False

    # Distance binning in mm
    x_bin_width_mm: float = 2.0
    x_min_mm: float = 0.0
    x_max_mm: float = 20.0

    # Plot options
    ci_conf: float = 0.95
    ymax: float | None = None
    subset_label: str | None = None


@dataclass(frozen=True)
class BetweenRewardConditionedDistTravResult:
    x_edges: np.ndarray
    x_centers: np.ndarray

    mean_total: np.ndarray
    ci_lo_total: np.ndarray
    ci_hi_total: np.ndarray

    mean_tail: np.ndarray
    ci_lo_tail: np.ndarray
    ci_hi_tail: np.ndarray

    n_units: np.ndarray
    meta: dict

    per_unit_total: np.ndarray | None = None  # (N_units, B)
    per_unit_tail: np.ndarray | None = None  # (N_units, B)
    per_unit_ids: np.ndarray | None = None  # (N_units,) stable IDs for pairing

    def validate(self) -> None:
        edges = np.asarray(self.x_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("x_edges must be 1D with >= 2 entries")
        B = int(edges.size - 1)

        def _check_1d(name: str) -> None:
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1 or arr.size != B:
                raise ValueError(f"{name} must be 1D with length {B}")

        for nm in (
            "x_centers",
            "mean_total",
            "ci_lo_total",
            "ci_hi_total",
            "mean_tail",
            "ci_lo_tail",
            "ci_hi_tail",
            "n_units",
        ):
            _check_1d(nm)

        if self.per_unit_total is not None:
            pu = np.asarray(self.per_unit_total, dtype=float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit_total must be 2D with shape (N, {B})")
        if self.per_unit_tail is not None:
            pu = np.asarray(self.per_unit_tail, dtype=float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit_tail must be 2D with shape (N, {B})")
        if self.per_unit_ids is not None:
            ids = np.asarray(self.per_unit_ids, dtype=object).ravel()
            # If per-unit arrays exist, ensure N matches
            if self.per_unit_total is not None:
                pu = np.asarray(self.per_unit_total, dtype=float)
                if ids.shape[0] != pu.shape[0]:
                    raise ValueError(
                        "per_unit_ids length must match per_unit_total rows"
                    )
            if self.per_unit_tail is not None:
                pu = np.asarray(self.per_unit_tail, dtype=float)
                if ids.shape[0] != pu.shape[0]:
                    raise ValueError(
                        "per_unit_ids length must match per_unit_tail rows"
                    )

    def save_npz(self, path: str) -> None:
        self.validate()
        kwargs = dict(
            x_edges=np.asarray(self.x_edges, dtype=float),
            x_centers=np.asarray(self.x_centers, dtype=float),
            mean_total=np.asarray(self.mean_total, dtype=float),
            ci_lo_total=np.asarray(self.ci_lo_total, dtype=float),
            ci_hi_total=np.asarray(self.ci_hi_total, dtype=float),
            mean_tail=np.asarray(self.mean_tail, dtype=float),
            ci_lo_tail=np.asarray(self.ci_lo_tail, dtype=float),
            ci_hi_tail=np.asarray(self.ci_hi_tail, dtype=float),
            n_units=np.asarray(self.n_units, dtype=int),
            meta=np.asarray([self.meta], dtype=object),
        )
        if self.per_unit_total is not None:
            kwargs["per_unit_total"] = np.asarray(self.per_unit_total, dtype=float)
        if self.per_unit_tail is not None:
            kwargs["per_unit_tail"] = np.asarray(self.per_unit_tail, dtype=float)
        if self.per_unit_ids is not None:
            kwargs["per_unit_ids"] = np.asarray(self.per_unit_ids, dtype=object)
        np.savez_compressed(path, **kwargs)

    @staticmethod
    def load_npz(path: str) -> "BetweenRewardConditionedDistTravResult":
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta_obj = z["meta"]
                meta = meta_obj.item() if hasattr(meta_obj, "item") else {}
            except Exception:
                meta = {}

        res = BetweenRewardConditionedDistTravResult(
            x_edges=np.asarray(z["x_edges"], dtype=float),
            x_centers=np.asarray(z["x_centers"], dtype=float),
            mean_total=np.asarray(z["mean_total"], dtype=float),
            ci_lo_total=np.asarray(z["ci_lo_total"], dtype=float),
            ci_hi_total=np.asarray(z["ci_hi_total"], dtype=float),
            mean_tail=np.asarray(z["mean_tail"], dtype=float),
            ci_lo_tail=np.asarray(z["ci_lo_tail"], dtype=float),
            ci_hi_tail=np.asarray(z["ci_hi_tail"], dtype=float),
            n_units=np.asarray(z["n_units"], dtype=int),
            meta=dict(meta) if isinstance(meta, dict) else {},
            per_unit_total=(
                np.asarray(z["per_unit_total"], dtype=float)
                if "per_unit_total" in z
                else None
            ),
            per_unit_tail=(
                np.asarray(z["per_unit_tail"], dtype=float)
                if "per_unit_tail" in z
                else None
            ),
            per_unit_ids=(
                np.asarray(z["per_unit_ids"], dtype=object)
                if "per_unit_ids" in z
                else None
            ),
        )
        res.validate()
        return res


class BetweenRewardConditionedDistTravPlotter:
    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardConditionedDistTravConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = "btw_rwd_dist_binned_disttrav"

    def _video_base(self, va: "VideoAnalysis") -> str:
        fn = getattr(va, "fn", None)
        if fn:
            try:
                return os.path.splitext(os.path.basename(str(fn)))[0]
            except Exception:
                pass
        return f"va_{id(va)}"

    def _fly_id(self, va: "VideoAnalysis", role_idx: int, trx_idx: int) -> int:
        """
        Return chamber/grid fly ID for this unit.

        Typically va.f is a scalar (one per exp-only video or per exp+yok pair).
        If va.f is sequence-like, we index by role_idx (fallback to trx_idx).
        """
        f = getattr(va, "f", None)
        if f is None:
            return -1

        # sequence-like case
        try:
            if isinstance(f, (list, tuple, np.ndarray)):
                if len(f) > role_idx:
                    return int(f[role_idx])
                if len(f) > trx_idx:
                    return int(f[trx_idx])
        except Exception:
            pass

        # scalar-ish fallback
        try:
            return int(f)
        except Exception:
            return -1

    def _fly_role(self, role_idx: int) -> str:
        return "exp" if int(role_idx) == 0 else "yok"

    def _write_sampled_segments_tsv(self, path: str) -> None:
        """
        Write per-bin top-K and random-K segment examples.
        Uses dt_total and dt_tail computed with the same masking as the metric.
        """
        import heapq

        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        top_k = int(
            getattr(self.opts, "btw_rwd_conditioned_disttrav_segs_top_k", 10) or 10
        )
        rand_k = int(
            getattr(self.opts, "btw_rwd_conditioned_disttrav_segs_rand_k", 10) or 10
        )
        seed = int(getattr(self.opts, "btw_rwd_conditioned_disttrav_segs_seed", 0) or 0)
        rng = np.random.default_rng(seed)

        # Heaps for top-K (min-heap): store (value, rowdict)
        top_total = [[] for _ in range(B)]
        top_tail = [[] for _ in range(B)]

        # Reservoir samples per bin: list[dict], and seen count
        rand = [[] for _ in range(B)]
        seen = np.zeros((B,), dtype=int)

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )

        exclude_nonwalk = bool(
            getattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
        )
        min_walk_frames = int(
            getattr(self.opts, "btw_rwd_conditioned_min_walk_frames", 2) or 2
        )

        t_idx = int(self.cfg.training_index)
        warned_missing_wc = [False]

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

            for role_idx, trx_idx in enumerate(va.flies):
                if not va.noyc and role_idx != 0:
                    continue

                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=trx_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = wall_contact_mask(
                    self.opts,
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    log_tag=self.log_tag,
                    warned_missing_wc=warned_missing_wc,
                )
                nonwalk_mask = build_nonwalk_mask(self.opts, va, trx_idx, fi, n_frames)

                traj = va.trx[trx_idx]
                dist_stats = ("median", "max")
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
                    exclude_nonwalk=exclude_nonwalk,
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=min_walk_frames,
                    dist_stats=dist_stats,
                    debug=False,
                    yield_skips=False,
                ):
                    x = float(getattr(seg, "max_d_mm", np.nan))
                    if not np.isfinite(x):
                        continue

                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue

                    s = int(getattr(seg, "s", -1))
                    e = int(getattr(seg, "e", -1))
                    if e <= s + 1:
                        continue

                    max_i = getattr(seg, "max_d_i", None)
                    if max_i is None:
                        continue
                    max_i = int(max_i)

                    px_per_mm = self._px_per_mm(va)
                    dt_total = dist_traveled_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        start_override=None,
                        min_keep_frames=min_walk_frames,
                    )
                    dt_tail = dist_traveled_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        start_override=max_i,
                        min_keep_frames=min_walk_frames,
                    )

                    if not (np.isfinite(dt_total) and np.isfinite(dt_tail)):
                        continue

                    fly_id = self._fly_id(va, role_idx=role_idx, trx_idx=trx_idx)

                    row = dict(
                        bin_lo_mm=float(edges[j]),
                        bin_hi_mm=float(edges[j + 1]),
                        x_center_mm=float(centers[j]),
                        dt_total_mm=float(dt_total),
                        dt_tail_mm=float(dt_tail),
                        max_d_mm=float(x),
                        s=int(s),
                        e=int(e),
                        max_d_i=int(max_i),
                        b_idx=int(getattr(seg, "b_idx", -1)),
                        video_id=str(video_id),
                        fly_id=int(fly_id),
                        trx_idx=int(trx_idx),
                        role_idx=int(role_idx),
                        fly_role=str(self._fly_role(role_idx)),
                        training_index=int(self.cfg.training_index),
                    )

                    # --- update top-K heaps
                    def _push_top(heap, value):
                        if not np.isfinite(value):
                            return
                        if len(heap) < top_k:
                            heapq.heappush(heap, (float(value), row))
                        else:
                            if float(value) > heap[0][0]:
                                heapq.heapreplace(heap, (float(value), row))

                    _push_top(top_total[j], dt_total)
                    _push_top(top_tail[j], dt_tail)

                    # --- reservoir sample
                    seen[j] += 1
                    if rand_k > 0:
                        if len(rand[j]) < rand_k:
                            rand[j].append(row)
                        else:
                            # replace with prob rand_k/seen
                            r = int(rng.integers(0, seen[j]))
                            if r < rand_k:
                                rand[j][r] = row
        # --- write TSV
        util.ensureDir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "sample_type",
                        "rank",
                        "bin_lo_mm",
                        "bin_hi_mm",
                        "x_center_mm",
                        "dt_total_mm",
                        "dt_tail_mm",
                        "max_d_mm",
                        "s",
                        "e",
                        "max_d_i",
                        "b_idx",
                        "video_id",
                        "fly_id",
                        "trx_idx",
                        "role_idx",
                        "fly_role",
                        "training_index",
                    ]
                )
                + "\n"
            )

            # top_total per bin
            for j in range(B):
                heap = sorted(top_total[j], key=lambda t: t[0], reverse=True)
                for rnk, (_val, row) in enumerate(heap, start=1):
                    f.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    "top_total",
                                    rnk,
                                    *[
                                        row["bin_lo_mm"],
                                        row["bin_hi_mm"],
                                        row["x_center_mm"],
                                        row["dt_total_mm"],
                                        row["dt_tail_mm"],
                                        row["max_d_mm"],
                                        row["s"],
                                        row["e"],
                                        row["max_d_i"],
                                        row["b_idx"],
                                        row["video_id"],
                                        row["fly_id"],
                                        row["trx_idx"],
                                        row["role_idx"],
                                        row["fly_role"],
                                        row["training_index"],
                                    ],
                                ],
                            )
                        )
                        + "\n"
                    )

            # top_tail per bin
            for j in range(B):
                heap = sorted(top_tail[j], key=lambda t: t[0], reverse=True)
                for rnk, (_val, row) in enumerate(heap, start=1):
                    f.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    "top_tail",
                                    rnk,
                                    *[
                                        row["bin_lo_mm"],
                                        row["bin_hi_mm"],
                                        row["x_center_mm"],
                                        row["dt_total_mm"],
                                        row["dt_tail_mm"],
                                        row["max_d_mm"],
                                        row["s"],
                                        row["e"],
                                        row["max_d_i"],
                                        row["b_idx"],
                                        row["video_id"],
                                        row["fly_id"],
                                        row["trx_idx"],
                                        row["role_idx"],
                                        row["fly_role"],
                                        row["training_index"],
                                    ],
                                ],
                            )
                        )
                        + "\n"
                    )

            # random per bin
            for j in range(B):
                for rnk, row in enumerate(rand[j], start=1):
                    f.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    "random",
                                    rnk,
                                    *[
                                        row["bin_lo_mm"],
                                        row["bin_hi_mm"],
                                        row["x_center_mm"],
                                        row["dt_total_mm"],
                                        row["dt_tail_mm"],
                                        row["max_d_mm"],
                                        row["s"],
                                        row["e"],
                                        row["max_d_i"],
                                        row["b_idx"],
                                        row["video_id"],
                                        row["fly_id"],
                                        row["trx_idx"],
                                        row["role_idx"],
                                        row["fly_role"],
                                        row["training_index"],
                                    ],
                                ],
                            )
                        )
                        + "\n"
                    )

        print(f"[{self.log_tag}] wrote segment-samples TSV: {path}")

    def _x_edges(self) -> np.ndarray:
        return make_x_edges(
            x_bin_width_mm=float(self.cfg.x_bin_width_mm),
            x_min_mm=float(self.cfg.x_min_mm),
            x_max_mm=float(self.cfg.x_max_mm),
        )

    def _px_per_mm(self, va: "VideoAnalysis") -> float | None:
        try:
            px_per_mm = float(va.xf.fctr * va.ct.pxPerMmFloor())
            if np.isfinite(px_per_mm) and px_per_mm > 0:
                return px_per_mm
        except Exception:
            pass
        return None

    def _collect_per_fly_binned_means(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        edges = self._x_edges()
        B = int(max(1, edges.size - 1))

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        exclude_nonwalk = bool(
            getattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
        )
        min_walk_frames = int(
            getattr(self.opts, "btw_rwd_conditioned_min_walk_frames", 2) or 2
        )

        per_fly_total: list[np.ndarray] = []
        per_fly_tail: list[np.ndarray] = []

        unit_info: list[dict] = []
        unit_ids: list[str] = []

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

                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=trx_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = wall_contact_mask(
                    self.opts,
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    log_tag=self.log_tag,
                    warned_missing_wc=warned_missing_wc,
                )
                nonwalk_mask = build_nonwalk_mask(self.opts, va, trx_idx, fi, n_frames)

                traj = va.trx[trx_idx]

                fly_id = self._fly_id(va, role_idx=role_idx, trx_idx=trx_idx)

                bin_vals_total: list[list[float]] = [[] for _ in range(B)]
                bin_vals_tail: list[list[float]] = [[] for _ in range(B)]

                # We need max_d_mm and max_d_i
                dist_stats = ("median", "max")

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
                    exclude_nonwalk=exclude_nonwalk,
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=min_walk_frames,
                    dist_stats=dist_stats,
                    debug=False,
                    yield_skips=False,
                ):
                    x = float(getattr(seg, "max_d_mm", np.nan))
                    if not np.isfinite(x):
                        continue

                    s = int(getattr(seg, "s", -1))
                    e = int(getattr(seg, "e", -1))
                    if e <= s + 1:
                        continue

                    max_i = getattr(seg, "max_d_i", None)
                    if max_i is None:
                        continue
                    max_i = int(max_i)

                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue

                    px_per_mm = self._px_per_mm(va)
                    dt_total = dist_traveled_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        start_override=None,
                        min_keep_frames=min_walk_frames,
                    )
                    dt_tail = dist_traveled_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        start_override=max_i,
                        min_keep_frames=min_walk_frames,
                    )

                    if np.isfinite(dt_total):
                        bin_vals_total[j].append(float(dt_total))
                    if np.isfinite(dt_tail):
                        bin_vals_tail[j].append(float(dt_tail))

                vec_total = np.full((B,), np.nan, dtype=float)
                vec_tail = np.full((B,), np.nan, dtype=float)

                for j in range(B):
                    if bin_vals_total[j]:
                        vv = np.asarray(bin_vals_total[j], dtype=float)
                        vv = vv[np.isfinite(vv)]
                        if vv.size:
                            vec_total[j] = float(np.mean(vv))
                    if bin_vals_tail[j]:
                        vv = np.asarray(bin_vals_tail[j], dtype=float)
                        vv = vv[np.isfinite(vv)]
                        if vv.size:
                            vec_tail[j] = float(np.mean(vv))

                if np.any(np.isfinite(vec_total)) or np.any(np.isfinite(vec_tail)):
                    per_fly_total.append(vec_total)
                    per_fly_tail.append(vec_tail)

                    video_id = self._video_base(va)
                    uid = f"{video_id}|fly={int(fly_id)}|trx={int(trx_idx)}"
                    unit_ids.append(uid)

                    unit_info.append(
                        dict(
                            video_id=self._video_base(va),
                            fly_id=int(fly_id),
                            trx_idx=int(trx_idx),
                            role_idx=int(role_idx),
                            fly_role=str(self._fly_role(role_idx)),
                            training_index=int(self.cfg.training_index),
                        )
                    )

        if not per_fly_total:
            meta = {
                "log_tag": self.log_tag,
                "training_index": int(self.cfg.training_index),
                "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
                "use_reward_exclusion_mask": bool(self.cfg.use_reward_exclusion_mask),
                "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
                "x_min_mm": float(self.cfg.x_min_mm),
                "x_max_mm": float(self.cfg.x_max_mm),
                "ci_conf": float(self.cfg.ci_conf),
                "n_fly_units": 0,
                "exclude_wall_contact": bool(
                    getattr(self.opts, "com_exclude_wall_contact", False)
                ),
                "exclude_nonwalking_frames": bool(exclude_nonwalk),
                "min_walk_frames": int(min_walk_frames),
                "units": "mm",
                "unit_info": unit_info,
            }
            return (
                np.empty((0, B), dtype=float),
                np.empty((0, B), dtype=float),
                np.empty((0,), dtype=object),
                meta,
            )

        Y_total = np.stack(per_fly_total, axis=0)
        Y_tail = np.stack(per_fly_tail, axis=0)

        meta = {
            "log_tag": self.log_tag,
            "training_index": int(self.cfg.training_index),
            "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
            "use_reward_exclusion_mask": bool(self.cfg.use_reward_exclusion_mask),
            "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
            "x_min_mm": float(self.cfg.x_min_mm),
            "x_max_mm": float(self.cfg.x_max_mm),
            "ci_conf": float(self.cfg.ci_conf),
            "n_fly_units": int(Y_total.shape[0]),
            "exclude_wall_contact": bool(
                getattr(self.opts, "com_exclude_wall_contact", False)
            ),
            "exclude_nonwalking_frames": bool(exclude_nonwalk),
            "min_walk_frames": int(min_walk_frames),
            "units": "mm",
            "unit_info": unit_info,
        }
        ids = np.asarray(unit_ids, dtype=object)
        return Y_total, Y_tail, ids, meta

    def _write_quantiles_tsv(
        self, path: str, res: BetweenRewardConditionedDistTravResult
    ) -> None:
        """
        Write per-bin distribution quantiles across fly-units for both metrics (total + tail).
        Requires res.per_unit_total and res.per_unit_tail (present when computed from raw data).
        """
        if res.per_unit_total is None or res.per_unit_tail is None:
            print(
                f"[{self.log_tag}] quantiles TSV: missing per_unit arrays; cannot write {path}"
            )
            return

        edges = np.asarray(res.x_edges, dtype=float)
        centers = np.asarray(res.x_centers, dtype=float)

        Yt = np.asarray(res.per_unit_total, dtype=float)  # (N, B)
        Yh = np.asarray(res.per_unit_tail, dtype=float)  # (N, B)
        B = int(centers.size)

        # quantiles to report
        qs = [25, 50, 75, 90, 95]

        util.ensureDir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "bin_lo_mm",
                        "bin_hi_mm",
                        "x_center_mm",
                        # totals
                        "n_units_total",
                        "mean_total",
                        "ci_lo_total",
                        "ci_hi_total",
                        "q25_total",
                        "median_total",
                        "q75_total",
                        "q90_total",
                        "q95_total",
                        # tail
                        "n_units_tail",
                        "mean_tail",
                        "ci_lo_tail",
                        "ci_hi_tail",
                        "q25_tail",
                        "median_tail",
                        "q75_tail",
                        "q90_tail",
                        "q95_tail",
                        # meta
                        "training_index",
                        "skip_first_sync_buckets",
                        "exclude_wall_contact",
                        "exclude_nonwalking_frames",
                        "min_walk_frames",
                    ]
                )
                + "\n"
            )

            for j in range(B):
                # totals
                col_t = Yt[:, j]
                col_t = col_t[np.isfinite(col_t)]
                qt = ["nan"] * len(qs)
                if col_t.size:
                    qt_vals = np.nanpercentile(col_t, qs)
                    qt = [f"{float(v)}" for v in qt_vals]

                # tail
                col_h = Yh[:, j]
                col_h = col_h[np.isfinite(col_h)]
                qh = ["nan"] * len(qs)
                if col_h.size:
                    qh_vals = np.nanpercentile(col_h, qs)
                    qh = [f"{float(v)}" for v in qh_vals]

                f.write(
                    "\t".join(
                        map(
                            str,
                            [
                                float(edges[j]),
                                float(edges[j + 1]),
                                float(centers[j]),
                                int(res.n_units[j]),
                                (
                                    float(res.mean_total[j])
                                    if np.isfinite(res.mean_total[j])
                                    else "nan"
                                ),
                                (
                                    float(res.ci_lo_total[j])
                                    if np.isfinite(res.ci_lo_total[j])
                                    else "nan"
                                ),
                                (
                                    float(res.ci_hi_total[j])
                                    if np.isfinite(res.ci_hi_total[j])
                                    else "nan"
                                ),
                                qt[0],
                                qt[1],
                                qt[2],
                                qt[3],
                                qt[4],
                                int(res.n_units[j]),
                                (
                                    float(res.mean_tail[j])
                                    if np.isfinite(res.mean_tail[j])
                                    else "nan"
                                ),
                                (
                                    float(res.ci_lo_tail[j])
                                    if np.isfinite(res.ci_lo_tail[j])
                                    else "nan"
                                ),
                                (
                                    float(res.ci_hi_tail[j])
                                    if np.isfinite(res.ci_hi_tail[j])
                                    else "nan"
                                ),
                                qh[0],
                                qh[1],
                                qh[2],
                                qh[3],
                                qh[4],
                                int(self.cfg.training_index),
                                int(self.cfg.skip_first_sync_buckets),
                                int(
                                    bool(
                                        getattr(
                                            self.opts, "com_exclude_wall_contact", False
                                        )
                                    )
                                ),
                                int(
                                    bool(
                                        getattr(
                                            self.opts,
                                            "btw_rwd_conditioned_exclude_nonwalking_frames",
                                            False,
                                        )
                                    )
                                ),
                                int(
                                    getattr(
                                        self.opts,
                                        "btw_rwd_conditioned_min_walk_frames",
                                        2,
                                    )
                                    or 2
                                ),
                            ],
                        )
                    )
                    + "\n"
                )
        print(f"[{self.log_tag}] wrote quantiles TSV: {path}")

    def compute_result(self) -> BetweenRewardConditionedDistTravResult:
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        Y_total, Y_tail, ids, meta = self._collect_per_fly_binned_means()

        mean_total = np.full((B,), np.nan, dtype=float)
        lo_total = np.full((B,), np.nan, dtype=float)
        hi_total = np.full((B,), np.nan, dtype=float)

        mean_tail = np.full((B,), np.nan, dtype=float)
        lo_tail = np.full((B,), np.nan, dtype=float)
        hi_tail = np.full((B,), np.nan, dtype=float)

        n_units = np.zeros((B,), dtype=int)

        if Y_total.size:
            for j in range(B):
                m, lo, hi, n = util.meanConfInt(
                    Y_total[:, j], conf=float(self.cfg.ci_conf)
                )
                mean_total[j], lo_total[j], hi_total[j] = float(m), float(lo), float(hi)
                n_units[j] = int(n)

                m, lo, hi, _n = util.meanConfInt(
                    Y_tail[:, j], conf=float(self.cfg.ci_conf)
                )
                mean_tail[j], lo_tail[j], hi_tail[j] = float(m), float(lo), float(hi)

        return BetweenRewardConditionedDistTravResult(
            x_edges=np.asarray(edges, dtype=float),
            x_centers=np.asarray(centers, dtype=float),
            mean_total=mean_total,
            ci_lo_total=lo_total,
            ci_hi_total=hi_total,
            mean_tail=mean_tail,
            ci_lo_tail=lo_tail,
            ci_hi_tail=hi_tail,
            n_units=n_units,
            meta=meta,
            per_unit_total=(Y_total if Y_total.size else None),
            per_unit_tail=(Y_tail if Y_tail.size else None),
            per_unit_ids=(ids if ids.size else None),
        )

    def plot(self) -> None:
        res = self.compute_result()
        res.validate()

        x = np.asarray(res.x_centers, dtype=float)
        edges = np.asarray(res.x_edges, dtype=float)
        widths = edges[1:] - edges[:-1]

        # derive two output paths from cfg.out_file
        out0 = str(self.cfg.out_file)
        root, ext = os.path.splitext(out0)
        if not ext:
            # if user passed something without an extension, fall back to opts.imageFormat
            ext = "." + str(getattr(self.opts, "imageFormat", "png")).lstrip(".")
        out_total = f"{root}_total{ext}"
        out_tail = f"{root}_tail{ext}"

        def _plot_one(*, y, lo, hi, title, ylabel, out_file) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

            if not np.any(np.isfinite(y)):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
            else:
                fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(widths)

                # Histogram-like bars: one bar per distance bin
                ax.bar(
                    x[fin],
                    y[fin],
                    width=0.92 * widths[fin],
                    align="center",
                    alpha=0.75,
                    linewidth=0.8,
                )

                fin_ci = fin & np.isfinite(lo) & np.isfinite(hi)
                if fin_ci.any():
                    yerr = np.vstack([y[fin_ci] - lo[fin_ci], hi[fin_ci] - y[fin_ci]])
                    ax.errorbar(
                        x[fin_ci],
                        y[fin_ci],
                        yerr=yerr,
                        fmt="none",
                        elinewidth=1.1,
                        capsize=2.0,
                        alpha=0.9,
                    )

                ax.set_xlabel(
                    maybe_sentence_case("max distance from reward center [mm] (binned)")
                )
                ax.set_ylabel(maybe_sentence_case(ylabel))

                if edges.size >= 2 and np.all(np.isfinite(edges[[0, -1]])):
                    ax.set_xlim(float(edges[0]), float(edges[-1]))

                ax.set_ylim(bottom=0)
                if self.cfg.ymax is not None:
                    ax.set_ylim(top=float(self.cfg.ymax))
                else:
                    y_top = (
                        np.nanmax(hi) if np.isfinite(np.nanmax(hi)) else np.nanmax(y)
                    )
                    if np.isfinite(y_top):
                        ax.set_ylim(top=float(y_top) * 1.12)

                ylim0, ylim1 = ax.get_ylim()
                y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
                for xi, hi_i, ni in zip(x, hi, res.n_units):
                    if np.isfinite(xi) and np.isfinite(hi_i) and int(ni) > 0:
                        util.pltText(
                            float(xi),
                            float(hi_i + y_off),
                            f"{int(ni)}",
                            ha="center",
                            size=self.customizer.in_plot_font_size,
                            color=".2",
                        )
                ax.set_title(maybe_sentence_case(title))

                # small annotation line
                parts = [f"T{int(self.cfg.training_index) + 1}"]
                if int(self.cfg.skip_first_sync_buckets) > 0:
                    parts.append(
                        f"skip first {int(self.cfg.skip_first_sync_buckets)} bucket(s)"
                    )
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
            writeImage(out_file, format=self.opts.imageFormat)
            plt.close(fig)
            print(f"[{self.log_tag}] wrote {out_file}")

        _plot_one(
            y=np.asarray(res.mean_total, dtype=float),
            lo=np.asarray(res.ci_lo_total, dtype=float),
            hi=np.asarray(res.ci_hi_total, dtype=float),
            title="between-reward distance traveled vs max distance-from-reward (total)",
            ylabel="mean distance traveled per fly [mm]",
            out_file=out_total,
        )

        _plot_one(
            y=np.asarray(res.mean_tail, dtype=float),
            lo=np.asarray(res.ci_lo_tail, dtype=float),
            hi=np.asarray(res.ci_hi_tail, dtype=float),
            title="between-reward distance traveled vs max distance-from-reward (max→end)",
            ylabel="mean distance traveled per fly [mm]",
            out_file=out_tail,
        )

        q_path = getattr(self.opts, "btw_rwd_conditioned_disttrav_quantiles_out", None)
        if q_path:
            try:
                self._write_quantiles_tsv(str(q_path), res)
            except Exception as e:
                print(f"[{self.log_tag}] WARNING: failed to write quantiles TSV: {e}")

        seg_path = getattr(self.opts, "btw_rwd_conditioned_disttrav_segs_out", None)
        if seg_path:
            try:
                self._write_sampled_segments_tsv(str(seg_path))
            except Exception as e:
                print(
                    f"[{self.log_tag}] WARNING: failed to write segment samples TSV: {e}"
                )


def plot_btw_rwd_conditioned_disttrav_overlay(
    *,
    results: Sequence[BetweenRewardConditionedDistTravResult],
    labels: Sequence[str],
    out_file: str,
    opts,
    customizer: PlotCustomizer,
    log_tag: str = "btw_rwd_dist_binned_disttrav",
) -> None:
    """
    Plot multiple cached BetweenRewardConditionedDistTravResult objects as grouped bars.

    Produces two images (like the single-group plotter):
      - <out_file>_total.<ext>
      - <out_file>_tail.<ext>

    `out_file` may include an extension; if so we preserve it. Otherwise we append opts.imageFormat.
    """
    if not results:
        raise ValueError("No results provided")
    if len(results) != len(labels):
        m = min(len(results), len(labels))
        results = list(results)[:m]
        labels = list(labels)[:m]

    # Derive output paths
    base = str(out_file)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = "." + str(getattr(opts, "imageFormat", "png")).lstrip(".")
        root = base
    out_total = f"{root}_total{ext}"
    out_tail = f"{root}_tail{ext}"

    # Reference x-axis
    ref = results[0]
    ref.validate()
    x = np.asarray(ref.x_centers, dtype=float)
    edges = np.asarray(ref.x_edges, dtype=float)
    widths = edges[1:] - edges[:-1]

    # Validate x_edges match across results
    for r, lab in zip(results, labels):
        r.validate()
        e2 = np.asarray(r.x_edges, dtype=float)
        if e2.shape != edges.shape or not np.allclose(e2, edges, equal_nan=True):
            raise ValueError(
                f"[{log_tag}] x_edges mismatch for {lab!r}; cannot overlay safely."
            )

    def _plot_one_metric(
        *,
        means: list[np.ndarray],
        lo: list[np.ndarray],
        hi: list[np.ndarray],
        n_units: list[np.ndarray],
        per_unit: list[np.ndarray | None],
        title: str,
        ylabel: str,
        out_path: str,
    ) -> None:

        n_groups = len(means)

        # grouped bar geometry
        fig_w, centers_x, bar_w, offsets, bin_ranges, xlim, categorical_used = (
            grouped_bar_layout_from_edges(edges, n_groups, categorical=True)
        )

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.4))

        if n_groups == 0:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            writeImage(out_path, format=opts.imageFormat)
            plt.close(fig)
            print(f"[{log_tag}] wrote {out_path}")
            return

        any_data = False
        pending_labels: list[tuple[float, float, int]] = []  # x, y_top, n
        xpos_by_group: list[np.ndarray] = []

        for gi in range(n_groups):
            y = np.asarray(means[gi], dtype=float)
            lo_i = np.asarray(lo[gi], dtype=float)
            hi_i = np.asarray(hi[gi], dtype=float)
            n_i = np.asarray(n_units[gi], dtype=int)

            xb = centers_x + offsets[gi]
            xpos_by_group.append(np.asarray(xb, float))

            fin = np.isfinite(xb) & np.isfinite(y) & np.isfinite(bar_w)
            if not fin.any():
                continue
            any_data = True

            ax.bar(
                xb[fin],
                y[fin],
                width=bar_w[fin],
                align="center",
                alpha=0.75,
                linewidth=0.8,
                label=str(labels[gi]),
            )

            fin_ci = fin & np.isfinite(lo_i) & np.isfinite(hi_i)
            if fin_ci.any():
                yerr = np.vstack([y[fin_ci] - lo_i[fin_ci], hi_i[fin_ci] - y[fin_ci]])
                ax.errorbar(
                    xb[fin_ci],
                    y[fin_ci],
                    yerr=yerr,
                    fmt="none",
                    elinewidth=1.1,
                    capsize=2.0,
                    alpha=0.9,
                )

            # queue per-bar n labels (added after we know y-lims)
            idxs = np.where(fin)[0]
            for j in idxs:
                nn = int(n_i[j])
                if nn <= 0:
                    continue
                y_top = float(hi_i[j]) if np.isfinite(hi_i[j]) else float(y[j])
                if np.isfinite(y_top):
                    pending_labels.append((float(xb[j]), y_top, nn))

        if not any_data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            ax.set_xlabel(
                maybe_sentence_case("max distance from reward center [mm] (binned)")
            )
            ax.set_ylabel(maybe_sentence_case(ylabel))

            ax.set_xlim(*xlim)
            ax.set_xticks(centers_x)

            # Tick labels: either bin ranges (0-2,2-4,...) or sparse numeric centers
            labels_xt = []
            for a, b in bin_ranges:
                if np.isclose(a, round(a)) and np.isclose(b, round(b)):
                    labels_xt.append(f"{int(round(a))}-{int(round(b))}")
                else:
                    labels_xt.append(f"{a:0.2f}-{b:0.2f}")
            ax.set_xticklabels(labels_xt, rotation=0, fontsize=8)

            ax.set_ylim(bottom=0)
            ymax = getattr(opts, "btw_rwd_conditioned_disttrav_ymax", None)
            if ymax is not None:
                try:
                    ax.set_ylim(top=float(ymax))
                except Exception:
                    pass

            ids_by_group = [r.per_unit_ids for r in results]

            # --- stats annotations (one-way ANOVA + Holm-corrected post-hoc) ---
            do_stats = bool(getattr(opts, "btw_rwd_conditioned_disttrav_stats", False))
            do_paired = bool(
                getattr(opts, "btw_rwd_conditioned_disttrav_stats_paired", False)
            )
            if do_stats and not any(pu is None for pu in per_unit):
                # if paired requested, ensure ids exist; otherwise warn + fall back
                use_paired = do_paired and not any(ids is None for ids in ids_by_group)
                if do_paired and not use_paired:
                    print(
                        f"[{log_tag}] WARNING: paired stats requested but missing per_unit_ids in one or more cached results; falling back to Welch."
                    )
                cfg_stats = StatAnnotConfig(
                    alpha=float(
                        getattr(opts, "btw_rwd_conditioned_disttrav_stats_alpha", 0.05)
                        or 0.05
                    ),
                    nlabel_off_frac=0.04,
                )
                annotate_grouped_bars_per_bin(
                    ax,
                    x_centers=centers_x,
                    xpos_by_group=xpos_by_group,
                    per_unit_by_group=[np.asarray(pu, float) for pu in per_unit],
                    per_unit_ids_by_group=(
                        [np.asarray(ids, dtype=object) for ids in ids_by_group]
                        if use_paired
                        else None
                    ),
                    hi_by_group=hi,
                    group_names=[str(l) for l in labels],
                    cfg=cfg_stats,
                    paired=use_paired,
                    panel_label=title,
                    debug=bool(
                        getattr(opts, "btw_rwd_conditioned_disttrav_stats_debug", False)
                    ),
                )

            ax.legend(loc="best", fontsize=customizer.in_plot_font_size)
            ax.set_title(maybe_sentence_case(title))

            # place n labels
            if pending_labels:
                ylim0, ylim1 = ax.get_ylim()
                y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
                for xb, y_top, nn in pending_labels:
                    util.pltText(
                        xb,
                        y_top + y_off,
                        f"{nn}",
                        ha="center",
                        size=customizer.in_plot_font_size,
                        color=".2",
                    )

        if customizer.font_size_customized:
            customizer.adjust_padding_proportionally()
        fig.tight_layout(rect=(0, 0, 1, 1))
        writeImage(out_path, format=opts.imageFormat)
        plt.close(fig)
        print(f"[{log_tag}] wrote {out_path}")

    # Build arrays for total
    per_unit_total = [r.per_unit_total for r in results]
    means_total = [np.asarray(r.mean_total, dtype=float) for r in results]
    lo_total = [np.asarray(r.ci_lo_total, dtype=float) for r in results]
    hi_total = [np.asarray(r.ci_hi_total, dtype=float) for r in results]
    n_total = [np.asarray(r.n_units, dtype=int) for r in results]

    _plot_one_metric(
        means=means_total,
        lo=lo_total,
        hi=hi_total,
        n_units=n_total,
        per_unit=per_unit_total,
        title="between-reward distance traveled vs max distance-from-reward (total)",
        ylabel="mean distance traveled per fly [mm]",
        out_path=out_total,
    )

    # Build arrays for tail
    per_unit_tail = [r.per_unit_tail for r in results]
    means_tail = [np.asarray(r.mean_tail, dtype=float) for r in results]
    lo_tail = [np.asarray(r.ci_lo_tail, dtype=float) for r in results]
    hi_tail = [np.asarray(r.ci_hi_tail, dtype=float) for r in results]
    n_tail = [np.asarray(r.n_units, dtype=int) for r in results]

    _plot_one_metric(
        means=means_tail,
        lo=lo_tail,
        hi=hi_tail,
        n_units=n_tail,
        per_unit=per_unit_tail,
        title="between-reward distance traveled vs max distance-from-reward (max→end)",
        ylabel="mean distance traveled per fly [mm]",
        out_path=out_tail,
    )
