from __future__ import annotations

import os
import numpy as np

from src.plotting.between_reward_segment_binning import (
    sync_bucket_window,
    build_nonwalk_mask,
    wall_contact_mask,
)
from src.plotting.between_reward_segment_metrics import dist_traveled_mm_masked


class ReturnLegDistPerFlyCollector:
    """
    Shared collector for between-reward return-leg (tail) distance *scalars*:

    - Returns per-training lists of (unit_id, mean_return_leg_distance_mm_per_segment)
    - Honors cfg.skip_first_sync_buckets / cfg.keep_first_sync_buckets
    - Uses the same segment iterator + masking semantics as between_reward_conditioned_disttrav.py
    - Consumers (plotters/exporters) may select trainings or pool them.
    """

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        video_fn = getattr(va, "fn", None)
        base = os.path.basename(str(video_fn)) if video_fn else "unknown_video"
        va_id = int(getattr(va, "f", 0) or 0)
        return f"{base}|va_tag={va_id}|trx_idx={int(f)}"

    def _effective_keep_first_sync_buckets(self) -> int:
        ckeep = int(getattr(self.cfg, "keep_first_sync_buckets", 0) or 0)
        return 0 if ckeep < 0 else ckeep

    def _effective_skip_first_sync_buckets(self) -> int:
        cskip = int(getattr(self.cfg, "skip_first_sync_buckets", 0) or 0)
        return 0 if cskip < 0 else cskip

    def _effective_sync_bucket_window(self) -> tuple[int, int]:
        return (
            self._effective_skip_first_sync_buckets(),
            self._effective_keep_first_sync_buckets(),
        )

    def _n_trainings(self) -> int:
        return max((len(getattr(va, "trns", [])) for va in self.vas), default=0)

    def _exclude_nonwalk(self) -> bool:
        if hasattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames"):
            return bool(
                getattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames")
            )
        return bool(
            getattr(self.opts, "btw_rwd_return_leg_dist_exclude_nonwalking_frames", False)
        )

    def _min_walk_frames(self) -> int:
        if hasattr(self.opts, "btw_rwd_conditioned_min_walk_frames"):
            return int(getattr(self.opts, "btw_rwd_conditioned_min_walk_frames", 2) or 2)
        return int(getattr(self.opts, "btw_rwd_return_leg_dist_min_walk_frames", 2) or 2)

    def _px_per_mm(self, va) -> float | None:
        try:
            _px_per_mm = float(va.xf.fctr * va.ct.pxPerMmFloor())
            if np.isfinite(_px_per_mm) and _px_per_mm > 0:
                return _px_per_mm
        except Exception:
            pass
        return None

    def _collect_return_leg_means_by_training_per_fly(
        self,
    ) -> list[list[tuple[str, float]]]:
        """
        Returns
        -------
        list[list[tuple[str, float]]]
            Outer list length n_trainings.
            For each training t:
              out[t] is a list of (unit_id, mean_dt_tail_mm_per_segment) for flies with >=1 valid segment.
        """
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float]]] = [[] for _ in range(n_trn)]

        # Shared knobs (same names/behavior as the binned disttrav plotter)
        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )

        exclude_nonwalk = self._exclude_nonwalk()
        min_walk_frames = self._min_walk_frames()

        warned_missing_wc = [False]
        dist_stats = ("median", "max")

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue

            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):

                for f in va.flies:
                    # Experimental fly only when yoked controls are present
                    if not va.noyc and f != 0:
                        continue

                    skip_first, keep_first = self._effective_sync_bucket_window()
                    fi, df, n_buckets, complete = sync_bucket_window(
                        va,
                        trn,
                        t_idx=t_idx,
                        f=f,
                        skip_first=skip_first,
                        keep_first=keep_first,
                        use_exclusion_mask=False,
                    )
                    if n_buckets <= 0:
                        continue

                    n_frames = int(max(1, n_buckets * df))

                    wc = wall_contact_mask(
                        self.opts,
                        va,
                        f,
                        fi=fi,
                        n_frames=n_frames,
                        log_tag="btw_rwd_return_leg_dist",
                        warned_missing_wc=warned_missing_wc,
                    )
                    nonwalk_mask = build_nonwalk_mask(self.opts, va, f, fi, n_frames)

                    traj = va.trx[f]
                    px_per_mm = self._px_per_mm(va)
                    if px_per_mm is None:
                        continue

                    tail_vals: list[float] = []

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
                        exclude_nonwalk=exclude_nonwalk,
                        nonwalk_mask=nonwalk_mask,
                        min_walk_frames=min_walk_frames,
                        dist_stats=dist_stats,
                        debug=False,
                        yield_skips=False,
                    ):
                        s = int(getattr(seg, "s", -1))
                        e = int(getattr(seg, "e", -1))
                        if e <= s + 1:
                            continue

                        max_i = getattr(seg, "max_d_i", None)
                        if max_i is None:
                            continue
                        max_i = int(max_i)

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
                        if np.isfinite(dt_tail):
                            tail_vals.append(float(dt_tail))

                    if not tail_vals:
                        continue

                    uid = self._unit_id(va, f=f)
                    out[t_idx].append(
                        (uid, float(np.mean(np.asarray(tail_vals, dtype=float))))
                    )
        return out

    def collect_return_leg_sync_bucket_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return per-video per-training per-sync-bucket mean return-leg distance arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            mean_exp, mean_ctrl, n_exp, n_ctrl
            with shapes (n_videos, n_trainings, n_buckets).
        """
        n_videos = len(self.vas)
        n_trn = self._n_trainings()
        skip_first, keep_first = self._effective_sync_bucket_window()

        def _theoretical_window(va, trn) -> tuple[int | None, int, list[bool], int]:
            try:
                df = int(va._numRewardsMsg(True, silent=True))
                fi0, n_raw, _on = va._syncBucket(trn, df)
            except Exception:
                return None, 1, [], 0

            if fi0 is None or n_raw is None:
                return fi0, df, [], 0

            n_raw = max(0, int(n_raw))
            df = max(1, int(df))
            skip_k = max(0, int(skip_first))
            keep_k = max(0, int(keep_first))

            if skip_k >= n_raw:
                return int(fi0) + skip_k * df, df, [], 0

            fi = int(fi0) + skip_k * df
            n_eff = n_raw - skip_k
            if keep_k > 0:
                n_eff = min(n_eff, keep_k)

            starts = [int(fi + k * df) for k in range(n_eff)]
            complete = [(int(trn.stop) - s) >= df for s in starts]
            return fi, df, complete, int(n_eff)

        n_buckets = 0
        for va in self.vas:
            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):
                _fi, _df, _complete, nb_here = _theoretical_window(va, trn)
                n_buckets = max(n_buckets, int(nb_here))

        mean_exp = np.full((n_videos, n_trn, n_buckets), np.nan, dtype=float)
        mean_ctrl = np.full((n_videos, n_trn, n_buckets), np.nan, dtype=float)
        n_exp = np.zeros((n_videos, n_trn, n_buckets), dtype=int)
        n_ctrl = np.zeros((n_videos, n_trn, n_buckets), dtype=int)

        if n_trn == 0 or n_buckets == 0:
            return mean_exp, mean_ctrl, n_exp, n_ctrl

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        exclude_nonwalk = self._exclude_nonwalk()
        min_walk_frames = self._min_walk_frames()

        warned_missing_wc = [False]
        dist_stats = ("median", "max")

        for vi, va in enumerate(self.vas):
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue

            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):
                for fly_key, f in (("exp", 0), ("ctrl", 1)):
                    if f >= len(getattr(va, "trx", [])):
                        continue

                    traj = va.trx[f]
                    if traj is None or traj.bad():
                        continue

                    fi, df, complete, nb_here = _theoretical_window(va, trn)
                    if nb_here <= 0:
                        continue

                    n_frames = int(max(1, nb_here * df))
                    wc = wall_contact_mask(
                        self.opts,
                        va,
                        f,
                        fi=fi,
                        n_frames=n_frames,
                        log_tag="btw_rwd_return_leg_dist",
                        warned_missing_wc=warned_missing_wc,
                    )
                    nonwalk_mask = build_nonwalk_mask(self.opts, va, f, fi, n_frames)

                    px_per_mm = self._px_per_mm(va)
                    if px_per_mm is None:
                        continue

                    sums = np.zeros(nb_here, dtype=float)
                    counts = np.zeros(nb_here, dtype=int)

                    for seg in va._iter_between_reward_segment_com(
                        trn,
                        f,
                        fi=fi,
                        df=df,
                        n_buckets=nb_here,
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
                        s = int(getattr(seg, "s", -1))
                        e = int(getattr(seg, "e", -1))
                        if e <= s + 1:
                            continue

                        b_idx = int(getattr(seg, "b_idx", -1))
                        if b_idx < 0 or b_idx >= nb_here:
                            continue

                        max_i = getattr(seg, "max_d_i", None)
                        if max_i is None:
                            continue

                        dt_tail = dist_traveled_mm_masked(
                            traj=traj,
                            s=s,
                            e=e,
                            fi=fi,
                            nonwalk_mask=nonwalk_mask,
                            exclude_nonwalk=exclude_nonwalk,
                            px_per_mm=px_per_mm,
                            start_override=int(max_i),
                            min_keep_frames=min_walk_frames,
                        )
                        if not np.isfinite(dt_tail):
                            continue

                        sums[b_idx] += float(dt_tail)
                        counts[b_idx] += 1

                    means = np.full(nb_here, np.nan, dtype=float)
                    keep = counts > 0
                    means[keep] = sums[keep] / counts[keep]

                    tgt_mean = mean_exp if fly_key == "exp" else mean_ctrl
                    tgt_n = n_exp if fly_key == "exp" else n_ctrl
                    tgt_mean[vi, t_idx, :nb_here] = means
                    tgt_n[vi, t_idx, :nb_here] = counts

        return mean_exp, mean_ctrl, n_exp, n_ctrl
