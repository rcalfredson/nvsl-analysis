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

        # Nonwalking flags are expected to already be mapped into these generic names
        exclude_nonwalk = bool(
            getattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
        )
        min_walk_frames = int(
            getattr(self.opts, "btw_rwd_conditioned_min_walk_frames", 2) or 2
        )

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
