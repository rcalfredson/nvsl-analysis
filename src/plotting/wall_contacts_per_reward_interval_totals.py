from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)


@dataclass
class WallContactsPerRewardIntervalTotalsConfig(TrainingMetricScalarBarsConfig):
    pass


class WallContactsPerRewardIntervalPerFlyCollector:
    """
    Shared collector for wall-contact counts per between-reward interval, reduced to
    one scalar per fly:

    - For each fly/training, compute the mean number of wall-contact events across
      valid between-reward intervals in the selected sync-bucket window.
    - Returns per-training lists of (unit_id, mean_wall_contacts_per_interval).
    - Consumers may further subset trainings or pool them.
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

    @staticmethod
    def _wall_contact_regions(trj) -> list:
        try:
            return trj.boundary_event_stats["wall"]["all"]["edge"][
                "boundary_contact_regions"
            ]
        except Exception:
            return []

    @staticmethod
    def _sanitize_reward_frames(rf, *, start: int, stop: int) -> np.ndarray:
        if rf is None:
            return np.zeros(0, dtype=np.int64)
        rf = np.asarray(rf, dtype=float)
        rf = rf[np.isfinite(rf)]
        if rf.size == 0:
            return np.zeros(0, dtype=np.int64)

        rf = rf[(rf >= start) & (rf < stop)]
        if rf.size == 0:
            return np.zeros(0, dtype=np.int64)

        rf = np.unique(rf.astype(np.int64))
        rf.sort()
        return rf

    @staticmethod
    def _count_regions_by_reward_interval_start(
        regions, rf: np.ndarray
    ) -> np.ndarray:
        """
        Count regions by which inter-reward interval contains region.start.
        Intervals are [rf[i], rf[i+1]) for i=0..n_rewards-2.
        """
        if rf is None or rf.size < 2:
            return np.zeros(0, dtype=np.int32)

        starts = rf[:-1]
        ends = rf[1:]
        counts = np.zeros(starts.size, dtype=np.int32)
        if not regions:
            return counts

        for region in regions:
            sf = int(region.start)
            if sf < int(starts[0]) or sf >= int(ends[-1]):
                continue
            idx = int(np.searchsorted(starts, sf, side="right") - 1)
            if idx < 0 or idx >= starts.size:
                continue
            if sf < int(ends[idx]):
                counts[idx] += 1

        return counts

    def _collect_wall_contact_interval_means_by_training_per_fly(
        self,
    ) -> list[list[tuple[str, float]]]:
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float]]] = [[] for _ in range(n_trn)]

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
                    # Match existing scalar-bar conventions: experimental fly only
                    # when yoked controls are present.
                    if not va.noyc and f != 0:
                        continue

                    skip_first, keep_first = self._effective_sync_bucket_window()
                    fi, df, n_buckets, _complete = sync_bucket_window(
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

                    end = int(fi + n_buckets * df)
                    reward_frames = self._sanitize_reward_frames(
                        va._getOn(trn, calc=False, f=f),
                        start=int(fi),
                        stop=end,
                    )
                    counts = self._count_regions_by_reward_interval_start(
                        self._wall_contact_regions(va.trx[f]),
                        reward_frames,
                    )
                    if counts.size == 0:
                        continue

                    uid = self._unit_id(va, f=f)
                    out[t_idx].append((uid, float(np.mean(counts.astype(float)))))

        return out


class WallContactsPerRewardIntervalTotalsPlotter(
    TrainingMetricScalarBarsPlotter, WallContactsPerRewardIntervalPerFlyCollector
):
    def __init__(
        self,
        vas,
        opts,
        gls,
        customizer,
        cfg: WallContactsPerRewardIntervalTotalsConfig,
    ):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="wall_contacts_per_reward_interval_total",
            y_label="Mean wall contacts per reward interval",
            base_title="Wall contacts per reward interval (per fly)",
        )

    def _collect_values_by_training_per_fly_scalar(self):
        return self._collect_wall_contact_interval_means_by_training_per_fly()
