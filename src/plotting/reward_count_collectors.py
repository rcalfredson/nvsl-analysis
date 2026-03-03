from __future__ import annotations

import os

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window


class RewardCountPerFlyCollector:
    """
    Shared collector for reward-count *scalars*:
    - returns per-training lists of (unit_id, total_reward_count)
    - honors cfg.skip_first_sync_buckets / cfg.keep_first_sync_buckets
    - honors cfg.trainings selection / pooling at the *consumer* level (plotters)
    """

    @staticmethod
    def _reward_count_unit_id(va, *, f: int) -> str:
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

    def _reward_count_effective_sync_bucket_window(self) -> tuple[int, int]:
        return (
            self._effective_skip_first_sync_buckets(),
            self._effective_keep_first_sync_buckets(),
        )

    @staticmethod
    def _filter_on_to_window(on: np.ndarray, *, fi: int, end: int) -> np.ndarray:
        # assuming on is frame indices (int)
        on = np.asarray(on)
        if on.size == 0:
            return on
        return on[(on >= fi) & (on < end)]

    def _collect_reward_totals_by_training_per_fly(self):
        """
        Collect total reward counts per fly, grouped by training.

        Returns
        -------
        list[list[tuple[str, float]]]
            Outer list has length n_trainings (as defined by self._n_trainings()).
            For each training index t:
                out[t] is a list of (unit_id, total_reward_count),
            where total_reward_count is the number of reward-on events within the
            effective sync-bucket window for that fly/training.

        Notes
        -----
        - Sync-bucket windowing uses cfg.skip_first_sync_buckets / cfg.keep_first_sync_buckets.
        - Only the experimental fly is included when yoked controls are present
          (i.e., when not va.noyc, only f == 0 contributes).
        - Consumers (plotters/exporters) may further select trainings or pool them;
          this collector always returns per-training values.
        """
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float]]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):

                # Experimental fly only
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    skip_first, keep_first = (
                        self._reward_count_effective_sync_bucket_window()
                    )
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

                    end = int(fi + n_buckets * df)

                    on = va._getOn(trn, False, f=f)
                    if on is None:
                        continue

                    on_win = self._filter_on_to_window(on, fi=fi, end=end)
                    n_rewards = float(len(on_win))

                    unit_id = self._reward_count_unit_id(va, f=f)
                    out[t_idx].append((unit_id, float(n_rewards)))

        return out
