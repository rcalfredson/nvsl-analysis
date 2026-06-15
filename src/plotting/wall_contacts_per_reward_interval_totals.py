from __future__ import annotations

import csv
from dataclasses import dataclass
import os

import numpy as np

from src.analysis.between_reward_filters import (
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sync_bucket_presence_filters import exp_target_sync_bucket_eligibility_mask
from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)


@dataclass
class WallContactsPerRewardIntervalTotalsConfig(TrainingMetricScalarBarsConfig):
    metric: str = "mean_contacts"


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

    @staticmethod
    def _training_name(trn, *, t_idx: int) -> str:
        try:
            return str(trn.name())
        except Exception:
            return f"training {t_idx + 1}"

    @staticmethod
    def _max_reward_distances_mm(va, trn, *, f: int, start: int, stop: int):
        try:
            cx, cy, radius_px = trn.circles(f)[0]
            px_per_mm = float(va.xf.fctr) * float(va.ct.pxPerMmFloor())
            traj = va.trx[f]
            x = np.asarray(traj.x[start:stop], dtype=float)
            y = np.asarray(traj.y[start:stop], dtype=float)
        except Exception:
            return np.nan, np.nan
        if x.size == 0 or y.size == 0 or not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return np.nan, np.nan
        radial_px = np.hypot(x - float(cx), y - float(cy))
        finite = radial_px[np.isfinite(radial_px)]
        if finite.size == 0:
            return np.nan, np.nan
        max_center_mm = float(np.max(finite) / px_per_mm)
        max_past_perimeter_mm = max(
            0.0,
            float(np.max(finite) - float(radius_px)) / px_per_mm,
        )
        return max_center_mm, max_past_perimeter_mm

    def _collect_wall_contact_interval_episode_rows(
        self,
        *,
        include_target_ineligible: bool,
        include_geometry: bool = False,
    ) -> list[dict]:
        rows: list[dict] = []
        n_trn = self._n_trainings()
        skip_first, keep_first = self._effective_sync_bucket_window()

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            target_eligible = bool(
                exp_target_sync_bucket_eligibility_mask([va], self.opts)[0]
            )
            if not target_eligible and not include_target_ineligible:
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue

            video_fn = str(getattr(va, "fn", "") or "")
            video_basename = os.path.basename(video_fn)
            va_tag = int(getattr(va, "f", 0) or 0)
            fps = float(getattr(va, "fps", np.nan))
            trns = getattr(va, "trns", [])

            for t_idx, trn in enumerate(trns[:n_trn]):
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

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
                    for episode_idx, (start_frame, end_frame, wall_count) in enumerate(
                        zip(reward_frames[:-1], reward_frames[1:], counts),
                        start=1,
                    ):
                        start_frame = int(start_frame)
                        end_frame = int(end_frame)
                        start_bucket = (
                            skip_first + int((start_frame - int(fi)) // int(df)) + 1
                        )
                        end_bucket = (
                            skip_first
                            + int((max(start_frame, end_frame - 1) - int(fi)) // int(df))
                            + 1
                        )
                        if include_geometry:
                            max_center_mm, max_past_perimeter_mm = (
                                self._max_reward_distances_mm(
                                    va,
                                    trn,
                                    f=f,
                                    start=start_frame,
                                    stop=end_frame,
                                )
                            )
                        else:
                            max_center_mm, max_past_perimeter_mm = np.nan, np.nan
                        rows.append(
                            {
                                "video_path": video_fn,
                                "video_basename": video_basename,
                                "unit_id": uid,
                                "va_tag": va_tag,
                                "trajectory_index": int(f),
                                "fly_role": "exp" if int(f) == 0 else "yok",
                                "training_index": int(t_idx + 1),
                                "training_name": self._training_name(
                                    trn, t_idx=t_idx
                                ),
                                "window_start_frame": int(fi),
                                "window_end_frame": end,
                                "skip_first_sync_buckets": int(skip_first),
                                "keep_first_sync_buckets": int(keep_first),
                                "episode_index_in_window": int(episode_idx),
                                "start_reward_frame": start_frame,
                                "end_reward_frame": end_frame,
                                "start_sync_bucket": int(start_bucket),
                                "end_sync_bucket": int(end_bucket),
                                "start_time_s": (
                                    float(start_frame / fps)
                                    if np.isfinite(fps) and fps > 0
                                    else np.nan
                                ),
                                "end_time_s": (
                                    float(end_frame / fps)
                                    if np.isfinite(fps) and fps > 0
                                    else np.nan
                                ),
                                "duration_s": (
                                    float((end_frame - start_frame) / fps)
                                    if np.isfinite(fps) and fps > 0
                                    else np.nan
                                ),
                                "wall_contact_event_count": int(wall_count),
                                "contains_wall_contact": bool(wall_count > 0),
                                "contactless": bool(wall_count == 0),
                                "wall_contact_assignment_rule": (
                                    "contact_region_start_in_"
                                    "[start_reward_frame,end_reward_frame)"
                                ),
                                "max_distance_from_reward_center_mm": max_center_mm,
                                "max_distance_past_reward_perimeter_mm": (
                                    max_past_perimeter_mm
                                ),
                                "passes_target_sync_bucket_filter": target_eligible,
                            }
                        )
        return rows

    def _collect_wall_contact_interval_aggregates_by_training_per_fly(
        self,
    ) -> list[list[tuple[str, float, int, int]]]:
        """
        Return wall-contact sums, contactless counts, and interval counts per
        fly/training.

        The count is the number of valid between-reward intervals in the selected
        sync-bucket window, so it is also the denominator for the episode filter.
        """
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float, int, int]]] = [[] for _ in range(n_trn)]
        grouped: dict[tuple[int, str], list[dict]] = {}
        for row in self._collect_wall_contact_interval_episode_rows(
            include_target_ineligible=False
        ):
            key = (int(row["training_index"]) - 1, str(row["unit_id"]))
            grouped.setdefault(key, []).append(row)

        for (t_idx, uid), rows in grouped.items():
            if t_idx < 0 or t_idx >= n_trn:
                continue
            wall_counts = np.asarray(
                [row["wall_contact_event_count"] for row in rows],
                dtype=int,
            )
            out[t_idx].append(
                (
                    uid,
                    float(np.sum(wall_counts)),
                    int(np.count_nonzero(wall_counts == 0)),
                    int(wall_counts.size),
                )
            )

        return out

    def _collect_wall_contact_interval_means_by_training_per_fly(
        self,
    ) -> list[list[tuple[str, float]]]:
        """
        Return per-training metrics after applying the between-reward episode filter.
        """
        min_intervals = min_between_reward_sync_bucket_trajectories(self.opts)
        out: list[list[tuple[str, float]]] = []

        for panel in self._collect_wall_contact_interval_aggregates_by_training_per_fly():
            vals: list[tuple[str, float]] = []
            for uid, total, contactless_count, count in panel:
                if count >= min_intervals and count > 0:
                    numerator = (
                        contactless_count
                        if self.cfg.metric == "contactless_fraction"
                        else total
                    )
                    vals.append((uid, float(numerator) / float(count)))
                elif count > 0:
                    vals.append((uid, np.nan))
            out.append(vals)

        return out

    def _collect_wall_contact_interval_pooled_means_per_fly(
        self, training_indices
    ) -> list[tuple[str, float]]:
        """
        Return one scalar per fly after pooling selected trainings at episode level.
        """
        min_intervals = min_between_reward_sync_bucket_trajectories(self.opts)
        by_training = self._collect_wall_contact_interval_aggregates_by_training_per_fly()
        sums: dict[str, float] = {}
        contactless_counts: dict[str, int] = {}
        counts: dict[str, int] = {}

        for t_idx in training_indices:
            if t_idx < 0 or t_idx >= len(by_training):
                continue
            for uid, total, contactless_count, count in by_training[t_idx]:
                sums[uid] = sums.get(uid, 0.0) + float(total)
                contactless_counts[uid] = (
                    contactless_counts.get(uid, 0) + int(contactless_count)
                )
                counts[uid] = counts.get(uid, 0) + int(count)

        out: list[tuple[str, float]] = []
        for uid in sorted(sums):
            count = counts.get(uid, 0)
            if count >= min_intervals and count > 0:
                numerator = (
                    contactless_counts.get(uid, 0)
                    if self.cfg.metric == "contactless_fraction"
                    else sums[uid]
                )
                out.append((uid, float(numerator) / float(count)))
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
        if cfg.metric == "contactless_fraction":
            y_label = "Fraction of trajectories without wall contact"
            base_title = "Contactless between-reward trajectories (per fly)"
        else:
            y_label = "Mean wall contacts per reward interval"
            base_title = "Wall contacts per reward interval (per fly)"
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="wall_contacts_per_reward_interval_total",
            y_label=y_label,
            base_title=base_title,
        )

    def _collect_values_by_training_per_fly_scalar(self):
        return self._collect_wall_contact_interval_means_by_training_per_fly()

    def _collect_pooled_values_per_fly_scalar(self, training_indices):
        return self._collect_wall_contact_interval_pooled_means_per_fly(training_indices)

    def export_episode_csv(self, out_csv: str) -> None:
        rows = self._collect_wall_contact_interval_episode_rows(
            include_target_ineligible=True,
            include_geometry=True,
        )
        n_trn = self._n_trainings()
        if self.cfg.trainings:
            selected = {
                int(training)
                for training in self.cfg.trainings
                if 1 <= int(training) <= n_trn
            }
            rows = [
                row for row in rows if int(row["training_index"]) in selected
            ]

        min_intervals = min_between_reward_sync_bucket_trajectories(self.opts)
        group_label = str(getattr(self.opts, "export_group_label", "") or "group")
        grouped: dict[tuple, list[dict]] = {}
        for row in rows:
            key = (
                (str(row["unit_id"]),)
                if self.cfg.pool_trainings
                else (
                    str(row["unit_id"]),
                    int(row["training_index"]),
                )
            )
            grouped.setdefault(key, []).append(row)

        output_rows = []
        for key in sorted(grouped):
            unit_rows = grouped[key]
            n_episodes = len(unit_rows)
            n_contactless = sum(bool(row["contactless"]) for row in unit_rows)
            passes_minimum = n_episodes >= min_intervals
            passes_target = all(
                bool(row["passes_target_sync_bucket_filter"]) for row in unit_rows
            )
            included = passes_minimum and passes_target
            fraction = (
                float(n_contactless) / float(n_episodes)
                if n_episodes > 0
                else np.nan
            )
            for row in unit_rows:
                output_rows.append(
                    {
                        "group": group_label,
                        **row,
                        "pooled_episode_count": int(n_episodes),
                        "pooled_contactless_episode_count": int(n_contactless),
                        "pooled_contactless_fraction": fraction,
                        "minimum_episode_count": int(min_intervals),
                        "passes_minimum_episode_filter": bool(passes_minimum),
                        "included_in_metric": bool(included),
                        "exclusion_reason": (
                            ""
                            if included
                            else (
                                "target_sync_bucket_filter"
                                if not passes_target
                                else "minimum_episode_count"
                            )
                        ),
                    }
                )

        fieldnames = [
            "group",
            "video_path",
            "video_basename",
            "unit_id",
            "va_tag",
            "trajectory_index",
            "fly_role",
            "training_index",
            "training_name",
            "window_start_frame",
            "window_end_frame",
            "skip_first_sync_buckets",
            "keep_first_sync_buckets",
            "episode_index_in_window",
            "start_reward_frame",
            "end_reward_frame",
            "start_sync_bucket",
            "end_sync_bucket",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "wall_contact_event_count",
            "contains_wall_contact",
            "contactless",
            "wall_contact_assignment_rule",
            "max_distance_from_reward_center_mm",
            "max_distance_past_reward_perimeter_mm",
            "passes_target_sync_bucket_filter",
            "pooled_episode_count",
            "pooled_contactless_episode_count",
            "pooled_contactless_fraction",
            "minimum_episode_count",
            "passes_minimum_episode_filter",
            "included_in_metric",
            "exclusion_reason",
        ]
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(
            "[wall_contacts_per_reward_interval_total] wrote episode provenance "
            f"CSV {out_csv} ({len(output_rows)} rows)"
        )
