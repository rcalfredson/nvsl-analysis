# src/plotting/reward_raster_plotter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.reward_window_utils import (
    cumulative_window_seconds_for_frame,
    effective_sync_bucket_window,
    frames_in_windows,
    selected_training_indices,
    selected_windows_for_va,
    training_window_label,
    window_duration_s,
)


@dataclass(frozen=True)
class RewardRasterConfig:
    out_file: str
    nflies: int = 0
    seed: int | None = None
    per_training: bool = False
    marker_size: float = 6.0
    trainings: Sequence[int] | None = None
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    first_n_rewards: int = 0
    subset_label: str | None = None
    sort_by: str = "none"
    sli_values: Sequence[float] | None = None


@dataclass(frozen=True)
class RewardRasterFlyEntry:
    va: object
    reward_times_s: np.ndarray
    total_rewards_in_window: int
    time_to_first_reward_s: float
    sli: float | None


class RewardRasterPlotter:
    """
    Raster plot of optogenetic reward times (LED pulses) across flies.

    - One row per fly (one VideoAnalysis instance)
    - X axis is time in seconds
    - A marker is drawn at each reward time (va._getOn(trn, calc=False, f=0))
    """

    def __init__(self, vas, opts, gls, cfg: RewardRasterConfig):
        self.vas = list(vas)
        self.opts = opts
        self.gls = gls
        self.cfg = cfg

    def _sample_entries(self, entries: list[RewardRasterFlyEntry]) -> list[RewardRasterFlyEntry]:
        if not entries:
            return []
        n_cfg = int(self.cfg.nflies)
        if n_cfg <= 0 or n_cfg >= len(entries):
            return list(entries)
        n = min(n_cfg, len(entries))
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(len(entries), size=n, replace=False)
        return [entries[i] for i in idx]

    @staticmethod
    def _get_reward_frames_for_training(va, trn, f_idx: int = 0) -> np.ndarray:
        # calc=False => "real rewards" (LED pulse frame indices)
        try:
            rf = va._getOn(trn, calc=False, f=f_idx)
        except TypeError:
            raise

        if rf is None:
            return np.array([], dtype=int)

        rf = np.asarray(rf, dtype=int)
        rf = rf[np.isfinite(rf)]
        rf = rf[rf >= 0]
        return rf.astype(int, copy=False)

    def _effective_skip_first_sync_buckets(self) -> int:
        skip, _keep = effective_sync_bucket_window(
            getattr(self.cfg, "skip_first_sync_buckets", 0),
            getattr(self.cfg, "keep_first_sync_buckets", 0),
        )
        return skip

    def _effective_keep_first_sync_buckets(self) -> int:
        _skip, keep = effective_sync_bucket_window(
            getattr(self.cfg, "skip_first_sync_buckets", 0),
            getattr(self.cfg, "keep_first_sync_buckets", 0),
        )
        return keep

    def _selected_training_indices(self, ref_va) -> list[int]:
        return selected_training_indices(
            ref_va,
            self.cfg.trainings,
            log_tag="reward_raster",
        )

    def _selected_windows_for_va(self, va, selected_trainings: Sequence[int]):
        return selected_windows_for_va(
            va,
            selected_trainings,
            skip_first_sync_buckets=self._effective_skip_first_sync_buckets(),
            keep_first_sync_buckets=self._effective_keep_first_sync_buckets(),
            f=0,
        )

    @staticmethod
    def _window_duration_s(fi: int, end: int, fps: float) -> float:
        return window_duration_s(fi, end, fps)

    def _collect_windowed_entry(
        self, va, *, selected_trainings: Sequence[int], sli_value: float | None
    ) -> RewardRasterFlyEntry | None:
        windows = self._selected_windows_for_va(va, selected_trainings)
        if not windows:
            return None

        fps = float(getattr(va, "fps", 1.0) or 1.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 1.0

        reward_frames = frames_in_windows(va, windows, calc=False, ctrl=False, f=0)
        all_times = np.asarray(
            [
                cumulative_window_seconds_for_frame(windows, int(frame), fps=fps)
                for frame in reward_frames
            ],
            dtype=float,
        )

        first_n = int(getattr(self.cfg, "first_n_rewards", 0) or 0)
        if first_n > 0 and all_times.size > first_n:
            all_times = all_times[:first_n]

        t_first = float(all_times[0]) if all_times.size else np.inf
        sli = None if sli_value is None else float(sli_value)

        return RewardRasterFlyEntry(
            va=va,
            reward_times_s=all_times,
            total_rewards_in_window=int(reward_frames.size),
            time_to_first_reward_s=t_first,
            sli=sli,
        )

    def _build_windowed_entries(
        self, selected_trainings: Sequence[int]
    ) -> list[RewardRasterFlyEntry]:
        entries: list[RewardRasterFlyEntry] = []
        sli_values = list(self.cfg.sli_values) if self.cfg.sli_values is not None else None

        for i, va in enumerate(self.vas):
            if getattr(va, "_skipped", False):
                continue
            try:
                if va.trx[0].bad():
                    continue
            except Exception:
                pass

            sli_val = None
            if sli_values is not None and i < len(sli_values):
                try:
                    if np.isfinite(sli_values[i]):
                        sli_val = float(sli_values[i])
                except Exception:
                    sli_val = None

            entry = self._collect_windowed_entry(
                va, selected_trainings=selected_trainings, sli_value=sli_val
            )
            if entry is not None:
                entries.append(entry)

        return entries

    def _sort_entries(self, entries: list[RewardRasterFlyEntry]) -> list[RewardRasterFlyEntry]:
        mode = str(getattr(self.cfg, "sort_by", "none") or "none").lower()
        if mode == "none":
            return list(entries)

        if mode == "total_rewards":
            return sorted(
                entries,
                key=lambda e: (
                    -int(e.total_rewards_in_window),
                    float(e.time_to_first_reward_s),
                ),
            )
        if mode == "time_to_first_reward":
            return sorted(
                entries,
                key=lambda e: (
                    np.isinf(e.time_to_first_reward_s),
                    float(e.time_to_first_reward_s),
                    -int(e.total_rewards_in_window),
                ),
            )
        if mode == "sli":
            return sorted(
                entries,
                key=lambda e: (
                    e.sli is None or not np.isfinite(e.sli),
                    -(float(e.sli) if e.sli is not None and np.isfinite(e.sli) else -np.inf),
                    float(e.time_to_first_reward_s),
                ),
            )

        print(f"[reward_raster] WARNING: unknown sort mode '{mode}'; using input order.")
        return list(entries)

    def _training_label(self, selected_trainings: Sequence[int]) -> str:
        return training_window_label(selected_trainings)

    def _plot_windowed(
        self, ref_va, selected_trainings: Sequence[int]
    ) -> plt.Figure | None:
        entries = self._build_windowed_entries(selected_trainings)
        entries = self._sample_entries(entries)
        entries = self._sort_entries(entries)

        if not entries:
            print("[reward_raster] No flies had an included analysis window to plot.")
            return None

        nrows = len(entries)
        fig, ax = plt.subplots(
            1, 1, figsize=(max(8, 0.18 * nrows + 6.0), max(4, 0.18 * nrows + 2.0))
        )

        xs_all = []
        ys_all = []
        xmax = 0.0
        for yi, entry in enumerate(entries):
            xs = np.asarray(entry.reward_times_s, dtype=float)
            if xs.size:
                xs_all.append(xs)
                ys_all.append(np.full(xs.shape, yi, dtype=float))
                xmax = max(xmax, float(xs[-1]))

        if xs_all:
            ax.scatter(
                np.concatenate(xs_all),
                np.concatenate(ys_all),
                s=self.cfg.marker_size,
                marker="|",
                linewidths=0.8,
            )

        ax.set_xlabel("Time from selected window start (s)")
        ax.set_ylabel("Fly")
        ax.set_ylim(-1, nrows)
        ax.invert_yaxis()
        ax.grid(False)
        if xmax > 0:
            ax.set_xlim(0, xmax * 1.02)

        title = "Reward raster"
        title += f" ({self._training_label(selected_trainings)})"
        first_n = int(getattr(self.cfg, "first_n_rewards", 0) or 0)
        if first_n > 0:
            title += f", first {first_n} rewards"
        ax.set_title(title)

        meta = []
        if self._effective_skip_first_sync_buckets() > 0:
            meta.append(f"skip first {self._effective_skip_first_sync_buckets()} sync buckets")
        keep_first = self._effective_keep_first_sync_buckets()
        if keep_first > 0:
            meta.append(f"keep first {keep_first} sync buckets")
        if self.cfg.subset_label:
            meta.append(str(self.cfg.subset_label))
        sort_by = str(getattr(self.cfg, "sort_by", "none") or "none")
        if sort_by != "none":
            meta.append(f"sorted by {sort_by.replace('_', ' ')}")
        if meta:
            fig.suptitle(" | ".join(meta), y=0.995, fontsize=10)

        fig.tight_layout()
        return fig

    def _plot_legacy(self) -> plt.Figure | None:
        picked = self._sample_entries(
            [RewardRasterFlyEntry(va=va, reward_times_s=np.zeros((0,)), total_rewards_in_window=0, time_to_first_reward_s=np.inf, sli=None) for va in self.vas]
        )
        picked = [entry.va for entry in picked]
        if not picked:
            print("[reward_raster] No videos to plot.")
            return None

        # Use the first VA as the reference for training structure.
        ref_va = picked[0]
        trns = getattr(ref_va, "trns", None)
        if not trns:
            print("[reward_raster] No trainings found; skipping.")
            return None

        nrows = len(picked)

        if self.cfg.per_training:
            ncols = len(trns)
            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(max(6, 3.2 * ncols), max(4, 0.18 * nrows + 2.0)),
                sharey=True,
            )
            if ncols == 1:
                axes = [axes]

            for ti, trn in enumerate(trns):
                ax = axes[ti]
                xs_all = []
                ys_all = []

                for yi, va in enumerate(picked):
                    fps = float(getattr(va, "fps", 1.0))
                    rf = self._get_reward_frames_for_training(va, trn, f_idx=0)
                    # Restrict to training window, then align to training start
                    rf = rf[(rf >= trn.start) & (rf < trn.stop)]
                    xs = (rf - trn.start) / fps
                    if xs.size:
                        xs_all.append(xs)
                        ys_all.append(np.full(xs.shape, yi, dtype=float))

                if xs_all:
                    x = np.concatenate(xs_all)
                    y = np.concatenate(ys_all)
                    ax.scatter(x, y, s=self.cfg.marker_size, marker="|", linewidths=0.8)

                ax.set_title(f"Training {ti+1}")
                ax.set_xlabel("Time in training (s)")
                ax.grid(False)

            axes[0].set_ylabel("Fly (random subset)")
            axes[0].set_ylim(-1, nrows)
            axes[0].invert_yaxis()
        else:
            fig, ax = plt.subplots(
                1, 1, figsize=(max(8, 0.18 * nrows + 6.0), max(4, 0.18 * nrows + 2.0))
            )

            xs_all = []
            ys_all = []

            # Plot rewards across full session time
            for yi, va in enumerate(picked):
                fps = float(getattr(va, "fps", 1.0))
                for trn in trns:
                    rf = self._get_reward_frames_for_training(va, trn, f_idx=0)
                    rf = rf[(rf >= trn.start) & (rf <= trn.stop)]
                    xs = rf / fps
                    if xs.size:
                        xs_all.append(xs)
                        ys_all.append(np.full(xs.shape, yi, dtype=float))

            if xs_all:
                x = np.concatenate(xs_all)
                y = np.concatenate(ys_all)
                ax.scatter(x, y, s=self.cfg.marker_size, marker="|", linewidths=0.8)

            # Marking training boundaries using ref_va's times
            fps0 = float(getattr(ref_va, "fps", 1.0))
            for ti, trn in enumerate(trns):
                ax.axvline(trn.start / fps0, linestyle=":", linewidth=1)
                ax.axvline(trn.stop / fps0, linestyle=":", linewidth=1)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fly (random subset)")
            ax.set_ylim(-1, nrows)
            ax.invert_yaxis()
            ax.grid(False)

        fig.tight_layout()
        out = self.cfg.out_file
        fig.savefig(out, dpi=200)
        print(f"[reward_raster] saved: {out}")
        return fig

    def plot(self) -> plt.Figure | None:
        if not self.vas:
            print("[reward_raster] No videos to plot.")
            return None

        ref_va = self.vas[0]
        trns = getattr(ref_va, "trns", None)
        if not trns:
            print("[reward_raster] No trainings found; skipping.")
            return None

        selected_trainings = self._selected_training_indices(ref_va)

        use_windowed_mode = bool(
            self.cfg.trainings
            or self._effective_skip_first_sync_buckets() > 0
            or self._effective_keep_first_sync_buckets() > 0
            or int(getattr(self.cfg, "first_n_rewards", 0) or 0) > 0
            or str(getattr(self.cfg, "sort_by", "none") or "none").lower() != "none"
        )
        if self.cfg.per_training and use_windowed_mode:
            print(
                "[reward_raster] NOTE: --reward-raster-per-training is ignored for "
                "windowed/first-n raster mode; plotting a single selected-window axis."
            )

        if use_windowed_mode:
            fig = self._plot_windowed(ref_va, selected_trainings)
            if fig is None:
                return None
            out = self.cfg.out_file
            fig.savefig(out, dpi=200)
            print(f"[reward_raster] saved: {out}")
            return fig

        return self._plot_legacy()
