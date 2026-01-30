# src/plotting/reward_raster_plotter.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RewardRasterConfig:
    out_file: str
    nflies: int = 50
    seed: int | None = None
    per_training: bool = False
    marker_size: float = 6.0


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

    def _sample_vas(self) -> list:
        if not self.vas:
            return []
        n = min(int(self.cfg.nflies), len(self.vas))
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(len(self.vas), size=n, replace=False)
        return [self.vas[i] for i in idx]

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

    def plot(self) -> plt.Figure | None:
        picked = self._sample_vas()
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
