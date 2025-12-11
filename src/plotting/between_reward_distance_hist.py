# src/plotting/between_reward_distance_hist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage


@dataclass
class BetweenRewardDistanceHistogramConfig:
    out_file: str
    bins: int = 30  # could make this configurable later if desired
    xmax: float | None = None


class BetweenRewardDistanceHistogramPlotter:
    """
    Aggregate between-reward distances across VideoAnalysis instances and
    plot histograms separated by training.

    Uses only experimental flies (f == 0 in multi-fly recordings).
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardDistanceHistogramConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg

        # assume all videos share the same training structure
        self.trns = vas[0].trns

    # ---------- data collection ----------

    def _collect_dists_by_training(self) -> list[np.ndarray]:
        """Return a list of length n_trainings, each a 1D array of distances."""
        all_by_trn: list[list[float]] = [[] for _ in self.trns]

        for va in self.vas:
            # Skip VAs that were skipped or have bad main trajectory
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            for t_idx, t in enumerate(self.trns):
                # Loop over flies, but only use experimental flies
                for f in va.flies:
                    if not va.noyc and f != 0:
                        # multi-fly (exp + yoked); keep only experimental fly
                        continue

                    # False → use actual rewards, not "calculated" rewards
                    on = va._getOn(t, False, f=f)
                    if on is None or len(on) < 2:
                        continue

                    dists_px = va._distTrav(f, on)
                    if not dists_px:
                        continue

                    trj = va.trx[f]
                    px_per_mm = trj.pxPerMmFloor * va.xf.fctr
                    dists_mm = np.array(dists_px, dtype=float) / px_per_mm

                    all_by_trn[t_idx].extend(dists_mm)

        return [np.asarray(xs, dtype=float) for xs in all_by_trn]

    # ---------- plotting ----------

    def plot_histograms(self) -> None:
        dists_by_trn = self._collect_dists_by_training()

        if not any(len(d) for d in dists_by_trn):
            print(
                "[btw_rwd_dists] no between-reward distance data found; skipping plot."
            )
            return

        n_trn = len(self.trns)
        fig, axes = plt.subplots(
            1,
            n_trn,
            figsize=(4.0 * n_trn, 4.0),
            squeeze=False,
            sharey=True,
        )
        axes = axes[0]

        for idx, (ax, dists, t) in enumerate(zip(axes, dists_by_trn, self.trns)):
            if dists.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            # Apply hard cutoff, if requested
            if self.cfg.xmax is not None:
                orig_n = dists.size
                dists = dists[dists <= self.cfg.xmax]
                dropped = orig_n - dists.size
                if dropped > 0:
                    print(
                        f"[btw-rwd-dists] {t.name()}: "
                        f"dropped {dropped} segments above {self.cfg.xmax} mm"
                    )

            if dists.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data ≤ cutoff", ha="center", va="center")
                continue

            # Basic histogram; bins could be made configurable via opts later
            ax.hist(dists, bins=self.cfg.bins)

            if self.cfg.xmax is not None:
                ax.set_xlim(0, self.cfg.xmax)

            ax.set_title(t.name())
            ax.set_xlabel("distance between rewards (mm)")
            if idx == 0:
                ax.set_ylabel("# between-reward segments")

        fig.suptitle("Between-reward distances per training (experimental flies only)")
        fig.tight_layout()

        # Use your existing helper so imageFormat is respected
        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)

        print(f"[btw_rwd_dists] wrote {self.cfg.out_file}")
