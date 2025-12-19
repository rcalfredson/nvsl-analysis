from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage


@dataclass
class TrainingMetricHistogramConfig:
    out_file: str
    bins: int = 30
    xmax: float | None = None
    normalize: bool = False
    pool_trainings: bool = False
    subset_label: str | None = None
    ymax: float | None = None


class TrainingMetricHistogramPlotter:
    """
    Generic "one panel per training" histogram plotter.
    Subclasses only need to implement _collect_values_by_training().
    """

    def _n_trainings(self) -> int:
        return max((len(getattr(va, "trns", [])) for va in self.vas), default=0)

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: TrainingMetricHistogramConfig,
        *,
        log_tag: str,
        x_label: str,
        base_title: str,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = log_tag
        self.x_label = x_label
        self.base_title = base_title

    def _training_labels(self, n_trn: int) -> list[str]:
        """
        Build per-training labels without assuming share Training objects.
        Uses the most common Training.name() at each training index across VAs.
        """
        labels: list[str] = []
        for i in range(n_trn):
            names = []
            for va in self.vas:
                trns = getattr(va, "trns", None)
                if not trns or i >= len(trns):
                    continue
                try:
                    names.append(trns[i].name())
                except Exception:
                    continue
            if names:
                labels.append(Counter(names).most_common(1)[0][0])
            else:
                labels.append(f"training {i + 1}")
        return labels

    def _collect_values_by_training(self) -> list[np.ndarray]:
        raise NotImplementedError

    def plot_histograms(self) -> None:
        vals_by_trn = self._collect_values_by_training()

        if not any(len(v) for v in vals_by_trn):
            print(f"[{self.log_tag}] no data found; skipping plot.")
            return

        if self.cfg.pool_trainings:
            pooled = np.concatenate([v for v in vals_by_trn if v.size > 0])
            vals_by_trn = [pooled]
            trn_labels = ["all trainings combined"]
        else:
            trn_labels = self._training_labels(len(vals_by_trn))

        n_trn = len(vals_by_trn)
        fig, axes = plt.subplots(
            1,
            n_trn,
            figsize=(4.0 * n_trn if n_trn > 1 else 6.5, 4.0),
            squeeze=False,
            sharey=True,
        )
        axes = axes[0]

        for idx, (ax, vals, label) in enumerate(zip(axes, vals_by_trn, trn_labels)):
            if vals.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            if self.cfg.xmax is not None:
                orig_n = vals.size
                vals = vals[vals <= self.cfg.xmax]
                dropped = orig_n - vals.size
                if dropped > 0:
                    print(
                        f"[{self.log_tag}] {label}: dropped {dropped} values "
                        f"above {self.cfg.xmax}"
                    )

            if vals.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data ≤ cutoff", ha="center", va="center")
                continue

            if self.cfg.normalize:
                counts, bin_edges = np.histogram(vals, bins=self.cfg.bins)
                total = counts.sum()
                if total == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                proportions = counts / total
                bin_widths = np.diff(bin_edges)
                ax.bar(bin_edges[:-1], proportions, width=bin_widths, align="edge")
            else:
                ax.hist(vals, bins=self.cfg.bins)

            if self.cfg.xmax is not None:
                ax.set_xlim(0, self.cfg.xmax)
            if self.cfg.ymax is not None:
                ax.set_ylim(top=self.cfg.ymax)

            ax.set_title(label)
            ax.set_xlabel(self.x_label)
            if idx == 0:
                ax.set_ylabel(
                    "proportion of segments" if self.cfg.normalize else "# segments"
                )
        title = self.base_title
        if self.cfg.subset_label:
            title = f"{title}\n{self.cfg.subset_label}"
        fig.suptitle(title)
        fig.tight_layout()

        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
