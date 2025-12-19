# src/plotting/overlay_training_metric_hist.py

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExportedTrainingHistogram:
    group: str
    panel_labels: list[str]
    counts: np.ndarray  # (n_panels, bins)
    bin_edges: np.ndarray  # (n_panels, bins+1)
    n_used: np.ndarray  # (n_panels,)
    meta: dict

    @property
    def bins(self) -> int:
        return int(self.meta.get("bins"))

    @property
    def xmax_effective(self) -> float | None:
        return self.meta.get("xmax_effective", None)

    @property
    def pool_trainings(self) -> bool:
        return bool(self.meta.get("pool_trainings", False))


def load_export_npz(group: str, path: str) -> ExportedTrainingHistogram:
    d = np.load(path, allow_pickle=True)
    panel_labels = [str(x) for x in list(d["panel_labels"])]
    counts = np.asarray(d["counts"])
    bin_edges = np.asarray(d["bin_edges"])
    n_used = np.asarray(d["n_used"])
    meta = json.loads(str(d["meta_json"]))
    return ExportedTrainingHistogram(
        group=group,
        panel_labels=panel_labels,
        counts=counts,
        bin_edges=bin_edges,
        n_used=n_used,
        meta=meta,
    )


def _fmt_mismatch(name: str, vals: list) -> str:
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return f"{name} differs across inputs: {uniq}"


def validate_alignment(hists: list[ExportedTrainingHistogram]) -> None:
    if len(hists) < 2:
        return

    bins = [h.bins for h in hists]
    if len(set(bins)) != 1:
        raise ValueError(_fmt_mismatch("bins", bins))

    pools = [h.pool_trainings for h in hists]
    if len(set(pools)) != 1:
        raise ValueError(_fmt_mismatch("pool_trainings", pools))

    xmax_eff = [h.xmax_effective for h in hists]
    # For numeric xmax, require close equality; for None, all must be None.
    if any(x is None for x in xmax_eff):
        if not all(x is None for x in xmax_eff):
            raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))
    else:
        # All numeric
        x0 = float(xmax_eff[0])
        for x in xmax_eff[1:]:
            if not np.isclose(float(x), x0, rtol=0, atol=1e-9):
                raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))

    # Panel labels must match exactly
    labels0 = hists[0].panel_labels
    for h in hists[1:]:
        if h.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [hh.panel_labels for hh in hists])
            )

    # Bin edges must match per panel
    edges0 = hists[0].bin_edges
    for h in hists[1:]:
        if h.bin_edges.shape != edges0.shape:
            raise ValueError("bin_edges shape differs across inputs")
        if not np.allclose(h.bin_edges, edges0, rtol=0, atol=1e-12):
            raise ValueError(
                "bin_edges differ across inputs. Re-export with the same --*-max and bins."
            )


def plot_overlays(
    hists: list[ExportedTrainingHistogram],
    *,
    mode: str,  # "pdf" or "cdf"
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
) -> plt.Figure:
    if mode not in ("pdf", "cdf"):
        raise ValueError("mode must be 'pdf' or 'cdf'")

    validate_alignment(hists)
    panel_labels = hists[0].panel_labels
    n_panels = len(panel_labels)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.0 * n_panels if n_panels > 1 else 6.5, 4.0),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    edges = hists[0].bin_edges
    # For a step plot, we’ll use edges and a y of length bins+1
    for p_idx, (ax, plabel) in enumerate(zip(axes, panel_labels)):
        any_data = False
        for h in hists:
            counts = np.asarray(h.counts[p_idx], dtype=float)
            total = counts.sum()
            if total <= 0:
                continue
            any_data = True

            if mode == "pdf":
                y_bins = counts / total
                # convert to step y with length bins+1
                y_step = np.concatenate([y_bins, [y_bins[-1]]])
                ax.step(
                    edges[p_idx],
                    y_step,
                    where="post",
                    label=f"{h.group} (n={int(h.n_used[p_idx])})",
                )
            else:
                cdf = np.cumsum(counts) / total
                y_step = np.concatenate([[0.0], cdf])
                ax.step(
                    edges[p_idx],
                    y_step,
                    where="post",
                    label=f"{h.group} (n={int(h.n_used[p_idx])})",
                )

        if not any_data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        ax.set_title(plabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if p_idx == 0:
            if ylabel:
                ax.set_ylabel(ylabel)

        if ymax is not None:
            ax.set_ylim(top=float(ymax))

        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
