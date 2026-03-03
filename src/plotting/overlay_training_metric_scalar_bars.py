# src/plotting/overlay_training_metric_scalar_bars.py

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.utils.util import meanConfInt


@dataclass(frozen=True)
class ExportedTrainingScalarBars:
    group: str
    panel_labels: list[str]

    # object arrays length n_panels:
    #   per_unit_values_panel[p] -> (N_panel,) float
    #   per_unit_ids_panel[p] -> (N_panel,) object
    per_unit_values_panel: np.ndarray
    per_unit_ids_panel: np.ndarray

    # (n_panels,)
    mean: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    n_units_panel: np.ndarray

    meta: dict

    @property
    def pool_trainings(self) -> bool:
        return bool(self.meta.get("pool_trainings", False))

    @property
    def ci_conf(self) -> float:
        return float(self.meta.get("ci_conf", 0.95))


def _maybe_none_array(x) -> np.ndarray | None:
    """
    np.savez sometimes stores None as a 0-d object array.
    Normalize those cases back to Python None.
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            if arr.shape == () and arr.item() is None:
                return None
            if arr.size == 1 and arr.ravel()[0] is None:
                return None
        except Exception:
            pass
    return arr


def _mean_ci_from_util(x: np.ndarray, conf: float) -> tuple[float, float, float, int]:
    """
    Mean and t CI across finite x, via src.utils.util.meanConfInt.
    Returns (mean, lo, hi, n).
    """
    m, lo, hi, n = meanConfInt(np.asarray(x, float), conf=float(conf), asDelta=False)
    return float(m), float(lo), float(hi), int(n)


def load_export_npz(group: str, path: str) -> ExportedTrainingScalarBars:
    d = np.load(path, allow_pickle=True)

    panel_labels = [str(x) for x in list(d["panel_labels"])]

    per_unit_values_panel = np.asarray(d["per_unit_values_panel"], dtype=object)
    per_unit_ids_panel = np.asarray(d["per_unit_ids_panel"], dtype=object)

    mean = np.asarray(d["mean"], dtype=float)
    ci_lo = np.asarray(d["ci_lo"], dtype=float)
    ci_hi = np.asarray(d["ci_hi"], dtype=float)
    n_units_panel = np.asarray(d["n_units_panel"], dtype=int)

    meta_json = d["meta_json"].item() if "meta_json" in d.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(meta_json)

    return ExportedTrainingScalarBars(
        group=group,
        panel_labels=panel_labels,
        per_unit_values_panel=per_unit_values_panel,
        per_unit_ids_panel=per_unit_ids_panel,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_units_panel=n_units_panel,
        meta=meta,
    )


def _fmt_mismatch(name: str, vals: list) -> str:
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return f"{name} differs across inputs: {uniq}"


def validate_alignment(xs: list[ExportedTrainingScalarBars]) -> None:
    if len(xs) < 2:
        return

    pools = [x.pool_trainings for x in xs]
    if len(set(pools)) != 1:
        raise ValueError(_fmt_mismatch("pool_trainings", pools))

    labels0 = xs[0].panel_labels
    for x in xs[1:]:
        if x.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [y.panel_labels for y in xs])
            )


def _panel_n(x: ExportedTrainingScalarBars, p_idx: int) -> int | None:
    if x.n_units_panel is None:
        return None
    if p_idx < 0 or p_idx >= int(x.n_units_panel.shape[0]):
        return None
    try:
        return int(x.n_units_panel[p_idx])
    except Exception:
        return None


def _legend_n_for_group(x: ExportedTrainingScalarBars) -> int | None:
    """
    Return a single n to show in the legend if it's unambiguous.
    - If only one panel, use that panel's n.
    - If multiple panels, use n only if it's constant (ignoring zeros/None).
    Otherwise return None (legend should omit n).
    """
    P = len(x.panel_labels)
    if P <= 0:
        return None
    if P == 1:
        return _panel_n(x, 0)

    if x.n_units_panel is None:
        return None
    n = np.asarray(x.n_units_panel, int).ravel()
    n = n[np.isfinite(n) & (n > 0)]
    if n.size == 0:
        return None
    uniq = np.unique(n)
    if uniq.size == 1:
        return int(uniq[0])
    return None


def _panel_n_label(x: ExportedTrainingScalarBars, p_idx: int) -> str:
    if x.n_units_panel is not None and x.n_units_panel.shape[0] > p_idx:
        return str(int(x.n_units_panel[p_idx]))
    return "?"


def _group_union_matrix(
    x: ExportedTrainingScalarBars,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (N_union, P) matrix of per-unit values and a parallel (N_union,) id array.
    Missing values are NaN.
    """
    P = len(x.panel_labels)

    # union of IDs across panels (finite values only)
    union = set()
    for p in range(P):
        ids = np.asarray(x.per_unit_ids_panel[p], dtype=object).ravel()
        vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
        mask = np.isfinite(vals) & (ids != None)
        union |= {str(i) for i in ids[mask]}

    union_ids = sorted(union)
    idx = {k: i for i, k in enumerate(union_ids)}
    M = np.full((len(union_ids), P), np.nan, dtype=float)

    for p in range(P):
        ids = np.asarray(x.per_unit_ids_panel[p], dtype=object).ravel()
        vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
        for i0, v0 in zip(ids, vals):
            if i0 is None:
                continue
            v = float(v0)
            if not np.isfinite(v):
                continue
            r = idx.get(str(i0), None)
            if r is not None:
                M[r, p] = v

    return M, np.asarray(union_ids, dtype=object)


def _paired_filter_across_groups_for_panel(
    mats: list[np.ndarray],
    ids: list[np.ndarray],
    *,
    p_idx: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    For a single panel index p_idx:
      - Find IDs present (finite) in ALL groups
      - Return a list of vectors (one per group) filtered to those IDs
      - Return the common IDs as object array
    """
    sets = []
    for M, I in zip(mats, ids):
        col = np.asarray(M[:, p_idx], float)
        mask = np.isfinite(col)
        sets.append({str(i) for i in I[mask] if i is not None})

    common = set.intersection(*sets) if sets else set()
    common_ids = np.asarray(sorted(common), dtype=object)

    out_vecs = []
    for M, I in zip(mats, ids):
        idx_map = {str(i): ii for ii, i in enumerate(I) if i is not None}
        v = np.full((common_ids.size,), np.nan, float)
        for k_idx, k in enumerate(common_ids):
            ii = idx_map.get(str(k), None)
            if ii is None:
                continue
            vv = float(M[ii, p_idx])
            if np.isfinite(vv):
                v[k_idx] = vv
        out_vecs.append(v)

    return out_vecs, common_ids


def plot_overlays(
    xs: list[ExportedTrainingScalarBars],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
) -> plt.Figure:
    validate_alignment(xs)

    panel_labels = xs[0].panel_labels
    P = len(panel_labels)
    G = len(xs)

    fig_w = max(6.0, 1.2 * P)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.5))

    x_centers = np.arange(P, dtype=float)

    # grouped/dodged bars
    group_band = 0.80
    bar_w = group_band / max(1, G)
    gpos = np.arange(G) - (G - 1) / 2.0
    offsets = gpos * bar_w  # (G,)

    # Per-group union matrices for stats (and paired recompute)
    mats = []
    ids_union = []
    for x in xs:
        M, I = _group_union_matrix(x)
        mats.append(M)
        ids_union.append(I)

    # If paired plotting, recompute displayed mean/CI per panel per group from paired-only
    paired_n_per_panel = None
    means_plot = []
    lo_plot = []
    hi_plot = []

    if stats and stats_paired:
        paired_n_per_panel = np.zeros((P,), dtype=int)
        for p in range(P):
            vecs, common_ids = _paired_filter_across_groups_for_panel(
                mats, ids_union, p_idx=p
            )
            paired_n_per_panel[p] = int(common_ids.size)

            for gi in range(G):
                if p == 0:
                    means_plot.append(np.full((P,), np.nan, float))
                    lo_plot.append(np.full((P,), np.nan, float))
                    hi_plot.append(np.full((P,), np.nan, float))

                m, lo, hi, _n = _mean_ci_from_util(vecs[gi], conf=xs[0].ci_conf)
                means_plot[gi][p] = m
                lo_plot[gi][p] = lo
                hi_plot[gi][p] = hi
    else:
        for x in xs:
            means_plot.append(np.asarray(x.mean, float))
            lo_plot.append(np.asarray(x.ci_lo, float))
            hi_plot.append(np.asarray(x.ci_hi, float))

    xpos_by_group = []
    hi_by_group = []
    per_unit_by_group = []
    per_unit_ids_by_group = []

    for gi, x in enumerate(xs):
        xg = x_centers + offsets[gi]
        xpos_by_group.append(np.asarray(xg, float))

        y = np.asarray(means_plot[gi], float)
        y_plot = np.where(np.isfinite(y), y, 0.0)
        n_leg = _legend_n_for_group(x)
        label = f"{x.group} (n={n_leg})" if n_leg is not None else f"{x.group}"
        ax.bar(xg, y_plot, width=bar_w, align="center", label=label)

        # CI whiskers (only if export had CI; in paired-mode we recomputed them anyway)
        lo = np.asarray(lo_plot[gi], float)
        hi = np.asarray(hi_plot[gi], float)
        mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if np.any(mask):
            ax.errorbar(
                xg[mask],
                y[mask],
                yerr=np.vstack([y[mask] - lo[mask], hi[mask] - y[mask]]),
                fmt="none",
                ecolor="0.15",
                capsize=3,
                capthick=1.0,
                elinewidth=1.0,
                alpha=0.9,
                zorder=3,
            )

        # baseline for brackets
        hi_by_group.append(np.where(np.isfinite(hi), hi, y))

        # stats payload
        per_unit_by_group.append(mats[gi])  # (N_union, P)
        per_unit_ids_by_group.append(ids_union[gi])  # (N_union,)

    # ---- per-panel n labels (only when legend n is omitted) ----
    # Show per-panel n centered on each tick, above the tallest bar/CI in that panel.
    need_per_panel_n = (
        (P > 1)
        and any(_legend_n_for_group(x) is None for x in xs)
        and not (stats and stats_paired)
    )

    if need_per_panel_n:
        ylim0, ylim1 = ax.get_ylim()
        y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
        y_pad = 0.015 * y_rng

        for p in range(P):
            # if all groups share same n at this panel, show a single "n=.."
            ns = []
            for x in xs:
                npp = _panel_n(x, p)
                if npp is not None and npp > 0:
                    ns.append(npp)
            if not ns:
                continue

            uniq = sorted(set(ns))
            n_text = (
                f"n={uniq[0]}" if len(uniq) == 1 else "n=" + "/".join(map(str, uniq))
            )

            # baseline above tallest bar/CI at this panel
            y_top = np.nan
            for gi in range(G):
                if p < hi_by_group[gi].shape[0] and np.isfinite(hi_by_group[gi][p]):
                    y_top = (
                        float(hi_by_group[gi][p])
                        if not np.isfinite(y_top)
                        else max(float(y_top), float(hi_by_group[gi][p]))
                    )
            if not np.isfinite(y_top):
                continue

            ax.text(
                float(x_centers[p]),
                float(y_top + y_pad),
                n_text,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.2",
                clip_on=False,
                zorder=9,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(panel_labels, rotation=30, ha="right")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))

    if stats:
        cfg_stats = StatAnnotConfig(
            alpha=float(stats_alpha),
            min_n_per_group=3,
            nlabel_off_frac=0.0,
        )
        annotate_grouped_bars_per_bin(
            ax,
            x_centers=x_centers,
            xpos_by_group=xpos_by_group,
            per_unit_by_group=per_unit_by_group,
            per_unit_ids_by_group=per_unit_ids_by_group,
            hi_by_group=hi_by_group,
            group_names=[x.group for x in xs],
            cfg=cfg_stats,
            paired=bool(stats_paired),
            panel_label=None,
            debug=debug,
        )

        # Optional: if paired and n varies by panel, annotate n above bars
        if stats_paired and paired_n_per_panel is not None:
            nz = paired_n_per_panel[paired_n_per_panel > 0]
            uniq = np.unique(nz) if nz.size else np.asarray([], int)
            n_constant = int(uniq[0]) if uniq.size == 1 else None

            if n_constant is None:
                ylim0, ylim1 = ax.get_ylim()
                y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
                y_pad = 0.015 * y_rng
                for p in range(P):
                    npp = int(paired_n_per_panel[p])
                    if npp <= 0:
                        continue
                    y_top = np.nan
                    for gi in range(G):
                        if p < hi_by_group[gi].shape[0] and np.isfinite(
                            hi_by_group[gi][p]
                        ):
                            y_top = (
                                hi_by_group[gi][p]
                                if not np.isfinite(y_top)
                                else max(y_top, hi_by_group[gi][p])
                            )
                    if np.isfinite(y_top):
                        ax.text(
                            float(x_centers[p]),
                            float(y_top + y_pad),
                            f"n={npp}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="0.2",
                            clip_on=False,
                            zorder=9,
                        )

    if title:
        fig.suptitle(title)

    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
