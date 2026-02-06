# src/plotting/wall_contacts_pmf.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.utils.common import maybe_sentence_case, writeImage
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin


_EPISODE_LABELS = {
    "sync_bucket": "wall contacts per sync bucket",
    "reward_interval": "wall contacts per reward interval",
}


@dataclass
class WallContactsPMFResult:
    """
    Cached export for wall-contact counts per episode.

    Supported NPZ formats:
      A) Sync bucket:
         - counts_per_bucket: (N, B) int
         - role_idx: (N,) int (0=exp, 1=yok)
         - optional: n_buckets, training_idx, training_name, video_basename

      B) Reward interval:
         - counts_per_interval: (N, K) int
         - role_idx: (N,) int
         - optional: n_intervals, n_rewards, training_idx, training_name, video_basename
    """

    counts: np.ndarray
    role_idx: np.ndarray
    episode_kind: str  # "sync_bucket" | "reward_interval"

    training_idx: int | None = None
    training_name: str | None = None
    video_basename: np.ndarray | None = None
    n_episodes: np.ndarray | None = None
    n_rewards: np.ndarray | None = None  # reward_interval only

    @classmethod
    def load_npz(cls, path: str) -> "WallContactsPMFResult":
        d = np.load(path, allow_pickle=True)

        if "counts_per_bucket" in d:
            counts = np.asarray(d["counts_per_bucket"])
            kind = "sync_bucket"
            n_episodes = (
                np.asarray(d["n_buckets"], dtype=int) if "n_buckets" in d else None
            )
            n_rewards = None
        elif "counts_per_interval" in d:
            counts = np.asarray(d["counts_per_interval"])
            kind = "reward_interval"
            n_episodes = (
                np.asarray(d["n_intervals"], dtype=int) if "n_intervals" in d else None
            )
            n_rewards = (
                np.asarray(d["n_rewards"], dtype=int) if "n_rewards" in d else None
            )
        else:
            raise ValueError(
                "Unrecognized NPZ format: expected counts_per_bucket or counts_per_interval"
            )

        role_idx = np.asarray(d["role_idx"])
        training_idx = (
            int(np.asarray(d["training_idx"]).reshape(-1)[0])
            if "training_idx" in d
            else None
        )
        training_name = (
            str(np.asarray(d["training_name"]).reshape(-1)[0])
            if "training_name" in d
            else None
        )
        video_basename = (
            np.asarray(d["video_basename"], dtype=object)
            if "video_basename" in d
            else None
        )

        res = cls(
            counts=counts,
            role_idx=role_idx,
            episode_kind=str(kind),
            training_idx=training_idx,
            training_name=training_name,
            video_basename=video_basename,
            n_episodes=n_episodes,
            n_rewards=n_rewards,
        )
        res.validate()
        return res

    def validate(self) -> None:
        c = np.asarray(self.counts)
        r = np.asarray(self.role_idx)
        if c.ndim != 2:
            raise ValueError("counts must be 2D (N, E)")
        if r.ndim != 1:
            raise ValueError("role_idx must be 1D (N,)")
        if c.shape[0] != r.shape[0]:
            raise ValueError("counts and role_idx must agree on N")

        if (
            self.n_episodes is not None
            and np.asarray(self.n_episodes).shape[0] != c.shape[0]
        ):
            raise ValueError("n_episodes must have length N")
        if (
            self.video_basename is not None
            and np.asarray(self.video_basename).shape[0] != c.shape[0]
        ):
            raise ValueError("video_basename must have length N")
        if (
            self.n_rewards is not None
            and np.asarray(self.n_rewards).shape[0] != c.shape[0]
        ):
            raise ValueError("n_rewards must have length N")

        if self.episode_kind not in _EPISODE_LABELS:
            raise ValueError(f"Unknown episode_kind={self.episode_kind!r}")

        # counts should be integer-like
        if not np.issubdtype(c.dtype, np.integer):
            if np.any(np.abs(c - np.round(c)) > 1e-6):
                raise ValueError("counts must be integer-valued")


def _rebin_pmf_2d(pmf: np.ndarray, bin_w: int) -> np.ndarray:
    if bin_w <= 1:
        return pmf
    X = np.asarray(pmf, dtype=float)
    N, K = X.shape
    K2 = (K + bin_w - 1) // bin_w
    out = np.full((N, K2), np.nan, dtype=float)
    for j in range(K2):
        a = j * bin_w
        b = min(K, (j + 1) * bin_w)
        out[:, j] = np.nansum(X[:, a:b], axis=1)
    return out


def _sanitize_bin_edges(edges) -> np.ndarray:
    """
    Convert user edges into a strictly increasing int array of edges in k-space.
    Ensures first edge is 0 (prepends if needed).
    """
    if edges is None:
        return np.zeros((0,), dtype=int)

    e = np.asarray(edges, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        return np.zeros((0,), dtype=int)

    # convert to int k edges (floor is typical; you can also round)
    e = np.floor(e).astype(int)

    # keep non-negative
    e = e[e >= 0]
    if e.size < 2:
        return np.zeros((0,), dtype=int)

    # sort + unique
    e = np.unique(e)
    e.sort()

    # require strictly increasing by construction (unique)
    if e.size < 2:
        return np.zeros((0,), dtype=int)

    # ensure starts at 0 for PMF indexing
    if e[0] != 0:
        e = np.concatenate(([0], e))

    return e


def _rebin_pmf_2d_by_edges(
    pmf: np.ndarray, edges: np.ndarray, *, overflow: bool
) -> np.ndarray:
    """
    Rebin a per-fly PMF over integer k bins into custom k-ranges given by edges.

    pmf: (N, K) for k=0..K-1
    edges: (M,) integer edges, e.g. [0,10,20] -> bins [0..9], [10..19]
    overflow:
      - False: only bins defined by edges (last is [edges[-2], edges[-1]))
      - True:  last bin is [edges[-1], +inf) (using available K support)
    """
    X = np.asarray(pmf, dtype=float)
    if X.ndim != 2:
        raise ValueError("pmf must be 2D (N, K)")
    if edges.size < 2:
        return X  # nothing to do

    N, K = X.shape

    # Determine number of output bins
    if overflow:
        n_bins = (edges.size - 1) + 1  # edge-defined bins + final overflow bin
    else:
        n_bins = edges.size - 1

    out = np.full((N, n_bins), np.nan, dtype=float)

    # Sum probability mass over each bin’s k support
    # Bin j: [edges[j], edges[j+1]) for j in 0..edges.size-2
    for j in range(edges.size - 1):
        a = int(edges[j])
        b = int(edges[j + 1])
        if a >= K:
            # entirely beyond support
            out[:, j] = 0.0
            continue
        a = max(0, a)
        b = max(a, min(K, b))
        out[:, j] = np.nansum(X[:, a:b], axis=1) if b > a else 0.0

    if overflow:
        a = int(edges[-1])
        if a >= K:
            out[:, -1] = 0.0
        else:
            out[:, -1] = np.nansum(X[:, a:K], axis=1)

    return out


def _edge_xticklabels(edges: np.ndarray, *, overflow: bool) -> list[str]:
    """
    Human-friendly x tick labels for edge bins.
    """
    labs = []
    for j in range(edges.size - 1):
        a = int(edges[j])
        b = int(edges[j + 1]) - 1
        labs.append(f"{a}-{b}" if a != b else f"{a}")
    if overflow:
        labs.append(f"{int(edges[-1])}+")
    return labs


def _select_counts_by_role(res: WallContactsPMFResult, role: str) -> np.ndarray:
    c = np.asarray(res.counts)
    r = np.asarray(res.role_idx, dtype=int)
    role = str(role)

    if role == "exp":
        return c[r == 0]
    if role == "yok":
        return c[r == 1]
    if role == "both":
        return c
    raise ValueError(f"Unknown role={role!r} (expected exp|yok|both)")


def _pooled_kmax(
    series_counts: Sequence[np.ndarray],
    *,
    pctl: float,
    cap: int | None,
) -> int:
    pooled = []
    for c in series_counts:
        c = np.asarray(c)
        if c.size == 0:
            continue
        pooled.append(c.reshape(-1))
    if not pooled:
        return 1

    allc = np.concatenate(pooled, axis=0).astype(float)
    allc = allc[np.isfinite(allc)]
    allc = allc[allc >= 0]
    if allc.size == 0:
        return 1

    k = int(np.floor(np.percentile(allc, float(pctl))))
    if cap is not None:
        k = min(int(cap), k)
    return max(1, k)


def _counts_to_pmf(counts_2d: np.ndarray, *, k_max: int, overflow: bool) -> np.ndarray:
    c = np.asarray(counts_2d)
    if c.size == 0:
        return np.zeros((0, k_max + 1), dtype=float)

    c = np.asarray(np.round(c), dtype=int)

    N, _E = c.shape
    K = k_max + 1
    pmf = np.full((N, K), np.nan, dtype=float)

    for i in range(N):
        row = c[i, :]
        row = row[np.isfinite(row)]
        row = row[row >= 0]
        denom = int(row.size)
        if denom <= 0:
            continue

        if overflow:
            clipped = np.clip(row, 0, k_max)
            h = np.bincount(clipped, minlength=K)[:K]
            pmf[i, :] = h / float(denom)
        else:
            row2 = row[row <= k_max]
            denom2 = int(row2.size)
            if denom2 <= 0:
                continue
            h = np.bincount(row2, minlength=K)[:K]
            pmf[i, :] = h / float(denom2)

    return pmf


def _mean_and_ci_tbased(
    pmf: np.ndarray, *, conf: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    X = np.asarray(pmf, dtype=float)
    row_ok = np.any(np.isfinite(X), axis=1)
    X = X[row_ok]
    n = int(X.shape[0])

    K = int(pmf.shape[1])
    if n == 0:
        nanv = np.full((K,), np.nan, dtype=float)
        return nanv, nanv, nanv, 0

    mean = np.full((K,), np.nan, dtype=float)
    lo = np.full((K,), np.nan, dtype=float)
    hi = np.full((K,), np.nan, dtype=float)

    for k in range(K):
        m, l, h, _ = util.meanConfInt(X[:, k], conf=conf, asDelta=False)
        mean[k] = float(m)
        lo[k] = float(l)
        hi[k] = float(h)

    return mean, lo, hi, n


def plot_wall_contacts_pmf_overlay(
    *,
    results: Sequence[WallContactsPMFResult],
    labels: Sequence[str],
    out_file: str,
    opts,
    customizer: PlotCustomizer,
    log_tag: str = "wall_contacts_pmf",
) -> None:
    """
    Generic PMF overlay plot over k wall contacts per episode (episode type inferred from NPZ).

    Each fly contributes its own PMF over k (based on its episodes),
    then we average across flies (equal fly weighting) and show t-based CI.
    """
    if not results:
        raise ValueError("No results provided")

    if len(results) != len(labels):
        m = min(len(results), len(labels))
        results = list(results)[:m]
        labels = list(labels)[:m]

    role = str(getattr(opts, "wall_contacts_pmf_role", "exp"))
    pctl = float(getattr(opts, "wall_contacts_pmf_kmax_pctl", 99.0))
    cap = getattr(opts, "wall_contacts_pmf_kmax_cap", 30)
    cap = int(cap) if cap is not None else None
    overflow = bool(getattr(opts, "wall_contacts_pmf_overflow", False))
    overflow = bool(getattr(opts, "wall_contacts_pmf_overflow", False))

    # binning: either custom edges OR uniform width
    edges_raw = getattr(opts, "wall_contacts_pmf_bin_edges", None)
    edges = _sanitize_bin_edges(edges_raw)

    bin_w = int(getattr(opts, "wall_contacts_pmf_bin_width", 1))
    use_edges = edges.size >= 2

    # output extension handling
    base = str(out_file)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = "." + str(getattr(opts, "imageFormat", "png")).lstrip(".")
        root = base
    out_path = f"{root}{ext}"

    # --- select counts by role ---
    group_counts: list[np.ndarray] = []
    group_labels: list[str] = []
    kinds: list[str] = []

    for res, lab in zip(results, labels):
        res.validate()
        c = _select_counts_by_role(res, role=role)
        if c.ndim != 2 or c.shape[1] <= 0 or c.shape[0] <= 0:
            continue
        kinds.append(res.episode_kind)
        group_counts.append(np.asarray(c))
        group_labels.append(str(lab))

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

    if not group_counts:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        writeImage(out_path, format=opts.imageFormat)
        plt.close(fig)
        print(f"[{log_tag}] wrote {out_path}")
        return

    # enforce consistent episode kind across overlay
    kind0 = kinds[0] if kinds else "sync_bucket"
    if any(k != kind0 for k in kinds):
        raise ValueError(f"Mixed episode kinds in overlay: {sorted(set(kinds))}")

    episode_label = _EPISODE_LABELS.get(kind0, "wall contacts per episode")

    # pooled k_max across groups
    k_max = _pooled_kmax(group_counts, pctl=pctl, cap=cap)

    # If user edges extend beyond pooled k_max, expand support so bins exist.
    if use_edges:
        # Ensure we can represent up to at least the last edge (and ideally the overflow start).
        k_max = max(int(k_max), int(edges[-1]))

    K = k_max + 1

    per_unit: list[np.ndarray] = []
    means: list[np.ndarray] = []
    lo: list[np.ndarray] = []
    hi: list[np.ndarray] = []
    ns: list[int] = []

    for c in group_counts:
        pmf = _counts_to_pmf(c, k_max=k_max, overflow=overflow)
        if use_edges:
            pmf = _rebin_pmf_2d_by_edges(pmf, edges, overflow=overflow)
        else:
            pmf = _rebin_pmf_2d(pmf, bin_w)

        per_unit.append(np.asarray(pmf, float))
        m, l, h, n = _mean_and_ci_tbased(pmf)
        means.append(m)
        lo.append(l)
        hi.append(h)
        ns.append(n)

    K2 = int(means[0].size) if means else 0
    x = np.arange(K2, dtype=float)
    widths = np.ones_like(x, dtype=float)

    n_groups = len(means)
    frac = 0.86
    bar_w = frac * widths / max(1, n_groups)

    any_data = False
    xpos_by_group: list[np.ndarray] = []

    for gi in range(n_groups):
        y = np.asarray(means[gi], dtype=float)
        lo_i = np.asarray(lo[gi], dtype=float)
        hi_i = np.asarray(hi[gi], dtype=float)

        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        xb = x + offset
        xpos_by_group.append(np.asarray(xb, float))

        fin = np.isfinite(y) & np.isfinite(xb)
        if not fin.any():
            continue
        any_data = True

        ax.bar(
            xb[fin],
            y[fin],
            width=bar_w[fin],
            align="center",
            alpha=0.75,
            linewidth=0.8,
            label=f"{group_labels[gi]} (n={ns[gi]})",
        )

        fin_ci = fin & np.isfinite(lo_i) & np.isfinite(hi_i)
        if fin_ci.any():
            yerr = np.vstack([y[fin_ci] - lo_i[fin_ci], hi_i[fin_ci] - y[fin_ci]])
            ax.errorbar(
                xb[fin_ci],
                y[fin_ci],
                yerr=yerr,
                fmt="none",
                elinewidth=1.1,
                capsize=2.0,
                alpha=0.9,
            )

    if not any_data:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        if use_edges:
            labels_xt = _edge_xticklabels(edges, overflow=overflow)
        else:
            labels_xt = []
            for j in range(K2):
                a = j * bin_w
                b = min(K, (j + 1) * bin_w) - 1
                labels_xt.append(str(a) if bin_w == 1 else f"{a}-{b}")
            if overflow and labels_xt:
                a_last = (K2 - 1) * bin_w
                labels_xt[-1] = f"{a_last}+"

        ax.set_xticks(x)
        ax.set_xticklabels(labels_xt)

        ax.set_xlabel(maybe_sentence_case(episode_label))
        ax.set_ylabel(maybe_sentence_case("mean probability"))
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.6, float(K2 - 1) + 0.6)

        do_stats = bool(getattr(opts, "wall_contacts_pmf_stats", False))
        alpha = float(getattr(opts, "wall_contacts_pmf_stats_alpha", 0.05) or 0.05)

        if do_stats:
            cfg_stats = StatAnnotConfig(
                alpha=alpha,
                min_n_per_group=3,
                nlabel_off_frac=0.0,
                headroom_frac=0.25,
            )
            annotate_grouped_bars_per_bin(
                ax,
                x_centers=x,
                xpos_by_group=xpos_by_group,
                per_unit_by_group=per_unit,
                hi_by_group=hi,
                group_names=[str(l) for l in group_labels],
                cfg=cfg_stats,
            )

        ax.legend(loc="best", fontsize=customizer.in_plot_font_size)
        ax.set_title(maybe_sentence_case(f"{episode_label} ({role})"))

    if customizer.font_size_customized:
        customizer.adjust_padding_proportionally()
    fig.tight_layout()
    writeImage(out_path, format=opts.imageFormat)
    plt.close(fig)
    print(f"[{log_tag}] wrote {out_path}")
