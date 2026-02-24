from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, ttest_rel

from src.utils.util import p2stars


def _match_by_id(
    a: np.ndarray, a_ids: np.ndarray, b: np.ndarray, b_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # map id -> value
    da = {
        str(i): float(v) for i, v in zip(a_ids, a) if np.isfinite(v) and i is not None
    }
    db = {
        str(i): float(v) for i, v in zip(b_ids, b) if np.isfinite(v) and i is not None
    }
    common = sorted(set(da.keys()) & set(db.keys()))
    if not common:
        return np.asarray([], float), np.asarray([], float)
    xa = np.asarray([da[k] for k in common], float)
    xb = np.asarray([db[k] for k in common], float)
    return xa, xb


def holm_adjust(pvals: list[float]) -> list[float]:
    """
    Holm–Bonferroni adjustment.

    - Adjusts only finite p-values.
    - Preserves NaN positions as-is, and converts
      inf → NaN.
    """
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return []

    out = np.full_like(p, np.nan)  # default to NaN; we'll fill finite slots

    finite_mask = np.isfinite(p)
    if not np.any(finite_mask):
        # all NaN/inf -> preserve as NaN (or you could copy infs, but NaN is usually best)
        return out.tolist()

    pf = p[finite_mask]

    # Optional: clamp tiny numerical negatives and >1 values into [0, 1]
    # (helpful if upstream occasionally produces -0.0 or 1.0000000002)
    pf = np.clip(pf, 0.0, 1.0)

    m = pf.size
    order = np.argsort(pf)
    p_sorted = pf[order]

    adj_sorted = np.empty_like(p_sorted)
    running_max = 0.0
    for i, pv in enumerate(p_sorted):
        factor = m - i
        adj = float(factor * pv)
        if adj > running_max:
            running_max = adj
        adj_sorted[i] = min(1.0, running_max)

    # unsort back to the finite subset, then scatter back into full output
    adj_f = np.empty_like(adj_sorted)
    adj_f[order] = adj_sorted
    out[finite_mask] = adj_f

    return out.tolist()


def anova_and_posthoc(
    group_samples: list[np.ndarray],
    *,
    cfg: StatAnnotConfig,
    group_names: list[str],
    paired: bool = False,
    group_ids: list[np.ndarray | None] | None = None,
    debug: bool = False,
    debug_ctx: str = "",
) -> tuple[float, dict[tuple[str, str], float]]:
    # ANOVA
    try:
        clean = [g[np.isfinite(g)] for g in group_samples]
        clean = [g for g in clean if g.size >= cfg.min_n_per_group]
        if len(clean) >= 2:
            _, p_anova = f_oneway(*clean)
        else:
            p_anova = np.nan
    except Exception:
        p_anova = np.nan

    # Pairwise Welch t-tests
    pairs: list[tuple[int, int]] = []
    p_raw: list[float] = []
    G = len(group_samples)

    for i in range(G):
        for j in range(i + 1, G):
            p = np.nan
            try:
                if paired:
                    if (
                        group_ids is None
                        or group_ids[i] is None
                        or group_ids[j] is None
                    ):
                        p = np.nan
                    else:
                        xi, xj = _match_by_id(
                            np.asarray(group_samples[i], float),
                            np.asarray(group_ids[i], dtype=object),
                            np.asarray(group_samples[j], float),
                            np.asarray(group_ids[j], dtype=object),
                        )
                        n_overlap = int(xi.size)
                        n_i = int(
                            np.isfinite(np.asarray(group_samples[i], float)).sum()
                        )
                        n_j = int(
                            np.isfinite(np.asarray(group_samples[j], float)).sum()
                        )
                        used = (
                            "paired"
                            if (n_overlap >= cfg.min_n_per_group)
                            else "welch_fallback"
                        )

                        if debug and used != "paired":
                            print(
                                f"[stats] {debug_ctx} {group_names[i]} vs {group_names[j]}: "
                                f"paired overlap n={n_overlap} (A n={n_i}, B n={n_j}) -> Welch"
                            )

                        if used == "paired":
                            _, p = ttest_rel(xi, xj, nan_policy="omit")
                        else:
                            # fallback to Welch if insufficient pairs
                            _, p = ttest_ind(
                                group_samples[i],
                                group_samples[j],
                                equal_var=False,
                                nan_policy="omit",
                            )
                else:
                    _, p = ttest_ind(
                        group_samples[i],
                        group_samples[j],
                        equal_var=False,
                        nan_policy="omit",
                    )
            except Exception:
                p = np.nan
            pairs.append((i, j))
            p_raw.append(float(p))

    # Holm correction is per-bin across group pairs; no correction across bins
    p_adj = holm_adjust(p_raw)

    out: dict[tuple[str, str], float] = {}
    for (i, j), pa in zip(pairs, p_adj):
        out[(group_names[i], group_names[j])] = float(pa)
    return float(p_anova), out


def draw_sig_bracket(
    ax: plt.Axes,
    *,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    lw: float = 1.2,
    color: str = "0.15",
    zorder: int = 10,
) -> None:
    if not text:
        return
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        linewidth=lw,
        color=color,
        zorder=zorder,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2.0,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
        zorder=zorder + 1,
        clip_on=False,
    )


@dataclass(frozen=True)
class StatAnnotConfig:
    alpha: float = 0.05
    min_n_per_group: int = 3
    headroom_frac: float = 0.25  # add to y-lim to avoid clipping
    bracket_h_frac: float = 0.012
    stack_gap_frac: float = 0.060
    gap_above_bars_frac: float = 0.045
    nlabel_off_frac: float = 0.04  # set 0.0 if no n labels


def annotate_grouped_bars_per_bin(
    ax: plt.Axes,
    *,
    x_centers: np.ndarray,  # (B,)
    xpos_by_group: list[np.ndarray],  # list of (B,) x positions
    per_unit_by_group: list[np.ndarray | None],  # list of (N_g, B)
    per_unit_ids_by_group: list[np.ndarray | None] | None = None,
    hi_by_group: list[np.ndarray],  # list of (B,) upper CI (or bar tops)
    group_names: list[str],
    cfg: StatAnnotConfig,
    paired: bool = False,
    panel_label: str | None = None,
    debug: bool = False,
) -> None:
    # Expand ylim for headroom
    ylim0, ylim1 = ax.get_ylim()
    y_rng0 = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    ax.set_ylim(ylim0, ylim1 + cfg.headroom_frac * y_rng0)

    B = int(x_centers.size)
    if any(x.shape[0] != B for x in xpos_by_group):
        return
    if any(h.shape[0] != B for h in hi_by_group):
        return

    ylim0, ylim1 = ax.get_ylim()
    y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    bracket_h = cfg.bracket_h_frac * y_rng
    step = bracket_h + cfg.stack_gap_frac * y_rng

    gidx = {name: i for i, name in enumerate(group_names)}
    for j in range(B):
        # Collect samples for this bin
        samples: list[np.ndarray] = []
        ids_for_samples: list[np.ndarray | None] = []
        ok = True

        for gi, pu in enumerate(per_unit_by_group):
            if pu is None:
                ok = False
                break
            v = np.asarray(pu[:, j], float)
            mask = np.isfinite(v)
            v = v[mask]
            if v.size < cfg.min_n_per_group:
                ok = False
                break
            samples.append(v)

            if per_unit_ids_by_group is not None:
                ids = per_unit_ids_by_group[gi]
                if ids is None:
                    ids_for_samples.append(None)
                else:
                    ids = np.asarray(ids, dtype=object).ravel()
                    ids_for_samples.append(ids[mask])
            else:
                ids_for_samples.append(None)
        if not ok:
            continue

        debug_ctx = ""
        if debug:
            debug_ctx = (
                f"panel={panel_label} bin={j}"
                if panel_label is not None
                else f"bin={j}"
            )

        _p_anova, p_adj_pairs = anova_and_posthoc(
            samples,
            cfg=cfg,
            group_names=group_names,
            paired=paired,
            group_ids=ids_for_samples if paired else None,
            debug=debug,
            debug_ctx=debug_ctx,
        )

        sig_pairs = [
            (pair, p)
            for pair, p in p_adj_pairs.items()
            if np.isfinite(p) and p < cfg.alpha
        ]
        if not sig_pairs:
            continue

        # baseline from CI tops
        y_base = np.nan
        for gi in range(len(hi_by_group)):
            y_top = (
                float(hi_by_group[gi][j]) if np.isfinite(hi_by_group[gi][j]) else np.nan
            )
            if np.isfinite(y_top):
                y_base = y_top if not np.isfinite(y_base) else max(y_base, y_top)
        if not np.isfinite(y_base):
            continue

        # Put brackets above n-label region if you have n labels; otherwise set nlabel_off_frac=0
        label_off = cfg.nlabel_off_frac * y_rng
        gap = cfg.gap_above_bars_frac * y_rng
        bracket_base = float(y_base + label_off + gap)

        # Optional: stack narrow first, wide last
        def _span(pair_item):
            (a, b), _p = pair_item
            i = gidx[a]
            k = gidx[b]
            return abs(float(xpos_by_group[i][j] - xpos_by_group[k][j]))

        sig_pairs = sorted(sig_pairs, key=_span)

        level = 0
        for (name_i, name_j), p in sig_pairs:
            i = gidx[name_i]
            k = gidx[name_j]

            x1 = float(xpos_by_group[i][j])
            x2 = float(xpos_by_group[k][j])
            if x2 < x1:
                x1, x2 = x2, x1

            stars = p2stars(p)
            if not stars:
                continue

            y = float(bracket_base + level * step)
            draw_sig_bracket(ax, x1=x1, x2=x2, y=y, h=bracket_h, text=stars)
            level += 1
