from __future__ import annotations

import os
import numpy as np


def _dbg(opts, msg: str) -> None:
    if opts is not None and bool(
        getattr(opts, "btw_rwd_shortest_tail_export_debug", False)
    ):
        print(msg)


def _safe_group_label(opts, gls):
    """
    Best-effort group label for exports. Mirrors the logic you use elsewhere,
    but avoids importing other bundle helpers to keep this exporter standalone.
    """
    # If you already have a canonical helper elsewhere, feel free to replace this.
    try:
        if opts is not None and getattr(opts, "groupLabel", None):
            return str(opts.groupLabel)
    except Exception:
        pass
    try:
        # Often gls is a list of labels; exporter is called per run/group.
        if isinstance(gls, (list, tuple)) and len(gls) == 1:
            return str(gls[0])
    except Exception:
        pass
    try:
        if isinstance(gls, str):
            return gls
    except Exception:
        pass
    return "group"


def _ensure_metric(
    va, *, trainings, q, n_min, k_floor, calc: bool, ctrl: bool, opts=None
) -> bool:
    """
    Ensure per-training shortest-tail metric arrays exist on this VideoAnalysis.
    Returns True if metric appears available after this call.
    """
    meta = getattr(va, "shortestTailMeta", None)
    if (
        meta
        and meta.get("q") == q
        and meta.get("n_min") == n_min
        and meta.get("k_floor") == k_floor
        and meta.get("trainings") == trainings
        and meta.get("calc") == calc
        and meta.get("ctrl") == ctrl
    ):
        return True

    try:
        va.byShortestBetweenRewardDistances(
            trainings=trainings,
            q=q,
            n_min=n_min,
            k_floor=k_floor,
            calc=calc,
            ctrl=ctrl,
        )
        return hasattr(va, "shortestTailMeanDistByTrn") and hasattr(
            va, "shortestTailMeanDistByTrn_mean"
        )
    except Exception as e:
        _dbg(
            opts,
            f"[btw-rwd-shortest-tail-export] {getattr(va, 'fn', 'va')}: FAILED compute ({type(e).__name__}: {e})",
        )
        return False


def _extract_training_names(va0):
    try:
        return np.array([t.name() for t in va0.trns], dtype=object)
    except Exception:
        return np.array([], dtype=object)


def _extract_video_ids(vas_ok):
    video_ids = []
    try:
        for i, va in enumerate(vas_ok):
            va_fn = getattr(va, "fn", f"va_{i}")
            va_f = getattr(va, "f", None)
            video_ids.append(f"{va_fn}__vaF{va_f}" if va_f is not None else va_fn)
        return np.array(video_ids, dtype=object)
    except Exception:
        return np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)


def export_btw_rwd_shortest_tail_bundle(vas, opts, gls, out_fn: str) -> None:
    """
    Export per-training shortest-tail between-reward distance summaries as NPZ.

    Saves:
      shortest_tail_exp:   (n_videos, n_trn) float  (mean across flies in each VA)
      shortest_tail_ctrl:  (n_videos, n_trn) float  (NaN if ctrl not present)
      shortest_tailN_exp:  (n_videos, n_trn) int    (# contributing flies for that VA/training)
      shortest_tailN_ctrl: (n_videos, n_trn) int
      training_names:      (n_trn,) object
      video_ids:           (n_videos,) object
      group_label:         object scalar
      q, n_min, k_floor:   scalars
      calc, ctrl:          scalars (for reproducibility)
    """
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    va0 = vas_ok[0]
    n_trn = max(len(getattr(va, "trns", [])) for va in vas_ok)

    # Parameters (prefer CLI flags; fall back to defaults)
    trainings = getattr(opts, "btw_rwd_shortest_tail_trainings", None)
    q = float(getattr(opts, "btw_rwd_shortest_tail_q", 0.05))
    n_min = int(getattr(opts, "btw_rwd_shortest_tail_n_min", 15))
    k_floor = int(getattr(opts, "btw_rwd_shortest_tail_k_floor", 3))

    # In v1 we export the delivered-reward metric (calc=False).
    # If later it's useful to add a calculated-reward export too, another flag can be added.
    calc = False
    ctrl_stream = False  # only used when calc=True

    group_label = _safe_group_label(opts, gls)

    _dbg(opts, f"[btw-rwd-shortest-tail-export] out={out_fn}")
    _dbg(
        opts,
        f"[btw-rwd-shortest-tail-export] group_label={group_label!r} n_videos={len(vas_ok)} n_trn={n_trn}",
    )
    _dbg(
        opts,
        f"[btw-rwd-shortest-tail-export] params: q={q} n_min={n_min} k_floor={k_floor} trainings={trainings}",
    )

    n_videos = len(vas_ok)
    shortest_tail_exp = np.full((n_videos, n_trn), np.nan, dtype=float)
    shortest_tail_ctrl = np.full((n_videos, n_trn), np.nan, dtype=float)
    shortest_tailN_exp = np.zeros((n_videos, n_trn), dtype=int)
    shortest_tailN_ctrl = np.zeros((n_videos, n_trn), dtype=int)

    for vi, va in enumerate(vas_ok):
        va_fn = getattr(va, "fn", f"va_{vi}")
        va_f = getattr(va, "f", None)
        vid = f"{va_fn}__vaF{va_f}" if va_f is not None else va_fn

        ok = _ensure_metric(
            va,
            trainings=trainings,
            q=q,
            n_min=n_min,
            k_floor=k_floor,
            calc=calc,
            ctrl=ctrl_stream,
            opts=opts,
        )
        if not ok:
            _dbg(
                opts, f"[btw-rwd-shortest-tail-export] {vid}: missing metric; skipping"
            )
            continue

        per_trn = getattr(va, "shortestTailMeanDistByTrn", None)
        per_trn_n = getattr(va, "shortestTailMeanDistByTrn_n", None)
        per_trn_mean = getattr(va, "shortestTailMeanDistByTrn_mean", None)

        if not isinstance(per_trn, list):
            _dbg(
                opts,
                f"[btw-rwd-shortest-tail-export] {vid}: invalid shortestTailMeanDistByTrn; skipping",
            )
            continue

        # Prefer per-training means/counts if available; otherwise compute from per-fly arrays.
        for ti in range(min(n_trn, len(per_trn))):
            vals = per_trn[ti]
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0:
                continue

            # exp: mean across flies for this VA/training
            if (
                isinstance(per_trn_mean, list)
                and ti < len(per_trn_mean)
                and np.isfinite(per_trn_mean[ti])
            ):
                shortest_tail_exp[vi, ti] = float(per_trn_mean[ti])
            else:
                if np.isfinite(vals).any():
                    shortest_tail_exp[vi, ti] = float(np.nanmean(vals))

            if isinstance(per_trn_n, list) and ti < len(per_trn_n):
                shortest_tailN_exp[vi, ti] = int(per_trn_n[ti])
            else:
                shortest_tailN_exp[vi, ti] = int(np.isfinite(vals).sum())

        _dbg(
            opts,
            f"[btw-rwd-shortest-tail-export] {vid}: exp finite={np.isfinite(shortest_tail_exp[vi]).sum()} / {shortest_tail_exp.shape[1]}",
        )

        # ctrl stream: only relevant if exporting calc=True, ctrl=True later.
        # For now, it stays NaN/0.

    training_names = _extract_training_names(va0)
    video_ids = _extract_video_ids(vas_ok)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        btw_rwd_shortest_tail_exp=shortest_tail_exp,
        btw_rwd_shortest_tail_ctrl=shortest_tail_ctrl,
        btw_rwd_shortest_tailN_exp=shortest_tailN_exp,
        btw_rwd_shortest_tailN_ctrl=shortest_tailN_ctrl,
        group_label=np.array([group_label], dtype=object),
        training_names=training_names,
        video_ids=video_ids,
        q=np.array(q, dtype=float),
        n_min=np.array(n_min, dtype=int),
        k_floor=np.array(k_floor, dtype=int),
        calc=np.array(bool(calc), dtype=bool),
        ctrl=np.array(bool(ctrl_stream), dtype=bool),
        trainings=np.array(trainings if trainings is not None else [], dtype=int),
    )
    print(f"[export] Wrote btw-rwd-shortest-tail bundle: {out_fn} (n={len(vas_ok)})")
