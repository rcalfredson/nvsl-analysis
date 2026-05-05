from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.exporting.bundle_utils import (
    build_metric_plus_sli_bundle,
    save_sli_bundle,
)
from src.plotting.btw_rwd_return_leg_dist_collectors import ReturnLegDistPerFlyCollector


def _parse_debug_bucket(raw) -> tuple[int, int] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace(":", ",").split(",")]
    if len(parts) != 2:
        raise ValueError(
            "--btw-rwd-return-leg-dist-sli-debug-bucket must be TRN,BKT, "
            f"got {raw!r}"
        )
    try:
        trn_1 = int(parts[0])
        bkt_1 = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            "--btw-rwd-return-leg-dist-sli-debug-bucket must contain integers, "
            f"got {raw!r}"
        ) from exc
    if trn_1 < 1 or bkt_1 < 1:
        raise ValueError(
            "--btw-rwd-return-leg-dist-sli-debug-bucket uses 1-based positive "
            f"indices, got {raw!r}"
        )
    return trn_1 - 1, bkt_1 - 1


def _video_label(va, fallback: str) -> str:
    fn = getattr(va, "fn", None)
    return str(fn) if fn else str(fallback)


def _va_tag(va) -> int:
    return int(getattr(va, "f", 0) or 0)


def _print_debug_bucket_rows(
    *,
    vas,
    opts,
    mean_exp: np.ndarray,
    mean_ctrl: np.ndarray,
    n_exp: np.ndarray,
    n_ctrl: np.ndarray,
) -> None:
    selected = _parse_debug_bucket(
        getattr(opts, "btw_rwd_return_leg_dist_sli_debug_bucket", None)
    )
    if selected is None:
        return

    trn_idx, bkt_idx = selected
    group_label = str(getattr(opts, "export_group_label", "") or "group")
    condition = str(
        getattr(opts, "btw_rwd_return_leg_dist_sli_debug_condition", "exp") or "exp"
    ).lower()
    conditions = ["exp", "ctrl"] if condition == "both" else [condition]
    top_n = max(
        0, int(getattr(opts, "btw_rwd_return_leg_dist_sli_debug_top_n", 0) or 0)
    )

    arrays = {
        "exp": (np.asarray(mean_exp, dtype=float), np.asarray(n_exp, dtype=int)),
        "ctrl": (np.asarray(mean_ctrl, dtype=float), np.asarray(n_ctrl, dtype=int)),
    }

    for cond in conditions:
        vals, counts = arrays[cond]
        if trn_idx >= vals.shape[1] or bkt_idx >= vals.shape[2]:
            print(
                "[btw_rwd_return_leg_dist_sli_debug] "
                f"{group_label} {cond}: requested T{trn_idx + 1} bucket {bkt_idx + 1}, "
                f"but array shape is {vals.shape}; skipping."
            )
            continue

        rows = []
        for vi, va in enumerate(vas):
            val = float(vals[vi, trn_idx, bkt_idx])
            n_seg = int(counts[vi, trn_idx, bkt_idx])
            rows.append(
                {
                    "va.f": _va_tag(va),
                    "video filename": _video_label(va, f"va_{vi}"),
                    "value_mm": val,
                    "n_segments": n_seg,
                }
            )

        finite_rows = [r for r in rows if np.isfinite(r["value_mm"])]
        finite_rows.sort(key=lambda r: r["value_mm"], reverse=True)
        missing_n = len(rows) - len(finite_rows)
        shown_rows = finite_rows[:top_n] if top_n > 0 else finite_rows

        header = (
            "[btw_rwd_return_leg_dist_sli_debug] "
            f"{group_label} {cond} T{trn_idx + 1} sync_bucket {bkt_idx + 1}: "
            f"{len(finite_rows)}/{len(rows)} finite"
        )
        if missing_n:
            header += f", {missing_n} NaN/missing"
        if top_n > 0:
            header += f", showing top {len(shown_rows)}"
        print(header)
        print(
            "[btw_rwd_return_leg_dist_sli_debug] "
            "rank\tvalue_mm\tn_segments\tvideo filename\tva.f"
        )
        for rank, row in enumerate(shown_rows, start=1):
            print(
                "[btw_rwd_return_leg_dist_sli_debug] "
                f"{rank}\t{row['value_mm']:.6g}\t{row['n_segments']}\t"
                f"{row['video filename']}\t{row['va.f']}"
            )


def _extract_btw_rwd_return_leg_dist_arrays(vas, opts):
    collector = ReturnLegDistPerFlyCollector()
    collector.vas = vas
    collector.opts = opts
    collector.cfg = SimpleNamespace(
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    mean_exp, mean_ctrl, n_exp, n_ctrl = collector.collect_return_leg_sync_bucket_arrays()
    _print_debug_bucket_rows(
        vas=vas,
        opts=opts,
        mean_exp=mean_exp,
        mean_ctrl=mean_ctrl,
        n_exp=n_exp,
        n_ctrl=n_ctrl,
    )
    return mean_exp, mean_ctrl, n_exp, n_ctrl



def build_btw_rwd_return_leg_dist_sli_bundle(vas, opts, gls) -> dict:
    def _extractor(vas_ok):
        mean_exp, mean_ctrl, n_exp, n_ctrl = _extract_btw_rwd_return_leg_dist_arrays(
            vas_ok, opts
        )
        return {
            "between_reward_return_leg_dist_exp": mean_exp,
            "between_reward_return_leg_dist_ctrl": mean_ctrl,
            "between_reward_return_leg_distN_exp": n_exp,
            "between_reward_return_leg_distN_ctrl": n_ctrl,
        }

    return build_metric_plus_sli_bundle(
        vas,
        opts,
        gls,
        extract_metric_arrays=_extractor,
        bucket_type="bysb2",
        print_label="between_reward_return_leg_dist",
    )


def export_btw_rwd_return_leg_dist_sli_bundle(vas, opts, gls, out_fn):
    bundle = build_btw_rwd_return_leg_dist_sli_bundle(vas, opts, gls)
    save_sli_bundle(bundle, out_fn, print_label="between_reward_return_leg_dist")
