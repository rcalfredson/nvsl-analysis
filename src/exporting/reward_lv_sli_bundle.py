# src/exporting/reward_lv_sli_bundle.py
from __future__ import annotations

import numpy as np

from src.exporting.bundle_utils import save_metric_plus_sli_bundle


def _extract_reward_lv_arrays(vas):
    """
    Returns:
      reward_lv_exp: (n_videos, n_trains, nb) float with NaNs allowed
      reward_lv_ctrl: (n_videos, n_trains, nb) float with NaNs allowed (all-NaN if absent)
    """
    n_videos = len(vas)

    # Find a reference VA that actually has syncRewardLV
    ref = None
    for va in vas:
        srlv = getattr(va, "syncRewardLV", None)
        if (
            srlv is not None
            and isinstance(srlv, (list, tuple))
            and len(srlv)
            and isinstance(srlv[0], dict)
        ):
            ref = va
            break

    if ref is None:
        return (
            np.full((n_videos, 0, 0), np.nan, dtype=float),
            np.full((n_videos, 0, 0), np.nan, dtype=float),
        )

    srlv_ref = getattr(ref, "syncRewardLV", []) or []
    n_trains = len(srlv_ref)

    # Infer nb by scanning for the first non-empty exp list.
    nb = 0
    for ti in range(n_trains):
        trn_dict = srlv_ref[ti] if isinstance(srlv_ref[ti], dict) else {}
        exp_vals = trn_dict.get("exp", None)
        if exp_vals is not None:
            try:
                nb = len(exp_vals)
            except Exception:
                nb = 0
        if nb > 0:
            break

    # Defensive fallback: scan other VAs in case ref has no full buckets
    if nb == 0:
        for va in vas:
            srlv = getattr(va, "syncRewardLV", None)
            if srlv is None or len(srlv) != n_trains:
                continue
            for ti in range(n_trains):
                trn_dict = srlv[ti] if isinstance(srlv[ti], dict) else {}
                exp_vals = trn_dict.get("exp", None)
                if exp_vals is None:
                    continue
                try:
                    nb = len(exp_vals)
                except Exception:
                    nb = 0
                if nb > 0:
                    break
            if nb > 0:
                break

    reward_lv_exp = np.full((n_videos, n_trains, nb), np.nan, dtype=float)
    reward_lv_ctrl = np.full((n_videos, n_trains, nb), np.nan, dtype=float)

    if nb == 0 or n_trains == 0:
        return reward_lv_exp, reward_lv_ctrl

    for vi, va in enumerate(vas):
        srlv = getattr(va, "syncRewardLV", None)
        if srlv is None or len(srlv) != n_trains:
            continue

        for ti in range(n_trains):
            trn_dict = srlv[ti] if isinstance(srlv[ti], dict) else {}

            exp_vals = trn_dict.get("exp", None)
            if exp_vals is not None:
                exp_vals = np.asarray(exp_vals, dtype=float).reshape(-1)
                reward_lv_exp[vi, ti, : min(nb, exp_vals.size)] = exp_vals[:nb]

            ctrl_vals = trn_dict.get("ctrl", None)
            if ctrl_vals is not None:
                ctrl_vals = np.asarray(ctrl_vals, dtype=float).reshape(-1)
                reward_lv_ctrl[vi, ti, : min(nb, ctrl_vals.size)] = ctrl_vals[:nb]

    return reward_lv_exp, reward_lv_ctrl


def export_reward_lv_sli_bundle(vas, opts, gls, out_fn):
    def _extractor(vas_ok):
        exp, ctrl = _extract_reward_lv_arrays(vas_ok)
        return {
            "reward_lv_exp": exp,
            "reward_lv_ctrl": ctrl,
        }

    save_metric_plus_sli_bundle(
        vas,
        opts,
        gls,
        out_fn,
        extract_metric_arrays=_extractor,
        bucket_type="rlv",
        print_label="reward_lv",
    )
