from __future__ import annotations

import numpy as np

from src.exporting.bundle_utils import save_metric_plus_sli_bundle


def _dbg(opts, msg: str) -> None:
    if opts is not None and bool(
        getattr(opts, "between_reward_maxdist_sli_debug", False)
    ):
        print(msg)


def _extract_between_reward_maxdist_arrays(vas, opts=None):
    """
    Returns:
      mean_exp:   (n_videos, n_trains, nb) float
      mean_ctrl:  (n_videos, n_trains, nb) float (all-NaN if absent)
      n_exp:      (n_videos, n_trains, nb) int
      n_ctrl:     (n_videos, n_trains, nb) int
    """
    n_videos = len(vas)
    va0 = vas[0]

    n_trains = len(getattr(va0, "trns", []) or [])
    nb = 0
    try:
        df = va0._numRewardsMsg(True, silent=True)
        for trn in getattr(va0, "trns", []) or []:
            _fi, n_buckets, _on = va0._syncBucket(trn, df)
            if n_buckets is not None:
                nb = max(nb, int(n_buckets))
    except Exception as exc:
        _dbg(
            opts,
            f"[between-reward-maxdist-export] WARNING: failed canonical bucket inference ({type(exc).__name__}: {exc})",
        )

    mean_exp = np.full((n_videos, n_trains, nb), np.nan, dtype=float)
    mean_ctrl = np.full((n_videos, n_trains, nb), np.nan, dtype=float)
    n_exp = np.zeros((n_videos, n_trains, nb), dtype=int)
    n_ctrl = np.zeros((n_videos, n_trains, nb), dtype=int)

    if n_trains == 0 or nb == 0:
        return mean_exp, mean_ctrl, n_exp, n_ctrl

    for vi, va in enumerate(vas):
        vid = getattr(va, "fn", f"va_{vi}")

        try:
            va.bySyncBucketMeanBetweenRewardMaxDist(
                exclude_wall_contact=bool(
                    getattr(
                        opts,
                        "between_reward_maxdist_exclude_wall_contact",
                        False,
                    )
                )
            )
        except Exception as exc:
            _dbg(
                opts,
                f"[between-reward-maxdist-export] {vid}: FAILED to compute bySyncBucketMeanBetweenRewardMaxDist ({type(exc).__name__}: {exc})",
            )
            continue

        means = getattr(va, "syncMeanBetweenRewardMaxDist", None)
        counts = getattr(va, "syncMeanBetweenRewardMaxDistN", None)
        if (
            not isinstance(means, list)
            or not isinstance(counts, list)
            or len(means) != n_trains
            or len(counts) != n_trains
        ):
            _dbg(
                opts,
                f"[between-reward-maxdist-export] {vid}: missing/invalid syncMeanBetweenRewardMaxDist(_N)",
            )
            continue

        for ti in range(n_trains):
            trn_means = means[ti] if isinstance(means[ti], dict) else {}
            trn_counts = counts[ti] if isinstance(counts[ti], dict) else {}

            exp_vals = np.asarray(trn_means.get("exp", []), dtype=float).reshape(-1)
            exp_n = np.asarray(trn_counts.get("exp", []), dtype=int).reshape(-1)
            ne = min(nb, exp_vals.size)
            if ne > 0:
                mean_exp[vi, ti, :ne] = exp_vals[:ne]
            ne_n = min(nb, exp_n.size)
            if ne_n > 0:
                n_exp[vi, ti, :ne_n] = exp_n[:ne_n]

            ctrl_vals = np.asarray(trn_means.get("ctrl", []), dtype=float).reshape(-1)
            ctrl_n = np.asarray(trn_counts.get("ctrl", []), dtype=int).reshape(-1)
            nc = min(nb, ctrl_vals.size)
            if nc > 0:
                mean_ctrl[vi, ti, :nc] = ctrl_vals[:nc]
            nc_n = min(nb, ctrl_n.size)
            if nc_n > 0:
                n_ctrl[vi, ti, :nc_n] = ctrl_n[:nc_n]

        _dbg(
            opts,
            f"[between-reward-maxdist-export] {vid}: exp finite={np.isfinite(mean_exp[vi]).sum()} / {mean_exp[vi].size}",
        )

    return mean_exp, mean_ctrl, n_exp, n_ctrl


def export_between_reward_maxdist_sli_bundle(vas, opts, gls, out_fn):
    def _extractor(vas_ok):
        mean_exp, mean_ctrl, n_exp, n_ctrl = _extract_between_reward_maxdist_arrays(
            vas_ok, opts
        )
        return {
            "between_reward_maxdist_exp": mean_exp,
            "between_reward_maxdist_ctrl": mean_ctrl,
            "between_reward_maxdistN_exp": n_exp,
            "between_reward_maxdistN_ctrl": n_ctrl,
        }

    save_metric_plus_sli_bundle(
        vas,
        opts,
        gls,
        out_fn,
        extract_metric_arrays=_extractor,
        bucket_type="bysb2",
        print_label="between_reward_maxdist",
    )
