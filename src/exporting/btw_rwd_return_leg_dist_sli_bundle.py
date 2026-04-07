from __future__ import annotations

from types import SimpleNamespace

from src.exporting.bundle_utils import save_metric_plus_sli_bundle
from src.plotting.btw_rwd_return_leg_dist_collectors import ReturnLegDistPerFlyCollector


def _extract_btw_rwd_return_leg_dist_arrays(vas, opts):
    collector = ReturnLegDistPerFlyCollector()
    collector.vas = vas
    collector.opts = opts
    collector.cfg = SimpleNamespace(
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    return collector.collect_return_leg_sync_bucket_arrays()


def export_btw_rwd_return_leg_dist_sli_bundle(vas, opts, gls, out_fn):
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

    save_metric_plus_sli_bundle(
        vas,
        opts,
        gls,
        out_fn,
        extract_metric_arrays=_extractor,
        bucket_type="bysb2",
        print_label="between_reward_return_leg_dist",
    )
