from src.plotting.first_n_reward_diagnostics import (
    FirstNRewardDiagnosticsConfig,
    FirstNRewardDiagnosticsPlotter,
)


def _plotter(cfg: FirstNRewardDiagnosticsConfig) -> FirstNRewardDiagnosticsPlotter:
    return FirstNRewardDiagnosticsPlotter(vas=[], opts=None, gls=None, cfg=cfg)


def test_sli_axis_label_uses_sli_selection_window_for_mean_over_first_bucket():
    cfg = FirstNRewardDiagnosticsConfig(
        csv_out="",
        trainings=(1,),
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
        sli_training_idx=0,
        sli_average_over_buckets=True,
        sli_skip_first_sync_buckets=0,
        sli_keep_first_sync_buckets=1,
    )

    assert _plotter(cfg)._sli_axis_label() == "SLI (T1, SB1)"


def test_sli_axis_label_uses_sli_training_context_not_reward_window():
    cfg = FirstNRewardDiagnosticsConfig(
        csv_out="",
        trainings=(2,),
        skip_first_sync_buckets=3,
        keep_first_sync_buckets=2,
        sli_training_idx=0,
        sli_average_over_buckets=True,
        sli_skip_first_sync_buckets=1,
        sli_keep_first_sync_buckets=4,
    )

    assert _plotter(cfg)._sli_axis_label() == "SLI (T1, mean, SB2-SB5)"
