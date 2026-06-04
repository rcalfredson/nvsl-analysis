from src.plotting.first_n_reward_diagnostics import (
    FirstNRewardDiagnosticsConfig,
    FirstNRewardDiagnosticsPlotter,
)
from src.plotting.cross_fly_correlations import (
    SLIContext,
    _format_labeled_corr_with_n,
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

    assert _plotter(cfg)._sli_axis_label() == "Mean SLI over T1 SB1"


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

    assert _plotter(cfg)._sli_axis_label() == "Mean SLI over T1 SB2–5"


def test_sli_axis_label_uses_total_bucket_count_when_keep_is_not_explicit():
    cfg = FirstNRewardDiagnosticsConfig(
        csv_out="",
        trainings=(2,),
        sli_training_idx=1,
        sli_average_over_buckets=True,
        sli_skip_first_sync_buckets=1,
        sli_keep_first_sync_buckets=0,
        sli_total_sync_buckets=6,
    )

    assert _plotter(cfg)._sli_axis_label() == "Mean SLI over T2 SB2–5"


def test_cross_fly_sli_context_axis_label_uses_explicit_mean_window_range():
    ctx = SLIContext(
        training_idx=1,
        average_over_buckets=True,
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=4,
    )

    assert ctx.axis_label() == "Mean SLI over T2 SB2–5"
    assert ctx.label_short() == "SLI (T2, mean, SB2-SB5)"


def test_cross_fly_sli_context_axis_label_uses_total_bucket_count_without_keep():
    ctx = SLIContext(
        training_idx=1,
        average_over_buckets=True,
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=0,
        total_sync_buckets=6,
    )

    assert ctx.axis_label() == "Mean SLI over T2 SB2–5"
    assert ctx.label_short() == "SLI (T2, mean, SB2-SB5)"


def test_fast_strong_correlation_annotation_places_n_in_label():
    assert (
        _format_labeled_corr_with_n(0.741, 2.33e-16, 87, label="All flies")
        == "All flies (n = 87): r = 0.741, p = 2.33e-16"
    )
