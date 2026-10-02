import pytest

from src.plotting.cross_fly_correlations import (
    SLIContext,
    _reward_rate_vs_sli_plot_key,
)
from src.plotting.palettes import CORRELATION_PLOT_COLORS, correlation_plot_color


def test_pre_training_speed_mean_t2_sli_has_unique_correlation_color():
    plot_key = "pre_training_speed_vs_mean_t2_sli"
    color = correlation_plot_color(plot_key)

    assert color not in {
        other_color
        for other_key, other_color in CORRELATION_PLOT_COLORS.items()
        if other_key != plot_key and not other_key.startswith("unused_")
    }


def test_concurrent_t1_sb1_reward_rate_has_unique_correlation_color():
    plot_key = "rewards_per_minute_vs_sli_t1_sb1"
    assert correlation_plot_color(plot_key) not in {
        color for key, color in CORRELATION_PLOT_COLORS.items() if key != plot_key
    }


@pytest.mark.parametrize(
    "ctx",
    [
        SLIContext(training_idx=0, explicit_bucket_idx=0),
        SLIContext(training_idx=0, keep_first_sync_buckets=1),
        SLIContext(
            training_idx=0, average_over_buckets=True, keep_first_sync_buckets=1
        ),
    ],
)
def test_equivalent_t1_sb1_selections_share_color(ctx):
    explicit = SLIContext(training_idx=0, explicit_bucket_idx=0)
    assert _reward_rate_vs_sli_plot_key(ctx, explicit) == (
        "rewards_per_minute_vs_sli_t1_sb1"
    )
    assert _reward_rate_vs_sli_plot_key(explicit, ctx) == (
        "rewards_per_minute_vs_sli_t1_sb1"
    )


@pytest.mark.parametrize(
    "other_ctx",
    [
        SLIContext(training_idx=1, explicit_bucket_idx=0),
        SLIContext(training_idx=0, explicit_bucket_idx=4),
        SLIContext(training_idx=0, total_sync_buckets=5),
        SLIContext(training_idx=0, average_over_buckets=True, keep_first_sync_buckets=5),
    ],
)
def test_other_or_mismatched_windows_keep_reward_rate_color(other_ctx):
    early = SLIContext(training_idx=0, explicit_bucket_idx=0)
    assert _reward_rate_vs_sli_plot_key(early, other_ctx) == "rewards_per_minute_vs_sli"
    assert _reward_rate_vs_sli_plot_key(other_ctx, early) == "rewards_per_minute_vs_sli"


def test_first_n_reward_rate_keeps_existing_color():
    early = SLIContext(training_idx=0, explicit_bucket_idx=0)
    assert _reward_rate_vs_sli_plot_key(early, early, first_n_rewards=10) == (
        "rewards_per_minute_vs_sli"
    )
