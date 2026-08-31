from src.plotting.palettes import CORRELATION_PLOT_COLORS, correlation_plot_color


def test_pre_training_speed_mean_t2_sli_has_unique_correlation_color():
    plot_key = "pre_training_speed_vs_mean_t2_sli"
    color = correlation_plot_color(plot_key)

    assert color not in {
        other_color
        for other_key, other_color in CORRELATION_PLOT_COLORS.items()
        if other_key != plot_key and not other_key.startswith("unused_")
    }
