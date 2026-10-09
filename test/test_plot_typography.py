import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.plotting.plot_customizer import PlotCustomizer, TIME_PLOT_STATS_FONT_SCALE


TIME_PLOT_FONT_SIZE = 27.0
CORRELATION_PLOT_FONT_SIZE = 20.0


def _rendered_font_sizes(base_font_size):
    """Return the text-role sizes produced by the shared plot customizer."""
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(base_font_size)

        fig, ax = plt.subplots()
        try:
            ax.plot([0, 1], [0, 1], label="group")
            ax.set_xlabel("x label")
            ax.set_ylabel("y label")
            ax.set_title("title")
            ordinary_text = ax.text(0.5, 0.5, "annotation")
            legend = ax.legend()
            fig.canvas.draw()

            return {
                "title": float(ax.title.get_fontsize()),
                "axis_label": float(ax.xaxis.label.get_fontsize()),
                "x_tick_label": float(ax.get_xticklabels()[0].get_fontsize()),
                "y_tick_label": float(ax.get_yticklabels()[0].get_fontsize()),
                "legend": float(legend.get_texts()[0].get_fontsize()),
                "ordinary_text": float(ordinary_text.get_fontsize()),
                "rc_font_size": float(plt.rcParams["font.size"]),
            }
        finally:
            plt.close(fig)


def test_time_plot_reference_typography_is_preserved():
    """The new proportional scale must not disturb the accepted time style."""
    sizes = _rendered_font_sizes(TIME_PLOT_FONT_SIZE)

    assert sizes["title"] == pytest.approx(30.0)
    assert sizes["axis_label"] == pytest.approx(29.0)
    assert sizes["x_tick_label"] == pytest.approx(25.0)
    assert sizes["y_tick_label"] == pytest.approx(25.0)
    assert sizes["legend"] == pytest.approx(24.0)


@pytest.mark.parametrize("baseline", [20, 26.5, 27, 30])
def test_plot_rewards_auc_and_abc_use_time_stats_ratio(baseline):
    import ast
    from pathlib import Path
    from src.plotting.annotation_layout import tick_label_fontsize

    source = ast.parse((Path(__file__).resolve().parents[1] / "analyze.py").read_text())
    plot_rewards = next(node for node in source.body
                        if isinstance(node, ast.FunctionDef) and node.name == "plotRewards")
    size_expressions = [keyword.value
                        for node in ast.walk(plot_rewards)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "_add_auc_text"
                        for keyword in node.keywords if keyword.arg == "size"]
    assert len(size_expressions) == 2
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(baseline)
        fig, ax = plt.subplots()
        try:
            namespace = dict(ax=ax, TIME_PLOT_STATS_FONT_SCALE=TIME_PLOT_STATS_FONT_SCALE,
                             customizer=customizer, pch=lambda legacy, current: current)
            for expression in size_expressions:
                size = eval(compile(ast.Expression(expression), "analyze.py", "eval"), namespace)
                label = ax.text(0.5, 0.5, "AUC/ABC", fontsize=size)
                assert label.get_fontsize() == pytest.approx(baseline * 24 / 27)
                assert tick_label_fontsize(ax) == pytest.approx(baseline * 25 / 27)
        finally:
            plt.close(fig)


def test_correlation_and_time_plots_share_text_role_proportions():
    """Changing the base size must not change the visual text hierarchy."""
    time_sizes = _rendered_font_sizes(TIME_PLOT_FONT_SIZE)
    correlation_sizes = _rendered_font_sizes(CORRELATION_PLOT_FONT_SIZE)

    for role in ("title", "axis_label", "x_tick_label", "y_tick_label", "legend"):
        assert correlation_sizes[role] / CORRELATION_PLOT_FONT_SIZE == pytest.approx(
            time_sizes[role] / TIME_PLOT_FONT_SIZE
        )


@pytest.mark.parametrize("base_font_size", [CORRELATION_PLOT_FONT_SIZE, TIME_PLOT_FONT_SIZE])
def test_configured_base_size_applies_to_ordinary_plot_text(base_font_size):
    """Unclassified annotations should not retain Matplotlib's default size."""
    sizes = _rendered_font_sizes(base_font_size)

    assert sizes["rc_font_size"] == pytest.approx(base_font_size)
    assert sizes["ordinary_text"] == pytest.approx(base_font_size)
