from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import src.plotting.first_n_reward_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def restore_matplotlib_rcparams():
    with diagnostics.plt.rc_context():
        yield


def _axis_size_inches(ax):
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted()
    )
    return bbox.width, bbox.height


def test_first_n_plot_uses_injected_typography(tmp_path, monkeypatch):
    cfg = diagnostics.FirstNRewardDiagnosticsConfig(
        csv_out="",
        plot_out=str(tmp_path / "first_n.png"),
        reward_event_type="calc",
        x_by="selected_reward_rate_to_nth_per_min",
        y_by="sli",
        color_by=None,
    )
    customizer = diagnostics.PlotCustomizer()
    customizer.update_font_size(20.0)
    plotter = diagnostics.FirstNRewardDiagnosticsPlotter(
        vas=[], opts=None, gls=None, cfg=cfg, customizer=customizer
    )
    rows = [
        SimpleNamespace(
            eligible_for_nth_reward_cutoff=True,
            first_n_selected_reward_span_s=90.0,
            time_to_nth_selected_reward_s=95.0,
            selected_reward_rate_to_nth_per_min=x,
            sli=y,
        )
        for x, y in ((4.0, 0.2), (6.0, 0.6), (8.0, 1.1))
    ]

    close_figure = diagnostics.plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda _fig: None)
    with diagnostics.plt.rc_context():
        diagnostics.plt.rcParams.update(
            {
                "font.size": 7.0,
                "axes.titlesize": 7.0,
                "axes.labelsize": 7.0,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
                "legend.fontsize": 7.0,
            }
        )
        plotter._write_plot(rows)

        fig = diagnostics.plt.gcf()
        ax = fig.axes[0]
        assert ax.title.get_fontsize() == pytest.approx(20.0 * 30.0 / 27.0)
        assert ax.xaxis.label.get_fontsize() == pytest.approx(20.0 * 29.0 / 27.0)
        assert ax.get_xticklabels()[0].get_fontsize() == pytest.approx(
            20.0 * 25.0 / 27.0
        )
        close_figure(fig)


@pytest.mark.parametrize("font_size", [12.0, 23.0])
def test_first_n_plot_constrains_stats_box_after_final_axis_sizing(
    tmp_path, monkeypatch, font_size
):
    cfg = diagnostics.FirstNRewardDiagnosticsConfig(
        csv_out="",
        plot_out=str(tmp_path / "first_n.png"),
        reward_event_type="calc",
        x_by="selected_reward_rate_to_nth_per_min",
        y_by="sli",
        color_by=None,
    )
    customizer = diagnostics.PlotCustomizer()
    customizer.update_font_size(font_size)
    plotter = diagnostics.FirstNRewardDiagnosticsPlotter(
        vas=[], opts=None, gls=None, cfg=cfg, customizer=customizer
    )
    monkeypatch.setattr(
        plotter, "_correlation_text",
        lambda *_args: r"n = 89, r = 0.424, p = $3.54 \times 10^{-5}$",
    )
    rows = [
        SimpleNamespace(
            eligible_for_nth_reward_cutoff=True,
            first_n_selected_reward_span_s=90.0,
            time_to_nth_selected_reward_s=95.0,
            selected_reward_rate_to_nth_per_min=x,
            sli=y,
        )
        for x, y in ((4.0, 0.2), (6.0, 0.6), (8.0, 1.1))
    ]

    calls = []
    original = diagnostics._add_smart_stats_box

    def recorded_constraint(ax, label, x, y, **kwargs):
        assert _axis_size_inches(ax) == pytest.approx(cfg.axis_size_inches)
        assert "fontsize" not in kwargs
        original_fontsize = ax.get_xticklabels()[0].get_fontsize()
        text = original(ax, label, x, y, **kwargs)
        assert text.get_fontsize() == original_fontsize
        renderer = ax.figure.canvas.get_renderer()
        patch = text.get_bbox_patch().get_window_extent(renderer)
        points = ax.transData.transform(np.column_stack([x, y]))
        assert not any(patch.contains(*point) for point in points)
        calls.append(
            (
                text,
                ax.get_window_extent(renderer=renderer).frozen(),
                text.get_bbox_patch().get_window_extent(renderer=renderer).frozen(),
            )
        )
        return text

    monkeypatch.setattr(
        diagnostics, "_add_smart_stats_box", recorded_constraint
    )
    plotter._write_plot(rows)

    assert len(calls) == 1
    fitted, axes_bbox, patch_bbox = calls[0]
    assert fitted
    assert patch_bbox.x0 > axes_bbox.x0
    assert patch_bbox.x1 < axes_bbox.x1
    assert patch_bbox.y0 > axes_bbox.y0
    assert patch_bbox.y1 < axes_bbox.y1


def test_first_n_plot_wraps_label_clipped_by_final_axis_sizing(
    tmp_path, monkeypatch
):
    cfg = diagnostics.FirstNRewardDiagnosticsConfig(
        csv_out="",
        plot_out=str(tmp_path / "first_n.png"),
        reward_event_type="calc",
        first_n_rewards=10,
        x_by="selected_reward_rate_to_nth_per_min",
        y_by="sli",
        color_by=None,
    )
    plotter = diagnostics.FirstNRewardDiagnosticsPlotter(
        vas=[], opts=None, gls=None, cfg=cfg
    )
    rows = [
        SimpleNamespace(
            eligible_for_nth_reward_cutoff=True,
            first_n_selected_reward_span_s=90.0,
            time_to_nth_selected_reward_s=95.0,
            selected_reward_rate_to_nth_per_min=x,
            sli=y,
        )
        for x, y in ((4.0, 0.2), (6.0, 0.6), (8.0, 1.1))
    ]

    close_figure = diagnostics.plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda _fig: None)
    with diagnostics.plt.rc_context({"font.size": 26}):
        plotter._write_plot(rows)

    fig = diagnostics.plt.gcf()
    ax = fig.axes[0]
    assert "\n" in ax.get_xlabel()
    assert _axis_size_inches(ax) == pytest.approx(cfg.axis_size_inches)
    rendered = diagnostics.plt.imread(cfg.plot_out)
    edge_pixels = np.concatenate(
        (
            rendered[0, :, :3],
            rendered[-1, :, :3],
            rendered[:, 0, :3],
            rendered[:, -1, :3],
        )
    )
    assert np.min(edge_pixels) > 0.95
    close_figure(fig)
