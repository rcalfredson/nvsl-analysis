from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import src.plotting.first_n_reward_diagnostics as diagnostics


def _axis_size_inches(ax):
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted()
    )
    return bbox.width, bbox.height


def test_first_n_plot_constrains_stats_box_after_final_axis_sizing(
    tmp_path, monkeypatch
):
    cfg = diagnostics.FirstNRewardDiagnosticsConfig(
        csv_out="",
        plot_out=str(tmp_path / "first_n.png"),
        reward_event_type="calc",
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

    calls = []
    original = diagnostics.keep_text_box_inside_axes

    def recorded_constraint(ax, text):
        result = original(ax, text)
        renderer = ax.figure.canvas.get_renderer()
        calls.append(
            (
                result,
                ax.get_window_extent(renderer=renderer).frozen(),
                text.get_bbox_patch().get_window_extent(renderer=renderer).frozen(),
            )
        )
        return result

    monkeypatch.setattr(
        diagnostics, "keep_text_box_inside_axes", recorded_constraint
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
