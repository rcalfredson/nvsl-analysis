import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting import cross_fly_correlations as corr


@pytest.mark.parametrize("values", [[-4, -2, 3], [-5, -3, -1], [1, 2, 4]])
def test_signed_axis_includes_data_and_zero_despite_fixed_limits(values):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 2)
    corr._show_signed_y_values(ax, np.asarray(values))
    low, high = ax.get_ylim()
    assert low < min(0, *values)
    assert high > max(0, *values)
    assert list(ax.lines[0].get_ydata()) == [0, 0]
    plt.close(fig)


def test_signed_scatter_keeps_negative_points_in_stats_and_render(tmp_path, monkeypatch):
    captured = {}

    def finalize(fig, customizer, **kwargs):
        ax = fig.axes[0]
        captured["limits"] = ax.get_ylim()
        captured["points"] = ax.collections[0].get_offsets().copy()
        captured["alpha"] = [
            txt.get_bbox_patch().get_alpha() for txt in ax.texts
            if txt.get_bbox_patch() is not None
        ]
        fig.canvas.draw()

    monkeypatch.setattr(corr, "_finalize_correlation_layout", finalize)
    cfg = corr.CorrelationPlotConfig(out_dir=tmp_path, ylim=(0, 10))
    result = corr.plot_correlation_scatter(
        x=np.array([0, 1, 2, 3]), y=np.array([-4, -2, 1, 3]),
        title="Signed RPD", x_label="SLI",
        y_label="Yoked-subtracted RPD\nfor T2 SB2–5 (m$^{-1}$)",
        cfg=cfg, filename="signed", customizer=None, y_zero_reference=True,
    )
    assert result.n == 4
    np.testing.assert_allclose(captured["points"][:, 1], [-4, -2, 1, 3])
    assert captured["limits"][0] < -4
    assert captured["alpha"] and all(a == 0.65 for a in captured["alpha"])
