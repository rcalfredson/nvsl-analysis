import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting import cross_fly_correlations as corr


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("values", [[-4, -2, 3], [-5, -3, -1], [1, 2, 4]])
def test_signed_axis_includes_data_and_zero_despite_fixed_limits(values, axis):
    fig, ax = plt.subplots()
    getattr(ax, f"set_{axis}lim")(0, 2)
    corr._show_signed_axis_values(ax, np.asarray(values), axis=axis)
    low, high = getattr(ax, f"get_{axis}lim")()
    assert low < min(0, *values)
    assert high > max(0, *values)
    assert list(getattr(ax.lines[0], f"get_{axis}data")()) == [0, 0]
    plt.close(fig)


def test_signed_scatter_keeps_negative_points_in_stats_and_render(tmp_path, monkeypatch):
    captured = {}

    def finalize(fig, customizer, **kwargs):
        ax = fig.axes[0]
        captured["limits"] = ax.get_ylim()
        captured["points"] = ax.collections[0].get_offsets().copy()
        fig.canvas.draw()

    original_write_image = corr.writeImage

    def save_plot(*args, **kwargs):
        ax = plt.gcf().axes[0]
        captured["alpha"] = [
            txt.get_bbox_patch().get_alpha() for txt in ax.texts
            if txt.get_bbox_patch() is not None
        ]
        return original_write_image(*args, **kwargs)

    monkeypatch.setattr(corr, "writeImage", save_plot)
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


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize(
    "x_reference,y_reference",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_scatter_zero_references_are_independent_opt_ins(
    tmp_path, monkeypatch, selected, x_reference, y_reference,
):
    captured = {}

    def finalize(fig, customizer, **kwargs):
        ax = fig.axes[0]
        captured["xlim"] = ax.get_xlim()
        captured["ylim"] = ax.get_ylim()
        captured["vertical"] = sum(
            np.array_equal(line.get_xdata(), [0, 0]) for line in ax.lines
        )
        captured["horizontal"] = sum(
            np.array_equal(line.get_ydata(), [0, 0]) for line in ax.lines
        )

    monkeypatch.setattr(corr, "_finalize_correlation_layout", finalize)
    monkeypatch.setattr(corr, "writeImage", lambda *args, **kwargs: None)
    kwargs = dict(
        x=np.array([-4, -2, 1, 3, np.nan]),
        y=np.array([-3, 1, -1, 2, -100]),
        title="test", x_label="SLI", y_label="RPD",
        filename="test", customizer=None,
    )
    # Omit false flags so the default behavior is covered too.
    if x_reference:
        kwargs["x_zero_reference"] = True
    if y_reference:
        kwargs["y_zero_reference"] = True
    if selected:
        corr.plot_selected_group_scatter(
            **kwargs, out_dir=tmp_path, xlim=(1, 2), ylim=(1, 2),
            bottom_idx=np.array([0]), top_idx=np.array([3]), mode="both",
        )
    else:
        cfg = corr.CorrelationPlotConfig(out_dir=tmp_path, xlim=(1, 2), ylim=(1, 2))
        result = corr.plot_correlation_scatter(**kwargs, cfg=cfg)
        assert result.n == 4

    assert captured["vertical"] == int(x_reference)
    assert captured["horizontal"] == int(y_reference)
    if x_reference:
        assert captured["xlim"][0] < -4
        assert captured["xlim"][1] > 3
    else:
        assert captured["xlim"] == (1, 2)
    if y_reference:
        assert -100 < captured["ylim"][0] < -3
        assert captured["ylim"][1] > 2
    else:
        assert captured["ylim"] == (1, 2)
