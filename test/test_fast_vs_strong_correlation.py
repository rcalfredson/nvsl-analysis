from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import pearsonr

import src.plotting.cross_fly_correlations as correlations
from src.plotting.plot_customizer import PlotCustomizer


@pytest.mark.parametrize(
    "x,y,fast,strong",
    [
        (list(range(11)), list(range(10, -1, -1)), [9, 10], [0, 1]),
        (list(range(10)), [0, 1, 2, 3, 4, *([np.nan] * 5)], [8, 9], [4]),
        # Counts use separate eligible populations, retaining original positions.
        ([0, np.nan, 1, np.inf, 2, 3, -np.inf, 4, 5, 6, 7],
         [0, 1, np.nan, 2, np.inf, 3, 4, 5, 6, 7, 8], [10], [10]),
        # Stable ascending sorting takes the later tied positions for the top group.
        ([0, 1, 2, 3, 4, 5, 6, 9, 9, 9], list(range(10)), [8, 9], [8, 9]),
        ([0, np.nan, 8, np.inf, 9], [9, 8, np.nan, 0, -np.inf], [4], [0]),
        ([np.nan, np.inf, -np.inf], [0, 1, 2], [], [2]),
        ([0, 1, 2], [np.nan, np.inf, -np.inf], [2], []),
        ([np.nan], [np.nan], [], []),
    ],
)
def test_fast_strong_selection_counts_indices_ties_and_independence(
    x, y, fast, strong,
):
    summary = correlations.summarize_fast_vs_strong(
        x, y, [object()] * len(x), SimpleNamespace(),
    )
    np.testing.assert_array_equal(summary["fast"], fast)
    np.testing.assert_array_equal(summary["strong"], strong)
    np.testing.assert_array_equal(summary["overlap"], sorted(set(fast) & set(strong)))
    assert all(indices.dtype.kind == "i" for indices in summary.values())


def test_classification_changes_leave_complete_pair_scatter_and_pearson_unchanged(
    monkeypatch, tmp_path,
):
    x = np.array([0, np.nan, 1, np.inf, 2, 3, 4, 5, 20, 8, 9, 10.0])
    y = np.array([0, 20, 2, 1, 1, 4, 3, 6, np.nan, 9, 10, 11.0])
    valid = np.isfinite(x) & np.isfinite(y)
    expected_r, expected_p = pearsonr(x[valid], y[valid])
    calls, overlays = [], []

    def record_pearson(x_values, y_values):
        result = pearsonr(x_values, y_values)
        calls.append((x_values.copy(), y_values.copy(), result))
        return result

    def inspect_overlay(ax, handles, stats_text, *args, **kwargs):
        artist = kwargs["scatter_artist"]
        overlays.append((stats_text, artist.get_facecolors().copy()))
        np.testing.assert_array_equal(artist.get_offsets(), np.column_stack((x[valid], y[valid])))

    monkeypatch.setattr(correlations, "pearsonr", record_pearson)
    monkeypatch.setattr(correlations, "_place_stacked_correlation_overlays", inspect_overlay)
    monkeypatch.setattr(correlations, "_finalize_correlation_layout", lambda *a, **k: None)
    monkeypatch.setattr(correlations, "writeImage", lambda *a, **k: None)

    for fraction in (0.2, 0.5):
        summary = correlations.summarize_fast_vs_strong(
            x, y, [object()] * len(x), SimpleNamespace(), frac=fraction,
        )
        # These flies have no complete pair, but retain independent membership.
        assert 8 in summary["fast"]
        assert 1 in summary["strong"]
        correlations.plot_fast_vs_strong_scatter(
            x, y, [object()] * len(x), summary["fast"], summary["strong"],
            tmp_path, fraction, PlotCustomizer(), strong_y_label="Later SLI",
            strong_title_suffix="", x_label="Initial SLI",
        )

    assert len(calls) == 2
    for x_values, y_values, result in calls:
        np.testing.assert_array_equal(x_values, x[valid])
        np.testing.assert_array_equal(y_values, y[valid])
        assert result.statistic == expected_r
        assert result.pvalue == expected_p
    assert overlays[0][0] == overlays[1][0] == correlations._format_corr_annotation(
        expected_r, expected_p, int(valid.sum()),
    )
    assert not np.array_equal(overlays[0][1], overlays[1][1])


@pytest.mark.parametrize("font_size", [12, 20, 23, 30])
@pytest.mark.parametrize("distribution", ["dense", "sparse", "small", "empty"])
def test_fast_vs_strong_always_stacks_stats_above_two_column_legend(
    monkeypatch, tmp_path, font_size, distribution,
):
    rng = np.random.default_rng(42)
    x = np.linspace(-0.2, 2.0, 50)
    y = x + rng.normal(0, 0.15, len(x))
    if distribution == "sparse":
        y[::2] -= 1.0
    elif distribution == "small":
        x, y = x[:2], y[:2]
    elif distribution == "empty":
        x, y = np.array([np.nan]), np.array([np.nan])
    checked = []

    def inspect_plot(*args, **kwargs):
        ax = plt.gcf().axes[0]
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = ax.get_legend()
        assert legend._ncols == 2
        assert [text.get_text() for text in legend.get_texts()] == [
            "Fast learners only", "Strong learners only", "Fast + strong", "Other flies",
        ]
        stats = [text for text in ax.texts if text.get_text().startswith("n = ")]
        assert len(stats) == 1
        stats_bbox = stats[0].get_bbox_patch().get_window_extent(renderer)
        legend_bbox = legend.get_window_extent(renderer)
        axes_bbox = ax.get_window_extent(renderer)
        assert stats_bbox.y0 > legend_bbox.y1
        for bbox in (stats_bbox, legend_bbox):
            assert bbox.x0 > axes_bbox.x0
            assert bbox.x1 < axes_bbox.x1
            assert bbox.y0 > axes_bbox.y0
            assert bbox.y1 < axes_bbox.y1
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            points = ax.transData.transform(np.column_stack([x[valid], y[valid]]))
            assert np.max(points[:, 1]) < legend_bbox.y0 - 2.0
        for line in ax.lines:
            vertices = line.get_path().transformed(line.get_transform()).vertices
            assert np.max(vertices[:, 1]) < legend_bbox.y0 - 2.0
        checked.append(True)

    monkeypatch.setattr(correlations, "writeImage", inspect_plot)
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(font_size)
        correlations.plot_fast_vs_strong_scatter(
            x, y, vas=[object()] * len(x),
            fast_idx=np.array([0]), strong_idx=np.array([len(x) - 1]),
            out_dir=tmp_path, frac=0.2, customizer=customizer,
            strong_y_label="Mean SLI over T2 SB2–5", strong_title_suffix="",
            x_label="SLI for T1 SB1",
        )
    assert checked == [True]


def test_fast_vs_strong_uses_one_all_fly_correlation_and_keeps_groups(
    monkeypatch, tmp_path
):
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([0.0, 1.0, 2.1, 3.0, 4.0, 5.0])
    calls = []
    overlays = []
    original_pearsonr = correlations.pearsonr

    def record_pearsonr(x_values, y_values):
        calls.append((np.asarray(x_values), np.asarray(y_values)))
        return original_pearsonr(x_values, y_values)

    def record_overlays(ax, handles, stats_text, x_values, y_values, **kwargs):
        overlays.append((ax, handles, stats_text, kwargs))

    monkeypatch.setattr(correlations, "pearsonr", record_pearsonr)
    monkeypatch.setattr(correlations, "_place_stacked_correlation_overlays", record_overlays)
    monkeypatch.setattr(correlations, "_finalize_correlation_layout", lambda *a, **k: None)
    monkeypatch.setattr(correlations, "writeImage", lambda *a, **k: None)

    correlations.plot_fast_vs_strong_scatter(
        x,
        y,
        vas=[object()] * len(x),
        fast_idx=np.array([0, 2, 5]),
        strong_idx=np.array([1, 2]),
        out_dir=tmp_path,
        frac=0.2,
        customizer=PlotCustomizer(),
        strong_y_label="Later SLI",
        strong_title_suffix="",
        x_label="Initial SLI",
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], x[:5])
    np.testing.assert_array_equal(calls[0][1], y[:5])
    assert len(overlays) == 1
    ax, handles, stats_text, kwargs = overlays[0]
    r, p = pearsonr(x[:5], y[:5])
    assert stats_text == correlations._format_corr_annotation(r, p, 5)
    assert kwargs.get("compact_stats_text") is None
    assert [handle.get_label() for handle in handles] == [
        "Fast learners only",
        "Strong learners only",
        "Fast + strong",
        "Other flies",
    ]
    expected_colors = [
        correlations.correlation_plot_color(f"fast_vs_strong_{group}")
        for group in ("fast", "strong", "overlap", "other", "other")
    ]
    actual_colors = kwargs["scatter_artist"].get_facecolors()
    np.testing.assert_allclose(
        actual_colors[:, :3], [mcolors.to_rgb(color) for color in expected_colors]
    )
    assert len(ax.lines) == 1


def test_fast_vs_strong_small_sample_shows_single_na_annotation(
    monkeypatch, tmp_path
):
    overlays = []
    monkeypatch.setattr(
        correlations,
        "pearsonr",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected correlation")),
    )
    monkeypatch.setattr(
        correlations,
        "_place_stacked_correlation_overlays",
        lambda ax, handles, stats_text, *args, **kwargs: overlays.append(
            (ax, handles, stats_text)
        ),
    )
    monkeypatch.setattr(correlations, "_finalize_correlation_layout", lambda *a, **k: None)
    monkeypatch.setattr(correlations, "writeImage", lambda *a, **k: None)

    correlations.plot_fast_vs_strong_scatter(
        np.array([0.0, 1.0, np.nan]),
        np.array([0.0, 1.0, 2.0]),
        vas=[object()] * 3,
        fast_idx=np.array([0]),
        strong_idx=np.array([1]),
        out_dir=tmp_path,
        frac=0.2,
        customizer=PlotCustomizer(),
        strong_y_label="Later SLI",
        strong_title_suffix="",
        x_label="Initial SLI",
    )

    assert overlays[0][2] == "n = 2, r = n/a, p = n/a"
    assert len(overlays[0][1]) == 4
    assert not overlays[0][0].lines
