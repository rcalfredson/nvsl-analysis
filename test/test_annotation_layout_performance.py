"""Check that skipping repaints preserves layout decisions and exported plots."""

import matplotlib
import numpy as np
import pytest

from scripts.benchmark_annotation_layout import make_figure, measure
from src.plotting import annotation_layout


@pytest.mark.parametrize("dpi,font_size", [(100, 11), (100, 23), (200, 23)])
def test_renderer_reuse_matches_full_redraws(monkeypatch, dpi, font_size):
    options = dict(dpi=dpi, font_size=font_size, panels=2, buckets=3, exports=True)
    with matplotlib.rc_context({"svg.hashsalt": "annotation-layout-test"}):
        fast = measure(annotation_layout, **options)
        monkeypatch.setattr(annotation_layout, "_can_reuse_annotation_renderer", lambda fig: False)
        full = measure(annotation_layout, **options)
    np.testing.assert_array_equal(fast[1], full[1])
    assert fast[0]["ylim"] == full[0]["ylim"]
    assert fast[2] == full[2]
    assert fast[3] == full[3]  # Byte-identical PNG, PDF, SVG, including glyphs.
    assert fast[0]["draws"] == 2  # One initial draw per resolver call.
    assert full[0]["draws"] > 10 * fast[0]["draws"]


@pytest.mark.parametrize(
    "effect", ["tight", "constrained", "callback", "custom_text", "annotation"],
)
def test_draw_time_effects_keep_full_redraws(effect):
    from unittest.mock import patch
    import matplotlib.pyplot as plt

    fig, annotations = make_figure(
        panels=1, buckets=2, layout=effect if effect in ("tight", "constrained") else None,
    )
    ax, texts = annotations[0]
    if effect == "callback":
        fig.canvas.mpl_connect("draw_event", lambda event: None)
    elif effect == "custom_text":
        # An unrelated label may move other artists from its draw hook.
        unrelated = ax.text(0, 0, "custom")
        original_draw = unrelated.draw
        unrelated.draw = lambda renderer: original_draw(renderer)
    elif effect == "annotation":
        # Annotation overrides Text.draw at the class level.
        ax.annotate("custom", (0, 0), xytext=(0.1, 0.1), arrowprops={"arrowstyle": "->"})
    try:
        assert not annotation_layout._can_reuse_annotation_renderer(fig)
        with patch.object(fig.canvas, "draw", wraps=fig.canvas.draw) as draw:
            annotation_layout.resolve_annotation_text_overlaps(ax, texts, [-0.5, 2.0])
        assert draw.call_count > 10
    finally:
        plt.close(fig)
