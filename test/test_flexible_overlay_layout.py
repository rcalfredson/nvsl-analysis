import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting.annotation_layout import place_flexible_overlay_texts


def test_flexible_overlay_bbox_stays_inside_axes():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylim(-0.5, 2.0)
    text = ax.text(
        0.03,
        0.97,
        "AUC (n=50,21): **** (p=1.81e-10)",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    place_flexible_overlay_texts(ax, [text], pad_px=4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    text_bbox = text.get_window_extent(renderer=renderer)

    assert text.get_transform() == ax.transAxes
    assert text_bbox.x0 >= axes_bbox.x0
    assert text_bbox.x1 <= axes_bbox.x1
    assert text_bbox.y0 >= axes_bbox.y0
    assert text_bbox.y1 <= axes_bbox.y1
    assert ax.get_ylim() == (-0.5, 2.0)
    plt.close(fig)


def test_flexible_overlay_avoids_legend_corner():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="learners")
    ax.legend(loc="upper left")
    text = ax.text(
        0.03,
        0.97,
        "AUC summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    place_flexible_overlay_texts(ax, [text])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not text.get_window_extent(renderer=renderer).overlaps(
        ax.get_legend().get_window_extent(renderer=renderer)
    )
    plt.close(fig)
