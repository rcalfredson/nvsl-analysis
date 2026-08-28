from src.plotting.sli_label_utils import sli_extreme_plot_specs


def test_sli_extremes_plot_top_solid_then_bottom_dashed():
    specs = sli_extreme_plot_specs(
        top=[4, 5],
        bottom=[0, 1],
        top_fraction=0.2,
        bottom_fraction=0.5,
    )

    assert specs == (
        ("top", [4, 5], "Top 20% learners", "-"),
        ("bottom", [0, 1], "Bottom 50% learners", "--"),
    )
