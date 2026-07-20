from types import SimpleNamespace

import numpy as np

import src.plotting.turnback_excursion_bin_sli_bundle_plotter as plotter


def test_turnback_bundle_plot_forwards_swarm_points(monkeypatch):
    bundle = {
        "group_label": np.asarray("Ctrl"),
        "video_ids": np.asarray(["v1", "v2", "v3"], dtype=object),
        "turnback_excursion_bin_ratio_exp": np.asarray(
            [[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]], dtype=float
        ),
        "turnback_excursion_bin_ratio_ctrl": np.zeros((3, 2), dtype=float),
        "turnback_excursion_bin_pair_inner_deltas_mm": np.asarray([3.0, 8.0]),
        "turnback_excursion_bin_pair_outer_deltas_mm": np.asarray([5.0, 10.0]),
    }
    captured = {}
    sentinel_figure = object()

    monkeypatch.setattr(plotter, "load_sli_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        plotter,
        "_selected_groups",
        lambda *_args, **_kwargs: [("Ctrl", np.arange(3, dtype=int))],
    )

    def fake_plot_overlays(exports, **kwargs):
        captured["exports"] = exports
        captured["kwargs"] = kwargs
        return sentinel_figure

    monkeypatch.setattr(plotter, "plot_overlays", fake_plot_overlays)
    monkeypatch.setattr(plotter, "writeImage", lambda *_args, **_kwargs: None)

    figure = plotter.plot_turnback_excursion_bin_sli_bundles(
        ["bundle.npz"],
        "plot.png",
        show_points=True,
        opts=SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None),
    )

    assert figure is sentinel_figure
    assert captured["kwargs"]["show_points"] is True
    np.testing.assert_allclose(
        np.asarray(
            captured["exports"][0].per_unit_values_panel[0], dtype=float
        ),
        [0.2, 0.3, 0.4],
    )
