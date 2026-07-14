from src.plotting.com_sli_bundle_plotter import _sample_size_label_kwargs


def test_agarose_sample_size_labels_get_translucent_white_background():
    kwargs = _sample_size_label_kwargs("agarose")

    assert kwargs["zorder"] == 5
    assert kwargs["bbox"] == {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
    }


def test_other_metric_sample_size_labels_keep_existing_style():
    assert _sample_size_label_kwargs("commag") == {}
