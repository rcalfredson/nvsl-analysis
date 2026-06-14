from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    normalize_sli_bundle,
    validate_post_wall_departure_tortuosity_bundle,
)
from src.exporting import post_wall_departure_tortuosity_examples as examples
from src.exporting import post_wall_departure_tortuosity_sli_bundle as exporter
from src.exporting.post_wall_departure_tortuosity_sli_bundle import (
    aggregate_post_wall_departure_tortuosity,
    departure_distance_to_reward_circle_mm,
    last_wall_departure_frame,
)
from src.plotting.post_wall_departure_tortuosity_sli_bundle_plotter import (
    _bundle_to_exported,
)


def _bundle():
    return {
        "sli": np.asarray([0.1, 0.2]),
        "group_label": np.asarray("control"),
        "bucket_len_min": np.asarray(np.nan),
        "training_names": np.asarray(["T1"]),
        "video_ids": np.asarray(["fly-a", "fly-b"]),
        "sli_training_idx": np.asarray(0),
        "sli_use_training_mean": np.asarray(False),
        "post_wall_departure_tortuosity_exp": np.asarray([[1.5], [2.0]]),
        "post_wall_departure_tortuosity_ctrl": np.asarray([[1.2], [np.nan]]),
        "post_wall_departure_tortuosityN_exp": np.asarray([[5], [7]]),
        "post_wall_departure_tortuosityN_ctrl": np.asarray([[5], [2]]),
        "btw_rwd_sync_bucket_min_trajectories": np.asarray(5),
    }


def test_last_wall_departure_uses_final_contact_and_absolute_frames():
    wall = np.asarray([False, True, True, False, False, True, False])

    assert last_wall_departure_frame(
        wall,
        segment_start=101,
        segment_stop=107,
        window_start=100,
    ) == (105, 106)
    assert (
        last_wall_departure_frame(
            np.zeros(7, dtype=bool),
            segment_start=100,
            segment_stop=107,
            window_start=100,
        )
        is None
    )
    assert (
        last_wall_departure_frame(
            np.asarray([False, False, True]),
            segment_start=100,
            segment_stop=103,
            window_start=100,
        )
        is None
    )


def test_departure_distance_targets_reward_circle_perimeter():
    traj = SimpleNamespace(
        x=np.asarray([10.0]),
        y=np.asarray([0.0]),
    )

    direct_mm, edge = departure_distance_to_reward_circle_mm(
        traj,
        departure_frame=0,
        reward_circle=(0.0, 0.0, 2.0),
        px_per_mm=2.0,
    )

    assert direct_mm == pytest.approx(4.0)
    assert edge == pytest.approx((2.0, 0.0))


def test_aggregation_applies_minimum_to_each_fly_and_role():
    records = [
        [[1.0, 2.0, 3.0], [2.0]],
        [[4.0], [3.0, 5.0, 7.0]],
    ]

    exp, ctrl, n_exp, n_ctrl = aggregate_post_wall_departure_tortuosity(
        records,
        min_episodes=3,
    )

    np.testing.assert_array_equal(n_exp, [[3], [1]])
    np.testing.assert_array_equal(n_ctrl, [[1], [3]])
    assert exp[0, 0] == pytest.approx(2.0)
    assert np.isnan(exp[1, 0])
    assert np.isnan(ctrl[0, 0])
    assert ctrl[1, 0] == pytest.approx(5.0)


def test_collector_measures_path_after_final_wall_departure(monkeypatch):
    traj = SimpleNamespace(
        x=np.asarray([1.0, 7.0, 6.0, 5.0, 3.0, 1.0]),
        y=np.zeros(6),
        d=np.asarray([6.0, 1.0, 1.0, 2.0, 2.0]),
        pxPerMmFloor=1.0,
        bad=lambda: False,
    )
    trn = SimpleNamespace(
        stop=6,
        isCircle=lambda: True,
        circles=lambda _f: [(0.0, 0.0, 1.0)],
        name=lambda: "T1",
    )
    va = SimpleNamespace(
        trx=[traj],
        trns=[trn],
        noyc=True,
        xf=SimpleNamespace(fctr=1.0),
        _iter_between_reward_segment_com=lambda *_args, **_kwargs: iter(
            [SimpleNamespace(s=0, e=5)]
        ),
    )
    monkeypatch.setattr(
        exporter,
        "sync_bucket_window",
        lambda *_args, **_kwargs: (0, 10, 1, [True]),
    )
    monkeypatch.setattr(
        exporter,
        "wall_contact_mask",
        lambda *_args, **_kwargs: np.asarray(
            [False, True, True, False, False, False]
        ),
    )

    records, details, _windows = exporter.collect_post_wall_departure_tortuosity(
        [va],
        SimpleNamespace(
            post_wall_departure_tortuosity_trainings=None,
            post_wall_departure_tortuosity_skip_first_sync_buckets=0,
            post_wall_departure_tortuosity_keep_first_sync_buckets=0,
            post_wall_departure_tortuosity_exclude_nonwalking_frames=False,
            post_wall_departure_tortuosity_min_walk_frames=2,
            post_wall_departure_tortuosity_min_direct_distance_mm=0.0,
        ),
    )

    assert records == [[[pytest.approx(1.0)], []]]
    assert details[0]["last_wall_contact_frame"] == 2
    assert details[0]["departure_frame"] == 3
    assert details[0]["path_mm"] == pytest.approx(4.0)
    assert details[0]["direct_mm"] == pytest.approx(4.0)


def test_bundle_validation_and_scalar_plot_conversion():
    bundle = _bundle()

    validate_post_wall_departure_tortuosity_bundle(bundle)
    normalized = normalize_sli_bundle(bundle)
    exported = _bundle_to_exported(
        normalized,
        label="control",
        mode="exp",
        sub_idx=np.asarray([0, 1]),
    )

    assert exported.panel_labels == ["Wall-contact trajectories"]
    assert exported.n_units_panel.tolist() == [2]
    assert exported.mean[0] == pytest.approx(1.75)


def test_example_export_ranks_and_draws_post_departure_path(monkeypatch, tmp_path):
    va = SimpleNamespace(_skipped=False, trns=[object()], gidx=0, fn="fly.avi")
    detail = {
        "va": va,
        "video_index": 0,
        "role_idx": 0,
        "trajectory_index": 0,
        "traj": SimpleNamespace(),
        "training_idx": 0,
        "segment_start": 10,
        "segment_stop": 30,
        "last_wall_contact_frame": 19,
        "departure_frame": 20,
        "metric_stop": 31,
        "path_mm": 8.0,
        "direct_mm": 4.0,
        "tortuosity": 2.0,
        "reward_edge_xy": (1.0, 1.0),
        "departure_xy": (5.0, 1.0),
        "exclude_nonwalk": False,
    }
    monkeypatch.setattr(
        examples,
        "collect_post_wall_departure_tortuosity",
        lambda _vas, _opts: ([[[2.0], []]], [detail], []),
    )
    monkeypatch.setattr(
        examples, "min_between_reward_sync_bucket_trajectories", lambda _opts: 1
    )
    monkeypatch.setattr(
        examples,
        "exp_target_sync_bucket_eligibility_mask",
        lambda _vas, _opts: [True],
    )
    plotted = []

    class FakePlotter:
        def __init__(self, **_kwargs):
            pass

        def plot_between_reward_interval(self, **kwargs):
            plotted.append(kwargs)

    monkeypatch.setattr(examples, "EventChainPlotter", FakePlotter)
    opts = SimpleNamespace(
        post_wall_departure_tortuosity_examples_role="exp",
        post_wall_departure_tortuosity_examples_num=4,
        post_wall_departure_tortuosity_examples_max_per_fly=1,
        post_wall_departure_tortuosity_examples_zoom_radius_mm=None,
        export_group_label="control",
        imageFormat="png",
    )

    examples.export_post_wall_departure_tortuosity_examples(
        [va], opts, ["control"], str(tmp_path)
    )

    assert len(plotted) == 1
    assert plotted[0]["highlight_start_frame"] == 20
    assert plotted[0]["highlight_stop_frame"] == 31
    assert plotted[0]["comparison_line_start_xy"] == (5.0, 1.0)
    assert plotted[0]["comparison_line_stop_xy"] == (1.0, 1.0)
    assert "2.0" in (tmp_path / "manifest.csv").read_text()
