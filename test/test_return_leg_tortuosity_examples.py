from types import SimpleNamespace

import numpy as np

from src.exporting import return_leg_tortuosity_excursion_bin_examples as examples


def _record(va, video_index, score, frame):
    return {
        "va": va,
        "video_index": video_index,
        "role_idx": 0,
        "trajectory_index": 0,
        "traj": SimpleNamespace(),
        "training_idx": 0,
        "segment_start": frame,
        "segment_stop": frame + 10,
        "global_max_frame": frame + 2,
        "metric_start": frame + 2,
        "radial_mm": 4.0,
        "tortuosity": float(score),
        "window_start": 0,
        "wall_mask": np.zeros(100, dtype=bool),
        "nonwalk_mask": None,
        "px_per_mm": 10.0,
        "reward_center_xy": (0.0, 0.0),
        "reward_radius_mm": 2.5,
        "exclude_nonwalk": False,
        "exclude_wall_frames": False,
        "min_walk_frames": 2,
        "metric_mode": "path_over_max_radius",
        "return_start_mode": "global_max",
        "legacy_distances": False,
    }


def test_example_export_matches_per_fly_tail_and_minimum(monkeypatch, tmp_path):
    vas = [
        SimpleNamespace(_skipped=False, trns=[object()], gidx=0, fn="fly_a.avi"),
        SimpleNamespace(_skipped=False, trns=[object()], gidx=0, fn="fly_b.avi"),
    ]
    details = [
        *[
            _record(vas[0], 0, score, 10 + i * 20)
            for i, score in enumerate((10, 9, 2, 1))
        ],
        *[
            _record(vas[1], 1, score, 100 + i * 20)
            for i, score in enumerate((8, 7, 6, 5))
        ],
    ]
    details[0]["exclude_wall_frames"] = True
    details[0]["wall_mask"][12] = True

    def fake_collect(_vas, _opts, *, episode_callback, **_kwargs):
        for record in details:
            episode_callback(record)
        records = [
            [[(record["radial_mm"], record["tortuosity"]) for record in details[:4]], []],
            [[(record["radial_mm"], record["tortuosity"]) for record in details[4:]], []],
        ]
        return records, []

    plotted = []

    class FakePlotter:
        def __init__(self, **_kwargs):
            pass

        def plot_between_reward_interval(self, **kwargs):
            plotted.append(kwargs)
            open(kwargs["out_path"], "wb").close()

    monkeypatch.setattr(examples, "_collect_records", fake_collect)
    monkeypatch.setattr(
        examples,
        "exp_target_sync_bucket_eligibility_mask",
        lambda _vas, _opts: [True, True],
    )
    monkeypatch.setattr(
        examples, "min_between_reward_sync_bucket_trajectories", lambda _opts: 3
    )
    monkeypatch.setattr(examples, "_metric_components", lambda _record: (10.0, 4.0, 4.0))
    monkeypatch.setattr(examples, "EventChainPlotter", FakePlotter)

    opts = SimpleNamespace(
        return_leg_tortuosity_excursion_bin_radius_pairs_mm="3:5",
        return_leg_tortuosity_excursion_bin_pairs_mm=None,
        return_leg_tortuosity_excursion_bin_trainings=None,
        return_leg_tortuosity_excursion_bin_top_fraction=0.5,
        return_leg_tortuosity_excursion_bin_examples_per_bin=3,
        return_leg_tortuosity_excursion_bin_examples_max_per_fly=1,
        return_leg_tortuosity_excursion_bin_examples_role="exp",
        return_leg_tortuosity_excursion_bin_examples_zoom_radius_mm=None,
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
        export_group_label="control",
        imageFormat="png",
    )

    examples.export_return_leg_tortuosity_excursion_bin_examples(
        vas, opts, ["control"], str(tmp_path)
    )

    assert len(plotted) == 2
    assert [call["highlight_start_frame"] for call in plotted] == [12, 102]
    assert plotted[0]["highlight_excluded_frame_mask"] is details[0]["wall_mask"]
    assert plotted[0]["highlight_excluded_frame_mask_start"] == 0
    assert plotted[1]["highlight_excluded_frame_mask"] is None
    manifest = (tmp_path / "manifest.csv").read_text()
    assert ",10.0,path_over_max_radius," in manifest
    assert ",8.0,path_over_max_radius," in manifest
    assert ",9.0,path_over_max_radius," not in manifest


def test_last_wall_frame_uses_absolute_window_coordinates():
    record = {
        "wall_mask": np.array([False, True, False, True, False]),
        "window_start": 100,
        "segment_start": 101,
        "segment_stop": 105,
    }
    assert examples._last_wall_frame(record) == 103
