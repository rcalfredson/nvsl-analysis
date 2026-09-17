from types import SimpleNamespace

import numpy as np

from src.analysis.trajectory import Trajectory


class _CircleTraining:
    def __init__(self, stop, *, number=1):
        self.n = int(number)
        self.start = 0
        self.stop = int(stop)
        self.postStop = int(stop)

    def isCircle(self):
        return True

    def circles(self, _fly_idx):
        return [(0.0, 0.0, 10.0)]

    def sname(self):
        return f"T{self.n}"


def test_reward_return_distance_entries_require_full_border_crossings():
    trj = object.__new__(Trajectory)
    trj.x = np.array([17.2, 16.5, 15.5, 12.0, 10.5, 9.5])
    trj.y = np.zeros_like(trj.x)
    trj.f = 0
    trj.va = SimpleNamespace(
        xf=SimpleNamespace(fctr=1.0),
        ct=SimpleNamespace(pxPerMmFloor=lambda: 1.0),
    )
    trn = _CircleTraining(stop=len(trj.x))

    episodes = trj.reward_return_distance_episodes_for_training(
        trn=trn,
        return_delta_mm=6.0,
        reward_delta_mm=0.0,
        border_width_mm=1.0,
        min_walking_frac=None,
    )

    assert [
        (ep["start"], ep["stop"], ep["reward_entry"], ep["success"])
        for ep in episodes
    ] == [(2, 5, 5, True)]
    assert episodes[0]["dist"] == 6.0  # Includes frame 4 -> reward-entry frame 5.


def test_circular_contact_region_spans_outer_to_inner_border_crossings(capsys):
    trj = object.__new__(Trajectory)
    trj.x = np.array([9.0, 10.5, 11.5, 10.5, 9.0])
    trj.y = np.zeros_like(trj.x)
    trj.f = 0
    trj._bad = False
    trj.opts = SimpleNamespace(outside_circle_radii=[10.0])
    trj.boundary_event_stats = {
        "wall": {"all": {"edge": {"boundary_contact_regions": []}}}
    }
    trn = _CircleTraining(stop=len(trj.x), number=2)
    trj.va = SimpleNamespace(
        circle=True,
        startPre=0,
        trns=[trn],
        fps=1.0,
        xf=SimpleNamespace(fctr=1.0),
        ct=SimpleNamespace(pxPerMmFloor=lambda: 1.0),
    )

    trj._calcOutsideCirclePeriods()

    stats = trj.boundary_event_stats["circle"]["ctr"]["ctr"][10.0]
    assert stats["boundary_contact"].tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert stats["boundary_contact_regions"] == [slice(2, 4)]
    assert stats["contact_start_idxs"].tolist() == [2]
    assert trj.outside_durations == [{10.0: [2]}]
    capsys.readouterr()


def test_reward_return_distance_counts_single_step_and_filters_endpoint():
    trj = object.__new__(Trajectory)
    trj.x = np.array([17.0, 15.0, 9.0])
    trj.y = np.zeros(3)
    trj.walking = np.array([True, True, False])
    trj.f = 0
    trj.va = SimpleNamespace(
        xf=SimpleNamespace(fctr=1.0),
        ct=SimpleNamespace(pxPerMmFloor=lambda: 1.0),
    )
    common = dict(trn=_CircleTraining(stop=3), return_delta_mm=6.0)
    episodes = trj.reward_return_distance_episodes_for_training(
        **common, min_walking_frac=None,
    )
    assert episodes[0]["dist"] == 6.0
    assert episodes[0]["walking_frac"] == 0.5
    episodes = trj.reward_return_distance_episodes_for_training(
        **common, min_walking_frac=0.6,
    )
    assert episodes[0]["end_reason"] == "dropped_low_walking"
    episodes = trj.reward_return_distance_episodes_for_training(
        **common, min_walking_frac=None, exclude_wall_contact=True,
        wall_contact_regions=[slice(2, 3)],
    )
    assert episodes[0]["end_reason"] == "dropped_wall"
