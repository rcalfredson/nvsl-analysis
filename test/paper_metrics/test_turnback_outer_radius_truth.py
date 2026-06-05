from types import SimpleNamespace

import numpy as np

from src.exporting.turnback_outer_radius_sli_bundle import (
    _compute_outer_radius_curves,
)


class _Training:
    def __init__(self, start=0, stop=100, *, circle=True, name="training"):
        self.start = start
        self.stop = stop
        self._circle = circle
        self._name = name

    def isCircle(self):
        return self._circle

    def name(self):
        return self._name


class _Trajectory:
    def __init__(self, episodes, *, bad=False):
        self._episodes = list(episodes)
        self._bad = bad

    def reward_turnback_dual_circle_episodes_for_training(self, **_kwargs):
        return list(self._episodes)


def _episode(stop, turns_back):
    return {"stop": int(stop), "turns_back": bool(turns_back)}


def _va(*, trx, noyc=True):
    return SimpleNamespace(
        fn="fake-video",
        trns=[_Training()],
        trx=list(trx),
        sync_bucket_ranges=[[(0, 50)]],
        noyc=bool(noyc),
        _skipped=False,
    )


def test_turnback_outer_radius_masks_below_min_episode_count():
    exp = _Trajectory(
        [
            _episode(10, True),
            _episode(20, False),
            _episode(30, True),
        ]
    )

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _compute_outer_radius_curves(
        [_va(trx=[exp])],
        outer_radii_mm=np.asarray([16.0], dtype=float),
        legacy_outer_radii=False,
        inner_radius_mm=None,
        inner_delta_mm=0.0,
        border_width_mm=0.1,
        radius_offset_px=0.0,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        debug=False,
        min_episodes=4,
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_array_equal(turn_exp, [[2]])
    np.testing.assert_array_equal(total_exp, [[3]])
