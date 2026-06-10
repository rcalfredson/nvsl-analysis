from types import SimpleNamespace

import numpy as np

from src.exporting.turnback_outer_radius_sli_bundle import (
    _compute_outer_radius_curves,
    export_turnback_outer_radius_sli_bundle,
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


def test_turnback_outer_radius_export_applies_exp_target_sync_bucket_filter(
    tmp_path, monkeypatch
):
    exp = _Trajectory([_episode(10, True), _episode(20, False)])
    va = _va(trx=[exp])
    va.reward_exclusion_mask = [[[True]]]
    va.sync_bucket_ranges = [[]]
    opts = SimpleNamespace(
        export_group_label="group",
        turnback_outer_radius_outer_radii_mm="16",
        turnback_outer_radius_outer_deltas_mm=None,
        turnback_outer_radius_trainings="1",
        turnback_outer_radius_skip_first_sync_buckets=0,
        turnback_outer_radius_keep_first_sync_buckets=0,
        turnback_outer_radius_last_sync_buckets=0,
        turnback_inner_radius_mm=None,
        turnback_inner_delta_mm=0.0,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
        turnback_outer_radius_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_turnback_episodes=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
    )
    monkeypatch.setattr(
        "src.exporting.turnback_outer_radius_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "turnback_outer_radius.npz"

    export_turnback_outer_radius_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        np.testing.assert_array_equal(
            bundle["exp_pi_threshold_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_pi_threshold_filter_reason"], ["target_sync_bucket_missing"]
        )
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
        assert np.isnan(bundle["turnback_outer_radius_ratio_exp"]).all()
        np.testing.assert_array_equal(bundle["turnback_outer_radius_total_exp"], [[2]])
        np.testing.assert_array_equal(bundle["turnback_outer_radius_turn_exp"], [[1]])
