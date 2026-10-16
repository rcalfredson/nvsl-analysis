from types import SimpleNamespace
import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.audit_walking_scale import (
    ROOT, aggregate_maps, diagnostic_environment, histogram, map_metrics,
    read_recipes, region_masks, walking_masks,
)
from scripts.summarize_walking_scale_audit import (
    number, regional_summary, scope_and_validation, validate_report_inputs,
)
from src.analysis.trajectory import Trajectory
from src.analysis.video_analysis import VideoAnalysis
from src.utils.common import CT, Xformer


@pytest.mark.parametrize("factor", [.8, 1., 1.2])
def test_masks_match_production_and_are_monotone(factor):
    ppmm = 8.
    speeds = np.array([0, 12.8, np.nextafter(16., 0), 16., np.nextafter(16., np.inf),
                       19.2, 20, np.nan, np.inf])
    current, scaled = walking_masks(speeds, ppmm, factor)
    trx = Trajectory.__new__(Trajectory)
    trx.sp = speeds.copy()
    trx.va = SimpleNamespace(ct=SimpleNamespace(pxPerMmFloor=lambda: ppmm))
    trx._setWalking()
    np.testing.assert_array_equal(current, trx.walking)
    assert not current[3]  # strict boundary
    assert current[4]
    assert not current[-2]
    assert current[-1]  # preserve production infinity behavior
    if factor < 1:
        assert np.all(scaled | ~current)
    elif factor > 1:
        assert np.all(current | ~scaled)
    else:
        np.testing.assert_array_equal(current, scaled)
    boundary = 2 * ppmm * factor
    _, exact = walking_masks(np.array([boundary, np.nextafter(boundary, np.inf)]), ppmm, factor)
    np.testing.assert_array_equal(exact, [False, True])


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_reject_invalid_scale(factor):
    with pytest.raises(ValueError):
        walking_masks([1], 8, factor)


def test_reuse_interpolation_and_missing_distance_policy():
    trx = Trajectory.__new__(Trajectory)
    trx.x = np.array([np.nan, 0., np.nan, 4., np.nan])
    trx.y = np.array([np.nan, 0., np.nan, 0., np.nan])
    trx.theta = None
    trx.va = SimpleNamespace(fps=10, on=np.array([]), openLoop=True)
    trx._max_proportion_lost_fms = .6
    trx._interpolate()
    trx._calcDistances()
    trx._calcSpeeds()
    np.testing.assert_array_equal(trx.nan, [True, False, True, False, True])
    np.testing.assert_allclose(trx.sp, [0, 0, 20, 20, 0])
    current, scaled = walking_masks(trx.sp, 8, .9)
    assert current[2] and scaled[2]  # originally missing, now interpolated: retained
    assert np.isnan(trx.x[0]) and np.isfinite(trx.sp[0])


def test_histogram_bounds_nan_post_mask_and_denominator():
    xy = [np.array([0., 1., 2., 3., np.nan, np.inf]), np.array([0., 1., 2., 1., np.nan, 1.])]
    mp, selected = histogram(xy, np.ones(6, bool), [2, 2], [[0, 2], [0, 2]], post_offset=1)
    assert selected == 4  # NaN excluded; infinity retained then rejected by histogram
    assert mp.sum() == 2  # rightmost histogram edge included
    assert (mp/6).sum() == pytest.approx(1/3)  # all window frames in denominator
    empty, _ = histogram(xy, np.zeros(6, bool), [2, 2], [[0, 2], [0, 2]])
    assert not empty.any()


def test_mass_and_shape_are_separate():
    a = np.array([[.1, .2], [0, .1]])
    metrics = map_metrics(a, a*1.1, 8)
    assert metrics["relative_mass_delta"] == pytest.approx(.1)
    assert metrics["shape_tv"] == pytest.approx(0, abs=1e-15)
    assert metrics["centroid_shift_mm"] == pytest.approx(0, abs=1e-15)


def test_notebook_extraction_preserves_current_recipes():
    recipes = read_recipes(ROOT/"notebooks/manuscript_figure_panels.ipynb")
    assert len(recipes) == 14
    assert "--pltHmVmin 1e-5" in recipes["extended_data_figure_13a_flat_pre"]["command"]
    assert "--rpib 3" in recipes["figure_4b_post"]["command"]


@pytest.mark.parametrize("ct,exp_fly,absolute_fly,role", [
    (CT.htl, 7, 17, 1), (CT.large2, 1, 1, 0), (CT.large2, 1, 3, 1),
])
def test_spatial_regions_partition_mirrored_chamber(ct, exp_fly, absolute_fly, role):
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.ct, va.ef, va.noyc = ct, exp_fly, False
    va.trxf = (exp_fly, absolute_fly) if role else (absolute_fly,)
    va.xf = Xformer(dict(fctr=.94, x=3, y=7), ct, np.zeros((720,720), np.uint8), True)
    trx = SimpleNamespace(pxPerMmFloor=ct.pxPerMmFloor())
    t = SimpleNamespace(circles=lambda f: [])
    lo, hi = va._heatmapBounds(role)
    shape = tuple(reversed([int(x) for x in (hi-lo)/2]))
    regions = region_masks(va,trx,t,role,lo,hi,shape)
    np.testing.assert_array_equal(sum(m.astype(int) for m in regions.values()), np.ones(shape))
    assert all(m.any() for m in regions.values())


def test_group_aggregation_preserves_separate_nonempty_eligibility():
    current = np.array([[[.1, 0]], [[0, 0]], [[.3, 0]]])
    scaled = np.array([[[0, 0]], [[0, .4]], [[.6, 0]]])
    group_current, group_scaled, keep_current, keep_scaled = aggregate_maps(current, scaled)
    np.testing.assert_allclose(group_current, [[.2, 0]])
    np.testing.assert_allclose(group_scaled, [[.3, .2]])
    np.testing.assert_array_equal(keep_current, [True, False, True])
    np.testing.assert_array_equal(keep_scaled, [False, True, True])


def test_empty_group_fails_instead_of_exporting_nan_maps():
    with pytest.raises(ValueError, match="No eligible"):
        aggregate_maps(np.zeros((2, 1, 1)), np.ones((2, 1, 1)))


@pytest.fixture
def report_scope():
    group = dict(
        panel="selected_panel", role=0, chamber="htl",
        relative_mass_delta=.2, shape_tv=.1, eligibility_added=1, eligibility_removed=2,
    )
    provenance = dict(
        recipes={"selected_recipe": {"panel": "selected_panel"}},
        recordings={"video.avi": {"cohorts": ["selected_cohort"]}},
        overlaps={},
        validation=dict(
            partial_run=False, walking_arrays_unchanged=True,
            exact_full_trajectory_mask_checks=3, exact_histogram_checks=3,
            cached_groups=[dict(panel="selected_panel", role=0, allclose=True, cache_mismatch=None)],
        ),
    )
    return provenance, [group]


def test_report_accepts_selected_recipe_subset_and_computes_eligibility(report_scope):
    provenance, groups = report_scope
    validate_report_inputs(provenance, groups)  # No requirement for 28 maps or two chambers.
    text = "\n".join(scope_and_validation(provenance, groups, []))
    assert "|htl|+20.000 to +20.000|10.000|1|2|" in text
    assert "No fly gains or loses" not in text
    assert "retaining the current classifier" not in text


@pytest.mark.parametrize("failure", ["partial", "missing", "mismatch", "duplicate", "changed_arrays"])
def test_report_rejects_unvalidated_inputs(report_scope, failure):
    provenance, groups = copy.deepcopy(report_scope)
    checks = provenance["validation"]
    if failure == "partial":
        checks["partial_run"] = True
    elif failure == "missing":
        checks["cached_groups"] = []
    elif failure == "mismatch":
        checks["cached_groups"][0]["cache_mismatch"] = "source changed"
    elif failure == "duplicate":
        checks["cached_groups"] *= 2
    else:
        checks["walking_arrays_unchanged"] = False
    with pytest.raises(ValueError):
        validate_report_inputs(provenance, groups)


def test_regional_summary_uses_rule_specific_denominators_and_undefined_ratios():
    groups = [dict(panel="panel", role=0, n_current=2, n_scaled=1)]
    regions = [
        dict(panel="panel", role=0, region="floor_interior", mass_current=.3, mass_scaled=.6, absolute_change=.3),
        dict(panel="panel", role=0, region="floor_edge_2mm", mass_current=.2, mass_scaled=0, absolute_change=.2),
        dict(panel="panel", role=0, region="outside_floor", mass_current=0, mass_scaled=0, absolute_change=0),
    ]
    interior, edge, outside = regional_summary(groups, regions)
    assert interior["relative_mass_delta"] == pytest.approx(3)
    assert edge["relative_mass_delta"] == pytest.approx(-1)
    assert outside["relative_mass_delta"] is None
    assert number(outside["relative_mass_delta"]) == "n/a"


def test_diagnostic_environment_restores_process_state_on_failure(tmp_path, monkeypatch):
    import __main__

    original_detector = object()
    fake_analyze = SimpleNamespace(FlyDetector=object())
    monkeypatch.setattr(__main__, "FlyDetector", original_detector, raising=False)
    monkeypatch.setitem(sys.modules, "analyze", fake_analyze)
    monkeypatch.setenv("MPLCONFIGDIR", "original_matplotlib_directory")
    (tmp_path / "work").mkdir()
    original_cwd, original_argv = Path.cwd(), sys.argv
    original_path, original_bytecode = sys.path.copy(), sys.dont_write_bytecode
    with pytest.raises(RuntimeError, match="deliberate"):
        with diagnostic_environment(tmp_path):
            assert Path.cwd() == tmp_path / "work"
            assert __main__.FlyDetector is fake_analyze.FlyDetector
            raise RuntimeError("deliberate failure")
    assert Path.cwd() == original_cwd
    assert sys.argv is original_argv
    assert sys.path == original_path
    assert sys.dont_write_bytecode == original_bytecode
    assert os.environ["MPLCONFIGDIR"] == "original_matplotlib_directory"
    assert __main__.FlyDetector is original_detector
