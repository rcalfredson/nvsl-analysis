"""Truth cases for the exhaustive reference-support census."""
import numpy as np
import pytest

from src.analysis.tracking_gap_support import audit_window, observed_references
from scripts.audit_tracking_gap_support import notebook_strings, admitted_trajectory, json_scalar


def fixture():
    raw = {name: np.ones(50)*10 for name in ('x', 'y', 'w', 'h', 'theta')}
    raw['x'][:] = 0
    for values in raw.values():
        values[5:7] = np.nan
    geom = dict(floor=((-10, 0), (10, 20)), scale=1., reward_center=(0, 10), wells=(1., [(0, 5), (0, 15)]))
    episodes = [dict(start=a, stop=a+2, support_start=a-1, support_stop=a+3,
        value=1., heading_frames=[a, a+1]) for a in (18, 23, 28, 33, 38)]
    return raw, geom, episodes


def test_census_matches_hand_enumerated_candidates_without_modifying_data():
    raw, geom, eps = fixture()
    copy = {name: a.copy() for name, a in raw.items()}
    stats, donors, refs = audit_window(raw, 10., 0, 50, geom,
        {('home_vector_alignment', '3/5mm'): eps}, padding_seconds=0)
    assert stats['gap_count'] == stats['donor_gaps'] == 1
    assert stats['lost_seconds'] == stats['donor_lost_seconds'] == .2
    assert len(refs) == 5
    row = donors[0]
    assert row['length_candidates'] == row['spatial_candidates'] == 42
    assert row['exposed_candidates'] == row['minimum_five_candidates'] == 25
    assert row['heading_candidates'] == 15
    assert row['reference_episode_indices'] == list(range(5))
    assert row['source_gap_sampling_probability'] == 1
    for name in raw:
        np.testing.assert_equal(raw[name], copy[name])


def test_local_minimum_and_radius_specific_exposure_are_not_union_targets():
    raw, geom, eps = fixture()
    _, rows, _ = audit_window(raw, 10., 0, 50, geom,
        {('turnback_ratio', '3/5mm'): eps[:1], ('turnback_ratio', '13/15mm'): eps}, padding_seconds=0)
    small, large = rows
    assert small['exposed_candidates'] == 5 and small['minimum_five_candidates'] == 0
    assert large['minimum_five_candidates'] == 25
    assert small['heading_candidates'] is None


def test_boundary_loss_is_not_silently_put_in_donor_denominator():
    raw, geom, eps = fixture()
    for a in raw.values():
        a[:2] = np.nan; a[49:] = np.nan
    stats, rows, _ = audit_window(raw, 10., 0, 50, geom, {('tortuosity', 'full'): eps})
    assert stats['gap_count'] == 3 and stats['donor_gaps'] == 1
    assert stats['excluded_gap_count'] == 2
    assert stats['excluded_lost_seconds'] == pytest.approx(.3)
    assert len(rows) == 1
    assert rows[0]['length_candidates'] == 0  # Two seconds of context on each side.


def test_body_field_loss_blocks_reference_and_cross_gap_episode_is_not_clipped():
    raw, _, eps = fixture()
    raw['w'][18] = np.nan
    crossing = dict(start=4, stop=8, support_start=3, support_stop=9)
    observed = np.logical_and.reduce([np.isfinite(a) for a in raw.values()])
    runs, retained = observed_references(observed, [crossing]+eps, 0, 50)
    assert (7, 18) in runs and (19, 50) in runs
    assert len(retained) == 4


def test_no_gaps_and_no_episodes_remain_zero_denominator_cases():
    raw, geom, _ = fixture()
    for a in raw.values():
        a[:] = 1.
    stats, rows, refs = audit_window(raw, 10., 0, 50, geom, {('tortuosity', 'full'): []})
    assert stats['fully_observed_window'] and stats['donor_gaps'] == 0
    assert rows == refs == []


def test_notebook_source_extraction_does_not_execute_code(tmp_path):
    import json
    path = tmp_path/'recipe.ipynb'
    path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [
        "VIDEO_GLOBS = '/archive/c*_*.avi'\nraise RuntimeError('never execute')\n"]}]}))
    assert notebook_strings(path)['VIDEO_GLOBS'] == '/archive/c*_*.avi'


def test_checkpoint_round_trip_preserves_numpy_scalar_counts_and_flags():
    import json
    encoded = json.dumps(dict(frame=np.int64(50), complete=np.bool_(True), seconds=np.float64(.2)), default=json_scalar)
    assert json.loads(encoded) == dict(frame=50, complete=True, seconds=.2)
    with pytest.raises(TypeError):
        json.dumps({'unsupported': object()}, default=json_scalar)


def test_matched_reference_counts_deduplicate_donor_reuse_but_keep_fly_identity():
    import pandas as pd
    from scripts.report_tracking_gap_support import matched_reference_counts
    base = dict(genotype='ctrl', chamber='flat', metric='turnback_ratio', variant='3/5mm',
        period='T2_SB2-5', baseline_eligible=True, video_path='/video.avi',
        pair=0, exposed_candidates=5, reference_episode_indices='[0, 1]')
    donors = pd.DataFrame([base, base, base | dict(pair=1, reference_episode_indices='[0]')])
    result = matched_reference_counts(donors)
    assert result.iloc[0].matched_reference_episodes == 3
    donors['exposed_candidates'] = 0
    assert matched_reference_counts(donors).empty


def test_source_overlap_is_reported_without_reassigning_membership():
    import pandas as pd
    from scripts.report_tracking_gap_support import membership_overlaps
    source = pd.DataFrame([
        dict(video_path='/shared.avi', pair=0, dataset='antennae_removed_flat'),
        dict(video_path='/shared.avi', pair=0, dataset='epg_kir_flat'),
        dict(video_path='/shared.avi', pair=1, dataset='epg_kir_flat'),
    ])
    overlap = membership_overlaps(source)
    assert len(overlap) == 1
    assert overlap.iloc[0].pair == 0
    assert overlap.iloc[0].dataset_memberships == 'antennae_removed_flat | epg_kir_flat'
    assert len(source) == 3


def test_admission_honors_note_override_after_production_checks(monkeypatch):
    from types import SimpleNamespace
    import scripts.audit_tracking_gap_support as audit
    class Trajectory:
        def _isEmpty(self): return False
        def _setLostFrameThreshold(self): pass
        def _interpolate(self): self._bad = False
        def _calcDistances(self): pass
        def _suspiciousJumps(self): self._bad = True
        def bad(self): return self._bad
    monkeypatch.setattr(audit, 'Trajectory', Trajectory)
    va = SimpleNamespace(opts=SimpleNamespace())
    def note(_): va.trx[0]._bad = False
    va._readNoteFile = note
    context = dict(analysis=va, x=np.ones(10), y=np.ones(10), physical_id=1, video_path='video.avi')
    assert admitted_trajectory(context) == (True, True)
    assert va.trxf == (1,)
