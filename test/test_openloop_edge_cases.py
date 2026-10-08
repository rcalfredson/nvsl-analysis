from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.inspect_openloop_edge_cases import apply_reference_exports, classify, led_side_preference, run
from scripts.inspect_openloop_walking import period_masks
from src.analysis.video_analysis import VideoAnalysis
from src.utils.common import CT
import src.utils.util as util


@pytest.mark.parametrize('missing', [False, True])
def test_preference_matches_existing_openloop_method(monkeypatch, missing):
    frames = dict(startTrain=[4], startPost=[44], v7=[4, 24], v0=[0, 14, 34])
    masks = period_masks(frames, 50)
    y = np.arange(50, dtype=float) % 13
    if missing:
        y[4] = np.nan
    trj = SimpleNamespace(y=y, bad=lambda: False)
    va = SimpleNamespace(trns=[SimpleNamespace(start=4, stop=44, yTop=5, yBottom=5, name=lambda: 'T1')],
                         startPre=0, alt=False, off=np.array(frames['v0']), cap=None,
                         ct=CT.regular, trx=[trj], flies=(0,),
                         _getOn=lambda t: np.array(frames['v7']), extractChamber=lambda img: img,
                         _append=lambda to, vals, f, n: to.append(vals))
    monkeypatch.setattr(util, 'readFrame', lambda cap, index: np.zeros((2, 2, 3), dtype=np.uint8))
    VideoAnalysis.posPrefOL(va)
    result = led_side_preference(trj, masks, 5)
    np.testing.assert_allclose([result['led_on'], result['led_off']], va.posPI[0][1:3], equal_nan=True)


def test_classification_preserves_threshold_and_missing_edge_cases():
    assert classify(-.2, 9.9, 10) == 'negative_PI_low_walking'
    assert classify(-.2, 10, 10) == 'negative_PI_other_walking'
    assert classify(0, 9.9, 10) == 'nonnegative_PI_low_walking'
    assert classify(.5, 9.9, 10) == 'nonnegative_PI_low_walking'
    assert classify(np.nan, 0, 10) == 'missing_measurement'


def reference_table():
    return pd.DataFrame(dict(panel=['1p', '1q', '1q'], subject_key=['standalone', 'exp', 'yoked'],
                             preference_pi_led_on=[-.6, -.5, .4], preference_pi_led_off=[-.4, -.6, .5]))


def test_configurable_reference_paths_validate_and_assign_roles(tmp_path):
    table = reference_table()
    opened = tmp_path / 'new preference export.csv'
    closed = tmp_path / 'new closed export.csv'
    table.loc[table.panel.eq('1q')].to_csv(opened, index=False)
    pd.DataFrame([dict(exp_subject_key='exp', yoked_subject_key='yoked')]).to_csv(closed, index=False)
    result = apply_reference_exports(table, open_loop_export=opened, closed_loop_export=closed)
    assert result.closed_loop_role.tolist() == ['standalone', 'experimental', 'yoked']
    assert 'closed_loop_role' not in table


def test_omitted_references_skip_lookup_explicitly(capsys):
    result = apply_reference_exports(reference_table())
    assert result.closed_loop_role.tolist() == ['standalone', 'not_checked', 'not_checked']
    assert 'validation skipped' in capsys.readouterr().out


def test_reference_inputs_are_independently_optional(tmp_path):
    table = reference_table()
    opened = tmp_path / 'preferences.csv'
    closed = tmp_path / 'roles.csv'
    table.loc[table.panel.eq('1q')].to_csv(opened, index=False)
    pd.DataFrame([dict(exp_subject_key='exp', yoked_subject_key='absent')]).to_csv(closed, index=False)
    assert apply_reference_exports(table, open_loop_export=opened).closed_loop_role.tolist() == [
        'standalone', 'not_checked', 'not_checked']
    assert apply_reference_exports(table, closed_loop_export=closed).closed_loop_role.tolist() == [
        'standalone', 'experimental', 'not_in_closed_export']


def test_supplied_reference_still_rejects_inconsistent_pi(tmp_path):
    opened = tmp_path / 'incorrect.csv'
    reference = reference_table().iloc[1:].copy()
    reference.loc[1, 'preference_pi_led_on'] = .9
    reference.to_csv(opened, index=False)
    with pytest.raises(ValueError, match='led_on PI differs'):
        apply_reference_exports(reference_table(), open_loop_export=opened)


@pytest.mark.parametrize('option', ['open_loop_export', 'closed_loop_export'])
def test_explicit_missing_reference_fails_before_video_processing(tmp_path, option):
    missing = tmp_path / 'missing.csv'
    with pytest.raises(FileNotFoundError, match='Reference export does not exist'):
        run(tmp_path / 'walking.csv', tmp_path / 'output', **{option: missing})
