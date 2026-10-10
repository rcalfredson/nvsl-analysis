"""Check cache fidelity against the manuscript renderer using synthetic counts."""

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting.heatmap_cache import load_heatmap_cache, render_heatmap_cache
from src.plotting.heatmap_style import apply_heatmap_text_layout, heatmap_log_formatter


@pytest.mark.parametrize('chamber', ['htl', 'large'])
@pytest.mark.parametrize('periods', [('training',), ('pre', 'training', 'post')])
def test_export_preserves_aggregate_probabilities_and_native_layout(tmp_path, chamber, periods):
    cv2 = pytest.importorskip('cv2')
    from src.utils import util
    from src.utils.common import CT
    from src.utils.constants import HEATMAP_DIV, POST_SYNC, ST

    # Load the production plot function without analyze.py's CLI/import side effects.
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'analyze.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'plotHeatmaps')
    cache = tmp_path / 'maps.npz'
    options = SimpleNamespace(
        hm_map_export=cache, bg=None, hm='O', fontSize=20, num_trainings=None,
        hm_periods=periods, hm_vmin='1e-6', hm_vmax='3e-3', imageFormat='png',
        hm_pre_minutes=10, rpiPostBucketLenMin=3,
    )
    scope = dict(
        opts=options, np=np, mpl=mpl, plt=plt, cv2=cv2, util=util,
        CT=CT, OP_LIN='I', P=False, F2T=True, HEATMAP_DIV=HEATMAP_DIV,
        POST_SYNC=POST_SYNC, ST=ST, COL_W=(255, 255, 255), COL_BK=(0, 0, 0),
        pch=lambda a, b: b, pcap=lambda s: s,
        prepare_paired_heatmaps=lambda vas, opts, indices: (vas, False),
        _resolve_heatmap_training_indices=lambda trns, selector: list(range(len(trns))),
        _parse_hm_bounds_arg=lambda value, flag: (float(value),) * 3,
        apply_heatmap_text_layout=apply_heatmap_text_layout,
        heatmap_log_formatter=heatmap_log_formatter,
        writeImage=lambda *args, **kwargs: None, HEATMAPS_IMG_FILE='heatmaps%s.png',
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<plotHeatmaps>', 'exec'), scope)
    counts = np.array([[0., 1., 100.], [10., 20., 200.]])
    training = SimpleNamespace(name=lambda: 'training 1', sname=lambda: 't1',
                               circles=lambda f: [(5, 6, 2)])
    def recording(length, multiplier):
        data = (counts * multiplier, length, (0., 0.))
        return SimpleNamespace(
            gidx=0, ct=getattr(CT, chamber), trns=[training], flies=[0, 1], noyc=False,
            heatmap=[[data], [data]], heatmapPost=[[data], [data]], heatmapPre=[data, data],
            xf=SimpleNamespace(f2t=lambda x, y, **kwargs: (x, y)), trxf=[0, 1],
            mirror=lambda xy: xy, heatmapOOB=False,
        )
    vas = [recording(1e8, 1), recording(2e8, 3)]
    with mpl.rc_context({'font.family': 'DejaVu Sans', 'font.size': 20}):
        scope['plotHeatmaps'](vas)
        native = plt.gcf()
        rows, layout, cmap = load_heatmap_cache(cache)
        expected = (counts / 1e8 + counts * 3 / 2e8) / 2
        for row in rows:
            for panel in row:
                np.testing.assert_array_equal(panel['probability'], expected)
        assert rows[0][0]['probability'][0, 0] == 0
        assert rows[0][0]['probability'][0, 1] < 1e-6
        rendered = render_heatmap_cache(cache, 1e-6, 3e-3, font_family='DejaVu Sans')
        try:
            assert len(native.axes) == len(rendered.axes)
            for original, cached in zip(native.axes, rendered.axes):
                np.testing.assert_allclose(original.get_position().bounds, cached.get_position().bounds)
                if original.images:
                    np.testing.assert_array_equal(original.images[0].get_array(), cached.images[0].get_array())
                    assert original.get_title(loc='left') == cached.get_title(loc='left')
                    assert original.get_title(loc='right') == cached.get_title(loc='right')
                    assert len(original.patches) == len(cached.patches)
                    for a, b in zip(original.patches, cached.patches):
                        np.testing.assert_allclose(a.center, b.center)
                        assert a.radius == b.radius
            lower_floor = render_heatmap_cache(cache, 1e-9, 3e-4, font_family='DejaVu Sans')
            try:
                image = next(ax.images[0] for ax in lower_floor.axes if ax.images)
                assert image.get_array()[0, 1] == expected[0, 1]
                assert image.norm.vmin == 1e-9
            finally:
                plt.close(lower_floor)
            after, _, _ = load_heatmap_cache(cache)
            np.testing.assert_array_equal(after[0][0]['probability'], expected)
        finally:
            plt.close(native)
            plt.close(rendered)


@pytest.mark.parametrize('bounds', [(0, 1), (1, 1), (-1, 1), (1, float('inf'))])
def test_bad_bounds_fail_before_loading_cache(bounds):
    with pytest.raises(ValueError, match='0 < vmin < vmax'):
        render_heatmap_cache('not-needed.npz', *bounds)


def test_notebook_reuses_maps_after_scale_changes_and_refreshes_changed_recipes(tmp_path, monkeypatch):
    import json
    from src.plotting.heatmap_cache import save_heatmap_cache

    notebook = Path(__file__).resolve().parents[1] / 'notebooks/heatmap_scale_experiment.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    scope = {}
    monkeypatch.chdir(notebook.parent.parent)
    exec(''.join(cells[1]['source']), scope)
    scope['display'] = lambda *args: None
    exec(''.join(cells[3]['source']), scope)
    scope.update(OUT=tmp_path, CACHE=tmp_path, RUN=True, FONT_FAMILY='DejaVu Sans')
    calls = []
    def fake_analysis(argv, **kwargs):
        calls.append(argv)
        output = argv[argv.index('--hm-map-export') + 1]
        save_heatmap_cache(output, [[{
            'probability': np.array([[0, 1e-8], [1e-5, 1e-3]]),
            'column': 0, 'fly': 0, 'header': 'Training 2', 'sample': 'n=2', 'circle': None,
        }]], {'figsize': [3.1, 6], 'nc': 1, 'nsr': 2, 'nsc': 1}, mpl.colormaps['jet'])
    monkeypatch.setattr(scope['subprocess'], 'run', fake_analysis)
    # First run: two analyses for eight candidate images.
    render_cell = ''.join(cells[4]['source'])
    exec(render_cell, scope)
    assert len(calls) == 2
    assert len(list(tmp_path.glob('*.png'))) == 8
    # A new scale only renders, even if the original recordings are unavailable.
    scope['CANDIDATES'] = [('ratio_50', 50)]
    scope['MAXIMA'] = {'small': 2e-3, 'large': 2e-4}
    exec(render_cell, scope)
    assert len(calls) == 2
    details = json.loads((tmp_path / 'small_flat_training_ratio_50.json').read_text())
    assert details['ratio'] == 50
    assert details['vmax'] == 2e-3
    # A changed analysis recipe refreshes that recipe only.
    scope['RECIPES']['small_flat_training'][1]['command'] += ' --hm-sync-bucket-tail-minutes 5'
    exec(render_cell, scope)
    assert len(calls) == 3
    scope['REBUILD_CACHE'] = True
    exec(render_cell, scope)
    assert len(calls) == 5
