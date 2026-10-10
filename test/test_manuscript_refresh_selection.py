"""Exercise notebook refresh routing without running analyses or copying files."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/manuscript_figure_panels.ipynb"


def _notebook_code():
    return [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def _preview_notebook(categories, *, run=True, rebuild=False):
    calls, exports, copies = [], [], []
    scope = {
        "REFRESH_SETTINGS": {
            "types": set(categories), "run": run, "copy": True,
            "rebuild_inputs": rebuild,
        },
    }
    for source in _notebook_code():
        if "def run_manuscript_recipe(" in source:
            exec(source, scope)
            scope["display"] = lambda *args: None
            scope["run_manuscript_recipe"] = lambda recipe, **kwargs: calls.append(
                (recipe, kwargs)
            )
            scope["subprocess"] = SimpleNamespace(
                run=lambda command, **kwargs: exports.append(command)
            )
        elif "def copy_manuscript_panels(" in source:
            tree = ast.parse(source)
            tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
            scope["copy_manuscript_panels"] = lambda recipes, *args, **kwargs: copies.extend(recipes)
            exec(compile(tree, "<copy-cell>", "exec"), scope)
        else:
            exec(source, scope)
    if "_refresh_controls" in scope:
        scope["_refresh_controls"].close()
    return scope, calls, exports, copies


@pytest.mark.parametrize("categories", [
    set(), {"correlation"}, {"time"}, {"data"},
    {"correlation", "time", "data"},
    {"correlation", "time", "data", "heatmap", "sharp_turn", "trajectory"},
])
def test_refresh_selection_routes_runs_and_copy_to_same_types(categories):
    scope, calls, exports, copies = _preview_notebook(categories)
    classify = scope["recipe_refresh_type"]
    for recipe, kwargs in calls:
        selected = classify(recipe) in categories
        assert kwargs["run"] == selected, recipe["title"]
        assert kwargs.get("run_export", False) == (
            selected and recipe.get("output_type") == "data"
        ), recipe["title"]
    assert copies == [
        recipe for recipe in scope["MANUSCRIPT_RECIPES"]
        if classify(recipe) in categories
    ]
    assert not exports  # Plot input bundles/CSVs are reused by default.


def test_rebuilding_inputs_only_exports_selected_plot_dependencies():
    scope, calls, exports, copies = _preview_notebook({"time"}, rebuild=True)
    assert len(exports) == 10  # Two ED17 cohort exports for each of five panels.
    assert all("--export-agarose-sli-bundle" in command for command in exports)
    assert not scope["RUN_FIGURE_1Q_EXPORTS"]
    assert not scope["REGENERATE_EXTENDED_DATA_FIGURE_9B_BUNDLE"]


def test_preview_disables_runs_and_dependency_exports():
    scope, calls, exports, copies = _preview_notebook(
        {"data", "time", "correlation", "sharp_turn"}, run=False, rebuild=True,
    )
    assert all(not kwargs["run"] and not kwargs.get("run_export", False)
               for recipe, kwargs in calls)
    assert not exports


def test_copy_allows_4i_data_and_time_but_rejects_duplicate_destinations():
    scope, _, _, _ = _preview_notebook({'data', 'time'}, run=False)
    source = next(s for s in _notebook_code() if 'def copy_manuscript_panels(' in s)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, '<copy-function>', 'exec'), scope)
    recipes = [scope['figure_4i'], scope['figure_4i_data']]
    scope['copy_manuscript_panels'](recipes, run=False)
    with pytest.raises(ValueError, match='Duplicate manuscript copy destination'):
        scope['copy_manuscript_panels'](recipes + [scope['figure_4i_data']], run=False)


def test_large_chamber_final_sli_is_separate_from_4i_mean_companion():
    scope, _, _, _ = _preview_notebook({'data'}, run=False)
    for name, panel in [('figure_4i_data', '4i'), ('figure_4v_data', '4v'),
                        ('extended_data_figure_16j', 'ED16j')]:
        recipe = scope[name]
        assert '--keyed-sli-csv' in recipe['export_command']
        assert 't2_sb5_final_sli' in recipe['panels'][panel]['output']
        assert all('--sli-min-valid-sync-buckets' not in stage['command']
                   for stage in recipe['analysis_stages'])
    companion = scope['figure_4i_mean_data']
    assert 'mean-sli-csv' in companion['export_command']
    assert '4i_mean' in companion['panels']
    assert companion in scope['MANUSCRIPT_RECIPES']
    assert companion['panels']['4i_mean']['output'] != scope['figure_4i_data']['panels']['4i']['output']


def test_copy_transfers_full_data_artifacts_and_preflights_missing_files(tmp_path):
    scope, _, _, _ = _preview_notebook({'data'}, run=False)
    source = next(s for s in _notebook_code() if 'def copy_manuscript_panels(' in s)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, '<copy-function>', 'exec'), scope)
    scope['ROOT'] = tmp_path
    recipe = scope['extended_data_figure_16j']
    target = tmp_path / 'network'
    target.mkdir()
    for artifact in recipe['artifacts']:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact)
    stats = recipe['analysis_stages'][0]['learning_stats_output']
    (tmp_path / stats).unlink()
    with pytest.raises(FileNotFoundError):
        scope['copy_manuscript_panels']([recipe], target, run=True)
    assert not list(target.iterdir())
    (tmp_path / stats).write_text(stats)
    scope['copy_manuscript_panels']([recipe], target, run=True)
    assert (target / 'edfig16/16j.csv').is_file()
    assert (target / 'edfig16/16j_data' / Path(stats).name).read_text() == stats
    assert len(list((target / 'edfig16/16j_data').iterdir())) == len(recipe['artifacts']) - 1


def test_checkboxes_combine_types_and_persist_on_cell_rerun():
    pytest.importorskip("ipywidgets")
    source = next(s for s in _notebook_code() if "REFRESH_TYPES =" in s)
    scope = {"display": lambda *args: None}
    exec(source, scope)
    assert scope["REFRESH_SETTINGS"]["types"] == {"correlation"}
    boxes = dict(zip(scope["REFRESH_TYPES"], scope["_type_checkboxes"]))
    boxes["time"].value = True
    boxes["data"].value = True
    boxes["correlation"].value = False
    scope["_action_checkboxes"][0].value = False
    assert scope["REFRESH_SETTINGS"]["types"] == {"time", "data"}
    assert not scope["REFRESH_SETTINGS"]["run"]
    exec(source, scope)
    assert scope["REFRESH_SETTINGS"]["types"] == {"time", "data"}
    assert [box.value for box in scope["_type_checkboxes"]] == [
        False, True, True, False, False, False,
    ]
    scope["_refresh_controls"].close()


def test_all_manuscript_heatmaps_use_shared_ratio_300():
    import math
    import shlex

    scope, _, _, _ = _preview_notebook({'heatmap'}, run=False)
    recipes = [r for r in scope['MANUSCRIPT_RECIPES'] if r.get('plot_type') == 'heatmap']
    assert len(recipes) == 14
    assert not scope['HEATMAP_CACHE_SETTINGS']['rebuild']
    for recipe in recipes:
        tokens = shlex.split(recipe['command'])
        minimum = float(tokens[tokens.index('--pltHmVmin') + 1])
        maximum = float(tokens[tokens.index('--pltHmVmax') + 1])
        assert math.isclose(maximum / minimum, 300)
        assert (minimum, maximum) == ((1e-5, 3e-3) if '--rmCC' in tokens else (1e-6, 3e-4))
        assert recipe['image_format'] == 'pdf'


def test_manuscript_heatmap_renders_cache_and_archives_expected_pdf(tmp_path, monkeypatch):
    import matplotlib as mpl
    import numpy as np
    from src.plotting.heatmap_cache import save_heatmap_cache

    scope, _, _, _ = _preview_notebook({'heatmap'}, run=False)
    source = next(s for s in _notebook_code() if 'def run_manuscript_recipe(' in s)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'run_manuscript_recipe']
    exec(compile(tree, '<run-recipe>', 'exec'), scope)
    scope['ROOT'] = tmp_path
    scope['PLOT_STYLE']['font_family'] = 'DejaVu Sans'
    calls = []
    def ensure(root, command, target, **kwargs):
        calls.append((command, target, kwargs))
        save_heatmap_cache(target, [[{
            'probability': np.array([[0., 1e-5], [1e-4, 1e-3]]),
            'column': 0, 'fly': 0, 'header': 'T2 SB5', 'sample': 'n=2', 'circle': None,
        }]], {'figsize': [3.1, 6], 'nc': 1, 'nsr': 2, 'nsc': 1}, mpl.colormaps['jet'])
        return target
    monkeypatch.setattr('src.plotting.heatmap_workflow.ensure_heatmap_cache', ensure)
    recipe = scope['extended_data_figure_13a_flat_training']
    scope['run_manuscript_recipe'](recipe, run=False)
    assert not calls
    assert not (tmp_path / 'imgs').exists()
    scope['run_manuscript_recipe'](recipe, run=True)
    assert len(calls) == 1
    assert calls[0][1].name == 'ED13a_flat_T2_SB5.npz'
    assert calls[0][2]['reuse_paths'][0].name == 'small_flat_training.npz'
    assert calls[0][2]['rebuild'] is False
    source_pdf = tmp_path / 'imgs/heatmaps2.pdf'
    output_pdf = tmp_path / recipe['panels']['ED13a_flat_T2_SB5']['output']
    assert source_pdf.read_bytes().startswith(b'%PDF')
    assert output_pdf.read_bytes() == source_pdf.read_bytes()
