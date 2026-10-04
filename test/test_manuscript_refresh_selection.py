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
