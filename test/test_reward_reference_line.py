import argparse
import ast
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from src.plotting.reward_reference_line import (
    add_reward_reference_line_arguments,
    draw_reward_reference_line,
)


@pytest.fixture
def parser():
    parser = argparse.ArgumentParser()
    add_reward_reference_line_arguments(parser)
    return parser


@pytest.mark.parametrize("tp", ["rpi", "rpip", "rpid", "rpipd", "rpd"])
def test_default_line_retains_color_and_width(parser, tp):
    fig, ax = plt.subplots()
    original = ax.axhline(color="k")
    line = draw_reward_reference_line(ax, tp, parser.parse_args([]))
    assert line.get_color() == original.get_color()
    assert line.get_linewidth() == original.get_linewidth()
    assert list(line.get_ydata()) == [0, 0]
    assert line.get_alpha() in (None, 1.0)
    plt.close(fig)


@pytest.mark.parametrize("tp", ["rpi", "rpip", "rpid", "rpipd", "rpd"])
def test_reward_pi_hidden_sli_softened_and_other_metrics_unchanged(parser, tp):
    opts = parser.parse_args(["--no-reward-pi-zero-line", "--sli-zero-line-alpha", "0.3"])
    fig, ax = plt.subplots()
    line = draw_reward_reference_line(ax, tp, opts)
    if tp in ("rpi", "rpip"):
        assert line is None
        assert not ax.lines
    elif tp in ("rpid", "rpipd"):
        assert line.get_alpha() == 0.3
    else:
        assert line.get_alpha() is None
    plt.close(fig)


def test_visibility_and_opacity_are_independent(parser):
    opts = parser.parse_args([
        "--no-sli-zero-line", "--reward-pi-zero-line-alpha", "0.4",
    ])
    fig, ax = plt.subplots()
    assert draw_reward_reference_line(ax, "rpid", opts) is None
    assert draw_reward_reference_line(ax, "rpi", opts).get_alpha() == 0.4
    # Callers without the new settings still get the original appearance.
    assert draw_reward_reference_line(ax, "rpi", SimpleNamespace()).get_alpha() == 1
    plt.close(fig)


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "bad"])
def test_invalid_opacity_rejected_by_cli(parser, value):
    with pytest.raises(SystemExit):
        parser.parse_args(["--sli-zero-line-alpha", value])


def test_manuscript_style_only_applies_to_direct_time_plots():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/manuscript_figure_panels.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {"Path": Path, "shlex": shlex}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Compile every cell, but execute only the style dictionary and function.
        tree = ast.parse(source)
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PLOT_STYLE" for t in node.targets)
            ) or (isinstance(node, ast.FunctionDef) and node.name == "styled_command"):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(notebook), "exec"), namespace)

    styled = namespace["styled_command"]
    recipe = {"command": "python analyze.py -v example.avi", "plot_type": "time"}
    command, _ = styled(recipe)
    tokens = shlex.split(command)
    assert "--no-reward-pi-zero-line" in tokens
    assert "--sli-zero-line" in tokens
    assert tokens[tokens.index("--sli-zero-line-alpha") + 1] == "0.3"

    # Explicit recipe overrides take precedence over the manuscript defaults.
    command, _ = styled(dict(recipe, command=recipe["command"] + " --reward-pi-zero-line --sli-zero-line-alpha 0.6"))
    tokens = shlex.split(command)
    assert "--no-reward-pi-zero-line" not in tokens
    assert tokens.count("--sli-zero-line-alpha") == 1
    assert tokens[tokens.index("--sli-zero-line-alpha") + 1] == "0.6"

    for command, plot_type in [
        ("python analyze.py -v example.avi", "correlation"),
        ("python scripts/plot_bundle.py --out example.pdf", "time"),
    ]:
        result, _ = styled({"command": command, "plot_type": plot_type})
        assert "zero-line" not in result
