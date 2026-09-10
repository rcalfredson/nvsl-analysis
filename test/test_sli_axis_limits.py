import warnings
from types import SimpleNamespace

import pytest

from src.plotting.sli_axis_limits import (
    load_plot_sli_axis_limits,
    load_sli_axis_limits,
    warn_if_sli_values_clipped,
)


def test_sli_axis_limits_default_to_dynamic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policy = load_sli_axis_limits()
    assert policy.mode == "dynamic"
    assert policy.limits is None


def test_sli_axis_limits_load_fixed_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=-0.5\nSLI_YLIM_MAX=2\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits()
    assert policy.fixed
    assert policy.limits == (-0.5, 2.0)


def test_cli_sli_limits_override_config_and_imply_fixed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=dynamic\nSLI_YLIM_MIN=-9\nSLI_YLIM_MAX=9\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits(minimum=-0.25, maximum=1.5)
    assert policy.fixed
    assert policy.limits == (-0.25, 1.5)


def test_selected_sli_limits_override_general_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=-0.2\nSLI_YLIM_MAX=1\n"
        "SLI_EXTREMES_YLIM_MODE=fixed\n"
        "SLI_EXTREMES_YLIM_MIN=-0.2\nSLI_EXTREMES_YLIM_MAX=2.1\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits(
        config_prefix="SLI_EXTREMES_YLIM",
        fallback_config_prefix="SLI_YLIM",
    )
    assert policy.fixed
    assert policy.limits == (-0.2, 2.1)


def test_selected_sli_policy_falls_back_to_general_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=-0.2\nSLI_YLIM_MAX=1\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits(
        config_prefix="SLI_EXTREMES_YLIM",
        fallback_config_prefix="SLI_YLIM",
    )
    assert policy.fixed
    assert policy.limits == (-0.2, 1.0)


def test_selected_cli_limits_override_selected_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_EXTREMES_YLIM_MODE=fixed\n"
        "SLI_EXTREMES_YLIM_MIN=-0.2\nSLI_EXTREMES_YLIM_MAX=2.1\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits(
        minimum=-0.5,
        maximum=3,
        config_prefix="SLI_EXTREMES_YLIM",
        fallback_config_prefix="SLI_YLIM",
    )
    assert policy.limits == (-0.5, 3.0)


def test_plot_scope_selects_general_or_extremes_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=-0.2\nSLI_YLIM_MAX=1\n"
        "SLI_EXTREMES_YLIM_MODE=fixed\n"
        "SLI_EXTREMES_YLIM_MIN=-0.2\nSLI_EXTREMES_YLIM_MAX=2.1\n",
        encoding="utf-8",
    )
    opts = SimpleNamespace(
        sli_ylim_mode=None,
        sli_ylim_min=None,
        sli_ylim_max=None,
        sli_extremes_ylim_mode=None,
        sli_extremes_ylim_min=None,
        sli_extremes_ylim_max=None,
    )
    assert load_plot_sli_axis_limits(opts, selected_groups=False).limits == (
        -0.2,
        1.0,
    )
    assert load_plot_sli_axis_limits(opts, selected_groups=True).limits == (
        -0.2,
        2.1,
    )


def test_cli_fixed_mode_can_use_configured_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(
        "SLI_YLIM_MODE=dynamic\nSLI_YLIM_MIN=-0.5\nSLI_YLIM_MAX=2\n",
        encoding="utf-8",
    )
    policy = load_sli_axis_limits(mode="fixed")
    assert policy.limits == (-0.5, 2.0)


def test_cli_dynamic_mode_rejects_limit_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="dynamic mode"):
        load_sli_axis_limits(mode="dynamic", minimum=-0.5, maximum=2.0)


@pytest.mark.parametrize(
    "text",
    [
        "SLI_YLIM_MODE=other\n",
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=-0.5\n",
        "SLI_YLIM_MODE=fixed\nSLI_YLIM_MIN=2\nSLI_YLIM_MAX=-0.5\n",
    ],
)
def test_sli_axis_limits_reject_invalid_config(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".analyze.local.env").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="SLI"):
        load_sli_axis_limits()


def test_fixed_sli_limits_warn_when_values_are_clipped():
    with pytest.warns(UserWarning, match="clip plotted values"):
        warn_if_sli_values_clipped([-0.6, 1.0], (-0.5, 2.0), context="test")


def test_fixed_sli_limits_do_not_warn_for_in_range_values():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_sli_values_clipped([-0.5, 2.0], (-0.5, 2.0), context="test")
    assert not caught
