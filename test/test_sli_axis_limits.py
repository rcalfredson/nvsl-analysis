import warnings

import pytest

from src.plotting.sli_axis_limits import (
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
