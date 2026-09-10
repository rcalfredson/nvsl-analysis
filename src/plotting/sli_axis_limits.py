from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from src.utils.local_config import load_local_analyze_config


@dataclass(frozen=True)
class SLIAxisLimits:
    mode: str
    limits: tuple[float, float] | None

    @property
    def fixed(self) -> bool:
        return self.mode == "fixed"


def load_plot_sli_axis_limits(opts, *, selected_groups: bool) -> SLIAxisLimits:
    """Resolve general or top/bottom-selected SLI limits for ``plotRewards``."""
    selected_args = (
        getattr(opts, "sli_extremes_ylim_mode", None),
        getattr(opts, "sli_extremes_ylim_min", None),
        getattr(opts, "sli_extremes_ylim_max", None),
    )
    use_selected_args = selected_groups and any(
        value is not None for value in selected_args
    )
    return load_sli_axis_limits(
        mode=(
            selected_args[0]
            if use_selected_args
            else getattr(opts, "sli_ylim_mode", None)
        ),
        minimum=(
            selected_args[1]
            if use_selected_args
            else getattr(opts, "sli_ylim_min", None)
        ),
        maximum=(
            selected_args[2]
            if use_selected_args
            else getattr(opts, "sli_ylim_max", None)
        ),
        config_prefix="SLI_EXTREMES_YLIM" if selected_groups else "SLI_YLIM",
        fallback_config_prefix="SLI_YLIM" if selected_groups else None,
    )


def load_sli_axis_limits(
    *,
    mode: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    config_prefix: str = "SLI_YLIM",
    fallback_config_prefix: str | None = None,
) -> SLIAxisLimits:
    """Resolve the shared y-axis policy for time-dependent SLI plots.

    Explicit arguments take precedence over ``.analyze.local.env``. Supplying
    either limit without a mode implies fixed mode, which makes the two CLI
    limit flags convenient to use on their own.
    """
    cfg = load_local_analyze_config()

    def _config_value(suffix):
        value = cfg.get(f"{config_prefix}_{suffix}")
        if value is None and fallback_config_prefix is not None:
            value = cfg.get(f"{fallback_config_prefix}_{suffix}")
        return value

    has_limit_override = minimum is not None or maximum is not None
    if mode is None:
        mode = "fixed" if has_limit_override else (_config_value("MODE") or "dynamic")
    mode = str(mode).strip().lower()
    if mode not in ("dynamic", "fixed"):
        raise ValueError(
            "SLI y-limit mode must be 'dynamic' or 'fixed'"
        )
    if mode == "dynamic":
        if has_limit_override:
            raise ValueError(
                "SLI y-limit bounds cannot be supplied with dynamic mode"
            )
        return SLIAxisLimits(mode=mode, limits=None)

    lo_value = minimum if minimum is not None else _config_value("MIN")
    hi_value = maximum if maximum is not None else _config_value("MAX")
    missing = []
    if lo_value is None:
        missing.append(f"{config_prefix}_MIN/y-limit minimum")
    if hi_value is None:
        missing.append(f"{config_prefix}_MAX/y-limit maximum")
    if missing:
        raise ValueError(
            "fixed SLI y-limit mode requires " + ", ".join(missing)
        )
    try:
        lo = float(lo_value)
        hi = float(hi_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SLI y-limit minimum and maximum must be numbers"
        ) from exc
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(
            "SLI y limits must be finite with maximum > minimum"
        )
    return SLIAxisLimits(mode=mode, limits=(lo, hi))


def warn_if_sli_values_clipped(values, limits: tuple[float, float], *, context: str):
    """Warn once when finite plotted geometry falls outside fixed SLI limits."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    lo, hi = limits
    data_lo = float(np.min(finite))
    data_hi = float(np.max(finite))
    if data_lo < lo or data_hi > hi:
        warnings.warn(
            f"{context}: fixed SLI y limits [{lo:g}, {hi:g}] clip plotted "
            f"values spanning [{data_lo:g}, {data_hi:g}]",
            UserWarning,
            stacklevel=2,
        )
