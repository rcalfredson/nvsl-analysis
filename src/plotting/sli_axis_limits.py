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


def load_sli_axis_limits(
    *,
    mode: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> SLIAxisLimits:
    """Resolve the shared y-axis policy for time-dependent SLI plots.

    Explicit arguments take precedence over ``.analyze.local.env``. Supplying
    either limit without a mode implies fixed mode, which makes the two CLI
    limit flags convenient to use on their own.
    """
    cfg = load_local_analyze_config()
    has_limit_override = minimum is not None or maximum is not None
    if mode is None:
        mode = "fixed" if has_limit_override else cfg.get("SLI_YLIM_MODE", "dynamic")
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

    lo_value = minimum if minimum is not None else cfg.get("SLI_YLIM_MIN")
    hi_value = maximum if maximum is not None else cfg.get("SLI_YLIM_MAX")
    missing = []
    if lo_value is None:
        missing.append("SLI_YLIM_MIN/--sli-ylim-min")
    if hi_value is None:
        missing.append("SLI_YLIM_MAX/--sli-ylim-max")
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
