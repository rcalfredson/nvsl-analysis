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


def load_sli_axis_limits() -> SLIAxisLimits:
    """Load and validate the shared y-axis policy for time-dependent SLI plots."""
    cfg = load_local_analyze_config()
    mode = cfg.get("SLI_YLIM_MODE", "dynamic").strip().lower()
    if mode not in ("dynamic", "fixed"):
        raise ValueError(
            ".analyze.local.env: SLI_YLIM_MODE must be 'dynamic' or 'fixed'"
        )
    if mode == "dynamic":
        return SLIAxisLimits(mode=mode, limits=None)

    missing = [
        key for key in ("SLI_YLIM_MIN", "SLI_YLIM_MAX") if key not in cfg
    ]
    if missing:
        raise ValueError(
            ".analyze.local.env: fixed SLI_YLIM_MODE requires " + ", ".join(missing)
        )
    try:
        lo = float(cfg["SLI_YLIM_MIN"])
        hi = float(cfg["SLI_YLIM_MAX"])
    except ValueError as exc:
        raise ValueError(
            ".analyze.local.env: SLI_YLIM_MIN and SLI_YLIM_MAX must be numbers"
        ) from exc
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(
            ".analyze.local.env: SLI y limits must be finite with MAX > MIN"
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
