from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.stats import f_oneway

from src.plotting.p_value_format import format_plot_p_value
import src.utils.util as util
from src.utils.common import (
    areaUnderCurve,
    pick_above_or_expand,
    ttest_1samp,
    ttest_ind,
)
from src.utils.local_config import load_local_analyze_config, parse_local_bool

SHOW_AUC_P_VALUES_KEY = "SHOW_AUC_P_VALUES"


@dataclass(frozen=True)
class AUCTestResult:
    p_value: float
    ns: tuple[int, ...]
    test: str


def signed_auc_values(series: np.ndarray) -> np.ndarray:
    """Return one signed trapezoidal AUC per row."""
    return areaUnderCurve(np.asarray(series, dtype=float))


def auc_or_abc_test_values(
    values: np.ndarray, *, between_curves: bool
) -> np.ndarray:
    """
    Preserve direction for AUC tests while retaining legacy ABC magnitudes.

    ``plotRewards`` historically applied an absolute-value transform to both
    AUC and area-between-curves (ABC) samples. AUC comparisons are now signed;
    ABC remains unchanged until its definition is considered separately.
    """
    values = np.asarray(values, dtype=float)
    return np.abs(values) if between_curves else values


def auc_samples(series_by_group: Sequence[np.ndarray]) -> list[np.ndarray]:
    samples = []
    for series in series_by_group:
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            samples.append(np.array([], dtype=float))
            continue
        aucs = signed_auc_values(arr)
        samples.append(aucs[np.isfinite(aucs)])
    return samples


def compute_auc_test(
    series_by_group: Sequence[np.ndarray], *, min_n_per_group: int = 2
) -> AUCTestResult | None:
    samples = auc_samples(series_by_group)
    ns = tuple(int(s.size) for s in samples)
    if len(samples) < 2 or any(n < int(min_n_per_group) for n in ns):
        return None

    try:
        if len(samples) == 2:
            _t, p, _n0, _n1, _msg = ttest_ind(
                samples[0], samples[1], silent=True, min_n=min_n_per_group
            )
            return AUCTestResult(float(p), ns, "Welch t-test")

        _f, p = f_oneway(*samples)
        return AUCTestResult(float(p), ns, "one-way ANOVA")
    except Exception:
        return None


def compute_single_group_auc_test(
    series: np.ndarray,
    *,
    reference_series: np.ndarray | None = None,
    min_n: int = 2,
) -> AUCTestResult | None:
    """Test one group's signed AUCs against zero or a paired reference.

    When ``reference_series`` is supplied, the test is performed on the
    within-video AUC differences.  This is equivalent to a paired t-test and
    keeps experimental/yoked pairs aligned when either member is missing.
    Without a reference, the plotted series can be an experimental-only
    measure or an already-computed experimental-minus-yoked difference.
    """
    try:
        aucs = signed_auc_values(np.asarray(series, dtype=float))
        test_name = "one-sample t-test"
        if reference_series is not None:
            reference_aucs = signed_auc_values(
                np.asarray(reference_series, dtype=float)
            )
            if aucs.shape != reference_aucs.shape:
                return None
            aucs = aucs - reference_aucs
            test_name = "paired t-test"

        _t, p, n = ttest_1samp(aucs, 0, min_n=min_n)
        if not np.isfinite(p):
            return None
        return AUCTestResult(float(p), (int(n),), test_name)
    except (IndexError, TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def auc_p_values_enabled() -> bool:
    cfg = load_local_analyze_config()
    raw_value = cfg.get(SHOW_AUC_P_VALUES_KEY)
    if raw_value is None:
        return True
    return parse_local_bool(raw_value, key_name=SHOW_AUC_P_VALUES_KEY)


def format_auc_stars(p_value: float, *, include_p_value: bool | None = None) -> str:
    if not np.isfinite(p_value):
        return ""
    stars = util.p2stars(p_value, nanR="")
    if include_p_value is None:
        include_p_value = auc_p_values_enabled()
    if not include_p_value:
        return stars
    return f"{stars} (p = {format_plot_p_value(p_value)})"


def format_auc_label(
    result: AUCTestResult, *, include_p_value: bool | None = None
) -> str:
    n_text = ",".join(str(n) for n in result.ns)
    return f"AUC (n={n_text}): {format_auc_stars(result.p_value, include_p_value=include_p_value)}"


def add_auc_label(
    ax,
    *,
    x: float,
    y_anchor: float,
    ylim: list[float],
    span: float,
    result: AUCTestResult,
    existing_ys: Sequence[float] = (),
    font_size: float | int = 10,
    top_pad: float | None = None,
):
    if result is None or not np.isfinite(result.p_value):
        return None
    if not np.isfinite(y_anchor):
        return None

    base_y = y_anchor + 0.16 * span
    avoid_ys = [y_anchor] + [
        float(y) for y in existing_ys if y is not None and np.isfinite(y)
    ]
    y_auc, va_align = pick_above_or_expand(
        base_y,
        avoid_ys,
        ylim,
        span_override=span,
    )
    if y_auc is None:
        return None

    plt_text = util.pltText(
        float(x),
        y_auc,
        format_auc_label(result),
        ha="left",
        va=va_align,
        size=font_size,
        color="0",
    )
    plt_text._y_ = float(y_anchor)
    plt_text._final_y_ = float(y_auc)

    if top_pad is None:
        top_pad = max(0.10, 0.006 * float(font_size))
    ylim[1] = max(float(ylim[1]), float(y_auc) + float(top_pad) * float(span))

    return plt_text
