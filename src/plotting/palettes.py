from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors
import seaborn as sns


# Seaborn-like muted categorical colors, used as the house palette for bars,
# histograms, and grouped scatter points.
MUTED_CATEGORICAL = (
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
)

NEUTRAL_DARK = "#44505C"
NEUTRAL_MID = "#73808C"
NEUTRAL_LIGHT = "#C6CDD3"
BRIGHT_YELLOW = "#F0E442"

FLY_COLS = (MUTED_CATEGORICAL[0], MUTED_CATEGORICAL[3])


def _adjust_lightness(color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * amount))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


def categorical_colors(n: int | None = None) -> list[str]:
    colors = list(MUTED_CATEGORICAL)
    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(max(0, int(n)))]


def group_color(index: int) -> str:
    return categorical_colors(index + 1)[index]


def group_accent_color(index: int) -> str:
    return _adjust_lightness(group_color(index), 0.82)


def group_fill_color(index: int) -> str:
    return _adjust_lightness(group_color(index), 1.12)


def stacked_group_colors(index: int) -> tuple[str, str, str]:
    base = group_fill_color(index)
    edge = group_accent_color(index)
    secondary = _adjust_lightness(base, 1.22)
    return base, secondary, edge


PAIRED = sns.color_palette(MUTED_CATEGORICAL[:8], 8)


METRIC_PALETTES = {
    "sli": ["#1f78b4", "#a6cee3"],
    "rpd": ["#33a02c", "#b2df8a"],
    "commag": ["#ff7f00", "#fdbf6f"],
    "meddist": ["#6a3d9a", "#cab2d6"],
    "agarose": [MUTED_CATEGORICAL[2], MUTED_CATEGORICAL[6]],
    "turnback": [BRIGHT_YELLOW, _adjust_lightness(BRIGHT_YELLOW, 1.12)],
    "between_reward_maxdist": [
        "#8B5A2B",
        _adjust_lightness("#8B5A2B", 1.35),
    ],
    "between_reward_return_leg_dist": [
        MUTED_CATEGORICAL[6],
        _adjust_lightness(MUTED_CATEGORICAL[6], 1.18),
    ],
}


# Correlation-plot palette built from the Okabe-Ito family plus a few deeper
# accents. This lets us keep a stable semantic color per plot type.
CORRELATION_PLOT_COLORS = {
    "rewards_per_distance_vs_sli": "#0072B2",
    "rewards_per_minute_vs_sli": "#D55E00",
    "first_n_reward_rate_vs_sli": "#44AA99",
    "median_distance_vs_sli": "#009E73",
    "baseline_pi_vs_sli": "#E69F00",
    "pre_training_exploration_vs_sli": "#56B4E9",
    "early_sli_vs_total_rewards": "#882255",
    "first_n_reward_timing_vs_sli": "#CC79A7",
    "fast_vs_strong_fast": "#332288",
    "fast_vs_strong_strong": "#009E73",
    "fast_vs_strong_overlap": "#CC79A7",
    "fast_vs_strong_other": "#999999",
    "selected_top": "#332288",
    "selected_bottom": "#882255",
    "selected_other": "#999999",
    "unused_bright_yellow": BRIGHT_YELLOW,
    "unused_slate_grey": "#999999",
}


_CORRELATION_METRIC_FAMILIES = {
    "sli": "sli",
    "first_n_reward_span_s": "first_n_reward_timing",
    "first_n_selected_reward_span_s": "first_n_reward_timing",
    "time_to_nth_actual_reward_s": "first_n_reward_timing",
    "time_to_nth_selected_reward_s": "first_n_reward_timing",
    "cutoff_time_since_selected_window_start_s": "first_n_reward_timing",
    "cutoff_time_since_cutoff_training_start_s": "first_n_reward_timing",
    "selected_reward_rate_to_nth_per_min": "first_n_reward_rate",
    "actual_reward_count_by_cutoff": "total_rewards",
    "control_reward_count_by_cutoff": "total_rewards",
}


def correlation_plot_color(
    plot_key: str, fallback: str | None = None
) -> str:
    if fallback is None:
        fallback = MUTED_CATEGORICAL[0]
    return CORRELATION_PLOT_COLORS.get(str(plot_key), fallback)


def correlation_plot_color_for_metrics(
    x_metric: str | None, y_metric: str | None, fallback: str | None = None
) -> str:
    if fallback is None:
        fallback = MUTED_CATEGORICAL[0]

    x_family = _CORRELATION_METRIC_FAMILIES.get(str(x_metric or ""), str(x_metric or ""))
    y_family = _CORRELATION_METRIC_FAMILIES.get(str(y_metric or ""), str(y_metric or ""))
    family_pair = frozenset((x_family, y_family))

    if family_pair == frozenset(("sli", "first_n_reward_timing")):
        return correlation_plot_color("first_n_reward_timing_vs_sli", fallback)
    if family_pair == frozenset(("sli", "first_n_reward_rate")):
        return correlation_plot_color("first_n_reward_rate_vs_sli", fallback)
    if family_pair == frozenset(("sli", "rewards_per_minute")):
        return correlation_plot_color("rewards_per_minute_vs_sli", fallback)
    if family_pair == frozenset(("sli", "total_rewards")):
        return correlation_plot_color("early_sli_vs_total_rewards", fallback)

    return fallback


def paired_slice(start, end, reverse=True):
    colors = PAIRED[start:end]
    return list(reversed(colors)) if reverse else colors


def get_palette(tp):
    """Return a pair of colors (exp, yok) appropriate for this metric type."""
    if tp in ("rpid", "rpipd"):
        return METRIC_PALETTES["sli"]
    elif tp in ("rpd", "rpd_exp_min_yok"):
        return METRIC_PALETTES["rpd"]
    elif tp in ("commag", "commag_exp_min_yok"):
        return METRIC_PALETTES["commag"]
    elif tp in ("meddist", "meddist_exp_min_yok"):
        return METRIC_PALETTES["meddist"]
    elif tp in ("agarose", "agarose_exp_min_yok"):
        return METRIC_PALETTES["agarose"]
    elif tp in ("turnback", "turnback_exp_min_yok"):
        return METRIC_PALETTES["turnback"]
    elif tp in ("between_reward_maxdist", "between_reward_maxdist_exp_min_yok"):
        return METRIC_PALETTES["between_reward_maxdist"]
    elif tp in (
        "between_reward_return_leg_dist",
        "between_reward_return_leg_dist_exp_min_yok",
    ):
        return METRIC_PALETTES["between_reward_return_leg_dist"]
    else:
        return FLY_COLS
