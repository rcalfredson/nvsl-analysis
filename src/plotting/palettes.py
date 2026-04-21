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

# Reusable accent colors, originally assembled for correlation plots but now
# shared more broadly across figure types.
ACCENT_BLUE = "#0072B2"
ACCENT_ORANGE = "#D55E00"
ACCENT_TEAL = "#44AA99"
ACCENT_GREEN = "#009E73"
ACCENT_GOLD = "#E69F00"
ACCENT_SKY = "#56B4E9"
ACCENT_WINE = "#882255"
ACCENT_PINK = "#CC79A7"
ACCENT_INDIGO = "#332288"
ACCENT_GREY = "#999999"

WALL_CONTACTS_CATEGORICAL = (
    ACCENT_BLUE,
    ACCENT_ORANGE,
    ACCENT_GREEN,
    ACCENT_WINE,
    ACCENT_INDIGO,
    ACCENT_GOLD,
)

NEUTRAL_DARK = "#44505C"
NEUTRAL_MID = "#73808C"
NEUTRAL_LIGHT = "#C6CDD3"
BRIGHT_YELLOW = "#F0E442"

FLY_COLS = (MUTED_CATEGORICAL[0], MUTED_CATEGORICAL[3])

NAMED_CATEGORICAL_PALETTES = {
    "default": MUTED_CATEGORICAL,
    "wall_contacts": WALL_CONTACTS_CATEGORICAL,
}


def _adjust_lightness(color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * amount))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


def resolve_categorical_palette(
    palette: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if palette is None:
        colors = list(MUTED_CATEGORICAL)
    elif isinstance(palette, str):
        colors = list(
            NAMED_CATEGORICAL_PALETTES.get(str(palette), MUTED_CATEGORICAL)
        )
    else:
        colors = [str(c) for c in palette]
        if not colors:
            colors = list(MUTED_CATEGORICAL)
    return colors


def categorical_colors(
    n: int | None = None,
    palette: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    colors = resolve_categorical_palette(palette)
    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(max(0, int(n)))]


def group_color(
    index: int, palette: str | list[str] | tuple[str, ...] | None = None
) -> str:
    return categorical_colors(index + 1, palette=palette)[index]


def group_accent_color(
    index: int, palette: str | list[str] | tuple[str, ...] | None = None
) -> str:
    return _adjust_lightness(group_color(index, palette=palette), 0.82)


def group_fill_color(
    index: int, palette: str | list[str] | tuple[str, ...] | None = None
) -> str:
    return _adjust_lightness(group_color(index, palette=palette), 1.12)


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
    "rewards_per_distance_vs_sli": ACCENT_BLUE,
    "rewards_per_minute_vs_sli": ACCENT_ORANGE,
    "first_n_reward_rate_vs_sli": ACCENT_TEAL,
    "median_distance_vs_sli": ACCENT_GREEN,
    "baseline_pi_vs_sli": ACCENT_GOLD,
    "pre_training_exploration_vs_sli": ACCENT_SKY,
    "early_sli_vs_total_rewards": ACCENT_WINE,
    "first_n_reward_timing_vs_sli": ACCENT_PINK,
    "fast_vs_strong_fast": ACCENT_INDIGO,
    "fast_vs_strong_strong": ACCENT_GREEN,
    "fast_vs_strong_overlap": ACCENT_PINK,
    "fast_vs_strong_other": ACCENT_GREY,
    "selected_top": ACCENT_INDIGO,
    "selected_bottom": ACCENT_WINE,
    "selected_other": ACCENT_GREY,
    "unused_bright_yellow": BRIGHT_YELLOW,
    "unused_slate_grey": ACCENT_GREY,
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
