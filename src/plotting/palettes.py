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
    "sli": [MUTED_CATEGORICAL[0], MUTED_CATEGORICAL[1]],
    "rpd": [MUTED_CATEGORICAL[2], MUTED_CATEGORICAL[3]],
    "commag": [MUTED_CATEGORICAL[4], MUTED_CATEGORICAL[8]],
    "meddist": [MUTED_CATEGORICAL[9], MUTED_CATEGORICAL[5]],
    "agarose": [MUTED_CATEGORICAL[2], MUTED_CATEGORICAL[6]],
}


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
    else:
        return FLY_COLS
