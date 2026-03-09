import seaborn as sns

FLY_COLS = ("#1f4da1", "#a00000")

PAIRED = sns.color_palette("Paired", 12)  # has 12 distinct colors


def paired_slice(start, end, reverse=True):
    colors = PAIRED[start:end]
    return list(reversed(colors)) if reverse else colors


METRIC_PALETTES = {
    "sli": paired_slice(0, 2),
    "rpd": paired_slice(2, 4),
    "commag": paired_slice(6, 8),
    "meddist": paired_slice(8, 10),
}


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
    else:
        return FLY_COLS