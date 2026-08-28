from __future__ import annotations


def pct_label(prefix, frac):
    if frac is None:
        return prefix
    return f"{prefix} {int(round(frac * 100))}% learners"


def sli_extreme_plot_specs(*, top, bottom, top_fraction, bottom_fraction):
    """Return learner extremes in their canonical plot and legend order."""
    return (
        ("top", top, pct_label("Top", top_fraction), "-"),
        ("bottom", bottom, pct_label("Bottom", bottom_fraction), "--"),
    )
