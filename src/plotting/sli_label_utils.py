from __future__ import annotations


def pct_label(prefix, frac):
    if frac is None:
        return prefix
    return f"{prefix} {int(round(frac * 100))}% learners"
