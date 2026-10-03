"""Zero reference line controls for reward PI and SLI in plotRewards."""

import argparse
import math


def _line_alpha(value):
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("opacity must be a number from 0 to 1") from exc
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise argparse.ArgumentTypeError("opacity must be a number from 0 to 1")
    return alpha


def add_reward_reference_line_arguments(parser):
    for prefix, label in (("reward-pi", "reward PI"), ("sli", "SLI")):
        parser.add_argument(
            f"--{prefix}-zero-line",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"Show the horizontal zero line in plotRewards {label} time plots "
            "(default: enabled; includes training, post-training, and selected groups).",
        )
        parser.add_argument(
            f"--{prefix}-zero-line-alpha",
            type=_line_alpha,
            default=1.0,
            metavar="ALPHA",
            help=f"Opacity of the plotRewards {label} zero line, from 0 to 1 "
            "(default: 1). Line color and width retain their existing defaults.",
        )


def draw_reward_reference_line(ax, tp, opts):
    """Draw the existing black line, with overrides only for reward PI/SLI."""
    prefix = (
        "reward_pi" if tp in ("rpi", "rpip")
        else "sli" if tp in ("rpid", "rpipd")
        else None
    )
    if prefix is None:
        return ax.axhline(color="k")
    if not getattr(opts, f"{prefix}_zero_line", True):
        return None
    return ax.axhline(
        color="k", alpha=getattr(opts, f"{prefix}_zero_line_alpha", 1.0)
    )
