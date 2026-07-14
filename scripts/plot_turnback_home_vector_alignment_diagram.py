#!/usr/bin/env python3
"""Draw a deterministic schematic of turnback home-vector alignment.

The diagram is intentionally independent of tracked trajectory data.  It shows a
successful dual-circle turnback and the two vectors evaluated at re-entry.  The
cosine of the angle between those vectors is the alignment metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path as MplPath


COLORS = {
    "trajectory": "#38BFA7",
    "inner_circle": "#46B8E9",
    "outer_circle": "#F2767A",
    "heading": "#3AAA35",
    "home": "#EF6C2C",
    "angle": "#42A9D6",
    "reward": "#6D3B8C",
}


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of a two-dimensional vector."""
    return vector / np.linalg.norm(vector)


def _draw_chamber(ax: plt.Axes) -> None:
    """Draw square outer and rounded inner chamber borders with matching strokes."""
    border_color = "0.12"
    border_width = 1.0
    outer = Rectangle(
        (-3.05, -3.05),
        6.10,
        6.10,
        facecolor="none",
        edgecolor=border_color,
        linewidth=border_width,
        zorder=0,
    )
    inner = FancyBboxPatch(
        (-2.88, -2.88),
        5.76,
        5.76,
        boxstyle="round,pad=0.0,rounding_size=0.23",
        facecolor="none",
        edgecolor=border_color,
        linewidth=border_width,
        zorder=0,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)


def _draw_trajectory(
    ax: plt.Axes,
    reentry: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    reward_radius: float,
) -> np.ndarray:
    """Draw a reward-to-reward successful turnback through the re-entry anchor."""
    reentry_heading = np.array(
        [np.cos(np.radians(-60.0)), np.sin(np.radians(-60.0))]
    )
    reentry_control = reentry - 0.20 * reentry_heading
    vertices = [
        (-0.08, 0.03),
        (-0.35, 0.20),
        (-0.45, 0.70),
        (-0.65, 1.05),
        (-0.90, 1.45),
        (-1.32, 1.22),
        (-1.55, 0.65),
        (-1.70, 0.25),
        (-1.65, -0.35),
        (-1.42, -0.45),
        (-1.28, -0.50),
        (-1.45, 0.45),
        (-1.35, 0.65),
        (-1.28, 0.72),
        tuple(reentry_control),
        tuple(reentry),
        (-0.88, 0.34),
        (-0.50, 0.12),
        (0.08, -0.02),
    ]
    # A Bezier curve lies within the convex hull of its control points. Because
    # the outer circle is convex, keeping every control point inside guarantees
    # that the illustrated episode never crosses the failure boundary.
    if any(np.hypot(x, y) >= outer_radius for x, y in vertices):
        raise ValueError("turnback trajectory must remain inside the outer circle")
    if any(np.hypot(*vertices[index]) >= reward_radius for index in (0, -1)):
        raise ValueError("between-reward trajectory endpoints must be in reward circle")

    # Sample through the designated re-entry (the end of the fifth cubic) and
    # verify that, after the first exit, the path never slips back inside early.
    episode_points = []
    for start in range(0, 15, 3):
        p0, p1, p2, p3 = (
            np.asarray(point, dtype=float) for point in vertices[start : start + 4]
        )
        for t in np.linspace(0.0, 1.0, 101):
            episode_points.append(
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t**2 * p2
                + t**3 * p3
            )
    episode_radii = np.linalg.norm(np.asarray(episode_points), axis=1)
    outside = np.flatnonzero(episode_radii >= inner_radius - 1e-6)
    if outside.size == 0 or np.any(
        episode_radii[outside[0] :] < inner_radius - 1e-6
    ):
        raise ValueError("turnback path re-enters the inner circle before its anchor")
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(vertices) - 1)
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor="none",
            edgecolor=COLORS["trajectory"],
            linewidth=2.2,
            capstyle="round",
            zorder=3,
        )
    )

    # Place a short direction arrow on the tangent of the second Bezier segment,
    # so its center is mathematically coincident with the trajectory.
    t = 0.48
    p0, p1, p2, p3 = (np.asarray(point, dtype=float) for point in vertices[3:7])
    point = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )
    tangent = _unit(
        3 * (1 - t) ** 2 * (p1 - p0)
        + 6 * (1 - t) * t * (p2 - p1)
        + 3 * t**2 * (p3 - p2)
    )
    ax.add_patch(
        FancyArrowPatch(
            point - 0.10 * tangent,
            point + 0.10 * tangent,
            arrowstyle="-|>",
            mutation_scale=12,
            color=COLORS["trajectory"],
            linewidth=1.6,
            zorder=4,
        )
    )
    return reentry_heading


def _draw_vector(
    ax: plt.Axes,
    origin: np.ndarray,
    vector: np.ndarray,
    *,
    color: str,
) -> None:
    endpoint = origin + vector
    ax.add_patch(
        FancyArrowPatch(
            origin,
            endpoint,
            arrowstyle="-|>",
            mutation_scale=15,
            color=color,
            linewidth=2.2,
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )


def _draw_alignment_angle(
    ax: plt.Axes,
    origin: np.ndarray,
    heading: np.ndarray,
    home: np.ndarray,
) -> float:
    heading_angle = float(np.degrees(np.arctan2(heading[1], heading[0]))) % 360
    home_angle = float(np.degrees(np.arctan2(home[1], home[0]))) % 360
    theta = (home_angle - heading_angle) % 360
    if theta > 180:
        heading_angle, home_angle = home_angle, heading_angle
        theta = 360 - theta

    radius = 0.48
    ax.add_patch(
        Arc(
            origin,
            2 * radius,
            2 * radius,
            theta1=heading_angle,
            theta2=home_angle,
            color=COLORS["angle"],
            linewidth=2.1,
            zorder=10,
        )
    )
    return theta


def _draw_fly_placeholder(
    ax: plt.Axes, origin: np.ndarray, heading: np.ndarray
) -> None:
    """Draw a minimal, replaceable fly marker at the metric anchor."""
    angle = float(np.degrees(np.arctan2(heading[1], heading[0])))
    ax.scatter(
        [origin[0]],
        [origin[1]],
        s=86,
        marker=(3, 0, angle - 90),
        facecolor="white",
        edgecolor="0.20",
        linewidth=1.1,
        zorder=11,
    )


def draw_diagram(
    output: Path,
    *,
    show_chamber: bool = True,
    show_fly: bool = True,
    show_title: bool = True,
) -> float:
    """Create the schematic, save it, and return the illustrated angle in degrees."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.6, 7.0))
    ax.set_aspect("equal")
    ax.set_xlim(-3.22, 3.22)
    ax.set_ylim(-4.55, 3.28)
    ax.axis("off")

    if show_chamber:
        _draw_chamber(ax)

    center = np.array([0.0, 0.0])
    reward_radius = 0.25
    inner_radius = 1.25
    outer_radius = 1.85
    reentry_angle = np.radians(150.0)
    reentry = inner_radius * np.array(
        [np.cos(reentry_angle), np.sin(reentry_angle)]
    )

    ax.add_patch(
        Circle(
            center,
            outer_radius,
            fill=False,
            edgecolor=COLORS["outer_circle"],
            linewidth=1.5,
            linestyle=(0, (2.0, 2.2)),
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            center,
            inner_radius,
            fill=False,
            edgecolor=COLORS["inner_circle"],
            linewidth=1.5,
            linestyle=(0, (2.0, 2.2)),
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            center,
            reward_radius,
            facecolor="white",
            edgecolor="0.40",
            linewidth=1.15,
            zorder=5,
        )
    )
    ax.plot(
        [-0.10, 0.10],
        [0.0, 0.0],
        color=COLORS["reward"],
        linewidth=1.6,
        zorder=6,
    )
    ax.plot(
        [0.0, 0.0],
        [-0.10, 0.10],
        color=COLORS["reward"],
        linewidth=1.6,
        zorder=6,
    )
    reentry_heading = _draw_trajectory(
        ax, reentry, inner_radius, outer_radius, reward_radius
    )

    home = _unit(center - reentry) * 1.28
    heading = reentry_heading * 1.28

    _draw_vector(
        ax,
        reentry,
        heading,
        color=COLORS["heading"],
    )
    _draw_vector(
        ax,
        reentry,
        home,
        color=COLORS["home"],
    )
    theta = _draw_alignment_angle(ax, reentry, heading, home)

    ax.scatter(
        [reentry[0]],
        [reentry[1]],
        s=24,
        facecolor="white",
        edgecolor="0.15",
        linewidth=1.0,
        zorder=10,
    )
    if show_fly:
        _draw_fly_placeholder(ax, reentry, heading)

    if show_title:
        ax.set_title(
            "Turnback home-vector alignment",
            fontsize=13,
            pad=8,
        )

    legend_handles = [
        Line2D([], [], color=COLORS["trajectory"], lw=2.2, label="Turnback path"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="0.40",
            label="Reward circle",
        ),
        Line2D(
            [],
            [],
            color=COLORS["inner_circle"],
            lw=1.5,
            linestyle=(0, (2.0, 2.2)),
            label="Inner circle (re-entry)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["outer_circle"],
            lw=1.5,
            linestyle=(0, (2.0, 2.2)),
            label="Outer circle",
        ),
        Line2D([], [], color=COLORS["heading"], lw=2.2, label="Heading vector"),
        Line2D([], [], color=COLORS["home"], lw=2.2, label="Home vector"),
        Line2D([], [], color=COLORS["angle"], lw=2.1, label=r"Alignment angle $\theta$"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.0, -3.58),
        bbox_transform=ax.transData,
        frameon=False,
        ncol=2,
        fontsize=9,
        handlelength=2.8,
    )
    ax.text(
        0.0,
        -3.28,
        r"Alignment $= \cos(\theta)$",
        fontsize=12,
        ha="center",
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return theta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("imgs/turnback_home_vector_alignment_diagram.pdf"),
        help="Output path; the suffix selects the format (default: %(default)s).",
    )
    parser.add_argument(
        "--no-chamber",
        action="store_true",
        help="Omit the removable chamber-border layer.",
    )
    parser.add_argument(
        "--no-fly",
        action="store_true",
        help="Omit the triangular fly placeholder.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the title for easier assembly into a multipanel figure.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    theta = draw_diagram(
        args.output,
        show_chamber=not args.no_chamber,
        show_fly=not args.no_fly,
        show_title=not args.no_title,
    )
    print(f"Wrote {args.output} (illustrated angle: {theta:.1f} degrees)")


if __name__ == "__main__":
    main()
