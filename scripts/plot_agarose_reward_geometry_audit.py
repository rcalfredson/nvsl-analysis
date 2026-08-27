#!/usr/bin/env python3
"""Plot reward-centered agarose geometry candidates and a complete null draw."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_reward_geometry_audit import select_maximin_complete_draw


CONFIG_FIELDS = ("video", "video_index", "va_fly", "trajectory_fly", "training")
SITE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"geometry audit CSV contains no rows: {path}")
    required = {
        *CONFIG_FIELDS,
        "physical_site",
        "candidate_x_px",
        "candidate_y_px",
        "physical_x_px",
        "physical_y_px",
        "passes_primary_geometry",
        "outer_radius_px",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "geometry audit CSV is missing plotting fields: " + ", ".join(missing)
        )
    return rows


def _config_key(row):
    return tuple(row[field] for field in CONFIG_FIELDS)


def _config_label(key):
    video, _video_index, va_fly, trajectory_fly, training = key
    return (
        f"{Path(video).name} | va_fly={va_fly}, trajectory_fly={trajectory_fly}, "
        f"training={training}"
    )


def _is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _select_complete_draw(rows, seed=101, strategy="maximin"):
    """Select one reproducible, mutually nonoverlapping primary draw."""
    by_site = {}
    for row in rows:
        if _is_true(row["passes_primary_geometry"]):
            by_site.setdefault(int(row["physical_site"]), []).append(row)
    expected_sites = sorted({int(row["physical_site"]) for row in rows})
    if any(site not in by_site for site in expected_sites):
        raise ValueError("this configuration has no primary candidate for at least one site")
    if strategy not in {"maximin", "random"}:
        raise ValueError("strategy must be 'maximin' or 'random'")
    outer_radius = float(rows[0]["outer_radius_px"])
    if strategy == "maximin":
        selected = select_maximin_complete_draw(
            [by_site[site] for site in expected_sites], outer_radius, seed=seed
        )
        if selected is None:
            raise ValueError(
                "this configuration has no nonoverlapping primary complete draw"
            )
        return {
            expected_sites[site_index]: row for site_index, row in selected.items()
        }

    rng = np.random.default_rng(int(seed))
    ordered_sites = sorted(expected_sites, key=lambda site: len(by_site[site]))
    for site in ordered_sites:
        rng.shuffle(by_site[site])

    def visit(site_index, selected):
        if site_index == len(ordered_sites):
            return selected
        site = ordered_sites[site_index]
        for candidate in by_site[site]:
            point = np.asarray(
                (float(candidate["candidate_x_px"]), float(candidate["candidate_y_px"]))
            )
            if all(
                np.linalg.norm(point - other_point) >= 2.0 * outer_radius - 1e-9
                for _other_site, _other_row, other_point in selected
            ):
                result = visit(site_index + 1, selected + [(site, candidate, point)])
                if result is not None:
                    return result
        return None

    selected = visit(0, [])
    if selected is None:
        raise ValueError("this configuration has no nonoverlapping primary complete draw")
    return {site: row for site, row, _point in selected}


def _base_chamber(ax, rows, *, title):
    first = rows[0]
    x0, y0, x1, y1 = (
        float(first[field])
        for field in ("floor_x0_px", "floor_y0_px", "floor_x1_px", "floor_y1_px")
    )
    nominal_radius = float(first["nominal_agarose_radius_px"])
    outer_radius = float(first["outer_radius_px"])
    reward = (
        float(first["reward_x_px"]),
        float(first["reward_y_px"]),
        float(first["reward_radius_px"]),
    )
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#fafafa", edgecolor="black")
    )
    physical_by_site = {}
    for row in rows:
        site = int(row["physical_site"])
        physical_by_site.setdefault(
            site, (float(row["physical_x_px"]), float(row["physical_y_px"]))
        )
    for site, center in sorted(physical_by_site.items()):
        color = SITE_COLORS[(site - 1) % len(SITE_COLORS)]
        ax.add_patch(Circle(center, nominal_radius, facecolor="#E69F00", alpha=0.35,
                            edgecolor=color, linewidth=1.5))
        ax.add_patch(Circle(center, outer_radius, fill=False, edgecolor=color,
                            linewidth=1.0, linestyle=":"))
        ax.text(center[0], center[1], str(site), ha="center", va="center", fontsize=8)
    ax.add_patch(
        Circle(reward[:2], reward[2], facecolor="#56B4E9", edgecolor="#0072B2", alpha=0.35)
    )
    ax.plot(reward[0], reward[1], marker="+", color="#0072B2", markersize=7)
    pad = 0.08 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")


def plot_geometry_audit(
    rows, *, output_path, seed=101, draw_selection="maximin", dpi=220
):
    selected = _select_complete_draw(rows, seed=seed, strategy=draw_selection)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), constrained_layout=True)
    _base_chamber(axes[0], rows, title="Candidate coverage")
    _base_chamber(
        axes[1], rows, title=f"{draw_selection.capitalize()} complete draw (seed {seed})"
    )

    for row in rows:
        site = int(row["physical_site"])
        color = SITE_COLORS[(site - 1) % len(SITE_COLORS)]
        axes[0].scatter(
            float(row["candidate_x_px"]),
            float(row["candidate_y_px"]),
            s=17 if _is_true(row["passes_primary_geometry"]) else 7,
            color=color if _is_true(row["passes_primary_geometry"]) else "#999999",
            alpha=0.8 if _is_true(row["passes_primary_geometry"]) else 0.16,
            linewidths=0,
        )

    outer_radius = float(rows[0]["outer_radius_px"])
    for site, row in sorted(selected.items()):
        color = SITE_COLORS[(site - 1) % len(SITE_COLORS)]
        center = (float(row["candidate_x_px"]), float(row["candidate_y_px"]))
        for ax in axes:
            ax.add_patch(
                Circle(
                    center,
                    outer_radius,
                    facecolor=color if ax is axes[1] else "none",
                    edgecolor=color,
                    alpha=0.16 if ax is axes[1] else 0.9,
                    linewidth=1.8,
                    linestyle="--",
                )
            )
            ax.plot(*center, marker="x", color=color, markersize=6)
        neighbor_sites = row.get("balanced_neighbor_sites", "")
        neighbor_label = (
            f" between A{neighbor_sites.replace('|', '/A')}" if neighbor_sites else ""
        )
        axes[1].text(
            center[0],
            center[1],
            f"V{site}{neighbor_label}\n"
            f"{float(row['angle_deg']):g}° | "
            f"gap {float(row['min_physical_agarose_gap_mm']):.1f} mm\n"
            f"{100.0 * float(row['outside_floor_area_fraction']):.1f}% out",
            ha="center",
            va="center",
            fontsize=7,
        )

    legend = [
        Line2D([], [], marker="o", linestyle="none", color="#999999", alpha=0.4,
               label="rejected candidate"),
        Line2D([], [], marker="o", linestyle="none", color="#0072B2",
               label="primary-eligible candidate"),
        Line2D([], [], linestyle="--", color="#333333", label="selected virtual outer circle"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=8)
    method = rows[0].get("candidate_method", "unknown")
    fig.suptitle(
        f"{_config_label(_config_key(rows[0]))} | candidates={method}", fontsize=11
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return selected


def plot_display_draw(rows, *, output_path, seed=101, dpi=220):
    """Plot a clean, publication-facing view of the selected analytical draw."""
    selected = _select_complete_draw(rows, seed=seed, strategy="maximin")
    first = rows[0]
    x0, y0, x1, y1 = (
        float(first[field])
        for field in ("floor_x0_px", "floor_y0_px", "floor_x1_px", "floor_y1_px")
    )
    nominal_radius = float(first["nominal_agarose_radius_px"])
    inner_radius = nominal_radius + (
        float(first.get("inner_radius_offset_mm", 0.0))
        * float(first["px_per_mm"])
    )
    outer_radius = float(first["outer_radius_px"])
    reward = (
        float(first["reward_x_px"]),
        float(first["reward_y_px"]),
        float(first["reward_radius_px"]),
    )
    physical_by_site = {}
    for row in rows:
        physical_by_site.setdefault(
            int(row["physical_site"]),
            (float(row["physical_x_px"]), float(row["physical_y_px"])),
        )

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="#f7f7f7",
            edgecolor="#555555",
            linewidth=1.2,
        )
    )
    for site, center in sorted(physical_by_site.items()):
        color = SITE_COLORS[(site - 1) % len(SITE_COLORS)]
        ax.add_patch(
            Circle(
                center,
                inner_radius,
                facecolor="#E69F00",
                alpha=0.30,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        ax.add_patch(
            Circle(
                center,
                outer_radius,
                fill=False,
                edgecolor=color,
                linewidth=1.1,
                linestyle=":",
            )
        )
        ax.text(
            center[0],
            center[1],
            str(site),
            ha="center",
            va="center",
            color="#222222",
            fontsize=11,
            fontweight="bold",
        )
    for site, row in sorted(selected.items()):
        color = SITE_COLORS[(site - 1) % len(SITE_COLORS)]
        center = (float(row["candidate_x_px"]), float(row["candidate_y_px"]))
        ax.add_patch(
            Circle(
                center,
                inner_radius,
                facecolor=color,
                alpha=0.16,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        ax.add_patch(
            Circle(
                center,
                outer_radius,
                fill=False,
                edgecolor=color,
                linewidth=1.6,
                linestyle=":",
            )
        )
        ax.text(
            center[0],
            center[1],
            f"V{site}",
            ha="center",
            va="center",
            color="#222222",
            fontsize=10,
            fontweight="bold",
        )
    ax.add_patch(
        Circle(
            reward[:2],
            reward[2],
            facecolor="white",
            edgecolor="#333333",
            linewidth=2.3,
        )
    )
    pad = 0.06 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Reward-matched virtual-circle placement", fontsize=15)
    ax.legend(
        handles=[
            Line2D([], [], color="#333333", linewidth=2.4, label="Reward circle"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="#E69F00",
                   markeredgecolor="#555555", alpha=0.45, label="Physical agarose positions"),
            Line2D([], [], color="#555555", linewidth=1.8, linestyle=":",
                   label="Selected virtual outer circle"),
            Line2D([], [], color="#555555", linewidth=1.1, linestyle=":",
                   label="Physical outer approach boundary"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Visualize reward-centered agarose geometry-audit candidates."
    )
    parser.add_argument("--candidates", required=True, help="Detailed audit CSV.")
    parser.add_argument(
        "--out", help="Output image path; required unless --list-configs is used."
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=1,
        help="1-based fly/training configuration to plot (default: 1).",
    )
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--draw-selection", choices=("maximin", "random"), default="maximin"
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--display-only",
        action="store_true",
        help="plot only a clean selected draw without audit labels",
    )
    args = parser.parse_args()

    rows = _read_rows(args.candidates)
    configs = sorted({_config_key(row) for row in rows})
    if args.list_configs:
        for index, key in enumerate(configs, 1):
            print(f"{index}: {_config_label(key)}")
        return 0
    if not args.out:
        parser.error("--out is required unless --list-configs is used")
    if args.config_index < 1 or args.config_index > len(configs):
        parser.error(f"--config-index must be between 1 and {len(configs)}")
    selected_key = configs[args.config_index - 1]
    selected_rows = [row for row in rows if _config_key(row) == selected_key]
    if args.display_only:
        selected = plot_display_draw(
            selected_rows,
            output_path=args.out,
            seed=args.seed,
            dpi=args.dpi,
        )
    else:
        selected = plot_geometry_audit(
            selected_rows,
            output_path=args.out,
            seed=args.seed,
            draw_selection=args.draw_selection,
            dpi=args.dpi,
        )
    print(f"Wrote {args.out}")
    print(
        "Selected angles: "
        + ", ".join(
            f"site {site}={float(row['angle_deg']):g}° "
            f"(gap={float(row['min_physical_agarose_gap_mm']):.2f} mm)"
            for site, row in sorted(selected.items())
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
