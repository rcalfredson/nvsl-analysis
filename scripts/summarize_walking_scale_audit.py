#!/usr/bin/env python3
"""Report saved walking-scale sensitivity outputs without accessing recordings.

Usage: python scripts/summarize_walking_scale_audit.py exports/diagnostics/walking_scale
The report describes measured changes. Scientific recommendations belong in a
reviewed audit note, not in generated text that could outlive its supporting data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shlex

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOOR_REGIONS = ("floor_interior", "floor_edge_2mm", "outside_floor")
COUNT_FIELDS = (
    "n_frames", "original_missing", "finite_positions", "finite_speeds", "finite_joint",
    "walking_current", "walking_scaled", "changed_frames", "added", "removed",
    "histogram_counts_current", "histogram_counts_scaled", "changed_originally_missing",
    "changed_step_touches_original_missing",
)
FRACTION_FIELDS = (
    "walking_current", "walking_scaled", "changed_frames",
    "histogram_counts_current", "histogram_counts_scaled",
)


def read_csv(path):
    """Empty skip lists need not have an existing CSV in older audit outputs."""
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def group_key(record):
    return record["panel"], int(record["role"])


def validate_report_inputs(provenance, groups):
    """Require complete, matching cache checks for the selected recipes, not 28 maps.

    Exact trajectory and histogram checks occur during collection. This second
    gate prevents reporting historical-cache agreement for partial, missing,
    duplicated or incompatible aggregate comparisons.
    """
    checks = provenance["validation"]
    if checks["partial_run"]:
        raise ValueError("Partial smoke runs cannot produce a cache-validated report")
    if not checks["walking_arrays_unchanged"]:
        raise ValueError("Production walking arrays were not verified unchanged")
    keys = [group_key(group) for group in groups]
    cached = checks["cached_groups"]
    cached_keys = [group_key(check) for check in cached]
    if not keys or len(keys) != len(set(keys)) or len(cached_keys) != len(set(cached_keys)):
        raise ValueError("Missing or duplicate group-map records")
    if set(keys) != set(cached_keys):
        raise ValueError("Missing historical cache comparisons for selected group maps")
    panels = {record["panel"] for record in provenance["recipes"].values()}
    if panels != {panel for panel, _ in keys}:
        raise ValueError("Not every selected recipe produced a group map")
    if not all(check["allclose"] and not check["cache_mismatch"] for check in cached):
        raise ValueError("Group maps or recipe/source/config provenance differ from historical caches")


def window_totals(groups, flies):
    """Pool frame counts for accounting; these are not equal-fly group-map means."""
    by_group = defaultdict(list)
    for row in flies:
        if row["status"] == "ok":
            by_group[group_key(row)].append(row)
    totals = []
    for group in groups:
        rows = by_group[group_key(group)]
        total = {name: group[name] for name in ("panel", "role", "cohort", "chamber", "period")}
        total.update(
            valid_fly_windows=len(rows),
            eligible_current=group["n_current"],
            eligible_scaled=group["n_scaled"],
        )
        for field in COUNT_FIELDS:
            total[field] = sum(int(row.get(field) or 0) for row in rows)
        for field in FRACTION_FIELDS:
            total[field + "_fraction"] = total[field] / total["n_frames"]
        totals.append(total)
    return totals


def regional_summary(groups, regions):
    """Compare region mass with the same separate eligibility as each group map.

    Absolute-change shares sum absolute per-fly changes before averaging. The
    three floor regions partition the map; reward regions overlap that partition.
    A relative change with zero baseline mass is undefined, represented by None.
    """
    by_group = defaultdict(list)
    for region in regions:
        by_group[group_key(region)].append(region)
    summary = []
    for group in groups:
        rows = by_group[group_key(group)]
        partition = [row for row in rows if row["region"] in FLOOR_REGIONS]
        total_mass = sum(float(row["mass_current"]) for row in partition)
        total_change = sum(float(row["absolute_change"]) for row in partition)
        for region in dict.fromkeys(row["region"] for row in rows):
            selected = [row for row in rows if row["region"] == region]
            current = sum(float(row["mass_current"]) for row in selected) / group["n_current"]
            scaled = sum(float(row["mass_scaled"]) for row in selected) / group["n_scaled"]
            change = sum(float(row["absolute_change"]) for row in selected)
            summary.append(dict(
                panel=group["panel"], role=group["role"], region=region,
                baseline_mass_share=current * group["n_current"] / total_mass if total_mass else 0,
                absolute_change_share=change / total_change if total_change else 0,
                relative_mass_delta=scaled / current - 1 if current else None,
            ))
    return summary


def number(value, *, scale=1, digits=3, signed=False):
    """Render undefined ratios as n/a rather than suggesting a zero change."""
    if value is None or not np.isfinite(value):
        return "n/a"
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value * scale, spec)


def role_name(role):
    return "exp" if int(role) == 0 else "yoked"


def table(headers, rows):
    return [
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
        *["|" + "|".join(map(str, row)) + "|" for row in rows],
    ]


def scope_and_validation(provenance, groups, skipped):
    checks = provenance["validation"]
    lines = [
        "# Walking-scale sensitivity analysis", "",
        "This generated report describes the measured sensitivity. It makes no recommendation "
        "about changing the production classifier and does not establish scientific equivalence.", "",
        "Both masks use identical production speed arrays. The collector verifies that production "
        "walking arrays remain unchanged and invokes no manuscript figure renderer.", "",
        f"Exact production-mask comparisons: **{checks['exact_full_trajectory_mask_checks']}** trajectories. "
        f"Exact baseline histogram comparisons: **{checks['exact_histogram_checks']}** fly/windows. "
        f"All **{len(checks['cached_groups'])}** selected aggregate baselines reproduce historical caches "
        "within rtol=1e-12, atol=1e-15, with matching recipe/source/config provenance.", "",
        f"Recordings: **{len(provenance['recordings'])} distinct files**, "
        f"**{sum(len(record['cohorts']) for record in provenance['recordings'].values())} cohort memberships**. "
        "Fly/window records repeated across recipes are not independent fly or calibration replicates.", "",
        "## Measured ranges by chamber", "",
    ]
    chamber_groups = defaultdict(list)
    for group in groups:
        chamber_groups[group["chamber"]].append(group)
    rows = []
    for chamber, selected in chamber_groups.items():
        mass = [group["relative_mass_delta"] for group in selected]
        spatial = [group["shape_tv"] for group in selected]
        added = sum(group["eligibility_added"] for group in selected)
        removed = sum(group["eligibility_removed"] for group in selected)
        rows.append([
            chamber,
            number(min(mass), scale=100, signed=True) + " to " + number(max(mass), scale=100, signed=True),
            number(max(spatial), scale=100), added, removed,
        ])
    lines += table(
        ["Chamber", "Group mass change %", "Maximum group shape TV %", "Eligibility additions", "Eligibility removals"],
        rows,
    )
    lines += ["", "Eligibility counts refer to fly/window records across selected panels, not distinct animals.", ""]
    reasons = Counter(row["reason"] for row in skipped)
    lines.append("Skipped initialization attempts: " + (
        "; ".join(f"{reason}: {count}" for reason, count in reasons.items()) if reasons else "none"
    ) + ".")
    overlaps = provenance.get("overlaps", {})
    lines += ["", f"Recording files shared across cohorts: **{len(overlaps)}**."]
    for path, cohorts in overlaps.items():
        lines.append(f"- `{path}`: {', '.join(cohorts)}")
    return lines


def group_tables(groups):
    lines = ["", "## Group maps", "",
        "Mass is histogram counts divided by all window frames, averaged equally across nonempty "
        "per-fly maps. Relative L1 includes both intensity and spatial changes. Shape TV compares "
        "maps after separately normalizing each to unit mass; 1% means 1% of conditional spatial "
        "probability would need reassignment between bins. That rescaling is diagnostic only.", ""]
    rows = []
    for group in groups:
        rows.append([
            group["panel"], role_name(group["role"]),
            f"{group['n_current']}→{group['n_scaled']}",
            number(group["mass_current"], digits=6) + "→" + number(group["mass_scaled"], digits=6),
            number(group["relative_mass_delta"], scale=100, signed=True),
            number(group["relative_l1"], scale=100),
            number(group["shape_tv"], scale=100),
            number(group["centroid_shift_mm"], digits=5),
        ])
    lines += table(
        ["Panel", "Role", "n current→scaled", "Mass current→scaled", "Mass change %",
         "Relative L1 %", "Shape TV %", "Centroid shift mm"], rows,
    )
    lines += ["", "## Across-fly distributions", "",
        "Walking changes are percentage points of all window frames. Distributions include all "
        "valid processed windows, including zero-count maps; group means above exclude empty maps.", ""]
    rows = []
    for group in groups:
        distributions = group["fly_distributions"]
        walking = distributions["walking_delta_pp"]
        changed = distributions["changed_fraction"]
        spatial = distributions["shape_tv"]
        rows.append([
            group["panel"], role_name(group["role"]),
            f"{number(walking.get('median'), signed=True)} "
            f"[{number(walking.get('p05'), signed=True)}, {number(walking.get('p95'), signed=True)}]",
            number(walking.get("min"), signed=True) + ", " + number(walking.get("max"), signed=True),
            number(changed.get("median"), scale=100) + " / " + number(changed.get("max"), scale=100),
            number(spatial.get("median"), scale=100) + " / " + number(spatial.get("max"), scale=100),
        ])
    lines += table(
        ["Panel", "Role", "Walking Δ pp median [p05, p95]", "Walking Δ pp min, max",
         "Changed fraction % median / max", "Per-fly shape TV % median / max"], rows,
    )
    return lines


def region_table(summary):
    lines = ["", "## Spatial regions", "",
        "Regions use histogram-bin centers, not the boundary-contact algorithm. Floor interior, "
        "the inward 2 mm edge band, and outside-floor bins partition the map. Reward regions "
        "overlap that partition; pre-training uses the upcoming T1 target. Physical labels assume "
        "template-based calibration. Absolute-change shares sum absolute per-fly changes before "
        "averaging so opposite directions across recordings do not cancel. Undefined ratios are n/a.", ""]
    rows = [
        [region["panel"], role_name(region["role"]), region["region"],
         number(region["baseline_mass_share"], scale=100, digits=2),
         number(region["absolute_change_share"], scale=100, digits=2),
         number(region["relative_mass_delta"], scale=100, signed=True)]
        for region in summary
    ]
    return lines + table(
        ["Panel", "Role", "Region", "Baseline mass share %", "Absolute-change share %", "Region mass change %"], rows,
    )


def individual_table(flies):
    selected = sorted(
        (row for row in flies if row["status"] == "ok"),
        key=lambda row: float(row["changed_fraction"]), reverse=True,
    )[:15]
    lines = ["", "## Largest individual walking changes", "",
        "Sparse maps can have large relative changes from only a few positions. Counts and "
        "original missingness are included here; interpolation is preserved, not excluded.", ""]
    rows = []
    for row in selected:
        rows.append([
            row["panel"], Path(row["video"]).name, row["absolute_fly"], role_name(row["role"]),
            number(float(row["factor"]), digits=6),
            number(float(row["changed_fraction"]), scale=100),
            number(float(row["walking_delta_pp"]), signed=True),
            row["histogram_counts_current"] + "→" + row["histogram_counts_scaled"],
            row["original_missing"], row.get("changed_step_touches_original_missing", "n/a"),
        ])
    return lines + table(
        ["Panel", "Video", "Absolute fly", "Role", "Factor", "Changed %", "Walking Δ pp",
         "Histogram positions current→scaled", "Originally missing frames", "Changed steps touching missing"], rows,
    )


def report_lines(out, provenance, groups, flies, regional, skipped):
    """Generate factual text only; no dataset-specific interpretation or recommendation."""
    lines = scope_and_validation(provenance, groups, skipped)
    lines += group_tables(groups)
    lines += region_table(regional)
    lines += individual_table(flies)
    output_arg = shlex.quote(str(out.relative_to(ROOT)))
    lines += ["", "## Conventions and limitations", "",
        "- This is a threshold sensitivity analysis; absolute floor calibration remains conditional on template calibration.",
        "- Production interpolation and NaN-distance replacement with zero are preserved. Finite speed alone is not valid tracking.",
        "- Exact windows, selected recipes, historical cache checks and input fingerprints are retained in the diagnostic files.",
        "- Summed window counts and equal-fly mean maps use different weighting; do not interchange them.",
        "- Relative changes in nearly empty bins can be large despite negligible probability mass.",
        "- No behavioral inference or formal scientific equivalence test follows from a small numerical or visual difference.",
        "", "## Reproduction", "", "Use the repository analysis environment; see `docs/walking_scale_audit.md`.",
        "", "```bash",
        f"python scripts/audit_walking_scale.py --output {output_arg}",
        f"python scripts/summarize_walking_scale_audit.py {output_arg}",
        "```", "",
        "For a selected-recipe run, repeat the recipe selection recorded in `provenance.json`.", "",
        "Files: `flies.csv`, `window_totals.csv`, `groups.json`, `regions.csv`, `regional_summary.json`, "
        "`scales.json`, `skipped.csv` when nonempty, `provenance.json`, and `maps/*.npz`. Sidecars are "
        "hashed; videos retain size/mtime rather than full-content hashes.", "",
    ]
    return lines


def render_diagnostics(out, groups):
    """Render new diagnostic charts only, with explicitly independent difference scales."""
    os.environ["MPLCONFIGDIR"] = str(out / "work/matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figure, axis = plt.subplots(figsize=(10, max(4, len(groups) * .35)))
    for index, group in enumerate(groups):
        values = group["fly_distributions"]["walking_delta_pp"]
        color = "C0" if group["role"] == 0 else "C1"
        axis.plot([values["p05"], values["p95"]], [index, index], color=color)
        axis.plot(values["median"], index, "o", color=color)
    axis.set(
        yticks=range(len(groups)),
        yticklabels=[group["panel"] + " " + role_name(group["role"]) for group in groups],
        xlabel="Walking fraction change (percentage points); median and 5th–95th percentiles",
    )
    axis.axvline(0, color="gray", linewidth=.7)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(out / "walking_change_distributions.png", dpi=160)
    plt.close(figure)

    with PdfPages(out / "diagnostic_difference_maps.pdf") as pdf:
        for panel in dict.fromkeys(group["panel"] for group in groups):
            selected = [group for group in groups if group["panel"] == panel]
            figure, axes = plt.subplots(len(selected), 2, figsize=(9, 4 * len(selected)), squeeze=False)
            for row, group in enumerate(selected):
                with np.load(out / "maps" / f"{panel}_role{group['role']}.npz") as archive:
                    current = archive["group_current"]
                    scaled = archive["group_scaled"]
                differences = [
                    (scaled - current) * 10000,
                    (scaled / scaled.sum() - current / current.sum()) * 10000,
                ]
                for column, difference in enumerate(differences):
                    limit = max(float(np.max(np.abs(difference))), 1e-12)
                    axis = axes[row, column]
                    image = axis.imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
                    kind = "occupancy Δ" if column == 0 else "unit-mass spatial Δ"
                    axis.set_title(role_name(group["role"]) + ": " + kind)
                    axis.set(xticks=[], yticks=[])
                    figure.colorbar(image, ax=axis, label="Probability change × 10,000", shrink=.8)
            figure.suptitle(panel + " — sensitivity diagnostic; each map has its own symmetric scale")
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def summarize(directory, *, plots=True):
    out = Path(directory).resolve()
    allowed = ROOT / "exports/diagnostics"
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Expected a dedicated subdirectory of exports/diagnostics")
    provenance = json.loads((out / "provenance.json").read_text())
    groups = json.loads((out / "groups.json").read_text())
    flies = read_csv(out / "flies.csv")
    regions = read_csv(out / "regions.csv")
    skipped = read_csv(out / "skipped.csv")
    validate_report_inputs(provenance, groups)

    totals = window_totals(groups, flies)
    regional = regional_summary(groups, regions)
    with (out / "window_totals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(totals[0]))
        writer.writeheader()
        writer.writerows(totals)
    (out / "regional_summary.json").write_text(json.dumps(regional, indent=2) + "\n")
    lines = report_lines(out, provenance, groups, flies, regional, skipped)
    (out / "REPORT.md").write_text("\n".join(lines))
    if plots:
        render_diagnostics(out, groups)
    print(out / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--no-plots", action="store_true", help="Write tables and text without diagnostic charts")
    arguments = parser.parse_args()
    summarize(arguments.directory, plots=not arguments.no_plots)
