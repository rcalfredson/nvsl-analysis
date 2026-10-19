#!/usr/bin/env python3
"""Survey tracking loss for three flat-large or flat-HTL Kir cohorts.

Reads notebook constants with ast.literal_eval (never executes notebook cells),
then reads local .data/.trx files without running the full analysis or decoding AVI.
Run with the project's analysis Python environment.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.agarose_preinvestigation import load_pickle, true_regions

COHORTS = {
    "regular": "FIGURE_4B_CTRL_KIR_VIDEO_GLOBS",
    "pfn_silenced": "FIGURE_4I_PFND_KIR_VIDEO_GLOBS",
    "antennae_removed": "FIGURE_4D_ANTENNAE_REMOVED_VIDEO_GLOBS",
}
HTL_COHORTS = {
    "regular": "FLAT_HTL_CTRL_KIR_VIDEO_GLOBS",
    "pfn_silenced": "FLAT_HTL_PFND_KIR_VIDEO_GLOBS",
    "antennae_removed": "FIGURE_2I_CTRL_KIR_ANTENNAE_REMOVED_VIDEO_GLOBS",
}


def notebook_lists(path, cohorts=None):
    cohorts = COHORTS if cohorts is None else cohorts
    values = {}
    for cell in json.loads(Path(path).read_text())["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in cohorts.values():
                        values[target.id] = ast.literal_eval(node.value)
    return {group: values[name] for group, name in cohorts.items()}


def loss_metrics(x, cutoff_pct=60):
    """Match Trajectory._isEmpty/_interpolate, including trailing missing data."""
    missing = np.isnan(np.asarray(x, dtype=float))
    if not missing.size:
        raise ValueError("zero-length trajectory")
    regions = true_regions(missing)
    initial = regions.pop(0).stop if regions and regions[0].start == 0 else 0
    lengths = [r.stop - r.start for r in regions]
    denominator = len(missing) - initial
    pct = 100 * sum(lengths) / denominator if denominator else float("nan")
    return dict(
        total_frames=len(missing), initial_missing_frames=initial,
        evaluated_frames=denominator, lost_frames=sum(lengths), loss_pct=pct,
        missing_pct_all_frames=100 * np.count_nonzero(missing) / len(missing),
        loss_sequences=len(lengths),
        mean_sequence_frames=float(np.mean(lengths)) if lengths else 0.0,
        max_sequence_frames=max(lengths, default=0),
        no_trajectory=bool(np.count_nonzero(missing) > 0.99 * len(missing)),
        exceeds_60_pct=bool(pct > 60),
        cutoff_pct=cutoff_pct,
        exceeds_cutoff=bool(pct > cutoff_pct),
    )


def survey(lists, chamber="large2"):
    expected_nef, cutoff = (10, 10) if chamber == "htl" else (2, 60)
    rows, manifest, unmatched = [], [], []
    for cohort, patterns in lists.items():
        videos = set()
        for pattern in patterns.split(","):
            matches = {Path(p) for p in glob.glob(pattern) if p.endswith(".avi")}
            if not matches:
                unmatched.append((cohort, pattern))
            videos.update(matches)
        for video in sorted(videos):
            data = load_pickle(video.with_suffix(".data"))
            trx = load_pickle(video.with_suffix(".trx"))
            proto = data["protocol"]
            fns = proto["frameNums"]
            nef = sum(bool(d) for d in fns) if isinstance(fns, list) else 1
            if len(trx["x"]) != 2 * expected_nef or nef != expected_nef:
                raise ValueError(f"Unexpected chamber/trajectory layout: {video}")
            manifest.append(dict(cohort=cohort, video=str(video)))
            for fly in range(expected_nef):
                mapping = proto.get("ef2yc", {})
                yc = mapping[fly] if mapping else fly + nef
                for role, index in (("exp", fly), ("yok", yc)):
                    rows.append(dict(cohort=cohort, video=str(video), fly=fly,
                                     role=role, trajectory_index=index,
                                     **loss_metrics(trx["x"][index], cutoff)))
    if unmatched:
        raise ValueError(f"Video patterns with no AVI matches: {unmatched}")
    return rows, manifest


def parse_loss_log(path):
    """Return logged percentages keyed by video basename, pair index and role."""
    records = {}
    video = fly = role = None
    for line in Path(path).read_text().splitlines():
        block = re.match(r"^=== analyzing (.*), fly (\d+) ===$", line)
        if block:
            video, fly, role = Path(block[1]).name, int(block[2]), None
        elif line in ("exp fly", "yok fly"):
            role = line.split()[0]
        elif video is not None and role is not None:
            loss = re.match(r"\s+lost: number frames: (\d+) \(([\d.]+)%\)", line)
            if loss:
                key = (video, fly, role)
                value = (int(loss[1]), float(loss[2]))
                if key in records and records[key] != value:
                    raise ValueError(f"Conflicting log entries: {key}")
                records[key] = value
    return records


def validate_log(rows, path):
    records = parse_loss_log(path)
    checked = 0
    for row in rows:
        key = (Path(row["video"]).name, row["fly"], row["role"])
        if key not in records:
            continue
        frames, pct = records[key]
        if frames != row["lost_frames"] or abs(pct - row["loss_pct"]) > 0.00501:
            raise ValueError(f"Log disagrees with source tracking: {key}")
        checked += 1
    if not checked:
        raise ValueError(f"No survey trajectories matched log {path}")
    return checked


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows):
    result = []
    for cohort in COHORTS:
        for role in ("exp", "yok", "all"):
            subset = [r for r in rows if r["cohort"] == cohort
                      and (role == "all" or r["role"] == role)]
            vals = np.array([r["loss_pct"] for r in subset if not r["no_trajectory"]])
            if not vals.size:
                raise ValueError(f"No measurable trajectories for {cohort}/{role}")
            result.append(dict(
                cohort=cohort, role=role, n=len(subset), measured_n=len(vals),
                no_trajectory=sum(r["no_trajectory"] for r in subset),
                mean_pct=float(np.mean(vals)), median_pct=float(np.median(vals)),
                p90_pct=float(np.quantile(vals, .9)), p95_pct=float(np.quantile(vals, .95)),
                max_pct=float(np.max(vals)),
                **{f"above_{t}_pct": int(np.count_nonzero(vals > t)) for t in (10, 30, 50, 60)},
            ))
    return result


def write_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, role in zip(axes, ("exp", "yok")):
        for cohort in COHORTS:
            vals = np.sort([r["loss_pct"] for r in rows if r["cohort"] == cohort
                            and r["role"] == role and not r["no_trajectory"]])
            ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post",
                    label=cohort.replace("_", " "))
        for threshold in (10, 60):
            ax.axvline(threshold, color="gray", linestyle="--", linewidth=.8)
        ax.set(title={"exp": "Experimental flies", "yok": "Yoked flies"}[role],
               xlabel="Tracking loss (%)", xlim=(0, 100), ylim=(0, 1.02))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative fraction of trajectories")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_html(path, title, summary, rows, manifest, cutoff):
    columns = [
        ("cohort", "Cohort"), ("role", "Role"), ("measured_n", "Measurable"),
        ("n", "Total"), ("no_trajectory", "No trajectory"),
        ("median_pct", "Median %"), ("p95_pct", "P95 %"), ("max_pct", "Max %"),
        *[(f"above_{t}_pct", f">{t}%") for t in (10, 30, 50, 60)],
    ]
    parts = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{html.escape(title)}</title>", "<style>",
             "body {font-family:system-ui,sans-serif;margin:2rem;color:#222}",
             "table {border-collapse:collapse;font-variant-numeric:tabular-nums}",
             "th,td {border:1px solid #ccc;padding:.6rem .8rem;text-align:right;white-space:nowrap}",
             "th {background:#f0f3f6} td:nth-child(-n+2) {text-align:left}",
             ".combined {background:#f7f8fa;font-weight:600} p {max-width:85ch;line-height:1.5}",
             "</style></head><body>", f"<h1>{html.escape(title)}</h1>",
             '<div style="overflow-x:auto"><table><thead><tr>',
             *[f'<th scope="col">{html.escape(label)}</th>' for _, label in columns],
             "</tr></thead><tbody>"]
    for row in summary:
        parts.append('<tr class="combined">' if row["role"] == "all" else "<tr>")
        for key, _ in columns:
            value = f"{row[key]:.2f}" if key in ("median_pct", "p95_pct", "max_pct") else str(row[key])
            parts.append(f"<td>{html.escape(value)}</td>")
        parts.append("</tr>")
    parts += ["</tbody></table></div>",
              "<p>exp = experimental; yok = yoked; all = both roles. P95 is the 95th percentile of tracking-loss percentages. Threshold columns count measurable trajectories with strictly greater loss.</p>",
              "<p>Full recordings are measured. Initial missing frames are excluded from loss percentages. Trajectories with more than 99% missing positions are classified as no trajectory and excluded from percentage summaries and threshold counts.</p>",
              f"<p>Coverage: {len(manifest)} videos; {len(rows)} trajectories; {sum(r['no_trajectory'] for r in rows)} no-trajectory cases. The chamber cutoff is {cutoff}%. Other analyzer exclusions are not applied.</p>",
              '<p><a href="report.md">Full report and source lists</a> · <a href="trajectories.csv">Per-trajectory data</a> · <a href="loss_cdf.png">Distribution plot</a></p>',
              "</body></html>"]
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/manuscript_figure_panels.ipynb")
    parser.add_argument("--chamber", choices=("large2", "htl"), default="large2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    cohorts = HTL_COHORTS if args.chamber == "htl" else COHORTS
    if args.output_dir is None:
        args.output_dir = ROOT / "results" / ("frame_loss_survey_htl" if args.chamber == "htl" else "frame_loss_survey")
    lists = notebook_lists(args.notebook, cohorts)
    rows, manifest = survey(lists, args.chamber)
    checks = [(str(p), validate_log(rows, p)) for p in args.check_log]
    summary = summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectories.csv", rows)
    write_csv(args.output_dir / "videos.csv", manifest)
    write_csv(args.output_dir / "summary.csv", summary)
    write_plot(args.output_dir / "loss_cdf.png", rows)
    title = "Flat HTL tracking-loss survey" if args.chamber == "htl" else "Figure 4 tracking-loss survey"
    cutoff = 10 if args.chamber == "htl" else 60
    last_pair = 9 if args.chamber == "htl" else 1
    lines = [f"# {title}", "",
             f"Source: the three named video-list constants in the manuscript notebook; all matching AVI recordings, fly pairs 0–{last_pair}, both exp and yok trajectories.", "",
             *[f"- {group}: `{name}`" for group, name in cohorts.items()], "",
             f"Chamber: {args.chamber}. The analyzer's tracking-loss cutoff is {cutoff}%; rejection requires strictly greater loss. This survey includes trajectories above that cutoff.", "",
             "Loss matches Trajectory._interpolate: NaN x positions after the initial missing prefix, divided by frames after that prefix. Trailing missing frames count. This is tracking loss, not camera frame loss. Full recordings are measured, not only Training 2. Trajectories with >99% missing positions are flagged as no trajectory and excluded from percentage summaries.", "",
             "| Cohort | Role | Measurable / total | No trajectory | Median % | P95 % | Max % | >10% | >30% | >50% | >60% |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['cohort']} | {s['role']} | {s['measured_n']} / {s['n']} | {s['no_trajectory']} | {s['median_pct']:.2f} | {s['p95_pct']:.2f} | {s['max_pct']:.2f} | {s['above_10_pct']} | {s['above_30_pct']} | {s['above_50_pct']} | {s['above_60_pct']} |")
    lines += ["", f"Coverage: {len(manifest)} videos; {len(rows)} trajectories; all patterns matched.",
              f"No-trajectory cases: {sum(r['no_trajectory'] for r in rows)}.", "",
              "These counts precede other analyzer exclusions (suspicious jumps, manual exclusions, reward criteria, etc.). A pair can be unusable because one member fails; individual trajectory counts are not retained-pair sample sizes.", "",
              "## Highest-loss trajectories", "",
              "| Cohort | Video | Pair | Role | Loss % | Longest gap (frames) |",
              "|---|---|---:|---|---:|---:|"]
    for r in sorted((r for r in rows if not r["no_trajectory"]), key=lambda r: r["loss_pct"], reverse=True)[:15]:
        lines.append(f"| {r['cohort']} | {Path(r['video']).name} | {r['fly']} | {r['role']} | {r['loss_pct']:.2f} | {r['max_sequence_frames']} |")
    lines += ["", "## Validation", ""]
    lines += [f"- {count} trajectory counts and rounded percentages agree with `{path}`." for path, count in checks]
    if not checks:
        lines.append("No saved-log comparison requested. Automated tests compare the survey calculation with Trajectory._interpolate for both chamber types.")
    lines += ["", "## Reproduce", "", "```bash",
              f"/home/tracking/miniconda3/envs/analysis3.13/bin/python scripts/survey_frame_loss.py --chamber {args.chamber}", "```", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    write_html(args.output_dir / "summary.html", title, summary, rows, manifest, cutoff)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
