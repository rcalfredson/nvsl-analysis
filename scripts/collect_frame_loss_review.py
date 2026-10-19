#!/usr/bin/env python3
"""Collect video/fly review identities from the large-chamber SLI sweep."""
import csv
import html
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/sli_frame_loss_sweep"
OUTPUT = ROOT / "results/frame_loss_review"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    subjects = {}
    for r in read(SOURCE / "baseline_subjects.csv"):
        key = (r["cohort"], r["video"], r["fly"], r["endpoint"])
        assert key not in subjects, key
        subjects[key] = r
    rows, empty = [], []
    for loss in read(SOURCE / "tracking_loss.csv"):
        key = (loss["cohort"], loss["video"], loss["fly"])
        final = subjects[key + ("final_sli",)]
        mean = subjects[key + ("mean_sli",)]
        assert Path(loss["video"]).is_file(), loss["video"]
        eligible = final["baseline_included"] == "True"
        row = dict(
            cohort=loss["cohort"], video_path=loss["video"],
            player_pair_id=int(loss["fly"]), role=loss["role"],
            physical_tracking_id=int(loss["trajectory_index"]),
            loss_pct=float(loss["loss_pct"]),
            experimental_loss_pct=float(final["exp_loss_pct"]),
            yoked_loss_pct=float(final["yok_loss_pct"]),
            max_gap_frames=int(loss["max_sequence_frames"]),
            final_sli_included=eligible, mean_sli_included=mean["baseline_included"] == "True",
            final_sli_exclusion_reason=final["baseline_exclusion_reason"],
            # Only supply launcher commands for baseline-eligible pairs.
            player_command=(shlex.join([
                "/home/tracking/miniconda3/envs/analysis3.13/bin/python", str(ROOT / "analyze.py"),
                "-v", loss["video"], "-f", loss["fly"], "--rCC", "15", "--num-trainings", "2", "--play"
            ]) if eligible else ""),
        )
        (empty if loss["no_trajectory"] == "True" else rows).append(row)
    rows.sort(key=lambda r: r["loss_pct"], reverse=True)
    eligible = [r for r in rows if r["final_sli_included"]]
    excluded = [r for r in rows if not r["final_sli_included"]]
    selections = {}

    def select(candidates, reason):
        for r in candidates:
            key = (r["video_path"], r["physical_tracking_id"])
            if key not in selections:
                selections[key] = dict(r, selection_reason=reason)
            elif reason not in selections[key]["selection_reason"]:
                selections[key]["selection_reason"] += "; " + reason

    select(eligible[:12], "top 12 final-SLI-eligible trajectories")
    for cohort in ("regular", "pfn_silenced", "antennae_removed"):
        select([r for r in eligible if r["cohort"] == cohort][:3], "top 3 eligible within cohort")
    select(excluded[:6], "top 6 final-SLI-excluded comparison trajectories")
    shortlist = sorted(selections.values(), key=lambda r: (not r["final_sli_included"], -r["loss_pct"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "review_shortlist.csv", shortlist)
    write(OUTPUT / "all_measurable_ranked.csv", rows)
    write(OUTPUT / "empty_trajectories.csv", empty)
    text = ["# Large-chamber tracking-loss video review", "",
            "Start with final-SLI-eligible trajectories: these pairs passed the shared SLI gate used by Figures 4j–l. Metric-specific criteria can still exclude them. The comparison list contains trajectories whose pairs did not contribute to final SLI.", "",
            "IDs are zero-based. `player_pair_id` is the value for `analyze.py -f`; the player loads that experimental/yoked pair. `physical_tracking_id` identifies the particular tracked fly to inspect. For these recordings, pair 0 loads exp 0 / yok 2; pair 1 loads exp 1 / yok 3. Do not pass a yoked physical ID of 2 or 3 as the pair ID.", "",
            "Loss percentages exclude the initial missing prefix, matching the analyzer. Essentially empty trajectories are listed separately and are not ranked. Player commands are provided for eligible pairs; excluded trajectories may have incomplete initialized annotations. Commands have not been GUI-tested. Run them from the repository directory.", ""]
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Tracking-loss video review</title><style>',
             'body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5}table{border-collapse:collapse}th,td{padding:.5rem;border:1px solid #ccc;text-align:left}th{background:#eef2f6}code{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow-x:auto}',
             '</style></head><body><h1>Large-chamber tracking-loss video review</h1>',
             *[f'<p>{html.escape(p)}</p>' for p in text[2:] if p]]
    for included, title in ((True, "Review first: final-SLI-eligible"), (False, "Comparison: final-SLI-excluded")):
        text += [f"## {title}", ""]
        parts.append(f'<h2>{title}</h2><div class="scroll"><table><thead><tr><th>Cohort</th><th>Loss %</th><th>Video path</th><th>Player pair ID</th><th>Role</th><th>Physical tracking ID</th></tr></thead><tbody>')
        for r in shortlist:
            if r["final_sli_included"] != included:
                continue
            text += [f"- **{r['cohort']}: {r['loss_pct']:.2f}% loss** — player pair **{r['player_pair_id']}**, **{r['role']}**, physical tracking ID **{r['physical_tracking_id']}**.",
                     f"  Video: `{r['video_path']}`", ""]
            if r["player_command"]:
                text += ["  ```bash", "  " + r["player_command"], "  ```", ""]
            else:
                text += [f"  Final-SLI exclusion: {r['final_sli_exclusion_reason']}.", ""]
            parts.append('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in (
                r['cohort'], f"{r['loss_pct']:.2f}", r['video_path'], r['player_pair_id'], r['role'], r['physical_tracking_id']))+'</tr>')
        parts.append('</tbody></table></div>')
    parts.append('<h2>Player commands for eligible pairs</h2>')
    for r in shortlist:
        if r["player_command"]:
            parts.append(f'<p>{html.escape(Path(r["video_path"]).name)} — pair {r["player_pair_id"]}, inspect {r["role"]} physical ID {r["physical_tracking_id"]}</p><pre><code>{html.escape(r["player_command"])}</code></pre>')
    parts += ['<p><a href="review_shortlist.csv">Shortlist CSV</a> · <a href="all_measurable_ranked.csv">All measurable trajectories, ranked</a> · <a href="empty_trajectories.csv">Empty trajectories</a></p>', '</body></html>']
    (OUTPUT / "review_list.md").write_text("\n".join(text)+"\n")
    (OUTPUT / "review_list.html").write_text("\n".join(parts)+"\n")
    print(f"Wrote {len(shortlist)} shortlisted trajectories ({sum(r['final_sli_included'] for r in shortlist)} eligible), {len(rows)} ranked measurable trajectories, {len(empty)} empty trajectories.")


if __name__ == "__main__":
    main()
