#!/usr/bin/env python3
"""Post hoc paired tracking-loss sensitivity of archived Figure 4 SLI results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.survey_frame_loss import COHORTS, notebook_lists, survey, write_csv

CUTOFFS = (60, 50, 40, 30, 20, 10)
GROUPS = {
    "regular": ("Normal antennae | Ctrl>Kir", "normal_ctrl_kir", "Control"),
    "pfn_silenced": ("Normal antennae | PFNd>Kir", "normal_pfnd_kir", "PFNd>Kir"),
    "antennae_removed": ("Antennae removed | Control", "antennae_removed_control", "Antennae removed"),
}
ENDPOINTS = {
    "final_sli": ("fig4i_t2_sb5_final_sli_audit.csv", "exp_minus_yoked", "sli_t2_sb5"),
    "mean_sli": ("fig4i_t2_sb2_sb5_mean_sli_audit.csv", "sli_t2_sb2_sb5_mean", "sli_t2_sb2_sb5_mean"),
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return float(value) if value not in ("", None) else math.nan


def equal(a, b):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)


def unique_index(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f"Duplicate identity: {k}")
        result[k] = row
    return result


def load_subjects(root, losses):
    """Validate source/archive/audit/GraphPad consistency before joining outcomes."""
    folder = root / "exports/manuscript"
    labels = {v[0]: k for k, v in GROUPS.items()}
    loss_index = unique_index(losses, lambda r: (r["cohort"], r["video"], int(r["fly"]), r["role"]))
    archives, provenance = {}, []
    for cohort, (label, slug, _) in GROUPS.items():
        source = folder / f"fig4i_{slug}_sli_source.csv"
        archive = folder / f"fig4i_{slug}_sli_archive.csv"
        prov_path = folder / f"fig4i_{slug}_sli_archive.provenance.json"
        prov = json.loads(prov_path.read_text())
        if sha256(source) != prov["source_sha256"] or sha256(archive) != prov["source_sha256"]:
            raise ValueError(f"Archive/source provenance mismatch: {slug}")
        if sha256(root / prov["learning_stats"]) != prov["learning_stats_sha256"]:
            raise ValueError(f"Learning-statistics provenance mismatch: {slug}")
        archives[cohort] = unique_index(read_csv(archive), lambda r: (r["closed_video"], int(r["exp_fly_id"])))
        provenance.append(dict(cohort=cohort, **prov))
    endpoint_rows = {}
    for endpoint, (filename, value_field, archive_field) in ENDPOINTS.items():
        audits = [r for r in read_csv(folder / filename) if r["group"] in labels]
        index = unique_index(audits, lambda r: (labels[r["group"]], r.get("video", r.get("closed_video")),
                                               int(r.get("fly", r.get("exp_fly_id")))))
        expected = {(c, v, f) for c, archive in archives.items() for v, f in archive}
        if set(index) != expected:
            raise ValueError(f"Archive/audit identity mismatch: {endpoint}")
        graphpad = read_csv(folder / filename.replace("_audit.csv", "_graphpad.csv"))
        for cohort, (label, _, _) in GROUPS.items():
            expected_values = [number(r[value_field]) for r in audits if r["group"] == label and r["included"] == "True"]
            actual = [number(r[label]) for r in graphpad if r[label].strip()]
            if len(actual) != len(expected_values) or not all(equal(a, b) for a, b in zip(actual, expected_values)):
                raise ValueError(f"GraphPad baseline mismatch: {endpoint}/{cohort}")
        for (cohort, video, fly), row in index.items():
            archive = archives[cohort][video, fly]
            value = number(row[value_field])
            final = endpoint == "final_sli"
            exp = number(archive["exp_reward_pi_t2_sb5" if final else "exp_reward_pi_t2_sb2_sb5_mean"])
            yok = number(archive["yoked_reward_pi_t2_sb5" if final else "yoked_reward_pi_t2_sb2_sb5_mean"])
            included = math.isfinite(exp) and math.isfinite(yok)
            if not final:
                if int(archive["sli_min_valid_sync_buckets"]) != 3:
                    raise ValueError("Mean SLI archive does not specify 3 valid buckets")
                included &= int(archive["sli_t2_sb2_sb5_valid_bucket_count"]) >= 3
            if included != (row["included"] == "True") or not equal(value, number(archive[archive_field])):
                raise ValueError(f"Archive/audit eligibility or value mismatch: {cohort}/{video}/{fly}")
            if included and not equal(value, exp - yok):
                raise ValueError("SLI component difference does not match export")
            for role, id_field in (("exp", "exp_fly_id"), ("yok", "yoked_fly_id")):
                loss = loss_index[cohort, video, fly, role]
                if int(loss["trajectory_index"]) != int(archive[id_field]):
                    raise ValueError("Physical trajectory identity mismatch")
            if int(row["yoked_fly_id"]) != int(archive["yoked_fly_id"]):
                raise ValueError("Audit yoked identity mismatch")
        endpoint_rows[endpoint] = index
    subjects = []
    for cohort, video, fly, role in loss_index:
        if role != "exp":
            continue
        exp, yok = (loss_index[cohort, video, fly, r] for r in ("exp", "yok"))
        for endpoint, index in endpoint_rows.items():
            row = index.get((cohort, video, fly))
            value_field = ENDPOINTS[endpoint][1]
            subjects.append(dict(
                cohort=cohort, video=video, fly=fly, yoked_fly_id=yok["trajectory_index"], endpoint=endpoint,
                exp_loss_pct=exp["loss_pct"], yok_loss_pct=yok["loss_pct"],
                exp_no_trajectory=exp["no_trajectory"], yok_no_trajectory=yok["no_trajectory"],
                baseline_included=bool(row and row["included"] == "True"),
                baseline_exclusion_reason=(row["exclusion_reason"] if row else "absent from analyzer export"),
                value=number(row[value_field]) if row else math.nan,
            ))
    return subjects, provenance


def passes(subject, cutoff):
    return (not subject["exp_no_trajectory"] and not subject["yok_no_trajectory"]
            and subject["exp_loss_pct"] <= cutoff and subject["yok_loss_pct"] <= cutoff)


def inclusion_audit(subjects, cutoffs=CUTOFFS):
    result = []
    for subject in subjects:
        for cutoff in cutoffs:
            exp_fail = subject["exp_no_trajectory"] or not subject["exp_loss_pct"] <= cutoff
            yok_fail = subject["yok_no_trajectory"] or not subject["yok_loss_pct"] <= cutoff
            included = subject["baseline_included"] and not exp_fail and not yok_fail
            if not subject["baseline_included"]:
                reason = "baseline: " + subject["baseline_exclusion_reason"]
            elif exp_fail and yok_fail:
                reason = "both exceed cutoff"
            elif exp_fail:
                reason = "experimental exceeds cutoff"
            elif yok_fail:
                reason = "yoked exceeds cutoff"
            else:
                reason = ""
            result.append(dict(**subject, cutoff_pct=cutoff, included=included, exclusion_reason=reason))
    return result


def cluster_bootstrap(values, cluster_ids, counts):
    """Counts resample recording clusters; ratio preserves a fly-weighted mean."""
    totals = np.bincount(cluster_ids, weights=values, minlength=counts.shape[1])
    sizes = np.bincount(cluster_ids, minlength=counts.shape[1])
    numerator, denominator = counts @ totals, counts @ sizes
    return np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)


def ci(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return math.nan, math.nan, 0
    low, high = np.quantile(finite, [.025, .975])
    return float(low), float(high), len(finite)


def summarize(subjects, draws=10000, seed=20261018, cutoffs=CUTOFFS):
    rng = np.random.default_rng(seed)
    summaries, boots = [], {}
    for cohort in GROUPS:
        videos = sorted({s["video"] for s in subjects if s["cohort"] == cohort})
        positions = {v: i for i, v in enumerate(videos)}
        # One set of cluster draws per cohort, shared by both endpoints and all cutoffs.
        counts = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=draws)
        for endpoint in ENDPOINTS:
            baseline = [s for s in subjects if s["cohort"] == cohort and s["endpoint"] == endpoint and s["baseline_included"]]
            if any(not passes(s, 60) for s in baseline):
                raise ValueError("60% strict filter differs from archived baseline; inspect overrides")
            baseline_mean = float(np.mean([s["value"] for s in baseline]))
            baseboot = cluster_bootstrap(np.array([s["value"] for s in baseline]),
                                        np.array([positions[s["video"]] for s in baseline], dtype=int), counts)
            for cutoff in cutoffs:
                retained = [s for s in baseline if passes(s, cutoff)]
                vals = np.array([s["value"] for s in retained])
                bootstrap = cluster_bootstrap(vals, np.array([positions[s["video"]] for s in retained], dtype=int), counts)
                boots[cohort, endpoint, cutoff] = bootstrap
                low, high, valid = ci(bootstrap)
                shift_low, shift_high, _ = ci(bootstrap - baseboot)
                mean = float(np.mean(vals)) if len(vals) else math.nan
                exp_only = sum(not passes(s, cutoff) and s["exp_loss_pct"] > cutoff and s["yok_loss_pct"] <= cutoff for s in baseline)
                yok_only = sum(not passes(s, cutoff) and s["yok_loss_pct"] > cutoff and s["exp_loss_pct"] <= cutoff for s in baseline)
                both = sum(s["exp_loss_pct"] > cutoff and s["yok_loss_pct"] > cutoff for s in baseline)
                summaries.append(dict(cohort=cohort, endpoint=endpoint, cutoff_pct=cutoff,
                                      baseline_n=len(baseline), n=len(vals), retention_fraction=len(vals)/len(baseline),
                                      n_videos=len({s["video"] for s in retained}), source_videos=len(videos),
                                      excluded_exp_only=exp_only, excluded_yok_only=yok_only, excluded_both=both,
                                      mean=mean, sd=float(np.std(vals, ddof=1)) if len(vals)>1 else math.nan,
                                      median=float(np.median(vals)) if len(vals) else math.nan,
                                      ci_low=low, ci_high=high, bootstrap_valid_draws=valid,
                                      mean_shift_from_60=mean-baseline_mean,
                                      shift_ci_low=shift_low, shift_ci_high=shift_high))
    lookup = {(s["cohort"], s["endpoint"], s["cutoff_pct"]): s for s in summaries}
    contrasts = []
    for cohort in ("pfn_silenced", "antennae_removed"):
        for endpoint in ENDPOINTS:
            base_diff = lookup[cohort, endpoint, 60]["mean"] - lookup["regular", endpoint, 60]["mean"]
            baseboot = boots[cohort, endpoint, 60] - boots["regular", endpoint, 60]
            for cutoff in cutoffs:
                group, control = lookup[cohort, endpoint, cutoff], lookup["regular", endpoint, cutoff]
                diff = group["mean"] - control["mean"]
                bootstrap = boots[cohort, endpoint, cutoff] - boots["regular", endpoint, cutoff]
                low, high, valid = ci(bootstrap)
                shift_low, shift_high, _ = ci(bootstrap - baseboot)
                contrasts.append(dict(comparison=cohort+"_minus_regular", endpoint=endpoint, cutoff_pct=cutoff,
                                      n_group=group["n"], n_control=control["n"], difference=diff,
                                      ci_low=low, ci_high=high, bootstrap_valid_draws=valid,
                                      difference_shift_from_60=diff-base_diff,
                                      shift_ci_low=shift_low, shift_ci_high=shift_high))
    return summaries, contrasts


def plots(out, summaries, contrasts, subjects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(GROUPS, ("#2878b5", "#dc7828", "#389445")))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, endpoint in enumerate(ENDPOINTS):
        for cohort, (_, _, label) in GROUPS.items():
            rs = sorted([s for s in summaries if s["cohort"] == cohort and s["endpoint"] == endpoint], key=lambda s:s["cutoff_pct"])
            x = [s["cutoff_pct"] for s in rs]
            axes[i, 0].plot(x, [s["n"] for s in rs], "o-", label=label, color=colors[cohort])
            axes[i, 1].plot(x, [s["mean"] for s in rs], "o-", label=label, color=colors[cohort])
            axes[i, 1].fill_between(x, [s["ci_low"] for s in rs], [s["ci_high"] for s in rs], alpha=.12, color=colors[cohort])
            if cohort != "regular":
                cr = sorted([s for s in contrasts if s["comparison"] == cohort+"_minus_regular" and s["endpoint"] == endpoint], key=lambda s:s["cutoff_pct"])
                axes[i, 2].plot(x, [s["difference"] for s in cr], "o-", label=label+" − Control", color=colors[cohort])
                axes[i, 2].fill_between(x, [s["ci_low"] for s in cr], [s["ci_high"] for s in cr], alpha=.12, color=colors[cohort])
        for j, label in enumerate(("Retained flies", "Group mean SLI", "Difference in mean SLI")):
            axes[i, j].set(title=("Final SLI (T2 SB5)" if i==0 else "Mean SLI (T2 SB2–5)")+"\n"+label,
                           xlabel="Maximum allowed tracking loss (%)", xticks=sorted(CUTOFFS))
            axes[i, j].invert_xaxis()
            axes[i, j].grid(alpha=.2)
        axes[i, 2].axhline(0, color="gray", lw=.8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Paired tracking-loss sensitivity • shading: 95% video-cluster bootstrap intervals")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sli_sensitivity.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for i, endpoint in enumerate(ENDPOINTS):
        for j, role in enumerate(("exp", "yok")):
            for cohort, (_, _, label) in GROUPS.items():
                rs = [s for s in subjects if s["baseline_included"] and s["cohort"]==cohort and s["endpoint"]==endpoint]
                axes[i,j].scatter([s[role+"_loss_pct"] for s in rs], [s["value"] for s in rs],
                                  color=colors[cohort], label=label, alpha=.65, s=18)
            axes[i,j].set(xlabel=("Experimental" if role=="exp" else "Yoked")+" tracking loss (%)",
                          ylabel="Final SLI" if i==0 else "Mean SLI")
            axes[i,j].axvline(10, color="gray", linestyle="--", lw=.8)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sli_vs_tracking_loss.png", dpi=160)
    plt.close(fig)


def fmt(value):
    return f"{value:.3f}" if math.isfinite(value) else "—"


def report(out, summaries, contrasts, validation, draws, seed):
    lookup = {(s["cohort"], s["endpoint"], s["cutoff_pct"]):s for s in summaries}
    sections = []
    text = ["# Large-chamber SLI tracking-loss sensitivity", "",
            "Post hoc paired loss filter applied to existing Figure 4 exports, for the three previously surveyed flat-large cohorts. Individual retained SLI values are unchanged. Analyzer cutoff defaults and manuscript exports were not modified.", "",
            "Final SLI: T2 SB5 experimental minus yoked reward PI, both finite. Mean SLI: mean paired finite bucket differences in T2 SB2–5, with at least 3 valid buckets. Each endpoint retains its own eligibility; mean SLI is not additionally conditioned on final SLI.", "",
            "Both members must pass the cutoff (equality passes). Empty trajectories and pre-existing endpoint exclusions remain excluded. The 60% baseline reproduces existing GraphPad values and inclusion IDs for these groups.", ""]
    for endpoint, heading in (("final_sli", "Final SLI (T2 SB5)"), ("mean_sli", "Mean SLI (T2 SB2–5)")):
        headers = ["Cutoff", "Control: n; mean", "PFNd>Kir: n; mean", "Antennae removed: n; mean"]
        table = []
        text += [f"## {heading}", "", "| " + " | ".join(headers) + " |", "|---|---|---|---|"]
        for cutoff in CUTOFFS:
            cells = [f"{cutoff}%"]
            for cohort in GROUPS:
                s = lookup[cohort, endpoint, cutoff]
                cells.append(f"{s['n']}; {fmt(s['mean'])}")
            table.append(cells)
            text.append("| " + " | ".join(cells) + " |")
        text += [""]
        sections.append((heading, headers, table))
    contrast_lookup = {(s["comparison"], s["endpoint"], s["cutoff_pct"]):s for s in contrasts}
    for endpoint, label in (("final_sli", "Final SLI"), ("mean_sli", "Mean SLI")):
        heading = label + " group differences (95% confidence intervals)"
        headers = ["Cutoff", "PFNd>Kir minus Control", "Antennae removed minus Control"]
        table = []
        text += [f"## {heading}", "", "| " + " | ".join(headers) + " |", "|---|---|---|"]
        for cutoff in CUTOFFS:
            cells = [f"{cutoff}%"]
            for cohort in ("pfn_silenced", "antennae_removed"):
                s = contrast_lookup[cohort+"_minus_regular", endpoint, cutoff]
                cells.append(f"{fmt(s['difference'])} [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]"
                             if math.isfinite(s["difference"]) else "—")
            table.append(cells)
            text.append("| " + " | ".join(cells) + " |")
        text += [""]
        sections.append((heading, headers, table))
    strict_counts = "; ".join(f"{e.replace('_', ' ')}: {lookup['antennae_removed', e, 10]['n']}" for e in ENDPOINTS)
    intermediate_counts = "; ".join(f"{e.replace('_', ' ')}: {lookup['antennae_removed', e, 20]['n']}" for e in ENDPOINTS)
    direction = "negative" if all(c["difference"] < 0 for c in contrasts if math.isfinite(c["difference"])) else "not uniformly negative"
    text += ["## Interpretation and limits", "",
             f"- Antennae-removed retention at 10%: {strict_counts}. With zero eligible flies, the mean and group contrast cannot be estimated.",
             f"- Antennae-removed retention at 20%: {intermediate_counts}. Compare this attrition with the other cohorts before interpreting mean shifts.",
             f"- The selected group-minus-control mean differences are {direction} wherever estimable. Their magnitude changes; this is not proof that the original effect size is insensitive to tracking quality.",
             "- The mean-SLI and final-SLI populations are distinct. Use final-SLI retention when anticipating the shared eligibility gate for Figures 4j–l; additional metric-specific exclusions still apply.",
             "- This measures sensitivity to excluding high-loss flies. It does not recompute rewards or establish the accuracy of interpolation. Equivalence to changing the analyzer cutoff has not yet been tested with direct analyzer runs.", "",
             "## Uncertainty", "",
             f"95% percentile intervals use {draws:,} bootstrap resamples of full source videos within each cohort (seed {seed}), keeping both pairs from each recording together. Means remain fly-weighted. The same cluster draws are reused across thresholds and endpoints; group draws are independent. Baseline shifts and contrast shifts use those paired draws. Empty resamples are omitted and their valid draw counts are exported. With zero retained flies, estimates and intervals are undefined. These are exploratory pointwise intervals, not simultaneous bands or a cutoff-selection test.", "",
             "## Validation", "", *[f"- {v}" for v in validation], "",
             "## Outputs", "",
             "- `summary.csv`: sample sizes, videos represented, means, SDs, medians, intervals, shifts, and attrition by experimental/yoked loss.",
             "- `contrasts.csv`: group-minus-control means, intervals, and shifts from 60%.",
             "- `inclusion_audit.csv`: every source pair, endpoint, and threshold, including baseline exclusions and absent exports.",
             "- `baseline_subjects.csv`, `tracking_loss.csv`, `videos.csv`: joined endpoint data and refreshed tracking measurements.",
             "- `sli_sensitivity.png/.pdf`, `sli_vs_tracking_loss.png`: threshold sweeps and individual loss/outcome relationships.",
             "- `provenance.json`: input hashes, tracking-file metadata, source-export provenance, parameters, and note-file snapshot.", "",
             "## Reproduce", "", "```bash",
             "MPLCONFIGDIR=/tmp/frame-loss-mpl /home/tracking/miniconda3/envs/analysis3.13/bin/python scripts/sweep_sli_frame_loss.py", "```", ""]
    (out / "report.md").write_text("\n".join(text))
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Large-chamber SLI loss sensitivity</title><style>',
             'body{font-family:system-ui,sans-serif;margin:2rem;color:#222;max-width:1200px}p,li{line-height:1.5}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:.6rem;border:1px solid #ccc;text-align:right}th{background:#eef2f6}img{max-width:100%}',
             '</style></head><body><h1>Large-chamber SLI tracking-loss sensitivity</h1>',
             '<p>Existing Figure 4 exports filtered by both experimental and yoked tracking loss. Values are unchanged for retained flies. Final and mean SLI retain their separate eligibility rules.</p>',
             f'<p><strong>Antennae-removed retention at 10%: {html.escape(strict_counts)}.</strong> A missing mean is undefined, not zero.</p>']
    for heading, headers, table in sections:
        parts += [f'<h2>{html.escape(heading)}</h2><table><thead><tr>',
                  *[f'<th scope="col">{html.escape(h)}</th>' for h in headers], '</tr></thead><tbody>']
        for row in table:
            parts.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in row)+'</tr>')
        parts.append('</tbody></table>')
    parts += ['<h2>Means, retention, and contrasts</h2><img src="sli_sensitivity.png" alt="Cutoff sensitivity plots">',
              '<h2>Individual outcomes versus tracking loss</h2><img src="sli_vs_tracking_loss.png" alt="SLI versus experimental and yoked tracking loss">',
              '<p>Shading shows pointwise 95% video-cluster bootstrap intervals. This is a sample-selection sensitivity analysis; direct analyzer reruns and episode-level interpolation checks remain separate work.</p>',
              '<p><a href="report.md">Full report and methods</a> · <a href="summary.csv">Full statistics and confidence intervals</a> · <a href="contrasts.csv">Group differences</a> · <a href="inclusion_audit.csv">Inclusion audit</a></p>',
              '</body></html>']
    (out / "report.html").write_text("\n".join(parts))
    print("\n".join(text[:text.index("## Interpretation and limits")]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sli_frame_loss_sweep")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("Use at least 100 bootstrap draws")
    notebook = ROOT / "notebooks/manuscript_figure_panels.ipynb"
    losses, manifest = survey(notebook_lists(notebook))
    # Ensure this run measures the same tracking inputs as the earlier survey.
    prior = read_csv(ROOT / "results/frame_loss_survey/trajectories.csv")
    key = lambda r: (r["cohort"], r["video"], int(r["fly"]), r["role"])
    index = unique_index(prior, key)
    if set(index) != {key(r) for r in losses}:
        raise ValueError("Current cohort coverage differs from earlier loss survey")
    for row in losses:
        old = index[key(row)]
        if not equal(float(old["loss_pct"]), row["loss_pct"]) or int(old["lost_frames"]) != row["lost_frames"]:
            raise ValueError("Source tracking differs from earlier survey")
    subjects, source_provenance = load_subjects(ROOT, losses)
    audit = inclusion_audit(subjects)
    summaries, contrasts = summarize(subjects, args.bootstrap_draws, args.seed)
    notes, tracking_files = [], []
    for row in manifest:
        video = Path(row["video"])
        note = video.with_name(video.stem + "__note.txt")
        notes.append(dict(video=str(video), note_exists=note.exists(), text=note.read_text() if note.exists() else ""))
        for suffix in (".data", ".trx"):
            path = video.with_suffix(suffix)
            stat = path.stat()
            tracking_files.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    validation = [
        f"Refreshed {len(losses)} trajectory measurements across {len(manifest)} videos; all identities, loss counts, and percentages match the earlier survey.",
        "All three archived source CSVs and learning-statistics files match their recorded SHA-256 provenance; source and archive hashes agree.",
        "Final and mean SLI audits exactly cover archive identities; physical experimental/yoked IDs agree with tracking. Eligibility and component differences match archived values, and included values match existing GraphPad columns.",
        "At 60%, all baseline-included subjects pass the strict paired filter, reproducing baseline inclusion IDs and means.",
        f"Manual note files present: {sum(n['note_exists'] for n in notes)} out of {len(notes)} source videos.",
    ]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("contrasts", contrasts), ("inclusion_audit", audit),
                       ("baseline_subjects", subjects), ("tracking_loss", losses), ("videos", manifest)):
        write_csv(out / f"{name}.csv", rows)
    inputs = [notebook, Path(__file__).resolve(), ROOT / "scripts/survey_frame_loss.py", ROOT / ".analyze.local.env"]
    inputs += [ROOT / "exports/manuscript" / filename for filename, _, _ in ENDPOINTS.values()]
    for _, slug, _ in GROUPS.values():
        inputs += list((ROOT / "exports/manuscript").glob(f"fig4i_{slug}_*"))
    for filename, _, _ in ENDPOINTS.values():
        inputs.append(ROOT / "exports/manuscript" / filename.replace("_audit.csv", "_graphpad.csv"))
    prov = dict(generated_utc=datetime.now(timezone.utc).isoformat(), cutoffs=list(CUTOFFS),
                bootstrap_draws=args.bootstrap_draws, seed=args.seed, cohorts=COHORTS,
                input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(inputs)) if p.is_file()},
                tracking_files=tracking_files, note_files=notes, source_export_provenance=source_provenance,
                validation=validation, analysis="post hoc paired loss filter; no direct analyzer cutoff reruns")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2)+"\n")
    plots(out, summaries, contrasts, subjects)
    report(out, summaries, contrasts, validation, args.bootstrap_draws, args.seed)


if __name__ == "__main__":
    main()
