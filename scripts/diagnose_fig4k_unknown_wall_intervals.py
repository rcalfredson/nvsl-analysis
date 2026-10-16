#!/usr/bin/env python3
"""Opt-in Figure 4k audit; never calls production postAnalyze/export routines.

Run with the analysis Python environment from any directory. Inputs are resolved
against --repo; all new artifacts go to a new --out directory. Cached wall states
are never read or written. Only the cleanup method is wrapped, temporarily, to
measure fresh states before they are discarded; the original cleanup still runs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np

COHORTS = {"ctrl": "Ctrl", "pfnd_kir": "PFNd>Kir", "antennae_removed": "Antennae removed"}


def interval_observation_stats(trace, starts, ends):
    """Evaluate precisely [start,end), rejecting malformed or nonbinary traces."""
    trace = np.asarray(trace)
    starts, ends = np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)
    if trace.ndim != 1 or starts.shape != ends.shape:
        raise ValueError("invalid trace/boundary shapes")
    if np.any(starts < 0) or np.any(ends > trace.size) or np.any(ends <= starts):
        raise ValueError("empty, reversed, or out-of-bounds reward interval")
    finite = np.isfinite(trace)
    if np.any(finite & (trace != 0) & (trace != 1)):
        raise ValueError("finite wall states must be 0 or 1 (cache states forbidden)")
    prefix = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    observed = prefix[ends] - prefix[starts]
    length = ends - starts
    return observed, observed / length, observed == length


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return (row["unit_id"], int(row["training_index"]),
            int(row["start_reward_frame"]), int(row["end_reward_frame"]))


def summarize(flies, intervals):
    ids = {r["unit_id"] for r in flies}
    intervals = [r for r in intervals if r["unit_id"] in ids]
    paired = [r for r in flies if np.isfinite(r["strict_mean"])]
    delta = np.asarray([r["strict_mean"] - r["baseline_mean"] for r in paired])
    fractions = np.asarray([r["classifiable_fraction"] for r in intervals])
    n = len(intervals)
    excluded = sum(not r["strict_eligible"] for r in intervals)
    return {
        "n_flies": len(flies), "n_intervals": n, "excluded_intervals": excluded,
        "excluded_percent": 100 * excluded / n if n else None,
        "entirely_unknown_intervals": sum(r["classifiable_frames"] == 0 for r in intervals),
        "episodes_discarded": sum(r["wall_contact_event_count"] for r in intervals if not r["strict_eligible"]),
        "no_eligible_interval_flies": sum(r["n_eligible"] == 0 for r in flies),
        "below_five_eligible_flies": sum(r["n_eligible"] < 5 for r in flies),
        "paired_flies": len(paired), "changed_mean_flies": int(np.count_nonzero(np.abs(delta) > 1e-12)),
        "increased_mean_flies": int(np.count_nonzero(delta > 1e-12)),
        "decreased_mean_flies": int(np.count_nonzero(delta < -1e-12)),
        "mean_baseline": float(np.mean([r["baseline_mean"] for r in flies])) if flies else None,
        "mean_strict_paired": float(np.mean([r["strict_mean"] for r in paired])) if paired else None,
        "mean_paired_change": float(np.mean(delta)) if delta.size else None,
        "median_paired_change": float(np.median(delta)) if delta.size else None,
        "min_paired_change": float(np.min(delta)) if delta.size else None,
        "max_paired_change": float(np.max(delta)) if delta.size else None,
        "classifiable_fraction_quantiles_0_25_50_75_100": np.quantile(fractions, [0, .25, .5, .75, 1]).tolist() if n else [],
    }


def run_cohort(repo, out, slug, limit):
    label = COHORTS[slug]
    stem = repo / "exports/manuscript" / f"fig4k_{slug}_wall_contacts"
    provenance_path = Path(str(stem) + "_archive.provenance.json")
    provenance = json.loads(provenance_path.read_text())
    command = shlex.split(provenance["analysis_command"])[2:]
    # Importing analyze parses argv but does not execute its __main__ workflow.
    sys.argv = [str(repo / "analyze.py"), *command, "--no-wall-cache"]
    analysis = importlib.import_module("analyze")
    # Acquisition pickles refer to this legacy class in the entry-point module.
    # Expose the exact production class; do not rewrite any input pickles.
    sys.modules["__main__"].FlyDetector = analysis.FlyDetector
    opts = analysis.p.parse_args(sys.argv[1:])
    analysis._normalize_com_options(opts)
    analysis._normalize_learner_metric_table_options(opts)
    # These are the relevant __main__ normalizations for the archived commands.
    opts.cTurnAnlyz = False
    opts.rTurnAnlyz = False
    opts.wall = analysis.WALL_CONTACT_DEFAULT_THRESH_STR
    opts.wall_cache = False
    analysis.opts = opts

    from src.analysis import boundary_contact
    from src.analysis.sync_bucket_presence_filters import exp_target_sync_bucket_eligibility_mask
    from src.exporting.com_sli_bundle import _compute_sli_scalar_and_timeseries_from_rpid
    from src.exporting.graphpad_csv import _scalar_sli_eligibility, _scalar_subject_key
    from src.plotting.overlay_training_metric_scalar_bars import load_export_npz
    from src.plotting.wall_contacts_per_reward_interval_totals import (
        WallContactsPerRewardIntervalTotalsConfig, WallContactsPerRewardIntervalTotalsPlotter,
    )

    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png", trainings=[2], skip_first_sync_buckets=1,
        keep_first_sync_buckets=4, metric="mean_contacts",
    )
    intervals, flies, attempts = [], [], []
    original_cleanup = analysis.VideoAnalysis.clean_up_boundary_contact_data

    def capture(va, **kwargs):
        collector = WallContactsPerRewardIntervalTotalsPlotter(
            vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg,
        )
        rows = collector._collect_wall_contact_interval_episode_rows(include_target_ineligible=True)
        rows = [r for r in rows if r["training_index"] == 2]
        if rows:
            trace = va.trx[0].boundary_event_stats["wall"]["all"]["edge"]["boundary_contact"]
            if np.asarray(trace).dtype.kind != "f":
                raise ValueError("fresh floating wall trace required")
            observed, fractions, eligible = interval_observation_stats(
                trace, [r["start_reward_frame"] for r in rows], [r["end_reward_frame"] for r in rows],
            )
            for r, o, frac, ok in zip(rows, observed, fractions, eligible):
                r.update(group=label, classifiable_frames=int(o), classifiable_fraction=float(frac),
                         strict_eligible=bool(ok), entirely_unknown=bool(o == 0))
            counts = np.asarray([r["wall_contact_event_count"] for r in rows])
            _, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid([va], opts)
            target = bool(exp_target_sync_bucket_eligibility_mask([va], opts)[0])
            final_sli = float(sli_ts[0, 1, 4]) if target else np.nan
            # Validate our reduction against the unchanged production collector.
            production = collector._collect_wall_contact_interval_means_by_training_per_fly()[1]
            expected_value = float(np.mean(counts)) if target and len(rows) >= 5 else np.nan
            actual_value = production[0][1] if production else np.nan
            if not np.isclose(expected_value, actual_value, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise AssertionError("diagnostic reduction differs from production collector")
            flies.append({
                "group": label, "unit_id": rows[0]["unit_id"], "video_path": va.fn,
                "n_intervals": len(rows), "n_eligible": int(eligible.sum()),
                "n_unknown": int((~eligible).sum()), "n_entirely_unknown": int((observed == 0).sum()),
                "episodes_total": int(counts.sum()), "episodes_discarded": int(counts[~eligible].sum()),
                "baseline_mean": float(np.mean(counts)),
                "strict_mean": float(np.mean(counts[eligible])) if eligible.any() else np.nan,
                "no_eligible_intervals": not bool(eligible.any()),
                "target_eligible": target, "final_sli": final_sli,
                "scalar_included": target and len(rows) >= 5,
                "manuscript_included": target and len(rows) >= 5 and bool(np.isfinite(final_sli)),
                "strict_manuscript_included": target and int(eligible.sum()) >= 5 and bool(np.isfinite(final_sli)),
                **{f"classifiable_fraction_q{q}": float(v) for q, v in zip(
                    [0, 25, 50, 75, 100], np.quantile(fractions, [0, .25, .5, .75, 1]))},
            })
            intervals.extend(rows)
        original_cleanup(va, **kwargs)

    paths = {}
    for entry in opts.video.split(","):
        path, raw_ids = analysis.detect_and_split_input(entry)
        ids = analysis.util.parseIntList(raw_ids or opts.fly)
        for fn in analysis.util.fileList(path, "analyze", pattern=analysis.AVI_X):
            paths.setdefault(fn, []).extend(ids)
    ordered = sorted(paths, key=lambda fn: (int(analysis.util.firstGroup(analysis.CAM_NUM, analysis.util.basename(fn))), fn))
    tasks = [(fn, f) for fn in ordered for f in paths[fn]]
    if len(tasks) != len(set(tasks)):
        raise ValueError("duplicate video/fly inputs")
    if limit:
        tasks = tasks[:limit]
    analysis.VideoAnalysis.clean_up_boundary_contact_data = capture
    try:
        for idx, (fn, f) in enumerate(tasks, 1):
            print(f"DIAGNOSTIC {slug} {idx}/{len(tasks)} {fn} fly={f}", flush=True)
            va = analysis.VideoAnalysis(fn, 0, opts, f)
            attempts.append({"video_path": fn, "va_tag": f, "skipped": va.skipped()})
            va.cap.release()
            del va
            # Keep raw input caching bounded without changing any derived data.
            if idx == len(tasks) or tasks[idx][0] != fn:
                analysis.VideoAnalysis.fileCache.clear()
    finally:
        analysis.VideoAnalysis.clean_up_boundary_contact_data = original_cleanup

    archived_rows = read_csv(Path(str(stem) + "_episodes.csv"))
    archived = load_export_npz(label, str(stem) + "_archive.npz")
    bundle_path = repo / "exports/manuscript" / f"fig4k_{slug}_sli_eligibility.npz"
    with np.load(bundle_path, allow_pickle=True) as b:
        known, sli_eligible = _scalar_sli_eligibility([label], [(label, b)], "final", None)[label]
    scalar_archive = dict(zip(map(str, archived.per_unit_ids_panel[0]), map(float, archived.per_unit_values_panel[0])))
    scalar_archive = {k: v for k, v in scalar_archive.items() if np.isfinite(v)}
    final_archive = {k: v for k, v in scalar_archive.items() if _scalar_subject_key(k) in sli_eligible}
    current, old = {key(r): r for r in intervals}, {key(r): r for r in archived_rows}
    if len(current) != len(intervals) or len(old) != len(archived_rows):
        raise ValueError("duplicate interval identities")
    differences = []
    for k in sorted(current.keys() | old.keys()):
        if k not in current or k not in old:
            differences.append({"kind": "interval_membership", "identity": str(k), "archived": k in old, "recomputed": k in current})
            continue
        for field in ["wall_contact_event_count", "window_start_frame", "window_end_frame", "passes_target_sync_bucket_filter"]:
            a, b = str(old[k][field]), str(current[k][field])
            if a != b:
                differences.append({"kind": field, "identity": str(k), "archived": a, "recomputed": b})
    current_scalar = {r["unit_id"]: r["baseline_mean"] for r in flies if r["scalar_included"]}
    current_final = {r["unit_id"]: r["baseline_mean"] for r in flies if r["manuscript_included"]}
    for kind, before, after in [("scalar", scalar_archive, current_scalar), ("final_manuscript", final_archive, current_final)]:
        for uid in sorted(before.keys() | after.keys()):
            if uid not in before or uid not in after or not np.isclose(before[uid], after[uid], rtol=1e-10, atol=1e-12):
                differences.append({"kind": kind, "identity": uid, "archived": before.get(uid), "recomputed": after.get(uid)})
    for r in flies:
        r["archived_scalar_included"] = r["unit_id"] in scalar_archive
        r["archived_manuscript_included"] = r["unit_id"] in final_archive
        r["mean_change"] = r["strict_mean"] - r["baseline_mean"]
    # Check the actual final GraphPad input, preserving archive source order.
    graphpad_path = repo / "exports/manuscript/fig4k_t2_sb2_sb5_wall_contacts_sli_eligible_t2_sb5_graphpad.csv"
    graphpad = [float(r[label]) for r in read_csv(graphpad_path) if r[label].strip()]
    expected = list(final_archive.values())
    graphpad_matches = len(graphpad) == len(expected) and bool(np.allclose(graphpad, expected, rtol=1e-10, atol=1e-12))
    if not graphpad_matches:
        differences.append({"kind": "archived_graphpad_values", "identity": label})
    result = {
        "cohort": label, "partial_run": bool(limit),
        "baseline_verified": not differences and not limit,
        "baseline_discrepancies": len(differences),
        "archived_interval_count": len(old), "recomputed_interval_count": len(current),
        "archived_interval_flies": len({r["unit_id"] for r in archived_rows}),
        "recomputed_interval_flies": len(flies),
        "archived_scalar_n": len(scalar_archive), "recomputed_scalar_n": len(current_scalar),
        "archived_manuscript_n": len(final_archive), "recomputed_manuscript_n": len(current_final),
        "archived_graphpad_matches": graphpad_matches,
        "all_interval_flies": summarize(flies, intervals),
        "baseline_manuscript_flies": summarize([r for r in flies if r["manuscript_included"]], intervals),
        "strict_manuscript_flies": summarize([r for r in flies if r["strict_manuscript_included"]], intervals),
        "source_command": provenance["analysis_command"], "effective_argv": sys.argv,
        "boundary_extension": boundary_contact.__file__, "boundary_extension_sha256": sha256(Path(boundary_contact.__file__)),
        "input_sha256": {str(p): sha256(p) for p in [provenance_path, Path(str(stem) + "_episodes.csv"), Path(str(stem) + "_archive.npz"), bundle_path, graphpad_path]},
        "python": sys.executable, "numpy": np.__version__,
        "interpretation": "Strict results are provisional until baseline_verified is true; no discrepancies are reconciled.",
    }
    write_csv(out / "intervals.csv", intervals)
    write_csv(out / "flies.csv", flies)
    write_csv(out / "attempts.csv", attempts)
    write_csv(out / "baseline_discrepancies.csv", differences)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True, help="New diagnostic-only directory")
    p.add_argument("--cohort", choices=list(COHORTS), required=True)
    p.add_argument("--limit", type=int, default=0, help="Pilot only; cannot verify the full baseline")
    args = p.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = str(out / "mplconfig")
    # Isolate incidental relative paths; never enter production postAnalyze.
    os.chdir(out)
    start = time.monotonic()
    with (out / "analysis.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = run_cohort(repo, out, args.cohort, args.limit)
    print(json.dumps({"out": str(out), "baseline_verified": result["baseline_verified"],
                      "elapsed_seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
