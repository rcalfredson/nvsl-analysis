"""Explicit cross-recording pairing for experimental-fly training heatmaps."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(video, fly):
    return str(Path(video).expanduser().resolve()), str(fly)


def skipped_eligibility_record(va, video, fly):
    """Record an attempted fly without accessing uninitialized analysis fields."""
    trajectories = getattr(va, "trx", ())
    if trajectories and trajectories[0].bad():
        reason = "rejected_trajectory"
    elif getattr(va, "isyc", False):
        reason = "not_experimental_fly"
    else:
        reason = "analysis_skipped"
    video, fly = identity(video, fly)
    return dict(video=video, fly=fly, occupancy_ok=False, heatmap_ok=False,
                reason=reason, heatmap_digest=None)


def eligibility_report(vas, opts, training_index):
    """Use occupancy's full-bucket validity, then check the plotted subwindow."""
    records = []
    for va in vas:
        trn = va.trns[training_index]
        trx = va.trx[0]
        df = va._min2f(opts.syncBucketLenMin)
        start, count, _ = va._syncBucket(trn, df)
        reason = ""
        if trx.bad():
            reason = "rejected_trajectory"
        elif start is None or not np.isfinite(start):
            reason = "missing_sync"
        else:
            first = int(start + (opts.hm_sync_bucket - 1) * df)
            last = first + df
            limit = min(trn.stop, int(trn.start + count * df))
            if opts.hm_sync_bucket > count or last > limit:
                reason = "incomplete_bucket"
            elif not np.any(~trx.nan[first:last]):
                reason = "no_valid_bucket_frames"
        occupancy_ok = not reason
        mp = va.heatmap[0][training_index][0]
        heatmap_ok = mp is not None and bool(np.sum(mp) > 0)
        if not reason and not heatmap_ok:
            reason = "empty_heatmap_window"
        video, fly = identity(va.fn, va.f)
        records.append(dict(video=video, fly=fly, occupancy_ok=occupancy_ok,
                            heatmap_ok=heatmap_ok, reason=reason,
                            heatmap_digest=None if mp is None else hashlib.sha256(
                                np.asarray(mp, dtype="<f8").tobytes()).hexdigest()))
    records.extend(getattr(opts, "_hm_pair_skipped_records", ()))
    report = dict(version=2, window=dict(
        training=training_index + 1, bucket=opts.hm_sync_bucket,
        bucket_minutes=opts.syncBucketLenMin,
        head_minutes=opts.hm_sync_bucket_head_minutes,
        tail_minutes=opts.hm_sync_bucket_tail_minutes), records=records)
    _index(report)
    return report


def _index(report):
    if report.get("version") != 2:
        raise ValueError("Unsupported heatmap eligibility report version; regenerate both reports to include early-skipped flies")
    result = {}
    for row in report["records"]:
        key = identity(row["video"], row["fly"])
        if key in result:
            raise ValueError(f"Duplicate heatmap recording/fly: {key}")
        result[key] = row
    return result


def paired_keys(manifest, before, after, side):
    """Return a shared intersection and an audit row for every explicit pair."""
    if side not in ("before", "after"):
        raise ValueError("Pair side must be before or after")
    indices = {"before": _index(before), "after": _index(after)}
    seen = {"before": set(), "after": set()}
    pair_ids, keep, audit = set(), set(), []
    with open(manifest, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "before_video", "before_fly", "after_video", "after_fly"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Pair manifest requires columns: {sorted(required)}")
        for row in reader:
            pair_id = row["pair_id"]
            if not pair_id or pair_id in pair_ids:
                raise ValueError(f"Missing or duplicate pair_id: {pair_id!r}")
            pair_ids.add(pair_id)
            keys, reasons = {}, {}
            for label in ("before", "after"):
                if not row[f"{label}_video"] or not row[f"{label}_fly"]:
                    raise ValueError(f"Missing {label} identity for {pair_id}")
                keys[label] = identity(row[f"{label}_video"], row[f"{label}_fly"])
                if keys[label] in seen[label]:
                    raise ValueError(f"Recording/fly reused in manifest: {keys[label]}")
                seen[label].add(keys[label])
                record = indices[label].get(keys[label])
                reasons[label] = "missing_recording" if record is None else record["reason"]
            included = not any(reasons.values())
            if included:
                keep.add(keys[side])
            audit.append({**row, "included": included,
                          "before_reason": reasons["before"], "after_reason": reasons["after"]})
    return keep, audit


def prepare_paired_heatmaps(vas, opts, train_indices):
    """Export eligibility only, or validate reports and select a paired cohort."""
    export = getattr(opts, "hm_pair_export", None)
    manifest = getattr(opts, "hm_pair_manifest", None)
    if not export and not manifest:
        if any(getattr(opts, name, None) for name in (
            "hm_pair_side", "hm_pair_before_report", "hm_pair_after_report", "hm_pair_audit"
        )):
            raise ValueError("Heatmap pairing options require --hm-pair-manifest")
        return vas, False
    if export and manifest:
        raise ValueError("Use --hm-pair-export separately from --hm-pair-manifest")
    if (tuple(opts.hm_periods) != ("training",) or len(train_indices) != 1
            or opts.hm_sync_bucket is None or opts.hm_sync_bucket < 1):
        raise ValueError("Heatmap pairing requires training-only heatmaps, one training, and --hm-sync-bucket")
    current = eligibility_report(vas, opts, train_indices[0])
    if export:
        Path(export).parent.mkdir(parents=True, exist_ok=True)
        Path(export).write_text(json.dumps(current, indent=2) + "\n")
        print(f"[heatmap pairing] exported {len(current['records'])} records to {export}; no heatmap plotted")
        return vas, True
    if not all(getattr(opts, name, None) for name in (
        "hm_pair_side", "hm_pair_before_report", "hm_pair_after_report", "hm_pair_audit"
    )):
        raise ValueError("Paired heatmaps require --hm-pair-side, --hm-pair-before-report, --hm-pair-after-report, and --hm-pair-audit")
    before = json.loads(Path(opts.hm_pair_before_report).read_text())
    after = json.loads(Path(opts.hm_pair_after_report).read_text())
    _index(before)
    _index(after)
    saved = before if opts.hm_pair_side == "before" else after
    if current["window"] != saved["window"] or _index(current) != _index(saved):
        raise ValueError("Current heatmap selection or eligibility differs from saved report; regenerate both reports")
    keep, audit = paired_keys(manifest, before, after, opts.hm_pair_side)
    path = Path(opts.hm_pair_audit)
    if path.resolve() in {Path(p).resolve() for p in (
        manifest, opts.hm_pair_before_report, opts.hm_pair_after_report
    )}:
        raise ValueError("Pair audit output must not overwrite the manifest or eligibility reports")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        fields = ["pair_id", "before_video", "before_fly", "after_video", "after_fly",
                  "included", "before_reason", "after_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit)
    selected = [va for va in vas if identity(va.fn, va.f) in keep]
    if not selected:
        raise ValueError(f"No eligible heatmap pairs; see {path}")
    print(f"[heatmap pairing] included {len(selected)}/{len(audit)} pairs; audit: {path}")
    return selected, False
