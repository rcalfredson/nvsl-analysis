#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


PREFIX_DEFAULT = "exp_target_sync_bucket_filter"
PREFIX_LEGACY = "exp_pi_threshold_filter"
MISSING_TARGET_REASONS = {
    "missing_sync_bucket_data",
    "target_training_missing",
    "target_sync_bucket_missing",
    "target_sync_bucket_invalid",
    # Legacy reasons from exp_pi_threshold_filter bundles.
    "target_training_or_fly_missing",
    "target_sync_bucket_nan",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def _array(data: np.lib.npyio.NpzFile, key: str, default=None) -> np.ndarray:
    if key not in data.files:
        if default is None:
            return np.asarray([])
        return np.asarray(default)
    return np.asarray(data[key])


def _decode_strings(values: np.ndarray) -> list[str]:
    out = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _episode_filter_overlap(data: np.lib.npyio.NpzFile, n_videos: int) -> dict:
    counts = _array(data, "turnback_excursion_bin_total_exp")
    if counts.shape[:1] != (n_videos,):
        return {
            "min_episodes": 0,
            "counts": None,
            "pass_any": np.full(n_videos, False, dtype=bool),
            "pass_all": np.full(n_videos, False, dtype=bool),
        }

    min_episodes = int(_scalar(data, "min_turnback_episodes", 0) or 0)
    if min_episodes <= 0:
        passed = np.ones(counts.shape, dtype=bool)
    else:
        passed = np.asarray(counts, dtype=int) >= min_episodes
    return {
        "min_episodes": min_episodes,
        "counts": np.asarray(counts, dtype=int),
        "pass_any": np.any(passed, axis=tuple(range(1, passed.ndim))),
        "pass_all": np.all(passed, axis=tuple(range(1, passed.ndim))),
    }


def _bundle_label(path: Path, data: np.lib.npyio.NpzFile) -> str:
    label = _scalar(data, "group_label", None)
    if label is None:
        return path.stem
    if isinstance(label, bytes):
        return label.decode("utf-8", errors="replace")
    label = str(label)
    return label if label else path.stem


def _resolve_prefix(data: np.lib.npyio.NpzFile, prefix: str) -> str:
    if f"{prefix}_reason" in data.files and f"{prefix}_eligible" in data.files:
        return prefix
    if prefix == PREFIX_DEFAULT:
        legacy_reason = f"{PREFIX_LEGACY}_reason"
        legacy_eligible = f"{PREFIX_LEGACY}_eligible"
        if legacy_reason in data.files and legacy_eligible in data.files:
            return PREFIX_LEGACY
    return prefix


def summarize_bundle(
    path: Path, *, prefix: str = PREFIX_DEFAULT
) -> tuple[dict, list[dict]]:
    with np.load(path, allow_pickle=True) as data:
        prefix = _resolve_prefix(data, prefix)
        reason_key = f"{prefix}_reason"
        eligible_key = f"{prefix}_eligible"
        if reason_key not in data.files or eligible_key not in data.files:
            raise ValueError(
                f"{path} does not contain {prefix!r} reason/eligible arrays"
            )

        reasons = _decode_strings(_array(data, reason_key))
        eligible = np.asarray(_array(data, eligible_key), dtype=bool).reshape(-1)
        if len(reasons) != eligible.size:
            raise ValueError(
                f"{path} has {len(reasons)} reasons but {eligible.size} eligibility values"
            )

        video_ids = _decode_strings(_array(data, "video_ids", np.arange(eligible.size)))
        if len(video_ids) != eligible.size:
            video_ids = [str(i) for i in range(eligible.size)]
        available_sync_buckets = _array(
            data, f"{prefix}_available_sync_buckets", np.full(eligible.size, np.nan)
        ).reshape(-1)
        target_bucket_starts = _array(
            data, f"{prefix}_target_bucket_start", np.full(eligible.size, np.nan)
        ).reshape(-1)
        target_bucket_stops = _array(
            data, f"{prefix}_target_bucket_stop", np.full(eligible.size, np.nan)
        ).reshape(-1)
        if available_sync_buckets.size != eligible.size:
            available_sync_buckets = np.full(eligible.size, np.nan)
        if target_bucket_starts.size != eligible.size:
            target_bucket_starts = np.full(eligible.size, np.nan)
        if target_bucket_stops.size != eligible.size:
            target_bucket_stops = np.full(eligible.size, np.nan)
        episode_overlap = _episode_filter_overlap(data, eligible.size)

        counts = Counter(reasons)
        excluded_reasons = Counter(
            reason for reason, keep in zip(reasons, eligible) if not keep
        )
        missing_target_bucket = sum(
            excluded_reasons.get(reason, 0) for reason in MISSING_TARGET_REASONS
        )
        other_excluded = int((~eligible).sum()) - missing_target_bucket
        excluded_mask = ~eligible
        min_episode_failed_all = int(
            np.count_nonzero(excluded_mask & ~episode_overlap["pass_any"])
        )
        min_episode_failed_some = int(
            np.count_nonzero(excluded_mask & ~episode_overlap["pass_all"])
        )

        summary = {
            "bundle": str(path),
            "label": _bundle_label(path, data),
            "prefix": prefix,
            "enabled": bool(_scalar(data, f"{prefix}_enabled", False)),
            "training": int(_scalar(data, f"{prefix}_training", 0) or 0),
            "sync_bucket": int(_scalar(data, f"{prefix}_sync_bucket", 0) or 0),
            "n_videos": int(eligible.size),
            "n_eligible": int(eligible.sum()),
            "n_excluded": int((~eligible).sum()),
            "n_missing_target_bucket": int(missing_target_bucket),
            "n_other_excluded": int(other_excluded),
            "min_episodes": int(episode_overlap["min_episodes"]),
            "n_excluded_failing_min_episodes_all_pairs": min_episode_failed_all,
            "n_excluded_failing_min_episodes_some_pair": min_episode_failed_some,
            "reason_counts": dict(counts),
            "excluded_reason_counts": dict(excluded_reasons),
        }

        rows = []
        for idx, (
            video_id,
            keep,
            reason,
            available_buckets,
            bucket_start,
            bucket_stop,
        ) in enumerate(
            zip(
                video_ids,
                eligible,
                reasons,
                available_sync_buckets,
                target_bucket_starts,
                target_bucket_stops,
            )
        ):
            if keep:
                continue
            if reason in MISSING_TARGET_REASONS:
                category = "missing_target_bucket"
            else:
                category = "other_excluded"
            rows.append(
                {
                    "bundle": str(path),
                    "label": summary["label"],
                    "video_index": idx,
                    "video_id": video_id,
                    "eligible": bool(keep),
                    "reason": reason,
                    "category": category,
                    "training": summary["training"],
                    "sync_bucket": summary["sync_bucket"],
                    "available_sync_buckets": available_buckets,
                    "target_bucket_start": bucket_start,
                    "target_bucket_stop": bucket_stop,
                    "min_episodes": summary["min_episodes"],
                    "turnback_episode_counts": (
                        ""
                        if episode_overlap["counts"] is None
                        else ";".join(
                            map(str, episode_overlap["counts"][idx].reshape(-1))
                        )
                    ),
                    "min_episode_pass_any_pair": bool(
                        episode_overlap["pass_any"][idx]
                    ),
                    "min_episode_pass_all_pairs": bool(
                        episode_overlap["pass_all"][idx]
                    ),
                }
            )

    return summary, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_reason_counts(counts: dict) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the experimental target sync-bucket filter accounting stored "
            "in one or more exported .npz bundles."
        )
    )
    parser.add_argument("bundles", nargs="+", help="Bundle .npz files to inspect.")
    parser.add_argument(
        "--prefix",
        default=PREFIX_DEFAULT,
        help=(
            f"Filter payload prefix in the bundle (default: {PREFIX_DEFAULT}; "
            f"falls back to legacy {PREFIX_LEGACY} when using the default)."
        ),
    )
    parser.add_argument(
        "--excluded-csv",
        default=None,
        help="Optional per-excluded-video CSV output path.",
    )
    args = parser.parse_args()

    summaries = []
    excluded_rows = []
    for bundle in args.bundles:
        summary, rows = summarize_bundle(Path(bundle), prefix=args.prefix)
        summaries.append(summary)
        excluded_rows.extend(rows)

    for summary in summaries:
        print(f"\n{summary['label']}")
        print(f"  bundle: {summary['bundle']}")
        print(
            "  target:"
            f" T{summary['training']} SB{summary['sync_bucket']},"
            f" enabled={summary['enabled']}"
        )
        print(
            "  videos:"
            f" total={summary['n_videos']},"
            f" eligible={summary['n_eligible']},"
            f" excluded={summary['n_excluded']}"
        )
        print(
            "  excluded breakdown:"
            f" missing_target_bucket={summary['n_missing_target_bucket']},"
            f" other={summary['n_other_excluded']}"
        )
        if summary["min_episodes"] > 0 and summary["n_excluded"] > 0:
            print(
                "  min-episode overlap among excluded:"
                f" threshold={summary['min_episodes']},"
                " fail_all_pairs="
                f"{summary['n_excluded_failing_min_episodes_all_pairs']},"
                " fail_some_pair="
                f"{summary['n_excluded_failing_min_episodes_some_pair']}"
            )
        print(f"  all reasons: {_fmt_reason_counts(summary['reason_counts'])}")
        if summary["excluded_reason_counts"]:
            print(
                "  excluded reasons:"
                f" {_fmt_reason_counts(summary['excluded_reason_counts'])}"
            )

    if len(summaries) > 1:
        total_videos = sum(s["n_videos"] for s in summaries)
        total_eligible = sum(s["n_eligible"] for s in summaries)
        total_excluded = sum(s["n_excluded"] for s in summaries)
        total_missing = sum(s["n_missing_target_bucket"] for s in summaries)
        total_other = sum(s["n_other_excluded"] for s in summaries)
        total_min_fail_all = sum(
            s["n_excluded_failing_min_episodes_all_pairs"] for s in summaries
        )
        total_min_fail_some = sum(
            s["n_excluded_failing_min_episodes_some_pair"] for s in summaries
        )
        print("\nTotal")
        print(
            "  videos:"
            f" total={total_videos}, eligible={total_eligible},"
            f" excluded={total_excluded}"
        )
        print(
            "  excluded breakdown:"
            f" missing_target_bucket={total_missing},"
            f" other={total_other}"
        )
        if any(s["min_episodes"] > 0 for s in summaries):
            print(
                "  min-episode overlap among excluded:"
                f" fail_all_pairs={total_min_fail_all},"
                f" fail_some_pair={total_min_fail_some}"
            )

    if args.excluded_csv:
        _write_csv(Path(args.excluded_csv), excluded_rows)
        print(f"\nWrote excluded-video rows: {args.excluded_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
