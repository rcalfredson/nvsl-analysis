"""Keyed exports and matching for closed/open-loop correlation analyses."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_CAMERA_RE = re.compile(r"^(c\d+)", re.IGNORECASE)

CLOSED_FIELDS = (
    "recording_date",
    "camera_id",
    "pair_id",
    "pair_key",
    "exp_subject_key",
    "yoked_subject_key",
    "closed_video",
    "group_index",
    "exp_fly_id",
    "yoked_fly_id",
    "sli_training",
    "sli_sync_bucket",
    "sli_t2_sb5",
    "sli_t2_sb2_sb5_mean",
    "exp_reward_pi_t2_sb5",
    "yoked_reward_pi_t2_sb5",
    "exp_reward_pi_t2_sb2_sb5_mean",
    "yoked_reward_pi_t2_sb2_sb5_mean",
)

OPEN_FIELDS = (
    "recording_date",
    "camera_id",
    "fly_id",
    "subject_key",
    "open_loop_video",
    "group_index",
    "preference_pi_pre",
    "preference_pi_led_on",
    "preference_pi_led_off",
)

AUDIT_FIELDS = (
    "status",
    "detail",
    "pair_key",
    "exp_subject_key",
    "closed_exp_subject_key",
    "yoked_subject_key",
    "open_loop_subject_key",
    "recording_date",
    "camera_id",
    "open_loop_recording_date",
    "open_loop_camera_id",
    "pair_id",
    "exp_fly_id",
    "yoked_fly_id",
    "open_loop_fly_id",
    "closed_video",
    "open_loop_video",
    "match_rule",
    "match_validated",
    "sli_t2_sb5",
    "sli_t2_sb2_sb5_mean",
    "preference_pi_led_on",
    "preference_pi_led_off",
    "finite_sli",
    "finite_mean_sli",
    "finite_led_on",
    "finite_led_off",
    "included_led_on",
    "included_led_off",
    "included_mean_sli_led_on",
    "included_mean_sli_led_off",
)


def recording_identity(video_path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return ``(YYYY-MM-DD, camera_id)`` parsed from a video path."""
    path = str(video_path)
    date_match = _DATE_RE.search(path)
    camera_match = _CAMERA_RE.search(Path(path).name)
    if date_match is None:
        raise ValueError(f"could not find YYYY-MM-DD recording date in {path!r}")
    if camera_match is None:
        raise ValueError(f"could not find leading camera id in {Path(path).name!r}")
    return date_match.group(1), camera_match.group(1).lower()


def subject_key(recording_date: str, camera_id: str, fly_id: int) -> str:
    return f"{recording_date}::{str(camera_id).lower()}::f{int(fly_id)}"


def pair_key(recording_date: str, camera_id: str, pair_id: int) -> str:
    return f"{recording_date}::{str(camera_id).lower()}::pair{int(pair_id)}"


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _physical_fly_ids(va) -> tuple[int, ...]:
    ids = getattr(va, "trxf", None)
    if ids is not None:
        return tuple(int(value) for value in ids)
    fly_id = getattr(va, "f", None)
    if fly_id is None:
        label = getattr(va, "fn", "<unknown>")
        raise ValueError(f"video analysis {label!r} has no fly id")
    return (int(fly_id),)


def build_closed_loop_sli_rows(
    vas: Sequence,
    raw_reward_pi: np.ndarray,
    *,
    training_idx: int = 1,
    sync_bucket_idx: int = 4,
) -> list[dict[str, object]]:
    """Build one final and mean T2 SLI row per closed-loop exp+yoked pair."""
    values = np.asarray(raw_reward_pi, dtype=float)
    if values.ndim != 4 or values.shape[0] != len(vas):
        raise ValueError(
            "closed-loop reward PI must have shape "
            "(videos, trainings, flies, sync_buckets) aligned with vas"
        )
    if values.shape[1] <= training_idx or values.shape[2] < 2:
        raise ValueError("closed-loop export requires T2 experimental+yoked reward PI")
    if values.shape[3] <= max(sync_bucket_idx, 4):
        raise ValueError("closed-loop export requires sync bucket 5 in T2")

    rows: list[dict[str, object]] = []
    for idx, va in enumerate(vas):
        video = str(getattr(va, "fn", ""))
        recording_date, camera_id = recording_identity(video)
        physical_ids = _physical_fly_ids(va)
        if len(physical_ids) != 2:
            raise ValueError(
                f"closed-loop video analysis {video!r} must resolve to exactly "
                f"experimental+yoked physical fly ids; got {physical_ids}"
            )
        exp_fly_id, yoked_fly_id = physical_ids
        exp_value = float(values[idx, training_idx, 0, sync_bucket_idx])
        yoked_value = float(values[idx, training_idx, 1, sync_bucket_idx])
        exp_mean = float(np.nanmean(values[idx, training_idx, 0, 1:5]))
        yoked_mean = float(np.nanmean(values[idx, training_idx, 1, 1:5]))
        rows.append(
            {
                "recording_date": recording_date,
                "camera_id": camera_id,
                "pair_id": exp_fly_id,
                "pair_key": pair_key(recording_date, camera_id, exp_fly_id),
                "exp_subject_key": subject_key(recording_date, camera_id, exp_fly_id),
                "yoked_subject_key": subject_key(
                    recording_date, camera_id, yoked_fly_id
                ),
                "closed_video": video,
                "group_index": int(getattr(va, "gidx", 0)),
                "exp_fly_id": exp_fly_id,
                "yoked_fly_id": yoked_fly_id,
                "sli_training": training_idx + 1,
                "sli_sync_bucket": sync_bucket_idx + 1,
                "sli_t2_sb5": exp_value - yoked_value,
                "sli_t2_sb2_sb5_mean": exp_mean - yoked_mean,
                "exp_reward_pi_t2_sb5": exp_value,
                "yoked_reward_pi_t2_sb5": yoked_value,
                "exp_reward_pi_t2_sb2_sb5_mean": exp_mean,
                "yoked_reward_pi_t2_sb2_sb5_mean": yoked_mean,
            }
        )
    return rows


def build_open_loop_preference_rows(vas: Sequence) -> list[dict[str, object]]:
    """Build one pre/on/off positional-PI row per open-loop physical fly."""
    rows: list[dict[str, object]] = []
    for va in vas:
        video = str(getattr(va, "fn", ""))
        if not bool(getattr(va, "openLoop", False)):
            raise ValueError(f"open-loop export received non-open-loop video {video!r}")
        if bool(getattr(va, "alt", True)):
            raise ValueError(
                f"open-loop export requires an LED on/off protocol, not an "
                f"alternating-side protocol: {video!r}"
            )
        physical_ids = _physical_fly_ids(va)
        if len(physical_ids) != 1:
            raise ValueError(
                f"open-loop video analysis {video!r} must resolve to one physical "
                f"fly id; got {physical_ids}"
            )
        preferences = list(getattr(va, "posPI", []))
        if len(preferences) != 1:
            raise ValueError(
                f"open-loop video analysis {video!r} must contain exactly one "
                f"pre/on/off preference record; got {len(preferences)}"
            )
        pi = np.asarray(preferences[0], dtype=float).reshape(-1)
        if pi.size != 3:
            raise ValueError(
                f"open-loop preference record for {video!r} must contain "
                f"pre/on/off values; got {pi.size}"
            )
        recording_date, camera_id = recording_identity(video)
        fly_id = physical_ids[0]
        rows.append(
            {
                "recording_date": recording_date,
                "camera_id": camera_id,
                "fly_id": fly_id,
                "subject_key": subject_key(recording_date, camera_id, fly_id),
                "open_loop_video": video,
                "group_index": int(getattr(va, "gidx", 0)),
                "preference_pi_pre": float(pi[0]),
                "preference_pi_led_on": float(pi[1]),
                "preference_pi_led_off": float(pi[2]),
            }
        )
    return rows


def write_rows_csv(
    rows: Iterable[Mapping[str, object]],
    out_csv: str | os.PathLike[str],
    fieldnames: Sequence[str],
) -> Path:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_closed_loop_sli_csv(
    vas: Sequence, raw_reward_pi: np.ndarray, out_csv: str | os.PathLike[str]
) -> Path:
    rows = build_closed_loop_sli_rows(vas, raw_reward_pi)
    path = write_rows_csv(rows, out_csv, CLOSED_FIELDS)
    print(f"[cross-experiment] wrote closed-loop SLI export {path} (n={len(rows)})")
    return path


def export_open_loop_preference_csv(
    vas: Sequence, out_csv: str | os.PathLike[str]
) -> Path:
    rows = build_open_loop_preference_rows(vas)
    path = write_rows_csv(rows, out_csv, OPEN_FIELDS)
    print(f"[cross-experiment] wrote open-loop preference export {path} (n={len(rows)})")
    return path


def read_rows_csv(
    path: str | os.PathLike[str], required_fields: Sequence[str]
) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def join_closed_to_experimental_open_loop(
    closed_rows: Sequence[Mapping[str, object]],
    open_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join each closed pair to its experimental fly's open-loop record.

    Returns ``(matched_rows, audit_rows)``. Duplicate keys are rejected; unmatched
    closed and unused open-loop records are retained in the audit.
    """
    open_by_key: dict[str, Mapping[str, object]] = {}
    for row in open_rows:
        key = str(row["subject_key"])
        if key in open_by_key:
            raise ValueError(f"duplicate open-loop subject_key: {key}")
        open_by_key[key] = row

    seen_pairs: set[str] = set()
    used_open: set[str] = set()
    yoked_subject_keys = {str(row["yoked_subject_key"]) for row in closed_rows}
    matched: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for closed in closed_rows:
        pkey = str(closed["pair_key"])
        if pkey in seen_pairs:
            raise ValueError(f"duplicate closed-loop pair_key: {pkey}")
        seen_pairs.add(pkey)
        exp_key = str(closed["exp_subject_key"])
        opened = open_by_key.get(exp_key)
        base = {field: closed.get(field, "") for field in AUDIT_FIELDS}
        base.update(
            closed_exp_subject_key=exp_key,
            match_rule="recording_date+camera_id+physical_fly_id",
            match_validated=False,
        )
        if opened is None:
            base.update(
                status="unmatched_closed",
                detail="no experimental-fly open-loop record",
            )
            audit.append(base)
            continue
        used_open.add(exp_key)
        joined = dict(closed)
        joined.update(
            open_loop_video=opened.get("open_loop_video", ""),
            closed_exp_subject_key=exp_key,
            open_loop_subject_key=opened.get("subject_key", ""),
            open_loop_recording_date=opened.get("recording_date", ""),
            open_loop_camera_id=opened.get("camera_id", ""),
            open_loop_fly_id=opened.get("fly_id", ""),
            match_rule="recording_date+camera_id+physical_fly_id",
            match_validated=(str(opened.get("subject_key", "")) == exp_key),
            preference_pi_led_on=opened.get("preference_pi_led_on", "nan"),
            preference_pi_led_off=opened.get("preference_pi_led_off", "nan"),
            preference_pi_pre=opened.get("preference_pi_pre", "nan"),
        )
        matched.append(joined)
        finite_sli = _is_finite(joined.get("sli_t2_sb5"))
        finite_mean_sli = _is_finite(joined.get("sli_t2_sb2_sb5_mean"))
        finite_led_on = _is_finite(joined.get("preference_pi_led_on"))
        finite_led_off = _is_finite(joined.get("preference_pi_led_off"))
        invalid = []
        if not finite_sli:
            invalid.append("nonfinite SLI")
        if not finite_mean_sli:
            invalid.append("nonfinite mean SLI")
        if not finite_led_on:
            invalid.append("nonfinite LED-on PI")
        if not finite_led_off:
            invalid.append("nonfinite LED-off PI")
        audit.append(
            {
                **{field: joined.get(field, "") for field in AUDIT_FIELDS},
                "status": "matched",
                "detail": "; ".join(invalid),
                "finite_sli": finite_sli,
                "finite_mean_sli": finite_mean_sli,
                "finite_led_on": finite_led_on,
                "finite_led_off": finite_led_off,
                "included_led_on": finite_sli and finite_led_on,
                "included_led_off": finite_sli and finite_led_off,
                "included_mean_sli_led_on": finite_mean_sli and finite_led_on,
                "included_mean_sli_led_off": finite_mean_sli and finite_led_off,
            }
        )

    for key, opened in open_by_key.items():
        if key in used_open:
            continue
        is_yoked_record = key in yoked_subject_keys
        audit.append(
            {
                "status": (
                    "available_yoked_open_loop"
                    if is_yoked_record
                    else "unused_open_loop"
                ),
                "detail": (
                    "retained but excluded from primary experimental-fly correlations"
                    if is_yoked_record
                    else "no closed-loop pair uses this fly as experimental"
                ),
                "exp_subject_key": key,
                "open_loop_subject_key": key,
                "recording_date": opened.get("recording_date", ""),
                "camera_id": opened.get("camera_id", ""),
                "open_loop_recording_date": opened.get("recording_date", ""),
                "open_loop_camera_id": opened.get("camera_id", ""),
                "exp_fly_id": opened.get("fly_id", ""),
                "open_loop_fly_id": opened.get("fly_id", ""),
                "open_loop_video": opened.get("open_loop_video", ""),
                "match_rule": "recording_date+camera_id+physical_fly_id",
                "match_validated": False,
                "preference_pi_led_on": opened.get("preference_pi_led_on", ""),
                "preference_pi_led_off": opened.get("preference_pi_led_off", ""),
            }
        )
    return matched, audit
