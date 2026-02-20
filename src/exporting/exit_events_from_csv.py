from __future__ import annotations

import os
import re
import csv
from dataclasses import dataclass

import numpy as np

from src.utils.common import writeImage
import src.utils.util as util

_SAN = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(s: str) -> str:
    s = (s or "").strip()
    s = _SAN.sub("_", s)
    return s.strip("_") or "unknown"


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _parse_int(v, *, default=None):
    if _is_blank(v):
        return default
    try:
        return int(float(v))
    except Exception:
        return default


@dataclass
class ExitEventImageConfig:
    out_root: str = "exports/exit_events"
    max_frames: int = 40  # cap per event to avoid monster strips
    stride: int = 1  # sample every N frames
    nc: int = 10  # columns in combineImgs
    d: int = 10  # spacing in combineImgs
    resize_fctr: float | None = None
    crop_radius_px: int | None = None  # if None, VA decides
    include_headers: bool = True


def export_exit_event_images_from_csv(
    *, csv_path: str, vas: list, cfg: ExitEventImageConfig
):
    """
    Reads the exit-event CSV, finds matching VideoAnalysis instances,
    renders cropped annotated frames per event, stitches, and writes images.
    """
    # Map (video_abs_path, fly_idx) -> VideoAnalysis
    # Note: VideoAnalysis stores analyzed fly as self.f, which matches the CLI fly id.
    va_map = {}
    for va in vas:
        key = (os.path.abspath(va.fn), int(va.f) if va.f is not None else 0)
        va_map[key] = va

    os.makedirs(cfg.out_root, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader_peek = f.read(2048)
        delim = "\t" if "\t" in reader_peek else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)

        n_total = 0
        n_written = 0
        n_skipped = 0

        for row in reader:
            n_total += 1

            video = row.get("video", "")
            fly_idx = _parse_int(row.get("fly_idx"), default=0)
            trx_idx = _parse_int(row.get("trx_idx"), default=0)

            if _is_blank(video):
                n_skipped += 1
                continue

            video_abs = os.path.abspath(video)
            va = va_map.get((video_abs, int(fly_idx)))
            if va is None:
                # CSV may include videos/flies not in this run's video list
                n_skipped += 1
                continue

            reject_reason = (row.get("reject_reason") or "").strip()
            is_large_turn = _is_blank(reject_reason)

            if is_large_turn:
                evt_type = "large_turn"
                has_lt = (row.get("has_large_turn") or "").strip().lower()
                if has_lt not in ("true", "1", "t", "yes", "y"):
                    # not fatal, but likely indicates CSV mismatch
                    n_skipped += 1
                    continue
                start = _parse_int(row.get("turn_start_idx"))
                end = _parse_int(row.get("turn_end_idx"))
            else:
                evt_type = _slug(reject_reason)
                start = _parse_int(row.get("reject_turn_start_idx"))
                end = _parse_int(row.get("reject_turn_end_idx"))

            if start is None or end is None or end < start:
                n_skipped += 1
                continue

            # Frame sampling
            frame_idxs = list(range(start, end + 1, max(1, int(cfg.stride))))
            if cfg.max_frames and len(frame_idxs) > cfg.max_frames:
                # Uniform subsample across the range
                frame_idxs = (
                    np.linspace(start, end, cfg.max_frames).round().astype(int).tolist()
                )

            tiles = []
            hdrs = []

            ok = True
            for fi in frame_idxs:
                try:
                    tile = va.render_annotated_frame(
                        fi,
                        trx_idx=trx_idx,
                        crop=True,
                        crop_radius_px=cfg.crop_radius_px,
                        include_training_annotation=True,
                        include_text_overlay=False,
                    )
                except Exception:
                    ok = False
                    break

                tiles.append(tile)
                if cfg.include_headers:
                    hdrs.append(str(fi))
            if not ok or not tiles:
                n_skipped += 1
                continue

            # Stitch
            stitched, _ = util.combineImgs(
                list(zip(tiles, hdrs)) if cfg.include_headers else tiles,
                nc=int(cfg.nc),
                d=int(cfg.d),
                resizeFctr=cfg.resize_fctr,
            )

            # Output path
            out_dir = os.path.join(cfg.out_root, evt_type)
            os.makedirs(out_dir, exist_ok=True)

            base = _slug(os.path.splitext(os.path.basename(video_abs))[0])
            fn_out = f"{base}__fly{int(fly_idx)}__{int(start)}-{int(end)}.png"
            out_path = os.path.join(out_dir, fn_out)

            writeImage(out_path, stitched)
            n_written += 1
        print(
            f"[exit_events] rows={n_total}, written={n_written}, skipped={n_skipped}, out={cfg.out_root}"
        )
