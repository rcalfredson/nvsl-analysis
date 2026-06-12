from __future__ import annotations

from typing import Any

import numpy as np


def wall_contact_regions_for_trj(
    trj,
    *,
    enabled: bool,
    warned_missing: list[bool] | None,
    log_tag: str,
):
    if not enabled:
        return None
    try:
        return trj.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]
    except (KeyError, TypeError, AttributeError):
        if warned_missing is not None and not warned_missing[0]:
            print(
                f"[{log_tag}] warning: requested wall-contact exclusion, but "
                "wall-contact regions were missing for some trajectories; "
                "those trajectories will be left unfiltered."
            )
            warned_missing[0] = True
        return None


def _region_to_start_stop(region: Any) -> tuple[int, int | None] | None:
    if region is None:
        return None

    if isinstance(region, slice):
        start = 0 if region.start is None else int(region.start)
        stop = None if region.stop is None else int(region.stop)
        return start, stop

    if isinstance(region, (tuple, list, np.ndarray)):
        try:
            if len(region) != 2:
                return None
            start = 0 if region[0] is None else int(region[0])
            stop = None if region[1] is None else int(region[1])
            return start, stop
        except Exception:
            return None

    if hasattr(region, "start") and hasattr(region, "stop"):
        try:
            start = 0 if getattr(region, "start") is None else int(region.start)
            stop = None if getattr(region, "stop") is None else int(region.stop)
            return start, stop
        except Exception:
            return None

    return None


def episode_overlaps_wall_contact(ep, wall_regions) -> bool:
    if not wall_regions:
        return False

    stop = int(ep.get("stop", 0))
    start = int(ep.get("start", max(0, stop - 1)))
    if stop <= start:
        stop = start + 1

    for region in wall_regions:
        ab = _region_to_start_stop(region)
        if ab is None:
            continue
        wall_start, wall_stop = ab
        wall_stop = stop if wall_stop is None else int(wall_stop)
        if min(wall_stop, stop) > max(wall_start, start):
            return True
    return False
