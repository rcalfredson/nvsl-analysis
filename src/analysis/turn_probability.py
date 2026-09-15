from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def turn_probabilities_in_range(
    contact_start_data,
    turn_results: Mapping[int, bool],
    measurement_range: slice,
    *,
    min_contact_events: int,
) -> tuple[float, float]:
    """Return toward/away sharp-turn probabilities using metric-local eligibility."""
    start = measurement_range.start
    stop = measurement_range.stop
    if not (np.isfinite(start) and np.isfinite(stop) and stop > start):
        return np.nan, np.nan

    contact_starts = np.asarray(contact_start_data)
    num_contact_events = int(
        np.count_nonzero((contact_starts >= start) & (contact_starts <= stop))
    )
    if num_contact_events == 0 or num_contact_events < int(min_contact_events):
        return np.nan, np.nan

    start = int(start)
    stop = int(stop + 1)
    num_turns_toward = sum(
        1 for frame in range(start, stop) if bool(turn_results.get(frame, False))
    )
    num_turns_away = sum(
        1
        for frame in range(start, stop)
        if frame in turn_results and not bool(turn_results[frame])
    )
    return num_turns_toward / num_contact_events, num_turns_away / num_contact_events
