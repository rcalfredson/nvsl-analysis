"""Utilities for reproducible sampling of non-overlapping frame windows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_non_overlapping_frame_windows(
    *,
    domain_start: int,
    domain_stop: int,
    window_span: int,
    count: int,
    rng: np.random.Generator,
    resolve_bounds: Callable[[int, int], tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Sample inclusive windows whose resolved output bounds do not overlap.

    ``window_span`` follows the trajectory plotter's convention: a request with
    start 63000 and stop 63100 has a span of 100 frames. ``resolve_bounds`` can
    account for plot padding or expansion at behavior-state turn boundaries.
    """
    return sample_non_overlapping_frame_windows_from_domains(
        frame_domains=[(domain_start, domain_stop)],
        window_span=window_span,
        count=count,
        rng=rng,
        resolve_bounds=resolve_bounds,
    )


def sample_non_overlapping_frame_windows_from_domains(
    *,
    frame_domains: list[tuple[int, int]],
    window_span: int,
    count: int,
    rng: np.random.Generator,
    resolve_bounds: Callable[[int, int], tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Sample windows wholly contained in one of several inclusive domains."""
    window_span = int(window_span)
    count = int(count)

    if count < 1:
        raise ValueError("random sample count must be at least 1")
    if window_span < 1:
        raise ValueError("random window span must be at least 1 frame")

    if resolve_bounds is None:
        resolve_bounds = lambda start, stop: (start, stop)

    candidates = []
    normalized_domains = [
        (int(domain_start), int(domain_stop))
        for domain_start, domain_stop in frame_domains
        if int(domain_stop) >= int(domain_start)
    ]
    for domain_start, domain_stop in normalized_domains:
        max_start = domain_stop - window_span
        for start in range(domain_start, max_start + 1):
            stop = start + window_span
            resolved_start, resolved_stop = resolve_bounds(start, stop)
            resolved_start, resolved_stop = int(resolved_start), int(resolved_stop)
            # Plot padding and turn-boundary expansion are part of the output.
            # Reject a candidate if either would push it outside its training.
            if resolved_start < domain_start or resolved_stop > domain_stop:
                continue
            candidates.append((start, stop, resolved_start, resolved_stop))

    if not candidates:
        domain_text = ", ".join(
            f"{start}-{stop}" for start, stop in normalized_domains
        ) or "none"
        raise ValueError(
            f"no {window_span}-frame windows fit wholly within frame domains "
            f"{domain_text} after plot-boundary expansion"
        )

    def greedy(candidate_order):
        selected = []
        for idx in candidate_order:
            candidate = candidates[int(idx)]
            resolved_start, resolved_stop = candidate[2], candidate[3]
            if all(
                resolved_stop < other[2] or resolved_start > other[3]
                for other in selected
            ):
                selected.append(candidate)
                if len(selected) == count:
                    return selected
        return selected

    # A randomized greedy pass gives arbitrary frame alignment in the common
    # case. Retry because an expanded turn at one candidate can be much wider
    # than its neighbors and make an otherwise feasible random pass fail.
    selected = []
    for _ in range(32):
        selected = greedy(rng.permutation(len(candidates)))
        if len(selected) == count:
            break

    if len(selected) < count:
        # Earliest-finish interval scheduling determines the maximum feasible
        # set. A seeded subset of it is a deterministic fallback.
        ordered = sorted(
            range(len(candidates)),
            key=lambda idx: (candidates[idx][3], candidates[idx][2]),
        )
        maximal = []
        last_stop = None
        for idx in ordered:
            candidate = candidates[idx]
            if last_stop is None or candidate[2] > last_stop:
                maximal.append(candidate)
                last_stop = candidate[3]

        if len(maximal) < count:
            domain_text = ", ".join(
                f"{start}-{stop}" for start, stop in normalized_domains
            )
            raise ValueError(
                f"cannot fit {count} non-overlapping {window_span}-frame windows "
                f"in frame domains {domain_text}; at most "
                f"{len(maximal)} fit after plot-boundary expansion"
            )
        chosen = rng.choice(len(maximal), size=count, replace=False)
        selected = [maximal[int(idx)] for idx in chosen]

    return sorted((candidate[0], candidate[1]) for candidate in selected)
