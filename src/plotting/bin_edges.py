# src/plotting/bin_edges.py
from __future__ import annotations
from typing import Union
import numpy as np

Edges = Union[np.ndarray, list[np.ndarray]]  # flat or grouped


def is_grouped_edges(x) -> bool:
    """
    Return True if `x` encodes "grouped" bin edges rather than a single flat edge vector.

    In this codebase, "grouped edges" means one panel's bins are composed of multiple
    disjoint edge arrays (e.g., to represent gaps or heterogeneous binning schemes)
    that should not be bridged by step plots and should be treated as separate
    contiguous segments.

    Accepted grouped representations:
      - list/tuple of 1D arrays: [edges_group0, edges_group1, ...]
      - numpy 1D object array containing arrays/lists
      - numpy numeric 2D array where each row is a group's 1D edge vector

    Flat (non-grouped) representations:
      - numpy 1D numeric array of edges [e0, e1, ..., eN]

    Notes:
      - This is a *shape/type* check, not a semantic validation of monotonicity.
      - Empty inputs are treated as non-grouped.
    """
    if isinstance(x, (list, tuple)):
        return len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray))

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        if x.dtype == object:
            first = x.flat[0]
            return isinstance(first, (list, tuple, np.ndarray))
        # numeric 2D -> treat rows as groups
        if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= 2:
            return True
    return False


def normalize_panel_edges(e_item) -> Edges:
    """
    Normalize a single "panel's" bin-edge specification into a canonical format.

    This function exists because different exporters/plotters may store edges in
    slightly different shapes (lists, object arrays, 2D arrays). Plotting and
    statistics code wants one consistent representation.

    Parameters
    ----------
    e_item
        One panel's bin edges, in any of these forms:
          Flat edges:
            - 1D numeric array-like of length (B + 1)
          Grouped edges (multiple disjoint edge arrays):
            - list/tuple of 1D array-likes
            - 1D object ndarray of 1D array-likes
            - 2D numeric ndarray where rows are groups

    Returns
    -------
    Edges
        If the input is flat: a 1D float ndarray of length (B + 1).
        If the input is grouped: a list of 1D float ndarrays, one per group.

    Intended semantics
    ------------------
    - Each group represents a *contiguous* run of bins. The caller should generally
      avoid drawing plot elements that "bridge" between groups.
    - This function does not enforce monotonicity or non-overlap; it only normalizes
      representation.
    """
    if isinstance(e_item, np.ndarray) and e_item.ndim == 2 and e_item.dtype != object:
        return [
            np.asarray(e_item[i, :], dtype=float).ravel()
            for i in range(e_item.shape[0])
        ]

    if is_grouped_edges(e_item):
        groups = (
            list(e_item)
            if isinstance(e_item, np.ndarray) and e_item.dtype == object
            else list(e_item)
        )
        return [np.asarray(g, dtype=float).ravel() for g in groups]

    return np.asarray(e_item, dtype=float).ravel()


def geom_from_edges(e_item: Edges):
    """
    Compute plotting-friendly bin geometry from a bin-edge specification.

    Given either a flat edge vector or a grouped edge specification, this function
    returns per-bin widths and centers, plus human-friendly bin ranges and global
    x-limits.

    This is used primarily for:
      - grouped/dodged bar layouts (bar width, x positions)
      - x-tick labeling (bin range strings)
      - setting consistent x-limits for plots

    Parameters
    ----------
    e_item : Edges
        Either:
          - flat edges: 1D array of length (B + 1)
          - grouped edges: list of 1D arrays (each length >= 2), where each group
            contributes its own contiguous set of bins. Groups are concatenated in
            order for widths/centers/ranges.

    Returns
    -------
    widths : np.ndarray, shape (B,)
        Bin widths (diff of edges). For grouped edges, this is the concatenation
        of widths from each group.
    centers : np.ndarray, shape (B,)
        Bin centers (midpoint of each bin). For grouped edges, concatenated.
    ranges : list[tuple[float, float]], length B
        List of (left_edge, right_edge) for each bin, in the same order as widths.
        Useful for generating labels like "10-12".
    B : int
        Number of bins (len(widths) == len(centers) == len(ranges)).
    x0 : float
        Global minimum x for the panel (left edge of first group / first edge).
    x1 : float
        Global maximum x for the panel (right edge of last group / last edge).

    Notes
    -----
    - For grouped edges, `x0` and `x1` reflect the extreme edges across all groups.
    - The returned arrays do not preserve per-group boundaries; callers that need
      group boundaries should also keep the grouped `e_item` itself.
    - This function assumes each provided edge array is ordered; it does not validate.
    """
    e_item = normalize_panel_edges(e_item)
    if isinstance(e_item, list):
        widths_list, centers_list, ranges = [], [], []
        x0 = float(e_item[0][0])
        x1 = float(e_item[-1][-1])
        for g in e_item:
            w = np.diff(g)
            c = 0.5 * (g[:-1] + g[1:])
            widths_list.append(w)
            centers_list.append(c)
            ranges.extend([(float(a), float(b)) for a, b in zip(g[:-1], g[1:])])
        widths = (
            np.concatenate(widths_list, axis=0)
            if widths_list
            else np.zeros((0,), float)
        )
        centers = (
            np.concatenate(centers_list, axis=0)
            if centers_list
            else np.zeros((0,), float)
        )
        return widths, centers, ranges, int(centers.size), x0, x1
    else:
        e = np.asarray(e_item, dtype=float)
        widths = np.diff(e)
        centers = 0.5 * (e[:-1] + e[1:])
        ranges = [(float(a), float(b)) for a, b in zip(e[:-1], e[1:])]
        return widths, centers, ranges, int(centers.size), float(e[0]), float(e[-1])
