# src/plotting/bin_edges.py
from __future__ import annotations
from typing import Union
import numpy as np

Edges = Union[np.ndarray, list[np.ndarray]]  # flat or grouped


def is_grouped_edges(x) -> bool:
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
    Accepts:
      - flat edges: 1D numeric array
      - grouped edges:
          * list/tuple of arrays
          * 1D object array of arrays/lists
          * numeric 2D array (rows are groups)
    Returns:
      - flat: 1D float ndarray
      - grouped: list of 1D float ndarrays
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
