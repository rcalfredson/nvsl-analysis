"""Dependency-light chamber-wise heatmap color rendering."""

import cv2
import numpy as np


def render_heatmap(mp, xform=np.log, xy_seps=None, thres=None,
                   colormap=cv2.COLORMAP_JET):
    """Transform and normalize each chamber independently, then apply a color map.

    Counts, smoothing, and any baseline correction belong to the caller. This
    preserves the renderer historically exposed by ``util.heatmap``.
    """
    xy_seps = xy_seps or ([], [])
    h, w = mp.shape[:2]
    image = np.full((h, w, 3), 255, np.uint8)
    x1 = 0
    for x2 in xy_seps[0] + [w]:
        y1 = 0
        for y2 in xy_seps[1] + [h]:
            values = mp[y1:y2, x1:x2]
            if thres is not None:
                size = values.size
                total = np.sum(values) - (np.amin(values) == 1) * size
                threshold = thres * total / float(size)
                values = cv2.threshold(values, threshold, threshold, cv2.THRESH_TRUNC)[1]
            if xform is not None:
                values = xform(values)
            colors = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image[y1:y2, x1:x2] = cv2.applyColorMap(colors, colormap)
            y1 = y2
        x1 = x2
    return image
