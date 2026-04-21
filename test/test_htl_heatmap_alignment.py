import numpy as np
import pytest

pytest.importorskip("cv2")

from src.analysis.video_analysis import VideoAnalysis
from src.utils.common import CT, Xformer


class _FakeTrajectory:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


def _make_htl_va(abs_fly: int) -> VideoAnalysis:
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.ct = CT.htl
    va.trxf = (abs_fly,)
    va.xf = Xformer(
        {"fctr": 1.0, "x": 0.0, "y": 0.0},
        CT.htl,
        np.zeros((720, 720), dtype=np.uint8),
        True,
    )
    return va


def test_htl_heatmap_bounds_use_padded_floor_geometry():
    va = _make_htl_va(abs_fly=0)

    xym, xyM = va._heatmapBounds(0)

    np.testing.assert_allclose(xym, (-15.0, -15.0))
    np.testing.assert_allclose(xyM, (95.0, 143.0))


def test_htl_heatmap_coords_recover_canonical_local_points_across_grid_positions():
    canonical_x = np.array([10.0, 40.0, 70.0, 15.0, 65.0])
    canonical_y = np.array([10.0, 64.0, 120.0, 100.0, 25.0])

    for abs_fly in (0, 9, 17, 19):
        va = _make_htl_va(abs_fly=abs_fly)

        frame_pts = [va.xf.t2f(x, y, f=abs_fly) for x, y in zip(canonical_x, canonical_y)]
        trx = _FakeTrajectory(
            x=[pt[0] for pt in frame_pts],
            y=[pt[1] for pt in frame_pts],
        )

        recovered_x, recovered_y = va._heatmapCoords(trx, 0, 0, len(canonical_x))

        np.testing.assert_allclose(recovered_x, canonical_x, atol=1e-6)
        np.testing.assert_allclose(recovered_y, canonical_y, atol=1e-6)

        cx_frame, cy_frame = va.xf.t2f(*CT.htl.center(), f=abs_fly)
        cx_local, cy_local = va.xf.f2t(cx_frame, cy_frame, f=abs_fly)
        np.testing.assert_allclose((cx_local, cy_local), CT.htl.center(), atol=1e-6)
