import math
from math import sin, cos
import os
import random
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.utils.common import writeImage
from src.utils.constants import CONTACT_BUFFER_OFFSETS


class EventChainPlotter:
    def __init__(self, trj, va, y_bounds=None, x=None, y=None, image_format="png"):
        self.trj = trj
        self.va = va
        self.y_bounds = y_bounds
        self.x = np.array(trj.x) if x is None else x
        self.y = np.array(trj.y) if y is None else y
        self.image_format = image_format

        # track which between-reward intervals have already been plotted
        # key: (trn_index, bucket_index) -> set of (start_reward, end_reward)
        self._used_between_reward_pairs = {}
        self._used_reward_return_episodes = {}
        self._used_return_prob_episodes = {}

    @staticmethod
    def _clamp_point_to_floor(x, y, top_left, bottom_right):
        x = max(min(float(x), float(bottom_right[0])), float(top_left[0]))
        y_min = min(float(top_left[1]), float(bottom_right[1]))
        y_max = max(float(top_left[1]), float(bottom_right[1]))
        y = max(min(float(y), y_max), y_min)
        return x, y

    @staticmethod
    def _rotate_local_point(dx, dy, angle_deg):
        angle = np.deg2rad(float(angle_deg))
        ca = float(np.cos(angle))
        sa = float(np.sin(angle))
        return (dx * ca - dy * sa, dx * sa + dy * ca)

    @classmethod
    def _draw_fly_icon(
        cls,
        ax,
        x,
        y,
        *,
        angle_deg=25.0,
        body_len=10.0,
        body_w=4.8,
        wing_len=7.0,
        wing_w=3.8,
        body_fc="#202020",
        wing_fc="#d9ecff",
    ):
        x = float(x)
        y = float(y)
        body_len = float(body_len)
        body_w = float(body_w)
        wing_len = float(wing_len)
        wing_w = float(wing_w)

        thorax_len = body_len * 0.42
        thorax_w = body_w * 1.08
        abdomen_len = body_len * 0.72
        abdomen_w = body_w * 0.88
        head_r = body_w * 0.30

        thorax_offset = cls._rotate_local_point(0.0, 0.0, angle_deg)
        abdomen_offset = cls._rotate_local_point(0.0, -body_len * 0.32, angle_deg)
        head_offset = cls._rotate_local_point(0.0, body_len * 0.34, angle_deg)
        wing_base_left = cls._rotate_local_point(
            -body_w * 0.20, body_len * 0.14, angle_deg
        )
        wing_base_right = cls._rotate_local_point(
            body_w * 0.20, body_len * 0.14, angle_deg
        )

        def _wing_patch(base_x, base_y, side_sign):
            wing_points = [
                (0.0, 0.0),
                (side_sign * wing_w * 0.52, -wing_len * 0.04),
                (side_sign * wing_w * 0.78, -wing_len * 0.44),
                (side_sign * wing_w * 0.34, -wing_len * 0.98),
                (0.0, -wing_len * 0.84),
                (-side_sign * wing_w * 0.14, -wing_len * 0.36),
            ]
            spread_deg = 16.0 * side_sign
            pts = []
            for dx, dy in wing_points:
                rx, ry = cls._rotate_local_point(dx, dy, angle_deg + spread_deg)
                pts.append((base_x + rx, base_y + ry))
            return patches.Polygon(
                pts,
                closed=True,
                facecolor=wing_fc,
                edgecolor="#7fa9c9",
                linewidth=0.55,
                alpha=0.55,
                zorder=11,
                joinstyle="round",
            )

        wing_left = _wing_patch(x + wing_base_left[0], y + wing_base_left[1], -1.0)
        wing_right = _wing_patch(x + wing_base_right[0], y + wing_base_right[1], 1.0)
        abdomen = patches.Ellipse(
            (x + abdomen_offset[0], y + abdomen_offset[1]),
            width=abdomen_w,
            height=abdomen_len,
            angle=angle_deg,
            facecolor="#2a211f",
            edgecolor="#f2e4cf",
            linewidth=0.5,
            zorder=8,
        )
        thorax = patches.Ellipse(
            (x + thorax_offset[0], y + thorax_offset[1]),
            width=thorax_w,
            height=thorax_len,
            angle=angle_deg,
            facecolor=body_fc,
            edgecolor="#f2e4cf",
            linewidth=0.55,
            zorder=9,
        )
        head = patches.Circle(
            (x + head_offset[0], y + head_offset[1]),
            radius=head_r,
            facecolor="#1a1a1a",
            edgecolor="#f2e4cf",
            linewidth=0.45,
            zorder=10,
        )

        leg_color = "#3c312d"
        for side in (-1.0, 1.0):
            x_anchor = side * body_w * 0.34
            leg_specs = (
                (-body_len * 0.12, side * body_w * 0.90, -body_len * 0.28),
                (0.00, side * body_w * 1.05, -body_len * 0.02),
                (body_len * 0.16, side * body_w * 0.96, body_len * 0.18),
            )
            for y_anchor, x_tip, y_tip in leg_specs:
                p0 = cls._rotate_local_point(x_anchor, y_anchor, angle_deg)
                p1 = cls._rotate_local_point(x_tip, y_tip, angle_deg)
                ax.plot(
                    [x + p0[0], x + p1[0]],
                    [y + p0[1], y + p1[1]],
                    color=leg_color,
                    linewidth=0.8,
                    alpha=0.95,
                    zorder=6,
                    solid_capstyle="round",
                )

        for side in (-1.0, 1.0):
            p0 = cls._rotate_local_point(side * head_r * 0.45, body_len * 0.48, angle_deg)
            p1 = cls._rotate_local_point(side * head_r * 1.10, body_len * 0.70, angle_deg)
            ax.plot(
                [x + p0[0], x + p1[0]],
                [y + p0[1], y + p1[1]],
                color="#282828",
                linewidth=0.55,
                alpha=0.9,
                zorder=10,
                solid_capstyle="round",
            )

        ax.add_patch(abdomen)
        ax.add_patch(thorax)
        ax.add_patch(head)
        ax.add_patch(wing_left)
        ax.add_patch(wing_right)

    @staticmethod
    def _between_reward_maxdist_geometry(
        x,
        y,
        start_reward,
        end_reward,
        reward_circle,
        *,
        exclude_reward_endpoints=False,
    ):
        if reward_circle is None:
            return None

        sr = int(start_reward)
        er = int(end_reward)
        s_seg = sr + 1 if exclude_reward_endpoints else sr
        e_seg = er - 1 if exclude_reward_endpoints else er
        if e_seg <= s_seg:
            return None

        xs = np.asarray(x[s_seg:e_seg], dtype=float)
        ys = np.asarray(y[s_seg:e_seg], dtype=float)
        if xs.size == 0 or ys.size == 0:
            return None

        idx = np.arange(s_seg, e_seg, dtype=int)
        fin = np.isfinite(xs) & np.isfinite(ys)
        if not fin.any():
            return None

        xs = xs[fin]
        ys = ys[fin]
        idx = idx[fin]

        cx, cy, r = reward_circle
        d = np.hypot(xs - float(cx), ys - float(cy))
        if d.size == 0 or not np.isfinite(d).any():
            return None

        try:
            k = int(np.nanargmax(d))
        except Exception:
            return None

        if k < 0 or k >= d.size:
            return None

        return {
            "frame": int(idx[k]),
            "x": float(xs[k]),
            "y": float(ys[k]),
            "d_px": float(d[k]),
            "cx": float(cx),
            "cy": float(cy),
            "r_px": float(r),
        }

    def _overlay_maxdist_schematic(
        self,
        ax,
        *,
        reward_circle,
        start_reward,
        end_reward,
        top_left,
        bottom_right,
        px_per_mm,
        show_label=True,
    ):
        geom = self._between_reward_maxdist_geometry(
            self.x,
            self.y,
            start_reward,
            end_reward,
            reward_circle,
            exclude_reward_endpoints=False,
        )
        if geom is None:
            return None

        cx = geom["cx"]
        cy = geom["cy"]
        x_max = geom["x"]
        y_max = geom["y"]

        ax.plot(
            [cx, x_max],
            [cy, y_max],
            linestyle=(0, (5, 3)),
            color="#c45a1c",
            linewidth=2.0,
            zorder=5,
            label="Dmax",
        )
        ax.scatter(
            [x_max],
            [y_max],
            s=44,
            color="#c45a1c",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="Max-distance point",
        )

        fly_angle = np.degrees(np.arctan2(y_max - cy, x_max - cx)) - 90.0
        self._draw_fly_icon(
            ax,
            x_max,
            y_max,
            angle_deg=float(fly_angle),
            body_len=10.0,
            body_w=5.2,
            wing_len=7.4,
            wing_w=4.2,
        )

        if show_label:
            midx = 0.5 * (cx + x_max)
            midy = 0.5 * (cy + y_max)
            dx = x_max - cx
            dy = y_max - cy
            norm = float(np.hypot(dx, dy))
            if norm > 0:
                off_x = -dy / norm * 12.0
                off_y = dx / norm * 12.0
            else:
                off_x = 0.0
                off_y = -12.0

            tx, ty = self._clamp_point_to_floor(
                midx + off_x, midy + off_y, top_left, bottom_right
            )
            if px_per_mm is not None and np.isfinite(geom["d_px"]):
                dmm = float(geom["d_px"]) / float(px_per_mm)
                label = f"dMax = {dmm:.2f} mm"
            else:
                label = "dMax"

            ax.text(
                tx,
                ty,
                label,
                fontsize=9,
                color="#7a3100",
                ha="center",
                va="center",
                zorder=9,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#c45a1c",
                    linewidth=0.9,
                    alpha=0.92,
                ),
            )

        return geom

    @staticmethod
    def _between_reward_return_leg_geometry(
        x,
        y,
        start_reward,
        end_reward,
        reward_circle,
    ):
        geom = EventChainPlotter._between_reward_maxdist_geometry(
            x,
            y,
            start_reward,
            end_reward,
            reward_circle,
            exclude_reward_endpoints=False,
        )
        if geom is None:
            return None

        sr = max(0, int(start_reward))
        er = max(sr, int(end_reward))
        max_frame = int(geom["frame"])
        if max_frame > er:
            return None

        idx = np.arange(max_frame, er + 1, dtype=int)
        if idx.size < 2:
            return None

        xs = np.asarray(x[idx], dtype=float)
        ys = np.asarray(y[idx], dtype=float)
        fin = np.isfinite(xs) & np.isfinite(ys)
        if fin.sum() < 2:
            return None

        pair_fin = fin[:-1] & fin[1:]
        if not pair_fin.any():
            return None
        step_dx = np.diff(xs)
        step_dy = np.diff(ys)
        d_px = float(np.sum(np.hypot(step_dx[pair_fin], step_dy[pair_fin])))
        if not np.isfinite(d_px):
            return None

        valid_last = int(np.flatnonzero(fin)[-1])

        return {
            **geom,
            "return_x": xs,
            "return_y": ys,
            "return_idx": idx,
            "return_d_px": d_px,
            "end_x": float(xs[valid_last]),
            "end_y": float(ys[valid_last]),
        }

    def _overlay_return_leg_dist_schematic(
        self,
        ax,
        *,
        reward_circle,
        start_reward,
        end_reward,
        top_left,
        bottom_right,
        px_per_mm,
        show_label=True,
    ):
        geom = self._between_reward_return_leg_geometry(
            self.x,
            self.y,
            start_reward,
            end_reward,
            reward_circle,
        )
        if geom is None:
            return None

        xs = np.asarray(geom["return_x"], dtype=float)
        ys = np.asarray(geom["return_y"], dtype=float)
        fin = np.isfinite(xs) & np.isfinite(ys)
        if xs.size < 2 or ys.size < 2 or fin.sum() < 2:
            return None
        valid_idx = np.flatnonzero(fin)
        xs_valid = xs[valid_idx]
        ys_valid = ys[valid_idx]

        ax.plot(
            xs,
            ys,
            color="#d26a1b",
            linewidth=2.7,
            alpha=0.98,
            zorder=5,
            solid_capstyle="round",
            label="Return leg",
        )
        ax.scatter(
            [geom["x"]],
            [geom["y"]],
            s=44,
            color="#c45a1c",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="dMax point",
        )

        if xs_valid.size >= 3:
            step = max(1, int((xs_valid.size - 1) / 3))
            for i in range(0, xs_valid.size - 1, step):
                j = min(xs_valid.size - 1, i + 1)
                if j <= i:
                    continue
                ax.annotate(
                    "",
                    xy=(xs_valid[j], ys_valid[j]),
                    xytext=(xs_valid[i], ys_valid[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#d26a1b",
                        lw=1.1,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        alpha=0.92,
                    ),
                    zorder=5,
                )

        k0 = 0
        k1 = min(len(xs_valid) - 1, 1)
        tan_dx = float(xs_valid[k1] - xs_valid[k0])
        tan_dy = float(ys_valid[k1] - ys_valid[k0])
        if np.hypot(tan_dx, tan_dy) > 1e-6:
            fly_angle = np.degrees(np.arctan2(tan_dy, tan_dx)) - 90.0
        else:
            fly_angle = np.degrees(
                np.arctan2(geom["end_y"] - ys_valid[0], geom["end_x"] - xs_valid[0])
            ) - 90.0
        self._draw_fly_icon(ax, xs_valid[0], ys_valid[0], angle_deg=float(fly_angle))

        if show_label:
            mid_i = min(len(xs_valid) - 1, max(0, int(len(xs_valid) * 0.5)))
            midx = float(xs_valid[mid_i])
            midy = float(ys_valid[mid_i])
            dx = float(geom["end_x"] - xs_valid[0])
            dy = float(geom["end_y"] - ys_valid[0])
            norm = float(np.hypot(dx, dy))
            if norm > 0:
                off_x = -dy / norm * 12.0
                off_y = dx / norm * 12.0
            else:
                off_x = 0.0
                off_y = -12.0

            tx, ty = self._clamp_point_to_floor(
                midx + off_x, midy + off_y, top_left, bottom_right
            )
            if px_per_mm is not None and np.isfinite(geom["return_d_px"]):
                dmm = float(geom["return_d_px"]) / float(px_per_mm)
                label = f"Return leg = {dmm:.2f} mm"
            else:
                label = "Return leg"

            ax.text(
                tx,
                ty,
                label,
                fontsize=9,
                color="#7a3100",
                ha="center",
                va="center",
                zorder=9,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#d26a1b",
                    linewidth=0.9,
                    alpha=0.92,
                ),
            )

        return geom

    @staticmethod
    def _catmull_rom_chain(points, *, samples_per_seg=24):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            return pts

        out = []
        n_pts = pts.shape[0]
        for i in range(n_pts - 1):
            p0 = pts[max(0, i - 1)]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[min(n_pts - 1, i + 2)]
            ts = np.linspace(0.0, 1.0, int(samples_per_seg), endpoint=(i == n_pts - 2))
            for t in ts:
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out.append(point)
        return np.asarray(out, dtype=float)

    @staticmethod
    def _draw_reward_center_marker(
        ax,
        cx,
        cy,
        *,
        ring_color="#143642",
        label=None,
        zorder=7,
        dot_radius=1.9,
        show_halo=True,
    ):
        if show_halo:
            halo = patches.Circle(
                (float(cx), float(cy)),
                radius=max(2.6, float(dot_radius) + 1.2),
                facecolor="white",
                edgecolor="none",
                alpha=0.95,
                zorder=zorder,
            )
            ax.add_patch(halo)
        inner = patches.Circle(
            (float(cx), float(cy)),
            radius=float(dot_radius),
            facecolor=ring_color,
            edgecolor="none",
            zorder=zorder + (1 if show_halo else 0),
        )
        ax.add_patch(inner)
        ax.plot(
            [float(cx) - 7.0, float(cx) + 7.0],
            [float(cy), float(cy)],
            color=ring_color,
            linewidth=0.85,
            alpha=0.55,
            zorder=zorder - 1,
        )
        ax.plot(
            [float(cx), float(cx)],
            [float(cy) - 7.0, float(cy) + 7.0],
            color=ring_color,
            linewidth=0.85,
            alpha=0.55,
            zorder=zorder - 1,
        )
        if label:
            ax.text(
                float(cx),
                float(cy) - 11.0,
                label,
                fontsize=8,
                color=ring_color,
                ha="center",
                va="top",
                zorder=zorder + 2,
            )

    @staticmethod
    def _synthetic_maxdist_path(
        reward_circle, *, variant=0, rng=None, excursion_scale=1.0
    ):
        if reward_circle is None:
            return None

        rng = rng or random.Random(variant)
        cx, cy, r = (float(v) for v in reward_circle)
        excursion_scale = float(excursion_scale)
        theta0 = np.deg2rad(rng.uniform(105.0, 170.0))
        drift_sign = rng.choice((-1.0, 1.0))
        theta1 = theta0 + drift_sign * np.deg2rad(rng.uniform(95.0, 175.0))

        start_rad = rng.uniform(0.54, 0.66)
        end_rad = rng.uniform(0.56, 0.70)
        peak_target = max(
            2.10,
            (2.20 + rng.uniform(-0.20, 0.60)) * excursion_scale,
        )

        current_theta = float(theta0)
        current_rad = float(start_rad)
        control_polar = [(current_theta, current_rad)]

        state = "run_out"
        loop_budget = rng.randint(1, 2)
        n_mid_states = rng.randint(3, 5)

        def _append_polar(theta, radial):
            control_polar.append(
                (
                    float(theta),
                    float(np.clip(radial, 0.50, 3.90)),
                )
            )

        for i in range(n_mid_states):
            if state == "run_out":
                current_theta += drift_sign * np.deg2rad(rng.uniform(18.0, 34.0))
                current_rad += rng.uniform(0.34, 0.62) * excursion_scale
                _append_polar(current_theta, current_rad)
                state = "loop" if loop_budget > 0 and rng.random() < 0.70 else "bend"
                continue

            if state == "bend":
                current_theta += drift_sign * np.deg2rad(rng.uniform(28.0, 58.0))
                current_rad += (
                    rng.uniform(-0.10, 0.18) * excursion_scale
                    + (0.14 * excursion_scale if current_rad < peak_target * 0.82 else 0.0)
                )
                _append_polar(current_theta, current_rad)
                if loop_budget > 0 and i < (n_mid_states - 1):
                    state = "loop" if rng.random() < 0.60 else "bend"
                else:
                    state = "bend"
                continue

            loop_dir = drift_sign * (-1.0 if rng.random() < 0.58 else 1.0)
            loop_budget -= 1
            for ang_deg, rad_delta in (
                (rng.uniform(24.0, 38.0), rng.uniform(0.10, 0.22)),
                (rng.uniform(36.0, 52.0), rng.uniform(-0.08, 0.10)),
                (rng.uniform(28.0, 42.0), rng.uniform(-0.05, 0.14)),
            ):
                current_theta += loop_dir * np.deg2rad(ang_deg)
                current_rad += rad_delta * excursion_scale
                _append_polar(current_theta, current_rad)
            state = "bend"

        max_rad_now = max(rad for _, rad in control_polar)
        if max_rad_now < peak_target:
            current_theta += drift_sign * np.deg2rad(rng.uniform(18.0, 32.0))
            current_rad = peak_target
            _append_polar(current_theta, current_rad)

        pre_return_theta = theta1 - drift_sign * np.deg2rad(rng.uniform(12.0, 30.0))
        pre_return_rad = max(
            0.95,
            min(
                max(current_rad, 1.10) - rng.uniform(0.45, 0.80) * excursion_scale,
                1.60,
            ),
        )
        _append_polar(pre_return_theta, pre_return_rad)
        _append_polar(theta1, end_rad)

        control_xy = np.asarray(
            [
                [cx + (r * rad) * np.cos(theta), cy + (r * rad) * np.sin(theta)]
                for theta, rad in control_polar
            ],
            dtype=float,
        )
        path_xy = EventChainPlotter._catmull_rom_chain(
            control_xy,
            samples_per_seg=max(18, int(22 * excursion_scale)),
        )
        if path_xy.shape[0] < 6:
            path_xy = control_xy

        xs = np.asarray(path_xy[:, 0], dtype=float)
        ys = np.asarray(path_xy[:, 1], dtype=float)

        # Preserve explicit start/end positions inside the reward circle.
        xs[0] = cx + (r * start_rad) * np.cos(theta0)
        ys[0] = cy + (r * start_rad) * np.sin(theta0)
        xs[-1] = cx + (r * end_rad) * np.cos(theta1)
        ys[-1] = cy + (r * end_rad) * np.sin(theta1)

        # Light smoothing keeps the path fluid without creating flat-radius arcs.
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        kernel /= kernel.sum()
        for _ in range(2):
            xs_pad = np.pad(xs, (2, 2), mode="edge")
            ys_pad = np.pad(ys, (2, 2), mode="edge")
            xs = np.convolve(xs_pad, kernel, mode="valid")
            ys = np.convolve(ys_pad, kernel, mode="valid")
            xs[0] = cx + (r * start_rad) * np.cos(theta0)
            ys[0] = cy + (r * start_rad) * np.sin(theta0)
            xs[-1] = cx + (r * end_rad) * np.cos(theta1)
            ys[-1] = cy + (r * end_rad) * np.sin(theta1)

        # Confirm there is a real excursion outside the reward circle.
        if not np.any(np.hypot(xs - cx, ys - cy) > (1.05 * r)):
            k_mid = int(0.56 * (len(xs) - 1))
            ang_mid = np.arctan2(ys[k_mid] - cy, xs[k_mid] - cx)
            force_rad = r * max(2.20, 2.75 * excursion_scale)
            xs[k_mid] = cx + force_rad * np.cos(ang_mid)
            ys[k_mid] = cy + force_rad * np.sin(ang_mid)

        d = np.hypot(xs - cx, ys - cy)
        k = int(np.argmax(d))
        return {
            "x": xs,
            "y": ys,
            "start_x": float(xs[0]),
            "start_y": float(ys[0]),
            "end_x": float(xs[-1]),
            "end_y": float(ys[-1]),
            "max_x": float(xs[k]),
            "max_y": float(ys[k]),
            "cx": cx,
            "cy": cy,
            "r_px": r,
            "k": k,
        }

    def _overlay_synthetic_maxdist_schematic(
        self,
        ax,
        *,
        reward_circle,
        top_left,
        bottom_right,
        rng=None,
        excursion_scale=1.0,
        variant=0,
    ):
        synth = self._synthetic_maxdist_path(
            reward_circle,
            variant=variant,
            rng=rng,
            excursion_scale=excursion_scale,
        )
        if synth is None:
            return None

        xs = np.asarray(synth["x"], dtype=float)
        ys = np.asarray(synth["y"], dtype=float)
        ax.plot(
            xs,
            ys,
            color="#1f5a7a",
            linewidth=2.1,
            zorder=4,
            label="Synthetic path",
        )

        step = max(1, int(len(xs) / 5))
        for i in range(step, len(xs) - 1, step):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1f5a7a",
                    lw=1.0,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    alpha=0.85,
                ),
                zorder=4,
            )

        ax.scatter(
            [synth["start_x"]],
            [synth["start_y"]],
            s=30,
            color="#2c9b45",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label="Segment start",
        )
        ax.scatter(
            [synth["end_x"]],
            [synth["end_y"]],
            s=30,
            color="#b33c2f",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label="Segment end",
        )

        cx = synth["cx"]
        cy = synth["cy"]
        x_max = synth["max_x"]
        y_max = synth["max_y"]
        k = int(synth["k"])
        ax.plot(
            [cx, x_max],
            [cy, y_max],
            linestyle=(0, (5, 3)),
            color="#c45a1c",
            linewidth=2.1,
            zorder=5,
            label="Dmax",
        )
        ax.scatter(
            [x_max],
            [y_max],
            s=44,
            color="#c45a1c",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="Max-distance point",
        )

        k0 = max(0, k - 1)
        k1 = min(len(xs) - 1, k + 1)
        tan_dx = float(xs[k1] - xs[k0])
        tan_dy = float(ys[k1] - ys[k0])
        if np.hypot(tan_dx, tan_dy) > 1e-6:
            fly_angle = np.degrees(np.arctan2(tan_dy, tan_dx)) - 90.0
        else:
            fly_angle = np.degrees(np.arctan2(y_max - cy, x_max - cx)) - 90.0
        self._draw_fly_icon(ax, x_max, y_max, angle_deg=float(fly_angle))

        midx = 0.5 * (cx + x_max)
        midy = 0.5 * (cy + y_max)
        dx = x_max - cx
        dy = y_max - cy
        norm = float(np.hypot(dx, dy))
        if norm > 0:
            off_x = -dy / norm * 12.0
            off_y = dx / norm * 12.0
        else:
            off_x = 0.0
            off_y = -12.0
        tx, ty = self._clamp_point_to_floor(
            midx + off_x, midy + off_y, top_left, bottom_right
        )
        ax.text(
            tx,
            ty,
            "Dmax",
            fontsize=9,
            color="#7a3100",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#c45a1c",
                linewidth=0.9,
                alpha=0.92,
            ),
        )

        return synth

    def _overlay_synthetic_return_leg_dist_schematic(
        self,
        ax,
        *,
        reward_circle,
        top_left,
        bottom_right,
        rng=None,
        excursion_scale=1.0,
        variant=0,
    ):
        synth = self._synthetic_maxdist_path(
            reward_circle,
            variant=variant,
            rng=rng,
            excursion_scale=excursion_scale,
        )
        if synth is None:
            return None

        xs = np.asarray(synth["x"], dtype=float)
        ys = np.asarray(synth["y"], dtype=float)
        k = int(synth["k"])
        pre_x = xs[: k + 1]
        pre_y = ys[: k + 1]
        tail_x = xs[k:]
        tail_y = ys[k:]

        ax.plot(
            xs,
            ys,
            color="#1f5a7a",
            linewidth=2.1,
            zorder=4,
            label="Synthetic path",
        )
        if tail_x.size >= 2:
            ax.plot(
                tail_x,
                tail_y,
                color="#d26a1b",
                linewidth=2.8,
                alpha=0.98,
                zorder=5,
                solid_capstyle="round",
                label="Synthetic return leg",
            )

        if pre_x.size >= 2:
            step = max(1, int((pre_x.size - 1) / 4))
            for i in range(0, pre_x.size - 1, step):
                j = min(pre_x.size - 1, i + 1)
                if j <= i:
                    continue
                dx = pre_x[j] - pre_x[i]
                dy = pre_y[j] - pre_y[i]
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue
                ax.annotate(
                    "",
                    xy=(pre_x[j], pre_y[j]),
                    xytext=(pre_x[i], pre_y[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#1f5a7a",
                        lw=1.0,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        alpha=0.85,
                    ),
                    zorder=4,
                )

        if tail_x.size >= 3:
            tail_step = max(1, int((tail_x.size - 1) / 3))
            for i in range(0, tail_x.size - 1, tail_step):
                j = min(tail_x.size - 1, i + 1)
                if j <= i:
                    continue
                ax.annotate(
                    "",
                    xy=(tail_x[j], tail_y[j]),
                    xytext=(tail_x[i], tail_y[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#d26a1b",
                        lw=1.1,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        alpha=0.92,
                    ),
                    zorder=5,
                )

        ax.scatter(
            [synth["start_x"]],
            [synth["start_y"]],
            s=30,
            color="#2c9b45",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label="Segment start",
        )
        ax.scatter(
            [synth["end_x"]],
            [synth["end_y"]],
            s=30,
            color="#b33c2f",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label="Segment end",
        )
        ax.scatter(
            [synth["max_x"]],
            [synth["max_y"]],
            s=44,
            color="#c45a1c",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="dMax point",
        )

        k0 = int(k)
        k1 = min(len(xs) - 1, k + 1)
        tan_dx = float(xs[k1] - xs[k0])
        tan_dy = float(ys[k1] - ys[k0])
        if np.hypot(tan_dx, tan_dy) > 1e-6:
            fly_angle = np.degrees(np.arctan2(tan_dy, tan_dx)) - 90.0
        else:
            fly_angle = np.degrees(
                np.arctan2(
                    synth["end_y"] - synth["max_y"], synth["end_x"] - synth["max_x"]
                )
            ) - 90.0
        self._draw_fly_icon(
            ax, synth["max_x"], synth["max_y"], angle_deg=float(fly_angle)
        )

        if tail_x.size >= 2:
            mid_i = min(tail_x.size - 1, max(0, int(tail_x.size * 0.5)))
            midx = float(tail_x[mid_i])
            midy = float(tail_y[mid_i])
            dx = float(tail_x[-1] - tail_x[0])
            dy = float(tail_y[-1] - tail_y[0])
        else:
            midx = float(synth["max_x"])
            midy = float(synth["max_y"])
            dx = float(synth["end_x"] - synth["max_x"])
            dy = float(synth["end_y"] - synth["max_y"])

        norm = float(np.hypot(dx, dy))
        if norm > 0:
            off_x = -dy / norm * 12.0
            off_y = dx / norm * 12.0
        else:
            off_x = 0.0
            off_y = -12.0
        tx, ty = self._clamp_point_to_floor(
            midx + off_x, midy + off_y, top_left, bottom_right
        )
        ax.text(
            tx,
            ty,
            "Return leg",
            fontsize=9,
            color="#7a3100",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#d26a1b",
                linewidth=0.9,
                alpha=0.92,
            ),
        )

        return synth

    def _synthetic_commag_bucket_segments(
        self, reward_circle, *, variant=0, rng=None, agg_mode="vector_mean"
    ):
        if reward_circle is None:
            return None

        cx, cy, r = (float(v) for v in reward_circle)
        rng = rng or random.Random()
        colors = ("#4e79a7", "#e15759", "#59a14f")
        rot_deg = (int(variant) % 3 - 1) * 5.0
        agg_mode = str(agg_mode or "vector_mean").strip().lower()

        def _rotated_xy(radius, theta):
            dx, dy = self._rotate_local_point(
                float(radius * np.cos(theta)), float(radius * np.sin(theta)), rot_deg
            )
            return (cx + dx, cy + dy)

        def _build_path(control_pts):
            path = self._catmull_rom_chain(np.asarray(control_pts, dtype=float), samples_per_seg=28)
            if path.shape[0] < 3:
                path = np.asarray(control_pts, dtype=float)
            return np.asarray(path, dtype=float)

        def _segment_payload(path, color, idx):
            xs = np.asarray(path[:, 0], dtype=float)
            ys = np.asarray(path[:, 1], dtype=float)
            mx = float(np.nanmean(xs))
            my = float(np.nanmean(ys))
            return {
                "x": xs,
                "y": ys,
                "mx": mx,
                "my": my,
                "vx": float(mx - cx),
                "vy": float(my - cy),
                "mag": float(np.hypot(mx - cx, my - cy)),
                "color": color,
                "idx": idx + 1,
            }

        base_rot = rng.uniform(0.0, 2.0 * np.pi)

        segments = []
        if agg_mode == "mean_magnitude":
            mix_mode = rng.choices(
                ("mixed", "near_zero_emphasis", "lost_emphasis"),
                weights=(0.48, 0.28, 0.24),
                k=1,
            )[0]
            if mix_mode == "near_zero_emphasis":
                family_choices = ["near_zero", "near_zero", rng.choice(("short_skewed", "lost"))]
            elif mix_mode == "lost_emphasis":
                family_choices = ["lost", rng.choice(("short_skewed", "lost")), "short_skewed"]
            else:
                family_choices = [
                    rng.choices(
                        ("short_skewed", "near_zero", "lost"),
                        weights=(0.46, 0.30, 0.24),
                        k=1,
                    )[0]
                    for _ in colors
                ]
            rng.shuffle(family_choices)

            for idx, color in enumerate(colors):
                theta0 = base_rot + idx * (2.0 * np.pi / 3.0) + rng.uniform(-0.22, 0.22)
                direction = rng.choice((-1.0, 1.0))
                family = family_choices[idx]
                inner_r_start = r * rng.uniform(0.16, 0.28)
                inner_r_end = r * rng.uniform(0.16, 0.28)
                phase = rng.uniform(-0.9, 0.9)
                pts = []

                if family == "near_zero":
                    loop_count = rng.uniform(2.1, 3.2)
                    arc_span = direction * loop_count * 2.0 * np.pi
                    outer_r = r * rng.uniform(2.15, 2.95)
                    outer_r += (idx - 1) * r * rng.uniform(0.16, 0.34)
                    drift = r * rng.uniform(-0.16, 0.16)
                    path_style = rng.choice(("loose_orbit", "flower_orbit", "meander_orbit"))
                    num_pts = rng.choice((13, 14, 15))
                    weave_phase = phase + idx * rng.uniform(0.8, 1.5)
                    cross_amp = r * rng.uniform(0.18, 0.34)
                    for t in np.linspace(0.0, 1.0, num_pts):
                        theta = theta0 + arc_span * t
                        theta += 0.18 * np.sin(2.0 * np.pi * t + phase)
                        theta += 0.06 * np.sin(7.0 * np.pi * t + 0.3 * phase)
                        if t < 0.10:
                            radius = inner_r_start + (outer_r - inner_r_start) * (t / 0.10)
                        elif t > 0.90:
                            radius = outer_r + (inner_r_end - outer_r) * ((t - 0.90) / 0.10)
                        else:
                            if path_style == "loose_orbit":
                                radius = outer_r
                                radius += cross_amp * np.sin(2.3 * np.pi * t + weave_phase)
                                radius += 0.12 * r * np.cos(5.2 * np.pi * t - 0.4 * phase)
                                radius += 0.07 * r * np.sin(8.0 * np.pi * t + 0.5 * weave_phase)
                            elif path_style == "flower_orbit":
                                radius = outer_r
                                radius += 0.14 * r * np.sin(4.0 * np.pi * t + weave_phase)
                                radius += cross_amp * np.sin(7.0 * np.pi * t - 0.2 * phase)
                                radius += 0.06 * r * np.cos(9.0 * np.pi * t + 0.3 * weave_phase)
                            else:
                                radius = outer_r
                                radius += drift * (2.0 * t - 1.0)
                                radius += cross_amp * np.sin(3.0 * np.pi * t + weave_phase)
                                radius += 0.08 * r * np.cos(6.5 * np.pi * t + 0.4 * phase)
                                radius += 0.06 * r * np.sin(8.5 * np.pi * t - 0.2 * weave_phase)
                        pts.append(_rotated_xy(radius, theta))

                elif family == "lost":
                    exit_r = r * rng.uniform(1.35, 1.75)
                    exit_theta = theta0 + direction * rng.uniform(0.18 * np.pi, 0.42 * np.pi)
                    exit_pt = np.array(_rotated_xy(exit_r, exit_theta), dtype=float)

                    n_out = rng.choice((2, 3))
                    n_wander = rng.choice((6, 7, 8))
                    n_return = rng.choice((3, 4))

                    # 1) Exit the reward-centered neighborhood.
                    for t in np.linspace(0.0, 1.0, n_out, endpoint=False):
                        theta = theta0 + (exit_theta - theta0) * t
                        theta += 0.08 * np.sin(np.pi * t + phase)
                        radius = inner_r_start + (exit_r - inner_r_start) * (t ** 0.82)
                        radius += 0.03 * r * np.sin(2.0 * np.pi * t + 0.5 * phase)
                        pts.append(_rotated_xy(radius, theta))

                    # 2) Persistent random walk while "lost", only later regaining a homeward bias.
                    current_pt = exit_pt.copy()
                    heading = exit_theta + rng.uniform(-1.4, 1.4)
                    for j in range(n_wander):
                        frac = (j + 1) / max(1, n_wander)
                        home_theta = np.arctan2(cy - current_pt[1], cx - current_pt[0])
                        if frac < 0.55:
                            heading += rng.uniform(-1.35, 1.35)
                        else:
                            heading = 0.72 * heading + 0.28 * (
                                home_theta + rng.uniform(-0.95, 0.95)
                            )
                        step_len = r * rng.uniform(0.42, 0.88)
                        trial_pt = current_pt + step_len * np.array(
                            [np.cos(heading), np.sin(heading)]
                        )
                        if float(np.hypot(trial_pt[0] - cx, trial_pt[1] - cy)) < 1.55 * r:
                            push_theta = heading + np.pi + rng.uniform(-0.7, 0.7)
                            trial_pt = current_pt + r * rng.uniform(0.55, 0.95) * np.array(
                                [np.cos(push_theta), np.sin(push_theta)]
                            )
                        if float(np.hypot(trial_pt[0] - cx, trial_pt[1] - cy)) > 3.8 * r:
                            rebound_theta = np.arctan2(cy - current_pt[1], cx - current_pt[0])
                            trial_pt = current_pt + r * rng.uniform(0.45, 0.85) * np.array(
                                [
                                    np.cos(rebound_theta + rng.uniform(-0.9, 0.9)),
                                    np.sin(rebound_theta + rng.uniform(-0.9, 0.9)),
                                ]
                            )
                        current_pt = trial_pt
                        pts.append(tuple(current_pt))

                    # 3) Regain the reward and return with a delayed, imperfect homing phase.
                    last_pt = np.array(pts[-1], dtype=float)
                    return_theta = np.arctan2(cy - last_pt[1], cx - last_pt[0]) + rng.uniform(
                        -0.45, 0.45
                    )
                    reentry_anchor = np.array(
                        _rotated_xy(r * rng.uniform(0.92, 1.12), return_theta), dtype=float
                    )
                    for t in np.linspace(0.0, 1.0, n_return, endpoint=False):
                        blend = t
                        bridge = last_pt + blend * (reentry_anchor - last_pt)
                        bridge += r * rng.uniform(0.18, 0.42) * np.array(
                            [
                                np.cos(return_theta + np.pi / 2 + 2.5 * blend + 0.4 * phase),
                                np.sin(return_theta + np.pi / 2 + 2.1 * blend - 0.3 * phase),
                            ]
                        )
                        bridge += r * rng.uniform(0.08, 0.20) * np.array(
                            [
                                np.cos(return_theta - 1.3 + 3.6 * blend),
                                np.sin(return_theta + 0.9 - 2.8 * blend),
                            ]
                        )
                        pts.append(tuple(bridge))
                    pts.append(_rotated_xy(inner_r_end, return_theta + rng.uniform(-0.18, 0.18)))

                else:
                    outer_r = r * rng.uniform(1.04, 1.34)
                    arc_span = direction * rng.uniform(0.45 * np.pi, 0.95 * np.pi)
                    path_style = rng.choice(("hook_loop", "bean_loop", "comma_loop"))
                    num_pts = rng.choice((8, 9))
                    for t in np.linspace(0.0, 1.0, num_pts):
                        theta = theta0 + arc_span * t
                        theta += 0.12 * np.sin(2.0 * np.pi * t + phase)
                        if path_style == "bean_loop":
                            theta += 0.08 * np.sin(4.2 * np.pi * t - 0.3 * phase)
                        elif path_style == "comma_loop":
                            theta += 0.10 * (t - 0.5) ** 3
                        if t < 0.16:
                            radius = inner_r_start + (outer_r - inner_r_start) * (t / 0.16)
                        elif t > 0.84:
                            radius = outer_r + (inner_r_end - outer_r) * ((t - 0.84) / 0.16)
                        else:
                            if path_style == "hook_loop":
                                radius = outer_r
                                radius += 0.10 * r * np.sin(np.pi * t) ** 2
                                radius += 0.05 * r * np.sin(3.0 * np.pi * t + phase)
                            elif path_style == "bean_loop":
                                radius = outer_r
                                radius += 0.14 * r * np.sin(np.pi * t)
                                radius += 0.05 * r * np.cos(4.0 * np.pi * t + phase)
                            else:
                                radius = outer_r
                                radius += 0.08 * r * (2.0 * t - 1.0)
                                radius += 0.05 * r * np.sin(3.4 * np.pi * t - 0.5 * phase)
                        pts.append(_rotated_xy(radius, theta))

                segments.append(_segment_payload(_build_path(pts), color, idx))
        else:
            target_vecs = []
            for base_angle in np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False):
                theta = base_rot + base_angle + rng.uniform(-0.32, 0.32)
                mag = r * rng.uniform(0.42, 0.82)
                target_vecs.append(np.array([mag * np.cos(theta), mag * np.sin(theta)]))
            target_vecs = np.asarray(target_vecs, dtype=float)
            target_vecs -= np.mean(target_vecs, axis=0, keepdims=True)
            offset_mag = rng.choice(
                (
                    0.0,
                    r * rng.uniform(0.10, 0.22),
                    r * rng.uniform(0.22, 0.38),
                )
            )
            offset_theta = rng.uniform(0.0, 2.0 * np.pi)
            offset_vec = offset_mag * np.array([np.cos(offset_theta), np.sin(offset_theta)])
            target_vecs += offset_vec
            mags = np.hypot(target_vecs[:, 0], target_vecs[:, 1])
            if np.max(mags) > 1.05 * r:
                target_vecs *= (1.05 * r) / np.max(mags)
            target_vecs += offset_vec - np.mean(target_vecs, axis=0)

            for idx, (target_vec, color) in enumerate(zip(target_vecs, colors)):
                target_theta = float(np.arctan2(target_vec[1], target_vec[0]))
                outer_r = min(1.12 * r, max(0.72 * r, 1.22 * float(np.hypot(*target_vec))))
                arc_span = rng.uniform(0.72 * np.pi, 1.08 * np.pi)
                theta_start = target_theta - 0.5 * arc_span
                theta_end = target_theta + 0.5 * arc_span
                inner_r_start = r * rng.uniform(0.16, 0.28)
                inner_r_end = r * rng.uniform(0.16, 0.28)
                phase = rng.uniform(-0.8, 0.8)

                pts = []
                for t in np.linspace(0.0, 1.0, 7):
                    theta = theta_start + (theta_end - theta_start) * t
                    theta += 0.08 * np.sin(2.0 * np.pi * t + phase)
                    if t < 0.18:
                        radius = inner_r_start + (outer_r - inner_r_start) * (t / 0.18)
                    elif t > 0.82:
                        radius = outer_r + (inner_r_end - outer_r) * ((t - 0.82) / 0.18)
                    else:
                        radius = outer_r + 0.05 * r * np.sin(np.pi * t + phase)
                    pts.append(_rotated_xy(radius, theta))

                path = _build_path(pts)
                xs = np.asarray(path[:, 0], dtype=float)
                ys = np.asarray(path[:, 1], dtype=float)
                segments.append(
                    {
                        "x": xs,
                        "y": ys,
                        "mx": float(np.nanmean(xs)),
                        "my": float(np.nanmean(ys)),
                        "vx": float(np.nanmean(xs) - cx),
                        "vy": float(np.nanmean(ys) - cy),
                        "mag": float(np.hypot(np.nanmean(xs) - cx, np.nanmean(ys) - cy)),
                        "color": color,
                        "idx": idx + 1,
                    }
                )

        mean_vx = float(np.mean([seg["vx"] for seg in segments]))
        mean_vy = float(np.mean([seg["vy"] for seg in segments]))
        return {
            "cx": cx,
            "cy": cy,
            "r_px": r,
            "segments": segments,
            "mean_vx": mean_vx,
            "mean_vy": mean_vy,
            "mean_mag": float(np.hypot(mean_vx, mean_vy)),
            "mean_x": float(cx + mean_vx),
            "mean_y": float(cy + mean_vy),
        }

    def _plot_between_reward_commag_schematic(
        self,
        *,
        trn_index,
        bucket_index,
        role_idx,
        image_format,
        out_dir,
        seed,
        reward_circle,
        floor_coords,
        agg_mode="vector_mean",
    ):
        agg_mode = str(agg_mode or "vector_mean").strip().lower()
        if agg_mode not in ("vector_mean", "mean_magnitude"):
            agg_mode = "vector_mean"

        if seed is None:
            synth_rng = random.Random()
        else:
            synth_rng = random.Random(
                int(seed) + 1009 * int(bucket_index) + 9173 * max(0, int(trn_index))
            )

        synth = self._synthetic_commag_bucket_segments(
            reward_circle, variant=bucket_index, rng=synth_rng, agg_mode=agg_mode
        )
        if synth is None:
            print("[plot_between_reward_chain] Unable to build COM schematic geometry.")
            return

        top_left, bottom_right = floor_coords[0], floor_coords[1]
        cx = synth["cx"]
        cy = synth["cy"]
        reward_r_px = synth["r_px"]
        mean_color = "#143642"
        mean_mag = float(np.mean([seg["mag"] for seg in synth["segments"]]))

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14.4, 5.0),
            gridspec_kw={"width_ratios": [1.12, 0.98, 0.90]},
        )
        panel_title_pad = 10
        panel_note_y = 0.955
        panel_note_bbox = dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="#cbd5e1",
            linewidth=0.8,
            alpha=0.92,
        )

        def _set_local_view(ax, xs, ys, *, include_points=(), pad_mult=0.32):
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            extra_x = [float(p[0]) for p in include_points]
            extra_y = [float(p[1]) for p in include_points]
            all_x = np.concatenate((xs, np.asarray(extra_x, dtype=float)))
            all_y = np.concatenate((ys, np.asarray(extra_y, dtype=float)))

            x_min = float(np.nanmin(all_x))
            x_max = float(np.nanmax(all_x))
            y_min = float(np.nanmin(all_y))
            y_max = float(np.nanmax(all_y))
            span_x = max(x_max - x_min, 2.2 * reward_r_px)
            span_y = max(y_max - y_min, 2.2 * reward_r_px)
            pad_x = max(10.0, pad_mult * span_x)
            pad_y = max(10.0, pad_mult * span_y)

            floor_x_min = min(float(top_left[0]), float(bottom_right[0]))
            floor_x_max = max(float(top_left[0]), float(bottom_right[0]))
            floor_y_min = min(float(top_left[1]), float(bottom_right[1]))
            floor_y_max = max(float(top_left[1]), float(bottom_right[1]))

            ax.set_xlim(max(floor_x_min, x_min - pad_x), min(floor_x_max, x_max + pad_x))
            ax.set_ylim(min(floor_y_max, y_max + pad_y), max(floor_y_min, y_min - pad_y))

        def _setup_arena_panel(ax, *, show_floor=False):
            rect = patches.FancyBboxPatch(
                (top_left[0], top_left[1]),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1.0,
                edgecolor="#374151",
                facecolor="#fbfcfd",
                zorder=0 if show_floor else -5,
                alpha=1.0 if show_floor else 0.0,
            )
            ax.add_patch(rect)
            reward_patch = patches.Circle(
                (cx, cy),
                reward_r_px,
                facecolor="#fff6db",
                edgecolor="#c59d3d",
                linewidth=1.5,
                alpha=0.95,
                zorder=1,
            )
            ax.add_patch(reward_patch)
            self._draw_reward_center_marker(
                ax,
                cx,
                cy,
                ring_color=mean_color,
                label=None,
                dot_radius=1.45,
                show_halo=False,
            )
            ax.text(
                cx,
                cy + reward_r_px + 7.0,
                "reward circle",
                fontsize=8,
                color="#8a6a12",
                ha="center",
                va="center",
                zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.72,
                ),
            )
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        ax = axes[0]
        _setup_arena_panel(ax)
        panel_a_x = []
        panel_a_y = []
        for seg in synth["segments"]:
            xs = np.asarray(seg["x"], dtype=float)
            ys = np.asarray(seg["y"], dtype=float)
            panel_a_x.extend(xs.tolist())
            panel_a_y.extend(ys.tolist())
            color = str(seg["color"])
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.5,
                alpha=0.95,
                zorder=3,
                solid_capstyle="round",
                label="Between-reward segment" if seg["idx"] == 1 else None,
            )
            ax.scatter(
                [xs[0], xs[-1]],
                [ys[0], ys[-1]],
                s=18,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
            ax.annotate(
                "",
                xy=(seg["mx"], seg["my"]),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.7,
                    color=color,
                    linestyle=(0, (4, 3)),
                    alpha=0.92,
                    shrinkA=2.0,
                    shrinkB=3.5,
                ),
                zorder=4,
            )
            ax.scatter(
                [seg["mx"]],
                [seg["my"]],
                s=54,
                color=color,
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
                label="Segment COM" if seg["idx"] == 1 else None,
            )
            badge_dx = 7.0 if seg["vx"] >= 0 else -7.0
            badge_dy = -7.0 if seg["vy"] >= 0 else 7.0
            ax.text(
                float(seg["mx"]) + badge_dx,
                float(seg["my"]) + badge_dy,
                f"{seg['idx']}",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="circle,pad=0.18",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )

        panel_a_points = [(cx, cy)]
        for seg in synth["segments"]:
            panel_a_points.append((seg["mx"], seg["my"]))
        _set_local_view(ax, panel_a_x, panel_a_y, include_points=panel_a_points, pad_mult=0.40)
        ax.text(
            0.03,
            panel_note_y,
            "Several between-reward segments\nwithin one sync bucket",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#334155",
            zorder=7,
            bbox=panel_note_bbox,
        )
        ax.set_title("A. Segment COM Vectors", fontsize=10, pad=panel_title_pad, loc="left")

        ax = axes[1]
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#fbfcfd")
        vec_extent = max(
            [seg["mag"] for seg in synth["segments"]] + [synth["mean_mag"], mean_mag, reward_r_px]
        )
        lim = max(1.15 * vec_extent, reward_r_px * 0.9)
        if agg_mode == "mean_magnitude":
            row_y = np.linspace(-0.40 * lim, 0.44 * lim, len(synth["segments"]) + 1)
            scalar_rows = row_y[: len(synth["segments"])]
            mag_y = row_y[-1]
            axis_y = 0.76 * lim
            ax.annotate(
                "",
                xy=(1.18 * lim, axis_y),
                xytext=(0.0, axis_y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.1,
                    color="#cbd5e1",
                    shrinkA=0.0,
                    shrinkB=0.0,
                ),
                zorder=0,
            )
            ax.text(
                0.0,
                axis_y + 0.12 * lim,
                "distance from reward center",
                fontsize=8,
                color="#475569",
                ha="left",
                va="bottom",
                zorder=4,
            )
            for seg, row_y in zip(synth["segments"], scalar_rows):
                color = str(seg["color"])
                ax.annotate(
                    "",
                    xy=(seg["mag"], row_y),
                    xytext=(0.0, row_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.0,
                        color=color,
                        alpha=0.9,
                        shrinkA=0.0,
                        shrinkB=3.0,
                    ),
                    zorder=2,
                )
                ax.scatter(
                    [seg["mag"]],
                    [row_y],
                    s=50,
                    color=color,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=3,
                )
                ax.text(
                    seg["mag"] + 0.04 * lim,
                    row_y,
                    f"{seg['idx']}",
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox=dict(
                        boxstyle="circle,pad=0.18",
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.94,
                    ),
                )
            ax.annotate(
                "",
                xy=(mean_mag, mag_y),
                xytext=(0.0, mag_y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=3.0,
                    color=mean_color,
                    alpha=0.98,
                    shrinkA=0.0,
                    shrinkB=4.0,
                ),
                zorder=5,
            )
            ax.scatter(
                [mean_mag],
                [mag_y],
                s=72,
                color=mean_color,
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
            )
            ax.text(
                mean_mag + 0.06 * lim,
                mag_y - 0.06 * lim,
                "mean distance",
                fontsize=8.5,
                color=mean_color,
                ha="left",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )
            ax.text(
                0.05,
                panel_note_y,
                r"Use COM magnitudes only, then average",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color="#334155",
                zorder=7,
                bbox=panel_note_bbox,
            )
            ax.set_xlim(-0.08 * lim, 1.25 * lim)
            ax.set_ylim(0.98 * lim, -0.8 * lim)
        else:
            ax.axhline(0.0, color="#cbd5e1", linewidth=1.0, zorder=0)
            ax.axvline(0.0, color="#cbd5e1", linewidth=1.0, zorder=0)
            ax.scatter(
                [0.0],
                [0.0],
                s=42,
                color=mean_color,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.text(
                0.0,
                0.12 * max(reward_r_px, 1.0),
                "reward center as origin",
                fontsize=8,
                color=mean_color,
                ha="center",
                va="bottom",
                zorder=4,
            )
            for seg in synth["segments"]:
                color = str(seg["color"])
                ax.annotate(
                    "",
                    xy=(seg["vx"], seg["vy"]),
                    xytext=(0.0, 0.0),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.0,
                        color=color,
                        alpha=0.9,
                        shrinkA=0.0,
                        shrinkB=3.0,
                    ),
                    zorder=2,
                )
                ax.scatter(
                    [seg["vx"]],
                    [seg["vy"]],
                    s=50,
                    color=color,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=3,
                )
                ax.text(
                    seg["vx"] * 1.04,
                    seg["vy"] * 1.04,
                    f"{seg['idx']}",
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox=dict(
                        boxstyle="circle,pad=0.18",
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.94,
                    ),
                )
            ax.annotate(
                "",
                xy=(synth["mean_vx"], synth["mean_vy"]),
                xytext=(0.0, 0.0),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=3.0,
                    color=mean_color,
                    alpha=0.98,
                    shrinkA=0.0,
                    shrinkB=4.0,
                ),
                zorder=5,
            )
            ax.scatter(
                [synth["mean_vx"]],
                [synth["mean_vy"]],
                s=72,
                color=mean_color,
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
            )
            ax.text(
                synth["mean_vx"] + 0.10 * lim,
                synth["mean_vy"] - 0.09 * lim,
                "mean of segment COM vectors",
                fontsize=8.5,
                color=mean_color,
                ha="left",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    linewidth=0.8,
                    alpha=0.92,
                    ),
                )
            ax.set_xlim(-0.2 * lim, 1.25 * lim)
            ax.set_ylim(1.2 * lim, -0.8 * lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            "B. Aggregate Across Segments", fontsize=10, pad=panel_title_pad, loc="left"
        )

        ax = axes[2]
        _setup_arena_panel(ax)
        panel_c_ring_radius = mean_mag if agg_mode == "mean_magnitude" else synth["mean_mag"]
        if agg_mode == "mean_magnitude":
            mag_ring = patches.Circle(
                (cx, cy),
                radius=mean_mag,
                fill=False,
                linestyle=(0, (3, 3)),
                linewidth=1.6,
                edgecolor=mean_color,
                alpha=0.50,
                zorder=2,
            )
            ax.add_patch(mag_ring)
            rep_x = cx + mean_mag
            rep_y = cy
            ax.annotate(
                "",
                xy=(rep_x, rep_y),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=3.0,
                    color=mean_color,
                    alpha=0.95,
                    shrinkA=2.0,
                    shrinkB=4.0,
                ),
                zorder=4,
            )
            ax.scatter(
                [rep_x],
                [rep_y],
                s=74,
                color=mean_color,
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
            ax.text(
                cx + 0.58 * mean_mag,
                cy - 0.18 * reward_r_px,
                "mean segment COM distance",
                fontsize=8.5,
                color=mean_color,
                ha="left",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=mean_color,
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )
            panel_c_x = [cx, rep_x]
            panel_c_y = [cy, rep_y]
            panel_c_points = (
                (cx, cy),
                (rep_x, rep_y),
                (cx - panel_c_ring_radius, cy),
                (cx + panel_c_ring_radius, cy),
                (cx, cy - panel_c_ring_radius),
                (cx, cy + panel_c_ring_radius),
            )
        else:
            mag_ring = patches.Circle(
                (cx, cy),
                radius=synth["mean_mag"],
                fill=False,
                linestyle=(0, (3, 3)),
                linewidth=1.4,
                edgecolor=mean_color,
                alpha=0.35,
                zorder=2,
            )
            ax.add_patch(mag_ring)
            ax.annotate(
                "",
                xy=(synth["mean_x"], synth["mean_y"]),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=3.0,
                    color=mean_color,
                    alpha=0.98,
                    shrinkA=2.0,
                    shrinkB=4.0,
                ),
                zorder=4,
            )
            ax.scatter(
                [synth["mean_x"]],
                [synth["mean_y"]],
                s=74,
                color=mean_color,
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
            mid_x = 0.5 * (cx + synth["mean_x"])
            mid_y = 0.5 * (cy + synth["mean_y"])
            ax.text(
                mid_x + 8.5,
                mid_y - 7.0,
                "bucket COM vector",
                fontsize=8.5,
                color=mean_color,
                ha="left",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=mean_color,
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )
            ax.text(
                0.03,
                panel_note_y,
                r"bucket COM distance = $\left\|\mathrm{mean}(m_x, m_y)\right\|$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="#334155",
                zorder=7,
                bbox=panel_note_bbox,
            )
            panel_c_x = [cx, synth["mean_x"]]
            panel_c_y = [cy, synth["mean_y"]]
            panel_c_points = (
                (cx, cy),
                (synth["mean_x"], synth["mean_y"]),
                (cx - panel_c_ring_radius, cy),
                (cx + panel_c_ring_radius, cy),
                (cx, cy - panel_c_ring_radius),
                (cx, cy + panel_c_ring_radius),
            )
        ax.set_title("C. Final Magnitude", fontsize=10, pad=panel_title_pad, loc="left")
        _set_local_view(
            ax,
            panel_c_x,
            panel_c_y,
            include_points=panel_c_points,
            pad_mult=0.48,
        )

        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#4e79a7",
                lw=2.4,
                label="Between-reward segments",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=mean_color,
                markeredgecolor="white",
                markersize=7,
                label="Reward center / COM points",
            ),
            plt.Line2D(
                [0],
                [0],
                color=mean_color,
                lw=3.0,
                label=(
                    "Mean segment COM distance"
                    if agg_mode == "mean_magnitude"
                    else "Bucket-mean COM vector"
                ),
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )

        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f
        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)
        fly_role = "exp" if role_idx == 0 else "yok"
        seed_str = f"{seed}" if seed is not None else "rand"

        fig.suptitle(
            "Between-reward COM schematic\n"
            f"{video_id}, fly {fly_idx}, {fly_role}\n"
            f"trn {trn_index + 1}, bucket {bucket_index + 1} | synthetic "
            f"{'mean-magnitude' if agg_mode == 'mean_magnitude' else 'vector-mean'} COM aggregation example",
            fontsize=12,
            y=0.985,
        )

        output_dir = out_dir or "imgs/between_rewards"
        output_path = (
            f"{output_dir}/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_bkt{bucket_index + 1}_"
            f"N3_seed{seed_str}_schematic-"
            f"{'commag_synth_mean_magnitude' if agg_mode == 'mean_magnitude' else 'commag_synth_vector_mean'}."
            f"{image_format}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.subplots_adjust(left=0.03, right=0.985, top=0.73, bottom=0.12, wspace=0.10)
        writeImage(output_path, format=image_format)
        plt.close(fig)

    def _turnback_schematic_geometry(self, reward_circle):
        if reward_circle is None or self.va is None:
            return None

        try:
            px_per_mm = float(self.va.ct.pxPerMmFloor()) * float(self.va.xf.fctr)
        except Exception:
            return None
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return None

        opts = getattr(self.va, "opts", None)
        inner_delta_mm = float(getattr(opts, "turnback_inner_delta_mm", 0.0) or 0.0)
        outer_delta_mm = float(getattr(opts, "turnback_outer_delta_mm", 2.0) or 2.0)
        border_width_mm = float(
            getattr(opts, "turnback_border_width_mm", 0.1) or 0.1
        )
        inner_radius_offset_px = float(
            getattr(opts, "turnback_inner_radius_offset_px", 0.0) or 0.0
        )

        cx, cy, reward_r_px = (float(v) for v in reward_circle)
        inner_r_px = (
            float(reward_r_px)
            + float(inner_delta_mm) * float(px_per_mm)
            + float(inner_radius_offset_px)
        )
        outer_r_px = float(reward_r_px) + float(outer_delta_mm) * float(px_per_mm)
        border_width_px = float(border_width_mm) * float(px_per_mm)

        if not np.isfinite(inner_r_px) or not np.isfinite(outer_r_px):
            return None
        if outer_r_px <= inner_r_px:
            return None

        return {
            "cx": cx,
            "cy": cy,
            "reward_r_px": float(reward_r_px),
            "inner_r_px": float(inner_r_px),
            "outer_r_px": float(outer_r_px),
            "border_width_px": float(max(0.0, border_width_px)),
            "px_per_mm": float(px_per_mm),
        }

    @staticmethod
    def _polar_path_to_xy(cx, cy, polar_points):
        return np.asarray(
            [
                [float(cx) + float(rad) * np.cos(theta), float(cy) + float(rad) * np.sin(theta)]
                for theta, rad in polar_points
            ],
            dtype=float,
        )

    @staticmethod
    def _first_state_transition(mask, *, start_idx=1, from_state, to_state):
        arr = np.asarray(mask, dtype=bool)
        if arr.size < 2:
            return None

        lo = max(1, int(start_idx))
        for idx in range(lo, arr.size):
            if bool(arr[idx - 1]) == bool(from_state) and bool(arr[idx]) == bool(to_state):
                return int(idx)
        return None

    @classmethod
    def _turnback_path_event_indices(cls, xs, ys, cx, cy, inner_r_px, outer_r_px):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
            return None

        d = np.hypot(xs - float(cx), ys - float(cy))
        in_inner = d <= float(inner_r_px)
        in_outer = d <= float(outer_r_px)

        exit_inner_idx = cls._first_state_transition(
            in_inner, start_idx=1, from_state=True, to_state=False
        )
        if exit_inner_idx is None:
            outside_inner = np.flatnonzero(~in_inner)
            if outside_inner.size == 0:
                return None
            exit_inner_idx = int(outside_inner[0])

        reenter_idx = cls._first_state_transition(
            in_inner, start_idx=exit_inner_idx + 1, from_state=False, to_state=True
        )
        exit_outer_idx = cls._first_state_transition(
            in_outer, start_idx=exit_inner_idx + 1, from_state=True, to_state=False
        )

        return {
            "exit_inner_idx": int(exit_inner_idx),
            "reenter_idx": None if reenter_idx is None else int(reenter_idx),
            "exit_outer_idx": None if exit_outer_idx is None else int(exit_outer_idx),
        }

    def _synthetic_turnback_ratio_paths(self, reward_circle, *, variant=0, rng=None):
        geom = self._turnback_schematic_geometry(reward_circle)
        if geom is None:
            return None

        rng = rng or random.Random(int(variant) + 17)
        cx = geom["cx"]
        cy = geom["cy"]
        inner_r_px = geom["inner_r_px"]
        outer_r_px = geom["outer_r_px"]

        gap_px = max(outer_r_px - inner_r_px, 1.0)
        success_mid_r = min(
            0.86 * outer_r_px,
            inner_r_px + rng.uniform(0.42, 0.72) * gap_px,
        )
        success_wide_r = min(
            0.90 * outer_r_px,
            inner_r_px + rng.uniform(0.55, 0.84) * gap_px,
        )
        fail_mid_r = min(
            0.92 * outer_r_px,
            inner_r_px + rng.uniform(0.54, 0.82) * gap_px,
        )
        fail_exit_r = max(
            1.03 * outer_r_px,
            outer_r_px + rng.uniform(0.05, 0.18) * gap_px,
        )
        fail_far_r = max(
            1.10 * outer_r_px,
            outer_r_px + rng.uniform(0.12, 0.28) * gap_px,
        )

        success_pts = [
            (
                np.deg2rad(rng.uniform(122.0, 168.0)),
                rng.uniform(0.48, 0.64) * inner_r_px,
            ),
            (
                np.deg2rad(rng.uniform(152.0, 192.0)),
                rng.uniform(1.00, 1.08) * inner_r_px,
            ),
            (
                np.deg2rad(rng.uniform(184.0, 226.0)),
                success_mid_r,
            ),
            (
                np.deg2rad(rng.uniform(220.0, 278.0)),
                success_wide_r,
            ),
            (
                np.deg2rad(rng.uniform(256.0, 320.0)),
                rng.uniform(0.88, 0.98) * inner_r_px,
            ),
            (
                np.deg2rad(rng.uniform(286.0, 338.0)),
                rng.uniform(0.48, 0.68) * inner_r_px,
            ),
        ]
        fail_pts = [
            (
                np.deg2rad(rng.uniform(10.0, 52.0)),
                rng.uniform(0.50, 0.66) * inner_r_px,
            ),
            (
                np.deg2rad(rng.uniform(2.0, 30.0)),
                rng.uniform(1.00, 1.08) * inner_r_px,
            ),
            (
                np.deg2rad(rng.uniform(-18.0, 16.0)),
                fail_mid_r,
            ),
            (
                np.deg2rad(rng.uniform(-52.0, -18.0)),
                fail_exit_r,
            ),
            (
                np.deg2rad(rng.uniform(-72.0, -34.0)),
                fail_far_r,
            ),
        ]

        success_xy = self._catmull_rom_chain(
            self._polar_path_to_xy(cx, cy, success_pts), samples_per_seg=26
        )
        fail_xy = self._catmull_rom_chain(
            self._polar_path_to_xy(cx, cy, fail_pts), samples_per_seg=24
        )
        if success_xy.shape[0] < 2 or fail_xy.shape[0] < 2:
            return None

        success_events = self._turnback_path_event_indices(
            success_xy[:, 0],
            success_xy[:, 1],
            cx,
            cy,
            inner_r_px,
            outer_r_px,
        )
        fail_events = self._turnback_path_event_indices(
            fail_xy[:, 0],
            fail_xy[:, 1],
            cx,
            cy,
            inner_r_px,
            outer_r_px,
        )
        if success_events is None or fail_events is None:
            return None

        success_cross_idx = success_events.get("reenter_idx")
        if success_cross_idx is None:
            return None
        fail_cross_idx = fail_events.get("exit_outer_idx")
        if fail_cross_idx is None:
            return None

        return {
            "geom": geom,
            "success": {
                "x": np.asarray(success_xy[:, 0], dtype=float),
                "y": np.asarray(success_xy[:, 1], dtype=float),
                "start_idx": int(success_events["exit_inner_idx"]),
                "event_idx": int(success_cross_idx),
                "event_label": "re-enter inner",
                "label": "turnback",
                "callout": "turnback\nre-enters inner circle",
                "color": "#1f8a70",
                "event_color": "#0d5c4f",
            },
            "failure": {
                "x": np.asarray(fail_xy[:, 0], dtype=float),
                "y": np.asarray(fail_xy[:, 1], dtype=float),
                "start_idx": int(fail_events["exit_inner_idx"]),
                "event_idx": int(fail_cross_idx),
                "event_label": "exit outer",
                "label": "no turnback",
                "callout": "no turnback\nexits outer circle",
                "color": "#d95f02",
                "event_color": "#8c3b00",
            },
        }

    def _overlay_synthetic_turnback_ratio_schematic(
        self,
        ax,
        *,
        reward_circle,
        top_left,
        bottom_right,
        rng=None,
        variant=0,
    ):
        synth = self._synthetic_turnback_ratio_paths(
            reward_circle,
            variant=variant,
            rng=rng,
        )
        if synth is None:
            return None

        geom = synth["geom"]
        cx = geom["cx"]
        cy = geom["cy"]
        reward_r_px = geom["reward_r_px"]
        inner_r_px = geom["inner_r_px"]
        outer_r_px = geom["outer_r_px"]
        border_width_px = geom["border_width_px"]

        inner_band = patches.Circle(
            (cx, cy),
            inner_r_px,
            fill=False,
            linestyle="-",
            linewidth=max(1.6, 0.45 * border_width_px),
            edgecolor="#3a7ca5",
            alpha=0.92,
            zorder=4,
            label="Inner circle",
        )
        outer_band = patches.Circle(
            (cx, cy),
            outer_r_px,
            fill=False,
            linestyle=(0, (6, 3)),
            linewidth=max(1.8, 0.35 * border_width_px),
            edgecolor="#6b7280",
            alpha=0.95,
            zorder=3,
            label="Outer circle",
        )
        reward_ring = patches.Circle(
            (cx, cy),
            reward_r_px,
            fill=False,
            linestyle="-",
            linewidth=1.5,
            edgecolor="#b8b8b8",
            alpha=0.95,
            zorder=2,
            label="Reward circle",
        )
        ax.add_patch(reward_ring)
        ax.add_patch(outer_band)
        ax.add_patch(inner_band)

        def _label_anchor_for_event(x_event, y_event):
            dx = float(x_event) - float(cx)
            dy = float(y_event) - float(cy)
            norm = float(np.hypot(dx, dy))
            if norm <= 1e-6:
                ux, uy = 1.0, 0.0
            else:
                ux, uy = dx / norm, dy / norm

            # Push labels clearly outside the outer circle so they read as callouts.
            radial_pad = max(22.0, 0.28 * outer_r_px)
            tangential_pad = 0.22 * outer_r_px
            side = -1.0 if ux < 0 else 1.0
            tx = float(x_event) + ux * radial_pad
            ty = float(y_event) + uy * radial_pad + side * tangential_pad
            return self._clamp_point_to_floor(tx, ty, top_left, bottom_right)

        for key in ("success", "failure"):
            spec = synth[key]
            xs = np.asarray(spec["x"], dtype=float)
            ys = np.asarray(spec["y"], dtype=float)
            color = str(spec["color"])
            event_idx = int(np.clip(spec["event_idx"], 0, len(xs) - 1))
            start_idx = int(np.clip(spec.get("start_idx", 0), 0, event_idx))

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.5,
                zorder=5,
                label=spec["label"],
            )

            step = max(1, int((len(xs) - 1) / 4))
            for i in range(0, len(xs) - 1, step):
                j = min(len(xs) - 1, i + 1)
                if j <= i:
                    continue
                ax.annotate(
                    "",
                    xy=(xs[j], ys[j]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.1,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        alpha=0.88,
                    ),
                    zorder=5,
                )

            ax.scatter(
                [xs[start_idx]],
                [ys[start_idx]],
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
            )
            ax.scatter(
                [xs[event_idx]],
                [ys[event_idx]],
                s=48,
                color=spec["event_color"],
                edgecolors="white",
                linewidths=0.9,
                zorder=7,
            )

            if event_idx > 0:
                dx = float(xs[event_idx] - xs[event_idx - 1])
                dy = float(ys[event_idx] - ys[event_idx - 1])
            else:
                dx = float(xs[min(len(xs) - 1, 1)] - xs[0])
                dy = float(ys[min(len(xs) - 1, 1)] - ys[0])
            if np.hypot(dx, dy) > 1e-6:
                fly_angle = np.degrees(np.arctan2(dy, dx)) - 90.0
                self._draw_fly_icon(
                    ax,
                    float(xs[event_idx]),
                    float(ys[event_idx]),
                    angle_deg=float(fly_angle),
                )

            label_x = float(xs[event_idx])
            label_y = float(ys[event_idx])
            end_tx, end_ty = _label_anchor_for_event(label_x, label_y)
            ha = "right" if end_tx < label_x else "left"
            ax.text(
                end_tx,
                end_ty,
                spec["callout"],
                fontsize=8.5,
                color=color,
                ha=ha,
                va="center",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.82,
                ),
            )
            ax.annotate(
                "",
                xy=(label_x, label_y),
                xytext=(end_tx, end_ty),
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.9,
                    alpha=0.62,
                    shrinkA=4.0,
                    shrinkB=4.0,
                ),
                zorder=7,
            )

        return synth

    def _get_bucket_range(
        self, *, trn_index: int, bucket_index: int
    ) -> Optional[Tuple[int, int]]:
        """
        Return (start, stop) absolute frame indices for a (training, bucket) pair.

        Prefers va.sync_bucket_ranges[t][b] = (start, stop) if available.
        Falls back to va.buckets[t] boundaries like [b0, b1, ..., bN].
        """
        sync_ranges = getattr(self.va, "sync_bucket_ranges", None)
        if sync_ranges is not None:
            if trn_index < 0 or trn_index >= len(sync_ranges):
                return None
            ranges = sync_ranges[trn_index] or []
            if bucket_index < 0 or bucket_index >= len(ranges):
                return None
            sb_start, sb_stop = ranges[bucket_index]
            return int(sb_start), int(sb_stop)

        if not hasattr(self.va, "buckets") or trn_index >= len(self.va.buckets):
            return None
        buckets = self.va.buckets[trn_index]
        if bucket_index < 0 or bucket_index >= len(buckets) - 1:
            return None
        return int(buckets[bucket_index]), int(buckets[bucket_index + 1])

    def _return_prob_windowing(self) -> Tuple[int, int, int]:
        opts = getattr(self.va, "opts", None)
        raw_skip = getattr(opts, "return_prob_outer_radius_skip_first_sync_buckets", None)
        raw_keep = getattr(opts, "return_prob_outer_radius_keep_first_sync_buckets", None)
        skip_first = (
            getattr(opts, "skip_first_sync_buckets", 0) if raw_skip is None else raw_skip
        )
        keep_first = (
            getattr(opts, "keep_first_sync_buckets", 0) if raw_keep is None else raw_keep
        )
        last_sync = int(
            getattr(opts, "return_prob_outer_radius_last_sync_buckets", 0) or 0
        )
        return max(0, int(skip_first or 0)), max(0, int(keep_first or 0)), max(
            0, last_sync
        )

    def _selected_return_prob_windows_for_training(self, trn_index: int):
        sync_ranges = getattr(self.va, "sync_bucket_ranges", None)
        if trn_index < 0 or trn_index >= len(getattr(self.va, "trns", [])):
            return []
        trn = self.va.trns[trn_index]
        if trn is None or not trn.isCircle():
            return []

        skip_first, keep_first, last_sync_buckets = self._return_prob_windowing()
        if sync_ranges and trn_index < len(sync_ranges) and sync_ranges[trn_index]:
            rr = list(sync_ranges[trn_index])
            rr = rr[max(0, int(skip_first)) :]
            if keep_first > 0:
                rr = rr[: int(keep_first)]
            if last_sync_buckets > 0:
                rr = rr[-int(last_sync_buckets) :]
            if not rr:
                return []
            return [(int(a), int(b)) for (a, b) in rr]

        return [(int(trn.start), int(trn.stop))]

    @staticmethod
    def _frame_in_ranges(frame: int, ranges) -> bool:
        for a, b in ranges:
            if int(a) <= frame < int(b):
                return True
        return False

    def _return_prob_window_counts(
        self,
        *,
        trn_index: int,
        outer_delta_mm: float,
        reward_delta_mm: float,
        border_width_mm: float,
        ctrl: bool,
    ) -> Tuple[Optional[Tuple[int, int, int]], str]:
        ranges = self._selected_return_prob_windows_for_training(trn_index)
        if not ranges:
            return None, "no selected window"

        trn = self.va.trns[trn_index]
        episodes = self.trj.reward_return_probability_episodes_for_training(
            trn=trn,
            outer_delta_mm=outer_delta_mm,
            reward_delta_mm=reward_delta_mm,
            border_width_mm=border_width_mm,
            ctrl=ctrl,
            debug=False,
        )
        if not episodes:
            return (0, 0, 0), self._return_prob_window_label(ranges)

        succ = 0
        fail = 0
        for ep in episodes:
            event_t = int(ep["stop"]) - 1
            if not self._frame_in_ranges(event_t, ranges):
                continue
            if bool(ep.get("returns", False)):
                succ += 1
            else:
                fail += 1
        return (succ, fail, succ + fail), self._return_prob_window_label(ranges)

    @staticmethod
    def _return_prob_window_label(ranges) -> str:
        if not ranges:
            return "window"
        if len(ranges) == 1:
            a, b = ranges[0]
            return f"[{int(a)},{int(b)})"
        return f"{len(ranges)} buckets"

    def draw_custom_arrowhead(
        self,
        ax,
        x_mid,
        y_mid,
        dx,
        dy,
        color,
        length=1.1,
        angle=30,
        shift_factor=-0.08,
        linewidth=1.0,
    ):
        """
        Draws a custom arrowhead using two line segments that converge slightly past the midpoint.

        Parameters:
        - ax: Matplotlib axis object
        - x_mid, y_mid: Midpoint coordinates of the trajectory segment
        - dx, dy: Direction vector of the trajectory segment
        - color: Color of the arrowhead lines
        - length: Length of the arrowhead lines (default: 1.1)
        - angle: Angle at which the arrowhead lines deviate from the trajectory (default: 30)
        - shift_factor: Fraction of the segment length to move the arrowhead towards the end
        (default: -0.08)
        - linewidth: Line width of the arrowhead segments (default: 1.0)

        Returns:
        - None
        """

        # Convert the angle to radians
        angle_rad = np.radians(angle)
        dx = -dx
        dy = -dy

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dy**2)
        if norm != 0:  # Avoid division by zero
            dx /= norm
            dy /= norm

        # Move the midpoint slightly towards the second endpoint
        x_mid_shifted = x_mid + dx * shift_factor * norm
        y_mid_shifted = y_mid + dy * shift_factor * norm

        # Calculate the coordinates for the two arrowhead lines
        left_dx = dx * cos(angle_rad) - dy * sin(angle_rad)
        left_dy = dx * sin(angle_rad) + dy * cos(angle_rad)
        right_dx = dx * cos(-angle_rad) - dy * sin(-angle_rad)
        right_dy = dx * sin(-angle_rad) + dy * cos(-angle_rad)

        # Scale the direction vectors by the desired length of the arrowhead
        left_x = x_mid_shifted + left_dx * length
        left_y = y_mid_shifted + left_dy * length
        right_x = x_mid_shifted + right_dx * length
        right_y = y_mid_shifted + right_dy * length

        # Draw the two line segments that form the arrowhead
        ax.add_line(
            plt.Line2D(
                [x_mid_shifted, left_x],
                [y_mid_shifted, left_y],
                color=color,
                lw=linewidth,
                zorder=5,
            )
        )
        ax.add_line(
            plt.Line2D(
                [x_mid_shifted, right_x],
                [y_mid_shifted, right_y],
                color=color,
                lw=linewidth,
                zorder=5,
            )
        )

    def _draw_arrow_for_speed(
        self,
        i,
        x_start,
        x_end,
        y_start,
        y_end,
        last_arrow_idx,
        arrow_interval,
        speed,
        arrow_kwargs=None,
        arrow_color="black",
    ):
        """Draws an arrow if the conditions for speed are met."""
        if last_arrow_idx is None or i >= last_arrow_idx + arrow_interval:
            x_mid = (x_start + x_end) / 2
            y_mid = (y_start + y_end) / 2
            dx = x_end - x_start
            dy = y_end - y_start
            kw = arrow_kwargs or {}
            self.draw_custom_arrowhead(
                plt.gca(), x_mid, y_mid, dx, dy, arrow_color, **kw
            )
            return i  # Update last_arrow_idx
        return last_arrow_idx

    def plot_first_n_between_reward_training_segments(
        self,
        trn_index: int,
        *,
        first_n: int = 10,
        image_format: str | None = None,
        role_idx: int | None = None,
        zoom: bool = False,
        zoom_radius_mm: float | None = None,
        zoom_radius_mult: float = 3.0,
        out_dir: str | None = None,
    ):
        """
        Plot the first N between-reward trajectory segments within a training on a
        single arena plot, using a distinct color for each segment.
        """

        image_format = image_format or self.image_format
        first_n = max(1, int(first_n))

        if trn_index < 0 or trn_index >= len(self.va.trns):
            print(
                f"[plot_first_n_between_reward_training_segments] Invalid trn_index={trn_index}; "
                f"valid range is 0..{len(self.va.trns) - 1}"
            )
            return

        trn = self.va.trns[trn_index]
        f_idx = self.trj.f

        try:
            reward_frames = np.array(self.va._getOn(trn, calc=True, f=f_idx), dtype=int)
        except Exception as e:
            print(
                "[plot_first_n_between_reward_training_segments] "
                f"Error getting rewards for fly {f_idx}, training {trn_index + 1}: {e}"
            )
            return

        if reward_frames.size == 0:
            print(
                "[plot_first_n_between_reward_training_segments] "
                f"No rewards for fly {f_idx}, training {trn_index + 1}"
            )
            return

        in_training = (reward_frames > int(trn.start)) & (reward_frames <= int(trn.stop))
        training_rewards = np.array(reward_frames[in_training], dtype=int)
        training_rewards.sort()

        if training_rewards.size < 2:
            print(
                "[plot_first_n_between_reward_training_segments] "
                f"Not enough rewards in training {trn_index + 1} for fly {f_idx} "
                f"(found {training_rewards.size})"
            )
            return

        reward_pairs = list(zip(training_rewards[:-1], training_rewards[1:]))
        selected_pairs = reward_pairs[:first_n]
        if not selected_pairs:
            print(
                "[plot_first_n_between_reward_training_segments] "
                f"No between-reward intervals found in training {trn_index + 1} for fly {f_idx}"
            )
            return

        n_frames = len(self.x)
        selected_segments = []
        for start_reward, end_reward in selected_pairs:
            start_frame = max(0, int(start_reward))
            end_frame = min(n_frames - 1, int(end_reward))
            if start_frame < end_frame:
                selected_segments.append((start_reward, end_reward, start_frame, end_frame))

        if not selected_segments:
            print(
                "[plot_first_n_between_reward_training_segments] "
                f"No valid frame ranges for fly {f_idx}, training {trn_index + 1}"
            )
            return

        # Conversion: px -> mm for floor coords
        px_per_mm = self.va.ct.pxPerMmFloor() * self.va.xf.fctr
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            px_per_mm = None

        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        reward_circle = None
        try:
            reward_circle = trn.circles(self.trj.f)[0]
        except Exception:
            reward_circle = None

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        def _ylim_is_inverted_for_full_view() -> bool:
            yA = bottom_right[1] - padding_y
            yB = top_left[1] + padding_y
            return yA > yB

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))
        plt.sca(ax)

        rect = patches.FancyBboxPatch(
            (top_left[0], top_left[1]),
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
            boxstyle="round,pad=0.05,rounding_size=2",
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)

        self._draw_sidewall_contact_region(
            lower_left_x=top_left[0],
            lower_left_y=top_left[1],
            top_left=top_left,
            bottom_right=bottom_right,
            contact_buffer_px=contact_buffer_px,
        )

        if reward_circle is not None:
            rcx, rcy, rcr = reward_circle
            rc_patch = plt.Circle(
                (rcx, rcy),
                rcr,
                color="lightgray",
                fill=False,
                linestyle="-",
                linewidth=1.5,
                zorder=3,
                label="Reward circle",
            )
            ax.add_patch(rc_patch)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        x0 = x1 = y0 = y1 = None
        if zoom and reward_circle is not None and px_per_mm is not None:
            rcx, rcy, rcr = reward_circle

            if zoom_radius_mm is not None:
                win_rad_px = float(zoom_radius_mm) * float(px_per_mm)
            else:
                win_rad_px = float(rcr) * float(zoom_radius_mult)

            win_rad_px = max(win_rad_px, float(rcr) * 1.25)

            floor_y_min = min(top_left[1], bottom_right[1])
            floor_y_max = max(top_left[1], bottom_right[1])
            y0 = max(floor_y_min, rcy - win_rad_px)
            y1 = min(floor_y_max, rcy + win_rad_px)

            floor_x_min = min(top_left[0], bottom_right[0])
            floor_x_max = max(top_left[0], bottom_right[0])
            x0 = max(floor_x_min, rcx - win_rad_px)
            x1 = min(floor_x_max, rcx + win_rad_px)

            if (x1 - x0) < 5 or (y1 - y0) < 5:
                ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
                ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)
                x0 = x1 = y0 = y1 = None
            else:
                ax.set_xlim(x0, x1)
                if _ylim_is_inverted_for_full_view():
                    ax.set_ylim(y1, y0)
                else:
                    ax.set_ylim(y0, y1)
        else:
            ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
            ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

        arrow_kwargs = {"length": 1.5, "linewidth": 1.0} if zoom else {
            "length": 3.0,
            "linewidth": 2.0,
        }

        cmap = plt.get_cmap("tab10")
        handles = []
        labels = []

        for idx, (start_reward, end_reward, start_frame, end_frame) in enumerate(
            selected_segments
        ):
            color = cmap(idx % 10)
            last_arrow_idx = None
            arrow_interval = 3

            for i in range(start_frame, end_frame):
                if (
                    np.isnan(self.x[i])
                    or np.isnan(self.y[i])
                    or np.isnan(self.x[i + 1])
                    or np.isnan(self.y[i + 1])
                ):
                    continue

                x_start, x_end = self.x[i], self.x[i + 1]
                y_start, y_end = self.y[i], self.y[i + 1]

                x_start = max(min(x_start, bottom_right[0]), top_left[0])
                x_end = max(min(x_end, bottom_right[0]), top_left[0])

                ax.plot(
                    [x_start, x_end],
                    [y_start, y_end],
                    color=color,
                    linewidth=1.3,
                    alpha=0.95,
                    zorder=4,
                )

                if getattr(self.trj, "walking", None) is not None and not self.trj.walking[i + 1]:
                    continue

                speed = np.hypot(x_end - x_start, y_end - y_start)
                try:
                    last_arrow_idx = self._draw_arrow_for_speed(
                        i,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        last_arrow_idx,
                        arrow_interval,
                        speed,
                        arrow_kwargs=arrow_kwargs,
                        arrow_color=color,
                    )
                except Exception:
                    pass

            ax.plot(
                self.x[start_reward],
                self.y[start_reward],
                marker="o",
                color=color,
                markerfacecolor="white",
                markersize=5,
                zorder=5,
            )
            ax.plot(
                self.x[end_reward],
                self.y[end_reward],
                marker="o",
                color=color,
                markersize=5,
                zorder=5,
            )

            handles.append(plt.Line2D([0], [0], color=color, lw=2))
            labels.append(f"seg {idx + 1}: {start_reward}->{end_reward}")

        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)

        fly_role = "exp" if role_idx == 0 else "yok"
        fig.suptitle(
            "First between-reward trajectories in training\n"
            f"{video_id}, fly {fly_idx}, {fly_role} | trn {trn_index + 1} | "
            f"first {len(selected_segments)} segments",
            fontsize=12,
        )

        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=8,
            )

        output_dir = out_dir or "imgs/between_rewards"
        output_path = (
            f"{output_dir}/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_first{len(selected_segments)}"
            f"{'_zoom' if zoom else ''}."
            f"{image_format}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.subplots_adjust(left=0.04, right=0.80, top=0.90, bottom=0.04)
        writeImage(output_path, format=image_format)
        plt.close(fig)

        print(
            "[plot_first_n_between_reward_training_segments] wrote "
            f"{output_path}"
        )

    def _draw_sidewall_contact_region(
        self, lower_left_x, lower_left_y, top_left, bottom_right, contact_buffer_px
    ):
        """Draws the sidewall contact region."""
        # Outer gray rectangle
        plt.gca().add_patch(
            patches.FancyBboxPatch(
                (lower_left_x, lower_left_y),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1,
                edgecolor="none",
                facecolor="gray",
                alpha=0.3,
                zorder=1,
            )
        )

        # Inner white rectangle
        plt.gca().add_patch(
            patches.FancyBboxPatch(
                (lower_left_x + contact_buffer_px, lower_left_y + contact_buffer_px),
                (bottom_right[0] - top_left[0]) - 2 * contact_buffer_px,
                (bottom_right[1] - top_left[1]) - 2 * contact_buffer_px,
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1,
                edgecolor="none",
                facecolor="white",
                zorder=2,
            )
        )

    def _draw_circle_overlays(self, radius_stats, cx=None, cy=None, trn_index=0):
        """Draw the analysis circle used for sharp-turn detection."""

        r_px = radius_stats["circle_radius_px"]
        r_mm = radius_stats.get("circle_radius_mm", None)
        if cx is None or cy is None:
            # fallback: reward circle center from training
            cx, cy, _ = self.va.trns[trn_index].circles(self.trj.f)[0]

        analysis_patch = plt.Circle(
            (cx, cy),
            r_px,
            color="black",
            fill=False,
            linestyle="--",
            linewidth=2,
            zorder=2,
            label=f"Analysis Circle (r={r_px:.1f}px)",
        )
        ax = plt.gca()
        ax.add_patch(analysis_patch)

        cx_trn, cy_trn, r_trn = self.va.trns[trn_index].circles(self.trj.f)[0]

        training_patch = plt.Circle(
            (cx_trn, cy_trn),
            r_trn,
            color="lightgray",
            fill=False,
            linestyle=":",
            linewidth=2,
            zorder=1,
            label=f"Training Circle (r={r_trn:.1f}px)",
        )
        ax.add_patch(training_patch)

        if r_mm is not None:
            plt.text(
                1.02,
                0.5,
                f"r = {r_mm:.1f} mm",
                transform=ax.transAxes,
                va="center",
                fontsize=12,
                color="black",
                rotation=90,
                zorder=3,
            )

    def _draw_wall_overlays(self, top_left, bottom_right, contact_buffer_px):
        if self.y_bounds is not None:
            plt.axhline(y=self.y_bounds[0], color="k", linestyle="--", zorder=3)
            plt.axhline(y=self.y_bounds[1], color="k", linestyle="--", zorder=3)

        self._draw_sidewall_contact_region(
            lower_left_x=top_left[0],
            lower_left_y=top_left[1],
            top_left=top_left,
            bottom_right=bottom_right,
            contact_buffer_px=contact_buffer_px,
        )

    def _setup_plot_and_axes(self, top_left, bottom_right, padding_x, padding_y):
        """Sets up the plot and axis properties."""

        # Draw the rounded floor box
        rect = patches.FancyBboxPatch(
            (top_left[0], top_left[1]),
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
            boxstyle="round,pad=0.05,rounding_size=2",
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            zorder=4,
        )
        plt.gca().add_patch(rect)
        plt.gca().axis("off")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
        plt.ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

    def _add_legend_entry(self, handles, labels, label, color):
        """Adds a new legend entry if it's not already present."""
        if label and label not in labels:
            labels.append(label)
            handles.append(plt.Line2D([0], [0], color=color, lw=2))

    def _plot_large_turn_event_chain(
        self,
        exits,
        trn_index,
        turning_idxs_filtered,
        turn_circle_index_mapping,
        rejection_reasons,
        plot_mode="all_types",
        start_frame=None,
        stop_frame=None,
        color_map=None,
        image_format=None,
    ):
        """
        Plots a trajectory for large turns with color-coded events, applying
        a mode to control the display of events and showing the heading angle at each node.

        Parameters:
        - exits: list of frame indices for all circle exits within the timeframe
        - trn_index: index of the training associated with the reward circle exits
        - turning_idxs_filtered: list of (start, end) indices of large turns
        - turn_circle_index_mapping: list of circle exit indices corresponding to large turns
        - rejection_reasons: dict of rejection reasons and start/end indices for rejected events
        - plot_mode: determines which frames to include in the plot:
                    'all_types' - show two large turns and all events between them.
                    'turn_plus_1' - show one large turn and one non-turn event following it.
        - start_frame: starting frame of the plot (optional)
        - stop_frame: stopping frame of the plot (optional)
        - color_map: a dictionary mapping rejection reasons to colors
        - image_format: format for saving the plot (default: "png")
        """

        image_format = image_format or self.image_format

        plt.figure(figsize=(12, 8))

        # Define default color_map if none is provided
        if color_map is None:
            color_map = {
                "no_event": "black",
                "large_turn": "red",
                "small_angle_reentry": "blue",
                "wall_contact": "orange",
                "too_little_walking": "purple",
                "low_displacement": "lime",
            }

        handles = []
        labels = []

        # Get the floor coordinates
        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        # Sidewall contact region
        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        self._draw_wall_overlays(top_left, bottom_right, contact_buffer_px)

        # Define the frame range
        if start_frame is None:
            start_frame = 0
        if stop_frame is None:
            stop_frame = len(self.x)

        # Retrieve the reward circle for the current training index
        if trn_index >= 0:
            reward_circle = self.va.trns[trn_index].circles(self.trj.f)[0]
            reward_circle_x, reward_circle_y, reward_circle_radius = reward_circle

            # Plot the reward circle on the figure
            reward_circle_patch = plt.Circle(
                (reward_circle_x, reward_circle_y),
                reward_circle_radius,
                color="lightgray",
                fill=False,
                linestyle="--",
                linewidth=2,
                zorder=2,
                label="Reward Circle",
            )
            plt.gca().add_patch(reward_circle_patch)

        # Set chain length based on mode
        chain_length = 2 if plot_mode == "all_types" else 1

        # Ensure there are enough large turns available
        if len(turning_idxs_filtered) >= chain_length:
            start_idx = random.randint(0, len(turning_idxs_filtered) - chain_length)
            selected_turns = turning_idxs_filtered[start_idx : start_idx + chain_length]
        else:
            return  # Not enough turns, exit early

        selected_events = []

        # For 'turn_plus_1' mode, ensure the next event after the selected turn is a non-turn event
        if plot_mode == "turn_plus_1":
            selected_turn = selected_turns[0]
            selected_events.append((selected_turn[0], selected_turn[1], "large_turn"))
            print("Added turn from", selected_turn[0], "to", selected_turn[1])

            # Find the next non-turn event after the selected turn
            next_event = None
            for idx, exit_frame in enumerate(exits):
                if exit_frame > selected_turn[1]:
                    if idx not in turn_circle_index_mapping:
                        rejection_reason, (event_start, event_end) = (
                            rejection_reasons.get(
                                idx, ("no_event", (exit_frame, exit_frame + 1))
                            )
                        )
                        next_event = (event_start, event_end, rejection_reason)
                        selected_events.append(next_event)
                        break
            if not next_event:
                return  # No non-turn event found after the selected turn, exit early

            print("The start frame is", start_frame)
            print("The selected turn start is", selected_turn[0])
            start_frame = max(start_frame, selected_turn[0] - 5)
            stop_frame = min(stop_frame, next_event[1] + 5)

        else:
            selected_turn = selected_turns[0]
            selected_events.append((selected_turn[0], selected_turn[1], "large_turn"))

            for idx, exit_frame in enumerate(exits):
                if exit_frame > selected_turn[1]:
                    if idx in turn_circle_index_mapping:
                        print("Exit frame:", exit_frame)
                        print("Index:", idx)
                        print("turn circle index mapping", turn_circle_index_mapping)
                        print(
                            "Length of index mapping:", len(turn_circle_index_mapping)
                        )
                        print(
                            "Length of turning indices filtered:",
                            len(turning_idxs_filtered),
                        )
                        next_turn_idx = turn_circle_index_mapping.index(idx)
                        print("next turn index:", next_turn_idx)
                        next_turn = turning_idxs_filtered[next_turn_idx]
                        print("next turn:", next_turn)
                        selected_events.append(
                            (next_turn[0], next_turn[1], "large_turn")
                        )
                        break
                    else:
                        rejection_reason, (event_start, event_end) = (
                            rejection_reasons.get(
                                idx, ("no_event", (exit_frame, exit_frame + 1))
                            )
                        )
                        selected_events.append(
                            (event_start, event_end, rejection_reason)
                        )

            start_frame = selected_turn[0] - 5
            print("selected events before stop frame:", selected_events)
            stop_frame = min(stop_frame, selected_events[-1][1] + 5)

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1
        self._setup_plot_and_axes(top_left, bottom_right, padding_x, padding_y)

        # Get the start of the first selected turn
        first_turn_start = selected_events[0][0]

        # Ensure we don't go out of bounds when including the five frames before the first event
        pre_turn_start = max(0, first_turn_start - 6)

        # Plot the trajectory leading up to the first event in simple black, no arrows
        plt.plot(
            self.x[pre_turn_start : first_turn_start + 1],
            self.y[pre_turn_start : first_turn_start + 1],
            color="black",
            linewidth=0.75,
            zorder=2,
        )

        last_turn_end = selected_events[-1][1]
        post_turn_end = min(len(self.x), last_turn_end + 5)

        # Plot the trajectory leading up to the first event in simple black, no arrows
        plt.plot(
            self.x[last_turn_end : post_turn_end + 1],
            self.y[last_turn_end : post_turn_end + 1],
            color="black",
            linewidth=0.75,
            zorder=2,
        )

        # Now plot the selected events
        for event_start, event_end, event_type in selected_events:
            if (
                np.isnan(self.x[event_start : event_end + 1]).any()
                or np.isnan(self.y[event_start : event_end + 1]).any()
            ):
                continue  # Skip events with invalid coordinates

            if event_type == "large_turn":
                color = color_map["large_turn"]
                label = "Large turn"
            else:
                color = color_map.get(event_type, "black")
                label = event_type.replace("_", " ").capitalize()

            self._add_legend_entry(handles, labels, label, color)

            plt.plot(
                self.x[event_start : event_end + 1],
                self.y[event_start : event_end + 1],
                color=color,
                zorder=3,
            )

            # Plot the portions between events in simple black, with small speed arrows
            for i in range(len(selected_events) - 1):
                # Define the end of the current event and the start of the next
                current_event_end = selected_events[i][1]
                next_event_start = selected_events[i + 1][0]

                # Plot the segment between events if there's a gap
                if current_event_end + 1 < next_event_start:
                    plt.plot(
                        self.x[current_event_end : next_event_start + 1],
                        self.y[current_event_end : next_event_start + 1],
                        color="black",
                        linewidth=0.75,
                        zorder=2,
                    )

                    # Draw small speed arrows for the mid-length of the segments
                    last_arrow_idx = None  # Reset arrow index for this gap segment
                    for j in range(current_event_end, next_event_start):
                        x_start = self.x[j]
                        x_end = self.x[j + 1]
                        y_start = self.y[j]
                        y_end = self.y[j + 1]

                        # Calculate speed
                        speed = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

                        # Only draw arrows if the fly is walking
                        if self.trj.walking[j + 1]:
                            last_arrow_idx = self._draw_arrow_for_speed(
                                j,
                                x_start,
                                x_end,
                                y_start,
                                y_end,
                                last_arrow_idx,
                                3,  # Adjust arrow interval as needed
                                speed,
                            )

            # Plot the trajectory and draw velocity angle arrows
            last_arrow_idx = None  # Reset arrow index for each event
            for i in range(event_start, event_end):
                x_start = self.x[i]
                x_end = self.x[i + 1]
                y_start = self.y[i]
                y_end = self.y[i + 1]

                # Plot trajectory

                # Draw velocity angle arrows for the mid-length of the segments
                speed = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
                if not self.trj.walking[i + 1]:
                    continue
                last_arrow_idx = self._draw_arrow_for_speed(
                    i,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    last_arrow_idx,
                    3,
                    speed,  # Adjust arrow_interval if needed
                )

            # Plot heading angle at each node between segments
            for i in range(event_start, event_end + 1):
                x_node = self.x[i]
                y_node = self.y[i]
                theta_deg = self.trj.theta[i]
                if not self.trj.walking[i + 1]:
                    continue

                # Convert angle from degrees to radians
                theta_rad = np.radians(theta_deg)

                # Subtle arrow size adjustments
                arrow_length = 3  # Reduce the length to make arrows smaller
                head_width = 1.5  # Subtler arrowhead
                head_length = 2.5  # Smaller arrowhead length

                # Calculate the arrow direction based on the heading angle
                dx = arrow_length * np.sin(theta_rad)
                dy = -arrow_length * np.cos(theta_rad)

                # Draw the heading direction as a subtle arrow with lighter color
                plt.arrow(
                    x_node,
                    y_node,
                    dx,
                    dy,
                    head_width=head_width,
                    head_length=head_length,
                    fc="gray",  # Lighter color for subtler arrows
                    ec="gray",  # Edge color also lighter
                    linewidth=0.75,  # Thin arrow lines
                    zorder=4,  # Still keep arrows on top of other plot elements
                )

        plt.title(
            f"Large Turn Events with Heading Angles, Frames {start_frame} to {stop_frame}"
        )

        plt.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.045),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

        output_path = f"imgs/large_turn_plot_with_heading_f{self.trj.f}_{start_frame}_{stop_frame}.{image_format}"
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        writeImage(output_path, format=image_format)
        plt.close()

    def plot_between_reward_interval(
        self,
        trn_index: int,
        start_reward: int,
        end_reward: int,
        *,
        seed: int | None = None,
        image_format: str | None = None,
        role_idx: int | None = None,
        pad: int = 5,
        zoom: bool = False,
        zoom_radius_mm: float | None = None,
        zoom_radius_mult: float = 3.0,
        max_dist_mm: float | None = None,
        short_strict: bool = False,
        out_path: str | None = None,
        title_suffix: str = "",
    ):
        """
        Plot exactly one between-reward trajectory segment for this fly, defined by
        (start_reward -> end_reward). Intended for debugging / explainability.

        Parameters
        ----------
        trn_index : int
            0-based index into va.trns.
        start_reward, end_reward : int
            Frame indices of the two successive reward events.
        seed : int | None
            Included for interface consistency (not used for selection).
        image_format : str | None
            Output image format. Defaults to self.image_format.
        role_idx : int | None
            Role index (0 exp / 1 yok). If None, will attempt to infer via va.flies.index(trj.f).
        pad : int
            Frames of padding on each side of the interval for plotting context.
        zoom : bool
            If True, zoom around the reward circle.
        zoom_radius_mm : float | None
            If provided, zoom window radius in mm. Otherwise use zoom_radius_mult * reward_radius.
        zoom_radius_mult : float
            Multiplier on reward radius (in px) for default zoom window.
        max_dist_mm : float | None
            If provided, compute segment distance (mm) and optionally reject if > max_dist_mm.
            (Mostly useful if you want to keep this consistent with other filters.)
        short_strict : bool
            If True and max_dist_mm is provided and the segment is too long, skip output.
        out_path : str | None
            If provided, write exactly here. Otherwise uses the standard imgs/between_rewards/ pattern.
        title_suffix : str
            Extra text appended to per-figure title (useful for tagging q / group / etc.).
        """

        image_format = image_format or self.image_format

        # --- Basic checks ----------------------------------------------------------
        if trn_index < 0 or trn_index >= len(self.va.trns):
            print(
                f"[plot_between_reward_interval] Invalid trn_index={trn_index}; "
                f"valid range is 0..{len(self.va.trns) - 1}"
            )
            return

        n_frames = len(self.x)
        sr = int(start_reward)
        er = int(end_reward)

        if sr < 0 or er < 0 or sr >= n_frames or er >= n_frames:
            print(
                f"[plot_between_reward_interval] Reward frames out of bounds "
                f"(start={sr}, end={er}, n_frames={n_frames})."
            )
            return
        if er <= sr:
            print(
                f"[plot_between_reward_interval] Invalid interval: end_reward ({er}) "
                f"must be > start_reward ({sr})."
            )
            return

        # Conversion: px -> mm for floor coords
        px_per_mm = self.va.ct.pxPerMmFloor() * self.va.xf.fctr
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            px_per_mm = None

        def _segment_dist_mm(start_reward_i: int, end_reward_i: int) -> float:
            start_frame_i = max(0, int(start_reward_i))
            end_frame_i = min(n_frames - 1, int(end_reward_i))
            if end_frame_i <= start_frame_i:
                return np.nan
            d_px = self.trj.distTrav(start_frame_i, end_frame_i)
            if px_per_mm is None or not np.isfinite(d_px):
                return np.nan
            return float(d_px) / float(px_per_mm)

        # Optional max-dist filter (debug consistency)
        dmm = _segment_dist_mm(sr, er) if (max_dist_mm is not None) else np.nan
        if max_dist_mm is not None:
            max_dist_mm = float(max_dist_mm)
            if np.isfinite(dmm) and dmm > max_dist_mm:
                msg = (
                    f"[plot_between_reward_interval] Segment dist {dmm:.2f} mm exceeds "
                    f"max_dist_mm={max_dist_mm:g} (fly {self.trj.f}, trn {trn_index + 1})."
                )
                if short_strict:
                    print(msg + " short_strict=True; skipping.")
                    return
                print(msg + " Proceeding anyway (short_strict=False).")

        # Compute plotted window
        start_frame = max(0, sr - int(pad))
        end_frame = min(n_frames - 1, er + int(pad))
        if start_frame >= end_frame:
            print(
                f"[plot_between_reward_interval] Collapsed plotted window "
                f"({start_frame}..{end_frame}); skipping."
            )
            return

        # --- Arena / floor geometry ------------------------------------------------
        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        reward_circle = None
        try:
            reward_circle = self.va.trns[trn_index].circles(self.trj.f)[0]
        except Exception:
            reward_circle = None

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        def _ylim_is_inverted_for_full_view() -> bool:
            yA = bottom_right[1] - padding_y
            yB = top_left[1] + padding_y
            return yA > yB

        # Arrow styles (copied from your existing method)
        arrow_kwargs_default = {"length": 3.0, "linewidth": 2.0}
        arrow_kwargs_zoomed = {"length": 1.5, "linewidth": 1.0}

        def _choose_arrow_kwargs_for_view(x0, x1, y0, y1) -> dict:
            floor_w = float(abs(bottom_right[0] - top_left[0]))
            floor_h = float(abs(top_left[1] - bottom_right[1]))
            if floor_w <= 0 or floor_h <= 0:
                return arrow_kwargs_default
            win_w = float(abs(x1 - x0))
            win_h = float(abs(y1 - y0))
            frac = max(win_w / floor_w, win_h / floor_h)
            return arrow_kwargs_zoomed if frac <= 0.60 else arrow_kwargs_default

        # --- Figure ----------------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
        plt.sca(ax)

        # Floor box
        rect = patches.FancyBboxPatch(
            (top_left[0], top_left[1]),
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
            boxstyle="round,pad=0.05,rounding_size=2",
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)

        # Sidewall contact region
        try:
            self._draw_sidewall_contact_region(
                lower_left_x=top_left[0],
                lower_left_y=top_left[1],
                top_left=top_left,
                bottom_right=bottom_right,
                contact_buffer_px=contact_buffer_px,
            )
        except Exception as e:
            print(
                f"[plot_between_reward_interval] Warning: failed to draw contact region: {e}"
            )

        # Reward circle
        if reward_circle is not None:
            rcx, rcy, rcr = reward_circle
            rc_patch = plt.Circle(
                (rcx, rcy),
                rcr,
                color="lightgray",
                fill=False,
                linestyle="-",
                linewidth=1.5,
                zorder=3,
                label="Reward circle",
            )
            ax.add_patch(rc_patch)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Viewport (zoom or full)
        x0 = x1 = y0 = y1 = None  # define for arrow style logic
        if zoom and reward_circle is not None and px_per_mm is not None:
            rcx, rcy, rcr = reward_circle

            if zoom_radius_mm is not None:
                win_rad_px = float(zoom_radius_mm) * float(px_per_mm)
            else:
                win_rad_px = float(rcr) * float(zoom_radius_mult)

            win_rad_px = max(win_rad_px, float(rcr) * 1.25)

            floor_y_min = min(top_left[1], bottom_right[1])
            floor_y_max = max(top_left[1], bottom_right[1])
            y0 = max(floor_y_min, rcy - win_rad_px)
            y1 = min(floor_y_max, rcy + win_rad_px)

            floor_x_min = min(top_left[0], bottom_right[0])
            floor_x_max = max(top_left[0], bottom_right[0])
            x0 = max(floor_x_min, rcx - win_rad_px)
            x1 = min(floor_x_max, rcx + win_rad_px)

            if (x1 - x0) < 5 or (y1 - y0) < 5:
                ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
                ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)
                x0 = x1 = y0 = y1 = None
            else:
                ax.set_xlim(x0, x1)
                if _ylim_is_inverted_for_full_view():
                    ax.set_ylim(y1, y0)
                else:
                    ax.set_ylim(y0, y1)

                eps = 0.01
                ax.add_patch(
                    patches.Rectangle(
                        (2 * eps, 2 * eps),
                        1 - 4 * eps,
                        1 - 4 * eps,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=1.0,
                        linestyle="--",
                        edgecolor="0.6",
                        zorder=10,
                    )
                )
                ax.text(
                    0.03,
                    0.97,
                    "zoom",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="0.35",
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
        else:
            ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
            ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

        # Choose arrow kwargs based on zoomed viewport
        if (
            zoom
            and reward_circle is not None
            and px_per_mm is not None
            and x0 is not None
            and x1 is not None
            and y0 is not None
            and y1 is not None
            and (x1 - x0) > 5
            and (y1 - y0) > 5
        ):
            arrow_kwargs = _choose_arrow_kwargs_for_view(x0, x1, y0, y1)
        else:
            arrow_kwargs = arrow_kwargs_default

        # --- Draw trajectory segment ------------------------------------------------
        last_arrow_idx = None
        arrow_interval = 3

        for i in range(start_frame, end_frame):
            if (
                np.isnan(self.x[i])
                or np.isnan(self.y[i])
                or np.isnan(self.x[i + 1])
                or np.isnan(self.y[i + 1])
            ):
                continue

            x_start, x_end = self.x[i], self.x[i + 1]
            y_start, y_end = self.y[i], self.y[i + 1]

            # clamp x to floor bounds like existing code
            x_start = max(min(x_start, bottom_right[0]), top_left[0])
            x_end = max(min(x_end, bottom_right[0]), top_left[0])

            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color="black",
                linewidth=0.75,
                zorder=3,
            )

            if getattr(self.trj, "walking", None) is not None:
                if not self.trj.walking[i + 1]:
                    continue

            speed = np.hypot(x_end - x_start, y_end - y_start)
            try:
                last_arrow_idx = self._draw_arrow_for_speed(
                    i,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    last_arrow_idx,
                    arrow_interval,
                    speed,
                    arrow_kwargs=arrow_kwargs,
                )
            except Exception:
                # if arrow helper isn't available / fails, just skip arrows
                pass

        # Mark the two reward frames
        ax.plot(
            self.x[sr],
            self.y[sr],
            marker="o",
            color="green",
            markersize=7,
            zorder=4,
            label="Reward (start)",
        )
        ax.plot(
            self.x[er],
            self.y[er],
            marker="o",
            color="red",
            markersize=7,
            zorder=4,
            label="Reward (end)",
        )

        # --- Titles / legend --------------------------------------------------------
        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)

        fly_role = "exp" if role_idx == 0 else "yok"

        dist_line = ""
        if px_per_mm is not None:
            dmm2 = _segment_dist_mm(sr, er)
            if np.isfinite(dmm2):
                dist_line = f", dist {dmm2:.2f} mm"

        suffix = f" {title_suffix}".rstrip()
        global_title = (
            "Between-reward trajectory (selected interval)\n"
            f"{video_id}, fly {fly_idx}, {fly_role} | trn {trn_index + 1}\n"
            f"rewards {sr}->{er} (frames {start_frame}-{end_frame}){dist_line}{suffix}"
        )
        fig.suptitle(global_title, fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                fancybox=True,
                shadow=True,
                ncol=3,
                fontsize=9,
            )

        fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.16)

        # --- Output path ------------------------------------------------------------
        if out_path is None:
            seed_str = f"{seed}" if seed is not None else "na"
            zoom_str = ""
            if zoom:
                if zoom_radius_mm is not None:
                    zoom_str = f"_zoom{float(zoom_radius_mm):g}mm"
                else:
                    zoom_str = f"_zoomx{float(zoom_radius_mult):g}"

            out_path = (
                f"imgs/between_rewards/"
                f"{video_id}__fly{fly_idx}_role{role_idx}_"
                f"trn{trn_index + 1}_"
                f"rw{sr}-{er}_pad{int(pad)}_seed{seed_str}"
                f"{zoom_str}."
                f"{image_format}"
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writeImage(out_path, format=image_format)
        plt.close(fig)

        print(f"[plot_between_reward_interval] wrote {out_path}")

    def plot_between_reward_chain(
        self,
        trn_index,
        bucket_index,
        seed=None,
        image_format=None,
        role_idx=None,
        num_examples=1,
        max_dist_mm: float | None = None,
        short_strict: bool = False,
        zoom: bool = False,
        zoom_radius_mm: float | None = None,
        zoom_radius_mult: float = 3.0,
        schematic_metric: str | None = None,
        out_dir: str | None = None,
    ):
        """
        Plot one or more between-reward trajectory segments for this fly, sampled
        randomly within the specified training and sync bucket.

        Parameters
        ----------
        trn_index : int
            0-based index of the training in va.trns.
        bucket_index : int
            0-based index of the sync bucket within the training.
        seed : int or None
            Random seed for reproducible selection of between-reward segments.
        image_format : str or None
            Image format for output (defaults to self.image_format).
        role_idx : int or None
            Experimental role index for this fly (e.g., 0 = experimental,
            1 = yoked control). If None, this method will attempt to infer it
            via self.va.flies.index(self.trj.f).
        num_examples : int
            Maximum number of between-reward segments to show as subplots.
        """

        image_format = image_format or self.image_format
        num_examples = max(1, int(num_examples))
        metric_key = str(schematic_metric or "").lower()

        # --- Basic safety checks ------------------------------------------------
        if trn_index < 0 or trn_index >= len(self.va.trns):
            print(
                f"[plot_between_reward_chain] Invalid trn_index={trn_index}; "
                f"valid range is 0..{len(self.va.trns) - 1}"
            )
            return

        if not hasattr(self.va, "buckets") or trn_index >= len(self.va.buckets):
            print(
                f"[plot_between_reward_chain] No buckets info for trn_index={trn_index}"
            )
            return

        buckets = self.va.buckets[trn_index]
        if bucket_index < 0 or bucket_index >= len(buckets) - 1:
            print(
                f"[plot_between_reward_chain] Invalid bucket_index={bucket_index}; "
                f"valid range is 0..{len(buckets) - 2}"
            )
            return

        bkt_start = buckets[bucket_index]
        bkt_end = buckets[bucket_index + 1]

        # --- Get reward frame indices for this training and fly -----------------
        trn = self.va.trns[trn_index]
        f_idx = self.trj.f

        try:
            reward_frames = np.array(self.va._getOn(trn, calc=True, f=f_idx), dtype=int)
        except Exception as e:
            print(
                f"[plot_between_reward_chain] Error getting rewards for fly {f_idx}, "
                f"training {trn_index}: {e}"
            )
            return

        if reward_frames.size == 0:
            print(
                f"[plot_between_reward_chain] No rewards for fly {f_idx}, "
                f"training {trn_index + 1}"
            )
            return

        in_bucket = (reward_frames > bkt_start) & (reward_frames <= bkt_end)
        bucket_rewards = reward_frames[in_bucket]

        if bucket_rewards.size < 2:
            print(
                f"[plot_between_reward_chain] Not enough rewards in bucket "
                f"{bucket_index + 1} for fly {f_idx}, training {trn_index + 1} "
                f"(found {bucket_rewards.size})"
            )
            return

        bucket_rewards.sort()
        reward_pairs = list(zip(bucket_rewards[:-1], bucket_rewards[1:]))

        if not reward_pairs:
            print(
                f"[plot_between_reward_chain] No between-reward intervals found "
                f"in bucket {bucket_index + 1} for fly {f_idx}, training {trn_index + 1}"
            )
            return

        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        reward_circle = None
        if trn_index >= 0:
            try:
                reward_circle = self.va.trns[trn_index].circles(self.trj.f)[0]
            except Exception:
                reward_circle = None

        if metric_key in (
            "commag_synth",
            "commag_synth_vector_mean",
            "commag_synth_mean_magnitude",
        ):
            if metric_key == "commag_synth_mean_magnitude":
                commag_agg_mode = "mean_magnitude"
            elif metric_key == "commag_synth_vector_mean":
                commag_agg_mode = "vector_mean"
            else:
                commag_agg_mode = str(
                    getattr(getattr(self.va, "opts", None), "com_per_segment_agg", "vector_mean")
                ).strip().lower()
            self._plot_between_reward_commag_schematic(
                trn_index=trn_index,
                bucket_index=bucket_index,
                role_idx=role_idx,
                image_format=image_format,
                out_dir=out_dir,
                seed=seed,
                reward_circle=reward_circle,
                floor_coords=floor_coords,
                agg_mode=commag_agg_mode,
            )
            return

        key = (trn_index, bucket_index)
        used_pairs = self._used_between_reward_pairs.setdefault(key, set())

        # Only consider unused intervals
        candidate_pairs = [p for p in reward_pairs if p not in used_pairs]
        if not candidate_pairs:
            print(
                f"[plot_between_reward_chain] All between-reward intervals already "
                f"used for fly {f_idx}, training {trn_index + 1}, bucket {bucket_index + 1}; "
                f"no unique intervals left to plot."
            )
            return

        rng = random.Random(seed) if seed is not None else random

        pad = 5
        n_frames = len(self.x)

        # Conversion: px -> mm for floor coords
        px_per_mm = self.va.ct.pxPerMmFloor() * self.va.xf.fctr
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            px_per_mm = None

        def _segment_dist_mm(start_reward: int, end_reward: int) -> float:
            # Distance over the between-reward interval itself (no padding).
            # This matches the semantic "segment length" rather than the plotted window.
            start_frame = max(0, int(start_reward))
            end_frame = min(n_frames - 1, int(end_reward))
            if end_frame <= start_frame:
                return np.nan
            d_px = self.trj.distTrav(start_frame, end_frame)
            if px_per_mm is None or not np.isfinite(d_px):
                return np.nan
            return float(d_px) / float(px_per_mm)

        filtered_pairs = candidate_pairs
        if max_dist_mm is not None:
            max_dist_mm = float(max_dist_mm)
            tmp = []
            for p in candidate_pairs:
                dmm = _segment_dist_mm(p[0], p[1])
                if np.isfinite(dmm) and dmm <= max_dist_mm:
                    tmp.append(p)

            if not tmp:
                msg = (
                    f"[plot_between_reward_chain] No between-reward segments under "
                    f"{max_dist_mm:g} mm for fly {f_idx}, trn {trn_index + 1}, "
                    f"bucket {bucket_index + 1} (candidates={len(candidate_pairs)})."
                )
                if short_strict:
                    print(msg + " short_strict=True; skipping.")
                    return
                print(msg + " Falling back to full candidate pool.")
            else:
                filtered_pairs = tmp

        rng.shuffle(filtered_pairs)

        # --- Select up to num_examples valid segments ---------------------------
        selected_segments = []
        for start_reward, end_reward in filtered_pairs:
            start_frame = max(0, start_reward - pad)
            end_frame = min(n_frames - 1, end_reward + pad)
            if start_frame < end_frame:
                selected_segments.append(
                    (start_reward, end_reward, start_frame, end_frame)
                )
                used_pairs.add((start_reward, end_reward))
                if len(selected_segments) >= num_examples:
                    break

        if not selected_segments:
            print(
                f"[plot_between_reward_chain] No valid between-reward frame ranges "
                f"found after trying all unused intervals for fly {f_idx}, "
                f"training {trn_index + 1}, bucket {bucket_index + 1}."
            )
            return

        if len(selected_segments) < num_examples:
            print(
                f"[plot_between_reward_chain] Requested {num_examples} examples but only "
                f"found {len(selected_segments)} unique intervals for fly {f_idx}, "
                f"training {trn_index + 1}, bucket {bucket_index + 1}."
            )

        n_examples = len(selected_segments)
        synth_master_rng = random.Random(seed) if seed is not None else random.Random()

        # --- Prepare arena / floor and overlays shared across subplots ----------
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        # --- Figure + axes grid -------------------------------------------------
        n_cols = min(5, n_examples)
        n_rows = int(math.ceil(n_examples / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        # Arrow styles
        arrow_kwargs_default = {
            "length": 3.0,
            "linewidth": 2.0,
        }
        arrow_kwargs_zoomed = {
            "length": 1.5,
            "linewidth": 1.0,
        }

        def _choose_arrow_kwargs_for_view(x0, x1, y0, y1) -> dict:
            """
            Choose arrow style based on how zoomed-in the current viewport is.
            Uses the effective window size (after clamping) relative to the full floor size.
            """
            floor_w = float(abs(bottom_right[0] - top_left[0]))
            floor_h = float(abs(top_left[1] - bottom_right[1]))
            if floor_w <= 0 or floor_h <= 0:
                return arrow_kwargs_default

            win_w = float(abs(x1 - x0))
            win_h = float(abs(y1 - y0))
            frac = max(win_w / floor_w, win_h / floor_h)

            # "Big enough zoom" == sufficiently tight viewport (close-up).
            return arrow_kwargs_zoomed if frac <= 0.60 else arrow_kwargs_default

        def _ylim_is_inverted_for_full_view() -> bool:
            yA = bottom_right[1] - padding_y
            yB = top_left[1] + padding_y
            return yA > yB  # Matplotlib interprets this as an inverted y-axis

        # --- Plot each selected segment in its own subplot ----------------------
        for idx, (start_reward, end_reward, start_frame, end_frame) in enumerate(
            selected_segments
        ):
            ax = axes[idx]
            plt.sca(ax)

            # Floor box
            rect = patches.FancyBboxPatch(
                (top_left[0], top_left[1]),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1,
                edgecolor="black",
                facecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

            # Sidewall contact region
            self._draw_sidewall_contact_region(
                lower_left_x=top_left[0],
                lower_left_y=top_left[1],
                top_left=top_left,
                bottom_right=bottom_right,
                contact_buffer_px=contact_buffer_px,
            )

            # Reward circle patch (slightly thinner, solid)
            if reward_circle is not None:
                rcx, rcy, rcr = reward_circle
                rc_patch = plt.Circle(
                    (rcx, rcy),
                    rcr,
                    color="lightgray",
                    fill=False,
                    linestyle="-",
                    linewidth=1.5,
                    zorder=3,
                    label="Reward circle",
                )
                ax.add_patch(rc_patch)

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            if zoom and reward_circle is not None and px_per_mm is not None:
                rcx, rcy, rcr = reward_circle

                if zoom_radius_mm is not None:
                    win_rad_px = float(zoom_radius_mm) * float(px_per_mm)
                else:
                    win_rad_px = float(rcr) * float(zoom_radius_mult)

                # Safety floor: avoid absurdly tiny windows
                win_rad_px = max(win_rad_px, float(rcr) * 1.25)

                # Clamp to floor rectangle so we don't zoom outside the arena too much
                floor_y_min = min(top_left[1], bottom_right[1])
                floor_y_max = max(top_left[1], bottom_right[1])
                y0 = max(floor_y_min, rcy - win_rad_px)
                y1 = min(floor_y_max, rcy + win_rad_px)

                floor_x_min = min(top_left[0], bottom_right[0])
                floor_x_max = max(top_left[0], bottom_right[0])
                x0 = max(floor_x_min, rcx - win_rad_px)
                x1 = min(floor_x_max, rcx + win_rad_px)

                # If clamping collapsed the window, fall back to full view
                if (x1 - x0) < 5 or (y1 - y0) < 5:
                    ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
                    ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)
                else:
                    ax.set_xlim(x0, x1)
                    if _ylim_is_inverted_for_full_view():
                        ax.set_ylim(y1, y0)
                    else:
                        ax.set_ylim(y0, y1)

                eps = 0.01
                ax.add_patch(
                    patches.Rectangle(
                        (2 * eps, 2 * eps),
                        1 - 4 * eps,
                        1 - 4 * eps,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=1.0,
                        linestyle="--",
                        edgecolor="0.6",  # neutral gray
                        zorder=10,
                    )
                )
                ax.text(
                    0.03,
                    0.97,
                    "zoom",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="0.35",
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )

            else:
                ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
                ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

            if (
                zoom
                and reward_circle is not None
                and px_per_mm is not None
                and (x1 - x0) > 5
                and (y1 - y0) > 5
            ):
                arrow_kwargs = _choose_arrow_kwargs_for_view(x0, x1, y0, y1)
            else:
                arrow_kwargs = arrow_kwargs_default

            if metric_key == "maxdist_synth":
                synth_rng = random.Random(synth_master_rng.random() + idx)
                self._overlay_synthetic_maxdist_schematic(
                    ax,
                    reward_circle=reward_circle,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    rng=synth_rng,
                    excursion_scale=(2.10 if not zoom else 1.0),
                    variant=idx,
                )
            elif metric_key == "return_leg_dist_synth":
                synth_rng = random.Random(synth_master_rng.random() + idx)
                self._overlay_synthetic_return_leg_dist_schematic(
                    ax,
                    reward_circle=reward_circle,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    rng=synth_rng,
                    excursion_scale=(2.10 if not zoom else 1.0),
                    variant=idx,
                )
            elif metric_key == "turnback_ratio_synth":
                synth_rng = random.Random(synth_master_rng.random() + idx)
                self._overlay_synthetic_turnback_ratio_schematic(
                    ax,
                    reward_circle=reward_circle,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    rng=synth_rng,
                    variant=idx,
                )
            else:
                # Base trajectory line + arrows
                last_arrow_idx = None
                arrow_interval = 3

                for i in range(start_frame, end_frame):
                    if (
                        np.isnan(self.x[i])
                        or np.isnan(self.y[i])
                        or np.isnan(self.x[i + 1])
                        or np.isnan(self.y[i + 1])
                    ):
                        continue

                    x_start, x_end = self.x[i], self.x[i + 1]
                    y_start, y_end = self.y[i], self.y[i + 1]

                    x_start = max(min(x_start, bottom_right[0]), top_left[0])
                    x_end = max(min(x_end, bottom_right[0]), top_left[0])

                    ax.plot(
                        [x_start, x_end],
                        [y_start, y_end],
                        color="black",
                        linewidth=0.75,
                        zorder=3,
                    )

                    if getattr(self.trj, "walking", None) is not None:
                        if not self.trj.walking[i + 1]:
                            continue

                    speed = np.hypot(x_end - x_start, y_end - y_start)
                    last_arrow_idx = self._draw_arrow_for_speed(
                        i,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        last_arrow_idx,
                        arrow_interval,
                        speed,
                        arrow_kwargs=arrow_kwargs,
                    )

                # Mark the two reward frames themselves
                start_y = self.y[start_reward]
                end_y = self.y[end_reward]
                ax.plot(
                    self.x[start_reward],
                    start_y,
                    marker="o",
                    color="green",
                    markersize=6,
                    zorder=4,
                    label="Reward (start)",
                )
                ax.plot(
                    self.x[end_reward],
                    end_y,
                    marker="o",
                    color="red",
                    markersize=6,
                    zorder=4,
                    label="Reward (end)",
                )

            if metric_key == "maxdist":
                self._overlay_maxdist_schematic(
                    ax,
                    reward_circle=reward_circle,
                    start_reward=start_reward,
                    end_reward=end_reward,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    px_per_mm=px_per_mm,
                    show_label=True,
                )
            elif metric_key == "return_leg_dist":
                self._overlay_return_leg_dist_schematic(
                    ax,
                    reward_circle=reward_circle,
                    start_reward=start_reward,
                    end_reward=end_reward,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    px_per_mm=px_per_mm,
                    show_label=True,
                )
            # Per-subplot title: just the varying info
            dmm = (
                _segment_dist_mm(start_reward, end_reward)
                if max_dist_mm is not None
                else None
            )
            dist_line = (
                f"\ndist {dmm:.2f} mm" if dmm is not None and np.isfinite(dmm) else ""
            )
            metric_line = (
                "\nmetric schematic: max distance"
                if metric_key == "maxdist"
                else (
                    "\nmetric schematic: synthetic max distance example"
                    if metric_key == "maxdist_synth"
                    else (
                        "\nmetric schematic: return-leg distance"
                        if metric_key == "return_leg_dist"
                        else (
                            "\nmetric schematic: synthetic return-leg distance example"
                            if metric_key == "return_leg_dist_synth"
                            else (
                                "\nmetric schematic: synthetic turnback-ratio example"
                                if metric_key == "turnback_ratio_synth"
                                else ""
                            )
                        )
                    )
                )
            )
            title_line = (
                f"seg {idx + 1}: synthetic example\nsymbolic trajectory"
                if metric_key
                in ("maxdist_synth", "return_leg_dist_synth", "turnback_ratio_synth")
                else (
                    f"seg {idx + 1}: frames {start_frame}-{end_frame}\n"
                    f"rewards {start_reward}->{end_reward}{dist_line}"
                )
            )
            ax.set_title(
                f"{title_line}{metric_line}",
                fontsize=9,
            )

            # Put a legend only on the first subplot to avoid clutter
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(0.7, -0.15),
                        fancybox=True,
                        shadow=True,
                        ncol=3,
                        fontsize=8,
                    )

        # Hide any unused axes
        for ax in axes[n_examples:]:
            ax.axis("off")

        # --- Build output filename + global title -------------------------------
        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)

        seed_str = f"{seed}" if seed is not None else "rand"
        fly_role = "exp" if role_idx == 0 else "yok"

        global_title = (
            (
                "Between-reward schematic examples\n"
                if str(schematic_metric or "").lower()
                in ("maxdist_synth", "return_leg_dist_synth", "turnback_ratio_synth")
                else "Between-reward trajectories\n"
            )
            + f"{video_id}, fly {fly_idx}, {fly_role}\n"
            + (
                f"trn {trn_index + 1}, bucket {bucket_index + 1} | synthetic Dmax example"
                if str(schematic_metric or "").lower() == "maxdist_synth"
                else (
                    f"trn {trn_index + 1}, bucket {bucket_index + 1} | synthetic return-leg example"
                    if str(schematic_metric or "").lower() == "return_leg_dist_synth"
                    else (
                        f"trn {trn_index + 1}, bucket {bucket_index + 1} | synthetic turnback-ratio example"
                        if str(schematic_metric or "").lower()
                        == "turnback_ratio_synth"
                        else f"trn {trn_index + 1}, bucket {bucket_index + 1}"
                    )
                )
            )
        )
        fig.suptitle(global_title, fontsize=12)

        short_str = f"_maxd{max_dist_mm:g}mm" if max_dist_mm is not None else ""
        schematic_str = (
            f"_schematic-{str(schematic_metric).lower()}"
            if str(schematic_metric or "").strip()
            else ""
        )
        zoom_str = ""
        if zoom:
            if zoom_radius_mm is not None:
                zoom_str = f"_zoom{zoom_radius_mm:g}mm"
            else:
                zoom_str = f"_zoomx{zoom_radius_mult:g}"

        output_dir = out_dir or "imgs/between_rewards"
        output_path = (
            f"{output_dir}/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_bkt{bucket_index + 1}_"
            f"N{n_examples}_seed{seed_str}"
            f"{short_str}{schematic_str}{zoom_str}."
            f"{image_format}"
        )
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        fig.subplots_adjust(
            left=0.04, right=0.98, top=0.9, bottom=0.20, wspace=0.01, hspace=0.25
        )
        writeImage(output_path, format=image_format)
        plt.close(fig)

    def plot_reward_return_chain(
        self,
        trn_index: int,
        bucket_index: int,
        *,
        return_delta_mm: float,
        reward_delta_mm: float,
        min_inside_return_frames: int = 1,
        border_width_mm: float = 0.1,
        exclude_wall_contact: bool = False,
        seed: Optional[int] = None,
        image_format: Optional[str] = None,
        role_idx: Optional[int] = None,
        num_examples: int = 1,
        include_failures: bool = False,
        pad_frames: int = 5,
    ) -> None:
        """
        Plot one or more reward-return trajectory segments for this fly, sampled
        within the specified training and sync bucket.

        A plotted segment spans:
          start_frame = (return-entry start) - pad_frames
          end_frame   = (reward_entry if success else episode stop) + pad_frames

        Parameters mirror Trajectory.reward_return_distance_episodes_for_training().
        """
        image_format = image_format or self.image_format
        num_examples = max(1, int(num_examples))
        pad_frames = max(0, int(pad_frames))

        if trn_index < 0 or trn_index >= len(getattr(self.va, "trns", [])):
            print(
                f"[plot_reward_return_chain] Invalid trn_index={trn_index}; "
                f"valid range is 0..{len(self.va.trns) - 1}"
            )
            return

        trn = self.va.trns[trn_index]
        if trn is None or not trn.isCircle():
            print(
                f"[plot_reward_return_chain] Training {trn_index + 1} is not a circle."
            )
            return

        bucket_range = self._get_bucket_range(
            trn_index=trn_index, bucket_index=bucket_index
        )
        if bucket_range is None:
            print(
                f"[plot_reward_return_chain] Invalid bucket_index={bucket_index} "
                f"for trn_index={trn_index}."
            )
            return
        bkt_start, bkt_end = bucket_range

        # wall-contact regions (optional)
        wall_regions = None
        if exclude_wall_contact:
            try:
                wall_regions = self.trj.boundary_event_stats["wall"]["all"]["edge"].get(
                    "boundary_contact_regions", None
                )
            except Exception:
                wall_regions = None

        episodes = self.trj.reward_return_distance_episodes_for_training(
            trn=trn,
            return_delta_mm=return_delta_mm,
            reward_delta_mm=reward_delta_mm,
            min_inside_return_frames=min_inside_return_frames,
            border_width_mm=border_width_mm,
            exclude_wall_contact=exclude_wall_contact,
            wall_contact_regions=wall_regions,
            debug=False,
        )
        if not episodes:
            print(
                f"[plot_reward_return_chain] No reward-return episodes for fly {self.trj.f}, "
                f"training {trn_index + 1}."
            )
            return

        # filter by bucket (episode start must fall inside bucket)
        eps_in_bucket = []
        for ep in episodes:
            s = int(ep["start"])
            if bkt_start <= s < bkt_end:
                if include_failures or ep.get("dist", None) is not None:
                    eps_in_bucket.append(ep)

        if not eps_in_bucket:
            mode = "incl failures" if include_failures else "success-only"
            print(
                f"[plot_reward_return_chain] No reward-return episodes in bucket "
                f"{bucket_index + 1} ({mode}) for fly {self.trj.f}, trn {trn_index + 1}."
            )
            return

        key = (trn_index, bucket_index)
        used = self._used_reward_return_episodes.setdefault(key, set())

        def _ep_key(ep) -> Tuple[int, int, str]:
            return (
                int(ep["start"]),
                int(ep.get("stop", -1)),
                str(ep.get("end_reason", "")),
            )

        candidates = [ep for ep in eps_in_bucket if _ep_key(ep) not in used]
        if not candidates:
            print(
                f"[plot_reward_return_chain] All reward-return episodes already used "
                f"for fly {self.trj.f}, trn {trn_index + 1}, bucket {bucket_index + 1}."
            )
            return

        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(candidates)

        n_frames = len(self.x)
        selected = []
        for ep in candidates:
            s_abs = int(ep["start"])
            end_abs = (
                int(ep["reward_entry"])
                if ep.get("reward_entry") is not None
                else int(ep["stop"])
            )
            start_frame = max(0, s_abs - pad_frames)
            end_frame = min(n_frames - 1, end_abs + pad_frames)
            if start_frame < end_frame:
                selected.append((ep, start_frame, end_frame, end_abs))
                used.add(_ep_key(ep))
                if len(selected) >= num_examples:
                    break

        if not selected:
            print(
                f"[plot_reward_return_chain] No valid frame ranges after padding for fly {self.trj.f}, "
                f"trn {trn_index + 1}, bucket {bucket_index + 1}."
            )
            return

        # Arena geometry (shared)
        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        reward_circle = None
        try:
            reward_circle = trn.circles(self.trj.f)[0]
        except Exception:
            reward_circle = None

        # return circle radius computation (px in same space as x/y)
        px_per_mm = float(self.va.ct.pxPerMmFloor()) * float(
            getattr(self.va.xf, "fctr", 1.0) or 1.0
        )
        return_circle = None
        if reward_circle is not None:
            rcx, rcy, rcr = reward_circle
            return_circle = (rcx, rcy, float(rcr) + float(return_delta_mm) * px_per_mm)
            reward_circle = (rcx, rcy, float(rcr) + float(reward_delta_mm) * px_per_mm)

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        n_examples = len(selected)
        n_cols = min(5, n_examples)
        n_rows = int(math.ceil(n_examples / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        big_arrow_kwargs = {"length": 3.0, "linewidth": 2.0}

        for idx, (ep, start_frame, end_frame, end_abs) in enumerate(selected):
            ax = axes[idx]
            plt.sca(ax)

            # Floor box
            rect = patches.FancyBboxPatch(
                (top_left[0], top_left[1]),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1,
                edgecolor="black",
                facecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

            # Sidewall contact region
            self._draw_sidewall_contact_region(
                lower_left_x=top_left[0],
                lower_left_y=top_left[1],
                top_left=top_left,
                bottom_right=bottom_right,
                contact_buffer_px=contact_buffer_px,
            )

            # Reward + Return circle overlays
            if reward_circle is not None:
                rcx, rcy, rcr = reward_circle
                ax.add_patch(
                    plt.Circle(
                        (rcx, rcy),
                        rcr,
                        color="lightgray",
                        fill=False,
                        linestyle="-",
                        linewidth=1.5,
                        zorder=3,
                        label="Reward circle",
                    )
                )
            if return_circle is not None:
                rx, ry, rr = return_circle
                ax.add_patch(
                    plt.Circle(
                        (rx, ry),
                        rr,
                        color="lightgray",
                        fill=False,
                        linestyle="--",
                        linewidth=1.2,
                        zorder=3,
                        label="Return circle",
                    )
                )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
            ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

            # Base trajectory line + arrows
            last_arrow_idx = None
            arrow_interval = 3

            for i in range(start_frame, end_frame):
                if (
                    np.isnan(self.x[i])
                    or np.isnan(self.y[i])
                    or np.isnan(self.x[i + 1])
                    or np.isnan(self.y[i + 1])
                ):
                    continue

                x_start, x_end = self.x[i], self.x[i + 1]
                y_start, y_end = self.y[i], self.y[i + 1]

                x_start = max(min(x_start, bottom_right[0]), top_left[0])
                x_end = max(min(x_end, bottom_right[0]), top_left[0])

                ax.plot([x_start, x_end], [y_start, y_end], linewidth=0.75, zorder=3)

                if getattr(self.trj, "walking", None) is not None:
                    if not self.trj.walking[i + 1]:
                        continue

                speed = np.hypot(x_end - x_start, y_end - y_start)
                last_arrow_idx = self._draw_arrow_for_speed(
                    i,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    last_arrow_idx,
                    arrow_interval,
                    speed,
                    arrow_kwargs=big_arrow_kwargs,
                )

            # Mark episode start (return-entry) and endpoint
            s_abs = int(ep["start"])
            s_ok = 0 <= s_abs < len(self.x)
            e_ok = 0 <= end_abs < len(self.x)

            if s_ok:
                ax.plot(
                    self.x[s_abs],
                    self.y[s_abs],
                    marker="o",
                    markersize=6,
                    zorder=4,
                    label="Return entry (start)",
                )

            success = (
                bool(ep.get("success", False)) and ep.get("reward_entry") is not None
            )
            if e_ok:
                ax.plot(
                    self.x[end_abs],
                    self.y[end_abs],
                    marker="o",
                    markersize=6,
                    zorder=4,
                    label=(
                        "Reward entry (end)"
                        if success
                        else "Episode end (exit/trn_end)"
                    ),
                )

            # Title per subplot
            end_reason = str(ep.get("end_reason", ""))
            ax.set_title(
                f"seg {idx + 1}: frames {start_frame}-{end_frame}\n"
                f"start {s_abs} → end {end_abs} ({end_reason})",
                fontsize=9,
            )

            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(0.7, -0.15),
                        fancybox=True,
                        shadow=True,
                        ncol=2,
                        fontsize=8,
                    )

        for ax in axes[n_examples:]:
            ax.axis("off")

        # output filename + title
        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)

        seed_str = f"{seed}" if seed is not None else "rand"
        fly_role = "exp" if role_idx == 0 else "yok"

        mode = "succ+fail" if include_failures else "succ"
        fig.suptitle(
            "Reward-return trajectories\n"
            f"{video_id}, fly {fly_idx}, {fly_role}\n"
            f"trn {trn_index + 1}, bucket {bucket_index + 1} ({mode})",
            fontsize=12,
        )

        output_path = (
            f"imgs/reward_return_distance/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_bkt{bucket_index + 1}_"
            f"N{n_examples}_seed{seed_str}_{mode}."
            f"{image_format}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.subplots_adjust(
            left=0.04, right=0.98, top=0.9, bottom=0.20, wspace=0.01, hspace=0.25
        )
        writeImage(output_path, format=image_format)
        plt.close(fig)

    def plot_return_prob_chain(
        self,
        trn_index: int,
        bucket_index: int,
        *,
        outer_delta_mm: float,
        reward_delta_mm: float = 0.0,
        border_width_mm: float = 0.1,
        seed: Optional[int] = None,
        image_format: Optional[str] = None,
        role_idx: Optional[int] = None,
        num_examples: int = 1,
        include_failures: bool = False,
        pad_frames: int = 5,
    ) -> None:
        """
        Plot one or more return-probability trajectory segments for this fly,
        sampled within the specified training and sync bucket.

        Episodes are sourced from Trajectory.reward_return_probability_episodes_for_training().
        Bucket assignment follows the exported metric: an episode belongs to a sync bucket
        according to its outcome frame (stop - 1).
        """
        image_format = image_format or self.image_format
        num_examples = max(1, int(num_examples))
        pad_frames = max(0, int(pad_frames))

        if trn_index < 0 or trn_index >= len(getattr(self.va, "trns", [])):
            print(
                f"[plot_return_prob_chain] Invalid trn_index={trn_index}; "
                f"valid range is 0..{len(self.va.trns) - 1}"
            )
            return

        trn = self.va.trns[trn_index]
        if trn is None or not trn.isCircle():
            print(
                f"[plot_return_prob_chain] Training {trn_index + 1} is not a circle."
            )
            return

        bucket_range = self._get_bucket_range(
            trn_index=trn_index, bucket_index=bucket_index
        )
        if bucket_range is None:
            print(
                f"[plot_return_prob_chain] Invalid bucket_index={bucket_index} "
                f"for trn_index={trn_index}."
            )
            return
        bkt_start, bkt_end = bucket_range

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = 0
        ctrl = bool(role_idx == 1)
        window_counts, window_label = self._return_prob_window_counts(
            trn_index=trn_index,
            outer_delta_mm=outer_delta_mm,
            reward_delta_mm=reward_delta_mm,
            border_width_mm=border_width_mm,
            ctrl=ctrl,
        )

        episodes = self.trj.reward_return_probability_episodes_for_training(
            trn=trn,
            outer_delta_mm=outer_delta_mm,
            reward_delta_mm=reward_delta_mm,
            border_width_mm=border_width_mm,
            ctrl=ctrl,
            debug=False,
        )
        if not episodes:
            print(
                f"[plot_return_prob_chain] No return-probability episodes for fly {self.trj.f}, "
                f"training {trn_index + 1}."
            )
            return

        eps_in_bucket = []
        for ep in episodes:
            event_t = int(ep["stop"]) - 1
            if bkt_start <= event_t < bkt_end:
                if include_failures or bool(ep.get("returns", False)):
                    eps_in_bucket.append(ep)

        if not eps_in_bucket:
            mode = "incl failures" if include_failures else "success-only"
            print(
                f"[plot_return_prob_chain] No return-probability episodes in bucket "
                f"{bucket_index + 1} ({mode}) for fly {self.trj.f}, trn {trn_index + 1}."
            )
            return

        key = (trn_index, bucket_index, float(outer_delta_mm), bool(ctrl))
        used = self._used_return_prob_episodes.setdefault(key, set())

        def _ep_key(ep) -> Tuple[int, int, str]:
            return (
                int(ep["start"]),
                int(ep.get("stop", -1)),
                str(ep.get("end_reason", "")),
            )

        candidates = [ep for ep in eps_in_bucket if _ep_key(ep) not in used]
        if not candidates:
            print(
                f"[plot_return_prob_chain] All return-probability episodes already used "
                f"for fly {self.trj.f}, trn {trn_index + 1}, bucket {bucket_index + 1}."
            )
            return

        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(candidates)

        n_frames = len(self.x)
        selected = []
        for ep in candidates:
            s_abs = int(ep["start"])
            end_abs = int(ep["stop"]) - 1
            start_frame = max(0, s_abs - pad_frames)
            end_frame = min(n_frames - 1, end_abs + pad_frames)
            if start_frame < end_frame:
                selected.append((ep, start_frame, end_frame, end_abs))
                used.add(_ep_key(ep))
                if len(selected) >= num_examples:
                    break

        if not selected:
            print(
                f"[plot_return_prob_chain] No valid frame ranges after padding for fly {self.trj.f}, "
                f"trn {trn_index + 1}, bucket {bucket_index + 1}."
            )
            return

        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        reward_circle = None
        try:
            reward_circle = trn.circles(self.trj.f)[0]
        except Exception:
            reward_circle = None

        px_per_mm = float(self.va.ct.pxPerMmFloor()) * float(
            getattr(self.va.xf, "fctr", 1.0) or 1.0
        )
        outer_circle = None
        if reward_circle is not None:
            rcx, rcy, rcr = reward_circle
            outer_circle = (rcx, rcy, float(rcr) + float(outer_delta_mm) * px_per_mm)
            reward_circle = (rcx, rcy, float(rcr) + float(reward_delta_mm) * px_per_mm)

        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        n_examples = len(selected)
        n_cols = min(5, n_examples)
        n_rows = int(math.ceil(n_examples / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        big_arrow_kwargs = {"length": 3.0, "linewidth": 2.0}

        for idx, (ep, start_frame, end_frame, end_abs) in enumerate(selected):
            ax = axes[idx]
            plt.sca(ax)

            rect = patches.FancyBboxPatch(
                (top_left[0], top_left[1]),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1,
                edgecolor="black",
                facecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

            self._draw_sidewall_contact_region(
                lower_left_x=top_left[0],
                lower_left_y=top_left[1],
                top_left=top_left,
                bottom_right=bottom_right,
                contact_buffer_px=contact_buffer_px,
            )

            if reward_circle is not None:
                rcx, rcy, rcr = reward_circle
                ax.add_patch(
                    plt.Circle(
                        (rcx, rcy),
                        rcr,
                        color="lightgray",
                        fill=False,
                        linestyle="-",
                        linewidth=1.5,
                        zorder=3,
                        label="Reward circle",
                    )
                )
            if outer_circle is not None:
                ox, oy, orad = outer_circle
                ax.add_patch(
                    plt.Circle(
                        (ox, oy),
                        orad,
                        color="lightgray",
                        fill=False,
                        linestyle="--",
                        linewidth=1.2,
                        zorder=3,
                        label="Outer circle",
                    )
                )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.set_xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
            ax.set_ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

            last_arrow_idx = None
            arrow_interval = 3

            for i in range(start_frame, end_frame):
                if (
                    np.isnan(self.x[i])
                    or np.isnan(self.y[i])
                    or np.isnan(self.x[i + 1])
                    or np.isnan(self.y[i + 1])
                ):
                    continue

                x_start, x_end = self.x[i], self.x[i + 1]
                y_start, y_end = self.y[i], self.y[i + 1]

                x_start = max(min(x_start, bottom_right[0]), top_left[0])
                x_end = max(min(x_end, bottom_right[0]), top_left[0])

                ax.plot([x_start, x_end], [y_start, y_end], linewidth=0.75, zorder=3)

                if getattr(self.trj, "walking", None) is not None:
                    if not self.trj.walking[i + 1]:
                        continue

                speed = np.hypot(x_end - x_start, y_end - y_start)
                last_arrow_idx = self._draw_arrow_for_speed(
                    i,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    last_arrow_idx,
                    arrow_interval,
                    speed,
                    arrow_kwargs=big_arrow_kwargs,
                )

            s_abs = int(ep["start"])
            s_ok = 0 <= s_abs < len(self.x)
            e_ok = 0 <= end_abs < len(self.x)

            if s_ok:
                ax.plot(
                    self.x[s_abs],
                    self.y[s_abs],
                    marker="o",
                    markersize=6,
                    zorder=4,
                    label="Reward exit (start)",
                )

            success = bool(ep.get("returns", False))
            if e_ok:
                ax.plot(
                    self.x[end_abs],
                    self.y[end_abs],
                    marker="o",
                    markersize=6,
                    zorder=4,
                    label=("Reward return" if success else "Outer-circle exit"),
                )

            end_reason = str(ep.get("end_reason", ""))
            ax.set_title(
                f"seg {idx + 1}: frames {start_frame}-{end_frame}\n"
                f"start {s_abs} → end {end_abs} ({end_reason})",
                fontsize=9,
            )

            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(0.7, -0.15),
                        fancybox=True,
                        shadow=True,
                        ncol=2,
                        fontsize=8,
                    )

        for ax in axes[n_examples:]:
            ax.axis("off")

        video_id = os.path.splitext(os.path.basename(self.va.fn))[0]
        fly_idx = self.va.f

        if role_idx is None:
            try:
                role_idx = self.va.flies.index(self.trj.f)
            except Exception:
                role_idx = int(self.trj.f)

        seed_str = f"{seed}" if seed is not None else "rand"
        fly_role = "exp" if role_idx == 0 else "yok"
        mode = "succ+fail" if include_failures else "succ"
        window_line = ""
        if window_counts is not None:
            succ_n, fail_n, total_n = window_counts
            window_line = (
                f"\nwindow {window_label}: succ={succ_n} fail={fail_n} total={total_n}"
            )
        fig.suptitle(
            "Return-probability trajectories\n"
            f"{video_id}, fly {fly_idx}, {fly_role}\n"
            f"trn {trn_index + 1}, bucket {bucket_index + 1}, outer+{float(outer_delta_mm):g} mm ({mode})"
            f"{window_line}",
            fontsize=12,
        )

        output_path = (
            f"imgs/return_probability/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_bkt{bucket_index + 1}_"
            f"outer{float(outer_delta_mm):g}mm_"
            f"N{n_examples}_seed{seed_str}_{mode}."
            f"{image_format}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.subplots_adjust(
            left=0.04, right=0.98, top=0.9, bottom=0.20, wspace=0.01, hspace=0.25
        )
        writeImage(output_path, format=image_format)
        plt.close(fig)

    def plot_sharp_turn_chain_wall(
        self,
        ellipse_ref_pt,
        bcr,
        turning_idxs,
        rejection_reasons,
        frames_to_skip,
        start_frame=None,
        mode="all_types",
        image_format=None,
    ):
        image_format = image_format or self.image_format

        def overlays(top_left, bottom_right, contact_buffer_px, _trn_index):
            self._draw_wall_overlays(top_left, bottom_right, contact_buffer_px)

        self._plot_event_chain_core(
            ellipse_ref_pt=ellipse_ref_pt,
            bcr=bcr,
            turning_idxs=turning_idxs,
            rejection_reasons=rejection_reasons or {},
            frames_to_skip=frames_to_skip or set(),
            start_frame=start_frame,
            mode=mode,
            image_format=image_format,
            overlays=overlays,
        )

    def plot_sharp_turn_chain_circle(
        self, radius_stats, trn_index, start_frame, mode, image_format=None
    ):
        image_format = image_format or self.image_format
        bcr = radius_stats["boundary_contact_regions"]
        turning_idxs = radius_stats["turning_indices"]
        rejection_reasons = radius_stats.get("rejection_reasons", {})
        frames_to_skip = radius_stats.get("frames_to_skip", set())

        cx, cy, _ = self.va.trns[trn_index].circles(self.trj.f)[0]

        def overlays(top_left, bottom_right, contact_buffer_px, _trn_index):
            self._draw_circle_overlays(radius_stats, cx=cx, cy=cy, trn_index=_trn_index)

        self._plot_event_chain_core(
            ellipse_ref_pt="ctr",
            bcr=bcr,
            turning_idxs=turning_idxs,
            rejection_reasons=rejection_reasons,
            frames_to_skip=frames_to_skip,
            start_frame=start_frame,
            mode=mode,
            image_format=image_format,
            overlays=overlays,
        )

    def _plot_event_chain_core(
        self,
        ellipse_ref_pt,
        bcr,
        turning_idxs,
        rejection_reasons,
        frames_to_skip,
        start_frame=None,
        mode="all_types",
        image_format=None,
        overlays=None,
        trn_index=-1,
    ):
        """
        Plots a chain of sharp turn events in a single figure, applying
        a color-coded scheme based on the trajectory's characteristics and
        adds arrows along the trajectory to indicate the direction of time progression.

        Parameters:
        - ellipse_ref_pt: reference point on the ellipse
        - bcr: list of boundary contact regions
        - turning_idxs_filtered: list of indices of sharp turns
        - rejection_reasons: list indexed identically to bcr, where each element provides
                             a categorical explanation or justification for why the
                             corresponding event wasn't classified as a sharp turn. It
                             serves as a detailed record of the specific criteria that
                             disqualified each event from being labeled as a turn.
        - frames_to_skip: list of indices of frames to skip (here, due to start of wall-contact event)
        - start_frame: optional starting frame to begin the search for events
        - mode: determines which frames to include in the plot:
                'all_types' - show two sharp turns along with all parts of the trajectory between them.
                'turn_plus_1' - show a single sharp turn and one non-turn event following it, with distinct colors.
        """
        image_format = image_format or self.image_format
        speed_threshold_high = 18
        speed_threshold_low = 6
        # Number of sharp turns to chain together
        if mode == "all_types":
            chain_length = 2
        elif mode == "turn_plus_1":
            chain_length = 1

        if start_frame is not None:
            # Find the first sharp turn after the start_frame
            start_idx = next(
                (
                    idx
                    for idx, turn_idx in enumerate(turning_idxs)
                    if bcr[turn_idx].start >= start_frame
                ),
                None,
            )

            if start_idx is None or start_idx + chain_length > len(turning_idxs):
                raise ValueError(
                    "No sufficient sharp turns found after the specified start_frame"
                )

            selected_turns = turning_idxs[start_idx : start_idx + chain_length]
        else:
            # Ensure there are enough sharp turns to select from
            if len(turning_idxs) < chain_length:
                chain_length = len(turning_idxs)

            # Randomly select a starting index
            start_idx = random.randint(0, len(turning_idxs) - chain_length)

            # Select the range of sharp turns
            selected_turns = turning_idxs[start_idx : start_idx + chain_length]

        # Define the start and end frames for the selected chain of turns
        start_frame = bcr[selected_turns[0]].start
        end_frame = bcr[selected_turns[-1]].stop

        if mode == "all_types":
            frames_range = range(
                max(0, start_frame - 5), min(len(self.x), end_frame + 5)
            )
        elif mode == "turn_plus_1":
            # Find the first event after the selected sharp turn that is not a sharp turn
            next_event_idx = None
            for i in range(selected_turns[-1] + 1, len(bcr)):
                if i not in turning_idxs:
                    next_event_idx = i
                    break
                else:
                    return

            # If a non-turn event is found, update end_frame to include it
            if next_event_idx is not None:
                end_frame = bcr[next_event_idx].stop

                # Include all frames between the sharp turn and the following non-turn event
                frames_range = range(
                    max(0, start_frame - 5), min(len(self.x), end_frame + 5)
                )
            else:
                # If no non-turn event is found, skip this segment
                return

        plt.figure(figsize=(12, 8))

        arrow_interval = 3  # Interval between arrows

        # Get the floor coordinates
        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]
        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        if overlays:
            overlays(top_left, bottom_right, contact_buffer_px, trn_index)

        # For the legend, we'll collect the labels and colors used
        handles = []
        labels = []

        last_arrow_idx = None

        # Initialize the count for successive slow frames
        successive_slow_frames = 0
        max_slow_frames = 15  # Adjust this value based on your preference
        current_bcr_index = None

        frames_to_mark = []

        for i in frames_range:
            if (
                np.isnan(self.x[i])
                or self.x[i] == 0
                or np.isnan(self.y[i])
                or self.y[i] == 0
            ):
                continue

            # Initialize defaults
            color = "black"
            label = None
            rejection_reason = None

            # Determine if the current frame is part of a sharp turn
            is_turn = any(bcr[j].start - 1 <= i < bcr[j].stop for j in selected_turns)

            # Mode-specific logic for color and label
            if mode == "turn_plus_1":
                if is_turn:
                    color = "red"
                    label = "Sharp turn"
                else:
                    is_event = any(
                        (
                            bcr[j].start - 1 <= i < bcr[j].stop
                            and bcr[j].start - 1 >= frames_range.start
                            and bcr[j].stop <= frames_range.stop
                        )
                        for j in range(len(bcr))
                        if j not in selected_turns
                    )
                    if is_event:
                        color = "blue"
                        label = "Boundary crossing w/out sharp turn"

            elif mode == "all_types":
                if is_turn:
                    color = "red"
                    label = "Sharp turn"
                else:
                    for j in range(len(bcr)):
                        if (
                            bcr[j].start - 1 <= i < bcr[j].stop
                            and bcr[j].start - 1 >= frames_range.start
                            and bcr[j].stop <= frames_range.stop
                        ):
                            rejection_reason = rejection_reasons[j]
                            current_bcr_index = j
                            color_label_map = {
                                "too_long": ("blue", "Duration > 0.75 s"),
                                "too_little_velocity_angle_change": (
                                    "orange",
                                    "Sum of vel. angle deltas < 90°",
                                ),
                                "sidewall_contact": ("purple", "Sidewall contact"),
                            }
                            color, label = color_label_map.get(
                                rejection_reason, ("black", None)
                            )
                            break

            if i in frames_to_skip and color != "black":
                frames_to_mark.append((self.trj.x[i], self.trj.y[i]))

            # Add to the legend only if it hasn't been added yet
            self._add_legend_entry(handles, labels, label, color)

            x = self.trj.x
            y = self.trj.y

            # Clamp the X coordinates to the camera limits
            x_start = max(min(x[i], bottom_right[0]), top_left[0])
            x_end = max(min(x[i + 1], bottom_right[0]), top_left[0])

            turn_too_long = rejection_reason == "too_long"

            if not turn_too_long or (
                turn_too_long
                and not (
                    self.trj.nan[i] and self.trj.nan[i + 1] and self.trj.nan[i + 2]
                )
            ):
                plt.plot([x_start, x_end], [y[i], y[i + 1]], color=color, zorder=3)

            if (
                turn_too_long
                and current_bcr_index is not None
                and not (
                    # i == bcr[current_bcr_index].start - 1 or
                    i
                    == bcr[current_bcr_index].stop - 1
                )
            ):
                # Set the lighter color for the short segments
                lighter_color = "lightblue"  # You can use 'lightblue' or an RGBA tuple for a lighter shade

                # Calculate the direction of the segment
                dx = self.x[i + 1] - self.x[i]
                dy = self.y[i + 1] - self.y[i]

                # Normalize direction vector to create a unit vector
                norm = np.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm

                # Define a constant length for the short segments
                fixed_segment_length = (
                    0.125  # Fixed length for short segments (adjust as needed)
                )

                # Calculate the start and end points of the short segment at the end of the main segment
                x_start_short = x_end - dx * fixed_segment_length
                y_start_short = y[i + 1] - dy * fixed_segment_length

                # Draw the short segment at the end of the main segment
                plt.plot(
                    [x_end, x_start_short],
                    [y[i + 1], y_start_short],
                    color=lighter_color,
                    linewidth=plt.gca()
                    .lines[-1]
                    .get_linewidth(),  # Match the width of the main line
                    zorder=4,
                )

            # Calculate the speed for this segment
            dx = self.x[i + 1] - self.x[i]
            dy = self.y[i + 1] - self.y[i]
            speed = np.sqrt(dx**2 + dy**2)

            # Set the arrow interval based on the speed
            if speed > speed_threshold_high:
                arrow_interval = 3  # Frequent arrows at high speeds
                successive_slow_frames = 0  # Reset the slow frame counter
            elif speed > speed_threshold_low:
                arrow_interval = 6  # Moderate arrows at medium speeds
                successive_slow_frames = 0  # Reset the slow frame counter
            else:
                arrow_interval = 20  # Less frequent arrows at low speeds
                successive_slow_frames += 1  # Increment the slow frame counter

            # Skip drawing arrows if there are too many successive slow frames
            if successive_slow_frames >= max_slow_frames:
                continue  # Skip the current frame for arrow placement

            last_arrow_idx = self._draw_arrow_for_speed(
                i, x_start, x_end, y[i], y[i + 1], last_arrow_idx, arrow_interval, speed
            )

        # Set plot limits with padding
        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1
        self._setup_plot_and_axes(top_left, bottom_right, padding_x, padding_y)

        # Plot a horizontal line at the vertical midpoint
        vertical_midpoint = (top_left[1] + bottom_right[1]) / 2
        plt.axhline(
            y=vertical_midpoint, color="black", linestyle=":", linewidth=2, zorder=4
        )

        for x, y in frames_to_mark:
            plt.plot(
                x,
                y,
                marker="o",
                color="green",
                markersize=6,
                zorder=5,
                label="Sidewall contact start",
            )
        if len(frames_to_mark) > 0:
            labels.append("Sidewall contact start")
            handles.append(
                plt.Line2D([0], [0], marker="o", color="green", lw=0, markersize=6)
            )

        # Add the legend outside the plot area
        plt.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

        plt.xlabel("")
        plt.ylabel("")

        plt.title(
            f"Boundary contact events and sharp turns, {start_frame} to {end_frame}"
        )

        output_path = f"imgs/turn__{ellipse_ref_pt}_ref_pt/chained_turn_{start_idx}_f{self.trj.f}.{image_format}"
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        writeImage(output_path, format=image_format)
        plt.close()
