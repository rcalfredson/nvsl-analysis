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
    ):
        """Draws an arrow if the conditions for speed are met."""
        if last_arrow_idx is None or i >= last_arrow_idx + arrow_interval:
            x_mid = (x_start + x_end) / 2
            y_mid = (y_start + y_end) / 2
            dx = x_end - x_start
            dy = y_end - y_start
            kw = arrow_kwargs or {}
            self.draw_custom_arrowhead(plt.gca(), x_mid, y_mid, dx, dy, "black", **kw)
            return i  # Update last_arrow_idx
        return last_arrow_idx

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
                role_idx = self.va.flies.index(fly_idx)
            except Exception:
                role_idx = 0

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

        # --- Prepare arena / floor and overlays shared across subplots ----------
        floor_coords = list(
            self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        )
        top_left, bottom_right = floor_coords[0], floor_coords[1]

        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = (
            self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm
        )

        reward_circle = None
        if trn_index >= 0:
            try:
                reward_circle = self.va.trns[trn_index].circles(self.trj.f)[0]
            except Exception:
                reward_circle = None

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

            # Per-subplot title: just the varying info
            dmm = (
                _segment_dist_mm(start_reward, end_reward)
                if max_dist_mm is not None
                else None
            )
            dist_line = (
                f"\ndist {dmm:.2f} mm" if dmm is not None and np.isfinite(dmm) else ""
            )
            ax.set_title(
                f"seg {idx + 1}: frames {start_frame}-{end_frame}\n"
                f"rewards {start_reward}->{end_reward}{dist_line}",
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
                role_idx = self.va.flies.index(fly_idx)
            except Exception:
                role_idx = 0  # fallback if not resolvable, but keep going

        seed_str = f"{seed}" if seed is not None else "rand"
        fly_role = "exp" if role_idx == 0 else "yok"

        global_title = (
            "Between-reward trajectories\n"
            f"{video_id}, fly {fly_idx}, {fly_role}\n"
            f"trn {trn_index + 1}, bucket {bucket_index + 1}"
        )
        fig.suptitle(global_title, fontsize=12)

        short_str = f"_maxd{max_dist_mm:g}mm" if max_dist_mm is not None else ""
        zoom_str = ""
        if zoom:
            if zoom_radius_mm is not None:
                zoom_str = f"_zoom{zoom_radius_mm:g}mm"
            else:
                zoom_str = f"_zoomx{zoom_radius_mult:g}"

        output_path = (
            f"imgs/between_rewards/"
            f"{video_id}__fly{fly_idx}_role{role_idx}_"
            f"trn{trn_index + 1}_bkt{bucket_index + 1}_"
            f"N{n_examples}_seed{seed_str}"
            f"{short_str}{zoom_str}."
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
                role_idx = self.va.flies.index(fly_idx)
            except Exception:
                role_idx = 0

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
