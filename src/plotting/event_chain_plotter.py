from math import sin, cos
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.utils.common import writeImage
from src.utils.constants import CONTACT_BUFFER_OFFSETS


class EventChainPlotter:
    def __init__(self, trj, va, y_bounds=None, x=None, y=None):
        self.trj = trj
        self.va = va
        self.y_bounds = y_bounds
        self.x = np.array(trj.x) if x is None else x
        self.y = np.array(trj.y) if y is None else y

    def draw_custom_arrowhead(
        self, ax, x_mid, y_mid, dx, dy, color, length=1.1, angle=30, shift_factor=-0.08
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
                lw=1,
                zorder=5,
            )
        )
        ax.add_line(
            plt.Line2D(
                [x_mid_shifted, right_x],
                [y_mid_shifted, right_y],
                color=color,
                lw=1,
                zorder=5,
            )
        )

    def _draw_arrow_for_speed(
        self, i, x_start, x_end, y_start, y_end, last_arrow_idx, arrow_interval, speed
    ):
        """Draws an arrow if the conditions for speed are met."""
        if last_arrow_idx is None or i >= last_arrow_idx + arrow_interval:
            x_mid = (x_start + x_end) / 2
            y_mid = (y_start + y_end) / 2
            dx = x_end - x_start
            dy = y_end - y_start
            self.draw_custom_arrowhead(plt.gca(), x_mid, y_mid, dx, dy, "black")
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
        image_format="png",
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

        plt.figure(figsize=(12, 8))

        # Define default color_map if none is provided
        if color_map is None:
            color_map = {
                "no_event": "black",
                "large_turn": "red",
                "reward_circle_entry": "blue",
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

    def plot_sharp_turn_chain_wall(
        self,
        ellipse_ref_pt,
        bcr,
        turning_idxs,
        rejection_reasons,
        frames_to_skip,
        start_frame=None,
        mode="all_types",
        image_format="png",
    ):
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
        self, radius_stats, trn_index, start_frame, mode, image_format
    ):
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
        image_format="png",
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

        output_path = f"imgs/turn__{ellipse_ref_pt}_ref_pt/chained_turn_{start_idx}_f{self.trj.f}.png"
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        writeImage(output_path, format=image_format)
        plt.close()
