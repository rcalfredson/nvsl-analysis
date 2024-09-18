import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from math import radians


def visualize_ellipse_and_boundaries(
    center_x,
    center_y,
    theta,
    semimaj_ax,
    semimin_ax,
    boundaries,
    boundary_index,
    points_to_check,
):
    """
    Visualizes the boundaries and ellipse for debugging purposes.

    This function plots an ellipse along with its boundary lines to visually verify 
    whether the ellipse is within the specified boundaries. It highlights the four 
    critical points on the ellipse (top, bottom, left, right) and the start and end 
    positions of the boundary.

    Parameters:
    - center_x : float
        X-coordinate of the ellipse center.
    - center_y : float
        Y-coordinate of the ellipse center.
    - theta : float
        Angle of the ellipse (orientation) in degrees.
    - semimaj_ax : float
        Length of the semi-major axis of the ellipse.
    - semimin_ax : float
        Length of the semi-minor axis of the ellipse.
    - boundaries : dict
        Dictionary of the start and end positions of the boundaries. Keys are 
        'left_start', 'right_start', 'top_start', 'bottom_start', etc.
    - boundary_index : int
        Index representing which boundary is being analyzed (0 for left, 1 for bottom, 
        2 for right, 3 for top).
    - points_to_check : list of tuples
        List of (x, y) points on the ellipse that are being checked for boundary proximity.

    Returns:
    - None
        The function creates a plot but does not return any value.
    """
    fig, ax = plt.subplots()

    # Plot the boundaries
    if boundary_index in [0, 2]:  # Left-Right boundary
        ax.axvline(
            x=boundaries["left_start"],
            color="r",
            linestyle="--",
            label="Left Boundary Start",
        )
        ax.axvline(
            x=boundaries["right_start"],
            color="g",
            linestyle="--",
            label="Right Boundary Start",
        )
        ax.axvline(
            x=boundaries["left_end"],
            color="r",
            linestyle="-",
            label="Left Boundary End",
        )
        ax.axvline(
            x=boundaries["right_end"],
            color="g",
            linestyle="-",
            label="Right Boundary End",
        )
    elif boundary_index in [1, 3]:  # Top-Bottom boundary
        ax.axhline(
            y=boundaries["top_start"],
            color="b",
            linestyle="--",
            label="Top Boundary Start",
        )
        ax.axhline(
            y=boundaries["bottom_start"],
            color="y",
            linestyle="--",
            label="Bottom Boundary Start",
        )
        ax.axhline(
            y=boundaries["top_end"], color="b", linestyle="-", label="Top Boundary End"
        )
        ax.axhline(
            y=boundaries["bottom_end"],
            color="y",
            linestyle="-",
            label="Bottom Boundary End",
        )

    # Plot the ellipse using the Ellipse patch
    ellipse_patch = patches.Ellipse(
        (center_x, center_y),
        width=2 * semimin_ax,
        height=2 * semimaj_ax,
        angle=theta,
        edgecolor="b",
        facecolor="none",
        label="Ellipse",
    )
    ax.add_patch(ellipse_patch)

    # Plot the four points on the ellipse
    for x, y in points_to_check:
        ax.plot(x, y, "ro")  # Red dots for the points on the ellipse

    # Set plot limits for better visualization
    ax.set_xlim([center_x - 2 * semimaj_ax, center_x + 2 * semimaj_ax])
    ax.set_ylim([center_y - 2 * semimaj_ax, center_y + 2 * semimaj_ax])

    ax.set_aspect("equal", "box")
    plt.legend()
    plt.title(f"Ellipse and Boundaries (Boundary Index: {boundary_index})")
    plt.show()


def ellipse_edge_points_within_boundaries(
    center_x,
    center_y,
    theta,
    semimaj_ax,
    semimin_ax,
    boundary_index,
    start_boundaries,
    boundary_combo,
    x_bounds_orig,
    y_bounds,
    visualize=False
):
    """
    Checks whether all points on the edge of an ellipse are within the specified boundaries.

    This function calculates four key points on the ellipse's edge (top, bottom, left, right) 
    and checks whether these points fall within the specified start boundaries. If any of 
    the points fall outside the boundaries, it returns False.

    Parameters:
    - center_x : float
        X-coordinate of the ellipse center.
    - center_y : float
        Y-coordinate of the ellipse center.
    - theta : float
        Angle of the ellipse (orientation) in degrees.
    - semimaj_ax : float
        Length of the semi-major axis of the ellipse.
    - semimin_ax : float
        Length of the semi-minor axis of the ellipse.
    - boundary_index : int
        Index representing which boundary is being analyzed (0 for left, 1 for bottom, 
        2 for right, 3 for top).
    - start_boundaries : dict
        Dictionary of the start boundaries for each wall.
    - boundary_combo : str
        Specifies which combination of boundaries is being considered (e.g., 'all', 'tb').
    - x_bounds_orig : list
        Original x-boundaries for the chamber.
    - y_bounds : list
        y-boundaries for the chamber.
    - visualize : bool, optional (default: False)
        Whether to visualize the ellipse and boundaries if points are found outside the boundaries.

    Returns:
    - bool
        Returns True if all points on the ellipse are within the boundaries, False otherwise.
    """
    if np.any(np.isnan([semimaj_ax, semimin_ax])):
        return

    theta_rad = radians(theta)
    R = np.array(
        [
            [-np.sin(theta_rad), np.cos(theta_rad)],
            [np.cos(theta_rad), np.sin(theta_rad)],
        ]
    )

    edge_points = np.array(
        [[0, semimin_ax], [0, -semimin_ax], [semimaj_ax, 0], [-semimaj_ax, 0]]
    )
    rotated_points = np.dot(edge_points, R.T)

    points_to_check = [
        (round(center_x + rotated_points[i, 0]), round(center_y + rotated_points[i, 1]))
        for i in range(4)
    ]

    if boundary_combo == "all" and "opposite":
        boundaries_to_check = [
            round(x_bounds_orig[0]),
            round(y_bounds[1]),
            round(x_bounds_orig[1]),
            round(y_bounds[0]),
        ]
    else:
        boundaries_to_check = start_boundaries

    for ellipse_x, ellipse_y in points_to_check:
        if boundary_index in [0, 2]:  # Left or Right boundary
            if ellipse_x < boundaries_to_check[0] or ellipse_x > boundaries_to_check[2]:
                if visualize:
                    visualize_ellipse_and_boundaries(
                        center_x, center_y, theta, semimaj_ax, semimin_ax, start_boundaries, boundary_index, points_to_check
                    )
                return False
        elif boundary_index in [1, 3]:  # Top or Bottom boundary
            if ellipse_y > boundaries_to_check[1] or ellipse_y < boundaries_to_check[3]:
                if visualize:
                    visualize_ellipse_and_boundaries(
                        center_x, center_y, theta, semimaj_ax, semimin_ax, start_boundaries, boundary_index, points_to_check
                    )
                return False

    return True
