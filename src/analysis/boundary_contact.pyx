#cython: language_level=3
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from collections import defaultdict
import os
from math import radians, sin, cos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
cimport numpy as cnp
import random

from src.utils.common import writeImage
from src.utils.constants import CONTACT_BUFFER_OFFSETS
from src.utils.util import closest_pts_to_lines, rotate_pts, trueRegions

cnp.import_array()

WALL_ORIENTATION_FOR_TURN = {"wall": "all", "agarose": "tb", "boundary": "tb"}

def flip_sign(double[:] mv):
    """
    Inverts the sign of each element in a given array in-place.

    This function iterates over each element of the input array `mv` and multiplies it by -1,
    effectively flipping its sign. This operation is performed in-place, meaning the input
    array is directly modified.

    Parameters:
    - mv : memoryview of double
        A 1D array (memoryview) of double precision floating-point numbers. Each element's
        sign in this array will be flipped.

    Returns:
    - None
        The function modifies the input array in-place and does not return any value.
    """
    cdef int i
    for i in range(mv.shape[0]):
        mv[i] = -mv[i]

def recursive_defaultdict():
    """
    Creates a recursive defaultdict, where each missing key initializes
    another defaultdict of the same type.

    Returns:
    - defaultdict: A recursive defaultdict structure.
    """
    return defaultdict(recursive_defaultdict)

def draw_custom_arrowhead(ax, x_mid, y_mid, dx, dy, color, length=1.1, angle=30, shift_factor=-0.08):
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
    ax.add_line(plt.Line2D([x_mid_shifted, left_x], [y_mid_shifted, left_y], color=color, lw=1, zorder=5))
    ax.add_line(plt.Line2D([x_mid_shifted, right_x], [y_mid_shifted, right_y], color=color, lw=1, zorder=5))

cpdef runBndContactAnalysisForCtrReferencePt(trj, va, offset, opts, find_turns=True):
    """
    Conducts a boundary contact analysis centered around the centroid (center) of the ellipse.
    This analysis identifies interactions between the trajectory and the boundary and calculates
    the distances from the centroid to the boundary.

    Parameters:
    - trj : Trajectory
        An instance of the Trajectory class representing the trajectory of an entity across
        frames in a video.
    - va : VideoAnalysis
        An instance of the VideoAnalysis class that has processed the video from which the
        trajectory data was obtained. It provides context and additional data required for the
        analysis.
    - offset : float
        A distance offset for defining the boundary relative to the chamber's top/bottom walls.
    - opts : argparse.Namespace
        A Namespace object containing various options and flags that influence the analysis 
        process. These options may include, but are not limited to, minimum turn speed, the 
        inclusion of specific types of boundaries in the analysis, and parameters defining what 
        constitutes a turn or boundary contact event.
    - find_turns : bool, optional
        A flag indicating whether to identify turning events based on boundary interactions. 
        Default is True.

    Returns:
    - dict
        A dictionary containing the results of the boundary contact analysis. This includes
        calculated distances from the ellipse centroid to the boundary, identified turns, and
        other relevant information extracted during the analysis process.

    Raises:
    - AttributeError
        If certain required attributes are missing from the opts parameter.

    This function initializes an EllipseToBoundaryDistCalculator object with the provided
    Trajectory and VideoAnalysis instances, along with other necessary parameters. It sets up
    the event thresholds using predefined constants and calculates distances from the ellipse 
    centroid to the specified boundary. If the `find_turns` flag is set to True, the function 
    also identifies turning events. The original options for turn detection are restored after 
    the analysis.
    """
    # Initialize the distance calculator
    dist_calc = EllipseToBoundaryDistCalculator(trj, va, min_turn_speed=opts.min_turn_speed)

    dist_calc.detectWallContact("closest")
    
    # Set the relevant properties
    dist_calc.boundary_type = 'boundary'
    dist_calc.boundary_combo = 'tb'
    
    # Set up event thresholds using the constant
    dist_calc.event_thresholds = {
        "vert": [
            CONTACT_BUFFER_OFFSETS["boundary"]["min"],
            CONTACT_BUFFER_OFFSETS["boundary"]["max"]
        ],
        "horiz": [
            CONTACT_BUFFER_OFFSETS["boundary"]["min"],
            CONTACT_BUFFER_OFFSETS["boundary"]["max"]
        ]
    }
    
    # Preserve the original turn options
    turn_opts_orig = opts.turn
    
    # Ensure boundary turns are considered if not already specified
    if not opts.turn:
        opts.turn = ['boundary']
    
    # Calculate distances with the center of the ellipse as the reference point
    dist_calc.calc_dist_boundary_to_ellipse(
        boundary_type='boundary',
        boundary_combo='tb',
        offset=offset,
        event_thresholds=dist_calc.event_thresholds,  # Use the thresholds set up above
        ellipse_ref_pt='ctr'  # Center reference point
    )
    
    # If specified, find the turning events
    if find_turns:
        findTurns(va, opts, dist_calc, 'boundary', 'tb', 'ctr')
    
    # Restore the original turn options
    opts.turn = turn_opts_orig
    
    # Return the analysis data
    return dist_calc.return_data

cpdef runBoundaryContactAnalyses(trj, va, offsets, thresholds, opts):
    """
    Conducts boundary contact analysis on a given trajectory to identify interactions with
    predefined boundary types (e.g., walls, agarose, boundaries) and calculates distances to
    these boundaries. This function iterates over different boundary types specified in the
    options, calculates distances to these boundaries using both the center and edge of the
    ellipse as reference points, and identifies turning events based on the calculated distances
    and the specified thresholds.

    Parameters:
    - trj : Trajectory
        An instance of the Trajectory class representing the trajectory of an entity across
        frames in a video.
    - va : VideoAnalysis
        An instance of the VideoAnalysis class that has processed the video from which the
        trajectory data was obtained. It provides context and additional data required for the
        analysis.
    - offsets : dict
        A dictionary containing distances from the top/bottom walls of the chamber at which to
        define the boundaries. Note: units are mm, and these offsets are scaled relative to the
        (HTL) chamber height of 16mm.
    - thresholds : dict
        A dictionary containing threshold values for different boundary types. These
        thresholds are used to determine proximity to boundaries and to identify significant
        interactions.
    - opts : argparse.Namespace
        A Namespace object resulting from parsing command-line arguments. This object contains
        various options and flags that influence the analysis process. These options include,
        but are not limited to, minimum turn speed, the inclusion of specific types of
        boundaries in the analysis, and parameters defining what constitutes a turn or
        boundary contact event.

    Returns:
    - dict
        A dictionary containing the results of the boundary contact analysis. This includes
        calculated distances to boundaries, identified turns, and other relevant information
        extracted during the analysis process.

    Raises:
    - AttributeError: If certain required attributes are missing from the opts parameter.

    The function works by initializing an EllipseToBoundaryDistCalculator object with the
    provided Trajectory and VideoAnalysis instances, along with other necessary parameters. It
    iterates through specified boundary types (e.g., walls, agarose, boundaries), calculates
    distances to these boundaries using both the center and edge of the ellipse as reference
    points, and identifies turning events based on the analysis of these distances and the
    application of specified thresholds. Special handling is included for walls, with different
    combinations of boundary orientations and special thresholds for scenarios such as sidewall
    exclusion.
    """
    boundary_dist_calc = EllipseToBoundaryDistCalculator(
        trj,
        va,
        min_turn_speed=opts.min_turn_speed,
        wall_debug=opts.wall_debug
    )
    ellipse_ref_pts = ["ctr", "edge"]

    if (
        not any([getattr(opts, bnd_tp) for bnd_tp in ("wall", "agarose", "boundary")])
        and
        not opts.excl_wall_for_spd
    ):
            return {}
    if opts.excl_wall_for_spd:
        boundary_dist_calc.detectWallContact("opposite")

    for bnd_tp in ("wall", "agarose", "boundary"):
        if not getattr(opts, bnd_tp):
            continue
        if bnd_tp == "wall":
            boundary_combos = ["lr", "agarose_adj", "all"]
            if opts.wall_orientation not in boundary_combos:
                boundary_combos.append(opts.wall_orientation)
            boundary_dist_calc.return_data["wall_orientations"] = boundary_combos
            for boundary_combo in boundary_combos:
                if boundary_combo == "agarose_adj":
                    wall_thr = {
                        "horiz": thresholds["agarose"],
                        "vert": thresholds["wall"],
                    }
                else:
                    wall_thr = thresholds["wall"]
                
                # Calculating distances using the edge reference point
                boundary_dist_calc.calc_dist_boundary_to_ellipse(
                    boundary_type=bnd_tp,
                    boundary_combo=boundary_combo,
                    offset=offsets[bnd_tp],
                    event_thresholds=wall_thr,
                    ellipse_ref_pt="edge"  # Use edge reference point for walls
                )
        else:
            boundary_combos = ["tb"]
            boundary_dist_calc.detectWallContact("opposite")
            boundary_dist_calc.detectWallContact("closest")
            
            # Calculate distances using both center and edge reference points
            if bnd_tp == 'boundary':
                for pt in ellipse_ref_pts:
                    boundary_dist_calc.calc_dist_boundary_to_ellipse(
                        boundary_type=bnd_tp,
                        boundary_combo=boundary_combos[0],
                        offset=offsets[bnd_tp],
                        event_thresholds=thresholds[bnd_tp],
                        ellipse_ref_pt=pt
                    )
            elif bnd_tp == 'agarose':
                boundary_dist_calc.calc_dist_boundary_to_ellipse(
                    boundary_type=bnd_tp,
                    boundary_combo=boundary_combos[0],
                    offset=offsets[bnd_tp],
                    event_thresholds=thresholds[bnd_tp],
                    ellipse_ref_pt='edge'
                )
                boundary_dist_calc.get_ellipse_ctr_boundary_crossings(
                    offset=offsets[bnd_tp]
                )
            for pt in ellipse_ref_pts:
                findTurns(
                    va, opts, boundary_dist_calc, bnd_tp, boundary_combos[0], ellipse_ref_pt=pt
                )

    return boundary_dist_calc.return_data

cpdef findTurns(va, opts, boundary_dist_calc, bnd_tp, boundary_combo, ellipse_ref_pt):
    # Identifies turning events within boundary contact events based on specified criteria.
    # This function is integral to the boundary contact analysis, determining which boundary
    # contacts involve significant turns that meet the criteria defined in the options.
    #
    # Parameters:
    # - va : VideoAnalysis
    #     An instance of the VideoAnalysis class that provides context and additional data
    #     required for the analysis, such as frames per second (fps), which is crucial for
    #     determining the duration in frames for turning events.
    # - opts : argparse.Namespace
    #     A Namespace object containing various options and flags that influence the analysis
    #     process, including minimum velocity angle delta for turns, the duration threshold for
    #     turning events, and the specific boundaries to consider for turning events.
    # - boundary_dist_calc : EllipseToBoundaryDistCalculator
    #     An instance of the EllipseToBoundaryDistCalculator class that calculates distances to
    #     boundaries and is now used to identify turning events based on those calculations.
    # - bnd_tp : str
    #     The type of boundary being analyzed (e.g., "wall", "agarose"). This parameter is used
    #     to filter and analyze turning events relevant to the specific boundary type.
    # - boundary_combo : str
    #     The specific combination of boundaries being considered in this part of the analysis.
    #     This parameter refines which turning events are relevant based on the boundary
    #     configuration (e.g., adjacent to agarose, left-right walls).
    # - ellipse_ref_pt : str
    #     Reference point on the ellipse used to determine boundary-crossing events, either "edge"
    #     or "center."
    #
    # This function checks if turning analysis is enabled and relevant for the given boundary
    # type and combination as specified in the options. If so, it calculates the duration in 
    # frames for turning events based on the video's fps and the duration threshold in the opts, 
    # and then calls the `find_subset_at_or_below_duration` method of the `boundary_dist_calc` 
    # instance to identify and mark turning events within the boundary contact analysis.
    #
    # Note:
    # - This function does not return any value but influences the results of the boundary contact
    #   analysis by identifying and marking turning events, which are then reflected in the output
    #   of the `runBoundaryContactAnalyses` function.
    # - It relies on the `WALL_ORIENTATION_FOR_TURN` dictionary to match boundary types and
    #   combinations with those specified for turn analysis in the opts, ensuring that only
    #   relevant turning events are identified.
    # - If boundary contact plots are enabled (`opts.bnd_ct_plots`), the function will also
    #   trigger visualization of the turns.
    if not va or not opts.turn or boundary_combo != WALL_ORIENTATION_FOR_TURN[bnd_tp]:
        return
    
    duration_in_frames = int(round(opts.turn_duration_thresh * va.fps))
    boundary_dist_calc.find_subset_at_or_below_duration(
        duration_in_frames,
        min_vel_angle_delta=opts.min_vel_angle_delta,
        ellipse_ref_pt=ellipse_ref_pt
    )
    
    if opts.bnd_ct_plots is None:
        return
    mode = opts.bnd_ct_plots if opts.bnd_ct_plots else "troubleshooting"
    
    boundary_dist_calc.visualize_turns(ellipse_ref_pt, opts, mode=mode)

cdef class EllipseToBoundaryDistCalculator:
    cdef object trj
    cdef object va
    cdef double min_turn_speed
    cdef double[:] rot_angles
    cdef double[:] sin_of_angles
    cdef double[:] neg_sin_of_angles
    cdef double[:] cos_of_angles
    cdef int[2][2] x_align_fl
    cdef int[2][2][2] y_align_fls
    cdef float[2] chamber_center
    cdef float[2] max_dists_from_ctr
    cdef double[2][2] bounds_orig
    cdef double[2][2] offsets
    cdef double[2] x_bounds_orig
    cdef double[2] y_bounds_orig
    cdef double[2] y_bounds
    cdef double[:] x
    cdef double[:] y
    cdef double[:, :] origins
    cdef cnp.ndarray semimaj_ax
    cdef cnp.ndarray semimin_ax
    cdef cnp.ndarray slope_intercept_sets
    cdef cnp.ndarray min_distances
    cdef cnp.ndarray out_of_bounds
    cdef cnp.ndarray best_wall_indices
    cdef cnp.ndarray bounds
    cdef cnp.ndarray oob_on_ignored_wall
    cdef cnp.ndarray dist_to_wall
    cdef cnp.ndarray wall_is_vert
    cdef str boundary_combo
    cdef public str boundary_type
    cdef str ellipse_edge_pt
    cdef str ellipse_ref_pt
    cdef dict best_pts
    cdef bint wall_debug
    cdef dict event_thresholds
    cdef public dict return_data
    cdef dict oob_test_pts, ell_pts, distances, wall_pts, angle_to_best_ell_pt
    cdef object bounds_agarose, bounds_full, y_bounds_agarose, y_bounds_full

    def __cinit__(
        self,
        trj,
        va,
        min_turn_speed=0,
        wall_debug=False
    ):
        # Initialize the EllipseToBoundaryDistCalculator with specific trajectory and video
        # analysis instances, along with a threshold for minimum turning speed. This constructor
        # sets up the calculator's initial state, enabling it to compute distances from ellipses
        # (representing entities in the video) to predefined boundaries based on trajectory data
        # and video analysis.
        #
        # Parameters:
        # - trj : Trajectory
        #     An instance of the Trajectory class representing the trajectory of an entity
        #     (e.g., a fly) across frames in a video. This object contains positional data
        #     (x, y), orientations (theta), dimensions (width and height), among other pieces of
        #     information necessary for the distance calculations.
        #
        # - va : VideoAnalysis
        #     An instance of the VideoAnalysis class that has processed the video from which the
        #     trajectory data was obtained. It provides context and additional data required for
        #     the analysis, such as pixel-per-millimeter ratios, which may be used to adjust the
        #     minimum turning speed, and other metrics that might influence boundary distance
        #     calculations.
        #
        # - min_turn_speed : double, optional
        #     The minimum turning speed threshold, used to filter out certain movements based on
        #     their velocity. This value may be adjusted based on pixel-per-millimeter ratios
        #     from the video analysis to maintain consistency in units or measurement scales.
        #     Defaults to 0, indicating that by default, no velocity-based filtering is applied.
        self.trj = trj
        self.va = va
        self.min_turn_speed = min_turn_speed
        self.wall_debug = wall_debug
        if hasattr(self.trj, "pxPerMmFloor"):
            self.min_turn_speed = self.trj.pxPerMmFloor * min_turn_speed
        self._reorient_ellipse_thetas()
        self._set_trj_data()
        self.return_data = {"boundary_event_stats": defaultdict(recursive_defaultdict)}

    @staticmethod
    def expected_dist_thresholds(boundary_combo):
        # Determines the expected distance thresholds based on the specified boundary
        # combination.
        #
        # This method maps the boundary combination to the orientations (vertical or horizontal)
        # that are relevant for calculating the distances to boundaries. This mapping helps in
        # configuring event thresholds for distance calculations and boundary contact analysis.
        #
        # Parameters:
        # - boundary_combo : str
        #     A string identifier for the boundary combination. Possible values include:
        #     - 'lr': indicating left-right boundaries,
        #     - 'tb': indicating top-bottom boundaries,
        #     - 'all': indicating all boundaries,
        #     - 'agarose_adj': like "all", but with top-bottom boundaries offset to the inner
        #                      border of the agarose.
        #
        # Returns:
        # - tuple of str
        #     A tuple containing the orientations ('vert', 'horiz') that are relevant for the
        #     given boundary combination. This information is used to define the event
        #     thresholds for boundary distance calculations.
        #     
        # Examples:
        # - For 'lr' (left-right), the method returns ('vert',), indicating that vertical
        #   boundaries are relevant.
        # - For 'tb' (top-bottom), it returns ('horiz',), pointing to the relevance of
        #   horizontal boundaries.
        # - For 'all' or 'agarose_adj', it returns ('vert', 'horiz'), indicating that both
        #   orientations are relevant.

        if boundary_combo == "lr":
            return ("vert",)
        elif boundary_combo == "tb":
            return ("horiz",)
        elif boundary_combo in ("all", "agarose_adj"):
            return ("vert", "horiz")

    def _set_event_thresholds(self, boundary_combo, event_thresholds, near_contact):
        # Establishes the thresholds for detecting boundary contact events during distance
        # calculations, tailored to the specified boundary combination and event thresholds.
        # This method configures the thresholds needed to identify when an object is in
        # proximity to or in contact with a boundary, making adjustments for scenarios where
        # near contact is considered.
        #
        # Parameters:
        # - boundary_combo : str
        #     Identifies the set of boundaries (e.g., 'lr' for left-right, 'tb' for top-bottom,
        #     'all', or 'agarose_adj') to be considered in the distance calculations.
        #
        # - event_thresholds : dict, list, or tuple
        #     Specifies the thresholds for initiating and concluding boundary contact events.
        #     Accepts a dictionary with 'vert' and 'horiz' keys defining start and end
        #     thresholds for vertical and horizontal orientations, respectively, or a list/tuple
        #     of two elements specifying universal start and end thresholds applicable to all
        #     orientations unless overridden.
        #
        # - near_contact : bool
        #     Indicates whether to adjust thresholds for near contact detection. When True,
        #     increases the start threshold slightly to accommodate close proximity without
        #     actual contact.
        #
        # Raises:
        # - ValueError: If using a dictionary for `event_thresholds` and it lacks entries for
        #               any orientation required by `boundary_combo`.
        # - TypeError: If `event_thresholds` is neither a list, tuple, nor dict.
        if type(event_thresholds) is list or type(event_thresholds) is tuple:
            self.event_thresholds = {
                orientation: {"start": event_thresholds[0], "end": event_thresholds[1]}
                for orientation in ("vert", "horiz")
            }
        elif type(event_thresholds) is dict:
            orientations = self.expected_dist_thresholds(boundary_combo)
            for ori in orientations:
                if ori not in event_thresholds:
                    ValueError(
                        "When using boundary type %s, event thresholds must be"
                        " specified for %s boundaries" % (self.boundary_type, ori)
                    )
            self.event_thresholds = {
                ori: {
                    "start": event_thresholds[ori][0],
                    "end": event_thresholds[ori][1],
                }
                for ori in orientations
            }
        else:
            raise TypeError("event_thresholds must be of type list, dict, or tuple")
        if near_contact:
            for ori in self.event_thresholds:
                self.event_thresholds[ori]["start"] += 0.1

    def _setup_boundary_dist_data(self):
        # Prepares the data required for calculating distances from ellipses to boundaries. This
        # method initializes arrays and performs initial calculations to set up the necessary
        # data structures for distance calculations. Specifically, it computes semi-major and
        # semi-minor axes of ellipses, prepares data structures for storing calculation results
        # (like best points on the ellipse and boundary, minimum distances, out-of-bound
        # statuses, and best wall indices), and initializes slope-intercept sets for boundary
        # lines. This setup is crucial for subsequent distance and boundary interaction
        # analyses.
        #
        # The method leverages trajectory data (`self.trj`) to determine the sizes of the
        # ellipses represented by the trajectory's height (`h`) and width (`w`), treating them
        # as semi-major and semi-minor axes respectively. It accounts for missing data by
        # substituting NaNs with default values. The approach facilitates handling various
        # boundary interaction scenarios, including calculations of closest points, minimum
        # distances to boundaries, and out-of-bound conditions.
        #
        # Note:
        # - This method is intended to be called internally during the initialization phase of
        #   the class or before performing detailed boundary distance calculations.
        # - It initializes multiple NumPy arrays and dictionaries that store information related
        #   to the ellipses' positions relative to boundaries, including their best points on
        #   both the ellipses and the boundaries, the minimum distances to the boundaries, and
        #   indices of the closest walls.
        # - Assumes that the trajectory data (`self.trj`) has been set and contains necessary
        #   information about the entity's positions and dimensions (height and width) across
        #   frames.
        #
        # Modifies:
        # - Initializes and sets `self.semimaj_ax`, `self.semimin_ax`,
        #   `self.slope_intercept_sets`, `self.best_pts`, `self.min_distances`,
        #   `self.out_of_bounds`, and `self.best_wall_indices` with appropriate sizes and
        #   default values based on the trajectory data.
        #
        # Raises:
        # - No explicit exception raising is done in this method, but it depends on the
        #   integrity of trajectory data (`self.trj`). Missing or malformed data may lead to
        #   incorrect calculations or initialization.
        if self.trj.h is None:
            return

        self.semimaj_ax = np.array(self.trj.h) / 2
        self.semimin_ax = np.array(self.trj.w) / 2
        self.slope_intercept_sets = np.empty((len(self.x), 4, 2))
        self.best_pts = {k: np.empty((len(self.x), 2)) for k in ("ellipse", "boundary")}
        self.min_distances = np.zeros((len(self.x)))
        self.out_of_bounds = np.full(len(self.x), False)
        self.best_wall_indices = np.full(len(self.x), np.nan)

    def _reorient_ellipse_thetas(self):
        # Adjusts the orientations (thetas) of ellipses from trajectory data to align correctly
        # for distance calculations. This method normalizes theta values within a specific range
        # and converts them to radians, ensuring the semi-major and semi-minor axes of ellipses
        # are properly oriented relative to the calculation coordinate system. Minor adjustments
        # are made to theta values to avoid computational issues when orientations align exactly
        # with axes.
        #
        # This adjustment is crucial for accurate distance measurements from ellipses to
        # boundaries, affecting how closest points on the ellipses are determined relative to
        # those boundaries.
        #
        # The method modifies class attributes to store adjusted orientations and their
        # trigonometric functions, supporting subsequent distance calculations.
        #
        # Note:
        # - Assumes `self.trj.theta` contains valid orientation data.
        # - If `self.trj.theta` is `None`, no adjustments are performed.
        #
        # Modifies:
        # - Adjusts `self.rot_angles` to store theta values in radians.
        # - Sets `self.sin_of_angles`, `self.neg_sin_of_angles`, and `self.cos_of_angles` for use in 
        # distance calculations.
        if self.trj.theta is None:
            return
        self.rot_angles = np.array(self.trj.theta)
        rot_angles_array = np.asarray(self.rot_angles)
        rot_angles_array = np.where(
            np.mod(rot_angles_array, 90) == 0, rot_angles_array + 1e-6, rot_angles_array
        )
        rot_angles_array = np.where(
            rot_angles_array > 0, rot_angles_array - 180, rot_angles_array
        )
        rot_angles_array = rot_angles_array * np.pi / 180
        self.sin_of_angles = np.sin(-rot_angles_array)
        self.neg_sin_of_angles = np.copy(self.sin_of_angles)
        flip_sign(self.neg_sin_of_angles)
        self.cos_of_angles = np.cos(-rot_angles_array)
        self.rot_angles = rot_angles_array

    cdef _set_trj_data(self):
        # Sets up the trajectory data and initializes boundary conditions for the
        # EllipseToBoundaryDistCalculator. This method processes video analysis data to
        # establish initial conditions related to the chamber's geometry, aligns boundary origin
        # points based on video analysis, calculates chamber center and maximum distances from
        # the center, and adjusts boundary positions considering offsets. It also directly
        # copies trajectory X and Y positions into class attributes for subsequent calculations.
        #
        # The process involves:
        # - Extracting floor alignment coordinates to calculate the chamber's center.
        # - Determining the maximum distances from the chamber center to its boundaries to help
        #   adjust boundary positions.
        # - Calculating offsets for original boundary positions based on predefined factors,
        #   ensuring boundaries are dynamically adjusted relative to the chamber center and
        #   maximum distances.
        # - Copying trajectory X and Y data into class attributes for use in distance
        #   calculations to boundaries.
        #
        # Modifies:
        # - `self.x_align_fl`, `self.y_align_fls`: Stores alignment information for calculating
        #                                          the chamber center.
        # - `self.chamber_center`: The calculated center of the chamber based on alignment
        #                          information.
        # - `self.max_dists_from_ctr`: Maximum distances from the chamber center to its edges,
        #                              used for boundary adjustment.
        # - `self.bounds_orig` and `self.offsets`: Original boundary positions and their
        #                                          adjustments.
        # - `self.x`, `self.y`: Direct copying of trajectory data for use in distance
        #                       calculations.
        #
        # Raises:
        # - Does not explicitly raise exceptions but relies on the integrity and availability of
        #   `self.va` attributes.
        if not hasattr(self.va, "ct"):
            return
        gen = self.va.ct.floor(self.va.xf, f=2)
        for i, (x, y) in enumerate(gen):
            self.x_align_fl[i][0] = x
            self.x_align_fl[i][1] = y
        lists = [list(self.va.ct.floor(self.va.xf, f=f)) for f in (5, 10)]
        for i, lst in enumerate(lists):
            for j, (x, y) in enumerate(lst):
                self.y_align_fls[i][j][0] = x
                self.y_align_fls[i][j][1] = y
        self.chamber_center[0] = 0.5 * (self.x_align_fl[0][0] + self.x_align_fl[1][0])
        self.chamber_center[1] = 0.5 * (self.y_align_fls[0][1][1] + self.y_align_fls[1][0][1])
        self.max_dists_from_ctr[0] = (
            self.chamber_center[0] - list(self.va.ct.floor(self.va.xf, f=0))[0][0]
        )
        self.max_dists_from_ctr[1] = self.chamber_center[1] - self.x_align_fl[0][1]
        gen = self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef)
        for i, (x, y) in enumerate(gen):
            self.bounds_orig[i][0] = x
            self.bounds_orig[i][1] = y
        for i in range(len(self.bounds_orig)):
            for j in range(2):
                # apply maximum of 0.4% offset to compensate for left-right and top-bottom
                # biases in boundary contact events caused by uneven visibility of sidewalls
                self.offsets[i][j] = 0.004 * (                   
                    self.bounds_orig[i][j] - self.chamber_center[j]
                ) / self.max_dists_from_ctr[j] * abs(
                    self.bounds_orig[i][j] - self.chamber_center[j]
                )
        for i in range(len(self.bounds_orig)):
            self.bounds_orig[i][0] += self.offsets[i][0]
            self.bounds_orig[i][1] += self.offsets[i][1]
        self.x_bounds_orig = [self.bounds_orig[0][0], self.bounds_orig[1][0]]
        self.y_bounds_orig = [self.bounds_orig[0][1], self.bounds_orig[1][1]]
        self.x = np.array(self.trj.x)
        self.y = np.array(self.trj.y)
        self.origins = np.vstack((self.x, self.y)).T

    def _rotate_chamber_bounds_about_ellipses(self, offset=0):
        # Rotates the chamber boundaries to align with the orientation of ellipses, based on the
        # specified boundary type and combination. This method adjusts the chamber's original
        # boundaries to match the orientation of ellipses representing entities in a video,
        # facilitating accurate distance calculations between these entities and the chamber
        # walls. It supports different configurations for agarose and non-agarose (wall)
        # boundaries, including adjustments for 'agarose_adj' boundary combinations.
        #
        # Depending on the `boundary_type` ('agarose' or 'wall') and `boundary_combo`
        # ('agarose_adj'), this method selects the appropriate pre-cached boundary values or
        # calculates new boundaries by rotating the original chamber bounds to align with the
        # ellipses' orientations. It also applies an offset for the 'agarose_adj' type to
        # account for the specific agarose area configuration.
        #
        # Parameters:
        # - offset: float (default: 0)
        #     Distance from the top/bottom walls of the chamber at which to define the boundaries.
        #     Note: units are mm, and these offsets are scaled relative to the
        #     (HTL) chamber height of 16mm.
        #
        # Note:
        # - This method assumes that original boundary positions and ellipse orientations
        #   (`rot_angles`, `sin_of_angles`, and `cos_of_angles`) have already been set during
        #   the initialization or prior processing steps.
        # - The rotation adjusts both the x and y bounds of the chamber to reflect the
        #   orientation of ellipses, enhancing the precision of distance-to-boundary
        #   calculations.
        # - After rotation, boundaries are either stored for future use (`bounds_agarose` and
        #   `y_bounds_agarose` for agarose-related calculations, `bounds_full` and
        #   `y_bounds_full` for wall-related calculations) or applied immediately for the
        #   current analysis.
        #
        # Modifies:
        # - `self.bounds`: The rotated boundaries aligned with the ellipses' orientations,
        #                   structured as a 4x2 numpy array representing the corner points of the chamber.
        # - `self.y_bounds`: The y-axis boundaries after potential adjustment for the agarose
        #                     area, reflecting the vertical span of the chamber after rotation.
        # - Caches the calculated boundaries in `self.bounds_agarose`, `self.y_bounds_agarose`,
        #   `self.bounds_full`, or `self.y_bounds_full` for efficiency in subsequent analyses.
        #
        # Raises:
        # - No explicit exceptions are raised by this method, but it relies on the presence and
        #   correctness of several class attributes (`self.bounds_orig`, `self.x_bounds_orig`,
        #   `self.y_bounds_orig`, etc.) set by other methods in the class.
        if self.boundary_type == 'agarose' or (
            self.boundary_type == 'wall' and self.boundary_combo == 'agarose_adj'
        ):
            cached_bounds = self.bounds_agarose
            cached_y_bounds = self.y_bounds_agarose
        else: # wall-contact analysis, either for all walls or left-right only
            cached_bounds = self.bounds_full
            cached_y_bounds = self.y_bounds_full
        if cached_bounds is not None and cached_y_bounds is not None:
            self.bounds = np.copy(cached_bounds)
            self.y_bounds = np.copy(cached_y_bounds)
            return
        self.y_bounds[0] = self.bounds_orig[0][1]
        self.y_bounds[1] = self.bounds_orig[1][1]

        if (
            self.boundary_combo == "agarose_adj" or
            self.boundary_type in ("agarose", "boundary")
        ):
            bounds_offset = (offset / 16) * (
                self.y_bounds_orig[1] - self.y_bounds_orig[0]
            )
            self.y_bounds[0] += bounds_offset
            self.y_bounds[1] -= bounds_offset
        self.bounds = np.full((len(self.x), 4, 2), np.nan)
        self.bounds = np.array(
            [
                [self.x_bounds_orig[0], self.y_bounds[0]],
                [self.x_bounds_orig[0], self.y_bounds[1]],
                [self.x_bounds_orig[1], self.y_bounds[1]],
                [self.x_bounds_orig[1], self.y_bounds[0]],
            ]
        )

        self.bounds = rotate_pts(
            self.bounds,
            self.cos_of_angles,
            self.sin_of_angles,
            self.origins,
            multiple_pts_per_rotation=True,
        )
        if self.boundary_type == 'agarose' or (
            self.boundary_type == 'wall' and self.boundary_combo == 'agarose_adj'
        ):
            self.bounds_agarose = np.copy(self.bounds)
            self.y_bounds_agarose = np.copy(self.y_bounds)
        else:
            self.bounds_full = np.copy(self.bounds)

    cdef void calculate_slope_intercept(
        self,
        int wall_idx,
        int upper_index,
        int lower_index,
    ):
        # Calculates the slope and intercept for the line representing a specified wall of the chamber.
        #
        # This helper function computes the slope and intercept of the line that defines one of the
        # four chamber walls. The wall is identified by `wall_idx`, with the slope and intercept
        # calculated using the coordinates of two corner points, determined by `upper_index` and 
        # `lower_index`, from the `bounds` array. The calculated slope and intercept are stored 
        # in the `slope_intercept_sets` array for use in subsequent distance calculations and 
        # boundary contact analyses.
        #
        # The `bounds` array defines the chamber corners in the following order:
        # - [x_min, y_min] (upper left)
        # - [x_min, y_max] (lower left)
        # - [x_max, y_max] (lower right)
        # - [x_max, y_min] (upper right)
        #
        # Wall indices (`wall_idx`) are mapped as follows:
        # - 0: Left wall (from upper left to lower left)
        # - 1: Bottom wall (from lower left to lower right)
        # - 2: Right wall (from lower right to upper right)
        # - 3: Top wall (from upper right to upper left)
        #
        # Parameters:
        # - wall_idx : int
        #     The index representing the wall for which the slope and intercept are being calculated.
        #     This determines the entry in the `slope_intercept_sets` array that will be updated.
        # - upper_index : int
        #     The index representing the boundary point with the smaller Y value (upper) used in the 
        #     slope calculation. This point is one of the chamber corners defined in the `bounds` array.
        # - lower_index : int
        #     The index representing the boundary point with the larger Y value (lower) used in the 
        #     slope calculation. This point is one of the chamber corners defined in the `bounds` array.
        #
        # Modifies:
        # - `self.slope_intercept_sets`: Updates the slope and intercept values for the line 
        #   representing the wall specified by `wall_idx`. The slope is stored in the first 
        #   position and the intercept in the second position of the respective slice for `wall_idx`.
        self.slope_intercept_sets[:, wall_idx, 0] = (
            self.bounds[:, upper_index, 1] - self.bounds[:, lower_index, 1]
        ) / (self.bounds[:, upper_index, 0] - self.bounds[:, lower_index, 0])
        self.slope_intercept_sets[:, wall_idx, 1] = (
            self.bounds[:, upper_index, 1]
            - self.slope_intercept_sets[:, wall_idx, 0]
            * self.bounds[:, upper_index, 0]
        )

    cdef cnp.ndarray[double, ndim=1] calculate_angle_to_best_ell_pt(
        self,
        cnp.ndarray[double, ndim=1] semimaj_ax,
        cnp.ndarray[double, ndim=1] semimin_ax,
        int wall_idx
    ):
        # Calculates the angle to the best (closest) point on an ellipse relative to a specified wall.
        #
        # This method calculates the angle between the semi-major axis of the ellipse and the slope
        # of the wall specified by `wall_idx`. The calculation uses the slope of the wall, derived 
        # from the previously calculated slope and intercept values, and the semi-minor axis of the 
        # ellipse. The result is the angle between the semi-major axis and the line representing the 
        # wall, which is used to determine the best (closest) points on the ellipse relative to the 
        # wall.
        #
        # Parameters:
        # - semimaj_ax : cnp.ndarray[double, ndim=1]
        #     A NumPy array containing the lengths of the semi-major axes of the ellipses. These 
        #     values are used to calculate the angles relative to the wall.
        # - semimin_ax : cnp.ndarray[double, ndim=1]
        #     A NumPy array containing the lengths of the semi-minor axes of the ellipses. These 
        #     values are used in conjunction with the slope of the wall to calculate the angles.
        # - wall_idx : int
        #     The index of the wall for which the angle is being calculated. This index corresponds 
        #     to the walls as represented in the `slope_intercept_sets` array, with wall_idx=0 for 
        #     the left wall, wall_idx=1 for the bottom wall, wall_idx=2 for the right wall, and 
        #     wall_idx=3 for the top wall.
        #
        # Returns:
        # - cnp.ndarray[double, ndim=1]
        #     A NumPy array containing the calculated angles for each ellipse, representing the 
        #     angle between the semi-major axis of the ellipse and the line representing the specified 
        #     wall.
        return np.arctan2(-semimaj_ax, self.slope_intercept_sets[:, wall_idx, 0] * semimin_ax)

    cdef tuple calculate_ellipse_points(
        self,
        angle_to_best_ell_pt,
        cnp.ndarray[double, ndim=1] semimin_ax,
        cnp.ndarray[double, ndim=1] semimaj_ax
    ):
        # Calculates the baseline and alternate points on an ellipse relative to a specified wall.
        #
        # This method computes two sets of points on an ellipse: baseline points, which are the closest 
        # points on the ellipse relative to the wall, and alternate points, which are calculated by 
        # reflecting the baseline points across the ellipse's center. These points are used to evaluate 
        # the proximity of the ellipse to the specified wall and to check for out-of-bound conditions.
        #
        # Parameters:
        # - angle_to_best_ell_pt : array-like
        #     The angles between the semi-major axes of the ellipses and the line representing the specified 
        #     wall. These angles are used to determine the direction of the closest points on the ellipses.
        # - semimin_ax : cnp.ndarray[double, ndim=1]
        #     A NumPy array containing the lengths of the semi-minor axes of the ellipses. These values are 
        #     used to calculate the x-coordinates of the baseline points.
        # - semimaj_ax : cnp.ndarray[double, ndim=1]
        #     A NumPy array containing the lengths of the semi-major axes of the ellipses. These values are 
        #     used to calculate the y-coordinates of the baseline points.
        #
        # Returns:
        # - tuple
        #     A tuple containing two NumPy arrays:
        #     1. Baseline points (array of shape [n, 2]): The closest points on the ellipses relative to 
        #        the specified wall.
        #     2. Alternate points (array of shape [n, 2]): Points calculated by reflecting the baseline 
        #        points across the center of the ellipses.
        # Initialize arrays for baseline and alternate points
        cdef cnp.ndarray[double, ndim=2] baseline_pts = np.empty((len(self.x), 2))
        cdef cnp.ndarray[double, ndim=2] alternate_pts
        
        # Calculate baseline points
        baseline_pts[:, 0] = semimin_ax * np.cos(angle_to_best_ell_pt) + self.x
        baseline_pts[:, 1] = semimaj_ax * np.sin(angle_to_best_ell_pt) + self.y
        
        # Calculate alternate points by reflecting the baseline points
        alternate_pts = np.vstack(
            (
                baseline_pts[:, 0] + 2 * (self.x - baseline_pts[:, 0]),
                baseline_pts[:, 1] + 2 * (self.y - baseline_pts[:, 1]),
            )
        ).T
        
        return baseline_pts, alternate_pts

    cdef void _calc_distances_to_single_wall(self, str wall):
        # Calculates distances from ellipses to a specified wall within a predefined boundary.
        #
        # This method calculates the distances between ellipses, representing entities within
        # the space, and a specific wall (left, right, top, or bottom) of the chamber boundary.
        # It considers various factors such as the boundary combination (e.g., top-bottom,
        # left-right), the orientation and position of the ellipses, and out-of-bound conditions.
        # The method ensures accurate distance measurements by dynamically calculating and using
        # slope and intercept values, angles, and best points on the ellipses relative to the wall.
        #
        # Parameters:
        # - wall : str
        #     The wall to which distances are being calculated. Can be 'left', 'right', 'top',
        #     or 'bottom'. The method also supports specifying walls flexibly by including 'bottom'
        #     as part of a substring.
        #
        # Modifies:
        # - `self.slope_intercept_sets` : 
        #     Calculates and updates slope and intercept values for the line representing the specified wall.
        # - `self.angle_to_best_ell_pt[wall]` : 
        #     Determines and stores the angles to the points on the ellipses closest to the specified wall, 
        #     based on the wall's orientation and the ellipses' geometry.
        # - `self.ell_pts[wall]` : 
        #     Updates with the points on ellipses that are closest to the specified wall, including both 
        #     baseline and alternate points for handling out-of-bound conditions.
        # - `self.distances[wall]` : 
        #     Records calculated distances from ellipses to the closest points on the specified wall, 
        #     which are essential for boundary analysis and contact detection.
        # - `self.wall_pts[wall]` : 
        #     Stores the coordinates of points on the specified wall that are closest to the ellipses, 
        #     which are crucial for accurate distance measurement and boundary event processing.
        # - `self.oob_test_pts[wall]` : 
        #     Updates with points used to test whether an ellipse is out of bounds relative to the specified wall, 
        #     aiding in boundary condition evaluation.
        # - `self.best_wall_indices`, `self.min_distances` : 
        #     Updates these arrays to indicate the closest wall and the minimum distances to the wall for each 
        #     ellipse, based on the latest calculations.
        # - `self.best_pts["ellipse"]`, `self.best_pts["boundary"]` : 
        #     Updates with the best points of contact between ellipses and boundaries, reflecting the closest 
        #     interactions with the specified wall.
        # - `self.out_of_bounds` : 
        #     Updates flags indicating whether any part of an ellipse is out of bounds, which is crucial for 
        #     boundary crossing detection and handling.
        # - `self.oob_on_ignored_wall` : 
        #     Specifically updates for cases where the boundary combination excludes consideration of certain 
        #     walls, indicating out-of-bound conditions on ignored boundaries.
        #
        # Notes:
        # - The method dynamically calculates slope and intercept values for the line representing the walls, 
        #   and uses these to determine the closest points on the ellipses to the walls.
        # - It handles special cases based on the boundary combination (e.g., ignoring certain walls if the 
        #   combination is top-bottom or left-right) and adjusts calculations accordingly.
        # - The method also updates indicators for out-of-bound conditions, which are crucial for determining 
        #   if an entity represented by an ellipse crosses the boundaries of the space.

        cdef int wall_idx, first_idx, upper_index, lower_index
        cdef double[:] angle_to_best_ell_pt
        # Determine wall index
        if wall == 'left':
            wall_idx = 0
        elif 'bottom' in wall:
            wall_idx = 1
        elif wall == 'right':
            wall_idx = 2
        else:
            wall_idx = 3


        # Determine boundary combination indices
        first_idx = 1 if self.boundary_combo in ("tb", "agarose_adj") else 0
        if self.boundary_combo in ("tb", "agarose_adj") and wall_idx in (0, 2):
            return
        if self.boundary_combo == 'lr' and wall_idx in (1, 3):
            return

        upper_index = wall_idx + 1 if wall_idx < 3 else 0
        lower_index = wall_idx

        # Calculate slope and intercept for the wall if not already done
        if not wall in self.angle_to_best_ell_pt:
            self.calculate_slope_intercept(wall_idx, upper_index, lower_index)
            angle_to_best_ell_pt = self.calculate_angle_to_best_ell_pt(
                self.semimaj_ax, self.semimin_ax, wall_idx
            )
            self.angle_to_best_ell_pt[wall] = angle_to_best_ell_pt
        else:
            angle_to_best_ell_pt = self.angle_to_best_ell_pt[wall]

        # Calculate ellipse points if not already done
        if self.ellipse_ref_pt == 'edge' and not wall_idx in self.ell_pts:
            baseline_pts, alternate_pts = self.calculate_ellipse_points(
                angle_to_best_ell_pt, self.semimin_ax, self.semimaj_ax
            )
            self.ell_pts[wall] = [baseline_pts, alternate_pts]
        for k, ellipse_pts in enumerate(self.ell_pts[wall]
                if self.ellipse_ref_pt == 'edge' else [np.vstack((self.x, self.y)).T]):
            if not wall in self.distances or not k in self.distances[wall]:
                distances, wall_pts = closest_pts_to_lines(
                    self.bounds[:, lower_index, :], self.bounds[:, upper_index, :], ellipse_pts
                )

                # For out of bounds test
                pts_for_oob_test = rotate_pts(
                    ellipse_pts, self.cos_of_angles, self.neg_sin_of_angles, self.origins
                )

                if not wall in self.distances:
                    self.distances[wall] = {k: distances}
                    self.wall_pts[wall] = {k: wall_pts}
                    self.oob_test_pts[wall] = {k: pts_for_oob_test}
                else:
                    self.distances[wall][k] = distances
                    self.wall_pts[wall][k] = wall_pts
                    self.oob_test_pts[wall][k] = pts_for_oob_test
            else:
                distances = self.distances[wall][k]
                wall_pts = self.wall_pts[wall][k]
                pts_for_oob_test = self.oob_test_pts[wall][k]

            if wall_idx == first_idx and k == 0:
                dist_comp = np.full_like(distances, True)
            else:
                dist_comp = distances < self.min_distances

            self.best_wall_indices = np.where(dist_comp, wall_idx, self.best_wall_indices)
            if self.ellipse_ref_pt == 'edge':
                if self.ellipse_edge_pt == "opposite":
                    chosen_ellipse_pts = self.ell_pts[wall][1 - k]  # Swap the index
                else:
                    chosen_ellipse_pts = self.ell_pts[wall][k]  # Use the current index
            else:
                chosen_ellipse_pts = ellipse_pts
            self.min_distances = np.where(dist_comp, distances, self.min_distances)

            self.best_pts["ellipse"] = np.where(
                np.expand_dims(dist_comp, 1),
                chosen_ellipse_pts,
                self.best_pts["ellipse"]
            )
            self.best_pts["boundary"] = np.where(np.expand_dims(dist_comp, 1), wall_pts, self.best_pts["boundary"])

            if self.boundary_combo == "tb":
                x_offset = (1 / 10) * (self.x_bounds_orig[1] - self.x_bounds_orig[0])
            else:
                x_offset = 0

            oob_xmin = pts_for_oob_test[:, 0] < (self.x_bounds_orig[0] + x_offset)
            oob_xmax = pts_for_oob_test[:, 0] > (self.x_bounds_orig[1] - x_offset)
            oob_ymin = pts_for_oob_test[:, 1] < self.y_bounds[0]
            oob_ymax = pts_for_oob_test[:, 1] > self.y_bounds[1]

            if self.boundary_combo not in ("all", "lr"):
                self.oob_on_ignored_wall = np.logical_or(
                    self.oob_on_ignored_wall,
                    (oob_xmin | oob_xmax)
                    if self.boundary_combo == "tb"
                    else (oob_ymin | oob_ymax),
                )
            self.out_of_bounds = np.logical_or(
                self.out_of_bounds,
                (False if self.boundary_combo == "tb" else oob_xmin)
                | (False if self.boundary_combo == "tb" else oob_xmax)
                | (False if self.boundary_combo == "lr" else oob_ymin)
                | (False if self.boundary_combo == "lr" else oob_ymax),
            )
    def _find_boundary_contact_events(self, near_contact):
        # Detects boundary contact events by analyzing the ellipse's positions relative 
        # to the specified boundaries.
        #
        # This method identifies when an ellipse crosses into or out of a boundary-
        # defined region within the chamber. It calculates the relevant distances to 
        # walls, checks for boundary crossings, and tracks the start and end of 
        # boundary contact events. The method supports both close contact events (near 
        # contact) and standard events, adjusting thresholds and distances as needed.
        #
        # Parameters:
        # - near_contact : bool
        #     A flag indicating whether to adjust thresholds for detecting near contact 
        #     events. If True, the method increases sensitivity to boundary proximity 
        #     by slightly adjusting the start and end thresholds.
        #
        # Modifies:
        # - `self.dist_to_wall` : 
        #     Calculates and stores the distances from ellipses to the closest walls, 
        #     normalized by pixel-to-millimeter conversion and a scaling factor. If 
        #     this attribute already exists, it reuses the stored distances.
        # - `self.wall_is_vert` : 
        #     Stores a boolean array indicating whether the closest wall for each 
        #     ellipse is vertical (left or right).
        # - `self.return_data` : 
        #     Updates with information related to detected boundary contact events, 
        #     including which frames involve contact and whether the events are near 
        #     contact or standard.
        #
        # Notes:
        # - The method first checks if the distance to the wall has already been 
        #   calculated. If not, it computes and stores the distances, accounting for 
        #   out-of-bound conditions by setting distances to zero where applicable.
        # - The method then establishes start and end thresholds for each boundary 
        #   based on the specified boundary combination (e.g., left-right or top-bottom 
        #   boundaries).
        # - The method iterates through each frame, checking whether the ellipse has 
        #   crossed the boundary start or end points, and flags the frame as a boundary 
        #   contact if the criteria are met.
        # - The method differentiates between standard and near contact events, 
        #   allowing for finer detection of close proximity to boundaries.

        if not hasattr(self, "distToWall"):
            dist_to_wall = (
                np.where(self.out_of_bounds, 0, self.min_distances)
                / self.va.ct.pxPerMmFloor()
                / self.va.xf.fctr
            )
            wall_is_vert = (
                (self.best_wall_indices == 0) | (self.best_wall_indices == 2)
            )
            self.dist_to_wall = dist_to_wall
            self.wall_is_vert = wall_is_vert
        else:
            dist_to_wall = self.dist_to_wall
            wall_is_vert = self.wall_is_vert

        # Calculate start and end boundaries based on boundary_combo
        boundaries = {}
        if self.boundary_combo in ['all', 'lr']:  # Left and Right boundaries
            boundaries['left_start'] = (
                self.x_bounds_orig[0]
                + self.event_thresholds["vert"]["start"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['right_start'] = (
                self.x_bounds_orig[1]
                - self.event_thresholds["vert"]["start"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['left_end'] = (
                self.x_bounds_orig[0]
                + self.event_thresholds["vert"]["end"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['right_end'] = (
                self.x_bounds_orig[1]
                - self.event_thresholds["vert"]["end"] * self.va.ct.pxPerMmFloor()
            )

        if self.boundary_combo in ['all', 'tb', 'agarose_adj']:  # Top and Bottom boundaries
            boundaries['top_start'] = (
                self.y_bounds[0]
                + self.event_thresholds["horiz"]["start"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['bottom_start'] = (
                self.y_bounds[1]
                - self.event_thresholds["horiz"]["start"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['top_end'] = (
                self.y_bounds[0]
                + self.event_thresholds["horiz"]["end"] * self.va.ct.pxPerMmFloor()
            )
            boundaries['bottom_end'] = (
                self.y_bounds[1]
                - self.event_thresholds["horiz"]["end"] * self.va.ct.pxPerMmFloor()
            )

        # Map boundaries to indices for easy access
        start_boundaries = {}
        end_boundaries = {}

        if self.boundary_combo in ['all', 'lr']:
            start_boundaries[0] = boundaries['left_start']  # Left boundary
            start_boundaries[2] = boundaries['right_start']  # Right boundary
            end_boundaries[0] = boundaries['left_end']  # Left boundary
            end_boundaries[2] = boundaries['right_end']  # Right boundary

        if self.boundary_combo in ['all', 'tb', 'agarose_adj']:
            start_boundaries[1] = boundaries['bottom_start']  # Bottom boundary
            start_boundaries[3] = boundaries['top_start']  # Top boundary
            end_boundaries[1] = boundaries['bottom_end']  # Bottom boundary
            end_boundaries[3] = boundaries['top_end']  # Top boundary

        def visualize_ellipse_and_boundaries(center_x, center_y, theta, semimaj_ax, semimin_ax, boundaries, boundary_index, points_to_check):
            fig, ax = plt.subplots()
            
            # Plot the boundaries
            if boundary_index in [0, 2]:  # Left-Right boundary
                ax.axvline(x=boundaries['left_start'], color='r', linestyle='--', label='Left Boundary Start')
                ax.axvline(x=boundaries['right_start'], color='g', linestyle='--', label='Right Boundary Start')
                ax.axvline(x=boundaries['left_end'], color='r', linestyle='-', label='Left Boundary End')
                ax.axvline(x=boundaries['right_end'], color='g', linestyle='-', label='Right Boundary End')
            elif boundary_index in [1, 3]:  # Top-Bottom boundary
                ax.axhline(y=boundaries['top_start'], color='b', linestyle='--', label='Top Boundary Start')
                ax.axhline(y=boundaries['bottom_start'], color='y', linestyle='--', label='Bottom Boundary Start')
                ax.axhline(y=boundaries['top_end'], color='b', linestyle='-', label='Top Boundary End')
                ax.axhline(y=boundaries['bottom_end'], color='y', linestyle='-', label='Bottom Boundary End')

            # Plot the ellipse using the Ellipse patch
            ellipse_patch = patches.Ellipse(
                (center_x, center_y), width=2*semimin_ax, height=2*semimaj_ax, angle=theta, edgecolor='b', facecolor='none', label='Ellipse'
            )
            ax.add_patch(ellipse_patch)

            # Plot the four points on the ellipse
            for (x, y) in points_to_check:
                ax.plot(x, y, 'ro')  # Red dots for the points on the ellipse

            # Set plot limits for better visualization
            ax.set_xlim([center_x - 2*semimaj_ax, center_x + 2*semimaj_ax])
            ax.set_ylim([center_y - 2*semimaj_ax, center_y + 2*semimaj_ax])
            
            ax.set_aspect('equal', 'box')
            plt.legend()
            plt.title(f'Ellipse and Boundaries (Boundary Index: {boundary_index})')
            plt.show()

        def ellipse_edge_points_within_boundaries(center_x, center_y, theta, semimaj_ax, semimin_ax, boundary_index):
            """Helper function to check if all points on the ellipse's edge are within the boundaries."""
            # Convert theta to radians for calculation
            if np.any(np.isnan([semimaj_ax, semimin_ax])):
                return
            theta_rad = radians(theta)
    
            # Rotation matrix for transforming the ellipse
            R = np.array([[-np.sin(theta_rad), np.cos(theta_rad)], 
                        [np.cos(theta_rad), np.sin(theta_rad)]])
            
            # Generate the four extreme points of the ellipse before rotation
            edge_points = np.array([[0, semimin_ax],  # Point along the semi-minor axis
                                    [0, -semimin_ax],
                [semimaj_ax, 0],  # Point along the semi-major axis
                                    [-semimaj_ax, 0],  # Opposite point on the semi-major axis
                                    ])  # Opposite point on the semi-minor axis
            
            # Apply the rotation matrix to each point to get the correct coordinates
            rotated_points = np.dot(edge_points, R.T)
            
            # Translate the points to the ellipse's center
            points_to_check = [
                (round(center_x + rotated_points[i, 0]), round(center_y + rotated_points[i, 1]))
                for i in range(4)
            ]


            if self.boundary_combo == 'all' and self.ellipse_edge_pt == 'opposite':
                boundaries_to_check = [
                    round(self.x_bounds_orig[0]),
                    round(self.y_bounds[1]),
                    round(self.x_bounds_orig[1]),
                    round(self.y_bounds[0]),
                ]
            else:
                boundaries_to_check = start_boundaries
            # Check if all these points are within the respective boundaries
            for (ellipse_x, ellipse_y) in points_to_check:
                if boundary_index in [0, 2]:  # Left or Right boundary
                    if ellipse_x < boundaries_to_check[0] or ellipse_x > boundaries_to_check[2]:
                        visualize_ellipse_and_boundaries(
                            center_x, center_y, theta, semimaj_ax, semimin_ax, boundaries, boundary_index, points_to_check
                        )
                        return False
                elif boundary_index in [1, 3]:  # Top or Bottom boundary
                    if ellipse_y > boundaries_to_check[1] or ellipse_y < boundaries_to_check[3]:
                        visualize_ellipse_and_boundaries(
                            center_x, center_y, theta, semimaj_ax, semimin_ax, boundaries, boundary_index, points_to_check
                        )
                        return False

            return True

        # Initialize event status
        in_event = False
        wall_contact = np.zeros_like(self.best_pts['ellipse'][:, 0], dtype=bool)

        # Loop through frames to detect boundary crossing events
        for i in range(len(self.best_pts['ellipse'])):
            ellipse_x, ellipse_y = self.best_pts['ellipse'][i]
            ellipse_center_x = self.trj.x[i]
            ellipse_center_y = self.trj.y[i]
            boundary_index = self.best_wall_indices[i]

            if in_event and not self.out_of_bounds[i]:
                if self.ellipse_ref_pt == 'edge' and self.ellipse_edge_pt == 'opposite':
                    if boundary_index in [0, 2]:  # Vertical boundaries
                        if boundary_index == 0 and ellipse_center_x > start_boundaries[0]:
                            in_event = False
                        elif boundary_index == 2 and ellipse_center_x < start_boundaries[2]:
                            in_event = False
                    else:  # Horizontal boundaries
                        if boundary_index == 3 and ellipse_center_y > start_boundaries[3]:
                            in_event = False
                        elif boundary_index == 1 and ellipse_center_y < start_boundaries[1]:
                            in_event = False
                else:
                    if self._crossed_boundary(
                        ellipse_x,
                        ellipse_y,
                        end_boundaries[boundary_index], 
                        boundary_index, toward_center=True
                    ):
                        # Ensure the ellipse center is now fully inside the start boundaries to end the event
                        in_event = False
                if (
                    self.wall_debug and 
                    not in_event and
                    self.boundary_combo == 'all' and
                    self.boundary_type == 'wall'
                ):
                    within_bounds = ellipse_edge_points_within_boundaries(
                        center_x=self.trj.x[i],
                        center_y=self.trj.y[i],
                        theta=self.trj.theta[i],
                        semimaj_ax=self.semimaj_ax[i],
                        semimin_ax=self.semimin_ax[i],
                        boundary_index=boundary_index
                    )
                    if within_bounds is not None and not within_bounds:
                        raise AssertionError(
                            f"Ellipse not fully contained within boundaries after wall-contact event at frame {i}. "
                            f"Center (x, y): ({self.trj.x[i]}, {self.trj.y[i]}), Rotation: {self.trj.theta[i]}, "
                            f"Semi-major axis: {self.semimaj_ax[i]}, Semi-minor axis: {self.semimin_ax[i]}, "
                            f"Boundary index: {boundary_index}, Fly index: {self.trj.f}, Boundary combination: "
                            f"{self.boundary_combo}, Ellipse ref pt: {self.ellipse_ref_pt}, Ellipse edge pt: "
                            f"{self.ellipse_edge_pt}"
                        )

            # Start a new event if it hasn't started yet
            if not in_event and self._crossed_boundary(
                ellipse_x, ellipse_y, start_boundaries[boundary_index], 
                boundary_index
            ):
                in_event = True

            # Mark wall contact if currently in an event
            if in_event:
                wall_contact[i] = True

        self.update_return_data_for_boundary_contact_stats(
            wall_contact, near_contact, dist_to_wall
        )

    def _crossed_boundary(self, x, y, boundary, boundary_index, toward_center=False):
        # Determines if an ellipse has crossed a specified boundary in the chamber.
        #
        # This method checks whether the point (x, y) on an ellipse has crossed a boundary defined by the
        # boundary index and the given boundary value. It considers both vertical and horizontal boundaries
        # and can determine crossing either towards the center of the chamber or away from it.
        #
        # Parameters:
        # - x : float
        #     The x-coordinate of the point on the ellipse being checked.
        # - y : float
        #     The y-coordinate of the point on the ellipse being checked.
        # - boundary : float
        #     The boundary value against which the point is being checked. This could represent the x-coordinate 
        #     for vertical boundaries (left or right) or the y-coordinate for horizontal boundaries (top or bottom).
        # - boundary_index : int
        #     The index representing the boundary being checked. Vertical boundaries use indices 0 (left) and 2 (right), 
        #     while horizontal boundaries use indices 1 (bottom) and 3 (top).
        # - toward_center : bool, optional
        #     A flag indicating the direction of the check. If True, the method checks if the point is crossing the 
        #     boundary towards the center of the chamber. If False, it checks if the point is crossing away from the center.
        #
        # Returns:
        # - bool
        #     True if the point has crossed the boundary in the specified direction; False otherwise.
        #
        # Notes:
        # - Vertical boundaries are checked by comparing the x-coordinate to the boundary value, while horizontal boundaries 
        #   are checked using the y-coordinate.
        # - The direction of crossing (toward or away from the center) is determined by the `toward_center` flag.
        if boundary_index in [0, 2]:  # Vertical boundaries: 0 (left) and 2 (right)
            if toward_center:
                if boundary_index == 0:  # Left boundary
                    return x > boundary
                else:  # Right boundary
                    return x < boundary
            else:
                if boundary_index == 0:  # Left boundary
                    return x < boundary
                else:  # Right boundary
                    return x > boundary
        else:  # Horizontal boundaries: 1 (bottom) and 3 (top)
            if toward_center:
                if boundary_index == 3:  # Top boundary
                    return y > boundary
                else:  # Bottom boundary
                    return y < boundary
            else:
                if boundary_index == 3:  # Top boundary
                    return y < boundary
                else:  # Bottom boundary
                    return y > boundary

    def update_return_data_for_boundary_contact_stats(self, wall_contact, near_contact, dist_to_wall=None):
        # Updates the return data structure with statistics related to boundary contact events.
        #
        # This method processes identified boundary contact events, whether they are general wall contacts
        # or near contacts, and updates the relevant entries in the return data structure with detailed 
        # statistics. This includes information such as contact regions, start indices, closest boundary indices,
        # and optionally distances to the boundaries if provided.
        #
        # Parameters:
        # - wall_contact : ndarray of bool
        #     An array indicating the frames where boundary contact events have occurred.
        # - near_contact : bool
        #     A flag indicating whether the contacts being processed are near contacts. This flag adjusts how 
        #     contact events are recorded, preventing double-counting and refining near contact regions.
        # - dist_to_wall : ndarray of float, optional
        #     An array representing the distances to the wall at each frame. This parameter is optional and 
        #     relevant primarily for 'edge' boundary contacts where distance data is necessary.
        #
        # Modifies:
        # - `self.return_data["boundary_event_stats"]` : 
        #     Updates this nested dictionary structure with detailed statistics for boundary events, including
        #     contact regions, start indices, closest boundary indices, contact counts by direction, and distances 
        #     to the boundary if provided.
        # - `self.return_data["best_pts"]` : 
        #     Updates the best points for the ellipse and boundary contact.
        # - `self.return_data["bounds_orig"]` : 
        #     Records the original boundary bounds for reference.
        # - `self.return_data["<threshold_keys>"]` : 
        #     Adds entries for event start and end distance thresholds based on the boundary type and combination.
        #
        # Notes:
        # - The method first ensures that the input `wall_contact` is in boolean format, which it uses to determine
        #   contact regions and refine the contact data based on the type of contact (standard or near).
        # - Near contact events are handled with additional checks to prevent double-counting and to ensure that 
        #   only the most accurate regions are recorded.
        # - For standard contact events, the method also records start indices and direction-specific contact counts,
        #   and updates the return data with all relevant statistics.
        # - Event thresholds, if available, are also added to the return data to reflect the start and end distances 
        #   used for event detection.
        #
        # Raises:
        # - This method does not explicitly raise exceptions but relies on the correct initialization and integrity 
        #   of required data structures. Missing or incorrect data may result in inaccurate event statistics.
        if wall_contact.dtype == bool:
            wall_contact_bool = wall_contact
        else:
            wall_contact_bool = wall_contact == 1

        # Store the original wall_contact_bool before any modifications
        original_wall_contact_bool = wall_contact_bool.copy()

        contact_regions = trueRegions(wall_contact_bool)[1:]
        if self.boundary_type == 'wall' and self.boundary_combo == "tb":
            for reg in contact_regions:
                if self.oob_on_ignored_wall[reg.start] or self.return_data[
                    'boundary_event_stats'
                ]['wall']['all']['edge']['boundary_contact'][reg.start]:
                    wall_contact_bool[reg] = False
            contact_regions = trueRegions(wall_contact_bool)[1:]
        if self.boundary_combo == "lr":
            for reg in contact_regions:
                if self.best_wall_indices[reg.start] in (1, 3):
                    wall_contact_bool[reg] = False
            contact_regions = trueRegions(wall_contact_bool)[1:]
        if self.ellipse_ref_pt == "edge":
            if self.ellipse_edge_pt == 'opposite':
                self.ellipse_ref_pt = "opp_edge"
        if near_contact:
            dict_to_update = self.return_data["boundary_event_stats"][
                self.boundary_type
            ][self.boundary_combo][self.ellipse_ref_pt]
            for reg in contact_regions:
                if np.any(dict_to_update["boundary_contact"][reg.start : reg.stop]):
                    wall_contact_bool[reg.start : reg.stop] = False
            dict_to_update["near_contact_regions"] = trueRegions(wall_contact_bool)
            dict_to_update["near_contact_start_idxs"] = np.array(
                [s.start for s in dict_to_update["near_contact_regions"]]
            )
            dict_to_update["near_contact"] = wall_contact_bool
            if self.boundary_type != "agarose":
                self.return_data["best_pts"] = self.return_data["boundary_event_stats"][
                    self.boundary_type
                ][self.boundary_combo][self.ellipse_ref_pt]["ellipse_and_boundary_pts"]
        else:
            wall_contact_start_idxs = np.array([s.start for s in contact_regions])
            try:
                if self.best_wall_indices is None:
                    raise AttributeError
                pre_trn_best_wall_indices = self.best_wall_indices[
                    [s for s in wall_contact_start_idxs if s < self.va.trns[0].start]
                ]
                h = np.histogram(pre_trn_best_wall_indices, bins=4, range=(-0.1, 4))
                has_best_wall_indices = True
            except AttributeError:
                has_best_wall_indices = False
            event_stats = {
                "bounds": {"x": self.x_bounds_orig, "y": self.y_bounds},
                "boundary_contact": (
                    wall_contact_bool
                    if wall_contact.dtype == bool
                    else wall_contact
                ),
                "original_boundary_contact": original_wall_contact_bool,
                "boundary_contact_regions": contact_regions,
                "contact_start_idxs": wall_contact_start_idxs,
                "closest_boundary_indices": self.best_wall_indices,
                "ellipse_and_boundary_pts": self.best_pts,
            }
            if has_best_wall_indices:
                event_stats["boundary_touch_by_direction"] = {
                    "left": h[0][0],
                    "right": h[0][2],
                    "top": h[0][-1],
                    "bottom": h[0][1],
                }
            if dist_to_wall is not None:
                event_stats["dist_to_boundary"] = dist_to_wall
            self.return_data["boundary_event_stats"][self.boundary_type][
                self.boundary_combo
            ][self.ellipse_ref_pt] = event_stats
            self.return_data["bounds_orig"] = self.bounds_orig
            try:
                if self.event_thresholds is None:
                    raise AttributeError
                for ori in self.event_thresholds:
                    k = "%s_%s_event_start_dist_threshold_%s" % (
                        self.boundary_type,
                        self.boundary_combo,
                        ori,
                    )
                    self.return_data[k] = self.event_thresholds[ori]["start"]
                    k = "%s_%s_event_end_dist_threshold_%s" % (
                        self.boundary_type,
                        self.boundary_combo,
                        ori,
                    )
                    self.return_data[k] = self.event_thresholds[ori]["end"]
            except AttributeError:
                print('No event thresholds available for', self.boundary_type, self.boundary_combo)

    def visualize_turns(self, ellipse_ref_pt, opts, mode='troubleshooting'):
        """
        Visualizes boundary contact events and turns by plotting the trajectory and 
        marking key events.
        """
        bcr = self.return_data["boundary_event_stats"][self.boundary_type][
            self.boundary_combo
        ][ellipse_ref_pt]["boundary_contact_regions"]
        turning_idxs_filtered = self.return_data["boundary_event_stats"][
            self.boundary_type
        ][self.boundary_combo][ellipse_ref_pt]["turning_indices"]

        start_frame = opts.bnd_ct_plot_start_fm
        plot_mode = opts.bnd_ct_plot_mode

        if mode == 'troubleshooting':
            for idx in range(len(bcr)):
                self._plot_single_event(idx, ellipse_ref_pt, bcr, turning_idxs_filtered, opts)
        elif mode == 'display':
            self._plot_event_chain(
                ellipse_ref_pt, bcr,
                turning_idxs_filtered,
                start_frame=start_frame,
                mode=plot_mode,
                image_format=opts.imageFormat
            )

    def _plot_single_event(self, idx, ellipse_ref_pt, bcr, turning_idxs_filtered, opts):
        cdef int i, start_frame, end_frame
        start_frame = bcr[idx].start
        end_frame = bcr[idx].stop

        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / self.va.fps
        frames_range = range(
            max(0, start_frame - 5), min(len(self.x), end_frame + 5)
        )

        plt.figure(figsize=(8, 8))

        # Filter out NaN or zero values for the green trajectory line
        filtered_x = [
            self.x[i] for i in frames_range if not (
                np.isnan(self.x[i]) or self.x[i] == 0 or 
                np.isnan(self.y[i]) or self.y[i] == 0
            )
        ]
        filtered_y = [
            self.y[i] for i in frames_range if not (
                np.isnan(self.x[i]) or self.x[i] == 0 or 
                np.isnan(self.y[i]) or self.y[i] == 0
            )
        ]

        # Plot the green trajectory line with valid points only
        plt.plot(filtered_x, filtered_y, 'g-', label='Trajectory')
        plt.xlim(self.x_bounds_orig[0], self.x_bounds_orig[1])

        # Iterate through the frames and classify each point based on speed threshold
        for i in range(start_frame + 1, end_frame + 1):
            if (
                np.isnan(self.x[i]) or self.x[i] == 0 or 
                np.isnan(self.y[i]) or self.y[i] == 0 or 
                np.isnan(self.x[i - 1]) or self.x[i - 1] == 0 or 
                np.isnan(self.y[i - 1]) or self.y[i - 1] == 0
            ):
                continue
            if self.trj.sp[i] >= self.min_turn_speed:
                plt.plot(
                    [self.x[i - 1], self.x[i]], 
                    [self.y[i - 1], self.y[i]], 
                    'r-'
                )
            else:
                plt.plot(
                    [self.x[i - 1], self.x[i]], 
                    [self.y[i - 1], self.y[i]], 
                    'b-'
                )

        # Mark the start of the contact event
        plt.scatter(
            [self.x[start_frame]], [self.y[start_frame]], 
            color='r', s=100, label='Contact Start'
        )

        # Mark the end of the turn (or contact event if not a turn)
        plt.scatter(
            [self.x[end_frame]], 
            [self.y[end_frame]], 
            color='b', s=100, 
            label='Turn End' if idx in turning_idxs_filtered else 'Contact End'
        )

        # Add horizontal lines to depict the Y range of the boundary
        plt.axhline(
            y=self.y_bounds[0], color='k', linestyle='--', 
            label='Boundary Min Y'
        )
        plt.axhline(
            y=self.y_bounds[1], color='k', linestyle='--', 
            label='Boundary Max Y'
        )

        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        # Title the plot based on whether the event is classified as a turn or just a contact event
        is_turn = idx in turning_idxs_filtered
        if is_turn:
            plt.title(
                f'Turn Visualization from Frame {start_frame} to '
                f'{end_frame}'
            )
        else:
            plt.title(
                f'Contact Event Visualization from Frame {start_frame} to '
                f'{end_frame}'
            )

        # Position the legend inside the plot area
        plt.legend(loc='best')

        total_vel_angle_deltas = self.return_data["boundary_event_stats"][
            self.boundary_type
        ][self.boundary_combo][ellipse_ref_pt]['total_vel_angle_deltas']
        current_vel_angle_delta = total_vel_angle_deltas[idx]

        # Annotations inside the plot area, adjusting based on y-coordinate range
        annotation_text = (
            f'Duration: {duration_seconds:.2f} s\n' +
            f'Angle Delta Sum: '
            f'{180 * current_vel_angle_delta / np.pi:.2f} deg'
        )
        if not is_turn:
            if duration_seconds > opts.turn_duration_thresh:
                annotation_text += '\nToo long for turn'
            else:
                annotation_text += '\nToo little angular change'

        # Determine the y-coordinate position for the annotation text
        y_min = min(self.y[min(frames_range):max(frames_range) + 1])
        y_max = max(self.y[min(frames_range):max(frames_range) + 1])

        # Place annotation high if points are near the bottom, and low if points are near the top
        if (y_min - self.y_bounds[0]) < (self.y_bounds[1] - y_max):
            text_y = 0.9  # Place annotation above the points
            va = 'top'
        else:  # Closer to top
            text_y = 0.1  # Place annotation below the points
            va = 'bottom'

        plt.text(
            0.3, text_y, annotation_text, fontsize=12, 
            transform=plt.gca().transAxes, ha='left', 
            va=va, color='black', 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )

        output_path = (
            f'{"turn_evts" if is_turn else "bnd_ct_evts"}'
            f'/{ellipse_ref_pt}_ref_pt'
            f'/evt_{idx}.png'
        )

        # Ensure the directory structure exists before saving the figure
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure to the specified path and close the plot
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


    def _plot_event_chain(
        self, ellipse_ref_pt, bcr, turning_idxs_filtered, start_frame=None, mode='all_types', image_format='png'
    ):
        """
        Plots a chain of sharp turn events in a single figure, applying 
        a color-coded scheme based on the trajectory's characteristics and 
        adds arrows along the trajectory to indicate the direction of time progression.

        Parameters:
        - ellipse_ref_pt: reference point on the ellipse
        - bcr: list of boundary contact regions
        - turning_idxs_filtered: list of indices of sharp turns
        - start_frame: optional starting frame to begin the search for events
        - mode: determines which frames to include in the plot:
                'all_types' - show two sharp turns along with all parts of the trajectory between them.
                'turn_plus_1' - show a single sharp turn and one non-turn event following it, with distinct colors.
        """
        cdef int i
        speed_threshold_high = 18
        speed_threshold_low = 6
        # Number of sharp turns to chain together
        if mode == 'all_types':
            chain_length = 2
        elif mode == 'turn_plus_1':
            chain_length = 1

        if start_frame is not None:
            # Find the first sharp turn after the start_frame
            start_idx = next((idx for idx, turn_idx in enumerate(turning_idxs_filtered) 
                              if bcr[turn_idx].start >= start_frame), None)
            
            if start_idx is None or start_idx + chain_length > len(turning_idxs_filtered):
                raise ValueError("No sufficient sharp turns found after the specified start_frame")
            
            selected_turns = turning_idxs_filtered[start_idx:start_idx + chain_length]
        else:
            # Ensure there are enough sharp turns to select from
            if len(turning_idxs_filtered) < chain_length:
                chain_length = len(turning_idxs_filtered)

            # Randomly select a starting index
            start_idx = random.randint(0, len(turning_idxs_filtered) - chain_length)

            # Select the range of sharp turns
            selected_turns = turning_idxs_filtered[start_idx:start_idx + chain_length]

        # Define the start and end frames for the selected chain of turns
        start_frame = bcr[selected_turns[0]].start
        end_frame = bcr[selected_turns[-1]].stop

        # Retrieve rejection reasons
        rejection_reasons = self.return_data["boundary_event_stats"][self.boundary_type][
            self.boundary_combo
        ][ellipse_ref_pt].get('rejection_reasons', [])

        if mode == 'all_types':
            frames_range = range(max(0, start_frame - 5), min(len(self.x), end_frame + 5))
        elif mode == 'turn_plus_1':
            # Find the first event after the selected sharp turn that is not a sharp turn
            next_event_idx = None
            for i in range(selected_turns[-1] + 1, len(bcr)):
                if i not in turning_idxs_filtered:
                    next_event_idx = i
                    break
                else:
                    return

            # If a non-turn event is found, update end_frame to include it
            if next_event_idx is not None:
                end_frame = bcr[next_event_idx].stop

                # Include all frames between the sharp turn and the following non-turn event
                frames_range = range(max(0, start_frame - 5), min(len(self.x), end_frame + 5))
            else:
                # If no non-turn event is found, skip this segment
                return

        plt.figure(figsize=(12, 8))

        arrow_interval = 3  # Interval between arrows

        # Get the floor coordinates
        floor_coords = list(self.va.ct.floor(self.va.xf, f=self.va.nef * (self.trj.f) + self.va.ef))
        top_left, bottom_right = floor_coords[0], floor_coords[1]
        
        # Adjust for coordinate system where Y-axis points down
        lower_left_x = top_left[0]
        lower_left_y = top_left[1]

        # Calculate sidewall contact region
        contact_buffer_mm = CONTACT_BUFFER_OFFSETS["wall"]["max"]
        contact_buffer_px = self.va.ct.pxPerMmFloor() * self.va.xf.fctr * contact_buffer_mm

        # Draw the outer gray rectangle for sidewall contact region
        plt.gca().add_patch(
            patches.FancyBboxPatch(
                (lower_left_x, lower_left_y),  # Lower-left corner
                bottom_right[0] - top_left[0],  # Width
                bottom_right[1] - top_left[1],  # Height
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1, edgecolor='none', facecolor='gray', alpha=0.3, zorder=1
            )
        )

        # Draw the inner white rectangle to "cut out" the center, leaving a gray band
        plt.gca().add_patch(
            patches.FancyBboxPatch(
                (lower_left_x + contact_buffer_px, lower_left_y + contact_buffer_px),  # Lower-left corner
                (bottom_right[0] - top_left[0]) - 2 * contact_buffer_px,  # Width
                (bottom_right[1] - top_left[1]) - 2 * contact_buffer_px,  # Height
                boxstyle="round,pad=0.05,rounding_size=2",
                linewidth=1, edgecolor='none', facecolor='white', zorder=2
            )
        )

        # For the legend, we'll collect the labels and colors used
        handles = []
        labels = []

        last_arrow_idx = None

        # Initialize the count for successive slow frames
        successive_slow_frames = 0
        max_slow_frames = 15  # Adjust this value based on your preference
        current_bcr_index = None

        frames_to_skip = self.return_data["boundary_event_stats"][self.boundary_type][
            self.boundary_combo
        ][ellipse_ref_pt].get('frames_to_skip', set())
        frames_to_mark = []

        for i in frames_range:
            if np.isnan(self.x[i]) or self.x[i] == 0 or np.isnan(self.y[i]) or self.y[i] == 0:
                continue

            # Determine if the current frame is part of a sharp turn
            is_turn = any(bcr[j].start - 1 <= i < bcr[j].stop for j in selected_turns)

            if i in frames_to_skip and color != 'black':
                frames_to_mark.append((self.trj.x[i], self.trj.y[i]))

            # Initialize default color and label
            color = 'black'
            label = None
            rejection_reason = None

            # Mode-specific logic for color and label
            if mode == 'turn_plus_1':
                if is_turn:
                    color = 'red'
                    label = 'Sharp turn'
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
                        color = 'blue'
                        label = 'Boundary crossing w/out sharp turn'

            elif mode == 'all_types':
                if is_turn:
                    color = 'red'
                    label = 'Sharp turn'
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
                                "too_long": ('blue', 'Duration > 0.75 s'),
                                "too_little_velocity_angle_change": ('orange', 'Sum of vel. angle deltas < 90°'),
                                "sidewall_contact": ('purple', 'Sidewall contact')
                            }
                            color, label = color_label_map.get(rejection_reason, ('black', None))
                            break

            # Add to the legend only if it hasn't been added yet
            if label and label not in labels:
                labels.append(label)
                handles.append(plt.Line2D([0], [0], color=color, lw=2))

            x = self.trj.x
            y = self.trj.y

            # Clamp the X coordinates to the camera limits
            x_start = max(min(x[i], bottom_right[0]), top_left[0])
            x_end = max(min(x[i + 1], bottom_right[0]), top_left[0])

            turn_too_long = rejection_reason == 'too_long'

            if not turn_too_long or (
                turn_too_long and not (self.trj.nan[i] and self.trj.nan[i+1] and self.trj.nan[i + 2])
            ):
                plt.plot(
                    [x_start, x_end],
                    [y[i], y[i + 1]],
                    color=color,
                    zorder=3
                )

            if turn_too_long and current_bcr_index is not None and not (
                    # i == bcr[current_bcr_index].start - 1 or
                    i == bcr[current_bcr_index].stop - 1
            ):
                # Set the lighter color for the short segments
                lighter_color = 'lightblue'  # You can use 'lightblue' or an RGBA tuple for a lighter shade

                # Calculate the direction of the segment
                dx = self.x[i + 1] - self.x[i]
                dy = self.y[i + 1] - self.y[i]

                # Normalize direction vector to create a unit vector
                norm = np.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm

                # Define a constant length for the short segments
                fixed_segment_length = 0.125  # Fixed length for short segments (adjust as needed)
                
                # Calculate the start and end points of the short segment at the end of the main segment
                x_start_short = x_end - dx * fixed_segment_length
                y_start_short = y[i + 1] - dy * fixed_segment_length

                # Draw the short segment at the end of the main segment
                plt.plot(
                    [x_end, x_start_short],
                    [y[i + 1], y_start_short],
                    color=lighter_color,
                    linewidth=plt.gca().lines[-1].get_linewidth(),  # Match the width of the main line
                    zorder=4
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

            # Determine if it's time to draw an arrow
            if last_arrow_idx is None or i >= last_arrow_idx + arrow_interval:
                # Calculate the midpoint of the segment
                x_mid = (x_start + x_end) / 2
                y_mid = (y[i] + y[i + 1]) / 2

                # Calculate the direction of the segment
                dx = x_end - x_start
                dy = y[i + 1] - y[i]

                # Draw custom arrowhead at the midpoint
                draw_custom_arrowhead(plt.gca(), x_mid, y_mid, dx, dy, color)

                # Update the position of the last index where an arrow was drawn
                last_arrow_idx = i


        plt.axhline(y=self.y_bounds[0], color='k', linestyle='--', zorder=3)
        plt.axhline(y=self.y_bounds[1], color='k', linestyle='--', zorder=3)

        # Draw the rounded floor box
        rect = patches.FancyBboxPatch(
            (lower_left_x, lower_left_y),  # Lower-left corner
            bottom_right[0] - top_left[0],  # Width
            bottom_right[1] - top_left[1],  # Height
            boxstyle="round,pad=0.05,rounding_size=2",
            linewidth=1, edgecolor='black', facecolor='none', zorder=4
        )
        plt.gca().add_patch(rect)
        plt.gca().axis('off')

        # Plot a horizontal line at the vertical midpoint
        vertical_midpoint = (top_left[1] + bottom_right[1]) / 2
        plt.axhline(y=vertical_midpoint, color='black', linestyle=':', linewidth=2, zorder=4)

        # Set equal scaling for both axes
        plt.gca().set_aspect('equal', adjustable='box')

        # Set plot limits with padding
        padding_x = (bottom_right[0] - top_left[0]) * 0.1
        padding_y = (top_left[1] - bottom_right[1]) * 0.1

        plt.xlim(top_left[0] - padding_x, bottom_right[0] + padding_x)
        plt.ylim(bottom_right[1] - padding_y, top_left[1] + padding_y)

        for (x, y) in frames_to_mark:
                plt.plot(
                    x,
                    y,
                    marker='o',
                    color='green',
                    markersize=6,
                    zorder=5,
                    label='Sidewall contact start'
                )
        if len(frames_to_mark) > 0:
            labels.append('Sidewall contact start')
            handles.append(plt.Line2D([0], [0], marker='o', color='green', lw=0, markersize=6))

        # Add the legend outside the plot area
        plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                    fancybox=True, shadow=True, ncol=2)

        plt.xlabel('')
        plt.ylabel('')

        plt.title(f'Boundary contact events and sharp turns, {start_frame} to {end_frame}')

        output_path = f'display/{ellipse_ref_pt}_ref_pt/chained_turn_{start_idx}_f{self.trj.f}.png'
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        writeImage(output_path, format=image_format)
        plt.close()

    def detectWallContact(self, ellipse_edge_pt="closest"):
        # Map the `ellipse_edge_pt` to the corresponding nested key in the return_data
        edge_type_key = "opp_edge" if ellipse_edge_pt == "opposite" else "closest"
        
        # Attempt to access the nested key without creating it if it doesn't exist
        boundary_contact = self.return_data.get("boundary_event_stats", {}).get(
            "wall", {}).get("all", {}).get(edge_type_key, {}).get("boundary_contact", None)
        
        if boundary_contact is None:
            # Run your logic since the key doesn't exist
            wall_thresholds = [
                CONTACT_BUFFER_OFFSETS["wall"]["min"],
                CONTACT_BUFFER_OFFSETS["wall"]["max"]
            ]
            self.calc_dist_boundary_to_ellipse(
                boundary_type="wall",
                boundary_combo="all",
                offset=0,
                event_thresholds={
                    "horiz": wall_thresholds,
                    "vert": wall_thresholds
                },
                ellipse_ref_pt="edge",
                ellipse_edge_pt=ellipse_edge_pt
            )

    def get_ellipse_ctr_boundary_crossings(self, offset):
        # Calculates and records the number of times an ellipse crosses boundary areas
        # within a specified well configuration during tracking. This method updates
        # an internal structure to reflect boundary crossings at each frame based on
        # specified boundary types and positional offsets.
        #
        # The method operates by:
        # 1. Setting up the initial zeroed counters for boundary crossings.
        # 2. Adjusting the offset for well lengths based on a pixels-per-millimeter (ppmm) value.
        # 3. Configuring well dimensions using the adjusted offset and querying for well boundary
        #    positions.
        # 4. Iteratively checking each frame to determine if the tracked position of the ellipse
        #    is within any well boundaries and updating the counter accordingly.
        #
        # Parameters:
        # - boundary_type : str
        #     The type of boundary (e.g., 'tb' for top-bottom, 'lr' for left-right) that should be
        #     considered for counting crossings.
        # - offset : float
        #     The offset from the well center, scaled by the `ppmm` value, used to determine the
        #     size of the wells for boundary checks.
        #
        # Modifies:
        # - self.return_data["boundary_event_stats"][boundary_type]["tb"]["boundary_contact_ctr"]
        #     This structure is updated to record the number of crossings for the specified boundary
        #     type at each frame.
        #
        # Raises:
        # - This method does not explicitly raise exceptions but assumes that all necessary class
        #   variables and data structures are correctly initialized and populated.

        boundary_contact = np.zeros(len(self.x))
        well_kwargs = {"wellLength": offset * self.va.ct.htl.value["ppmm"]} if offset else {}
        wells = self.va.ct.arenaWells(
            self.va.xf, self.va.f + (0 if self.va.noyc else self.trj.f * self.va.nef),
            **well_kwargs
        )
        self.ellipse_ref_pt = 'ctr'
        def checkFrame(i):
            x, y = self.trj.x[i], self.trj.y[i]
            if np.isnan(self.trj.x[i]):
                boundary_contact[i] = np.nan
                return
            for well in wells:
                inside_x_now = x >= well[0] and x <= well[2]
                inside_y_now = y >= well[1] and y <= well[3]
                condition_met = inside_x_now and inside_y_now
                if condition_met:
                    boundary_contact[i] = 1
                    return
            boundary_contact[i] = 0
        for i in range(len(self.x)):
            checkFrame(i)
        self.update_return_data_for_boundary_contact_stats(boundary_contact, False)

    def find_subset_at_or_below_duration(
        self, duration_in_frames, min_vel_angle_delta, ellipse_ref_pt
    ):
        wall_contact_regions = self.return_data[
            "boundary_event_stats"
        ]["wall"]["all"]["edge"]["boundary_contact_regions"]

        bcr = self.return_data["boundary_event_stats"][self.boundary_type][
            self.boundary_combo
        ][ellipse_ref_pt]["boundary_contact_regions"]
        is_turning = np.zeros(len(self.x))
        turning_idxs_filtered = []

        rejection_reasons = []  # Structure to hold rejection reasons

        # Initialize frames_to_skip key in return_data before the loop
        if 'frames_to_skip' not in self.return_data[
            "boundary_event_stats"
        ][self.boundary_type][self.boundary_combo][ellipse_ref_pt]:
            self.return_data[
                "boundary_event_stats"
            ][self.boundary_type][self.boundary_combo][ellipse_ref_pt]['frames_to_skip'] = set()

        # Reference to frames_to_skip for easy access and modification
        frames_to_skip = self.return_data[
            "boundary_event_stats"
        ][self.boundary_type][self.boundary_combo][ellipse_ref_pt]['frames_to_skip']

        for idx in range(len(bcr)):
            start_frame = bcr[idx].start
            end_frame = bcr[idx].stop

            if end_frame - start_frame > duration_in_frames:
                rejection_reasons.append("too_long")
                continue

            # Determine frames to skip based on wall contact and speed threshold
            for region in wall_contact_regions:
                if start_frame <= region.start < end_frame:
                    frames_to_skip.add(region.start)

            vel_angle_deltas = []
            for i in range(start_frame - 1, end_frame):
                if i in frames_to_skip or self.trj.sp[i] < self.min_turn_speed:
                    continue

                upper_index = i + 1
                while upper_index < end_frame and (
                    upper_index in frames_to_skip or
                    self.trj.sp[upper_index] < self.min_turn_speed
                ):
                    upper_index += 1

                if upper_index >= end_frame:
                    break

                vel_angle_delta = abs(
                    (self.trj.velAngles[upper_index] - self.trj.velAngles[i])
                )
                alt_vel_angle = 2 * np.pi - vel_angle_delta
                vel_angle_deltas.append(min(vel_angle_delta, alt_vel_angle))

            total_vel_angle_delta = np.abs(np.sum(vel_angle_deltas))

            if 'total_vel_angle_deltas' not in self.return_data["boundary_event_stats"][
                self.boundary_type][self.boundary_combo][ellipse_ref_pt]:
                self.return_data["boundary_event_stats"][
                    self.boundary_type][self.boundary_combo][ellipse_ref_pt]['total_vel_angle_deltas'] = []
            self.return_data["boundary_event_stats"][
                self.boundary_type][self.boundary_combo][ellipse_ref_pt][
                    'total_vel_angle_deltas'].append(total_vel_angle_delta)

            if total_vel_angle_delta >= np.pi * min_vel_angle_delta / 180:
                is_turning[start_frame:end_frame] = 1
                turning_idxs_filtered.append(idx)
                rejection_reasons.append("turn")
            else:
                rejection_reasons.append("too_little_velocity_angle_change")

        # Save the rejection reasons to the return data structure
        self.return_data["boundary_event_stats"][self.boundary_type][
            self.boundary_combo
        ][ellipse_ref_pt]['rejection_reasons'] = rejection_reasons

        is_near_turning = np.zeros(len(self.x))
        for idx in turning_idxs_filtered:
            start_frame = bcr[idx].start
            end_frame = bcr[idx].stop
            if np.any(
                (self.trj.y[start_frame:end_frame] < self.y_bounds[0])
                | (self.trj.y[start_frame:end_frame] > self.y_bounds[1])
            ):
                is_turning[start_frame:end_frame] = 2

        for key_name, var in (
            ("turning_indices", turning_idxs_filtered),
            ("turning", is_turning),
            ("near_turning", is_near_turning),
        ):
            if ellipse_ref_pt in self.return_data["boundary_event_stats"][self.boundary_type][
                self.boundary_combo
            ]:
                self.return_data["boundary_event_stats"][self.boundary_type][
                    self.boundary_combo
                ][ellipse_ref_pt][key_name] = var
            else:
                self.return_data["boundary_event_stats"][self.boundary_type][
                    self.boundary_combo
                ][ellipse_ref_pt] = {key_name: var}

    cdef initialize_attributes(self):
        # Initializes various attributes of the EllipseToBoundaryDistCalculator class with
        # default values if they are not already set. This method ensures that the class has all
        # necessary data structures ready for performing its calculations, particularly for
        # handling out-of-bounds (OOB) test points, ellipse points, distances, wall points, and
        # angles to the best ellipse point, as well as boundary configurations for different
        # experimental setups.

        # The method sets default values for the following attributes if they are None:
        # - `oob_test_pts`: A dictionary to store out-of-bounds test points. Default is an empty
        #   dictionary.
        # - `ell_pts`: A dictionary to store points on the ellipse. Default is an empty
        #   dictionary.
        # - `distances`: A dictionary to store distances from the ellipse to the boundary.
        #   Default is an empty dictionary.
        # - `wall_pts`: A dictionary to store points on the wall near the ellipse. Default is an
        #   empty dictionary.
        # - `angle_to_best_ell_pt`: A dictionary to store angles to the best point on the
        #   ellipse from the boundary. Default is an empty dictionary.
        # - `bounds_agarose`: The boundary configuration for agarose setups. Default is None.
        # - `bounds_full`: The boundary configuration for full setups. Default is None.
        # - `y_bounds_agarose`: The Y-axis boundary configuration for agarose setups. Default is
        #   None.
        # - `y_bounds_full`: The Y-axis boundary configuration for full setups. Default is None.

        # This method is typically called during the initialization phase of each boundary
        # distance calculation to ensure that all attributes are properly set up before any
        # calculations or analyses are performed.
        default_values = {
            "oob_test_pts": {},
            "ell_pts": {},
            "distances": {},
            "wall_pts": {},
            "angle_to_best_ell_pt": {},
            "bounds_agarose": None,
            "bounds_full": None,
            "y_bounds_agarose": None,
            "y_bounds_full": None,
        }
        if self.oob_test_pts is None:
            self.oob_test_pts = default_values["oob_test_pts"]
        if self.ell_pts is None:
            self.ell_pts = default_values["ell_pts"]
        if self.distances is None:
            self.distances = default_values['distances']
        if self.wall_pts is None:
            self.wall_pts = default_values['wall_pts']
        if self.angle_to_best_ell_pt is None:
            self.angle_to_best_ell_pt = default_values['angle_to_best_ell_pt']

    cpdef calc_dist_boundary_to_ellipse(
        self,
        boundary_type,
        boundary_combo,
        offset,
        event_thresholds,
        ellipse_edge_pt='closest',
        ellipse_ref_pt='edge'
    ):
        # Calculates the minimum distance of ellipses (representing entities) to specified
        # boundaries within a defined chamber area. This method adjusts the chamber bounds and
        # computes distances based on the current orientation and position of the ellipses,
        # taking into account the boundary type and combination specified. It initializes and
        # updates several attributes related to the distances of ellipses to these boundaries,
        # handling different configurations and types of boundaries.

        # Parameters:
        # - boundary_type : str
        #     The type of boundary to consider for distance calculations. This could represent
        #     different sections or types of walls within the chamber, such as 'agarose' or
        #     'full' chamber boundaries.
        # - boundary_combo : str
        #     Specifies the combination of boundaries to consider for the calculations. Examples
        #     include 'all' for all boundaries, 'lr' for left-right boundaries, 'tb' for
        #     top-bottom boundaries, and 'agarose_adj' for agarose-adjusted boundaries. This
        #     parameter influences which chamber walls are considered in the distance
        #     calculations.
        # - offset : float
        #     Distance from the top/bottom walls of the chamber at which to define the boundaries.
        #     Note: units are mm, and these offsets are scaled relative to the
        #     (HTL) chamber height of 16mm.
        # - event_thresholds : dict
        #     A dictionary containing threshold values used to determine boundary contact
        #     events. These thresholds are used to fine-tune the identification of events based
        #     on the distances of the ellipses to the boundaries. The dictionary keys correspond
        #     to different types of events or boundary orientations (e.g., 'vertical' or
        #     'horizontal'), and the values are the specific thresholds used.
        # - ellipse_edge_pt : str
        #     Specifies the ellipse point with respect to which distances to the boundary will be
        #     calculated. Choices are 'closest', to use the point on the ellipse closest to the
        #     boundary, or 'opposite', to use the point on the ellipse opposite the closest point.

        # Modifies:
        # - Initializes and updates internal attributes related to boundary distances, such as
        #   minimum distances to each boundary (`self.min_distances`), flags indicating
        #   out-of-bounds conditions (`self.oob_on_ignored_wall`), and the configuration of the
        #   chamber bounds based on the specified `boundary_combo`. This includes rotating the
        #   chamber bounds to align with the orientation of the ellipses and calculating
        #   distances accordingly.
        # - Updates `self.return_data` with information related to the boundary distance
        #   calculations, including the types of boundaries considered, the combination of
        #   boundaries, and the specific distances calculated for each ellipse to the nearest
        #   boundary.

        # Note:
        # - This method relies on several preparatory steps, including setting up the trajectory
        #   data, reorienting ellipse orientations, and initializing attributes related to the
        #   chamber configuration and boundary types. It is designed to be called after these
        #   initializations have been performed.
        # - The method supports handling different boundary combinations and types by adjusting
        #   the chamber bounds and considering specific walls within the chamber for distance
        #   calculations. This flexibility allows for detailed analysis of entity positions
        #   relative to boundaries in various experimental setups.
        self._setup_boundary_dist_data()
        if self.trj.theta is None:
            return
        self.boundary_type = boundary_type
        self.boundary_combo = boundary_combo
        self.ellipse_edge_pt = ellipse_edge_pt
        self.ellipse_ref_pt = ellipse_ref_pt

        self.initialize_attributes()

        self.min_distances = np.zeros((len(self.x)))
        self._rotate_chamber_bounds_about_ellipses(offset)
        self.oob_on_ignored_wall = np.full(len(self.x), False)

        # Determine which walls to check based on the boundary combination
        if self.boundary_combo in ('all', 'lr'):
            walls_to_check = ('left', 'bottom', 'right', 'top')
        elif self.boundary_combo == 'agarose_adj':
            walls_to_check = ('left', 'bottom_agarose', 'right', 'top_agarose')
        elif self.boundary_combo == 'tb':
            walls_to_check = ('bottom_agarose', 'top_agarose')

        # Calculate distances for each wall
        for wall in walls_to_check:
            self._calc_distances_to_single_wall(wall)

        # Rotate best points back to original orientation
        for k in self.best_pts:
            self.best_pts[k] = rotate_pts(
                self.best_pts[k], self.cos_of_angles, self.neg_sin_of_angles, self.origins
            )

        # Set event thresholds and find boundary contact events
        for near_contact in (False, True):
            self._set_event_thresholds(
                boundary_combo, event_thresholds, near_contact=near_contact
            )
            self._find_boundary_contact_events(near_contact)
