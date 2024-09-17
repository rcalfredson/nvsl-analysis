#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.math cimport (fabs, nan)

from common_cython cimport (
    angleDiff,
    compute_distance_or_nan,
    euclidean_norm,
    in_range,
    ndarray_long_to_vector,
    ndarray_float_to_vector,
    PI,
    sign,
)

cdef struct TurnData:
    # A structure to store start and end points of turns within a time division.
    #
    # Attributes:
    # - start (vector[vector[double]]): A nested vector structure where the outer level
    #   represents time divisions and the inner level contains the start points of individual
    #   turns.
    # - end (vector[vector[double]]): Similar to `start`, but contains the end points of
    #   individual turns.
    vector[vector[double]] start
    vector[vector[double]] end

cdef struct HistData:
    # A structure for storing histogram data related to start and end points of turns.
    #
    # Attributes:
    # - start (vector[pair[vector[double], vector[double]]]): A vector where each element is a
    #   pair; the first element of the pair is a vector of double representing bin counts, and
    #   the second element is a vector of double representing bin edges, for the start points of
    #   turns.
    # - end (vector[pair[vector[double], vector[double]]]): Similar to `start`, but for the end
    #   points of turns.
    vector[pair[vector[double], vector[double]]] start
    vector[pair[vector[double], vector[double]]] end

cdef class RewardCircleAnchoredTurnFinder:
    cdef object va
    cdef double[:] x_view
    cdef double[:] y_view
    cdef double[:] d_view
    cdef bint[:] walking_view
    cdef vector[bint] wall_contact
    cdef double min_turn_speed
    cdef double min_turn_speed_px
    cdef int num_hist_bins
    cdef bint end_turn_before_recontact
    cdef vector[TurnData] lg_turn_dists
    cdef vector[HistData] large_turn_hist_counts_edges
    cdef vector[vector[double]] turn_to_exit_ratios
    cdef vector[vector[pair[int, int]]] indices_of_turns
    cdef vector[pair[int, int]] trn_ranges
    cdef long[:] circle_ctr
    cdef double circle_rad
    cdef double[:] dist_from_ctr_at_end
    cdef vector[vector[vector[double]]] large_turn_stats

    def __cinit__(
        self,
        object va,
        double min_turn_speed,
        int num_hist_bins,
        int end_turn_before_recontact
    ):
        # Initializes the RewardCircleAnchoredTurnFinder with required parameters.
        #
        # Parameters:
        # - va (VideoAnalysis): A VideoAnalysis instance representing data from an experiment.
        # - min_turn_speed (double): The minimum speed threshold to consider a movement as a
        #   turn.
        # - num_hist_bins (int): The number of histogram bins to use for analyzing turn data.
        # - end_turn_before_recontact (int): A flag indicating whether to end turn tracking
        #   before recontact with a boundary.
        self.va = va
        self.min_turn_speed = min_turn_speed
        self.num_hist_bins = num_hist_bins
        self.lg_turn_dists = vector[TurnData]()
        self.large_turn_hist_counts_edges = vector[HistData]()
        self.turn_to_exit_ratios = vector[vector[double]]()
        self.indices_of_turns = vector[vector[pair[int, int]]]()
        self.end_turn_before_recontact = end_turn_before_recontact

    def calcLargeTurnsAfterCircleExit(self):
        # Calculates large turns occurring after exiting a circle by analyzing the trajectory
        # data.
        #
        # This method identifies ranges of trajectory indices that represent turns after exiting
        # a predefined circle. It updates internal structures with the details of these turns,
        # such as their start and stop indices, and computes statistics related to these large
        # turns for further analysis.
        self.trn_ranges = vector[pair[int, int]]()
        self.trn_ranges.push_back(pair[int, int](
            self.va.startPre, self.va.trns[0].start
        ))
        for t in self.va.trns:
            self.trn_ranges.push_back(pair[int, int](t.start, t.stop))
        self.getTurnsForTrx()
        self.va.lg_turn_dists = self.lg_turn_dists
        self.va.large_turn_stats = np.array(self.large_turn_stats)

    cdef getTurnsForTrx(self):
        # Iterates through all trajectories to identify and process large turning movements.
        #
        # This method loops through each trajectory in the VideoAnalysis object (`self.va.trx`),
        # identifying large turning movements and then computes large turn statistics for each
        # trajectory. It serves as an entry point for processing individual trajectories and
        # aggregating their turn-related data.
        for trj in self.va.trx:
            self.getTurnsForTrj(trj)
        self.compute_large_turn_stats()

    cdef getTurnsForTrj(self, trj):
        # Processes a single trajectory to identify large turns and compute related statistics.
        #
        # For a given trajectory that is not marked as bad, this method prepares trajectory
        # data, such as positions, directions, walking status, and minimum turn speed in pixels
        # per millimeter, for analysis. It initializes containers for storing turn data and
        # histogram data for large turns. Then, it iterates over a series of indices
        # representing training sessions, the primary division of time in the data being
        # analyzed, processing each to collect turn data and statistics.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object to be processed, containing positional, directional, and
        #     movement data.
        if not trj.bad():
            self.wall_contact = trj.boundary_event_stats[
                'wall'
            ]['all']['edge']['boundary_contact']
            self.x_view = trj.x
            self.y_view = trj.y
            self.d_view = trj.d
            self.walking_view = trj.walking.astype(np.int32)
            self.min_turn_speed_px = self.min_turn_speed * trj.pxPerMmFloor * self.va.xf.fctr

        self.turn_to_exit_ratios.push_back(vector[double]())
        self.indices_of_turns.push_back(vector[pair[int, int]]())
        self.lg_turn_dists.push_back(
            TurnData(start=vector[vector[double]](), end=vector[vector[double]]())
        )
        self.large_turn_hist_counts_edges.push_back(
            HistData(
                start=vector[pair[vector[double], vector[double]]](),
                end=vector[pair[vector[double], vector[double]]]()
            )
        )
        for i, trn_range in enumerate(self.trn_ranges):
            self.getTurnsForRange(trj, i, trn_range)

    @staticmethod
    cdef compute_statistic(vector[double] &vals, vector[double] &edges):
        # Computes the sum and median of a set of values based on histogram data.
        #
        # This static method calculates the total sum of all values in a given vector and
        # determines the median value based on histogram edges. It is used for processing
        # statistical data related to large turns.
        #
        # Parameters:
        # - vals : vector[double]
        #     A vector of doubles representing the values for which statistics are computed.
        # - edges : vector[double]
        #     A vector of doubles representing the histogram edges associated with the values.
        #
        # Returns:
        # - tuple
        #     A tuple containing the total sum of values and the median value as determined by
        #     the histogram edges. If the input vector of values is empty, the median is
        #     returned as NaN.
        cdef double sum_val = 0.0
        cdef int peak_index
        cdef double median
        if vals.size() > 0:
            for k in range(vals.size()):
                sum_val += vals[k]
            peak_index = np.argmax(vals)
            median = 0.5 * (edges[peak_index] + edges[peak_index + 1])
        else:
            median = nan("")
        return sum_val, median

    cdef compute_large_turn_stats(self):
        # Computes summary statistics for large turns across all training sessions and
        # trajectories.
        #
        # For each training session and trajectory, this method calculates the sum and median
        # distance of large turns, as well as the ratio of turns to exits. These statistics are
        # stored in a structured format that facilitates easy access and analysis.
        #
        # This computation relies on previously gathered histogram data on large turn distances
        # and the relationship between turns and exits, iterating through each training session
        # and trajectory to populate the large turn statistics with these derived metrics.
        cdef int i, j
        cdef int num_trns = len(self.va.trns)
        cdef double sum_, med_dist_

        self.large_turn_stats.resize(len(self.va.trns))
        for i in range(num_trns):
            self.large_turn_stats[i].resize(len(self.va.trx))
            for j in range(len(self.va.trx)):
                self.large_turn_stats[i][j].resize(3, nan(""))

            for j in range(2):
                sum_, med_dist_ = RewardCircleAnchoredTurnFinder.compute_statistic(
                    self.large_turn_hist_counts_edges[j].end[i].first,
                    self.large_turn_hist_counts_edges[j].end[i].second
                )

                self.large_turn_stats[i][j][0] = sum_
                self.large_turn_stats[i][j][1] = med_dist_
                self.large_turn_stats[i][j][2] = self.turn_to_exit_ratios[j][i + 1]


    cdef getTurnsForRange(self, trj, i, trn_range):
        # Identifies and processes large turns within a specified range of a trajectory.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - i : int
        #     The index of the current training session within the analysis.
        # - trn_range : tuple
        #     The start and end points defining the range of the trajectory to be analyzed for
        #     large turns.
        #
        # This method first checks if the trajectory is marked as bad, skipping further
        # processing if so. For valid trajectories, it initializes and updates data structures
        # to track large turns and their characteristics (e.g., distances and exit ratios)
        # within the specified range. It includes identifying exit and entry points for turns
        # and calculating histograms for turn distances.
        if trj.bad():
            self.initialize_invalid_trajectory_data(trj.f)
            return

        self.set_circle_data(trj, i)
        exits = in_range(trj.ex[0], trn_range[0], trn_range[1])
        entries = in_range(trj.en[0], trn_range[0], trn_range[1])
        self.initialize_lg_turn_dists(trj.f, len(exits))

        for j, ex_fm in enumerate(exits):
            self.run_turn_search(trj, i, ex_fm, j, entries)
        self.turn_to_exit_ratios[trj.f].push_back(self.calc_turn_to_exit_ratio(trj, i))

        self.compute_histogram(trj, i)

    cdef compute_histogram(self, trj, trn_idx):
        # Computes histograms for the start and end points of large turns within a trajectory
        # for a specific training session.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - trn_idx : int
        #     The index of the training session within the analysis.
        #
        # This method internally calls a helper function to compute histograms for both the
        # start and end points of large turns identified in the specified trajectory and
        # training session. It organizes the histogram data for further analysis and
        # visualization.
        self._inner_compute_histogram(trj, trn_idx, "start")
        self._inner_compute_histogram(trj, trn_idx, "end")

    cdef _inner_compute_histogram(self, trj, trn_idx, tp):
        # Computes and stores histograms for the start or end points of large turns within a
        # specified trajectory and training session.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - trn_idx : int
        #     The index of the training session within the analysis.
        # - tp : str
        #     Type of data point to compute histogram for, either "start" or "end" of the large
        #     turn.
        #
        # This method calculates histograms based on the distances at the start or end of large
        # turns, utilizing numpy for histogram computation. The results are then converted and
        # stored in vectors for further statistical analysis. This process supports the analysis
        # of large turn dynamics by providing quantitative measurements of their spatial
        # characteristics.
        cdef np.ndarray bcs
        cdef np.ndarray bins
        if tp == "start":
            data = self.lg_turn_dists[trj.f].start[trn_idx]
        elif tp == "end":
            data = self.lg_turn_dists[trj.f].end[trn_idx]
        else:
            raise ValueError("Invalid type")
        np_data = np.asarray(data)
        bcs, bins = np.histogram(np_data, bins=self.num_hist_bins, range=(0, 8))
        cdef vector[double] cts_vec = ndarray_long_to_vector(bcs)
        cdef vector[double] bins_vec = ndarray_float_to_vector(bins)
        
        hist_entry = pair[vector[double], vector[double]](
            cts_vec, bins_vec
        )
        
        if tp == "start":
            self.large_turn_hist_counts_edges[trj.f].start.push_back(hist_entry)
        elif tp == "end":
            self.large_turn_hist_counts_edges[trj.f].end.push_back(hist_entry)

    cdef void initialize_invalid_trajectory_data(self, int f):
        # Initializes data structures for an invalid trajectory with NaNs and empty vectors.
        #
        # Parameters:
        # - f : int
        #     The index of the trajectory considered invalid.
        #
        # For trajectories marked as invalid, this method ensures that data structures related
        # to large turn analysis (such as turn-to-exit ratios and histograms) are properly
        # initialized with placeholder values (NaNs for ratios and empty vectors for
        # distributions), preventing errors in subsequent analysis steps.
        self.turn_to_exit_ratios[f].push_back(nan(""))
        self.lg_turn_dists[f].start.push_back(vector[double]())
        self.lg_turn_dists[f].end.push_back(vector[double]())
        self.large_turn_hist_counts_edges[f].start.push_back(
            pair[vector[double], vector[double]]()
        )
        self.large_turn_hist_counts_edges[f].end.push_back(
            pair[vector[double], vector[double]]()   
        )

    cdef void set_circle_data(self, trj, int i):
        # Sets the circle data (center and radius) for the current trajectory and training
        # session.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - i : int
        #     The index of the current training session.
        #
        # This method extracts and sets the circle data (center and radius) for the trajectory's
        # current training session. It adjusts the index for accessing the circle data and
        # retrieves this information from the trajectory's associated training session, storing
        # it for use in large turn analysis.
        if i > 0:
            i -= i
        circle = self.va.trns[i].circles(trj.f)[0]
        self.circle_ctr = np.array(circle[:2])
        self.circle_rad = circle[2]

    cdef void initialize_lg_turn_dists(self, int f, int length):
        # Initializes distance vectors for large turn start and end points with NaN values for a
        # specific trajectory.
        #
        # Parameters:
        # - f : int
        #     The index of the trajectory for which to initialize the distance vectors.
        # - length : int
        #     The length of the distance vectors to be initialized.
        #
        # This method ensures that each trajectory has a pre-allocated vector of NaN values for
        # both the start and end distances of large turns, facilitating subsequent calculations
        # and analyses.
        self.lg_turn_dists[f].start.push_back(vector[double](length, nan("")))
        self.lg_turn_dists[f].end.push_back(vector[double](length, nan("")))

    cdef double calc_turn_to_exit_ratio(self, trj, trn_idx):
        # Calculates the ratio of exits from the reward circle that were followed by a large
        # turn.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - trn_idx : int
        #     The index of the training session within the analysis.
        #
        # Returns:
        # - ratio : double
        #     The ratio of exits from the reward circle that led to a large turn, with NaN
        #     values in `lg_turn_dists` indicating exits that did not result in a large turn.
        #     The length of `lg_turn_dists` corresponds to the number of exits from the reward
        #     circle, and non-NaN values indicate exits that were followed by a large turn.
        #
        # This method enhances the understanding of trajectory dynamics by quantifying the
        # proportion of exits from a reward circle that result in significant turning behavior,
        # which is crucial for analyzing the spatial decision-making processes of the subject.
        cdef int count_non_nan = 0
        cdef int total
        cdef double ratio

        total = len(self.lg_turn_dists[trj.f].end[trn_idx])
        for val in self.lg_turn_dists[trj.f].end[trn_idx]:
            if val == val:
                count_non_nan += 1
        if total != 0:
            ratio = count_non_nan / total
        else:
            ratio = nan("")
        return ratio

    cdef run_turn_search(self, trj, trn_idx, circle_exit_frame, circle_exit_idx, entries, bint debug=False):
        """
        Executes a streamlined search for the start and end indices of a large turn within a trajectory,
        using a single forward pass based on circle exit points and potential turn entry points.
        
        Parameters:
        - trj : Trajectory
            The trajectory object being analyzed.
        - trn_idx : int
            The index of the training session within the trajectory.
        - circle_exit_frame : int
            The frame index at which the trajectory exits the analysis circle.
        - circle_exit_idx : int
            The index of the exit point within the vector of exit points.
        - entries : list
            A list of potential entry points for the turn within the trajectory.
        - debug : bool
            When True, debug statements will be printed during execution.
        """
        SET_TURN_END_AT_90_DEG = True

        cdef int turn_st_idx = -1
        cdef int turn_end_idx = -1
        cdef double cw_ang_del_sum = 0.0
        cdef double ccw_ang_del_sum = 0.0
        cdef double displacement
        cdef vector[double] angle_deltas = vector[double]()
        cdef double abs_angle_delta
        cdef int fm_i = circle_exit_frame - 1
        cdef int cw_initial_frame = -1
        cdef int ccw_initial_frame = -1
        cdef double angle_delta_degrees
        cdef int turn_direction = 0  # 1 for CW, -1 for CCW
        cdef bint threshold_crossed = False  # Track if 90-degree threshold has been crossed

        dist_threshold = 1 * trj.pxPerMmFloor * self.va.xf.fctr
        spf = 1 / self.va.fps

        # Single forward pass to find initial turn frames and determine turn start and end
        debug_frame_range_low = 25600
        debug_frame_range_high = 25700
        while True:
            fm_i += 1

            if fm_i + 2 >= len(trj.sp):
                return  # Exit if we exceed trajectory length

            # Use the new helper function to check the speed threshold
            if self.check_speed_threshold(
                fm_i + 1, trj, debug, debug_frame_range_low, debug_frame_range_high
            ):
                continue

            upper_index = fm_i + 1
            while (
                upper_index < len(trj.sp) and
                self.check_speed_threshold(
                    upper_index + 1, trj, debug, debug_frame_range_low, debug_frame_range_high
                )
            ):
                upper_index += 1

            if upper_index >= len(trj.sp):
                return

            if self.check_exit_condition(
                upper_index, threshold_crossed, circle_exit_idx, entries, max(cw_ang_del_sum, ccw_ang_del_sum),
                angle_deltas, debug, debug_frame_range_low, debug_frame_range_high
            ):
                return

            self.update_angle_and_sums(
                fm_i, upper_index, trj, angle_deltas, &cw_ang_del_sum, &ccw_ang_del_sum,
                &cw_initial_frame, &ccw_initial_frame, debug, debug_frame_range_low, debug_frame_range_high
            )

            if not threshold_crossed and fabs(cw_ang_del_sum) >= 90 and sign(cw_ang_del_sum) > 0:
                turn_st_idx = cw_initial_frame
                turn_direction = 1  # CW turn
                threshold_crossed = True
                if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                    print(f"Threshold crossed at frame {fm_i}, CW Turn Start Index = {turn_st_idx}")
                break

            if not threshold_crossed and fabs(ccw_ang_del_sum) >= 90 and sign(ccw_ang_del_sum) < 0:
                turn_st_idx = ccw_initial_frame
                turn_direction = -1  # CCW turn
                threshold_crossed = True
                if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                    print(f"Threshold crossed at frame {fm_i}, CCW Turn Start Index = {turn_st_idx}")
                break

        if turn_st_idx == -1:
            return

        if SET_TURN_END_AT_90_DEG:
            turn_end_idx = upper_index
        else:
            fm_i = upper_index
            while True:
                fm_i += 1

                if fm_i + 1 >= len(trj.sp):
                    return

                if trj.sp[fm_i + 1] < self.min_turn_speed_px:
                    if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                        print(f"Frame {fm_i} skipped: Speed = {trj.sp[fm_i + 1]:.2f} (below threshold)")
                    continue

                theta_1 = trj.theta[fm_i] * PI / 180
                theta_2 = trj.theta[fm_i + 1] * PI / 180
                angle_delta = angleDiff(
                    theta_2, theta_1, absVal=False, useRadians=True
                )
                angle_delta_degrees = angle_delta * (180.0 / PI)

                if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                    print(f"Frame {fm_i}: Speed = {trj.sp[fm_i + 1]:.2f}, Angle Delta = {angle_delta_degrees:.2f}")

                if self.check_exit_condition(
                    upper_index, threshold_crossed, circle_exit_idx, entries, max(cw_ang_del_sum, ccw_ang_del_sum),
                    angle_deltas, debug, debug_frame_range_low, debug_frame_range_high
                ):
                    return

                if sign(angle_delta_degrees) != turn_direction:
                    turn_end_idx = fm_i
                    if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                        print(f"Turn end detected at frame {fm_i}, Turn End Index = {turn_end_idx}")
                    break

        if turn_st_idx == -1 or turn_end_idx == -1:
            return

        displacement = euclidean_norm(
            self.x_view[turn_st_idx], self.y_view[turn_st_idx],
            self.x_view[turn_end_idx], self.y_view[turn_end_idx]
        )
        dist_trav = self.distTrav(turn_st_idx, turn_end_idx)

        if dist_trav < dist_threshold:
            return

        if self.too_little_walking(turn_st_idx, turn_end_idx):
            return

        if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
            print(f"Turn start frame = {turn_st_idx}, Turn end frame = {turn_end_idx}")
            print(f"Displacement = {displacement:.2f}, Distance Traveled = {dist_trav:.2f}")
            input()

        self.lg_turn_dists[trj.f].start[trn_idx][circle_exit_idx] = compute_distance_or_nan(
            self.circle_ctr[0], self.circle_ctr[1], self.x_view[turn_st_idx], self.y_view[turn_st_idx]
        ) / (trj.pxPerMmFloor * self.va.xf.fctr)

        self.lg_turn_dists[trj.f].end[trn_idx][circle_exit_idx] = compute_distance_or_nan(
            self.circle_ctr[0], self.circle_ctr[1], self.x_view[turn_end_idx], self.y_view[turn_end_idx]
        ) / (trj.pxPerMmFloor * self.va.xf.fctr)

        self.indices_of_turns[trj.f].push_back(pair[int, int](turn_st_idx, turn_end_idx))

    cdef bint too_little_walking(self, int turn_st_idx, int turn_end_idx):
        # Determines if the proportion of frames identified as 'walking' between the start and
        # end indices of a turn is below a threshold.
        #
        # Parameters:
        # - turn_st_idx : int
        #     The start index of the turn in the trajectory.
        # - turn_end_idx : int
        #     The end index of the turn in the trajectory.
        #
        # Returns:
        # - bint
        #     True if the proportion of 'walking' frames is less than 30%, indicating
        #     insufficient walking activity to constitute a significant turn. Otherwise, False.
        cdef int count_true = 0
        cdef int idx

        for idx in range(turn_st_idx, turn_end_idx):
            count_true += self.walking_view[idx]

        return count_true / (turn_end_idx - turn_st_idx) < 0.30

    cdef void update_angle_and_sums(self, int fm_i, int upper_index, trj, 
                                vector[double] angle_deltas, 
                                double* cw_ang_del_sum, double* ccw_ang_del_sum,
                                int* cw_initial_frame, int* ccw_initial_frame,
                                bint debug=False, int debug_frame_range_low=0, int debug_frame_range_high=0):
        """
        Helper function to calculate angle deltas and update cumulative sums for CW and CCW turns.
        """
        theta_1 = trj.theta[fm_i] * PI / 180
        theta_2 = trj.theta[upper_index] * PI / 180
        angle_delta = angleDiff(
            theta_2, theta_1, absVal=False, useRadians=True
        )
        angle_delta_degrees = angle_delta * (180.0 / PI)
        abs_angle_delta = fabs(angle_delta_degrees)

        angle_deltas.push_back(angle_delta_degrees)

        if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
            print(f"Frame {fm_i}: Speed = {trj.sp[fm_i + 1]:.2f}, Upper Frame = {upper_index}, Speed (upper) = {trj.sp[upper_index + 1]:.2f}")
            print(f'Lower angle: {trj.theta[fm_i]:.2f}, Upper angle: {trj.theta[upper_index]:.2f}')
            print(f"Angle Delta (degrees) = {angle_delta_degrees:.2f}, Absolute Angle Delta = {abs_angle_delta:.2f}")
            print(f"CW Cumulative Angle: {cw_ang_del_sum[0]:.2f} degrees, CCW Cumulative Angle: {ccw_ang_del_sum[0]:.2f} degrees")

        if abs_angle_delta >= 18:
            if sign(angle_delta_degrees) > 0 and cw_initial_frame[0] == -1:
                cw_initial_frame[0] = fm_i
                if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                    print(f"CW Turn started at frame {cw_initial_frame[0]}")
            elif sign(angle_delta_degrees) < 0 and ccw_initial_frame[0] == -1:
                ccw_initial_frame[0] = fm_i
                if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                    print(f"CCW Turn started at frame {ccw_initial_frame[0]}")

        if cw_initial_frame[0] != -1:
            cw_ang_del_sum[0] += angle_delta_degrees

        if ccw_initial_frame[0] != -1:
            ccw_ang_del_sum[0] += angle_delta_degrees

    cdef bint check_speed_threshold(
        self, int frame_idx, trj, bint debug, int debug_frame_range_low, int debug_frame_range_high
    ):
        """
        Checks whether the speed at a given frame index is below the minimum turn speed threshold.

        Parameters:
        - frame_idx : int
            The frame index being checked.
        - trj : Trajectory
            The trajectory object containing speed data.
        - debug : bool
            Whether debug output is enabled.
        - debug_frame_range_low : int
            Lower bound of the debug frame range.
        - debug_frame_range_high : int
            Upper bound of the debug frame range.

        Returns:
        - bint : True if the speed is below the threshold, False otherwise.
        """
        if trj.sp[frame_idx] < self.min_turn_speed_px:
            if debug and debug_frame_range_low <= frame_idx <= debug_frame_range_high:
                print(f"Frame {frame_idx} skipped: Speed = {trj.sp[frame_idx]:.2f} (below threshold of {self.min_turn_speed_px:.2f})")
            return True
        return False

    cdef check_exit_condition(self, int fm_i, bint threshold_crossed, int circle_exit_idx, 
                          np.ndarray entries, double angle_delta_degrees, vector[double] angle_deltas,
                          bint debug=False, int debug_frame_range_low=0, int debug_frame_range_high=0):
        """
        Helper function to check for turn exit conditions
        """
        exit_result = self.check_turn_exit_conditions(
            fm_i, threshold_crossed, circle_exit_idx, entries, angle_delta_degrees, angle_deltas
        )
        if exit_result:
            if debug and debug_frame_range_low <= fm_i <= debug_frame_range_high:
                print(f'Event-ending condition detected at frame {fm_i}; no turn.')
            return True
        return False

    def check_turn_exit_conditions(
        self,
        int fm_i,
        bint threshold_crossed,
        int circle_exit_idx,
        np.ndarray entries,
        double angle_delta_sum,
        vector[double] angle_deltas,
    ):
        # Evaluates conditions to determine the end of a turn or to invalidate a turn based on
        # spatial criteria and interaction with environmental boundaries.
        #
        # Parameters:
        # - fm_i : int
        #     The current frame index being evaluated.
        # - threshold_crossed : bool
        #     Whether a significant turning threshold has been crossed.
        # - circle_exit_idx : int
        #     The index of the exit from the reward circle.
        # - entries : np.ndarray
        #     Indices of entries into the reward circle.
        # - angle_delta_sum : double
        #     The cumulative sum of angle deltas for the current turn.
        # - angle_deltas : list of double
        #     The sequence of angle deltas for the current turn.
        #
        # Returns:
        # - tuple or None
        #     A tuple containing the updated frame index, angle_delta_sum, and angle_deltas if
        #     conditions are met; None otherwise.

        cdef bint after_reentry_frame = circle_exit_idx + 1 < len(entries) and entries[circle_exit_idx + 1] <= fm_i
        cdef bint inside_circle = euclidean_norm(
            self.circle_ctr[0], 
            self.circle_ctr[1],
            self.x_view[fm_i],
            self.y_view[fm_i]
        ) < self.circle_rad
        
        if after_reentry_frame or inside_circle or self.wall_contact[fm_i]:
            if threshold_crossed:
                return fm_i - self.end_turn_before_recontact, angle_delta_sum, angle_deltas
            else:
                return -1, -1, angle_deltas
        
        return None

    cdef double distTrav(self, int i1, int i2):
        # Calculates the total distance traveled between two indices in the trajectory.
        #
        # Parameters:
        # - i1 : int
        #     The starting index for the distance calculation.
        # - i2 : int
        #     The ending index for the distance calculation.
        #
        # Returns:
        # - double
        #     The total distance traveled between the two indices.
        #
        # This method sums the distances traveled between consecutive frames from the starting
        # index (i1) to the ending index (i2), providing a measure of how much distance the
        # subject has covered over that portion of the trajectory.
        cdef double result = 0.0
        cdef int i
        for i in range(i1, i2):
            result += self.d_view[i]
        return result
