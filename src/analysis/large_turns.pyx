#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.math cimport (fabs, nan, cos, sin, acos, sqrt)

from src.plotting.event_chain_plotter import EventChainPlotter
from src.utils.common_cython cimport (
    angleDiff,
    compute_distance_or_nan,
    euclidean_norm,
    in_range,
    ndarray_long_to_vector,
    ndarray_float_to_vector,
    PI,
    sign,
)

# ---- helpers (module scope) ----
cdef inline int _sign_with_deadband(double x, double eps):
    if x > eps:
        return 1
    elif x < -eps:
        return -1
    else:
        return 0

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
    cdef str trj_plot_mode
    cdef str image_format
    cdef vector[TurnData] lg_turn_dists
    cdef vector[HistData] large_turn_hist_counts_edges
    cdef vector[vector[double]] turn_to_exit_ratios
    cdef list indices_of_turns
    cdef list turn_circle_index_mapping
    cdef vector[pair[int, int]] trn_ranges
    cdef long[:] circle_ctr
    cdef double circle_rad
    cdef double[:] dist_from_ctr_at_end
    cdef vector[vector[vector[double]]] large_turn_stats
    cdef bint collect_exit_events
    cdef double weaving_max_outside_mm
    cdef double tangent_thresh_deg
    cdef double weaving_tangent_flip_buffer_deg
    cdef double backward_frac_thresh

    def __cinit__(
        self,
        object va,
        double min_turn_speed,
        int num_hist_bins,
        int end_turn_before_recontact,
        str trj_plot_mode=None,
        str image_format='png',
        bint collect_exit_events=False,
        double weaving_max_outside_mm=1.5,
        double tangent_thresh_deg=30.0,
        double weaving_tangent_flip_buffer_deg=5.0,
        double backward_frac_thresh=0.6
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
        # - trj_plot_mode (string): Mode for large turn trajectory plot: 'all_types' to include
        #   all frames between two large turns or 'turn_plus_1' to show a single large turn
        #   and a subsequent non-turn event. Defaults to None (no plotting)
        self.va = va
        self.min_turn_speed = min_turn_speed
        self.num_hist_bins = num_hist_bins
        self.lg_turn_dists = vector[TurnData]()
        self.large_turn_hist_counts_edges = vector[HistData]()
        self.turn_to_exit_ratios = vector[vector[double]]()
        self.end_turn_before_recontact = end_turn_before_recontact
        self.trj_plot_mode = trj_plot_mode
        self.image_format = image_format
        self.collect_exit_events = collect_exit_events

        # High-performing learning strategy parameters
        self.weaving_max_outside_mm = weaving_max_outside_mm
        self.tangent_thresh_deg = tangent_thresh_deg
        self.weaving_tangent_flip_buffer_deg = weaving_tangent_flip_buffer_deg
        self.backward_frac_thresh = backward_frac_thresh

    def calcLargeTurnsAfterCircleExit(self):
        # Calculates large turns occurring after exiting a circle by analyzing the trajectory
        # data.
        #
        # This method identifies ranges of trajectory indices that represent turns after exiting
        # a predefined circle. It updates internal structures with the details of these turns,
        # such as their start and stop indices, and computes statistics related to these large
        # turns for further analysis.

        # Reset lightweight per-fly/per-training weaving summary for this analysis run.
        # Shape after analysis:
        #   va.weaving_exit_stats[fly_idx][training_idx] = (weaving_count, total_exits)
        # where training_idx indexes trainings 0..N-1 (pre is excluded).
        self.va.weaving_exit_stats = []

        # Reset lightweight per-fly/per-training small-angle re-entry summary.
        # Shape after analysis:
        #   va.small_angle_exit_stats[fly_idx][training_idx] = (small_angle_count, total_exits)
        self.va.small_angle_exit_stats = []

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

        if self.collect_exit_events:
            if not hasattr(self.va, "lg_turn_exit_events"):
                self.va.lg_turn_exit_events = []
            while len(self.va.lg_turn_exit_events) <= trj.f:
                self.va.lg_turn_exit_events.append([])

        if not hasattr(self.va, 'lg_turn_rejection_reasons'):
            self.va.lg_turn_rejection_reasons = []
        if len(self.va.lg_turn_rejection_reasons) <= trj.f:
            self.va.lg_turn_rejection_reasons.append([])

        self.turn_to_exit_ratios.push_back(vector[double]())
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
        cdef int trn_i, fly_i
        cdef int num_trns = len(self.va.trns)
        cdef int num_trx = len(self.va.trx)
        cdef int range_idx
        cdef double sum_, med_dist_

        # One entry per training session (pre is omitted from large_turn_stats)
        self.large_turn_stats.resize(num_trns)

        for trn_i in range(num_trns):
            # Map training index (0..num_trns-1) to trn_ranges index (1..num_trns)
            # 0 == pre, so skip that and use i+1
            range_idx = trn_i + 1

            self.large_turn_stats[trn_i].resize(num_trx)

            for fly_i in range(num_trx):
                # [0] = total large-turn count, [1] = median distance, [2] = turn/exit ratio
                self.large_turn_stats[trn_i][fly_i].resize(3, nan(""))

                # Guard in case a trajectory has fewer ranges (shouldn't normally happen,
                # but safer if some trajectories were skipped or truncated)
                if range_idx >= self.large_turn_hist_counts_edges[fly_i].end.size():
                    continue

                sum_, med_dist_ = RewardCircleAnchoredTurnFinder.compute_statistic(
                    self.large_turn_hist_counts_edges[fly_i].end[range_idx].first,
                    self.large_turn_hist_counts_edges[fly_i].end[range_idx].second
                )

                self.large_turn_stats[trn_i][fly_i][0] = sum_
                self.large_turn_stats[trn_i][fly_i][1] = med_dist_

                if range_idx < self.turn_to_exit_ratios[fly_i].size():
                    self.large_turn_stats[trn_i][fly_i][2] = self.turn_to_exit_ratios[fly_i][range_idx]


    cdef getTurnsForRange(self, trj, i, trn_range):
        cdef object rejection_map = None
        cdef dict exit_to_turn
        cdef int j, k
        cdef int turn_st_idx, turn_end_idx
        cdef int seg_start, seg_end
        cdef int weaving_nonlarge = 0
        cdef int small_angle_nonlarge = 0
        cdef int total_exits

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

        # Initialize rejection reasons for this timeframe
        # Each trajectory (`trj.f`) will have its own sublist for each timeframe's exits
        if len(self.va.lg_turn_rejection_reasons[trj.f]) <= i:
            self.va.lg_turn_rejection_reasons[trj.f].append({})

        self.turn_circle_index_mapping = []  # Reset for each training session
        self.indices_of_turns = []

        debug_flag = bool(getattr(self.va, "debug_large_turns", False))
        for j, ex_fm in enumerate(exits):
            self.run_turn_search(trj, i, ex_fm, j, entries, debug_flag)

        # ------------------ summarize exits for this fly / time range ------------------
        # Build a mapping: exit_idx -> (turn_start_idx, turn_end_idx)
        exit_to_turn = {}
        for k, exit_idx_for_turn in enumerate(self.turn_circle_index_mapping):
            turn_st_idx, turn_end_idx = self.indices_of_turns[k]
            exit_to_turn[exit_idx_for_turn] = (turn_st_idx, turn_end_idx)

        # Convenience handle for rejection reasons for this fly / time range
        if trj.f < len(self.va.lg_turn_rejection_reasons):
            if i < len(self.va.lg_turn_rejection_reasons[trj.f]):
                rejection_map = self.va.lg_turn_rejection_reasons[trj.f][i]

        # Per-range weaving summary
        total_exits = len(exits)

        debug_exits = bool(getattr(self.va, "debug_large_turns", False))

        # One record per reward-circle exit in this time range (pre or training)
        for exit_idx, ex_fm in enumerate(exits):
            if exit_idx in exit_to_turn:
                turn_st_idx, turn_end_idx = exit_to_turn[exit_idx]
                has_turn = True
            else:
                turn_st_idx = -1
                turn_end_idx = -1
                has_turn = False

            # Defaults for strategy-related fields
            reentry_frame = None
            max_outside_mm = None
            angle_to_tangent_deg = None
            frac_backward = None
            strategy_weaving = False
            strategy_backward = False
            reason = None

            # Use the rejection map and cached metrics, if present
            if rejection_map is not None and exit_idx in rejection_map:
                rej_val = rejection_map[exit_idx]
                reason = rej_val[0]
                evt_start_idx, evt_end_idx = rej_val[1]

                if reason in ("small_angle_reentry", "weaving", "backward_walking"):
                    # Re-entry type event
                    reentry_frame = int(evt_end_idx)

                # If metrics were cached during classification, fetch them
                if hasattr(self.va, "lg_turn_exit_metrics"):
                    per_fly_metrics = self.va.lg_turn_exit_metrics
                    if trj.f < len(per_fly_metrics):
                        per_trn = per_fly_metrics[trj.f]
                        if i < len(per_trn):
                            per_exit = per_trn[i]
                            if exit_idx in per_exit:
                                m = per_exit[exit_idx]
                                max_outside_mm = m.get("max_outside_mm")
                                angle_to_tangent_deg = m.get("angle_to_tangent_deg")
                                frac_backward = m.get("frac_backward")
                                strategy_weaving = bool(m.get("is_weaving", False))
                                strategy_backward = bool(m.get("is_backward", False))

                # Ensure booleans are at least consistent with the reason label,
                # even if metrics dict is missing for some reason.
                if reason == "weaving":
                    strategy_weaving = True
                elif reason == "backward_walking":
                    strategy_backward = True

            # --- Summary: count weaving / small-angle among NON-large-turn exits only ---
            if (not has_turn) and (reason == "weaving"):
                weaving_nonlarge += 1
            elif (not has_turn) and (reason == "small_angle_reentry"):
                small_angle_nonlarge += 1

            # --- Optional: heavy per-exit record, only if dump flag is enabled ---
            if self.collect_exit_events:
                self.va.lg_turn_exit_events[trj.f].append(
                    {
                        "trn_range_idx": i,            # 0 = pre, 1..N = trn_ranges 1..N
                        "exit_idx": exit_idx,          # index within 'exits' for this range
                        "exit_frame": int(ex_fm),      # absolute frame index
                        "has_large_turn": has_turn,    # True if a large turn was accepted
                        "turn_start_idx": (
                            int(turn_st_idx) if has_turn else None
                        ),
                        "turn_end_idx": (
                            int(turn_end_idx) if has_turn else None
                        ),
                        # Strategy-related fields (if available):
                        "reentry_frame": reentry_frame,
                        "strategy_weaving": bool(strategy_weaving),
                        "strategy_backward_walking": bool(strategy_backward),
                        "max_outside_mm": max_outside_mm,
                        "angle_to_tangent_deg": angle_to_tangent_deg,
                        "frac_backward_frames": frac_backward,
                    }
                )

                if debug_exits:
                    if bool(strategy_weaving):
                        print(f"weaving re-entry detected, f{trj.f}, {int(ex_fm)} to {reentry_frame}")
                        input()
                    elif (not has_turn) and (reason == 'small_angle_reentry'):
                        print(f"small-angle re-entry detected, f{trj.f},  {int(ex_fm)} to {reentry_frame}")
                        input()
                    elif has_turn:
                        print(f"large turn detected, f{trj.f}, {int(turn_st_idx)} to {int(turn_end_idx)}")
                        input()

        # Accumulate a light-weight per-fly/per-training summary, independent
        # of whether per-exit dumps are enabled.
        self._accumulate_weaving_stats(trj.f, i, weaving_nonlarge, total_exits)
        self._accumulate_small_angle_stats(trj.f, i, small_angle_nonlarge, total_exits)

        # Large turn-to-exit ratio calculation
        self.turn_to_exit_ratios[trj.f].push_back(self.calc_turn_to_exit_ratio(trj, i))
        self.compute_histogram(trj, i)

        # Check if plotting flag is enabled
        if self.trj_plot_mode:
            # Call to EventChainPlotter, passing large turn-specific variables
            large_turn_plotter = EventChainPlotter(trj, self.va, image_format=self.image_format)

            # Define variables that map to large turn events
            lg_turn_idxs = self.indices_of_turns
            turn_circle_index_mapping = self.turn_circle_index_mapping
            start_frame = trn_range[0]
            stop_frame = trn_range[1]
            if len(lg_turn_idxs) == 0:
                return
            large_turn_plotter._plot_large_turn_event_chain(
                exits=exits,
                trn_index=i - 1,
                turning_idxs_filtered=lg_turn_idxs,
                turn_circle_index_mapping=turn_circle_index_mapping,
                rejection_reasons=self.va.lg_turn_rejection_reasons[trj.f][i],
                plot_mode=self.trj_plot_mode,
                start_frame=start_frame,
                stop_frame=stop_frame,
            )

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

    cdef void set_circle_data(self, trj, int trn_range_idx):
        # Sets the circle data (center and radius) for the current trajectory and training
        # session.
        #
        # Parameters:
        # - trj : Trajectory
        #     The trajectory object being analyzed.
        # - trn_range_idx : int
        #     index into self.trn_ranges
        #     0      -> pre
        #     1..N   -> training 1..N (self.va.trns[0..N-1])
        #
        # This method extracts and sets the circle data (center and radius) for the trajectory's
        # current training session. It adjusts the index for accessing the circle data and
        # retrieves this information from the trajectory's associated training session, storing
        # it for use in large turn analysis.
        cdef int trn_idx

        if trn_range_idx <= 0:
            # Pre period: use the first training's circle (adjust if you prefer something else)
            trn_idx = 0
        else:
            trn_idx = trn_range_idx - 1

        circle = self.va.trns[trn_idx].circles(trj.f)[0]
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
        debug_frame_range_low = 0
        debug_frame_range_high = 1000000
        if debug and debug_frame_range_low <= circle_exit_frame <= debug_frame_range_high:
            print(f"starting turn search for frame {circle_exit_frame}")
        while True:
            fm_i += 1

            if fm_i + 2 >= len(trj.sp):
                return  # Exit if we exceed trajectory length

            # Use the new helper function to check the speed threshold
            if self.check_speed_threshold(
                fm_i + 1, trj, debug, debug_frame_range_low, debug_frame_range_high
            ):
                if debug and debug_frame_range_low <= (fm_i + 1) <= debug_frame_range_high:
                    print(f"continuing; frame {fm_i + 1} below speed threshold")
                continue

            upper_index = fm_i + 1
            while (
                upper_index + 1 < len(trj.sp) and
                self.check_speed_threshold(
                    upper_index + 1, trj, debug, debug_frame_range_low, debug_frame_range_high
                )
            ):
                if debug and debug_frame_range_low <= upper_index <= debug_frame_range_high:
                    print(f"frame {upper_index} - speed too low. Skipping.")
                upper_index += 1

            if upper_index >= len(trj.sp):
                return

            if self.check_exit_condition(
                upper_index,
                circle_exit_frame if turn_st_idx == -1 else turn_st_idx,
                threshold_crossed,
                circle_exit_idx,
                trn_idx,
                entries,
                max(cw_ang_del_sum, ccw_ang_del_sum),
                angle_deltas,
                trj,
                debug,
                debug_frame_range_low,
                debug_frame_range_high,
                update_rejection_reasons=True
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
                    upper_index,
                    turn_st_idx,
                    threshold_crossed,
                    circle_exit_idx,
                    trn_idx,
                    entries,
                    max(cw_ang_del_sum, ccw_ang_del_sum),
                    angle_deltas,
                    trj,
                    debug,
                    debug_frame_range_low,
                    debug_frame_range_high,
                    update_rejection_reasons=True
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
            self.va.lg_turn_rejection_reasons[trj.f][trn_idx][circle_exit_idx] =(
                "low_displacement", (turn_st_idx, turn_end_idx)
            )
            return

        if self.too_little_walking(turn_st_idx, turn_end_idx):
            self.va.lg_turn_rejection_reasons[trj.f][trn_idx][circle_exit_idx] = (
                'too_little_walking', (turn_st_idx, turn_end_idx)
            )
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

        self.indices_of_turns.append((turn_st_idx, turn_end_idx))
        self.turn_circle_index_mapping.append(circle_exit_idx)

    cdef inline double walking_fraction(self, int start_idx, int end_idx):
        cdef int k
        cdef int n = end_idx - start_idx
        cdef int count_true = 0
        if n <= 0:
            return 0.0
        for k in range(start_idx, end_idx):
            count_true += self.walking_view[k]
        return count_true / <double> n

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

    cdef check_exit_condition(
        self,
        int event_end_idx,
        int event_start_idx,
        bint threshold_crossed,
        int circle_exit_idx,
        int trn_idx,
        np.ndarray entries,
        double angle_delta_degrees,
        vector[double] angle_deltas,
        trj,
        bint debug=False,
        int debug_frame_range_low=0,
        int debug_frame_range_high=0,
        bint update_rejection_reasons=False
    ):
        """
        Helper function to check for turn exit conditions
        """
        exit_result = self.check_turn_exit_conditions(
            event_end_idx,
            event_start_idx,
            threshold_crossed,
            circle_exit_idx,
            trn_idx,
            entries,
            angle_delta_degrees,
            angle_deltas,
            trj,
            update_rejection_reasons
        )
        if exit_result:
            if debug and debug_frame_range_low <= event_end_idx <= debug_frame_range_high:
                print(f'Event-ending condition detected at frame {event_end_idx}; no turn.')
            return True
        return False

    def check_turn_exit_conditions(
        self,
        int event_end_idx,
        int event_start_idx,
        bint threshold_crossed,
        int circle_exit_idx,
        int trn_idx,
        np.ndarray entries,
        double angle_delta_sum,
        vector[double] angle_deltas,
        trj,
        bint update_rejection_reasons=False
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
        # - trj : Trajectory
        #     The Trajectory instance for the fly associated with the event
        # - update_rejection_reasons : bint
        #     Whether to update the entry in the list of rejection reasons for this event
        #     if an exit condition is detected.
        # 
        #
        # Returns:
        # - tuple or None
        #     A tuple containing the updated frame index, angle_delta_sum, and angle_deltas if
        #     conditions are met; None otherwise.
        cdef int reentry_frame = -1
        cdef bint reached_reentry = False
        cdef int eff_end_idx
        cdef bint weaving_geom_ok = False
        cdef bint weaving_motion_ok = False

        # For exit index k, we assume:
        #   entries[k]     = entry *before* the exit
        #   entries[k + 1] = first entry *after* the exit (the re-entry we care about)
        if circle_exit_idx + 1 < entries.shape[0]:
            reentry_frame = <int> entries[circle_exit_idx + 1]
            reached_reentry = (event_end_idx >= reentry_frame)

        # Use the *true* re-entry frame as the effective end of the segment
        if reached_reentry:
            eff_end_idx = reentry_frame
        else:
            eff_end_idx = event_end_idx

        cdef bint inside_circle = euclidean_norm(
            self.circle_ctr[0],
            self.circle_ctr[1],
            self.x_view[eff_end_idx],
            self.y_view[eff_end_idx],
        ) < self.circle_rad

        cdef double max_outside_mm = 0.0
        cdef double angle_to_tangent_deg = nan("")
        cdef double frac_backward = 0.0
        cdef double px_per_mm = 0.0
        cdef double dist_trav_mm = 0.0
        cdef double walk_frac = 0.0
        cdef bint is_weaving = False
        cdef bint is_backward = False
        cdef object reason

        # Event ends because:
        #   - we have reached the known re-entry frame, OR
        #   - we are inside the circle at eff_end_idx, OR
        #   - we hit the wall at eff_end_idx
        if reached_reentry or inside_circle or self.wall_contact[eff_end_idx]:
            if reached_reentry or inside_circle:
                # Classify using the *true exit→re-entry* segment:
                (
                    max_outside_mm,
                    angle_to_tangent_deg,
                    frac_backward,
                    is_weaving,
                    is_backward,
                    weaving_geom_ok,
                    weaving_motion_ok,
                ) = self.classify_weaving_and_backward_for_segment(
                    trj, event_start_idx, eff_end_idx
                )

                if is_weaving:
                    reason = "weaving"
                elif is_backward:
                    reason = "backward_walking"
                elif weaving_geom_ok and (not weaving_motion_ok):
                    px_per_mm = trj.pxPerMmFloor * self.va.xf.fctr
                    dist_trav_mm = self.distTrav(event_start_idx, eff_end_idx) / px_per_mm
                    walk_frac = self.walking_fraction(event_start_idx, eff_end_idx)
                    if dist_trav_mm < 1.0:
                        reason = "low_displacement"
                    elif walk_frac < 0.30:
                        reason = "too_little_walking"
                    else:
                        reason = "small_angle_reentry"
                else:
                    reason = "small_angle_reentry"

                if update_rejection_reasons:
                    self.va.lg_turn_rejection_reasons[trj.f][trn_idx][circle_exit_idx] = (
                        reason,
                        (event_start_idx, eff_end_idx),
                    )

                # Only cache metrics if requested; this will be used later in
                # getTurnsForRange to build lg_turn_exit_events without
                # recomputing classify_weaving_and_backward_for_segment().
                if self.collect_exit_events:
                    self._store_strategy_metrics(
                        trj.f,
                        trn_idx,
                        circle_exit_idx,
                        eff_end_idx,
                        max_outside_mm,
                        angle_to_tangent_deg,
                        frac_backward,
                        is_weaving,
                        is_backward,
                    )
            else:
                # Pure wall-contact-terminated event; no re-entry classification.
                if update_rejection_reasons:
                    self.va.lg_turn_rejection_reasons[trj.f][trn_idx][circle_exit_idx] = (
                        "wall_contact",
                        (event_start_idx, eff_end_idx),
                    )

            if threshold_crossed:
                return event_end_idx - self.end_turn_before_recontact, angle_delta_sum, angle_deltas
            else:
                # No large turn: this exit is "rejected" (weaving/backward/small-angle or wall)
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

    cdef void _store_strategy_metrics(
        self,
        int fly_idx,
        int trn_idx,
        int exit_idx,
        int reentry_frame,
        double max_outside_mm,
        double angle_to_tangent_deg,
        double frac_backward,
        bint is_weaving,
        bint is_backward,
    ):
        """
        Internal helper to cache weaving/backward metrics for one exit→re-entry
        segment so that we can reuse them when building lg_turn_exit_events.
        Shape: va.lg_turn_exit_metrics[fly_idx][trn_idx][exit_idx] -> dict
        """
        cdef list per_fly
        cdef list per_trn
        cdef dict per_exit
        cdef dict metrics

        if not hasattr(self.va, "lg_turn_exit_metrics"):
            self.va.lg_turn_exit_metrics = []

        per_fly = self.va.lg_turn_exit_metrics

        # Ensure per-fly list exists
        while len(per_fly) <= fly_idx:
            per_fly.append([])

        per_trn = per_fly[fly_idx]
        # Ensure per-training-range list exists
        while len(per_trn) <= trn_idx:
            per_trn.append({})

        per_exit = per_trn[trn_idx]

        metrics = {
            "reentry_frame": int(reentry_frame),
            "max_outside_mm": max_outside_mm,
            "angle_to_tangent_deg": angle_to_tangent_deg,
            "frac_backward": frac_backward,
            "is_weaving": bool(is_weaving),
            "is_backward": bool(is_backward),
        }
        per_exit[exit_idx] = metrics

    cdef void _accumulate_weaving_stats(
        self,
        int fly_idx,
        int trn_range_idx,
        int weaving_nonlarge,
        int total_exits,
    ):
        """
        Accumulate per-fly, per-training weaving-exit counts and total exits,
        independent of whether per-exit dumps are enabled.

        Layout on VideoAnalysis:
            va.weaving_exit_stats[fly_idx][training_idx] = (weaving_count, total_exits)

        where:
            trn_range_idx = 0  -> pre   (ignored here)
                        1..N -> training 1..N
            training_idx  = trn_range_idx - 1  (0-based)
        """
        cdef int training_idx
        cdef list per_fly
        cdef list per_trn
        cdef tuple old_val
        cdef int prev_weaving, prev_total

        # Map range index (0 = pre, 1..N = trainings) to training index 0..N-1.
        if trn_range_idx <= 0:
            return

        training_idx = trn_range_idx - 1

        if not hasattr(self.va, "weaving_exit_stats"):
            self.va.weaving_exit_stats = []

        per_fly = self.va.weaving_exit_stats

        # Ensure per-fly list exists
        while len(per_fly) <= fly_idx:
            per_fly.append([])

        per_trn = per_fly[fly_idx]

        # Ensure per-training index slot exists
        while len(per_trn) <= training_idx:
            per_trn.append((0, 0))

        old_val = per_trn[training_idx]
        prev_weaving = int(old_val[0])
        prev_total = int(old_val[1])

        per_trn[training_idx] = (
            prev_weaving + weaving_nonlarge,
            prev_total + total_exits,
        )

    cdef void _accumulate_small_angle_stats(
        self,
        int fly_idx,
        int trn_range_idx,
        int small_angle_nonlarge,
        int total_exits,
    ):
        """
        Accumulate per-fly, per-training small-angle re-entry counts and
        total exits, independent of whether per-exit dumps are enabled.

        Layout on VideoAnalysis:
            va.small_angle_exit_stats[fly_idx][training_idx] = (small_angle_count, total_exits)

        where:
            trn_range_idx = 0  -> pre   (ignored here)
                        1..N -> training 1..N
            training_idx  = trn_range_idx - 1  (0-based)
        """
        cdef int training_idx
        cdef list per_fly
        cdef list per_trn
        cdef tuple old_val
        cdef int prev_small, prev_total

        # Map range index (0 = pre, 1..N = trainings) to training index 0..N-1.
        if trn_range_idx <= 0:
            return

        training_idx = trn_range_idx - 1

        if not hasattr(self.va, "small_angle_exit_stats"):
            self.va.small_angle_exit_stats = []

        per_fly = self.va.small_angle_exit_stats

        # Ensure per-fly list exists
        while len(per_fly) <= fly_idx:
            per_fly.append([])

        per_trn = per_fly[fly_idx]

        # Ensure per-training index slot exists
        while len(per_trn) <= training_idx:
            per_trn.append((0, 0))

        old_val = per_trn[training_idx]
        prev_small = int(old_val[0])
        prev_total = int(old_val[1])

        per_trn[training_idx] = (
            prev_small + small_angle_nonlarge,
            prev_total + total_exits,
        )

    cdef bint heading_tangent_side_consistent_for_segment(
        self,
        trj,
        int start_idx,
        int end_idx,
        double sign_eps
    ):
        """
        Returns True iff the sign of (heading · tangent_unit) stays consistent over
        the segment, ignoring frames where abs(dot) <= sign_eps (near 90°).
        """
        cdef int k
        cdef double dx, dy, tx, ty, t_norm
        cdef double heading_rad, hx, hy
        cdef double dot_ht, cos_raw
        cdef int side_ref = 0
        cdef int side_cur

        for k in range(start_idx, end_idx + 1):
            dx = self.x_view[k] - self.circle_ctr[0]
            dy = self.y_view[k] - self.circle_ctr[1]

            # Tangent vector (CCW): (-dy, dx)
            tx = -dy
            ty = dx
            t_norm = sqrt(tx * tx + ty * ty)
            if t_norm <= 1e-12:
                continue

            heading_rad = trj.theta[k] * PI / 180.0
            hx = sin(heading_rad)
            hy = -cos(heading_rad)

            dot_ht = hx * tx + hy * ty
            cos_raw = dot_ht / t_norm  # hx,hy is unit; divide by |t|

            side_cur = _sign_with_deadband(cos_raw, sign_eps)
            if side_cur == 0:
                continue

            if side_ref == 0:
                side_ref = side_cur
            elif side_cur != side_ref:
                return False
        
        # If we never got a confident sign (all near 90°), treat as consistent.
        return True

    cdef tuple classify_weaving_and_backward_for_segment(
        self,
        trj,
        int start_idx,
        int end_idx
    ):
        """
        Classify a single exit→re-entry segment as weaving and/or backward walking.

        Returns:
            (max_outside_mm, angle_to_tangent_deg, frac_backward,
             is_weaving, is_backward,
             weaving_geom_ok, weaving_motion_ok
             )
        """
        cdef int n_frames = len(self.x_view)
        cdef int k
        cdef double px_per_mm
        cdef double rad_mm
        cdef double dist_ctr_px, dist_ctr_mm, outside_mm
        cdef double max_outside_mm = 0.0
        cdef double angle_to_tangent_deg = nan("")
        cdef double frac_backward = 0.0
        cdef int n_steps
        cdef int backward_count = 0

        cdef double dx, dy, radial_angle
        cdef double tx, ty
        cdef double dot_ht, cos_diff
        cdef double heading_rad, tangent_angle, diff_rad

        cdef double v_norm, vx, vx_hat, vy, vy_hat, hx, hy, dot

        cdef double dist_trav_px
        cdef double dist_trav_mm = 0.0
        cdef double walk_frac = 0.0
        cdef bint weaving_motion_ok = True
        cdef bint weaving_geom_ok = False

        cdef bint tangent_side_ok = True
        cdef double flip_buf_deg
        cdef double sign_eps

        # Clamp indices defensively
        if start_idx < 0:
            start_idx = 0
        if end_idx >= n_frames:
            end_idx = n_frames - 1
        if end_idx <= start_idx:
            return (max_outside_mm, angle_to_tangent_deg, frac_backward, False, False, False, False)

        px_per_mm = trj.pxPerMmFloor * self.va.xf.fctr
        rad_mm = self.circle_rad / px_per_mm

        # --- 1) Max distance outside circle (mm)
        for k in range(start_idx, end_idx + 1):
            dist_ctr_px = euclidean_norm(
                self.circle_ctr[0], self.circle_ctr[1],
                self.x_view[k], self.y_view[k]
            )
            dist_ctr_mm = dist_ctr_px / px_per_mm
            outside_mm = dist_ctr_mm - rad_mm
            if outside_mm > max_outside_mm:
                max_outside_mm = outside_mm

        if max_outside_mm < 0:
            max_outside_mm = 0.0

        # --- 2) Heading vs tangent at re-entry (end_idx)
        dx = self.x_view[end_idx] - self.circle_ctr[0]
        dy = self.y_view[end_idx] - self.circle_ctr[1]

        # Tangent vector (any orientation; we use CCW: (-dy, dx))
        tx = -dy
        ty = dx

        # Heading unit vector from theta in screen coords (x right, y down)
        heading_rad = trj.theta[end_idx] * PI / 180.0
        hx = sin(heading_rad)
        hy = -cos(heading_rad)

        # Compute angle between heading and tangent
        cdef double t_norm = sqrt(tx * tx + ty * ty)
        if t_norm > 0:
            dot_ht = hx * tx + hy * ty
            cos_diff = dot_ht / t_norm # heading is unit length

            if cos_diff > 1.0:
                cos_diff = 1.0
            elif cos_diff < -1.0:
                cos_diff = -1.0

            cos_diff = fabs(cos_diff)
            
            diff_rad = acos(cos_diff)
            angle_to_tangent_deg = diff_rad * (180.0 / PI)
        else:
            angle_to_tangent_deg = nan("")

        # --- 2b) Heading-to-tangent side consistency over the whole segment
        # Define deadband around 90° to avoid dot sign flicker.
        flip_buf_deg = self.weaving_tangent_flip_buffer_deg
        sign_eps = sin(flip_buf_deg * PI / 180.0) # dot threshold equivalent to (90° ± flip_buf_deg)
        if sign_eps < 0:
            sign_eps = -sign_eps
        if sign_eps > 1.0:
            sign_eps = 1.0
        tangent_side_ok = self.heading_tangent_side_consistent_for_segment(
            trj, start_idx, end_idx, sign_eps
        )

        # --- 2c) Motion-quality gates (for weaving acceptance only)
        # Match large-turn standards:
        #   - min distance traveled: 1.0 mm
        #   - min walking fraction: 0.30
        dist_trav_px = self.distTrav(start_idx, end_idx)
        dist_trav_mm = dist_trav_px / px_per_mm
        walk_frac = self.walking_fraction(start_idx, end_idx)
        weaving_motion_ok = (dist_trav_mm >= 1.0) and (walk_frac >= 0.30)

        # --- 3) Backward walking fraction (velocity opposite heading)
        n_steps = end_idx - start_idx
        if n_steps <= 0:
            return (max_outside_mm, angle_to_tangent_deg, frac_backward, False, False, False, False)

        for k in range(start_idx, end_idx):
            vx = self.x_view[k + 1] - self.x_view[k]
            vy = self.y_view[k + 1] - self.y_view[k]

            heading_rad = trj.theta[k] * PI / 180.0
            hx = sin(heading_rad)
            hy = -cos(heading_rad)

            v_norm = sqrt(vx * vx + vy * vy) + 1e-9
            vx_hat = vx / v_norm
            vy_hat = vy / v_norm
            dot = vx_hat * hx + vy_hat * hy
            if dot < -0.5:
                backward_count += 1
        
        frac_backward = backward_count / <double> n_steps

        # --- 4) Apply thresholds
        weaving_geom_ok = (
            max_outside_mm <= self.weaving_max_outside_mm
            and angle_to_tangent_deg <= self.tangent_thresh_deg
            and tangent_side_ok
        )
        cdef bint is_weaving = weaving_geom_ok and weaving_motion_ok
        cdef bint is_backward = (frac_backward >= self.backward_frac_thresh)

        return (
            max_outside_mm,
            angle_to_tangent_deg,
            frac_backward,
            is_weaving,
            is_backward,
            weaving_geom_ok,
            weaving_motion_ok
        )
