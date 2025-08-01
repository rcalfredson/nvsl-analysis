import numpy as np

from src.analysis.reward_range_calculator import RewardRangeCalculator
from src.utils.util import inRange


class ContactEventTrainingComparison:
    def __init__(
        self,
        N,
        t,
        n_bkts,
        rewards_for_fly,
        trj,
        evt_name,
        boundary_tp,
        boundary_orientation,
        event_indices,
        opts,
        save_stats=False,
        aux_data={},
    ):
        """
        Initializes a ContactEventTrainingComparison instance, which is designed to analyze and compare
        contact events with various boundaries relative to rewards received by Drosophila during spatial
        learning tasks within a confined chamber. This analysis is part of a behavioral study to
        understand how flies interact with their environment, especially in terms of seeking rewards,
        and to calculate ratios of contact events per reward.

        Args:
        - N (int): Number of rewards over which to average boundary-contact events at the beginning
                and end of each analyzed period of the experiment.
        - t (Training): An instance representing a time period when the virtual reward circle was
                        active, and the fly could pursue rewards.
        - n_bkts (int): The number of synchronization buckets (internal time divisions) in the training,
                        typically representing ten-minute intervals.
        - rewards_for_fly (array): A nested array containing the counts of rewards the fly received,
                                organized by training session and within by sync bucket.
        - trj (Trajectory): An instance containing the raw movement data of the fly within the chamber,
                            as captured by a video tracking system.
        - evt_name (str): The type of contact event to analyze (e.g., "wall_contact", "agarose_contact",
                        or a turn event like "inside_line_turn").
        - boundary_tp (str): The type of boundary to track, which can vary depending on the event
                            name and the specific experimental setup.
        - boundary_orientation (str): Specifies which orientation of walls to analyze (e.g., "all" for all
                        walls, "tb" for top and bottom walls, "agarose_adj" for boundaries adjacent to
                        agarose but with specific offsets).
        - event_indices (array): Indices marking the start times of contact events, aligned with the
                                time point system used in the Trajectory instance.
        - opts (argparse.Namespace): Configuration options passed from analyze.py.
        - save_stats (bool, optional): Whether to save statistical results to an attribute of the
                                    VideoAnalysis instance for later reference, particularly when
                                    outputting results to a CSV file. Defaults to False.
        - aux_data (dict, optional): A dictionary for any auxiliary data that might be needed for
                                    analysis. Defaults to an empty dictionary.

        This class primarily operates through its `calcContactMetrics` method, which calculates the
        ratios of the fly's contact events with various boundaries to its number of rewards, aiding
        in the analysis of spatial learning and behavior.
        """
        self.N = N
        self.t = t
        self.n_bkts = n_bkts
        self.rewards_for_fly = rewards_for_fly
        self.trj = trj
        self.va = trj.va
        self.evt_name = evt_name
        self.boundary_tp = boundary_tp
        self.boundary_orientation = boundary_orientation
        self.event_indices = event_indices
        self.opts = opts
        self.save_stats = save_stats
        self.aux_data = aux_data
        self.reward_range_calculator = RewardRangeCalculator(self.va, self.opts)

    def calcContactMetricRewardRangeLaterTraining(self):
        """
        Calculates the reward range for contact metrics during later training sessions,
        adapting the range based on the current session and event details.

        This method adjusts the reward range for the trajectory (`trj`) based on the event
        indices, event name (`evt_name`), boundary orientation (`boundary_orientation`),
        and training session details.

        The method updates `self.trj.reward_range` with the calculated slice indicating
        the range of frames eligible for reward calculation.

        Additionally, if no relevant changes in boundary events relative to the wall
        type are detected, the method sets `self.N` to 0, effectively ignoring the
        session for further reward calculation.

        Parameters:
        - None directly. Operates on attributes of the instance, particularly:
        - `opts`: Options dict containing configurations.
        - `trj`: Trajectory object that stores various trajectory-related data,
            including `va` for velocity/acceleration data and `reward_range` for
            determining which frames are considered for reward calculation.

        Returns:
        - None. Modifies the instance's `trj.reward_range` directly based on the
        available data. Also, updates `self.N` and potentially other attributes
        depending on the reward calculation logic.

        Note:
        - This method is specifically designed for handling reward range calculation
        during later training sessions, where previous session data may influence
        the calculation. It is part of a larger class that manages contact event
        training and comparison, indicating its use in a context where flies'
        interactions with boundaries or walls are significant for behavioral analysis.
        """
        self.reward_range_calculator.calc_reward_range_later_training(self.t)
        if len(self.va.reward_ranges) > self.t.n:
            self.trj.reward_range = self.va.reward_ranges[self.t.n]
        rewards_for_trn = self.rewards_for_fly[self.t.n - 1]
        if len(rewards_for_trn) > 0:
            self.N = rewards_for_trn[-1]
        else:
            self.N = 0

        if (
            np.all(
                np.isnan(
                    self.trj.va.boundary_event_rel_changes[self.evt_name][
                        self.boundary_orientation
                    ]
                )
            )
            and len(
                self.trj.va.boundary_event_rel_changes[self.evt_name][
                    self.boundary_orientation
                ]
            )
            > 1
        ):
            setattr(
                self.trj.va,
                self.va.csv_attr_name,
                np.array([[np.nan] * 2 * len(self.va.flies) * len(self.va.trns)]),
            )
            self.N = 0

    def calcContactMetricRewardRangeFirstTraining(self):
        """
        Establishes the reward range for contact metrics specifically for the first training session,
        setting up necessary data structures and calculating the initial reward range.

        This method performs several key steps:
        - Initializes or resets the `boundary_event_rel_raw_stats` attribute for the current
        event and wall type, ensuring a clean slate for the first training session's data.
        - Sets up `boundary_event_rel_changes` if it doesn't already exist, to track relative changes
        in boundary contact events over time. This attribute is crucial for understanding how
        interactions with the environment evolve across different training sessions and potentially
        influence the fly's behavior and learning process.
        - Calculates the pre-reward range as a period leading up to the first reward event,
        facilitating analysis of behavior immediately before reward acquisition.
        - Determines the reward range for the first training session.

        The reward range determination is crucial for subsequent analysis, as it dictates which
        frames from the trajectory data (trj) are considered when calculating contact metrics.
        This method sets the groundwork for analyzing how initial interactions with the environment
        potentially influence the fly's learning and behavior.

        Parameters:
        - None directly. Operates on attributes of the instance, such as `opts`, `trj`, `va`, etc.,
        to perform its calculations and updates.

        Returns:
        - None. Directly modifies several attributes of the instance, including
        `trj.boundary_event_rel_raw_stats`, `trj.pre_reward_range`, `trj.reward_range`, and potentially
        others based on the options and data passed to the instance.

        Note:
        - This method is specifically tailored for initial training sessions, focusing on setting up
        and calculating metrics that will be used to analyze the fly's behavior in relation to
        rewards and boundary contacts. It is part of a larger class dedicated to understanding
        spatial learning and behavior in Drosophila through contact event analysis.
        """

        if not hasattr(self.trj, "boundary_event_rel_raw_stats"):
            self.trj.boundary_event_rel_raw_stats = {
                self.evt_name: {self.boundary_orientation: {}}
            }
        if self.evt_name not in self.trj.boundary_event_rel_raw_stats:
            self.trj.boundary_event_rel_raw_stats[self.evt_name] = {
                self.boundary_orientation: {}
            }
        else:
            self.trj.boundary_event_rel_raw_stats[self.evt_name][
                self.boundary_orientation
            ] = {}
        if self.trj.f == 0:
            if not hasattr(self.trj.va, "boundary_event_rel_changes"):
                self.trj.va.boundary_event_rel_changes = {
                    self.evt_name: {self.boundary_orientation: []}
                }
            else:
                if self.evt_name in self.trj.va.boundary_event_rel_changes:
                    self.trj.va.boundary_event_rel_changes[self.evt_name][
                        self.boundary_orientation
                    ] = []
                else:
                    self.trj.va.boundary_event_rel_changes[self.evt_name] = {
                        self.boundary_orientation: []
                    }
        self.reward_range_calculator.calc_reward_range_first_training(self.t)
        if len(self.va.reward_ranges) > self.t.n:
            self.trj.pre_reward_range = self.va.reward_ranges[0]
            self.trj.reward_range = self.va.reward_ranges[1]
        self.N = (
            self.rewards_for_fly[self.t.n - 1][0]
            if len(self.rewards_for_fly[self.t.n - 1]) > 0
            else 0
        )

        if not hasattr(self.va, "reward_ranges"):
            self.va.reward_ranges = [self.trj.pre_reward_range, self.trj.reward_range]

        self.trj.boundary_event_rel_raw_stats[self.evt_name][self.boundary_orientation][
            "N"
        ] = self.N
        self.trj.boundary_event_rel_raw_stats[self.evt_name][self.boundary_orientation][
            "denoms"
        ] = []
        self.trj.boundary_event_rel_raw_stats[self.evt_name][self.boundary_orientation][
            "counts"
        ] = []

    def fillRawStats(self):
        """
        Fills in raw statistical data related to contact events for each training session. This method
        calculates and stores the count of contact events and evaluates whether the session meets
        predefined criteria for inclusion in further analysis. It operates based on the reward ranges
        established for each training session, counting contact events within these ranges and
        determining their adequacy based on the experiment's parameters.

        This method specifically handles:
        - Counting the number of contact events within the pre-determined reward ranges.
        - Calculating the statistics for contact events related to specified boundaries or events
        (e.g., wall contact, agarose contact, or turns) and storing them in the class's data
        structures.
        - Evaluating the sessions based on the number of contact events and the experiment's
        predefined thresholds, marking them as acceptable or not for further analysis.

        The method updates the `boundary_event_rel_raw_stats` with numerical and boolean indicators
        for each session, reflecting the raw counts of events, whether the session met the criteria,
        and the calculated ratios of contact events per reward or other relevant measures.

        Parameters:
        - None directly. Uses the instance's attributes to access necessary data, including the
        trajectory (`trj`), event indices, and options (`opts`).

        Returns:
        - None. Modifies the instance's `boundary_event_rel_raw_stats` and potentially other
        attributes to store calculated statistics and indicators of session validity.

        Note:
        - This method is crucial for preprocessing and preparing data for detailed analysis in later
        stages. It segregates sessions based on their statistical relevance and prepares the ground
        for calculating metrics that offer insights into the flies' learning behavior and interaction
        with the environment.
        - It assumes that reward ranges and other necessary preprocessing steps have been completed.
        """
        reward_ranges = (
            self.va.reward_ranges[:2]
            if self.t.n == 1
            else [self.va.reward_ranges[self.t.n]]
        )

        for i, reward_range in enumerate(reward_ranges):
            rewards = [
                self.va._countOn(
                    reward_range.start,
                    reward_range.stop,
                    calc=True,
                    ctrl=ctrl,
                    f=self.trj.f,
                )
                for ctrl in (False, True)
            ]
            reward_count = np.sum(rewards)
            key_to_save = 0 if self.t.n == 1 and i == 0 else self.t.n

            stats_numer = inRange(
                self.event_indices, reward_range.start, reward_range.stop, count=True
            )
            if "_turn" not in self.evt_name:
                is_ok = rewards[0] >= self.opts.piTh
                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ][key_to_save] = (
                    (stats_numer / rewards[0]) if rewards[0] > 0 else np.nan
                )

                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ]["counts"].append(
                    stats_numer if reward_count > self.opts.piTh else np.nan
                )

                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ]["denoms"].append([rewards[0], is_ok])

            else:
                stats_denom = inRange(
                    self.aux_data["contact_idxs"],
                    reward_range.start,
                    reward_range.stop,
                    count=True,
                )
                is_ok = (
                    stats_denom >= self.opts.turn_contact_thresh
                    and reward_count >= self.opts.piTh
                )
                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ][key_to_save] = (
                    (stats_numer / stats_denom) if stats_denom > 0 else np.nan
                )
                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ]["denoms"].append([stats_denom, is_ok])

    def calcContactMetrics(self):
        """
        Calculates the contact metrics for Drosophila within a spatial learning environment, focusing on
        the relationship between contact events and received rewards across different training sessions.
        This method serves as the core analytical procedure, integrating data prepared by previous steps
        to compute the metrics that quantify learning and behavior in response to the environment.

        The method performs several key operations:
        - Establishes the entries (en) into the reward circle to focus on, filtering the trajectory (trj)
        events that occur after the start of the virtual training session.
        - Calls the appropriate method to calculate the reward range for the first or subsequent
        training sessions, preparing the dataset for detailed analysis.
        - Fills in raw statistical data for contact events within the calculated reward ranges.
        - Computes and stores changes in contact event metrics over time, especially for subsequent
        training sessions, to assess learning and adaptation behavior.
        - Optionally, prepares data for output, organizing and formatting the analysis results for
        further review or reporting.

        This method is critical for transitioning from raw data collection to meaningful insights into
        the flies' behavior, encapsulating the logic needed to assess how contact with different
        boundaries correlates with the pursuit and acquisition of rewards in a controlled environment.

        Parameters:
        - None. Utilizes the class attributes set during initialization and updated through other
        methods to access and manipulate the necessary data.

        Returns:
        - None. This method directly modifies class attributes to store the calculated metrics and
        potentially formats data for output based on the `save_stats` flag.

        Note:
        - This method assumes that initial preprocessing and setup tasks have been completed by
        calling the `calcContactMetricRewardRangeFirstTraining`, `calcContactMetricRewardRangeLaterTraining`,
        and `fillRawStats` methods as appropriate.
        - The effectiveness of this analysis is dependent on the quality and completeness of the input data,
        including the accuracy of event indices and reward counts.
        """

        # Establish the entries (en) into the reward circle to focus on
        self.en = self.trj.en[0][self.trj.en[0] >= self.va.trns[0].start]

        # Calculate the reward range for the first or subsequent training sessions
        if self.t.n == 1:
            self.calcContactMetricRewardRangeFirstTraining()
        else:
            self.calcContactMetricRewardRangeLaterTraining()

        # Fill in raw statistical data for contact events within the calculated reward ranges
        self.fillRawStats()

        # Compute and store changes in contact event metrics over time
        if self.t.n == 2:
            previous_values = [
                self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ]["denoms"][i][1]
                for i in range(1, 3)
            ]
            if (
                False in previous_values
                or self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ][1]
                == 0
            ):
                val = np.nan
            else:
                val = (
                    self.trj.boundary_event_rel_raw_stats[self.evt_name][
                        self.boundary_orientation
                    ][2]
                    - self.trj.boundary_event_rel_raw_stats[self.evt_name][
                        self.boundary_orientation
                    ][1]
                ) / self.trj.boundary_event_rel_raw_stats[self.evt_name][
                    self.boundary_orientation
                ][
                    1
                ]
            self.trj.va.boundary_event_rel_changes[self.evt_name][
                self.boundary_orientation
            ].append(val)

        # Prepare data for output if needed
        if self.t.n == len(self.va.trns) and self.save_stats:
            setattr(self.va, self.va.csv_attr_name, [[]])
            if self.evt_name == "wall_contact":
                wall_orientations = ("all", "lr", "agarose_adj")
            elif self.evt_name in ("agarose_contact", "boundary_contact"):
                wall_orientations = ("tb",)
            elif "_turn" in self.evt_name:
                wall_orientations = ("all",) if "wall" in self.evt_name else ("tb",)

            for wall_orientation in wall_orientations:
                for i in range(0, len(self.va.trns) + 1):
                    for trj in self.va.trx:
                        evt_ratio = trj.boundary_event_rel_raw_stats[self.evt_name][
                            wall_orientation
                        ][i]
                        denom = trj.boundary_event_rel_raw_stats[self.evt_name][
                            wall_orientation
                        ]["denoms"][i]
                        slc = slice(trj.f, None, len(self.va.flies))
                        if i >= 1:
                            rwds_by_bkt = trj.va.numRewardsTot[1][0][slc][i - 1]
                            not_enough_bkts = (
                                len(rwds_by_bkt) == 0
                                if i == 1
                                else len(rwds_by_bkt) < self.n_bkts - 1
                            )
                        else:
                            not_enough_bkts = False

                        if self.evt_name in ("agarose_contact", "boundary_contact"):
                            if not_enough_bkts:
                                getattr(trj.va, self.va.csv_attr_name)[-1].append(
                                    np.nan
                                )
                            else:
                                getattr(trj.va, self.va.csv_attr_name)[-1].append(
                                    trj.boundary_event_rel_raw_stats[self.evt_name][
                                        wall_orientation
                                    ]["counts"][i]
                                )
                        if not_enough_bkts or not denom[1]:
                            evt_ratio = np.nan
                        getattr(trj.va, self.va.csv_attr_name)[-1].append(evt_ratio)

            setattr(
                self.va,
                self.va.csv_attr_name,
                np.array(getattr(trj.va, self.va.csv_attr_name)),
            )
