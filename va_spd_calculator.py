import numpy as np
from trajectory import Trajectory
from video_analysis_interface import VideoAnalysisInterface
from reward_range_calculator import RewardRangeCalculator


class VASpeedCalculator:
    def __init__(self, va: VideoAnalysisInterface, opts):
        """
        Initializes the VASpeedCalculator with an instance of VideoAnalysisInterface.

        Parameters:
            va (VideoAnalysisInterface): an instance of VideoAnalysisInterface
            opts (argparse.Namespace): Configuration options.
        """
        self.va = va
        self.opts = opts
        self.reward_range_calculator = RewardRangeCalculator(self.va, self.opts)
        self.reward_range_calculator.calculate_reward_ranges()

    def calcSpeedsOverSBs(self):
        """
        Calculates the average speeds over specified segments for each Trajectory instance.

        This method iterates over the timeframes given, calculates average speed,
        and stores it in an array format suitable for CSV output.
        """
        all_speeds = []

        # Pre-training (first element of reward_ranges)
        pre_training_range = self.va.reward_ranges[0]
        pre_training_speeds = self._calc_average_speeds(
            pre_training_range.start, pre_training_range.stop, 0
        )
        all_speeds.append(pre_training_speeds)

        # Training and post-training segments
        for t_idx, t in enumerate(self.va.trns):
            reward_range = self.va.reward_ranges[
                t_idx + 1
            ]  # Use t_idx + 1 for training timeframes
            all_speeds.append(
                self._calc_average_speeds(
                    reward_range.start, reward_range.stop, t_idx + 1
                )
            )

            post_training_speeds_1 = self._calc_post_training_speeds(t.stop, 3)
            all_speeds.append(post_training_speeds_1)

            post_training_speeds_2 = self._calc_post_training_speeds(
                t.stop + self.va._min2f(3), 3
            )
            all_speeds.append(post_training_speeds_2)

        # Store the results in the VideoAnalysis instance attribute
        self.va.speeds_over_sbs = [list(np.concatenate(all_speeds))]

    def _calc_average_speeds(self, start_frame, end_frame, index=None):
        """
        Calculates the average speeds between start_frame and end_frame for each Trajectory instance.

        Parameters:
            start_frame (int): The starting frame index.
            end_frame (int): The ending frame index.
            index (int): The index of the timeframe (doesn't apply for post-training periods).

        Returns:
            list: A list of average speeds for each Trajectory.
        """
        if (
            np.isnan(start_frame)
            or np.isnan(end_frame)
            or (index is not None and self.va.pair_exclude[index] is True)
        ):
            return len(self.va.trx) * [np.nan]
        start_frame, end_frame = int(start_frame), int(end_frame)
        speeds = []

        for traj in self.va.trx:
            traj: Trajectory
            if traj.bad():
                speeds.append(np.nan)
                continue

            if self.opts.excl_wall_for_spd:
                boundary_contact = traj.boundary_event_stats["wall"]["all"]["opp_edge"][
                    "boundary_contact"
                ]

                valid_indices = ~boundary_contact[start_frame:end_frame]
                speed_elements = traj.sp[start_frame:end_frame][valid_indices]
            else:
                speed_elements = traj.sp[start_frame:end_frame]

            if len(speed_elements) == 0:
                speeds.append(np.nan)
                continue

            avg_speed = sum(speed_elements) / len(speed_elements) / traj.pxPerMmFloor
            speeds.append(avg_speed)

        return speeds

    def _calc_post_training_speeds(self, start_frame, minutes):
        """
        Calculates the average speeds for a specified number of minutes after training for each Trajectory instance.

        Parameters:
            start_frame (int): The starting frame index.
            minutes (int): The duration in minutes for which to calculate the average speeds.

        Returns:
            list: A list of average speeds for each Trajectory.
        """
        end_frame = start_frame + self.va._min2f(minutes)
        return self._calc_average_speeds(start_frame, end_frame)
