import numpy as np
from src.analysis.video_analysis_interface import VideoAnalysisInterface


class RewardRangeCalculator:
    def __init__(self, va: VideoAnalysisInterface, opts):
        """
        Initializes a RewardRangeCalculator instance.

        Args:
        - va (VideoAnalysis): An instance of the VideoAnalysis class.
        - opts (argparse.Namespace): Configuration options.
        """
        self.va = va
        self.rewards_for_fly = self.va.numRewardsTot[1][0][0 :: len(self.va.flies)]
        self.opts = opts
        self.expected_num_ranges = len(self.va.trns) + 1  # Number of trainings plus 1
        self.apply_pi_pre = getattr(self.opts, "apply_pi_pre", False)

    def _nan_equal(self, x, y):
        """
        Check if two values are equal, considering NaN values.

        Args:
        - x: The first value.
        - y: The second value.

        Returns:
        - bool: True if the values are equal or both are NaN, False otherwise.
        """
        try:
            if np.isnan(x) and np.isnan(y):
                return True
            else:
                return x == y
        except TypeError:
            return x == y

    def _slice_exists(self, new_slice, slice_list):
        """
        Checks if a given slice already exists in a list of slices.

        Args:
        - new_slice (slice): The slice to check.
        - slice_list (list): The list of slices.

        Returns:
        - bool: True if the slice exists, False otherwise.
        """
        return any(
            self._nan_equal(s.start, new_slice.start)
            and self._nan_equal(s.stop, new_slice.stop)
            and self._nan_equal(s.step, new_slice.step)
            for s in slice_list
        )

    def _add_unique_range(self, reward_range, is_pre_training=False):
        """
        Adds a reward range to the va.reward_ranges list if it is not already present,
        allowing duplicate NaN reward ranges as long as the total number is below the expected number.

        Args:
        - reward_range (slice): The reward range to add.
        - is_pre_training (bool): Whether this range is for the pre-training period,
          to bypass exclusion logic.
        """
        range_added = False
        if not hasattr(self.va, "reward_ranges"):
            self.va.reward_ranges = [reward_range]
            range_added = True
        else:
            if not self._slice_exists(reward_range, self.va.reward_ranges):
                self.va.reward_ranges.append(reward_range)
                range_added = True
            elif np.isnan(reward_range.start) and np.isnan(reward_range.stop):
                if len(self.va.reward_ranges) < self.expected_num_ranges:
                    self.va.reward_ranges.append(reward_range)
                    range_added = True

        if range_added:
            if not hasattr(self.va, "pair_exclude"):
                self.va.pair_exclude = []
            if is_pre_training and not self.apply_pi_pre:
                # Pre is always kept when the flag is off
                self.va.pair_exclude.append(False)
            else:
                self.va.pair_exclude.append(
                    self._check_paired_threshold_result(
                        reward_range.start, reward_range.stop
                    )
                )

    def _check_paired_threshold_result(self, start_frame, end_frame):
        result = self._apply_threshold_check(start_frame, end_frame, paired=True)
        if np.isnan(result[0]) and np.isnan(result[1]):
            return True
        return False

    def _apply_threshold_check(self, start_frame, end_frame, paired=False):
        """
        Applies the event threshold check to determine if NaNs should be used for the slice.

        Args:
        - start_frame (int): The starting frame index.
        - end_frame (int): The ending frame index.

        Returns:
        - tuple: The potentially modified start and end frames.
        """
        if np.isnan(start_frame) or np.isnan(end_frame):
            return np.nan, np.nan

        if paired:
            for f in self.va.flies:
                onEvents = [
                    self.va._countOn(start_frame, end_frame, calc=True, ctrl=ctrl, f=f)
                    for ctrl in (False, True)
                ]
                if np.sum(onEvents) < self.va.opts.piTh:
                    return np.nan, np.nan
        else:
            onEvents = [
                self.va._countOn(start_frame, end_frame, calc=True, ctrl=ctrl, f=0)
                for ctrl in (False, True)
            ]
            if np.sum(onEvents) < self.va.opts.piTh:
                return np.nan, np.nan

        return start_frame, end_frame

    def calc_reward_range_first_training(self, t):
        """
        Calculates the reward range for the first training session.

        Args:
        - t (Training): The current training session.
        """
        ten_min_in_frames = self.va.fps * 10 * 60
        pre_reward_range = slice(t.start - ten_min_in_frames, t.start)

        # Pre-training slice
        pre_start, pre_end = pre_reward_range.start, pre_reward_range.stop
        if self.apply_pi_pre:
            pre_start, pre_end = self._apply_threshold_check(
                pre_start, pre_end, paired=True
            )

        # Training 1 slice
        start_index = self.va.buckets[0][0]
        end_index = self.va.buckets[0][1]
        training_start, training_end = self._apply_threshold_check(
            start_index, end_index
        )

        self._add_unique_range(slice(pre_start, pre_end), is_pre_training=True)
        self._add_unique_range(
            slice(training_start, training_end), is_pre_training=False
        )

    def calc_reward_range_later_training(self, t):
        """
        Calculates the reward range for later training sessions.

        Args:
        - t (Training): The current training session.
        """
        buckets = self.va.buckets[t.n - 1]
        finite_buckets = buckets[~np.isnan(buckets)]
        start_index = buckets[-3] if len(finite_buckets) >= 2 else np.nan
        end_index = buckets[-2] if len(finite_buckets) >= 1 else np.nan
        start_index, end_index = self._apply_threshold_check(start_index, end_index)
        reward_range = slice(start_index, end_index)

        self._add_unique_range(reward_range)

    def calculate_reward_ranges(self):
        """
        Main method to calculate reward ranges based on each training session.
        """
        df = self.va._min2f(self.opts.syncBucketLenMin)
        for t in self.va.trns:
            self.n_bkts = self.va._syncBucket(t, df)
            if t.n == 1:
                self.calc_reward_range_first_training(t)
            else:
                self.calc_reward_range_later_training(t)
