import numpy as np

from boundary_contact import runBndContactAnalysisForCtrReferencePt
from common import CT
from reward_range_calculator import RewardRangeCalculator
from video_analysis_interface import VideoAnalysisInterface


class VATurnProbabilityDistanceCollator:
    def __init__(self, va: VideoAnalysisInterface, opts):
        if va.ct != CT.htl:
            raise NotImplementedError(
                "Only HTL chamber is supported for analysis"
                " of turn probability by distance."
            )
        self.va = va
        self.distances = opts.turn_prob_by_dist
        self.min_vel_angle_delta = opts.min_vel_angle_delta
        self.opts = opts
        self.invert_distances()

    def invert_distances(self):
        """
        Adjusts the reference point for the distances to match the upper and lower edges
        of the chamber, conforming to the expected format for EllipseToBoundaryDistCalculator.
        """
        self.distances_inv = [8 - dist for dist in self.distances]

    def calcTurnProbabilitiesByDistance(self):
        if not hasattr(self.va, "reward_ranges"):
            rr_calc = RewardRangeCalculator(self.va, self.opts)
            rr_calc.calculate_reward_ranges()
        self.va.turn_prob_by_distance = {}
        for i, dist in enumerate(self.distances):
            self.va.turn_prob_by_distance[dist] = []
            for traj in self.va.trx:
                if traj.bad():
                    self.va.turn_prob_by_distance[dist].append(
                        len(self.va.reward_ranges) * [(np.nan, np.nan)]
                    )
                    continue
                self.va.turn_prob_by_distance[dist].append([])
                stats = runBndContactAnalysisForCtrReferencePt(
                    traj, self.va, self.distances_inv[i], self.opts
                )["boundary_event_stats"]
                turn_results = self.va.determineTurnDirectionality(
                    "boundary", "ctr", stats, traj
                )["boundary"]["ctr"]["all"]
                contact_start_data = stats["boundary"]["tb"]["ctr"][
                    "contact_start_idxs"
                ]

                for j, reward_range in enumerate(self.va.reward_ranges):
                    if self.va.pair_exclude[j]:
                        self.va.turn_prob_by_distance[dist][-1].append((np.nan, np.nan))
                        continue
                    num_contact_evts = len(
                        contact_start_data[
                            (contact_start_data >= reward_range.start)
                            & (contact_start_data <= reward_range.stop)
                        ]
                    )
                    if num_contact_evts < self.opts.turn_contact_thresh:
                        self.va.turn_prob_by_distance[dist][-1].append((np.nan, np.nan))
                        continue
                    start = int(reward_range.start)
                    stop = int(reward_range.stop + 1)
                    num_turns_toward = sum(
                        1
                        for frame in range(start, stop)
                        if turn_results.get(frame, False) == True
                    )
                    num_turns_away = sum(
                        1
                        for frame in range(start, stop)
                        if frame in turn_results and turn_results[frame] == False
                    )
                    ratio_toward = num_turns_toward / num_contact_evts
                    ratio_away = num_turns_away / num_contact_evts
                    self.va.turn_prob_by_distance[dist][-1].append(
                        (ratio_toward, ratio_away)
                    )
