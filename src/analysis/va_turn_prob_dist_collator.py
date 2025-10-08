import numpy as np

from src.analysis.boundary_contact import runBndContactAnalysisForCtrReferencePt
from src.analysis.circle_contact import runCircleContactAnalysisUsingOutside
from src.utils.common import CT
from src.plotting.event_chain_plotter import EventChainPlotter
from src.analysis.reward_range_calculator import RewardRangeCalculator
from src.analysis.video_analysis_interface import VideoAnalysisInterface


class VATurnProbabilityDistanceCollator:
    def __init__(self, va: VideoAnalysisInterface, opts):
        if va.ct not in (CT.htl, CT.large, CT.large2):
            raise NotImplementedError(
                "Only HTL and large chambers are supported for analysis"
                " of turn probability by distance."
            )
        self.va = va
        self.geom = getattr(opts, "contact_geometry", "horizontal")
        if self.geom == "circular":
            self.distances = opts.outside_circle_radii
        else:
            self.distances = opts.turn_prob_by_dist
            self.invert_distances()
        self.min_vel_angle_delta = opts.min_vel_angle_delta
        self.opts = opts

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
                if self.geom == "horizontal":
                    be_stats = runBndContactAnalysisForCtrReferencePt(
                        traj, self.va, self.distances_inv[i], self.opts
                    )["boundary_event_stats"]
                    btp, bcombo, ref_pt = "boundary", "tb", "ctr"
                else:  # circular
                    be_stats = runCircleContactAnalysisUsingOutside(
                        traj, self.va, self.distances[i], self.opts
                    )
                    btp, bcombo, ref_pt = "circle", "ctr", "ctr"
                    radius_stats = be_stats[btp][bcombo][ref_pt]

                    if self.opts.bnd_ct_plots:
                        plotter = EventChainPlotter(
                            traj, self.va, image_format=self.opts.imageFormat
                        )

                        if self.opts.bnd_ct_plots == "troubleshooting":
                            raise NotImplementedError(
                                "Circle-based sharp turn troubleshooting plots are not implemented yet."
                            )
                        for trn_index, trn in enumerate(self.va.trns):
                            bcr_all = radius_stats["boundary_contact_regions"]
                            turning_all = radius_stats["turning_indices"]

                            bcr_filtered = [
                                ev
                                for ev in bcr_all
                                if ev.start >= trn.start and ev.stop <= trn.stop
                            ]
                            turning_filtered = [
                                new_idx
                                for new_idx, ev in enumerate(bcr_filtered)
                                if ev in [bcr_all[idx] for idx in turning_all]
                            ]

                            if not turning_filtered:
                                continue

                            rs_filtered = {
                                **radius_stats,
                                "boundary_contact_regions": bcr_filtered,
                                "turning_indices": turning_filtered,
                                "circle_radius_mm": self.opts.outside_circle_radii[i],
                            }

                            plotter.plot_sharp_turn_chain_circle(
                                radius_stats=rs_filtered,
                                trn_index=trn_index,
                                start_frame=self.opts.bnd_ct_plot_start_fm,
                                mode=self.opts.bnd_ct_plot_mode,
                            )

                turn_results = self.va.determineTurnDirectionality(
                    btp, ref_pt, ext_data=be_stats, trj=traj, boundary_combo=bcombo
                )[btp]["ctr"]["all"]
                contact_start_data = be_stats[btp][bcombo][ref_pt]["contact_start_idxs"]

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
