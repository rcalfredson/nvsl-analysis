import numpy as np
from src.analysis.video_analysis_interface import VideoAnalysisInterface


class VATurnDirectionalityCollator:
    def __init__(self, va: VideoAnalysisInterface, opts):
        self.va = va
        self.opts = opts

    def calcTurnDirectionalityMetrics(self):
        self.va.turn_dir_metrics = {}
        boundary_tps = getattr(self.opts, "turn")
        for boundary_tp in boundary_tps:
            if boundary_tp == "wall":
                continue
            self.va.turn_dir_metrics[boundary_tp] = {}
            for ellipse_ref_pt in ("ctr", "edge"):
                direction_ratios = self.calculate_direction_ratios(
                    boundary_tp, ellipse_ref_pt
                )
                self.va.turn_dir_metrics[boundary_tp][ellipse_ref_pt] = direction_ratios

    def calculate_direction_ratios(self, boundary_tp, ellipse_ref_pt):
        direction_ratios = []
        turn_tps = ["all"]
        if ellipse_ref_pt == "edge":
            turn_tps.extend(["inside", "outside"])

        for turn_tp in turn_tps:
            if not hasattr(self.va, "reward_ranges"):
                direction_ratios.extend(
                    len(self.va.trx) * (len(self.va.trns) + 1) * [np.nan]
                )
                continue
            for reward_range in self.va.reward_ranges:
                for trj in self.va.trx:
                    if trj.bad():
                        direction_ratios.extend([np.nan, np.nan])
                        continue
                    reward_count = len(
                        trj.en[False][
                            (trj.en[False] >= reward_range.start)
                            & (trj.en[False] < reward_range.stop)
                        ]
                    )
                    if reward_count < self.opts.piTh:
                        direction_ratios.extend([np.nan, np.nan])
                        continue

                    frame_data, num_contact_evts = self.get_frame_and_contact_data(
                        trj, boundary_tp, ellipse_ref_pt, reward_range, turn_tp
                    )

                    if num_contact_evts == 0:
                        direction_ratios.extend([np.nan, np.nan])
                        continue

                    num_turns_toward, num_turns_away = self.count_turns(frame_data)
                    direction_ratios.append(num_turns_toward / num_contact_evts)
                    direction_ratios.append(num_turns_away / num_contact_evts)

        return direction_ratios

    def get_frame_and_contact_data(
        self, trj, boundary_tp, ellipse_ref_pt, reward_range, turn_tp
    ):
        frame_data = trj.boundary_event_stats[boundary_tp]["tb"][ellipse_ref_pt][
            "turn_direction_toward_ctr"
        ][turn_tp]
        contact_start_data = trj.boundary_event_stats[boundary_tp]["tb"][
            ellipse_ref_pt
        ]["contact_start_idxs"]
        num_contact_evts = len(
            contact_start_data[
                (contact_start_data >= reward_range.start)
                & (contact_start_data <= reward_range.stop)
            ]
        )

        frame_data = {
            frame: data
            for frame, data in frame_data.items()
            if reward_range.start <= frame <= reward_range.stop
        }
        return frame_data, num_contact_evts

    def count_turns(self, frame_data):
        num_turns_toward = sum(1 for _, toward in frame_data.items() if toward)
        num_turns_away = sum(1 for _, toward in frame_data.items() if not toward)
        return num_turns_toward, num_turns_away
