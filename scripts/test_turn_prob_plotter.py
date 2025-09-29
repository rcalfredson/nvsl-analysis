import numpy as np
import os

from analyze import TurnProbabilityByDistancePlotter
from src.plotting.plot_customizer import PlotCustomizer


def test_turn_probability_two_groups():
    """
    Contrived test harness for TurnProbabilityByDistancePlotter.
    Creates fake data for 2 groups, exp vs yoked, across distances
    and timeframes. Values are noisy but systematic so plots look realistic.
    """

    rng = np.random.default_rng()
    customizer = PlotCustomizer()
    customizer.update_font_size(30)
    customizer.update_font_family("Arial")

    # --- Define fake distances & timeframes ---
    distances = [2, 4, 6, 8, 10]  # mm
    timeframes = ["pre_trn", "t1_start", "t2_end", "t3_end"]

    # --- Dummy VA object with required attributes ---
    class DummyVA:
        def __init__(self, gidx):
            self.gidx = gidx
            self.turn_prob_by_distance = {}

            # Define systematic baselines per group
            if gidx == 0:  # Group A (control-like)
                # base_curve = np.linspace(0.25, 0.45, len(distances))
                base_curve = np.linspace(0.45, 0.25, len(distances))
            else:  # Group B (PFNd-like)
                # base_curve = np.linspace(0.15, 0.35, len(distances))
                base_curve = np.linspace(0.35, 0.15, len(distances))

            for d_idx, d in enumerate(distances):
                # Experimental = follow baseline + small noise
                exp_vals = [
                    [
                        base_curve[d_idx] + rng.normal(0, 0.03),  # toward
                        base_curve[d_idx] + rng.normal(0, 0.03),
                    ]  # away
                    for _ in timeframes
                ]
                # Control = slightly shifted baseline
                ctrl_vals = [
                    [
                        base_curve[d_idx] + rng.normal(0, 0.03) - 0.05,  # toward
                        base_curve[d_idx] + rng.normal(0, 0.03) - 0.05,
                    ]  # away
                    for _ in timeframes
                ]

                self.turn_prob_by_distance[d] = [exp_vals, ctrl_vals]

    # --- Construct fake VA instances for 2 groups ---
    gls = ["Group A", "Group B"]
    va_instances = []
    for gidx in range(len(gls)):
        for _ in range(10):  # 10 flies per group
            va_instances.append(DummyVA(gidx))

    # --- Minimal opts stub ---
    class DummyOpts:
        def __init__(self):
            self.use_union_filter = False
            self.contact_geometry = "circular"
            self.imageFormat = "png"

    opts = DummyOpts()

    # --- Instantiate plotter & run ---
    plotter = TurnProbabilityByDistancePlotter(
        va_instances=va_instances,
        gls=gls,
        opts=opts,
    )
    plotter.average_turn_probabilities()
    plotter.plot_turn_probabilities()

    print("Plots generated in imgs/turn_probability_*.png")


if __name__ == "__main__":
    test_turn_probability_two_groups()
