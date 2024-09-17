import matplotlib.pyplot as plt
import numpy as np

from common import writeImage
from util import slugify


class OutsideCircleDurationPlotter:
    PATCH_PALETTE = ["tomato", "forestgreen"]
    EDGE_PALETTE = ["darkred", "darkgreen"]
    OUTSIDE_CIRCLE_NBINS = 15  # Number of bins for histogram plotting
    MIN_NUM_DURATIONS = (
        10  # Minimum number of durations required for inclusion in plots
    )

    def __init__(self, vas, opts, gls):
        """
        Initializes the OutsideCircleDurationPlotter instance for plotting outside-circle durations.

        Parameters:
        - vas (list[VideoAnalysis]): A list of VideoAnalysis instances, each containing
          trajectory data (`trx`) and a group index (`gidx`) for the flies observed.
        - opts (argparse.Namespace): Parsed command-line options from the analyze script,
          containing various configuration settings for the analysis.
        - gls (list[str] or None): Group labels specifying the names of each group involved in
          the analysis. If `None`, groups will not be distinguished by label.

        Attributes:
        - vas (list[VideoAnalysis]): The provided VideoAnalysis instances for processing.
        - opts (argparse.Namespace): The command-line options used to control the analysis parameters.
        - gls (list[str] or None): The group labels for differentiating between groups in the analysis.
        - n_bins (int): The number of bins to use for histogram plotting.
        - bins (numpy.ndarray): An array representing the bin edges for histogram calculations.
        - num_plot_failures (int): Counter for the number of plots skipped due to insufficient data.
        """
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.n_bins = self.OUTSIDE_CIRCLE_NBINS
        self.bins = np.linspace(0, 60, self.n_bins + 1)
        self.num_plot_failures = 0

    def calculate_histograms(self, durations, normalized=True):
        """
        Calculates histograms for outside-circle durations.

        Parameters:
        - durations (list[float]): A list of durations for a specific fly, radius, and training session.
        - normalized (bool): Whether to normalize the histogram by bin width.

        Returns:
        - numpy.ndarray: The histogram of outside-circle durations.
        """
        hist, bin_edges = np.histogram(durations, bins=self.bins)
        if normalized:
            bin_widths = np.diff(bin_edges)
            hist = hist / bin_widths / np.sum(hist)
        return hist

    def plot_distributions(self, normalized=True):
        """
        Plots averaged distributions of outside-circle durations for each radius and training period.

        Parameters:
        - normalized (bool): Whether to plot normalized histograms.
        """
        radii_mm = self.vas[0].trx[0].opts.outside_circle_radii  # Radii from options

        for radius in radii_mm:
            self.plot_for_radius_and_training(
                radius, 0, is_pre_training=True, normalized=normalized
            )
            for trn_idx in range(1, len(self.vas[0].trns) + 1):
                self.plot_for_radius_and_training(
                    radius, trn_idx, is_pre_training=False, normalized=normalized
                )

    def plot_for_radius_and_training(
        self, radius, trn_idx, is_pre_training=False, normalized=True
    ):
        """
        Plots the averaged distribution of outside-circle durations for a specific radius and training period.

        Parameters:
        - radius (float): The radius being analyzed.
        - trn_idx (int): Index of the training period being analyzed.
        - is_pre_training (bool): Flag indicating if the data is from pre-training period.
        - normalized (bool): Whether to plot normalized histograms.
        """
        histograms_by_group = {0: [], 1: []}  # 0 for experimental, 1 for control
        is_multi_group = self.gls is not None and len(self.gls) > 1

        for va in self.vas:
            for trj in va.trx:
                if trj.bad() or (is_multi_group and trj.f == 1):
                    continue
                group = va.gidx if is_multi_group else trj.f
                durations_idx = 0 if is_pre_training else trn_idx

                if radius not in trj.outside_durations[durations_idx]:
                    continue

                durations = trj.outside_durations[durations_idx][radius]

                if len(durations) < self.MIN_NUM_DURATIONS:
                    continue

                hist = self.calculate_histograms(durations, normalized=normalized)
                histograms_by_group[group].append(hist)

        mean_distributions = {
            group: (
                np.mean(histograms, axis=0)
                if histograms
                else np.zeros(len(self.bins) - 1)
            )
            for group, histograms in histograms_by_group.items()
        }

        self.draw_histogram(
            mean_distributions,
            radius,
            trn_idx,
            histograms_by_group,
            is_multi_group,
            is_pre_training,
            normalized,
        )

    def draw_histogram(
        self,
        mean_distributions,
        radius,
        trn_idx,
        histograms_by_group,
        is_multi_group,
        is_pre_training,
        normalized,
    ):
        """
        Draws the histogram for the averaged distributions of outside-circle durations.

        Parameters:
        - mean_distributions (dict): Dictionary containing averaged histograms for each group.
        - radius (float): The radius being analyzed.
        - trn_idx (int): Index of the training period being analyzed.
        - histograms_by_group (dict): Dictionary containing histograms for each fly by group.
        - is_multi_group (bool): Flag indicating if the analysis is a multi-group comparison.
        - is_pre_training (bool): Flag indicating if the data is from pre-training period.
        - normalized (bool): Whether to plot normalized histograms.
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        colors = [
            (self.PATCH_PALETTE[0], self.EDGE_PALETTE[0]),
            (self.PATCH_PALETTE[1], self.EDGE_PALETTE[1]),
        ]
        labels = self.gls if is_multi_group else ["Experimental", "Control"]

        for group in range(2):
            ax.bar(
                self.bins[:-1],
                mean_distributions[group],
                width=np.diff(self.bins),
                align="edge",
                color=colors[group][0],
                alpha=0.5,
                label=f"{labels[group]} (n={len(histograms_by_group[group])})",
                edgecolor=colors[group][1],
                linewidth=0.5,
            )

        ax.set_xlabel("Duration Outside Circle (seconds)")
        ax.set_ylabel("Frequency" if normalized else "Count")
        title_period = "Pre-Training" if is_pre_training else f"Training {trn_idx}"
        ax.set_title(
            f"Outside-Circle Durations for Radius {radius} mm and {title_period} ({'Normalized' if normalized else 'Absolute'})"
        )
        ax.legend()
        # Determine if group labels should be included in the filename
        group_label_part = ""
        if is_multi_group and self.gls:
            group_label_part = "_".join(slugify(label) for label in self.gls)

        # Generate filename
        normalized_str = "normalized" if normalized else "absolute"
        pre_training_str = "pre_training" if is_pre_training else f"training_{trn_idx}"
        filename = (
            f"outside_circle_duration_radius_{radius}mm_"
            f"{pre_training_str}_{normalized_str}_{group_label_part}.png"
        )
        filename = f"imgs/{slugify(filename)}"

        writeImage(filename)

        plt.close(fig)
