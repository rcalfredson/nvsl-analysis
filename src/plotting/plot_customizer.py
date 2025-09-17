import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as mtransforms


class PlotCustomizer:
    """
    A class to customize the appearance of matplotlib plots, including font size and family.

    Attributes:
        font_size_default (float): The default font size from matplotlib's current rcParams.
        font_size (float): The current font size used in plots.
        font_family_default (str): The default font family from matplotlib's current rcParams.
        font_size_customized (bool): Indicates whether the font size has been customized.
        font_family_customized (bool): Indicates whether the font family has been customized.
        text_objects (list): A list of text objects for further customization.
        in_plot_font_size (float): A font size specifically used for in-plot text, such as
                                   legends and labels.
        increase_factor (float): A factor representing how much the current font size has
                                 increased relative to the default font size. This is used
                                 to adjust plot padding proportionally. Initialized to 1,
                                 indicating no increase.
    """

    def __init__(self, in_plot_font_size=None):
        """
        Initializes the PlotCustomizer object with optional in-plot font size customization.

        Parameters:
            in_plot_font_size (float, optional): The font size to be used for in-plot text.
                                                 If None, defaults to 3 less than the current
                                                 matplotlib font size.
        """
        self.font_size_default = plt.rcParams["font.size"]
        self.font_size = self.font_size_default
        self.font_family_default = self._get_font_family()
        self.font_size_customized = False
        self.font_family_customized = False
        self.text_objects = []
        self.in_plot_font_size = (
            self.font_size - 3 if in_plot_font_size is None else in_plot_font_size
        )
        self.increase_factor = 1

    def _get_font_family(self):
        """
        Retrieves the default font family from matplotlib's configuration.

        Returns:
            str: The default font family.
        """
        return (
            plt.rcParams["font.family"][0]
            if isinstance(plt.rcParams["font.family"], list)
            else plt.rcParams["font.family"]
        )

    def update_font_size(self, new_font_size):
        """
        Updates the font size for various plot components and recalculates the in-plot font size.

        Parameters:
            new_font_size (float): The new font size to apply to the plot.
        """
        self.in_plot_font_size = new_font_size - 3
        plt.rc("axes", titlesize=new_font_size + 3)
        plt.rc("axes", labelsize=new_font_size + 2)
        plt.rc("xtick", labelsize=new_font_size - 2)
        plt.rc("ytick", labelsize=new_font_size - 2)
        plt.rc("figure", titlesize=new_font_size)
        plt.rc("legend", fontsize=self.in_plot_font_size)

        if self.font_size != new_font_size:
            self.font_size_customized = True
        self.font_size = new_font_size
        self.increase_factor = self.font_size / self.font_size_default

    def update_font_family(self, new_font_family):
        """
        Updates the font family used in the plot if the new font family is different from the default.

        Parameters:
            new_font_family (str): The new font family to use in the plot.
        """
        if new_font_family and new_font_family != self.font_family_default:
            plt.rcParams.update({"font.family": new_font_family})
            self.font_family_customized = True

    @property
    def customized(self):
        """
        Checks if either font size or font family has been customized.

        Returns:
            bool: True if either font size or font family has been customized, False otherwise.
        """
        return self.font_family_customized or self.font_size_customized

    @property
    def font_size_diff(self):
        """
        Calculates the difference between the current font size and the default font size.

        Returns:
            float: The difference in font size.
        """
        return self.font_size - self.font_size_default

    def adjust_aspect_ratio(self, ax, target_aspect=1):
        """
        Adjusts the aspect ratio of a subplot based on its axis limits to ensure
        the plot rectangles look consistent.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object to adjust.
            target_aspect (float): The desired aspect ratio of the plot rectangle.
        """
        ax.set_box_aspect(target_aspect)

    def adjust_padding_proportionally(
        self, aspect_ratio=0.75, wspace=0.08, base_hspace=0.35
    ):
        """
        Adjusts the figure size and subplot padding proportionally to the font size.
        Enlarges the figure instead of shrinking the axes area when fonts are big.
        Ensures X tick spacing <= 10 and Y tick spacing <= 20.
        Also inserts newlines into oversized text boxes and axis labels.

        Parameters
        ----------
        aspect_ratio : float
            Target box aspect ratio for each subplot.
        wspace : float
            Horizontal spacing between subplots, as a fraction of axis width.
        base_hspace : float
            Baseline vertical spacing between subplot rows.
        """
        fig = plt.gcf()

        # --- Step 1: Scale figure size based on font size increase (scaled down by 2)
        w, h = fig.get_size_inches()
        effective_factor_x = 1 + 0.6 * (self.increase_factor - 1)
        effective_factor_y = effective_factor_x
        new_w = w * effective_factor_x / 2
        new_h = h * effective_factor_y / 2

        if new_w > w or new_h > h:
            fig.set_size_inches(new_w, new_h, forward=True)

        # --- Step 2: Ensure X-axis tick spacing is not greater than 10
        for ax in fig.get_axes():
            xlim = ax.get_xlim()
            xticks = ax.get_xticks()
            if len(xticks) >= 2:
                spacing = xticks[1] - xticks[0]
                if spacing > 10:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
                    ax.set_xlim(left=xlim[0])

        # --- Step 3: Ensure Y-axis tick spacing is set proportionally based on font size
        for ax in fig.get_axes():
            _, height = (
                ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).size
            )
            fontsize = ax.yaxis.get_label().get_fontsize()

            # Rough heuristic: one label needs about 1.5 x fontsize in points of vertical space
            label_height_inches = 1.5 * fontsize / 72
            max_ticks = max(2, int(height / label_height_inches))

            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))

        # --- Step 4: Add newlines for long texts and axis labels ---
        font_threshold = 20

        def split_evenly(s: str) -> str:
            """Split string into two roughly even parts by word count."""
            words = s.split()
            if len(words) < 4:
                return s
            mid = len(words) // 2
            left, right = words[:mid], words[mid:]
            if len(left) < 2 or len(right) < 2:
                return s
            return " ".join(left) + "\n" + " ".join(right)

        def split_at_colon(s: str) -> str:
            """Split at the last colon, but only if both sides have ≥2 words."""
            if ":" not in s:
                return s
            head, tail = s.rsplit(":", 1)
            head_words, tail_words = head.split(), tail.split()
            if len(head_words) < 2 or len(tail_words) < 2:
                return s
            return head + ":\n  " + tail.strip()

        for ax in fig.get_axes():
            for label in [ax.xaxis.get_label(), ax.yaxis.get_label()]:
                if label.get_text() and label.get_fontsize() > font_threshold:
                    s = label.get_text()
                    if "\n" not in s:
                        label.set_text(split_evenly(s))

        # Text boxes other than titles and tick labels
        for txt in fig.findobj(match=plt.Text):
            if txt in [ax.title for ax in fig.get_axes()] + [
                lab
                for ax in fig.get_axes()
                for lab in ax.get_xticklabels() + ax.get_yticklabels()
            ]:
                continue

            if txt.get_fontsize() > font_threshold and ":" in txt.get_text():
                new_s = split_at_colon(txt.get_text())
                if new_s != txt.get_text():
                    x, y = txt.get_position()
                    txt.set_text(new_s)
                    txt.set_ha("left")
                    txt.set_va("top")
                    txt.set_position((x, y))

        # --- Step 5: Shared axis label logic for large fonts ---
        axes = fig.get_axes()
        if not axes:
            return

        large_font = any(
            ax.yaxis.get_label().get_fontsize() >= font_threshold for ax in axes
        )

        if large_font:
            # --- Step 5a: Single Y label (use leftmost subplot) ---
            left_ax = min(axes, key=lambda ax: ax.get_position().x0)
            shared_y_label = next(
                (
                    ax.yaxis.get_label().get_text()
                    for ax in axes
                    if ax.yaxis.get_label().get_text()
                ),
                "",
            )
            fontsize = left_ax.yaxis.get_label().get_fontsize()

            # Hide all native labels excluding Y tick labels for leftmost subplot
            for ax in axes:
                ax.yaxis.label.set_visible(False)
                if ax is not left_ax:
                    ax.set_yticklabels([])

            if shared_y_label:
                pad_pts = 2.75 * fontsize + 15
                trans = left_ax.transAxes + mtransforms.ScaledTranslation(
                    -pad_pts / 72.0, 0.0, fig.dpi_scale_trans
                )
                fig.text(
                    0.0,
                    0.5,
                    shared_y_label,
                    transform=trans,
                    rotation="vertical",
                    va="center",
                    ha="center",
                    fontsize=fontsize,
                )

            # --- Step 5b: Reduce horizontal spacing if shared Y label is used ---
            wspace *= 0

            # --- Step 5c: Single X label (centered) ---
            shared_x_label = next(
                (
                    ax.xaxis.get_label().get_text()
                    for ax in axes
                    if ax.xaxis.get_label().get_text()
                ),
                "",
            )
            if shared_x_label:
                for ax in axes:
                    ax.xaxis.label.set_visible(False)
                fig.text(
                    0.5,
                    0.0,
                    shared_x_label,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                )

        # --- Step 6: Normalize subplot box aspect ---
        for ax in fig.get_axes():
            ax.set_box_aspect(aspect_ratio)

        # --- Step 7: Apply subplot spacing ---
        fig.subplots_adjust(left=0.12, right=0.88, wspace=wspace, hspace=base_hspace)
