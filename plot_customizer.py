import matplotlib.pyplot as plt


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
        plt.rc("axes", titlesize=new_font_size)
        plt.rc("axes", labelsize=new_font_size)
        plt.rc("xtick", labelsize=new_font_size - 2)
        plt.rc("ytick", labelsize=new_font_size - 2)
        plt.rc("figure", titlesize=new_font_size)
        plt.rc("legend", fontsize=self.in_plot_font_size)

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

    def adjust_padding_proportionally(self, aspect_ratio=0.75):
        """
        Adjusts the plot padding proportionally and ensures a consistent aspect ratio for a given font size.
        """
        fig = plt.gcf()
        for ax in fig.get_axes():
            self.adjust_aspect_ratio(ax, aspect_ratio)

        padding_increase = (self.increase_factor - 1) * 0.05

        current_wspace = plt.rcParams["figure.subplot.wspace"]
        current_hspace = plt.rcParams["figure.subplot.hspace"]

        plt.subplots_adjust(
            wspace=current_wspace + padding_increase,
            hspace=current_hspace + padding_increase,
        )
