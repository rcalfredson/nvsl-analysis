import textwrap
import warnings

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import matplotlib.transforms as mtransforms
import numpy as np


FIXED_Y_MAX_INTERVALS = 6
FIXED_Y_MAX_DECIMALS = 3

LEGEND_HANDLE_MAX_POINTS = 30.0
LEGEND_HANDLE_TEXT_PAD_MAX_POINTS = 5.0
LEGEND_BORDER_PAD_MAX_POINTS = 4.0
LEGEND_BORDER_AXES_PAD_MAX_POINTS = 2.0
LARGE_FONT_LEGEND_MIN_POINTS = 18.0
LEGEND_BORDER_AXES_PAD_RELAXED_POINTS = 6.0


def compact_legend_spacing(font_size, *, handlelength=None):
    """Keep legend chrome from growing excessively with large fonts.

    Matplotlib expresses legend spacing in multiples of the legend font size.
    Preserve the configured/default spacing at ordinary sizes, but cap each
    dimension at a readable physical size for large-font plots.
    """
    font_size_points = float(
        FontProperties(size=font_size).get_size_in_points()
    )
    if not np.isfinite(font_size_points) or font_size_points <= 0:
        raise ValueError("legend font size must be positive and finite")

    base_handlelength = float(
        plt.rcParams["legend.handlelength"]
        if handlelength is None
        else handlelength
    )

    if font_size_points < LARGE_FONT_LEGEND_MIN_POINTS:
        return {
            "handlelength": base_handlelength,
            "handletextpad": float(plt.rcParams["legend.handletextpad"]),
            "borderpad": float(plt.rcParams["legend.borderpad"]),
            "borderaxespad": float(plt.rcParams["legend.borderaxespad"]),
        }

    def _capped(rc_key, max_points):
        return min(float(plt.rcParams[rc_key]), max_points / font_size_points)

    return {
        "handlelength": min(
            base_handlelength,
            LEGEND_HANDLE_MAX_POINTS / font_size_points,
        ),
        "handletextpad": _capped(
            "legend.handletextpad", LEGEND_HANDLE_TEXT_PAD_MAX_POINTS
        ),
        "borderpad": _capped("legend.borderpad", LEGEND_BORDER_PAD_MAX_POINTS),
        "borderaxespad": _capped(
            "legend.borderaxespad", LEGEND_BORDER_AXES_PAD_MAX_POINTS
        ),
    }


def apply_adaptive_legend_axes_edge_inset(ax, legend):
    """Relax a large-font legend's axes-edge inset when its width permits.

    The available horizontal slack is split between the two sides conceptually,
    then clamped to a 2--6 point range. Short legends therefore retain a
    comfortable inset, while legends that nearly fill (or exceed) an axes use
    the tight inset needed to maximize usable width.

    Returns the chosen physical inset in points.
    """
    font_size_points = float(legend.prop.get_size_in_points())
    if not np.isfinite(font_size_points) or font_size_points <= 0:
        raise ValueError("legend font size must be positive and finite")

    if font_size_points < LARGE_FONT_LEGEND_MIN_POINTS:
        return float(legend.borderaxespad) * font_size_points

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_width_px = float(ax.get_window_extent(renderer=renderer).width)
    legend_width_px = float(legend.get_window_extent(renderer=renderer).width)
    slack_points = (axes_width_px - legend_width_px) * 72.0 / float(fig.dpi)
    inset_points = float(
        np.clip(
            0.5 * slack_points,
            LEGEND_BORDER_AXES_PAD_MAX_POINTS,
            LEGEND_BORDER_AXES_PAD_RELAXED_POINTS,
        )
    )
    legend.borderaxespad = inset_points / font_size_points
    return inset_points


def _faithful_tick_precision(ticks, max_precision):
    ticks = np.asarray(ticks, dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    if ticks.size == 0:
        return 1

    unique_ticks = np.unique(ticks)
    spacings = np.diff(unique_ticks)
    positive_spacings = spacings[spacings > 0]
    spacing = float(np.min(positive_spacings)) if positive_spacings.size else 1.0
    magnitude = max(1.0, float(np.max(np.abs(unique_ticks))))
    tolerance = max(
        spacing * 1e-8,
        float(np.spacing(magnitude)) * 4.0,
    )

    for precision in range(max_precision + 1):
        rounded = np.asarray(
            [float(f"{tick:.{precision}f}") for tick in unique_ticks]
        )
        if np.all(np.abs(rounded - unique_ticks) <= tolerance):
            return precision
    return None


def _tick_decimal_precision(ticks, max_precision=8):
    """Return fixed-point precision that faithfully represents tick positions."""
    precision = _faithful_tick_precision(ticks, max_precision)
    if precision is not None:
        return precision
    return max_precision


def _fixed_endpoint_ticks(
    limits,
    max_intervals=FIXED_Y_MAX_INTERVALS,
    max_precision=FIXED_Y_MAX_DECIMALS,
):
    """Return an exact, endpoint-anchored grid within a precision budget."""
    lower, upper = (float(value) for value in limits)
    max_intervals = max(1, int(max_intervals))
    max_precision = max(0, int(max_precision))
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        raise ValueError("fixed axis limits must be finite and increasing")

    nice_mantissas = np.asarray([1.0, 2.0, 2.5, 5.0, 10.0])
    endpoint_precision = _faithful_tick_precision((lower, upper), 8)
    precision_budget = max(
        max_precision,
        8 if endpoint_precision is None else endpoint_precision,
    )

    # Prefer conventional decimal steps, then accept any exactly displayable
    # grid. In both passes, reduce density instead of rounding labels away
    # from their true coordinates.
    for require_nice_step in (True, False):
        for interval_count in range(max_intervals, 0, -1):
            step = span / interval_count
            exponent = np.floor(np.log10(step))
            mantissa = step / (10.0**exponent)
            nice_step = np.any(
                np.isclose(mantissa, nice_mantissas, rtol=1e-10, atol=1e-12)
            )
            if require_nice_step and not nice_step:
                continue

            ticks = np.linspace(lower, upper, interval_count + 1)
            ticks[0], ticks[-1] = lower, upper
            ticks[np.abs(ticks) <= span * 1e-12] = 0.0
            if _faithful_tick_precision(ticks, precision_budget) is not None:
                return ticks

    # A single interval always preserves the requested endpoints. This is
    # reachable only for limits requiring more than eight decimal places.
    return np.asarray([lower, upper], dtype=float)


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

    @staticmethod
    def _max_y_tick_intervals(ax):
        _, height = (
            ax.get_window_extent().transformed(
                ax.figure.dpi_scale_trans.inverted()
            ).size
        )
        fontsize = ax.yaxis.get_label().get_fontsize()
        label_height_inches = 1.5 * fontsize / 72
        return max(2, int(height / label_height_inches))

    @staticmethod
    def _set_adaptive_y_tick_formatter(ax, ticks):
        if all(abs(tick - round(tick)) < 1e-8 for tick in ticks):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(x)}")
            )
            return

        precision = _tick_decimal_precision(ticks)

        def _adaptive_fmt(x, _, precision=precision):
            return f"{x:.{precision}f}"

        ax.yaxis.set_major_formatter(FuncFormatter(_adaptive_fmt))

    def set_fixed_y_axes(
        self,
        axes,
        limits,
        *,
        max_intervals=FIXED_Y_MAX_INTERVALS,
        max_precision=FIXED_Y_MAX_DECIMALS,
    ):
        """Set deterministic fixed limits, showing ticks on the left column."""
        axes = list(axes)
        if not axes:
            return

        endpoint_precision = _faithful_tick_precision(limits, 8)
        if endpoint_precision is None or endpoint_precision > max_precision:
            required = ">8" if endpoint_precision is None else str(endpoint_precision)
            warnings.warn(
                "fixed y-axis endpoints require "
                f"{required} decimal places; exceeding the configured "
                f"{max_precision}-decimal tick-label budget to preserve them",
                UserWarning,
                stacklevel=2,
            )

        ticks = _fixed_endpoint_ticks(
            limits,
            max_intervals=max_intervals,
            max_precision=max_precision,
        )
        left_edge = min(float(ax.get_position().x0) for ax in axes)
        left_axes = {
            ax
            for ax in axes
            if np.isclose(float(ax.get_position().x0), left_edge, atol=1e-8)
        }

        for ax in axes:
            ax.set_ylim(*limits)
            ax.set_yticks(ticks)
            self._set_adaptive_y_tick_formatter(ax, ticks)
            show_ticks = ax in left_axes
            ax.tick_params(
                axis="y",
                which="both",
                left=show_ticks,
                labelleft=show_ticks,
                right=False,
                labelright=False,
            )
            # Setting ticks may expand an axis if floating-point noise puts an
            # endpoint microscopically outside it, so restore the limits last.
            ax.set_ylim(*limits)

    def set_fixed_y_axis(self, ax, limits, **kwargs):
        """Compatibility wrapper for a single fixed-y axis."""
        self.set_fixed_y_axes([ax], limits, **kwargs)

    def adjust_padding_proportionally(
        self,
        aspect_ratio=0.75,
        wspace=0.08,
        base_hspace=0.35,
        wrap_legend_labels=True,
        wrap_axis_labels=True,
        wrap_x_axis_labels=None,
        wrap_y_axis_labels=None,
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
        wrap_axis_labels : bool
            Whether to insert newlines into long axis labels at larger font sizes.
        wrap_x_axis_labels : bool
            Whether to insert newlines into long X-axis labels. Defaults to
            wrap_axis_labels.
        wrap_y_axis_labels : bool
            Whether to insert newlines into long Y-axis labels. Defaults to
            wrap_axis_labels.
        """
        fig = plt.gcf()
        if wrap_x_axis_labels is None:
            wrap_x_axis_labels = wrap_axis_labels
        if wrap_y_axis_labels is None:
            wrap_y_axis_labels = wrap_axis_labels

        # --- Step 1: Optionally wrap long legend labels instead of shrinking legend font ---
        renderer = None
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None

        def _wrap_legend_text(text_obj, max_width_px):
            raw = text_obj.get_text()
            if (
                renderer is None
                or not raw
                or "\n" in raw
                or max_width_px is None
                or max_width_px <= 0
            ):
                return

            bbox = text_obj.get_window_extent(renderer=renderer)
            if bbox.width <= max_width_px:
                return

            words = raw.split()
            if len(words) < 2:
                return

            wrapped = raw
            for width in range(len(raw), 1, -1):
                candidate = textwrap.fill(
                    raw,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                text_obj.set_text(candidate)
                candidate_bbox = text_obj.get_window_extent(renderer=renderer)
                if candidate_bbox.width <= max_width_px:
                    wrapped = candidate
                    break

            text_obj.set_text(wrapped)

        if wrap_legend_labels:
            for ax in fig.get_axes():
                leg = ax.get_legend()
                if leg is not None:
                    ax_bbox = ax.get_window_extent(renderer=renderer)
                    if ax_bbox.width <= 0:
                        continue

                    anchor = leg.get_bbox_to_anchor()
                    outside_right = False
                    if anchor is not None:
                        anchor_box = anchor.transformed(fig.transFigure.inverted())
                        outside_right = anchor_box.x0 >= 0.95 * ax.get_position().x1

                    if outside_right:
                        fig_px_width = fig.get_window_extent(renderer=renderer).width
                        max_width_px = max(fig_px_width * 0.26, ax_bbox.width * 0.22)
                    else:
                        max_width_px = ax_bbox.width * 0.45

                    for text in leg.get_texts():
                        _wrap_legend_text(text, max_width_px)

        # --- Step 2: Scale figure size only modestly as fonts increase.
        # Keep most of the visual effect in the text itself instead of making
        # the whole canvas grow in near-lockstep, which would preserve the
        # text-to-plot ratio when the image is viewed "fit to window".
        w, h = fig.get_size_inches()
        effective_factor_x = min(1 + 0.18 * (self.increase_factor - 1), 1.25)
        effective_factor_y = min(1 + 0.12 * (self.increase_factor - 1), 1.18)
        new_w = w * effective_factor_x
        new_h = h * effective_factor_y

        if new_w > w or new_h > h:
            fig.set_size_inches(new_w, new_h, forward=True)

        # --- Step 3: Ensure automatic X-axis tick spacing is not greater than 10.
        # Explicit tick grids encode data positions (for example, sync-bucket
        # endpoints), so they must not be replaced during large-font layout.
        for ax in fig.get_axes():
            if isinstance(ax.xaxis.get_major_locator(), FixedLocator):
                continue
            xlim = ax.get_xlim()
            xticks = ax.get_xticks()
            if len(xticks) >= 2:
                spacing = xticks[1] - xticks[0]
                if spacing > 10:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
                    ax.set_xlim(left=xlim[0])

        # --- Step 4: Ensure Y-axis tick spacing is set proportionally based on font size
        for ax in fig.get_axes():
            max_ticks = self._max_y_tick_intervals(ax)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))

            yticks = ax.get_yticks()
            if len(yticks) > 0:
                # Preserve the actual tick coordinates. Checking only for
                # unique labels can misrepresent 0.15-spaced ticks as
                # alternating 0.1 and 0.2 increments after rounding.
                self._set_adaptive_y_tick_formatter(ax, yticks)

        # --- Step 5: Add newlines for long axis labels ---
        if wrap_x_axis_labels or wrap_y_axis_labels:
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

            for ax in fig.get_axes():
                labels = []
                if wrap_x_axis_labels:
                    labels.append(ax.xaxis.get_label())
                if wrap_y_axis_labels:
                    labels.append(ax.yaxis.get_label())
                for label in labels:
                    if label.get_text() and label.get_fontsize() > font_threshold:
                        s = label.get_text()
                        if "\n" not in s:
                            label.set_text(split_evenly(s))

        # --- Step 6: Shared axis label logic when font size exceeds the default ---
        axes = fig.get_axes()
        if not axes:
            return

        font_larger_than_default = self.font_size > self.font_size_default

        if font_larger_than_default and len(axes) > 1:
            # --- Step 6a: Single Y label (use leftmost subplot) ---
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

            # --- Step 6b: Single X label (centered) ---
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

        # --- Step 7: Normalize subplot box aspect ---
        for ax in fig.get_axes():
            ax.set_box_aspect(aspect_ratio)

        # --- Step 8: Apply subplot spacing ---
        # When the figure is enlarged for bigger fonts, keep subplot gaps from
        # growing too much by scaling wspace back down proportionally.
        scaled_wspace = wspace / max(effective_factor_x, 1.0)
        fig.subplots_adjust(
            left=0.12,
            right=0.88,
            wspace=scaled_wspace,
            hspace=base_hspace,
        )
