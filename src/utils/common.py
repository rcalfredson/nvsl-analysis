# -*- coding: utf-8 -*-
#
# common for real-time tracker and analysis
#
# 12 Mar 2019 by Ulrich Stern
#

# standard libraries
import enum
import math
import os

# third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# custom modules and constants
from src.utils.constants import P
from src.utils.numba_loader import jit

import src.utils.util as util
from src.utils.util import error

WELCH = True  # whether to use Welch's t-test

# - - -


# numerical utilities
def is_nan(value):
    return np.isnan(value) or (value == "nan")


def propagate_nans(array):
    # Ensure the input is a NumPy array
    array = np.array(array)

    # Determine the size of the inner dimension
    inner_size = array.shape[-1]
    assert inner_size % 2 == 0, "The inner dimension size must be even."

    half_size = inner_size // 2

    # Iterate through the video and stages
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            first_set = array[i, j, :half_size]
            second_set = array[i, j, half_size:]

            # Find where NaNs are in the first and second sets
            first_set_nans = np.isnan(first_set)
            second_set_nans = np.isnan(second_set)

            # Propagate NaNs to the counterpart elements
            second_set[first_set_nans] = np.nan
            first_set[second_set_nans] = np.nan

            # Assign the updated sets back to the array
            array[i, j, :half_size] = first_set
            array[i, j, half_size:] = second_set

    return array


# dict utilities
def deep_access(x, attr, keylist, new_val=None):
    val = getattr(x, attr)
    keylist = keylist.split(".")
    for i, key in enumerate(keylist):
        if new_val is not None:
            if i == len(keylist) - 1:
                val[key] = new_val
                continue
            if i > len(keylist) - 1:
                continue
        val = val[key]
    return val


# chamber type
class CT(enum.Enum):
    regular = dict(ppmm=8.4, center=(45, 70))
    htl = dict(ppmm=8.0, nc=5, name="HtL", floor=((0, 0), (80, 128)), center=(40, 64))
    large = dict(ppmm=7.0, nc=2, floor=((0, 0), (244, 244)), center=(122, 122))

    # returns chamber type given the number of flies per camera
    @staticmethod
    def get(numFlies):
        return {2: CT.regular, 4: CT.large, 20: CT.htl}[numFlies]

    def __str__(self):
        s = self.value.get("name")
        return self.name if s is None else s

    # returns number of pixels per mm at floor of chamber
    def pxPerMmFloor(self):
        return self.value["ppmm"]

    # returns number of chambers per row (in camera view)
    def numCols(self):
        return self.value.get("nc")

    # returns geometric specs of the wells of the two floor styles of the large or
    # HTL chamber.
    #
    # Arguments:
    #   1) xf: instance of Xformer class
    #   2) f:  fly number (zero-indexed, row-dominant)
    #   3) tp: floor style
    #          'A'- agarose wells (for either large or HTL)
    #          'F'- pits on flat floor (for large only)
    #   4) c:  camera number (1 or 2); for flat chamber only
    #   5) wellLength: length of the well in mm (for HTL only)
    #                  Note: if a value other than the default
    #                  is used, then this represents a "virtual"
    #                  well that doesn't correspond with a
    #                  physically-defined boundary.
    #
    # Return format for circular wells:
    #   - well radius (float)
    #   - coordinates of each well, e.g.,
    #       tuple((well1_x, well1_y), (well2_x, well2_y), ...)
    #
    # Return format for rectangular wells:
    #   - coordinates of upper-left and lower-right corners of well, e.g.,
    #       tuple((well1_x1, well1_y1, well1_x2, well1_y2), ...)
    def arenaWells(self, xf, f, tp="A", c=1, wellLength=24):
        if self not in (CT.large, CT.htl):
            error("only implemented for large or HTL chamber")
        args = (xf, f) + (
            (
                "A",
                c,
            )
            if self is CT.large
            else (wellLength,)
        )
        return {CT.large: self._arenaWellsLarge, CT.htl: self._arenaWellsHTL}[self](
            *args
        )

    # returns agarose wells for HTL chamber (see arenaWells)
    def _arenaWellsHTL(self, xf, f, wellLength=24):
        flr = list(self.floor(xf, f=f))
        upperWell = (flr[0][0], flr[0][1], flr[1][0], flr[0][1] + wellLength)
        lowerWell = (flr[0][0], flr[1][1] - wellLength, flr[1][0], flr[1][1])
        return (upperWell, lowerWell)

    # returns agarose wells for large chamber (see arenaWells)
    def _arenaWellsLarge(self, xf, f, tp="A", c=1):
        wellRadius = (4 if tp == "A" else 2) * self.pxPerMmFloor()
        flrCrds = [
            (crd[0], crd[1] + 284 * (1 if f > 1 else 0))
            for crd in self.floor(xf, np.mod(f, 2))
        ]
        xHalf, yHalf = np.mean(flrCrds, axis=0)
        if tp == "A":
            wellCoords = (
                (flrCrds[0][0] + wellRadius, yHalf),
                (xHalf, flrCrds[0][1] + wellRadius),
                (flrCrds[1][0] - wellRadius, yHalf),
                (xHalf, flrCrds[1][1] - wellRadius),
            )
        else:
            if c == 1:
                wellCoords = () if f < 2 else ((xHalf, yHalf),)
            else:
                sideGap = 12 * self.pxPerMmFloor()
                diagonalPits = (
                    (flrCrds[0][0] + sideGap, flrCrds[0][1] + sideGap),
                    (flrCrds[1][0] - sideGap, flrCrds[1][1] - sideGap),
                )
                wellCoords = diagonalPits + (
                    ()
                    if f < 2
                    else ((flrCrds[1][0] - sideGap, flrCrds[0][1] + sideGap),)
                )
        return wellRadius, wellCoords

    # returns tl and br frame coordinates for chamber floor for the given fly;
    #  if no fly is given or for regular chamber, tl and br for the full frame
    #  is returned
    def floor(self, xf, f=None):
        if f is None or self is CT.regular:
            return (0, 0), xf.frameSize
        elif self in (CT.htl, CT.large):
            return (xf.t2f(*xy, f=f, noMirr=True) for xy in self.value["floor"])
        else:
            error("not yet implemented")

    # sets (floor) width and height
    def __getattr__(self, name):
        if name in ("width", "height"):
            self.width, self.height = (self.value["floor"][1][i] for i in (0, 1))
            return getattr(self, name)
        raise AttributeError(name)

    # returns frame coordinates of the center of the chamber floor
    def center(self):
        return self.value["center"]

    @staticmethod
    def _test():
        for ct in CT:
            tlBr = ct.value.get("floor")
            if tlBr:
                util.test(ct.center, [], np.array(tlBr).mean(axis=0))


# - - -


# coordinate transformer between template and frame coordinates
# * for large and high-throughput chambers, template coordinates are for the
#  top left chamber with 0 representing the top left corner of the floor
class Xformer:
    # tm: dictionary with template match values (keys: fctr, x, and y)
    def __init__(self, tm, ct, frame, fy):
        self.ct, self.nc, self.frameSize = ct, ct.numCols(), util.imgSize(frame)
        if tm is not None:
            self.init(tm)
        self.fy = fy

    # called explicitly if template match values calculated (the values were not
    #  saved by early versions of rt-trx)
    def init(self, tm):
        rf = tm.get("rszFctr", 1.5 if self.frameSize[1] == 1080 else 1)
        self.fctr, self.x, self.y = rf * tm["fctr"], tm["x"], tm["y"]

    def initialized(self):
        return hasattr(self, "x")

    # shifts template coordinates between top left (TL) and the given fly's
    #  chambers
    def _shift(self, xy, f, toTL):
        if self.ct in (CT.htl, CT.large) and f is not None:
            r, c = divmod(f, self.nc)
            tf = util.tupleSub if toTL else util.tupleAdd
            if self.ct is CT.htl:
                xy = tf(xy, (144 * c + 5, [4, 180, 355, 531][r]))
            else:
                xy = tf(xy, (284 * c + 4, 284 * r + 4))
        return xy

    # mirrors template coordinates, with 0 representing top left corner of floor
    def _mirror(self, xy, f, noMirr=False):
        if self.ct in (CT.htl, CT.large) and f is not None:
            r, c = divmod(f, self.nc)
            x, y = xy
            if self.ct is CT.htl:
                xy = (
                    x if c < 3 or noMirr else self.ct.width - x,
                    y if r < 2 or not self.fy or noMirr else self.ct.height - y,
                )
            else:
                xy = (x if c == 0 or noMirr else self.ct.width - x, y)
        return xy

    # xforms template coordinates to int-rounded frame coordinates, possibly
    #  changing coordinates for top left chamber (with 0 representing top left
    #  corner of floor) into coordinates for the given fly (used for placing
    #  circles, etc.)
    def t2f(self, x, y, f=None, noMirr=False):
        xy = self._shift(self._mirror((x, y), f, noMirr), f, toTL=False)
        return util.intR(util.tupleAdd(util.tupleMul(xy, self.fctr), (self.x, self.y)))

    # convenience functions
    def t2fX(self, x, f=None):
        return self.t2f(x, 0, f)[0]

    def t2fY(self, y, f=None):
        return self.t2f(0, y, f)[1]

    def t2fR(self, r):
        return util.intR(r * self.fctr)

    # xforms frame coordinates to template coordinates; x and y can be ndarrays
    def f2t(self, x, y, f=None, noMirr=False):
        xy = (x - self.x) / self.fctr, (y - self.y) / self.fctr
        return self._mirror(self._shift(xy, f, toTL=True), f, noMirr=noMirr)


# - - - output formatting
def cVsA(calc, ctrl=False, abb=True):
    return (
        ("ctrl." if abb else "__control__")
        if ctrl
        else (("calc." if abb else "__calculated__") if calc else "actual")
    )


def cVsA_l(calc, ctrl=False):
    return cVsA(calc, ctrl, False)


def flyDesc(f):
    return "yok" if f else "exp"


def frame2hm(nf, fps):
    h = nf / fps / 3600
    return "%.1fh" % h if h >= 1 else "%s min" % util.formatFloat(h * 60, 1)


def pch(a, b):
    return a if P else b


def skipMsg(mins):
    return "(first %s min of each bucket skipped)" % util.formatFloat(mins, 1)


# - - - signal processing
def filterDataAndCalcDiffs(data_to_flt):
    flt = {}
    diff = {}

    # Filtering and smoothing data
    for k in ("theta", "x", "y"):
        if data_to_flt[k] is not None:
            flt[k] = util.butter(data_to_flt[k])

    # Calculating smoothed distances and differences
    flt["dist"] = util.distances((flt["x"], flt["y"]))
    flt["diff"] = {k: np.diff(flt[k]) for k in ["x", "y"]}
    acc_magnitude_sm = np.sqrt(flt["diff"]["x"] ** 2 + flt["diff"]["y"] ** 2)

    # Calculating differences for unsmoothed data
    if data_to_flt["theta"] is not None:
        diff["theta"] = util.angleDiff(
            data_to_flt["theta"][1:], data_to_flt["theta"][:-1], absVal=False
        )

    # Velocity angles for both smoothed and unsmoothed data
    vel_angles = np.zeros(len(data_to_flt["x"]))
    vel_angles_sm = np.zeros(len(data_to_flt["x"]))
    vel_angles[:-1] = util.velocityAngles(
        np.array([data_to_flt["x"], data_to_flt["y"]]).T
    )
    vel_angles_sm[:-1] = util.velocityAngles(np.array([flt["x"], flt["y"]]).T)

    # Acceleration angles for both smoothed and unsmoothed data
    acc_angles = np.zeros(len(data_to_flt["x"]))
    acc_angles_sm = np.zeros(len(data_to_flt["x"]))
    acc_angles[:-2] = util.velocityAngles(
        np.array([np.diff(data_to_flt["x"]), np.diff(data_to_flt["y"])]).T
    )
    acc_angles_sm[:-2] = util.velocityAngles(
        np.array([flt["diff"]["x"], flt["diff"]["y"]]).T
    )

    # Adjusting acceleration angles for proper alignment
    acc_angles = np.roll(acc_angles, 1)
    acc_angles_sm = np.roll(acc_angles_sm, 1)

    return {
        "flt": flt,
        "diff": diff,
        "velAngles": vel_angles,
        "velAnglesSm": vel_angles_sm,
        "accMag": acc_magnitude_sm,
        "accMagSm": acc_magnitude_sm,
        "accAngles": acc_angles,
        "accAnglesSm": acc_angles_sm,
    }


@jit(nopython=True)
def resolveAngles180To360(theta, x, y, start_index, min_jump):
    # define several parameters using default values from Ctrax
    max_velocity_angle_weight = 0.25
    velocity_angle_weight = 0.05
    theta_360 = np.asarray(theta)
    N = len(theta)
    state_prev = np.zeros((N - 1, 2), dtype=np.uint8)
    tmp_cost = np.zeros(2)
    cost_prev_new = np.zeros(2)
    cost_prev = np.zeros(2)
    angle_flip_range = list(range(2))
    rad_to_deg_fctr = 180 / np.pi

    for tloc in range(1, N):
        if tloc < start_index + 1:
            continue
        x_curr = x[tloc]
        y_curr = y[tloc]
        x_prev = x[tloc - 1]
        y_prev = y[tloc - 1]
        v_x = x_curr - x_prev
        v_y = y_prev - y_curr
        d_center = math.sqrt(v_x**2 + v_y**2)
        velocity_angle = 90 - rad_to_deg_fctr * np.arctan2(v_y, v_x)
        if d_center >= min_jump:
            w = 0
        else:
            angle_wt = velocity_angle_weight * d_center
            if angle_wt < max_velocity_angle_weight:
                w = angle_wt
            else:
                w = max_velocity_angle_weight
        w_compl = 1 - w
        for s_curr in angle_flip_range:
            theta_curr = theta_360[tloc] + s_curr * 180
            for s_prev in angle_flip_range:
                theta_prev = theta_360[tloc - 1] + s_prev * 180
                cost_curr = w_compl * util.angleDiff(
                    theta_prev, theta_curr, absVal=True
                ) + w * util.angleDiff(theta_curr, velocity_angle, absVal=True)
                tmp_cost[s_prev] = cost_prev[s_prev] + cost_curr
            if tmp_cost[0] < tmp_cost[1]:
                s_prev = 0
            else:
                s_prev = 1
            state_prev[tloc - 1, s_curr] = s_prev
            cost_prev_new[s_curr] = tmp_cost[s_prev]
        cost_prev = cost_prev_new
    if cost_prev[0] < cost_prev[1]:
        s_curr = 0
    else:
        s_curr = 1
    if s_curr == 1:
        theta_360[N - 1] += 180
        theta_360[tloc] = util.normAngles(theta_360[tloc], useRadians=False)

    for tloc in range(N - 2, -1, -1):
        s_curr = state_prev[tloc, s_curr]
        if s_curr == 1:
            theta_360[tloc] += 180
            theta_360[tloc] = util.normAngles(theta_360[tloc], useRadians=False)
    return theta_360


# - - - stats


# returns t, p, na, nb
def ttest_rel(a, b, msg=None, min_n=2, conf_int=False, silent=False):
    return ttest(a, b, True, msg, min_n, conf_int, silent)


def ttest_ind(a, b, msg=None, min_n=2, conf_int=False, silent=False):
    return ttest(a, b, False, msg, min_n, conf_int, silent)


def ttest(a, b, paired, msg=None, min_n=2, conf_int=False, silent=False):
    if paired:
        abFinite = np.isfinite(a) & np.isfinite(b)
    a, b = (x[abFinite if paired else np.isfinite(x)] for x in (a, b))
    na, nb = len(a), len(b)
    if min(na, nb) < min_n:
        return np.nan, np.nan, na, nb, None
    with np.errstate(all="ignore"):
        t, p = st.ttest_rel(a, b) if paired else st.ttest_ind(a, b, equal_var=not WELCH)
    if msg:
        means = []
        for vals in (a, b):
            mci = util.meanConfInt(vals, asDelta=True)
            means.append("%.3g" % mci[0])
            if conf_int:
                means[-1] += " ±%.2f" % mci[1]
        msg = "%spaired t-test -- %s:" % ("" if paired else "un", msg)
        msg += "\n  n = %s means: %s, %s; t-test: p = %.5f, t = %.3f" % (
            "%d," % na if paired else "%d, %d;" % (na, nb),
            means[0],
            means[1],
            p,
            t,
        )
        if not silent:
            print(msg)
    return t, p, na, nb, msg


# returns t, p, na
def ttest_1samp(a, val, msg=None, min_n=2):
    a = a[np.isfinite(a)]
    na = len(a)
    if na < min_n:
        return np.nan, np.nan, na
    with np.errstate(all="ignore"):
        t, p = st.ttest_1samp(a, val)
    if msg:
        print("one-sample t-test -- %s:" % msg)
        print(
            "  n = %d, mean: %.3g, value: %.1g; t-test: p = %.5f, t = %.3f"
            % (na, np.mean(a), val, p, t)
        )
    return t, p, na


# - - - file and image


def writeImage(fn, img=None, format="png"):
    file_extension = f".{format}"
    base, ext = os.path.splitext(fn)

    if ext.lower() != file_extension.lower():
        fn = base + file_extension
        print(
            f"The file extension has been changed to {file_extension} to coincide with the specified format."
        )

    print(f"writing {fn}...")

    if img is None:
        plt.savefig(fn, bbox_inches="tight", format=format)
    else:
        cv2.imwrite(fn, img)


def adjustLegend(legend, axs, all_line_vals, legend_loc="best"):
    """
    Increases Y axis limit based on detected overlaps with other objects in the plot.

    Parameters:
    - legend: The legend object to adjust.
    - axs: The array of axes objects to adjust.
    - all_line_vals: List of line values to check for overlaps.
    """
    if legend is None:
        return
    kwargs = {}
    prop_dict = {"style": "italic"}
    kwargs["prop"] = prop_dict
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()

    overlap_detected = True

    while overlap_detected:
        overlap_detected = False
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(
            plt.gca().transAxes.inverted()
        )
        plt.sca(axs[0, 0])
        ax = plt.gca()
        ymin_data = ax.transAxes.transform((0, legend_bbox.ymin))[1]
        ymax_data = ax.transAxes.transform((0, legend_bbox.ymax))[1]

        ymin_data = ax.transData.inverted().transform((0, ymin_data))[1]
        ymax_data = ax.transData.inverted().transform((0, ymax_data))[1]

        current_ylim = ax.get_ylim()

        # Calculate the increment size based on Y-axis tick size
        ticks = ax.get_yticks()
        tick_size = (
            ticks[1] - ticks[0]
            if len(ticks) > 1
            else (current_ylim[1] - current_ylim[0]) * 0.1
        )

        for line_vals in all_line_vals:
            for val in line_vals:
                if ymin_data * 0.8 <= val <= ymax_data * 1.2:
                    overlap_detected = True
                    for i in range(axs.shape[1]):
                        plt.sca(axs[0, i])
                        ax = plt.gca()
                        current_ylim = ax.get_ylim()
                        ax.set_ylim(current_ylim[0], current_ylim[1] + tick_size)
                if overlap_detected:
                    plt.sca(axs[0, 0])
                    ax = plt.gca()
                    legend = plt.legend(loc=legend_loc, **kwargs)
                    break
            if overlap_detected:
                break


def draw_text_with_bg(
    img, text, org, fontFace, fontScale, color, thickness, lineType, bgColor, alpha=0.6
):
    (text_w, text_h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

    # Create a transparent overlay of the same size as the original image
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (org[0] - 5, org[1] - text_h - 5),
        (org[0] + text_w + 5, org[1] + 5),
        bgColor,
        -1,
    )

    # Blend the original image and the overlay to create a semi-transparent effect
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add the text on top
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)


# - - -

if __name__ == "__main__":
    print("testing")
    CT._test()
    print("done")
