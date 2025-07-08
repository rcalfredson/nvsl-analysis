# standard libraries
import math

# third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# custom modules and constants
from src.analysis.boundary_contact import EllipseToBoundaryDistCalculator
from src.utils.common import CT, Xformer, cVsA, flyDesc, writeImage
from src.utils.common_cython import resolveAngles180To360
from src.utils.constants import BORDER_WIDTH, _RDP_PKG
from src.analysis.training import Training
from src.analysis.video_analysis_interface import VideoAnalysisInterface

import src.utils.util as util
from src.utils.util import COL_B, COL_BK, COL_G, COL_O, COL_R, COL_W, COL_Y, COL_Y_D
from src.utils.util import error

CALC_REWARDS_IMG_FILE = "imgs/%s_rewards_fly_%d.png"


class Trajectory:
    """
    A class to represent the trajectory of a fly, including its movements, angles,
    speeds, and interactions with boundaries over time.

    Attributes:
        x, y (np.array): Arrays representing the x and y coordinates of the fly's trajectory.
        w, h, theta (tuple of np.array or None): Width, height, and angle of the fly at each point, if available.
        f (int): Indicates the type of fly (0 for experimental, 1 for yoked control).
        va: VideoAnalysis instance, used for detailed analysis if provided.
        ts (np.array): Timestamps for each point in the trajectory.
        opts (object): Options for processing and analyzing the trajectory.
        _bad (bool): Indicates if the trajectory is considered invalid or problematic.

    Parameters:
        xy (tuple of np.array): Tuple of numpy arrays for x and y coordinates.
        opts (argparse.Namespace): Configuration options for trajectory processing.
        wht (tuple of np.array, optional): Tuple containing width, height, and theta arrays. Defaults to None.
        f (int, optional): Fly type indicator (0 for experimental, 1 for yoked control). Defaults to 0.
        va (VideoAnalysis, optional): VideoAnalysis instance to use for detailed analysis. Defaults to None.
        ts (np.array, optional): Timestamps for each coordinate. Defaults to None.
    """

    JMP_LEN_TH, DIST_TH = 30, 10
    SUSP_FRAC_TH, SUSP_NUM_TH = 0.03, 3
    VEL_A_MIN_D = 3
    _DEBUG = False

    def __init__(self, xy, opts, wht=None, f=0, va=None, ts=None):
        """
        Initializes the Trajectory object with coordinates, dimensions, and analysis options.
        Processes the trajectory data by interpolating missing points, calculating distances,
        speeds, areas, identifying suspicious jumps, and calculating rewards.
        Optionally, plots issues and chooses orientations based on options provided.
        """
        self.x, self.y = xy
        (self.w, self.h, self.theta) = wht if wht else 3 * (None,)
        self.f, self.va, self.ts = f, va, ts
        self._p("%s fly" % flyDesc(f))
        self.opts = opts
        if self._isEmpty():
            return
        self._setLostFrameThreshold()
        self._interpolate()
        if self._bad:
            return
        self._calcDistances()
        if self.va:
            self._calcSpeeds()
            self._calcAreas()
            self._setWalking()
            self._setOnBottom()
        self._suspiciousJumps()
        self._calcRewards()
        if opts.showTrackIssues:
            self._plotIssues()
        if opts.chooseOrientations:
            self._chooseOrientations()

    def _setLostFrameThreshold(self):
        """
        Sets the threshold for the maximum proportion of lost frames in the video,
        beyond which the trajectory is marked as "bad".

        By default, the threshold is 0.1. For large-chamber videos—where tracking is
        more likely to lose flies—a higher threshold of 0.5 is used.

        Side effects:
            - Sets the `_max_proportion_lost_fms` attribute based on the chamber type.

        Returns:
            None
        """
        self._max_proportion_lost_fms = 0.1
        if self.va and self.va.ct is CT.large2:
            self._max_proportion_lost_fms = 0.6

    def _p(self, s):
        """
        Prints a message if the VideoAnalysis instance is available.

        Parameters:
            s (str): The message to print.
        """
        if self.va:
            print(s)

    def _isEmpty(self):
        """
        Checks if the trajectory is mostly empty (i.e., contains mostly NaN values).

        Returns:
            bool: True if the trajectory is considered empty, False otherwise. Sets the `_bad`
            attribute to True if the trajectory is empty.
        """
        if np.count_nonzero(np.isnan(self.x)) > 0.99 * len(self.x):
            self._p("  no trajectory")
            self._bad = True
            return True
        return False

    def _interpolate(self):
        """
        Interpolates missing points in the trajectory where the fly's position is unknown (NaN values).
        This method first identifies regions with missing data and then uses linear interpolation
        to estimate the positions within these regions. It also adjusts for sequences where data
        loss occurs at the beginning of the trajectory and marks the trajectory as bad if the
        proportion of missing data exceeds a certain threshold.

        Side effects:
            - Updates the object's position arrays (x, y, theta) with interpolated values.
            - Sets the `_bad` attribute to True if the proportion of missing data is too high.
            - Adjusts the `trxStartIdx` to account for initial missing data.
        """
        self.nan = np.isnan(self.x)
        self.nanrs = nanrs = util.trueRegions(self.nan)
        self.trxStartIdx = 0
        if len(nanrs) and nanrs[0].start == 0:
            self.trxStartIdx = nanrs[0].stop
            del nanrs[0]
        ls = [r.stop - r.start for r in nanrs]
        lf = sum(ls) / (len(self.x) - self.trxStartIdx)
        self._bad = (
            self.va and lf > self._max_proportion_lost_fms and not self.va.openLoop
        )
        self._p(
            "  lost: number frames: %d (%s)%s"
            % (
                sum(ls),
                "{:.2%}".format(lf),
                (
                    ""
                    if not ls
                    else (
                        " *** bad ***"
                        if self._bad
                        else ", sequence length: avg: %.1f, max: %d"
                        % (np.mean(ls), max(ls))
                    )
                ),
            )
        )
        if self._bad:
            return

        # lost during "on"
        if self.va:
            if self.va.on.size > 0:
                msk, nfon = np.zeros_like(self.x, bool), 2
                for d in range(nfon):
                    on = self.va.on + 1 + d
                    if on.size > 0:
                        msk[on[on < len(msk)]] = True
                nf, nl = np.sum(msk), np.sum(msk & self.nan)
                if nf:
                    print(
                        '    during "on" (%d frames, %d per "on" cmd): %d (%s)'
                        % (nf, nfon, nl, "{:.2%}".format(nl / nf))
                    )
            else:
                print('    No "on" frames detected.')

        self._p("    interpolating...")
        if self.theta is not None:
            self._transformAngles()
        for r in nanrs:
            f, t = r.start, r.stop
            assert f > 0
            for a in [a for a in (self.x, self.y, self.theta) if a is not None]:
                a[r] = np.interp(
                    list(range(f, t)),
                    [f - 1, t],
                    [a[f - 1], a[t] if t < len(a) else a[f - 1]],
                )

    def receiveDataByKey(self, data):
        """
        Updates the Trajectory object's attributes based on a dictionary of new data.

        Parameters:
            data (dict): A dictionary where keys correspond to attribute names of the Trajectory
            object and values are the new data to be assigned to these attributes.

        Side effects:
            - Updates the attributes of the Trajectory object with the values provided in `data`.
        """
        for k in data:
            setattr(self, k, data[k])

    # note: self.d is forward-looking (i.e., between current and next position),
    #  self.dBw is backward-looking
    def _calcDistances(self):
        """
        Calculates the distances between consecutive points in the trajectory. It computes both
        forward-looking distances (from current to next position) and backward-looking distances
        (from previous to current position). The method also calculates the mean and standard
        deviation of these distances, excluding NaN values, and updates the trajectory data
        accordingly.

        Side effects:
            - Updates the `d` attribute with forward-looking distances.
            - Updates the `dBw` attribute with backward-looking distances.
            - Computes and stores the mean (`mean_d`) and standard deviation (`std_d`) of the
              forward-looking distances.
        """
        self.d = np.full_like(self.x, np.nan, dtype=np.float64)
        self.d[:-1] = util.distances((self.x, self.y))
        self.mean_d, self.std_d = np.nanmean(self.d), np.nanstd(self.d)
        self.d[np.isnan(self.d)] = 0
        self.dBw = np.zeros_like(self.x)
        self.dBw[1:] = self.d[:-1]

    def _calcSpeeds(self):
        """
        Calculates the speeds of the fly at each point in the trajectory based on the backward-looking
        distances and the frame rate of the video analysis interface.

        Side effects:
            - Updates the `sp` attribute with calculated speeds for each point in the trajectory.
        """
        self.sp = self.dBw * self.va.fps

    def _calcAreas(self):
        """
        Calculates the areas of the fly's body at each point in the trajectory assuming an elliptical
        body shape. This calculation is not interpolated and directly uses the width and height
        attributes.

        Side effects:
            - Updates the `ar` attribute with calculated areas for each point in the trajectory.
        """
        self.ar = self.w * self.h * np.pi / 4

    def _setWalking(self):
        """
        Determines whether the fly is walking based on a speed threshold. The threshold is set to
        2 mm/s, converted to pixels per second using the pixel per millimeter floor scale from the
        video analysis interface.

        Side effects:
            - Updates the `walking` attribute with a boolean array indicating whether the fly is
              considered to be walking at each point in the trajectory.
        """
        self.pxPerMmFloor = self.va.ct.pxPerMmFloor()
        self.walking = self.sp > 2 * self.pxPerMmFloor  # 2 mm/s * px_per_mm
        # note: could write code to automaticalliy fix c4__2015-09-16__10-15-38.avi
        #  problem (fly resting and tracking problem makes it look like back and
        #  forth jumps).  E.g., resting, movement, flag all immediately following
        #  (resting allowed) movements that match or reverse match this movement
        #  (and the movement itself)

    # notes:
    # * False if data missing
    # * constants for both versions of "on bottom" calculation determined using
    #  _playAnnotatedVideo(), see yanglab Wiki
    # * onBottomPre: current and preceding frames are "on bottom" ("matching"
    #  self.sp)
    # TODO
    # * exclude sidewall for HtL chamber
    # * rename onBottom -> onFloor
    def _setOnBottom(self):
        """
        Determines whether the fly is "on the bottom" of the arena based on its position, area, and
        movements. The criteria for being "on bottom" vary by chamber type (CT) and are based on
        specific constants and conditions detailed in the method's implementation.

        Side effects:
            - Updates the `onBottom` attribute with a boolean array indicating whether the fly is
              considered to be on the bottom at each point in the trajectory.
            - Updates the `onBottomPre` attribute, which considers both the current and preceding
              frames for determining the "on bottom" status.
        """
        if self.va.ct is CT.regular:
            v = 2  # version (1 or 2)
            xf, dx, useMinMax = self.va.xf, 15, True
            xm, ym = xf.t2f((4, 109 + dx)[self.f], 2.5)
            xM, yM = xf.t2f((86 - dx, 191)[self.f], 137)
            xmin, xmax = np.nanmin(self.x), np.nanmax(self.x)
            if useMinMax:
                xm = xmin + dx if self.f == 1 else xm
                xM = xmax - dx if self.f == 0 else xM
            with np.errstate(invalid="ignore"):  # suppress warnings due to NaNs
                onB = (
                    (xm < self.x)
                    & (self.x < xM)
                    & (ym < self.y)
                    & (self.y < yM)
                    & (self.ar < (300 if v == 1 else 310))
                )
            if v == 2:
                onB &= self.d < 30  # exclude jumps
                for s in util.trueRegions(onB):
                    ar = self.ar[s.start : s.stop]
                    mar = np.mean(ar)
                    if mar < 210 or mar < 240 and len(ar) > 2:
                        idxs = np.flatnonzero(ar < 260)  # exclude large start or stop
                        onB[s.start : s.start + idxs[0]] = False
                        onB[s.start + idxs[-1] + 1 : s.stop] = False
                        continue
                    onB[s.start : s.stop] = False
        elif self.va.ct is CT.htl:
            onB = ~self.nan
        elif self.va.ct in (CT.large, CT.large2):
            onB = ~self.nan
        else:
            error("not yet implemented")
        self.onBottom = onB
        self.onBottomPre = np.zeros_like(self.x, dtype=bool)
        self.onBottomPre[1:] = self.onBottom[:-1]
        self.onBottomPre &= self.onBottom
        assert np.count_nonzero(self.onBottom != self.onBottomPre) == len(
            util.trueRegions(self.onBottom)
        )
        if self.va.ct is CT.regular:
            self.dltX = np.abs(self.x - xf.t2fX((86, 109)[self.f]))
            self.dltX2 = np.abs(self.x - (xmin if self.f else xmax))

    def _transformAngles(self):
        """
        Corrects the orientation angle (theta) for cases where OpenCV's fitEllipse method might switch
        the major axis between height and width when the ellipse (representing the fly) is oriented
        towards quadrants I and III. If the height is found to be less than the width for a given frame,
        indicating a potential misidentification of the major axis, this method swaps the height and width
        values and adjusts the orientation angle by adding 90 degrees. This ensures the angle accurately
        represents the fly's orientation relative to its actual shape and direction.

        Side effects:
            - Corrects the `h` (height), `w` (width), and `theta` (orientation angle) attributes of the
              trajectory to accurately reflect the fly's orientation at each point.
        """
        if self.h is None:
            return
        with np.errstate(invalid="ignore"):
            heightLtWidth = np.less(self.h, self.w)
            self.oldH = np.array(self.h)
            self.h = np.where(heightLtWidth, self.w, self.oldH)
            self.w = np.where(heightLtWidth, self.oldH, self.w)
            self.theta = np.where(
                heightLtWidth,
                util.normAngles(self.theta + 90, useRadians=False),
                self.theta,
            )
        self.theta[0 : self.trxStartIdx - 1] = 90

    def _chooseOrientations(self):
        """
        Applies the Viterbi algorithm to convert angular orientations from a [0, 180) degree range
        to a [0, 360) degree range, considering the fly's movements to resolve ambiguities in
        orientation. This method is especially useful for ensuring consistency in angular data
        across frames, particularly after transformations that may introduce discontinuities.

        Side effects:
            - Updates the `theta` attribute with angles adjusted to a [0, 360) degree range, providing
              a more accurate representation of the fly's orientation throughout the trajectory.
        """
        if self.h is None:
            return

        # 30 mm / s
        min_jump = (
            30 * self.va.ct.pxPerMmFloor() / self.va.fps if self.va else self.JMP_LEN_TH
        )
        self.theta = resolveAngles180To360(
            self.theta, self.x, self.y, self.trxStartIdx, min_jump
        )

    def _suspiciousJumps(self):
        """
        Identifies and records suspicious jumps in the fly's trajectory based on distance thresholds.
        A jump is considered suspicious if the sum of the distances from the end of one jump to the start
        of the next, and vice versa, is below a certain threshold. The method updates the trajectory's
        status to bad if the fraction and number of suspicious jumps exceed specified thresholds.

        Side effects:
            - Populates the `susp` attribute with indices of suspicious jumps.
            - Updates the `_bad` attribute if the criteria for suspicious activity are met, indicating
              potential issues with the trajectory data.
        """
        self.susp = []
        jis = (self.d > self.JMP_LEN_TH).nonzero()[0]
        # indexes of jump start points; jis+1 gives jump end points
        ns, nj = 0, len(jis)
        for i, ji in enumerate(jis):
            if i > 0:
                pji = jis[i - 1]
                if self._DEBUG and i < 10 and self.f == 1:
                    print(i, ji, self.d[ji - 2 : ji + 2])
                if self.dist(pji + 1, ji) + self.dist(pji, ji + 1) < self.DIST_TH:
                    self.susp.extend((pji, ji))
                    ns += 1
        sf = ns / nj if nj else 0
        self._bad = sf >= self.SUSP_FRAC_TH and ns >= self.SUSP_NUM_TH
        self._p(
            "  long (>%d) jumps: %d, suspicious: %d%s%s"
            % (
                self.JMP_LEN_TH,
                nj,
                ns,
                " ({:.1%})".format(sf) if nj else "",
                " *** bad ***" if self._bad else "",
            )
        )

    def _checkRewards(self, t, en):
        """
        Compares calculated rewards with actual ones for the first fly only, identifying any discrepancies.
        It specifically looks for instances where the video analysis might have dropped frames, resulting
        in mismatches between expected and observed rewards. The method also skips differences of a single
        frame and the last frame of a training session to avoid false positives.

        Parameters:
            t (object): An object containing start and stop times for the analysis period.
            en (np.array): Array of frames where rewards were expected.

        Side effects:
            - Updates the `no_en` and `no_on` attributes with the count of discrepancies between expected
              rewards (en) and observed rewards (on).
            - Optionally displays images highlighting the discrepancies if `showRewardMismatch` option
              is enabled.
        """
        if self.f != 0:  # only for fly 1
            return
        en = util.inRange(en, t.start, t.stop)
        on = self.va._getOn(t)
        if np.array_equal(en, on):
            return
        enS, onS = set(en), set(on)
        sd = np.array(sorted(enS ^ onS))

        # skip 1-frame differences
        d1 = (np.diff(sd) == 1).nonzero()[0]
        sdS = set(np.delete(sd, np.concatenate((d1, d1 + 1))))
        # skip last training frame
        sdS -= {t.stop - 1}

        self.no_en += len(sdS & enS)
        self.no_on += len(sdS & onS)
        if self.opts.showRewardMismatch:
            imgs, hdrs, nr = [], [], 4
            for j, fi in enumerate(sorted(sdS)):
                i1, i2 = fi - 2, fi + 3
                imgs.extend(self._annImgs(i1, i2, show="d"))
                for i in range(i1, i2):
                    if i == fi:
                        hdr = "f %d only %s" % (i, cVsA(fi in enS))
                    else:
                        hdr = "f %+d" % (i - fi)
                        if i == i1 and j % nr == 0:
                            hdr += "  (t %d-%d)" % (t.start, t.stop)
                    hdrs.append(hdr)
                if (j + 1) % nr == 0 or j + 1 == len(sdS):
                    self.rmImNum += 1
                    cv2.imshow(
                        "reward mismatch %d" % self.rmImNum,
                        util.combineImgs(imgs, hdrs=hdrs, nc=i2 - i1)[0],
                    )
                    del imgs[:], hdrs[:]

    def calcRewardsImg(self):
        """
        Generates images illustrating the points at which rewards were either expected or actually occurred,
        for both control and post-experiment phases. The method processes each training session to highlight
        reward entries within specified intervals, appending annotations to differentiate between control
        and post-reward visuals.

        Side effects:
            - Generates and saves annotated images to disk, showing reward occurrences for control
              (ctrl == True) and post-experiment (ctrl == False) phases. The images are saved with
              filenames indicating the control or post status and the fly number.
        """
        for ctrl in (False, True):
            # post rewards shown for ctrl == False
            imgs, hdrs = [], []
            for t in self.va.trns:
                if not hasattr(self, "en"):
                    continue
                en = self.en[ctrl]
                fi, la = (t.start, t.stop) if ctrl else (t.stop, t.postStop)
                en = util.inRange(en, fi, la)
                tSfx = ("" if ctrl else " post") + ", "
                for j, eni in enumerate(en[:2]):
                    i1, i2 = eni - 1, eni + 1
                    imgs.extend(self._annImgs(i1, i2, show="d", ctrl=ctrl))
                    for i in range(i1, i2):
                        hdr = ""
                        if i == i1:
                            hdr = "%sf %+d" % (
                                t.sname() + tSfx if j == 0 else "",
                                i - fi,
                            )
                        elif i == eni:
                            hdr = "enter"
                        hdrs.append(hdr)
            if imgs:
                img = util.combineImgs(
                    imgs, hdrs=hdrs, nc=(i2 - i1) * 2, hdrL=util.basename(self.va.fn)
                )[0]
                fn = CALC_REWARDS_IMG_FILE % ("ctrl" if ctrl else "post", self.f + 1)
                writeImage(fn, img)

    def _calcPercentInCircle(self, t, inC, inCPre=None):
        bl_3_min = self.va._min2f(3)
        bl_10_min = self.va._min2f(10)
        bl_1_min = self.va._min2f(1)
        bl_2_min = self.va._min2f(2)

        inCRngs, nanRngs = self._append_pct_circle_pre_ranges(
            t, inC, inCPre, bl_3_min, bl_10_min
        )
        inC_custom, inCPre_custom = self._calculate_custom_circle(t, inCPre)

        self._append_pct_circle_post_ranges(
            t, inC, inCRngs, nanRngs, bl_1_min, bl_2_min
        )

        if t.n == 1 and inCPre is not None:
            inCRngs.insert(2, inCPre)
            nanRngs.insert(2, slice(self.va.startPre, t.start))
        elif t.n == len(self.va.trns):
            inCRngs.append(inC[t.stop - t.start :])
            nanRngs.append(slice(t.stop, t.postStop))

        self._calculate_circle_percentages_for_ranges(
            t, inCRngs, nanRngs, inC_custom, inCPre_custom, bl_3_min
        )

    def _append_pct_circle_pre_ranges(self, t, inC, inCPre, bl_3_min, bl_10_min):
        inCRngs = [inC[0 : t.stop - t.start]]
        nanRngs = [slice(t.start, t.stop)]

        if t.n == 1 and inCPre is not None:
            # First 3 minutes of pre-training
            inCRngs.insert(0, inCPre[:bl_3_min])
            nanRngs.insert(0, slice(self.va.startPre, self.va.startPre + bl_3_min))

            # Final 10 minutes of pre-training
            inCRngs.insert(1, inCPre[-bl_10_min:])
            nanRngs.insert(1, slice(t.start - bl_10_min, t.start))

        return inCRngs, nanRngs

    def _calculate_custom_circle(self, t, inCPre):
        if self.opts.pctTimeCircleRad:
            customRad = util.intR(
                self.opts.pctTimeCircleRad * self.va.ct.pxPerMmFloor() * self.va.xf.fctr
            )
            x, y = self.xy(t.start, t.postStop)
            cx, cy, r = t.circles(self.f)[0]  # Original reward circle
            inC_custom = self.calc_in_circle(x, y, cx, cy, customRad)
            if t.n == 1 and inCPre is not None:
                x_pre, y_pre = self.xy(self.va.startPre, t.start)
                inCPre_custom = self.calc_in_circle(x_pre, y_pre, cx, cy, customRad)
            else:
                inCPre_custom = None
        else:
            inC_custom = None
            inCPre_custom = None

        return inC_custom, inCPre_custom

    def _append_pct_circle_post_ranges(
        self, t, inC, inCRngs, nanRngs, bl_1_min, bl_2_min
    ):
        startPost = self.va.fns["startPost"][t.n - 1]

        # Add ranges for first 1 minute and first 2 minutes of the post period
        inCRngs.append(inC[startPost - t.start : startPost - t.start + bl_1_min])
        nanRngs.append(slice(startPost, startPost + bl_1_min))
        inCRngs.append(inC[startPost - t.start : startPost - t.start + bl_2_min])
        nanRngs.append(slice(startPost, startPost + bl_2_min))

        # Existing post-buckets for each training
        bl = self.va._min2f(self.opts.postBucketLenMin)
        for _ in range(2):
            inCRngs.append(inC[startPost - t.start : startPost - t.start + bl])
            nanRngs.append(slice(startPost, startPost + bl))
            startPost += bl

    def _calculate_circle_percentages_for_ranges(
        self, t, inCRngs, nanRngs, inC_custom, inCPre_custom, bl_3_min
    ):
        for j, inCRng in enumerate(inCRngs):
            numNaN = np.count_nonzero(~self.nan[nanRngs[j]])
            if numNaN > 0:
                pctInOriginalCircle = np.count_nonzero(inCRng == 2) / numNaN
                self.pctInC["rwd"].append(pctInOriginalCircle)

                if inC_custom is not None:
                    if t.n == 1:
                        if (
                            j == 0 and inCPre_custom is not None
                        ):  # First 3 minutes of pre-training
                            pctInCustomCircle = (
                                np.count_nonzero(inCPre_custom[:bl_3_min] == 2) / numNaN
                            )
                        elif (
                            j == 1 and inCPre_custom is not None
                        ):  # Final 10 minutes of pre-training
                            bl_10_min = len(inCRng)  # Length of the last 10 minutes
                            pctInCustomCircle = (
                                np.count_nonzero(inCPre_custom[-bl_10_min:] == 2)
                                / numNaN
                            )
                        elif (
                            j == 2 and inCPre_custom is not None
                        ):  # Entire pre-training period
                            pctInCustomCircle = (
                                np.count_nonzero(inCPre_custom == 2) / numNaN
                            )
                        else:  # Post-training and custom timeframes
                            pctInCustomCircle = (
                                np.count_nonzero(
                                    inC_custom[
                                        nanRngs[j].start
                                        - t.start : nanRngs[j].stop
                                        - t.start
                                    ]
                                    == 2
                                )
                                / numNaN
                            )
                    else:
                        # Post-training and other periods for non-pre-training
                        pctInCustomCircle = (
                            np.count_nonzero(
                                inC_custom[
                                    nanRngs[j].start
                                    - t.start : nanRngs[j].stop
                                    - t.start
                                ]
                                == 2
                            )
                            / numNaN
                        )
                    self.pctInC["custom"].append(pctInCustomCircle)
            else:
                self.pctInC["rwd"].append(np.nan)
                if inC_custom is not None:
                    self.pctInC["custom"].append(np.nan)

    @staticmethod
    def _calcEnEx(inC, start, mode="en"):
        """
        Calculates the indices of frames where the fly enters (mode="en") or exits (mode="ex") the reward
        circle, based on transitions in the fly's position relative to the circle's border.

        Parameters:
            inC (np.array): An array indicating the fly's position relative to the reward circle for each frame,
                            where 0 represents outside, 1 on the border, and 2 inside the border.
            start (int): The index of the first frame in `inC` to consider in the calculations.
            mode (str, optional): Specifies the mode of calculation, either "en" for entries or "ex" for exits.
                                  Default is "en".

        Returns:
            np.array: An array of indices (adjusted for the start index) where entries or exits occur,
                      depending on the specified mode.

        Note:
            The method is static and does not modify the state of the object.
        """
        idxs = np.arange(len(inC))[inC != 1]
        sign = 1 if mode == "en" else -1
        return idxs[np.flatnonzero(np.diff(inC[inC != 1]) == sign * 2) + 1] + start

    def calc_in_circle(self, x, y, cx, cy, r):
        """
        Calculates the position of the fly relative to a circle defined by its center and radius.

        Parameters:
            x (np.array): X-coordinates of the fly's position.
            y (np.array): Y-coordinates of the fly's position.
            cx (float): X-coordinate of the circle's center.
            cy (float): Y-coordinate of the circle's center.
            r (float): Radius of the circle.

        Returns:
            np.array: An array indicating the fly's position relative to the circle for each frame.
                    0 represents outside, 1 on the border, and 2 inside the border.
        """
        dc = np.linalg.norm([x - cx, y - cy], axis=0)
        inC = (dc < r).astype(int) + (dc < r + BORDER_WIDTH)
        return inC

    def _calcOutsideCirclePeriods(self, debug=False):
        """
        Calculates the duration of periods when the fly is outside a series of circles of increasing
        radius, concentric with the reward circle, and outputs summary statistics for manual checks.
        """
        if not self.va or not self.va.circle or self.bad():
            return

        # Convert wall contact regions into a set of frames for quick lookup
        wall_contact_frames = set()
        for region in self.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]:
            wall_contact_frames.update(range(region.start, region.stop))
        print(f"{flyDesc(self.f)}")

        if debug:
            print(
                f"[INFO] Wall contact frames (total {len(wall_contact_frames)}):",
                sorted(wall_contact_frames),
            )

        radii_mm = self.opts.outside_circle_radii  # Radii in mm
        # Convert radii from mm to pixels
        radii_px = [el * self.va.ct.pxPerMmFloor() * self.va.xf.fctr for el in radii_mm]

        if debug:
            print(f"[INFO] Radii (mm): {radii_mm}")
            print(f"[INFO] Radii (px): {radii_px}")

        outside_durations = []

        def calculate_durations(ex_evts, en_evts, trn_idx, fly_type):
            """Helper function to calculate valid durations."""
            valid_pairs = []
            i, j = 0, 0

            if debug:
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Start of calculate_durations."
                )
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Exit events:",
                    ex_evts,
                )
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Entry events:",
                    en_evts,
                )

            while i < len(ex_evts) and j < len(en_evts):
                if ex_evts[i] < en_evts[j]:
                    if not any(
                        frame in wall_contact_frames
                        for frame in range(ex_evts[i], en_evts[j])
                    ):
                        valid_pairs.append((ex_evts[i], en_evts[j]))
                    elif debug:
                        print(
                            f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Skipped due to wall contact between {ex_evts[i]} and {en_evts[j]}"
                        )
                    i += 1
                    j += 1
                else:
                    j += 1

            if debug:
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Valid exit-entry pairs:",
                    valid_pairs,
                )

            durations = [en - ex for ex, en in valid_pairs if en - ex > 0]

            if debug:
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Durations:",
                    durations,
                )
                print(
                    f"[DEBUG] [Training Index: {trn_idx}, Fly Type: {self.f}] Mean duration:",
                    np.mean(durations) if durations else "N/A",
                )

            return durations

        for t_idx, t in enumerate(self.va.trns):
            start = t.start if t.n > 1 else self.va.startPre
            cx, cy, r_smallest = list(t.circles(self.f))[0]

            if debug:
                print(
                    f"[INFO] Training session: {t.n}, Start frame: {start}, Circle center: ({cx}, {cy}), Smallest radius: {r_smallest}"
                )

            if t.n == 1:
                outside_durations.append(
                    {r_mm: [] for r_mm in radii_mm}
                )  # For pre-training
                outside_durations.append(
                    {r_mm: [] for r_mm in radii_mm}
                )  # For training 1
            else:
                outside_durations.append(
                    {r_mm: [] for r_mm in radii_mm}
                )  # For other training periods

            x, y = self.xy(start, t.postStop)
            for r_mm, r_px in zip(radii_mm, radii_px):
                inC = self.calc_in_circle(x, y, cx, cy, r_px)

                if debug:
                    print(
                        f"[INFO] [Training Index: {t_idx}, Radius: {r_mm} mm] Radius (mm): {r_mm}, Radius (px): {r_px}"
                    )
                    print(
                        f"[INFO] [Training Index: {t_idx}, Radius: {r_mm} mm] Calculated in-circle values (first 20): {inC[:20]}"
                    )

                if t.n == 1:
                    inCPre = inC[0 : t.start - self.va.startPre]
                    inC = inC[t.start - self.va.startPre :]

                    # Pre-training durations (index 0)
                    ex_evts_pre = self._calcEnEx(inCPre, start, mode="ex")
                    en_evts_pre = self._calcEnEx(inCPre, start, mode="en")

                    if debug:
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Exit events: {ex_evts_pre}"
                        )
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Entry events: {en_evts_pre}"
                        )
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] inC pre:",
                            inCPre,
                        )
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Number of frames inside circle:",
                            np.count_nonzero(inCPre),
                        )

                    outside_durations[0][r_mm].extend(
                        calculate_durations(
                            ex_evts_pre, en_evts_pre, trn_idx=0, fly_type="Pre-training"
                        )
                    )

                    # Training 1 durations (index 1)
                    ex_evts = self._calcEnEx(inC, t.start, mode="ex")
                    en_evts = self._calcEnEx(inC, t.start, mode="en")

                    if debug:
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Exit events: {ex_evts}"
                        )
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Entry events: {en_evts}"
                        )

                    outside_durations[1][r_mm].extend(
                        calculate_durations(
                            ex_evts, en_evts, trn_idx=1, fly_type="Training 1"
                        )
                    )
                else:
                    # Other training periods
                    ex_evts = self._calcEnEx(inC, t.start, mode="ex")
                    en_evts = self._calcEnEx(inC, t.start, mode="en")

                    if debug:
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Exit events: {ex_evts}"
                        )
                        print(
                            f"[DEBUG] [Training Index: {t_idx}, Fly Type: {self.f}] Entry events: {en_evts}"
                        )

                    outside_durations[-1][r_mm].extend(
                        calculate_durations(
                            ex_evts, en_evts, trn_idx=t_idx, fly_type=f"{self.f}"
                        )
                    )

        self.outside_durations = outside_durations

        # Temporary output for manual sanity checks
        fps = self.va.fps  # Frames per second

        # Print a summary for each radius
        for i, phase_durations in enumerate(outside_durations):
            phase_name = (
                "Pre-training"
                if i == 0
                else "Training 1" if i == 1 else f"Training {i}"
            )
            print(f"{phase_name}:")
            for r_mm, durations in phase_durations.items():
                if durations:
                    mean_duration_frames = np.mean(durations)
                    mean_duration_seconds = mean_duration_frames / fps
                    print(
                        f"  Radius {r_mm:.2f} mm: Mean Duration = {mean_duration_seconds:.2f} seconds "
                        f"({mean_duration_frames:.2f} frames)"
                    )
                else:
                    print(f"  Radius {r_mm:.2f} mm: No outside periods recorded")

    def _calcRewards(self):
        """
        Calculates the occurrences of reward entries (circle enter events) for both the actual training
        circle of fly 1, a virtual training circle for fly 2, and control circles if defined. This method
        differentiates between calculated rewards and control rewards, taking into account the presence
        of a border width around the reward circle. It also compares calculated rewards against actual
        rewards to identify mismatches.

        Side effects:
            - Updates several attributes related to rewards, including `en` for enter events, `ex` for
            exit events if turn analysis is opted, `no_en` and `no_on` for mismatch statistics, and
            `pctInC` for the percentage of time spent in the reward circle.
            - Generates statistics on the total calculated and control rewards, and prints information
            about reward mismatches and the total control rewards during training sessions.
            - Optionally, displays images highlighting discrepancies between calculated and actual rewards
            if `showRewardMismatch` option is enabled.
        """
        if not self.va or not self.va.circle:
            return
        ens = [[], []]  # enter events
        if self.opts.cTurnAnlyz:
            exs = [[], []]  # exit events
        self.no_en = self.no_on = 0  # statistics for mismatch calc. vs. actual
        self.rmImNum, nEnT, nEn0T, twc = 0, [0, 0], 0, []
        self.pctInC = {"rwd": [], "custom": []}
        if self.opts.cTurnAnlyz:
            self.rCirExits = []
        for t in self.va.trns:
            start = t.start if t.n > 1 else self.va.startPre
            x, y = self.xy(start, t.postStop)
            for i, (cx, cy, r) in enumerate(t.circles(self.f)):
                inC = self.calc_in_circle(x, y, cx, cy, r)
                if self.opts.pctTimeCircleRad and i == 0:
                    customRad = self.opts.pctTimeCircleRad
                    inC_custom = self.calc_in_circle(x, y, cx, cy, customRad)
                for s in util.trueRegions(self.nan[start : t.postStop]):
                    inC[s] = inC[s.start - 1] if s.start > 0 else False
                    if self.opts.pctTimeCircleRad and i == 0:
                        inC_custom[s] = (
                            inC_custom[s.start - 1] if s.start > 0 else False
                        )
                if t.n == 1:
                    inCPre = inC[0 : t.start - self.va.startPre]
                    inC = inC[t.start - self.va.startPre :]
                    if self.opts.pctTimeCircleRad:
                        inC_custom = inC_custom[t.start - self.va.startPre :]
                    en = np.hstack(
                        (self._calcEnEx(inCPre, start), self._calcEnEx(inC, t.start))
                    )
                    if self.opts.cTurnAnlyz:
                        ex = np.hstack(
                            (
                                self._calcEnEx(inCPre, start, mode="ex"),
                                self._calcEnEx(inC, t.start, mode="ex"),
                            )
                        )
                else:
                    en = self._calcEnEx(inC, t.start)
                    if self.opts.cTurnAnlyz:
                        ex = self._calcEnEx(inC, t.start, mode="ex")

                ctrl = i > 0
                ens[ctrl].append(en)
                if self.opts.cTurnAnlyz:
                    exs[ctrl].append(ex)

                if i == 0:
                    if not self._bad:
                        self._calcPercentInCircle(t, inC, inCPre if t.n == 1 else None)
                    en0 = (
                        (np.diff((inC > 1).astype(int)) == 1).nonzero()[0] + 1 + t.start
                    )
                    self._checkRewards(t, en0)
                    nEn0T += util.inRange(en0, t.start, t.stop, count=True)
                    if BORDER_WIDTH == 0:
                        assert np.array_equal(en, en0)
                elif i == 1:
                    twc.append(t.n)
                nEnT[ctrl] += util.inRange(en, t.start, t.stop, count=True)

        self.en = [np.sort(np.concatenate(en)) for en in ens]
        if self.opts.cTurnAnlyz:
            self.ex = [np.sort(np.concatenate(ex)) for ex in exs]
        nt = nEnT[0]
        print("  total calculated rewards during training: %d" % nt)
        if self.f == 0:
            bw0 = BORDER_WIDTH == 0
            if not bw0:
                print(
                    "    for zero-width border: %d%s"
                    % (nEn0T, "" if nt == 0 else " (+{:.1%})".format((nEn0T - nt) / nt))
                )
            msg = []
            for no, tp in ((self.no_en, "calc."), (self.no_on, "actual")):
                if no:
                    msg.append("only %s: %d" % (tp, no))
            print(
                "%s    compared with actual ones: %s"
                % ("" if bw0 else "  ", ", ".join(msg) if msg else "identical")
            )
            if msg and self.opts.showRewardMismatch:
                cv2.waitKey(0)
        print(
            "  total control rewards during training%s %s: %d"
            % (util.pluralS(len(twc)), util.commaAndJoin(twc), nEnT[1])
        )

    def _plotIssues(self):
        """
        Visualizes tracking issues within the fly's trajectory, specifically highlighting suspicious jumps
        and periods where the fly is lost (i.e., not tracked). This method plots these issues over the
        first frame of the video for visual inspection, using different colors to denote suspicious jumps
        and lost periods.

        Side effects:
            - Generates a plot overlaying the trajectory issues on the first frame of the video, which is
              particularly useful for visually identifying potential data quality issues.
            - Prints details about suspicious jumps, including the timestamps and frame numbers where these
              jumps occur.
        """
        if not self.va:
            return
        susT, susC, losT, losC = "suspicious jump", "w", "lost", "y"
        if self.f == 0:
            plt.figure(util.basename(self.va.fn) + " Tracking Issues")
            plt.imshow(cv2.cvtColor(self.va.frame, cv2.COLOR_BGR2RGB))
            plt.axis("image")
            tx = plt.gca().transAxes
            for x, c, t in ((0.25, susC, susT), (0.75, losC, losT)):
                plt.text(x, 0.05, t, color=c, transform=tx, ha="center")
        for ji in self.susp:
            plt.plot(self.x[ji : ji + 2], self.y[ji : ji + 2], color=susC)
        print(
            "    suspicious jumps: %s"
            % ", ".join(
                "%s (%d)" % (util.s2time(ji / self.va.fps), ji) for ji in self.susp[::2]
            )
        )
        for r in self.nanrs:
            f, t = r.start, r.stop
            plt.plot(self.x[f:t], self.y[f:t], color=losC, marker="o", ms=3, mew=0)

    def _annImgs(self, i1, i2, show="", ctrl=False):
        """
        Generates a list of images annotated with various markers and text for frames within
        the specified range. The annotations can include differences in timestamps between
        frames, distance relations to a control point, and the experimental fly number,
        depending on the 'show' argument. Additionally, control and training period markers
        can be annotated.

        Parameters:
            i1 (int): The starting frame index for generating annotations.
            i2 (int): The ending frame index (exclusive) for generating annotations.
            show (str, optional): Controls what additional information is included in the
                                  text annotations. Can include:
                                  't' for the time difference between the current frame and
                                      the previous frame,
                                  'd' for indicating distance relation (less than or greater
                                      than/equal to) to a reference point,
                                  and 'f' for displaying the experimental fly number.
            ctrl (bool, optional): Indicates whether control circle annotations should be
                                   applied. This is typically used to differentiate between
                                   actual and virtual training scenarios.

        Returns:
            list: A list of images (as numpy arrays) with the specified annotations applied,
                  including training period markers, control indicators, and optional text
                  annotations based on the 'show' parameter.

        Notes:
            - This method reads frames from the video, applies ellipse and text annotations
              based on the specified parameters, and extracts the chamber-specific area of
              each annotated frame.
            - The method supports detailed analysis by visually marking key aspects of the
              fly's behavior and environmental interactions within the specified frame range.
        """
        imgs = []
        for i in range(i1, i2):
            img = util.readFrame(self.va.cap, i)
            t, cpr = Training.get(self.va.trns, i, includePost=True), None
            if t:
                cpr = t.annotate(img, ctrl=ctrl, f=self.f)
            ellDrwn = self.annotate(img, i, boundary_contact_markup=False)
            img = self.va.extractChamber(img)
            self.annotateTxt(img, i, show, cpr)
            # uncomment to show ellipse params:
            # TODO: move to player?
            # if ellDrwn:
            #   (x, y), (w, h), theta = self.ellipse(i)
            #   putText(img, "w = %.1f, h = %.1f, theta = %.1f" %(w, h, theta),
            #     (5,5), (0,1), textStyle(color=COL_W))
            imgs.append(img)
        return imgs

    def annotate_boundary_contact(self, img, i, boundary_tp, evt_type):
        """
        Annotates an image with visual markers indicating boundary contact events, such as wall contacts
        or turns near the boundary. The method draws boundary lines and marks points of contact or near
        contact between the fly and the boundary.

        Parameters:
            img (numpy.ndarray): The image to be annotated.
            i (int): The frame index corresponding to the image being annotated.
            boundary_tp (str): The type of boundary event being annotated ("wall" or other types indicating
                               proximity to the bottom or top boundaries).
            evt_type (str): The event type to annotate ("contact" for direct boundary contacts or "turn"
                            for turning events near the boundary).

        Side effects:
            - Draws boundary lines, contact lines, and adds text annotations directly onto the input image,
              modifying it in place.
            - The color and content of annotations vary depending on the distance to the boundary and the
              type of event being annotated.

        Notes:
            - This method supports detailed analysis of boundary interactions by visually distinguishing
              between different types of boundary events and their locations relative to predefined boundary
              zones.
        """
        if boundary_tp == "wall":
            boundary_orientation = self.opts.wall_orientation
        else:
            boundary_orientation = "tb"
        stats = self.boundary_event_stats[boundary_tp][boundary_orientation]
        dists = stats["dist_to_boundary"]
        if evt_type == "contact":
            contact = stats["boundary_contact"]
            near_contact = stats["near_contact"]
        elif evt_type == "turn":
            contact = stats["turning"]
            near_contact = stats["near_turning"]
        bnds = stats["bounds"]
        corners = {
            "tl": [bnds["x"][0], bnds["y"][0]],
            "bl": [bnds["x"][0], bnds["y"][1]],
            "tr": [bnds["x"][1], bnds["y"][0]],
            "br": [bnds["x"][1], bnds["y"][1]],
        }
        for k in corners:
            corners[k] = [int(round(el)) for el in corners[k]]
        for segment in (("tl", "bl"), ("tl", "tr"), ("tr", "br"), ("bl", "br")):
            cv2.line(
                img, tuple(corners[segment[0]]), tuple(corners[segment[1]]), COL_R, 1
            )

        if not self.nan[i]:
            if dists[i] > 0:
                pts = [
                    tuple(
                        [
                            int(round(el))
                            for el in stats["ellipse_and_boundary_pts"][side][i]
                        ]
                    )
                    for side in ("boundary", "ellipse")
                ]
                cv2.line(img, pts[0], pts[1], COL_G, 1)
            text_coords = tuple(
                [corners["bl"][0] - 5, corners["bl"][1] + 4]
                if self.va.ef // self.va.ct.numCols() == 0 and self.f == 0
                else [corners["tl"][0] - 5, corners["tl"][1] - 12]
            )
            if not self.boundary_text_added:
                if near_contact[i]:
                    txt_col = COL_B
                elif contact[i]:
                    txt_col = COL_W
                else:
                    txt_col = COL_BK
                util.putText(
                    img,
                    "%.2fmm" % dists[i],
                    text_coords,
                    (0, 1),
                    util.textStyle(color=txt_col),
                )
                self.boundary_text_added = True

    def annotate(
        self,
        img,
        i,
        tlen=1,
        col=COL_Y,
        boundaries={},
        flagged_regions=None,
        flagged_cols=None,
        evt_type="contact",
        boundary_contact_markup=True,
    ):
        """
        Annotates a given image with an ellipse representing the fly's position and orientation at frame i,
        optionally drawing an arrow to indicate the velocity direction or head orientation. It also supports
        marking boundary contacts and drawing partial trajectories leading up to the current frame.

        Parameters:
            img (numpy.ndarray): The image to be annotated.
            i (int): The frame index corresponding to the image being annotated.
            tlen (int, optional): The length of the trajectory to draw leading up to frame i. Default is 1.
            col (tuple, optional): The color for the ellipse. Default is yellow (COL_Y).
            boundaries (dict, optional): A dictionary specifying the types of boundaries to annotate.
            flagged_regions (list, optional): Regions within the trajectory that should be highlighted.
            flagged_cols (list, optional): Colors to use for flagged regions.
            evt_type (str, optional): The type of boundary event to annotate. Default is "contact".
            boundary_contact_markup (bool, optional): Whether to include boundary contact annotations. Default is True.

        Returns:
            bool: True if the ellipse was drawn, False otherwise (e.g., if the frame is marked as NaN).

        Side effects:
            - Directly modifies the input image by drawing annotations.
            - Marks boundary contacts and draws trajectory segments or velocity vectors as specified.
        """
        nn = not self.nan[i]
        if nn:
            if hasattr(self, "velAngles") and not np.isnan(self.velAngles[i]):
                cv2.arrowedLine(
                    img,
                    (int(round(self.x[i])), int(round(self.y[i]))),
                    (
                        int(round(self.x[i] + 50 * math.cos(self.velAngles[i]))),
                        int(round(self.y[i] + 50 * math.sin(self.velAngles[i]))),
                    ),
                    COL_O,
                    1,
                )
            if boundary_contact_markup and (
                not self.opts.agarose
                or (
                    self.opts.agarose
                    and not self.boundary_event_stats["agarose"]["tb"][
                        "boundary_contact"
                    ][i]
                )
            ):
                floor_coords = list(
                    self.va.ct.floor(self.va.xf, f=self.va.nef * (self.f) + self.va.ef)
                )
                top_left, bottom_right = floor_coords[0], floor_coords[1]
                cv2.rectangle(img, top_left, bottom_right, COL_G)
                cv2.ellipse(img, self.ellipse(i), col, 1)
            else:
                ellipse = self.ellipse(i, rounded=True, semi_axes=True)
                for j in range(2):
                    cv2.ellipse(
                        img,
                        ellipse[0],
                        ellipse[1],
                        ellipse[2],
                        j * 180 + 50,
                        j * 180 + 130,
                        col,
                        1,
                    )
            if self.opts.chooseOrientations:
                arrowedLinePoints = self.centerToHead(i)
                cv2.arrowedLine(
                    img, arrowedLinePoints[0], arrowedLinePoints[1], COL_R, 2
                )
        if len(boundaries) == 0:
            for bnd_tp in ("wall", "agarose"):
                boundaries[bnd_tp] = getattr(self.opts, bnd_tp)
        self.boundary_text_added = False
        for k in boundaries:
            if (
                not boundary_contact_markup
                or not boundaries[k]
                or (not getattr(self.opts, "%s_eg" % k) and not self.opts.play)
            ):
                continue

            self.annotate_boundary_contact(img, i, k, evt_type)
        i1 = max(i - tlen, 0)
        xy = self.xy(i1, i + 1)
        xy = [a[~np.isnan(a)] for a in xy]
        if len(xy) > 1:
            self.annotateTrx(img, xy, i, i1, flagged_regions, flagged_cols)
        return nn

    def annotateTrx(self, img, xy, i, i1, flagged_regions, flagged_cols):
        """
        Draws the trajectory on the annotated video frame from index i1 to i. This method supports highlighting
        specific regions of interest within the trajectory and can differentiate trajectory segments based on
        circular motion predictions, using different colors to indicate looping or non-looping motion.

        Parameters:
            img (numpy.ndarray): The image on which to draw the trajectory.
            xy (list): A list containing the x and y coordinates of the trajectory to be drawn.
            i (int): The ending frame index of the trajectory segment.
            i1 (int): The starting frame index of the trajectory segment.
            flagged_regions (list, optional): A list of regions (as slice objects) within the trajectory that
                                               should be highlighted, indicating areas of specific interest.
            flagged_cols (list, optional): A list of colors corresponding to each flagged region, used to
                                           visually distinguish these areas on the image.

        Side effects:
            - Directly modifies the input image by drawing the specified trajectory segments.
            - If `opts.circle` is True, segments the trajectory based on circular motion predictions and
              applies different colors to indicate looping (circular) versus non-looping (straight) motion.
            - Highlights specified regions of the trajectory with the provided colors for flagged regions.

        Notes:
            - The method dynamically adjusts to the presence of flagged regions, altering the trajectory
              visualization to emphasize these areas.
            - The circular motion differentiation is contingent on the `opts.circle` setting, aiming to provide
              visual cues for analyzing motion patterns and behaviors.
        """
        if flagged_regions:

            def polyline(pts, col):
                cv2.polylines(img, pts, False, col)

            pts_to_plot = util.xy2Pts(*xy)
            if flagged_regions[0].start > 0:
                polyline(pts_to_plot[:, : flagged_regions[0].start + 1], COL_BK)
            for j, reg in enumerate(flagged_regions):
                if reg.stop - reg.start == 1:
                    cv2.circle(
                        img, tuple(pts_to_plot[:, reg.start][0]), 2, flagged_cols[j], -1
                    )
                    nonflagged_offset = 1
                else:
                    polyline(pts_to_plot[:, reg.start : reg.stop], flagged_cols[j])
                    nonflagged_offset = 0
                polyline(
                    pts_to_plot[
                        :,
                        max(reg.stop - 1 - nonflagged_offset, 0) : (
                            flagged_regions[j + 1].start + 1
                            if j + 1 < len(flagged_regions)
                            else i - i1 + 1
                        ),
                    ],
                    COL_BK,
                )
            return
        if self.opts.circle is True:
            predictions = self.circlePredictions[i1 : i + 1]
            if np.any([np.size(a) < len(predictions) for a in xy]):
                return
            regions = np.insert(
                np.unique([[r.start, r.stop] for r in util.trueRegions(predictions)]),
                0,
                0,
            ).astype(int)
            for i in range(0, len(regions)):
                u_idx = regions[i + 1] + 1 if i < len(regions) - 1 else None
                cv2.polylines(
                    img,
                    util.xy2Pts(*[a[regions[i] : u_idx] for a in xy]),
                    False,
                    COL_R if predictions[regions[i]] == True else COL_Y_D,
                )
        cv2.polylines(img, util.xy2Pts(*xy), False, COL_Y_D)

    def annotateTxt(self, img, i=None, show="", cpr=None):
        """
        Annotates an image with various text-based information depending on the 'show'
        argument. This can include the difference in timestamp between the current and
        previous frame, distance comparison to the reward circle, and the experimental fly
        number.

        Parameters:
            img (numpy.ndarray): The image to annotate.
            i (int, optional): Frame index for which annotations are being made. Required for
                               timestamp and distance annotations.
            show (str, optional): Controls what information to show on the image:
                                  't' includes timestamp difference,
                                  'd' includes a comparison of the fly's distance to the
                                      reward circle's radius ('<' means inside, '>=' means
                                      outside or on the border),
                                  'f' includes the experimental fly number.
            cpr (tuple, optional): Center point and radius of the reward circle for distance
                                   comparison. Required if 'd' is in the show argument.

        Side effects:
            - Directly modifies the input image by adding text annotations.
        """
        txt, alrt = [], False
        if i and i > 0 and "t" in show:
            dt, dt0 = self.ts[i] - self.ts[i - 1], 1 / self.va.fps
            alrt = abs(dt - dt0) / dt0 > 0.1
            txt.append("+%.2fs" % dt)
        if cpr and "d" in show:
            txt.append(
                "d %s r"
                % ("<" if util.distance(self.xy(i), cpr[:2]) < cpr[2] else ">=")
            )
        if "f" in show:
            txt.append("%d" % self.va.ef)
        if txt:
            util.putText(
                img,
                ", ".join(txt),
                (5, 5),
                (0, 1),
                util.textStyle(color=COL_Y if alrt else COL_W),
            )

    # - - -

    @staticmethod
    def _test(opts):
        """
        A static method for testing various functionalities within the Trajectory class, such as interpolation,
        boundary contact calculations, and utility functions. This method serves as a self-contained unit test
        for the class, ensuring correctness and stability of its methods.

        Parameters:
            opts: A configuration object containing options and settings used within the tests.

        Note:
            - This method should not be used in production code and is intended for development and testing purposes only.
        """
        nan = np.nan
        xy = (np.array(e) for e in ([nan, 1, nan, 2, nan], [nan, 2, nan, 4, nan]))
        t = Trajectory(xy, opts)
        util.requireClose((t.x, t.y), ([nan, 1, 1.5, 2, 2], [nan, 2, 3, 4, 4]))
        util.requireClose(t.d, [0, np.sqrt(0.5**2 + 1), np.sqrt(0.5**2 + 1), 0, 0])
        util.requireClose(t.d[1], t.dist(1, 2))
        Trajectory._testBoundaryContactCalcs()

    @staticmethod
    def _testBoundaryContactCalcs():
        """
        A static method specifically designed to test the boundary contact calculations within the Trajectory class.
        It uses mock data and configurations to verify the accuracy of distance and event detection related to
        boundary contacts.

        Note:
            - This method focuses on testing the functionality related to detecting and calculating boundary contacts,
              such as distance to agarose or wall boundaries.
            - Intended for internal testing and verification purposes only.
        """

        class TestVideoAnalysis(VideoAnalysisInterface):
            def __init__(self):
                pass

        class TestTrajectory(Trajectory):
            def __init__(self, xy, wht=None, f=0, va=None, ts=None):
                self.x, self.y = xy
                (self.w, self.h, self.theta) = wht if wht else 3 * (None,)
                self.f, self.va, self.ts = f, va, ts
                self.nan = np.isnan(self.x)

        mockVA = TestVideoAnalysis()
        mockVA.circle = True
        mockVA.nef = 10
        mockVA.ef = 0
        mockVA.trxf = (0,)
        mockVA.ct = CT.htl
        mockVA.on = np.array([100, 200])
        mockVA.trns = [slice(0, 5)]
        mockVA.fps = 7.5
        mockXF = Xformer(
            {"fctr": 1, "x": 0, "y": 0}, CT.htl, np.zeros((720, 720)), False
        )
        mockVA.xf = mockXF

        # distance-to-agarose tests
        xs = [25, 40, 30]
        ys = [25, 40, 100]
        xy = (np.array(e, dtype=np.double) for e in (xs, ys))
        agarose_dist_trj = TestTrajectory(
            xy,
            (np.array(e) for e in ([[6] * len(xs), [15] * len(ys), [0.0, 45.0, -58]])),
            va=mockVA,
        )
        dist_calculator = EllipseToBoundaryDistCalculator(agarose_dist_trj, mockVA)
        dist_calculator.calc_dist_boundary_to_ellipse("agarose", "tb", 3, (0.5, 0.8))
        util.requireClose(
            dist_calculator.return_data["boundary_event_stats"]["agarose"]["tb"][
                "edge"
            ]["dist_to_boundary"],
            [0, 0.93046135, 0.33005768],
        )

        # distance-to-wall tests
        xs = [43.96897561, 74.24995122, 41, 24]
        ys = [67.10194504, 12.69, 120, 79]
        xy = (np.array(e) for e in (xs, ys))
        wall_dist_trj = TestTrajectory(
            xy,
            (
                np.array(e)
                for e in ([[6] * len(xs), [15] * len(ys), [0.0, 45.0, -17.7, 72.9]])
            ),
            va=mockVA,
        )
        dist_calculator = EllipseToBoundaryDistCalculator(wall_dist_trj, mockVA)
        dist_calculator.calc_dist_boundary_to_ellipse("wall", "all", 0, (0.7, 1.0))
        util.requireClose(
            dist_calculator.return_data["boundary_event_stats"]["wall"]["all"]["edge"][
                "dist_to_boundary"
            ],
            [4.66012195, 0.53602127, 0.53886833, 1.63618514],
        )

    # - - -

    def distTrav(self, i1, i2):
        """
        Calculates the cumulative distance traveled between two frame indices.

        Parameters:
            i1 (int): Starting frame index.
            i2 (int): Ending frame index.

        Returns:
            float: The total distance traveled between the two frames.
        """
        return np.sum(self.d[i1:i2]) if hasattr(self, "d") else np.nan

    def dist(self, i1=None, i2=None):
        """
        Calculates the distance between two given frames or returns an array of distances
        between consecutive frames.

        Parameters:
            i1 (int, optional): Starting frame index. If only i1 is provided, returns the
                                coordinates at this index.
            i2 (int, optional): Ending frame index. If both i1 and i2 are provided, calculates
                                the distance between these frames.

        Returns:
            float or numpy.ndarray: Distance between i1 and i2 if both are provided, otherwise
                                    an array of distances.
        """
        return (
            self.d
            if i1 is None
            else util.distance((self.x[i1], self.y[i1]), (self.x[i2], self.y[i2]))
        )

    def xy(self, i1=None, i2=None):
        """
        Retrieves the x and y coordinates for a specified range of frames or for the entire
        trajectory.

        Parameters:
            i1 (int, optional): Starting frame index.
            i2 (int, optional): Ending frame index.

        Returns:
            tuple: A tuple of numpy arrays (x, y) containing the coordinates for the specified
                   frame range or the entire trajectory.
        """
        return (
            (self.x, self.y)
            if i1 is None
            else (
                (self.x[i1], self.y[i1])
                if i2 is None
                else (self.x[i1:i2], self.y[i1:i2])
            )
        )

    def xyRdp(self, i1, i2, epsilon):
        """
        Applies the Ramer-Douglas-Peucker algorithm to simplify the trajectory between two frame indices.

        Parameters:
            i1 (int): Starting frame index.
            i2 (int): Ending frame index.
            epsilon (float): The maximum distance threshold for simplification.

        Returns:
            tuple: Simplified x and y coordinates after applying the RDP algorithm.
        """
        return util.xy2T(util.rdp(self.xy(i1, i2), epsilon, _RDP_PKG))

    def ellipse(self, i, rounded=False, semi_axes=False):
        """
        Returns the parameters of the ellipse representing the fly at a given frame.

        Parameters:
            i (int): Frame index.
            rounded (bool, optional): If True, rounds the ellipse parameters to integers. Default is False.
            semi_axes (bool, optional): If True, returns semi-axes lengths instead of full axes lengths. Default is False.

        Returns:
            tuple: A tuple containing the ellipse parameters (center coordinates, axes lengths, orientation angle).
        """
        axes = (self.w[i], self.h[i])
        if semi_axes:
            axes = [a / 2 for a in axes]
        coords = (self.x[i], self.y[i])
        theta = self.theta[i]
        if rounded:
            axes, coords = [tuple([int(round(a)) for a in l]) for l in (axes, coords)]
            theta = int(round(theta))
        return (coords, axes, theta)

    def centerToHead(self, i):
        """
        Calculates the points defining a line from the center of the fly to a point along its heading direction,
        scaled by the ellipse's height, for a given frame.

        Parameters:
            i (int): The frame index.

        Returns:
            list: A list containing two tuples, each representing an (x, y) point: the center and a point in the
                  heading direction.
        """
        return list(
            map(
                util.intR,
                [
                    (self.x[i], self.y[i]),
                    (
                        self.x[i] + self.h[i] * np.sin(np.radians(self.theta[i])),
                        self.y[i] - self.h[i] * np.cos(np.radians(self.theta[i])),
                    ),
                ],
            )
        )

    def bad(self, bad=None):
        """
        Gets or sets the trajectory's 'bad' status, indicating whether it contains too many suspicious jumps
        or other issues.

        Parameters:
            bad (bool, optional): The value to set for the trajectory's 'bad' status. If None, the method
                                  functions as a getter.

        Returns:
            bool: The current 'bad' status of the trajectory if 'bad' is None. Otherwise, sets the 'bad' status.
        """
        if bad is not None:
            self._bad = bad
        return self._bad
