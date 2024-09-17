#
# Robert's plotting code
#
# 31 Oct 2019 by Robert Alfredson
#

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from common import *
from util import *

ANGULAR_VEL_IMG_FILE = "imgs/angular_vel_f%s_%s.png"
ANGULAR_VEL_HIST_IMG_FILE = "imgs/angular_vel_hist%s.png"
TURN_RADIUS_HIST_IMG_FILE = "imgs/turn_radius_hist%s.png"


def vaSlicesByTraining(vas):
    """
    iterates through vas and returns array of dictionaries
    whose keys are VideoAnalysis objects and values are slices to index by
    training. first element of returned array corresponds to entire experiment,
    and thus is slice(None).
    """
    slcGrps = [{va: slice(None) for va in vas}]
    for i in range(len(vas[0].trns)):
        slcGrps.append({va: slice(va.trns[i].start, va.trns[i].stop) for va in vas})
    return slcGrps


def plotTurnRadiusHist(vas, gls=None, yoked=False):
    xMin = 0
    xMax = 10
    for si, slc in enumerate(vaSlicesByTraining(vas)):
        hist = Histogram(vas, gls, si, xMin, xMax)
        for j in range(hist.ng):
            for k, va in enumerate(hist.vaLists[j]):
                dataset = np.array(
                    [
                        turn.radius
                        for turn in va.trx[yoked if hist.multiGroup() else j].turns
                        if si == 0
                        or (turn.start >= slc[va].start and turn.start < slc[va].stop)
                    ]
                )
                hist._updateHistForSingleCol(dataset, j, k)
        hist._calcTTestPVals()
        nVids = [len(vaList) for vaList in hist.vaLists]
        plotHistByGroup(
            hist.bins,
            wts=hist.means,
            err=hist.cis,
            overflow=hist.overflowRange,
            nVids=nVids if nVids != [1] else None,
            lbls=hist.labels,
            xLabel="Turn radius (mm)",
            yLabel="Count%s" % ("" if si == 0 else " (normalized)"),
            title="Turn radius, %s" % ("entire experiment" if si == 0 else "T%i" % si),
            titleSize=10 if si > 0 else 14,
            pVals=hist.pVals,
        )
        if si in [0, len(vas[0].trns)]:
            writeImage(
                TURN_RADIUS_HIST_IMG_FILE
                % (
                    (
                        "%s%s"
                        % (
                            ("" if si == 0 else "_by_trns"),
                            ("_" + basename(va.fn)) if len(vas) == 1 else "",
                        )
                    )
                )
            )


# plot histogram and time-dependent line plot of angular velocity
def plotAngularVelocity(vas, opts, gls=None):
    plotAngularVelocityHist(vas, gls, opts.yoked)
    if opts.angVelOverTime:
        for va in vas:
            for i in range(len(va.trx)):
                plotAngularVelocityOverTime(va, i, opts.angVelRewards)


# plot histogram of angular velocity
def plotAngularVelocityHist(vas, gls=None, yoked=False):
    xMin = 0
    xMax = 600
    subtitles = ["", ", only circular motion"]
    file_suffixes = ["", "_circle"]
    width = 3
    data = []
    for i in range(2):  # all frames, then only those with circular motion
        for si, slc in enumerate(vaSlicesByTraining(vas)):
            hist = Histogram(vas, gls, si, xMin, xMax)
            for j in range(hist.ng):
                if i == 0 and si == 0:
                    data.append([])
                    for va in hist.vaLists[j]:
                        trj = va.trx[yoked if hist.multiGroup() else j]
                        if not np.any(trj.circlePredictions):
                            continue
                        data[j].append(
                            movingAverage(np.abs(trj.diff["theta"]), width) * va.fps
                        )
                for k, va in enumerate(hist.vaLists[j]):
                    trj = va.trx[yoked if hist.multiGroup() else j]
                    if si == 0:
                        dataSlc = slc[va]
                        predictSlc = slice(width, None)
                    else:
                        dataSlc = slice(slc[va].start - width, slc[va].stop - width)
                        predictSlc = slc[va]
                    dataset = (
                        data[j][k][dataSlc]
                        if i == 0
                        else data[j][k][dataSlc][
                            np.flatnonzero(trj.circlePredictions[predictSlc])
                        ]
                    )
                    hist._updateHistForSingleCol(dataset, j, k)
            hist._calcTTestPVals()
            nVids = [len(grpData) for grpData in data]
            plotHistByGroup(
                hist.bins,
                wts=hist.means,
                err=hist.cis,
                overflow=hist.overflowRange,
                nVids=nVids if nVids != [1] else None,
                lbls=hist.labels,
                xLabel="Angular velocity (degrees / second)",
                yLabel="Count%s" % ("" if si == 0 else " (normalized)"),
                title="Angular velocity, %s%s (frameset width: %i)"
                % (
                    ("entire experiment" if si == 0 else "T%i" % si),
                    subtitles[i],
                    width,
                ),
                titleSize=10 if si > 0 else 14,
                pVals=hist.pVals,
            )
            if si in [0, len(vas[0].trns)]:
                writeImage(
                    ANGULAR_VEL_HIST_IMG_FILE
                    % (
                        file_suffixes[i]
                        + (
                            "%s%s"
                            % (
                                ("" if si == 0 else "_by_trns"),
                                ("_" + basename(va.fn)) if len(vas) == 1 else "",
                            )
                        )
                    )
                )


def plotAngularVelocityOverTime(va, iTrx, widthMins=5, angVelRewards=False):
    xLabel = "Duration (min)"
    yLabel = "Angular velocity (degrees / second)"
    trj = va.trx[iTrx]
    if angVelRewards:
        widthMins = 2 / (va.fps * 60)
    width = int(va.fps * widthMins * 60)
    angVel = movingAverage(np.abs(trj.diff["theta"]), width) * va.fps
    numPts = len(angVel)
    tScaler = va.fps * 60
    if not angVelRewards:
        fig = plt.figure(figsize=(18.5, 10.5))
        for trn in va.trns:
            plt.axvspan(
                trn.start / tScaler, trn.stop / tScaler, alpha=0.3, color="gray"
            )
        plt.plot(
            np.linspace(widthMins, widthMins + numPts / tScaler, numPts), angVel, "r-"
        )
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        writeImage(ANGULAR_VEL_IMG_FILE % (trj.f, basename(va.fn)))
    else:
        for i, trn in enumerate(va.trns):
            all_rewards = va._getOn(trn, False, False, i)
            fig = plt.figure(figsize=(18.5, 10.5))
            plt.plot(
                np.linspace(trn.start / tScaler, trn.stop / tScaler, trn.len()),
                np.take(
                    angVel,
                    np.linspace(trn.start, trn.stop, trn.len()).astype(int) - width,
                ),
            )
            plt.plot(
                all_rewards / va.fps / 60.0,
                np.take(angVel, all_rewards - width),
                "x",
                color="#d99102",
                markersize=10,
                mew=3,
            )
            plt.xlim(left=(trn.stop / tScaler - 2, trn.stop / tScaler))
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.title("Training %s (last two minutes)" % (i + 1))
            writeImage(
                ANGULAR_VEL_IMG_FILE % (trj.f, "t%s_%s" % (i + 1, basename(va.fn)))
            )


# plot histogram, using pre-binned data, for one or more groups.
# "wts" (list of numpy ndarrays, max 2) determines bar heights;
# "labels" and "nVids" (both lists) required if len(wts) > 1.
# optional support for error bars and overflow bin range (min and max).
# adapted from this source: https://stackoverflow.com/a/51050772
def plotHistByGroup(
    bins,
    wts,
    lbls=None,
    nVids=None,
    err=None,
    overflow=None,
    xLabel="",
    yLabel="",
    title="",
    titleSize=10,
    pVals=None,
):
    multiGroup = len(wts) > 1
    if multiGroup and None in [lbls, nVids]:
        raise ArgumentError("lbls and nVids required for multi-group histograms")
    limLabels = ["lower", "upper"]
    colormap = plt.cm.brg
    plt.gca().set_prop_cycle(
        plt.cycler("color", colormap(np.linspace(0, 0.9, len(wts))))
    )
    n, bins, patchList = plt.hist(
        [bins[:-1] for _ in wts],
        bins=bins,
        stacked=not multiGroup,
        edgecolor="k",
        label=lbls if multiGroup else None,
        weights=wts,
    )

    def get_lgd():
        return plt.legend(fontsize="x-small")

    def appendLgd(lgd, idx, newText):
        text = lgd.get_texts()[idx]
        text.set_text("%s %s" % (text.get_text(), newText))

    if multiGroup or nVids:
        lgd = get_lgd()
    if nVids:
        if not multiGroup:
            patchList[0].set_label(" ")
            lgd = get_lgd()
        for i, vidCt in enumerate(nVids):
            appendLgd(lgd, i, ("(%s)" if multiGroup else "%s") % ("%s videos" % vidCt))
    if overflow:
        for i in range(len(wts)):
            patches = patchList[i] if type(patchList) is list else patchList
            for j, label in enumerate(limLabels):
                if overflow[i][j] != bins[-j]:
                    binRange = label + " bin range: ({:.2f}, {:.2f})".format(
                        *[overflow[i][j], bins[-j]][::-j]
                    )
                    if multiGroup or (nVids is not None and len(nVids) == 1):
                        appendLgd(lgd, i, "- {}".format(binRange))
                    else:
                        patches[-j].set_label(binRange)
                        get_lgd()
    if err is not None:
        for j, wtSet in enumerate(wts):
            plt.errorbar(
                bins[:-1] + (j + 1) * (bins[1] - bins[0]) / (len(wts) + 1),
                wts[j],
                yerr=err[j],
                ds="steps-mid",
                ecolor="#333c4a",
                lw=1,
                fmt="none",
            )
        plt.ylim(bottom=0)
    if pVals is not None:
        maxHts = np.amax(wts, axis=0)
        starOffset = (bins[1] - bins[0]) * 1 / len(wts)
        for j, pVal in enumerate(pVals):
            strs = p2stars(pVal, nanR="")
            if not strs.startswith("*"):
                continue
            pltText(
                bins[j] + starOffset,
                1.4 * maxHts[j],
                strs,
                ha="center",
                size="small",
                color="0",
                weight="bold",
            )
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title, fontSize=titleSize)


class Histogram:
    """
    computes histogram bins and configures plots
    """

    def __init__(self, vas, gls, si, xMin, xMax):
        self.vas = vas
        self.gls = gls
        self.si = si
        self.xMin = xMin
        self.xMax = xMax
        self._histGroupsAndLabels()
        self.overflowRange = [[xMin, xMax] for _ in range(self.ng)]
        self._initHistArrsAndFigure()

    def multiGroup(self):
        """
        returns boolean indicating whether group labels have been set
        """
        return self.gls is not None and len(self.gls) > 1

    # divides a column of histogram counts by its sum to update the sum to unity.
    def _normalize(self, group, col):
        if self.si:
            self.cts[group][:, col] /= np.sum(self.cts[group][:, col])

    # returns number of groups, labels for histogram legend, and VideoAnalysis
    # lists through which to iterate when binning histograms
    def _histGroupsAndLabels(self):
        if self.multiGroup():
            self.ng, self.labels = len(self.gls), self.gls
            self.vaLists = [
                [va for va in self.vas if va.gidx == i] for i in range(self.ng)
            ]
        else:
            self.ng = len(self.vas[0].trx)
            self.labels, self.vaLists = (
                ["Experimental", "Yoked"] if self.ng > 1 else None
            ), [self.vas] * self.ng

    # initializes numpy arrays for bin heights and errors and sets up figure
    # in which histogram is plotted
    def _initHistArrsAndFigure(self):
        self.nBins = 20 if self.si else 60
        self.binW = (self.xMax - self.xMin) / self.nBins
        self.means = [np.zeros((1, len(self.vaLists[i]))) for i in range(self.ng)]
        self.cis = [np.zeros((self.nBins, 1)) for _ in range(self.ng)]
        self.cts = [
            np.zeros((self.nBins, len(self.vaLists[j]))) for j in range(self.ng)
        ]
        if self.si in [0, 1]:
            fig = plt.figure(figsize=(18.5, 7.5 if self.si else 10.5))
        if self.si > 0:
            plt.subplot(1, len(self.vaLists[0][0].trns), self.si)

    # computes binning/normalization and updates overflow range for one
    # iteration through a column of counts (i.e. single VideoAnalysis object)
    def _updateHistForSingleCol(self, data, j, k):
        data = data[np.isfinite(data)]
        self.cts[j][:, k], self.bins = np.histogram(
            data,
            np.arange(self.xMin, self.xMax + 0.5 * self.binW, self.binW),
            density=self.si != 0,
        )
        self._normalize(j, k)
        self._updateOverflowRange(data, j, k)
        self.means[j], self.cis[j] = [
            np.asarray(result)
            for result in zip(
                *[
                    meanConfInt(self.cts[j][row, :], asDelta=True)[0:2]
                    for row in range(np.shape(self.cts[j])[0])
                ]
            )
        ]

    # apply t-test to each bin for the first and last group in the histogram.
    def _calcTTestPVals(self):
        self.pVals = [
            ttest_ind(self.cts[0][iBin, :], self.cts[-1][iBin, :])[1]
            for iBin in range(self.cts[0].shape[0])
        ]

    # detects data points outside range [xMin, xMax]; augments upper/lower
    # bin counts accordingly, and updates the running overflow range for group j.
    def _updateOverflowRange(self, data, j, k):
        limits, overflow = [self.xMin, self.xMax], 2 * [False]
        s, extrema = [1, -1], [np.nanmin(data), np.nanmax(data)]
        for l, limit in enumerate(limits):
            if not limit or s[l] * limit < s[l] * extrema[l]:
                limits[l] = extrema[l]
            else:
                overflow[l] = True
            if overflow[l]:
                scaleF = (
                    self.cts[j][:, k][self.cts[j][:, k] > 0].min() if self.si else 1
                )
                self.cts[j][-l, k] = (
                    (self.cts[j][-l, k] / scaleF)
                    + (s[l] * data < s[l] * limits[l]).sum()
                ) * scaleF
                self._normalize(j, k)
            if s[l] * extrema[l] < s[l] * self.overflowRange[j][l]:
                self.overflowRange[j][l] = extrema[l]
