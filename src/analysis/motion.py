#
# # -*- coding: utf-8 -*-
#
# circular motion analysis
#
# 31 Oct 2019 by Robert Alfredson
#


import numpy as np
import scipy.io as sio

from src.utils.common import *
from src.utils.util import *


class CircularMotionDetector:
    """
    detects circular motion
    """

    CIRCLE_PREDICT_FILE = "Circle_2019_07_11T1006.jab"
    CIRCULAR_TRX_FILE = "imgs/circular_trx%s.png"

    def __init__(self, trj, opts):
        """
        creates a new circular motion detector
        """
        self.trj = trj
        self.va = self.trj.va
        self.opts = opts

    def analyzeTurns(self):
        """
        labels turns as sub-regions of circular motion consisting of 3 or more
        frames of angular change in a single direction (left or right);
        also computes each turn's radius and determines if the turn is a pivot.
        """
        self.trj.turns = []

        cPred = self.trj.circlePredictions
        thetaDiff = self.trj.diff["theta"]
        with np.errstate(invalid="ignore"):
            sameTurnDRegions = trueRegions(thetaDiff > 0) + trueRegions(thetaDiff < 0)
        for r in sameTurnDRegions:
            cPredSlc = cPred[r.start : r.stop + 1]
            if not np.any(cPredSlc):
                continue
            cRegions = trueRegions(cPredSlc)
            for cReg in cRegions:
                stop = r.start + cReg.stop - 1
                start = r.start + cReg.start
                if (stop - start) > 1:
                    assert (
                        np.unique(np.sign(self.trj.diff["theta"][start:stop])).size == 1
                    )
                    self.trj.turns.append(
                        Turn(
                            start,
                            stop,
                            (
                                "l"
                                if np.sum(self.trj.diff["theta"][start:stop]) < 0
                                else "r"
                            ),
                        )
                    )
        self.calcTurnRadiusAndDetectPivots()

    def calcTurnRadiusAndDetectPivots(self):
        """
        1) calculates radii of labelled turns and 2) classifies pivots, i.e., turns
        having sufficiently small radius and large angular displacement.
        """
        rMax = 1.2  # mm; maximum turn radius for a pivot
        for i, turn in enumerate(self.trj.turns):
            slc = slice(turn.start, turn.length + turn.start)
            angDisp = np.abs(np.sum(self.trj.diff["theta"][slc]))
            self.trj.turns[i].radius = (
                np.sum(self.trj.distFlt[slc])
                / (angDisp / 360)
                / (2 * np.pi * self.va.ct.pxPerMmFloor())
                if angDisp != 0
                else np.nan
            )
            self.trj.turns[i].isPivot = (
                self.trj.turns[i].radius <= rMax and angDisp >= 180
            )

    def classifierLbl(self):
        """
        returns file suffix for JAABA-related files.
        specifies fly number and type, e.g., " _f1_yoked."
        """
        return "_f%s_%s" % (self.va.ef, "exp" if self.trj.f == 0 else "yoked")

    def predictCircularMotion(self):
        """
        detects regions of circular motion in fly trajectory, using either:
          1) numerical algorithm (default)
          2) JAABA, a machine-learning framework that requires a behavior classifier
             file (see variable circlePredictFile)

        if using JAABA, results are saved to disk, but are not analyzed or plotted.
        """
        if self.opts.useJAABA:
            self.predictCircularMotionViaJAABA()
        else:
            flt = self.trj.flt
            d_theta_min = 0.2 * 180 / np.pi
            dist_min = self.va.ct.pxPerMmFloor() * 2 / 7
            maxSlowRatio = 0.9
            min_frames = 4
            self.trj.circlePredictions = np.full(len(self.trj.theta), False)
            distFltFin = toFinite(self.trj.distFlt)
            slowRatio = np.count_nonzero(distFltFin < dist_min) / distFltFin.size
            if slowRatio >= maxSlowRatio:
                print(
                    "insufficient motion for angular analysis:"
                    + " {:.1%} of calc. distances < {}px".format(slowRatio, dist_min)
                )
                return
            for i in range(1, len(self.trj.theta) - 1):
                if True not in np.isnan([flt["theta"][i - 1], flt["theta"][i]]):
                    if self.trj.distFlt[i] < dist_min:
                        continue
                    fwd_idx = min(len(self.trj.theta) - 2, i + 5)
                    bkwd_idx = max(0, i - 4)
                    thetaSlc = self.trj.diff["theta"][bkwd_idx:fwd_idx]
                    thetaSlcFin = toFinite(thetaSlc)
                    numRej = thetaSlc.size - thetaSlcFin.size
                    self.trj.circlePredictions[i] = (
                        np.sum(np.abs(thetaSlcFin) > d_theta_min) > 6 - numRej
                        and np.sum(self.trj.distFlt[bkwd_idx:fwd_idx] >= dist_min)
                        > 6 - numRej
                    )
            isolated_idxs = np.array([], int)
            for i in range(1, min_frames):
                isolated_idxs = np.append(
                    isolated_idxs,
                    seqMatches(
                        self.trj.circlePredictions,
                        np.concatenate(([False], np.full(i, True), [False])),
                    ),
                )
            np.put(self.trj.circlePredictions, isolated_idxs, False)

    def predictCircularMotionViaJAABA(self):
        self.writeJaabaTrxFile()
        dirname = os.path.dirname(self.va.fn)
        perframe_dir = os.path.join(dirname, "perframe")
        scores_file = "scores_Circle.mat"
        predict_file = os.path.join("JAABA", "data", self.opts.circlePredictFile)
        matlab_command_args = [
            "matlab",
            "-nodisplay",
            "-nodesktop",
            "-nosplash",
            "-wait",
            "-r",
        ]
        shutil.copyfile(self.jaabaFilename(), os.path.join(dirname, "trx.mat"))
        if os.path.exists(os.path.join(dirname, scores_file)):
            os.remove(os.path.join(dirname, scores_file))
        if os.path.exists(perframe_dir):
            shutil.rmtree(perframe_dir)
        else:
            os.mkdir(perframe_dir)

        execute(
            matlab_command_args
            + [
                "addpath('JAABA/JAABA-master/JAABA-master/perframe');"
                + "JAABADetect('"
                + dirname
                + "', 'jabfiles',"
                + "'"
                + predict_file
                + "'); copyfile("
                + fullfile(dirname, scores_file)
                + ","
                + fullfile(dirname, "scores_Circle" + self.classifierLbl() + ".mat")
                + "); exit"
            ]
        )

    def circularTrxImg(self):
        """
        saves image of randomly-sampled regions of circular motion
        """
        imgs, hdrs = [], []
        start_idxs = np.array(
            [r.start for r in trueRegions(self.trj.circlePredictions)]
        )
        subsets = [
            [
                start_idxs[idx],
                np.argwhere(
                    self.trj.circlePredictions[start_idxs[idx] :] == False
                ).flatten()[0]
                + start_idxs[idx],
            ]
            for idx in np.random.choice(list(range(0, len(start_idxs))), 40)
        ]
        for idx, subset in enumerate(subsets):
            img = readFrame(self.va.cap, subset[0])
            xy = self.trj.xy(subset[0], subset[1])
            cv2.polylines(img, xy2Pts(*xy), False, COL_R)
            imgs.append(self.va.extractChamber(img))
            hdrs.append(
                "%sm (%d-%d)"
                % (round(self.va._f2min(subset[1] - 0), 1), subset[0], subset[1])
            )
        img = combineImgs(imgs, hdrs=hdrs, hdrL=self.va.fn)[0]
        writeImage(CircularMotionDetector.CIRCULAR_TRX_FILE % self.classifierLbl(), img)

    def jaabaFilename(self):
        """
        returns filename used for trajectory data that is passed to JAABA
        """
        return replaceCheck(AVI_X, self.classifierLbl() + "_JAABA.mat", self.va.fn)

    def writeJaabaTrxFile(self):
        """
        saves fly trajectory data in a format readable by JAABA
        """
        ofn = self.jaabaFilename()
        print("\nwriting JAABA-compatible trajectory file %s..." % basename(ofn))
        start_index = self.trj.trxStartIdx
        fxy = self.va.xf.f2t(self.trj.x, self.trj.y)
        n_frames, px_per_mm = len(fxy[0][start_index:]), 1
        x, y = fxy[0][start_index:], fxy[1][start_index:]
        a, b = self.trj.h[start_index:] / 4, self.trj.w[start_index:] / 4
        self.trj._chooseOrientations()
        theta = math.pi * self.trj.theta[start_index:] / 180.0
        d = dict(
            trj=dict(
                x=x,
                x_mm=x / px_per_mm,
                y=y,
                y_mm=y / px_per_mm,
                theta=theta,
                theta_mm=theta,
                a=a,
                a_mm=a / px_per_mm,
                b=b,
                b_mm=b / px_per_mm,
                endframe=n_frames,
                nframes=n_frames,
                firstframe=1,
                pxpermm=px_per_mm,
                fps=self.va.fps,
                off=0,
                start_index=start_index,
                dt=np.full((1, n_frames - 1), 1 / self.va.fps),
            )
        )
        sio.savemat(ofn, d)


class DataCombiner:
    """
    calculates and outputs motion analytics aggregated across multiple flies
    """

    def __init__(self, va, post_bucket_len_min):
        """
        Creates a new DataCombiner

        Arguments:
          - va: VideoAnalysis instance containing data to be analyzed
          - post_bucket_len_min: length of post-training buckets for number of rewards
                              (in minutes)
        """
        self.va = va
        self.post_bucket_len_min = post_bucket_len_min

    def combineResults(
        self,
        attr,
        key,
        sv_name,
        header="",
        silent=False,
        inclPreAndPost=False,
        postOffsets=None,
        asPct=False,
    ):
        """
        calculates averages by sync bucket for analysis data stored per video frame;
        see combineAgaroseResults and combineCircleResults

        Arguments:
          - attr: attribute of the Trajectory instance where the data are stored.
          - key: key to use for accessing the data from the Trajectory instance attribute.
                 If nested, use periods to separate the levels, e.g., key1.key2
          - tp: data type name (e.g. "onAgarose") to use for saved, combined results
          - header: header to use if printing results to console
          - silent: whether to print results to console
          - inclPreAndPost: whether to analyze pre- and post-training data
          - postOffsets: number of frames to trim from the start (i.e. ignore)
                         when analyzing post-training
        """
        if not silent:
            assert len(header) > 0
            print("\n%% frames %s:" % header)
        self.predictions = []
        for trj in self.va.trx:
            if trj.bad():
                self.predictions.append([])
                continue
            self.predictions.append(deep_access(trj, attr, key))
        SB, pre, post, postBB = [
            "%s%s" % (sv_name, sect) for sect in ("SB", "Pre", "Post", "PostByBucket")
        ]
        df = self.va._numRewardsMsg(True, silent=True)
        if postOffsets is None:
            postOffsets = [0 for _ in range(len(self.va.flies))]
        setattr(self.va, SB, [])
        if inclPreAndPost:
            for sect in (pre, post, postBB):
                setattr(self.va, sect, [[] for _ in range(len(self.va.flies))])
        for i, t in enumerate(self.va.trns):
            getattr(self.va, SB).append([])
            for j, f in enumerate(self.va.flies):
                postEnd = (
                    self.va.trns[i + 1].start
                    if i < len(self.va.trns) - 1
                    else self.va.nf
                )
                fi, n, _ = self.va._syncBucket(t, df)
                la = min(t.stop, int(t.start + n * df))
                if fi is None:
                    self.va._append(
                        getattr(self.va, SB)[i],
                        [np.nan for _ in range(int(n - 1))],
                        0,
                        n - 1,
                    )
                    getattr(self.va, pre)[j].append(np.nan)
                    getattr(self.va, post)[j].append(np.nan)
                    continue
                resByTrn = []
                if inclPreAndPost:
                    if i == 0:
                        self.calcRatio(
                            j,
                            self.va.startPre,
                            t.start,
                            getattr(self.va, pre)[j],
                            asPct=asPct,
                        )
                    startPost = self.va.fns["startPost"][i] + postOffsets[j]
                    self.calcRatio(
                        j, startPost, postEnd, getattr(self.va, post)[j], asPct=asPct
                    )
                    self.addPostResultsByBucket(j, startPost, postBB)
                while fi + df <= la:
                    self.calcRatio(j, fi, fi + df, resByTrn, asPct=asPct)
                    fi += df
                self.va._append(getattr(self.va, SB)[i], resByTrn, 0, n - 1)
                if silent:
                    continue
                print(t.name())
                self.va._printBucketVals(
                    getattr(self.va, SB)[i][j], f, msg="f%d" % (f + 1), prec=3
                )

    def _visit_durations_in_range(self, region_slices, start, stop):
        return [
            sl.stop - sl.start for sl in region_slices if sl.start >= start and sl.start < stop and sl.stop <= stop
        ]

    def calcRatio(self, fIdx, fi, la, results, asPct=False):
        """
        Calculates average value of self.predictions over the given range of video
        frames and appends to results array

        Arguments:
          - fIdx: fly index (0 for experimental, 1 for yoked control)
          - fi: index of starting frame of video range
          - la: index of ending frame of video range
          - results: array to which to append results
        """
        pSlice = self.predictions[fIdx][fi:la]
        results.append(np.nanmean(pSlice) * (100 if asPct else 1))

    def addPostResultsByBucket(self, fIdx, startPost, attrName):
        """
        calculates averages by post bucket for per-frame data

        Arguments:
          - fIdx: fly index (0 for experimental, 1 for yoked control)
          - startPost: index of starting frame of post-training
          - attrName: attribute of VideoAnalysis to update
        """
        bucketLn = self.va._min2f(self.post_bucket_len_min)
        for _ in range(2):
            self.calcRatio(
                fIdx, startPost, startPost + bucketLn, getattr(self.va, attrName)[fIdx]
            )
            startPost += bucketLn

    def regionContactByTp(self, region_label, tp):
        reg_label_caps = region_label.capitalize()
        return {
            "edge": {
                "key": "boundary_contact",
                "sv_name": f"on{reg_label_caps}",
                "desc": "ellipse edge",
            },
            "ctr": {
                "key": "boundary_contact_ctr",
                "sv_name": f"on{reg_label_caps}Ctr",
                "desc": "body center",
            },
        }[tp]
    
    def combineOnRegionVisitDurations(self, region_label, tp):
        """
        Calculates the mean duration (in seconds) of each visit across a region
        border, broken down exactly the same way as on-region percentages
        (sync buckets, pre-training, post-training, post buckets).

        Results are stored in attributes whose base name is the *same* as the
        percentage metric but with the suffix 'Dur', e.g.
            onAgaroseSB      →  onAgaroseDurSB
            onAgarosePre     →  onAgaroseDurPre  …etc.
        """
        reg_caps = region_label.capitalize()
        base = f"on{reg_caps}Edge" if tp == "edge" else f"on{reg_caps}Ctr"
        sv_name = f"{base}Dur"          # attribute stem for durations

        # Attribute scaffolding
        SB, pre, post, postBB = [
            f"{sv_name}{sect}" for sect in ("SB", "Pre", "Post", "PostByBucket")
        ]
        for attr in (SB, pre, post, postBB):
            setattr(self.va, attr, [])
            # input(f'set up attribute {attr}')

        # pre‑training range
        pre_intvl = slice(self.va.startPre, self.va.trns[0].start)

        df = self.va._numRewardsMsg(True, silent=True)     # frames / bucket
        for i, trn in enumerate(self.va.trns):

            getattr(self.va, SB).append([])                # bucket container

            # frame indices that define sync buckets in this training
            fi_start, n_buckets, _ = self.va._syncBucket(trn, df)
            la = min(trn.stop, int(trn.start + n_buckets * df))

            # loop over flies
            for j, trj in enumerate(self.va.trx):

                # ensure container rows exist
                for attr in (pre, post, postBB):
                    if len(getattr(self.va, attr)) <= j:
                        getattr(self.va, attr).append([])

                # skip bad flies
                if fi_start is None or trj.bad():
                    getattr(self.va, SB)[i].append(n_buckets * [np.nan])
                    getattr(self.va, pre)[j].append(np.nan)
                    getattr(self.va, post)[j].append(np.nan)
                    continue

                # ---- extract all visit slices for this fly ----
                visit_slices = trj.boundary_event_stats[region_label]["tb"][tp][
                    "boundary_contact_regions"
                ]

                # pre‑training mean
                pre_durs = self._visit_durations_in_range(
                    visit_slices, pre_intvl.start, pre_intvl.stop
                )
                getattr(self.va, pre)[j].append(
                    np.nan
                    if len(pre_durs) == 0
                    else self.va._f2s(np.mean(pre_durs))
                )

                # post‑training mean (whole post period)
                post_start = self.va.fns["startPost"][i]
                post_end = (
                    self.va.trns[i + 1].start
                    if i < len(self.va.trns) - 1
                    else self.va.nf
                )
                post_durs = self._visit_durations_in_range(
                    visit_slices, post_start, post_end
                )
                getattr(self.va, post)[j].append(
                    np.nan
                    if len(post_durs) == 0
                    else self.va._f2s(np.mean(post_durs))
                )

                # post‑training, bucketed
                bucket_len = self.va._min2f(self.post_bucket_len_min)
                for _ in range(2):
                    pb_durs = self._visit_durations_in_range(
                        visit_slices, post_start, post_start + bucket_len
                    )
                    getattr(self.va, postBB)[j].append(
                        np.nan
                        if len(pb_durs) == 0
                        else self.va._f2s(np.mean(pb_durs))
                    )
                    post_start += bucket_len

                # ---- sync buckets inside training ----
                bucket_means = []
                fi = fi_start
                while fi + df <= la:
                    sb_durs = self._visit_durations_in_range(
                        visit_slices, fi, fi + df
                    )
                    bucket_means.append(
                        np.nan if len(sb_durs) == 0 else self.va._f2s(np.mean(sb_durs))
                    )
                    fi += df
                n_missing = n_buckets - len(bucket_means)
                if n_missing > 0:
                    bucket_means.extend([np.nan] * n_missing)
                getattr(self.va, SB)[i].append(bucket_means)

    def combineOnRegionResults(self, region_label, tp):
        """
        calculates 1) average time fly spent over an ROI for each sync bucket,
        2) average pre-training dwell time on an ROI, and 3) number of pre-
        training ROI visits.

        Parameters:
        - region_label: the label associated with the ROI.
        """
        tRg = list(range(len(self.va.trns)))
        fRg = list(range(len(self.va.flies)))
        fHdr = "  f%d: "
        preTrnVisits = []
        for trj in self.va.trx:
            if trj.bad():
                preTrnVisits.append([])
                continue
            preTrnVisits.append(
                trueRegions(
                    trj.boundary_event_stats[region_label]["tb"][tp][
                        "boundary_contact"
                    ][self.va.startPre : self.va.trns[0].start]
                    == True
                )
            )
        visitAvgs = np.array(
            [
                meanConfInt([(v.stop - v.start) for v in visitL], asDelta=True)
                for visitL in preTrnVisits
            ]
        )
        postOffsets = [
            (
                intR(visitAvg[0] + visitAvg[1])
                if np.count_nonzero(np.isnan(visitAvg)) == 0
                else 0
            )
            for visitAvg in visitAvgs
        ]

        def flyMsg(data, prec=3, newline=False):
            for j, f in enumerate(self.va.flies):
                if isinstance(data[j], (list, np.ndarray, tuple)):
                    val, delta = data[j][0:2]
                else:
                    val, delta = data[j], None
                fmt = "%%.%df" % prec
                print(
                    fHdr % (f + 1)
                    + fmt % val
                    + (" ± " + fmt % delta if delta is not None else "")
                )
            if newline:
                print("")

        self.combineResults(
            "boundary_event_stats",
            "%s.tb.%s.%s" % (region_label, tp, "boundary_contact"),
            self.regionContactByTp(region_label, tp)["sv_name"],
            silent=True,
            inclPreAndPost=True,
            postOffsets=postOffsets,
            asPct=True,
        )
        sv_name = self.regionContactByTp(region_label, tp)["sv_name"]
        sb = getattr(self.va, "%sSB" % sv_name)
        pre = getattr(self.va, "%sPre" % sv_name)
        post = getattr(self.va, "%sPost" % sv_name)

        print(
            "\n%% frames with fly positioned past %s border by training (%s):"
            % (region_label, self.regionContactByTp(region_label, tp)["desc"])
        )
        print("pre-training:")
        flyMsg([getattr(self.va, "%sPre" % sv_name)[j][0] for j in fRg])
        for i, t in enumerate(self.va.trns):
            print(t.name())
            for j in range(len(self.va.flies)):
                print(
                    fHdr % (j + 1)
                    + "%.3f (post: %.3f)"
                    % (
                        np.mean(getattr(self.va, "%sSB" % sv_name)[i][j]),
                        getattr(self.va, "%sPost" % sv_name)[j][i],
                    )
                )
        print("averages across all trainings:")
        flyMsg([meanConfInt([sb[i][j] for i in tRg]) for j in fRg])
        print("averages across all non-training periods:")
        flyMsg([meanConfInt([pre[j][0]] + [post[j][i] for i in tRg]) for j in fRg])
        print(f"Number pre-training visits across {region_label} border:")
        flyMsg([len(visitList) for visitList in preTrnVisits], prec=0)
        print(f"Avg duration (secs) of visits across {region_label} border:")
        flyMsg([self.va._f2s(visitAvg) for visitAvg in visitAvgs], newline=True)

    def combineCircleResults(self):
        """
        for each sync bucket, calculates the percentage of fly's trajectory where
        circular motion was detected
        """
        self.combineResults(
            "circlePredictions", "with circular behavior by sync bucket"
        )

    def combineTurnResults(self):
        """
        determines which turns coincide with rewards, and computes:
          1) left vs. right proportion of turns for each sync bucket and for the
             first fifteen rewarded turns for each training
          2) average turn radius for each sync bucket
        """

        def wrapBucketPrint(resultType, i, j, f, prec):
            self.va._printBucketVals(
                self.va.turnResults[resultType][i][j], f, msg="f%d" % (f + 1), prec=prec
            )

        initialSkewLim = 16
        print("\nproportion rewarded left turns by sync bucket:")
        df = self.va._numRewardsMsg(True, silent=True)
        self.va.turnResults = {"lTurnPct": [], "meanTurnRadii": [], "numPivots": []}
        for i, t in enumerate(self.va.trns):
            print(t.name())
            for res in self.va.turnResults:
                self.va.turnResults[res].append([])
            for j, f in enumerate(self.va.flies):
                lAndRTurns = {
                    d: [turn for turn in self.va.trx[j].turns if turn.direction is d]
                    for d in ["l", "r"]
                }
                fi, n, _ = self.va._syncBucket(t, df)
                la = min(t.stop, int(t.start + n * df))
                # indexing scheme for turns and initialSkew is (0: left, 1: right)
                rewardIdxs, turns = self.va._getOn(t, False, False, f), [[], []]
                initialSkew = np.array([0, 0])
                startIdxs = np.zeros(2, int)
                nearestTurns = np.ones(2, int)
                for rIdx in rewardIdxs:
                    for k, d in enumerate(lAndRTurns):
                        for s in range(startIdxs[k], len(lAndRTurns[d])):
                            if (
                                lAndRTurns[d][s].start > rIdx
                                and lAndRTurns[d][s - 1].start < rIdx
                            ):
                                nearestTurns[k] = s - 1
                                startIdxs[k] = s - 1
                                break
                    nearestTurnsIdxAndLen = [
                        [
                            getattr(lAndRTurns[key][nearestTurns[idx]], idxAttr)
                            for idxAttr in ["start", "length"]
                        ]
                        for idx, key in enumerate(lAndRTurns)
                    ]
                    maxIdx = np.argmax(
                        [fmAndLn[0] for fmAndLn in nearestTurnsIdxAndLen]
                    )
                    if rIdx < np.sum(nearestTurnsIdxAndLen[maxIdx]):
                        turns[maxIdx].append(nearestTurnsIdxAndLen[maxIdx][0])
                        if np.count_nonzero(np.sum(turns)) < initialSkewLim:
                            initialSkew[maxIdx] += 1
                resultsByTraining = {key: [] for key in self.va.turnResults}
                if fi is None:
                    for res in self.va.turnResults:
                        self.va._append(
                            self.va.turnResults[res][i][j], [], 0, int(n - 1)
                        )
                    continue
                while fi + df <= la:
                    syncTurns = [
                        np.flatnonzero(
                            np.all([turns[k] >= fi, turns[k] < fi + df], axis=0)
                        ).size
                        for k in range(2)
                    ]
                    resultsByTraining["lTurnPct"].append(
                        np.nan
                        if np.sum(syncTurns) == 0
                        else syncTurns[0] / np.sum(syncTurns)
                    )
                    turnsInSB = [
                        turn
                        for turn in self.va.trx[j].turns
                        if turn.start >= fi
                        and turn.stop <= fi + df
                        and not np.isnan(turn.radius)
                    ]
                    resultsByTraining["meanTurnRadii"].append(
                        np.mean([turn.radius for turn in turnsInSB])
                    )
                    resultsByTraining["numPivots"].append(
                        np.count_nonzero([turn.isPivot for turn in turnsInSB])
                    )
                    fi += df
                for res in self.va.turnResults:
                    self.va._append(
                        self.va.turnResults[res][i],
                        resultsByTraining[res],
                        0,
                        int(n - 1),
                    )
                wrapBucketPrint("lTurnPct", i, j, f, 3)
                skewTot = np.sum(initialSkew)
                print(
                    "  proportion of left turns among the first %i" % (skewTot),
                    "rewarded turns: %.2f" % (initialSkew[0] / skewTot),
                )
                print("  total rewarded turns: %i" % np.count_nonzero(np.sum(turns)))
        for result in [
            {"name": "average turn radius", "type": "meanTurnRadii", "prec": 2},
            {"name": "number pivots", "type": "numPivots", "prec": 0},
        ]:
            print("\n%s by sync bucket" % result["name"])
            for i, t in enumerate(self.va.trns):
                print(t.name())
                for j, f in enumerate(self.va.flies):
                    wrapBucketPrint(result["type"], i, j, f, result["prec"])


class Turn:
    """
    single turn of a fly
    """

    def __init__(self, start, stop, direction):
        """
        create a new turn

        arguments
          - start and stop: frame indices of the beginning/end of turn
          - direction: 'l' or 'r'
        """
        self.start = start
        self.stop = stop
        self.length = stop - start
        self.direction = direction  # 'l' or 'r' for left or right
