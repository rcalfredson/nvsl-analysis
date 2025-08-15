# standard libraries
import collections
import csv
import gc
from multiprocessing.pool import ThreadPool
import os
import random
import re
import sys
import textwrap
import timeit

# third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylru
import scipy.io as sio

# custom modules and functions
from src.analysis.boundary_contact import runBoundaryContactAnalyses
from src.utils.common import (
    CT,
    Xformer,
    cVsA,
    cVsA_l,
    filterDataAndCalcDiffs,
    flyDesc,
    frame2hm,
    skipMsg,
)
from src.utils.constants import (
    CONTACT_BUFFER_OFFSETS,
    MIDLINE_BOUNDARY_DIST,
    AGAROSE_BOUNDARY_DIST,
    HEATMAP_DIV,
    LGC2,
    MIDLINE_XING2,
    POST_SYNC,
    RDP_MIN_LINES,
    RI_START,
    RI_START_POST,
    SPEED_ON_BOTTOM,
    ST,
    VBA,
)
from src.analysis.contact_event_training_comparison import (
    ContactEventTrainingComparison,
)
from src.analysis.ellipse_to_boundary_dist import (
    TrjDataContainer,
    VaDataContainer,
    WALL_ORIENTATION_FOR_TURN,
)
from src.analysis.large_turns import RewardCircleAnchoredTurnFinder
from src.analysis.motion import CircularMotionDetector, DataCombiner
from src.analysis.trajectory import Trajectory
from src.analysis.training import Training
from src.plotting.trx_plotter import TrxPlotter
import src.utils.util as util
from src.utils.util import (
    ArgumentError,
    VideoError,
    AVI_X,
    COL_G,
    COL_R,
    COL_W,
    COL_Y,
    DIGITS_ONLY,
    SPACES_AFTER_TAB,
)
from src.utils.util import error
from src.analysis.va_spd_calculator import VASpeedCalculator
from src.analysis.va_turn_directionality_collator import VATurnDirectionalityCollator
from src.analysis.va_turn_prob_dist_collator import VATurnProbabilityDistanceCollator

BACKGROUND_CHANNEL = 0  # blue (default for tracking)
SYNC_CTRL = False  # whether to start sync buckets after control reward
DELAY_IMG_FILE = "imgs/delay.png"
TRX_IMG_FILE = "imgs/%s__t%d.png"

per_va_processing_times = []
per_bnd_processing_times = []
per_lgt_processing_times = []


# analysis of a single video
class VideoAnalysis:
    _ON_KEY = re.compile(r"^v[1-9]\d*(\.\d+)?$")  # excludes, e.g., 'v0'

    numPostBuckets, numNonPostBuckets = None, 4
    rpiNumPostBuckets, rpiNumNonPostBuckets = None, 0

    fileCache = pylru.lrucache(1)
    currFn, currAImg = None, None

    # f: fly to analyze, e.g., for HtL (0-19); None: regular chamber
    def __init__(self, fn, gidx, opts, f=None):
        """
        Initializes the VideoAnalysis class, setting up the analysis environment for a given video.

        This constructor takes the filename of the video to be analyzed, a group index, analysis options,
        and optionally, a specific fly to analyze. It performs initial setup tasks such as loading data,
        initializing trajectories, and setting default options based on the analysis requirements. It also
        handles conditional analyses such as circular and turning motion analysis, boundary contact analyses,
        and the generation of various outputs (e.g., annotated videos, MAT files, JAABA trx files).

        Parameters:
        - fn (str): Filename of the video to be analyzed.
        - gidx (int): Group index used to identify the video within a set or batch of analyses.
        - opts (object): An object containing options that dictate how the analysis should be performed.
        - f (int, optional): Specifies a particular fly to analyze within the video. If None, a regular
                             chamber analysis is assumed (a single-fly video)

        The method prints the start of the analysis process and skips the analysis if the trajectory data
        for the first fly is marked as bad. It performs a series of analyses based on the provided options,
        including reward calculations, boundary contact events, turning motion analysis, and more. It
        concludes by cleaning up boundary contact data and logging the processing time if timing measurements
        are enabled.

        Note:
        This method is crucial for initializing the analysis process, setting up the data and options
        required for a comprehensive study of fly behavior within the experimental video.
        """
        if opts.timeit:
            start_t = timeit.default_timer()
        print(
            "=== analyzing %s%s ===\n"
            % (util.basename(fn), "" if f is None else ", fly %d" % f)
        )

        self.opts = opts
        self.gidx, self.f = gidx, f
        self._skipped = True  # for early returns
        self._loadData(fn)
        if self.isyc:
            print("yoked control")
            return
        self.flies = (0,) if self.noyc else (0, 1)
        if opts.annotate:
            self._writeAnnotatedVideo()
            return
        self.setOptionDefaults()

        needs_tp = (
            opts.contact_geometry == "horizontal" and opts.turn_prob_by_dist
        ) or (opts.contact_geometry == "circular" and opts.outside_circle_radii)
        if needs_tp:
            opts.chooseOrientations = True

        if (opts.cTurnAnlyz or opts.outside_circle_radii) and not opts.wall:
            setattr(
                opts,
                "wall",
                f"{CONTACT_BUFFER_OFFSETS['wall']['min']}|"
                f"{CONTACT_BUFFER_OFFSETS['wall']['max']}",
            )
            opts.chooseOrientations = True
        if opts.wall or opts.boundary or opts.agarose:
            opts.chooseOrientations = True
        self._initTrx()
        self._readNoteFile(fn)  # possibly overrides whether trajectories bad
        if opts.circle:
            self._analyzeCircularAndTurningMotion()
        self.runBoundaryContactAnalyses()
        if opts.jaabaOut:
            for trj in self.trx:
                CircularMotionDetector(trj, opts).writeJaabaTrxFile()
            return
        elif opts.matFile:
            self._writeMatFile()
            return
        elif opts.play:
            self._playAnnotatedVideo()
            return
        if self.trx[0].bad():
            print("\n*** skipping analysis ***")
            return
        print()

        self._skipped = False
        if self.circle or self.choice:
            self._analysisImage()
        self.byBucket()
        if self.circle:
            self.calcRewardsPre()
            self.bySyncBucket()
            self.bySyncBucket2()  # pass True to get maximum distance reached
            self.byPostBucket()
            self.bySyncBucketMedDist()
            self.byReward()
            self.byTraining()
            if opts.plotTrx:
                TrxPlotter(self, self.opts).plotTrx(mode="grid")
            if opts.plotThm:
                TrxPlotter(self, self.opts).plotTrx(mode="hm")
            if opts.plotThmNorm:
                TrxPlotter(self, self.opts).plotTrx(mode="hm_norm")
            if opts.rdp:
                self.rdpAnalysis()
            self.speed()
            VASpeedCalculator(self, opts).calcSpeedsOverSBs()
            self.rewardsPerMinute()
            if opts.outside_circle_radii:
                self.analyzeOutsideCirclePeriods()
            if opts.prefCircleSlideRad:
                self._aggregate_slide_circle_metrics()
            if opts.rpd:
                self._rewards_per_distance()
        for opt, evt_name in (
            ("wall", "wall_contact"),
            ("agarose", "agarose_contact"),
            ("boundary", "boundary_contact"),
            ("turn", "turn"),
        ):
            if getattr(opts, opt):
                if opt == "turn":
                    boundary_tps = getattr(opts, opt)
                    for boundary_tp in boundary_tps:
                        if boundary_tp == "wall":
                            continue
                        turn_tps = ["inside_line", "outside_line", "edge", "ctr"]
                        for turn_tp in turn_tps:
                            self.contactEventsBySyncBucket(
                                "%s_%s_turn" % (turn_tp, boundary_tp)
                            )
                            if opts.turn_dir and boundary_tp != "wall":
                                self.determineTurnDirectionality(boundary_tp, turn_tp)
                else:
                    self.contactEventsBySyncBucket(evt_name)
        if opts.turn_dir:
            turn_dir_collator = VATurnDirectionalityCollator(self, self.opts)
            turn_dir_collator.calcTurnDirectionalityMetrics()

        for bnd_tp in ("agarose", "boundary"):
            if not getattr(opts, bnd_tp):
                continue
            self._analyzeRegionPreference(bnd_tp)
        if opts.cTurnAnlyz:
            if opts.timeit:
                lg_turn_start = timeit.default_timer()
            turn_finder = RewardCircleAnchoredTurnFinder(
                self,
                opts.min_turn_speed,
                opts.lg_turn_nbins,
                int(opts.end_turn_before_recontact),
                opts.lg_turn_plots,
            )
            turn_finder.calcLargeTurnsAfterCircleExit()
            if opts.timeit:
                per_lgt_processing_times.append(timeit.default_timer() - lg_turn_start)
        if needs_tp:
            turn_prob_dist_collator = VATurnProbabilityDistanceCollator(self, opts)
            turn_prob_dist_collator.calcTurnProbabilitiesByDistance()
        if self.choice:
            if self.openLoop:
                self.posPrefOL()
            else:
                self.posPref()
            if self.rectangle:
                self.speed()
            if opts.plotTrx:
                if opts.ol:
                    self.plotTrx()
                else:
                    self.plotYOverTime()
            if opts.ol:
                self.bySyncBucket2(True)
        if opts.move:
            self.distance()
        if opts.hm:
            self.calcHm()

        if opts.delayCheckMult is not None:
            self.delayCheck()

        self.clean_up_boundary_contact_data(verbose=False)
        if opts.timeit:
            proc_time = timeit.default_timer() - start_t
            print("va proc time:", proc_time)
            per_va_processing_times.append(proc_time)

    # set option defaults depending on protocol
    def setOptionDefaults(self):
        if hasattr(self.opts, "_dfltsSet"):
            return
        self.opts._dfltsSet = True
        if self.opts.numBuckets is None:
            self.opts.numBuckets = 1 if self.choice else 12
        if self.opts.piBucketLenMin is None:
            self.opts.piBucketLenMin = 10 if self.choice else 2

    def clean_up_boundary_contact_data(self, *, verbose: bool = False):
        """
        Remove heavy per-frame / per-event data that are no longer needed.
        If verbose=True, print a human-readable report of everything removed.
        """

        # ─────────────────────────────────────────────────────────────
        # 1.  Keys that can be safely discarded
        # ─────────────────────────────────────────────────────────────
        boundary_related_keys = (
            "dist_to_boundary",
            "ellipse_and_boundary_pts",
            "bounds",
            "original_boundary_contact",
            "boundary_contact",
            "near_contact",
            "boundary_contact_regions",
            "near_contact_regions",
            "contact_start_idxs",
            "near_contact_start_idxs",
            "closest_boundary_indices",
            # new since circular-contact analysis
            "training_ranges",
            "circle_radius_px",
            "circle_centre_px",
        )

        turn_related_keys = (
            "turning",
            "turning_indices",
            "near_turning",
            "near_turning_indices",
            "rejection_reasons",
            "total_vel_angle_deltas",
            "frames_to_skip",
        )

        # ─────────────────────────────────────────────────────────────
        # 2.  Helpers
        # ─────────────────────────────────────────────────────────────
        def _prune(
            stats_dict: dict, delete_turn: bool, path: str, log: collections.defaultdict
        ):
            """
            Remove keys from ONE leaf dict; update log if verbose.
            """
            for k in boundary_related_keys:
                if k in stats_dict:
                    if verbose:
                        log[path][k] = sys.getsizeof(stats_dict[k])
                    stats_dict.pop(k)

            if delete_turn:
                for k in turn_related_keys:
                    if k in stats_dict:
                        if verbose:
                            log[path][k] = sys.getsizeof(stats_dict[k])
                        stats_dict.pop(k)

        # where we accumulate bytes removed when verbose
        removal_log = collections.defaultdict(dict)

        # ─────────────────────────────────────────────────────────────
        # 3.  Walk through every trajectory
        # ─────────────────────────────────────────────────────────────
        for trj_idx, trj in enumerate(self.trx):
            if not hasattr(trj, "boundary_event_stats"):
                continue

            for boundary_tp, orient_dict in trj.boundary_event_stats.items():

                # honour “export raw events” flags (e.g.  --wall_eg)
                if getattr(self.opts, f"{boundary_tp}_eg", None):
                    continue

                for boundary_orientation, ep_dict in orient_dict.items():
                    for ellipse_ref_pt, leaf in ep_dict.items():

                        # Decide whether turn-specific keys should be removed
                        delete_turn_here = False
                        if self.opts.turn:
                            for turn_boundary in self.opts.turn:
                                if (
                                    boundary_tp == turn_boundary
                                    and boundary_orientation
                                    == WALL_ORIENTATION_FOR_TURN[turn_boundary]
                                ):
                                    delete_turn_here = True
                                    break

                        path = (
                            f"fly{trj_idx}.{boundary_tp}"
                            f".{boundary_orientation}.{ellipse_ref_pt}"
                        )

                        # ─── normal branches (wall / boundary / agarose …) ───
                        if boundary_tp != "circle":
                            _prune(leaf, delete_turn_here, path, removal_log)
                            continue

                        # ─── circle branch: one more level for each radius ───
                        for r_mm, stats in leaf.items():  # stats is per-radius dict
                            _prune(
                                stats, delete_turn_here, f"{path}.r{r_mm}", removal_log
                            )

        # ─────────────────────────────────────────────────────────────
        # 4.  Optional GC kick & verbose summary
        # ─────────────────────────────────────────────────────────────
        gc.collect()

        if verbose and removal_log:
            total_bytes = 0
            print("\n✂  Clean-up report:")
            for scope, kv in removal_log.items():
                freed = sum(kv.values())
                total_bytes += freed
                print(f"  • {scope}: removed {len(kv)} keys, ~{freed/1e6:.2f} MB")
                for k, sz in kv.items():
                    print(f"      – {k:<24} {sz/1e6:.3f} MB")
            print(f"  → TOTAL freed ≈ {total_bytes/1e6:.2f} MB\n")

    # returns whether analysis was skipped
    def skipped(self):
        return self._skipped

    # writes images with some calculated rewards
    def calcRewardsImgs(self):
        for trx in self.trx:
            trx.calcRewardsImg()

    # note: called for non-skipped analysis only
    def _analysisImage(self):
        if self.fn != self.currFn:
            VideoAnalysis.currFn = self.fn
            img = self.aimg = VideoAnalysis.currAImg = self.frame.copy()
        else:
            img, self.aimg = self.currAImg, None
        for t in self.trns:
            t.annotate(img, ctrl=None, verbose=VBA)

    # extractChamber() extracts the experimental fly's chamber floor plus the
    #  given border from the given frame
    def _createExtractChamber(self):
        (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.ef)
        bw = {CT.regular: 0, CT.htl: 15, CT.large: 35, CT.large2: 39}[self.ct]

        def exCh(frame, borderw=bw):
            return util.subimage(
                frame, (xm - borderw, ym - borderw), (xM + borderw, yM + borderw)
            )

        self.extractChamber = exCh

    def _loadData(self, fn):
        """
        Loads video data and initializes trajectory and training session information.

        This method performs several key initializations for video analysis:
        - Opens the video file and retrieves its frame rate and the initial frame.
        - Checks the cache for pre-loaded data (such as trajectory and experimental data)
          associated with the video file to avoid redundant loading. If not found in cache,
          it loads and caches this data.
        - Extracts and processes trajectory raw data and experimental protocol information,
          setting up various attributes such as the number of frames, experimental flies,
          chamber type, and specific protocol parameters (e.g., circle or open loop).
        - Determines the type of experimental protocol (regular, circle, open loop) and
          initializes `Training` instances for each training session defined in the protocol.
        - Calculates and sets up frame indices for reward events and, if applicable, off events
          for open-loop protocols.
        - Prints video analysis initialization information, including video length, frame rate,
          and chamber type, as well as prepares a report on the training sessions.

        Parameters:
        - fn (str): The filename of the video to be analyzed.

        This method heavily relies on utility functions from the `util` module for data loading
        and processing, demonstrating tight integration with custom modules for handling
        video and experimental data.

        Raises:
        - AssertionError: If any consistency checks fail (e.g., unexpected number of experimental
          flies based on protocol information).
        - Various custom exceptions for specific error conditions (e.g., out-of-range fly numbers),
          handled by utility functions.

        Note:
        This method is designed to be called once during the initialization of a `VideoAnalysis`
        instance, setting the stage for further detailed analysis of video and trajectory data.
        """
        self.cap = util.videoCapture(fn)
        self.fps, self.fn = util.frameRate(self.cap), fn
        self.frame, self.bg = util.readFrame(self.cap, 0), None

        if fn not in self.fileCache:
            self.fileCache[fn] = [
                util.unpickle(util.replaceCheck(AVI_X, x, fn))
                for x in (".data", ".trx")
            ]
        self.dt, self.trxRw = self.fileCache[fn]
        x, proto = self.trxRw["x"], self.dt["protocol"]
        nfls, self.nf = len(x), len(x[0])
        self.ct = CT.get(nfls, LGC2)
        self.fns, self.info = (proto[k] for k in ("frameNums", "info"))
        multEx = isinstance(self.fns, list)
        nef = self.nef = sum(bool(d) for d in self.fns) if multEx else 1
        self.noyc, self.ef, self.isyc = nfls == nef, self.f or 0, False
        assert self.noyc or nef == int(nfls / 2)
        ef2yc = self.ef2yc = proto.get("ef2yc", {})
        oor, err = self.ef >= nfls, None
        if oor or not ef2yc and self.ef >= nef:
            err = "fly number %d out of range (only %s)" % (
                self.ef,
                util.nItems(
                    nfls if oor else nef, ("" if oor else "experimental ") + "fly"
                ),
            )
            self.isyc = not oor
        elif ef2yc and self.ef not in ef2yc:
            err = "fly number %d not among experimental flies (%s)" % (
                self.ef,
                util.commaAndJoin(list(ef2yc.keys())),
            )
            self.isyc = True
        if err:
            if self.opts.allowYC and self.isyc:
                return
            error(err)
        yTop, yBottom = (
            (proto["lines"][k] for k in ("yTop", "yBottom"))
            if "lines" in proto
            else (None, None)
        )
        if self.f is None:
            if multEx:
                error(
                    "more than one experimental fly and no fly numbers; use "
                    + "-v with : or -f"
                )
            assert self.ct == CT.regular
        elif multEx:
            self.fns, self.info = self.fns[self.ef], self.info[self.ef]
            if yTop:
                yTop, yBottom = yTop[self.ef], yBottom[self.ef]
        area, self.pt = "area" in proto, proto.get("pt")
        self.xf = Xformer(proto.get("tm"), self.ct, self.frame, proto.get("fy", False))
        self.circle = area or self.pt == "circle"
        self.openLoop = self.pt == "openLoop"
        self.rectangle = self.pt == "rectangle"
        self.trns = []
        tms = list(zip(self.fns["startTrain"], self.fns["startPost"] + [self.nf]))
        if len(self.fns["startTrain"]) == 0 or not "startPre" in self.fns:
            print(
                "warning: no indices found for training and/or pre-training. Skipping analysis."
            )
            self._skipped = True
            return
        self.startPre = self.fns["startPre"][0]
        # note: some older experiments used 'startPre' more than once
        if self.circle:
            r = proto["area" if area else "circle"]["r"]
            rl = self.info.get("r", [])
            if len(rl) == len(tms):
                r = rl
            else:
                assert all(r1 == r for r1 in rl)
            cPos = self.info["cPos"]
        if self.openLoop:
            self.alt = proto.get("alt", True)
        for i, (st, spst) in enumerate(tms):
            if self.circle:
                trn = Training(
                    i + 1,
                    st,
                    spst,
                    self,
                    self.opts,
                    circle=(cPos[i], r if np.isscalar(r) else r[i]),
                )
            else:
                trn = Training(
                    i + 1,
                    st,
                    spst,
                    self,
                    opts=self.opts,
                    ytb=None if yTop is None else (yTop, yBottom),
                )
            self.trns.append(trn)
        # frame indexes of rewards
        on = [self.fns[k] for k in self.fns if self._ON_KEY.match(k)]
        self.on = np.sort(np.concatenate(on)) if on else np.array([])
        if self.openLoop:
            self.off = np.array(self.fns["v0"])
            assert np.array_equal(self.off, np.sort(self.off))

        print(
            "  video length: %s, frame rate: %s fps, chamber type: %s"
            % (
                frame2hm(self.nf, self.fps),
                util.formatFloat(self.fps, 1),
                self.ct,
            )
        )
        print("  (pre: %s)" % frame2hm(self.trns[0].start - self.startPre, self.fps))
        Training.processReport(self.trns, self.on, self.nf, self.opts)
        self.choice = all(t.tp is t.TP.choice for t in self.trns) and not self.ct in (
            CT.large,
            CT.large2,
        )
        # note: also used for "choice type" open-loop protocols
        self._createExtractChamber()

    def _initTrx(self):
        """
        Initializes trajectory objects for the video analysis.

        This method processes raw trajectory data to create `Trajectory` objects, encapsulating
        movement data for tracked objects in the video. The process includes:
        - Initialization of an empty list for storing `Trajectory` objects.
        - Extraction of position (x, y), width (w), height (h), and orientation (theta) data
          from raw trajectory data.
        - Creation of `Trajectory` objects for each set of data, incorporating relevant options
          and timestamps.

        The method facilitates subsequent analyses by organizing and storing positional and
        geometrical data for tracked objects. Trajectories are indexed from 0 for identification
        purposes.

        Note:
        The method assumes that raw trajectory data (`self.trxRw`) has been loaded and is accessible
        within the instance.
        """
        print("\nprocessing trajectories...")
        self.trx, ts = [], self.trxRw.get("ts")
        self.trxf = (
            (self.ef,)
            if self.noyc
            else (self.ef, self.ef2yc[self.ef] if self.ef2yc else self.ef + self.nef)
        )
        for f in self.trxf:
            x, y, w, h, theta = (
                np.array(self.trxRw[xy][f], dtype=np.float64)
                for xy in ("x", "y", "w", "h", "theta")
            )
            self.trx.append(
                Trajectory(
                    (x, y), self.opts, (w, h, theta), len(self.trx), va=self, ts=ts
                )
            )

    def _rewards_per_distance(self):
        print("\nrewards per distance traveled [m⁻¹]")
        self.rwdsPerDist = []
        exp_trj = self.trx[0]
        ctrl_trj = self.trx[1] if len(self.trx) > 1 else None
        num_displayed_buckets = 0

        def _row(trj: Trajectory, trn: Training):
            """Safe fetch of one training row"""
            nonlocal num_displayed_buckets
            df = self._numRewardsMsg(True, silent=True)
            fi_start, n_buckets, _ = self._syncBucket(trn, df)
            nan_row = [np.nan] * n_buckets
            row = []

            if trj is None or fi_start is None or getattr(trj, "_bad", True):
                return nan_row

            fi = fi_start
            la = min(trn.stop, int(trn.start + n_buckets * df))
            fiRi = util.none2val(self._idxSync(RI_START, trn, fi, la), la)
            n_calc = self._countOnByBucket(
                fi, la, df, calc=True, ctrl=False, f=trj.f, fiCount=fiRi
            )

            while fi + df < la:
                i = len(row)

                # get distance traveled in meters
                dist_trav_meters = (
                    trj.distTrav(fi, fi + df)
                    / (self.xf.fctr * self.ct.pxPerMmFloor())
                    / 1000
                )

                row.append(n_calc[i] / dist_trav_meters)

                fi += df

            num_displayed_buckets = max(num_displayed_buckets, len(row))
            row.extend([np.nan] * (n_buckets - len(row)))
            return row

        for trn in self.trns:
            print(trn.name())
            rows = (_row(exp_trj, trn), _row(ctrl_trj, trn))
            for i, row in enumerate(rows):
                self._printBucketVals(
                    row[:num_displayed_buckets], i, flyDesc(i), prec=2
                )
            self.rwdsPerDist.extend(rows)

    def _aggregate_slide_circle_metrics(self):
        """Collate slide-circle %-in metrics from every Trajectory."""
        if not self.opts.prefCircleSlideRad:
            return

        self.slidePctConc = []
        self.slidePctShift = []

        exp_trj = self.trx[0]  # experimental
        ctrl_trj = self.trx[1] if len(self.trx) > 1 else None

        # helper ---------------------------------------------------------
        def _row(trj, lbl, trn):
            """Safe fetch of one training row; NaNs if anything is missing."""
            # how many buckets does this training *expect*
            df = self._numRewardsMsg(True, silent=True)
            _, n_buckets, _ = self._syncBucket(trn, df)
            nan_row = [np.nan] * n_buckets

            if (
                trj is None
                or getattr(trj, "_bad", True)
                or not hasattr(trj, "pctInC_SB")
                or lbl not in trj.pctInC_SB
                or len(trj.pctInC_SB[lbl]) <= trn.n - 1
            ):
                return nan_row

            row = trj.pctInC_SB[lbl][trn.n - 1]
            # pad / trim to the expected length just in case
            if len(row) < n_buckets:
                row = row + [np.nan] * (n_buckets - len(row))
            elif len(row) > n_buckets:
                row = row[:n_buckets]
            return row

        # ---------------------------------------------------------------

        for trn in self.trns:
            # concentric
            self.slidePctConc.append(
                (
                    _row(exp_trj, "slideConc", trn),
                    _row(ctrl_trj, "slideConc", trn),
                )
            )
            # shifted
            self.slidePctShift.append(
                (
                    _row(exp_trj, "slideShift", trn),
                    _row(ctrl_trj, "slideShift", trn),
                )
            )

    # note file
    # * overrides, e.g., suspicious jump exclusion
    # * e.g., "e0,i2": exclude fly 0, include fly 2
    # * fly numbering is yoked control-independent (e.g., fly 0 is experimental
    #  fly for regular chamber)
    _EI_NUM = re.compile(r"^(e|i)(\d+)$")

    def _readNoteFile(self, fn):
        nfn = util.replaceCheck(AVI_X, "__note.txt", fn)
        note = util.readFile(nfn)
        if note is not None:
            print("\nreading %s:" % util.basename(nfn))
            note, ov = note.strip(), False
            note = note.decode()
            for ps in note.split(","):
                mo = self._EI_NUM.match(ps)
                try:
                    excl, f1 = mo.group(1) == "e", int(mo.group(2))
                except:
                    error('cannot parse "%s"' % note)
                if f1 in self.trxf:
                    f = self.trxf.index(f1)
                    if self.trx[f].bad() != excl:
                        self.trx[f].bad(excl)
                        print(
                            "  %scluding %s fly" % ("ex" if excl else "in", flyDesc(f))
                        )
                        ov = True
            if not ov:
                print("  no override")

    # - - -

    def _writeAnnotatedVideo(self):
        """
        Writes an annotated version of the original video with visual markers or annotations
        corresponding to the analyzed trajectories and events.

        This method creates a new video file that includes the original video content with
        additional annotations (such as trajectories, events, or specific markers) overlaid on each
        frame. The annotations are applied based on the analysis conducted by the `Training` objects
        associated with this video analysis instance. The steps involved are:
        - Determines the output filename by appending "__ann.avi" to the original video filename.
        - Initializes a video writer object using OpenCV, configured to match the original video's
          frame rate and size, and to output in the MJPG format.
        - Iterates through each frame of the original video:
          - Retrieves the current frame.
          - Checks if there is a `Training` object associated with the current frame index and,
            if so, calls its `annotate` method to apply annotations to the frame.
          - Writes the annotated frame to the output video file.
        - Releases the video writer object once all frames have been processed and annotated.

        The method assumes that the original video file has been successfully opened and that
        `Training` objects with annotation capabilities have been properly initialized and are
        available in `self.trns`.

        Note:
        The output video provides a visual representation of the analysis results, making it easier
        to review and understand the movements and behaviors identified during the analysis. This
        can be particularly useful for presentations, further analysis, or verification of the
        automated analysis results.
        """
        ofn = util.replaceCheck(AVI_X, "__ann.avi", self.fn)
        print("\nwriting annotated video %s..." % util.basename(ofn))
        out = cv2.VideoWriter(
            ofn, util.cvFourcc("MJPG"), self.fps, util.imgSize(self.frame), isColor=True
        )
        i = 0
        util.setPosFrame(self.cap, i)
        while True:
            ret, frm = self.cap.read()
            if not ret:
                break
            t = Training.get(self.trns, i)
            if t:
                t.annotate(frm)
            out.write(frm)
            i += 1
        out.release()

    def analyzeOutsideCirclePeriods(self):
        print(
            "\nmean duration of periods spent outside circle (concentric w/ reward circle)"
        )
        for trj in self.trx:
            trj._calcOutsideCirclePeriods()

    def runBoundaryContactAnalyses(self):
        """
        Executes boundary contact analysis for each trajectory in the video.

        This method handles the preparation and execution of boundary contact analyses based on
        the options specified in `self.opts`. It dynamically sets distance thresholds for different
        boundary types (e.g., agarose, wall) if they are specified for analysis and ensures that
        only the high-throughput learning (HTL) chamber type is analyzed for boundary contacts,
        raising an error otherwise.

        The analysis process includes:
        - Parsing and setting distance thresholds for boundary contact analysis.
        - Filtering trajectory data and calculating differences necessary for boundary contact detection.
        - Running the boundary contact analysis either in parallel using a thread pool or sequentially,
        depending on whether the `self.opts.bnd_ct_plots` flag is set.

        Results of the analysis are stored in the `self.trx` list, updating each `Trajectory` object
        with boundary contact information. Additionally, if the 'wall_orientations' data is generated
        during the analysis, it is stored as an attribute of the `VideoAnalysis` instance.
        """
        if self.opts.turn:
            for boundary_type in self.opts.turn:
                if not getattr(self.opts, boundary_type):
                    event_thresholds = (
                        f"{CONTACT_BUFFER_OFFSETS[boundary_type]['min']}|"
                        f"{CONTACT_BUFFER_OFFSETS[boundary_type]['max']}"
                    )
                    setattr(
                        self.opts,
                        boundary_type,
                        (
                            f"{MIDLINE_BOUNDARY_DIST}|"
                            if boundary_type == "boundary"
                            else ""
                        )
                        + event_thresholds,
                    )

        boundary_tps = ("boundary", "agarose", "wall")
        if (
            any(getattr(self.opts, opt) for opt in boundary_tps)
            or self.opts.turn_prob_by_dist
        ):
            if self.ct not in (CT.htl, CT.large, CT.large2):
                raise NotImplementedError(
                    "Only HTL and large chamber types supported for boundary-contact analysis"
                )
        else:
            return

        thresholds = {}
        self.boundary_contact_offsets = {}
        for bnd_tp in boundary_tps:
            if getattr(self.opts, bnd_tp):
                params = getattr(self.opts, bnd_tp)
                params = params.split("|")
                if bnd_tp == "boundary":
                    if len(params) == 3:
                        offset = params[0]
                        thr_for_bnd = params[1:]
                    elif len(params) == 1:
                        offset = params[0]
                        thr_for_bnd = [
                            CONTACT_BUFFER_OFFSETS[bnd_tp]["min"],
                            CONTACT_BUFFER_OFFSETS[bnd_tp]["max"],
                        ]
                    else:
                        raise ArgumentError(
                            "Args for --boundary-contact must be either"
                            " a single value (for distance) or"
                            " three values (distance + start/end thresholds)"
                        )
                else:
                    offset = AGAROSE_BOUNDARY_DIST
                    thr_for_bnd = params
                self.boundary_contact_offsets[bnd_tp] = float(offset)
                thresholds[bnd_tp] = [float(el) for el in thr_for_bnd]
            else:
                thresholds[bnd_tp] = [
                    CONTACT_BUFFER_OFFSETS[bnd_tp]["min"],
                    CONTACT_BUFFER_OFFSETS[bnd_tp]["max"],
                ]
                self.boundary_contact_offsets[bnd_tp] = {
                    "boundary": MIDLINE_BOUNDARY_DIST,
                    "agarose": AGAROSE_BOUNDARY_DIST,
                    "wall": 0,
                }[bnd_tp]

        if (
            self.opts.bnd_ct_plots
            or self.opts.wall_debug
            or self.opts.turn_prob_by_dist
        ):
            for i, trj in enumerate(self.trx):
                if trj.bad():
                    continue

                # Filter trajectory data and calculate differences
                trj_data = {k: getattr(trj, k) for k in ("theta", "x", "y")}
                filtered_data = filterDataAndCalcDiffs(trj_data)
                self.trx[i].receiveDataByKey(filtered_data)

                # Run boundary contact analysis for each trajectory
                trj_proxy = TrjDataContainer(trj)
                va_proxy = VaDataContainer(self)
                result = runBoundaryContactAnalyses(
                    trj_proxy,
                    va_proxy,
                    self.boundary_contact_offsets,
                    thresholds,
                    self.opts,
                )

                for k in result:
                    if k == "wall_orientations":
                        setattr(self, k, result[k])
                    else:
                        setattr(self.trx[i], k, result[k])

            if self.opts.timeit:
                per_bnd_processing_times.append(timeit.default_timer() - bnd_start_t)
        else:
            with ThreadPool() as pool:
                input_data = [
                    {k: getattr(trj, k) for k in ("theta", "x", "y")}
                    for trj in self.trx
                ]
                results = pool.map(filterDataAndCalcDiffs, input_data)
            for i, res in enumerate(results):
                self.trx[i].receiveDataByKey(res)

            if self.opts.timeit:
                bnd_start_t = timeit.default_timer()

            input_data = []
            for trj in self.trx:
                if trj.bad():
                    continue
                trj_proxy = TrjDataContainer(trj)
                va_proxy = VaDataContainer(self)
                input_data.append(
                    (
                        trj_proxy,
                        va_proxy,
                        self.boundary_contact_offsets,
                        thresholds,
                        self.opts,
                    )
                )
            with ThreadPool() as pool:
                results = pool.starmap(runBoundaryContactAnalyses, input_data)
            for i, res in enumerate(results):
                for k in res:
                    if k == "wall_orientations":
                        setattr(self, k, res[k])
                    else:
                        setattr(self.trx[i], k, res[k])

            if self.opts.timeit:
                per_bnd_processing_times.append(timeit.default_timer() - bnd_start_t)

    def _analyzeCircularAndTurningMotion(self):
        """
        Detects and analyzes circular motion and turning behavior within trajectories.

        This method:
        1. Detects circular motion patterns within trajectories using `CircularMotionDetector`.
        2. Analyzes turning movements within detected circular motion.
        3. Consolidates analysis results across all trajectories using `DataCombiner`.

        The analysis aims to identify and quantify behaviors related to circular motion and turning,
        providing insights into movement dynamics and navigational strategies of tracked objects.

        Note:
        Relies on `CircularMotionDetector` and `DataCombiner` for motion detection and result consolidation.
        Analysis results can inform further behavioral analysis or experimental setup adjustments.
        """
        for trj in self.trx:
            circularMD = CircularMotionDetector(trj, self.opts)
            circularMD.predictCircularMotion()
            circularMD.analyzeTurns()
        collator = DataCombiner(self, self.opts)
        collator.combineCircleResults()
        collator.combineTurnResults()

    def calcOnRegionVisitDurationsForCsv(self, region_label):
        """
        Builds a CSV-ready flat list of all mean visit durations (seconds) per interval,
        one value per fly per interval, mirroring regionPercentagesCsv.

        Definition: visits whose start is within the interval and whose stop does not cross the interval end.
        """
        # scaffold
        if not hasattr(self, "regionVisitDurCsv"):
            self.regionVisitDurCsv = {region_label: {"ctr": [], "edge": []}}
        else:
            self.regionVisitDurCsv[region_label] = {"ctr": [], "edge": []}

        if not hasattr(self, "reward_ranges"):
            for tp in ("ctr", "edge"):
                self.regionVisitDurCsv[region_label][tp].extend(
                    len(self.trx) * (len(self.trns) + 1) * [np.nan]
                )
            return

        def durations_in_interval(region_slices, start, stop):
            return [
                sl.stop - sl.start
                for sl in region_slices
                if sl.start >= start and sl.start < stop and sl.stop <= stop
            ]

        for tp in ("ctr", "edge"):
            vals = []
            for i, intvl in enumerate(self.reward_ranges):
                try:
                    intvl = slice(int(intvl.start), int(intvl.stop))
                except (TypeError, ValueError):
                    vals.extend(len(self.trx) * [np.nan])
                    continue

                for trj in self.trx:
                    if trj.bad() or self.pair_exclude[i]:
                        vals.append(np.nan)
                        continue

                    # get visit slices for this fly/region/type
                    region_slices = trj.boundary_event_stats[region_label]["tb"][tp][
                        "boundary_contact_regions"
                    ]

                    durs = durations_in_interval(region_slices, intvl.start, intvl.stop)

                    if len(durs) == 0:
                        vals.append(np.nan)
                    else:
                        vals.append(self._f2s(np.mean(durs)))
            self.regionVisitDurCsv[region_label][tp].extend(vals)

    def calcOnRegionProportionsForCsv(self, region_label):
        """
        Calculates the percentages of contacts with regions of interest (typically an agarose area,
        though also potentially a pair of "virtual" regions with a similar shape as an agarose area,
        but with a different length), distinguishing between center crossing and edge crossing, for
        CSV output.

        This method evaluates the extent of interaction flies have with ROIs within the experimental
        chamber, differentiating between two types of contacts:
        - "ctr": The fly's body center crosses the border between the center floor and the ROI.
        - "edge": The edge of the fly's fitted ellipse crosses the ROI border, but the body center
                does not.

        Frames where sidewall contact is detected are excluded from these calculations if the region
        label is not "agarose." The exclusion is performed using the "opp_edge" key in the `boundary_event_stats`
        dictionary, which indicates sidewall contact.

        For each type of contact, the method computes the percentage of time spent in contact with the
        ROI during specified reward intervals. These calculations are stored in a dictionary,
        `self.onRegionPercentagesCsv`, indexed by the region label with sub-indices of 'ctr' and
        'edge' for easy reference and subsequent CSV output.

        Parameters:
        - region_label: the label associated with the ROI. Determines which data to reference in
                        boundary_event_stats, as well as where to store it in regionPercentagesCsv.

        The computation process includes:
        - Parsing through specified reward ranges to calculate contact percentages for each interval.
        - Handling trajectories marked as 'bad' by skipping them or assigning NaN values, ensuring data
        integrity.
        - For valid trajectories, calculating the percentage based on the number of frames showing contact
        with the ROIs, adjusted for the specific type of contact (center or edge). Sidewall contact
        exclusion is applied when the region is not labeled "agarose."
        - Aggregating these percentages across all trajectories for both types of contacts, facilitating
        analysis on how flies interact with treated areas based on their movement and position relative
        to the ROI border.

        Note:
        This nuanced analysis allows for a detailed understanding of fly behavior in relation to ROI areas,
        providing insights into their movement patterns and preferences within the experimental setup. The
        method's output is tailored for CSV format, aiding in the visualization and further statistical
        analysis of the data.
        """
        if not hasattr(self, "regionPercentagesCsv"):
            self.regionPercentagesCsv = {region_label: {"ctr": [], "edge": []}}
        else:
            self.regionPercentagesCsv[region_label] = {"ctr": [], "edge": []}
        for tp in self.regionPercentagesCsv[region_label]:
            vals = []
            if not hasattr(self, "reward_ranges"):
                # default reward range has intervals for the pre-period and each training
                vals.extend(len(self.trx) * (len(self.trns) + 1) * [np.nan])
                continue
            for i, intvl in enumerate(self.reward_ranges):
                try:
                    intvl = slice(int(intvl.start), int(intvl.stop))
                except ValueError:
                    vals.extend(len(self.trx) * [np.nan])
                    continue
                for trj in self.trx:
                    if trj.bad():
                        vals.append(np.nan)
                        continue
                    if self.pair_exclude[i]:
                        vals.append(np.nan)
                        continue
                    boundary_contact = trj.boundary_event_stats[region_label]["tb"][tp][
                        "original_boundary_contact"
                    ][intvl]

                    if region_label != "agarose":
                        wall_contact = trj.boundary_event_stats["wall"]["all"][
                            "opp_edge"
                        ]["boundary_contact"][intvl]
                        valid_contact = boundary_contact * (
                            ~wall_contact.astype(bool)
                        ).astype(int)
                    else:
                        valid_contact = boundary_contact

                    if region_label == "agarose":
                        intvl_len = np.sum(~trj.nan[intvl])
                    else:
                        intvl_len = intvl.stop - intvl.start

                    vals.append(100 * np.count_nonzero(valid_contact) / intvl_len)
            self.regionPercentagesCsv[region_label][tp].extend(vals)

    # measures % of time on agarose, or in a defined region, by sync bucket.
    def _analyzeRegionPreference(self, region_label):
        self.calcOnRegionProportionsForCsv(region_label)
        self.calcOnRegionVisitDurationsForCsv(region_label)
        combiner = DataCombiner(self, self.opts.postBucketLenMin)
        combiner.combineOnRegionResults(region_label, "edge")
        combiner.combineOnRegionResults(region_label, "ctr")
        combiner.combineOnRegionVisitDurations(region_label, "edge")
        combiner.combineOnRegionVisitDurations(region_label, "ctr")

    def _writeMatFile(self):
        """
        Saves analyzed trajectory and event data to a MATLAB-compatible file.

        This method compiles trajectory data and training session intervals into a dictionary
        and writes this information to a .mat file, facilitating further analysis in MATLAB. The
        process includes:
        - Creating a 'mat' directory if it does not already exist to store the output file.
        - Generating the output filename based on the original video file name with a .mat extension.
        - Extracting and transforming trajectory data for each fly, considering only valid (non-'bad') trajectories.
        - Compiling training session start and stop times, along with frame indices where specific
          events (e.g., 'on' events) occurred, adjusting indices for MATLAB's 1-based indexing.
        - Writing the data to the .mat file using the SciPy I/O library.

        The output file contains individual fly trajectory coordinates (x, y) and arrays detailing
        the start and stop frames of each training session and event occurrence times, providing
        a comprehensive dataset for post-processing or detailed analysis within MATLAB.

        Note:
        This method assumes that there are two flies (or tracked objects) in the video, handling
        cases where a fly's data might be flagged as invalid due to tracking errors or other issues.
        """
        matDir = "mat"
        if not os.path.exists(matDir):
            os.makedirs(matDir)
        ofn = os.path.join(
            matDir, util.basename(util.replaceCheck(AVI_X, ".mat", self.fn))
        )
        print("\nwriting MATLAB file %s..." % ofn)
        t = []
        for f in (0, 1):
            trx = self.trx[f]
            t.append([[], []] if trx.bad() else self.xf.f2t(trx.x, trx.y))
        d = dict(
            f1x=t[0][0],
            f1y=t[0][1],
            f2x=t[1][0],
            f2y=t[1][1],
            trainings=np.array([[t.start, t.stop] for t in self.trns]) + 1,
            on=self.on + 1,
        )
        sio.savemat(ofn, d)

    # - - -

    _DLT = 100
    _ARROW_KEY_MAP = {
        83: 1,
        84: _DLT,
        81: -1,
        82: -_DLT,
        ord("."): 1,
        ord(">"): _DLT,
        ord(","): -1,
        ord("<"): -_DLT,
    }
    # note: arrow keys not seen by OpenCV on Mac
    _HLP = re.sub(
        SPACES_AFTER_TAB,
        "",
        textwrap.dedent(
            """\
    keyboard commands:
    h or ?\t                    toggle show help
    q\t                         quit
    <frame|time> + g\t          go to frame or time (hh:mm:ss)
    <frames|time> + l\t         set length of trajectory shown
    s\t                         toggle show stats
    right, left arrows or .,\t  next, previous frame
    down, up arrows or ><\t     frame +100, -100"""
        ),
    )

    # play video
    def _playAnnotatedVideo(self):
        reg = self.ct is CT.regular
        i = ip = 0
        trx, tlen, s, show, hlp = self.trx, self._DLT, "", False, False
        previous_i = None
        while True:
            try:
                if previous_i is None or i != previous_i:
                    raw_frm = util.readFrame(self.cap, i)
                frm = np.array(raw_frm)
            except VideoError:
                i = ip
                continue
            ip = i
            t, cpr = Training.get(self.trns, i), None
            if t:
                cpr = t.annotate(frm)
            for trx in self.trx:
                if self.opts.wall:
                    fly_color = (
                        COL_G
                        if trx.boundary_event_stats["wall"][
                            self.wall_orientations[
                                self.wall_orientations.index(self.opts.wall_orientation)
                            ]
                        ]["boundary_contact"][i]
                        else COL_Y
                    )
                else:
                    fly_color = COL_Y if True else COL_R
                    if hasattr(trx, "walking"):
                        print("frame:", i)
                        print("position:", trx.x[i], trx.y[i])
                        print("walking:", trx.walking[i])
                        print("speed:", trx.sp[i])
                        print("speed threshold for walking:", 2 * trx.pxPerMmFloor)
                        print("theta:", trx.theta[i])

                trx.annotate(frm, i, tlen, fly_color)
            if reg:
                frm = cv2.resize(frm, (0, 0), fx=2, fy=2)
            if show:
                txt = []
                for f, trx in enumerate(self.trx):
                    txt1 = []
                    txt1.append("%s:" % flyDesc(f))
                    txt1.append("d=%.1f" % trx.d[i])
                    txt1.append("ar=%.1f" % trx.ar[i])
                    txt1.append("onB=%s" % ("T" if trx.onBottom[i] else "F"))
                    if reg:
                        # txt1.append('dx=%.1f' %trx.dltX[i])
                        txt1.append("dx2=%.1f" % trx.dltX2[i])
                    txt.append(" ".join(txt1))
                util.putText(
                    frm, "  ".join(txt), (5, 5), (0, 1), util.textStyle(color=COL_W)
                )
            elif hlp:
                util.putText(
                    frm, self._HLP, (5, 5), (0, 1), util.textStyle(color=COL_W)
                )
            else:
                self.trx[0].annotateTxt(frm, i, "td", cpr)
            hdr = "%s (%d) tlen=%d" % (util.s2time(i / self.fps), i, tlen)
            img = util.combineImgs(((frm, hdr),))[0]
            cv2.imshow(util.basename(self.fn), img)

            # if key "press" (possibly due to auto repeat) happened before waitKey(),
            #  waitKey() does *not* process events and the window is not updated;
            #  the following code makes sure event processing is done
            eventProcessingDone = False
            previous_i = i
            while True:
                k = cv2.waitKey(1)
                if k == -1:
                    eventProcessingDone = True
                elif eventProcessingDone:
                    break
            k &= 255
            dlt, kc = self._ARROW_KEY_MAP.get(k), chr(k)
            if kc == "q":
                break
            elif kc in ("h", "?"):
                hlp = not hlp
            elif kc in ("g", "l"):
                n = None
                if DIGITS_ONLY.match(s):
                    n = int(s)
                else:
                    try:
                        n = int(util.time2s(s) * self.fps)
                    except ArgumentError:
                        pass
                if n is not None:
                    if kc == "g":
                        i = n
                    else:
                        tlen = n
                s = ""
            elif kc == "s":
                show = not show
            elif kc in "0123456789:":
                s += kc
            elif dlt:
                i += dlt

    # - - -

    def _bad(self, f):
        return self.trx[0 if f is None else f].bad()

    # returns frame indexes of all rewards during the given training
    # note: post not used
    def _getOn(self, trn, calc=False, ctrl=False, f=None, post=False):
        if len(self.trx) > 0 and self._bad(f):
            return []
        on = self.trx[f].en[ctrl] if calc else self.on
        if trn is None:
            return on
        fi, la = (trn.stop, trn.postStop) if post else (trn.start, trn.stop)
        return util.inRange(on, fi, la)

    # returns number of rewards in the given frame index range
    def _countOn(self, fi, la, calc=False, ctrl=False, f=None):
        on = self._getOn(None, calc, ctrl, f)
        if len(on) == 0:
            on = np.array([])
        count = util.inRange(on, fi, la, count=True)

        if (
            ctrl
            and self.opts.controlCircleInCorner
            and not self.opts.disableCornerCircleScaling
        ):
            count /= 4
        return count

    # returns number of rewards by bucket; fiCount can be used to make
    #  counting start later than fi
    def _countOnByBucket(
        self, fi, la, df, calc=False, ctrl=False, f=None, fiCount=None, record=False
    ):
        nOns, fi0 = [], fi
        while fi + df <= la:
            nOns.append(
                self._countOn(
                    fi if fiCount is None else max(fi, fiCount), fi + df, calc, ctrl, f
                )
            )
            fi += df
            if record:
                self.buckets[-1].append(fi)
        if fiCount is None:
            no = self._countOn(fi0, fi, calc, ctrl, f)
            assert np.isnan(no) or sum(nOns) == no
        return nOns

    # returns frame index of first reward in the given frame index range
    def _idxFirstOn(self, fi, la, calc, ctrl, f=0):
        on = util.inRange(self._getOn(None, calc, ctrl, f), fi, la)
        return on[0] if len(on) else None

    # returns frame index of first frame where fly 0 is on control side (across
    #  midline) in the given frame range
    def _idxFirstCtrlSide(self, fi, la, trn):
        assert trn.hasSymCtrl()
        if MIDLINE_XING2:
            tc, cc = trn.circles()[:2]
            assert util.distance(tc, cc) > 2 * trn.r
            xy = (self.trx[0].x[fi:la], self.trx[0].y[fi:la])
            onCs = util.distances(xy, pnt=cc) < util.distances(xy, pnt=tc)
        else:
            # y-coordinate-based calculation which requires horizontal midline.
            # there can be small differences (due to rounding) compared to new
            #  calculation; kept this around since it was used for the 2019 paper.

            yc, ym, ys = trn.circles()[0][1], trn.cntr[1], self.trx[0].y[fi:la]
            assert abs(yc - ym) > trn.r
            onCs = ys > ym if yc < ym else ys < ym
        idx = np.argmax(onCs)
        return fi + idx if onCs[idx] else None

    # returns whether the first reward in first bucket for fly 0 is control
    def _firstRewardCtrl(self, fi, la, df):
        if fi is None or fi + df > la:  # consistent with _countOnByBucket()
            return None
        calc = True
        ic, inc = (self._idxFirstOn(fi, fi + df, calc, ctrl) for ctrl in (True, False))
        return (
            (None if inc is None else 0)
            if ic is None
            else (1 if inc is None else int(ic < inc))
        )

    # returns whether fly 0 crossed midline before first reward in first bucket
    def _xedMidlineBefore(self, fi, la, df, trn):
        if fi is None or fi + df > la or not trn.hasSymCtrl():
            # consistent with _countOnByBucket()
            return None
        on1 = self._idxFirstOn(fi, fi + df, calc=True, ctrl=False)
        im = self._idxFirstCtrlSide(fi, fi + df, trn)
        return (
            (None if on1 is None else 0)
            if im is None
            else (1 if on1 is None else int(im < on1))
        )

    # appends n of the given values to "to"
    def _append(self, to, vals, f=0, n=2):
        if np.isscalar(vals) or vals is None:
            n, vals = 1, [vals]
        else:
            n = int(n)
        t = (
            n * (np.nan,)
            if self._bad(f)
            else tuple(vals[:n]) + (n - len(vals)) * (np.nan,)
        )
        assert len(t) == n
        to.append(t)

    def _min2f(self, m):
        return util.intR(m * 60 * self.fps)

    def _f2min(self, a):
        return a / (60 * self.fps)

    def _f2s(self, a):
        return a / self.fps

    def _f2ms(self, a):
        return util.time2str(a / self.fps, "%M:%S", utc=True)

    def _vals(self, vals, f):
        return "trajectory bad" if self._bad(f) else vals

    def _printBucketVals(self, vs, f, msg=None, nParen=0, prec=None):
        """
        Prints bucket values or a placeholder message.

        This method prints values representing bucket contents or a placeholder message if the
        trajectory is marked as 'bad'. It formats and prints the provided values `vs` as a comma-separated
        list, with optional precision formatting specified by the `prec` parameter. The `msg` parameter,
        if provided, serves as a prefix to the printed values.

        Parameters:
        - vs (list): A list of values representing bucket contents or data to be printed.
        - f (int or None): The index of the trajectory being processed, or None if unspecified.
        - msg (str or None): Optional message prefix to be printed along with the values.
        - nParen (int): The number of values to be enclosed in parentheses.
        - prec (int or None): Optional precision for formatting numerical values.

        Note:
        - If the trajectory is marked as 'bad' (via the `_bad` method), a placeholder message indicating
          absence of full bucket data is printed instead of the actual values.
        """
        if prec is not None:
            frm = "%%.%df" % prec
            vs = [frm % v for v in vs]
        vs = ["(%s)" % v if i < nParen else v for i, v in enumerate(vs)]
        print(
            "  %s%s"
            % (
                "%s: " % msg if msg else "",
                self._vals(util.join(", ", vs, 10) if vs else "no full bucket", f),
            )
        )

    def _rewardType(self, calc, ctrl, f):
        """
        Generates a label indicating reward type based on calculation mode and control status.

        This method generates a label indicating the reward type based on the calculation mode
        (`calc`) and control status (`ctrl`) provided. The label distinguishes between actual,
        calculated, and control rewards and includes additional information if specified.

        Parameters:
        - calc (bool): Flag indicating whether the reward is calculated.
        - ctrl (bool): Flag indicating whether the reward is from the control group.
        - f (int or None): The index of the trajectory being processed, or None if unspecified.

        Returns:
        - str: A string representing the reward type label, including calculation mode and
          control status information if applicable.
        """
        return "%s%s" % (cVsA(calc, ctrl), " %s" % flyDesc(f) if calc or ctrl else "")

    # returns bucket length in frames as int
    def _numRewardsMsg(self, sync, silent=False):
        blm = self.opts.syncBucketLenMin if sync else self.opts.postBucketLenMin
        if silent == False:
            print(
                "\nnumber%s rewards by %s bucket (%s min):"
                % (
                    "" if sync else " " + cVsA_l(True),
                    "sync" if sync else "post",
                    util.formatFloat(blm, 1),
                )
            )
        return self._min2f(blm)

    # default: skip frame of first reward
    def _syncBucket(self, trn, df=np.nan, skip=1):
        on = self._getOn(trn)  # sync buckets determined using actual rewards
        fi = on[0] + skip if len(on) else None
        if SYNC_CTRL:
            fi = (
                fi
                if fi is None
                else util.noneadd(
                    self._idxFirstOn(fi, trn.stop, calc=True, ctrl=True), skip
                )
            )
        n = np.ceil(trn.len() / df - 0.01).astype(int)
        return fi, n, on

    # returns SyncType (tp)-dependent frame index in the given frame index range
    # note: skip applies only to sync on control circle
    def _idxSync(self, tp, trn, fi, la, skip=1):
        if tp is ST.fixed or fi is None or np.isnan(fi):
            return fi
        elif tp is ST.control or not trn.hasSymCtrl():
            return util.noneadd(self._idxFirstOn(fi, la, calc=True, ctrl=True), skip)
        else:
            assert tp is ST.midline
            return self._idxFirstCtrlSide(fi, la, trn)

    # returns start frame of first post bucket
    def _postSyncBucket(self, trn, skip=1):
        return self._idxSync(POST_SYNC, trn, trn.stop, trn.postStop, skip)

    # - - -

    # number of rewards by bucket
    def byBucket(self):
        if self.rectangle:
            self.numRewards = [  # only tracking real rewards (not calc)
                [[]]  # no control rewards
            ]
        tnOn = 0
        for i, t in enumerate(self.trns):
            df = t.len() / self.opts.numBuckets
            if self.opts.showByBucket:
                if i == 0:
                    print("number rewards: (bucket: %s)" % frame2hm(df, self.fps))
                print(t.name())
            la, nOns = t.start, []
            for i in range(self.opts.numBuckets):
                fi, la = la, t.start + util.intR((i + 1) * df)
                nOns.append(self._countOn(fi, la))
            snOn = sum(nOns)
            assert la == t.stop and self._countOn(t.start, t.stop) == snOn
            tnOn += snOn
            if self.opts.showByBucket:
                print("  %s  (sum: %d)" % (", ".join(map(str, nOns)), snOn))
            if self.rectangle:
                self.numRewards[0][0].append(nOns)
        print(
            "total rewards training: %d, non-training: %d" % (tnOn, len(self.on) - tnOn)
        )
        self.totalTrainingNOn = tnOn

    # number of rewards by sync bucket
    def bySyncBucket(self):
        df = self._numRewardsMsg(True)
        self.numRewards = [[[]], [[], []]]  # idxs: calc, ctrl
        self.numRewardsTot = [[[]], [[], []]]
        self.rewardPI, self.rewardPITrns = [], []
        self.firstRewardCtrl, self.xedMidlineBefore = [], []
        self.buckets = []
        for t in self.trns:
            print(t.name())
            fi, n, on = self._syncBucket(t, df)
            self.buckets.append([fi if fi is not None else np.nan])
            la = min(t.stop, int(t.start + n * df))
            fiRi = util.none2val(self._idxSync(RI_START, t, fi, la), la)
            self.rewardPITrns.append(t)
            self._append(self.firstRewardCtrl, self._firstRewardCtrl(fi, la, df))
            self._append(self.xedMidlineBefore, self._xedMidlineBefore(fi, la, df, t))
            for calc, f in ((False, None), (True, 0), (True, 1)):
                if self.noyc and f == 1:
                    continue
                for ctrl in (False, True) if calc else (False,):
                    nOns = (
                        []
                        if fi is None
                        else self._countOnByBucket(
                            fi,
                            la,
                            df,
                            calc,
                            ctrl,
                            f,
                            fiRi if calc else None,
                            record=(not calc and not ctrl),
                        )
                    )
                    if len(self.buckets[-1]) - 1 < n:
                        self.buckets[-1].extend(
                            [np.nan] * int(n - len(self.buckets[-1]) + 1)
                        )
                    self._printBucketVals(nOns, f, msg=self._rewardType(calc, ctrl, f))
                    self._append(self.numRewards[calc][ctrl], nOns, f)
                    self._append(self.numRewardsTot[calc][ctrl], nOns, f, len(nOns))
                    if ctrl:
                        pis = util.prefIdx(nOnsP, nOns, n=self.opts.piTh)
                        self._printBucketVals(pis, f, msg="  PI", prec=2)
                        self._append(self.rewardPI, pis, f, n=n)
                    nOnsP = nOns

    # distance traveled or maximum distance reached between (actual) rewards
    #  by sync bucket
    # notes:
    # * reward that starts sync bucket included here (skip=0) so that
    #  distance to the next reward is included in average; this differs from
    #  bySyncBucket() but matches byActualReward()
    # * also used for "open loop" analysis, where sync buckets equal buckets
    def bySyncBucket2(self, maxD=False):
        hdr = "\naverage %s between actual rewards by %sbucket:" % (
            "maximum distance reached" if maxD else "distance traveled",
            "" if self.opts.ol else "sync ",
        )
        print(hdr)
        self.bySB2Header, self.bySB2 = hdr, []
        df = self._min2f(self.opts.syncBucketLenMin)
        for t in self.trns:
            print(t.name())
            fi, n, on = self._syncBucket(t, df, skip=0)
            assert not self.opts.ol or fi == t.start
            la = min(t.stop, int(t.start + n * df))
            nOns, adb = [], [[], []]
            if fi is not None:
                nOns1 = self._countOnByBucket(fi, la, df)
                while fi + df <= la:
                    onb = util.inRange(on, fi, fi + df)
                    nOn = len(onb)
                    for f in self.flies:
                        if maxD:
                            maxDs = []
                            for i, f1 in enumerate(onb[:-1]):
                                xy = self.trx[f].xy(f1, onb[i + 1])
                                maxDs.append(np.max(util.distances(xy, True)))
                            adb[f].append(
                                np.nan if nOn < self.opts.adbTh else np.mean(maxDs)
                            )
                        else:
                            adb[f].append(
                                np.nan
                                if nOn < self.opts.adbTh
                                else self.trx[f].distTrav(onb[0], onb[-1]) / (nOn - 1)
                            )
                    nOns.append(len(onb))
                    fi += df
                assert nOns == nOns1
            for f in self.flies:
                self._printBucketVals(adb[f], f, msg=flyDesc(f), prec=1)
                self._append(self.bySB2, adb[f], f, n=n if self.opts.ol else n - 1)
        self.buckets = np.array(self.buckets)

    # --------------- new median-based distance method ----------------
    def bySyncBucketMedDist(self, min_no_contact_s=None):
        """
        For each sync-bucket, compute the median of per-frame
        distances to the reward-circle center, optionally filtering out
        brief no-contact bouts as in cleanNonContactMask().
        Results in self.syncMedDist: a list (per training) of dicts
        with 'exp' and optionally 'ctrl' keys mapping to median distances.
        """
        df = self._numRewardsMsg(True, silent=True)
        self.syncMedDist = []

        for trn in self.trns:
            # build bucket boundaries from the first‐reward frame
            fi, n_buckets, _ = self._syncBucket(trn, df)
            n_buckets = int(n_buckets)
            this_training = {}
            if fi is None:
                for fly_key in ("exp", "ctrl"):
                    this_training[fly_key] = [np.nan for _ in range(n_buckets)]
                self.syncMedDist.append(this_training)
                continue
            starts = [int(fi + k * df) for k in range(int(n_buckets))]
            ends = [s + df for s in starts]
            # la guards against partial buckets
            la = min(trn.stop, int(trn.start + n_buckets * df))
            buckets = [(s, e) for s, e in zip(starts, ends) if e <= la]

            for fly_key, traj in (("exp", self.trx[0]),) + (
                (("ctrl", self.trx[1]),) if len(self.trx) > 1 else ()
            ):
                # precompute mask if requested
                if min_no_contact_s is not None:
                    mask = traj.cleanNonContactMask(min_no_contact_s)

                # find correct reward‐circle center for this fly
                fly_idx = 0 if fly_key == "exp" else 1
                if self.trx[fly_idx]._bad:
                    this_training[fly_key] = [np.nan] * n_buckets
                    continue
                cx, cy, _ = trn.circles(fly_idx)[0]

                med_vals = []
                for s, e in buckets:
                    # pick valid frame indices
                    if min_no_contact_s is not None:
                        idxs = np.nonzero(mask[s:e])[0] + s
                    else:
                        idxs = np.arange(s, e)
                    xs = traj.x[idxs]
                    ys = traj.y[idxs]
                    # per-frame distances
                    ds = np.hypot(xs - cx, ys - cy)
                    if ds.size == 0:
                        med_vals.append(np.nan)
                    else:
                        med_vals.append(
                            np.nanmedian(ds) / (self.xf.fctr * self.ct.pxPerMmFloor())
                        )

                missing = int(n_buckets) - len(med_vals)
                if missing > 0:
                    med_vals.extend([np.nan] * missing)
                this_training[fly_key] = med_vals

            self.syncMedDist.append(this_training)

    # calculates 1) number of calculated and control rewards during entire pre-
    # training and 2) reward PI for final 10 minutes of pre-training
    def calcRewardsPre(self):
        self.numRewardsTotPrePost = [[] for _ in range(4)]
        self.numPreCrossings = [[] for _ in range(len(self.flies))]
        t, self.rewardPIPre = self.trns[0], []
        (
            start,
            stop,
        ) = (
            self.startPre,
            t.start,
        )
        for f in self.flies:
            preOns = [
                self._countOn(
                    t.start - self._min2f(10), t.start, calc=True, ctrl=ctrl, f=f
                )
                for ctrl in (False, True)
            ]
            self.numPreCrossings[f].append(np.sum(preOns))
            self.rewardPIPre.append(util.prefIdx(*preOns, n=self.opts.piTh))
            for j, ctrl in enumerate((False, True)):
                self.numRewardsTotPrePost[j].append(
                    self._countOn(start, stop, calc=True, ctrl=ctrl, f=f)
                )

    def calcContactlessRewardsForCSV(
        self, evt_name, boundary_orientation, t, n_bkts, trj, contact_idxs
    ):
        """
        Calculates contactless rewards for CSV output based on event and training phase.

        This method computes contactless rewards for CSV output, focusing on specific events
        and training phases within the experimental setup. The calculations are performed as follows:

        - If the event name does not match "wall_contact" or the boundary orientation is not present in the
          contactless rewards data, the method returns without further action.

        - If the training session is in the pre-reward phase:
          - Initializes pre-reward session data for contactless rewards.
          - Computes contactless rewards for the pre-reward phase using the
            `calcContactlessRewardsForSyncBucket` method.

        - If the training session is the final segment of the last training session:
          - Aggregates pre-reward session data.
          - Computes contactless rewards for CSV output by combining pre-reward and final training
            session data.
          - Converts the aggregated data into numeric values suitable for CSV output using the
            `convert_rewards_to_numeric` method.

        Parameters:
        - evt_name (str): The name of the event for which contactless rewards are calculated.
        - boundary_orientation (str): The orientation of boundaries associated with the event.
        - t (Training): The training session during which the event occurred.
        - n_bkts (int): The number of buckets used for analysis.
        - trj (Trajectory): The trajectory object associated with the event.
        - contact_idxs (list): Indices of contact events related to the event.

        Note:
        This method provides essential functionality for analyzing contactless rewards within
        the experimental setup. It handles different event scenarios and training phases to
        ensure accurate computation and representation of contactless reward data for CSV output.
        """
        if (
            evt_name != "wall_contact"
            or boundary_orientation not in self.contactless_rewards
        ):
            return
        if t.n == 1:
            self.contactless_rewards[boundary_orientation]["pre"].append(0)
            self.contactless_trajectory_lengths[boundary_orientation]["pre"].append([])
            self.contactless_max_dists[boundary_orientation]["pre"].append([])
            self.calcContactlessRewardsForSyncBucket(
                trj,
                contact_idxs,
                trj.en[False],
                trj.pre_reward_range.start,
                trj.pre_reward_range.stop,
                self.contactless_rewards[boundary_orientation]["pre"],
                self.contactless_trajectory_lengths[boundary_orientation]["pre"],
                self.contactless_max_dists[boundary_orientation]["pre"],
            )
        elif t.n == 3 and trj.f + 1 == len(self.trx):
            self.contactless_rewards[boundary_orientation]["csv"] = [
                list(self.contactless_rewards[boundary_orientation]["pre"])
            ]
            nf = len(self.flies)
            for i in range(len(self.trns)):
                if i == 0:
                    for f in self.flies:
                        self.contactless_rewards[boundary_orientation]["csv"][
                            -1
                        ].append(
                            self.contactless_rewards[boundary_orientation]["trn"][
                                i * nf + f
                            ][0]
                        )
                if i > 0:
                    for f in self.flies:
                        self.contactless_rewards[boundary_orientation]["csv"][
                            -1
                        ].append(
                            self.contactless_rewards[boundary_orientation]["trn"][
                                i * nf + f
                            ][n_bkts - 2]
                        )
            for period_tp in ("pre", "trn"):
                self.convert_rewards_to_numeric(boundary_orientation, period_tp)

    def floorCenter(self, f=0):
        if not hasattr(self, "_floor_center") or f not in self._floor_center:
            floor = list(self.ct.floor(self.xf, f=self.trxf[f]))
            floor_center = np.transpose(
                np.expand_dims(
                    (
                        0.5 * (floor[0][0] + floor[1][0]),
                        0.5 * (floor[0][1] + floor[1][1]),
                    ),
                    0,
                )
            )
            if not hasattr(self, "_floor_center"):
                self._floor_center = {f: floor_center}
            else:
                self._floor_center[f] = floor_center
        else:
            floor_center = self._floor_center[f]
        return floor_center

    def calcContactlessRewardsForSyncBucket(
        self,
        trj,
        wall_contact_idxs,
        reward_idxs,
        fi,
        la,
        contactless_rewards,
        contactless_trajectory_lengths,
        contactless_max_dists,
    ):
        """
        Calculates contactless rewards and related metrics for a sync bucket.

        This method computes contactless rewards and associated metrics for a specific sync bucket
        within the experimental setup. It processes trajectory data and event indices to derive
        contactless reward values, trajectory lengths, and maximum distances from a designated
        floor center point.

        The calculation involves the following steps:
        - Determining the floor center coordinates based on the trajectory's floor position.
        - Filtering reward indices within the specified bucket time range.
        - Iterating through reward indices to compute trajectory lengths and distances traveled
          without wall contact.
        - Calculating the average contactless reward value and related metrics for the bucket.
        - Handling cases where no rewards or insufficient rewards are present.

        Parameters:
        - trj (Trajectory): The trajectory object associated with the synchronous time bucket.
        - wall_contact_idxs (list): Indices of wall contact events within the trajectory.
        - reward_idxs (ndarray): Indices of reward events within the specified time bucket.
        - fi (int): The start index of the time bucket.
        - la (int): The end index of the time bucket.
        - contactless_rewards (list): List to store contactless reward values for the bucket.
        - contactless_trajectory_lengths (list): List to store trajectory lengths for the bucket.
        - contactless_max_dists (list): List to store maximum distances from the reward circle
          center during contactless reward trajectories in the bucket.

        Note:
        This method plays a crucial role in analyzing contactless behavior during specific
        time intervals within the experimental sessions. It provides essential insights into
        fly movements and interactions with experimental elements, contributing to a deeper
        understanding of behavioral patterns and responses.
        """
        floor_center = self.floorCenter(trj.f)
        reward_idxs = reward_idxs[(reward_idxs >= fi) & (reward_idxs < la)]
        num_rwds = len(reward_idxs)
        if num_rwds == 0:
            contactless_trajectory_lengths[-1] = np.mean(
                contactless_trajectory_lengths[-1]
            )
            contactless_max_dists[-1] = np.mean(contactless_max_dists[-1])
            contactless_rewards[-1] = np.nan
            return
        prev_rew = reward_idxs[0]
        indices = []
        for rew_idx in reward_idxs[1:]:
            if (
                len(
                    wall_contact_idxs[
                        (wall_contact_idxs > prev_rew) & (wall_contact_idxs < rew_idx)
                    ]
                )
                == 0
            ):
                trj_xy = trj.xy(i1=prev_rew, i2=rew_idx)
                dist_trav = trj.distTrav(prev_rew, rew_idx)
                indices.append(rew_idx)
                contactless_trajectory_lengths[-1].append(dist_trav)
                contactless_rewards[-1] += 1
                contactless_max_dists[-1].append(
                    np.amax(np.linalg.norm(np.array(trj_xy) - floor_center, axis=0))
                )
            prev_rew = rew_idx
        contactless_rewards[-1] /= num_rwds
        if len(contactless_trajectory_lengths[-1]) > 0:
            contactless_trajectory_lengths[-1] = np.mean(
                contactless_trajectory_lengths[-1]
            )
            contactless_max_dists[-1] = np.mean(contactless_max_dists[-1])
        else:
            contactless_trajectory_lengths[-1] = np.nan
            contactless_max_dists[-1] = np.nan

        if num_rwds < self.opts.piTh:
            contactless_rewards[-1] = np.nan

    @staticmethod
    def centre_for(training_ranges, frame_idx):
        # ranges are (start, stop, cx, cy)
        for s, e, cx, cy in training_ranges:
            if s <= frame_idx < e:
                return cx, cy
        # fallback – shouldn’t happen
        return training_ranges[-1][2], training_ranges[-1][3]

    @staticmethod
    def val_to_numeric(entry):
        """
        Converts a value to its numeric equivalent.

        This static method is responsible for converting a given value to its numeric equivalent.
        It handles various data types such as floats and strings, including special cases like
        'nan' values.

        Parameters:
        - entry (float, str): The value to be converted to numeric.

        Returns:
        - float: The numeric equivalent of the input value.

        Note:
        This method provides a robust conversion mechanism, ensuring compatibility and consistency
        in handling different types of data within the context of contactless rewards calculation
        and processing.
        """
        if type(entry) == float:
            return entry
        if type(entry) == str:
            if "nan" in entry:
                return np.nan
            else:
                return float(entry.split(" (")[0])

    def convert_rewards_to_numeric(self, boundary_orientation, period_tp):
        """
        Converts contactless rewards to numeric values.

        This method converts contactless rewards stored in string format to their numeric equivalents.
        It iterates through the specified rewards data structure and applies the `val_to_numeric`
        function to each entry, converting strings representing numeric values to actual floats.

        Parameters:
        - boundary_orientation (str): The orientation of the boundary being analyzed for contact events.
        - period_tp (str): The period type indicating whether rewards are for the pre-reward or
          training phase.

        Note:
        The conversion to numeric values facilitates further analysis and processing of contactless
        rewards, ensuring consistency and compatibility with numerical data operations and computations.
        """
        for i, entry in enumerate(
            self.contactless_rewards[boundary_orientation][period_tp]
        ):
            if type(entry) != tuple and type(entry) != list:
                self.contactless_rewards[boundary_orientation][period_tp][i] = (
                    self.val_to_numeric(entry)
                )
            else:
                self.contactless_rewards[boundary_orientation][period_tp][i] = list(
                    entry
                )
                for j, subentry in enumerate(entry):
                    self.contactless_rewards[boundary_orientation][period_tp][i][j] = (
                        self.val_to_numeric(subentry)
                    )

    def get_boundary_orientations(self, evt_name):
        """
        Determines the boundary orientations relevant to a given event name for analysis.

        This method identifies the applicable boundary orientations for various types of events, such as wall contacts,
        agarose contacts, and turning events. Each event type is associated with specific orientations of boundaries that
        are crucial for its analysis. The method returns a tuple of strings indicating these orientations, facilitating
        targeted analysis based on the event's nature.

        Parameters:
        - evt_name (str): The name of the event for which to determine the relevant boundary orientations. It interprets
                        specific keywords within the event name (e.g., "wall_contact", "agarose", "turn") to categorize
                        the event and determine the appropriate orientations for analysis.

        Returns:
        - tuple: A tuple of strings representing the boundary orientations relevant to the specified event. Possible values
                include identifiers for specific orientations or categories of boundaries (e.g., "tb" for top and bottom,
                "all" for all boundaries).

        Examples:
        - Calling `get_boundary_orientations("wall_contact")` would return the orientations associated with wall contact events.
        - Calling `get_boundary_orientations("agarose_contact")` would return `("tb",)`, indicating top and bottom boundaries
          for agarose contact events.
        - Calling `get_boundary_orientations("inside_line_turn")` would return `("all",)`, indicating all boundaries are
          considered for turning events.
        """
        if evt_name == "wall_contact":
            return self.wall_orientations
        elif "agarose" in evt_name or "boundary" in evt_name:
            return ("tb",)
        elif "turn" in evt_name:
            return ("all",)

    def setup_sb_contact_events(self, evt_name, boundary_orientations, is_turn):
        """
        Initializes the setup for analyzing contact events or turns within synchronization buckets (SBs)
        based on the event type and boundary orientations.

        This method configures various attributes essential for the detailed analysis and reporting of
        contact events (e.g., wall or agarose contacts) or turning events. It sets up logging headers, CSV
        attribute names for data export, and initializes data structures for tracking specific metrics such
        as rewards without contact, trajectory lengths, and maximum distances without contact for wall contact
        events.

        Parameters:
        - evt_name (str): The name of the event, indicating the type of contact or turning event being analyzed.
        - boundary_orientations (tuple): A tuple of strings indicating the orientations of boundaries involved
                                        in the event, facilitating targeted analysis.
        - is_turn (bool): A flag indicating whether the event is a turning event, which requires additional
                        setup for duration tracking.
        """
        if evt_name in ("wall_contact", "agarose_contact", "boundary_contact"):
            self.boundary_tp = "_".join(evt_name.split("_")[:-1])
            self.csv_attr_name = "%s_contact_evts_for_csv" % self.boundary_tp
            self.sb_contact_evt_log_header = (
                "%s-contact events per reward" % self.boundary_tp
            )
        elif "_turn" in evt_name:
            self.boundary_tp = evt_name.split("_turn")[0].split("_")[-1]
            self.csv_attr_name = "%s_evts_for_csv" % evt_name
            self.sb_contact_evt_log_header = "%s turns per %s-contact event" % (
                self.boundary_tp,
                self.boundary_tp,
            )
            if "_line" in evt_name:
                self.sb_contact_evt_log_header = "%s-line %s" % (
                    evt_name.split("_line")[0],
                    self.sb_contact_evt_log_header,
                )
            else:
                self.ellipse_ref_pt = evt_name.split("_")[0]
                self.sb_contact_evt_log_header += (
                    f" (contact events begin when {self.ellipse_ref_pt} of"
                    " fitted ellipse crosses boundary)"
                )
        if evt_name == "wall_contact":
            self.contactless_rewards = {
                boundary_orientation: {"pre": [], "trn": [], "csv": []}
                for boundary_orientation in ("all", "agarose_adj")
            }
            self.contactless_trajectory_lengths = {
                boundary_orientation: {"pre": [], "trn": [], "csv": []}
                for boundary_orientation in ("all", "agarose_adj")
            }
            self.contactless_max_dists = {
                boundary_orientation: {"pre": [], "trn": [], "csv": []}
                for boundary_orientation in ("all", "agarose_adj")
            }
        self.N = self.opts.n_rewards_start_end_avg
        self.evts = []
        if is_turn:
            self.evt_durations = []
            if hasattr(self, "evts_counts"):
                delattr(self, "evts_counts")
        else:
            self.evts_counts = []
        print(("\n%s:" % self.sb_contact_evt_log_header))
        if evt_name == "wall_contact":
            self.wall_orientation_idx = boundary_orientations.index(
                self.opts.wall_orientation
            )
        self.firsts_by_trn = []

    def determineTurnDirectionality(
        self, boundary_tp, turn_tp, ext_data=None, trj=None, boundary_combo="tb"
    ):
        """
        Determines the turn directionality towards the center based on trajectory data.

        Parameters:
        - boundary_tp (str): The type of boundary interaction involved (e.g., "tb" for top-bottom).
        - turn_tp (str): The type of turn involved, which could be "inside" or "outside".
        - ext_data (dict, optional): If provided, uses this data to determine turning, instead of internal data.

        Returns:
        - dict or None: Returns a dictionary containing the turn directionality results if ext_data is provided,
          otherwise updates internal state.
        """
        debug = False
        results = {} if ext_data is not None else None

        trx = [trj] if trj is not None else self.trx

        for trj in trx:
            if trj.bad():
                continue

            if "line" in turn_tp:
                ellipse_ref_pt = "edge"
                turn_type = "inside" if "inside" in turn_tp else "outside"
                # Use data from ext_data if provided, otherwise use internal data
                if ext_data:
                    turn_boolean = ext_data[boundary_tp][boundary_combo][
                        ellipse_ref_pt
                    ]["turning"] == (1 if turn_type == "inside" else 2)
                else:
                    turn_boolean = trj.boundary_event_stats[boundary_tp][
                        boundary_combo
                    ][ellipse_ref_pt]["turning"] == (1 if turn_type == "inside" else 2)
            else:
                ellipse_ref_pt = turn_tp
                turn_type = "all"
                # Use data from ext_data if provided, otherwise use internal data
                if ext_data:
                    turn_boolean = (
                        ext_data[boundary_tp][boundary_combo][ellipse_ref_pt]["turning"]
                        > 0
                    )
                else:
                    turn_boolean = (
                        trj.boundary_event_stats[boundary_tp][boundary_combo][
                            ellipse_ref_pt
                        ]["turning"]
                        > 0
                    )

            turn_starts_ends = util.trueRegions(turn_boolean)

            if boundary_tp == "circle":
                floor_center_x, floor_center_y = None, None
                training_ranges = ext_data["circle"]["ctr"]["ctr"]["training_ranges"]
            else:
                floor_center = self.floorCenter(trj.f)
                floor_center_x, floor_center_y = floor_center[0][0], floor_center[1][0]
                training_ranges = None

            if results is not None:
                turn_directionality_data = self.collectTurnDirectionalityStats(
                    trj,
                    turn_starts_ends,
                    floor_center_x,
                    floor_center_y,
                    debug,
                    training_ranges,
                )
                results.setdefault(boundary_tp, {}).setdefault(ellipse_ref_pt, {})[
                    turn_type
                ] = turn_directionality_data
            else:
                self.updateTurnDirectionalityStats(
                    trj,
                    boundary_tp,
                    boundary_combo,
                    ellipse_ref_pt,
                    turn_type,
                    turn_starts_ends,
                    floor_center_x,
                    floor_center_y,
                    debug,
                    training_ranges,
                )

        return results

    def collectTurnDirectionalityStats(
        self,
        trj,
        turn_starts_ends,
        floor_center_x,
        floor_center_y,
        debug,
        training_ranges=None,
    ):
        """
        Collects the boundary event statistics for turn directionality based on the provided parameters.

        Returns:
        - dict: Collected turn directionality statistics for the trajectory.
        """
        turn_data = {}
        debug_rows = []  # For CSV output
        max_debug_rows = 250

        for start_end in turn_starts_ends:
            start, end = start_end.start, start_end.stop
            if start == end or end == len(trj.x):
                continue
            # raw positions
            sx, sy = trj.x[start], trj.y[start]
            ex, ey = trj.x[end], trj.y[end]

            if training_ranges is not None:
                cx, cy = VideoAnalysis.centre_for(training_ranges, start)
            else:
                cx, cy = floor_center_x, floor_center_y

            # compute vectors
            svx, svy = cx - sx, cy - sy
            evx, evy = cx - ex, cy - ey

            sh = round(np.degrees(trj.velAngles[start]), 2)
            eh = round(np.degrees(trj.velAngles[end]), 2)

            turn_toward = self.calculate_turn_toward_center(
                np.array([svx, svy]), np.array([evx, evy]), sh, eh
            )

            turn_data[start] = turn_toward

            if debug:
                print(f"start/end index: {start}, {end}")
                print(
                    f"start pos: ({sx:.2f}, {sy:.2f})   end pos: ({ex:.2f}, {ey:.2f})"
                )
                print(f"start heading: {sh}°   end heading: {eh}°")
                print(
                    f"start vec: ({svx:.2f}, {svy:.2f})   end vec: ({evx:.2f}, {evy:.2f})"
                )
                print(f"is turn toward center? {turn_toward}")
                print("-" * 40)

                if len(debug_rows) < max_debug_rows:
                    debug_rows.append(
                        {
                            "start_index": start,
                            "end_index": end,
                            "start_x": round(sx, 2),
                            "start_y": round(sy, 2),
                            "end_x": round(ex, 2),
                            "end_y": round(ey, 2),
                            "start_heading_deg": sh,
                            "end_heading_deg": eh,
                            "start_vec_x": round(svx, 2),
                            "start_vec_y": round(svy, 2),
                            "end_vec_x": round(evx, 2),
                            "end_vec_y": round(evy, 2),
                            "toward_center": turn_toward,
                        }
                    )

        # write out CSV if in debug mode
        if debug and debug_rows:
            csv_path = (
                f"turn_direction_debug_{trj.id}.csv"
                if hasattr(trj, "id")
                else "turn_direction_debug.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=debug_rows[0].keys())
                writer.writeheader()
                writer.writerows(debug_rows)
            print(f"\nWrote {len(debug_rows)} debug events to {csv_path}\n")

        return turn_data

    def updateTurnDirectionalityStats(
        self,
        trj,
        boundary_tp,
        boundary_combo,
        ellipse_ref_pt,
        turn_type,
        turn_starts_ends,
        floor_center_x,
        floor_center_y,
        debug,
        training_ranges=None,
    ):
        """
        Updates the boundary event statistics for turn directionality based on the provided parameters.

        Handles the update of internal state statistics based on turning events, processing each
        start and stop event to calculate directional vectors and updating the stats dictionary.

        Parameters:
        - trj: The trajectory object being processed.
        - boundary_tp (str): Type of boundary interaction.
        - ellipse_ref_pt (str): Reference point on the ellipse involved in the turn.
        - turn_type (str): Specifies whether the turn is "inside" or "outside".
        - turn_starts_ends: Regions where turning starts and ends.
        - floor_center_x (float): X-coordinate of the floor center.
        - floor_center_y (float): Y-coordinate of the floor center.
        - debug (bool): Flag to control debug output.

        Returns:
        - None: Directly modifies the trajectory object's internal statistics.
        """

        if ellipse_ref_pt not in trj.boundary_event_stats[boundary_tp][boundary_combo]:
            trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt] = {}
        if (
            "turn_direction_toward_ctr"
            not in trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt]
        ):
            trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt][
                "turn_direction_toward_ctr"
            ] = {}
        if (
            turn_type
            not in trj.boundary_event_stats[boundary_tp][boundary_combo][
                ellipse_ref_pt
            ]["turn_direction_toward_ctr"]
        ):
            trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt][
                "turn_direction_toward_ctr"
            ][turn_type] = {}

        for start_end in turn_starts_ends:
            start, end = start_end.start, start_end.stop
            if start == end or end == len(trj.x):
                continue
            start_pos_x, start_pos_y = trj.x[start], trj.y[start]
            end_pos_x, end_pos_y = trj.x[end], trj.y[end]

            if training_ranges is not None:
                cx, cy = VideoAnalysis.centre_for(training_ranges, start)
            else:
                cx, cy = floor_center_x, floor_center_y

            start_vector = np.array([cx - start_pos_x, cy - start_pos_y])
            start_heading_angle = np.degrees(trj.velAngles[start])
            end_vector = np.array([cx - end_pos_x, cy - end_pos_y])

            end_heading_angle = np.degrees(trj.velAngles[end])
            turn_direction_toward_ctr = self.calculate_turn_toward_center(
                start_vector, end_vector, start_heading_angle, end_heading_angle
            )

            trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt][
                "turn_direction_toward_ctr"
            ][turn_type][start] = turn_direction_toward_ctr

            if debug:
                print("Debug Info:")
                print("floor center:", floor_center_x, floor_center_y)
                print("start/end index:", start, end)
                print("start position:", start_pos_x, start_pos_y)
                print("before degree conversion:", trj.velAngles[start])
                print("start heading angle:", start_heading_angle)
                print("start vector to center:", start_vector)
                print("end position:", end_pos_x, end_pos_y)
                print("end heading angle:", end_heading_angle)
                print("end vector to center:", end_vector)
                print("is turn toward center?", turn_direction_toward_ctr)
                input()

        if debug:
            print("all results for turn direction toward center:")
            print(
                trj.boundary_event_stats[boundary_tp][boundary_combo][ellipse_ref_pt][
                    "turn_direction_toward_ctr"
                ][turn_type]
            )

    def calculate_turn_toward_center(
        self, start_vector, end_vector, start_angle, end_angle
    ):
        """
        Calculates if the turn is towards the center based on the provided vectors and angles.

        Parameters:
        - start_vector (np.array): The vector from the start position to the center.
        - end_vector (np.array): The vector from the end position to the center.
        - start_angle (float): The heading angle at the start of the turn.
        - end_angle (float): The heading angle at the end of the turn.

        Returns:
        - bool: True if the turn is toward the center, False otherwise.
        """
        start_to_center_angle = np.degrees(np.arctan2(start_vector[1], start_vector[0]))
        end_to_center_angle = np.degrees(np.arctan2(end_vector[1], end_vector[0]))
        start_delta = util.angleDiff(start_to_center_angle, start_angle, absVal=True)
        end_delta = util.angleDiff(end_to_center_angle, end_angle, absVal=True)
        return end_delta < start_delta

    def process_sb_contact_events(self, evt_name, boundary_orientations, is_turn):
        """
        Processes synchronization bucket (SB) contact events for each trajectory in the analysis,
        calculating statistics such as event counts, durations, and specific metrics for contactless
        events, based on the event type and boundary orientations.

        This method iterates over each training session and trajectory, computing metrics for contact
        events (or turns, if applicable) within each synchronization bucket. It supports detailed analysis
        for both standard contact events (wall or agarose) and turning events, adjusting the calculation
        based on the event's nature and the specified boundary orientations. For wall contact events,
        additional metrics like contactless rewards, trajectory lengths, and maximum distances without
        contact are calculated.

        Parameters:
        - evt_name (str): The name of the event being analyzed, which determines the type of analysis
                          performed and the metrics calculated.
        - boundary_orientations (tuple): Specifies the orientations of boundaries involved in the event,
                                         facilitating targeted analysis for each specified orientation.
        - is_turn (bool): Indicates whether the event is a turning event, affecting the metrics and
                          calculations performed.

        The processing involves several steps:
        - For each trajectory, synchronization buckets are defined and analyzed to count the occurrences
          of the specified event within each bucket.
        - For turning events, durations of turns within each bucket are also calculated.
        - Specific handling for wall contact events includes calculating metrics for periods without
          contact, aiding in understanding the flies' behavior in relation to the chamber boundaries.
        - Metrics are prepared for export in CSV format and for detailed logging.

        This method is designed to handle a wide range of event types and boundary orientations, ensuring
        flexibility in analyzing different aspects of the flies' interactions with chamber boundaries.
        """
        for t in self.trns:
            print(t.name())
            self.evts.append([])
            if is_turn:
                self.evt_durations.append([])
            else:
                self.evts_counts.append([])
            for i, trj in enumerate(self.trx):
                calc_rewards = self._getOn(t, calc=True, f=trj.f)
                ctrl_entries = self._getOn(t, ctrl=True, calc=True, f=trj.f)
                df = self._min2f(self.opts.syncBucketLenMin)
                fi, n, _ = self._syncBucket(t, df)
                n = int(n)
                orig_fi = fi
                self.firsts_by_trn.append(orig_fi)
                rewards_for_fly = self.numRewardsTot[1][0][trj.f :: len(self.flies)]
                if trj._bad or fi is None or None in self.firsts_by_trn:
                    if "turn" in evt_name:
                        n_measures_per_trn = 1
                    else:
                        n_measures_per_trn = 2
                    placeholder_nans = np.array(
                        [
                            [np.nan]
                            * len(self.flies)
                            * (len(self.trns) + 1)
                            * n_measures_per_trn
                        ]
                    )
                    for boundary_orientation in boundary_orientations:
                        if (
                            evt_name == "wall_contact"
                            and boundary_orientation in self.contactless_rewards
                        ):
                            self._append(
                                self.contactless_rewards[boundary_orientation]["trn"],
                                [np.nan] * n,
                                trj.f,
                                n,
                            )
                            self._append(
                                self.contactless_trajectory_lengths[
                                    boundary_orientation
                                ]["trn"],
                                [np.nan] * n,
                                trj.f,
                                n,
                            )
                            self._append(
                                self.contactless_max_dists[boundary_orientation]["trn"],
                                [np.nan] * n,
                                trj.f,
                                n,
                            )
                            self.convert_rewards_to_numeric(boundary_orientation, "trn")
                            trj.va.contactless_rewards[boundary_orientation]["csv"] = [
                                placeholder_nans[0][: int(len(placeholder_nans[0]) / 2)]
                            ]
                    if t.n == len(self.trns):
                        for boundary_orientation in boundary_orientations:
                            if (
                                evt_name == "wall_contact"
                                and boundary_orientation
                                in self.contactless_trajectory_lengths
                            ):
                                self.contactless_trajectory_lengths[
                                    boundary_orientation
                                ]["csv"] = np.array([np.nan] * 2)
                                self.contactless_max_dists[boundary_orientation][
                                    "csv"
                                ] = np.array([np.nan] * 2)
                            if not hasattr(trj.va, "boundary_event_rel_changes"):
                                trj.va.boundary_event_rel_changes = {
                                    evt_name: {boundary_orientation: []}
                                }
                            if evt_name not in trj.va.boundary_event_rel_changes:
                                trj.va.boundary_event_rel_changes[evt_name] = {
                                    boundary_orientation: []
                                }
                            if (
                                boundary_orientation
                                not in trj.va.boundary_event_rel_changes[evt_name]
                            ):
                                trj.va.boundary_event_rel_changes[evt_name][
                                    boundary_orientation
                                ] = []
                            if (
                                len(
                                    trj.va.boundary_event_rel_changes[evt_name][
                                        boundary_orientation
                                    ]
                                )
                                < i + 1
                            ):
                                trj.va.boundary_event_rel_changes[evt_name][
                                    boundary_orientation
                                ].append(np.nan)
                    setattr(trj.va, self.csv_attr_name, placeholder_nans)
                    for _ in boundary_orientations:
                        self.evts[-1].append(np.full(int(n), np.nan))
                        if is_turn:
                            self.evt_durations[-1].append(np.full(int(n), np.nan))
                        else:
                            self.evts_counts[-1].append(np.full(int(n), np.nan))
                    self._printBucketVals(
                        self.evts[-1][-1], trj.f, msg=flyDesc(trj.f), prec=2
                    )
                    if not hasattr(trj, "boundary_event_rel_raw_stats"):
                        trj.boundary_event_rel_raw_stats = {
                            evt_name: {
                                tp: {"start": np.nan} for tp in boundary_orientations
                            }
                        }
                    else:
                        trj.boundary_event_rel_raw_stats[evt_name] = {
                            tp: {"start": np.nan} for tp in boundary_orientations
                        }

                    continue
                num_rewards = rewards_for_fly[t.n - 1]
                la = min(t.stop, t.start + n * df)
                for j, boundary_orientation in enumerate(boundary_orientations):
                    if evt_name == "wall_contact":
                        contactless_rewards = []
                        contactless_trajectory_lengths = []
                        contactless_max_dists = []
                    if boundary_orientation == self.opts.wall_orientation:
                        self.wall_orientation_idx = j
                    evts_for_trj = []
                    durations_for_trj = []
                    fi = orig_fi
                    if not hasattr(self, "ellipse_ref_pt") or "_line" in evt_name:
                        self.ellipse_ref_pt = "edge"
                    idxs_all = trj.boundary_event_stats[self.boundary_tp][
                        boundary_orientation
                    ][self.ellipse_ref_pt]["contact_start_idxs"]
                    if "_turn" in evt_name:
                        contact_idxs = idxs_all
                        if "_line" in evt_name:
                            turning_indices = []
                            turn_type = 1 if "inside" in evt_name else 2
                            for idx in trj.boundary_event_stats[self.boundary_tp][
                                boundary_orientation
                            ]["edge"]["turning_indices"]:
                                if (
                                    trj.boundary_event_stats[self.boundary_tp][
                                        boundary_orientation
                                    ]["edge"]["turning"][idxs_all[idx]]
                                    == turn_type
                                ):
                                    turning_indices.append(idx)
                            idxs_all = idxs_all[turning_indices]
                        else:
                            turning_indices = trj.boundary_event_stats[
                                self.boundary_tp
                            ][boundary_orientation][self.ellipse_ref_pt][
                                "turning_indices"
                            ]
                            contact_idxs = idxs_all
                            idxs_all = idxs_all[turning_indices]

                    calc_entries_per_bucket = []
                    ctrl_entries_per_bucket = []
                    while fi + df <= la:
                        mask_for_indexing = (idxs_all >= fi) & (idxs_all < fi + df)
                        if "_turn" not in evt_name:
                            raw_count = int(np.count_nonzero(mask_for_indexing))

                            # per-bucket counts of calc rewards and control entries
                            n_calc = int(
                                np.count_nonzero(
                                    (calc_rewards >= fi) & (calc_rewards < fi + df)
                                )
                            )
                            n_ctrl = int(
                                np.count_nonzero(
                                    (ctrl_entries >= fi) & (ctrl_entries < fi + df)
                                )
                            )

                            # stash temporarily; we’ll threshold after the loop
                            evts_for_trj.append(raw_count)
                            # make sure these two lists exist before the loop (see Part B below)
                            calc_entries_per_bucket.append(n_calc)
                            ctrl_entries_per_bucket.append(n_ctrl)

                            if (
                                evt_name == "wall_contact"
                                and boundary_orientation in self.contactless_rewards
                            ):
                                contactless_rewards.append(0)
                                contactless_trajectory_lengths.append([])
                                contactless_max_dists.append([])
                                self.calcContactlessRewardsForSyncBucket(
                                    trj,
                                    idxs_all,
                                    calc_rewards,
                                    fi,
                                    fi + df,
                                    contactless_rewards,
                                    contactless_trajectory_lengths,
                                    contactless_max_dists,
                                )
                        else:
                            num_turns = len(idxs_all[mask_for_indexing])
                            num_contacts = len(
                                contact_idxs[
                                    (contact_idxs >= fi) & (contact_idxs < fi + df)
                                ]
                            )
                            num_events = (
                                num_turns / num_contacts
                                if num_contacts >= self.opts.turn_contact_thresh
                                else np.nan
                            )
                            durations_for_bucket = [
                                [sl.stop - sl.start]
                                for sl in [
                                    trj.boundary_event_stats[self.boundary_tp][
                                        boundary_orientation
                                    ][self.ellipse_ref_pt]["boundary_contact_regions"][
                                        idx
                                    ]
                                    for idx in np.array(turning_indices)[
                                        mask_for_indexing
                                    ]
                                ]
                            ]
                            durations_for_trj.append(np.mean(durations_for_bucket))
                            evts_for_trj.append(num_events)
                        fi += df
                    if (
                        evt_name == "wall_contact"
                        and boundary_orientation in self.contactless_rewards
                    ):
                        self._append(
                            self.contactless_rewards[boundary_orientation]["trn"],
                            contactless_rewards,
                            trj.f,
                            n,
                        )
                        self._append(
                            self.contactless_trajectory_lengths[boundary_orientation][
                                "trn"
                            ],
                            contactless_trajectory_lengths,
                            trj.f,
                            n,
                        )
                        self._append(
                            self.contactless_max_dists[boundary_orientation]["trn"],
                            contactless_max_dists,
                            trj.f,
                            n,
                        )
                        if t.n == len(self.trns):
                            self.contactless_trajectory_lengths[boundary_orientation][
                                "csv"
                            ].append(
                                self.contactless_trajectory_lengths[
                                    boundary_orientation
                                ]["trn"][len(self.flies) * (t.n - 1) + trj.f][n - 2]
                            )
                            self.contactless_max_dists[boundary_orientation][
                                "csv"
                            ].append(
                                self.contactless_max_dists[boundary_orientation]["trn"][
                                    len(self.flies) * (t.n - 1) + trj.f
                                ][n - 2]
                            )
                    if "_turn" not in evt_name:
                        # Base arrays
                        counts_raw = np.asarray(
                            evts_for_trj, dtype=float
                        )  # raw event counts (no gating yet)
                        calc_arr = np.asarray(calc_entries_per_bucket, dtype=int)
                        ctrl_arr = np.asarray(ctrl_entries_per_bucket, dtype=int)

                        # ---- 1) COUNTS METRIC (relaxed gate: calc + ctrl) ----
                        counts_ok = (calc_arr + ctrl_arr) >= self.opts.piTh
                        counts_out = counts_raw.copy()
                        counts_out[~counts_ok] = np.nan
                        counts_out = np.where(np.isinf(counts_out), np.nan, counts_out)

                        pad = int(n - len(counts_out))
                        self.evts_counts[-1].append(
                            np.hstack((counts_out, [np.nan] * pad))
                            if pad > 0
                            else counts_out
                        )

                        # ---- 2) RATIO METRIC (status quo, calc-only gate) ----
                        # num_rewards should be per-bucket; align defensively
                        nr = np.asarray(num_rewards, dtype=float)[: len(counts_raw)]

                        ratio_ok = calc_arr >= self.opts.piTh
                        bad_div = (nr == 0) | ~np.isfinite(nr)

                        ev = counts_raw.astype(float)
                        # apply calc-only threshold first
                        ev[~ratio_ok] = np.nan
                        # divide where both threshold passes and denominator is valid
                        do_div = ratio_ok & ~bad_div
                        ev[do_div] = ev[do_div] / nr[do_div]
                        # invalid denominators become NaN (already handled by ~do_div)

                        ev = np.where(np.isinf(ev), np.nan, ev)
                        ev = np.hstack((ev, [np.nan] * pad)) if pad > 0 else ev

                        # store the ratio metric
                        self.evts[-1].append(ev)
                        evts_for_trj = ev
                    else:
                        evts_for_trj = np.where(
                            evts_for_trj == np.inf, np.nan, evts_for_trj
                        )
                        n_missing_buckets = int(n - len(evts_for_trj))
                        self.evts[-1].append(
                            np.hstack((evts_for_trj, n_missing_buckets * [np.nan]))
                        )
                    if is_turn:
                        self.evt_durations[-1].append(
                            np.hstack((durations_for_trj, n_missing_buckets * [np.nan]))
                        )
                    if (
                        self.boundary_tp == "agarose"
                        or self.boundary_tp == "boundary"
                        or (
                            self.boundary_tp == "wall"
                            and boundary_orientation == self.opts.wall_orientation
                        )
                    ):
                        self._printBucketVals(
                            evts_for_trj, trj.f, msg=flyDesc(trj.f), prec=2
                        )

                    ContactEventTrainingComparison(
                        self.N,
                        t,
                        n,
                        rewards_for_fly,
                        trj,
                        evt_name,
                        self.boundary_tp,
                        boundary_orientation,
                        event_indices=idxs_all,
                        opts=self.opts,
                        save_stats=i + 1 == len(self.trx)
                        and j + 1 == len(boundary_orientations),
                        aux_data=(
                            {"contact_idxs": contact_idxs}
                            if "_turn" in evt_name
                            else {}
                        ),
                    ).calcContactMetrics()
                    self.calcContactlessRewardsForCSV(
                        evt_name, boundary_orientation, t, n, trj, idxs_all
                    )

    def print_sb_contact_output(self, evt_name, is_turn):
        """
        Prints the output of the synchronization bucket (SB) contact event analysis, including
        relative changes and metrics for each fly, based on the specified event type and whether
        it involves turning events.

        This method displays detailed results for the contact or turning events analyzed,
        indicating the relative change in contact events or turn durations across the experimental
        period. It adapts the output based on the type of boundary interaction (e.g., agarose or wall contacts)
        and the specific orientation of the wall involved in the event. Additionally, it handles the
        presentation of data based on the chosen reward boundary range option set in the analysis options.

        Parameters:
        - evt_name (str): The name of the event, determining the type of contact event and influencing
                        the output content.
        - is_turn (bool): Indicates whether the event is a turning event, affecting the calculation and
                        presentation of turn duration data.

        The method first determines the description of the boundary contact range based on the analysis options.
        It then iterates over each fly, printing the relative change in contact events or turn durations, along
        with additional statistics such as the start and end points of the analyzed range. The method also
        prepares and stores the events and, if applicable, turn durations for potential further analysis or export.

        Note:
        This method serves as the concluding step in presenting the analysis of contact events by SB, offering
        a clear and concise summary of the findings for each fly involved in the study.
        """
        self.bnd_contact_range_desc = "T1 first bucket to T2 last bucket"
        print(
            (
                "\nrelative change, %s, %s:"
                % (self.sb_contact_evt_log_header, self.bnd_contact_range_desc)
            )
        )
        if self.boundary_tp in ("agarose", "boundary"):
            tp_to_print = "tb"
        elif "turn" in evt_name:
            if "agarose" in evt_name:
                tp_to_print = "tb"
            else:
                tp_to_print = "all"
        else:
            tp_to_print = self.opts.wall_orientation
        for i, val in enumerate(self.boundary_event_rel_changes[evt_name][tp_to_print]):
            print(
                (
                    "  %s: %.2f" % (flyDesc(i), val * 100)
                    + ("%" if ~np.isnan(val) else "")
                    + (
                        (
                            " (%.2f to %.2f)"
                            % (
                                self.trx[i].boundary_event_rel_raw_stats[evt_name][
                                    tp_to_print
                                ][1],
                                self.trx[i].boundary_event_rel_raw_stats[evt_name][
                                    tp_to_print
                                ][2],
                            )
                        )
                        if 1
                        in self.trx[i].boundary_event_rel_raw_stats[evt_name][
                            tp_to_print
                        ]
                        and not np.isnan(
                            self.boundary_event_rel_changes[evt_name][tp_to_print][i]
                        )
                        else ""
                    )
                )
            )
        if not hasattr(self, "boundary_events"):
            self.boundary_events = {evt_name: np.array(self.evts)}
        else:
            self.boundary_events[evt_name] = np.array(self.evts)
        if (not is_turn) and hasattr(self, "evts_counts"):
            if not hasattr(self, "boundary_event_counts"):
                self.boundary_event_counts = {evt_name: np.array(self.evts_counts)}
            else:
                self.boundary_event_counts[evt_name] = np.array(self.evts_counts)
        if is_turn:
            array_to_save = np.array(self.evt_durations) / self.fps
            if not hasattr(self, "boundary_event_durations"):
                self.boundary_event_durations = {evt_name: array_to_save}
            else:
                self.boundary_event_durations[evt_name] = array_to_save

    def contactEventsBySyncBucket(self, evt_name):
        """
        Analyzes and processes contact events by synchronization bucket for a specified event type.

        This method organizes contact events, including turns if specified in the event name, by sync buckets,
        facilitating the alignment of events across different trajectories or instances for comparative analysis. It encompasses
        a three-step process: setting up the contact events, processing them, and then printing the output.

        Parameters:
        - evt_name (str): The name of the event to analyze, indicating the type of contact event, with the presence of '_turn'
                        signifying whether the event is a turning event.

        The method first identifies the boundary orientations involved in the event, sets up the data structures for analyzing
        contact events by synchronization buckets, processes these events to generate relevant statistics, and finally prints
        the results in a structured format.

        Note:
        This method is integral to the analysis of contact events in video trajectories, where synchronization buckets are
        used to segment and analyze the temporal distribution and frequency of these events within the video data.
        """
        is_turn = "_turn" in evt_name
        boundary_orientations = self.get_boundary_orientations(evt_name)

        self.setup_sb_contact_events(evt_name, boundary_orientations, is_turn)

        self.process_sb_contact_events(evt_name, boundary_orientations, is_turn)

        self.print_sb_contact_output(evt_name, is_turn)

    def byPostBucket(self):
        """
        Executes a series of analyses related to the post-reward phase of the experiment, focusing on
        the positional and reward preference indices, as well as the calculation of rewards obtained by
        flies during this phase.

        This method serves as a high-level orchestrator that sequentially calls:
        - `positionalPiPost()` to calculate and report the positional preference index (PI) of flies,
          indicating their preference for specific areas within the arena during the post-reward phase.
        - `calcRewardsPost()` to calculate and report the number of rewards (entries into reward areas) obtained
          by the flies during the post-reward phase.
        - `rewardPiPost()` to calculate and report the reward PI, analyzing the flies' preference for one type
          of reward over another during the post-reward phase.

        Parameters:
        - None

        By aggregating these analyses, this method provides a comprehensive overview of flies' behavior
        and preferences in the post-reward phase, encapsulating their positional preferences, engagement
        with reward areas, and reward preferences in a structured and sequential manner.
        """
        self.positionalPiPost()
        self.calcRewardsPost()
        self.rewardPiPost()

    FRAC_OF_BUCK_FOR_PI = 0.05

    def positionalPiPost(self):
        """
        Calculates and reports the positional preference index (PI) of flies in post-reward
        conditions by analyzing their proximity to defined areas (circles) within the arena.

        This method evaluates the flies' positional preferences during the post-reward phase of
        the experiment, using a defined radius multiplier to determine the effective radius around
        points of interest (usually reward sites). The PI is calculated for each synchronization
        bucket during the post-reward phase, indicating the relative preference for one area over
        another, based on the fly's locations within these defined radii.

        The method iterates over each training session that includes a symmetric control (symmetric
        placement of rewards or other stimuli) and calculates the PI based on the fly's positions
        relative to the centers of interest. The PI is a ratio indicating the preference for one
        area over another, adjusted for the total number of frames the fly is within the defined
        radius of any area of interest.

        Notes:
        - The PI is calculated as `(nf[0] - nf[1]) / nfsum`, where `nf[0]` and `nf[1]` are the
          frame counts within the radius of the first and second areas of interest, respectively,
          and `nfsum` is the sum of these counts.
        - The method prints the PI values for each analyzed bucket in the post-reward phase, along
          with a summary that includes the total post-reward duration and the radius multiplier used
          for defining the area of interest.
        - Only the first fly's trajectory is analyzed in this implementation, and the method asserts
          that the trajectory data should not be marked as 'bad' to proceed with the analysis.
        """
        blm, rm = self.opts.piBucketLenMin, self.opts.radiusMult
        df = self._min2f(blm)
        self.posPI, self.posPITrns = [], []
        print(
            "\npositional PI (r*%s) by post bucket (%s min):"
            % (util.formatFloat(rm, 2), util.formatFloat(blm, 1))
        )

        trx = self.trx[0]  # fly 1
        (x, y), bad = trx.xy(), trx.bad()
        assert not bad
        for t in self.trns:
            if not t.hasSymCtrl():
                continue
            fi, la, pis, r = t.stop, t.postStop, [], t.r * rm
            print("%s (total post: %s)" % (t.name(), frame2hm(la - fi, self.fps)))
            while fi + df <= la:
                xb, yb = x[fi : fi + df], y[fi : fi + df]
                nf = [
                    np.count_nonzero(np.linalg.norm([xb - cx, yb - cy], axis=0) < r)
                    for (cx, cy, __) in t.circles()
                ]
                nfsum = sum(nf)
                pis.append(
                    np.nan
                    if nfsum < self.FRAC_OF_BUCK_FOR_PI * df
                    else (nf[0] - nf[1]) / nfsum
                )
                fi += df
            self._printBucketVals(["%.2f" % pi for pi in pis], f=0)
            self.posPITrns.append(t)
            self.posPI.append((pis[0] if pis and not bad else np.nan,))

    def calcRewardsPost(self):
        """
        Calculates and reports the number of calculated rewards obtained by flies during the post-reward
        phase of the experiment. This method focuses exclusively on rewards defined by the flies' entries
        into a defined reward area.

        The method iterates over each training session and fly, counting the entries into the reward area
        during the post-reward phase. The count starts from the transition point between the training and
        post-reward phases, adjusted for the number of non-post buckets to ensure focus on the post-reward
        analysis. Each fly's reward count is reported, providing insight into their engagement with the
        reward areas after the training sessions have concluded.
        """
        calc, ctrl, nnpb = True, False, self.numNonPostBuckets
        df = self._numRewardsMsg(False)
        self.numRewardsPost, self.numRewardsPostPlot = [], []
        for i, t in enumerate(self.trns):
            print(
                t.name()
                + (
                    "  (values in parentheses are still training)"
                    if i == 0 and nnpb > 0
                    else ""
                )
            )
            for f in self.flies:
                nOns = self._countOnByBucket(
                    t.stop - df * nnpb, t.postStop, df, calc, ctrl, f
                )
                if self.numPostBuckets is None:
                    VideoAnalysis.numPostBuckets = len(nOns)
                nOns1 = nOns[nnpb - 1 :]
                self._printBucketVals(
                    nOns1, f, msg=self._rewardType(calc, ctrl, f), nParen=1
                )
                self._append(self.numRewardsPost, nOns1, f, n=4)
                self._append(self.numRewardsPostPlot, nOns, f, n=self.numPostBuckets)

    def distBetweenCalcForCSV(self):
        """
        Prepares and returns a list of average distances between calculated rewards, along with the
        number of rewards, formatted for CSV export. This method processes the distances calculated
        before the experiment (pre-condition) and during each training session, adjusting for rewards
        obtained and handling cases with insufficient data by marking them as NaN.

        The method iterates over each training session and fly, extracting the average distances
        between calculated rewards and the number of rewards obtained. It handles various scenarios:
        - When no rewards are calculated or data is insufficient, it marks the distance as NaN and
          includes the count of rewards calculated.
        - It adjusts the values based on the training session and concatenates them to a list intended
          for CSV export.

        Parameters:
        - None

        Returns:
        - A list of strings, each representing the average distance between calculated rewards for a
          fly in a specific training session, formatted as "<average distance> (<number of rewards>)"
          or just "<average distance>" if the number of rewards is not applicable.

        This method facilitates the analysis and reporting of how the distance between rewards changes
        over the course of the experiment, providing insights into flies' interactions with the reward
        areas in relation to the number of rewards they obtained.
        """
        values = self.avgDistBtwnCalcPre
        n_bkts = int(len(self.avgDistancesByBkt[0]) / len(self.flies))
        for t in self.trns:
            bucket_offset = 0 if t.n == 1 else n_bkts - 2
            for f in self.flies:
                distBtwnCalcVals = self.avgDistBtwnCalc[(t.n - 1) * len(self.flies) + f]
                if np.isnan(distBtwnCalcVals[0]):
                    values.extend([np.nan, np.nan])
                elif np.isnan(distBtwnCalcVals[1]):
                    values.extend(
                        [
                            distBtwnCalcVals[0],
                            np.nan,
                        ]
                    )
                else:
                    values.extend(
                        list(self.avgDistBtwnCalc[(t.n - 1) * len(self.flies) + f])
                    )
            for f in self.flies:
                dist = self.avgDistancesByBkt[t.n - 1][n_bkts * f + bucket_offset]
                rewards_for_fly = self.numRewardsTot[1][0][f :: len(self.flies)]
                if not (
                    ~np.isnan(dist)
                    or len(rewards_for_fly[t.n - 1]) < bucket_offset + 1
                    or np.isnan(rewards_for_fly[t.n - 1][bucket_offset])
                ):
                    dist = np.nan
                values.append(dist)

        return values

    # returns list containing per-fly lists of reward PI values in the following
    # order: last 10 min of pre, T1's first sync bucket, T1's first two
    # post-buckets, T2's last sync bucket, T2's first two post-buckets, T3's last
    # sync bucket, and T3's first two post-buckets
    def rewardPiCombined(self):
        pis = []
        spacing = 1 if self.noyc else 2
        for f in self.flies:
            rpis, rpips = self.rewardPI[f::spacing], self.rewardPiPst[f::spacing]
            alt_rpips = self.rewardPiPstNoSync[f::spacing]
            postVals = []
            for i, rpip in enumerate(rpips):
                if i == 2:
                    # special format for training 3 - includes single bucket without sync
                    postVals.append([rpip[0], alt_rpips[i][0], rpip[1]])
                else:
                    postVals.append(rpip[0:2])
            postVals = util.concat(postVals)
            pis.append([self.rewardPIPre[f], rpis[0][0]] + postVals)
            if np.all(np.isnan(pis[f])):
                self.trx[0].bad(True)
            for tIdx, valsIdx in ((1, 4), (2, 7)):
                if tIdx + 1 > len(self.trns):
                    continue
                if np.all(np.isnan(self.buckets)):
                    nBuckets = 0
                else:
                    # buckets marks the start/end indices, so the
                    # quantity is one fewer than the length.
                    nBuckets = len(self.buckets[tIdx]) - 1
                rpisForTrn = np.array(rpis[tIdx])
                pis[f].insert(
                    valsIdx,
                    # select one bucket back from the theoretical final bucket
                    (np.nan if nBuckets == 0 else rpisForTrn[nBuckets - 2]),
                )
        interspersed_pis = []
        for i in range(len(pis[0])):
            for j in range(len(pis)):
                interspersed_pis.append(pis[j][i])
        return [interspersed_pis]

    def rewardPiPost(self):
        """
        Calculates and reports the reward preference index (PI) for flies during the post-reward
        phase of the experiment. This method evaluates the flies' reward preferences by analyzing
        their entries into reward areas, with consideration for both synchronized and non-synchronized
        reward delivery times.

        The reward PI is calculated for each fly across all training sessions, with additional
        analysis for reward crossings. The method differentiates between synchronized (aligned with
        specific experimental events) and non-synchronized (independent of such events) reward
        deliveries, calculating PIs for both scenarios. It prints the calculated PIs, along with
        the number of reward crossings, providing insights into flies' behavior in relation to
        the provided rewards during the post-reward phase.

        Parameters:
          - None

        The method iterates over each training session and each fly, calculating the reward PI
        and the total number of reward crossings, adjusted for synchronization with experimental
        events. The results are printed for each fly, highlighting their preference for one reward
        type over another in the post-reward phase and indicating the extent of their engagement
        with the reward areas.

        Notes:
          - The reward PI provides a quantifiable measure of the flies' reward preferences, offering
            valuable insights into the effectiveness of the training and their subsequent behavior.
          - The method accounts for the possibility of continued training activity affecting the
            reward PI calculations, especially in the initial training session.
        """
        calc, blm, nnpb = True, self.opts.rpiPostBucketLenMin, self.rpiNumNonPostBuckets
        print(
            "\nreward PI by post %sbucket (%s min)"
            % ("" if POST_SYNC is ST.fixed else "sync ", util.formatFloat(blm, 1))
        )
        df = self._min2f(blm)
        self.rewardPiPst = []
        self.rewardPiPstNoSync = []
        self.numPostCrossings = []
        self.numPostCrossingsNoSync = []
        for i, t in enumerate(self.trns):
            print(
                t.name()
                + (
                    "  (values in parentheses are still training)"
                    if i == 0 and nnpb > 0
                    else ""
                )
            )
            pfi = util.none2val(self._postSyncBucket(t), t.postStop)
            fiRi = util.none2val(
                self._idxSync(RI_START_POST, t, pfi, t.postStop), t.postStop
            )
            for f in self.flies:
                for useSync in (True, False):
                    nOns = []
                    crossings = []
                    for ctrl in (False, True):
                        nOns.append(
                            util.concat(
                                self._countOnByBucket(fi, la, df, calc, ctrl, f, fiC)
                                for fi, la, fiC in (
                                    (t.stop - df * nnpb, t.stop, None),
                                    (pfi, t.postStop, fiRi if useSync else None),
                                )
                            )
                        )

                        if useSync and ctrl and t.n == len(self.trns):
                            self.numRewardsTotPrePost[3].append(np.sum(nOns[1]))
                    pis = util.prefIdx(nOns[0], nOns[1], n=self.opts.piTh)
                    crossings = np.add(nOns[0], nOns[1])
                    if self.rpiNumPostBuckets is None:
                        VideoAnalysis.rpiNumPostBuckets = (
                            nnpb + (t.len(post=True) - 3 * self.fps) // df
                        )
                    self._append(
                        self.rewardPiPst if useSync else self.rewardPiPstNoSync,
                        pis,
                        f,
                        n=self.rpiNumPostBuckets,
                    )
                    self._append(
                        (
                            self.numPostCrossings
                            if useSync
                            else self.numPostCrossingsNoSync
                        ),
                        crossings,
                        f,
                        n=self.rpiNumPostBuckets,
                    )
                    if useSync:
                        self._printBucketVals(
                            pis, f, msg=flyDesc(f), prec=2, nParen=nnpb
                        )

    # - - -

    # analyzes, e.g., time between rewards
    def byReward(self):
        self.byActualReward()
        self.byCalcReward()

    def byTraining(self):
        """
        Collects data having one value per training (and potentially per pre- and post-period).
        Updates the `pctInC` attribute based on the contents of the `pctInC` attribute of `Trajectory`
        instances in `self.trx`.

        The `pctInC` attribute is a dictionary with keys 'rwd' and 'custom', each containing
        lists of percentages for the reward circle and custom circle, respectively.
        """
        self.pctInC = {}

        # Iterate over the keys in the pctInC dictionary ('rwd' and 'custom')
        for key in ["rwd", "custom"]:
            self.pctInC[key] = [
                [trj.pctInC[key][i] for trj in self.trx if not trj._bad]
                for i in range(len(self.trx[0].pctInC[key]))
            ]

    def _byRewardMsg(self, calc):
        """
        Prints a message specifying the criteria for comparing rewards in the experiment based on
        the calculation type, and returns the number of rewards to compare.

        This method outputs a message to indicate the comparison being made between sets of rewards
        within the experiment, specifically stating whether the comparison is based on calculated or
        actual rewards. It details the number of rewards in the first set versus the next set for
        comparison, as specified in the experiment options.

        Parameters:
        - calc (bool): Indicates whether the comparison is based on calculated rewards (True) or actual
                       rewards (False).

        Returns:
        - int: The number of rewards specified for comparison, as defined in the experiment options.

        The method is primarily used for logging purposes, to clearly communicate the basis of reward
        comparisons being conducted as part of the experiment's analysis.
        """
        nrc = self.opts.numRewardsCompare
        print("\nby %s reward: (first %d vs. next %d)" % (cVsA_l(calc), nrc, nrc))
        return nrc

    def _plot(self, sp, data, title, xlabel, ylabel, ylim, f=None):
        """
        Creates a subplot with data and optional moving averages, customized with titles, labels, and limits.

        This method generates a plot or a subplot within a larger figure, plotting the raw data along with
        moving averages for specified window sizes. It's designed to be flexible, allowing customization of
        the plot's appearance, including the color and style of the lines, the plot title, axis labels, and
        the y-axis limits. Special handling is provided for certain cases, such as when plotting data for a
        single fly.

        Parameters:
        - sp (tuple): Subplot configuration as a tuple (nrows, ncols, index), specifying the layout and
                      position of the subplot within a figure.
        - data (list or np.array): The data to be plotted.
        - title (str): The title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - ylim (tuple): A tuple specifying the limits of the y-axis (min, max).
        - f (int, optional): Specifies whether the plot is for a single fly (f=1) or for aggregated data.
                             Certain lines or averages may be omitted based on this parameter.

        The method supports visualization for analyzing trends in the data, with conditional plotting of
        moving averages using different window sizes (e.g., 25, 50, 100, 200) to smooth the data and reveal
        underlying patterns. The color and style of these lines can be adjusted, and specific configurations
        are applied based on the context of the data being plotted (e.g., data for a single fly versus
        aggregated data).

        Note:
        This method is intended for internal use within the class for data visualization purposes, especially
        when analyzing time series data or distances in an experimental context.
        """

        def xrng(ys, off=0):
            return list(range(1 + off, len(ys) + 1 + off))

        ax = plt.subplot(*sp)
        if f != 1:
            plt.plot(xrng(data), data, color="0.5")
        for i, (n, c) in enumerate(
            ((25, (0.5, 0.5, 0)), (50, "g"), (100, "b"), (200, "r"))
        ):
            if f == 1 and n != 100:
                continue
            if len(data) > n:
                avgs = np.convolve(data, np.ones(n) / n, mode="valid")
                plt.plot(xrng(avgs), avgs, color=c, linestyle="--" if f == 1 else "-")
                if sp[2] == 1:
                    plt.text(
                        0.75,
                        0.85 - i * 0.08,
                        "n = %d" % n,
                        color=c,
                        transform=ax.transAxes,
                    )
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(*ylim)

    def _firstNVsNext(self, data, n, lbl, appendTo, f=None):
        """
        Compares and reports the averages of the first N and the next N data points in a given dataset,
        optionally for a specific entity (e.g., a fly). This method calculates the mean of these segments
        and appends the results to a provided list. It also prints a summary of the comparison, indicating
        average values or noting if the data is considered 'bad' and thus not suitable for analysis.

        Parameters:
        - data (list or np.array): The dataset from which to extract and compare segments.
        - n (int): The number of data points in each segment for comparison.
        - lbl (str): A label describing the data being compared (e.g., "time between [s]", "distance between").
        - appendTo (list): A list to which the comparison results (tuples of averages) will be appended.
        - f (int, optional): An identifier for a specific entity within the dataset, such as a fly number.
                             If provided, the comparison is considered in the context of this entity.

        The method segments the data into the first N and next N points, calculates the mean of each segment,
        and handles cases where the dataset is insufficient (marking such comparisons as NaN). It's designed
        to support analyses where comparing sequential segments of data is meaningful, such as analyzing
        behavioral changes or performance over time.

        Note:
        This method is particularly useful in experimental contexts where understanding the change or
        difference between sequential segments of data can provide insights into behavioral or performance
        trends, especially when analyzing time series data or spatial measurements.
        """
        bad = self._bad(f)
        a = tuple(
            (
                np.mean(data[i * n : (i + 1) * n])
                if not self._bad(f) and (i + 1) * n <= len(data)
                else np.nan
            )
            for i in range(2)
        )
        appendTo.append(a)
        print(
            "  avg. %s%s: %s"
            % (
                lbl,
                "" if f is None else " (%s)" % (flyDesc(f)),
                self._vals("%.1f vs. %.1f" % a, f),
            )
        )

    def _distTrav(self, f, on):
        """
        Calculates the distances traveled by a fly between consecutive rewards, ensuring the sum of
        these distances matches the total distance traveled from the first to the last reward.

        This method iterates through pairs of consecutive rewards, calculating the distance traveled
        by the fly between each pair. It is designed to work with individual flies, identified by
        their index, and uses the trajectory data stored in the class to perform the calculations.

        Parameters:
        - f (int): The index of the fly for which to calculate distances.
        - on (list or np.array): A list or array of reward occurrence times for the specified fly.

        Returns:
        - list: A list of distances traveled between consecutive rewards. If the fly's trajectory is
                marked as 'bad' or if there are no rewards, an empty list is returned.

        This method is crucial for analyzing the spatial dynamics of how flies interact with rewards
        over time, providing a detailed look at the distances covered in pursuit of these rewards.

        Note:
        - The method asserts that the sum of individual distances matches the total distance from the
          first to the last reward, serving as a consistency check for the trajectory data.
        """
        if self._bad(f):
            return []
        trx, db = self.trx[f], []
        for fi, la in zip(on[:-1], on[1:]):
            db.append(trx.distTrav(fi, la))
        dt = trx.distTrav(on[0], on[-1]) if len(on) else np.nan
        assert not db or np.isnan(dt) or np.isclose(sum(db), dt)
        return db

    def byActualReward(self):
        """
        Analyzes and compares the time and distance between actual rewards obtained by flies
        during training sessions. This method calculates the average time and distance between
        the first N rewards and the subsequent N rewards, where N is specified by the experiment's
        configuration. It also generates plots for these metrics if visualization is enabled.

        The method iterates over each training session, extracting the actual reward occurrences
        for each fly and then calculating the time intervals and distances between rewards. It
        leverages the `_firstNVsNext` method to compare these intervals and distances for the first
        N rewards versus the next N rewards, appending the results to class attributes for further
        analysis or reporting. If plotting is enabled, it visualizes these comparisons for each
        training session and each fly, showcasing the temporal and spatial dynamics of reward
        interactions.

        Parameters:
        - None

        This method provides a detailed look at how flies' interactions with rewards evolve over
        the course of training, highlighting any changes in behavior that may occur as flies become
        more experienced or fatigued. By comparing the first N rewards to the next N, it offers
        insights into learning patterns, efficiency changes, or preferences that develop over time.

        Note:
        - The comparison metric (time or distance between rewards) and the number of rewards to compare
          (N) are configurable, allowing for flexible analysis tailored to specific experimental goals.
        - Visualization of these metrics is optional and can be controlled through the experiment's
          configuration settings, aiding in the interpretation and presentation of the findings.
        """
        nrc = self._byRewardMsg(False)
        self.avgTimeBetween, self.avgDistBetween = [], []
        if self.opts.showPlots:
            plt.figure(util.basename(self.fn), (20, 10))
        for i, t in enumerate(self.trns):
            print(t.name())
            tnl, xlbl = t.name(short=False), "reward"
            on = self._getOn(t)
            nr = len(on) if self.opts.plotAll else nrc * 2 + 1
            on1 = on[:nr]

            ylbl = "time between [s]"
            tb = np.diff(on1) / self.fps
            self._firstNVsNext(tb, nrc, ylbl, self.avgTimeBetween)
            if self.opts.showPlots:
                self._plot((2, 3, 1 + i), tb, tnl, xlbl, ylbl, (0, 40))

            ylbl = "distance between"
            for f in self.flies:
                db = self._distTrav(f, on1)
                self._firstNVsNext(db, nrc, ylbl, self.avgDistBetween, f)
                if self.opts.showPlots and not self._bad(f):
                    self._plot((2, 3, 4 + i), db, None, xlbl, ylbl, (0, 1600), f)

    def calcDistanceBetweenRewardsPre(self):
        """
        Calculates the average distance traveled by each fly between rewards in the pre-reward phase,
        specifically within a ten-minute window leading up to the start of the first training session.

        This method assesses the flies' movement in relation to reward locations before any training has
        commenced, offering insights into their natural movement patterns or initial exploratory behavior.
        For each fly, it filters reward events occurring within the ten-minute pre-training window and
        calculates the average distance traveled between these reward points.

        The method accounts for cases where flies are marked as having bad data or where the number of
        reward events does not meet a predefined threshold (`piTh`), setting the average distance for
        these cases to NaN. The calculated averages are stored for each fly, providing a basis for
        comparison with post-training behavior or for analyzing initial exploratory tendencies.

        Parameters:
        - None

        Notes:
        - The analysis is limited to a ten-minute window immediately preceding the first training session,
          focusing on pre-training exploratory behavior.
        - Flies with insufficient reward events or marked as 'bad' are excluded from the calculation, ensuring
          the reliability of the data considered in the analysis.
        - The resulting average distances are appended to `avgDistBtwnCalcPre`, offering a concise summary of
          pre-training movement in relation to reward locations for each fly.
        """
        self.avgDistBtwnCalcPre = []
        ten_min_in_frames = self.fps * 10 * 60
        reward_range = slice(self.trns[0].start - ten_min_in_frames, self.trns[0].start)
        for f in self.flies:
            if self.trx[f].bad():
                self.avgDistBtwnCalcPre.append(np.nan)
                continue
            on = self.trx[f].en[False]
            on = on[(on >= reward_range.start) & (on < reward_range.stop)]
            if len(on) < self.opts.piTh:
                self.avgDistBtwnCalcPre.append(np.nan)
                continue
            db = np.array(self._distTrav(f, on))
            self.avgDistBtwnCalcPre.append(np.mean(db))

    def byCalcReward(self):
        """
        Analyzes calculated rewards obtained by flies, evaluating the time and distance between
        these rewards. This method extends the analysis to include pre-experiment calculations
        and comparisons for each training session.

        Initially, the method calculates the average distance between calculated rewards before
        the experiment starts. It then iterates through each training session, analyzing the
        calculated rewards obtained by each fly. For each fly, it calculates:
        - The time intervals between consecutive calculated rewards.
        - The distances traveled between consecutive calculated rewards.

        The analysis compares the first N calculated rewards to the subsequent N, where N is a
        configurable number specified in the experiment's settings. The method prints the training
        session names and computes the average time and distance between calculated rewards, appending
        these averages to class attributes for further analysis or reporting.

        Parameters:
        - None

        This method provides insights into the behavioral dynamics of flies in relation to calculated
        rewards, highlighting how their interactions with reward areas evolve over the course of the
        training. By focusing on calculated rewards, it offers an understanding of flies' movements
        and decisions based on their perception of reward locations, rather than actual reward delivery.

        Note:
        - The method prepares data for potential visualization, plotting the results if enabled in
          the experiment's configuration settings. It is particularly useful for assessing learning
          patterns, efficiency changes, or preferences that develop over time in response to calculated
        rewards.
        """
        nrc = self._byRewardMsg(True)
        self.avgTimeBtwnCalc, self.avgDistBtwnCalc = [], []
        self.avgDistancesByBkt = []
        self.calcDistanceBetweenRewardsPre()
        for t in self.trns:
            print(t.name())
            self.avgDistancesByBkt.append([])
            on = [self._getOn(t, True, f=f) for f in self.flies]
            db = [np.array(self._distTrav(f, on[f])) for f in self.flies]
            for f in self.flies:
                tb = np.diff(on[f][: nrc * 2 + 1]) / self.fps
                self._firstNVsNext(tb, nrc, "time between [s]", self.avgTimeBtwnCalc, f)
            for f in self.flies:
                if t.n == len(self.trns) and f == 0:
                    self.t_lengths_for_plot = []
                    if self.opts.wall:
                        self.t_length_contactless_mask = []
                self._firstNVsNext(
                    db[f], nrc, "distance between", self.avgDistBtwnCalc, f
                )
                if self._bad(f):
                    self.avgDistancesByBkt[-1].extend(
                        [np.nan] * (len(self.buckets[t.n - 1]) - 1)
                    )
                    continue
                num_rewards = self.numRewardsTot[1][0][f :: len(self.flies)][t.n - 1]
                for i, bkt in enumerate(self.buckets[t.n - 1][:-1]):
                    bkt_range = [bkt, self.buckets[t.n - 1][i + 1]]
                    if (
                        None in bkt_range
                        or np.any(np.isnan(bkt_range))
                        or num_rewards[i] < self.opts.piTh
                    ):
                        self.avgDistancesByBkt[-1].append(np.nan)
                        continue
                    distance_mask = (on[f] > bkt_range[0]) & (on[f] <= bkt_range[1])
                    if np.all(distance_mask == False):
                        self.avgDistancesByBkt[-1].append(np.nan)
                        continue
                    distance_mask[util.trueRegions(distance_mask)[0].stop - 1] = False
                    if t.n == 3 and f == 0:
                        new_t_lengths = db[f][distance_mask[:-1]]
                        self.t_lengths_for_plot.extend(new_t_lengths)
                        reward_idxs = on[f][distance_mask]

                        if self.opts.wall:
                            bcis = self.trx[f].boundary_event_stats["wall"]["all"][
                                "edge"
                            ]["contact_start_idxs"]
                            new_t_length_contactless_mask = np.zeros(
                                (len(new_t_lengths)), dtype=int
                            )
                            prev_rew = reward_idxs[0]
                            for j, rew_idx in enumerate(reward_idxs[1:]):
                                if len(bcis[(bcis > prev_rew) & (bcis < rew_idx)]) == 0:
                                    new_t_length_contactless_mask[j] = 1
                                prev_rew = rew_idx
                            self.t_length_contactless_mask.extend(
                                new_t_length_contactless_mask
                            )

                    self.avgDistancesByBkt[-1].append(
                        np.mean(db[f][distance_mask[:-1]])
                    )
                if t.n == len(self.trns) and f == 0:
                    self.t_lengths_for_plot = np.array(self.t_lengths_for_plot)
                    if self.opts.wall:
                        self.t_length_contactless_mask = np.array(
                            self.t_length_contactless_mask
                        ).astype(bool)

    # - - -

    # - - -

    # analyze after RDP simplification
    def rdpAnalysis(self):
        blm, eps, t = 10, self.opts.rdp, self.trns[-1]
        print("\nanalysis after RDP simplification (epsilon %.1f)" % eps)
        self.rdpInterval = "last %s min of %s" % (util.formatFloat(blm, 1), t.name())
        print(self.rdpInterval)
        assert self.circle and len(self.trns) == 3 and t.tp is t.TP.center
        self.rdpAvgLL, self.rdpTA = [], []
        on = self._getOn(t)
        f1, d, ta = None, [[], []], [[], []]
        for f2 in util.inRange(on, t.stop - self._min2f(blm), t.stop):
            if f1:
                for f in self.flies:
                    sxy = self.trx[f].xyRdp(f1, f2 + 1, epsilon=eps)
                    d[f].extend(util.distances(sxy[0]))
                    ta[f].append(util.turnAngles(sxy[0]))
            f1 = f2
        print("avg. line length")
        for f in self.flies:
            mll = np.mean(d[f]) if len(d[f]) >= RDP_MIN_LINES else np.nan
            print("  %s: %.1f" % (flyDesc(f), mll))
            self._append(self.rdpAvgLL, mll, f)
        print("turn analysis")
        for f in self.flies:
            nt, ndc = 0, 0
            for ta1 in ta[f]:
                tas = np.sign(ta1)
                assert np.count_nonzero(tas) == len(tas) == len(ta1)
                # note: RDP should guarantee there are no 0-degree turns
                nt += len(tas)
                ndc += np.count_nonzero(np.diff(tas))
            print(
                "  %s: same direction: %s  number turns: %d"
                % (flyDesc(f), "{:.2%}".format((nt - ndc) / nt) if nt else "-", nt)
            )
            self.rdpTA.append(None if self._bad(f) else ta[f])

    # - - -

    # calculate chamber background
    # note: used for both heatmaps and LED detector; only one background saved
    #  currently (correct only if heatmaps and LED detector not used together)
    def background(self, channel=BACKGROUND_CHANNEL, indent=0):
        if self.bg is None:
            print(" " * indent + "calculating background (channel: %d)..." % channel)
            n, nf, nmax, frames = 0, 11, self.trns[-1].postStop, []
            dn = nmax * 0.8 / nf
            for i in range(nf):
                n += random.randint(util.intR(0.2 * dn), util.intR(1.8 * dn))
                frames.append(
                    util.toChannel(util.readFrame(self.cap, min(n, nmax - 1)), channel)
                )
            self.bg = np.median(frames, axis=0)
        return self.bg

    # note: assumes template coordinates
    #  e.g., for large chamber w/out yoked controls, mirror() makes flies 1-3
    #   look like fly 0
    # TODO: replace with Xformer's built-in _mirror()?
    def mirror(self, xy):
        if self.ct in (CT.large, CT.large2):
            sep = 321 if self.ct is CT.large2 else 268
            return [
                2 * sep - xy[0] if self.ef % 2 else xy[0],
                2 * sep - xy[1] if self.noyc and self.ef > 1 else xy[1],
            ]
        else:
            return xy

    # calculate maps for heatmaps
    def calcHm(self):
        self.heatmap, self.heatmapPost = [[], []], [[], []]  # index: fly, training
        self.heatmapOOB = False
        startPost = RI_START_POST if self.circle else ST.fixed
        for i, t in enumerate(self.trns):
            for f in self.flies:
                if self.ct is CT.regular:
                    xym = np.array(((-30, 108)[f], -24))
                    xyM = np.array(((90, 228)[f], 164))
                elif self.ct is CT.large:
                    sw = 36
                    xym = np.array((4 - sw, (4 - sw, 286)[f]))
                    xyM = np.array((250, (250, 532 + sw)[f]))
                else:
                    error("heatmap not yet implemented")
                bins, rng = [int(el) for el in (xyM - xym) / HEATMAP_DIV], np.vstack(
                    (xym, xyM)
                ).T
                trx = self.trx[f]
                for j, hm in enumerate((self.heatmap, self.heatmapPost)):
                    if j == 0:
                        fi, la, skip = t.start, t.stop, False
                    else:
                        # note: should there be limit how late fi can be?
                        fi = util.none2val(self._postSyncBucket(t, skip=0))
                        la = fi + self._min2f(self.opts.rpiPostBucketLenMin)
                        if la > t.postStop:
                            error(
                                "post training bucket length (%g min) exceeds post period"
                                % self.opts.rpiPostBucketLenMin
                            )
                        fiRi = util.none2val(
                            self._idxSync(startPost, t, fi, la, skip=0), la
                        )
                        skip = not la <= t.postStop  # correct also if la is NaN
                    if trx.bad() or skip:
                        hm[f].append((None, None, xym))
                        continue
                    xy = self.mirror([a[fi:la] for a in self.xf.f2t(trx.x, trx.y)])
                    for a, m, M in zip(xy, xym, xyM):
                        if not (m < np.nanmin(a) and np.nanmax(a) < M):
                            self.heatmapOOB = True
                        if j:
                            a[0 : fiRi - fi] = np.nan
                    xy = [a[trx.walking[fi:la]] for a in xy]
                    assert np.array_equal(np.isnan(xy[0]), np.isnan(xy[1]))
                    xy = [a[~np.isnan(a)] for a in xy]
                    # due to interpolation, there should be no NaNs due to lost flies
                    mp = np.histogram2d(xy[0], xy[1], bins=bins, range=rng)[0]
                    hm[f].append((mp.T, la - fi, xym))

    # - - -

    # positional preference
    def posPref(self):
        blm, numB = self.opts.piBucketLenMin, self.opts.numBuckets
        print(
            "\npositional preference (for top), including "
            + util.formatFloat(blm, 1)
            + "-min post buckets:"
        )
        if self.opts.skip:
            print("  " + skipMsg(self.opts.skip))
        self.posPI, sf = [], self._min2f(self.opts.skip)
        self.botToTopCrossings = []

        for t in self.trns:
            print(t.name())
            self.botToTopCrossings.append([])
            for f in self.flies:
                fi, la, df = t.start, t.postStop, t.len() / numB
                pis, o = [], []
                n_top_crossings = 0
                while fi + df <= la:
                    fiI, skip = util.intR(fi), False
                    ivs = (
                        [(fiI, fiI + sf)] if self.opts.skip and self.opts.skipPI else []
                    ) + [(fiI + sf, util.intR(fi + df))]
                    for i, (f1, f2) in enumerate(ivs):
                        y = self.trx[f].y[f1:f2]
                        inT, inB = y < t.yTop, y > t.yBottom
                        vt, vb = (len(util.trueRegions(a)) for a in (inT, inB))
                        if len(o) < numB:
                            n_top_crossings += vt
                        elif len(o) == numB:
                            n_top_crossings = vt
                        nt, nb = (np.count_nonzero(a) for a in (inT, inB))
                        if i == len(ivs) - 1:
                            skip |= vt < self.opts.minVis or vb < self.opts.minVis
                        if len(ivs) > 1 and i == 0:
                            skip |= nt == 0 or nb == 0
                    pi = util.prefIdx(nt, nb)
                    pis.append(np.nan if skip else pi)
                    o.append("%s%.2f" % ("post: " if len(o) == numB else "", pi))
                    if len(o) >= numB:
                        self.botToTopCrossings[-1].append(n_top_crossings)
                    fi += df
                    if len(o) == numB:
                        df = self._min2f(blm)
                        assert np.isclose(fi, t.stop)
                self._append(self.posPI, pis, f, n=2)
                print("  %s: %s" % (flyDesc(f), ", ".join(o)))
                self.printBotToTopCrossings(blm)

    def printBotToTopCrossings(self, blm):
        print(
            "\n# crossings from bottom to top, including "
            + util.formatFloat(blm, 1)
            + "-min post buckets:"
        )
        for t in self.trns:
            print(t.name())
            for f in self.flies:
                ct_list = self.botToTopCrossings[t.n - 1]
                out = [
                    f"{'post: ' if i + 1 == len(ct_list) else ''}{item}"
                    for i, item in enumerate(ct_list)
                ]
                print(f"  {flyDesc(f)}: {', '.join(out)}")

    # positional preference for open loop protocols (both on-off and alternating
    #  side)
    def posPrefOL(self):
        print("\npositional preference for LED side:")
        self.posPI = []
        for t in self.trns:
            print(t.name())
            assert t.yTop == t.yBottom
            ivs = ((self.startPre + 1, t.start), (t.start, t.stop))
            # y coordinate of trajectory can be NaN for frame startPre
            on = self._getOn(t)
            if not self.alt:
                off = util.inRange(self.off, t.start, t.stop)
            img = self.extractChamber(util.readFrame(self.cap, on[0] + 2))
            if self.ct is not CT.regular:
                self.trx[0].annotateTxt(img, show="f")
            self.olimg = img
            assert on[0] + 1 < on[1] and on[0] <= t.start + 1 and on[-1] <= t.stop

            def posPrefByFly():
                with np.errstate(invalid="ignore"):  # suppress warnings due to NaNs
                    inT, pis = self.trx[f].y < t.yTop, []
                if self.alt:
                    for i in range(1, len(on), 2):
                        inT[on[i] : on[i + 1] if i + 1 < len(on) else t.stop] ^= True
                else:
                    mask = np.zeros_like(inT, dtype=int)
                    mask[on] = 1
                    mask[off] -= 1
                    mask = np.cumsum(mask)
                    assert mask.min() == 0 and mask.max() == 1
                for i, (f1, f2) in enumerate(ivs):
                    inT1, pre, onOff = inT[f1:f2], i == 0, i == 1 and not self.alt
                    useMask = pre or onOff
                    # for HtL, tracking may not have started at beginning of pre period
                    if useMask:
                        mask1 = ~np.isnan(self.trx[f].y[f1:f2]) if pre else mask[f1:f2]
                    maskFlagRange = 2 if onOff else 1
                    for j in range(maskFlagRange):
                        if self.trx[f].bad():
                            pis.append(np.nan)
                        else:
                            if not (
                                pre
                                or j == 1
                                or not np.any(np.isnan(self.trx[f].y[f1:f2]))
                            ):
                                return [np.nan] * len(ivs) * maskFlagRange
                            if j == 1:
                                mask1 ^= 1
                            nt = np.count_nonzero(inT1 & mask1 if useMask else inT1)
                            nb = (np.count_nonzero(mask1) if useMask else f2 - f1) - nt
                            pis.append(util.prefIdx(nt, nb))
                return pis

            for f in self.flies:
                pis = posPrefByFly()
                self._append(self.posPI, pis, f, n=2 if self.alt else 3)
                print(
                    "  %s: %.2f (pre), %.2f%s"
                    % (
                        flyDesc(f),
                        pis[0],
                        pis[1],
                        "" if self.alt else " (on), %.2f (off)" % pis[2],
                    )
                )

    def plotYOverTime(self):
        df, nr, fn = (
            self._min2f(self.opts.piBucketLenMin),
            4,
            util.basename(self.fn, False),
        )
        ledC = "#70e070" if self.opts.green else "#ff8080"
        for t in self.trns:
            assert t.ct is CT.regular
            plt.figure(figsize=(20, 4 * nr))
            yc = self.xf.t2fY(70)
            for f in self.flies:
                fi, la = t.start, t.stop
                dm = max(
                    abs(y - yc)
                    for y in util.minMax(self.trx[f].y[t.start : t.postStop])
                )
                ymm = (yc - dm, yc + dm)
                for post in (False, True):
                    plt.subplot(nr, 1, 1 + 2 * f + post)
                    plt.yticks([])
                    plt.ylim(ymm[::-1])
                    if post:
                        fi, la = t.stop, min(t.stop + df, t.postStop)
                    x = self._f2min(np.arange(fi, la))
                    xmm = x[[0, -1]]
                    plt.xlim(xmm)
                    y = self.trx[f].y[fi:la]
                    for e in self._f2min(util.inRange(self.on, fi, la)):
                        plt.plot((e, e), ymm, color=ledC)
                    plt.plot(x, y, color=".2")
                    if hasattr(t, "yTop"):
                        for y in (t.yTop, t.yBottom):
                            plt.plot(xmm, (y, y), color=".5", ls="--")
                    plt.title(
                        "post"
                        if post
                        else "fly %d%s" % (f + 1, " [%s]" % t.name() if f == 0 else "")
                    )
            plt.savefig(TRX_IMG_FILE % (fn, t.n), bbox_inches="tight")
            plt.close()

    # - - -

    def distance(self):
        """
        Calculates and reports the distance traveled by each fly within each training session,
        segmented according to a predefined number of intervals (buckets).

        This method iterates through each training session and divides the session into equal-sized
        intervals based on the number of buckets specified in the options. It calculates the distance
        traveled by each fly within these intervals, summing up these distances to obtain the total
        distance traveled during the session. The method verifies the consistency of these calculations
        by comparing the sum of interval distances to the total distance computed directly from the
        start to the stop of the training session.
        """
        numB = self.opts.numBuckets
        print("\ndistance traveled:")
        for t in self.trns:
            print(t.name())
            df = t.len() / numB
            for f in self.flies:
                la, ds, trx = t.start, [], self.trx[f]
                for i in range(numB):
                    fi, la = la, t.start + util.intR((i + 1) * df)
                    ds.append(trx.distTrav(fi, la))
                td = sum(ds)
                assert np.isclose(trx.distTrav(t.start, t.stop), td)
                self._printBucketVals(ds, f, "%s (%.0f)" % (flyDesc(f), td), prec=0)

    # - - -

    # speed stats
    def speed(self):
        preLenMin, spMinNFrms, bt = 10, 100, SPEED_ON_BOTTOM
        print(
            "\nspeed stats (with values for "
            + util.formatFloat(preLenMin, 1)
            + "-min pre period first):"
        )
        df = self._min2f(preLenMin)
        self.speed, self.stopFrac = [], []
        self.speedLbl = "speed %s[%s/s]" % (
            "bottom " if bt else "",
            "mm" if bt else "px",
        )
        fi = 0
        for t in self.trns:
            print(t.name())
            # check whether pulse in pre period
            on = util.inRange(self.on, fi, t.start)
            pls = on[-1] if len(on) else t.start
            assert len(on) <= 1  # at most one pulse in pre period
            fi = t.stop + 1  # pulse can happen on t.stop frame
            for f in self.flies:
                trx = self.trx[f]
                noSp = not hasattr(trx, "sp")
                sps, stpFs = [], []
                for pre in (True, False):
                    if noSp:
                        continue
                    f1, f2 = (pls - df, pls) if pre else (t.start, t.stop)
                    sp1 = trx.sp[f1:f2]
                    if bt:
                        sp1 = sp1[trx.onBottomPre[f1:f2]] / trx.pxPerMmFloor
                        # print ">>>", t.n, f, pre, len(sp1)
                    sps.append(np.nan if len(sp1) < spMinNFrms else np.mean(sp1))
                    nw, df12 = np.count_nonzero(trx.walking[f1:f2]), f2 - f1
                    stpFs.append((df12 - nw) / df12)
                print(
                    "  %s: %s"
                    % (
                        flyDesc(f),
                        (
                            "avg. %s: " % self.speedLbl
                            + "%s, stop fraction: %s"
                            % (
                                self._vals(util.join(", ", sps, p=1), f),
                                self._vals(util.join(", ", stpFs, p=2), f),
                            )
                        ),
                    )
                )
                self._append(self.speed, sps, f)
                self._append(self.stopFrac, stpFs, f)

    # rewards per minute
    def rewardsPerMinute(self):
        self.rewardsPerMin = []
        for t in self.trns:
            fi, la = self._syncBucket(t, skip=0)[0], t.stop
            rpm = (
                np.nan
                if fi is None
                else self._countOn(fi, la, calc=True, ctrl=False, f=0)
                / self._f2min(la - fi)
            )
            self._append(self.rewardsPerMin, rpm, f=0)

    # - - -

    def initLedDetector(self):
        """
        Initializes the LED detection algorithm by determining the threshold needed to distinguish between
        LED on and off states in the video frames. This method selects a specific algorithm based on the
        configuration and calculates a brightness threshold using a sample of video frames.

        Two algorithms are available for LED detection:
        - Version 1 focuses on the maximum brightness value in a specified channel.
        - Version 2 calculates the difference from the background, identifying the kth-highest value
          as the feature of interest.

        The method computes the average "LED off" brightness level and establishes a threshold above
        which the LED is considered to be on. This threshold is calculated as the mean "LED off" brightness
        plus a multiple of its standard deviation, where the multiplier is specified in the options.

        Parameters:
        - None

        This initialization process involves selecting a portion of the frame to analyze, adjusting for
        background brightness if necessary, and calculating the threshold based on the variability of
        brightness levels observed in a sample of frames. This approach ensures that subsequent analyses
        can accurately detect when the LED is activated, a critical factor in experiments where LED signals
        are used as rewards or cues.

        Note:
        - The choice of algorithm and the calculation of the threshold are designed to accommodate different
          experimental setups and lighting conditions, ensuring reliable LED detection across various scenarios.
        """
        v, ch = 2, 2  # version (1 or 2)
        assert v in (1, 2)
        (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.ef)
        if v == 1:
            k, bg = 1, 0
            print("  algorithm: max (channel: %d)" % ch)
        else:
            k = 10
            print("  algorithm: background difference, kth-highest value (k=%d)" % k)
            bg = self.background(channel=ch, indent=2)[ym:yM, xm:xM]
        k1 = (xM - xm) * (yM - ym) - k

        # closure stores, e.g., which part of frame to use
        def feature(frame):
            return np.partition(frame[ym:yM, xm:xM, ch] - bg, k1, axis=None)[k1]

        self.feature = feature
        print('  reading frames to learn "LED off"...')
        vs = [self.feature(util.readFrame(self.cap, n + 20)) for n in range(100)]
        self.ledOff = np.mean(vs)
        self.ledTh = self.ledOff + self.opts.delayCheckMult * np.std(vs)

    # returns combined image if no key given; otherwise, memorizes the given
    #  frame sequence and increments c[key]
    def _delayImg(self, i1=None, i2=None, key=None, c=None):
        if not hasattr(self, "_dImgs"):
            self._dImgs, self._dHdrs = {}, {}
            self._dNc = None if i1 is None else i2 - i1
        if key is None:
            imgs, hdrs = [], []
            for key in sorted(self._dImgs):
                imgs.extend(self._dImgs[key])
                hdrs.extend(self._dHdrs[key])
            return util.combineImgs(imgs, hdrs=hdrs, nc=self._dNc)[0] if imgs else None

        if c is not None:
            c[key] += 1

        assert i1 is not None and i2 is not None
        if key not in self._dImgs:
            self._dImgs[key], self._dHdrs[key] = [], []
        imgs, hdrs = self._dImgs[key], self._dHdrs[key]
        n = len(imgs) / (i2 - i1)
        if n > 1:
            return
        trx = self.trx[0]
        imgs.extend(trx._annImgs(i1, i2, show="td"))
        for i in range(i1, i2):
            if i == i1:
                hdr = "frame %d" % i
            else:
                hdr = key if i == i1 + 1 and n == 0 else ""
            hdrs.append(hdr)

    def _delayCheckError(self, msg, i1, i2, data, expl=""):
        self._delayImg(i1, i2, msg)
        cv2.imwrite(DELAY_IMG_FILE, self._delayImg())
        error("\n%s %s%s" % (msg, data, "\n" + expl if expl else ""))

    def delayCheck(self):
        """
        Performs a delay check for "LED on" events, analyzing the timing between a fly's entry into the reward
        circle and the LED activation to ensure proper synchronization of reward delivery.

        This method iterates through "LED on" events, reading frames around each event to detect the fly's
        entry into the reward circle and the subsequent LED activation. It checks for anomalies such as long
        delays in LED activation, missing frames, or premature LED activation. The method prints a summary
        of the findings, including the average delay, minimum and maximum delays, and instances of long
        delays. If any issues are detected, such as missing "on" signals or unexpected delays, appropriate
        error handling or logging is performed.

        Parameters:
        - None

        The method initializes LED detection parameters and proceeds to analyze each "LED on" event, taking
        into account the specific training session and the location of the reward circle. It calculates the
        delay between the fly entering the circle and the LED turning on, logging any deviations from expected
        behavior. The results include a detailed report on the average delay, the distribution of delays,
        and specific cases of long delays or other anomalies.

        Notes:
        - This analysis is crucial for ensuring the integrity of the experiment's design, particularly the
          timing of reward delivery, which may affect the flies' learning and behavior.
        - The method generates a comprehensive report on the delay check, including statistics on measured
          delays and any anomalies detected during the analysis.
        """
        print('\n"LED on" delay check')
        trx = self.trx[0]  # fly 1
        ts = trx.ts
        if ts is None:
            print("  skipped (timestamps missing)")
            return
        self.initLedDetector()

        print('  reading frames around each "LED on" event...')
        kLd, kM = "long delay", "measured"
        c, dlts, preD, ledMax, npr = collections.Counter(), [], 2, 0, 0
        ldfs = [[] for t in self.trns]
        for i, fi in enumerate(self.on):
            npr += 1  # events processed
            util.printF("\r  %d: %d" % (i, fi))
            t = Training.get(self.trns, fi)
            if not t:
                c["not training (wake-up)"] += 1
                continue
            f1, f2 = fi - preD, fi + 3
            cx, cy, r = t.circles()[False]
            isIn = [util.distance(trx.xy(j), (cx, cy)) < r for j in range(f1, f2)]
            en = np.nonzero(np.diff(np.array(isIn, np.int)) == 1)[0]
            if en.size != 1:
                self._delayImg(f1, f2, "%d enter events" % en.size, c)
                continue
            ts1, en = ts[f1:f2], en[0] + 1
            if np.any(np.diff(ts1) > 1.5 / self.fps):
                self._delayImg(f1, f2, "missing frame", c)
                continue
            vs = [self.feature(util.readFrame(self.cap, j)) for j in range(f1 + en, f2)]
            ledMax = max(ledMax, max(vs))
            isOn = [v > self.ledTh for v in vs]
            if isOn[0]:
                self._delayImg(f1, f2, "not off at enter", c)
                continue
            if np.any(trx.nan[f1 : f1 + en + 1]):
                self._delayImg(f1, f2, "fly lost", c)
                continue
            on = np.nonzero(isOn)[0]
            if not on.size:
                expl = (
                    '  "on" hard to detect for HtL corner/side chambers, '
                    + "possibly adjust --dlyCk"
                    if self.ct is CT.htl
                    else ""
                )
                self._delayCheckError('missing "on"', f1, f2, (isIn, en, isOn), expl)
            else:
                dlt = ts1[on[0] + en] - ts1[en]
                c[kM] += 1
                if dlt < 0.5 / self.fps:
                    self._delayCheckError('"on" too soon', f1, f2, (isIn, en, isOn))
                if dlt > 1.5 / self.fps:
                    self._delayImg(f1, f2, kLd, c)
                    ldfs[t.n - 1].append(fi)
                dlts.append(dlt)

        tc = sum(c[k] for k in c if k not in (kLd, kM))
        assert tc + c[kM] == npr
        print(
            '\n  skipped "LED on" events:%s'
            % (" ({:.1%})".format(tc / npr) if tc else "")
        )
        if tc:
            for k in sorted(c):
                if k != kM:
                    print(
                        "    %d (%s): %s%s"
                        % (
                            c[k],
                            "{:.1%}".format(c[k] / npr),
                            k,
                            " (not skipped)" if k == kLd else "",
                        )
                    )
        else:
            print("    none")
        print(
            "  classifier: avg. off: %.1f, threshold: %.1f, max. on: %.1f"
            % (self.ledOff, self.ledTh, ledMax)
        )
        print('  "LED on" events measured: %d' % c[kM])
        if c[kM]:
            print(
                "    delay: mean: %.3fs, min: %.3fs, max: %.3fs  (1/fps: %.3fs)"
                % (np.mean(dlts), np.amin(dlts), np.amax(dlts), 1 / self.fps)
            )
            if c[kLd]:
                print("    long delays (> 1.5/fps): {:.1%}".format(c[kLd] / c[kM]))
                for i, t in enumerate(self.trns):
                    if ldfs[i]:
                        print(
                            "      t%d: %s"
                            % (t.n, util.join(", ", ldfs[i], lim=8, end=True))
                        )
        img = self._delayImg()
        if img is not None:
            cv2.imwrite(DELAY_IMG_FILE, img)
