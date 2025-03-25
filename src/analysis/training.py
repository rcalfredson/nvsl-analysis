# standard libraries
import enum

# third-party libraries
import cv2
import numpy as np

# custom modules and constants
from src.utils.common import CT, frame2hm, pch
from src.utils.constants import LEGACY_YC_CIRCLES
import src.utils.util as util
from src.utils.util import COL_W, error

POST_TIME_MIN = False


# minimal wrapper for training
# notes:
# * data attributes (e.g., start, stop, etc.) are accessed w/out method
# * naming virtual vs. control circles, see comment at beginning of file
class Training:
    TP = enum.Enum("TrainingType", "bottom top center circle choice move")
    # circle is used for various positions in large chamber
    HAS_SYM_CTRL = {TP.bottom, TP.top}

    _exp, _expVals = None, None

    # n = 1, 2, ...
    def __init__(self, n, start, stop, va, opts, circle=None, ytb=None):
        self.n, self.start, self.stop, self.va = n, start, stop, va
        self.ct, self.xf, self.fps, self.yc = va.ct, va.xf, va.fps, not va.noyc
        (self.cx, self.cy), self.r = circle if circle else ((None, None), None)
        (self.yTop, self.yBottom) = ytb if ytb else (None, None)
        self.cs, self.v_cs = [], []  # training and control circles for each fly
        self._setCntr()
        self.sym = False
        self.opts = opts

    def _setCntr(self):
        if not hasattr(self, "cntr") and self.xf.initialized():
            self.cntr = self.xf.t2f(*self.ct.center(), f=self.va.ef)

    def isCircle(self):
        return self.cx is not None

    # length in frames
    def len(self, post=False):
        return self.postStop - self.stop if post else self.stop - self.start

    # returns whether this training has symmetrical control circle
    def hasSymCtrl(self):
        return self.tp in self.HAS_SYM_CTRL or self.sym

    # returns training and control circle(s) for the given fly
    def circles(self, f=0):
        return self.v_cs if f == 1 else self.cs

    # returns name (short version: e.g., "training 1")
    def name(self, short=True):
        if not short:
            tt, pt = (frame2hm(self.len(x), self.fps) for x in (False, True))
        return "%s %d%s" % (
            pch("session", "training"),
            self.n,
            "" if short else ": %s, %s (post: %s)" % (tt, self.tpS, pt),
        )

    # returns short name (e.g., "t1")
    def sname(self):
        return "t%d" % self.n

    # draws, e.g., circles on the given image
    # ctrl: False: exp. circle, True: control circle, None: all circles
    # returns cx, cy, and r in case of single circle
    def annotate(self, img, ctrl=False, col=COL_W, f=0):
        if self.cs:
            cs = (
                self.cs + self.v_cs
                if ctrl is None
                else self.circles(f)[ctrl : ctrl + 1]
            )
            for cx, cy, r in cs:
                cv2.circle(img, (cx, cy), r, col)
            if len(cs) == 1:
                return cs[0]
        elif self.tp is self.TP.choice:
            for y in (self.yTop, self.yBottom):
                (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.va.ef)
                bw = {CT.regular: -12, CT.htl: 15, CT.large: 35}[self.ct]
                cv2.line(img, (xm - bw, y), (xM + bw, y), col)

    # returns the training for the given frame index, None for non-training
    @staticmethod
    def get(trns, fi, includePost=False):
        for t in trns:
            if t.start <= fi < (t.postStop if includePost else t.stop):
                return t
        return None

    @staticmethod
    def _addCircle(t, cx, cy, r, xf, tmFct=None, tmX=None, isCntr=False):
        t.cs.append((cx, cy, r))
        if t.ct is CT.regular:
            assert xf is not None and tmFct is not None and tmX is not None
            ccx = 150.5 if isCntr else 192 - 22
            ccx = util.intR(ccx * tmFct + tmX) if LEGACY_YC_CIRCLES else xf.t2fX(ccx)
            t.v_cs.append((ccx, cy, r))
        elif t.yc and t.ct is CT.large:
            t.v_cs.append((cx, 2 * xf.t2fY(268) - cy, r))
        elif t.yc and t.ct is CT.large2:
            t.v_cs.append((cx, cy + xf.t2fY(-xf.y, f=t.va.nef), r))
        elif t.yc and t.ct is CT.htl:
            t.v_cs.append((cx, cy + xf.t2fY(-xf.y, f=t.va.nef), r))

    @staticmethod
    def _getCornerCoords(t, xf, fly_index):
        floor_tl, floor_br = t.ct.floor(xf, f=fly_index)
        return {
            "top-left": (floor_tl[0] + t.r, floor_tl[1] + t.r),
            "top-right": (floor_br[0] - t.r, floor_tl[1] + t.r),
            "bottom-left": (floor_tl[0] + t.r, floor_br[1] - t.r),
            "bottom-right": (floor_br[0] - t.r, floor_br[1] - t.r),
        }
    
    @staticmethod
    def _getRadiusMultCC(opts, ct):
        if opts.radiusMultCC:
            return opts.radiusMultCC
        return 2.5 if ct is CT.htl else 3

    # sets training, control, and virtual (yoked control) circles
    @staticmethod
    def _setCircles(trns, cyu, opts):
        if not any(t.isCircle() for t in trns):
            return
        calcTm, xf = len(cyu) == 3, trns[0].xf
        if calcTm and trns[0].ct is CT.regular:  # calculate template match values
            tmFct = (cyu[2] - cyu[0]) / (112.5 - 27.5)
            xm, ym = [
                min(t.cx if i else t.cy for t in trns if t.isCircle()) for i in (1, 0)
            ]
            tmX, tmY = xm - (4 + 22) * tmFct, ym - 27.5 * tmFct
            if not xf.initialized():
                xf.init(dict(fctr=tmFct, x=tmX, y=tmY))
                for t in trns:
                    t._setCntr()
            else:
                errs = abs(xf.x - tmX), abs(xf.y - tmY), abs(xf.fctr - tmFct) / tmFct
                assert all(err < 0.7 for err in errs[:2]) and errs[2] < 0.01
        else:
            tmFct, tmX = xf.fctr, xf.x
        for t in trns:
            if t.isCircle():
                isCntr = t.tp is t.TP.center

                # Add primary experimental circle
                Training._addCircle(t, t.cx, t.cy, t.r, xf, tmFct, tmX, isCntr)

                # Determine and explicitly add the control circle
                if opts.controlCircleInCorner:
                    # Get floor coordinates for experimental fly
                    corner_coords = Training._getCornerCoords(t, xf, t.va.ef)

                    # Add control circle explicitly
                    for corner, (ctrl_cx, ctrl_cy) in corner_coords.items():
                        Training._addCircle(t, ctrl_cx, ctrl_cy, t.r, xf)
                else:
                    if t.tp is t.TP.circle:
                        if t.ct in (CT.large, CT.large2):
                            if t.ct is CT.large2 and opts.rotateControlCircle:
                                Training._addCircle(
                                    t,
                                    2 * t.cntr[0] - t.cx,
                                    2 * t.cntr[1] - t.cy,
                                    t.r,
                                    xf,
                                )
                            else:
                                rm = Training._getRadiusMultCC(opts, t.ct)
                                Training._addCircle(t, t.cx, t.cy, util.intR(t.r * rm), xf)
                        elif t.ct is CT.htl:
                            Training._addCircle(t, t.cx, 2 * t.cntr[1] - t.cy, t.r, xf)
                            t.sym = True
                        else:
                            error(
                                "TrainingType circle not implemented for %s chamber"
                                % t.ct
                            )
                    elif isCntr:
                        rm = Training._getRadiusMultCC(opts, t.ct)
                        Training._addCircle(
                            t, t.cx, t.cy, util.intR(t.r * rm), xf, tmFct, tmX, isCntr
                        )
                    else:
                        if len(cyu) == 3:
                            assert t.cy == cyu[0] or t.cy == cyu[2]
                            ccy = cyu[2] if t.cy == cyu[0] else cyu[0]
                        elif t.tp in (t.TP.bottom, t.TP.top):
                            assert t.ct is CT.regular
                            ccy = xf.t2fY(112.5 if t.tp is t.TP.top else 27.5)
                        else:
                            error("not yet implemented")
                        Training._addCircle(t, t.cx, ccy, t.r, xf, tmFct, tmX, isCntr)

    @staticmethod
    def _setYTopBottom(trns):
        for t in trns:
            if t.tp is t.TP.choice and t.yTop is None:
                t.yTop = t.yBottom = t.xf.t2fY(t.ct.center()[1], f=t.va.ef)

    # to catch cases where the different videos (experiments) do not match
    # descriptor examples:
    # * bottom 1.0h, top 1.0h, center 1.0h
    # * 10 min x3
    @staticmethod
    def _setExperimentDescriptor(trns, opts):
        if trns[0].isCircle():
            exp = ", ".join("%s %s" % (t.tpS, frame2hm(t.len(), t.fps)) for t in trns)
        else:
            tms = util.repeats([frame2hm(t.len(), t.fps) for t in trns])
            exp = ", ".join("%s%s" % (t, " x%d" % r if r > 1 else "") for (t, r) in tms)
        expVals = util.concat(t.expVals for t in trns)
        if Training._exp is None:
            Training._exp, Training._expVals = exp, expVals
        else:
            em = exp == Training._exp
            evm = util.isClose(expVals, Training._expVals, atol=1)
            if (
                not (em and evm)
                and not opts.annotate
                and not opts.rdp
                and not opts.allowMismatch
            ):
                error(
                    "\nexperiment%s do not match (%s vs. %s)"
                    % (
                        ("s", '"%s"' % exp, '"%s"' % Training._exp)
                        if not em
                        else (
                            " values",
                            "[%s]" % util.join(", ", expVals, p=0),
                            "[%s]" % util.join(", ", Training._expVals, p=0),
                        )
                    )
                )

    # post stops on possible wake-up pulse
    @staticmethod
    def _setPostStop(trns, on, nf, opts):
        for i, t in enumerate(trns):
            t.postStop = trns[i + 1].start if i + 1 < len(trns) else nf
            on = on[(t.stop < on) & (on < t.postStop)]
            if len(on):
                t.postStop = on[0]
            if POST_TIME_MIN and not opts.move and t.postStop - t.stop < 10 * t.fps:
                error("less than 10s post time for %s" % t.name())

    # processes all trainings and reports trainings
    # note: call before calling instance methods
    @staticmethod
    def processReport(trns, on, nf, opts):
        assert all(t.n == i + 1 for i, t in enumerate(trns))
        Training._setPostStop(trns, on, nf, opts)
        cx_values = [t.cx for t in trns if t.cx is not None]
        cy_values = [t.cy for t in trns if t.cy is not None]

        if cx_values and cy_values:
            cxu, cyu = np.unique(cx_values), np.unique(cy_values)
        else:
            cxu, cyu = np.array([]), np.array([])

        # set training type
        for t in trns:
            if opts.move:
                t.tp = t.TP.move
            elif t.cx is None:
                t.tp = t.TP.choice
            else:
                cir = t.tp = "circle x=%d,y=%d,r=%d" % (t.cx, t.cy, t.r)
            if t.isCircle():
                if t.ct in (CT.large, CT.large2):
                    t.tp = t.TP.circle
                elif len(cyu) == 3 and len(cxu) == 2:
                    if t.cy == cyu[2]:
                        t.tp = t.TP.bottom
                    elif t.cy == cyu[0]:
                        t.tp = t.TP.top
                    else:
                        t.tp = t.TP.center
                else:

                    def equal1(tp1, tp2):  # possibly move to util.py
                        return all(abs(e1 - e2) <= 1 for e1, e2 in zip(tp1, tp2))

                    cc = (t.cx, t.cy)
                    if t.ct is CT.htl:
                        if equal1(cc, t.cntr):
                            t.tp = t.TP.center
                        else:
                            t.tp = t.TP.circle
                    else:
                        assert t.ct is CT.regular
                        if equal1(cc, t.cntr):
                            t.tp = t.TP.center
                        elif equal1(cc, t.xf.t2f(26, 112.5)):
                            t.tp = t.TP.bottom
                        elif equal1(cc, t.xf.t2f(26, 27.5)):
                            t.tp = t.TP.top
                        else:
                            error("not yet implemented")
            t.expVals = (
                t.xf.f2t(t.cx, t.cy, f=t.va.ef) + (t.r,) if t.tp is t.TP.circle else ()
            )
            t.tpS = t.tp if isinstance(t.tp, str) else t.tp.name
            print(
                "  %s%s" % (t.name(short=False), " (%s)" % cir if t.isCircle() else "")
            )
        Training._setCircles(trns, cyu, opts)
        Training._setYTopBottom(trns)
        Training._setExperimentDescriptor(trns, opts)
