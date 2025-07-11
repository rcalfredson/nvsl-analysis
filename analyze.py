# -*- coding: utf-8 -*-
#
# analyze learning experiments
#
# 18 Sep 2015 by Ulrich Stern
#
# notes:
# * naming:
#  calculated reward: entering of actual or virtual (fly 2) training circle
#  control reward: entering of control circle ("top vs. bottom")
#
# TODO
# * always for new analysis: make sure bad trajectory data skipped
#  - check this for recent additions
#  - if checkValues() is used, this is checked
# * rewrite to store data for postAnalyze() and writeStats() in dict?
# * rename reward -> response (where better)
# * write figures
# * compare tracking with Ctrax?
# * separate options for RDP and epsilon
# * fly 0, 1, and 2 used in comments
# * move CT to common.py?
#

# standard libraries
import argparse
import collections
import csv
import itertools
import logging
import os
import random
import re
import sys
import timeit
import warnings

# third-party libraries
import cv2
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# custom modules and functions
from src.utils.common import (
    CT,
    cVsA_l,
    adjustLegend,
    flyDesc,
    is_nan,
    pch,
    propagate_nans,
    skipMsg,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    writeImage,
)
from src.utils.constants import (
    CONTACT_BUFFER_OFFSETS,
    MIDLINE_BOUNDARY_DIST,
    HEATMAP_DIV,
    POST_SYNC,
    RDP_MIN_TURNS,
    SPEED_ON_BOTTOM,
    ST,
)
from src.analysis.motion import CircularMotionDetector
from src.plotting.outside_circle_duration_plotter import OutsideCircleDurationPlotter
from src.utils.parsers import parse_distances
from src.plotting.plot import plotAngularVelocity, plotTurnRadiusHist
from src.plotting.plot_customizer import PlotCustomizer
from src.analysis.sli_tools import (
    compute_sli_per_fly,
    plot_sli_extremes,
    select_extremes,
)
from src.analysis.training import Training
from src.analysis.trajectory import Trajectory
from src.plotting.turn_directionality_plotter import TurnDirectionalityPlotter
from src.plotting.turn_prob_dist_plotter import TurnProbabilityByDistancePlotter
import src.utils.util as util
from src.utils.util import (
    ArgumentError,
    Tee,
    VideoError,
    AVI_X,
    COL_B,
    COL_BK,
    COL_W,
    WINDOWS,
)
from src.utils.util import error, warn
from src.analysis.video_analysis import (
    VideoAnalysis,
    per_va_processing_times,
    per_bnd_processing_times,
    per_lgt_processing_times,
)

CAM_NUM = re.compile(r"^c(\d+)__")
LOG_FILE = "__analyze.log"
STATS_FILE, VIDEO_COL = "learning_stats.csv", True
ANALYSIS_IMG_FILE = "imgs/analysis.png"
CIRCULAR_MOTION_IMG_FILE = "imgs/circular_motion__%s_min_buckets.png"
PIVOT_IMG_FILE = "imgs/pivots__%s_min_buckets.png"
CONTACT_EVENT_IMG_FILE = "imgs/%s_%s_contact__%%s_min_buckets.png"
DIST_BTWN_REWARDS_FILE = "imgs/dist_btwn_%srewards__%%s_min_buckets.png"
DIST_BTWN_REWARDS_HIST_FILE = "imgs/trajectory_len_dist%s.png"
MAX_DIST_REWARDS_FILE = "imgs/max_dist_from_center__%s_min_buckets.png"
CONTACTLESS_RWDS_IMG_FILE = "imgs/contactless_rewards_%s__%%s_min_buckets.png"
TURN_RADIUS_IMG_FILE = "imgs/turn_radius__%s_min_buckets.png"
REWARD_PI_IMG_FILE = "imgs/reward_pi__%s_min_buckets.png"
REWARD_PI_DIFF_IMG_FILE = "imgs/reward_pi_diff__%s_min_buckets.png"
REWARD_PI_POST_IMG_FILE = "imgs/reward_pi_post__%s_min_buckets.png"
REWARD_PI_POST_DIFF_IMG_FILE = "imgs/reward_pi_post_diff__%s_min_buckets.png"
DIST_BTWN_REWARDS_LABEL = "mean dist. between calc. rewards%s"
TURN_IMG_FILE = "imgs/%s%s_turn%s%s__%%s_min_buckets.png"
TURN_LABEL = "turns per %s-contact event%s"
TURN_DURATION_LABEL = "mean turn duration (s)%s"
CONTACTLESS_RWDS_LABEL = "proportion contactless rewards, %s walls"
REWARDS_IMG_FILE = "imgs/rewards__%s_min_buckets.png"
RUN_LENGTHS_IMG_FILE = "imgs/run_lengths.png"
TURN_ANGLES_IMG_FILE = "imgs/turn_angles.png"
HEATMAPS_IMG_FILE = "imgs/heatmaps%s.png"
OPEN_LOOP_IMG_FILE = "imgs/open_loop.png"

P = False  # whether to use paper style for plots
F2T = True  # whether to show only first two trainings for paper
LEG = False  # whether to show legend for paper

BORDER_WIDTH = 1


OP_LIN, OP_LOG = "I", "O"
OPTS_HM = (OP_LIN, OP_LOG)

customizer = PlotCustomizer(pch(11, "x-small"))

# - - -


class FlyDetector:
    pass  # required for unpickle()


# - - -

p = argparse.ArgumentParser(description="Analyze learning experiments.")

p.add_argument(
    "-v",
    dest="video",
    default=None,
    metavar="N",
    help="video filename, directory name, or comma-separated list of names "
    + "(names can contain wildcards *, ?, and []); use | to separate "
    + "video groups (for rewards plots, etc.); use : to give fly number "
    + "range (overriding -f for the video)",
)
p.add_argument(
    "-f",
    dest="fly",
    default=None,
    metavar="N",
    help='fly numbers in case of HtL or large chamber (e.g., "0-19" or '
    + '"6-8,11-13"); use | for by-group fly numbers',
)
p.add_argument(
    "--gl",
    dest="groupLabels",
    default=None,
    metavar="N",
    help="labels for video groups (bar-separated)",
)
p.add_argument(
    "--aem",
    dest="allowMismatch",
    action="store_true",
    help="allow experiment descriptor mismatch, which leads to error otherwise",
)
p.add_argument(
    "--ayc",
    dest="allowYC",
    action="store_true",
    help="allow yoked controls among fly numbers, which leads to error otherwise",
)

g = p.add_argument_group("specialized analysis")
g.add_argument(
    "--move",
    dest="move",
    action="store_true",
    help='analyze "move" experiments (not auto-recognized)',
)
g.add_argument(
    "--ol",
    dest="ol",
    action="store_true",
    help='analyze "open loop" experiments; not needed for on-off and '
    + "alternating side protocols",
)
g.add_argument(
    "--best-worst-sli",
    action="store_true",
    help=(
        "Plot the SLI (Spatial Learning Index) across the full experiment "
        "for the top and bottom 10%% of learners, selected based on their SLI "
        "score in the final sync bucket of training session 2, or the session "
        "specified by --best-worst-trn."
    ),
)

g.add_argument(
    "--best-worst-trn",
    type=int,
    default=2,
    help=(
        "Specify which training session index to use when determining the top "
        "and bottom 10%% of learners for --best-worst-sli. The SLI is taken "
        "from the final sync bucket of the specified training session. Default is 2."
    ),
)
g.add_argument(
    "--best-worst-fraction",
    type=float,
    default=0.1,
    help="Fraction of flies to use for SLI cutoff when plotting best/worst learners."
    " Default: %(default)0.1f.",
)
g.add_argument(
    "--rdp",
    dest="rdp",
    type=float,
    metavar="F",
    nargs="?",
    const=3.0,
    default=0,
    help="analyze trajectories simplified using RDP with the given epsilon "
    + "(default: %(const)s)",
)
g.add_argument(
    "--pct-time-circle-rad",
    dest="pctTimeCircleRad",
    type=float,
    help="measure the percentage of time the fly spends within a circle"
    " of the specified radius (in mm), concentric with the reward circle.",
)
g.add_argument(
    "--wall",
    nargs="?",
    help="detect when the fly makes contact with the wall. An optional value"
    " of two numbers enclosed in quotes and separated by a vertical divider,"
    " e.g., 'x1|x2', is used to set the distance from the wall at which"
    " contact events begin and end (defaults:"
    f" {CONTACT_BUFFER_OFFSETS['wall']['min']}mm and"
    f" {CONTACT_BUFFER_OFFSETS['wall']['max']}mm).",
    default=None,
    const=f"{CONTACT_BUFFER_OFFSETS['wall']['min']}|"
    f"{CONTACT_BUFFER_OFFSETS['wall']['max']}",
)
g.add_argument(
    "--wall_orientation",
    choices=("lr", "tb", "all", "agarose_adj"),
    default="all",
    help="choose a specific set of walls from which to calculate wall-"
    + "contact events - 'lr' for left-right, 'tb' for top-bottom, "
    + "'all' for all walls, and 'agarose_adj' for all walls with "
    " top and bottom shifted to align with the agarose border.",
)
g.add_argument(
    "--wall_eg", action="store_true", help="save example images of wall-contact events"
)
g.add_argument(
    "--excl-wall-for-spd",
    action="store_true",
    help="Exclude frames where wall walking is detected when calculating speed statistics.",
)
g.add_argument(
    "--agarose",
    nargs="?",
    help="[HTL chamber only] detect when the fly makes contact with agarose."
    " An optional value of two numbers enclosed in quotes and separated by a"
    " vertical divider, e.g., 'x1|x2', is used to set the distance from the"
    " agarose boundary at which contact events begin and end (defaults:"
    f" {CONTACT_BUFFER_OFFSETS['agarose']['min']}mm and {CONTACT_BUFFER_OFFSETS['agarose']['max']}mm).",
    default=None,
    const=f"{CONTACT_BUFFER_OFFSETS['agarose']['min']}|{CONTACT_BUFFER_OFFSETS['agarose']['max']}",
)
g.add_argument(
    "--boundary-contact",
    dest="boundary",
    nargs="?",
    help="[HTL chamber only] detect when the fly makes contact with a boundary"
    " at a specified distance from the top/bottom walls. Defaults to using the"
    " two midlines (4mm from the center in either direction). Optional argument"
    " can consist either of just the distance, or, additionally, the distances"
    " from the boundary at which contact events begin and end (defaults: "
    f"{CONTACT_BUFFER_OFFSETS['boundary']['min']}mm and {CONTACT_BUFFER_OFFSETS['boundary']['max']}mm",
    default=None,
    const=f"{MIDLINE_BOUNDARY_DIST}|"
    f"{CONTACT_BUFFER_OFFSETS['boundary']['min']}|{CONTACT_BUFFER_OFFSETS['boundary']['max']}",
)
g.add_argument(
    "--wall-debug",
    action="store_true",
    help="[HTL chamber only] run assertions to confirm that fly is completely within floor bounds at"
    " the end of every wall-contact event.",
)
g.add_argument(
    "--lg-turn-plots",
    choices=("all_types", "turn_plus_1"),
    help="Generate large turn trajectory plot. Two modes are supported: 'all_types' includes all"
    " frames between two large turns, whereas 'turn_plus_1' shows a single sharp turn and a"
    " subsequent non-turn event.",
)
g.add_argument(
    "--bnd-ct-plots",
    type=str,
    choices=["troubleshooting", "display"],
    nargs="?",
    const="troubleshooting",
    default=None,
    help="Generate boundary contact/sharp turn trajectory plots using one of two modes:"
    " 'troubleshooting' for individual event plots, or 'display' for chaining events together.",
)
g.add_argument(
    "--bnd-ct-plot-mode",
    choices=["all_types", "turn_plus_1"],
    default="all_types",
    help="Mode for boundary contact/sharp turn trajectory plot. 'all_types' includes all frames"
    " between two sharp turns. 'turn_plus_1' shows a single sharp turn and the subsequent non-turn event.",
)
g.add_argument(
    "--bnd-ct-plot-start-fm",
    type=int,
    required=False,
    help="Specify the starting frame for the boundary contact trajectory plot "
    "when using 'display' mode. If not provided, the event chain will be "
    "selected randomly.",
)
g.add_argument(
    "--turn",
    help="detect turn events as a subset of boundary-contact events with respect either"
    " to the wall, to the plastic/agarose boundary, or the reward circle",
    choices=("agarose", "boundary", "circle"),
    nargs="+",
)
g.add_argument(
    "--turn-dir",
    action="store_true",
    help="analyze directionality of agarose- and/or boundary-anchored turns",
)
g.add_argument(
    "--contact-geometry",
    choices=("horizontal", "circular"),
    default="horizontal",
    help="[turn-probability] use horizontal centre-line analysis "
    "or concentric-circle analysis (default: horizontal)",
)
g.add_argument(
    "--turn-prob-by-dist",
    type=str,
    help="Generate a plot of turn probability as a function of distance from the"
    " horizontal center line of the chamber. Provide a comma-separated list of"
    " monotonically increasing values.",
)
g.add_argument(
    "--use-union-filter",
    dest="use_union_filter",
    action="store_true",
    default=False,
    help="Apply the union-based exclusion across all ticks (default: per-tick exclusion only).",
)
g.add_argument(
    "--outside-circle-radii",
    type=str,
    help="Comma-separated list of radii (mm) concentric with the reward circle. "
    "Used for (i) histograms of exit-to-entry durations AND "
    "(ii) turn-probability-by-distance when --contact-geometry circular.",
)
g.add_argument(
    "--outside-circle-norm",
    action="store_true",
    help="whether to normalize the histograms"
    " plotted for the outside-event event analysis (see --outside-circle-radii)",
)
g.add_argument(
    "--rc_turn_tests",
    help="which test(s) to apply for group comparisons of distance from reward circle"
    " at the start/end of turns. Choices: t-test ('t') and/or Mann-Whitney U test"
    " ('mwu'). For the t-test and Mann-Whitney U test, turns are averaged (one mean"
    " observation per fly per comparison).",
    default=("t"),
    choices=("t", "mwu"),
    nargs="+",
)
g.add_argument(
    "--turn_duration_thresh",
    help="maximum event duration of turn events, measured in seconds",
    type=float,
    default=0.75,
)
g.add_argument(
    "--end_turn_before_recontact",
    action="store_true",
    help="for turns anchored by the reward circle (--turn circle), whether to end"
    " the turns before re-entering the reward circle or re-contacting the wall"
    " (by default, turns end after those events)",
)
g.add_argument(
    "--minNumLT",
    type=int,
    help="minimum number of large turns per training for a fly to be included in plots",
    default=20,
)
g.add_argument("--turn_eg", action="store_true", help="save example images of turns")
g.add_argument(
    "--turn_contact_thresh",
    help="threshold of contact events needed for turn stats to be reported over a"
    " given period (i.e., a sync bucket). Default: 10",
    type=int,
    default=10,
)
g.add_argument(
    "--min_vel_angle_delta",
    help="minimum velocity angle delta (degrees) required to label a turn",
    type=float,
    default=90,
)
g.add_argument(
    "--min_turn_speed",
    help="minimum speed (units: mm/s) at which to consider a frame's data when"
    " labeling turning events.",
    default=2,
    type=float,
)
g.add_argument(
    "--agarose_eg",
    action="store_true",
    help="save example images of agarose-contact events",
)
g.add_argument(
    "--n_rewards_start_end_avg",
    type=int,
    default=50,
    help="number of rewards over which to average boundary-contact events at"
    + " the beginning and end of the experiment.",
)

g = p.add_argument_group("tweaking analysis")
g.add_argument(
    "--shBB",
    dest="showByBucket",
    action="store_true",
    help='show rewards by "bucket" (--nb per training)',
)
g.add_argument(
    "--nb",
    dest="numBuckets",
    type=int,
    default=None,
    metavar="N",
    help="number of buckets per training (default: 1 if choice else 12)",
)
g.add_argument(
    "--nrc",
    dest="numRewardsCompare",
    type=int,
    default=100,
    metavar="N",
    help="number of rewards to compare (default: %(default)s)",
)
g.add_argument(
    "--sb",
    dest="syncBucketLenMin",
    type=float,
    default=10,
    metavar="F",
    help="length of sync buckets (in minutes, default: %(default)s); "
    + "synchronized with first reward",
)
g.add_argument(
    "--piTh",
    dest="piTh",
    type=int,
    default=10,
    metavar="N",
    help="calculate reward PI only if sum is at least this number "
    + "(default: %(default)s)",
)
g.add_argument(
    "--independent-exclusion",
    action="store_true",
    help=(
        "Unlink the exclusion conditions between experimental and yoked control flies in the learning stats file. "
        "By default, if a sync bucket value is excluded for one fly (such as due to piTh), then it is also excluded "
        "for the other. Using this flag, each fly is included or excluded independently of its pair."
    ),
)
g.add_argument(
    "--adbTh",
    dest="adbTh",
    type=int,
    default=5,
    metavar="N",
    help="calculate average distance traveled (or maximum distance reached) "
    + "between rewards for sync buckets only "
    + "if number of rewards is at least this number (default: %(default)s)",
)
g.add_argument(
    "--pib",
    dest="piBucketLenMin",
    type=float,
    default=None,
    metavar="F",
    help="length of post training buckets for positional PI (in minutes, "
    + "default: 10 if choice else 2)",
)
g.add_argument(
    "--rm",
    dest="radiusMult",
    type=float,
    default=1.3,
    metavar="F",
    help="multiplier for radius for positional PI (default: %(default)s)",
)
g.add_argument(
    "--controlCircleInCorner",
    action="store_true",
    help="override control circle placement from saved training protocol,"
    " placing it in the corner of the chamber instead.",
)
g.add_argument(
    "--disableCornerCircleScaling",
    action="store_true",
    help="(TEMPORARY) Disable the scaling by (1/4) when using four control"
    " circles placed at the corners of the chamber.",
)
g1 = g.add_mutually_exclusive_group()
g1.add_argument(
    "--rmCC",
    dest="radiusMultCC",
    type=float,
    default=None,
    metavar="F",
    help="multiplier for radius for control circle (default: 2.5 if HtL else 3)",
)
g1.add_argument(
    "--rCC",
    dest="radiusCC",
    type=float,
    default=None,
    metavar="F",
    help="radius of control circle (in mm)",
)
g.add_argument(
    "--pb",
    dest="postBucketLenMin",
    type=float,
    default=3,
    metavar="F",
    help="length of post training buckets for number rewards (in minutes, "
    + "default: %(default)s)",
)
g.add_argument(
    "--rpib",
    dest="rpiPostBucketLenMin",
    type=float,
    default=3,
    metavar="F",
    help="length of post training buckets for reward PI (in minutes, "
    + "default: %(default)s)",
)
g.add_argument(
    "--skp",
    dest="skip",
    type=float,
    default=0,
    metavar="F",
    help="skip the given number of minutes from beginning of buckets "
    + "(default: %(default)s)",
)
g.add_argument(
    "--skpPi",
    dest="skipPI",
    action="store_true",
    help="if fly did not visit both top and bottom during bucket's "
    + "--skp period, skip bucket's PI in %s" % STATS_FILE,
)
g.add_argument(
    "--minVis",
    dest="minVis",
    type=int,
    default=0,
    metavar="N",
    help="skip bucket's PI in %s unless each top and bottom " % STATS_FILE
    + "were visited at least this many times (default: %(default)s)",
)
g.add_argument(
    "--yoked",
    dest="yoked",
    action="store_true",
    help="when running multi-group analysis, use data from yoked flies "
    + "instead of experimental.",
)

g = p.add_argument_group("plotting")
g.add_argument("--shPlt", dest="showPlots", action="store_true", help="show plots")
g.add_argument(
    "--fs",
    dest="fontSize",
    type=float,
    default=mpl.rcParams["font.size"],
    metavar="F",
    help="font size for plots (default: %(default)s)",
)
g.add_argument(
    "--fontFamily",
    dest="fontFamily",
    type=str,
    default=None,
    help="Override the default font family for plots (e.g., 'Arial').",
)
g.add_argument(
    "--imgFormat",
    dest="imageFormat",
    type=str,
    default="png",
    help="Desired format for plots saved by plotRewards (default: %(default)s)."
    " Other common options include 'svg' and 'pdf'.",
)
g.add_argument(
    "--ws",
    dest="wspace",
    type=float,
    default=mpl.rcParams["figure.subplot.wspace"],
    metavar="F",
    help="width of space between subplots (default: %(default)s)",
)
g.add_argument(
    "--hidePltTests",
    action="store_true",
    help="hide statistical tests displayed in plots saved by plotRewards"
    " (first-to-last bucket significance test and inline text with ABC/AUC test)",
)
g.add_argument("--pltAll", dest="plotAll", action="store_true", help="plot all rewards")
g.add_argument(
    "--pltTrx",
    dest="plotTrx",
    action="store_true",
    help="plot trajectories (plot depends on protocol)",
)
g.add_argument(
    "--cirTrx",
    dest="circleTrx",
    action="store_true",
    help="plot trajectories of randomly-sampled bouts of "
    + "circular motion (one file per video analysis)",
)
g.add_argument(
    "--av",
    dest="angVelOverTime",
    action="store_true",
    help="(requires --circle option) plot angular velocity over time, "
    + "in addition to default histogram plotting",
)
g.add_argument(
    "--avR",
    dest="angVelRewards",
    action="store_true",
    help="show rewards on the plot of angular velocity",
)
g.add_argument(
    "--pltThm", dest="plotThm", action="store_true", help="plot trajectory heatmaps"
)
g.add_argument(
    "--pltThmN",
    dest="plotThmNorm",
    action="store_true",
    help="plot normalized trajectory heatmaps",
)
g.add_argument(
    "--pltHm",
    dest="hm",
    choices=OPTS_HM,
    nargs="?",
    const=OP_LOG,
    default=None,
    help="plot heatmaps with linear (%s) or logarithmic (%s, default) colorbar"
    % OPTS_HM,
)
g.add_argument(
    "--bg",
    dest="bg",
    type=float,
    nargs="?",
    const=0.6,
    default=None,
    metavar="F",
    help="plot heatmaps on chamber background with the given alpha "
    + "(default: %(const)s); use 0 to show chamber background",
)
g.add_argument(
    "--grn", dest="green", action="store_true", help="use green for LED color"
)
g.add_argument(
    "--fix",
    dest="fixSeed",
    action="store_true",
    help="fix random seed for rewards images",
)
g.add_argument(
    "--tlen_nbins",
    type=int,
    default=40,
    help="number of bins to plot in histograms of between-"
    "reward trajectory lengths (default: %(default)s)",
)
g.add_argument(
    "--tlen_mincts",
    type=int,
    default=50,
    help="minimum number of between-reward trajectories per training for a fly to"
    " be included in plots (default: %(default)s)",
)
g.add_argument(
    "--tlen_hist_ubound",
    help="upper bounds to use for the between-reward trajectory length"
    " histograms. Must consist of three comma-separated values for the"
    " three subsets of plotted data: 1) all, 2) contactless only, and"
    " 3) with contact only. Default: %(default)s",
    default="1000,300,1000",
)
g.add_argument(
    "--lg_turn_nbins",
    type=int,
    default=15,
    help="number of bins to plot in histograms of distance from reward"
    " circle of large turns. Default: %(default)s",
)

g = p.add_argument_group('rt-trx "debug"')
g.add_argument(
    "--shTI", dest="showTrackIssues", action="store_true", help="show tracking issues"
)
g.add_argument(
    "--shRM",
    dest="showRewardMismatch",
    action="store_true",
    help="show mismatch between calculated and actual rewards "
    + "(typically due to dropped frames in rt-trx.py)",
)
g.add_argument(
    "--dlyCk",
    dest="delayCheckMult",
    type=float,
    metavar="F",
    nargs="?",
    const=3,
    default=None,
    help='check delay between response and "LED on," using the given '
    + 'standard deviation multiplier to set the "LED on" threshold '
    + "(default: %(const)s)",
)
g.add_argument("--timeit", action="store_true", help="log stats of processing times")

g = p.add_argument_group("specialized files and player")
g.add_argument(
    "--ann", dest="annotate", action="store_true", help="write annotated video"
)
g.add_argument(
    "--jaabaOut",
    dest="jaabaOut",
    action="store_true",
    help="write JAABA-compatible trajectory file",
)
g.add_argument(
    "--orient",
    dest="chooseOrientations",
    action="store_true",
    help="resolve head-tail orientations",
)
g.add_argument(
    "--circle",
    dest="circle",
    action="store_true",
    help="analyze trajectories for circular, pivoting, and turning motion "
    + "(defaults to heuristic algorithm), including turn direction and radius",
)
g.add_argument(
    "--jab",
    dest="useJAABA",
    action="store_true",
    help="use JAABA algorithm for circular motion analysis",
)
g.add_argument(
    "--circlePredictFile",
    dest="circlePredictFile",
    default=CircularMotionDetector.CIRCLE_PREDICT_FILE,
    metavar="file.jab",
    help=".jab file inside JAABA/data directory to use for circular analysis",
)
g.add_argument(
    "--mat",
    dest="matFile",
    action="store_true",
    help="write MATLAB file (see yanglab Wiki for fields)",
)
g.add_argument("--play", dest="play", action="store_true", help="play annotated video")
p.parse_args()

opts = p.parse_args()
if opts.timeit:
    start_t = timeit.default_timer()

# - - -


def pcap(s):
    """
    Capitalizes the first character of the given string if a certain condition is met.

    This function checks a global condition `P` and, if true, capitalizes the first character
    of the input string `s`. Otherwise, it returns the string unchanged. This is typically
    used for formatting output strings in a context where capitalization is conditional.

    Parameters:
    - s : str
        The string to potentially capitalize.

    Returns:
    - str
        The input string with the first character capitalized if the condition `P` is true;
        otherwise, the original string.

    Note:
    - The global variable `P` determines whether the capitalization occurs. This function
      assumes that `P` is defined elsewhere in the global scope.
    """
    return s[:1].upper() + s[1:] if P else s


def areaUnderCurve(a):
    """
    Calculates the area under the curve (AUC) for each row in the input array, handling missing
    values by returning NaN for rows entirely composed of NaNs.

    This function computes the AUC using the trapezoidal rule across all columns for each row
    in the input 2D array `a`. If the last column of a row consists entirely of NaN values,
    these are ignored in the computation. This is particularly useful for processing data where
    some rows may have missing or incomplete measurements.

    Parameters:
    - a : numpy.ndarray
        A 2D numpy array where each row represents a sequence of values for which the AUC is
        to be calculated.

    Returns:
    - numpy.ndarray
        A 1D numpy array of the same length as the number of rows in `a`, containing the AUC
        for each row. Rows entirely composed of NaNs are assigned NaN in the result.

    Raises:
    - AssertionError
        If the numpy operation to check for NaN values in the trapezoidal calculation fails.
        This serves as a sanity check for the input data's compatibility with the calculation
        method.

    Note:
    - This function is designed to work with numerical data where measurements might be missing
      or incomplete. It utilizes `numpy.trapz` for the AUC calculation, assuming that the input
      array `a` is properly formatted as a 2D numpy array.
    """
    if np.all(np.isnan(a[:, -1])):
        a = a[:, :-1]
    assert np.isnan(np.trapz([1, np.nan]))
    return np.trapz(a, axis=1)


def headerForType(va, tp, calc):
    """
    Generate a header string for a specified report type.

    This function dynamically constructs a header string based on the report type (`tp`), a
    flag indicating the nature of the reward (`calc`), and the characteristics of the `va`
    object. It supports a diverse range of report types, including averages, counts,
    event-specific metrics, and more, tailored to the analysis of experimental data,
    particularly in the context of Drosophila behavior studies.

    Parameters:
    - va : VideoAnalysis
        A VideoAnalysis instance encapsulating various attributes that may affect the header's
        format. Its structure and properties are dependent on the broader experimental and
        analytical context, such as the specific behavior being tracked or the experimental
        setup (e.g., open-loop or choice tasks).
    - tp : str
        A string identifier for the report type. This could represent a wide array of reports,
        such as 'atb' (average time between rewards), 'adb' (average distance between
        rewards), CSV reports for distinct events (e.g., turns, wall contacts), among others.
        The function adjusts the header output based on this identifier.
    - calc : bool or specific flag
        Indicates whether the report pertains to a "calculated" reward or an "actual" reward.
        A "calculated" reward refers to an event determined by the trajectory of the subject
        (e.g., Drosophila) entering a predefined training circle, as defined by a computerized
        tracking system. An "actual" reward refers to the provision of a tangible stimulus to
        the subject.

    Returns:
    - str
        A header string specifically formatted according to the report type, the nature of the
        reward (calculated vs. actual), and the attributes of the `va` object. If the report
        type is unrecognized, an ArgumentError is raised.

    Raises:
    - ArgumentError
        Triggered if the `tp` (type) parameter does not match any known report type
        configurations within the system.

    Examples:
    - Generating a header for a report on average time between calculated rewards:
        headerForType(va_object, 'atb', True) ->
        "\naverage time between __calculated__ rewards: ..."
    - For a CSV report detailing distances between actual rewards without preceding wall
        contact:
        headerForType(va_object, 'dbr_no_contact_csv', False) ->
        "\ndistance between actual rewards with no preceding wall contact"

    """
    if tp == "atb" or "adb" in tp:
        return "\naverage %s between %s rewards:" % (
            "time" if tp == "atb" else "distance traveled",
            cVsA_l(calc),
        )
    elif tp in ("nr", "nrc"):
        return "\nnumber %s rewards by sync bucket:" % cVsA_l(calc, tp == "nrc")
    elif (
        tp
        in (
            "%s_csv" % evt
            for evt in (
                "agarose",
                "boundary",
                "wall",
            )
        )
        or "agarose_turn_csv" in tp
        or "agarose_turn_dir_csv" in tp
        or "boundary_turn_csv" in tp
        or "boundary_turn_dir_csv" in tp
        or "wall_turn_csv" in tp
        or "wall_turn_dir_csv" in tp
    ):
        if "turn" in tp:
            ellipse_ref_pt, boundary_tp = tp.split("_")[:2]
            if ellipse_ref_pt == "ctr":
                ellipse_ref_pt = "center"
            metric = "directionality" if "turn_dir" in tp else "events"
            return "\n%s turn %s by training (border crossings based on ellipse %s)" % (
                boundary_tp,
                metric,
                ellipse_ref_pt,
            )
        return "\n%s-contact events by training:" % (tp.split("_")[0],)
    elif tp == "bot_top_cross":
        return "\n# bottom-to-top crossings:"
    elif tp == "lg_turn_rwd_csv":
        return "stats for turns >= 90 deg on exit from reward circle"
    if "max_ctr_d_no_contact" in tp:
        if "csv" not in tp:
            return ""
        return (
            "\nmean of max. dist. from center for each trajectory with no"
            " preceding wall contact"
        )
    if tp == "btwn_rwd_dist_from_ctr":
        return "\nmean of dist. from center between rewards"
    if "dbr" in tp:
        if "csv" not in tp:
            return ""
        return "\ndistance between calculated rewards" + (
            " with no preceding wall contact" if "no_contact" in tp else ""
        )
    elif "r_no_contact" in tp:
        if "csv" not in tp:
            return ""

        return (
            "\n%s by training"
            % CONTACTLESS_RWDS_LABEL
            % ("agarose-adjacent" if "agarose" in tp else "outer")
        )
    elif (
        tp in ("pcm", "turn", "pivot", "wall", "agarose", "r_no_contact", "boundary")
        or "_turn" in tp
    ):
        return ""
    elif tp == "ppi":
        return "\npositional PI (r*%s) by post bucket:" % util.formatFloat(
            opts.radiusMult, 2
        )
    elif tp in ("rpi", "rpi_combined"):
        return "\n%s reward PI by%s bucket:" % (
            cVsA_l(True),
            " sync" if tp == "rpi" else "",
        )
    elif tp == "rpip":
        return ""
    elif tp in ("rpid", "rpipd"):
        return ""
    elif tp == "nrp":
        return "\nnumber %s rewards by post bucket:" % cVsA_l(True)
    elif tp == "nrpp":
        return ""
    elif tp == "c_pi":
        if va.openLoop:
            return "\npositional preference for LED side:"
        else:
            h = "positional preference (for top)"
            h1 = "\n" + skipMsg(opts.skip) if opts.skip else ""
            return (
                "\n"
                + (
                    "%s by bucket:" % h
                    if opts.numBuckets > 1
                    else '"%s, including %s-min post bucket:"'
                    % (h, bucketLenForType(tp)[1])
                )
                + h1
            )
    elif tp == "rdp":
        return "\naverage RDP line length (epsilon %.1f)" % opts.rdp
    elif tp == "bysb2":
        return va.bySB2Header if hasattr(va, "bySB2Header") else None
    elif tp == "frc":
        return "\nfirst reward in first sync bucket is control:"
    elif tp == "xmb":
        return "\ncrossed midline before first reward in first sync bucket:"
    elif tp in ("spd", "spd_sb"):
        return "\naverage %s%s:" % (va.speedLbl, " (by SB)" if "sb" in tp else "")
    elif tp == "stp":
        return "\naverage stop fraction:"
    elif tp == "rpm":
        return "\nrewards per minute:"
    elif tp.startswith("agarose_pct_") or tp.startswith("boundary_pct_"):
        calc_method = {"ctr": "body center", "edge": "edge of fitted ellipse"}[
            tp.split("_")[-1]
        ]
        is_agarose = tp.startswith("agarose")

        return (
            f"\n% time {'over agarose' if is_agarose else 'past border'} (contact events begin when"
            + f" %s crosses {'agarose ' if is_agarose else ''}border)" % calc_method
        )
    elif "pic" in tp:
        modifier = "reward" if tp == "pic" else f"{opts.pctTimeCircleRad}mm"
        return f"\n% time in {modifier} circle:"
    else:
        raise ArgumentError(tp)


def fliesForType(va, tp, calc=None):
    """
    Determines the set of flies to be analyzed based on the report type and calculation flag.

    This function selects flies for analysis based on the specified report type (`tp`) and an
    optional calculation flag (`calc`). The selection criteria vary, emphasizing either all
    flies or a specific subset, depending on the analysis focus (e.g., calculated vs. actual
    rewards, specific events, or conditions).

    Parameters:
    - va : VideoAnalysis
        An instance of VideoAnalysis containing attributes and data related to the video
        analysis of Drosophila behavior.
    - tp : str
        A string identifier for the type of report or analysis being conducted. This parameter
        determines which flies are selected for analysis.
    - calc : bool, optional
        An optional flag indicating whether the analysis pertains to a calculated condition
        or measurement. The default is None, which applies specific default logic based on
        the report type.

    Returns:
    - tuple
        A tuple containing the indices of flies selected for the analysis. The content and
        length of the tuple vary based on the report type and the calculation flag.

    Raises:
    - ArgumentError
        If the `tp` (type) parameter does not match any recognized report type configurations
        within the system.

    Note:
    The function is designed to support a wide range of report types and scenarios in the
    behavioral analysis of Drosophila, factoring in both predefined conditions and the nature
    of rewards (calculated vs. actual).
    """
    if tp in ("atb", "nr", "nrc"):
        return va.flies if calc else (0,)
    elif (
        tp in ("ppi", "frc", "xmb", "rpm", "turn", "pivot", "pcm", "rpid", "rpipd")
        or "_exp_min_yok" in tp
    ):
        return (0,)
    elif (
        tp
        in (
            "adb",
            "nrp",
            "nrpp",
            "rpi",
            "rpip",
            "c_pi",
            "rdp",
            "bysb2",
            "spd",
            "stp",
            "agarose",
            "boundary",
            "wall",
        )
        or "r_no_contact" in tp
        or "dbr" in tp
        or "_turn" in tp
        or "max_ctr_d" in tp
    ):
        return va.flies
    else:
        raise ArgumentError(tp)


def bucketLenForType(tp):
    """
    Determines the bucket length for time-based analyses based on the report type.

    This function calculates the length of time buckets (intervals) for various types of
    analyses. The bucket length varies depending on the specific type of report or analysis
    being conducted, reflecting different temporal granularities required for accurate
    measurement and interpretation.

    Parameters:
    - tp : str
        A string identifier for the type of report or analysis, which dictates the bucket
        length used in the analysis.

    Returns:
    - tuple
        A tuple where the first element is the bucket length as a float and the second element
        is a formatted string representation of the bucket length. If the report type does not
        specify a bucket length, None is returned.

    Note:
    This function is crucial for analyses that segment data into temporal intervals, such as
    post-training rewards analysis, contact or turn event analysis, and positional preference
    indices. The bucket length determines the temporal resolution of the analysis, impacting
    the interpretation of Drosophila behavior over time.
    """
    bl = None
    if (
        tp
        in (
            "nr",
            "nrc",
            "rpi",
            "bysb2",
            "pcm",
            "turn",
            "pivot",
            "agarose",
            "boundary",
            "wall",
            "dbr",
            "rpid",
            "max_ctr_d_no_contact",
        )
        or "_turn" in tp
        or "r_no_contact" in tp
    ):
        bl = opts.syncBucketLenMin
    elif tp in ("ppi", "c_pi"):
        bl = opts.piBucketLenMin
    elif tp in ("nrp", "nrpp", "pic", "pic_custom"):
        bl = opts.postBucketLenMin
    elif tp in ("rpip", "rpi_combined", "rpipd", "spd_sb"):
        bl = opts.rpiPostBucketLenMin
    return bl, bl if bl is None else util.formatFloat(bl, 1)


def columnNamesForLargeTurnAnalysis(va):
    """
    Generates a list of column names for analyzing large turn events in relation to reward
    circle exits.

    This function constructs column names that combine metrics related to large turns, such as
    the number of large turns, the median distance from the reward circle during these turns,
    and the ratio of large turns to reward circle exits. Each column name is further specified
    by the trial number and a description of the fly involved.

    Parameters:
    - va (VideoAnalysis): A VideoAnalysis instance. It must have attributes `trns` (trainings)
                          and `trx` (trajectories), where each training and trajectory has
                          properties used in naming.

    Returns:
    - list of str: A list of column names. Each column name is formatted to include the metric
                   type, trial number, and a descriptive label for the fly involved in the
                   turn events.
    """
    types = (
        "# lg turns",
        "median distance from rwd circle for lg turns",
        "# lg turns / # rwd circle exits",
    )
    out = []
    for t in va.trns:
        for trj in va.trx:
            for tp in types:
                out.append("%s- T%i- %s" % (tp, t.n, flyDesc(trj.f)))
    return out


def columnNamesForBoundaryContactType(va, tp):
    """
    Generates a list of column names for analyzing contact events with boundaries or zones,
    specified by type.

    This function creates column names based on the type of boundary contact event being
    analyzed (e.g., wall, agarose), including specifics like contact type (all-wall, sidewall,
    agarose-adjacent) and context (e.g., / rwd for reward-related). Each column name is
    further detailed by the trial number, the start or end of the trial, and the fly involved.

    Parameters:
    - va (VideoAnalysis): A VideoAnalysis instance. It must have attributes `trns`
                          (trainings), `flies` (fly identifiers), and potentially other
                          properties relevant to boundary contact analysis.
    - tp (str): A string indicating the type of contact event to analyze (e.g., "wall",
                "agarose", "agarose_pct").

    Returns:
    - list of str: A list of column names tailored for the analysis of specific boundary
                   contact events. Each name incorporates details about the contact type,
                   trial information, and the fly involved, formatted for clarity and
                   specificity.
    """
    evt_tp = "_".join(tp.split("_")[:-1])
    directions = [""]
    if evt_tp == "wall":
        contact_types = ["all-wall", "sidewall", "agarose-adjacent"]
        suffixes = (" / rwd",)
    elif evt_tp in ("agarose", "boundary"):
        contact_types = [evt_tp]
        suffixes = ("", " / rwd")
    elif evt_tp == "agarose_pct":
        contact_types = ["% time spent on agarose"]
        suffixes = ("",)
    elif evt_tp == "boundary_pct":
        contact_types = ["% time spent past border"]
        suffixes = ("",)
    elif "_turn" in evt_tp:
        directional = "_dir" in evt_tp
        ellipse_ref_tp, boundary_tp = evt_tp.split("_turn")[0].split("_")
        variants = [""]
        if ellipse_ref_tp == "edge":
            variants.extend(["inside-line ", "outside-line "])
        if directional:
            directions = ["toward-center ", "away-from-center "]
        contact_types = ["%s%s turns" % (variant, boundary_tp) for variant in variants]
        suffixes = (" / contact evt",)
    else:
        contact_types = [""]
        suffixes = [""]
    if evt_tp in ("wall", "agarose", "boundary"):
        contact_types = ["%s contact evts" % el for el in contact_types]
    columns = []
    for contact_tp in contact_types:
        for trn_n in range(len(va.trns) + 1):
            for f in va.flies:
                for direction in directions:
                    for suffix in suffixes:
                        if trn_n >= 1:
                            trn_desc = "T%i %s" % (
                                trn_n,
                                "start" if trn_n == 1 else "end",
                            )
                        else:
                            trn_desc = "pre last 10m"
                        if len(contact_tp) == 0 and len(suffix) == 0:
                            prefix = ""
                        else:
                            prefix = "%s%s%s- " % (direction, contact_tp, suffix)
                        columns.append("%s%s (%s)" % (prefix, trn_desc, flyDesc(f)))
    return columns


def columnNamesForType(va, tp, calc, n):
    """
    Generates a list of column names for a specified analysis type, potentially
    including calculation-specific and training-specific variations.

    Parameters:
    - va (VideoAnalysis): An instance of the VideoAnalysis class, representing the
      context of the video analysis.
    - tp (str): The type of analysis being performed, which determines the format
      and content of the column names. Examples include 'atb', 'adb', 'nr', etc.
    - calc (bool): Indicates whether additional calculation-specific columns should
      be included. This often affects whether pre- and post-calculation columns are
      added.
    - n (int): A numeric parameter that influences the generation of certain column
      names, particularly in types that involve numerical thresholds or bucket counts.

    Returns:
    - list of str: A list containing the generated column names appropriate for the
      specified analysis type. The format and number of columns can vary significantly
      based on the type of analysis and the provided parameters.

    Raises:
    - ArgumentError: If the provided analysis type (`tp`) is not recognized or supported
      by the function.

    Note:
    This function is designed to be highly flexible to accommodate a wide range of analysis
    types and configurations. The returned column names are tailored to the specifics of
    the analysis, including distinctions between different types of data (e.g., behavioral
    analysis, contact analysis), calculation variations, and the structure of training
    sessions.
    """

    def fiNe(pst, f=None):
        if va.noyc and f == 1:
            return ()
        fly = "" if f is None else "%s fly " % (flyDesc(f))
        return "%sfirst%s" % (fly, pst), "%snext%s" % (fly, pst)

    bl = bucketLenForType(tp)[1]
    if tp in ("atb", "adb", "adb_csv"):
        nr = " %d" % n
        if tp == "adb_csv":
            pre_cols = ["pre last 10m (%s)" % flyDesc(f) for f in range(len(va.trx))]
            cols = (
                fiNe(nr, 0)
                + fiNe(nr, 1)
                + tuple(["%s bucket #%%i" % (flyDesc(f)) for f in range(len(va.trx))])
            )
            cols_per_trn = len(cols)
            cols = ",".join(cols)
            cols = duplicateColumnsAcrossTrns(cols, va.trns).split(",")
            for t in va.trns:
                for i in range(2 * len(va.flies), 3 * len(va.flies)):
                    cols[(t.n - 1) * cols_per_trn + i] = cols[
                        (t.n - 1) * cols_per_trn + i
                    ] % (1 if t.n == 1 else 5)
            return pre_cols + cols
        return fiNe(nr, 0) + fiNe(nr, 1) if calc or tp == "adb" else fiNe(nr)
    elif tp in ("nr", "nrc"):
        if opts.numBuckets == 1:
            return [f"{flyDesc(f)} fly" for f in range(len(va.trx))]
        bl = " %s min" % bl
        return fiNe(bl, 0) + fiNe(bl, 1) if calc else fiNe(bl)

    elif tp == "ppi":
        return ("post %s min" % bl,)
    elif tp in ("rpi", "bysb2"):
        n = len(vaVarForType(va, tp, calc)[0])
        bl = "%s min" % bl

        def cols(f):
            if va.noyc and f == 1:
                return ()
            cs = ["#%d" % (i + 1) for i in range(n)]
            cs[0] = "%s fly %s %s" % (flyDesc(f), bl, cs[0])
            return tuple(cs)

        return cols(0) + cols(1)
    elif tp == "nrp":
        bl = " %s min" % bl

        def cols(f):
            if va.noyc and f == 1:
                return ()
            cs = ("trn. last", "post 1st", "post 2nd", "post 3rd")
            return tuple("%s fly %s%s" % (flyDesc(f), c, bl) for c in cs)

        return cols(0) + cols(1)
    elif tp in ("dbr_no_contact_csv", "max_ctr_d_no_contact_csv"):
        return ["T%i end (%s)" % (len(va.trns), flyDesc(f)) for f in va.flies]
    elif (
        tp
        in (
            "agarose_csv",
            "boundary_csv",
            "wall_csv",
        )
        or ("r_no_contact" in tp and "csv" in tp)
        or "agarose_pct" in tp
        or "boundary_pct" in tp
        or "agarose_turn_csv" in tp
        or "agarose_turn_dir_csv" in tp
        or "boundary_turn_csv" in tp
        or "boundary_turn_dir_csv" in tp
        or "wall_turn_csv" in tp
        or "wall_turn_dir_csv" in tp
    ):
        return columnNamesForBoundaryContactType(va, tp)
    elif tp == "btwn_rwd_dist_from_ctr":
        pass
        return [
            f"{metric} dist - {timeframe} ({flyDesc(f)})"
            for f in va.flies
            for timeframe in ("first 24", "last 24")
            for metric in ("max", "mean")
        ]
    elif tp == "lg_turn_rwd_csv":
        return columnNamesForLargeTurnAnalysis(va)
    elif (
        tp
        in (
            "nrpp",
            "rpid",
            "rpip",
            "pcm",
            "turn",
            "pivot",
            "agarose",
            "boundary",
            "wall",
            "rpipd",
            "dbr",
            "dbr_no_contact",
            "max_ctr_d_no_contact",
        )
        or "_turn" in tp
        or "r_no_contact" in tp
    ):
        return None
    elif tp == "c_pi":
        if va.openLoop:
            ps = (" pre",) + (("",) if va.alt else (" on", " off"))
        else:
            ps = ("", " post") if opts.numBuckets == 1 else (" first", " next")

        def cols(f):
            if va.noyc and f == 1:
                return ()
            return tuple("%s fly %s" % (flyDesc(f), p) for p in ps)

        return cols(0) + cols(1)
    elif tp == "rdp":
        return "exp fly", "yok fly"
    elif tp in ("frc", "xmb", "rpm"):
        return ("exp fly",)
    elif tp == "bot_top_cross":
        cols = []
        for timeframe in ("", "post"):
            cols.extend([f"{flyDesc(f)} fly {timeframe}" for f in range(len(va.trx))])
        return cols
    elif tp in ("spd", "stp"):

        def cols(f):
            if va.noyc and f == 1:
                return ()
            f = "%s fly " % (flyDesc(f))
            return (f + "pre", f + "training")

        return cols(0) + cols(1)
    elif tp in ("pic", "pic_custom"):
        return colNamesForPrePostAndTraining(
            va,
            ((1, 1), (2, 1), (bl, 2)),
            pre_bl=(
                ((3, 1),),
                ((10, 1),),
            ),
            entireTrn=True,
        )
    elif tp == "rpi_combined":
        return colNamesForPrePostAndTraining(
            va,
            bl,
            entirePrePost=False,
            bySyncB=True,
            extendedPost=True,
            intersperseExpYok=True,
        )
    elif tp == "spd_sb":
        return colNamesForPrePostAndTraining(
            va,
            bl,
            entirePrePost=False,
            bySyncB=True,
            extendedPost=False,
            intersperseExpYok=True,
        )
    else:
        raise ArgumentError(tp)


def colNamesForPrePostAndTraining(
    va,
    post_bl,
    pre_bl=None,
    entireTrn=False,
    bySyncB=False,
    entirePrePost=True,
    extendedPost=False,
    intersperseExpYok=False,
):
    """
    Returns a tuple of column names organized by training sessions, with configurable
    entries for pre and post-analysis periods and an option to intersperse column names
    with experimental and yoked control groups.

    Parameters:
    - va (VideoAnalysis): An instance of the VideoAnalysis class, providing context.
    - post_bl (str, int, or tuple of tuples): The bucket length(s) for the post-period, influencing
      naming of columns related to time/duration. If tuple of tuples, each inner tuple should contain (bucket_length, number_of_periods).
    - pre_bl (str, int, tuple of tuples, or tuple of two tuples of tuples, optional):
      The bucket length(s) for the pre-period, influencing naming of columns related to time/duration.
      If a tuple of tuples, each inner tuple should contain (bucket_length, number_of_periods).
      If a tuple of two tuples of tuples, the first set is anchored to the start of the pre-period,
      and the second set is anchored to the end. Defaults to None, meaning only the entire pre-period will be used.
    - entireTrn (bool, optional): Include a column for the entire training period. Defaults to False.
    - bySyncB (bool, optional): Include columns for sync buckets of specific trainings. Defaults to False.
    - entirePrePost (bool, optional): Adds columns for the entire pre and post periods. Defaults to True.
    - extendedPost (bool, optional): Includes alternate post-bucket durations for the last training. Defaults to False.
    - intersperseExpYok (bool, optional): Intersperse column names with "(exp)" and "(yok)" to indicate experimental and yoked control groups. Defaults to False.

    Returns:
    - tuple of str: A tuple containing the generated column names, organized to reflect
      the structure of the analysis and the specified configuration options.
    """
    skipT = not entireTrn and not bySyncB
    if intersperseExpYok and len(va.flies) == 1:
        intersperseExpYok = False

    def create_col_name(base_name, is_exp, modifier=""):
        label = " (exp)" if is_exp else " (yok)"
        return (
            f"{base_name}{label}{modifier}"
            if intersperseExpYok
            else f"{base_name}{modifier}"
        )

    def ordinal(n):
        return f"{n}{'tsnrhtdd'[(n//10%10!=1)*(n%10<4)*n%10::4]}"

    def reverse_ordinal(n):
        reverse_ordinals = {1: "last", 2: "2nd-to-last", 3: "3rd-to-last"}
        return reverse_ordinals.get(n, f"{ordinal(n)}-to-last")

    cs = []

    if isinstance(post_bl, (str, int)):
        post_bl_info = [
            (post_bl, 2)
        ]  # Use default of two periods if post_bl is a single value
    else:
        post_bl_info = post_bl

    if pre_bl is None:
        pre_bl_info_start = []
        pre_bl_info_end = []
    elif isinstance(pre_bl, (str, int)):
        pre_bl_info_start = [(pre_bl, 1)]
        pre_bl_info_end = []
    elif isinstance(pre_bl[0][0], tuple):
        pre_bl_info_start = pre_bl[0]
        pre_bl_info_end = pre_bl[1]
    else:
        pre_bl_info_start = pre_bl
        pre_bl_info_end = []

    # Pre-training columns (start-anchored)
    for bl_len, periods in pre_bl_info_start:
        bl_str = f" {bl_len} min"
        for period in range(1, periods + 1):
            ordinal_period = ordinal(period)
            pre_name = f"pre {ordinal_period}{bl_str}"
            if intersperseExpYok:
                cs.append(create_col_name(pre_name, True))
                cs.append(create_col_name(pre_name, False))
            else:
                cs.append(pre_name)

    # Pre-training columns (end-anchored)
    total_periods = sum(p for _, p in pre_bl_info_end)
    for bl_len, periods in pre_bl_info_end:
        bl_str = f" {bl_len} min"
        for period in range(1, periods + 1):
            reverse_ord_period = reverse_ordinal(total_periods - period + 1)
            pre_name = f"pre {reverse_ord_period}{bl_str}"
            if intersperseExpYok:
                cs.append(create_col_name(pre_name, True))
                cs.append(create_col_name(pre_name, False))
            else:
                cs.append(pre_name)
        total_periods -= periods

    if entirePrePost:
        if intersperseExpYok:
            cs.append(create_col_name("entire pre", True))
            cs.append(create_col_name("entire pre", False))
        else:
            cs.append("entire pre")

    # Training and post-training columns
    for t in va.trns:
        if entireTrn:
            cs.extend(
                [create_col_name(t.name(), True), create_col_name(t.name(), False)]
                if intersperseExpYok
                else [t.name()]
            )
        modifiers = ("", "(no ctrl sync)") if extendedPost and t.n == 3 else ("",)

        for bl_len, periods in post_bl_info:
            bl_str = f" {bl_len} min"
            for period in range(1, periods + 1):
                ordinal_period = ordinal(period)
                for modifier in modifiers if ordinal_period == "1st" else ("",):
                    post_name = f"{t.name() + ' ' if period == 1 and not skipT else ''}post {ordinal_period}{bl_str}{modifier}"
                    if intersperseExpYok:
                        cs.append(create_col_name(post_name, True))
                        cs.append(create_col_name(post_name, False))
                    else:
                        cs.append(post_name)

    # Adjusting the insertion of sync bucket columns
    if bySyncB:
        for tIdx, t in enumerate(va.trns):
            sb_name = f"{t.name()} {'first' if tIdx == 0 else 'last'} SB"
            insert_pos = len(cs)
            for i, col in enumerate(cs):
                if t.name() in col:
                    insert_pos = i
                    break
            if intersperseExpYok:
                cs.insert(insert_pos, create_col_name(sb_name, True))
                cs.insert(insert_pos + 1, create_col_name(sb_name, False))
            else:
                cs.insert(insert_pos, sb_name)

    if entirePrePost:
        pre_post_cols = (
            [
                create_col_name(f"entire post-T{len(va.trns)}", True),
                create_col_name(f"entire post-T{len(va.trns)}", False),
            ]
            if intersperseExpYok
            else [f"entire post-T{len(va.trns)}"]
        )
        return tuple(cs + pre_post_cols)
    else:
        pre_col = (
            [
                create_col_name("pre last 10 min", True),
                create_col_name("pre last 10 min", False),
            ]
            if intersperseExpYok
            else ["pre last 10 min"]
        )
        return tuple(pre_col + cs)


def vaVarForType(va, tp, calc):
    """
    Retrieves the relevant variable(s) from a VideoAnalysis instance based on the specified
    analysis type and calculation flag.

    Parameters:
    - va (VideoAnalysis): The VideoAnalysis instance containing experimental data and analysis
                          results.
    - tp (str): The type of analysis, which determines which variable(s) to retrieve. Examples
                include 'atb' (average time between), 'adb' (average distance between), 'nr'
                (number of rewards), among others.
    - calc (bool): A flag indicating whether the calculation is for a calculated metric (True)
                   or an actual metric (False). This flag is relevant for certain types of
                   analysis where the distinction between calculated and actual metrics is
                   significant.

    Returns:
    - Variable(s): Depending on the analysis type and the calc flag, this function returns the
                   appropriate variable(s) from the VideoAnalysis instance. The return type
                   can vary (e.g., list, numpy array, scalar) based on the specific analysis
                   type.

    Raises:
    - ArgumentError: If the provided analysis type (`tp`) is not recognized or if the
                     calculation flag does not apply to the given type.

    Note:
    This function is a key component of a larger analysis pipeline, allowing for flexible
    retrieval of data based on analysis needs. The distinction between calculated and actual
    metrics, as well as the variety of analysis types, reflects the complex nature of
    behavioral data analysis.
    """
    if tp == "atb":
        return va.avgTimeBtwnCalc if calc else va.avgTimeBetween
    elif "adb" in tp:
        if calc:
            if "csv" in tp:
                return [va.distBetweenCalcForCSV()]
            else:
                return va.avgDistBtwnCalc
        else:
            return va.avgDistBetween
    elif tp in ("nr", "nrc"):
        return va.numRewards[calc][tp == "nrc"]
    elif tp == "ppi":
        return va.posPI
    elif tp == "rpi":
        return va.rewardPI
    elif tp == "rpid":
        return va.rewardPI
    elif tp == "rpi_combined":
        return va.rewardPiCombined()
    elif tp == "rpip":
        return va.rewardPiPst
    elif tp == "rpipd":
        return va.rewardPiPst
    elif tp == "nrp":
        return va.numRewardsPost
    elif tp == "nrpp":
        return va.numRewardsPostPlot
    elif tp == "c_pi":
        return va.posPI
    elif tp == "rdp":
        return va.rdpAvgLL
    elif tp == "bysb2":
        return va.bySB2
    elif tp == "frc":
        return va.firstRewardCtrl
    elif tp == "xmb":
        return va.xedMidlineBefore
    elif tp == "spd":
        return va.speed
    elif tp == "spd_sb":
        return va.speeds_over_sbs
    elif tp == "stp":
        return va.stopFrac
    elif tp == "rpm":
        return va.rewardsPerMin
    elif tp == "pcm":
        return va.circleSync
    elif tp == "turn":
        return va.turnResults["meanTurnRadii"]
    elif tp == "pivot":
        return va.turnResults["numPivots"]
    elif tp == "pic":
        return va.pctInC["rwd"]
    elif tp == "pic_custom":
        return va.pctInC["custom"]
    elif tp == "dbr":
        return va.avgDistancesByBkt
    elif tp == "dbr_no_contact_csv":
        return [va.contactless_trajectory_lengths[opts.wall_orientation]["csv"]]
    elif tp == "max_ctr_d_no_contact_csv":
        return [va.contactless_max_dists[opts.wall_orientation]["csv"]]
    elif tp == "dbr_no_contact":
        return va.contactless_trajectory_lengths[opts.wall_orientation]["trn"]
    elif tp == "max_ctr_d_no_contact":
        return va.contactless_max_dists[opts.wall_orientation]["trn"]
    elif "r_no_contact" in tp:
        if "agarose" in tp:
            boundary_orientation = "agarose_adj"
        else:
            boundary_orientation = "all"
        if "csv" in tp:
            return va.contactless_rewards[boundary_orientation]["csv"]
        return va.contactless_rewards[boundary_orientation]["trn"]
    elif tp == "btwn_rwd_dist_from_ctr":
        return va.btwnRwdDistsFromCtr
    elif tp in ("agarose", "wall", "boundary"):
        if tp == "agarose" or tp == "boundary":
            return va.boundary_events["%s_contact" % tp]
        return va.boundary_events["%s_contact" % tp][
            :, va.wall_orientation_idx :: len(va.wall_orientations), :
        ]
    elif tp == "bot_top_cross":
        return va.botToTopCrossings
    elif "agarose_pct" in tp or "boundary_pct" in tp:
        tp_split = tp.split("_")
        region_tp = "_".join(tp_split[:-2])
        contact_tp = tp_split[-1]
        return [va.regionPercentagesCsv[region_tp][contact_tp]]
    elif "turn_duration" in tp:
        return va.boundary_event_durations[tp.split("_duration")[0]]
    elif tp == "wall_csv":
        return va.wall_contact_evts_for_csv
    elif tp == "agarose_csv":
        return va.agarose_contact_evts_for_csv
    elif tp == "boundary_csv":
        return va.boundary_contact_evts_for_csv
    elif tp == "lg_turn_rwd_csv":
        return [va.large_turn_stats.flatten()]
    elif "_turn_csv" in tp:
        boundary_tp = tp.split("_turn_csv")[0]
        attrs_to_fetch = ["%s_turn_evts_for_csv" % boundary_tp]
        if "edge" in boundary_tp:
            boundary_tp = boundary_tp.split("_")[1]
            attrs_to_fetch.append("inside_line_%s_turn_evts_for_csv" % boundary_tp)
            attrs_to_fetch.append("outside_line_%s_turn_evts_for_csv" % boundary_tp)
        return np.hstack(
            [getattr(va, attr_to_fetch) for attr_to_fetch in attrs_to_fetch]
        )
    elif "turn_dir" in tp:
        tp_split = tp.split("_")
        ellipse_ref_pt = tp_split[0]
        boundary_tp = tp_split[1]
        return [va.turn_dir_metrics[boundary_tp][ellipse_ref_pt]]
    elif "_turn" in tp:
        if "_exp_min_yok" in tp:
            tp = tp.split("_exp_min_yok")[0]
        if "line" in tp:
            return va.boundary_events[tp]
        if "ctr" in tp or "edge" in tp:
            return va.boundary_events[tp]
    else:
        raise ArgumentError(tp)


def trnsForType(va, tp):
    """
    Selects and returns training sessions relevant to the specified analysis type.

    Parameters:
    - va (VideoAnalysis): The VideoAnalysis instance containing information about various
                          trials and experimental conditions.
    - tp (str): The type of analysis, which determines the set of training sessions to be
                returned. This influences which trials are considered relevant based on the
                analysis focus (e.g., positional PI, reward PI, etc.).

    Returns:
    - list: A list of trainings deemed relevant for the specified analysis type. The criteria
            for relevance vary by analysis type, with some types requiring specific subsets of
            trials or even excluding certain trials based on experimental conditions.

    Note:
    The returned list of trials is instrumental in conducting analyses that are tailored to
    specific experimental conditions or objectives. By filtering trials based on the analysis
    type, this function facilitates focused and relevant data analysis.
    """
    if tp == "ppi":
        return [] if opts.rdp else va.posPITrns
    elif tp in ("rpi", "rpid", "rpipd"):
        return va.rewardPITrns
    elif tp == "rdp":
        return va.trns[-1:] if opts.rdp else []
    else:
        return va.trns


def typeCalc(tc):
    """
    Parses a type-calculation string to identify the analysis type and whether it pertains to
    calculated rewards.

    Parameters:
    - tc (str): A string combining the analysis type with an optional 'c' flag, separated by a
                dash. The flag 'c' indicates that the analysis refers to calculated rewards,
                which are based on the trajectory of the fly entering the training circle,
                rather than actual rewards that involve the delivery of a stimulus.

    Returns:
    - (str, bool): A tuple where the first element is the analysis type and the second element
                   is a boolean indicating whether the analysis pertains to calculated rewards
                   (True if 'c' is present, otherwise False).

    Examples:
    - typeCalc("atb-c") returns ("atb", True), indicating an analysis type of 'atb' with
      calculated rewards.
    - typeCalc("adb") returns ("adb", False), indicating an analysis type of 'adb' without
      specifying reward calculation type.
    """
    ps = tc.split("-")
    return ps[0], ps[1] == "c" if len(ps) > 1 else False


def checkValues(vas, tp, calc, a):
    """
    Validates that data for 'bad' trajectories are marked as NaN.

    Parameters:
    - vas (list[VideoAnalysis]): A list of VideoAnalysis objects, each representing analysis
                                 results for a video.
    - tp (str): The type of analysis being conducted, which affects which trajectories are
                considered.
    - calc (bool): Indicates the nature of the rewards being analyzed: True for calculated
                   rewards based on trajectories, and False for actual rewards.
    - a (numpy.ndarray): An array containing the analysis data. It is expected to have
                         dimensions corresponding to [video index, analysis dimension,
                         trajectory/bucket index].

    Raises:
    - AssertionError: If any 'bad' trajectory identified by VideoAnalysis objects does not
                      have its corresponding data set to NaN in the array.

    """
    fs = fliesForType(vas[0], tp, calc)
    npf = int(a.shape[2] / len(fs))
    for i, va in enumerate(vas):
        for f in fs:
            if va._bad(f):
                assert np.all(np.isnan(a[i, :, f * npf : (f + 1) * npf]))


FLY_COLS = ("#1f4da1", "#a00000")


def drawLegend(ng, nf, nrp, gls, customizer):
    """
    Generates and adds a legend to the plot based on the experimental setup, including
    distinguishing between experimental and control groups, and accommodating for various
    plotting preferences such as font size and location.

    Parameters:
    - ng, nf, nrp, gls: As previously defined.
    - customizer: Plot customizer with font size attributes.
    """
    kwargs = {}
    prop_dict = {"style": "italic"}
    if ng == 1 and nf == 2 and not P:
        kwargs["labels"] = ["Experimental", "Yoked"]
        kwargs["handles"] = [plt.Line2D([], [], color=FLY_COLS[i]) for i in range(2)]
        kwargs["loc"] = "best"
    elif gls and (not P or LEG):
        if customizer.font_size_customized:
            kwargs["loc"] = "best"
        else:
            kwargs["loc"] = 1 if nrp else 4
            prop_dict["size"] = "medium"
    kwargs["prop"] = prop_dict
    if not gls and "labels" not in kwargs:
        return None
    return plt.legend(**kwargs)


def plotRewards(va, tp, a, trns, gis, gls, vas=None):
    """
    Plots reward data from Drosophila behavior analysis.

    This function visualizes the reward data collected from experiments involving Drosophila,
    grouping the data according to user-defined criteria and displaying it in a meaningful way
    to facilitate behavioral analysis.

    Parameters:
    - va (VideoAnalysis): An instance of VideoAnalysis representing the analysis of a single video.
    - tp (str): The type of plot to generate, which determines how the data is visualized.
    - a (numpy.ndarray): An array of data to be plotted, structured to match the requirements
                         of the specified plot type.
    - trns (list[Training]): A list of Training instances, each containing training data for
                             the Drosophila involved in the analysis.
    - gis (list[int]): Indices specifying the group to which each VideoAnalysis instance belongs.
    - gls (list[str]): Labels for each group, used for annotating the plot.
    - vas (list[VideoAnalysis]): A list of VideoAnalysis instances, each analysis corresponding
                                 to a different video.

    Returns:
    - The function does not return a value but generates and displays a plot based on the
      provided data and parameters.

    Raises:
    - ValueError: If any of the input parameters are invalid or not in the expected format.
    - RuntimeError: If there's an issue generating the plot.
    """
    nrp, rpip = tp == "nrpp", tp in ("rpip", "rpipd")
    r_diff = tp in ("rpid", "rpipd")
    diff_tp = r_diff or "exp_min_yok" in tp
    bnd_contact = tp in ("wall", "agarose", "boundary")
    bnd_turn = "_turn" in tp
    pcm, turn_rad, pivot = tp == "pcm", tp == "turn", tp == "pivot"
    circle = pcm or turn_rad or pivot
    txt_positioned_vert_mid = (
        circle or bnd_contact or bnd_turn or "dbr" in tp or "no_contact" in tp
    )
    post = nrp or rpip
    nnpb = va.rpiNumNonPostBuckets if rpip else va.numNonPostBuckets
    fs, ng = fliesForType(va, tp), gis.max() + 1
    nf = len(fs)
    nb, (meanC, fly2C) = int(a.shape[2] / nf), FLY_COLS

    def getVals(g, b=None, delta=False, f1=None):
        vis = np.flatnonzero(gis == g)

        def gvs(f):
            o = f * nb
            return a[vis, i, o : o + nb] if b is None else a[vis, i, o + b]

        return gvs(0) - gvs(1) if delta else gvs(f1 if f1 is not None else f)

    meanOnly, showN, showV, joinF, fillBtw = True, True, False, True, True
    showPG, showPP = True, True  # p values between groups, for post
    showPFL = (
        True if not opts.hidePltTests else False
    )  # p values between first and last buckets
    showPT = not P if not opts.hidePltTests else False  # p values between trainings
    showSS = not P  # speed stats
    if showSS and vas:
        speed, stpFr = (
            np.array([getattr(va, k) for va in vas]) for k in ("speed", "stopFrac")
        )
        speed, stpFr = (np.nanmean(a, axis=0) for a in (speed, stpFr))
    nr = 1 if joinF else nf
    legend = None
    bl, blf = bucketLenForType(tp)
    xs = (np.arange(nb) + (-(nnpb - 1) if post else 1)) * bl
    if nrp:
        ylim = [0, 60]
    elif bnd_contact:
        ylim = [0, 15]
    elif tp == "dbr":
        ylim = [0, 1600]
    elif tp == "dbr_no_contact":
        ylim = [0, 150]
    elif tp == "max_ctr_d_no_contact":
        ylim = [0, 40]
    elif "duration" in tp:
        if "_exp_min_yok" in tp:
            ylim = [-2, 2]
        else:
            ylim = [0, 2]
    elif r_diff:
        ylim = [-0.5, 1.5]
    else:
        ylim = [-1, 1]
    if circle:
        ylim[0] = 0
    lbls, fbv = {}, []
    tas = 2 * [None]  # index: 0:under curve, 1:between curves
    if P and F2T:
        trns = trns[:2]

    nc = len(trns)
    figsize = pch(([5.33, 11.74, 18.18][nc - 1], 4.68 * nr), (20, 5 * nr))
    if customizer.font_size_customized:
        figsize = list(figsize)
        figsize[0] += 0.2 * (customizer.font_size - customizer.font_size_default)
    axs = plt.subplots(nr, nc, figsize=figsize)[1]
    if nr == 1:
        if nc == 1:
            axs = np.array([[axs]])
        else:
            axs = axs[None]
    for f in fs:
        mc = fly2C if joinF and f == 1 else meanC
        for i, t in enumerate(trns):
            nosym = not t.hasSymCtrl()
            comparable = not (nf == 1 and nosym)
            ax = axs[0 if joinF else f, i]
            plt.sca(ax)
            if P and f == 0:
                plt.locator_params(axis="y", nbins=5)
            # delta: return difference between fly 0 and fly 1
            if not meanOnly:
                # plot line for each video
                assert ng == 1
                for v in range(a.shape[0]):
                    ys = a[v, i, f * nb : (f + 1) * nb]
                    fin = np.isfinite(ys)
                    plt.plot(
                        xs[fin],
                        ys[fin],
                        color="skyblue" if f == 0 else "tomato",
                        marker="o",
                        ms=3,
                    )
            if i == 0 and f == 0:
                if not rpip and not r_diff:
                    all_line_vals = []
                    for grp_idx in range(ng):
                        for f_idx in (0,) if diff_tp else va.flies:
                            line_vals = np.array(
                                [
                                    util.meanConfInt(
                                        getVals(grp_idx, b, False, f1=f_idx)
                                    )[0]
                                    for b in range(nb)
                                ]
                            ).T
                            all_line_vals.append(line_vals)
                else:
                    all_line_vals = []
            # plot mean and confidence interval
            for g in range(ng):  # group
                if pcm:
                    rewardsAvgs = np.mean(
                        [
                            list(rewards)
                            for rewards in [
                                va.numRewardsTot[1][0][slice(opts.yoked, None, 2)]
                                for va in np.asarray(vas)[np.flatnonzero(gis == g)]
                            ]
                            if len(
                                np.flatnonzero(
                                    [len(rewardGrp) == nb for rewardGrp in rewards]
                                )
                            )
                            == len(rewards)
                        ],
                        axis=0,
                    )
                    assert rewardsAvgs.shape == (nc, nb)
                mci = np.array([util.meanConfInt(getVals(g, b)) for b in range(nb)]).T
                if turn_rad or pivot:
                    ylim[1] = 8 if turn_rad else 50
                    # 4 rows: mean, lower bound, upper bound, number samples
                if not (rpip and f == 1 and not nosym):
                    for j in range(3):
                        ys = mci[j, :]
                        fin = np.isfinite(ys)
                        linestyles = ["-", "--", ":"]
                        if j == 0 or not fillBtw:
                            (line,) = plt.plot(
                                xs[fin],
                                ys[fin],
                                color=mc,
                                marker="o",
                                ms=3 if j == 0 else 2,
                                mec=mc,
                                linewidth=2 if j == 0 else 1,
                                linestyle=linestyles[g] if j == 0 else linestyles[1],
                            )
                            if i == 0 and j == 0 and f == 0 and gls:
                                line.set_label(gls[g] + (" yoked-ctrl" if f else ""))
                        if j == 2 and fillBtw:
                            plt.fill_between(
                                xs[fin], mci[1, :][fin], ys[fin], color=mc, alpha=0.15
                            )
                    # sample sizes
                    if showN and (not nrp or i == 0) and (ng == 1 or f == 0):
                        for j, n in enumerate(mci[3, :1] if nrp else mci[3, :]):
                            if n > 0:
                                lblText = (
                                    "%.2f" % rewardsAvgs[i + f][j] if pcm else "%d" % n
                                )
                                y, key, m = (
                                    mci[0, j],
                                    util.join("|", (i, j)),
                                    (ylim[1] - ylim[0]) / 2,
                                )
                                y_pos = y + 0.04 * m
                                txt = util.pltText(
                                    xs[j],
                                    y_pos,
                                    lblText,
                                    ha="center",
                                    size=customizer.in_plot_font_size,
                                    color=".2",
                                )
                                if i == 0:
                                    all_line_vals.append([y_pos])
                                txt1 = lbls.get(key)
                                if txt1:
                                    y1 = txt1._y_
                                    txt1._firstSm_ = y1 < y
                                    if (
                                        abs(y1 - y) < pch(0.14, 0.1) * m
                                    ):  # move label below
                                        txta, ya = (txt, y) if y1 > y else (txt1, y1)
                                        txta.set_y(ya - pch(0.04, 0.03) * m)
                                        txta.set_va("top")
                                        txta._ontp_ = False
                                else:
                                    txt._y_, txt._ontp_, txt._firstSm_ = y, True, False
                                    lbls[key] = txt
                    # values
                    if showV:
                        for j, y in enumerate(mci[0, :]):
                            if np.isfinite(y):
                                util.pltText(
                                    xs[j],
                                    y - 0.08 * (30 if nrp else 1),
                                    ("%%.%df" % (1 if nrp else 2)) % y,
                                    ha="center",
                                    size=customizer.in_plot_font_size,
                                    color=".2",
                                )
                # t-test p values
                if (
                    showPG
                    and ng == 2
                    and g == 1
                    and f == 0
                    or rpip
                    and showPP
                    and ng == 1
                    and f == nf - 1
                    and comparable
                    or r_diff
                    and ng == 1
                ) and not nrp:
                    if g == 1:
                        cmpg = True
                        dlt = nosym if nf == 2 else False
                    else:
                        cmpg = False
                        dlt = nosym if not r_diff else False
                    tpm = np.array(
                        [
                            (
                                ttest_ind(getVals(0, b, dlt), getVals(1, b, dlt))
                                if cmpg
                                else ttest_1samp(getVals(0, b, dlt, 0), 0)
                            )[:2]
                            + (np.nanmean(getVals(int(cmpg), b)),)
                            for b in range(nb)
                        ]
                    ).T
                    # 3 rows: t-test t and p and mean for g == int(cmpg)
                    assert util.isClose(mci[0, :], tpm[2, :])
                    font_size_adjustment_factor = 0.003
                    additional_space = (
                        font_size_adjustment_factor
                        * customizer.font_size_diff
                        * (ylim[1] - ylim[0])
                    )
                    for j, p in enumerate(tpm[1, :]):
                        txt = lbls.get(util.join("|", (i, j)))
                        if txt:
                            y, ontp, fs = txt._y_, txt._ontp_, txt._firstSm_
                            strs = util.p2stars(p, nanR="")
                            sws = strs.startswith("*")
                            if not cmpg and not nosym and not sws:
                                continue
                            y += 0 if sws else pch(0.02, 0.015) * m
                            ys = (
                                y - pch(0.15, 0.105) * m
                                if not ontp
                                else (
                                    y - pch(0.06, 0.045) * m
                                    if fs
                                    else y + pch(0.13, 0.1) * m
                                )
                            )
                            if ys > y:
                                ys += additional_space
                            else:
                                ys -= additional_space
                            util.pltText(
                                xs[j],
                                ys,
                                strs,
                                ha="center",
                                va=("baseline" if ys > y else "top"),
                                size=customizer.in_plot_font_size,
                                color="0",
                                weight="bold",
                            )
                            if i == 0:
                                all_line_vals.append([ys])
                    # AUC
                    if not (rpip) and not opts.hidePltTests:
                        if txt_positioned_vert_mid:
                            yp = 0.8 * (ylim[1] - ylim[0])
                        else:
                            yp = -0.79 if nosym else pch(-0.55, -0.46)
                        printed_header = False
                        for btwn in pch((False,), (False, True)):
                            if nosym and not btwn or nf == 1 and btwn:
                                continue
                            a_ = tuple(
                                areaUnderCurve(getVals(x, None, btwn)) for x in (0, 1)
                            )
                            if tas[btwn] is None:
                                tas[btwn] = a_
                            else:
                                tas[btwn] = util.tupleAdd(tas[btwn], a_)
                            for tot in (False, True):
                                if i == 0 and tot:
                                    continue

                                def getA(g):
                                    return (
                                        (tas[0][g] + a_[g] if nosym else tas[btwn][g])
                                        if tot
                                        else a_[g]
                                    )

                                try:
                                    a0, a1 = getA(0), getA(1)
                                except (
                                    TypeError
                                ):  # triggered, e.g., for 3x center training
                                    continue
                                nm = pcap(
                                    ("total " if tot else "")
                                    + (
                                        "AUC + ABC"
                                        if nosym and tot
                                        else ("ABC" if btwn else "AUC")
                                    )
                                )
                                tpn = ttest_ind(
                                    a0,
                                    a1,
                                    "%s, %s"
                                    % (
                                        "training 1-%d" % (i + 1) if tot else t.name(),
                                        nm,
                                    ),
                                    silent=True,
                                )
                                if i == 0 and not printed_header and tpn[4] is not None:
                                    print(
                                        "\narea under reward index curve or between curves "
                                        + "by group:"
                                    )
                                    printed_header = True
                                if tpn[4] is not None:
                                    print(tpn[4])
                                safety_margin = 0.05 * (ylim[1] - ylim[0])
                                for vals in all_line_vals:
                                    if len(vals) <= 1:
                                        continue
                                    valid_vals = sorted(vals[~np.isnan(vals)])
                                    if not valid_vals:
                                        continue
                                    closest_y_value = min(
                                        valid_vals, key=lambda y: abs(y - yp)
                                    )
                                    if abs(closest_y_value - yp) < safety_margin:
                                        if t.n == 1:
                                            yp = ylim[0] + safety_margin
                                        else:
                                            yp = ylim[1] - 2 * safety_margin
                                if i == 0:
                                    all_line_vals.append([yp])
                                util.pltText(
                                    xs[0],
                                    yp,
                                    "%s (n=%d,%d): %s"
                                    % (nm, tpn[2], tpn[3], util.p2stars(tpn[1], True)),
                                    size=pch(12, customizer.in_plot_font_size),
                                    color="0",
                                )
                                yp -= (
                                    pch(0.14, 0.11)
                                    if not txt_positioned_vert_mid
                                    else 0.05 * (ylim[1] - ylim[0])
                                )

                runPFL = showPFL and ng == 1 and f == 0 and not post and comparable
                runPT = showPT and ng == 1 and f == 0 and not post and comparable
                if runPFL or runPT:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", r"Mean of empty slice")
                        ms_group0 = np.array(
                            [np.nanmean(getVals(0, b)) for b in range(nb)]
                        )
                        if nf > 1:
                            ms_group1 = np.array(
                                [np.nanmean(getVals(0, b, f1=True)) for b in range(nb)]
                            )
                        else:
                            ms_group1 = None
                        if ms_group1 is not None:
                            max_mean_value = max(
                                np.nanmax(ms_group0), np.nanmax(ms_group1)
                            )
                            min_mean_value = min(
                                np.nanmin(ms_group0), np.nanmin(ms_group1)
                            )
                        else:
                            max_mean_value = max(ms_group0)
                            min_mean_value = min(ms_group0)
                # t-test first vs. last
                if showPFL and ng == 1 and f == 0 and not post and comparable:
                    lb = nb - 1
                    while True:
                        tpn = ttest_rel(getVals(0, 0, nosym), getVals(0, lb, nosym))
                        if tpn[3] < 2 and lb > 1:
                            lb = lb - 1
                        else:
                            break
                    assert util.isClose(mci[0, :], ms_group0)
                    x1, x2 = xs[0], xs[lb]
                    y, h, col = (
                        max_mean_value + pch(0.15, 0.13),
                        0.03,
                        "0",
                    )
                    if np.isfinite(y):
                        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                        util.pltText(
                            (x1 + x2) * 0.5,
                            y + h,
                            util.p2stars(tpn[1]),
                            ha="center",
                            va="bottom",
                            size=pch(11, customizer.in_plot_font_size),
                            color=col,
                            weight="bold",
                        )
                        bracket_top = y + h
                # t-test between trainings
                if showPT and ng == 1 and f == 0 and not post and comparable:
                    assert len(fbv) == i
                    fbv.append(getVals(0, 0, nosym))
                    if i > 0 and t.hasSymCtrl() == trns[0].hasSymCtrl():
                        tpn = ttest_rel(fbv[0], fbv[i])

                        util.pltText(
                            xs[1],
                            0.25 * (ylim[1] - ylim[0]) if circle else -0.7,
                            "1st bucket, t1 vs. t%d (n=%d): %s"
                            % (i + 1, min(tpn[2], tpn[3]), util.p2stars(tpn[1], True)),
                            size=customizer.in_plot_font_size,
                            color="0",
                        )
                # speed stats
                if showSS and ng == 1 and f == 1 and not post:
                    for f1 in va.flies:
                        i1 = i * 2 + f1
                        util.pltText(
                            xs[1],
                            -0.83 - f1 * 0.11,
                            "%s: %s/s: %s, stop: %s"
                            % (
                                flyDesc(f1),
                                "mm" if SPEED_ON_BOTTOM else "px",
                                util.join(", ", speed[i1], p=1),
                                util.join(", ", stpFr[i1], p=2),
                            ),
                            size=customizer.in_plot_font_size,
                            color="0",
                        )
            # labels etc.
            if f == 0 or not joinF:
                plt.title(
                    pcap(
                        ("post " if post else "")
                        + (
                            t.name()
                            if joinF
                            else (
                                ("%s " % t.name() if f == 0 else "")
                                + "fly %d" % (f + 1)
                            )
                        )
                    )
                )
                plt.xlabel(
                    pcap(
                        "end points [min] of %s min %sbuckets"
                        % (
                            blf,
                            (
                                ""
                                if post and not (rpip and POST_SYNC is not ST.fixed)
                                else "sync "
                            ),
                        )
                    )
                )
                if not P or i == 0:
                    ylabels = dict(
                        nrp="circle enter events",
                        turn="average turn radius (mm)",
                        pivot="number of pivots",
                        pcm="proportion circular motion",
                        wall="wall-contact events per reward",
                        agarose="agarose-contact events",
                        boundary="boundary-contact events",
                        dbr=DIST_BTWN_REWARDS_LABEL % "",
                        dbr_no_contact=DIST_BTWN_REWARDS_LABEL
                        % " (no preceding wall contact)",
                        max_ctr_d_no_contact="mean of max dists. from center\n(trajectories w/"
                        " no preceding wall contact)",
                        r_no_contact=CONTACTLESS_RWDS_LABEL,
                        rpid="reward index diff. (exp - yok)",
                        rpipd="post reward index diff. (exp - yok)",
                    )
                    if opts.turn:
                        label_variant = {"_exp_min_yok": ", exp - yoked", "": ""}
                        for turn_boundary in opts.turn:
                            for variant in label_variant:
                                ylabels["%s_turn%s" % (turn_boundary, variant)] = (
                                    TURN_LABEL % (turn_boundary, label_variant[variant])
                                )
                                ylabels[
                                    "%s_turn_duration%s" % (turn_boundary, variant)
                                ] = TURN_DURATION_LABEL % (label_variant[variant])
                                for turn_tp in ("inside", "outside"):
                                    ylabels[
                                        "%s_line_%s_turn%s"
                                        % (turn_tp, turn_boundary, variant)
                                    ] = TURN_LABEL % (
                                        turn_boundary,
                                        label_variant[variant],
                                    )
                                    ylabels[
                                        "%s_line_%s_turn_duration%s"
                                        % (turn_tp, turn_boundary, variant)
                                    ] = TURN_DURATION_LABEL % (label_variant[variant])
                    for name, boundary_orientation in (
                        ("agarose-adjacent", "agarose_adj"),
                        ("all", "all"),
                    ):
                        ylabels["r_no_contact_%s" % boundary_orientation] = (
                            CONTACTLESS_RWDS_LABEL % name
                        )

                    y_label_tp = tp
                    if y_label_tp.startswith("ctr_") or y_label_tp.startswith("edge_"):
                        y_label_tp = "_".join(y_label_tp.split("_")[1:])
                    plt.ylabel(pcap(ylabels.get(y_label_tp, "reward index")))
                plt.axhline(color="k")
                if post:
                    plt.xlim(xs[0] - bl, xs[-1] + bl)
                    plt.ylim(*ylim)
                    if nnpb > 0:  # "training end" line
                        xl = xs[nnpb - 1]
                        plt.plot(
                            [xl, xl],
                            ylim,
                            color="0.5",
                            linewidth=2,
                            linestyle="--",
                            zorder=1,
                        )
                else:
                    plt.xlim(0, xs[-1])
                    plt.ylim(*ylim)
            if i == 0 and f == 0:
                legend = drawLegend(ng, nf, nrp, gls, customizer)
    if not nrp:
        plt.subplots_adjust(wspace=opts.wspace)
    if tp in ("wall", "agarose"):
        if tp == "agarose":
            boundary_orientation = "tb"
        else:
            boundary_orientation = va.wall_orientations[va.wall_orientation_idx]
        plt.gcf().text(
            0.9,
            0.9,
            "contact-event thresholds:\nlower: %.2fmm horiz, %.2fmm vert"
            "\nupper: %.2fmm horiz, %.2fmm vert"
            % (
                getattr(
                    va.trx[0],
                    "%s_%s_event_start_dist_threshold_horiz"
                    % (tp, boundary_orientation),
                ),
                getattr(
                    va.trx[0],
                    "%s_%s_event_start_dist_threshold_vert"
                    % (tp, boundary_orientation),
                ),
                getattr(
                    va.trx[0],
                    "%s_%s_event_end_dist_threshold_horiz" % (tp, boundary_orientation),
                ),
                getattr(
                    va.trx[0],
                    "%s_%s_event_end_dist_threshold_vert" % (tp, boundary_orientation),
                ),
            ),
        )
    if tp == "wall":
        boundary_orientation = va.wall_orientations[va.wall_orientation_idx]
        if boundary_orientation == "all":
            wall_txt = "all"
        elif boundary_orientation == "tb":
            wall_txt = "top & bot."
        elif boundary_orientation == "lr":
            wall_txt = "left & right"
        elif boundary_orientation == "agarose_adj":
            wall_txt = "agarose-adjacent"
        plt.gcf().text(0.1, 0.95, "walls analyzed for contact events: %s" % wall_txt)
    imgFiles = dict(
        nrpp=REWARDS_IMG_FILE,
        pcm=CIRCULAR_MOTION_IMG_FILE,
        turn=TURN_RADIUS_IMG_FILE,
        rpi=REWARD_PI_IMG_FILE,
        rpid=REWARD_PI_DIFF_IMG_FILE,
        rpipd=REWARD_PI_POST_DIFF_IMG_FILE,
        rpip=REWARD_PI_POST_IMG_FILE,
        pivot=PIVOT_IMG_FILE,
        dbr=DIST_BTWN_REWARDS_FILE % "",
        dbr_no_contact=DIST_BTWN_REWARDS_FILE % "contactless_",
        max_ctr_d_no_contact=MAX_DIST_REWARDS_FILE,
    )

    if opts.turn:
        for boundary_tp in opts.turn:
            for ellipse_ref_pt in ("edge", "ctr"):
                for variant in ("", "_exp_min_yok"):
                    imgFiles[
                        "%s_%s_turn%s" % (ellipse_ref_pt, boundary_tp, variant)
                    ] = TURN_IMG_FILE % (
                        f"{ellipse_ref_pt}_",
                        boundary_tp,
                        "",
                        variant,
                    )
                    imgFiles[
                        "%s_%s_turn_duration%s" % (ellipse_ref_pt, boundary_tp, variant)
                    ] = TURN_IMG_FILE % (
                        f"{ellipse_ref_pt}_",
                        boundary_tp,
                        "_duration_",
                        variant,
                    )
                    for turn_tp in ("inside_line_", "outside_line_"):
                        imgFiles["%s%s_turn%s" % (turn_tp, boundary_tp, variant)] = (
                            TURN_IMG_FILE % (turn_tp, boundary_tp, "", variant)
                        )
                        imgFiles[
                            "%s%s_turn_duration%s" % (turn_tp, boundary_tp, variant)
                        ] = TURN_IMG_FILE % (
                            turn_tp,
                            boundary_tp,
                            "_duration_",
                            variant,
                        )
    for boundary_orientation in ("agarose_adj", "all"):
        imgFiles["r_no_contact_%s" % boundary_orientation] = (
            CONTACTLESS_RWDS_IMG_FILE % ("%s_walls" % boundary_orientation)
        )

    for bnd in (
        "agarose",
        "boundary",
        "wall",
    ):
        for ellipse_ref_pt in ("edge", "ctr"):
            imgFiles[bnd] = CONTACT_EVENT_IMG_FILE % (ellipse_ref_pt, bnd)
    if customizer.customized:
        customizer.adjust_padding_proportionally()
        adjustLegend(legend, axs, all_line_vals)
    writeImage(imgFiles[tp] % blf, format=opts.imageFormat)
    plt.close()


# - - -


# plot turn angles and run lengths
def plotRdpStats(vas, gls, tpTa=True):
    if tpTa:
        binW = 10
        bins = np.arange(-180, 180.1, binW)
        cntr, barW, barO = (bins[:-1] + bins[1:]) / 2, 0.35 * binW, 0.4 * binW
    else:
        cntr, barW, barO = np.array([0]), 0.35, 0.4
    nb, nf, flies = len(cntr), [], vas[0].flies
    plt.figure(figsize=(15 if tpTa else 4, 6))
    for f in flies:
        if gls and f == 1:
            continue
        for g in range(len(gls) if gls else 1):  # group
            byFly = []
            for va in vas:
                if gls and va.gidx != g:
                    continue
                if tpTa:
                    ta = va.rdpTA[f]
                    rdpTA = np.concatenate(ta if ta else [[]]) * 180 / np.pi
                    if len(rdpTA) >= RDP_MIN_TURNS:
                        hist, edg = np.histogram(rdpTA, bins=bins, density=True)
                        byFly.append(hist * binW)
                else:
                    mll = va.rdpAvgLL[f]
                    if not np.isnan(mll):
                        byFly.append(mll)
            nf.append(len(byFly))
            byFly = np.array(byFly)
            mci = np.array([util.meanConfInt(byFly[:, b]) for b in range(nb)]).T
            # 4 rows: see plotRewards()
            assert util.isClose(mci[0, :], np.mean(byFly, axis=0))
            bars = plt.bar(
                cntr + barO * (f + g - 0.5),
                mci[0],
                align="center",
                width=barW,
                color=FLY_COLS[f],
                edgecolor=FLY_COLS[f],
                linewidth=1,
                fill=False if g else True,
                yerr=[mci[0] - mci[1], mci[2] - mci[0]],
                ecolor=".6",
                capsize=0,
                error_kw=dict(elinewidth=2),
            )
            if gls:
                bars.set_label(gls[g])

    # labels etc.
    plt.title(va.rdpInterval)
    plt.xlabel("turn angle [degrees]" if tpTa else "")
    plt.ylabel("relative frequency" if tpTa else "average run length [pixels]")
    if not tpTa:
        plt.xlim(-2, 2)
        plt.ylim(0, plt.ylim()[1] * 1.2)
        plt.xticks([])
    tf = plt.gca().transAxes
    if gls:
        plt.legend(loc=1, prop=dict(size="medium", style="italic"))
        plt.text(
            0.9 if tpTa else 0.72,
            0.75,
            "n=%d,%d" % tuple(nf),
            size="small",
            color=".2",
            transform=tf,
        )
    else:
        for f in flies:
            yt = (0.85 if tpTa else 0.9) - f * 0.06
            plt.text(
                0.86 if tpTa else 0.6,
                yt,
                "fly %d" % (f + 1),
                color=FLY_COLS[f],
                transform=tf,
            )
            plt.text(
                0.915 if tpTa else 0.8,
                yt,
                "n=%d" % nf[f],
                size="small",
                color=".2",
                transform=tf,
            )

    writeImage(TURN_ANGLES_IMG_FILE if tpTa else RUN_LENGTHS_IMG_FILE)


# plot heatmaps
def plotHeatmaps(vas):
    if max(va.gidx for va in vas) > 0:
        return
    prob = True  # show probabilities (preferred)
    cmap = util.mplColormap()  # alternatives: inferno, gray, etc.
    usesb = False  # Seaborn heatmaps have lines for alpha < 1
    va0, alpha = vas[0], 1 if opts.bg is None else opts.bg
    trns, lin, flies = va0.trns, opts.hm == OP_LIN, va0.flies
    if P and F2T:
        trns = trns[:2]
    imgs, nc, nsc = [], len(trns), 2 if va0.ct is CT.regular else 1
    nsr, nf = 1 if va0.noyc else 3 - nsc, len(flies)
    if va0.ct is CT.regular:
        fig = plt.figure(figsize=(4 * nc, 6))
    elif va0.ct is CT.large:
        fig = plt.figure(figsize=(3.1 * nc, 6 * nsr))
    elif va0.ct is CT.large2:
        error("plotHeatmaps not yet implemented")
    gs = mpl.gridspec.GridSpec(
        2,
        nc + 1,
        wspace=0.2,
        hspace=0.2 / nsr,
        width_ratios=[1] * nc + [0.07],
        top=0.9,
        bottom=0.05,
        left=0.05,
        right=0.95,
    )
    cbar_ax = []
    for pst in (0, 1):

        def hm(va):
            return va.heatmapPost if pst else va.heatmap

        cbar_ax.append(fig.add_subplot(gs[pst, nc]))
        mpms, nfs, vmins = [], [], []
        for i, f in itertools.product(list(range(nc)), flies):
            mps, ls = [], []
            for va in vas:
                mp, l = hm(va)[f][i][:2]
                if mp is not None and np.sum(mp) > 0:
                    mps.append(mp / l if prob else mp)
                    ls.append(l)
            assert np.all(np.abs(np.diff(ls)) <= 2)  # about equal numbers of frames
            mpm = np.mean(mps, axis=0)
            mpms.append(mpm)
            nfs.append(len(mps))
            vmins.append(np.amin(mpm[mpm > 0]))
        vmin, vmax = np.amin(vmins), np.amax(mpms)
        vmin1 = 0 if lin else vmin / (vmax / vmin) ** 0.05  # .9*vmin not bad either
        for i, t in enumerate(trns):
            imgs1 = []
            gs1 = mpl.gridspec.GridSpecFromSubplotSpec(
                nsr,
                nsc,
                subplot_spec=gs[pst, i],
                wspace=0.06 if nsc > 1 else 0.0,
                hspace=0.045 if nsr > 1 else 0.0,
            )
            ttl = pcap(
                "post %s min%s"
                % (
                    util.formatFloat(opts.rpiPostBucketLenMin, 1),
                    "" if POST_SYNC is ST.fixed else " sync",
                )
                if pst
                else t.name()
            )
            for f in flies:
                mp = mpms[i * nf + f]
                mp = np.maximum(mp, vmin1)
                if f == 0:
                    ttln = "n=%d" % nfs[i * nf + f]
                img = cv2.resize(
                    util.heatmap(mp, xform=None if lin else np.log),
                    (0, 0),
                    fx=HEATMAP_DIV,
                    fy=HEATMAP_DIV,
                )
                ax = fig.add_subplot(gs1[f])
                if usesb:
                    sns.heatmap(
                        mp,
                        ax=ax,
                        alpha=alpha,
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                        cmap=cmap,
                        vmax=vmax,
                        vmin=vmin1,
                        norm=None if lin else mpl.colors.LogNorm(),
                        cbar=i == 0 and f == 0,
                        cbar_kws=(
                            None
                            if lin
                            else dict(
                                ticks=mpl.ticker.LogLocator(subs=(1.0, 3.0)),
                                format=mpl.ticker.LogFormatter(
                                    minor_thresholds=(10, 10)
                                ),
                            )
                        ),
                        cbar_ax=None if i or f else cbar_ax[pst],
                    )
                else:
                    ai = ax.imshow(
                        mp,
                        alpha=alpha,
                        cmap=cmap,
                        norm=(
                            None
                            if lin
                            else mpl.colors.LogNorm(
                                vmax=vmax,
                                vmin=vmin1,
                            )
                        ),
                        extent=[0, mp.shape[1], mp.shape[0], 0],
                    )
                    ax.set(xticks=[], yticks=[], aspect="equal")
                    ax.axis("off")
                    if i == 0 and f == 0:
                        kws = (
                            {}
                            if lin
                            else dict(
                                ticks=mpl.ticker.LogLocator(subs=(1.0, 3.0)),
                                format=mpl.ticker.LogFormatter(
                                    minor_thresholds=(10, 10)
                                ),
                            )
                        )
                        cb = ax.figure.colorbar(ai, cbar_ax[pst], ax, **kws)
                        cb.outline.set_linewidth(0)
                        cb.solids.set_alpha(1)
                        cb.solids.set_cmap(util.alphaBlend(cmap, alpha))
                xym = hm(va0)[f][i][2]
                if opts.bg is not None:  # add chamber background
                    wh = util.tupleMul(mp.shape[::-1], HEATMAP_DIV)
                    tl, br = (va0.xf.t2f(*xy) for xy in (xym, util.tupleAdd(xym, wh)))
                    ax.imshow(
                        va0.background()[tl[1] : br[1], tl[0] : br[0]],
                        extent=ax.get_xlim() + ax.get_ylim(),
                        cmap="gray",
                        vmin=0,
                        vmax=255,
                        zorder=-1,
                    )
                if f == 0:
                    plt.title(ttl, loc="left")
                if (f == 0) == (nsc == 1):
                    plt.title(ttln, loc="right", size="medium")
                if not pst and f == 0 and t.circles(f):
                    cx, cy, r = t.circles(f)[0]
                    cxy = util.tupleSub(va0.mirror(va0.xf.f2t(cx, cy)), xym)
                    cv2.circle(img, util.intR(cxy), r, COL_W if lin else COL_BK, 1)
                    ax.add_artist(
                        mpl.patches.Circle(
                            util.tupleMul(cxy, 1.0 / HEATMAP_DIV),
                            r / HEATMAP_DIV,
                            color="w" if lin else "k",
                            fill=False,
                            linewidth=0.8,
                        )
                    )
                imgs1.append(img)
            imgs.append((util.combineImgs(imgs1, nc=nsc, d=5)[0], ttl + " (%s)" % ttln))
    img = util.combineImgs(imgs, nc=nc)[0]
    writeImage(HEATMAPS_IMG_FILE % "", img)
    writeImage(HEATMAPS_IMG_FILE % 2, format=opts.imageFormat)
    oob = [util.basename(va.fn) for va in vas if va.heatmapOOB]
    if oob:
        warn("heatmaps out of bounds for %s" % util.commaAndJoin(oob))
    if False:  # for showing mean distance
        for f in flies:
            print(
                ">>> fly %d: %.3g"
                % (
                    f + 1,
                    np.mean([va.trx[f].mean_d for va in vas if not va.trx[f].bad()]),
                )
            )


def tlenVaVarForTp(va, tp):
    if tp == "combined":
        return va.t_lengths_for_plot
    elif tp == "contactless":
        return va.t_lengths_for_plot[va.t_length_contactless_mask]
    elif tp == "with_contact":
        return va.t_lengths_for_plot[~va.t_length_contactless_mask]


def tlenUpperBoundForTp(tp):
    combined, contactless, with_contact = [
        int(el) for el in opts.tlen_hist_ubound.split(",")
    ]
    return {
        "combined": combined,
        "contactless": contactless,
        "with_contact": with_contact,
    }[tp]


def tlenPlotTitleForTp(tp):
    return {
        "combined": "",
        "contactless": " \n(no contact between rewards)",
        "with_contact": " \n(only with contact between rewards)",
    }[tp]


def tlenPngFileForTp(tp):
    if tp == "combined":
        return ""
    return "_" + tp


def plot_angle_field_by_segment(vas, n_bins=10, data_type="velocity", gls=None):
    epsilon = 1e-10
    # note: no multi-group label support yet
    if gls is not None:
        gl = gls[0]
    else:
        gl = ""
    for i in range(2):  # Experimental and Control
        for j in range(len(vas[0].trns)):
            x = []
            y = []
            angles = []
            magnitudes = []
            circle = vas[0].trns[j].circles(i)[0]
            circle_ctr = vas[0].xf.f2t(circle[0], circle[1], f=vas[0].ef)
            circle_rad = circle[2]
            for va in vas:
                if not hasattr(va, "trns") or len(va.trns) == 0:
                    continue
                trj = va.trx[i]
                trn = va.trns[j]
                if trj.bad():
                    continue
                start, end = trn.start, trn.stop
                trj_x, trj_y = va.xf.f2t(
                    trj.flt["x"][start:end],
                    trj.flt["y"][start:end],
                    f=va.ef,
                )
                x.extend(trj_x[trj.walking[start:end]])
                y.extend(trj_y[trj.walking[start:end]])
                if data_type == "velocity":
                    angleTp = "vel"
                    magTp = "sp"
                elif data_type == "acceleration":
                    angleTp = "acc"
                    magTp = "accMag"
                angles.extend(
                    getattr(trj, "%sAngles" % angleTp)[start:end][
                        trj.walking[start:end]
                    ]
                )
                magnitudes.extend(
                    getattr(trj, magTp)[start:end][trj.walking[start:end]]
                )

            x_range = [np.nanmin(x), np.nanmax(x)]
            y_range = [np.nanmin(y), np.nanmax(y)]
            angles = np.array(angles)
            magnitudes = np.array(magnitudes)

            margin_x = 0.01 * (x_range[1] - x_range[0])
            margin_y = 0.01 * (y_range[1] - y_range[0])

            # Define spatial bins based on max x, y positions
            bins_x = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, n_bins)
            bins_y = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, n_bins)

            # Digitize the data to find the bin indices
            bin_idx_x = np.clip(np.digitize(x, bins_x) - 1, 0, None)
            bin_idx_y = np.clip(np.digitize(y, bins_y) - 1, 0, None)

            U = np.zeros((n_bins - 1, n_bins - 1))
            V = np.zeros((n_bins - 1, n_bins - 1))
            mean_vec_mag = np.zeros((n_bins - 1, n_bins - 1))
            avg_mag = np.zeros((n_bins - 1, n_bins - 1))

            # Use NumPy's unique to identify unique bin pairs
            unique_bins = np.unique(np.vstack([bin_idx_x, bin_idx_y]), axis=1)
            for bin_x, bin_y in unique_bins.T:
                mask = (bin_idx_x == bin_x) & (bin_idx_y == bin_y)
                angles_in_bin = angles[mask]
                if len(angles_in_bin):
                    x_avg = np.mean(np.cos(angles_in_bin))
                    y_avg = np.mean(np.sin(angles_in_bin))

                    mean_vec_mag[bin_y, bin_x] = np.sqrt(x_avg**2 + y_avg**2)
                    avg_mag[bin_y, bin_x] = np.mean(magnitudes[mask])

                    U[bin_y, bin_x] = x_avg
                    V[bin_y, bin_x] = y_avg

            mid_x = 0.5 * (bins_x[:-1] + bins_x[1:])
            mid_y = 0.5 * (bins_y[:-1] + bins_y[1:])
            X, Y = np.meshgrid(mid_x, mid_y)

            # Normalize U and V based on the magnitudes
            U /= mean_vec_mag + epsilon  # Add a small value to prevent division by zero
            V /= mean_vec_mag + epsilon

            plt.figure(figsize=(8, 10))
            plt.quiver(X, Y, U, V, avg_mag, cmap="viridis")
            plt.colorbar(label="Average %s Magnitude" % data_type.capitalize())
            plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.gca().invert_yaxis()
            circle = patches.Circle(
                circle_ctr, circle_rad, fill=False, edgecolor="red", linewidth=2
            )
            plt.gca().add_patch(circle)
            plt.title(
                "%s Trajectories, Training %i" % (("Experimental", "Control")[i], j + 1)
            )
            plt.savefig(
                "imgs/htl_%s_%s_%s_vec_fld__t%i.png"
                % (gl, ("exp", "ctl")[i], data_type[:3], j)
            )


class LgTurnPlotter:
    PATCH_PALETTE = ["tomato", "forestgreen"]
    EDGE_PALETTE = ["darkred", "darkgreen"]

    def __init__(self, vas, gls):
        """
        Initializes the LgTurnPlotter instance with necessary data for plotting large turn
        distances.

        Parameters:
        - vas (list[VideoAnalysis]): A list of VideoAnalysis instances, each containing
          trajectory data (`trx`) and a group index (`gidx`) for the flies observed.
        - gls (list[str] or None): Group labels specifying the names of each group involved in
          the analysis. If `None`, groups will not be distinguished by label.

        Attributes:
        - nf (int): The number of frames in the first video analysis instance's trajectory,
          used as a reference for processing.
        - n_bins (int): The number of bins to use when calculating histograms for turn
          distances, derived from global options.
        - bins (numpy.ndarray): An array representing the bin edges for histogram
          calculations, spanning from 0 to 60.
        - num_plot_failures (int): Counter to track the number of plots skipped due to
          insufficient data.

        """
        self.nf = len(vas[0].trx)
        self.vas = vas
        self.gls = gls
        self.n_bins = opts.lg_turn_nbins
        self.bins = np.linspace(0, 8, self.n_bins + 1)
        self.num_plot_failures = 0

    def report_plot_failures(self):
        """
        Reports the number of plot failures due to insufficient sample size after attempting
        to plot large-turn distance histograms.

        Notes:
        - This method should be called after all plotting attempts have been made to provide a
          summary of any data limitations encountered.
        - It advises on possible solutions, such as including more data or adjusting the
          `minNumLT` threshold.
        """
        if self.num_plot_failures == 0:
            return
        print(
            "\nWarning: skipped plotting of %i large-turn distance histograms"
            " due to insufficient sample size." % self.num_plot_failures
        )
        print(
            "Either include more experimental data or reduce minNumLT"
            " (current value: %i)." % opts.minNumLT
        )

    def plt_turn_dists(self, tp):
        """
        Plots histograms of turn distances for each turn captured in the video analysis data.

        Parameters:
        - tp (str): Indicates the phase of the turn being analyzed, either "start" or "end",
          to focus on the relevant part of the trajectory data.

        Notes:
        - This method generates plots for individual turns as well as comparative plots
          between different types of turns (e.g., first vs third turn).
        - It is designed to provide insights into the distribution of turn distances across
          different experimental conditions or time points.
        """
        if self.gls is None:
            gis = np.array([0 for _ in range(len(self.vas))])
        else:
            gis = np.array([v.gidx for v in self.vas])
        for i in range(len(self.vas[0].trns)):
            self.plot_turn_dist_for_trn(i, tp, gis)
            self.plot_turn_dist_for_trn(i, tp, gis, exp_only=True)
        self.plot_turn_dist_trn1_vs_trn3(gis)

    def calculate_histograms(self, fly_data):
        """
        Calculates histograms for turn distances, grouping data by experimental condition or
        fly type.

        Parameters:
        - fly_data (pandas.DataFrame): A data frame containing turn distance data and
                                       metadata for a single fly or group of flies.

        Returns:
        - (int, numpy.ndarray): A tuple containing the group index and the histogram of turn
                                distances, adjusted for bin width.

        """
        if len(self.t_idxs) > 1:
            group_idx = fly_data["trn"].iloc[0]
        else:
            if self.exp_only:
                group_idx = self.gls.index(fly_data["grp_lbl"].iloc[0])
            else:
                group_idx = fly_data["fly_tp"].iloc[0]
        hist, bin_edges = np.histogram(
            fly_data["distance"], bins=self.bins, density=True
        )
        bin_widths = np.diff(bin_edges)
        hist *= bin_widths
        return group_idx, hist

    def collect_raw_turns(self, t_idxs, gi=None, f=None, reset=False):
        """
        Collects raw turn data for specified turns, optionally filtering by group index and
        resetting the data collection.

        Parameters:
        - t_idxs (list[int]): Indices of turns to be collected.
        - gi (int, optional): Group index to filter the flies by their experimental group. If
                              `None`, no filtering is applied.
        - f (int, optional): Specifies the focus of the data collection, e.g., experimental
                             (0) or control (1) flies.
        - reset (bool, optional): If `True`, resets the current turn data storage before
                                  collecting new data.

        Notes:
        - This method is designed to aggregate turn data from multiple flies, considering
          their group and experimental condition.
        - It facilitates the detailed analysis of turn dynamics by collecting and organizing
          raw data for further processing.
        """
        if reset:
            self.turn_data = {
                "fly_id": [],
                "fly_tp": [],  # 0 for exp, 1 for ctrl
                "distance": [],
                "grp_lbl": [],
                "trn": [],
            }
        for va in self.vas:
            if va.gidx != gi:
                continue
            if len(self.turn_data["fly_id"]) == 0:
                fly_id = 0
            else:
                fly_id = self.turn_data["fly_id"][-1] + 1
            for i in t_idxs:
                ds = np.array(va.lg_turn_dists[f or 0][self.tp][i + 1])
                num_exits = len(ds)
                ds = ds[~np.isnan(ds)]
                num_turns = len(ds)
                if len(ds) < opts.minNumLT:
                    print(
                        "skipping fly",
                        f,
                        "in",
                        va.fn,
                        "-- too few turns for training",
                        i + 1,
                    )
                    continue
                if (
                    self.tp == "start"
                    and not hasattr(self, "cts_by_trn_df")
                    and not self.exp_only
                ):
                    self.cts_by_trn["fly_id"].append(fly_id)
                    self.cts_by_trn["fly_tp"].append(["exp", "yok"][f or 0])
                    self.cts_by_trn["Training"].append(i + 1)
                    self.cts_by_trn["grp_lbl"].append(
                        self.gls[gi] if self.gls is not None else None
                    )
                    self.cts_by_trn["n_exits"].append(num_exits)
                    self.cts_by_trn["n_turns"].append(num_turns)

                self.turn_data["fly_id"].extend([fly_id for _ in range(num_turns)])
                self.turn_data["fly_tp"].extend([f or 0 for _ in range(num_turns)])
                self.turn_data["distance"].extend(ds)
                self.turn_data["grp_lbl"].extend(
                    [
                        self.gls[gi] if self.gls is not None else None
                        for _ in range(num_turns)
                    ]
                )
                self.turn_data["trn"].extend([i for _ in range(num_turns)])

    def draw_dist_hist(self, t_idxs, gi=None):
        """
        Draws and saves histograms comparing the distribution of distances from the reward
        circle center during specified turns.

        This method performs a series of statistical analyses and visualizes the results in
        the form of histograms. It compares the average distance from the reward center for
        different turns or groups, calculates the statistical significance of differences
        observed, and visualizes overlap and confidence intervals.

        Parameters:
        - t_idxs (list[int]): Indices of turns to analyze. The method supports analyzing up to
                              two turns simultaneously.
        - gi (int, optional): Group index to filter the data by a specific experimental group.
                              This parameter is used when the data includes multiple groups
                              and only one group's data should be visualized.

        Notes:
        - This method handles data grouping, statistical testing (including t-tests and
          Mann-Whitney U tests), and visualization in one comprehensive procedure.
        - It dynamically adjusts the analysis based on the number of turns specified, whether
          the data pertains exclusively to experimental or control flies, and whether group
          labels are available.
        - It also incorporates error handling for unsupported turn index lengths and ensures
          data is appropriately filtered and categorized before analysis.
        - Visualization aspects include plotting histograms with error bars, indicating
          statistical significance directly on the plots, and generating a descriptive title
          and legend tailored to the specifics of the analysis being performed.
        - The method concludes by saving the generated figure to a designated directory with a
          filename reflecting the analysis details.

        Raises:
        - ValueError: If more than two turn indices are provided, indicating an attempt to
                      analyze more than is supported by the method's design.

        Example Usage:
        ```python
        plotter = LgTurnPlotter(vas=my_vas, gls=my_gls)
        plotter.draw_dist_hist(t_idxs=[0, 1], gi=0)
        ```

        This example would generate histograms comparing the first and second turns for flies
        in the first group, analyzing their distance from the reward circle center and saving
        the resulting plot.
        """
        if len(t_idxs) > 2:
            error("t_idx max allowed length: 2")
        self.turn_data = pd.DataFrame(self.turn_data)
        self.t_idxs = t_idxs
        if len(t_idxs) > 1:
            fixed_effect_var = "trn"
        else:
            if self.exp_only:
                fixed_effect_var = "grp_lbl"
            else:
                fixed_effect_var = "fly_tp"
        if self.gls is not None:
            self.turn_data["grp_lbl"] = pd.Categorical(self.turn_data["grp_lbl"])
        if len(t_idxs) > 1:
            self.labels = ["T%i" % (idx + 1) for idx in t_idxs]
        else:
            if self.exp_only:
                self.labels = self.gls
            else:
                self.labels = ["exp", "ctrl"][: self.nf]
        if len(self.labels) == 2 and (
            "t" in opts.rc_turn_tests or "mwu" in opts.rc_turn_tests
        ):
            if len(self.t_idxs) > 1:
                distinct_trn_counts = self.turn_data.groupby("fly_id")["trn"].nunique()
                self.turn_data = self.turn_data[
                    self.turn_data["fly_id"].map(distinct_trn_counts) >= 2
                ]
            average_distance_by_fly = self.turn_data.groupby(
                ["fly_id", "trn"], as_index=False
            ).agg(
                average_distance=("distance", "mean"),
                fly_tp=("fly_tp", "first"),
                grp_lbl=("grp_lbl", "first"),
            )
            if len(t_idxs) > 1:
                grp1_key = t_idxs[0]
                grp2_key = t_idxs[1]
            else:
                if self.exp_only:
                    grp1_key = self.gls[0]
                    grp2_key = self.gls[1]
                else:
                    grp1_key = 0
                    grp2_key = 1
            dist_grps = []
            samp_sizes = []
            for key in (grp1_key, grp2_key):
                dist_grps.append(
                    average_distance_by_fly[
                        average_distance_by_fly[fixed_effect_var] == key
                    ]
                )
                samp_sizes.append(len(dist_grps[-1].index))
            samp_sizes = np.array(samp_sizes)
            if not np.all(samp_sizes > 1):
                self.num_plot_failures += 1
                return
            if "t" in opts.rc_turn_tests:
                if len(t_idxs) > 1:
                    pivoted_data = average_distance_by_fly.pivot(
                        index="fly_id", columns="trn", values="average_distance"
                    )
                    pivoted_data = pivoted_data.dropna()
                    _, avg_dist_pvalue = st.ttest_rel(
                        pivoted_data[t_idxs[0]], pivoted_data[t_idxs[1]]
                    )
                else:
                    _, avg_dist_pvalue = st.ttest_ind(
                        dist_grps[0]["average_distance"],
                        dist_grps[1]["average_distance"],
                    )
            if "mwu" in opts.rc_turn_tests:
                mw_result = st.mannwhitneyu(
                    dist_grps[0]["average_distance"],
                    dist_grps[1]["average_distance"],
                )

        normalized_hist_data = [[] for _ in range(len(self.labels))]
        bin_centers = (self.bins[1:] + self.bins[:-1]) / 2
        hists_by_fly = (
            self.turn_data.groupby(["fly_id", "trn"])
            .apply(self.calculate_histograms)
            .tolist()
        )
        if len(t_idxs) > 1:
            for group_idx, hist in hists_by_fly:
                if group_idx in self.t_idxs:
                    pos = self.t_idxs.index(group_idx)
                    normalized_hist_data[pos].append(hist)
        else:
            group_count = -1
            added_groups = []
            for group_idx, hist in hists_by_fly:
                if group_idx not in added_groups:
                    added_groups.append(group_idx)
                    group_count += 1
                normalized_hist_data[added_groups.index(group_idx)].append(hist)
        normalized_hist_data = [np.array(hd) for hd in normalized_hist_data]
        mean_distributions = [np.mean(hd, axis=0) for hd in normalized_hist_data]
        cis = []
        for i, hd in enumerate(normalized_hist_data):
            st_sem = st.sem(hd)
            zero_scale_indices = st_sem == 0
            conf_int_lo_final = np.zeros(st_sem.shape)
            conf_int_hi_final = np.zeros(st_sem.shape)
            ci_l, ci_h = st.t.interval(
                0.95,
                len(hd) - 1,
                loc=mean_distributions[i][~zero_scale_indices],
                scale=st.sem(hd)[~zero_scale_indices],
            )
            conf_int_lo_final[~zero_scale_indices] = ci_l
            conf_int_hi_final[~zero_scale_indices] = ci_h
            cis.append((conf_int_lo_final, conf_int_hi_final))
        p_values = []
        if len(self.labels) > 1 and np.all(samp_sizes > 0):
            if len(self.t_idxs) > 1:
                test_func = ttest_rel
            else:
                test_func = ttest_ind
            for i in range(len(bin_centers)):
                p_val = test_func(
                    normalized_hist_data[0][:, i], normalized_hist_data[1][:, i]
                )[1]
                p_values.append(p_val)
            p_values = np.array(p_values)
            numeric_ps = ~np.isnan(p_values)
            if np.count_nonzero(numeric_ps):
                p_values[numeric_ps] = multipletests(
                    p_values[numeric_ps], method="bonferroni"
                )[1]
            overlap = np.minimum(mean_distributions[0], mean_distributions[1])
        fig = plt.figure(figsize=(9, 6))
        colors = (("tomato", "darkred"), ("forestgreen", "darkgreen"))[
            : len(self.labels)
        ]
        for i, colorset in enumerate(colors):
            plt.bar(
                self.bins[:-1],
                mean_distributions[i],
                width=np.diff(self.bins),
                align="edge",
                color=colorset[0],
                alpha=0.5,
                label=f"{self.labels[i]} (n={samp_sizes[i]})",
                edgecolor=colorset[1],
                linewidth=0.5,
            )
            plt.errorbar(
                bin_centers,
                mean_distributions[i],
                yerr=(
                    np.maximum(mean_distributions[i] - cis[i][0], 0),
                    np.maximum(cis[i][1] - mean_distributions[i], 0),
                ),
                fmt="none",
                ecolor=colorset[0],
                alpha=0.3,
                capsize=2,
            )
        if len(self.labels) > 1:
            plt.bar(
                self.bins[:-1],
                overlap,
                width=np.diff(self.bins),
                align="edge",
                color=(0.525, 0.647, 0.718),
                alpha=1.0,
                label="overlap",
                edgecolor="mediumblue",
                linewidth=0.5,
            )
        legend = plt.legend(prop={"style": "italic"})
        plt.xlabel("Distance from chamber center (mm)")
        plt.ylabel("Frequency")
        plt.xlim((0, 8))
        current_ylim = plt.ylim()
        plt.ylim((current_ylim[0], current_ylim[1] * 2))
        star_spacing = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
        for i, p_val in enumerate(p_values):
            plt.text(
                bin_centers[i],
                max(mean_distributions[0][i], mean_distributions[1][i]) + star_spacing,
                util.p2stars(p_val),
                ha="center",
                size=customizer.in_plot_font_size,
            )
        title = "Mean normalized distribution"
        descriptors = ["%s of turn" % self.tp]

        if len(t_idxs) == 1:
            descriptors.append("T%s" % str(t_idxs[0] + 1))
        else:
            descriptors.append("training %i vs %i" % (t_idxs[0] + 1, t_idxs[1] + 1))
        if fixed_effect_var != "grp_lbl" and self.gls is not None:
            descriptors.append("Group: %s" % self.gls[gi])
        if self.exp_only:
            descriptors.append("experimental flies")
        elif not self.exp_only and len(t_idxs) > 1:
            descriptors.append("control flies")
        if len(descriptors) > 0:
            title += ",\n" + (", ".join(descriptors))
        # keep variable for title, but don't display it for now.
        fig.tight_layout()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer).transformed(
            plt.gca().transAxes.inverted()
        )
        if not self.exp_only or len(self.labels) == 2:
            ttest_txt = ""
            if "t" in opts.rc_turn_tests:
                ttest_txt += "\n%st-test, mean distance from chamber center: %s" % (
                    ("Welch's " if len(self.t_idxs) == 1 else "paired "),
                    util.p2stars(avg_dist_pvalue),
                )
            if "mwu" in opts.rc_turn_tests:
                ttest_txt += "\nMann-Whitney U test: %s" % (
                    util.p2stars(mw_result.pvalue)
                )

        plt.gca().text(
            0.98,
            bbox.y0 - 0.02,
            ttest_txt,
            verticalalignment="top",
            horizontalalignment="right",
            transform=plt.gca().transAxes,
            fontsize=customizer.in_plot_font_size,
        )
        suffix = ""
        if self.exp_only:
            suffix += "_exp"
        if not self.exp_only and len(t_idxs) > 1:
            suffix += "_ctrl"
        if not self.exp_only or len(t_idxs) > 1:
            suffix += ("_%s" % self.gls[gi]) if self.gls is not None else ""
        if len(t_idxs) > 1:
            suffix += "_T1vsT3"
        save_path = "turn_%s_dist_from_rwd_ctr" % self.tp
        if len(t_idxs) == 1:
            save_path += "_T%i" % (t_idxs[0] + 1)
        writeImage(
            os.path.join(
                "imgs",
                util.get_valid_filename("%s%s.png" % (save_path, suffix)),
            ),
            format=opts.imageFormat,
        )
        plt.close("all")

    def plot_turn_dist_trn1_vs_trn3(self, gis):
        """
        Plots the comparison of turn distance histograms between the first and third turns
        across experimental groups.

        This method orchestrates the collection and plotting of turn distance data
        specifically for the first and third turns. It supports analyzing the data separately
        for experimental and control flies within each specified group.

        Parameters:
        - gis (numpy.ndarray): An array of group indices for the flies. This array helps
                               determine the number of groups to iterate over during the
                               plotting process.

        Notes:
        - The method resets the turn data storage before each new collection to ensure data
          integrity.
        - It leverages the `collect_raw_turns` method for data aggregation and the
          `draw_dist_hist` method for histogram plotting, performing these operations for both
          experimental only and combined (experimental and control) scenarios.
        """
        self.turn_data = {}
        range_len = len(self.gls) if ~np.all(gis == 0) else 1
        for exp_only in (True, False):
            self.exp_only = exp_only
            for i in range(range_len):
                self.collect_raw_turns((2, 0), i, reset=True, f=int(not exp_only))
                self.draw_dist_hist(t_idxs=(2, 0), gi=i)

    def _initialize_cts_by_trn(self):
        """
        Initializes the container for counts by training session data with empty lists for
        various metrics.

        This method sets up a dictionary to track metrics related to fly behavior and outcomes
        across different training sessions. It includes the fly ID, type, specific training
        session, group label, number of exits, and number of turns made. This structure
        supports the organization and analysis of data based on the training conditions
        experienced by the flies.

        The initialization facilitates subsequent operations that populate these lists with
        actual data, enabling detailed analysis and comparison of fly behavior across
        different training sessions.
        """
        self.cts_by_trn = {
            "fly_id": [],
            "fly_tp": [],
            "Training": [],
            "grp_lbl": [],
            "n_exits": [],
            "n_turns": [],
        }

    @staticmethod
    def annotate_sample_size(ax, dataset, hue_label, keys):
        """
        Annotates a plot with the sample sizes for the groups being compared.

        This static method adds a text annotation to a plot indicating the number of
        observations (sample size) for each group involved in the comparison. It's designed to
        enhance plot interpretability by providing context on the dataset size.

        Parameters:
        - ax (matplotlib.axes.Axes): The matplotlib Axes object to annotate.
        - dataset (pandas.DataFrame): The dataset from which sample sizes are calculated.
        - hue_label (str): The column name in `dataset` used to distinguish between groups.
        - keys (list[str]): The labels of the groups to display sample sizes for. Supports
          annotating one or two groups.

        Notes:
        - The method assumes `dataset` contains at least one group identified by `hue_label`
          and `keys`.
        - It places the annotation in the upper left corner of the plot by default, with a
          small margin from the edges.
        """
        group1_size = len(dataset[dataset[hue_label] == keys[0]])
        sample_size_txt = f"{keys[0]} (n={group1_size})"
        if len(keys) > 1:
            group2_size = len(dataset[dataset[hue_label] == keys[1]])
            sample_size_txt += f" vs {keys[1]} (n={group2_size})"
        else:
            group2_size = 0

        margin = 0.015
        ax.text(
            margin,
            1 - margin,
            sample_size_txt,
            fontsize=customizer.in_plot_font_size,
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
        )

    @staticmethod
    def style_swarmplot_edges(swarm, custom_palette, edge_palette):
        """
        Styles the edges of each point in a seaborn swarm plot according to a custom palette.

        This method iterates over the collections in a swarm plot and sets the edge colors of
        each point based on a matching color from the custom palette. If a matching color is
        found in the custom palette, the corresponding edge color from `edge_palette` is
        applied; otherwise, the edge color is set to 'none'.

        Parameters:
        - swarm (sns.categorical._SwarmPlotter): The swarm plotter instance containing the
          plotted data points.
        - custom_palette (dict): A dictionary mapping group keys to colors, used to determine
          the face color of each point in the plot.
        - edge_palette (dict): A dictionary mapping the same group keys to edge colors, used
          to style the edges of the points.

        Notes:
        - This method is particularly useful for distinguishing between groups in a swarm plot
          by using different edge colors, enhancing the plot's readability and aesthetic
          appeal.
        """
        for path_collection in swarm.collections:
            colors = path_collection.get_facecolors()
            edge_colors = []

            for color in colors:
                matching_key = next(
                    (
                        k
                        for k, v in custom_palette.items()
                        if np.allclose(color[:3], mcolors.to_rgb(v))
                    ),
                    None,
                )
                if matching_key:
                    edge_color = edge_palette[matching_key]
                    edge_colors.append(edge_color)
                else:
                    edge_colors.append("none")

            path_collection.set_edgecolors(edge_colors)

    @staticmethod
    def adjust_barplot_colors(ax, custom_palette, edge_palette):
        """
        Adjusts the edge colors of bars in a matplotlib bar plot to match a custom palette.

        After plotting, this method matches each bar's face color to the closest color in the
        custom palette and sets the bar's edge color accordingly from the `edge_palette`. It
        also adjusts the colors of associated lines (e.g., error bars) to match the edge
        colors.

        Parameters:
        - ax (matplotlib.axes.Axes): The Axes object containing the bar plot.
        - custom_palette (dict): A dictionary mapping categories to face colors for the bars.
        - edge_palette (dict): A dictionary mapping the same categories to edge colors for
                               styling the bars' edges.

        Notes:
        - The method ensures that both the bars and their corresponding error bars (or other
          line markers) are styled consistently according to the specified palettes.
        """
        bars_colors = [mcolors.to_rgba(v) for v in custom_palette.values()]
        line_index = 0
        line_range = 1 if len(ax.patches) == len(ax.lines) else 3
        for patch in ax.patches:
            face_color = patch.get_facecolor()

            closest_color = min(
                bars_colors,
                key=lambda col: np.linalg.norm(np.array(face_color) - np.array(col)),
            )
            matching_key = next(
                k
                for k, v in custom_palette.items()
                if np.allclose(closest_color, mcolors.to_rgba(v))
            )

            if matching_key:
                edge_color = edge_palette[matching_key]
                patch.set_edgecolor(edge_color)

            for _ in range(line_range):
                line = ax.lines[line_index]
                line.set_color(edge_color)
                line.set_mfc(edge_color)
                line.set_mec(edge_color)
                line_index += 1

    @staticmethod
    def annotate_ttest_results(ax, dataset, hue_label, keys):
        """
        Annotates a plot with the results of independent t-tests comparing two groups across
        different trainings.

        For each unique training session in the dataset, this method performs an independent
        t-test between the two specified groups (identified by `keys`) and annotates the plot
        with the p-value significance level (converted to asterisks) above the compared
        groups.

        Parameters:
        - ax (matplotlib.axes.Axes): The Axes object to annotate with t-test results.
        - dataset (pandas.DataFrame): The data containing the groups, training sessions, and
                                      measurements to be compared.
        - hue_label (str): The column name in `dataset` used to distinguish between the groups
                           for comparison.
        - keys (list[str]): The labels of the two groups to compare in the t-test.

        Notes:
        - This method is designed to visually summarize the statistical significance of
          differences between two groups across various training sessions directly on the
          plot.
        - It dynamically calculates the appropriate y-axis position for each annotation to
          avoid overlapping with the plotted data while ensuring visibility.
        """
        trainings = dataset["Training"].unique()
        for training in trainings:
            data_subsets = [
                dataset[
                    (dataset["Training"] == training) & (dataset[hue_label] == key)
                ]["proportion"].values
                for key in keys
            ]
            if len(data_subsets) != 2:
                continue

            p_value = ttest_ind(data_subsets[0], data_subsets[1])[1]

            y_max = max(
                dataset[dataset["Training"] == training]["proportion"].max(),
                ax.get_ylim()[1],
            )
            y_bracket = y_max + 0.02
            x_values = [
                i for i, x in enumerate(dataset["Training"].unique()) if x == training
            ]

            bracket_height = 0.015
            bracket_width = 0.45
            x_start = x_values[0] - bracket_width / 2
            x_end = x_values[-1] + bracket_width / 2

            ax.plot([x_start, x_end], [y_bracket, y_bracket], color="black")

            ax.plot(
                [x_start, x_start],
                [y_bracket - bracket_height, y_bracket],
                color="black",
            )

            ax.plot(
                [x_end, x_end], [y_bracket - bracket_height, y_bracket], color="black"
            )

            significance = util.p2stars(p_value)
            ax.text(
                sum(x_values) / len(x_values),
                y_bracket + 0.01,
                significance,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    @staticmethod
    def adjust_legend(ax):
        """
        Adjusts the legend of a plot to include only the first half of the handles and labels.

        This method can be used when a plot automatically generates a redundant or overly
        detailed legend. It simplifies the legend by keeping only the first half of the
        entries, which can be particularly useful in plots where the same categories are
        represented multiple times.

        Parameters:
        - ax (matplotlib.axes.Axes): The Axes object whose legend is to be adjusted.

        Notes:
        - This method is especially useful in seaborn plots where the legend includes entries
          for both the main plot and any additional elements like confidence intervals or
          subsets, leading to duplication.
        """
        handles, labels = ax.get_legend_handles_labels()
        l = len(handles) // 2
        ax.legend(handles[:l], labels[:l], loc="upper right")

    @staticmethod
    def save_figure(fig, filename, directory="imgs"):
        """
        Saves a figure to a specified directory with a given filename.

        If the target directory does not exist, it is created. The figure is then saved using
        the full path constructed from the directory and filename.

        Parameters:
        - fig (matplotlib.figure.Figure): The Figure object to save.
        - filename (str): The name of the file to save the figure as. Should include the file extension, e.g., '.png'.
        - directory (str, optional): The directory path where the figure should be saved. Defaults to 'imgs'.

        Notes:
        - This method streamlines the process of saving figures by handling directory creation
          and using a consistent saving routine.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        writeImage(os.path.join(directory, filename), format=opts.imageFormat)

    def plot_turn_dist_for_trn(self, t_idx, tp, gis=None, exp_only=False):
        """
        Plots the distribution of turn distances for specified training indices and group
        indices.

        This method can plot the distribution for either the start ('start') or end ('end') of
        turns, and it can focus on experimental flies only or include control flies for
        comparison. The method supports analyzing multiple groups and provides options for
        detailed comparison across training sessions.

        Parameters:
        - t_idx (int or tuple): Training index or indices to plot.
        - tp (str): Specifies whether to analyze the 'start' or 'end' of turns.
        - gis (numpy.array, optional): Array of group indices to include in the analysis. If
                                       not provided or if all values are 0, the analysis
                                       defaults to a single group.
        - exp_only (bool): If True, plots only experimental flies, comparing across multiple
                           groups. If False, compares experimental and control flies within
                           each group.

        Notes:
        - This method dynamically adjusts its behavior based on the provided arguments to
          offer flexible analysis of turn distributions across different conditions and
          groups.
        - It initializes necessary data structures and performs calculations to plot
          histograms and scatter plots summarizing the proportion of large turns to exits.
        """
        if not hasattr(self, "cts_by_trn"):
            self._initialize_cts_by_trn()
        self.turn_data = {}
        self.exp_only = exp_only
        self.tp = tp
        if not hasattr(self, "scatter_drawn"):
            self.scatter_drawn = set()
        range_len = len(self.gls) if ~np.all(gis == 0) else 1
        if exp_only:
            if self.gls is None:
                return
            for i in range(range_len):
                self.collect_raw_turns((t_idx,), i, reset=i == 0)
            self.draw_dist_hist(t_idxs=(t_idx,))
            if t_idx + 1 == len(self.vas[0].trns):
                if exp_only not in self.scatter_drawn:
                    self.draw_turn_ratio_scatter()
                    self.scatter_drawn.add(exp_only)
        else:
            for i in range(range_len):
                for j in range(len(self.vas[0].trx)):
                    self.collect_raw_turns((t_idx,), gi=i, f=j, reset=j == 0)
                self.draw_dist_hist(t_idxs=(t_idx,), gi=i)
            if t_idx + 1 == len(self.vas[0].trns):
                self.cts_by_trn_df = pd.DataFrame(self.cts_by_trn)
                self.cts_by_trn_df["proportion"] = (
                    self.cts_by_trn_df["n_turns"] / self.cts_by_trn_df["n_exits"]
                )
                self._initialize_cts_by_trn()
                if exp_only not in self.scatter_drawn:
                    for i in range(range_len):
                        self.draw_turn_ratio_scatter(gi=i)
                    self.scatter_drawn.add(exp_only)

    def draw_turn_ratio_scatter(self, gi=None):
        """
        Draws a scatter plot illustrating the ratio of large turns to exits for flies.

        This method plots data for a specified group index (if provided) and adjusts the plot
        based on whether only experimental flies are included. It annotates the plot with
        statistical test results and adjusts aesthetics for clarity and readability.

        Parameters:
        - gi (int, optional): Group index to filter the dataset for a specific group's data
                              before plotting. If None, the plot includes all available
                              groups.

        Notes:
        - The method applies custom color palettes for bars and points, adjusts edge colors
          for clarity, and annotates the plot with statistical significance markers.
        - It dynamically adjusts the plot's title and y-axis limits based on the data being
          visualized and the groups included in the analysis.
        - This method is designed to visually convey the proportion of large turns relative to
          exits, facilitating comparisons across conditions or groups.
        """
        fig, ax = plt.subplots()
        dataset = self.cts_by_trn_df
        if gi is not None and self.gls is not None:
            dataset = dataset[dataset["grp_lbl"] == self.gls[gi]]

        hue_label = "grp_lbl" if self.exp_only else "fly_tp"
        fig.set_size_inches(9, 7)
        if self.exp_only:
            keys = self.gls
        else:
            keys = dataset[hue_label].unique()

        LgTurnPlotter.annotate_sample_size(ax, dataset, hue_label, keys)

        custom_palette = dict(zip(keys, self.PATCH_PALETTE))
        edge_palette = dict(zip(keys, self.EDGE_PALETTE))

        sns.barplot(
            x="Training",
            hue=hue_label,
            y="proportion",
            data=dataset,
            palette=custom_palette,
            ax=ax,
            alpha=0.5,
            capsize=0.06,
        )

        swarm = sns.swarmplot(
            x="Training",
            hue=hue_label,
            y="proportion",
            data=dataset,
            dodge=True,
            size=5,
            zorder=3,
            palette=custom_palette,
            linewidth=2,
            ax=ax,
        )

        LgTurnPlotter.style_swarmplot_edges(swarm, custom_palette, edge_palette)
        LgTurnPlotter.adjust_barplot_colors(ax, custom_palette, edge_palette)
        LgTurnPlotter.annotate_ttest_results(ax, dataset, hue_label, keys)

        plt.ylabel("Proportion of Large Turns to Exits")
        if gi is not None and self.gls is not None:
            title_suffix = f", {self.gls[gi]}"
        else:
            title_suffix = ""
        plt.title("Turn Ratios" + title_suffix)
        current_ylim = plt.ylim()
        plt.ylim((current_ylim[0], current_ylim[1] * 1.2))

        LgTurnPlotter.adjust_legend(ax)

        plt.tight_layout()

        if self.exp_only:
            suffix = "_exp"
        else:
            suffix = (
                "_%s" % self.gls[gi] if gi is not None and self.gls is not None else ""
            )
        save_path = "turn_to_exit_ratio"
        LgTurnPlotter.save_figure(
            fig, util.get_valid_filename("%s%s.png" % (save_path, suffix))
        )


def plotTlenDist(vas, gis, gls, tp):
    fcs = [(0.0, 0.0, 1.0, 0.5), (0, 0.5, 0.5, 0.5)]
    n_bins = opts.tlen_nbins
    upper_bound = tlenUpperBoundForTp(tp)
    bin_width = upper_bound / opts.tlen_nbins
    upper_bound += bin_width
    n_bins += 1
    hist_bins = np.linspace(0, upper_bound, num=n_bins + 1)
    cts = [
        np.zeros((n_bins, len([va for va in vas if va.gidx == gi]))) for gi in set(gis)
    ]
    plt.figure(1, figsize=(9, 6))
    highest_cts = np.zeros((n_bins))
    combined_t_lengths = []
    for gi in set(gis):
        combined_t_lengths.append([])
        va_count = 0
        for i, va in enumerate(vas):
            if va.gidx == gi:
                data = tlenVaVarForTp(va, tp)
                if len(data) < opts.tlen_mincts:
                    continue
                plt.figure(2)
                cts_to_add, _, _ = plt.hist(
                    np.clip(data, hist_bins[0], hist_bins[-1]),
                    bins=hist_bins,
                    density=True,
                )
                plt.close(2)
                cts[gi][:, va_count] = cts_to_add
                va_count += 1

                combined_t_lengths[-1].extend(va.t_lengths_for_plot)
        plt.figure(1)
        normalized_means = np.nanmean(cts[gi], axis=1)
        normalized_means = normalized_means / np.sum(normalized_means)
        plt.figure(1)
        if gls is not None:
            label = gls[gi]
        else:
            label = None
        cts_for_grp, _, _ = plt.hist(
            hist_bins[:-1],
            weights=normalized_means,
            bins=hist_bins,
            density=False,
            label=label,
            linewidth=1,
            edgecolor="black",
            fc=fcs[gi],
        )
        highest_cts = np.where(cts_for_grp > highest_cts, cts_for_grp, highest_cts)

    max_cts = max(highest_cts)
    if len(set(gis)) > 1:
        p_vals_by_bin = np.zeros((n_bins))
        for i in range(n_bins):
            p_vals_by_bin[i] = ttest_ind(cts[0][i, :], cts[1][i, :])[1]
            star_txt = util.p2stars(p_vals_by_bin[i])
            if star_txt != "ns" and star_txt is not None:
                util.pltText(
                    0.5 * (hist_bins[i] + hist_bins[i + 1]),
                    min(highest_cts[i] + 0.05 * max_cts, max_cts),
                    star_txt,
                    fontsize="x-small",
                    ha="center",
                )

    plt.title("Trajectory length distribution, T3%s" % (tlenPlotTitleForTp(tp)))
    plt.xlabel("Trajectory length (px)")
    plt.ylabel("Proportion of trajectories")
    if gls is not None:
        plt.legend()
    writeImage(
        DIST_BTWN_REWARDS_HIST_FILE % tlenPngFileForTp(tp), format=opts.imageFormat
    )
    plt.close()


# "post analyze" the given VideoAnalysis objects
def postAnalyze(vas):
    if len(vas) <= 1:
        return

    run_turn_prob = (
        opts.contact_geometry == "horizontal" and opts.turn_prob_by_dist
    ) or (opts.contact_geometry == "circular" and opts.outside_circle_radii)

    print("\n\n=== all video analysis (%d videos) ===" % len(vas))
    print("\ntotal rewards training: %d" % sum(va.totalTrainingNOn for va in vas))

    n, va = opts.numRewardsCompare, vas[0]
    gis = np.array([va.gidx for va in vas])
    gls = opts.groupLabels and opts.groupLabels.split("|")
    ng = gis.max() + 1
    if gls and len(gls) != ng:
        error("numbers of groups and group labels differ")

    if opts.wall:
        plotTlenDist(vas, gis, gls, "contactless")
        plotTlenDist(vas, gis, gls, "with_contact")
    if opts.cTurnAnlyz:
        lg_turn_plotter = LgTurnPlotter(vas=vas, gls=gls)
        lg_turn_plotter.plt_turn_dists("start")
        lg_turn_plotter.plt_turn_dists("end")
        lg_turn_plotter.report_plot_failures()
    if opts.outside_circle_radii:
        # Integrate the new OutsideCircleDurationPlotter here
        outside_circle_plotter = OutsideCircleDurationPlotter(
            vas=vas, opts=opts, gls=gls
        )
        outside_circle_plotter.plot_distributions(normalized=opts.outside_circle_norm)
    if va.circle:
        plotTlenDist(vas, gis, gls, "combined")
    if opts.turn_dir:
        turn_dir_plotter = TurnDirectionalityPlotter(vas, gls, customizer, opts)
        turn_dir_plotter.plot_turn_directionality()

    if run_turn_prob:
        turn_prob_plotter = TurnProbabilityByDistancePlotter(vas, gls, opts, customizer)
        turn_prob_plotter.average_turn_probabilities()
        turn_prob_plotter.plot_turn_probabilities()

    if not (va.circle or va.choice):
        return

    tcs = (
        ("bysb2",)
        if va.choice
        else (
            "atb",
            "atb-c",
            "adb",
            "adb-c",
            "nr",
            "nr-c",
            "ppi",
            "rpi",
            "rpip",
            "nrp-c",
            "nrpp-c",
            "rdp",
            "bysb2",
            "spd",
            "stp",
            "rpm",
            "dbr",
        )
    )
    if opts.circle:
        tcs += (
            "pcm",
            "turn",
            "pivot",
        )
    if not va.noyc and not va.choice:
        tcs += ("rpid", "rpipd")
    for opt in ("wall", "agarose", "boundary", "turn"):
        if getattr(opts, opt):
            if opt == "turn":
                for boundary_tp in getattr(opts, opt):
                    for variant in ("", "_exp_min_yok"):
                        for turn_tp in ("inside_line", "outside_line"):
                            tcs += ("%s_%s_turn%s" % (turn_tp, boundary_tp, variant),)
                            tcs += (
                                "%s_%s_turn_duration%s"
                                % (turn_tp, boundary_tp, variant),
                            )
                    ellipse_ref_pts = ["edge"] + (
                        ["ctr"] if boundary_tp != "wall" else []
                    )
                    for ref_pt in ellipse_ref_pts:
                        tcs += (
                            "%s_%s_turn_duration%s" % (ref_pt, boundary_tp, variant),
                        )
                        tcs += ("%s_%s_turn%s" % (ref_pt, boundary_tp, variant),)
            else:
                tcs += (opt,)
            if opt == "wall":
                tcs += ("dbr_no_contact", "max_ctr_d_no_contact")
                for boundary_orientation in ("all", "agarose_adj"):
                    tcs += ("r_no_contact_%s" % boundary_orientation,)

    saved_bottom = saved_top = None
    for tc in tcs:
        tp, calc = typeCalc(tc)
        hdr = headerForType(va, tp, calc)
        if hdr is None:
            continue
        print(hdr)
        cns = columnNamesForType(va, tp, calc, n)
        nf = len(fliesForType(va, tp, calc))
        if cns:
            nb = int(len(cns) / nf)
        trns = trnsForType(va, tp)
        if not trns:
            print("skipped")
            continue
        a = np.array([vaVarForType(va, tp, calc) for va in vas])
        a = a.reshape((len(vas), len(trns), -1))
        if tp in ("rpid", "rpipd") or "exp_min_yok" in tp:
            raw_perf = a.copy()
            n_videos = len(vas)
            n_trains = len(trns)
            n_flies = len(va.flies)
            logging.debug("raw perf shape: %s", raw_perf.shape)
            logging.debug("n flies: %d", n_flies)
            nb = raw_perf.shape[2] // n_flies
            raw_4 = raw_perf.reshape((n_videos, n_trains, n_flies, nb))
            a = np.array(
                [
                    np.subtract(
                        a[i, :, 0 : int(a.shape[2] / 2)],
                        a[i, :, int(a.shape[2] / 2) :],
                    )
                    for i in range(a.shape[0])
                ]
            )

            # if the user asked for best/worst SLI, pick them once during TRAINING (rpid)
            selected_bottom = selected_top = None
            if opts.best_worst_sli:
                if tp == "rpid":
                    # *training* period: actually compute the extremes
                    sli_ser = compute_sli_per_fly(raw_4, opts.best_worst_trn - 1)
                    selected_bottom, selected_top = select_extremes(
                        sli_ser, fraction=opts.best_worst_fraction
                    )
                    # save them so we can reuse for the post period
                    saved_bottom, saved_top = selected_bottom, selected_top
                elif tp == "rpipd" and saved_bottom is not None:
                    # *post* period: reuse the exact same flies we picked above
                    selected_bottom, selected_top = saved_bottom, saved_top

            # now call the plotting function for either training or post
            if tp in ("rpid", "rpipd") and selected_bottom is not None:
                bl, _ = bucketLenForType(tp)
                logging.debug("raw 4 shape: %s", raw_4.shape)
                if tp == "rpid":
                    logging.debug("SLI series:\n%s", sli_ser)
                plot_sli_extremes(
                    perf4=raw_4,
                    trns=trns,
                    bucket_len_minutes=bl,
                    selected_bottom=selected_bottom,
                    selected_top=selected_top,
                    gis=gis,
                    gls=gls,
                    training_idx=opts.best_worst_trn - 1,
                    fraction=opts.best_worst_fraction,
                    tp=tp,
                    n_nonpost_buckets=(
                        va.rpiNumNonPostBuckets
                        if tp.startswith("rpip")
                        else va.numNonPostBuckets
                    ),
                    outdir="imgs",
                    title=f"SLI extremes ({'training' if tp=='rpid' else 'post'} {opts.best_worst_trn})",
                )
                logging.debug("  bottom flies: %s", selected_bottom)
                logging.debug("     top flies: %s", selected_top)

        a_orig = a.copy()

        if (
            len(va.flies) > 1
            and (
                tp
                in (
                    "rpi",
                    "rpip",
                    "dbr",
                    "dbr_no_contact",
                    "agarose",
                    "boundary",
                    "wall",
                    "max_ctr_d_no_contact",
                )
                or "r_no_contact_" in tp
                or "_turn" in tp
            )
            and not "exp_min_yok" in tp
        ):
            a = propagate_nans(a)
        # a's dimensions: video, training, bucket or fly x bucket
        assert cns is None or a.shape[2] == len(cns)
        checkValues(vas, tp, calc, a)
        if tp == "ppi":
            for i, t in enumerate(trns):
                ttest_1samp(a[:, i, 0], 0, "%s %s" % (t.name(), cns[0]))
        elif tp in ("rpi", "rpid"):
            a_for_ttest = a_orig if tp == "rpi" else a
            for i, t in enumerate(trns):
                if t.hasSymCtrl():
                    for j, cn in enumerate(cns or []):
                        ttest_1samp(a_for_ttest[:, i, j], 0, "%s %s" % (t.name(), cn))
            plotRewards(va, tp, a, trns, gis, gls, vas)
            if len(trns) > 1 and all(t.hasSymCtrl() for t in trns[:2]):
                test_hdr = "first sync bucket, training 1 vs. 2"
                if tp == "rpid":
                    test_hdr += " (exp - yok)"
                ttest_rel(a_for_ttest[:, 0, 0], a_for_ttest[:, 1, 0], test_hdr)
            for i, t in enumerate(trns):
                lb = nb - 1
                if nf == 1 and not t.hasSymCtrl() or lb == 0:
                    continue
                while True:
                    ab = [
                        (
                            a_for_ttest[:, i, b]
                            if t.hasSymCtrl()
                            else a_for_ttest[:, i, b] - a_for_ttest[:, i, b + nb]
                        )
                        for b in (0, lb)
                    ]
                    nbt = ttest_rel(
                        ab[0],
                        ab[1],
                        "%s, fly %s, bucket #%d vs. #%d"
                        % (t.name(), "1" if t.hasSymCtrl() else "delta", 1, lb + 1),
                    )[3]
                    if nbt < 2 and lb > 1:
                        lb = lb - 1
                    else:
                        break
        elif tp in ("rpip", "rpipd"):
            plotRewards(va, tp, a, trns, gis, gls)
        elif (
            tp in ("wall", "agarose", "boundary", "dbr", "max_ctr_d_no_contact")
            or "_turn" in tp
            or "r_no_contact" in tp
        ):
            plotRewards(va, tp, a, trns, gis, gls, vas)
        elif tp in ["pcm", "turn", "pivot"]:
            if not va.noyc:
                nPts = a.shape[2] // 2
                a = a[:, :, slice(nPts, None) if opts.yoked else slice(None, nPts)]
            plotRewards(va, tp, a, trns, gis, gls, vas)
        elif tp == "nrp":
            for i, t in enumerate(trns):
                for i1, i2 in ((0, 1), (4, 5), (0, 4), (1, 5), (2, 6), (3, 7)):
                    if i2 < a.shape[2]:
                        ttest_rel(
                            a[:, i, i1],
                            a[:, i, i2],
                            "training %d, %s vs. %s" % (t.n, cns[i1], cns[i2]),
                        )
        elif tp == "nrpp":
            plotRewards(va, tp, a, trns, gis, gls)
        elif tp == "rdp":
            ttest_rel(a[:, 0, 0], a[:, 0, 1], va.rdpInterval + ", %s vs. %s" % cns[:2])
            plotRdpStats(vas, gls, False)
        elif tp == "bysb2":
            for i, t in enumerate(trns):
                ab = [
                    np.hstack((a[:, i, b], a[:, i, b + nb])) if opts.ol else a[:, i, b]
                    for b in (0, nb - 1)
                ]
                ttest_rel(ab[0], ab[1], "%s, bucket #%d vs. #%d" % (t.name(), 1, nb))
        elif tp in ("spd", "stp", "rpm"):
            spst = tp in ("spd", "stp")
            fm = "{:.1f}" if tp == "rpm" else ("{:.2f}" if tp == "spd" else "{:.1%}")
            if ng == 1 and spst and nf == 2:
                for i, t in enumerate(trns):
                    ttest_rel(
                        a[:, i, 1],
                        a[:, i, 3],
                        "training %d, %s vs. %s" % (t.n, cns[1], cns[3]),
                    )
            print(
                "means with 95%% confidence intervals%s:"
                % (" (pre, training)" if spst else "")
            )
            if (
                tp == "spd"
                and va.ct in (CT.htl, CT.large, CT.large2)
                and SPEED_ON_BOTTOM
            ):
                print("note: sidewall and lid currently included")
            flies, groups = fliesForType(va, tp) if ng == 1 else (0,), list(range(ng))
            mgll = None if ng == 1 else max(len(g) for g in gls)
            ns = [np.count_nonzero(gis == g) for g in groups]
            print('  n = %s  (in "()" below if different)' % util.join(", ", ns))
            for i, t in enumerate(trns):
                for f, g in itertools.product(flies, groups):
                    txt = []
                    for b in range(nb):
                        ci = nb * f + b
                        mcn = util.meanConfInt(
                            a[np.flatnonzero(gis == g), i, ci], asDelta=True
                        )
                        sn = mcn[2] != ns[g]
                        txt.append(
                            ("%s ±%s%s" % (fm, fm, " ({})" if sn else "")).format(
                                *mcn[: 3 if sn else 2]
                            )
                        )
                    print(
                        "  %s %s: %s"
                        % (
                            "t%d," % t.n if f == 0 and g == 0 else " " * 3,
                            "%s fly" % flyDesc(f) if ng == 1 else gls[g].ljust(mgll),
                            ", ".join(txt),
                        )
                    )
        # handle "type codes" included in postAnalyze() for checkValues()
        elif tp == None:
            pass
        else:
            adba = tp == "adb" and not calc
            if (calc or adba) and nf == 2:
                assert nb == 2
                for b in range(1 + adba):
                    for i, t in enumerate(trns):
                        ttest_rel(
                            a[:, i, b],
                            a[:, i, b + nb],
                            "training %d, %s vs. %s" % (t.n, cns[b], cns[b + nb]),
                        )
                if not adba:
                    ttest_rel(a[:, 0, 2], a[:, 0, 3], "training 1, %s vs. %s" % cns[2:])
            if not calc:
                ttest_rel(a[:, 0, 0], a[:, 0, 1], "training 1, %s vs. %s" % cns[:2])
                if len(trns) > 1:
                    ttest_rel(a[:, 0, 0], a[:, 1, 0], "%s, training 1 vs. 2" % cns[0])
            if nf == 1 and calc:
                print("skipped")

    if opts.rdp:
        plotRdpStats(vas, gls)
    if opts.circle:
        plotAngularVelocity(vas, opts, gls)
        plotTurnRadiusHist(vas, gls, opts.yoked)


def frmStat(n):
    return "{:.3f}".format(n) if isinstance(n, (float, np.floating)) else str(n)


def basicStatsCols(va):
    return ("video," if VIDEO_COL else "") + ("fly," if va.f is not None else "")


def vidRow(va):
    return util.basename(va.fn) + "," if VIDEO_COL else ""


# save data to CSV with yoked-control flies on separate rows and with column
# names as given (i.e., columns unmapped to trainings or fly numbers)
def writeStatsOneRowPerFly(vas, cns, tp, calc, sf):
    sf.write(basicStatsCols(vas[0]) + cns + "\n")
    currBasename, byTraining = "", ("pic", "pic_custom", "nrtot", "nrcpp")

    def writeLines(lines):
        linesConcat = util.concat(lines)
        if len(linesConcat) == 0:
            return
        sf.write("\n".join(linesConcat) + "\n")

    for i, va in enumerate(vas):
        if util.basename(va.fn) != currBasename:
            if currBasename:
                writeLines(lines)
            lines = [[], []]
        currBasename, stats = util.basename(va.fn), vaVarForType(va, tp, calc)
        for f in va.flies:
            if va._bad(f):
                if f == va.flies[-1] and i == len(vas) - 1:
                    writeLines(lines)
                continue
            statForF = [st[f] for st in stats] if tp in byTraining else stats[f]
            lines[f].append(
                vidRow(va)
                + ("%i," % (va.f + va.nef * f) if va.f is not None else "")
                + ",".join(map(frmStat, statForF))
            )
            if f == va.flies[-1] and i == len(vas) - 1:
                writeLines(lines)


def duplicateColumnsAcrossTrns(cns, trns):
    return ",".join("%s %s" % (t.name(), cns) for t in trns)


def add_analysis_types_with_boundary_points(
    boundary_tp, analysis_types, incl_direction=False
):
    ellipse_ref_pts = ["edge"] + (["ctr"] if boundary_tp != "wall" else [])
    for ellipse_ref_pt in ellipse_ref_pts:
        analysis_types.append(f"{ellipse_ref_pt}_{boundary_tp}_turn_csv")
        if incl_direction and boundary_tp != "wall":
            analysis_types.append(f"{ellipse_ref_pt}_{boundary_tp}_turn_dir_csv")


def writeStats(vas, sf):
    print("\nwriting %s..." % STATS_FILE)

    def should_apply_pairwise_exclusion(va, tp, col_index=None):
        if col_index is not None:
            # Skip pairwise exclusion for the first pair in specific tables
            if tp in ["agarose_pct_edge", "agarose_pct_ctr"] and col_index == 0:
                return False
            return True

        return (
            not opts.independent_exclusion
            and len(va.flies) > 1
            and tp in analysis_types_with_training_number_columns
        )

    def should_apply_quadruple_exclusion(va, tp):
        return (
            not opts.independent_exclusion
            and len(va.flies) > 1
            and tp in analysis_types_with_quadruple_exclusion
        )

    def should_apply_even_odd_exclusion(va, tp):
        return (
            not opts.independent_exclusion
            and len(va.flies) > 1
            and tp in analysis_types_with_even_odd_exclusion
        )

    def apply_even_odd_exclusion(values):
        for i in range(0, len(values), 4):
            if i + 3 < len(values):
                if is_nan(values[i]) or is_nan(values[i + 2]):
                    values[i] = values[i + 2] = "nan"
                if is_nan(values[i + 1]) or is_nan(values[i + 3]):
                    values[i + 1] = values[i + 3] = "nan"
        return values

    util.writeCommand(sf, csvStyle=True)

    n, va = opts.numRewardsCompare, vas[0]
    for t in va.trns:
        sf.write('"%s"\n' % t.name(short=False))

    if not (va.circle or va.choice):
        return

    analysis_types = list(
        (
            ("c_pi", "bysb2")
            if va.choice
            else (
                "atb",
                "atb-c",
                "adb",
                "adb_csv-c",
                "nr",
                "nr-c",
                "nrc-c",
                "ppi",
                "nrp-c",
                "rdp",
                "bysb2",
                "frc",
                "xmb",
                "spd",
                "spd_sb",
                "rpi_combined-c",
                "pic",
            )
        )
    )
    if va.rectangle:
        analysis_types += ("spd", "nr", "bot_top_cross")
    one_row_per_fly_stats = ("pic", "pic_custom", "nrtot", "nrcpp")
    analysis_types_with_training_number_columns = [
        f"{tp}_csv"
        for tp in (
            "wall",
            "agarose",
            "boundary",
            "lg_turn_rwd",
            "adb",
            "dbr_no_contact",
            "max_ctr_d_no_contact",
        )
    ] + ["rpi_combined", "spd_sb"]

    for boundary_tp in ("wall", "boundary", "agarose"):
        add_analysis_types_with_boundary_points(
            boundary_tp,
            analysis_types_with_training_number_columns,
            incl_direction=opts.turn_dir,
        )

    if opts.wall:
        analysis_types += ("wall_csv", "dbr_no_contact_csv", "max_ctr_d_no_contact_csv")
        for boundary_orientation in ("all", "agarose_adj"):
            key = f"r_no_contact_{boundary_orientation}_csv"
            analysis_types += (key,)
            analysis_types_with_training_number_columns.append(key)

    if opts.agarose:
        analysis_types += ("agarose_csv", "agarose_pct_edge", "agarose_pct_ctr")
        analysis_types_with_training_number_columns.extend(analysis_types[-2:])

    if opts.boundary:
        analysis_types += ("boundary_csv", "boundary_pct_edge", "boundary_pct_ctr")
        analysis_types_with_training_number_columns.extend(analysis_types[-2:])

    if opts.turn:
        for boundary_tp in opts.turn:
            add_analysis_types_with_boundary_points(
                boundary_tp, analysis_types, incl_direction=opts.turn_dir
            )

    if opts.cTurnAnlyz:
        analysis_types += ("lg_turn_rwd_csv",)

    if opts.plotThm:
        analysis_types += ("btwn_rwd_dist_from_ctr",)
    if opts.pctTimeCircleRad:
        analysis_types += ("pic_custom",)

    analysis_types_with_quadruple_exclusion = [
        tp for tp in analysis_types if "turn_dir_csv" in tp
    ]
    analysis_types_with_quadruple_exclusion.extend(["agarose_csv", "boundary_csv"])

    analysis_types_with_even_odd_exclusion = ["agarose_csv", "boundary_csv"]

    for analysis_type in analysis_types:
        tp, calc = typeCalc(analysis_type)
        assert tp != "nrc" or calc is True
        hdr = headerForType(va, tp, calc)
        if hdr is None:
            continue
        sf.write(hdr + "\n")
        cns = ",".join(columnNamesForType(va, tp, calc, n))
        trns = trnsForType(va, tp)
        if not trns:
            sf.write("skipped\n")
            continue
        if (
            tp
            not in tuple(analysis_types_with_training_number_columns)
            + one_row_per_fly_stats
        ):
            cns = duplicateColumnsAcrossTrns(cns, trns)
        if tp in one_row_per_fly_stats:
            writeStatsOneRowPerFly(vas, cns, tp, calc, sf)
            continue
        sf.write(basicStatsCols(va) + cns + "\n")
        for va in vas:
            values = util.concat(vaVarForType(va, tp, calc), True)
            if should_apply_even_odd_exclusion(va, tp):
                values = apply_even_odd_exclusion(list(values))
            elif should_apply_quadruple_exclusion(va, tp):
                values = list(values)
                for i in range(0, len(values), 4):
                    if i + 3 < len(values):
                        if (
                            is_nan(values[i])
                            or is_nan(values[i + 1])
                            or is_nan(values[i + 2])
                            or is_nan(values[i + 3])
                        ):
                            values[i] = values[i + 1] = values[i + 2] = values[
                                i + 3
                            ] = "nan"
            elif should_apply_pairwise_exclusion(va, tp):
                values = list(values)
                for i in range(0, len(values), 2):
                    if should_apply_pairwise_exclusion(va, tp, col_index=i // 2):
                        if i + 1 < len(values):
                            if is_nan(values[i]) or (is_nan(values[i + 1])):
                                values[i] = values[i + 1] = "nan"
            else:
                values = list(values)

            sf.write(
                vidRow(va)
                + (f"{va.f}," if va.f is not None else "")
                + ",".join(map(frmStat, values))
                + "\n"
            )

    if hasattr(va, "avgMaxDist") and va.avgMaxDist[0]:
        sf.write(
            "\nheatmap analysis (epsilon %.1f; number traj.: %d)\n"
            % (opts.rdp, va.ntrx)
        )
        vs = (
            ("average maximum distance", "avgMaxDist"),
            ("average absolute first turn angle", "avgFirstTA"),
            ("average first run length", "avgFirstRL"),
        )
        ncols, ntrns = len(va.avgMaxDist[0]), len(va.trns)
        cols = "video,fly," + ",".join(
            ",".join(f"t{t+1} b{b+1}" for b in range(int(ncols / ntrns)))
            for t in range(ntrns)
        )
        for hdr, vn in vs:
            sf.write("\n" + hdr + "\n" + cols + "\n")
            for f in va.flies:
                for va in vas:
                    r = getattr(va, vn)[f]
                    assert len(r) == ncols
                    sf.write(
                        util.basename(va.fn) + f",{f+1}," + ",".join(map(str, r)) + "\n"
                    )


def analysisImage(vas):
    util.backup(ANALYSIS_IMG_FILE)
    imgs = [(va.aimg, util.basename(va.fn)) for va in vas if va.aimg is not None]
    img = util.combineImgs(imgs, nc=5)[0]
    writeImage(ANALYSIS_IMG_FILE, img)


_CAM_DATE = re.compile(r"^(c\d+__[\d-]+)")


def openLoopImage(vas):
    imgs = []
    for va in vas:
        if not hasattr(va, "olimg"):
            return
        bn = util.basename(va.fn)
        imgs.append(
            (va.olimg, bn if va.ct is CT.regular else util.firstGroup(_CAM_DATE, bn))
        )
    writeImage(OPEN_LOOP_IMG_FILE, util.combineImgs(imgs, nc=5)[0])


def circularTrxImages(vas):
    for va in vas:
        for trj in va.trx:
            CircularMotionDetector(trj, opts).circularTrxImg()


class BoundaryContactEventAnalyzer:
    def __init__(self, vas, boundary_tp, evt_name, save_imgs=True):
        """
        Initializes a BoundaryContactEventAnalyzer instance.

        This class is designed for analyzing boundary contact events in the context of
        Drosophila's goal-directed tasks in a controlled experimental environment. It
        processes video analysis data to understand the dynamics of walking trajectories and
        interactions with environment boundaries.

        Parameters:
        - vas (list[VideoAnalysis]): A list of VideoAnalysis objects, each representing
          analysis results for a video. These objects contain data on the Drosophila's walking
          trajectories and interactions with designated boundaries within the experimental
          setup.
        - boundary_tp (str): Specifies the type of boundary (e.g., "agarose", "turn") being
          analyzed. This affects how events are categorized and analyzed.
        - evt_name (str): The name of the event being analyzed. This is used to identify
          specific types of boundary contact events and to organize output data and images
          related to these events.
        - save_imgs (bool, optional): A flag indicating whether to save images related to the
          boundary contact events being analyzed. Defaults to True.

        Attributes:
        - num_rows (int): The number of rows of data to process. Default is set to 4.
        - num_imgs_saved (dict): A dictionary tracking the number of images saved for
          different types of events.
        - boundary_orientation (str): Determines the boundary orientation (e.g., "tb" for
          top-bottom) based on the event name, affecting how events are categorized.
        - log_header (str): A header for logging purposes, derived from the event name.
        - images_to_save (list): A list to hold images that are to be saved as part of the
          analysis.
        - image_headers (list): A list of headers corresponding to the images in
          `images_to_save`.
        - evt_key (str): A key representing the event statistics in the analysis results.
        - wall_pi (NoneType): Placeholder for wall position indicator, to be set to 'lr'
          (left-right) or 'tb' (top-bottom) during analysis. Initially None.

        Raises:
        - ValueError: If `evt_name` does not correspond to a recognized event type or if other
          input parameters do not meet required conditions (not explicitly checked in the
          provided code snippet but could be implemented).
        """
        self.vas = vas
        self.boundary_tp = boundary_tp
        self.evt_name = evt_name
        self.save_imgs = save_imgs
        self.num_rows = 4
        self.num_imgs_saved = {"pre_trn": 0, "t3": 0}
        if "agarose" in evt_name or "turn" in evt_name:
            self.boundary_orientation = "tb"
        else:
            self.boundary_orientation = opts.wall_orientation
        if "contact" in evt_name:
            self.log_header = "%s-contact" % evt_name.split("_")[0]
        else:
            self.log_header = evt_name.replace("_", " ")
        self.images_to_save = []
        self.image_headers = []
        self.evt_key = "%s_evt_stats" % evt_name

        # analysis for debugging
        self.wall_pi = None  # set to 'lr' (left-right) or 'tb' (top-bottom)
        # self.calc_contact_event_stats()

    def percent_change_ttests(self):
        """
        Performs t-tests on the percent change in boundary contact events across different
        groups of videos.

        This method calculates the statistical significance of the relative changes in
        boundary contact events between different experimental groups, identified by their
        unique group indexes. It is intended to help understand variations in Drosophila's
        behavior in response to different experimental conditions.

        Assumptions:
        - The method assumes that 'vas' has been populated with VideoAnalysis instances that
          include relevant boundary event data for the analysis.
        - Data for each VideoAnalysis instance includes a group index ('gidx') and boundary
          event relative changes keyed by event name and wall type.

        Outputs:
        - Prints the results of one-sample and independent two-sample t-tests, comparing the
          mean of the relative changes to 0 (null hypothesis) within groups and between
          groups, respectively.
        - Reports on insufficient data or NaN values in the dataset, skipping over those
          cases.

        Notes:
        - This method relies on external variables and settings, such as 'opts.groupLabels',
          for group labels and comparisons. It expects these settings to be predefined and
          relevant to the current analysis context.

        Raises:
        - RuntimeError: If the analysis prerequisites are not met, such as missing data or
          incorrect data formats (not explicitly checked in the provided code snippet but
          could be implemented).
        """
        gis = np.array([va.gidx for va in self.vas])
        combined_data = np.array(
            [
                va.boundary_event_rel_changes[self.evt_name][self.boundary_orientation]
                for va in self.vas
            ]
        )
        if combined_data.shape[0] == 1:
            return
        print(
            (
                "\n%s events, rel. change, %s"
                % (self.log_header, self.vas[0].bnd_contact_range_desc)
            )
        )
        vidx_gps = [np.flatnonzero(gis == gi) for gi in np.unique(gis)]
        if len(gis) > 1:
            gls = opts.groupLabels and opts.groupLabels.split("|")
        else:
            gls = None
        if np.all(np.isnan(combined_data)):
            print("  skipping analysis; insufficient data")
        for i, vidx_gp in enumerate(vidx_gps):
            if gls:
                print("\ngroup: %s" % gls[i])
            va_in_grp = self.vas[vidx_gp[0]]
            data_in_grp = combined_data[vidx_gp, :]
            for f in va_in_grp.flies:
                _, p, _ = ttest_1samp(data_in_grp[:, f], 0, msg=flyDesc(f))
                if np.isnan(p):
                    print("  insufficient data.")
        if gls and len(gls) > 1:
            print()
            for f in va_in_grp.flies:
                ttest_ind(
                    combined_data[vidx_gps[0], f],
                    combined_data[vidx_gps[1], f],
                    msg="comparison, %s vs. %s, %s" % (gls[0], gls[1], flyDesc(f)),
                    conf_int=True,
                )

    def sample_frames_surrounding_evt(
        self, trj, period_name, event_idx, event_len, event_degree, evt_type
    ):
        """
        Samples frames around a specified event to visualize the event's context within a
        trajectory. It marks the event and potentially the preceding event with different
        colors based on their proximity or significance.

        Parameters:
        - trj: Trajectory object containing the trajectory data and video capture object.
        - period_name: String name of the period during which the event occurs.
        - event_idx: The frame index where the event starts.
        - event_len: The length of the event in frames.
        - event_degree: A string indicating the proximity or significance of the event ("near"
          or "true").
        - evt_type: The type of event being sampled (e.g., "turn" or "contact").

        This method handles the visualization of events by marking them and optionally
        the events that preceded them. It also manages the creation and saving of
        images that visualize these events, including handling the spacing and
        organization of these images into a grid for display purposes.
        """

        def evt_color(degree):
            if degree == "near":
                return COL_B
            elif degree == "true":
                return COL_W

        show_evt_hist = hasattr(self, "previous_start_idx")
        if show_evt_hist:
            tween_frame_dist = event_idx - self.previous_start_idx
            tlen = tween_frame_dist
            if tween_frame_dist + 1 <= (self.context_frames - self.context_asymm):
                slice_offset = self.context_frames - tween_frame_dist + 1
                tlen = self.context_frames + 1
            else:
                slice_offset = 0
        else:
            tlen = 30
        for i in range(
            -self.context_frames + self.context_asymm,
            self.context_frames + 1 + self.context_asymm,
        ):
            frame_to_save = util.readFrame(trj.va.cap, event_idx + i)
            flagged_regions = []
            flagged_colors = []
            if (
                hasattr(self, "previous_evt_len")
                and self.previous_start_idx <= event_idx + i
            ):
                flagged_regions.append(
                    slice(0 + slice_offset, 0 + self.previous_evt_len + slice_offset)
                )
                flagged_colors.append(evt_color(self.previous_evt_degree))
            if i >= 1:
                flagged_regions.append(slice(tlen, event_len + tlen))
                flagged_colors.append(evt_color(event_degree))

            trj.annotate(
                frame_to_save,
                event_idx + i,
                tlen=tlen + i,
                boundaries={self.boundary_tp: True},
                flagged_regions=flagged_regions,
                flagged_cols=flagged_colors,
                evt_type=evt_type,
            )
            if event_idx > trj.va.trns[0].start:
                t = Training.get(trj.va.trns, event_idx)
                if t:
                    t.annotate(frame_to_save)
            self.images_to_save.append(
                frame_to_save[
                    self.crop_bounds[0] : self.crop_bounds[1],
                    self.crop_bounds[2] : self.crop_bounds[3],
                ]
            )
        self.image_headers.extend(
            [
                "%s, fly %i %s, fm %i, %s"
                % (
                    util.basename(trj.va.fn),
                    trj.va.ef,
                    "exp" if trj.f == 0 else "ctrl",
                    event_idx,
                    util.s2time((event_idx + i) / trj.va.fps),
                )
            ]
            + [None] * self.context_frames * 2
        )
        if len(self.images_to_save) % self.num_cols != 0:
            self.images_to_save.append(None)
            self.image_headers.append(None)
        if len(self.images_to_save) == self.num_rows * self.num_cols:
            combined = util.combineImgs(
                self.images_to_save,
                nc=self.num_cols,
                hdrs=self.image_headers,
                adjustHS=False,
            )[0]
            img_name = "imgs/%s_%s_examples_%s_%i.png" % (
                self.boundary_tp,
                evt_type,
                period_name,
                self.num_imgs_saved[period_name],
            )
            print(("writing %s..." % img_name))
            cv2.imwrite(img_name, combined)
            self.images_to_save = []
            self.image_headers = []
            self.num_imgs_saved[period_name] += 1

    def sample_all_contact_event_imgs(self, turning=False):
        """
        Initiates the sampling of contact event images for all trajectories within the
        dataset. Adjusts parameters based on whether the events to be sampled are turning
        events or not.

        Parameters:
        - turning: Boolean flag indicating whether the sampling is for turning events (True)
                or for contact events (False).

        This method sets up the context frames, the layout for the image sampling
        (number of columns, context asymmetry), and iterates through all valid
        trajectories to sample images for each.
        """
        self.events_per_row = 1 if turning else 2
        self.context_frames = 8 if turning else 3
        self.context_asymm = 6 if turning else 0
        self.num_cols = self.events_per_row * (self.context_frames * 2 + 2) - 1
        for va in self.vas:
            for trj in va.trx:
                if trj._bad:
                    continue
                self.sample_contact_events_imgs_for_trj(trj, turning)

    def sample_contact_events_imgs_for_trj(self, trj, turning):
        """
        Samples images for contact or turning events within a single trajectory. It adjusts
        the sampling process based on the type of event being visualized.

        Parameters:
        - trj: Trajectory object containing the trajectory data to be visualized.
        - turning: Boolean indicating whether to sample turning events (True) or
          based on the trajectory's position and whether it is in the first row or not.

        It then samples images for either contact or turning events based on the
        specified event type, using the specified temporal and spatial parameters to
        select and annotate the relevant frames.
        """
        in_first_row = trj.va.ef // trj.va.ct.numCols() == 0 and trj.f == 0
        y_ubound = max(trj.bounds_orig["y"][0] - (10 if in_first_row else 20), 0)
        y_lbound = trj.bounds_orig["y"][1] + (20 if in_first_row else 10)
        self.crop_bounds = [
            int(round(idx))
            for idx in (
                max(y_ubound, 0),
                y_lbound,
                max(trj.bounds_orig["x"][0] - 10, 0),
                trj.bounds_orig["x"][1] + 10,
            )
        ]
        near_event_idx = 0
        time_ranges = dict(
            list(
                zip(
                    list(self.num_imgs_saved.keys()),
                    ((trj.va.trns[2].start, trj.va.trns[2].stop),),
                )
            )
        )
        start_idxs = trj.boundary_event_stats[self.boundary_tp][
            self.boundary_orientation
        ]["edge"]["contact_start_idxs"]
        regions = trj.boundary_event_stats[self.boundary_tp][self.boundary_orientation][
            "edge"
        ]["boundary_contact_regions"]
        if turning:
            start_idxs_near = start_idxs[
                trj.boundary_event_stats[self.boundary_tp][self.boundary_orientation][
                    "near_turning_indices"
                ]
            ]
            near_regions = [
                regions[idx]
                for idx in trj.boundary_event_stats[self.boundary_tp][
                    self.boundary_orientation
                ]["near_turning_indices"]
            ]
            start_idxs = start_idxs[
                trj.boundary_event_stats[self.boundary_tp][self.boundary_orientation][
                    "turning_indices"
                ]
            ]
            regions = [
                regions[idx]
                for idx in trj.boundary_event_stats[self.boundary_tp][
                    self.boundary_orientation
                ]["turning_indices"]
            ]
        else:
            start_idxs_near = trj.boundary_event_stats[self.boundary_tp][
                self.boundary_orientation
            ]["near_contact_start_idxs"]
            near_regions = trj.boundary_event_stats[self.boundary_tp][
                self.boundary_orientation
            ]["near_contact_regions"]
        for k in time_ranges:
            if hasattr(self, "previous_start_idx"):
                del self.previous_start_idx
            if hasattr(self, "previous_evt_len"):
                del self.previous_evt_len
            self.images_to_save = []
            self.image_headers = []
            num_near_events_added = 0
            near_contact_event_idxs = []
            events_in_range = (start_idxs >= time_ranges[k][0]) & (
                start_idxs <= time_ranges[k][1]
            )
            if len(events_in_range) == 0:
                continue
            first_event_idx = util.firstTrue(events_in_range)
            events_in_range = start_idxs[events_in_range][:100]
            for i, event_idx in enumerate(events_in_range):
                while start_idxs_near[near_event_idx] < time_ranges[k][0]:
                    near_event_idx += 1
                near_contact_idx = start_idxs_near[near_event_idx]
                if near_contact_idx < event_idx:
                    evt_len = (
                        near_regions[near_event_idx].stop
                        - near_regions[near_event_idx].start
                    )
                    evt_degree = "near"
                    self.sample_frames_surrounding_evt(
                        trj,
                        k,
                        start_idxs_near[near_event_idx],
                        evt_len,
                        evt_degree,
                        "turn" if turning else "contact",
                    )
                    near_contact_event_idxs.append(start_idxs_near[near_event_idx])
                    self.previous_evt_degree = evt_degree
                    self.previous_start_idx = near_regions[near_event_idx].start
                    self.previous_evt_len = evt_len
                    num_near_events_added += 1
                    near_event_idx += 1
                if start_idxs_near[near_event_idx] > event_idx:
                    evt_len = (
                        regions[first_event_idx + i].stop
                        - regions[first_event_idx + i].start
                    )
                    evt_degree = "true"
                    self.sample_frames_surrounding_evt(
                        trj,
                        k,
                        event_idx,
                        evt_len,
                        evt_degree,
                        "turn" if turning else "contact",
                    )
                    self.previous_evt_degree = evt_degree
                    self.previous_start_idx = event_idx
                    self.previous_evt_len = evt_len

    def calc_contact_event_stats(self):
        """
        Calculates statistics for contact events between flies and walls or boundaries,
        generating a CSV file that summarizes the number of events per side (left, right,
        top, bottom) and a preference index (PI) for these interactions. This method also
        triggers the sampling of contact event images if image saving is enabled.

        The preference index (PI) is calculated as (E1 - E2) / (E1 + E2), where E1 and E2
        are the number of events on opposite sides (e.g., left vs. right or top vs. bottom).
        This index helps in understanding the flies' preference or avoidance behavior
        towards specific sides of the enclosure.

        The output CSV file, `wall_contact_stats.csv`, includes columns for video name,
        fly number, number of events for each side, and the preference index. The format
        and included data vary slightly depending on the wall interaction orientation
        (`lr` for left-right, `tb` for top-bottom).

        Note:
        - The calculation and image sampling only proceed for trajectories marked as valid
        (not `_bad`).
        - Events are filtered based on the fly's position in the enclosure, depending on the
        wall preference index orientation (left-right or top-bottom).

        Attributes:
            save_imgs (bool): A flag indicating whether to save images of contact events.
            wall_pi (str): Specifies the orientation for calculating the preference index
                        (`lr` for left-right, `tb` for top-bottom). If not set, the method
                        skips PI calculation.

        Side Effects:
            - Generates a CSV file named `wall_contact_stats.csv` in the current working directory.
            - May save images of contact events if `save_imgs` is True.

        Example Usage:
            >>> analyzer = ContactEventAnalyzer(...)
            >>> analyzer.calc_contact_event_stats()
        This will calculate the contact event statistics and, depending on configuration,
        save relevant images and output a CSV file summarizing the findings.
        """
        csv_data = [["vid. name", "fly no."]]
        if self.wall_pi == "lr":
            csv_data[-1].extend(["# left events", "# right events", "PI (l - r)"])
        elif self.wall_pi == "tb":
            csv_data[-1].extend(["# top events", "# bottom events", "PI (t - b)"])
        for va in self.vas:
            vid_name = util.basename(va.fn)
            if self.wall_pi == "lr" and not (
                va.ef % va.ct.numCols() == 0
                or va.ef % va.ct.numCols() == va.ct.numCols() - 1
            ):
                continue
            elif self.wall_pi == "tb" and not (va.ef < va.ct.numCols()):
                continue
            for trj in va.trx:
                if trj._bad:
                    continue
                if self.save_imgs:
                    self.sample_contact_events_imgs_for_trj(trj)
                if not self.wall_pi:
                    continue
                fn = "%i (%s)" % (va.f, "yc" if trj.f else "exp")
                if self.wall_pi == "lr":
                    if va.ef % va.ct.numCols() > 0:
                        evt_tp_1 = trj.wall_touch_by_direction["right"]
                        evt_tp_2 = trj.wall_touch_by_direction["left"]
                    else:
                        evt_tp_1 = trj.wall_touch_by_direction["left"]
                        evt_tp_2 = trj.wall_touch_by_direction["right"]
                if self.wall_pi == "tb":
                    if trj.f == 0:
                        evt_tp_1 = trj.wall_touch_by_direction["top"]
                        evt_tp_2 = trj.wall_touch_by_direction["bottom"]
                    else:
                        evt_tp_1 = trj.wall_touch_by_direction["bottom"]
                        evt_tp_2 = trj.wall_touch_by_direction["top"]
                pi = (evt_tp_1 - evt_tp_2) / (evt_tp_1 + evt_tp_2)
                csv_data.append([vid_name, fn, evt_tp_1, evt_tp_2, "%.3f" % pi])

            if not self.wall_pi:
                continue
            with open("wall_contact_stats.csv", "wt") as f:
                cw = csv.writer(f, delimiter=",")
                cw.writerows(csv_data)


# - - -


def detect_and_split_input(v):
    """
    Detects if 'v' is a Windows path with a drive and properly separates the indices.
    Returns the path and indices separately.
    """
    if WINDOWS and ":" in v and not v.startswith("/"):
        if "\\" in v:
            path_parts = v.split("\\")
        else:
            path_parts = v.split("/")
        first_part = path_parts[0]
        if len(first_part) == 2 and first_part[1] == ":":
            if ":" in path_parts[-1]:
                v, fs1 = "\\".join(path_parts[:-1]), path_parts[-1].split(":")[-1]
                return v, fs1
            else:
                return v, None
    else:
        vFs = v.split(":")
        if len(vFs) == 1:
            return v, None
        else:
            return vFs[0], vFs[1]
    return v, None


def analyze():
    customizer.update_font_size(opts.fontSize)
    customizer.update_font_family(opts.fontFamily)
    if P:
        mpl.rcParams.update(
            {
                "font.size": 12,  # ignore opts.fontSize
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
            }
        )
    mpl.rcParams.update({"axes.linewidth": 1, "lines.dashed_pattern": "3.05, 3."})
    vgs = opts.video.split("|")
    ng = len(vgs)
    # flies by group
    if opts.fly is None:
        fs = [[None]]
    else:
        fs = [util.parseIntList(v) for v in opts.fly.split("|")]
    if len(fs) == 1:
        fs = fs * ng
    if len(fs) != ng:
        error("fly numbers required for each group")

    # fn2fs: file name: list with the lists of fly numbers for each group

    fn2fs, fnf = collections.defaultdict(lambda: [[] for _ in range(ng)]), []
    for i, vg in enumerate(vgs):
        for entry in vg.split(","):
            path, raw_ids = detect_and_split_input(entry)
            if raw_ids:
                ids = util.parseIntList(raw_ids)
            else:
                ids = fs[i]
            if WINDOWS:
                path = util.unix_to_windows(path)
            for fn in util.fileList(path, "analyze", pattern=AVI_X):
                fn2fs[fn][i].extend(ids)
                # build flat list of filename:ID strings
                fnf.extend(fn if f is None else f"{fn}:{f}" for f in ids)
    dups = util.duplicates(fnf)
    if dups:
        error("duplicate: %s" % dups[0])
    fns = list(fn2fs.keys())
    if not fns:
        return
    cns = [int(util.firstGroup(CAM_NUM, util.basename(fn))) for fn in fns]
    vas, va = [], None
    for i, fn in enumerate([fn for (cn, fn) in sorted(zip(cns, fns))]):
        for gidx in range(ng):
            for f in fn2fs[fn][gidx]:
                if va:
                    print()
                va = VideoAnalysis(fn, gidx, opts, f)
                if not va.skipped():
                    vas.append(va)

    if vas:
        if opts.timeit:
            post_start = timeit.default_timer()
        va0 = vas[0]
        postAnalyze(vas)
        if opts.wall:
            analyzer = BoundaryContactEventAnalyzer(
                vas, boundary_tp="wall", evt_name="wall_contact"
            )
            analyzer.percent_change_ttests()
            if opts.wall_eg:
                analyzer.sample_all_contact_event_imgs()
        if opts.agarose:
            analyzer = BoundaryContactEventAnalyzer(
                vas, boundary_tp="agarose", evt_name="agarose_contact"
            )
            analyzer.percent_change_ttests()
            if opts.agarose_eg:
                analyzer.sample_all_contact_event_imgs()
        if opts.turn:
            analyzer = BoundaryContactEventAnalyzer(
                vas, boundary_tp=opts.turn, evt_name="%s_turn" % opts.turn
            )
            if opts.turn_eg:
                analyzer.sample_all_contact_event_imgs(turning=True)
        util.backup(STATS_FILE)
        with open(STATS_FILE, "w", 1) as sf:
            writeStats(vas, sf)
        if va0.circle or va0.choice:
            analysisImage(vas)
        if va0.circle:
            if opts.fixSeed:
                random.seed(101)
            try:

                random.choice(vas).calcRewardsImgs()
            except VideoError:
                print('some "rewards images" not written due to video error')
        if opts.hm:
            plotHeatmaps(vas)
        if va0.openLoop:
            openLoopImage(vas)
        if len(vas) == 1:
            if opts.circle:
                plotAngularVelocity(vas, opts)
                plotTurnRadiusHist(vas)
        if opts.circleTrx:
            circularTrxImages(vas)
    if opts.timeit:
        print("total post-analyze time:", timeit.default_timer() - post_start)
        print(
            "Mean per-bnd contact processing time:",
            util.meanConfInt(per_bnd_processing_times, asDelta=True),
        )
        print(
            "Mean per-large turn processing time:",
            util.meanConfInt(per_lgt_processing_times, asDelta=True),
        )
        print(
            "Mean per-VA processing time:",
            util.meanConfInt(per_va_processing_times, asDelta=True),
        )
    if opts.showPlots or opts.showTrackIssues:
        plt.show(block=False)
        input("\npress Enter to continue...")


# - - -


# self test
def test():
    Trajectory._test(opts)


# - - -
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    opts = p.parse_args()
    if opts.angVelOverTime:
        opts.circle = True
    if opts.turn and "circle" in opts.turn:
        opts.cTurnAnlyz = True
        opts.turn.remove("circle")
    else:
        opts.cTurnAnlyz = False

    if opts.turn_prob_by_dist:
        opts.turn_prob_by_dist = parse_distances(opts.turn_prob_by_dist)
    if opts.outside_circle_radii:
        opts.outside_circle_radii = parse_distances(opts.outside_circle_radii)

    if opts.turn_prob_by_dist or opts.outside_circle_radii:
        if opts.contact_geometry == "horizontal" and not opts.turn_prob_by_dist:
            error("--turn-prob-by-dist must be supplied for horizontal geometry")
        if opts.contact_geometry == "circular" and not opts.outside_circle_radii:
            error("--outside-circle-radii must be supplied for circular geometry")

    # - - -
    test()
    log = not (opts.showPlots or opts.showTrackIssues)
    # note: Tee makes plt.show(block=False) not work
    if log:
        util.backup(LOG_FILE)
    with open(LOG_FILE if log else os.devnull, "w", 1) as lf:
        util.writeCommand(lf)
        if log:
            sys.stdout = Tee([sys.stdout, lf])
        analyze()
        if opts.timeit:
            print("Script processing time:", timeit.default_timer() - start_t)
