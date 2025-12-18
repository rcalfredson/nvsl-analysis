import enum

# fixed: not behavior dependent
# midline defaults to control if training has no symmetric control circle
ST = enum.Enum("SyncType", "fixed midline control")

BORDER_WIDTH = 1
CAP_1ST_LTR = True
CONTACT_BUFFER_OFFSETS = {
    "wall": {"min": 0.7, "max": 1.0},
    "wall_contact": {"min": 0.4, "max": 0.8},
    "wall_walking": {"min": 1.2, "max": 9999},
    "agarose": {"min": 0.5, "max": 0.6},
    "boundary": {"min": 0.0, "max": 0.1},
}
WALL_CONTACT_DEFAULT_THRESH_STR = (
    f"{CONTACT_BUFFER_OFFSETS['wall']['min']}|{CONTACT_BUFFER_OFFSETS['wall']['max']}"
)
AGAROSE_BOUNDARY_DIST = 3
MIDLINE_BOUNDARY_DIST = 4
HEATMAP_DIV = 2
# whether to use calculated template match values for yoked control circles
LEGACY_YC_CIRCLES = False
LGC2 = True  # version 2 of large chamber (39x39 mm)
P = False  # whether to use paper style for plots
POST_SYNC = ST.fixed  # when to start post buckets
RDP_MIN_LINES = RDP_MIN_TURNS = 100  # for including fly in analysis
RI_START = ST.midline  # when to start RI calculation
RI_START_POST = ST.control  # ditto for post period
# whether to use circle-distance-based calculation for midline crossing
MIDLINE_XING2 = True
SPEED_ON_BOTTOM = True  # whether to measure speed only on bottom
VBA = False

SAVE_AUC_TYPES = {"rpid", "rpd", "meddist"}

_RDP_PKG = False

# Reason codes for why COM bucket became NaN (or was never computed)
COMR_OK = 0
COMR_NO_FULL_BUCKETS = 1
COMR_BAD_TRAJ = 2
COMR_EXCLUDED_PAIR = 3
COMR_EMPTY_BUCKET = 4
COMR_NAN_MEAN = 5
COMR_INCOMPLETE_BUCKET = 6
COMR_WALL_CONTACT = 7

# per-segment-specific
COMR_INSUFF_REWARDS = 10
COMR_SEG_TOO_SHORT = 11
COMR_SEG_MEDDIST_NAN = 12
COMR_SEG_MEDDIST_FILTER = 13
COMR_SEG_MEAN_NAN = 14
COMR_SEG_WALL_CONTACT_FILTER = 15
COMR_SEG_EMPTY_XY = 16
COMR_SEG_ALL_NAN_X_OR_Y = 17
