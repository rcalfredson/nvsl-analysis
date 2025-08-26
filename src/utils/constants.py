import enum

# fixed: not behavior dependent
# midline defaults to control if training has no symmetric control circle
ST = enum.Enum("SyncType", "fixed midline control")

BORDER_WIDTH = 1
CONTACT_BUFFER_OFFSETS = {
    "wall": {"min": 0.7, "max": 1.0},
    "wall_contact": {"min": 0.4, "max": 0.8},
    "wall_walking": {"min": 1.2, "max": 9999},
    "agarose": {"min": 0.5, "max": 0.6},
    "boundary": {"min": 0.0, "max": 0.1},
}
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

SAVE_AUC_TYPES = {"rpid", "rpd", "com"}

_RDP_PKG = False
