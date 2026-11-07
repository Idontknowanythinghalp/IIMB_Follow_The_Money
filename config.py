"""config.py - every tunable number lives here, so we tune without touching logic.

All values are PLACEHOLDERS (see the spec doc). Change them here and nowhere else.
Scoring thresholds will be added to this file when we build scoring.py.
"""
from datetime import datetime

# ---- Reproducibility: same seed = same data every run ----
SEED = 42

# ---- Simulation window (fictional): 8 weeks starting on a Monday ----
START_DATE = datetime(2026, 9, 7)
WEEKS = 8
FESTIVAL_WEEK = 6  # 0-based: week index 6 is the 7th week (fictional "Diwali week")

# ---- Fixtures (fictional) ----
FIXTURE_START_HOUR = 19
FIXTURE_START_MINUTE = 30
FIXTURE_LENGTH_HOURS = 4
# Three fixtures per week. The weekdays rotate each week, so every weekday
# also has plenty of NON-fixture days (needed for the matched-hours baseline).
FIXTURE_DAY_OFFSETS = (0, 3, 5)

# Fixture window = from WINDOW_BEFORE_HOURS before the start
# to WINDOW_AFTER_HOURS after the end (placeholder values).
WINDOW_BEFORE_HOURS = 3
WINDOW_AFTER_HOURS = 1

# ---- Output folder ----
DATA_DIR = "data"

# =====================================================================
# Scoring (added with scoring.py v1). ALL PLACEHOLDERS - tune here only.
# =====================================================================

# Calendar signal: ratio = money in fixture windows / expected money in the
# same clock hours on non-fixture days. 0 points at 1.0, 100 points at 4.0.
CAL_RATIO_SCALE = (1.0, 4.0)
# Minimum transactions inside fixture windows before the calendar signal is scored
# (tiny samples give wild ratios).
CAL_MIN_WINDOW_TXNS = 20

# Payer-pattern sub-metrics: (value that scores 0, value that scores 100).
PAYER_SCALES = {
    "top_payer_share":     (0.30, 0.80),  # share of money from the top 10% of payers
    "round_share":         (0.20, 0.80),  # share of txns that are multiples of 100
    "ticket_out_of_range": (0.10, 0.60),  # share of txns outside declared ticket range
    "volume_vs_declared":  (1.0, 3.0),    # monthly volume / declared monthly max
    "volume_vs_trend":     (1.3, 2.0),    # last-half volume / first-half volume
    "amount_clustering":   (0.15, 0.60),  # share of txns in the 3 most common amounts
    "outside_catchment":   (0.20, 0.80),  # share of payers outside the declared catchment
    "off_hours_share":     (0.05, 0.40),  # share of txns outside the declared busy hours
    "ticket_vs_trend":     (1.5, 4.0),    # last-half average ticket / first-half average ticket
    "refund_share":        (0.05, 0.25),  # refunded money / payment money
}
TOP_PAYER_FRACTION = 0.10
PAYER_TOP_K = 3  # s_payer = average of the K strongest sub-scores (avoids dilution)

# Two-key rule: if the calendar signal is the ONLY module at/above this,
# drift is capped (a calendar spike alone cannot prove betting).
TWO_KEY_THRESHOLD = 30
TWO_KEY_CAP = 59

# Confidence (minimum-evidence rule)
MIN_TXNS_TO_SCORE = 20
MIN_TXNS_CONFIDENT = 50
LOW_CONFIDENCE_MAX_RUNG = 2

# Response ladder: (minimum final score, rung, name), checked top to bottom.
LADDER = [
    (90, 4, "Review / offboard"),
    (75, 3, "Payout hold"),
    (60, 2, "Ask merchant"),
    (30, 1, "Monitor"),
    (0, 0, "Green"),
]

# ---- Added in scoring v2 ----
# Payer pattern is also computed on the fixture-window slice (needs this many txns).
MIN_SLICE_TXNS = 30

# Network booster: both merchants must look gambling-like and have unrelated
# category groups. They are then linked if they share a settlement account,
# OR share payers inside fixture windows (overlap must beat the same-hours baseline).
NETWORK_OVERLAP_MIN = 0.60   # shared payers / smaller merchant's payers (in fixture windows)
NETWORK_MIN_SHARED = 8       # at least this many bet-like payers shared (scaled to our small synthetic merchants)
NETWORK_LIFT = 1.5           # in-window overlap must beat the same-hours baseline by this factor
NETWORK_BONUS = 25