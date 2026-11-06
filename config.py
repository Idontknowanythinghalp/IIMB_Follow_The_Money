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
