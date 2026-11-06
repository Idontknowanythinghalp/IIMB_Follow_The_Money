"""generate_data.py - builds the synthetic data for the Follow the Money prototype (v1).

Run:   python generate_data.py
Makes: data/transactions.csv, data/merchants.csv, data/calendar.csv, data/ground_truth.csv

v1 covers: the calendar, M1 (normal bakery), M2 (Diwali sweet shop), M3 (betting bakery).
Later versions add M4, M5, M7 (then M6, M3b).

The scorer must NEVER read ground_truth.csv. It exists only so we can check
our results (and count false positives) afterwards.
"""
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import config

rng = np.random.default_rng(config.SEED)
OTHER_CITIES = ["MUM", "DEL", "HYD", "CHE"]


# ---------------------------------------------------------------- calendar
def build_calendar():
    """Fictional fixtures (3 per week, weekdays rotate) plus one festival week."""
    rows = []
    fixture_no = 0
    for week in range(config.WEEKS):
        week_start = config.START_DATE + timedelta(weeks=week)
        for offset in config.FIXTURE_DAY_OFFSETS:
            day = (week * 2 + offset) % 7  # rotates the weekday each week
            start = (week_start + timedelta(days=day)).replace(
                hour=config.FIXTURE_START_HOUR, minute=config.FIXTURE_START_MINUTE
            )
            end = start + timedelta(hours=config.FIXTURE_LENGTH_HOURS)
            fixture_no += 1
            rows.append(
                {"event_id": f"F{fixture_no:02d}", "type": "fixture",
                 "name": f"Fictional fixture {fixture_no}", "start": start, "end": end}
            )
    festival_start = config.START_DATE + timedelta(weeks=config.FESTIVAL_WEEK)
    rows.append(
        {"event_id": "FEST1", "type": "festival", "name": "Diwali week (fictional dates)",
         "start": festival_start, "end": festival_start + timedelta(days=7, seconds=-1)}
    )
    return pd.DataFrame(rows)


def fixture_windows(calendar):
    """List of (window_start, window_end) for every fixture."""
    windows = []
    for _, row in calendar[calendar["type"] == "fixture"].iterrows():
        windows.append(
            (row["start"] - timedelta(hours=config.WINDOW_BEFORE_HOURS),
             row["end"] + timedelta(hours=config.WINDOW_AFTER_HOURS))
        )
    return windows


# ----------------------------------------------------------------- helpers
def make_pool(prefix, size, home_city, p_home=0.95):
    """A pool of payers: {payer_id: payer_region}. Most live in the home city."""
    pool = {}
    for i in range(1, size + 1):
        city = home_city if rng.random() < p_home else str(rng.choice(OTHER_CITIES))
        pool[f"{prefix}{i:04d}"] = city
    return pool


def random_times(n, days, start_hour, end_hour):
    """n random timestamps: a random day from `days`, a random time between the hours."""
    day_idx = rng.integers(0, len(days), size=n)
    seconds = rng.integers(start_hour * 3600, end_hour * 3600, size=n)
    return [days[d] + timedelta(seconds=int(s)) for d, s in zip(day_idx, seconds)]


def times_in_windows(n, windows):
    """n random timestamps, each inside a randomly chosen fixture window."""
    out = []
    for i in rng.integers(0, len(windows), size=n):
        start, end = windows[i]
        span = (end - start).total_seconds()
        out.append(start + timedelta(seconds=float(rng.uniform(0, span))))
    return out


def varied_amounts(n, low, high):
    """Whole-rupee amounts that are NOT round (a bump keeps them off multiples of 100)."""
    amounts = rng.integers(low, high + 1, size=n)
    return np.where(amounts % 100 == 0, amounts + 7, amounts)


def make_txns(merchant_id, times, pool, amounts):
    """Turn times + amounts into transaction rows, picking a random payer from the pool."""
    payer_ids = rng.choice(list(pool.keys()), size=len(times))
    return pd.DataFrame(
        {"merchant_id": merchant_id,
         "timestamp": times,
         "amount": [int(a) for a in amounts],
         "payer_id": payer_ids,
         "payer_region": [pool[p] for p in payer_ids]}
    )


# --------------------------------------------------------------- merchants
MERCHANTS = [
    # id, name, category, mcc, group, ticket min/max, monthly min/max, hours, festival, catchment
    ("M1", "Bakery A", "bakery", 5462, "food", 30, 600, 40000, 80000, 7, 20, False, "BLR"),
    ("M2", "Sweet Shop", "sweets", 5441, "food", 100, 2000, 50000, 100000, 9, 21, True, "BLR"),
    ("M3", "Bakery C", "bakery", 5462, "food", 30, 600, 40000, 80000, 7, 20, False, "BLR"),
]
MERCHANT_COLUMNS = [
    "merchant_id", "name", "declared_category", "declared_mcc", "category_group",
    "declared_ticket_min", "declared_ticket_max", "declared_monthly_min",
    "declared_monthly_max", "declared_busy_start", "declared_busy_end",
    "festival_eligible", "declared_catchment",
]
GROUND_TRUTH = [
    ("M1", "normal bakery", False),
    ("M2", "genuine festival spike", False),
    ("M3", "betting bakery (fixture-linked)", True),
]


def main():
    calendar = build_calendar()
    windows = fixture_windows(calendar)
    all_days = [config.START_DATE + timedelta(days=d) for d in range(config.WEEKS * 7)]
    fest_first = config.FESTIVAL_WEEK * 7
    festival_days = all_days[fest_first:fest_first + 7]

    parts = []

    # M1: normal bakery. Mostly one-time payers, varied amounts, no link to fixtures.
    pool = make_pool("M1P", 400, "BLR")
    parts.append(make_txns("M1", random_times(600, all_days, 7, 20), pool,
                           varied_amounts(600, 40, 450)))

    # M2: sweet shop. Normal trade all 8 weeks, plus a big genuine Diwali-week surge.
    pool = make_pool("M2P", 700, "BLR")
    parts.append(make_txns("M2", random_times(300, all_days, 9, 21), pool,
                           varied_amounts(300, 100, 800)))
    parts.append(make_txns("M2", random_times(300, festival_days, 10, 21), pool,
                           varied_amounts(300, 200, 1500)))

    # M3: betting bakery. Normal bakery trade, plus bets from a small repeat pool,
    # round amounts, ~85% inside fixture windows.
    pool = make_pool("M3P", 250, "BLR")
    parts.append(make_txns("M3", random_times(300, all_days, 7, 20), pool,
                           varied_amounts(300, 40, 450)))
    bettors = make_pool("M3B", 60, "BLR", p_home=0.85)
    bet_times = times_in_windows(340, windows) + random_times(60, all_days, 18, 24)
    bet_amounts = rng.choice([100, 200, 500, 1000], size=400)
    parts.append(make_txns("M3", bet_times, bettors, bet_amounts))

    transactions = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    transactions.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(transactions) + 1)])

    merchants = pd.DataFrame(MERCHANTS, columns=MERCHANT_COLUMNS)
    truth = pd.DataFrame(GROUND_TRUTH, columns=["merchant_id", "scenario", "is_front"])

    os.makedirs(config.DATA_DIR, exist_ok=True)
    stamp = "%Y-%m-%dT%H:%M:%S"
    transactions.assign(timestamp=transactions["timestamp"].dt.strftime(stamp)).to_csv(
        f"{config.DATA_DIR}/transactions.csv", index=False)
    merchants.to_csv(f"{config.DATA_DIR}/merchants.csv", index=False)
    calendar.assign(start=calendar["start"].dt.strftime(stamp),
                    end=calendar["end"].dt.strftime(stamp)).to_csv(
        f"{config.DATA_DIR}/calendar.csv", index=False)
    truth.to_csv(f"{config.DATA_DIR}/ground_truth.csv", index=False)

    print_summary(transactions, windows)


def print_summary(transactions, windows):
    """Quick sanity check printed after every run."""
    in_window = pd.Series(False, index=transactions.index)
    for start, end in windows:
        in_window |= (transactions["timestamp"] >= start) & (transactions["timestamp"] <= end)
    transactions = transactions.assign(in_window=in_window)
    print(f"{'merchant':<9}{'txns':>6}{'total Rs':>11}{'payers':>8}{'in-window %':>13}")
    for mid, g in transactions.groupby("merchant_id"):
        share = 100 * g.loc[g["in_window"], "amount"].sum() / g["amount"].sum()
        print(f"{mid:<9}{len(g):>6}{g['amount'].sum():>11,}{g['payer_id'].nunique():>8}{share:>12.0f}%")
    print(f"\nWrote files to ./{config.DATA_DIR}/")


if __name__ == "__main__":
    main()
