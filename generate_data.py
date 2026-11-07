"""generate_data.py - builds the synthetic data for the Follow the Money prototype (v3).

Run:   python generate_data.py
Makes: data/transactions.csv, data/merchants.csv, data/calendar.csv, data/ground_truth.csv

Merchants:  M1 normal bakery            M2 genuine Diwali sweet shop
            M3 betting bakery           M4a/M4b split operator (shared bettor pool)
            M5a/M5b campus shops        M7 genuine match-night pizza shop
            M8 festival-hiding front    M9 evasive festival front
            M10a/M10b split operator linked only by a shared settlement account
(M6 roulette / session module and M3b smurfing variant come later.)

The scorer must NEVER read ground_truth.csv. It exists only so we can check
our results (and count false positives) afterwards.
"""
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import config

rng = np.random.default_rng(config.SEED)

SPLIT_POOL = 25        # shared bettor pool for the split operator (M4a/M4b)
SPLIT_BETS = 40        # bet-like txns per split shop
SPLIT_IN_WINDOW = 0.8  # share of those inside fixture windows
SPLIT_NORMAL = 250     # normal (honest-looking) txns per split shop
OTHER_CITIES = ["MUM", "DEL", "HYD", "CHE"]
NORMAL_REFUND_RATE = 0.03
REFUND_RATES = {"M3": 0.15}  # the betting bakery returns money through refunds


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
    """Turn times + amounts into payment rows, picking a random payer from the pool."""
    payer_ids = rng.choice(list(pool.keys()), size=len(times))
    return pd.DataFrame(
        {"merchant_id": merchant_id,
         "timestamp": times,
         "amount": [int(a) for a in amounts],
         "payer_id": payer_ids,
         "payer_region": [pool[p] for p in payer_ids],
         "txn_type": "payment"}
    )


def make_refunds(payments, rate):
    """Refund a random share of payments: same payer, same amount, 1-48 hours later."""
    n = int(round(rate * len(payments)))
    if n == 0:
        return payments.iloc[0:0]
    picked = payments.iloc[rng.choice(len(payments), size=n, replace=False)].copy()
    delays = rng.uniform(1, 48, size=n)
    picked["timestamp"] = [t + timedelta(hours=float(d)) for t, d in zip(picked["timestamp"], delays)]
    picked["txn_type"] = "refund"
    return picked


# --------------------------------------------------------------- merchants
MERCHANTS = [
    # id, name, category, mcc, group, ticket min/max, monthly min/max, hours, festival, catchment
    ("M1", "Bakery A", "bakery", 5462, "food", 30, 600, 40000, 80000, 7, 20, False, "BLR"),
    ("M2", "Sweet Shop", "sweets", 5441, "food", 100, 2000, 50000, 100000, 9, 21, True, "BLR"),
    ("M3", "Bakery C", "bakery", 5462, "food", 30, 600, 40000, 80000, 7, 20, False, "BLR"),
    ("M4a", "Bakery D", "bakery", 5462, "food", 30, 600, 30000, 70000, 7, 23, False, "BLR"),
    ("M4b", "Recharge Hub", "recharge", 4814, "telecom", 10, 1000, 30000, 70000, 8, 22, False, "BLR"),
    ("M5a", "Campus Canteen", "canteen", 5812, "food", 20, 400, 40000, 90000, 9, 17, False, "BLR"),
    ("M5b", "Campus Stationery", "stationery", 5943, "retail", 10, 500, 20000, 60000, 9, 17, False, "BLR"),
    ("M7", "Match-night Pizza", "restaurant", 5812, "food", 100, 1500, 100000, 200000, 17, 23, False, "BLR"),
    ("M8", "Sweet Shop B", "sweets", 5441, "food", 100, 2000, 50000, 100000, 9, 21, True, "BLR"),
    ("M9", "Sweet Shop C", "sweets", 5441, "food", 100, 2000, 50000, 100000, 9, 21, True, "BLR"),
    ("M10a", "Bakery E", "bakery", 5462, "food", 30, 600, 30000, 70000, 7, 23, False, "BLR"),
    ("M10b", "Recharge Point", "recharge", 4814, "telecom", 10, 1000, 30000, 70000, 8, 22, False, "BLR"),
]
MERCHANT_COLUMNS = [
    "merchant_id", "name", "declared_category", "declared_mcc", "category_group",
    "declared_ticket_min", "declared_ticket_max", "declared_monthly_min",
    "declared_monthly_max", "declared_busy_start", "declared_busy_end",
    "festival_eligible", "declared_catchment",
]
# Settlement account per merchant. Only M10a and M10b share one (same owner).
SHARED_ACCOUNTS = {"M10a": "ACC-OWNER-1", "M10b": "ACC-OWNER-1"}

GROUND_TRUTH = [
    ("M1", "normal bakery", False),
    ("M2", "genuine festival spike", False),
    ("M3", "betting bakery (fixture-linked)", True),
    ("M4a", "split operator (bakery side, shared bettors)", True),
    ("M4b", "split operator (recharge side, shared bettors)", True),
    ("M5a", "campus canteen (shared students)", False),
    ("M5b", "campus stationery (shared students)", False),
    ("M7", "genuine match-night pizza shop", False),
    ("M8", "festival-hiding front (round amounts, repeat payers)", True),
    ("M9", "evasive festival front (varied amounts, one-time payers)", True),
    ("M10a", "split operator (bakery side, linked by settlement account)", True),
    ("M10b", "split operator (recharge side, linked by settlement account)", True),
]


def main():
    calendar = build_calendar()
    windows = fixture_windows(calendar)
    all_days = [config.START_DATE + timedelta(days=d) for d in range(config.WEEKS * 7)]
    fest_first = config.FESTIVAL_WEEK * 7
    festival_days = all_days[fest_first:fest_first + 7]
    fest_lo = config.START_DATE + timedelta(weeks=config.FESTIVAL_WEEK)
    fest_hi = fest_lo + timedelta(days=7)
    festival_windows = [w for w in windows if w[0] < fest_hi and w[1] > fest_lo]
    weekdays_only = [d for d in all_days if d.weekday() < 5]
    fixture_dates = {(w[0] + timedelta(hours=config.WINDOW_BEFORE_HOURS)).date() for w in windows}

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

    # M4a + M4b: split operator. Each shop looks mild alone, but they share one bettor pool.
    shared_bettors = make_pool("SPL", SPLIT_POOL, "BLR")
    pool = make_pool("M4AP", 150, "BLR")
    parts.append(make_txns("M4a", random_times(SPLIT_NORMAL, all_days, 7, 20), pool, varied_amounts(SPLIT_NORMAL, 40, 450)))
    pool = make_pool("M4BP", 150, "BLR")
    parts.append(make_txns("M4b", random_times(SPLIT_NORMAL, all_days, 8, 22), pool, varied_amounts(SPLIT_NORMAL, 10, 500)))
    for mid in ("M4a", "M4b"):
        n_in = int(round(SPLIT_BETS * SPLIT_IN_WINDOW))
        times = times_in_windows(n_in, windows) + random_times(SPLIT_BETS - n_in, all_days, 10, 24)
        parts.append(make_txns(mid, times, shared_bettors, rng.choice([100, 200, 500], size=SPLIT_BETS)))

    # M5a + M5b: campus canteen and stationery. Same students, no gambling pattern.
    students = make_pool("STU", 150, "BLR", p_home=1.0)
    own = make_pool("M5AP", 100, "BLR", p_home=1.0)
    parts.append(make_txns("M5a", random_times(300, weekdays_only, 9, 17), students, varied_amounts(300, 20, 300)))
    parts.append(make_txns("M5a", random_times(200, weekdays_only, 9, 17), own, varied_amounts(200, 20, 300)))
    own = make_pool("M5BP", 180, "BLR", p_home=1.0)
    parts.append(make_txns("M5b", random_times(120, weekdays_only, 9, 17), students, varied_amounts(120, 10, 300)))
    parts.append(make_txns("M5b", random_times(180, weekdays_only, 9, 17), own, varied_amounts(180, 10, 300)))

    # M7: genuine pizza shop. Open 17:00-23:00 every day, ~40% busier on fixture evenings.
    pool = make_pool("M7P", 600, "BLR")
    pizza_times = []
    for day in all_days:
        orders = rng.poisson(8 * (1.4 if day.date() in fixture_dates else 1.0))
        pizza_times += random_times(int(orders), [day], 17, 23)
    parts.append(make_txns("M7", pizza_times, pool, varied_amounts(len(pizza_times), 150, 900)))

    # M8 / M9: sweet shops with a festival-week surge AND bets hidden inside it.
    for mid, bettor_prefix, n_bettors, round_amounts in (("M8", "M8B", 40, True), ("M9", "M9B", 150, False)):
        pool = make_pool(f"{mid}P", 700, "BLR")
        parts.append(make_txns(mid, random_times(300, all_days, 9, 21), pool, varied_amounts(300, 100, 800)))
        parts.append(make_txns(mid, random_times(300, festival_days, 10, 21), pool, varied_amounts(300, 200, 1500)))
        bettors = make_pool(bettor_prefix, n_bettors, "BLR")
        amounts = rng.choice([100, 200, 500, 1000], size=150) if round_amounts else varied_amounts(150, 120, 900)
        parts.append(make_txns(mid, times_in_windows(150, festival_windows), bettors, amounts))

    # M10a + M10b: split operator with SEPARATE bettor pools, linked only by a shared
    # settlement account. Tests the deterministic link (payer overlap finds nothing).
    pool = make_pool("M10AP", 150, "BLR")
    parts.append(make_txns("M10a", random_times(SPLIT_NORMAL, all_days, 7, 20), pool, varied_amounts(SPLIT_NORMAL, 40, 450)))
    pool = make_pool("M10BP", 150, "BLR")
    parts.append(make_txns("M10b", random_times(SPLIT_NORMAL, all_days, 8, 22), pool, varied_amounts(SPLIT_NORMAL, 10, 500)))
    for mid in ("M10a", "M10b"):
        bettors = make_pool(f"{mid.upper()}B", SPLIT_POOL, "BLR")
        n_in = int(round(SPLIT_BETS * SPLIT_IN_WINDOW))
        times = times_in_windows(n_in, windows) + random_times(SPLIT_BETS - n_in, all_days, 10, 24)
        parts.append(make_txns(mid, times, bettors, rng.choice([100, 200, 500], size=SPLIT_BETS)))

    payments = pd.concat(parts, ignore_index=True)

    # Refunds (added last, so the payments above stay identical to earlier versions).
    refund_parts = []
    for mid, g in payments.groupby("merchant_id"):
        refund_parts.append(make_refunds(g, REFUND_RATES.get(mid, NORMAL_REFUND_RATE)))

    transactions = pd.concat([payments] + refund_parts, ignore_index=True).sort_values("timestamp")
    transactions.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(transactions) + 1)])

    merchants = pd.DataFrame(MERCHANTS, columns=MERCHANT_COLUMNS)
    merchants["settlement_account_id"] = [
        SHARED_ACCOUNTS.get(m, f"ACC-{m}") for m in merchants["merchant_id"]]
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
    """Quick sanity check printed after every run (payments only)."""
    pay = transactions[transactions["txn_type"] == "payment"]
    refunds = transactions[transactions["txn_type"] == "refund"]
    in_window = pd.Series(False, index=pay.index)
    for start, end in windows:
        in_window |= (pay["timestamp"] >= start) & (pay["timestamp"] <= end)
    pay = pay.assign(in_window=in_window)
    print(f"{'merchant':<9}{'txns':>6}{'total Rs':>11}{'payers':>8}{'in-window %':>13}{'refund %':>10}")
    for mid, g in pay.groupby("merchant_id", sort=False):
        share = 100 * g.loc[g["in_window"], "amount"].sum() / g["amount"].sum()
        refund = 100 * refunds.loc[refunds["merchant_id"] == mid, "amount"].sum() / g["amount"].sum()
        print(f"{mid:<9}{len(g):>6}{g['amount'].sum():>11,}{g['payer_id'].nunique():>8}"
              f"{share:>12.0f}%{refund:>9.0f}%")
    print(f"\nWrote files to ./{config.DATA_DIR}/")


if __name__ == "__main__":
    main()