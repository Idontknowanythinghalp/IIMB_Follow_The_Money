"""scoring.py - turns transactions into one results row per merchant (v2).

Used by the dashboard:   from scoring import load_data, score_all
Run it directly to test: python scoring.py

v2 has: calendar signal (matched-hours baseline that ignores festival days),
payer pattern (10 sub-metrics, also computed on the fixture-window slice),
festival check, two-key rule, network bonus (settlement-account link OR payer
overlap), confidence, ladder.
Not built yet: session module (s_session is 0).

IMPORTANT: score_all() must never read ground_truth.csv. Only the test report
at the bottom of this file does, and only to compare results afterwards.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import config

EPOCH = pd.Timestamp("2000-01-01")
DAY = 86400.0


# ------------------------------------------------------------------ loading
def load_data(folder=None):
    folder = folder or config.DATA_DIR
    transactions = pd.read_csv(f"{folder}/transactions.csv", parse_dates=["timestamp"])
    if "txn_type" not in transactions.columns:
        transactions["txn_type"] = "payment"
    merchants = pd.read_csv(f"{folder}/merchants.csv")
    calendar = pd.read_csv(f"{folder}/calendar.csv", parse_dates=["start", "end"])
    return transactions, merchants, calendar


# ------------------------------------------------------------------ helpers
def secs(ts):
    """A timestamp as seconds since 2000-01-01 (plain numbers are easy to compare)."""
    return (pd.Timestamp(ts) - EPOCH).total_seconds()


def scale(value, lo, hi):
    """Linear 0-100 score: 0 at `lo`, 100 at `hi`, clamped. None stays None (skipped)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if np.isinf(value):
        return 100.0
    return float(np.clip((value - lo) / (hi - lo), 0, 1) * 100)


def get_windows(calendar):
    """Fixture windows as (start_sec, end_sec), and the festival window (or None)."""
    windows = []
    for _, row in calendar[calendar["type"] == "fixture"].iterrows():
        windows.append((secs(row["start"] - timedelta(hours=config.WINDOW_BEFORE_HOURS)),
                        secs(row["end"] + timedelta(hours=config.WINDOW_AFTER_HOURS))))
    fest = calendar[calendar["type"] == "festival"]
    festival = (secs(fest.iloc[0]["start"]), secs(fest.iloc[0]["end"])) if len(fest) else None
    return windows, festival


def window_shifts(windows, festival):
    """For each fixture window: the day-shifts that give a clean 'same hours' baseline.
    Clean = inside the simulation, not touching any fixture window, not in the festival.
    Same weekday (shift divisible by 7) is preferred if at least 3 such days exist."""
    span = config.WEEKS * 7
    sim_lo = secs(config.START_DATE)
    sim_hi = secs(config.START_DATE + timedelta(days=span))
    out = []
    for a, b in windows:
        clean = []
        for k in range(-(span - 1), span):
            if k == 0:
                continue
            a2, b2 = a + k * DAY, b + k * DAY
            if a2 < sim_lo or b2 > sim_hi:
                continue
            if any(a2 <= wb and b2 >= wa for wa, wb in windows):
                continue
            if festival and a2 <= festival[1] and b2 >= festival[0]:
                continue
            clean.append(k)
        same_weekday = [k for k in clean if k % 7 == 0]
        out.append(same_weekday if len(same_weekday) >= 3 else clean)
    return out


# ---------------------------------------------------------- calendar signal
def suspect_payers(sub):
    """Payers who look bet-like inside a slice: any round-amount payment, or 2+ payments.
    One-time customers paying varied amounts drop out, so a big normal customer
    base cannot dilute the payer-overlap measure."""
    repeats = sub["payer_id"].map(sub["payer_id"].value_counts())
    return set(sub.loc[(sub["amount"] % 100 == 0) | (repeats >= 2), "payer_id"])


def calendar_signal(g, windows, festival, shifts):
    """Returns (ratio, bet-like payers inside fixture windows, bet-like payers in the baseline hours).
    ratio = money inside fixture windows / expected money in the same clock hours
    on clean non-fixture days. In windows that overlap the festival, only
    repeat-payer or round-amount transactions count toward the actual figure."""
    t = (g["timestamp"] - EPOCH).dt.total_seconds().to_numpy()
    payer = g["payer_id"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(g["amount"].to_numpy(dtype=float))])

    def span(a, b):
        return np.searchsorted(t, a, "left"), np.searchsorted(t, b, "right")

    total_actual = total_expected = 0.0
    window_payers, baseline_payers = set(), set()
    n_in_windows = 0
    for (a, b), ks in zip(windows, shifts):
        lo, hi = span(a, b)
        n_in_windows += hi - lo
        window_payers.update(suspect_payers(g.iloc[lo:hi]))
        if festival and a <= festival[1] and b >= festival[0]:
            sub = g.iloc[lo:hi]
            repeats = sub["payer_id"].map(sub["payer_id"].value_counts())
            keep = (sub["amount"] % 100 == 0) | (repeats >= 2)
            actual = float(sub.loc[keep, "amount"].sum())
        else:
            actual = cum[hi] - cum[lo]
        if not ks:
            continue
        expected = []
        for k in ks:
            lo2, hi2 = span(a + k * DAY, b + k * DAY)
            expected.append(cum[hi2] - cum[lo2])
            baseline_payers.update(suspect_payers(g.iloc[lo2:hi2]))
        total_actual += actual
        total_expected += float(np.mean(expected))

    if n_in_windows < config.CAL_MIN_WINDOW_TXNS:
        ratio = 1.0  # too few transactions in the windows to say anything (avoids noise)
    elif total_expected == 0:
        ratio = np.inf if total_actual > 0 else 1.0
    else:
        ratio = total_actual / total_expected
    return ratio, window_payers, baseline_payers


# ------------------------------------------------------------- payer pattern
def payer_metrics(g, m, festival, full=True):
    """Raw values for the sub-metrics (None = cannot be computed, skipped).
    full=False is used for the fixture-window slice: trend, volume and refund
    metrics only make sense on the whole period, so they are skipped there."""
    amounts = g["amount"]
    days = config.WEEKS * 7
    raw = {}

    by_payer = g.groupby("payer_id")["amount"].sum().sort_values(ascending=False)
    k = max(1, int(np.ceil(config.TOP_PAYER_FRACTION * len(by_payer))))
    raw["top_payer_share"] = by_payer.iloc[:k].sum() / by_payer.sum()
    raw["round_share"] = float((amounts % 100 == 0).mean())
    raw["ticket_out_of_range"] = float(((amounts < m["declared_ticket_min"]) |
                                        (amounts > m["declared_ticket_max"])).mean())
    raw["amount_clustering"] = float(amounts.value_counts().head(3).sum() / len(amounts))

    # share of transactions outside the declared busy hours
    start, end = int(m["declared_busy_start"]), int(m["declared_busy_end"])
    hour = g["timestamp"].dt.hour
    if start == 0 and end >= 24:
        raw["off_hours_share"] = 0.0
    elif end > start:
        raw["off_hours_share"] = float(((hour < start) | (hour >= end)).mean())
    else:  # overnight opening hours
        raw["off_hours_share"] = float(((hour >= end) & (hour < start)).mean())

    # payers outside the declared catchment (skipped for national sellers / no data)
    raw["outside_catchment"] = None
    if "payer_region" in g.columns and str(m["declared_catchment"]).lower() != "national":
        regions = g.dropna(subset=["payer_region"]).groupby("payer_id")["payer_region"].first()
        if len(regions):
            raw["outside_catchment"] = float((regions != m["declared_catchment"]).mean())

    raw["volume_vs_declared"] = raw["volume_vs_trend"] = raw["ticket_vs_trend"] = None
    if full:
        half = config.START_DATE + timedelta(weeks=config.WEEKS // 2)
        first = g[g["timestamp"] < half]
        last = g[g["timestamp"] >= half]
        monthly = amounts.sum() / (days / 30)
        raw["volume_vs_declared"] = monthly / m["declared_monthly_max"]
        raw["volume_vs_trend"] = (last["amount"].sum() / first["amount"].sum()
                                  if first["amount"].sum() > 0 else np.inf)
        if len(first) and len(last):
            raw["ticket_vs_trend"] = last["amount"].mean() / first["amount"].mean()

        # festival check: a genuine festival surge explains away the two VOLUME metrics
        if festival is not None and bool(m["festival_eligible"]):
            t = (g["timestamp"] - EPOCH).dt.total_seconds()
            outside = g.loc[~((t >= festival[0]) & (t <= festival[1])), "amount"].sum()
            fest_days = (festival[1] - festival[0]) / DAY
            monthly_outside = outside / ((days - fest_days) / 30)
            if monthly_outside / m["declared_monthly_max"] <= 1.0:
                raw["volume_vs_declared"] = raw["volume_vs_trend"] = None
    return raw


def payer_score(raw):
    """Average of the PAYER_TOP_K strongest sub-scores (skipped ones ignored)."""
    scores = {}
    for name, value in raw.items():
        lo, hi = config.PAYER_SCALES[name]
        s = scale(value, lo, hi)
        if s is not None:
            scores[name] = s
    top = sorted(scores.values(), reverse=True)[:config.PAYER_TOP_K]
    return (float(np.mean(top)) if top else 0.0), scores


# ------------------------------------------------------------------- ladder
def ladder(final_score, confidence):
    rung, name = 0, "Green"
    for minimum, r, n in config.LADDER:
        if final_score >= minimum:
            rung, name = r, n
            break
    if confidence == "low" and rung > config.LOW_CONFIDENCE_MAX_RUNG:
        rung = config.LOW_CONFIDENCE_MAX_RUNG
        name = next(n for _, r, n in config.LADDER if r == rung)
    return rung, name


def shows_signal(s_calendar, s_payer, s_session):
    """For the network check one module at/above the threshold is enough:
    the link (shared account or shared bet-like payers) supplies the second key."""
    return max(s_calendar, s_payer, s_session) >= config.TWO_KEY_THRESHOLD


# ------------------------------------------------------------ network bonus
def find_links(info):
    """Pairs of merchants that strengthen each other. `info` maps merchant_id to a dict.
    Conditions (all must hold): both show a signal, category groups differ, and
    either they share a settlement account OR they share payers inside fixture
    windows (overlap high, enough shared payers, and clearly above the baseline)."""
    links = []
    ids = list(info)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            x, y = info[a], info[b]
            if not (x["gambling_like"] and y["gambling_like"]):
                continue
            if x["group"] == y["group"]:
                continue
            if x["account"] and x["account"] == y["account"]:
                links.append((a, b, "shared settlement account"))
                continue
            shared = x["w_payers"] & y["w_payers"]
            smaller = min(len(x["w_payers"]), len(y["w_payers"]))
            if smaller == 0 or len(shared) < config.NETWORK_MIN_SHARED:
                continue
            overlap = len(shared) / smaller
            base_small = min(len(x["b_payers"]), len(y["b_payers"]))
            base_overlap = len(x["b_payers"] & y["b_payers"]) / base_small if base_small else 0.0
            if overlap >= config.NETWORK_OVERLAP_MIN and (
                    base_overlap == 0 or overlap >= config.NETWORK_LIFT * base_overlap):
                links.append((a, b, f"shared payers ({overlap:.0%} overlap)"))
    return links


# ------------------------------------------------------------- main function
def score_all(transactions, merchants, calendar):
    """One row per merchant. The dashboard reads only this table."""
    windows, festival = get_windows(calendar)
    shifts = window_shifts(windows, festival)
    rows, info = {}, {}

    for _, m in merchants.iterrows():
        mid = m["merchant_id"]
        all_g = transactions[transactions["merchant_id"] == mid]
        g = all_g[all_g["txn_type"] == "payment"].sort_values("timestamp").reset_index(drop=True)
        refunds = all_g.loc[all_g["txn_type"] == "refund", "amount"].sum()
        n = len(g)
        row = {"merchant_id": mid, "name": m["name"], "txns": n,
               "network_bonus": 0.0, "network_partner": ""}

        if n < config.MIN_TXNS_TO_SCORE:
            row.update(s_calendar=0.0, s_payer=0.0, s_session=0.0, drift=0.0,
                       confidence="insufficient")
            rows[mid] = row
            continue

        ratio, w_payers, b_payers = calendar_signal(g, windows, festival, shifts)
        s_calendar = scale(ratio, *config.CAL_RATIO_SCALE)

        raw = payer_metrics(g, m, festival, full=True)
        raw["refund_share"] = float(refunds / g["amount"].sum())
        s_payer, subs = payer_score(raw)

        # same payer-pattern measures on the fixture-window slice only (fights dilution)
        in_window = np.zeros(n, dtype=bool)
        t = (g["timestamp"] - EPOCH).dt.total_seconds().to_numpy()
        for a, b in windows:
            in_window |= (t >= a) & (t <= b)
        if in_window.sum() >= config.MIN_SLICE_TXNS:
            slice_raw = payer_metrics(g[in_window], m, None, full=False)
            slice_raw["refund_share"] = None
            s_payer = max(s_payer, payer_score(slice_raw)[0])

        s_session = 0.0  # session module not built yet (add-on)
        drift = max(s_calendar, s_payer, s_session)
        # two-key rule: a calendar spike alone is capped
        if (s_calendar >= config.TWO_KEY_THRESHOLD and s_payer < config.TWO_KEY_THRESHOLD
                and s_session < config.TWO_KEY_THRESHOLD):
            drift = min(drift, config.TWO_KEY_CAP)

        row.update(s_calendar=round(s_calendar, 1), s_payer=round(s_payer, 1),
                   s_session=s_session, drift=round(drift, 1),
                   confidence="normal" if n >= config.MIN_TXNS_CONFIDENT else "low",
                   cal_ratio=round(float(ratio), 2) if np.isfinite(ratio) else ratio)
        row.update({f"sub_{k}": round(v, 1) for k, v in subs.items()})
        rows[mid] = row
        info[mid] = {"group": m["category_group"], "w_payers": w_payers, "b_payers": b_payers,
                     "account": str(m.get("settlement_account_id", "") or ""),
                     "gambling_like": shows_signal(s_calendar, s_payer, s_session)}

    # network booster: add bonus points to both merchants of every link
    for a, b, reason in find_links(info):
        for mine, other in ((a, b), (b, a)):
            rows[mine]["network_bonus"] = config.NETWORK_BONUS
            rows[mine]["network_partner"] = (rows[mine]["network_partner"] + "; " if rows[mine]["network_partner"] else "") + f"{other} ({reason})"

    for row in rows.values():
        final = min(100.0, row["drift"] + row["network_bonus"])
        rung, rung_name = ladder(final, row["confidence"])
        row.update(final_score=round(final, 1), rung=rung, rung_name=rung_name)

    first = ["merchant_id", "name", "txns", "s_calendar", "s_payer", "s_session", "drift",
             "network_bonus", "final_score", "confidence", "rung", "rung_name", "network_partner"]
    df = pd.DataFrame(list(rows.values()))
    return df[first + [c for c in df.columns if c not in first]]


# ------------------------------------------------- test report (not the scorer)
def print_report(results, folder=None):
    """Compares results with ground_truth.csv. TEST ONLY - never used for scoring."""
    folder = folder or config.DATA_DIR
    truth = pd.read_csv(f"{folder}/ground_truth.csv")
    merged = results.merge(truth, on="merchant_id")
    merged["scenario"] = merged["scenario"].str.slice(0, 34)
    cols = ["merchant_id", "scenario", "s_calendar", "s_payer", "network_bonus",
            "final_score", "rung_name"]
    print(merged[cols].to_string(index=False))
    flagged = merged[merged["rung"] >= 2]
    honest = merged[~merged["is_front"]]
    fronts = merged[merged["is_front"]]
    print(f"\nHonest merchants at 'Ask' or above: {int((~flagged['is_front']).sum())} of {len(honest)}")
    print(f"Fronts at 'Ask' or above:           {int(flagged['is_front'].sum())} of {len(fronts)}")


if __name__ == "__main__":
    tx, mer, cal = load_data()
    result = score_all(tx, mer, cal)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(result.drop(columns=["network_partner"]).to_string(index=False))
    print("\nNetwork links:")
    for _, r in result[result["network_partner"] != ""].iterrows():
        print(f"  {r['merchant_id']}: {r['network_partner']}")
    print()
    print_report(result)