"""app.py - Streamlit dashboard for the Follow the Money prototype.

Run:  streamlit run app.py
Needs the data/ folder made by generate_data.py.

The dashboard only VIEWS results: scoring.py decides the scores, this file shows them.
ground_truth.csv is used only for the optional "presenter view" (our answer key).

Ask-first ladder: the evidence supports a step (from the score), but a first
detection never goes past "Ask merchant". Escalation needs "no response", and
"merchant explains" steps back down.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
from scoring import load_data, score_all

RUNG_ICON = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "⛔"}
RUNG_NAME = {rung: name for _, rung, name in config.LADDER}
FIRST_DETECTION_CAP = 2  # a first detection goes no higher than "Ask merchant"

SUB_LABELS = {
    "top_payer_share": "A few payers give most of the money",
    "round_share": "Round-number payments",
    "ticket_out_of_range": "Payments outside the declared ticket range",
    "amount_clustering": "The same few amounts, again and again",
    "off_hours_share": "Payments outside the declared opening hours",
    "outside_catchment": "Payers outside the declared area",
    "volume_vs_declared": "Volume above the declared level",
    "volume_vs_trend": "Volume jump compared with its own history",
    "ticket_vs_trend": "Average payment jump compared with its own history",
    "refund_share": "High share of refunds",
    "peer_top_payer_share": "Payer concentration compared with peers",
    "peer_round_share": "Round-number payments compared with peers",
    "peer_txns_per_payer": "Payer repetition compared with peers",
}
SESS_LABELS = {
    "payers_with_session": "Payers with a burst (3+ payments within 2 hours)",
    "escalating_sessions": "Bursts where the amount escalates",
    "late_night_share": "Payments between 21:00 and 02:00",
}

st.set_page_config(page_title="Follow the Money", layout="wide")


# ------------------------------------------------------------------- data
@st.cache_data
def load_everything():
    tx, mer, cal = load_data()
    return tx, mer, cal, score_all(tx, mer, cal)


def load_truth():
    try:
        return pd.read_csv(f"{config.DATA_DIR}/ground_truth.csv")
    except FileNotFoundError:
        return None


def show_chart(chart):
    """Altair chart that fills the width (the parameter name changed between Streamlit versions)."""
    for options in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.altair_chart(chart, **options)
            return
        except TypeError:
            continue


def score_bars(values):
    """Horizontal bars on a fixed 0-100 axis, with the score written in each label."""
    scores = pd.DataFrame({"Measure": [f"{name} ({value:.0f})" for name, value in values.items()],
                           "Score": list(values.values())})
    chart = alt.Chart(scores).mark_bar().encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title=None),
        y=alt.Y("Measure:N", sort="-x", title=None, axis=alt.Axis(labelLimit=500)),
    ).properties(height=36 * len(scores))
    show_chart(chart)


def reload_data():
    st.cache_data.clear()
    st.session_state.clear()


# ------------------------------------------------------------ ladder state
def init_state(results):
    ids = list(results["merchant_id"])
    if "step" not in st.session_state or set(st.session_state["step"]) != set(ids):
        st.session_state["step"] = {r.merchant_id: min(int(r.rung), FIRST_DETECTION_CAP)
                                    for r in results.itertuples()}
        st.session_state["log"] = {m: [] for m in ids}


def merchant_explains(mid):
    state = st.session_state["step"]
    before = state[mid]
    state[mid] = max(before - 1, 0)
    st.session_state["log"][mid].append(
        f"Merchant explained, and we accepted it: {RUNG_NAME[before]} to {RUNG_NAME[state[mid]]}")


def no_response(mid, recommended):
    state = st.session_state["step"]
    before = state[mid]
    state[mid] = min(before + 1, recommended)
    st.session_state["log"][mid].append(
        f"No response or explanation rejected: {RUNG_NAME[before]} to {RUNG_NAME[state[mid]]}")


def reset_merchant(mid, recommended):
    st.session_state["step"][mid] = min(recommended, FIRST_DETECTION_CAP)
    st.session_state["log"][mid] = []


# ---------------------------------------------------------------- overview
def tab_overview(results, truth, presenter):
    steps = st.session_state["step"]
    current = results["merchant_id"].map(steps)
    cols = st.columns(5)
    cols[0].metric("Merchants", len(results))
    for rung, col in zip((0, 1, 2, 3), cols[1:]):
        col.metric(f"{RUNG_ICON[rung]} {RUNG_NAME[rung]}", int((current == rung).sum()))

    table = pd.DataFrame({
        "": current.map(RUNG_ICON),
        "Merchant": results["merchant_id"] + "  " + results["name"],
        "Score": results["final_score"],
        "Current step": current.map(RUNG_NAME),
        "Evidence supports": results["rung"].map(RUNG_NAME),
        "Confidence": results["confidence"],
        "Network": results["network_partner"],
    })
    if presenter and truth is not None:
        flags = results["merchant_id"].map(dict(zip(truth["merchant_id"], truth["is_front"])))
        table["Answer key"] = flags.map({True: "front", False: "honest"})
    st.dataframe(table.set_index("Merchant"))
    st.caption("Current step starts at 'Ask merchant' at most. Use the Merchant detail tab "
               "to simulate the merchant explaining, or not responding.")


# ------------------------------------------------------------------ detail
def why_lines(row):
    lines = []
    ratio = row.get("cal_ratio")
    if row["s_calendar"] >= 30 and ratio is not None and np.isfinite(ratio):
        lines.append(f"Money inside fixture windows is {ratio:.1f}x the usual level for the "
                     f"same hours on other days (calendar score {row['s_calendar']:.0f}).")
    subs = {k[4:]: row[k] for k in row.index if k.startswith("sub_") and pd.notna(row[k])}
    for key, value in sorted(subs.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        if value >= 20:
            lines.append(f"{SUB_LABELS[key]}: {value:.0f} / 100")
    if row["network_bonus"] > 0:
        lines.append(f"Linked with {row['network_partner']}: +{row['network_bonus']:.0f} points.")
    return lines or ["Nothing unusual compared with the declared profile and its own history."]


def daily_chart(pay, cal):
    daily = (pay.groupby(pay["timestamp"].dt.floor("D"))["amount"].sum()
             .rename("inflow").reset_index().rename(columns={"timestamp": "day"}))
    fixtures = cal[cal["type"] == "fixture"].copy()
    fixtures["start"] = fixtures["start"].dt.floor("D")
    fixtures["end"] = fixtures["start"] + pd.Timedelta(days=1)
    festival = cal[cal["type"] == "festival"][["start", "end"]]
    bars = alt.Chart(daily).mark_bar().encode(
        x=alt.X("day:T", title=None), y=alt.Y("inflow:Q", title="Rupees per day"))
    fixture_layer = alt.Chart(fixtures[["start", "end"]]).mark_rect(
        opacity=0.25, color="#e8863a").encode(x="start:T", x2="end:T")
    festival_layer = alt.Chart(festival).mark_rect(
        opacity=0.15, color="#2e9e5b").encode(x="start:T", x2="end:T")
    return alt.layer(festival_layer, fixture_layer, bars).properties(height=260)


def hour_profile(pay, cal):
    fixture_days = set(cal.loc[cal["type"] == "fixture", "start"].dt.floor("D"))
    all_days = config.WEEKS * 7
    day = pay["timestamp"].dt.floor("D")
    hour = pay["timestamp"].dt.hour
    on_fixture = day.isin(fixture_days)
    profile = pd.DataFrame({
        "Fixture days": pay[on_fixture].groupby(hour[on_fixture])["amount"].sum() / max(len(fixture_days), 1),
        "Other days": pay[~on_fixture].groupby(hour[~on_fixture])["amount"].sum() / max(all_days - len(fixture_days), 1),
    }).reindex(range(24)).fillna(0)
    profile.index.name = "Hour of day"
    return profile


def tab_detail(results, tx, cal):
    labels = {r.merchant_id: f"{r.merchant_id}  {r.name}" for r in results.itertuples()}
    mid = st.selectbox("Merchant", list(labels), format_func=labels.get)
    row = results[results["merchant_id"] == mid].iloc[0]
    recommended = int(row["rung"])
    step = st.session_state["step"][mid]

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{RUNG_ICON[step]} {RUNG_NAME[step]}")
        st.write(f"The evidence supports: **{RUNG_NAME[recommended]}**  ·  confidence: **{row['confidence']}**")
        m = st.columns(5)
        m[0].metric("Calendar", f"{row['s_calendar']:.0f}")
        m[1].metric("Payer pattern", f"{row['s_payer']:.0f}")
        m[2].metric("Session", f"{row['s_session']:.0f}")
        m[3].metric("Network bonus", f"+{row['network_bonus']:.0f}")
        m[4].metric("Final score", f"{row['final_score']:.0f}")
        st.markdown("**Why**")
        for line in why_lines(row):
            st.write("- " + line)
    with right:
        st.markdown("**Ask-first ladder**")
        st.button("Merchant explains (accepted)", on_click=merchant_explains, args=(mid,))
        st.button("No response / explanation rejected", on_click=no_response, args=(mid, recommended),
                  disabled=step >= recommended)
        st.button("Reset this merchant", on_click=reset_merchant, args=(mid, recommended))
        if step >= recommended and recommended > 0:
            st.caption("Already at the step the evidence supports.")
        for entry in st.session_state["log"][mid]:
            st.write("- " + entry)

    pay = tx[(tx["merchant_id"] == mid) & (tx["txn_type"] == "payment")]
    st.markdown("**Money in per day** (orange = fixture days, green = festival week)")
    show_chart(daily_chart(pay, cal))
    st.markdown("**Money in by hour of day** (average per day; fixture days vs other days, approximate)")
    st.line_chart(hour_profile(pay, cal))

    st.markdown("**Context**")
    flag = row.get("declaration_flag")
    if isinstance(flag, str) and flag:
        st.warning("Onboarding flag: " + flag + ".")
    peers = row.get("peer_group")
    if pd.notna(peers) and peers >= config.PEER_MIN_SIZE:
        st.write(f"- Compared with {int(peers)} peer merchants in the same category (MCC).")
    else:
        st.write("- Too few peers in this category, so fixed baselines are used.")
    if pd.notna(row.get("cluster_size")) and row["cluster_size"] > 1:
        st.write(f"- Part of a linked group of {int(row['cluster_size'])} merchants.")

    subs = {SUB_LABELS[k[4:]]: row[k] for k in row.index if k.startswith("sub_") and pd.notna(row[k])}
    if subs:
        st.markdown("**Payer-pattern sub-scores** (0 = normal, 100 = extreme; the strongest three form the payer score)")
        st.caption(f"Computed on: {row.get('payer_basis', 'whole period')}")
        if max(subs.values()) == 0:
            st.write("All sub-scores are 0: this merchant looks normal on every payer-pattern measure.")
        score_bars(subs)

    sess = {SESS_LABELS[k[5:]]: row[k] for k in row.index if k.startswith("sess_") and pd.notna(row[k])}
    if sess:
        st.markdown("**Session (velocity) sub-scores**")
        if row["s_session"] == 0:
            st.write("No bursts of repeat payments from the same payer: the session score is 0.")
        else:
            score_bars(sess)


# ----------------------------------------------------------------- network
def tab_network(results, tx, mer):
    st.write("Two merchants are linked only if **both show a signal**, their categories **differ**, "
             "and they share a **settlement account** or **bet-like payers inside fixture windows**. "
             "Shared customers alone never link anyone.")
    pay = tx[tx["txn_type"] == "payment"]
    payers = {m: set(g["payer_id"]) for m, g in pay.groupby("merchant_id")}
    info = results.set_index("merchant_id")
    signal = info[["s_calendar", "s_payer", "s_session"]].max(axis=1) >= config.TWO_KEY_THRESHOLD
    group = mer.set_index("merchant_id")["category_group"]
    account = mer.set_index("merchant_id")["settlement_account_id"] if "settlement_account_id" in mer else None

    rows, ids = [], list(info.index)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = len(payers.get(a, set()) & payers.get(b, set()))
            same_account = account is not None and account[a] == account[b]
            if shared < 10 and not same_account:
                continue
            linked = b in str(info.loc[a, "network_partner"])
            rows.append({
                "Pair": f"{a} + {b}",
                "Shared payers (whole period)": shared,
                "Same settlement account": "yes" if same_account else "no",
                "Categories differ": "yes" if group[a] != group[b] else "no",
                "Both show a signal": "yes" if (signal[a] and signal[b]) else "no",
                "Linked": "LINKED" if linked else "no",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Pair"))
    else:
        st.info("No merchant pairs share enough payers or a settlement account.")
    st.caption("Pairs like the campus canteen and stationery shop share many students but show no "
               "signal, so they are not linked. That is the false-positive protection.")


# -------------------------------------------------------------------- main
def main():
    st.title("Follow the Money: merchant drift monitor")
    st.caption("Synthetic data. Every threshold is a placeholder (see config.py).")

    try:
        tx, mer, cal, results = load_everything()
    except FileNotFoundError:
        st.error("No data found. Run `python generate_data.py` first, then reload this page.")
        st.stop()

    init_state(results)
    truth = load_truth()

    with st.sidebar:
        st.button("Reload data and reset", on_click=reload_data)
        presenter = st.checkbox("Presenter view (show answer key)", value=False)
        show_bg = st.checkbox("Include the background (peer) merchants", value=False)
        with st.expander("The ladder"):
            for minimum, rung, name in sorted(config.LADDER, key=lambda x: x[1]):
                st.write(f"{RUNG_ICON[rung]} {name}  (score {minimum}+)")
            st.caption("A first detection goes no higher than 'Ask merchant'.")

    view = results if show_bg else results[~results["merchant_id"].str.startswith("BG")]
    tab1, tab2, tab3 = st.tabs(["Overview", "Merchant detail", "Network"])
    with tab1:
        tab_overview(view, truth, presenter)
    with tab2:
        tab_detail(view, tx, cal)
    with tab3:
        tab_network(results, tx, mer)


main()