# Follow the Money: merchant drift monitor (prototype v3)

https://idontknowanythinghalp-iimb-follow-the-money-app-sek8ob.streamlit.app/

Detects merchants whose payments do not match what they declared (the "bakery that takes bets" problem),
using payment patterns only, and responds in steps (ask first) so honest merchants can recover.
All data is synthetic. All thresholds are placeholders tuned on seed 42.

## Quick start

```
pip install pandas numpy streamlit altair      # check versions with pip before pinning
python generate_data.py                        # writes data/*.csv (fixed seed from config.py)
python scoring.py                              # prints the score table and a ground-truth report
streamlit run app.py                           # dashboard
python robustness.py 50 1                      # 50 fresh seeds (about 7 minutes on one core)
python blind.py make --worlds 10 --first-seed 9001 && python blind.py predict && python blind.py evaluate
```

## What each file does

| File | Role |
|---|---|
| `config.py` | Every tunable number (scales, thresholds, ladder, network rules). No logic. |
| `generate_data.py` | Builds the synthetic world: transactions, merchants, calendar, and `ground_truth.csv` (answer key, never read by the scorer). |
| `scoring.py` | `load_data()` and `score_all()`: one results row per merchant. Also a ground-truth test report when run directly. |
| `app.py` | Streamlit dashboard. Only views results. Applies the ask-first ladder state (a first detection never goes past Ask). |
| `robustness.py` | Regenerates and scores many seeds; reports detection and false-positive rates per merchant and per seed. |
| `blind.py` | Blind evaluation: anonymised held-out worlds, sealed labels, frozen predictions, report with confidence intervals. |

## How scoring works (per merchant)

1. **Calendar signal** (`s_calendar`): money inside fixture windows (3 h before start to 1 h after end) divided by the expected money in the same clock hours on clean non-fixture days. Ratio 1 scores 0, ratio 4 scores 100. In windows that overlap the festival, only repeat-payer or round-amount payments count.
2. **Payer pattern** (`s_payer`): sub-metrics such as top-payer share, round-amount share, amount clustering, tickets outside the declared range, off-hours share, payers outside the catchment, volume vs declared, volume and ticket trend, refund share. With 8 or more same-MCC peers, payer concentration and round share are replaced by robust z-scores against peers, and payer repetition is added. `s_payer` is the average of the 3 strongest sub-scores. It is also computed on the fixture-window slice only, and the higher result is used. Declared ticket and volume maxima are trusted only up to 2x the MCC default.
3. **Session module** (`s_session`): a session is 3 or more payments from one payer within 2 hours. Scored on share of payers with a session, escalating amounts, and late-night share.
4. **Festival check**: a festival-eligible merchant whose volume outside the festival is within its declared max has the two volume sub-metrics switched off. A festival never explains calendar, payer-concentration or session signals.
5. **Drift** = max(`s_calendar`, `s_payer`, `s_session`). **Two-key rule**: if only the calendar is at or above 30, drift is capped at 59 (Monitor).
6. **Confidence**: under 20 payments, not scored; 20 to 49, low (rung capped at Ask).
7. **Network booster**: two merchants are linked only if their category groups differ, one has a signal of at least 30 and the other at least 20, and either (a) they share a settlement account and BOTH have payer or session evidence of at least 30, or (b) they share at least 8 bet-like payers inside fixture windows, at least 60% of the smaller set and at least 1.5x the same-hours baseline overlap. Linked merchants get +25. A payer index and seed-and-expand search avoid comparing every pair.
8. **Final score** = min(100, drift + bonus), then the ladder: Green (below 30), Monitor (30), Ask merchant (60), Payout hold (75), Review / offboard (90).

## Synthetic merchants

 M1 normal bakery, M2 genuine Diwali sweet shop, M5a/M5b campus shops sharing students, M7 match-night pizza shop, M12a/M12b honest owner with two match-night shops on one settlement account, BG01 to BG60 background peers.
Fronts (should reach Ask or above): M3 fixture-linked betting bakery, M4a/M4b split operator (shared bettors), M10a/M10b split operator (shared settlement account), M8 festival-hiding front (round amounts), M9 evasive festival front (varied amounts), M6 casino sessions without fixtures, M11 inflated declaration.

## Testing and what the numbers mean

- `python scoring.py` on seed 42: tuned seed, so the weakest evidence.
- `robustness.py`: fresh seeds, same scenario designs. Last run (49 untuned seeds plus seed 42, M12 included): 0 of 3,283 honest merchant-runs at Ask or above; fronts at Ask or above averaged 7.8 of 9 per world.
- `blind.py`: fresh seeds plus jittered generator parameters, anonymised IDs, sealed labels. If `config.py` or `scoring.py` logic changes after `predict`, `evaluate` marks the report NOT BLIND. Comment and docstring edits do not break the seal. First run: 76 of 90 fronts and 0 of 670 honest at Ask or above. That set is burned if anyone tunes after seeing it.

## Known limits

- **M9 (evasive festival front) is never caught.** The festival-overlap rule that protects M2 also drops one-time varied-amount bettors. Fixing it needs a new signal, not a threshold change.
- **M12b is held under Ask by the two-key cap (59), one point below the line.** It is protected by the rule, not by margin.
- **Split-operator shops are mostly Ask, rarely Payout hold.**
- **Untested path:** the payer-overlap link does not require payer or session evidence on both sides, only a signal of 30 and 20. Two calendar-only merchants of different categories sharing repeat customers could link and gain +25. No scenario exercises this yet.
- The scenarios were designed by us, so fresh seeds are blind to the numbers, not to the ideas. M3b (smurfing) is in the spec but not built.
- Peer baselines include any fronts in the peer set. The median and MAD tolerate this here; a real deployment with many fronts would need care.
- The spec doc in Drive is still the v1 draft and disagrees with the code (see the spec change list).
