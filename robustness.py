"""robustness.py - re-generate the synthetic world for many seeds, score each, compare with truth.

Usage:  python robustness.py [n_seeds] [first_seed]      (default 40 seeds starting at 1)
Writes: robustness_by_merchant.csv, robustness_by_seed.csv, and prints a summary.

Thresholds were tuned on SEED=42, so 42 is reported separately as the 'tuned seed';
the headline numbers come from the OTHER seeds only.
"""
import contextlib
import importlib
import io
import os
import sys
import tempfile
from multiprocessing import Pool

import pandas as pd

import config


def run_seed(seed):
    folder = tempfile.mkdtemp(prefix=f"ftm_{seed}_")
    config.SEED = seed
    config.DATA_DIR = folder
    import generate_data
    importlib.reload(generate_data)          # rng is created at import time, so reload per seed
    with contextlib.redirect_stdout(io.StringIO()):
        generate_data.main()
    from scoring import load_data, score_all
    tx, mer, cal = load_data(folder)
    res = score_all(tx, mer, cal)
    truth = pd.read_csv(f"{folder}/ground_truth.csv")
    out = res.merge(truth, on="merchant_id")
    out["seed"] = seed
    keep = ["seed", "merchant_id", "scenario", "is_front", "s_calendar", "s_payer", "s_session",
            "network_bonus", "final_score", "rung", "rung_name", "network_partner", "cluster_size"]
    return out[keep]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seeds = list(range(first, first + n))
    if 42 not in seeds:
        seeds.append(42)
    with Pool(min(os.cpu_count() or 1, 8)) as pool:
        frames = pool.map(run_seed, seeds)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("robustness_by_merchant.csv", index=False)

    named = df[~df["merchant_id"].str.startswith("BG")]
    bg = df[df["merchant_id"].str.startswith("BG")]

    per_seed = []
    for s, g in df.groupby("seed"):
        fronts, honest = g[g["is_front"]], g[~g["is_front"]]
        per_seed.append({"seed": s,
                         "fronts_ask_plus": int((fronts["rung"] >= 2).sum()), "fronts": len(fronts),
                         "honest_ask_plus": int((honest["rung"] >= 2).sum()), "honest": len(honest),
                         "honest_monitor_plus": int((honest["rung"] >= 1).sum()),
                         "honest_named_ask_plus": int((honest[~honest["merchant_id"].str.startswith("BG")]["rung"] >= 2).sum())})
    ps = pd.DataFrame(per_seed)
    ps.to_csv("robustness_by_seed.csv", index=False)

    other = named[named["seed"] != 42]
    n_other = other["seed"].nunique()
    print(f"Seeds: {n_other} untuned + seed 42 (tuned)\n")
    rows = []
    for mid, g in named.groupby("merchant_id", sort=False):
        o = g[g["seed"] != 42]
        t = g[g["seed"] == 42]
        rows.append({"merchant": mid, "scenario": g["scenario"].iloc[0][:40], "front": bool(g["is_front"].iloc[0]),
                     "seed42_rung": int(t["rung"].iloc[0]) if len(t) else None,
                     "ask+_rate": round(float((o["rung"] >= 2).mean()), 2),
                     "hold+_rate": round(float((o["rung"] >= 3).mean()), 2),
                     "monitor+_rate": round(float((o["rung"] >= 1).mean()), 2),
                     "score_median": round(float(o["final_score"].median()), 1),
                     "score_min": round(float(o["final_score"].min()), 1),
                     "score_max": round(float(o["final_score"].max()), 1)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    o_ps = ps[ps["seed"] != 42]
    print("\nPer-seed (untuned seeds only):")
    print(f"  fronts at Ask+ : mean {o_ps['fronts_ask_plus'].mean():.2f} of {o_ps['fronts'].iloc[0]}"
          f"  (min {o_ps['fronts_ask_plus'].min()}, max {o_ps['fronts_ask_plus'].max()})")
    print(f"  honest at Ask+ : seeds with >=1 false positive = {(o_ps['honest_ask_plus'] > 0).sum()} of {len(o_ps)}"
          f"; total false positives = {o_ps['honest_ask_plus'].sum()} over {int(o_ps['honest'].sum())} honest merchant-runs")
    print(f"  named honest (M1,M2,M5a/b,M7,M12a/b) at Ask+: {o_ps['honest_named_ask_plus'].sum()} over "
          f"{n_other * 7} merchant-runs")
    bg_o = bg[bg["seed"] != 42]
    print(f"  background honest at Ask+: {(bg_o['rung'] >= 2).sum()} of {len(bg_o)}; at Monitor+: {(bg_o['rung'] >= 1).sum()}")

    m12 = named[named["merchant_id"].isin(["M12a", "M12b"])]
    linked = m12[m12["network_partner"].fillna("") != ""]
    print(f"  M12 pair falsely linked in {linked['seed'].nunique()} of {named['seed'].nunique()} seeds")


if __name__ == "__main__":
    main()
