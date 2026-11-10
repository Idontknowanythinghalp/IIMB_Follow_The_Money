"""blind.py - blind evaluation harness for the Follow the Money prototype.

Three steps, run in this order:

  python blind.py make     --worlds 10 --first-seed 9001     # build anonymised held-out worlds + sealed labels
  python blind.py predict                                     # score them; freezes hashes of config.py / scoring.py
  python blind.py evaluate                                    # opens the labels; refuses if the code changed

What "blind" means here (and what it does not):
  * The scorer sees anonymised IDs (X001...), anonymised payer and account IDs, no names, and NO labels.
    Labels live in blind/sealed/ and are only read by `evaluate`.
  * Every world uses a fresh seed AND jittered generator parameters (pool sizes, in-window share,
    festival week), so thresholds tuned on SEED=42 are tested against a shifted world.
  * `predict` records SHA-256 hashes of config.py, scoring.py and the predictions. `evaluate` checks them.
    If you edit thresholds after `predict`, the report is stamped "NOT BLIND" and you must `make` fresh worlds.
  * LIMIT: the scenarios (fronts and honest look-alikes) were still designed by us. Fresh seeds are blind to
    the numbers, not to the ideas. Real blindness to scenario design needs a teammate to write scenarios:
    use `python blind.py make --import-folder DIR` with a folder holding transactions.csv, merchants.csv,
    calendar.csv and ground_truth.csv (columns: merchant_id, scenario, is_front) made WITHOUT reading config.py.
"""
import argparse
import hashlib
import importlib
import io
import contextlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

ROOT = "blind"
SEALED = f"{ROOT}/sealed"
PRED = f"{ROOT}/predictions"
CODE_FILES = ["config.py", "scoring.py"]

FAMILY = {
    "M1": "normal", "M2": "honest: genuine festival", "M3": "front: fixture-linked bettor",
    "M4a": "front: split, shared payers", "M4b": "front: split, shared payers",
    "M5a": "honest: shared students", "M5b": "honest: shared students",
    "M7": "honest: match-night shop", "M8": "front: festival-hiding (round)",
    "M9": "front: festival-hiding (evasive)", "M10a": "front: split, shared account",
    "M10b": "front: split, shared account", "M6": "front: casino sessions",
    "M11": "front: inflated declaration", "M12a": "honest: owner pair, shared account",
    "M12b": "honest: owner pair, shared account",
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def family_of(true_id):
    return FAMILY.get(true_id, "normal (background)" if true_id.startswith("BG") else "unknown")


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------- make
def anonymise(folder_in, folder_out, rng):
    """Copy a data folder with merchant, payer, account IDs and names replaced. Returns the id map."""
    tx = pd.read_csv(f"{folder_in}/transactions.csv")
    mer = pd.read_csv(f"{folder_in}/merchants.csv")
    cal = pd.read_csv(f"{folder_in}/calendar.csv")
    ids = list(mer["merchant_id"])
    order = rng.permutation(len(ids))
    mid_map = {ids[i]: f"X{n + 1:03d}" for n, i in enumerate(order)}
    payers = tx["payer_id"].unique()
    pay_map = dict(zip(payers, [f"P{i:06d}" for i in rng.permutation(len(payers))]))
    accounts = mer["settlement_account_id"].unique()
    acc_map = dict(zip(accounts, [f"A{i:04d}" for i in rng.permutation(len(accounts))]))

    tx["merchant_id"] = tx["merchant_id"].map(mid_map)
    tx["payer_id"] = tx["payer_id"].map(pay_map)
    tx["txn_id"] = [f"T{i:06d}" for i in range(1, len(tx) + 1)]
    mer["merchant_id"] = mer["merchant_id"].map(mid_map)
    mer["name"] = "Merchant " + mer["merchant_id"]
    mer["settlement_account_id"] = mer["settlement_account_id"].map(acc_map)
    mer = mer.sort_values("merchant_id").reset_index(drop=True)

    os.makedirs(folder_out, exist_ok=True)
    tx.to_csv(f"{folder_out}/transactions.csv", index=False)
    mer.to_csv(f"{folder_out}/merchants.csv", index=False)
    cal.to_csv(f"{folder_out}/calendar.csv", index=False)
    return mid_map


def make_world(seed, out_name):
    """One jittered synthetic world, anonymised. Returns its parameters."""
    jr = np.random.default_rng(seed + 10_000)           # jitter RNG (separate from the generator's)
    params = {
        "seed": seed,
        "SPLIT_POOL": int(jr.integers(18, 36)),
        "SPLIT_BETS": int(jr.integers(30, 56)),
        "SPLIT_IN_WINDOW": round(float(jr.uniform(0.65, 0.90)), 2),
        "SPLIT_NORMAL": int(jr.integers(200, 321)),
        "FESTIVAL_WEEK": int(jr.integers(3, 7)),
    }
    raw = f"{ROOT}/_raw_{seed}"
    config.SEED, config.DATA_DIR, config.FESTIVAL_WEEK = seed, raw, params["FESTIVAL_WEEK"]
    import generate_data
    importlib.reload(generate_data)
    for k in ("SPLIT_POOL", "SPLIT_BETS", "SPLIT_IN_WINDOW", "SPLIT_NORMAL"):
        setattr(generate_data, k, params[k])
    with contextlib.redirect_stdout(io.StringIO()):
        generate_data.main()
    truth = pd.read_csv(f"{raw}/ground_truth.csv")
    return raw, truth, params


def cmd_make(args):
    if os.path.exists(ROOT) and not args.force:
        raise SystemExit(f"{ROOT}/ already exists. Use --force to overwrite (this destroys the old sealed labels).")
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(SEALED)
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "worlds": {}}
    jobs = []
    if args.import_folder:
        jobs.append(("imported", None))
    else:
        jobs += [(f"world_{s}", s) for s in range(args.first_seed, args.first_seed + args.worlds)]
    for name, seed in jobs:
        rng = np.random.default_rng((seed or 7) + 555)
        if seed is None:
            raw, truth, params = args.import_folder, pd.read_csv(f"{args.import_folder}/ground_truth.csv"), {}
        else:
            raw, truth, params = make_world(seed, name)
        mid_map = anonymise(raw, f"{ROOT}/{name}", rng)
        labels = truth.assign(anon_id=truth["merchant_id"].map(mid_map),
                              family=truth["merchant_id"].map(family_of))
        lp = f"{SEALED}/{name}_labels.csv"
        labels.to_csv(lp, index=False)
        manifest["worlds"][name] = {"labels_sha256": sha(lp), "params": params, "merchants": len(labels)}
        if seed is not None:
            shutil.rmtree(raw, ignore_errors=True)
        print(f"{name}: {len(labels)} merchants, {int(labels['is_front'].sum())} fronts")
    json.dump(manifest, open(f"{ROOT}/manifest.json", "w"), indent=2)
    print(f"\nSealed labels are in {SEALED}/. Do NOT open them until after `predict`.")


# ---------------------------------------------------------------- predict
def cmd_predict(args):
    manifest = json.load(open(f"{ROOT}/manifest.json"))
    from scoring import load_data, score_all
    os.makedirs(PRED, exist_ok=True)
    frozen = {"predicted": datetime.now(timezone.utc).isoformat(),
              "code_sha256": {f: sha(f) for f in CODE_FILES}, "predictions_sha256": {}}
    keep = ["merchant_id", "final_score", "rung", "rung_name", "confidence", "s_calendar", "s_payer",
            "s_session", "network_bonus", "network_partner", "cluster_size"]
    for name in manifest["worlds"]:
        folder = f"{ROOT}/{name}"
        leaked = [f for f in os.listdir(folder) if "truth" in f.lower() or "label" in f.lower()]
        if leaked:
            raise SystemExit(f"{folder} contains {leaked}: the scorer must not see labels.")
        tx, mer, cal = load_data(folder)
        res = score_all(tx, mer, cal)[keep]
        p = f"{PRED}/{name}.csv"
        res.to_csv(p, index=False)
        frozen["predictions_sha256"][name] = sha(p)
        print(f"{name}: scored {len(res)} merchants")
    json.dump(frozen, open(f"{ROOT}/predictions_manifest.json", "w"), indent=2)
    print("\nFrozen. Editing config.py or scoring.py now makes `evaluate` report NOT BLIND.")


# --------------------------------------------------------------- evaluate
def rate_line(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "n/a"


def cmd_evaluate(args):
    manifest = json.load(open(f"{ROOT}/manifest.json"))
    frozen = json.load(open(f"{ROOT}/predictions_manifest.json"))
    problems = []
    for f in CODE_FILES:
        if sha(f) != frozen["code_sha256"][f]:
            problems.append(f"{f} changed after predict")
    for name, h in frozen["predictions_sha256"].items():
        if sha(f"{PRED}/{name}.csv") != h:
            problems.append(f"predictions for {name} were edited")
        if sha(f"{SEALED}/{name}_labels.csv") != manifest["worlds"][name]["labels_sha256"]:
            problems.append(f"labels for {name} were edited")
    status = "BLIND" if not problems else "NOT BLIND: " + "; ".join(problems)

    frames = []
    for name in manifest["worlds"]:
        pr = pd.read_csv(f"{PRED}/{name}.csv")
        lb = pd.read_csv(f"{SEALED}/{name}_labels.csv")
        frames.append(pr.merge(lb, left_on="merchant_id", right_on="anon_id", suffixes=("", "_true")).assign(world=name))
    df = pd.concat(frames, ignore_index=True)
    df["network_partner"] = df["network_partner"].fillna("")
    fronts, honest = df[df["is_front"]], df[~df["is_front"]]
    lines = [f"# Blind evaluation: {status}", "",
             f"Worlds: {df['world'].nunique()}; merchant-runs: {len(df)} ({len(fronts)} front, {len(honest)} honest)", "",
             "## Headline (operative threshold = Ask merchant or above, rung >= 2)",
             f"- Fronts caught at Ask+ : {rate_line(int((fronts['rung'] >= 2).sum()), len(fronts))}",
             f"- Honest wrongly at Ask+: {rate_line(int((honest['rung'] >= 2).sum()), len(honest))}",
             f"- Honest wrongly at Monitor+: {rate_line(int((honest['rung'] >= 1).sum()), len(honest))}",
             f"- Honest wrongly at Payout hold+: {rate_line(int((honest['rung'] >= 3).sum()), len(honest))}",
             f"- Honest merchants given a network link: {rate_line(int((honest['network_partner'] != '').sum()), len(honest))}",
             "", "## By scenario family (Ask+ rate)", "", "| family | n | caught/flagged at Ask+ | median score |", "|---|---|---|---|"]
    for fam, g in df.groupby("family"):
        k = int((g["rung"] >= 2).sum())
        lo, hi = wilson(k, len(g))
        lines.append(f"| {fam} | {len(g)} | {k}/{len(g)} ({100 * lo:.0f}-{100 * hi:.0f}%) | {g['final_score'].median():.0f} |")
    miss = fronts[fronts["rung"] < 2].groupby("family").size()
    if len(miss):
        lines += ["", "## Missed fronts by family", ""] + [f"- {fam}: {k}" for fam, k in miss.items()]
    fps = honest[honest["rung"] >= 2]
    if len(fps):
        lines += ["", "## False positives (honest at Ask+)", ""] + [
            f"- {r.world} {r.merchant_id} ({r.family}): score {r.final_score}, {r.rung_name}" for r in fps.itertuples()]
    report = "\n".join(lines)
    open(f"{ROOT}/blind_report.md", "w").write(report)
    df.to_csv(f"{ROOT}/blind_results.csv", index=False)
    print(report)
    if problems:
        print("\n!! This run is NOT blind. Do not quote these numbers as held-out. Make fresh worlds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--worlds", type=int, default=10)
    m.add_argument("--first-seed", type=int, default=9001)
    m.add_argument("--import-folder")
    m.add_argument("--force", action="store_true")
    sub.add_parser("predict")
    sub.add_parser("evaluate")
    a = ap.parse_args()
    {"make": cmd_make, "predict": cmd_predict, "evaluate": cmd_evaluate}[a.cmd](a)
