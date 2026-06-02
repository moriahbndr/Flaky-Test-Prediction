# Output:
#   results/tables/label_verification.csv  — all flagged rows with a
#   Reason column explaining which pattern triggered the flag.

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_label_verification():
    raw_path = "data/input_data/flakeflagger_results.csv"
    if not os.path.exists(raw_path):
        print(f"[Label verification] Raw data not found at {raw_path} — skipping.")
        return

    df = pd.read_csv(raw_path)

    df["IsFlaky"]        = pd.to_numeric(df["IsFlaky"],        errors="coerce")
    df["NumFailingRuns"] = pd.to_numeric(df["NumFailingRuns"], errors="coerce")
    df["NumPassingRuns"] = pd.to_numeric(df["NumPassingRuns"], errors="coerce")

    df = df.dropna(subset=["IsFlaky", "NumFailingRuns", "NumPassingRuns"])
    total = len(df)

    # --- flag 1: labeled flaky but never actually failed ---
    flag1 = df[(df["IsFlaky"] == 1) & (df["NumFailingRuns"] == 0)].copy()
    flag1["Reason"] = "labeled_flaky_never_failed"

    # --- flag 2: labeled non-flaky but had at least one failure ---
    flag2 = df[(df["IsFlaky"] == 0) & (df["NumFailingRuns"] > 0)].copy()
    flag2["Reason"] = "labeled_nonflaky_had_failures"

    # --- flag 3: labeled flaky but never passed ---
    flag3 = df[(df["IsFlaky"] == 1) & (df["NumPassingRuns"] == 0)].copy()
    flag3["Reason"] = "labeled_flaky_never_passed"

    # keep only the columns that are useful for review
    keep_cols = ["Project", "Test", "IsFlaky", "NumFailingRuns", "NumPassingRuns", "Reason"]
    flag1 = flag1[keep_cols]
    flag2 = flag2[keep_cols]
    flag3 = flag3[keep_cols]

    print(f"\n--------- Label Verification (FlakeFlagger) ---------")
    print(f"Total tests in dataset: {total}")
    print()
    print(f"  labeled_flaky_never_failed    : {len(flag1)} ({100 * len(flag1) / total:.2f}%)")
    print(f"  labeled_nonflaky_had_failures : {len(flag2)} ({100 * len(flag2) / total:.2f}%)")
    print(f"  labeled_flaky_never_passed    : {len(flag3)} ({100 * len(flag3) / total:.2f}%)")

    all_flagged = pd.concat([flag1, flag2, flag3], ignore_index=True)

    if len(all_flagged) > 0:
        os.makedirs("results/tables", exist_ok=True)
        all_flagged.to_csv("results/tables/label_verification.csv", index=False)
        print(f"\nFlagged rows saved to: results/tables/label_verification.csv")
    else:
        print("\nNo label inconsistencies found.")

run_label_verification()
