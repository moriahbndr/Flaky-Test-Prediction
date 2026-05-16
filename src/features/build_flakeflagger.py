###############################################################################################################
# Feature extraction from the FlakeFlagger dataset.
# Output: data/processed/full_features_v2.csv
#
# Static features  — derivable from test/class/package name alone
# Lightweight dynamic features — derived from existing FlakeFlagger run-outcome columns
###############################################################################################################

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_features import apply_static_features

df = pd.read_csv("data/input_data/flakeflagger_results.csv")

df["Project"] = df["Project"].astype(str)
df["Test"]    = df["Test"].astype(str)

numeric_cols = [
    "IsFlaky", "NumFailingRuns", "NumPassingRuns",
    "FirstFailingRunID", "FirstPassingRunID", "UniqueFailingExceptionTypes"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Class"]    = df["Test"].apply(lambda x: x.split("#")[0] if "#" in x else x)
df["Function"] = df["Test"].apply(lambda x: x.split("#")[1] if "#" in x else "")


# ==================== Static Features ==================== #

apply_static_features(df)


# ==================== Lightweight Dynamic Features ==================== #

df["TotalRuns"]   = df["NumFailingRuns"] + df["NumPassingRuns"]
df["FailRatio"]   = df["NumFailingRuns"] / df["TotalRuns"].replace(0, 1)
df["PassRatio"]   = df["NumPassingRuns"] / df["TotalRuns"].replace(0, 1)
df["AnyFailures"]    = (df["NumFailingRuns"] > 0).astype(int)
df["AnyPassingRuns"] = (df["NumPassingRuns"] > 0).astype(int)

# BothPassAndFail: passed at least once AND failed at least once — the core flakiness signal
# A truly flaky test must show both outcomes.
df["BothPassAndFail"] = ((df["NumPassingRuns"] > 0) & (df["NumFailingRuns"] > 0)).astype(int)

# AlwaysFails: never passed — likely a real (consistent) failure, not flaky
df["AlwaysFails"] = ((df["NumFailingRuns"] > 0) & (df["NumPassingRuns"] == 0)).astype(int)

# FailOnFirstRun: failure appeared on the very first run
df["FailOnFirstRun"] = (df["FirstFailingRunID"] == 1).astype(int)

# EarlyFailure: failure appeared within the first 3 runs
df["EarlyFailure"] = ((df["AnyFailures"] == 1) & (df["FirstFailingRunID"] <= 3)).astype(int)

# ExceptionDiversityRatio: how varied are the exception types per failing run?
# High ratio → many different exception types, potentially more environment-sensitive
df["ExceptionDiversityRatio"] = (
    df["UniqueFailingExceptionTypes"] / df["NumFailingRuns"].replace(0, 1)
).where(df["NumFailingRuns"] > 0, other=0)


# ==================== Output ==================== #

final_cols = [
    "Project", "Test", "IsFlaky",

    # Static — name lengths and structure
    "FunctionNameLength", "FunctionWordCount", "FunctionHasDigits",
    "ClassNameLength", "PackageLength",

    # Static — keywords in function name
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
    "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
    "UIBrowserInFunction", "RetryFlakeInFunction",

    # Static — keywords in class name
    "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
    "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass",

    # Static — keywords in package path
    "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",

    # Lightweight dynamic — raw dataset columns
    "NumFailingRuns", "NumPassingRuns",
    "FirstFailingRunID", "FirstPassingRunID", "UniqueFailingExceptionTypes",

    # Lightweight dynamic — derived
    "TotalRuns", "FailRatio", "PassRatio",
    "AnyFailures", "AnyPassingRuns",
    "BothPassAndFail", "AlwaysFails",
    "FailOnFirstRun", "EarlyFailure",
    "ExceptionDiversityRatio",
]

df = df[final_cols].dropna()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/full_features_v2.csv", index=False)

print("Saved to: data/processed/full_features_v2.csv")
print("Shape:", df.shape)
print("\nFeature count:", len(final_cols) - 3, "(excluding Project, Test, IsFlaky)")
print("\nSample:")
print(df.head())
