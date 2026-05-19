# this pulls features out of the FlakeFlagger dataset and saves them to a csv
# two kinds of features:
#   static  - derived from the test/class/package name alone
#   dynamic - derived from how many times the test passed or failed

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_features import apply_static_features

df = pd.read_csv("data/input_data/flakeflagger_results.csv")

df["Project"] = df["Project"].astype(str)
df["Test"]    = df["Test"].astype(str)

# assigning the numeric data in the columns to their names
numeric_cols = [
    "IsFlaky", "NumFailingRuns", "NumPassingRuns",
    "FirstFailingRunID", "FirstPassingRunID", "UniqueFailingExceptionTypes"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# obtaining class and function names
df["Class"]    = df["Test"].apply(lambda x: x.split("#")[0] if "#" in x else x)
df["Function"] = df["Test"].apply(lambda x: x.split("#")[1] if "#" in x else "")


# --- static features --- #

apply_static_features(df)

# --- lightweight dynamic features --- #

df["TotalRuns"]   = df["NumFailingRuns"] + df["NumPassingRuns"]
df["FailRatio"]   = df["NumFailingRuns"] / df["TotalRuns"].replace(0, 1)
df["PassRatio"]   = df["NumPassingRuns"] / df["TotalRuns"].replace(0, 1)
df["AnyFailures"]    = (df["NumFailingRuns"] > 0).astype(int)
df["AnyPassingRuns"] = (df["NumPassingRuns"] > 0).astype(int)
df["BothPassAndFail"] = ((df["NumPassingRuns"] > 0) & (df["NumFailingRuns"] > 0)).astype(int)
df["NeverPassed"] = ((df["NumFailingRuns"] > 0) & (df["NumPassingRuns"] == 0)).astype(int)
df["FailOnFirstRun"] = (df["FirstFailingRunID"] == 1).astype(int)
df["EarlyFailure"] = ((df["AnyFailures"] == 1) & (df["FirstFailingRunID"] <= 3)).astype(int)
df["ExceptionDiversityRatio"] = df["UniqueFailingExceptionTypes"] / df["NumFailingRuns"].replace(0, 1)
df.loc[df["NumFailingRuns"] == 0, "ExceptionDiversityRatio"] = 0
df["FailPassOdds"]    = df["NumFailingRuns"] / df["NumPassingRuns"].replace(0, 1)
df["FailPassDiff"]    = df["NumFailingRuns"] - df["NumPassingRuns"]
df["AbsFailPassDiff"] = df["FailPassDiff"].abs()
df["FlakinessBalance"] = 1 - (df["FailRatio"] - 0.5).abs() * 2
df["NeverFails"] = (df["NumFailingRuns"] == 0).astype(int)
df["LogNumFailingRuns"] = np.log1p(df["NumFailingRuns"])
df["LogTotalRuns"]      = np.log1p(df["TotalRuns"])


# --- output --- #
# if theres a better order/organization for our final cols feel free to change it

final_cols = [
    "Project", "Test", "IsFlaky",

    # static — name lengths and structure
    "FunctionNameLength", "FunctionWordCount", "FunctionHasDigits",
    "ClassNameLength", "PackageLength",

    # static — keywords in function name
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
    "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
    "UIBrowserInFunction", "RetryFlakeInFunction", "TestOrderInFunction",

    # static — keywords in class name
    "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
    "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass", "TestOrderInClass",

    # static — keywords in package path
    "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",

    # lightweight dynamic — raw dataset columns
    "NumFailingRuns", "NumPassingRuns",
    "FirstFailingRunID", "FirstPassingRunID", "UniqueFailingExceptionTypes",

    # lightweight dynamic — derived
    "TotalRuns", "FailRatio", "PassRatio",
    "AnyFailures", "AnyPassingRuns",
    "BothPassAndFail", "NeverPassed",
    "FailOnFirstRun", "EarlyFailure",
    "ExceptionDiversityRatio",
    "FailPassOdds", "FailPassDiff", "AbsFailPassDiff",
    "FlakinessBalance", "NeverFails",
    "LogNumFailingRuns", "LogTotalRuns",
]

df = df[final_cols].dropna()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/full_features.csv", index=False)

print("Saved to: data/processed/full_features.csv")
