###############################################################################################################
# Static feature extraction from the iDFlakies dataset.
# Mirrors the static feature set in build_flakeflagger.py so that a model trained on FlakeFlagger
# can be evaluated on iDFlakies for cross-project generalization.
#
# main differences from the FlakeFlagger pipeline:
#   - iDFlakies has NO run-history columns → only static features are extracted
#   - ALL entries in iDFlakies are confirmed flaky (IsFlaky = 1)
#   - Test name format: "packageName.ClassName.methodName" (split on last '.')
#   - Project is derived from the GitHub URL: "{owner}-{repo}"
#
# Output: data/processed/idflakies_features.csv
###############################################################################################################

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_features import apply_static_features

df = pd.read_csv("data/input_data/idFlakies_dataset.csv")

TEST_COL = "Fully-Qualified Test Name (packageName.ClassName.methodName)"

# --------------------------------------------------------------------------- #
# Parse test name → Class and Function                                        #
# iDFlakies format: packageName.ClassName.methodName                          #
# Split on the last '.' — everything before is the class, last part is method #
# --------------------------------------------------------------------------- #
df["Class"]    = df[TEST_COL].apply(lambda x: x.rsplit(".", 1)[0] if "." in str(x) else str(x))
df["Function"] = df[TEST_COL].apply(lambda x: x.rsplit(".", 1)[1] if "." in str(x) else "")
df["Test"]     = df[TEST_COL].astype(str)

# Derive a short project identifier from the GitHub URL: "owner-repo"
df["Project"] = df["Project URL"].apply(
    lambda url: "-".join(str(url).rstrip("/").split("/")[-2:]) if "/" in str(url) else str(url)
)

# All iDFlakies entries are confirmed flaky
df["IsFlaky"] = 1

# ==================== Static Features ==================== #

apply_static_features(df)


# ==================== Output ==================== #

final_cols = [
    "Project", "Test", "IsFlaky",
    "Category",   # keep iDFlakies flakiness category for reference / subgroup analysis

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
]

df = df.dropna(subset=["Test", "Function"])[final_cols]

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/idflakies_features.csv", index=False)

print("Saved to: data/processed/idflakies_features.csv")
print("Shape:", df.shape)
print(f"\nUnique projects: {df['Project'].nunique()}")
print(f"Total tests: {len(df)}  (all IsFlaky=1)")
print("\nFlakiness category breakdown:")
print(df["Category"].value_counts().head(10))
print("\nSample:")
print(df[["Project", "Test", "Category"]].head(5).to_string(index=False))
