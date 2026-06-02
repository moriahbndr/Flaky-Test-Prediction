###############################################################################################################
# Static feature extraction from the iDFlakies dataset.
# Mirrors the static feature set in build_flakeflagger.py so that a model trained on FlakeFlagger
# can be evaluated on iDFlakies for cross-project generalization.
#
# main differences from the FlakeFlagger pipeline:
#   - static features are the only thing being extracted
#   - all entries in iDFlakies are confirmed flaky (IsFlaky = 1)
#   - Test name format: "packageName.ClassName.methodName" (split on last '.')
#   - Project is derived from the GitHub URL: "{owner}-{repo}"
#
# Outputting to: data/processed/idflakies_features.csv
###############################################################################################################

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_features import apply_static_features


def main():
    df = pd.read_csv("data/input_data/idFlakies_dataset.csv")

    IDFLAKIES_TEST_COL = "Fully-Qualified Test Name (packageName.ClassName.methodName)"

    # obtaining class, function and test names
    df["Class"]    = df[IDFLAKIES_TEST_COL].apply(lambda x: x.rsplit(".", 1)[0] if "." in str(x) else str(x))
    df["Function"] = df[IDFLAKIES_TEST_COL].apply(lambda x: x.rsplit(".", 1)[1] if "." in str(x) else "")
    df["Test"]     = df[IDFLAKIES_TEST_COL].astype(str)

    # short project identifier from the GitHub URL: "owner-repo"
    df["Project"] = df["Project URL"].apply(
        lambda url: "-".join(str(url).rstrip("/").split("/")[-2:]) if "/" in str(url) else str(url)
    )

    # All iDFlakies entries are confirmed flaky
    df["IsFlaky"] = 1

    apply_static_features(df)

    final_cols = [
        "Project", "Test", "IsFlaky",
        "Category",   # keep iDFlakies flakiness category for reference / subgroup analysis

        # Static — name lengths and structure
        "FunctionNameLength", "FunctionWordCount", "FunctionHasDigits",
        "ClassNameLength", "PackageLength",

        # Static — keywords in function name
        "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
        "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
        "UIBrowserInFunction", "RetryFlakeInFunction", "TestOrderInFunction",

        # Static — keywords in class name
        "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
        "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass", "TestOrderInClass",

        # Static — keywords in package path
        "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",
    ]

    df = df.dropna(subset=["Test", "Function"])[final_cols]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/idflakies_features.csv", index=False)

    print("Saved to: data/processed/idflakies_features.csv")


if __name__ == "__main__":
    main()
