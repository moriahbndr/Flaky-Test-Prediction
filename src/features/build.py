###############################################################################################################
# creating the static and lightweight dynamic features in this file from the FlakeFlagger dataset
# (adding/correctly locating iDFlakies next to reference and use for creating those features in the future)
# columns are produced with the results and are saved to .....
# data/processed/full_features.csv
###############################################################################################################

import pandas as pd
import os
# TODO: attempt to locate and assign the FirstFailingRunID from the FlakeFlagger dataset (located in flakeflagger_results.csv)

# pandas DataFrame taking FlakeFlagger dataset, making sure column names will be taken as string
df = pd.read_csv("data/input_data/flakeflagger_results.csv")
df["Project"] = df["Project"].astype(str)
df["Test"] = df["Test"].astype(str)

# the column names (the FlakeFlagger dataset), 
# putting names to the numeric columns, names are from data/flakeflagger_results.csv line 1
numeric_cols = [
    "IsFlaky", "NumFailingRuns", "NumPassingRuns", "FirstFailingRunID", "FirstPassingRunID","UniqueFailingExceptionTypes"
]


# looping each column and converting to numeric
# errors = "coerce" is needed to transform any potential bad values and avoid failures
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors = "coerce")


#splitting into class or function first
df["Class"] = df["Test"].apply(lambda x: x.split("#")[0] if "#" in x else x)
df["Function"] = df["Test"].apply(lambda x: x.split("#")[1] if "#" in x else "")

# TODO: add to and save to new folder and add a seperate test case/area file
# -------- Consolidating and Filtering Data to only retrieve (NumFailingRuns > 0) --------- #
filtered_df = df[df["NumFailingRuns"] > 0]
#print("\n\nNumOfFailingRuns new data: \n", filtered_df)
print("\n\nNumOfFailingRuns new data: \n", filtered_df[numeric_cols])


# -------- Static Features --------- #

#Function and Class name length
df["ClassNameLength"] = df["Class"].apply(len)
df["FunctionNameLength"] = df["Function"].apply(len)

#Package path length
df["PackageLength"] = df["Class"].apply(lambda x: x.count("."))

# -- keywords in function names -- # 
# moriah - I might have missed a few keywords for each, if so, feel free to add

df["SleepOrWaitInFunction"] = df["Function"].apply(
    lambda x: 1 if any(word in x.lower() for word in ["sleep", "wait", "delay", "timeout"]) else 0)

df["AsyncInFunction"] = df["Function"].apply(
    lambda x: 1 if any(word in x.lower() for word in ["async", "concurrent", "thread"]) else 0)

df["TimeOrRandomInFunction"] = df["Function"].apply(
    lambda x: 1 if any(word in x.lower() for word in ["time", "random"]) else 0)

# -------- Lightweight Dynamic Features --------- #
# data/flakeflagger_results.csv line 1 for naming references #
 
df["TotalRuns"] = df["NumFailingRuns"] + df["NumPassingRuns"]
df["FailRatio"] = df["NumFailingRuns"] / df["TotalRuns"].replace(0, 1)
df["PassRatio"] = df["NumPassingRuns"] / df["TotalRuns"].replace(0, 1)
df["AnyFailures"] = df["NumFailingRuns"].apply(lambda x: 1 if x > 0 else 0)
df["AnyPassingRuns"] = df["NumPassingRuns"].apply(lambda x: 1 if x > 0 else 0)

final_cols = [
    "Project","Test","IsFlaky",
    
    # Static columns
    "FunctionNameLength", "ClassNameLength", "PackageLength",
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
        
    #lightweight dyamic colums
    "NumFailingRuns","NumPassingRuns",
    "FirstFailingRunID", "FirstPassingRunID","UniqueFailingExceptionTypes",
    "TotalRuns","FailRatio","PassRatio",
    "AnyFailures","AnyPassingRuns"
]

df = df[final_cols].dropna()

# processed dataset saved - feature extraction pipeline
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/full_features.csv", index=False)

#Confirmation message
print("Processed results have been saved to: data/processed/full_features.csv")
print("Shape:", df.shape)
print(df.head())