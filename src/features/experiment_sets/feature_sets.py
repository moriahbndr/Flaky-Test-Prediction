# organized the features into their sets for model training

# # to add a new static feature, add its column name in here
STATIC_COLS = [
    "FunctionNameLength", "FunctionWordCount", "FunctionHasDigits",
    "ClassNameLength", "PackageLength",
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
    "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
    "UIBrowserInFunction", "RetryFlakeInFunction", "TestOrderInFunction",
    "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
    "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass", "TestOrderInClass",
    "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",
]

# to add a new dynamic feature, add its column name here
DYNAMIC_COLS = [
    "TotalRuns",
    "LogTotalRuns",
    "FirstFailingRunID",
    "FirstPassingRunID",
    "EarlyFailure",
    "FailOnFirstRun",
    "ExceptionDiversityRatio",
]

FF_SMELL_COLS = [
    "assertion_roulette", "conditional_test_logic", "eager_test",
    "fire_and_forget", "indirect_testing", "mystery_guest",
    "resource_optimism", "test_run_war",
    "num_asserts", "test_length",
]

# baseline to be used by model - raw flake flagger data, smell only 
def build_baseline_sets(smell_cols=None):
    sets = {
        "flakeflagger_raw": [
            "NumFailingRuns", "NumPassingRuns",
            "FirstFailingRunID", "FirstPassingRunID",
            "UniqueFailingExceptionTypes",
        ],
    }
    if smell_cols:
        sets["smell_only"] = smell_cols
    return sets


def build_feature_sets(smell_cols=None):
    sets = {
        "static":              STATIC_COLS,
        "static_plus_dynamic": STATIC_COLS + DYNAMIC_COLS,
    }
    sets.update(build_baseline_sets(smell_cols))
    return sets
