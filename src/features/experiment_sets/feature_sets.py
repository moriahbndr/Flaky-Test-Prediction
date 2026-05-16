# Feature column lists and experiment definitions used by model_training.py.
# To add a new static feature: add it to static_features.py, then add the column name to STATIC_COLS.

STATIC_COLS = [
    "FunctionNameLength", "FunctionWordCount", "FunctionHasDigits",
    "ClassNameLength", "PackageLength",
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
    "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
    "UIBrowserInFunction", "RetryFlakeInFunction",
    "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
    "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass",
    "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",
]

FF_SMELL_COLS = [
    "assertion_roulette", "conditional_test_logic", "eager_test",
    "fire_and_forget", "indirect_testing", "mystery_guest",
    "resource_optimism", "test_run_war",
    "num_asserts", "test_length",
]

FF_CHURN_COLS = [
    "file_churn_window_5", "file_churn_window_10", "file_churn_window_25",
    "file_churn_window_50", "file_churn_window_75", "file_churn_window_100",
]


def build_feature_sets(available_smell_cols=None, available_churn_cols=None):
    """
    Build and return the FEATURE_SETS dictionary for model_training.py.
    smell_only and flakeflagger_static are only added when smell cols are available.
    """
    sets = {
        "flakeflagger_raw": [
            "NumFailingRuns", "NumPassingRuns",
            "FirstFailingRunID", "FirstPassingRunID",
            "UniqueFailingExceptionTypes",
        ],
        "static_v2": STATIC_COLS,
        "static_v2_plus_dynamic": STATIC_COLS + [
            "TotalRuns", "FailRatio", "PassRatio",
            "AnyFailures", "AnyPassingRuns",
            "BothPassAndFail", "AlwaysFails",
            "FailOnFirstRun", "EarlyFailure",
            "ExceptionDiversityRatio",
        ],
    }

    if available_smell_cols:
        sets["smell_only"] = available_smell_cols
        sets["flakeflagger_static"] = available_smell_cols + (available_churn_cols or [])

    return sets
