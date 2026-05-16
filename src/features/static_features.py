###############################################################################################################
# Shared static feature extraction logic.
# Used by both build_flakeflagger.py and build_idflakies.py.
#
# To add or change a keyword category, edit _KW here — both pipelines pick it up automatically.
###############################################################################################################

def _has_any(text, words):
    t = text.lower()
    return int(any(w in t for w in words))

_KW = {
    "sleep_wait":  ["sleep", "wait", "delay", "timeout"],
    "async":       ["async", "concurrent", "thread", "parallel"],
    "time_random": ["time", "random", "date", "timer"],
    "network":     ["http", "socket", "network", "request", "api", "url", "rest", "endpoint", "connect"],
    "fileio":      ["file", "io", "disk", "path", "directory", "read", "write", "stream"],
    "database":    ["database", "db", "sql", "query", "transaction", "persist", "dao", "repository"],
    "ui_browser":  ["ui", "browser", "selenium", "screen", "click", "render", "dom", "page"],
    "retry_flake": ["retry", "flaky", "unstable", "intermittent", "eventually"],
}


def apply_static_features(df):
    """
    Add all static name-derived features to df in-place.
    Requires df to have 'Class' and 'Function' string columns.
    Returns df.
    """
    pkg = df["Class"].apply(lambda x: x.rsplit(".", 1)[0] if "." in x else "")

    # Name lengths
    df["FunctionNameLength"] = df["Function"].apply(len)
    df["ClassNameLength"]    = df["Class"].apply(len)
    df["PackageLength"]      = df["Class"].apply(lambda x: x.count("."))

    # Function name structure
    df["FunctionWordCount"] = df["Function"].apply(
        lambda x: len([c for c in x if c.isupper()]) + 1 if x else 0)  # camelCase word count proxy
    df["FunctionHasDigits"] = df["Function"].apply(lambda x: int(any(c.isdigit() for c in x)))

    # Keywords in function name
    df["SleepOrWaitInFunction"]  = df["Function"].apply(lambda x: _has_any(x, _KW["sleep_wait"]))
    df["AsyncInFunction"]        = df["Function"].apply(lambda x: _has_any(x, _KW["async"]))
    df["TimeOrRandomInFunction"] = df["Function"].apply(lambda x: _has_any(x, _KW["time_random"]))
    df["NetworkInFunction"]      = df["Function"].apply(lambda x: _has_any(x, _KW["network"]))
    df["FileIOInFunction"]       = df["Function"].apply(lambda x: _has_any(x, _KW["fileio"]))
    df["DatabaseInFunction"]     = df["Function"].apply(lambda x: _has_any(x, _KW["database"]))
    df["UIBrowserInFunction"]    = df["Function"].apply(lambda x: _has_any(x, _KW["ui_browser"]))
    df["RetryFlakeInFunction"]   = df["Function"].apply(lambda x: _has_any(x, _KW["retry_flake"]))

    # Keywords in class name
    df["SleepOrWaitInClass"]  = df["Class"].apply(lambda x: _has_any(x, _KW["sleep_wait"]))
    df["AsyncInClass"]        = df["Class"].apply(lambda x: _has_any(x, _KW["async"]))
    df["TimeOrRandomInClass"] = df["Class"].apply(lambda x: _has_any(x, _KW["time_random"]))
    df["NetworkInClass"]      = df["Class"].apply(lambda x: _has_any(x, _KW["network"]))
    df["FileIOInClass"]       = df["Class"].apply(lambda x: _has_any(x, _KW["fileio"]))
    df["DatabaseInClass"]     = df["Class"].apply(lambda x: _has_any(x, _KW["database"]))
    df["UIBrowserInClass"]    = df["Class"].apply(lambda x: _has_any(x, _KW["ui_browser"]))

    # Keywords in package path (computed locally — not added as a df column)
    df["NetworkInPackage"]   = pkg.apply(lambda x: _has_any(x, _KW["network"]))
    df["FileIOInPackage"]    = pkg.apply(lambda x: _has_any(x, _KW["fileio"]))
    df["DatabaseInPackage"]  = pkg.apply(lambda x: _has_any(x, _KW["database"]))
    df["UIBrowserInPackage"] = pkg.apply(lambda x: _has_any(x, _KW["ui_browser"]))

    return df
