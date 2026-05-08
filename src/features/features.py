###############################################################################################################
# Extracts FlakeFlagger's static test-smell features for every test in the FlakeFlagger dataset.
#
# Source: the FlakeFlagger repo already contains each test method as an individual .java file under
#   flakiness-predicter/input_data/original_tests/{project}/{flakyMethods|nonFlakyMethods}/
# We fetch those files directly from raw.githubusercontent.com — no GitHub token required.
#
# Features extracted (mirrors FlakeFlaggerFeaturesTypes.csv static column):
#   assertion_roulette, conditional_test_logic, eager_test, fire_and_forget,
#   indirect_testing, mystery_guest, resource_optimism, test_run_war,
#   num_asserts, test_length
#
# Files are cached locally so re-runs are instant.
# Output: data/processed/flakeflagger_features.csv
###############################################################################################################

import requests
import pandas as pd
import re
import os
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# FlakeFlagger repo raw-file base URL                                         #
# --------------------------------------------------------------------------- #
FF_BASE = (
    "https://raw.githubusercontent.com/AlshammariA/FlakeFlagger"
    "/master/flakiness-predicter/input_data/original_tests"
)

# A handful of projects use a slightly different folder name in the FF repo
FF_PROJECT_NAME_MAP = {
    "jknack-handlebars.java": "jknack-handlebars",
}

CACHE_DIR = Path("data/source_cache/ff_methods")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Fetch a single test-method file from the FlakeFlagger repo                  #
# --------------------------------------------------------------------------- #

def fetch_method_source(project, class_name, method_name, is_flaky):
    """
    Download the test method's .java file from the FlakeFlagger repo.
    Files live at:
      {FF_BASE}/{project}/{flakyMethods|nonFlakyMethods}/{class_name}-{method_name}.java
    Returns source text, or None if the file does not exist.
    """
    ff_project = FF_PROJECT_NAME_MAP.get(project, project)
    folder     = "flakyMethods" if is_flaky else "nonFlakyMethods"
    filename   = f"{class_name}-{method_name}.java"

    cache_path = CACHE_DIR / ff_project / folder / filename
    miss_path  = cache_path.with_suffix(".miss")

    if miss_path.exists():
        return None
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"{FF_BASE}/{ff_project}/{folder}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        time.sleep(0.05)   # polite delay — raw.githubusercontent.com has no strict rate limit
    except requests.RequestException:
        return None

    if resp.status_code == 200:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        return resp.text

    miss_path.parent.mkdir(parents=True, exist_ok=True)
    miss_path.touch()
    return None


# --------------------------------------------------------------------------- #
# Test-smell detectors                                                         #
# Each function receives the full file content (which IS the method body)     #
# and returns an int (0 / 1) or a count.                                      #
# --------------------------------------------------------------------------- #

def smell_assertion_roulette(body):
    """Multiple assertions with no descriptive failure message."""
    all_asserts = re.findall(r'\bassert\w*\s*\(', body)
    if len(all_asserts) < 2:
        return 0
    with_msg = len(re.findall(r'\bassert\w*\s*\(\s*"', body))
    return int((len(all_asserts) - with_msg) > 1)


def smell_conditional_test_logic(body):
    """if / for / while / switch inside the test."""
    return int(bool(re.search(r'\b(if|for|while|switch)\s*\(', body)))


def smell_eager_test(body):
    """Test calls more than 5 distinct obj.method() combinations."""
    calls = re.findall(r'\b\w+\s*\.\s*\w+\s*\(', body)
    return int(len(set(calls)) > 5)


def smell_fire_and_forget(body):
    """Async operation started without a join / get / await."""
    has_async = bool(re.search(
        r'new\s+Thread\s*\(|\.execute\s*\(|\.submit\s*\(|'
        r'CompletableFuture\.(run|supply)Async', body
    ))
    has_wait = bool(re.search(r'\.(join|get|await|waitFor)\s*\(', body))
    return int(has_async and not has_wait)


def smell_indirect_testing(body):
    """No direct assertions; test relies on chained / delegated calls."""
    asserts = len(re.findall(r'\bassert\w*\s*\(', body))
    chains  = len(re.findall(r'\w+\s*\(\s*\)\s*\.\s*\w+', body))
    return int(asserts == 0 or chains > 2)


def smell_mystery_guest(body):
    """Test accesses files, databases, or network resources."""
    return int(bool(re.search(
        r'\b(File|FileInputStream|FileOutputStream|Socket|ServerSocket|'
        r'URL|HttpClient|HttpURLConnection|Connection|DataSource|DriverManager)\b', body
    )))


def smell_resource_optimism(body):
    """Test creates a File without checking existence first."""
    has_file  = bool(re.search(r'\bnew\s+File\s*\(', body))
    has_check = bool(re.search(r'\.exists\s*\(\s*\)|\.isFile\s*\(\s*\)', body))
    return int(has_file and not has_check)


def smell_test_run_war(body):
    """Test modifies shared / global state."""
    return int(bool(re.search(
        r'System\.(setProperty|clearProperty|setOut|setErr)|'
        r'Runtime\.getRuntime\s*\(\s*\)', body
    )))


def count_num_asserts(body):
    return len(re.findall(r'\bassert\w*\s*\(', body))


def count_test_length(body):
    return sum(1 for line in body.splitlines() if line.strip())


# --------------------------------------------------------------------------- #
# Combined extraction for one method file                                      #
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    "assertion_roulette", "conditional_test_logic", "eager_test",
    "fire_and_forget", "indirect_testing", "mystery_guest",
    "resource_optimism", "test_run_war",
    "num_asserts", "test_length",
]


def extract_features(body):
    return {
        "assertion_roulette":     smell_assertion_roulette(body),
        "conditional_test_logic": smell_conditional_test_logic(body),
        "eager_test":             smell_eager_test(body),
        "fire_and_forget":        smell_fire_and_forget(body),
        "indirect_testing":       smell_indirect_testing(body),
        "mystery_guest":          smell_mystery_guest(body),
        "resource_optimism":      smell_resource_optimism(body),
        "test_run_war":           smell_test_run_war(body),
        "num_asserts":            count_num_asserts(body),
        "test_length":            count_test_length(body),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    df = pd.read_csv("data/input_data/flakeflagger_results.csv")
    df["Class"]    = df["Test"].apply(lambda x: x.split("#")[0] if "#" in x else x)
    df["Function"] = df["Test"].apply(lambda x: x.split("#")[1] if "#" in x else "")

    total = len(df)
    print(f"Processing {total} tests — fetching from FlakeFlagger repo (no token required)...")

    rows  = []
    found = 0

    for i, row in df.iterrows():
        body = fetch_method_source(
            project     = row["Project"],
            class_name  = row["Class"],
            method_name = row["Function"],
            is_flaky    = row["IsFlaky"],
        )

        base = {"Project": row["Project"], "Test": row["Test"]}
        if body is not None:
            base.update(extract_features(body))
            found += 1
        else:
            base.update({c: None for c in FEATURE_COLS})

        rows.append(base)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total}  ({found} matched so far)")

    result = pd.DataFrame(rows)
    pct = 100 * found // total
    print(f"\nFeatures extracted for {found}/{total} rows ({pct}%)")

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv("data/processed/flakeflagger_features.csv", index=False)
    print("Saved to: data/processed/flakeflagger_features.csv")


if __name__ == "__main__":
    main()
