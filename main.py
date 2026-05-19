# runs all the scripts in order so nothing has to be run manually
#   1. build_flakeflagger.py - pulls out the static and run-based features
#   2. build_smells.py       - grabs the test smell features (needed for the baseline)
#   3. build_idflakies.py    - static features from iDFlakies for cross-project testing
#   4. model_training.py     - trains everything and saves the results
#
# just run: python main.py

import runpy
import sys
import os

# (name to print, path to the script)
STEPS = [
    ("Feature extraction (FlakeFlagger)", "src/features/build/build_flakeflagger.py"),
    ("Smell feature extraction",          "src/features/build/build_smells.py"),
    ("iDFlakies feature extraction",      "src/features/build/build_idflakies.py"),
    ("Model training + comparison",       "src/model_training.py"),
]

def run_step(label, path):
    print(f"\n--- {label} ({path}) ---")
    runpy.run_path(path, run_name="__main__")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for label, path in STEPS:
        try:
            run_step(label, path)
        except Exception as e:
            print(f"\n[ERROR] {label} failed: {e}")
            print("fix whatever broke above and try running this again")
            sys.exit(1)

    print("\n--- pipeline complete ---")
    print("  results/tables/model_metrics.csv")
    print("  results/tables/cross_project_metrics.csv")
