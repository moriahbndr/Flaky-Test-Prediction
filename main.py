###############################################################################################################
# Pipeline entry point — runs the full project in order:
#   1. src/features/build_flakeflagger.py — extract static + lightweight dynamic features
#   2. src/features/build_smells.py  — fetch Java source and extract test-smell features (smell_only baseline)
#   3. src/model_training.py       — train all experiments and compare against smell_only baseline
#
# Run from the project root:
#   python main.py
###############################################################################################################

import runpy
import sys
import os

STEPS = [
    ("Feature extraction (FlakeFlagger)", "src/features/build_flakeflagger.py"),
    ("Smell feature extraction",          "src/features/build_smells.py"),
    ("iDFlakies feature extraction",      "src/features/build_idflakies.py"),
    ("Model training + comparison",       "src/model_training.py"),
]

def run_step(label, path):
    print(f"\n{'#'*60}")
    print(f"  STEP: {label}")
    print(f"  FILE: {path}")
    print(f"{'#'*60}\n")
    runpy.run_path(path, run_name="__main__")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for label, path in STEPS:
        try:
            run_step(label, path)
        except Exception as e:
            print(f"\n[ERROR] {label} failed: {e}")
            print("Fix the error above and re-run main.py.")
            sys.exit(1)

    print(f"\n{'#'*60}")
    print("  Pipeline complete.")
    print("  Results saved to: results/tables/model_metrics.csv")
    print("  Cross-project results: results/tables/cross_project_metrics.csv")
    print(f"{'#'*60}\n")
