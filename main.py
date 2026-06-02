import runpy
import sys
import os

STEPS = [
    ("Label verification",               "src/label_verification.py"),
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
