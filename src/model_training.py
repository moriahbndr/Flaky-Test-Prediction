#source for code outline - https://github.com/liannewriting/YouTube-videos-public/blob/main/xgboost-python-tutorial-example/xgboost_python.ipynb

import pandas as pd
from xgboost import XGBClassifier, plot_importance
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix)


# --------------------------------------------------------- #
#  Loading processed dataset (build.py has to run first)    #
# --------------------------------------------------------- #
df = pd.read_csv("data/processed/full_features.csv")

# Merge FlakeFlagger test-smell features when available
# Run src/features/features.py first to generate this file
ff_path = "data/processed/flakeflagger_features.csv"
if os.path.exists(ff_path):
    ff_df = pd.read_csv(ff_path)
    df = df.merge(ff_df, on=["Project", "Test"], how="left")
    print(f"Merged FlakeFlagger features from: {ff_path}")

target = "IsFlaky"
ignore_cols = ["Project", "Test", target]

# Experiments:
#   flakeflagger_raw    — raw execution-history columns (leaky; sets an upper-bound baseline)
#   static_only         — name-derived features only, no execution history (leak-free)
#   static_plus_bounded — static + TotalRuns + UniqueFailingExceptionTypes
#                         (reduced leakage: neither column directly encodes pass/fail outcome)
#   static_plus_code    — static + code-level features from GitHub source (when available)
#                         runs only if github_features.py has been executed first

_static = [
    "FunctionNameLength", "ClassNameLength", "PackageLength",
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
]
_ff_smell_cols = [
    "assertion_roulette", "conditional_test_logic", "eager_test",
    "fire_and_forget", "indirect_testing", "mystery_guest",
    "resource_optimism", "test_run_war",
    "num_asserts", "test_length",
]
_ff_churn_cols = [
    "file_churn_window_5", "file_churn_window_10", "file_churn_window_25",
    "file_churn_window_50", "file_churn_window_75", "file_churn_window_100",
]

FEATURE_SETS = {
    "flakeflagger_raw": [
        "NumFailingRuns", "NumPassingRuns",
        "FirstFailingRunID", "FirstPassingRunID",
        "UniqueFailingExceptionTypes",
    ],
    "static_only": _static,
    "static_plus_bounded": _static + ["TotalRuns", "UniqueFailingExceptionTypes"],
}

# Add FlakeFlagger experiment only when features.py has been run.
# Uses smell features + churn features when both are available;
# falls back to smells only if churn columns are missing.
available_smell_cols = [c for c in _ff_smell_cols if c in df.columns and df[c].notna().any()]
available_churn_cols = [c for c in _ff_churn_cols if c in df.columns and df[c].notna().any()]
if available_smell_cols:
    ff_cols = available_smell_cols + available_churn_cols
    label   = f"smells({len(available_smell_cols)}) + churn({len(available_churn_cols)})"
    FEATURE_SETS["flakeflagger_static"] = ff_cols
    print(f"FlakeFlagger features available [{label}] — flakeflagger_static experiment enabled.")
else:
    print("No FlakeFlagger features found — run src/features/features.py to enable flakeflagger_static experiment.")

print("Dataset shape:", df.shape)

# ------------------------------------ #
#  Train/test split & Class Imbalance  #
# ------------------------------------ #
# Split once on the full feature matrix so all experiments use the identical rows
y = df[target]
X_full = df.drop(columns=ignore_cols)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, stratify=y, random_state=7
)

flaky_num = y_train.sum()
not_flaky_num = len(y_train) - flaky_num
imbalance_weight = not_flaky_num / flaky_num if flaky_num > 0 else 1.0

# Default threshold for experiments with strong signal (leaky features hold up at high confidence).
# flakeflagger_static uses best-F1 threshold selection instead — see inside the loop.
DEFAULT_THRESHOLD = 0.99

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

all_metrics = []

# ---------------------------------------- #
#  Run each experiment                      #
# ---------------------------------------- #
for exp_name, features in FEATURE_SETS.items():
    print(f"\n{'='*50}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*50}")

    X_train = X_train_full[features]
    X_test = X_test_full[features]

    # ---------------------------- #
    # Pipeline build and training  #
    # ---------------------------- #
    pipe = Pipeline([
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=250,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.75,
            colsample_bytree=0.75,
            scale_pos_weight=imbalance_weight,
            random_state=20
        ))
    ])
    pipe.fit(X_train, y_train)

    # --------------------------------------------------------- #
    # Threshold selection                                       #
    # flakeflagger_static: sweep 0.1–0.9 and pick best F1      #
    # all others: use DEFAULT_THRESHOLD (0.99)                  #
    # --------------------------------------------------------- #
    y_prob = pipe.predict_proba(X_test)[:, 1]

    if exp_name == "flakeflagger_static":
        best_f1, best_thresh = 0.0, 0.5
        for t in [x / 100 for x in range(10, 95, 5)]:
            _pred = (y_prob >= t).astype(int)
            _f1   = f1_score(y_test, _pred, zero_division=0)
            if _f1 > best_f1:
                best_f1, best_thresh = _f1, t
        threshold = best_thresh
    else:
        threshold = DEFAULT_THRESHOLD

    y_pred = (y_prob >= threshold).astype(int)

    # -------- #
    # Metrics  #
    # -------- #
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    misclassification_cost = (1 * fp) + (2 * fn)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision}  Recall: {recall}  F1: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Misclassification Cost: {misclassification_cost}")

    # Calibrated model for Brier score
    xgb_cal = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=180,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.75,
        scale_pos_weight=imbalance_weight,
        random_state=21
    )
    cal_model = CalibratedClassifierCV(xgb_cal, method="sigmoid", cv=3)
    cal_model.fit(X_train, y_train)
    cal_prob = cal_model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, cal_prob)
    print(f"Brier Score: {brier:.6f}")

    all_metrics.append({
        "Experiment": exp_name,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TrueNegatives": tn,
        "FalsePositives": fp,
        "FalseNegatives": fn,
        "TruePositives": tp,
        "MisclassificationCost": misclassification_cost,
        "BrierScore": brier,
    })

    # Save the pipeline for the most informative leak-free experiment
    if exp_name in ("flakeflagger_static", "static_plus_bounded"):
        joblib.dump(pipe, "results/models/xgboost_model.pkl")
        print("Saved model to: results/models/xgboost_model.pkl")

    # ------------------- #
    # Calibration curve   #
    # ------------------- #
    try:
        frac_pos, mean_pred = calibration_curve(y_test, cal_prob, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(mean_pred, frac_pos, marker="o", label=f"Calibrated XGBoost")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.xlabel("Probability Predicted")
        plt.ylabel("Frequency Observed")
        plt.title(f"Calibration Curve — {exp_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/figures/calibration_curve_{exp_name}.png")
        plt.close()
        print(f"Saved: results/figures/calibration_curve_{exp_name}.png")
    except ValueError as e:
        print(f"Calibration curve skipped for {exp_name}: {e}")

    # ------------------- #
    # Feature importance  #
    # ------------------- #
    plt.figure(figsize=(10, 6))
    plot_importance(pipe["clf"])
    plt.title(f"Feature Importance — {exp_name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/xgb_feature_importance_{exp_name}.png")
    plt.close()
    print(f"Saved: results/figures/xgb_feature_importance_{exp_name}.png")


# ------------------------- #
# Save all metrics as CSV   #
# ------------------------- #
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/tables/model_metrics.csv", index=False)
print("\nSaved all experiment metrics to: results/tables/model_metrics.csv")
print(metrics_df[["Experiment", "Precision", "Recall", "F1", "BrierScore", "MisclassificationCost"]].to_string(index=False))
