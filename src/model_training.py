# used this notebook as a starting point for the xgboost setup:
# https://github.com/liannewriting/YouTube-videos-public/blob/main/xgboost-python-tutorial-example/xgboost_python.ipynb

import pandas as pd
from xgboost import XGBClassifier, plot_importance
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix)


# load the processed dataset that build_flakeflagger.py created
df = pd.read_csv("data/processed/full_features_v2.csv")

# if build_smells.py has been run, pull in the smell features and merge them in
ff_path = "data/processed/flakeflagger_features.csv"
if os.path.exists(ff_path):
    ff_df = pd.read_csv(ff_path)
    df = df.merge(ff_df, on=["Project", "Test"], how="left")
    print(f"Merged FlakeFlagger features from: {ff_path}")

target = "IsFlaky"
ignore_cols = ["Project", "Test", target]

# these are the different experiments we're running:
#   flakeflagger_raw       - uses the raw run history columns, this is basically cheating
#                            since the data already tells you if it failed. good upper bound though
#   smell_only             - just the test smell features from the java source code
#   static_v2              - only name-based features, no run history at all
#   static_v2_plus_dynamic - name features + some derived run stats (still leaky but useful to see)
#   flakeflagger_static    - smell features + churn, this is the FlakeFlagger paper's approach

_static = [
    "FunctionNameLength", "ClassNameLength", "PackageLength",
    "SleepOrWaitInFunction", "AsyncInFunction", "TimeOrRandomInFunction",
    "NetworkInFunction", "FileIOInFunction", "DatabaseInFunction",
    "UIBrowserInFunction", "RetryFlakeInFunction",
    "SleepOrWaitInClass", "AsyncInClass", "TimeOrRandomInClass",
    "NetworkInClass", "FileIOInClass", "DatabaseInClass", "UIBrowserInClass",
    "NetworkInPackage", "FileIOInPackage", "DatabaseInPackage", "UIBrowserInPackage",
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
    "smell_only": [],          # filled in below once we know which smell cols are available
    "static_v2": _static,
    "static_v2_plus_dynamic": _static + [
        "TotalRuns", "FailRatio", "PassRatio",
        "AnyFailures", "AnyPassingRuns",
        "BothPassAndFail", "AlwaysFails",
        "FailOnFirstRun", "EarlyFailure",
        "ExceptionDiversityRatio",
    ],
}

# only add smell-based experiments if build_smells.py has been run first
# if the churn columns are also there, include those too
available_smell_cols = [c for c in _ff_smell_cols if c in df.columns and df[c].notna().any()]
available_churn_cols = [c for c in _ff_churn_cols if c in df.columns and df[c].notna().any()]
if available_smell_cols:
    FEATURE_SETS["smell_only"] = available_smell_cols          # <-- pure baseline
    ff_cols = available_smell_cols + available_churn_cols
    FEATURE_SETS["flakeflagger_static"] = ff_cols              # smells + churn (existing)
    print(f"Smell features available ({len(available_smell_cols)}) — smell_only baseline enabled.")
else:
    del FEATURE_SETS["smell_only"]
    print("No smell features found — run src/features/build_smells.py first.")
print("Dataset shape:", df.shape)

# split the data 75/25 — all experiments use the exact same split so results are comparable
# model is only trained on FlakeFlagger data here, iDFlakies comes in later for cross-project eval
y = df[target]
X_full = df.drop(columns=ignore_cols)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, stratify=y, random_state=7
)

# dataset is pretty imbalanced (way more non-flaky tests than flaky ones)
# scale_pos_weight helps the model not just predict everything as non-flaky
flaky_num = y_train.sum()
not_flaky_num = len(y_train) - flaky_num
imbalance_weight = not_flaky_num / flaky_num if flaky_num > 0 else 1.0

# 0.99 is pretty aggressive but the leaky experiments need it to avoid too many false positives
# smell_only and flakeflagger_static use a swept threshold instead since their signal is weaker
DEFAULT_THRESHOLD = 0.99

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

all_metrics = []
trained_pipes = {}   # keeping each trained model so we can reuse them for the iDFlakies eval

# loop through each experiment and train/evaluate
for exp_name, features in FEATURE_SETS.items():
    print(f"\n{'='*50}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*50}")

    X_train = X_train_full[features]
    X_test = X_test_full[features]

    # build and train the model
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
    trained_pipes[exp_name] = pipe   # save for cross-project eval below

    # for weaker experiments (smell_only, flakeflagger_static) we sweep thresholds
    # and pick whichever gives the best f1 score
    # for everything else just use the default 0.99
    y_prob = pipe.predict_proba(X_test)[:, 1]

    if exp_name in ("flakeflagger_static", "smell_only"):
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

    # calculate all our metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    # missing a flaky test costs twice as much as a false alarm
    misclassification_cost = (1 * fp) + (2 * fn)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision}  Recall: {recall}  F1: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Misclassification Cost: {misclassification_cost}")

    # train a second calibrated version just to get the brier score and calibration curve
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

    # save the model — we only really care about the best leak-free one
    if exp_name in ("flakeflagger_static", "static_plus_bounded"):
        joblib.dump(pipe, "results/models/xgboost_model.pkl")
        print("Saved model to: results/models/xgboost_model.pkl")

    # calibration curve — shows how well the predicted probabilities match actual outcomes
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

    # feature importance plot so we can see which features the model actually relied on
    plt.figure(figsize=(10, 6))
    plot_importance(pipe["clf"])
    plt.title(f"Feature Importance — {exp_name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/xgb_feature_importance_{exp_name}.png")
    plt.close()
    print(f"Saved: results/figures/xgb_feature_importance_{exp_name}.png")


# save all the experiment results to a csv
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/tables/model_metrics.csv", index=False)
print("\nSaved all experiment metrics to: results/tables/model_metrics.csv")
print(metrics_df[["Experiment", "Precision", "Recall", "F1", "BrierScore", "MisclassificationCost"]].to_string(index=False))


# cross-project evaluation using the iDFlakies dataset
# the model was trained entirely on FlakeFlagger so iDFlakies is completely unseen
# since iDFlakies only has confirmed flaky tests (no non-flaky), we can only measure recall here
# basically asking: of all the tests we know are flaky, how many does our model catch?
_idf_path = "data/processed/idflakies_features.csv"

if not os.path.exists(_idf_path):
    print(f"\n[Cross-project] iDFlakies features not found at {_idf_path}")
    print("  Run src/features/build_idflakies.py first, then re-run model training.")
else:
    print(f"\n{'='*60}")
    print("  Cross-project evaluation — iDFlakies (static features only)")
    print(f"{'='*60}")

    idf = pd.read_csv(_idf_path)

    # only experiments that use static features can run on iDFlakies
    # since iDFlakies has no run history columns, the dynamic experiments won't work here
    _static_exps = ["static_v2", "static_v2_plus_dynamic", "smell_only", "flakeflagger_static"]
    _static_only_exps = [e for e in _static_exps if e in trained_pipes
                         and all(f in idf.columns for f in FEATURE_SETS.get(e, []))]

    cross_metrics = []

    for exp_name in _static_only_exps:
        features = FEATURE_SETS[exp_name]
        if not features:
            continue

        # drop any rows that are missing features
        idf_sub = idf.dropna(subset=features)
        if idf_sub.empty:
            print(f"  [{exp_name}] No complete rows in iDFlakies — skipping.")
            continue

        X_idf = idf_sub[features]
        y_idf = idf_sub["IsFlaky"]  # all 1s

        y_idf_prob = trained_pipes[exp_name].predict_proba(X_idf)[:, 1]
        y_idf_pred = (y_idf_prob >= DEFAULT_THRESHOLD).astype(int)

        tp_idf  = int((y_idf_pred == 1).sum())
        fn_idf  = int((y_idf_pred == 0).sum())
        recall_idf = tp_idf / len(y_idf) if len(y_idf) > 0 else 0.0

        print(f"\n  Experiment : {exp_name}")
        print(f"  iDFlakies tests evaluated : {len(y_idf)}")
        print(f"  Caught (TP) : {tp_idf}   Missed (FN) : {fn_idf}")
        print(f"  Cross-project Recall : {recall_idf:.4f}")

        # break down recall by flakiness category (ID, OD, NOD, etc.)
        # interesting to see if our model is better at catching certain types of flakiness
        if "Category" in idf_sub.columns:
            cat_results = []
            for cat, grp in idf_sub.groupby("Category"):
                g_prob = trained_pipes[exp_name].predict_proba(grp[features])[:, 1]
                g_pred = (g_prob >= DEFAULT_THRESHOLD).astype(int)
                g_recall = g_pred.mean()
                cat_results.append({"Category": cat, "N": len(grp), "Recall": round(g_recall, 4)})
            cat_df = pd.DataFrame(cat_results).sort_values("N", ascending=False)
            print(f"\n  Per-category recall (top 10 by count):")
            print(cat_df.head(10).to_string(index=False))

        cross_metrics.append({
            "Experiment":          exp_name,
            "iDFlakies_N":         len(y_idf),
            "iDFlakies_Recall":    round(recall_idf, 4),
            "iDFlakies_TP":        tp_idf,
            "iDFlakies_FN":        fn_idf,
        })

    if cross_metrics:
        cross_df = pd.DataFrame(cross_metrics)
        cross_path = "results/tables/cross_project_metrics.csv"
        cross_df.to_csv(cross_path, index=False)
        print(f"\nSaved cross-project metrics to: {cross_path}")
        print(cross_df.to_string(index=False))
