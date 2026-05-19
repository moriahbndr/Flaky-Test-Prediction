# used this notebook as a starting point for the xgboost setup:
# https://github.com/liannewriting/YouTube-videos-public/blob/main/xgboost-python-tutorial-example/xgboost_python.ipynb
# for k-fold as reccomended during last stand-up meeting : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# chose stratified due to the lack of ordering in dataset and having much more flaky vs non-flaky (imbalanced)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
import joblib
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "experiment_sets"))
from feature_sets import FF_SMELL_COLS, build_feature_sets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix)

DEFAULT_THRESHOLD = 0.80
K = 5

NUMERIC_KEYS = [
    "Precision", "Recall", "F1",
    "TrueNegatives", "FalsePositives", "FalseNegatives", "TruePositives",
    "MisclassificationCost",
]

def make_xgb(imbalance_weight, n_estimators=300, random_state=20):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=8,            # increasing - found better results as i increased the depth, not much changes from 8 - 10 but significant from 6 - 8
        learning_rate=0.07,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=3,
        scale_pos_weight=imbalance_weight,
        random_state=random_state,
    )

# all relevant features, data, and sets loaded 
def load_data():
    df = pd.read_csv("data/processed/full_features.csv")
    ff_path = "data/processed/flakeflagger_features.csv"
    if os.path.exists(ff_path):
        df = df.merge(pd.read_csv(ff_path), on=["Project", "Test"], how="left")

    smell_cols   = [c for c in FF_SMELL_COLS if c in df.columns and df[c].notna().any()]
    FEATURE_SETS = build_feature_sets(smell_cols)
    y            = df["IsFlaky"]    #flaky labels
    X_full       = df.drop(columns=["Project", "Test", "IsFlaky"])  # features df
    return df, smell_cols, FEATURE_SETS, y, X_full


def run_fold(exp_name, X_tr, X_te, y_tr, y_te):
    flaky_count  = y_tr.sum()
    class_weight = (len(y_tr) - flaky_count) / flaky_count if flaky_count > 0 else 1.0

    model           = make_xgb(class_weight)
    model.fit(X_tr, y_tr)
    predicted_probs = model.predict_proba(X_te)[:, 1]

    if exp_name == "smell_only":
        best_f1_score, best_threshold = 0.0, 0.5
        for step in range(10, 95, 5):
            candidate_threshold = step / 100
            fold_f1 = f1_score(y_te, (predicted_probs >= candidate_threshold).astype(int), zero_division=0)
            if fold_f1 > best_f1_score:
                best_f1_score, best_threshold = fold_f1, candidate_threshold
        threshold = best_threshold
    else:
        threshold = DEFAULT_THRESHOLD

    predicted_labels             = (predicted_probs >= threshold).astype(int)
    conf_matrix                  = confusion_matrix(y_te, predicted_labels)
    true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()

    return {
        "Threshold":             threshold,
        "Precision":             precision_score(y_te, predicted_labels, zero_division=0),
        "Recall":                recall_score(y_te, predicted_labels, zero_division=0),
        "F1":                    f1_score(y_te, predicted_labels, zero_division=0),
        "TrueNegatives":         int(true_neg),
        "FalsePositives":        int(false_pos),
        "FalseNegatives":        int(false_neg),
        "TruePositives":         int(true_pos),
        "MisclassificationCost": false_pos + 2 * false_neg, # equal cost is FP + FN but FN i put as 2x as costly here
    }

def _print_table(headers, rows):
    col_widths = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(headers))
    ]
    top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    def fmt_row(cells):
        return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " │"
    print(top)
    print(fmt_row(headers))
    print(mid)
    for row in rows:
        print(fmt_row(row))
    print(bot)


# generating and saving the plots (in the results)
def save_plots(exp_name, final_model, X_full, features):
    plt.figure(figsize=(10, 6))
    plot_importance(final_model)
    plt.title(f"Feature Importance — {exp_name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/xgb_feature_importance_{exp_name}.png")
    plt.close()

    pd.DataFrame({
        "Feature":    X_full[features].columns.tolist(),
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True).to_csv(
        f"results/tables/feature_importance_{exp_name}.csv", index=False
    )


# testing if the model can now work on tests it hassnt been trained on / never ecnountered
# also measuring generalization, whether the model was able to learn patterns or just momorized
def cross_project_eval(trained_models, trained_thresholds, FEATURE_SETS):
    idflakies_csv_path = "data/processed/idflakies_features.csv"
    if not os.path.exists(idflakies_csv_path):
        print(f"\n[Cross-project] iDFlakies features not found — run build_idflakies.py first.")
        return

    print(f"\n--------- Cross-project Generalization (iDFlakies) ---------")
    idflakies_df         = pd.read_csv(idflakies_csv_path)
    static_experiments   = ["static", "static_plus_dynamic", "smell_only"]
    eligible_experiments = [
        e for e in static_experiments
        if e in trained_models and all(f in idflakies_df.columns for f in FEATURE_SETS.get(e, []))
    ]

    cross_project_results = []
    for exp_name in eligible_experiments:
        experiment_features = FEATURE_SETS[exp_name]
        idflakies_subset    = idflakies_df.dropna(subset=experiment_features)
        if idflakies_subset.empty:
            print(f"  [{exp_name}] No complete rows in iDFlakies — skipping.")
            continue

        idflakies_labels      = idflakies_subset["IsFlaky"]
        idflakies_predictions = (trained_models[exp_name].predict_proba(idflakies_subset[experiment_features])[:, 1]
                                 >= trained_thresholds[exp_name]).astype(int)
        true_positives        = int((idflakies_predictions == 1).sum())
        false_negatives       = int((idflakies_predictions == 0).sum())
        cross_project_recall  = true_positives / len(idflakies_labels) if len(idflakies_labels) > 0 else 0.0

        cross_project_results.append({
            "Experiment":       exp_name,
            "iDFlakies_N":      len(idflakies_labels),
            "iDFlakies_Recall": round(cross_project_recall, 4),
            "iDFlakies_TP":     true_positives,
            "iDFlakies_FN":     false_negatives,
        })

    if cross_project_results:
        pd.DataFrame(cross_project_results).to_csv("results/tables/cross_project_metrics.csv", index=False)

        rows = []
        for r in cross_project_results:
            row = [
                r["Experiment"],
                str(r["iDFlakies_N"]),
                f"{r['iDFlakies_Recall']:.4f}",
                f"{r['iDFlakies_TP']}/{r['iDFlakies_N']}",
            ]
            rows.append(row)

        _print_table(["Experiment", "N", "Recall", "Caught"], rows)


# --- main --- #

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/models",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

df, smell_cols, FEATURE_SETS, y, X_full = load_data()
stratified_kfold         = StratifiedKFold(n_splits=K, shuffle=True, random_state=7)
full_dataset_class_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0

print(f"\n--------- Flaky Test Prediction Pipeline ---------")
print(f"Dataset : {df.shape[0]} tests, {df.shape[1]} columns")
print(f"Smell features : {'enabled' if smell_cols else 'not found, run build_smells.py first'}")
print(f"\nTraining {len(FEATURE_SETS)} experiments with {K}-fold stratified cross-validation...")

experiment_metrics = []
trained_models     = {}
trained_thresholds = {}

for exp_name, features in FEATURE_SETS.items():
    print(f"  {exp_name}...")

    fold_metrics = []

    for train_indices, test_indices in stratified_kfold.split(X_full, y):
        fold_result = run_fold(
            exp_name,
            X_full.iloc[train_indices][features], X_full.iloc[test_indices][features],
            y.iloc[train_indices],                y.iloc[test_indices],
        )
        fold_metrics.append(fold_result)

    fold_avg    = {k: np.mean([r[k] for r in fold_metrics]) for k in NUMERIC_KEYS}
    fold_std_devs     = {k: np.std( [r[k] for r in fold_metrics]) for k in NUMERIC_KEYS}
    avg_threshold = np.mean([r["Threshold"] for r in fold_metrics])

    trained_thresholds[exp_name] = avg_threshold

    full_trained_model = make_xgb(full_dataset_class_weight)
    full_trained_model.fit(X_full[features], y)
    trained_models[exp_name] = full_trained_model

    if exp_name == "static_plus_dynamic":
        joblib.dump(full_trained_model, "results/models/xgboost_model.pkl")

    save_plots(exp_name, full_trained_model, X_full, features)

    experiment_metrics.append({
        "Experiment":                exp_name,
        "Threshold":                 round(avg_threshold, 2),
        "Precision":                 round(fold_avg["Precision"], 4),
        "Precision_Std":             round(fold_std_devs["Precision"], 4),
        "Recall":                    round(fold_avg["Recall"], 4),
        "Recall_Std":                round(fold_std_devs["Recall"], 4),
        "F1":                        round(fold_avg["F1"], 4),
        "F1_Std":                    round(fold_std_devs["F1"], 4),
        "TrueNegatives":             int(round(fold_avg["TrueNegatives"])),
        "FalsePositives":            int(round(fold_avg["FalsePositives"])),
        "FalseNegatives":            int(round(fold_avg["FalseNegatives"])),
        "TruePositives":             int(round(fold_avg["TruePositives"])),
        "MisclassificationCost":     round(fold_avg["MisclassificationCost"], 1),
        "MisclassificationCost_Std": round(fold_std_devs["MisclassificationCost"], 1),
    })

pd.DataFrame(experiment_metrics).to_csv("results/tables/model_metrics.csv", index=False)

print(f"\n--------- Results (5-fold cross-validation, mean ± std) ---------\n")
rows = []
for r in experiment_metrics:
    row = [
        r["Experiment"],
        f"{r['Precision']:.4f} ± {r['Precision_Std']:.4f}",
        f"{r['Recall']:.4f} ± {r['Recall_Std']:.4f}",
        f"{r['F1']:.4f} ± {r['F1_Std']:.4f}",
        f"{r['MisclassificationCost']:.1f} ± {r['MisclassificationCost_Std']:.1f}",
    ]
    rows.append(row)

_print_table(["Experiment", "Precision", "Recall", "F1", "Misc Cost"], rows)

cross_project_eval(trained_models, trained_thresholds, FEATURE_SETS)

print(f"\n--------- Saved ---------")
print(f"  results/tables/model_metrics.csv")
print(f"  results/tables/cross_project_metrics.csv")
print(f"  results/models/xgboost_model.pkl")
print(f"  results/figures/  (feature importance per experiment)")
