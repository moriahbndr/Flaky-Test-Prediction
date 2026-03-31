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

# focusing on isFlaky, ignoring else
target = "IsFlaky"
ignore_cols = ["Project", "Test", target]

# X - input, y - output here
X = df.drop(columns=ignore_cols)
y = df[target]

print("Dataset shape:", df.shape)


# ------------------------------------ #
#  Train/test split & Class Imbalance  #
# ------------------------------------ #
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, stratify=y, random_state=7)
flaky_num = y_train.sum()
not_flaky_num = len(y_train) - flaky_num
imbalance_weight = not_flaky_num / flaky_num if flaky_num > 0 else 1.0


# ---------------------------- #
# Pipeline Build and training  #
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


# ----------- #
# Prediction  #
# ----------- #
threshold = .99 # started from .4 and the results stayed the same from .4 - .99
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)


# -------- #
# Metrics  #
# -------- #
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tp = cm[1,1]
misclassification_cost = (1 * fp) + (2 * fn)    # false positive = 1 false negative = 2

print("\nCurrent hreshold:", threshold)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Confusion Matrix:")
print(cm)
print("Misclassification Cost:", misclassification_cost)

xgb_model = XGBClassifier(
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

cal_model = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3) #using sigmoid for now, but as more calibration data is added may want to use isotonic instead
cal_model.fit(X_train, y_train)
cal_prob = cal_model.predict_proba(X_test)[:, 1]    #using flaky probilitiy [:, 1]
brierScore = brier_score_loss(y_test, cal_prob) #lower is best for score


# ------------------------- #
# Saving the metrics as CSV #
# ------------------------- #
metric_results = "results/tables"
if not os.path.exists(metric_results):
    os.makedirs(metric_results, exist_ok=True)

metricScores = pd.DataFrame([{
    "Threshold": threshold,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "TrueNegatives": tn,
    "FalsePositives": fp,
    "FalseNegatives": fn,
    "TruePositives": tp,
    "MisclassificationCost": misclassification_cost,
    "BrierScore": brierScore
}])

metricScores.to_csv("results/tables/model_metrics.csv", index=False)
print("Saved metric results and data to: results/tables/model_metrics.csv")


# --------------------------------- #
# Trained model saved after running #
# --------------------------------- #
model_results = "results/models"
if not os.path.exists(model_results):
    os.makedirs(model_results, exist_ok=True)
joblib.dump(pipe, "results/models/xgboost_model.pkl")
print("Saved the model to: results/models/xgboost_model.pkl")

# saving the figure 
figureProduced = "results/figures"
if not os.path.exists(figureProduced):
    os.makedirs(figureProduced, exist_ok=True)

frac_pos, mean_pred = calibration_curve(y_test, cal_prob, n_bins=10)

# Matplot lib plots table 
width = 6
height = 6
plt.figure(figsize=(height, width))
plt.plot(mean_pred, frac_pos, marker="o", label="Calibrated XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Probability Predicted")
plt.ylabel("Frequency Observed")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/calibration_curve.png")
plt.close()

print("Saved plotted graph to: results/figures/calibration_curve.png")

xgb_model = pipe["clf"]

plt.figure(figsize=(10, 8))
plot_importance(xgb_model)
plt.tight_layout()
plt.savefig("results/figures/xgb_feature_importance.png") #source: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.plotting
plt.close()

print("Saved feature importance graph to: results/figures/xgb_feature_importance.png")
