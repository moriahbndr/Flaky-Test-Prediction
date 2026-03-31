# importing pandas (data loads),scikit-learn (metrics), joblib (saves the model), os (adding/creatinf folders)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# storing the created features (static and lightweight dynamic) - in the 
df = pd.read_csv("data/processed/full_features.csv")

# focusing on IsFlaky column and regarding ignore
target = "IsFlaky"     
ignore = ["Project", "Test", target]

# x - input, y - output
X = df.drop(columns=ignore)
y = df[target]

# 75/25 split where 25 goes to test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,random_state=7,stratify=y
)

# class imbalance measurements
flaky_num = y_train.sum()
notFlaky_num = len(y_train) - flaky_num

if flaky_num > 0:
    imbalanced_weight = notFlaky_num / flaky_num
else:
    imbalanced_weight = 1.0
    
# Build XGBoost model
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.08,
    subsample=0.75,
    colsample_bytree=0.75,
    n_estimators=250,
    max_depth=3,
    scale_pos_weight=imbalanced_weight,
    random_state=7
)

#Trainingg
xgb_model.fit(X_train, y_train)

#Prediction
y_prob = xgb_model.predict_proba(X_test)[:,1]
threshold = 0.40
y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred, zero_division=0)        # showing how many actual flaky tests out of all predicted flaky tests
recall = recall_score(y_test, y_pred, zero_division=0)              # how many flaky tests did the model catch from the true flaky tests
f1 = f1_score(y_test, y_pred, zero_division=0)                      # combined score that balances precision and recall
matrix = confusion_matrix(y_test, y_pred)                           # true positives and negatives , and false positives and negatives shown

print("Threshold:",threshold, " Precision: ", precision)
print("Recall:", recall, "F1: ", f1)
print("Confusion Matrix:")
print(matrix)

# Saving the model at end of running
os.makedirs("results/models", exist_ok=True)
joblib.dump(xgb_model, "results/models/xgboost_model.pkl")
print("Saved: results/models/xgboost_model.pkl")
