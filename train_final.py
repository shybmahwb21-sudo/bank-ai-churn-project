"""Train and evaluate the churn models with an imbalance-safe threshold."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data" / "Churn_Modelling.csv")
data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])
X, y = data.drop(columns=["Exited"]), data["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y)
X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_train, y_train, test_size=.20, random_state=42, stratify=y_train)

num = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
       "HasCrCard", "IsActiveMember", "EstimatedSalary"]
cat = ["Geography", "Gender"]
def make_pipeline(estimator):
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), num),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("preprocessor", prep), ("model", estimator)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                             class_weight="balanced", random_state=42, n_jobs=-1)}

def best_threshold(y_true, probabilities):
    """Select on validation only; test is never used to choose the cutoff."""
    candidates = np.linspace(.10, .90, 161)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])

rows, pipes, thresholds = [], {}, {}
for name, estimator in models.items():
    pipe = make_pipeline(estimator)
    pipe.fit(X_fit, y_fit)
    valid_prob = pipe.predict_proba(X_valid)[:, 1]
    threshold = best_threshold(y_valid, valid_prob)
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = test_prob >= threshold
    rows.append({
        "Model": name, "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1-score": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_prob)})
    pipes[name], thresholds[name] = pipe, threshold

results = pd.DataFrame(rows).sort_values(["F1-score", "Recall"], ascending=False).reset_index(drop=True)
best = str(results.loc[0, "Model"])
threshold = thresholds[best]
# Refit the selected pipeline on all labelled training rows after threshold selection.
pipes[best].fit(X_train, y_train)
joblib.dump(pipes[best], ROOT / "best_model.joblib")
metrics = {
    "best_model": best, "decision_threshold": threshold,
    "threshold_selection": "validation F1 (20% of training split; test untouched)",
    "class_weight": "balanced", "test_size": .20, "validation_size_of_train": .20,
    "positive_label": "Exited=1", "results": results.to_dict("records"),
    "test_confusion_matrix": confusion_matrix(
        y_test, pipes[best].predict_proba(X_test)[:, 1] >= threshold).tolist()}
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8")
print(results.to_string(index=False))
print(f"Selected: {best} | validation threshold: {threshold:.3f}")
print(f"Saved: {ROOT / 'best_model.joblib'}")
