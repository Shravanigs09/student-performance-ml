"""
src/train.py
Trains and benchmarks candidate models across both Math and Portuguese datasets,
exporting serialized champion pipelines for each subject.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from pipeline import create_full_pipeline

DATASETS = {
    "math": os.path.join("data", "raw", "student-mat.csv"),
    "por": os.path.join("data", "raw", "student-por.csv"),
}
MODEL_DIR = "models"


def load_and_prep_data(filepath: str):
    df = pd.read_csv(filepath, sep=";")
    df["passed"] = (df["G3"] >= 10).astype(int)
    X = df.drop(columns=["passed", "G3", "G2"])
    y = df["passed"]
    return X, y


def train_subject_pipeline(subject_key: str, filepath: str):
    print("\n" + "#" * 65)
    print(f"  TRAINING SUBJECT PIPELINE: {subject_key.upper()}")
    print("#" * 65)

    X, y = load_and_prep_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_clf = None, 0.0, None

    for name, clf in candidate_models.items():
        pipe = create_full_pipeline(clf)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1_macro", "roc_auc"])
        mean_f1 = res["test_f1_macro"].mean()
        mean_auc = res["test_roc_auc"].mean()
        print(f"{name:<22} | F1-Macro: {mean_f1:.4f} | ROC-AUC: {mean_auc:.4f}")

        if mean_f1 > best_score:
            best_score, best_name, best_clf = mean_f1, name, clf

    print(f"--> Champion Selected ({subject_key}): {best_name}")
    final_pipe = create_full_pipeline(best_clf)
    final_pipe.fit(X_train, y_train)

    y_pred = final_pipe.predict(X_test)
    y_prob = final_pipe.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["At-Risk (0)", "Passed (1)"]))
    print(f"Held-Out ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{subject_key}_pipeline.joblib")
    joblib.dump(final_pipe, save_path)
    print(f"Saved artifact: {save_path}")


def main():
    for subject_key, path in DATASETS.items():
        if os.path.exists(path):
            train_subject_pipeline(subject_key, path)
        else:
            print(f"Warning: {path} not found, skipping.")


if __name__ == "__main__":
    main()