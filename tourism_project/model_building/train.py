"""Tune, evaluate, track, and serialize the best classification model."""
from pathlib import Path
import json
import warnings
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DEPLOYMENT = ROOT / "tourism_project" / "deployment"
REPORTS = ROOT / "reports"
TARGET = "ProdTaken"

def main():
    train = pd.read_csv(ARTIFACTS / "train.csv")
    test = pd.read_csv(ARTIFACTS / "test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int)
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int)
    categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__max_features": ["sqrt"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="roc_auc",
        cv=cv, n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "model": "RandomForestClassifier",
    }
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, DEPLOYMENT / "model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # MLflow tracking is enabled in CI/local environments where MLflow is installed.
    try:
        import mlflow
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("visit-with-us-tourism-purchase")
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(DEPLOYMENT / "model.joblib"))
            mlflow.log_artifact(str(REPORTS / "metrics.json"))
    except ImportError:
        warnings.warn("MLflow not installed; metrics.json still records tuned parameters and metrics.")
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print("Test metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ["best_params", "classification_report", "confusion_matrix"]}, indent=2))
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("Classification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Saved model: {DEPLOYMENT / 'model.joblib'}")
    print(f"Saved metrics: {REPORTS / 'metrics.json'}")

if __name__ == "__main__":
    main()
