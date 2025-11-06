import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
DEFAULT_DATA_PATH = os.path.join("static", "titanic_survival.csv")
DEFAULT_TARGET = "Survived"


@dataclass
class ModelReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    fit_time_s: float
    best: bool = False


def _infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    features = df.drop(columns=[target])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def _load_dataset(data_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # Ensure target exists; allow some loose renaming just in case
    if target not in df.columns:
        # common typos seen earlier
        if "2urvived" in df.columns:
            df = df.rename(columns={"2urvived": target})
        else:
            raise ValueError(f"Target column '{target}' not found in {data_path}")
    return df


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(acc), float(prec), float(rec), float(f1)


def _model_candidates() -> Dict[str, object]:
    # Classification models suitable for binary Titanic target
    return {
        "LogReg_L2": LogisticRegression(
            solver="lbfgs", penalty="l2", max_iter=1000, random_state=42
        ),
        "LogReg_L1": LogisticRegression(
            solver="liblinear", penalty="l1", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }


def train_and_compare(
        data_path: str = DEFAULT_DATA_PATH,
        target: str = DEFAULT_TARGET,
        test_size: float = 0.25,
        random_state: int = 42
) -> Dict:
    df = _load_dataset(data_path, target)
    # Prefer a lean, reproducible feature set suitable for UI inputs
    preferred_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    available = [c for c in preferred_features if c in df.columns]
    if len(available) >= 4:  # require at least a reasonable subset
        df = df[[target] + available].copy()
    numeric_cols, categorical_cols = _infer_feature_types(df, target)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[target])
    y = df[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = _model_candidates()
    reports: Dict[str, ModelReport] = {}
    fitted_models: Dict[str, Pipeline] = {}

    # Fit and evaluate each base model
    for name, estimator in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time_s = time.time() - t0
        y_pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        acc, prec, rec, f1 = _evaluate(y_test, y_pred)
        auc = roc_auc_score(y_test, prob)
        reports[name] = ModelReport(
            accuracy=acc, precision=prec, recall=rec, f1=f1, auc=float(auc), fit_time_s=fit_time_s
        )
        fitted_models[name] = pipe

    # Build a soft VotingClassifier from the top performers (by AUC)
    top_names = sorted(reports.keys(), key=lambda k: reports[k].auc, reverse=True)[:3]
    voting_estimators = [(n, candidates[n]) for n in top_names]  # fresh, not fitted
    voting_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", VotingClassifier(estimators=voting_estimators, voting="soft"))
    ])
    t0 = time.time()
    voting_pipe.fit(X_train, y_train)
    voting_fit_time = time.time() - t0
    y_pred_ens = voting_pipe.predict(X_test)
    prob_ens = voting_pipe.predict_proba(X_test)[:, 1]
    acc, prec, rec, f1 = _evaluate(y_test, y_pred_ens)
    auc = roc_auc_score(y_test, prob_ens)
    reports["VotingClassifier"] = ModelReport(
        accuracy=acc, precision=prec, recall=rec, f1=f1, auc=float(auc), fit_time_s=voting_fit_time
    )
    fitted_models["VotingClassifier"] = voting_pipe

    # Select best by AUC, break ties by higher F1
    best_name = max(reports.keys(), key=lambda k: (reports[k].auc, reports[k].f1))
    for k in reports:
        reports[k].best = (k == best_name)
    best_model = fitted_models[best_name]

    # Persist model and metrics
    joblib.dump(best_model, MODEL_PATH)

    metrics_payload = {
        "dataset": {
            "path": data_path,
            "n_rows": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1),
            "target": target,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
        "split": {"test_size": test_size, "random_state": random_state},
        "metrics": {name: asdict(rep) for name, rep in reports.items()},
        "best_model": {"name": best_name, "criterion": "max_auc_then_max_f1"},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def ensure_model(data_path: str = DEFAULT_DATA_PATH, target: str = DEFAULT_TARGET) -> Dict:
    """
    Train on first use if artifacts are missing. Returns current metrics dict.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return train_and_compare(data_path=data_path, target=target)
