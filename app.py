import json
import os
from typing import Dict, Optional
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from train import ensure_model, train_and_compare, METRICS_PATH, MODEL_PATH

app = Flask(__name__)

_model_cache = None  # lazy-loaded trained pipeline
_metrics_cache: Optional[Dict] = None


def _load_metrics() -> Optional[Dict]:
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _metrics_cache = json.load(f)
            return _metrics_cache
    return None


def _load_or_train() -> Dict:
    """
    Ensure artifacts exist (train on first run), warm metrics cache.
    """
    global _metrics_cache
    metrics = ensure_model()
    _metrics_cache = metrics
    return metrics


def _load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            _load_or_train()
        _model_cache = joblib.load(MODEL_PATH)
        # If legacy regression artifact is loaded (no predict_proba), retrain to classification
        if not hasattr(_model_cache, "predict_proba"):
            train_and_compare()
            _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


@app.route("/", methods=["GET"])
def index():
    metrics = _load_metrics()
    best_model_name = metrics["best_model"]["name"] if metrics else None
    dataset = metrics["dataset"] if metrics else None
    return render_template("index.html", best_model=best_model_name, dataset=dataset)


@app.route("/predict", methods=["POST"])
def predict():
    model = _load_model()
    metrics = _load_metrics() or _load_or_train()

    # Expected raw feature columns (pre-preprocessing)
    numeric_cols = metrics["dataset"]["numeric_features"]
    categorical_cols = metrics["dataset"]["categorical_features"]
    expected_cols = numeric_cols + categorical_cols

    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        # Accept form submissions as well
        payload = {k: v for k, v in request.form.items()}

    # Build a single-row DataFrame with all expected columns
    row = {col: payload.get(col) for col in expected_cols}
    # Coerce numeric features
    for c in numeric_cols:
        val = row.get(c)
        if val is None or val == "":
            continue
        try:
            row[c] = float(val)
        except (TypeError, ValueError):
            # Let imputer handle NaN; leave as None
            row[c] = None
    df = pd.DataFrame([row])

    # Use classifier probability for the positive class
    prob = float(model.predict_proba(df)[0, 1])
    label = int(prob >= 0.5)
    result = {"predicted_score": prob, "predicted_label": label}

    if request.is_json:
        return jsonify(result)
    # For form submits, just return JSON for simplicity
    return jsonify(result)


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics = _load_metrics()
    if not metrics:
        return jsonify({"error": "No metrics available"}), 404
    return jsonify(metrics)


@app.route("/train", methods=["POST", "GET"])
def train():
    # Allow GET to ease UI button usage; still non-interactive
    metrics = train_and_compare()
    # Reset caches
    global _model_cache, _metrics_cache
    _model_cache = None
    _metrics_cache = metrics
    return jsonify({"status": "ok", "best_model": metrics["best_model"], "metrics": metrics["metrics"]})


@app.route("/compare", methods=["GET"])
def compare():
    metrics = _load_metrics() or _load_or_train()
    return render_template("compare.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
