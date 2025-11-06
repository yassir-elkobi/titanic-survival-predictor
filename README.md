## Titanic Survival Predictor (Flask + scikit-learn)

An educational, end-to-end classification project showing how different ML models learn to predict Titanic survival. It
includes:

- A clean preprocessing pipeline (imputation, scaling, one-hot encoding)
- Multiple models trained and compared automatically
- An ensemble (soft VotingClassifier) to combine strong models
- A Flask app with UI to predict and view comparison metrics

This repo is intentionally simple and readable to help newcomers (and busy managers) understand how ML fits together.

### Demo: What happens

1) Data is loaded (your CSV at `static/titanic_survival.csv`).
2) We split into train/test and build a preprocessing pipeline.
3) We train several models and evaluate them:
    - Logistic Regression (L2)
    - Logistic Regression (L1)
    - Random Forest
    - Gradient Boosting
    - Ensemble (Voting over top performers)
4) We compute metrics (Accuracy, Precision, Recall, F1, AUC) and pick the best model.
5) The best model is saved to `model.joblib`. A full report is saved to `metrics.json`.
6) The Flask app serves:
    - `/` — prediction form
    - `/compare` — comparison table of models
    - `/train` — retrain and refresh metrics
    - `/metrics` — raw JSON of results

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: put your dataset at static/titanic_survival.csv with target column 'Survived'
python app.py  # this will train on first run if model is missing
# Open http://127.0.0.1:5000
```

## Using the API

- Predict via JSON:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Pclass":3,"Sex":"male","Age":29,"SibSp":0,"Parch":0,"Fare":7.25,"Embarked":"S"}'
```

- Compare models (JSON): `GET /metrics`

## Educational Notes

- Why multiple models? Each algorithm has different bias/variance trade-offs. We compare them to learn which works best
  for this data.
- Why a pipeline? Consistent preprocessing avoids data leakage and keeps code maintainable.
- What metrics mean:
    - Accuracy: overall correctness
    - Precision: percent of predicted positives that are correct
    - Recall: percent of actual positives that are found
    - F1: balance between precision and recall
    - AUC: ability to discriminate between classes
- Ensemble: A soft VotingClassifier averaging strong models often performs as well as or better than a single best
  model.

## Project Layout

```
app.py          — Flask app (predict, compare, retrain, metrics)
train.py        — training and comparison logic; saves model.joblib and metrics.json
templates/      — UI pages (index.html, compare.html)
requirements.txt — Python dependencies
static/titanic_survival.csv — sample dataset (optional)
```

## Bring Your Own Data

Place a CSV at `static/titanic_survival.csv` with a target column:

- `Survived` (0/1)

Feature columns can include `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`. The code will infer types and
handle missing values.

## Reproducibility

We fix `random_state=42` and use a consistent test split (`test_size=0.25`). You can change these in
`train_and_compare()` in `train.py`.

## License

MIT