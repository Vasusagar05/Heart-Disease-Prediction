"""Load a saved pipeline and produce a non-diagnostic educational prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import ROOT

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def predict_patient(patient: dict, model_name: str = "XGBoost") -> dict:
    """Return class label and positive-class probability for one feature mapping."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {list(MODEL_FILES)}")
    artifact_path = ROOT / "models" / MODEL_FILES[model_name]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model file is missing: {artifact_path}. Run python src/train.py first.")
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    columns = artifact["feature_columns"]
    missing = set(columns) - set(patient)
    extra = set(patient) - set(columns)
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(sorted(missing))}")
    row = pd.DataFrame([{column: patient[column] for column in columns}], columns=columns)
    prediction = int(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    positive_index = list(pipeline.classes_).index(1)
    return {
        "prediction": prediction,
        "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "probability": float(probabilities[positive_index]),
        "model": model_name,
        "ignored_extra_features": sorted(extra),
    }


if __name__ == "__main__":
    example = {"age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
               "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
               "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2}
    result = predict_patient(example)
    print(f"Prediction: {result['label']}\nProbability: {result['probability']:.1%}\nModel: {result['model']}")
