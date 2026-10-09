"""Streamlit interface for educational heart disease risk classification."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import MODEL_FILES, predict_patient  # noqa: E402

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="<3", layout="wide")
st.title("Heart Disease Prediction System")
st.subheader("Machine Learning Based Risk Prediction")
st.warning("Medical Disclaimer: Educational and research use only. This model output is not a medical diagnosis. Consult a qualified healthcare professional for medical advice and diagnosis.")

LABELS = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure (mm Hg)", "chol": "Cholesterol (mg/dl)", "fbs": "Fasting blood sugar > 120 mg/dl", "restecg": "Resting ECG", "thalach": "Maximum heart rate", "exang": "Exercise-induced angina", "oldpeak": "Oldpeak", "slope": "ST segment slope", "ca": "Major vessels", "thal": "Thalassemia"}
from preprocessing import KNOWN_CATEGORICAL  # noqa: E402

data_path = ROOT / "data" / "heart.csv"
data_schema = pd.read_csv(data_path, na_values=["?", "NA", "N/A", ""])
if "thalch" in data_schema and "thalach" not in data_schema:
    data_schema = data_schema.rename(columns={"thalch": "thalach"})
target_column = next((name for name in ("target", "num", "goal") if name in data_schema), None)
feature_columns = [name for name in data_schema.columns if name not in {target_column, "id", "dataset"}]
valid_ranges = {"age": (1, 120), "trestbps": (50, 300), "chol": (50, 800), "thalach": (40, 300), "oldpeak": (-5.0, 20.0)}

st.header("Patient Information")
with st.form("patient_form"):
    columns = st.columns(3)
    patient = {}
    for index, feature in enumerate(feature_columns):
        with columns[index % 3]:
            label = LABELS.get(feature, feature.replace("_", " ").title())
            values = data_schema[feature].dropna().unique().tolist()
            if feature in KNOWN_CATEGORICAL or not pd.api.types.is_numeric_dtype(data_schema[feature]):
                values = sorted(values, key=str)
                patient[feature] = st.selectbox(label, values, key=feature)
            else:
                minimum, maximum = valid_ranges.get(feature, (float(min(values)), float(max(values))))
                default = float(data_schema[feature].median())
                default = min(maximum, max(minimum, default))
                step = 0.1 if feature == "oldpeak" else 1.0
                patient[feature] = st.number_input(label, min_value=float(minimum), max_value=float(maximum), value=default, step=step, key=feature)
    model_name = st.selectbox("Select Prediction Model", list(MODEL_FILES), index=2)
    submitted = st.form_submit_button("Predict Heart Disease", type="primary")

if submitted:
    try:
        result = predict_patient(patient, model_name)
        st.subheader("Prediction Result")
        if result["prediction"]:
            st.error("Heart Disease Detected by this model")
        else:
            st.success("No Heart Disease Detected by this model")
        st.metric("Model estimated probability for class 1", f"{result['probability']:.1%}")
        st.caption(f"Model used: {result['model']}. This probability is a model output, not a clinical risk estimate.")
    except ImportError as error:
        st.error(
            "Windows blocked a compiled scikit-learn file needed to load the model. "
            "The analytics dashboard can still be viewed, but prediction requires "
            "an approved Python environment or an administrator to allow the blocked file."
        )
        st.caption(str(error))
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))


@st.cache_data
def load_metrics():
    path = ROOT / "reports" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


st.header("Model Performance")
report = load_metrics()
if report:
    table = pd.DataFrame(report["metrics"]).T
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    st.info(f"Selected model: {report['best_model']}. {report['selection_method']}")
    for filename, caption in [("confusion_matrices.png", "Confusion matrices"), ("roc_curves.png", "ROC curves"), ("precision_recall_curves.png", "Precision-recall curves"), ("feature_importance.png", "Tree model feature importance")]:
        plot = ROOT / "reports" / "figures" / filename
        if plot.exists():
            image_width = 1100 if filename in {"roc_curves.png", "precision_recall_curves.png"} else 1200
            st.image(str(plot), caption=caption, width=image_width)
else:
    st.info("Train the models with `python src/train.py` to create the comparison table and analytics charts.")

