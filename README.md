# Heart Disease Prediction System

An educational machine-learning project that compares Logistic Regression, Random Forest, and XGBoost on the UCI Heart Disease Cleveland data. It includes leakage-safe preprocessing, exploratory analysis, test-set evaluation, prediction helpers, and a Streamlit dashboard.

## Objective

Explore whether patient measurements in this small historical dataset are associated with the presence of heart disease, and demonstrate a reproducible binary-classification workflow. The model output is not a diagnosis or a validated clinical risk estimate.

## Dataset inspection

`data/heart.csv` currently contains 920 records from four UCI Heart Disease sources (Cleveland, Hungary, Switzerland, and VA Long Beach), 13 patient features, an `id`, and a `dataset` source label. The pipeline excludes `id` and `dataset` from prediction features, and renames `thalch` to the commonly used `thalach`. Its outcome column is `num`; values 1-4 indicate disease severity. The project converts `num == 0` to class 0 and positive severity values to class 1. A file that already has a binary `target` column is also supported. The loader inspects shape, columns, types, missingness, duplicates, and unique values before training.

Source: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI: 10.24432/C52P4X.

Inspection of the current CSV found 920 rows and 16 columns: 13 patient features, `id`, `dataset`, and `num`. It includes four source labels. There are 411 records with `num = 0` and 509 with `num > 0`; training maps these to 0 and 1. Missing values are present in blood pressure (59), cholesterol (30), fasting blood sugar (90), resting ECG (2), maximum heart rate (55), exercise angina (55), oldpeak (62), slope (309), major vessels (611), and thalassemia (486). The file also contains zero values in cholesterol (172) and resting blood pressure (1); the preprocessing marks these as implausible and imputes them within the training pipeline. The `id` and source-center `dataset` fields are excluded from model inputs.

## Project structure

```text
Heart_D_prediction/
├── data/heart.csv
├── notebooks/heart_disease_analysis.ipynb
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── figures/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

The model and report files are created by training and are not source code. The Python scripts and app create the artifacts when you run the commands below.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Logistic Regression, Random Forest, XGBoost, Matplotlib, Seaborn, Streamlit, Joblib, Jupyter.

## Models

- **Logistic Regression**: scaled linear baseline.
- **Random Forest**: class-weighted ensemble of decision trees.
- **XGBoost**: gradient-boosted tree ensemble.

All transformations are fitted as part of each training pipeline on the training partition only. Continuous numeric values are median-imputed and standardized; UCI's coded categorical fields (such as chest pain type and thalassemia) are mode-imputed and one-hot encoded. An 80/20 split uses `random_state=42` and target stratification. Broad invalid-value checks for familiar UCI columns mark implausible inputs as missing for training-only imputation.

## Workflow

```text
UCI data -> inspection and cleaning -> exploratory analysis -> stratified split
          -> train preprocessing and three models -> held-out evaluation
          -> model comparison and saved artifacts -> Streamlit interface
```

## Evaluation and model selection

The same held-out test set is used for all models. The training script calculates accuracy, precision, recall/sensitivity, F1, ROC-AUC, specificity, confusion matrices, and classification reports; it also saves ROC and precision-recall comparisons. Since a false negative can represent a missed positive case, recall is an important part of the comparison. Model selection uses the lowest mean rank across accuracy, precision, recall, F1, ROC-AUC, and specificity, with each metric weighted equally; it does not optimize accuracy alone. This ranking is an educational comparison method, not clinical validation.

## Setup and run

From the project root, use Python 3.10 or later:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py src/train.py
py -m streamlit run app.py
```

On macOS/Linux, replace `py` with `python3` and activate with `source .venv/bin/activate`.

Open the Streamlit URL printed in the terminal. The application supports model choice, patient inputs, probability display, and a performance dashboard. To inspect the exploratory notebook, run `jupyter notebook notebooks/heart_disease_analysis.ipynb`.

## Outputs

`python src/train.py` creates three model pipelines under `models/`, plus:

- `reports/metrics.json` - calculated metrics, confusion matrices, reports, and selected model.
- `reports/model_comparison.csv` - generated comparison table.
- `reports/feature_importance.csv` - Random Forest and XGBoost transformed-feature importance.
- `reports/figures/` - confusion matrices, ROC curves, precision-recall curves, and feature-importance plots.

### Results from the current training run

The training output you provided used the same held-out test set for all three models (184 records: 82 without heart disease and 102 with heart disease). Values below are the measured results from that run, rounded to three decimals.

| Model | Accuracy | Precision | Recall / Sensitivity | F1 | ROC-AUC | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.815 | 0.827 | 0.843 | 0.835 | 0.894 | 0.780 |
| Random Forest | 0.804 | 0.817 | 0.833 | 0.825 | **0.904** | 0.768 |
| XGBoost | **0.832** | 0.814 | **0.902** | **0.856** | 0.898 | 0.744 |

The script selected **Logistic Regression** using its lowest mean rank across the six metrics, with each metric weighted equally. XGBoost achieved the highest recall, F1, and accuracy, while Random Forest achieved the highest ROC-AUC. The selection rule therefore reflects an overall balance rather than a claim that Logistic Regression leads every metric. If reducing false negatives is the main project priority, compare XGBoost's higher recall with its lower specificity and resulting higher false-positive rate.

Training writes the calculated results to `reports/metrics.json` and `reports/model_comparison.csv`; the Streamlit dashboard reads the saved results and plots. Re-run `python src/train.py` after changing the data or training code to refresh those artifacts. If the resulting metrics change, update this README table to match.

## Prediction module

After training, call `predict_patient` with a mapping containing all feature columns:

```python
from src.predict import predict_patient

patient = {
    "age": 55, "sex": "Male", "cp": "atypical angina",
    "trestbps": 140, "chol": 250, "fbs": False,
    "restecg": "normal", "thalach": 150, "exang": False,
    "oldpeak": 1.2, "slope": "flat", "ca": 0, "thal": "normal",
}
print(predict_patient(patient, "Random Forest"))
```

## Medical disclaimer

This application is developed for educational and research purposes only. The prediction is generated by a machine learning model and should not be considered a medical diagnosis. Always consult a qualified healthcare professional for medical advice and diagnosis.
