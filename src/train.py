"""Train, evaluate, compare and persist the project's three classifiers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise SystemExit("XGBoost is required. Install project dependencies with: pip install -r requirements.txt") from error

from preprocessing import ROOT, load_dataset, make_preprocessor, normalize_target, validate_and_split

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "figures"


def build_models():
    return {
        "Logistic Regression": ("logistic_regression.pkl", LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest": ("random_forest.pkl", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)),
        "XGBoost": ("xgboost.pkl", XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=42, eval_metric="logloss",
            n_jobs=-1, verbosity=0)),
    }


def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Specificity": specificity,
        "y_pred": predicted,
        "y_score": probabilities,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1], target_names=["No Heart Disease", "Heart Disease"],
            output_dict=True, zero_division=0),
    }


def write_figures(results, y_test):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, result) in zip(axes, results.items()):
        matrix = np.asarray(result["confusion_matrix"])
        ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(
            ax=ax, colorbar=False, cmap="Blues", include_values=False)
        cell_names = [["TN", "FP"], ["FN", "TP"]]
        for row in range(2):
            for column in range(2):
                ax.text(column, row, f"{cell_names[row][column]}\n{matrix[row, column]}",
                        ha="center", va="center", color="black", fontsize=10)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_score"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {result['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve Comparison")
    ax.legend(loc="lower right"); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result["y_score"])
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(PLOT_DIR / "precision_recall_curves.png", dpi=160); plt.close(fig)


def write_feature_importance(models, raw_columns):
    """Save tree-model importance values using the pipeline's transformed feature names."""
    rows = []
    for label in ("Random Forest", "XGBoost"):
        pipeline = models[label]
        estimator = pipeline.named_steps["classifier"]
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        for feature, importance in zip(names, estimator.feature_importances_):
            rows.append({"Model": label, "Feature": feature, "Importance": float(importance)})
    table = pd.DataFrame(rows).sort_values(["Model", "Importance"], ascending=[True, False])
    table.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, label in zip(axes, ("Random Forest", "XGBoost")):
        top = table[table.Model == label].nlargest(12, "Importance").sort_values("Importance")
        ax.barh(top.Feature.str.replace(r"^(numeric|categorical)__", "", regex=True), top.Importance)
        ax.set_title(f"{label} Feature Importance"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "feature_importance.png", dpi=160); plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 50 + "\nHEART DISEASE PREDICTION SYSTEM\n" + "=" * 50)
    frame, source_target = normalize_target(load_dataset())
    print(f"\nTarget column: {source_target}; binary class distribution:\n{frame[source_target].value_counts().sort_index()}")
    X_train, X_test, y_train, y_test = validate_and_split(frame, source_target)
    metrics, pipelines = {}, {}
    for index, (name, (filename, estimator)) in enumerate(build_models().items(), start=1):
        print(f"\n[{index}/3] {name}")
        pipeline = Pipeline([("preprocess", make_preprocessor(X_train)), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        result = evaluate(pipeline, X_test, y_test)
        pipelines[name] = pipeline
        metrics[name] = result
        joblib.dump({"pipeline": pipeline, "feature_columns": list(X_train.columns),
                     "source_target": source_target, "positive_class": 1}, MODEL_DIR / filename)
        print(classification_report(y_test, result["y_pred"], labels=[0, 1],
              target_names=["No Heart Disease", "Heart Disease"], zero_division=0))

    write_figures(metrics, y_test)
    write_feature_importance(pipelines, list(X_train.columns))
    columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Specificity"]
    comparison = pd.DataFrame({name: {key: value[key] for key in columns}
                               for name, value in metrics.items()}).T
    # Equal-weight average rank across all requested measures prevents accuracy-only selection.
    ranks = comparison.rank(ascending=False, method="average")
    comparison["Selection score"] = ranks.mean(axis=1)
    best = comparison["Selection score"].idxmin()
    comparison.drop(columns="Selection score").to_csv(REPORT_DIR / "model_comparison.csv", index_label="Model")
    payload = {
        "metrics": {name: {key: float(result[key]) for key in columns} for name, result in metrics.items()},
        "confusion_matrices": {name: result["confusion_matrix"] for name, result in metrics.items()},
        "classification_reports": {name: result["classification_report"] for name, result in metrics.items()},
        "best_model": best,
        "selection_method": "Lowest mean rank across accuracy, precision, recall, F1, ROC-AUC and specificity (equal weight).",
        "dataset_rows": int(len(frame)), "test_rows": int(len(y_test)),
        "source_target": source_target,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 50 + "\nMODEL PERFORMANCE\n" + "=" * 50)
    print(comparison.drop(columns="Selection score").round(3).to_string())
    print(f"\nBEST MODEL\nSelected Model: {best}\nReason: Lowest mean rank across six complementary measures; recall is included to limit missed positive cases.")
    print(f"\nModels and factual reports saved under {ROOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise
