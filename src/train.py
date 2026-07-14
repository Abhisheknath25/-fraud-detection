"""
train.py — Train and evaluate classification models for fraud detection.

Models evaluated:
  1. Logistic Regression (baseline)
  2. Random Forest Classifier
  3. XGBoost Classifier

The best model (by AUC‑ROC) is saved to models/fraud_model.pkl.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
from xgboost import XGBClassifier
import joblib

from src.preprocess import prepare_data, MODELS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """Return a dict of model_name → model_instance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"  AUC‑ROC          : {auc_roc:.4f}")
    print(f"  Avg Precision (PR): {avg_prec:.4f}")

    return {
        "model_name": model_name,
        "auc_roc": float(auc_roc),
        "avg_precision": float(avg_prec),
        "f1_score": float(f1),
        "precision_fraud": float(report["1"]["precision"]),
        "recall_fraud": float(report["1"]["recall"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_results(models_metrics: list, models_trained: dict,
                 X_test, y_test) -> None:
    """Generate ROC curve, Precision‑Recall curve, and confusion matrices."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # ---- ROC curves ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_trained.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC curve saved to {PLOTS_DIR}/roc_curves.png")

    # ---- Precision‑Recall curves ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_trained.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision‑Recall Curve Comparison")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pr_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] PR curve saved to {PLOTS_DIR}/pr_curves.png")

    # ---- Confusion matrices ----
    fig, axes = plt.subplots(1, len(models_trained), figsize=(6 * len(models_trained), 5))
    if len(models_trained) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models_trained.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrices saved to {PLOTS_DIR}/confusion_matrices.png")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(dataset_path: str | None = None):
    """
    Train all models, evaluate, select the best, and persist it.

    Returns
    -------
    best_model, all_metrics
    """
    # 1. Prepare data
    if dataset_path:
        X_train, X_test, y_train, y_test = prepare_data(dataset_path)
    else:
        X_train, X_test, y_train, y_test = prepare_data()

    # 2. Train & evaluate each model
    models = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name} …")
        model.fit(X_train, y_train)
        trained[name] = model
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    # 3. Select best model by AUC‑ROC
    best = max(results, key=lambda m: m["auc_roc"])
    best_model = trained[best["model_name"]]
    print(f"\n🏆 Best model: {best['model_name']}  (AUC‑ROC = {best['auc_roc']:.4f})")

    # 4. Save best model & metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    all_metrics = {"best_model": best["model_name"], "models": results}
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {METRICS_PATH}")

    # 5. Plots
    plot_results(results, trained, X_test, y_test)

    return best_model, all_metrics


if __name__ == "__main__":
    train()
    print("\n✅ Training pipeline complete.")
