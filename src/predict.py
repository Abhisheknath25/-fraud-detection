"""
predict.py — Prediction utilities for the fraud detection model.

Loads the trained model and scaler, accepts raw feature vectors,
and returns predictions with probability scores and risk levels.
"""

import os
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Feature names in the original dataset (in order)
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
NUM_FEATURES = len(FEATURE_NAMES)  # 30

# Risk thresholds
RISK_THRESHOLDS = {
    "HIGH": 0.75,
    "MEDIUM": 0.40,
    "LOW": 0.0,
}


# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------
_model = None
_scaler = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run `python -m src.train` first."
            )
        _model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    return _model


def _load_scaler():
    global _scaler
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                "Run `python -m src.train` first."
            )
        _scaler = joblib.load(SCALER_PATH)
        print(f"[INFO] Scaler loaded from {SCALER_PATH}")
    return _scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_risk(probability: float) -> str:
    """Map a fraud probability to a human‑readable risk level."""
    if probability >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif probability >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def predict(features: list[float]) -> dict:
    """
    Predict whether a single transaction is fraudulent.

    Parameters
    ----------
    features : list[float]
        A list of 30 numeric values corresponding to
        [Time, V1, V2, …, V28, Amount].

    Returns
    -------
    dict with keys:
        prediction       – "Fraudulent" or "Legitimate"
        fraud_probability – float ∈ [0, 1]
        risk_level        – "LOW" | "MEDIUM" | "HIGH"
    """
    if len(features) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, got {len(features)}. "
            f"Features should be: {FEATURE_NAMES}"
        )

    model = _load_model()
    scaler = _load_scaler()

    arr = np.array(features).reshape(1, -1)

    # Scale Time (index 0) and Amount (index 29) using the saved scaler
    time_amount = arr[:, [0, -1]]
    time_amount_scaled = scaler.transform(time_amount)
    arr[:, 0] = time_amount_scaled[:, 0]
    arr[:, -1] = time_amount_scaled[:, 1]

    prob = model.predict_proba(arr)[0, 1]
    pred_label = "Fraudulent" if prob >= 0.5 else "Legitimate"
    risk = classify_risk(prob)

    return {
        "prediction": pred_label,
        "fraud_probability": round(float(prob), 4),
        "risk_level": risk,
    }


def predict_batch(batch: list[list[float]]) -> list[dict]:
    """Predict fraud for a batch of transactions."""
    return [predict(row) for row in batch]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a random 30‑feature vector for testing
    dummy = np.random.randn(NUM_FEATURES).tolist()
    print("Input features (random):", dummy[:5], "…")
    result = predict(dummy)
    print("Prediction:", result)
