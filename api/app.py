"""
app.py — FastAPI REST API for the Credit Card Fraud Detection system.

Endpoints
---------
GET  /           → Web UI for testing
POST /predict    → Submit transaction features, get fraud prediction
GET  /health     → API health check
GET  /model-info → Loaded model metadata
GET  /docs       → Auto-generated Swagger documentation (built‑in)
"""

import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict as predict_fraud, NUM_FEATURES, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: eager‑load model so first request is fast
    try:
        from src.predict import _load_model, _load_scaler
        _load_model()
        _load_scaler()
        print("✅ Model and scaler loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   The API will start, but /predict will fail until the model is trained.")
    yield
    # Shutdown: nothing to clean up


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "A machine learning API that detects fraudulent credit card "
        "transactions in real time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """Input schema — 30 numeric features for one transaction."""
    features: list[float] = Field(
        ...,
        min_length=NUM_FEATURES,
        max_length=NUM_FEATURES,
        description=(
            f"A list of {NUM_FEATURES} numeric values: {FEATURE_NAMES}"
        ),
        json_schema_extra={
            "example": [0.0] + [0.0] * 28 + [149.62],
        },
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Fraudulent or Legitimate")
    fraud_probability: float = Field(..., description="Probability of fraud (0–1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the web testing interface."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud",
    description="Submit a transaction's 30 features and receive a fraud prediction.",
)
async def predict_endpoint(req: TransactionRequest):
    """Run the fraud detection model on a single transaction."""
    try:
        result = predict_fraud(req.features)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health():
    """Check whether the API and model are operational."""
    model_loaded = os.path.exists(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
    )


@app.get("/model-info", summary="Model information")
async def model_info():
    """Return training metrics for the loaded model."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Train the model first.",
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)
