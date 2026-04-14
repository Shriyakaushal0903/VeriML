"""
api/app.py
----------
FastAPI service exposing the Reliable-ML system.

Endpoints:
  POST /predict        – single prediction with uncertainty + calibration
  POST /predict/batch  – batch predictions
  GET  /health         – health check
  GET  /model/info     – model metadata

Run with:
  uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations
import os, sys, json, time
import numpy as np
from pathlib import Path
from typing import Optional, List

# ── FastAPI ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    # Stub so the module can be imported for documentation purposes
    class FastAPI:           # type: ignore
        def __init__(self, **kw): pass
        def get(self, *a, **kw):  return lambda f: f
        def post(self, *a, **kw): return lambda f: f

# ── Project imports ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Reliable ML API",
    description=(
        "Confidence-Aware Prediction System with Calibration and "
        "Failure Diagnostics. Returns predictions, uncertainty estimates, "
        "calibrated probabilities, and reject flags."
    ),
    version="1.0.0",
)

if _FASTAPI_OK:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )


# ══════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    """Features for a single credit-default prediction."""
    LIMIT_BAL:  float = Field(..., description="Credit limit (USD)")
    SEX:        int   = Field(..., ge=1, le=2)
    EDUCATION:  int   = Field(..., ge=1, le=4)
    MARRIAGE:   int   = Field(..., ge=1, le=3)
    AGE:        int   = Field(..., ge=18, le=100)
    PAY_0:      int   = Field(..., ge=-2, le=8,
                              description="Repayment status last month")
    PAY_2:      int   = Field(0, ge=-2, le=8)
    PAY_3:      int   = Field(0, ge=-2, le=8)
    BILL_AMT1:  float = Field(0.0, ge=0)
    BILL_AMT2:  float = Field(0.0, ge=0)
    PAY_AMT1:   float = Field(0.0, ge=0)
    PAY_AMT2:   float = Field(0.0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2,
                "MARRIAGE": 1, "AGE": 35, "PAY_0": 1, "PAY_2": 0,
                "PAY_3": 0, "BILL_AMT1": 12000, "BILL_AMT2": 8000,
                "PAY_AMT1": 500, "PAY_AMT2": 700,
            }
        }


class PredictionResponse(BaseModel):
    prediction:           int            # 0 or 1
    raw_probability:      float          # uncalibrated P(default)
    calibrated_probability: float        # calibrated P(default)
    uncertainty:          float          # ensemble std
    confidence_margin:    float          # |p - 0.5|
    rejected:             bool           # True = abstained
    risk_tier:            str            # LOW / MEDIUM / HIGH / UNCERTAIN
    latency_ms:           float


class BatchRequest(BaseModel):
    records: List[PredictionRequest]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size:  int
    latency_ms:  float


class HealthResponse(BaseModel):
    status:   str
    model_loaded: bool
    timestamp: float


# ══════════════════════════════════════════════════════════════════════
# Model singleton (lazy-loaded)
# ══════════════════════════════════════════════════════════════════════

_STATE: dict = {"loaded": False, "pipeline": None,
                "xgb": None, "ensemble": None,
                "platt": None, "reject": None}


def _load_models():
    """Load the trained artifacts into memory (called once at startup)."""
    import pickle
    artefact_dir = Path(__file__).parent.parent / "outputs"
    artefact_path = artefact_dir / "artefacts.pkl"

    if not artefact_path.exists():
        print(f"[API] Warning: {artefact_path} not found. "
              "Run train.py first to generate artefacts.")
        return

    with open(artefact_path, "rb") as f:
        state = pickle.load(f)

    _STATE.update(state)
    _STATE["loaded"] = True
    print("[API] Models loaded successfully.")


def _feature_vector(req: PredictionRequest) -> np.ndarray:
    import pandas as pd
    from data.pipeline import engineer_features
    row = pd.DataFrame([req.dict()])
    row = engineer_features(row)
    return _STATE["pipeline"].transform(row)


def _predict_one(req: PredictionRequest, t0: float) -> PredictionResponse:
    if not _STATE["loaded"]:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    X = _feature_vector(req)

    # XGBoost raw probability
    raw_prob = float(_STATE["xgb"].predict_proba(X)[0, 1])

    # Ensemble uncertainty
    mean_prob, unc = _STATE["ensemble"].predict_with_uncertainty(X)
    unc_val  = float(unc[0])

    # Calibrated probability
    cal_prob = float(_STATE["platt"].transform(np.array([raw_prob]))[0])

    # Reject option
    rc      = _STATE["reject"]
    preds   = rc.predict(np.array([cal_prob]), np.array([unc_val]))
    rejected = bool(preds[0] == rc.ABSTAIN)
    pred    = int(preds[0]) if not rejected else int(cal_prob >= 0.5)

    margin  = float(abs(cal_prob - 0.5))

    # Risk tier
    if rejected:
        tier = "UNCERTAIN"
    elif cal_prob < 0.3:
        tier = "LOW"
    elif cal_prob < 0.6:
        tier = "MEDIUM"
    else:
        tier = "HIGH"

    return PredictionResponse(
        prediction=pred,
        raw_probability=round(raw_prob, 4),
        calibrated_probability=round(cal_prob, 4),
        uncertainty=round(unc_val, 4),
        confidence_margin=round(margin, 4),
        rejected=rejected,
        risk_tier=tier,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    _load_models()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="ok", model_loaded=_STATE["loaded"],
                          timestamp=time.time())


@app.get("/model/info", tags=["System"])
async def model_info():
    if not _STATE["loaded"]:
        raise HTTPException(503, "Models not loaded.")
    return {
        "model":           "XGBoost + Deep Ensemble (5 members)",
        "calibration":     "Platt Scaling",
        "reject_option":   "Margin + Uncertainty threshold",
        "feature_count":   len(_STATE["pipeline"].feature_names),
        "features":        _STATE["pipeline"].feature_names,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    t0 = time.perf_counter()
    return _predict_one(request, t0)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    if len(batch.records) > 1000:
        raise HTTPException(400, "Batch size limited to 1000.")
    t0 = time.perf_counter()
    results = [_predict_one(r, t0) for r in batch.records]
    return BatchResponse(
        predictions=results,
        batch_size=len(results),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
