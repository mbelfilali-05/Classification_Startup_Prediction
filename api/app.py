"""
api/app.py
─────────────────────────────────────────────────────────────────────
FastAPI Prediction Service for Startup Classification.

Exposes a /predict endpoint that accepts startup features and returns
a classification with probabilities.

Uses the exact same pipeline as training (no feature duplication).

Usage:
  cd startup-classification
  uvicorn api.app:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"funding_total_usd": 5000000, "funding_rounds": 3, ...}'

Docs:
  http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from utils.feature_engineering import (
    RawCleaner, FeatureEngineer, CategoryGrouper, FeatureSelector,
)


# ═══════════════════════════════════════════════════════════════════
# Load pipeline components at startup
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ART_DIR = os.path.join(BASE_DIR, 'artifacts')
MOD_DIR = os.path.join(BASE_DIR, 'models')

cleaner = joblib.load(os.path.join(ART_DIR, 'cleaner.pkl'))
engineer = joblib.load(os.path.join(ART_DIR, 'engineer.pkl'))
grouper = joblib.load(os.path.join(ART_DIR, 'grouper.pkl'))
selector = joblib.load(os.path.join(ART_DIR, 'selector.pkl'))
preprocessor = joblib.load(os.path.join(ART_DIR, 'preprocessor.pkl'))
label_encoder = joblib.load(os.path.join(ART_DIR, 'label_encoder_target.pkl'))
model = joblib.load(os.path.join(MOD_DIR, 'xgb_model.pkl'))


# ═══════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════

class StartupInput(BaseModel):
    """Input schema matching the raw CSV columns."""
    name: Optional[str] = Field("Unknown", description="Startup name")
    homepage_url: Optional[str] = Field(None, description="Website URL")
    category_list: str = Field(..., description="Categories separated by |")
    funding_total_usd: float = Field(..., description="Total funding in USD")
    country_code: str = Field("USA", description="ISO country code")
    state_code: Optional[str] = Field("", description="State code")
    region: Optional[str] = Field("", description="Region name")
    city: Optional[str] = Field("", description="City name")
    funding_rounds: int = Field(1, description="Number of funding rounds")
    founded_at: Optional[str] = Field(None, description="Founded date (YYYY-MM-DD)")
    first_funding_at: Optional[str] = Field(None, description="First funding date")
    last_funding_at: Optional[str] = Field(None, description="Last funding date")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "TechStartup",
                "homepage_url": "https://techstartup.com",
                "category_list": "Software|SaaS",
                "funding_total_usd": 5000000,
                "country_code": "USA",
                "region": "SF Bay Area",
                "funding_rounds": 3,
                "founded_at": "2018-01-01",
                "first_funding_at": "2018-06-01",
                "last_funding_at": "2021-03-01",
            }
        }


class PredictionOutput(BaseModel):
    predicted_status: str
    confidence: float
    probabilities: dict
    low_confidence_flag: bool


# ═══════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Startup Classification API",
    description="Predict startup outcomes (operating/acquired/closed/ipo) using XGBoost.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Startup Classification API. Visit /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost", "features": preprocessor.n_features_in_}


@app.post("/predict", response_model=PredictionOutput)
def predict(startup: StartupInput):
    """
    Predict the status of a startup.

    The input passes through the EXACT same pipeline as training:
    RawCleaner → FeatureEngineer → CategoryGrouper → FeatureSelector → Preprocessor → Model
    """
    try:
        # Build a DataFrame row from the input
        row = startup.model_dump()
        row['funding_total_usd'] = str(row['funding_total_usd'])
        row['status'] = 'unknown'  # placeholder, not used
        df = pd.DataFrame([row])

        # Run the full pipeline
        df = cleaner.transform(df)
        df = engineer.transform(df)
        df = grouper.transform(df)
        df = selector.transform(df)
        X = preprocessor.transform(df)

        # Predict
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_encoder.classes_[pred_idx]
        confidence = float(proba[pred_idx])

        return PredictionOutput(
            predicted_status=pred_label,
            confidence=round(confidence, 4),
            probabilities={
                cls: round(float(p), 4)
                for cls, p in zip(label_encoder.classes_, proba)
            },
            low_confidence_flag=confidence < 0.6,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
