"""
REST API — Financial Risk & Anomaly Detection System
=====================================================
Serves the detection engine over HTTP using FastAPI.

Endpoints:
  POST /detect         — score a batch of transactions
  POST /detect/single  — score one transaction
  GET  /health         — liveness check
  GET  /metrics        — last evaluation metrics
  GET  /model/info     — model metadata

Usage:
    pip install fastapi uvicorn
    uvicorn src.api.app:app --reload --port 8000

    # Then POST to http://localhost:8000/detect
"""

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Stub classes so the module can be imported without FastAPI
    class BaseModel:
        pass
    def Field(*a, **k):
        return None

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class TransactionIn(BaseModel):
    transaction_id: Optional[str] = Field(None, example="TXN-000000000001")
    timestamp: str = Field(..., example="2024-06-15T14:32:00")
    account_id: str = Field(..., example="ACC0001")
    account_type: Optional[str] = Field("medium_spender")
    merchant: str = Field(..., example="Amazon")
    category: Optional[str] = Field("Online Shopping")
    amount: float = Field(..., gt=0, example=149.99)
    currency: str = Field("GBP", example="GBP")
    amount_gbp: Optional[float] = Field(None)
    channel: Optional[str] = Field("mobile_app")
    location: Optional[str] = Field("London")
    is_foreign: Optional[bool] = Field(False)
    device_id: Optional[str] = Field("DEV-ABCD1234")
    ip_country: Optional[str] = Field("GB")


class TransactionOut(BaseModel):
    transaction_id: str
    account_id: str
    timestamp: str
    amount_gbp: float
    ensemble_score: float
    is_anomaly: int
    risk_level: str
    alert_reason: str
    flag_over_10k: int
    flag_night_tx: int
    flag_foreign_tx: int
    flag_unknown_device: int
    compliance_score: float


class BatchRequest(BaseModel):
    transactions: List[TransactionIn] = Field(..., min_items=1, max_items=10000)


class BatchResponse(BaseModel):
    total: int
    anomalies_detected: int
    anomaly_rate_pct: float
    results: List[TransactionOut]
    processing_time_ms: float


# ─────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────

def create_app() -> "FastAPI":
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Financial Risk & Anomaly Detection API",
        description=(
            "Advanced transaction anomaly detection using a 5-model ensemble "
            "(Z-Score, IQR, Isolation Forest, PCA Autoencoder, LOF). "
            "Returns risk scores, risk tiers, and alert reasons."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Lazy-load model
    _state = {"detector": None, "last_metrics": {}}

    def _get_detector():
        if _state["detector"] is None:
            from src.detectors.ensemble_detector import EnsembleAnomalyDetector
            model_path = "models/saved/ensemble_detector.pkl"
            if Path(model_path).exists():
                logger.info("Loading saved model…")
                _state["detector"] = EnsembleAnomalyDetector.load(model_path)
            else:
                logger.warning("No saved model found. Auto-fitting on startup data…")
                from src.data_generator import generate_transactions
                df = generate_transactions(n_accounts=80, days=180)
                _state["detector"] = EnsembleAnomalyDetector()
                _state["detector"].fit(df)
        return _state["detector"]

    def _txns_to_df(txns: List[TransactionIn]) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(txns):
            rows.append({
                "transaction_id": t.transaction_id or f"TXN-API-{i:06d}",
                "timestamp":      t.timestamp,
                "account_id":     t.account_id,
                "account_type":   t.account_type or "medium_spender",
                "merchant":       t.merchant,
                "category":       t.category or "Other",
                "amount":         t.amount,
                "currency":       t.currency,
                "amount_gbp":     t.amount_gbp if t.amount_gbp else t.amount,
                "channel":        t.channel or "web",
                "location":       t.location or "Unknown",
                "is_foreign":     t.is_foreign or False,
                "device_id":      t.device_id or "N/A",
                "ip_country":     t.ip_country or "GB",
            })
        return pd.DataFrame(rows)

    # ── Routes ────────────────────────────────────────

    @app.get("/health", tags=["System"])
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "2.0.0"}

    @app.get("/model/info", tags=["System"])
    def model_info():
        det = _get_detector()
        return {
            "model": "EnsembleAnomalyDetector v2.0",
            "detectors": ["Z-Score", "IQR", "Isolation Forest", "PCA Autoencoder", "LOF"],
            "contamination": det.contamination,
            "score_threshold": det.score_threshold,
            "is_fitted": det.is_fitted,
            "detector_weights": {
                "zscore": 0.10, "iqr": 0.10, "iso_forest": 0.35,
                "autoencoder": 0.25, "lof": 0.20,
            },
        }

    @app.get("/metrics", tags=["System"])
    def get_metrics():
        metrics_path = Path("reports/metrics.json")
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return {"message": "No metrics available. Run main.py first."}

    @app.post("/detect", response_model=BatchResponse, tags=["Detection"])
    def detect_batch(req: BatchRequest):
        """Score a batch of transactions (up to 10,000)."""
        import time
        t0 = time.time()
        try:
            det = _get_detector()
            df = _txns_to_df(req.transactions)
            results = det.detect(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        out_cols = [
            "transaction_id", "account_id", "timestamp", "amount_gbp",
            "ensemble_score", "is_anomaly", "risk_level", "alert_reason",
            "flag_over_10k", "flag_night_tx", "flag_foreign_tx",
            "flag_unknown_device", "compliance_score",
        ]
        # Fill missing cols
        for c in out_cols:
            if c not in results.columns:
                results[c] = 0

        results_list = []
        for _, row in results[out_cols].iterrows():
            results_list.append(TransactionOut(
                transaction_id=str(row["transaction_id"]),
                account_id=str(row["account_id"]),
                timestamp=str(row["timestamp"]),
                amount_gbp=float(row["amount_gbp"]),
                ensemble_score=round(float(row["ensemble_score"]), 4),
                is_anomaly=int(row["is_anomaly"]),
                risk_level=str(row["risk_level"]),
                alert_reason=str(row["alert_reason"]),
                flag_over_10k=int(row["flag_over_10k"]),
                flag_night_tx=int(row["flag_night_tx"]),
                flag_foreign_tx=int(row["flag_foreign_tx"]),
                flag_unknown_device=int(row["flag_unknown_device"]),
                compliance_score=float(row["compliance_score"]),
            ))

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        anomaly_count = int(results["is_anomaly"].sum())
        return BatchResponse(
            total=len(results),
            anomalies_detected=anomaly_count,
            anomaly_rate_pct=round(anomaly_count / len(results) * 100, 2),
            results=results_list,
            processing_time_ms=elapsed_ms,
        )

    @app.post("/detect/single", tags=["Detection"])
    def detect_single(txn: TransactionIn):
        """Score a single transaction."""
        batch_req = BatchRequest(transactions=[txn])
        response = detect_batch(batch_req)
        return {
            "result": response.results[0],
            "processing_time_ms": response.processing_time_ms,
        }

    return app


# Only create app if FastAPI is available
if HAS_FASTAPI:
    app = create_app()
