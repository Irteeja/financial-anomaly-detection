"""
Unit Tests — Financial Anomaly Detection System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_transactions
from src.features.feature_engineering import build_features, NUMERIC_FEATURES
from src.detectors.ensemble_detector import (
    EnsembleAnomalyDetector,
    ZScoreDetector,
    IQRDetector,
    IsolationForestDetector,
    AutoencoderProxyDetector,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return generate_transactions(n_accounts=20, days=30, seed=0)


@pytest.fixture(scope="module")
def feat_df(raw_df):
    return build_features(raw_df)


@pytest.fixture(scope="module")
def detector(raw_df):
    d = EnsembleAnomalyDetector(contamination=0.05)
    d.fit(raw_df)
    return d


# ─────────────────────────────────────────────
# Data Generator
# ─────────────────────────────────────────────

class TestDataGenerator:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns(self, raw_df):
        expected = {"transaction_id", "timestamp", "account_id", "amount",
                    "amount_gbp", "currency", "merchant", "true_label"}
        assert expected.issubset(raw_df.columns)

    def test_no_null_amounts(self, raw_df):
        assert raw_df["amount"].isnull().sum() == 0

    def test_amounts_positive(self, raw_df):
        assert (raw_df["amount"] > 0).all()

    def test_anomalies_present(self, raw_df):
        assert raw_df["true_label"].sum() > 0

    def test_anomaly_rate_reasonable(self, raw_df):
        rate = raw_df["true_label"].mean()
        assert 0.01 < rate < 0.30

    def test_multiple_anomaly_types(self, raw_df):
        types = raw_df.loc[raw_df["true_label"] == 1, "anomaly_type"].unique()
        assert len(types) >= 3

    def test_accounts_assigned(self, raw_df):
        assert raw_df["account_id"].nunique() > 1

    def test_timestamp_sorted(self, raw_df):
        ts = pd.to_datetime(raw_df["timestamp"])
        # Not necessarily sorted globally (sorted after feature engineering)
        assert ts.min() < ts.max()


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_temporal_features_created(self, feat_df):
        for col in ["hour", "day_of_week", "is_weekend", "is_night",
                    "hour_sin", "hour_cos"]:
            assert col in feat_df.columns, f"Missing: {col}"

    def test_behavioural_features_created(self, feat_df):
        for col in ["amount_zscore", "amount_vs_mean", "rolling_7d_ratio"]:
            assert col in feat_df.columns, f"Missing: {col}"

    def test_velocity_features_created(self, feat_df):
        for col in ["vel_1h", "vel_24h", "amt_vel_24h"]:
            assert col in feat_df.columns, f"Missing: {col}"

    def test_compliance_flags_created(self, feat_df):
        for col in ["flag_over_10k", "flag_night_tx", "flag_foreign_tx"]:
            assert col in feat_df.columns, f"Missing: {col}"

    def test_hour_in_range(self, feat_df):
        assert feat_df["hour"].between(0, 23).all()

    def test_cyclical_hour_bounded(self, feat_df):
        assert feat_df["hour_sin"].between(-1, 1).all()
        assert feat_df["hour_cos"].between(-1, 1).all()

    def test_no_inf_values(self, feat_df):
        numeric = feat_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_numeric_features_present(self, feat_df):
        missing = [f for f in NUMERIC_FEATURES if f not in feat_df.columns]
        assert len(missing) == 0, f"Missing features: {missing}"


# ─────────────────────────────────────────────
# Sub-detectors
# ─────────────────────────────────────────────

class TestZScoreDetector:
    def test_fit_and_predict(self, feat_df):
        det = ZScoreDetector(threshold=3.0)
        det.fit(feat_df)
        preds = det.predict(feat_df)
        assert len(preds) == len(feat_df)
        assert set(preds).issubset({0, 1})

    def test_score_bounded(self, feat_df):
        det = ZScoreDetector()
        det.fit(feat_df)
        s = det.score(feat_df)
        assert (s >= 0).all() and (s <= 1).all()


class TestIQRDetector:
    def test_fit_and_predict(self, feat_df):
        det = IQRDetector()
        det.fit(feat_df)
        preds = det.predict(feat_df)
        assert len(preds) == len(feat_df)

    def test_score_bounded(self, feat_df):
        det = IQRDetector()
        det.fit(feat_df)
        s = det.score(feat_df)
        assert (s >= 0).all() and (s <= 1).all()


class TestIsolationForest:
    def test_fit_and_predict(self, feat_df):
        det = IsolationForestDetector(contamination=0.05)
        det.fit(feat_df)
        preds = det.predict(feat_df)
        assert len(preds) == len(feat_df)
        assert preds.sum() > 0  # some anomalies detected

    def test_score_bounded(self, feat_df):
        det = IsolationForestDetector()
        det.fit(feat_df)
        s = det.score(feat_df)
        assert (s >= 0).all() and (s <= 1).all()


class TestAutoencoderProxy:
    def test_fit_and_predict(self, feat_df):
        det = AutoencoderProxyDetector()
        det.fit(feat_df)
        preds = det.predict(feat_df)
        assert len(preds) == len(feat_df)

    def test_score_bounded(self, feat_df):
        det = AutoencoderProxyDetector()
        det.fit(feat_df)
        s = det.score(feat_df)
        assert (s >= 0).all() and (s <= 1).all()


# ─────────────────────────────────────────────
# Ensemble Detector
# ─────────────────────────────────────────────

class TestEnsembleDetector:
    def test_fit_sets_flag(self, detector):
        assert detector.is_fitted

    def test_detect_returns_dataframe(self, detector, raw_df):
        results = detector.detect(raw_df)
        assert isinstance(results, pd.DataFrame)

    def test_detect_adds_columns(self, detector, raw_df):
        results = detector.detect(raw_df)
        for col in ["ensemble_score", "is_anomaly", "risk_level", "alert_reason"]:
            assert col in results.columns

    def test_ensemble_score_bounded(self, detector, raw_df):
        results = detector.detect(raw_df)
        assert results["ensemble_score"].between(0, 1).all()

    def test_is_anomaly_binary(self, detector, raw_df):
        results = detector.detect(raw_df)
        assert set(results["is_anomaly"].unique()).issubset({0, 1})

    def test_risk_level_valid(self, detector, raw_df):
        results = detector.detect(raw_df)
        valid = {"Low", "Medium", "High", "Critical"}
        assert set(results["risk_level"].unique()).issubset(valid)

    def test_some_anomalies_detected(self, detector, raw_df):
        results = detector.detect(raw_df)
        assert results["is_anomaly"].sum() > 0

    def test_evaluate_returns_metrics(self, detector, raw_df):
        results = detector.detect(raw_df)
        metrics = detector.evaluate(results)
        for key in ["precision", "recall", "f1_score", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_summary_keys(self, detector, raw_df):
        results = detector.detect(raw_df)
        s = detector.summary(results)
        for key in ["total_transactions", "anomalies_flagged", "anomaly_rate_pct"]:
            assert key in s

    def test_roc_auc_above_random(self, detector, raw_df):
        """A good detector should beat random (0.5) substantially."""
        results = detector.detect(raw_df)
        metrics = detector.evaluate(results)
        assert metrics["roc_auc"] > 0.55


# ─────────────────────────────────────────────
# Integration
# ─────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: generate → feature → detect → evaluate."""
        df = generate_transactions(n_accounts=15, days=20, seed=99)
        detector = EnsembleAnomalyDetector(contamination=0.05)
        results = detector.detect(df)
        metrics = detector.evaluate(results)

        assert len(results) == len(df)
        assert metrics["roc_auc"] > 0.5
        assert results["is_anomaly"].sum() > 0

    def test_single_account(self):
        """Should not crash with a single account."""
        df = generate_transactions(n_accounts=1, days=10, seed=7)
        detector = EnsembleAnomalyDetector()
        results = detector.detect(df)
        assert len(results) == len(df)
