"""
Anomaly Detection Engine — Ensemble of 5 Methods
==================================================
Methods:
  1. Z-Score   (statistical)
  2. IQR fence (statistical)
  3. Isolation Forest (ML unsupervised)
  4. Autoencoder Reconstruction Error (deep-learning proxy via PCA)
  5. Local Outlier Factor (density-based)

Final score: weighted vote with calibration.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from scipy import stats
import joblib
import logging
from pathlib import Path

from src.features.feature_engineering import NUMERIC_FEATURES, build_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Sub-detectors
# ─────────────────────────────────────────────

class ZScoreDetector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit(self, df):
        return self

    def score(self, df) -> np.ndarray:
        """Return anomaly probability proxy: |z| normalised to [0,1]."""
        z = df["amount_zscore"].abs().fillna(0).values
        return np.clip(z / 10.0, 0, 1)

    def predict(self, df) -> np.ndarray:
        return (df["amount_zscore"].abs().fillna(0) > self.threshold).astype(int).values


class IQRDetector:
    def fit(self, df):
        self._stats = {}
        for acct, grp in df.groupby("account_id"):
            q1, q3 = grp["amount_gbp"].quantile([0.25, 0.75])
            iqr = q3 - q1
            self._stats[acct] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def score(self, df) -> np.ndarray:
        scores = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            bounds = self._stats.get(row.account_id, (-np.inf, np.inf))
            lo, hi = bounds
            if row.amount_gbp > hi:
                scores[i] = min((row.amount_gbp - hi) / (hi - lo + 1e-9), 1.0)
            elif row.amount_gbp < lo:
                scores[i] = min((lo - row.amount_gbp) / (hi - lo + 1e-9), 1.0)
        return np.clip(scores, 0, 1)

    def predict(self, df) -> np.ndarray:
        return (self.score(df) > 0).astype(int)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.contamination = contamination
        self.pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("model", IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def _X(self, df) -> np.ndarray:
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        return df[cols].fillna(0).values

    def fit(self, df):
        self.pipe.fit(self._X(df))
        return self

    def score(self, df) -> np.ndarray:
        """Normalise anomaly score to [0,1] (higher = more anomalous)."""
        raw = self.pipe.named_steps["model"].score_samples(
            self.pipe.named_steps["scaler"].transform(self._X(df))
        )
        # score_samples returns negative; more negative = more anomalous
        s = -raw
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    def predict(self, df) -> np.ndarray:
        preds = self.pipe.predict(self._X(df))
        return (preds == -1).astype(int)


class AutoencoderProxyDetector:
    """
    PCA-based reconstruction error as a lightweight autoencoder proxy.
    High reconstruction error → anomaly.
    """
    def __init__(self, n_components: int = 10, threshold_quantile: float = 0.95):
        self.n_components = n_components
        self.threshold_quantile = threshold_quantile
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.threshold_ = None

    def _X(self, df) -> np.ndarray:
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        return df[cols].fillna(0).values

    def fit(self, df):
        X = self.scaler.fit_transform(self._X(df))
        self.pca.fit(X)
        recon_err = self._recon_error(X)
        self.threshold_ = np.quantile(recon_err, self.threshold_quantile)
        return self

    def _recon_error(self, X_scaled: np.ndarray) -> np.ndarray:
        X_recon = self.pca.inverse_transform(self.pca.transform(X_scaled))
        return np.mean((X_scaled - X_recon) ** 2, axis=1)

    def score(self, df) -> np.ndarray:
        X = self.scaler.transform(self._X(df))
        err = self._recon_error(X)
        return np.clip(err / (err.max() + 1e-9), 0, 1)

    def predict(self, df) -> np.ndarray:
        X = self.scaler.transform(self._X(df))
        err = self._recon_error(X)
        return (err > self.threshold_).astype(int)


class LOFDetector:
    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.scaler = RobustScaler()
        self._scores = None

    def _X(self, df) -> np.ndarray:
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        return df[cols].fillna(0).values

    def fit(self, df):
        # LOF is transductive — fit and score simultaneously
        X = self.scaler.fit_transform(self._X(df))
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        self._preds = self.lof_.fit_predict(X)
        raw = -self.lof_.negative_outlier_factor_
        self._scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        self._train_size = len(df)
        return self

    def score(self, df) -> np.ndarray:
        if self._scores is not None and len(df) == self._train_size:
            return self._scores
        # Re-fit on new data (LOF is transductive)
        X = self.scaler.transform(self._X(df))
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        lof.fit_predict(X)
        raw = -lof.negative_outlier_factor_
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    def predict(self, df) -> np.ndarray:
        if self._scores is not None and len(df) == self._train_size:
            return (self._preds == -1).astype(int)
        X = self.scaler.transform(self._X(df))
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        preds = lof.fit_predict(X)
        return (preds == -1).astype(int)


# ─────────────────────────────────────────────
# Ensemble Detector
# ─────────────────────────────────────────────

DETECTOR_WEIGHTS = {
    "zscore":      0.10,
    "iqr":         0.10,
    "iso_forest":  0.35,
    "autoencoder": 0.25,
    "lof":         0.20,
}


class EnsembleAnomalyDetector:
    """
    Weighted ensemble of all 5 detectors.

    Produces:
      - Per-detector binary flags
      - Per-detector raw anomaly scores [0,1]
      - Weighted ensemble score [0,1]
      - Final binary prediction at configurable threshold
      - Risk tier: Critical / High / Medium / Low
    """

    def __init__(self, contamination: float = 0.05, score_threshold: float = 0.45):
        self.contamination = contamination
        self.score_threshold = score_threshold

        self.detectors = {
            "zscore":      ZScoreDetector(threshold=3.0),
            "iqr":         IQRDetector(),
            "iso_forest":  IsolationForestDetector(contamination=contamination),
            "autoencoder": AutoencoderProxyDetector(),
            "lof":         LOFDetector(contamination=contamination),
        }
        self.is_fitted = False

    # ──────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────

    def fit(self, df_raw: pd.DataFrame) -> "EnsembleAnomalyDetector":
        logger.info("Building features…")
        df = build_features(df_raw)

        logger.info("Fitting all detectors…")
        for name, det in self.detectors.items():
            logger.info(f"  → {name}")
            det.fit(df)

        self.is_fitted = True
        logger.info("Ensemble fitted ✓")
        return self

    # ──────────────────────────────────────────
    # Detect
    # ──────────────────────────────────────────

    def detect(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            logger.info("Auto-fitting on provided data…")
            self.fit(df_raw)

        df = build_features(df_raw)
        results = df.copy()

        # Per-detector scores and flags
        ensemble_score = np.zeros(len(df))
        for name, det in self.detectors.items():
            s = det.score(df)
            p = det.predict(df)
            results[f"score_{name}"] = s
            results[f"flag_{name}"] = p
            ensemble_score += DETECTOR_WEIGHTS[name] * s

        results["ensemble_score"] = ensemble_score
        results["is_anomaly"] = (ensemble_score >= self.score_threshold).astype(int)

        # Risk tier
        results["risk_level"] = pd.cut(
            ensemble_score,
            bins=[-0.001, 0.30, 0.45, 0.65, 1.01],
            labels=["Low", "Medium", "High", "Critical"],
        ).astype(str)
        # Override: rule-based critical triggers
        results.loc[results["flag_over_10k"] == 1, "risk_level"] = "Critical"
        results.loc[results["flag_unknown_device"] == 1, "risk_level"] = "High"

        # Human-readable reason
        results["alert_reason"] = results.apply(self._explain_row, axis=1)

        return results

    # ──────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────

    def evaluate(self, results: pd.DataFrame) -> dict:
        """Compute evaluation metrics (requires true_label column)."""
        if "true_label" not in results.columns:
            raise ValueError("true_label column required for evaluation.")

        y_true = results["true_label"].values
        y_pred = results["is_anomaly"].values
        y_score = results["ensemble_score"].values

        metrics = {
            "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":          round(roc_auc_score(y_true, y_score), 4),
            "avg_precision":    round(average_precision_score(y_true, y_score), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "total_flagged":    int(y_pred.sum()),
            "true_positives":   int(((y_pred == 1) & (y_true == 1)).sum()),
            "false_positives":  int(((y_pred == 1) & (y_true == 0)).sum()),
            "false_negatives":  int(((y_pred == 0) & (y_true == 1)).sum()),
        }

        # Per anomaly type breakdown
        if "anomaly_type" in results.columns:
            metrics["per_type"] = (
                results[results["true_label"] == 1]
                .groupby("anomaly_type")
                .apply(lambda g: {
                    "n":       len(g),
                    "caught":  int(g["is_anomaly"].sum()),
                    "recall":  round(g["is_anomaly"].mean(), 3),
                })
                .to_dict()
            )

        return metrics

    # ──────────────────────────────────────────
    # Reporting helpers
    # ──────────────────────────────────────────

    def summary(self, results: pd.DataFrame) -> dict:
        total = len(results)
        flagged = results["is_anomaly"].sum()
        return {
            "total_transactions":  total,
            "anomalies_flagged":   int(flagged),
            "anomaly_rate_pct":    round(flagged / total * 100, 2),
            "critical_alerts":     int((results["risk_level"] == "Critical").sum()),
            "high_alerts":         int((results["risk_level"] == "High").sum()),
            "medium_alerts":       int((results["risk_level"] == "Medium").sum()),
            "total_flagged_gbp":   round(results.loc[results["is_anomaly"] == 1, "amount_gbp"].sum(), 2),
            "accounts_flagged":    int(results.loc[results["is_anomaly"] == 1, "account_id"].nunique()),
        }

    @staticmethod
    def _explain_row(row) -> str:
        reasons = []
        if row.get("flag_zscore"):         reasons.append("High z-score")
        if row.get("flag_iqr"):            reasons.append("IQR outlier")
        if row.get("flag_iso_forest"):     reasons.append("Isolation Forest")
        if row.get("flag_autoencoder"):    reasons.append("Reconstruction error")
        if row.get("flag_lof"):            reasons.append("LOF outlier")
        if row.get("flag_over_10k"):       reasons.append("Amount ≥ £10,000")
        if row.get("flag_unknown_device"): reasons.append("Unknown device")
        if row.get("flag_high_vel_1h"):    reasons.append("High velocity (1h)")
        if row.get("flag_night_tx"):       reasons.append("Night transaction")
        if row.get("flag_foreign_tx"):     reasons.append("Foreign currency")
        return "; ".join(reasons) if reasons else "—"

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self, path: str = "models/saved/ensemble_detector.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/saved/ensemble_detector.pkl") -> "EnsembleAnomalyDetector":
        obj = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return obj
