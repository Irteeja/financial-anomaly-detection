# Financial Risk & Anomaly Detection System

A transaction monitoring system that flags suspicious financial activity using an ensemble of statistical and ML-based methods. Built with compliance monitoring and fraud detection in mind.

[![CI Pipeline](https://github.com/Irteeja/financial-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Irteeja/financial-anomaly-detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

The idea behind this project was to go beyond a single anomaly detection algorithm — in practice, no one method catches everything. So I combined five different approaches (two statistical, two ML-based, one density-based) into a weighted ensemble that scores each transaction from 0 to 1. Anything above the threshold gets flagged with a risk tier and a human-readable reason.

The system detects six types of anomalous behaviour:

- **Large fraud** — a single transaction way outside the account's normal spending range
- **Velocity abuse** — dozens of small transactions in a short window (classic card testing pattern)
- **Round-tripping** — money going out and coming back within hours (money laundering signal)
- **Night + foreign** — high-value foreign currency transactions at unusual hours
- **Structuring** — multiple transactions just under the £10,000 reporting threshold (deliberate)
- **Account takeover** — unknown device, foreign IP, and a large transfer all at once

---

## The Five Detectors

| Method | Type | What it catches |
|--------|------|----------------|
| Z-Score | Statistical | Amounts with \|z\| > 3.0 vs the account's own history |
| IQR Fence | Statistical | Per-account interquartile range violations |
| Isolation Forest | ML (unsupervised) | Globally unusual patterns across 35+ features |
| PCA Autoencoder proxy | Reconstruction error | Transactions that don't reconstruct well from principal components |
| Local Outlier Factor | Density-based | Transactions that look fine globally but are odd within their local neighbourhood |

Each detector outputs a score between 0 and 1. The final ensemble score is a weighted average — Isolation Forest carries the most weight (35%), followed by the PCA proxy (25%), LOF (20%), and the two statistical methods (10% each). The weights reflect how each method performed on labelled holdout data.

---

## Getting Started

```bash
git clone https://github.com/Irteeja/financial-anomaly-detection.git
cd financial-anomaly-detection
pip install -r requirements.txt
python main.py
```

The pipeline will generate synthetic transaction data, fit the ensemble, evaluate performance, and write an HTML report to `reports/html/report.html`.

You can tweak things via CLI flags:

```bash
# Larger dataset, longer time window
python main.py --accounts 120 --days 365

# More sensitive — catches more but increases false positives
python main.py --threshold 0.35 --contamination 0.08

# Skip the HTML report if you just want the terminal output
python main.py --no-report

# Save the fitted model, then reload it next time (skips re-fitting)
python main.py --save-model
python main.py --load-model
```

---

## Project Structure

```
financial-anomaly-detection/
│
├── main.py                          # Run everything from here
├── requirements.txt
├── config/settings.yaml             # All tunable parameters in one place
│
├── src/
│   ├── data_generator.py            # Synthetic data with realistic account profiles
│   ├── detectors/
│   │   └── ensemble_detector.py     # All 5 detectors + ensemble logic
│   ├── features/
│   │   └── feature_engineering.py   # 35+ features across 6 groups
│   ├── visualization/
│   │   └── plots.py                 # 6 dark-theme analysis plots
│   ├── reporting/
│   │   └── html_report.py           # Self-contained HTML risk report
│   └── api/
│       └── app.py                   # FastAPI REST endpoint
│
├── data/raw/                        # Generated transaction CSVs
├── data/processed/                  # Detection results with scores
├── models/saved/                    # Persisted model files (.pkl)
├── reports/html/                    # Generated HTML reports
├── notebooks/exploration.ipynb      # Interactive walkthrough notebook
├── tests/test_pipeline.py           # 30+ unit and integration tests
└── .github/workflows/ci.yml         # CI — runs tests on every push
```

---

## Feature Engineering

Raw transaction data doesn't carry much signal on its own. The engineering step builds context — comparing each transaction against the account's own history, measuring what's happened in the last hour, checking for device switches, and so on. There are 35+ features across six groups:

**Temporal** — hour of day encoded cyclically (so 23:00 and 00:00 are close together in feature space), day of week, weekend flag, night flag.

**Behavioural** — z-score vs the account's own history, ratio to rolling 7-day mean, deviation from account median, amount vs merchant category average.

**Velocity** — transaction count in 1h / 6h / 24h / 72h windows, total GBP spent in 1h and 24h windows. These are computed per account using a sliding window scan.

**Network** — merchant diversity per account, whether this is the first time a merchant has been used, merchant frequency, encoded categoricals.

**Device** — known device flag, unknown device flag, device-switched flag (changed from previous transaction).

**Compliance flags** — binary hard rules: amount ≥ £10k, round amount, foreign transaction, high velocity, night transaction, unknown device, foreign IP with known device (mismatch).

---

## Example Output

```
══════════════════════════════════════════════════════════
  FINANCIAL RISK & ANOMALY DETECTION — RESULTS
══════════════════════════════════════════════════════════
  Total transactions :     2,341
  Anomalies flagged  :       127  (5.4%)
  Critical alerts    :        18
  High alerts        :        41
  Total flagged £GBP :   £342,891.00
  Accounts flagged   :        35
────────────────────────────────────────────────────────
  Precision          :    0.7812
  Recall             :    0.8421
  F1 Score           :    0.8105
  ROC-AUC            :    0.9234
  Avg Precision      :    0.8671
  True Positives     :        96
  False Positives    :        27
  False Negatives    :        18
────────────────────────────────────────────────────────
  Detection by anomaly type:
    Large Fraud            ████████████████████ 100% (12/12)
    Account Takeover       ████████████████████  95% (19/20)
    Structuring            ████████████████      80% (16/20)
    Velocity Abuse         ██████████████        70% (14/20)
    Night Foreign          ████████████████████ 100% (10/10)
    Round Trip             ████████████████      80% (14/18)
══════════════════════════════════════════════════════════
```

---

## Architecture

```
Raw Transactions
      │
      ▼
Feature Engineering (35+ features)
      │
      ├─── Z-Score Detector ────────────────┐
      ├─── IQR Fence Detector ──────────────┤
      ├─── Isolation Forest ────────────────┤──► Weighted Ensemble Score [0,1]
      ├─── PCA Autoencoder (proxy) ─────────┤          │
      └─── Local Outlier Factor ────────────┘          ▼
                                               Binary Flag + Risk Tier
                                                         │
                                                         ▼
                                               HTML Report + CSV + JSON
```

---

## REST API

If you want to integrate this into another system, there's a FastAPI endpoint included:

```bash
pip install fastapi uvicorn
uvicorn src.api.app:app --reload --port 8000
```

Then POST transactions to `http://localhost:8000/detect` — it returns scores, risk tiers, and alert reasons for each one. Full docs at `/docs` once the server is running.

---

## Testing

```bash
# Run everything
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Skip the slower integration tests
pytest tests/ -v -k "not Integration"
```

There are 30+ tests covering the data generator, each individual detector, the feature engineering pipeline, and the full end-to-end flow.

---

## Dependencies

| Package | Used for |
|---------|---------|
| `scikit-learn` | Isolation Forest, LOF, PCA, scalers |
| `scipy` | Z-score, IQR |
| `pandas` | Data wrangling |
| `numpy` | Numerical ops |
| `matplotlib` | Plots |
| `joblib` | Model serialisation |
| `fastapi` + `uvicorn` | REST API (optional) |

---

## License

MIT — see [LICENSE](LICENSE)
