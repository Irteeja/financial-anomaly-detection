"""
Main Pipeline — Financial Risk & Anomaly Detection System
==========================================================
Entry point: generates data → engineers features → runs ensemble → evaluates → reports.

Usage:
    python main.py                    # full pipeline
    python main.py --accounts 120 --days 365
    python main.py --load-model       # load saved model
    python main.py --no-report        # skip HTML report
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Financial Anomaly Detection Pipeline")
    p.add_argument("--accounts",    type=int,   default=80,    help="Number of accounts")
    p.add_argument("--days",        type=int,   default=180,   help="Simulation period (days)")
    p.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    p.add_argument("--contamination",type=float,default=0.05,  help="Expected anomaly rate")
    p.add_argument("--threshold",   type=float, default=0.45,  help="Ensemble score threshold")
    p.add_argument("--load-model",  action="store_true",        help="Load saved model")
    p.add_argument("--save-model",  action="store_true",        help="Save fitted model")
    p.add_argument("--no-report",   action="store_true",        help="Skip HTML report")
    p.add_argument("--output-csv",  type=str,   default="data/processed/results.csv")
    p.add_argument("--output-html", type=str,   default="reports/html/report.html")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── 1. Generate data ────────────────────────────────────────────
    logger.info("━━━ Step 1: Generating synthetic transaction data ━━━")
    from src.data_generator import generate_transactions

    df = generate_transactions(
        n_accounts=args.accounts,
        days=args.days,
        seed=args.seed,
    )
    logger.info(
        f"Generated {len(df):,} transactions | "
        f"True anomalies: {df['true_label'].sum():,} "
        f"({df['true_label'].mean()*100:.1f}%)"
    )

    # Save raw data
    raw_path = Path("data/raw/transactions.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved → {raw_path}")

    # ── 2. Load or fit detector ─────────────────────────────────────
    logger.info("━━━ Step 2: Initialising Ensemble Detector ━━━")
    from src.detectors.ensemble_detector import EnsembleAnomalyDetector

    model_path = "models/saved/ensemble_detector.pkl"

    if args.load_model and Path(model_path).exists():
        logger.info(f"Loading saved model from {model_path}")
        detector = EnsembleAnomalyDetector.load(model_path)
    else:
        detector = EnsembleAnomalyDetector(
            contamination=args.contamination,
            score_threshold=args.threshold,
        )
        logger.info("Fitting ensemble (this may take ~30 seconds)…")
        detector.fit(df)

    if args.save_model:
        detector.save(model_path)

    # ── 3. Run detection ────────────────────────────────────────────
    logger.info("━━━ Step 3: Running detection pipeline ━━━")
    results = detector.detect(df)

    # Save processed results
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    # ── 4. Evaluate ─────────────────────────────────────────────────
    logger.info("━━━ Step 4: Evaluating performance ━━━")
    metrics = detector.evaluate(results)
    summary = detector.summary(results)

    # Print summary
    print("\n" + "═" * 56)
    print("  FINANCIAL RISK & ANOMALY DETECTION — RESULTS")
    print("═" * 56)
    print(f"  Total transactions :  {summary['total_transactions']:>8,}")
    print(f"  Anomalies flagged  :  {summary['anomalies_flagged']:>8,}  "
          f"({summary['anomaly_rate_pct']}%)")
    print(f"  Critical alerts    :  {summary['critical_alerts']:>8,}")
    print(f"  High alerts        :  {summary['high_alerts']:>8,}")
    print(f"  Total flagged £GBP :  £{summary['total_flagged_gbp']:>10,.2f}")
    print(f"  Accounts flagged   :  {summary['accounts_flagged']:>8,}")
    print("─" * 56)
    print(f"  Precision          :  {metrics['precision']:>8.4f}")
    print(f"  Recall             :  {metrics['recall']:>8.4f}")
    print(f"  F1 Score           :  {metrics['f1_score']:>8.4f}")
    print(f"  ROC-AUC            :  {metrics['roc_auc']:>8.4f}")
    print(f"  Avg Precision      :  {metrics['avg_precision']:>8.4f}")
    print(f"  True Positives     :  {metrics['true_positives']:>8,}")
    print(f"  False Positives    :  {metrics['false_positives']:>8,}")
    print(f"  False Negatives    :  {metrics['false_negatives']:>8,}")
    print("─" * 56)
    if "per_type" in metrics:
        print("  Detection by anomaly type:")
        for atype, info in metrics["per_type"].items():
            bar = "█" * int(info["recall"] * 20)
            print(f"    {atype:<22} {bar:<20} {info['recall']*100:.0f}% "
                  f"({info['caught']}/{info['n']})")
    print("═" * 56 + "\n")

    # Save metrics JSON
    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({**summary, **metrics}, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # ── 5. HTML Report ───────────────────────────────────────────────
    if not args.no_report:
        logger.info("━━━ Step 5: Generating HTML report ━━━")
        from src.reporting.html_report import generate_html_report

        report_path = generate_html_report(
            results=results,
            metrics=metrics,
            summary=summary,
            output_path=args.output_html,
        )
        logger.info(f"HTML report → {report_path}")

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s ✓")


if __name__ == "__main__":
    main()
