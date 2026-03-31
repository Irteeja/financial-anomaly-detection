"""
HTML Report Generator
=====================
Produces a self-contained, styled HTML risk report from detection results.
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


def _spark(values, width=80, height=24):
    """Tiny inline SVG sparkline."""
    if not values or max(values) == min(values):
        return ""
    mn, mx = min(values), max(values)
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * width if len(values) > 1 else 0
        y = height - (v - mn) / (mx - mn) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{" ".join(pts)}" fill="none" '
        f'stroke="#e74c3c" stroke-width="1.5" stroke-linejoin="round"/>'
        f"</svg>"
    )


def generate_html_report(
    results: pd.DataFrame,
    metrics: dict,
    summary: dict,
    output_path: str = "reports/html/report.html",
) -> str:
    """
    Generate a full HTML risk report.

    Parameters
    ----------
    results     : detect() output DataFrame
    metrics     : evaluate() dict
    summary     : summary() dict
    output_path : where to write the HTML file

    Returns
    -------
    str : path to generated report
    """
    ts = datetime.now().strftime("%d %B %Y, %H:%M")
    anomalies = results[results["is_anomaly"] == 1].copy()

    # Daily anomaly count for sparkline
    results["date_only"] = pd.to_datetime(results["timestamp"]).dt.date
    daily = results.groupby("date_only")["is_anomaly"].sum().values.tolist()

    # Top flagged accounts
    top_accounts = (
        anomalies.groupby("account_id")
        .agg(
            alerts=("is_anomaly", "sum"),
            total_gbp=("amount_gbp", "sum"),
            max_score=("ensemble_score", "max"),
        )
        .sort_values("alerts", ascending=False)
        .head(10)
        .reset_index()
    )

    # Risk distribution
    risk_counts = results["risk_level"].value_counts().to_dict()

    # Anomaly type breakdown (if available)
    type_rows = ""
    if "per_type" in metrics:
        for atype, info in metrics["per_type"].items():
            pct = info["recall"] * 100
            type_rows += f"""
            <tr>
              <td>{atype.replace("_", " ").title()}</td>
              <td>{info["n"]}</td>
              <td>{info["caught"]}</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar" style="width:{pct:.0f}%"></div>
                  <span>{pct:.0f}%</span>
                </div>
              </td>
            </tr>"""

    # Top anomaly rows
    cols_show = ["transaction_id", "account_id", "timestamp", "merchant",
                 "amount_gbp", "currency", "risk_level", "ensemble_score", "alert_reason"]
    cols_show = [c for c in cols_show if c in anomalies.columns]
    alert_rows = ""
    for _, row in anomalies.sort_values("ensemble_score", ascending=False).head(20).iterrows():
        risk_cls = {"Critical": "risk-crit", "High": "risk-high",
                    "Medium": "risk-med", "Low": "risk-low"}.get(row.get("risk_level", ""), "")
        alert_rows += f"""
        <tr>
          <td class="mono">{row.get("transaction_id","")}</td>
          <td class="mono">{row.get("account_id","")}</td>
          <td>{str(row.get("timestamp",""))[:16]}</td>
          <td>{row.get("merchant","")}</td>
          <td class="num">£{row.get("amount_gbp",0):,.2f}</td>
          <td><span class="badge {risk_cls}">{row.get("risk_level","")}</span></td>
          <td class="num">{row.get("ensemble_score",0):.3f}</td>
          <td class="reason">{row.get("alert_reason","")}</td>
        </tr>"""

    account_rows = ""
    for _, row in top_accounts.iterrows():
        account_rows += f"""
        <tr>
          <td class="mono">{row["account_id"]}</td>
          <td class="num">{int(row["alerts"])}</td>
          <td class="num">£{row["total_gbp"]:,.2f}</td>
          <td class="num">{row["max_score"]:.3f}</td>
        </tr>"""

    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    spark_svg = _spark(daily)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Risk & Anomaly Detection — Report</title>
<style>
  :root {{
    --bg: #0d0f14;
    --surface: #13161e;
    --surface2: #1a1e2b;
    --border: #252836;
    --accent: #e74c3c;
    --accent2: #f39c12;
    --green: #27ae60;
    --blue: #3498db;
    --text: #e8eaf0;
    --muted: #7b8197;
    --mono: "JetBrains Mono", "Courier New", monospace;
    --sans: "Inter", "Segoe UI", sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans);
          font-size: 14px; line-height: 1.6; }}
  a {{ color: var(--blue); text-decoration: none; }}

  /* ── Layout ── */
  .page {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px; }}
  header {{ display: flex; align-items: center; justify-content: space-between;
            border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 32px; }}
  .logo {{ display: flex; align-items: center; gap: 12px; }}
  .logo-icon {{ width: 36px; height: 36px; background: var(--accent);
               border-radius: 8px; display: flex; align-items: center;
               justify-content: center; font-size: 18px; }}
  h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }}
  .subtitle {{ color: var(--muted); font-size: 12px; }}
  .report-meta {{ text-align: right; color: var(--muted); font-size: 12px; }}

  /* ── KPI cards ── */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 20px; }}
  .kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                letter-spacing: 0.8px; margin-bottom: 8px; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; letter-spacing: -1px; }}
  .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .kpi.red .kpi-value   {{ color: var(--accent); }}
  .kpi.amber .kpi-value {{ color: var(--accent2); }}
  .kpi.green .kpi-value {{ color: var(--green); }}
  .kpi.blue .kpi-value  {{ color: var(--blue); }}

  /* ── Metrics row ── */
  .metrics-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 32px; }}
  .metric {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-name  {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                   letter-spacing: 0.6px; margin-bottom: 6px; }}
  .metric-value {{ font-size: 22px; font-weight: 700; }}
  .metric.good  {{ border-color: var(--green); }}
  .metric.warn  {{ border-color: var(--accent2); }}
  .metric.bad   {{ border-color: var(--accent); }}

  /* ── Panels ── */
  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }}
  .panel-header {{ padding: 16px 20px; border-bottom: 1px solid var(--border);
                   font-size: 13px; font-weight: 600; display: flex;
                   align-items: center; justify-content: space-between; }}
  .panel-body {{ padding: 20px; }}
  .full-width {{ grid-column: 1 / -1; }}

  /* ── Tables ── */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 12px; color: var(--muted);
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
        border-bottom: 1px solid var(--border); font-weight: 500; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .num {{ text-align: right; font-family: var(--mono); }}
  .mono {{ font-family: var(--mono); font-size: 12px; }}
  .reason {{ color: var(--muted); font-size: 12px; max-width: 280px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  /* ── Badges ── */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.4px; }}
  .risk-crit {{ background: rgba(231,76,60,0.2); color: #e74c3c; }}
  .risk-high {{ background: rgba(243,156,18,0.2); color: #f39c12; }}
  .risk-med  {{ background: rgba(52,152,219,0.2); color: #3498db; }}
  .risk-low  {{ background: rgba(39,174,96,0.2); color: #27ae60; }}

  /* ── Bar ── */
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 8px; background: var(--accent); border-radius: 4px; min-width: 2px; max-width: 160px; }}
  .bar-wrap span {{ color: var(--muted); font-size: 12px; }}

  /* ── Confusion matrix ── */
  .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; max-width: 300px; }}
  .cm-cell {{ padding: 20px; border-radius: 8px; text-align: center; }}
  .cm-label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .cm-val {{ font-size: 26px; font-weight: 700; }}
  .cm-tp {{ background: rgba(39,174,96,0.2); color: var(--green); }}
  .cm-fp {{ background: rgba(231,76,60,0.15); color: var(--accent); }}
  .cm-fn {{ background: rgba(243,156,18,0.15); color: var(--accent2); }}
  .cm-tn {{ background: rgba(52,152,219,0.1); color: var(--blue); }}

  /* ── Sparkline area ── */
  .spark-wrap {{ margin-top: 8px; }}

  /* ── Risk donut (CSS-only) ── */
  .risk-bars {{ display: flex; flex-direction: column; gap: 10px; }}
  .risk-row {{ display: flex; align-items: center; gap: 10px; }}
  .risk-label {{ width: 70px; font-size: 12px; color: var(--muted); }}
  .risk-bar {{ flex: 1; height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }}
  .risk-fill {{ height: 100%; border-radius: 5px; transition: width 0.3s; }}
  .risk-count {{ width: 40px; text-align: right; font-size: 12px; font-family: var(--mono); }}

  footer {{ border-top: 1px solid var(--border); margin-top: 32px;
            padding-top: 16px; color: var(--muted); font-size: 12px;
            display: flex; justify-content: space-between; }}
</style>
</head>
<body>
<div class="page">

  <header>
    <div class="logo">
      <div class="logo-icon">⚡</div>
      <div>
        <div class="subtitle">Financial Risk Intelligence</div>
        <h1>Anomaly Detection Report</h1>
      </div>
    </div>
    <div class="report-meta">
      Generated: {ts}<br>
      Transactions analysed: {summary["total_transactions"]:,}<br>
      Engine: Ensemble v2.0 (5 detectors)
    </div>
  </header>

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi red">
      <div class="kpi-label">Anomalies Detected</div>
      <div class="kpi-value">{summary["anomalies_flagged"]:,}</div>
      <div class="kpi-sub">{summary["anomaly_rate_pct"]}% of all transactions</div>
    </div>
    <div class="kpi amber">
      <div class="kpi-label">Critical + High Alerts</div>
      <div class="kpi-value">{summary["critical_alerts"] + summary["high_alerts"]:,}</div>
      <div class="kpi-sub">{summary["critical_alerts"]} Critical · {summary["high_alerts"]} High</div>
    </div>
    <div class="kpi red">
      <div class="kpi-label">Total Flagged Value</div>
      <div class="kpi-value">£{summary["total_flagged_gbp"]:,.0f}</div>
      <div class="kpi-sub">Across {summary["accounts_flagged"]} accounts</div>
    </div>
    <div class="kpi green">
      <div class="kpi-label">ROC-AUC Score</div>
      <div class="kpi-value">{metrics.get("roc_auc", 0):.3f}</div>
      <div class="kpi-sub">F1: {metrics.get("f1_score", 0):.3f}</div>
    </div>
  </div>

  <!-- Model Metrics -->
  <div class="metrics-row">
    <div class="metric good">
      <div class="metric-name">Precision</div>
      <div class="metric-value">{metrics.get("precision", 0):.3f}</div>
    </div>
    <div class="metric good">
      <div class="metric-name">Recall</div>
      <div class="metric-value">{metrics.get("recall", 0):.3f}</div>
    </div>
    <div class="metric good">
      <div class="metric-name">F1 Score</div>
      <div class="metric-value">{metrics.get("f1_score", 0):.3f}</div>
    </div>
    <div class="metric good">
      <div class="metric-name">ROC-AUC</div>
      <div class="metric-value">{metrics.get("roc_auc", 0):.3f}</div>
    </div>
    <div class="metric good">
      <div class="metric-name">Avg Precision</div>
      <div class="metric-value">{metrics.get("avg_precision", 0):.3f}</div>
    </div>
  </div>

  <!-- Panels row 1 -->
  <div class="panels">

    <!-- Risk distribution -->
    <div class="panel">
      <div class="panel-header">Risk Distribution</div>
      <div class="panel-body">
        <div class="risk-bars">
          {_risk_bar("Critical", risk_counts.get("Critical", 0), summary["anomalies_flagged"], "#e74c3c")}
          {_risk_bar("High",     risk_counts.get("High", 0),     summary["anomalies_flagged"], "#f39c12")}
          {_risk_bar("Medium",   risk_counts.get("Medium", 0),   summary["anomalies_flagged"], "#3498db")}
          {_risk_bar("Low",      risk_counts.get("Low", 0),       summary["anomalies_flagged"], "#27ae60")}
        </div>
      </div>
    </div>

    <!-- Confusion matrix -->
    <div class="panel">
      <div class="panel-header">Confusion Matrix</div>
      <div class="panel-body">
        <div class="cm-grid">
          <div class="cm-cell cm-tp">
            <div class="cm-label">True Positive</div>
            <div class="cm-val">{metrics.get("true_positives", cm[1][1] if len(cm) > 1 else 0)}</div>
          </div>
          <div class="cm-cell cm-fp">
            <div class="cm-label">False Positive</div>
            <div class="cm-val">{metrics.get("false_positives", cm[0][1] if len(cm) > 0 else 0)}</div>
          </div>
          <div class="cm-cell cm-fn">
            <div class="cm-label">False Negative</div>
            <div class="cm-val">{metrics.get("false_negatives", cm[1][0] if len(cm) > 1 else 0)}</div>
          </div>
          <div class="cm-cell cm-tn">
            <div class="cm-label">True Negative</div>
            <div class="cm-val">{cm[0][0] if len(cm) > 0 else 0}</div>
          </div>
        </div>
        <div class="spark-wrap">
          <div class="kpi-label" style="margin-top:16px">Daily anomaly trend</div>
          {spark_svg}
        </div>
      </div>
    </div>

    <!-- Anomaly type breakdown -->
    <div class="panel">
      <div class="panel-header">Detection by Anomaly Type</div>
      <div class="panel-body">
        <table>
          <thead><tr>
            <th>Type</th><th>Total</th><th>Caught</th><th>Recall</th>
          </tr></thead>
          <tbody>{type_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- Top accounts -->
    <div class="panel">
      <div class="panel-header">Top Flagged Accounts</div>
      <div class="panel-body">
        <table>
          <thead><tr>
            <th>Account</th><th>Alerts</th><th>Total GBP</th><th>Max Score</th>
          </tr></thead>
          <tbody>{account_rows}</tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- Alert table -->
  <div class="panel full-width" style="margin-bottom:32px">
    <div class="panel-header">
      Top 20 Alerts by Ensemble Score
      <span style="color:var(--muted);font-size:12px;font-weight:400">sorted by risk score ↓</span>
    </div>
    <div class="panel-body" style="padding:0">
      <table>
        <thead><tr>
          <th>Transaction ID</th><th>Account</th><th>Timestamp</th>
          <th>Merchant</th><th>Amount</th><th>Risk</th><th>Score</th><th>Reason</th>
        </tr></thead>
        <tbody>{alert_rows}</tbody>
      </table>
    </div>
  </div>

  <footer>
    <span>Financial Risk & Anomaly Detection System · Ensemble v2.0</span>
    <span>Report generated {ts}</span>
  </footer>
</div>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def _risk_bar(label, count, total, color):
    pct = round(count / max(total, 1) * 100)
    return f"""
    <div class="risk-row">
      <span class="risk-label">{label}</span>
      <div class="risk-bar">
        <div class="risk-fill" style="width:{pct}%;background:{color}"></div>
      </div>
      <span class="risk-count">{count}</span>
    </div>"""
