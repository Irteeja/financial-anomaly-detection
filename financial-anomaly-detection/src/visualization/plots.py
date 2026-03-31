"""
Visualization Module
====================
Generates all analytical plots:
  - Anomaly score distribution
  - ROC and Precision-Recall curves
  - Feature importance (permutation-based)
  - Temporal anomaly heatmap
  - Account risk scatter
  - Per-detector agreement matrix
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────
DARK_BG   = "#0d0f14"
SURFACE   = "#13161e"
SURFACE2  = "#1a1e2b"
BORDER    = "#252836"
RED       = "#e74c3c"
AMBER     = "#f39c12"
BLUE      = "#3498db"
GREEN     = "#27ae60"
PURPLE    = "#9b59b6"
TEXT      = "#e8eaf0"
MUTED     = "#7b8197"

PALETTE = [RED, BLUE, GREEN, AMBER, PURPLE, "#1abc9c", "#e67e22", "#95a5a6"]

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.facecolor":  SURFACE2,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT,
    })


# ─────────────────────────────────────────────
# 1. Score Distribution
# ─────────────────────────────────────────────

def plot_score_distribution(results: pd.DataFrame, out_dir: str = "reports") -> str:
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Ensemble Anomaly Score Distribution", color=TEXT, fontsize=14, fontweight="bold", y=1.02)

    # Left: histogram coloured by true label
    ax = axes[0]
    if "true_label" in results.columns:
        normal = results.loc[results["true_label"] == 0, "ensemble_score"]
        anomaly = results.loc[results["true_label"] == 1, "ensemble_score"]
        ax.hist(normal,  bins=50, alpha=0.7, color=BLUE,  label=f"Normal  (n={len(normal):,})",  density=True)
        ax.hist(anomaly, bins=50, alpha=0.8, color=RED,   label=f"Anomaly (n={len(anomaly):,})", density=True)
        ax.legend(fontsize=9)
    else:
        ax.hist(results["ensemble_score"], bins=50, color=BLUE, alpha=0.8, density=True)

    ax.axvline(results["ensemble_score"].median(), color=AMBER, linewidth=1.2,
               linestyle="--", label="Median")
    ax.set_xlabel("Ensemble Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Label", color=TEXT, pad=8)
    ax.grid(True, alpha=0.3)

    # Right: box plots per risk level
    ax2 = axes[1]
    risk_order = ["Low", "Medium", "High", "Critical"]
    colors_map  = {"Low": GREEN, "Medium": BLUE, "High": AMBER, "Critical": RED}
    data_by_risk = [
        results.loc[results["risk_level"] == r, "ensemble_score"].values
        for r in risk_order
        if r in results["risk_level"].values
    ]
    labels_present = [r for r in risk_order if r in results["risk_level"].values]
    bp = ax2.boxplot(data_by_risk, patch_artist=True, widths=0.5,
                     medianprops=dict(color=TEXT, linewidth=2))
    for patch, label in zip(bp["boxes"], labels_present):
        patch.set_facecolor(colors_map.get(label, BLUE))
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(MUTED)

    ax2.set_xticklabels(labels_present)
    ax2.set_ylabel("Ensemble Score")
    ax2.set_title("Score by Risk Level", color=TEXT, pad=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = f"{out_dir}/score_distribution.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 2. ROC + PR Curves
# ─────────────────────────────────────────────

def plot_roc_pr(results: pd.DataFrame, out_dir: str = "reports") -> str:
    if "true_label" not in results.columns:
        return ""
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    fig.suptitle("Model Performance Curves", color=TEXT, fontsize=14, fontweight="bold")

    y_true  = results["true_label"].values
    y_score = results["ensemble_score"].values

    # ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=RED, lw=2, label=f"ROC AUC = {roc_auc_val:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=RED)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", color=TEXT, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # PR
    ax2 = axes[1]
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = auc(rec, prec)
    baseline = y_true.mean()
    ax2.plot(rec, prec, color=BLUE, lw=2, label=f"PR AUC = {pr_auc_val:.4f}")
    ax2.fill_between(rec, prec, alpha=0.08, color=BLUE)
    ax2.axhline(baseline, color=MUTED, lw=1, linestyle="--",
                label=f"Baseline ({baseline:.2%})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", color=TEXT, pad=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    plt.tight_layout()
    path = f"{out_dir}/roc_pr_curves.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 3. Temporal Heatmap
# ─────────────────────────────────────────────

def plot_temporal_heatmap(results: pd.DataFrame, out_dir: str = "reports") -> str:
    _apply_dark_style()
    df = results.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df[df["is_anomaly"] == 1]
        .groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
        .pivot(index="day_of_week", columns="hour", values="count")
        .reindex(day_order)
        .fillna(0)
    )
    # Fill missing hours
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
    cmap = LinearSegmentedColormap.from_list("risk", [SURFACE2, AMBER, RED])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order, fontsize=9)
    ax.set_title("Anomaly Heatmap — Hour of Day × Day of Week", color=TEXT, fontsize=13, pad=10)
    ax.set_xlabel("Hour of Day", color=TEXT)
    ax.set_facecolor(SURFACE)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Anomaly Count", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
    plt.tight_layout()
    path = f"{out_dir}/temporal_heatmap.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 4. Detector Agreement
# ─────────────────────────────────────────────

def plot_detector_agreement(results: pd.DataFrame, out_dir: str = "reports") -> str:
    _apply_dark_style()
    det_cols = [c for c in results.columns if c.startswith("flag_") and
                c in ["flag_zscore", "flag_iqr", "flag_iso_forest",
                       "flag_autoencoder", "flag_lof"]]
    if len(det_cols) < 2:
        return ""

    labels = [c.replace("flag_", "").replace("_", " ").title() for c in det_cols]
    corr = results[det_cols].astype(float).corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Detector Agreement Analysis", color=TEXT, fontsize=14, fontweight="bold")

    # Correlation heatmap
    ax = axes[0]
    cmap2 = LinearSegmentedColormap.from_list("corr", [BLUE, SURFACE2, RED])
    im = ax.imshow(corr.values, cmap=cmap2, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color=TEXT)
    ax.set_title("Detector Correlation Matrix", color=TEXT, pad=8)
    cbar2 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=MUTED)

    # Bar: how many transactions each detector flagged
    ax2 = axes[1]
    counts = results[det_cols].sum().values
    bars = ax2.barh(labels, counts, color=PALETTE[:len(labels)], alpha=0.85, height=0.6)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(count):,}", va="center", fontsize=9, color=MUTED)
    ax2.set_xlabel("Transactions Flagged")
    ax2.set_title("Flags per Detector", color=TEXT, pad=8)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim(0, max(counts) * 1.15)

    plt.tight_layout()
    path = f"{out_dir}/detector_agreement.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 5. Account Risk Scatter
# ─────────────────────────────────────────────

def plot_account_risk_scatter(results: pd.DataFrame, out_dir: str = "reports") -> str:
    _apply_dark_style()
    acct = (
        results.groupby("account_id")
        .agg(
            total_txns=("transaction_id", "count"),
            anomaly_count=("is_anomaly", "sum"),
            max_score=("ensemble_score", "max"),
            total_gbp=("amount_gbp", "sum"),
        )
        .reset_index()
    )
    acct["anomaly_rate"] = acct["anomaly_count"] / acct["total_txns"]

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
    sc = ax.scatter(
        acct["total_txns"],
        acct["anomaly_rate"] * 100,
        s=np.sqrt(acct["total_gbp"]) * 0.4 + 20,
        c=acct["max_score"],
        cmap=LinearSegmentedColormap.from_list("r", [GREEN, AMBER, RED]),
        alpha=0.8,
        edgecolors=BORDER,
        linewidths=0.5,
        vmin=0, vmax=1,
    )

    # Annotate top-risk accounts
    top = acct.nlargest(8, "anomaly_count")
    for _, row in top.iterrows():
        ax.annotate(
            row["account_id"],
            (row["total_txns"], row["anomaly_rate"] * 100),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=MUTED,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Max Ensemble Score", color=TEXT, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
    cbar.ax.yaxis.set_tick_params(color=MUTED)

    ax.set_xlabel("Total Transactions per Account")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Account Risk Landscape\n(bubble size ∝ total spend)", color=TEXT, fontsize=13, pad=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f"{out_dir}/account_risk_scatter.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 6. Daily Anomaly Timeline
# ─────────────────────────────────────────────

def plot_daily_timeline(results: pd.DataFrame, out_dir: str = "reports") -> str:
    _apply_dark_style()
    df = results.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = (
        df.groupby("date")
        .agg(total=("transaction_id", "count"),
             anomalies=("is_anomaly", "sum"))
        .reset_index()
    )
    daily["rate"] = daily["anomalies"] / daily["total"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=DARK_BG, sharex=True)
    fig.suptitle("Daily Transaction Timeline", color=TEXT, fontsize=14, fontweight="bold")

    dates = range(len(daily))

    # Total txns
    ax = axes[0]
    ax.fill_between(dates, daily["total"], alpha=0.4, color=BLUE)
    ax.plot(dates, daily["total"], color=BLUE, lw=1.5)
    ax.set_ylabel("Total Txns")
    ax.set_title("Daily Transaction Volume", color=TEXT, fontsize=11, pad=4)
    ax.grid(True, alpha=0.2)

    # Anomaly count
    ax2 = axes[1]
    ax2.fill_between(dates, daily["anomalies"], alpha=0.4, color=RED)
    ax2.plot(dates, daily["anomalies"], color=RED, lw=1.5)
    ax2.set_ylabel("Anomalies")
    ax2.set_title("Daily Anomaly Count", color=TEXT, fontsize=11, pad=4)
    ax2.grid(True, alpha=0.2)

    # Anomaly rate
    ax3 = axes[2]
    ax3.fill_between(dates, daily["rate"], alpha=0.4, color=AMBER)
    ax3.plot(dates, daily["rate"], color=AMBER, lw=1.5)
    ax3.axhline(daily["rate"].mean(), color=MUTED, lw=1, linestyle="--",
                label=f"Mean {daily['rate'].mean():.1f}%")
    ax3.set_ylabel("Anomaly Rate %")
    ax3.set_title("Daily Anomaly Rate", color=TEXT, fontsize=11, pad=4)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlabel("Days")

    # X-axis labels every 2 weeks
    tick_indices = list(range(0, len(daily), 14))
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(
        [str(daily["date"].iloc[i]) for i in tick_indices],
        rotation=30, ha="right", fontsize=8
    )

    plt.tight_layout()
    path = f"{out_dir}/daily_timeline.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 7. Generate All Plots
# ─────────────────────────────────────────────

def generate_all_plots(results: pd.DataFrame, out_dir: str = "reports") -> dict:
    """Run all plot functions and return dict of {name: filepath}."""
    plots = {}
    fns = [
        ("score_distribution", plot_score_distribution),
        ("roc_pr_curves",      plot_roc_pr),
        ("temporal_heatmap",   plot_temporal_heatmap),
        ("detector_agreement", plot_detector_agreement),
        ("account_risk",       plot_account_risk_scatter),
        ("daily_timeline",     plot_daily_timeline),
    ]
    for name, fn in fns:
        try:
            path = fn(results, out_dir=out_dir)
            if path:
                plots[name] = path
                print(f"  ✓ {name} → {path}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
    return plots
