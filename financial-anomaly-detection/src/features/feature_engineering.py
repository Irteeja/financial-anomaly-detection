"""
Feature Engineering Pipeline
==============================
Transforms raw transaction records into a rich ML-ready feature matrix.

Feature groups:
  - Temporal   : hour, day-of-week, is_weekend, is_night, month
  - Behavioural: account-level rolling stats, z-scores, deviation ratios
  - Velocity   : per-account transaction counts across multiple windows
  - Network    : merchant / IP-country co-occurrence entropy
  - Device     : known-device flag, device-switching flag
  - Compliance : rule-based binary flags (round-amount, foreign, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Main pipeline entry-point
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df : raw transaction DataFrame (output of data_generator)

    Returns
    -------
    df_feat : DataFrame with all engineered features appended
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    df = _temporal_features(df)
    df = _behavioural_features(df)
    df = _velocity_features(df)
    df = _network_features(df)
    df = _device_features(df)
    df = _compliance_flags(df)

    return df


# ─────────────────────────────────────────────
# Feature Groups
# ─────────────────────────────────────────────

def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["day_of_month"]= df["timestamp"].dt.day
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    df["is_morning"]  = ((df["hour"] >= 6) & (df["hour"] <= 9)).astype(int)
    # Cyclical encoding so 23h → 0h is close in feature space
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account statistical features."""

    # Amount z-score (within account)
    df["amount_zscore"] = (
        df.groupby("account_id")["amount_gbp"]
        .transform(lambda x: stats.zscore(x, nan_policy="omit"))
        .fillna(0)
    )

    # Account-level global stats (on past data only — use expanding mean)
    grp = df.groupby("account_id")["amount_gbp"]
    df["acct_mean"]   = grp.transform(lambda x: x.expanding().mean().shift(1)).fillna(df["amount_gbp"])
    df["acct_std"]    = grp.transform(lambda x: x.expanding().std().shift(1)).fillna(1)
    df["acct_median"] = grp.transform(lambda x: x.expanding().median().shift(1)).fillna(df["amount_gbp"])
    df["acct_max"]    = grp.transform(lambda x: x.expanding().max().shift(1)).fillna(df["amount_gbp"])

    # Deviation from personal mean
    df["amount_vs_mean"]   = (df["amount_gbp"] - df["acct_mean"]).clip(-1e5, 1e5)
    df["amount_mean_ratio"] = (df["amount_gbp"] / df["acct_mean"].replace(0, 1)).clip(0, 50)

    # Rolling 7-day window stats
    df["date"] = df["timestamp"].dt.date
    df["rolling_7d_mean"] = (
        df.groupby("account_id")["amount_gbp"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
        .fillna(df["amount_gbp"])
    )
    df["rolling_7d_std"] = (
        df.groupby("account_id")["amount_gbp"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).std().shift(1))
        .fillna(1)
    )
    df["rolling_7d_ratio"] = (
        df["amount_gbp"] / df["rolling_7d_mean"].replace(0, 1)
    ).clip(0, 50)

    # Category amount z-score (amount vs that merchant category)
    df["cat_amount_zscore"] = (
        df.groupby("category")["amount_gbp"]
        .transform(lambda x: stats.zscore(x, nan_policy="omit"))
        .fillna(0)
    )

    return df


def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction frequency / velocity across multiple time windows."""

    df["ts_unix"] = df["timestamp"].astype(np.int64) // 10**9  # seconds

    for window_hours, col in [(1, "vel_1h"), (6, "vel_6h"), (24, "vel_24h"), (72, "vel_72h")]:
        window_sec = window_hours * 3600
        counts = []
        for _, grp in df.groupby("account_id"):
            grp = grp.sort_values("ts_unix")
            ts_vals = grp["ts_unix"].values
            c = np.array([
                np.searchsorted(ts_vals, ts_vals[i], side="left")
                - np.searchsorted(ts_vals, ts_vals[i] - window_sec, side="left")
                for i in range(len(ts_vals))
            ])
            counts.append(pd.Series(c, index=grp.index))
        df[col] = pd.concat(counts).reindex(df.index).fillna(1)

    # Amount velocity: total GBP in window
    for window_hours, col in [(1, "amt_vel_1h"), (24, "amt_vel_24h")]:
        window_sec = window_hours * 3600
        amts = []
        for _, grp in df.groupby("account_id"):
            grp = grp.sort_values("ts_unix")
            ts_vals = grp["ts_unix"].values
            amt_vals = grp["amount_gbp"].values
            s = np.array([
                amt_vals[
                    np.searchsorted(ts_vals, ts_vals[i] - window_sec, side="left"):i
                ].sum()
                for i in range(len(ts_vals))
            ])
            amts.append(pd.Series(s, index=grp.index))
        df[col] = pd.concat(amts).reindex(df.index).fillna(0)

    df["daily_tx_rank"] = df.groupby(["account_id", "date"])["ts_unix"].rank(method="first")

    return df


def _network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merchant diversity and IP-country entropy per account."""

    # Number of unique merchants per account (global)
    merchant_diversity = (
        df.groupby("account_id")["merchant"].nunique().rename("acct_merchant_diversity")
    )
    df = df.join(merchant_diversity, on="account_id")

    # Number of unique IP countries per account
    ip_diversity = (
        df.groupby("account_id")["ip_country"].nunique().rename("acct_ip_diversity")
    )
    df = df.join(ip_diversity, on="account_id")

    # Merchant frequency: how often this merchant appears for this account
    df["merchant_acct_freq"] = (
        df.groupby(["account_id", "merchant"])["transaction_id"]
        .transform("count")
    )

    # Is this a new merchant for the account? (first time seen)
    df["is_new_merchant"] = (
        df.groupby(["account_id", "merchant"]).cumcount() == 0
    ).astype(int)

    # Encode categorical (merchant, channel, ip_country)
    for col in ["merchant", "channel", "ip_country", "category"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


def _device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Device consistency signals."""

    # Most common device per account
    df["acct_primary_device"] = (
        df.groupby("account_id")["device_id"]
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
    )
    df["is_known_device"] = (df["device_id"] == df["acct_primary_device"]).astype(int)
    df["is_unknown_device"] = (df["device_id"] == "UNKNOWN-DEVICE").astype(int)

    # Device switch: different device from previous transaction on same account
    df["prev_device"] = df.groupby("account_id")["device_id"].shift(1)
    df["device_switched"] = (
        (df["device_id"] != df["prev_device"]) & df["prev_device"].notna()
    ).astype(int)

    return df


def _compliance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Hard rule-based binary compliance flags."""

    df["flag_over_10k"]       = (df["amount_gbp"] >= 10_000).astype(int)
    df["flag_over_1k"]        = (df["amount_gbp"] >= 1_000).astype(int)
    df["flag_round_amount"]   = ((df["amount_gbp"] % 500 == 0) & (df["amount_gbp"] >= 500)).astype(int)
    df["flag_foreign_tx"]     = df["is_foreign"].astype(int)
    df["flag_night_tx"]       = df["is_night"]
    df["flag_high_vel_1h"]    = (df["vel_1h"] >= 5).astype(int)
    df["flag_high_vel_24h"]   = (df["vel_24h"] >= 20).astype(int)
    df["flag_high_amt_1h"]    = (df["amt_vel_1h"] >= 5_000).astype(int)
    df["flag_unknown_device"] = df["is_unknown_device"]
    df["flag_new_merchant"]   = df["is_new_merchant"]
    df["flag_foreign_ip"]     = (df["ip_country"] != "GB").astype(int)
    df["flag_ip_device_mismatch"] = (
        (df["flag_foreign_ip"] == 1) & (df["is_known_device"] == 1)
    ).astype(int)
    df["compliance_score"]    = df[
        [c for c in df.columns if c.startswith("flag_")]
    ].sum(axis=1)

    return df


# ─────────────────────────────────────────────
# Feature list for ML models
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    # temporal
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_night", "is_morning", "month",
    # amount behavioural
    "amount_gbp", "amount_zscore", "amount_vs_mean", "amount_mean_ratio",
    "rolling_7d_ratio", "rolling_7d_std", "cat_amount_zscore",
    # velocity
    "vel_1h", "vel_6h", "vel_24h", "vel_72h",
    "amt_vel_1h", "amt_vel_24h", "daily_tx_rank",
    # network
    "acct_merchant_diversity", "acct_ip_diversity",
    "merchant_acct_freq", "is_new_merchant",
    "merchant_enc", "channel_enc", "ip_country_enc", "category_enc",
    # device
    "is_known_device", "is_unknown_device", "device_switched",
    # compliance
    "compliance_score",
]
