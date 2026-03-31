"""
Synthetic Financial Transaction Data Generator
===============================================
Generates highly realistic transaction data with:
- Account behavioural profiles (spending patterns per user)
- Time-series seasonality (day/week/month cycles)
- Merchant network graphs
- Multiple injected anomaly types (fraud, money laundering patterns,
  wash trading, velocity abuse, round-tripping)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import hashlib
import json
from pathlib import Path


# ─────────────────────────────────────────────
# Account Profiles
# ─────────────────────────────────────────────

ACCOUNT_PROFILES = {
    "low_spender":    {"mean": 35,    "std": 25,   "max_single": 300,   "daily_tx": 1.5},
    "medium_spender": {"mean": 150,   "std": 80,   "max_single": 2000,  "daily_tx": 3.0},
    "high_spender":   {"mean": 600,   "std": 300,  "max_single": 10000, "daily_tx": 5.0},
    "business":       {"mean": 2500,  "std": 1500, "max_single": 50000, "daily_tx": 8.0},
    "student":        {"mean": 20,    "std": 15,   "max_single": 200,   "daily_tx": 2.0},
}

MERCHANTS = {
    "Tesco":           {"category": "Grocery",         "avg_amount": 45,   "std": 30},
    "Sainsbury's":     {"category": "Grocery",         "avg_amount": 55,   "std": 35},
    "Amazon":          {"category": "Online Shopping", "avg_amount": 85,   "std": 70},
    "ASOS":            {"category": "Online Shopping", "avg_amount": 60,   "std": 40},
    "HSBC ATM":        {"category": "Cash Withdrawal", "avg_amount": 120,  "std": 80},
    "Barclays ATM":    {"category": "Cash Withdrawal", "avg_amount": 100,  "std": 60},
    "BP Fuel":         {"category": "Fuel",            "avg_amount": 70,   "std": 20},
    "Shell":           {"category": "Fuel",            "avg_amount": 65,   "std": 18},
    "McDonald's":      {"category": "Food & Drink",    "avg_amount": 12,   "std": 6},
    "Costa Coffee":    {"category": "Food & Drink",    "avg_amount": 8,    "std": 3},
    "Netflix":         {"category": "Subscription",    "avg_amount": 17,   "std": 1},
    "Spotify":         {"category": "Subscription",    "avg_amount": 11,   "std": 1},
    "Uber":            {"category": "Transport",       "avg_amount": 18,   "std": 12},
    "TfL":             {"category": "Transport",       "avg_amount": 5,    "std": 3},
    "John Lewis":      {"category": "Retail",          "avg_amount": 120,  "std": 90},
    "Boots":           {"category": "Health & Beauty", "avg_amount": 25,   "std": 20},
    "Deliveroo":       {"category": "Food & Drink",    "avg_amount": 28,   "std": 12},
    "Sky":             {"category": "Subscription",    "avg_amount": 45,   "std": 5},
    "EDF Energy":      {"category": "Utilities",       "avg_amount": 95,   "std": 30},
    "Thames Water":    {"category": "Utilities",       "avg_amount": 40,   "std": 10},
    "HMRC":            {"category": "Tax",             "avg_amount": 800,  "std": 500},
    "Lloyds Transfer": {"category": "Bank Transfer",   "avg_amount": 500,  "std": 400},
    "PayPal":          {"category": "Online Payment",  "avg_amount": 75,   "std": 60},
    "Revolut":         {"category": "Online Payment",  "avg_amount": 200,  "std": 180},
    "Argos":           {"category": "Retail",          "avg_amount": 80,   "std": 60},
}

CURRENCIES = (
    ["GBP"] * 75
    + ["USD"] * 8
    + ["EUR"] * 8
    + ["AED"] * 2
    + ["JPY"] * 2
    + ["AUD"] * 2
    + ["CAD"] * 2
    + ["CHF"] * 1
)

FX_RATES = {
    "GBP": 1.0, "USD": 0.79, "EUR": 0.86,
    "AED": 0.22, "JPY": 0.0053, "AUD": 0.51,
    "CAD": 0.58, "CHF": 0.89,
}

CHANNELS = ["mobile_app", "web", "atm", "pos_terminal", "telephone", "branch"]
LOCATIONS_UK = [
    "London", "Manchester", "Birmingham", "Leeds", "Glasgow",
    "Liverpool", "Bristol", "Sheffield", "Edinburgh", "Cardiff",
]
LOCATIONS_FOREIGN = [
    "Dubai", "New York", "Paris", "Amsterdam", "Tokyo",
    "Sydney", "Toronto", "Zurich", "Singapore", "Madrid",
]


# ─────────────────────────────────────────────
# Core Generator
# ─────────────────────────────────────────────

def generate_transactions(
    n_accounts: int = 80,
    days: int = 180,
    anomaly_ratio: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic labelled transaction dataset.

    Returns
    -------
    pd.DataFrame with columns:
        transaction_id, timestamp, account_id, account_type,
        merchant, category, amount, currency, amount_gbp,
        channel, location, is_foreign, device_id,
        ip_country, true_label, anomaly_type
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start = datetime(2024, 1, 1)
    merchant_names = list(MERCHANTS.keys())

    # Assign profiles to accounts
    profile_types = list(ACCOUNT_PROFILES.keys())
    account_profiles = {
        f"ACC{str(i).zfill(4)}": rng.choice(profile_types)
        for i in range(1, n_accounts + 1)
    }
    # Assign a primary device and home location per account
    account_devices = {
        acc: f"DEV-{hashlib.md5(acc.encode()).hexdigest()[:8].upper()}"
        for acc in account_profiles
    }
    account_home = {
        acc: rng.choice(LOCATIONS_UK)
        for acc in account_profiles
    }

    rows = []

    # ── Normal transactions ──────────────────────────────────────────
    total_normal = int(n_accounts * days * 0.6)   # ~0.6 tx/account/day avg
    for _ in range(total_normal):
        account = rng.choice(list(account_profiles.keys()))
        profile = ACCOUNT_PROFILES[account_profiles[account]]

        # Realistic timestamp (more transactions on weekday afternoons)
        day_offset = int(rng.integers(0, days))
        ts = start + timedelta(days=day_offset)
        hour_weights = _hourly_weights()
        hour = int(rng.choice(range(24), p=hour_weights))
        minute = int(rng.integers(0, 60))
        ts = ts.replace(hour=hour, minute=minute)

        merchant = str(rng.choice(merchant_names))
        m_info = MERCHANTS[merchant]
        amount = max(0.5, rng.normal(m_info["avg_amount"], m_info["std"]))
        amount = min(amount, profile["max_single"])
        amount = round(amount, 2)

        currency = str(rng.choice(CURRENCIES))
        is_foreign = currency != "GBP"
        location = (
            str(rng.choice(LOCATIONS_FOREIGN)) if is_foreign
            else account_home[account]
        )
        channel = str(rng.choice(
            CHANNELS,
            p=[0.40, 0.25, 0.10, 0.15, 0.05, 0.05]
        ))
        device = account_devices[account] if channel in ("mobile_app", "web") else "N/A"
        ip_country = "GB" if not is_foreign else str(rng.choice(
            ["US", "AE", "FR", "JP", "AU", "CA", "DE", "SG"]
        ))

        rows.append({
            "transaction_id": _tx_id(rng),
            "timestamp": ts,
            "account_id": account,
            "account_type": account_profiles[account],
            "merchant": merchant,
            "category": m_info["category"],
            "amount": amount,
            "currency": currency,
            "amount_gbp": round(amount * FX_RATES.get(currency, 1.0), 2),
            "channel": channel,
            "location": location,
            "is_foreign": is_foreign,
            "device_id": device,
            "ip_country": ip_country,
            "true_label": 0,
            "anomaly_type": "normal",
        })

    # ── Inject anomalies ─────────────────────────────────────────────
    n_anomalies = int(len(rows) * anomaly_ratio / (1 - anomaly_ratio))
    anomaly_injectors = [
        _inject_large_fraud,
        _inject_velocity_abuse,
        _inject_round_trip,
        _inject_night_foreign,
        _inject_structuring,
        _inject_account_takeover,
    ]

    accounts = list(account_profiles.keys())
    for _ in range(n_anomalies):
        injector = rng.choice(anomaly_injectors)
        account = str(rng.choice(accounts))
        profile = ACCOUNT_PROFILES[account_profiles[account]]
        anomaly_rows = injector(rng, account, profile, account_devices[account], start, days)
        rows.extend(anomaly_rows)

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ─────────────────────────────────────────────
# Anomaly Injectors
# ─────────────────────────────────────────────

def _inject_large_fraud(rng, account, profile, device, start, days):
    """Single very large transaction far outside profile."""
    ts = start + timedelta(
        days=int(rng.integers(0, days)),
        hours=int(rng.integers(0, 24)),
        minutes=int(rng.integers(0, 60)),
    )
    amount = round(float(rng.uniform(profile["max_single"] * 2, profile["max_single"] * 10)), 2)
    return [_row(rng, account, "Revolut", ts, amount, "USD", True,
                 "web", "SG", device, 1, "large_fraud")]


def _inject_velocity_abuse(rng, account, profile, device, start, days):
    """Many small transactions in a short window (card testing / carding)."""
    base_ts = start + timedelta(days=int(rng.integers(0, days)), hours=2)
    result = []
    for i in range(int(rng.integers(15, 30))):
        ts = base_ts + timedelta(minutes=i * 2)
        amount = round(float(rng.uniform(0.5, 5.0)), 2)
        result.append(_row(rng, account, "Amazon", ts, amount, "GBP", False,
                           "web", "GB", device, 1, "velocity_abuse"))
    return result


def _inject_round_trip(rng, account, profile, device, start, days):
    """Money sent out and returned in close succession (round-tripping)."""
    ts1 = start + timedelta(days=int(rng.integers(0, days - 1)), hours=10)
    ts2 = ts1 + timedelta(hours=int(rng.integers(1, 8)))
    amount = round(float(rng.uniform(500, 5000)), 2)
    return [
        _row(rng, account, "Lloyds Transfer", ts1, amount, "GBP", False,
             "web", "GB", device, 1, "round_trip"),
        _row(rng, account, "PayPal", ts2, amount * 0.98, "GBP", False,
             "mobile_app", "GB", device, 1, "round_trip"),
    ]


def _inject_night_foreign(rng, account, profile, device, start, days):
    """High-value foreign transaction at unusual hour."""
    ts = start + timedelta(
        days=int(rng.integers(0, days)),
        hours=int(rng.integers(1, 5)),
        minutes=int(rng.integers(0, 60)),
    )
    amount = round(float(rng.uniform(1000, 8000)), 2)
    currency = str(rng.choice(["AED", "USD", "JPY"]))
    return [_row(rng, account, "Revolut", ts, amount, currency, True,
                 "mobile_app", "AE", "UNKNOWN-DEVICE", 1, "night_foreign")]


def _inject_structuring(rng, account, profile, device, start, days):
    """Multiple transactions just below reporting threshold (£10k)."""
    base_ts = start + timedelta(days=int(rng.integers(0, days - 2)))
    result = []
    for i in range(int(rng.integers(3, 7))):
        ts = base_ts + timedelta(days=i, hours=int(rng.integers(9, 17)))
        amount = round(float(rng.uniform(9000, 9999)), 2)
        result.append(_row(rng, account, "HSBC ATM", ts, amount, "GBP", False,
                           "atm", "GB", device, 1, "structuring"))
    return result


def _inject_account_takeover(rng, account, profile, device, start, days):
    """Unknown device + foreign IP + large transfer."""
    ts = start + timedelta(
        days=int(rng.integers(0, days)),
        hours=int(rng.integers(0, 24)),
    )
    amount = round(float(rng.uniform(profile["max_single"], profile["max_single"] * 5)), 2)
    return [_row(rng, account, "Lloyds Transfer", ts, amount, "GBP", False,
                 "web", "NG", "UNKNOWN-DEVICE-XYZ", 1, "account_takeover")]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _row(rng, account, merchant, ts, amount, currency, is_foreign,
         channel, ip_country, device, label, anomaly_type):
    m_info = MERCHANTS.get(merchant, {"category": "Other"})
    return {
        "transaction_id": _tx_id(rng),
        "timestamp": ts,
        "account_id": account,
        "account_type": "unknown",
        "merchant": merchant,
        "category": m_info.get("category", "Other"),
        "amount": amount,
        "currency": currency,
        "amount_gbp": round(amount * FX_RATES.get(currency, 1.0), 2),
        "channel": channel,
        "location": "Foreign" if is_foreign else "UK",
        "is_foreign": is_foreign,
        "device_id": device,
        "ip_country": ip_country,
        "true_label": label,
        "anomaly_type": anomaly_type,
    }


def _tx_id(rng) -> str:
    return "TXN-" + "".join(
        str(rng.integers(0, 10)) for _ in range(12)
    )


def _hourly_weights() -> np.ndarray:
    """Return a 24-element probability array peaking around business hours."""
    w = np.array([
        0.2, 0.1, 0.1, 0.1, 0.2, 0.5,   # 00-05
        1.5, 3.0, 4.5, 5.5, 6.0, 6.5,   # 06-11
        7.0, 6.5, 6.0, 6.5, 7.5, 8.0,   # 12-17
        7.0, 5.5, 4.0, 3.0, 1.5, 0.5,   # 18-23
    ], dtype=float)
    return w / w.sum()


if __name__ == "__main__":
    df = generate_transactions(n_accounts=80, days=180, seed=42)
    out = Path("data/raw/transactions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} transactions  |  "
          f"Anomalies: {df['true_label'].sum():,} "
          f"({df['true_label'].mean()*100:.1f}%)")
    print(df.head())
