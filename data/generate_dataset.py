"""
data/generate_dataset.py
========================
Generates a synthetic 90-day EHR access log based on the CERT r4.1
insider threat schema, customized for healthcare environments.

Produces:
  - data/ehr_access_log.csv       — 3,757 EHR access events (primary dataset)
  - data/user_profiles.csv        — 150 user behavioral baseline profiles
  - data/ground_truth_labels.csv  — Ground-truth malicious flags for evaluation
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── Configuration ───────────────────────────────────────────────────────────
NUM_USERS         = 150
MALICIOUS_RATIO   = 0.06           # 6% malicious users (≈9 users)
SIM_DAYS          = 90
TARGET_EVENTS     = 3757
START_DATE        = datetime(2024, 1, 1)

ROLES = {
    "Doctor":           {"weight": 0.30, "record_limit": 20, "privilege": "high"},
    "Nurse":            {"weight": 0.30, "record_limit": 15, "privilege": "medium"},
    "Admin":            {"weight": 0.15, "record_limit": 50, "privilege": "high"},
    "Receptionist":     {"weight": 0.10, "record_limit":  5, "privilege": "low"},
    "ITSupport":        {"weight": 0.08, "record_limit": 10, "privilege": "medium"},
    "Pharmacist":       {"weight": 0.05, "record_limit":  8, "privilege": "medium"},
    "ChiefMedicalOfficer": {"weight": 0.02, "record_limit": 100, "privilege": "high"},
}

DEPARTMENTS = ["Cardiology", "Oncology", "Neurology", "Pediatrics",
               "Emergency", "Radiology", "Pharmacy", "IT", "Administration"]

ACCESS_TYPES = ["view", "edit", "download", "print", "export", "delete"]
DEVICES      = ["workstation", "laptop", "tablet", "mobile", "external_usb"]
LOCATIONS    = ["office", "remote", "vpn", "unknown"]


# ── Step 1: Generate User Profiles ─────────────────────────────────────────
def generate_users() -> pd.DataFrame:
    role_list   = list(ROLES.keys())
    role_weights = [ROLES[r]["weight"] for r in role_list]

    users = []
    malicious_indices = set(random.sample(range(NUM_USERS), int(NUM_USERS * MALICIOUS_RATIO)))

    for i in range(NUM_USERS):
        role = random.choices(role_list, weights=role_weights, k=1)[0]
        dept = random.choice(DEPARTMENTS)
        is_malicious = i in malicious_indices

        users.append({
            "user_id":          f"USR{i:04d}",
            "username":         fake.user_name(),
            "full_name":        fake.name(),
            "role":             role,
            "department":       dept,
            "privilege_level":  ROLES[role]["privilege"],
            "record_limit":     ROLES[role]["record_limit"],
            "is_malicious":     is_malicious,
            "hire_date":        fake.date_between(start_date="-5y", end_date="-6m"),
            "termination_flag": random.random() < 0.03 if is_malicious else False,
        })

    return pd.DataFrame(users)


# ── Step 2: Generate EHR Access Events ─────────────────────────────────────
def _off_hours(dt: datetime) -> bool:
    return not (7 <= dt.hour < 19) or dt.weekday() >= 5


def generate_events(users_df: pd.DataFrame) -> pd.DataFrame:
    events = []

    for _, user in users_df.iterrows():
        is_mal = user["is_malicious"]

        # Malicious users generate ~3× more events
        n_events = random.randint(30, 80) if is_mal else random.randint(10, 40)

        for _ in range(n_events):
            day_offset = random.randint(0, SIM_DAYS - 1)
            base_ts    = START_DATE + timedelta(days=day_offset)

            # Malicious users prefer off-hours
            if is_mal and random.random() < 0.55:
                hour = random.choice(list(range(0, 7)) + list(range(19, 24)))
            else:
                hour = random.randint(7, 18)

            ts = base_ts.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
            )

            records_accessed = (
                random.randint(user["record_limit"], user["record_limit"] * 4)
                if is_mal and random.random() < 0.4
                else random.randint(1, user["record_limit"])
            )

            access_type = (
                random.choices(ACCESS_TYPES, weights=[5, 5, 40, 15, 25, 10], k=1)[0]
                if is_mal
                else random.choices(ACCESS_TYPES, weights=[60, 20, 5, 5, 5, 5], k=1)[0]
            )

            device = (
                random.choices(DEVICES, weights=[10, 20, 10, 10, 50], k=1)[0]
                if is_mal
                else random.choices(DEVICES, weights=[60, 25, 8, 5, 2], k=1)[0]
            )

            location = (
                random.choices(LOCATIONS, weights=[10, 30, 20, 40], k=1)[0]
                if is_mal
                else random.choices(LOCATIONS, weights=[70, 15, 12, 3], k=1)[0]
            )

            session_duration = (
                random.randint(5, 30)
                if is_mal
                else random.randint(15, 120)
            )

            external_emails = (
                random.randint(3, 15) if is_mal and random.random() < 0.3 else 0
            )

            events.append({
                "event_id":              f"EVT{len(events):06d}",
                "timestamp":             ts,
                "user_id":               user["user_id"],
                "username":              user["username"],
                "role":                  user["role"],
                "department":            user["department"],
                "privilege_level":       user["privilege_level"],
                "record_limit":          user["record_limit"],
                "records_accessed":      records_accessed,
                "access_type":           access_type,
                "device_type":           device,
                "location":              location,
                "session_duration_min":  session_duration,
                "external_emails_sent":  external_emails,
                "off_hours_access":      int(_off_hours(ts)),
                "bulk_download_flag":    int(access_type in ["download", "export"] and records_accessed > 10),
                "unauthorized_access":   int(records_accessed > user["record_limit"]),
                "usb_connected":         int(device == "external_usb"),
                "failed_auth_attempts":  random.randint(3, 10) if is_mal and random.random() < 0.2 else random.randint(0, 1),
                "vpn_usage":             int(location == "vpn"),
                "is_malicious":          int(is_mal),
            })

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)

    # Trim/pad to TARGET_EVENTS
    if len(df) > TARGET_EVENTS:
        df = df.sample(TARGET_EVENTS, random_state=42).sort_values("timestamp").reset_index(drop=True)

    return df


# ── Step 3: Engineer Behavioral Features ───────────────────────────────────
def engineer_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Add 38 behavioral features required by the ML engine."""
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Rolling user-level aggregates (last 7 days per event)
    df = df.sort_values("timestamp")
    user_stats = (
        df.groupby("user_id")
        .agg(
            avg_records_per_session  = ("records_accessed", "mean"),
            max_records_session      = ("records_accessed", "max"),
            total_events             = ("event_id", "count"),
            off_hours_ratio          = ("off_hours_access", "mean"),
            bulk_download_ratio      = ("bulk_download_flag", "mean"),
            external_email_ratio     = ("external_emails_sent", lambda x: (x > 0).mean()),
            avg_session_duration     = ("session_duration_min", "mean"),
            usb_usage_count          = ("usb_connected", "sum"),
            failed_auth_ratio        = ("failed_auth_attempts", "mean"),
            unique_access_types      = ("access_type", "nunique"),
            vpn_ratio                = ("vpn_usage", "mean"),
            weekend_access_ratio     = ("is_weekend", "mean"),
            unauthorized_ratio       = ("unauthorized_access", "mean"),
            external_email_count     = ("external_emails_sent", "sum"),
        )
        .reset_index()
    )

    df = df.merge(user_stats, on="user_id", how="left")

    # Derived risk signals
    df["records_over_limit"]    = (df["records_accessed"] / df["record_limit"].clip(lower=1)).clip(upper=5)
    df["session_anomaly_score"] = (df["session_duration_min"] < 10).astype(int)
    df["after_midnight"]        = ((df["hour"] >= 0) & (df["hour"] < 5)).astype(int)
    df["privilege_risk"]        = df["privilege_level"].map({"high": 1.0, "medium": 0.5, "low": 0.2})
    df["device_risk"]           = df["device_type"].map({
        "external_usb": 1.0, "mobile": 0.7, "tablet": 0.5,
        "laptop": 0.3, "workstation": 0.1,
    })
    df["location_risk"]         = df["location"].map({
        "unknown": 1.0, "remote": 0.6, "vpn": 0.4, "office": 0.1,
    })
    df["access_type_risk"]      = df["access_type"].map({
        "delete": 1.0, "export": 0.9, "download": 0.8,
        "print": 0.5, "edit": 0.3, "view": 0.1,
    })

    return df


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports/incidents", exist_ok=True)

    print("🏥 Generating user profiles...")
    users_df = generate_users()
    users_df.to_csv("data/user_profiles.csv", index=False)
    print(f"   ✅ {len(users_df)} users created ({users_df['is_malicious'].sum()} malicious)")

    print("📋 Generating EHR access events...")
    events_df = generate_events(users_df)
    print(f"   ✅ {len(events_df)} raw events generated")

    print("🔧 Engineering behavioral features...")
    featured_df = engineer_features(events_df)
    featured_df.to_csv("data/ehr_access_log.csv", index=False)
    print(f"   ✅ {featured_df.shape[1]} features engineered")

    # Ground truth labels
    labels_df = featured_df[["event_id", "user_id", "timestamp", "is_malicious"]].copy()
    labels_df.to_csv("data/ground_truth_labels.csv", index=False)

    print("\n📊 Dataset Summary:")
    print(f"   Total Events : {len(featured_df):,}")
    print(f"   Malicious    : {featured_df['is_malicious'].sum():,} ({featured_df['is_malicious'].mean()*100:.1f}%)")
    print(f"   Features     : {featured_df.shape[1]}")
    print(f"   Date Range   : {featured_df['timestamp'].min()} → {featured_df['timestamp'].max()}")
    print("\n✅ Dataset saved to data/")


if __name__ == "__main__":
    main()
