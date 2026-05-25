"""
models/train_models.py
======================
Trains the Dual-Model ML Engine:
  1. Isolation Forest  (unsupervised anomaly detection, 40% weight)
  2. Random Forest     (supervised classifier, 60% weight)

Saves trained models + SHAP explainer to models/saved/
Outputs evaluation metrics to console.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ── Feature columns used by the ML engine ──────────────────────────────────
FEATURE_COLS = [
    # Raw event features
    "records_accessed", "session_duration_min", "external_emails_sent",
    "off_hours_access", "bulk_download_flag", "unauthorized_access",
    "usb_connected", "failed_auth_attempts", "vpn_usage",
    "is_weekend", "after_midnight",
    # Engineered user-level aggregates
    "avg_records_per_session", "max_records_session", "total_events",
    "off_hours_ratio", "bulk_download_ratio", "external_email_ratio",
    "avg_session_duration", "usb_usage_count", "failed_auth_ratio",
    "unique_access_types", "vpn_ratio", "weekend_access_ratio",
    "unauthorized_ratio", "external_email_count",
    # Derived risk signals
    "records_over_limit", "session_anomaly_score",
    "privilege_risk", "device_risk", "location_risk", "access_type_risk",
]

IF_WEIGHT  = 0.40   # Isolation Forest contribution
RF_WEIGHT  = 0.60   # Random Forest contribution

CONTAMINATION = 0.06  # matches malicious ratio
N_ESTIMATORS  = 200


def load_data(path: str = "data/ehr_access_log.csv"):
    df = pd.read_csv(path)
    # Fill any NaN values that may arise from user-level aggregates
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    print("  Training Isolation Forest (unsupervised)...")
    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    iso.fit(X)
    print(f"  ✅ Isolation Forest trained on {X.shape[0]:,} samples")
    return iso


def train_random_forest(X_train, y_train) -> Pipeline:
    print("  Training Random Forest (supervised)...")

    # SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"     After SMOTE: {X_res.shape[0]:,} samples "
          f"({int(y_res.sum())} malicious / {int((y_res==0).sum())} benign)")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_res, y_res)
    print(f"  ✅ Random Forest trained")
    return pipeline


def evaluate_models(iso, rf_pipeline, X_test, y_test, scaler):
    print("\n📊 Model Evaluation")
    print("=" * 50)

    # IF anomaly scores → [0,1] probability proxy
    if_scores = iso.decision_function(X_test)
    if_norm   = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

    # RF probabilities
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

    # Ensemble score
    ensemble_score = IF_WEIGHT * if_norm + RF_WEIGHT * rf_probs
    y_pred = (ensemble_score >= 0.50).astype(int)

    print("\n--- Random Forest ---")
    print(classification_report(y_test, rf_pipeline.predict(X_test),
                                 target_names=["Benign", "Malicious"]))
    print(f"ROC-AUC (RF):       {roc_auc_score(y_test, rf_probs):.4f}")
    print(f"Avg Precision (RF): {average_precision_score(y_test, rf_probs):.4f}")

    print("\n--- Ensemble (IF + RF) ---")
    print(classification_report(y_test, y_pred,
                                 target_names=["Benign", "Malicious"]))
    print(f"ROC-AUC (Ensemble): {roc_auc_score(y_test, ensemble_score):.4f}")
    print(f"Avg Precision (En): {average_precision_score(y_test, ensemble_score):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    metrics = {
        "roc_auc_rf":       round(roc_auc_score(y_test, rf_probs), 4),
        "roc_auc_ensemble": round(roc_auc_score(y_test, ensemble_score), 4),
        "avg_precision":    round(average_precision_score(y_test, ensemble_score), 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def build_shap_explainer(rf_pipeline, X_train_scaled: np.ndarray):
    """Build a TreeExplainer for fast SHAP computation at inference time."""
    clf = rf_pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf, feature_perturbation="interventional")
    return explainer


def save_artifacts(iso, rf_pipeline, scaler, explainer, metrics, feature_cols):
    os.makedirs("models/saved", exist_ok=True)

    joblib.dump(iso,         "models/saved/isolation_forest.pkl")
    joblib.dump(rf_pipeline, "models/saved/random_forest_pipeline.pkl")
    joblib.dump(scaler,      "models/saved/feature_scaler.pkl")
    joblib.dump(explainer,   "models/saved/shap_explainer.pkl")

    meta = {
        "feature_cols":    feature_cols,
        "if_weight":       IF_WEIGHT,
        "rf_weight":       RF_WEIGHT,
        "contamination":   CONTAMINATION,
        "n_estimators":    N_ESTIMATORS,
        "risk_thresholds": {
            "allow":    0.30,
            "restrict": 0.55,
            "block":    0.75,
        },
        "evaluation": metrics,
    }
    with open("models/saved/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n💾 Saved artifacts:")
    for fn in ["isolation_forest.pkl", "random_forest_pipeline.pkl",
               "feature_scaler.pkl", "shap_explainer.pkl", "model_metadata.json"]:
        print(f"   models/saved/{fn}")


def plot_feature_importance(rf_pipeline, feature_cols):
    clf = rf_pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.barh([feature_cols[i] for i in idx[::-1]],
             importances[idx[::-1]], color="steelblue")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("models/saved/feature_importance.png", dpi=150)
    plt.close()
    print("   models/saved/feature_importance.png")


def main():
    print("🏥 Healthcare Security — Dual-Model ML Training\n")

    print("📂 Loading dataset...")
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df["is_malicious"].values
    print(f"   Loaded {len(df):,} events | {int(y.sum())} malicious ({y.mean()*100:.1f}%)")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Shared scaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\n🤖 Training models...")
    iso        = train_isolation_forest(X_train_scaled)
    rf_pipeline = train_random_forest(X_train, y_train)

    print("\n📈 Evaluating...")
    metrics = evaluate_models(iso, rf_pipeline, X_test, y_test, scaler)

    print("\n🔍 Building SHAP explainer...")
    explainer = build_shap_explainer(rf_pipeline, X_train_scaled)
    print("   ✅ SHAP TreeExplainer ready")

    print("\n💾 Saving artifacts...")
    save_artifacts(iso, rf_pipeline, scaler, explainer, metrics, FEATURE_COLS)
    plot_feature_importance(rf_pipeline, FEATURE_COLS)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
