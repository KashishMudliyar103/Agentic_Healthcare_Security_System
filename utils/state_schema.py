"""
utils/state_schema.py
=====================
Defines the shared TypedDict state object that flows through every
agent in the LangGraph pipeline. Every agent reads from and writes
to this immutable-style state bag.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ── Enumerations ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyDecision(str, Enum):
    ALLOW            = "ALLOW"
    ALLOW_WITH_ALERT = "ALLOW_WITH_ALERT"
    RESTRICT         = "RESTRICT"
    BLOCK            = "BLOCK"


class HITLStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING      = "PENDING"
    APPROVED     = "APPROVED"
    REJECTED     = "REJECTED"
    TIMEOUT      = "TIMEOUT"


class MitigationAction(str, Enum):
    NONE             = "NONE"
    ALERT_SENT       = "ALERT_SENT"
    VIEW_ONLY_MODE   = "VIEW_ONLY_MODE"
    SESSION_REVOKED  = "SESSION_REVOKED"
    ACCOUNT_LOCKED   = "ACCOUNT_LOCKED"
    DOWNLOADS_DISABLED = "DOWNLOADS_DISABLED"


# ── Core State ────────────────────────────────────────────────────────────────

class EHRAccessEvent(TypedDict):
    """Raw input event from the EHR system."""
    event_id:             str
    timestamp:            str                  # ISO 8601
    user_id:              str
    username:             str
    role:                 str
    department:           str
    privilege_level:      str
    record_limit:         int
    records_accessed:     int
    access_type:          str
    device_type:          str
    location:             str
    session_duration_min: int
    external_emails_sent: int
    off_hours_access:     int
    bulk_download_flag:   int
    unauthorized_access:  int
    usb_connected:        int
    failed_auth_attempts: int
    vpn_usage:            int
    is_weekend:           int
    after_midnight:       int
    # Engineered features (may be pre-computed or computed on-the-fly)
    avg_records_per_session: float
    max_records_session:     float
    total_events:            int
    off_hours_ratio:         float
    bulk_download_ratio:     float
    external_email_ratio:    float
    avg_session_duration:    float
    usb_usage_count:         int
    failed_auth_ratio:       float
    unique_access_types:     int
    vpn_ratio:               float
    weekend_access_ratio:    float
    unauthorized_ratio:      float
    external_email_count:    int
    records_over_limit:      float
    session_anomaly_score:   int
    privilege_risk:          float
    device_risk:             float
    location_risk:           float
    access_type_risk:        float


class AgentState(TypedDict):
    """
    The unified state bag passed through every LangGraph node.
    Each agent adds to / updates specific fields.
    """
    # ── Input ──────────────────────────────────────────────────────────────
    event: EHRAccessEvent

    # ── Authentication Agent ───────────────────────────────────────────────
    auth_valid:           bool
    auth_flags:           List[str]           # e.g. ["off_hours_access", "unknown_location"]
    session_id:           str
    auth_timestamp:       str

    # ── Monitoring Agent ───────────────────────────────────────────────────
    monitoring_flags:     List[str]           # e.g. ["records_exceeded_limit", "bulk_download"]
    monitoring_severity:  str                 # "none" | "low" | "medium" | "high"
    rule_violations:      List[Dict[str, Any]]

    # ── Behavior Analysis Agent ────────────────────────────────────────────
    isolation_forest_score: float             # Raw IF anomaly score [0,1]
    random_forest_prob:     float             # RF malicious probability [0,1]
    ensemble_risk_score:    float             # Weighted ensemble [0,1]
    risk_level:             RiskLevel
    shap_values:            List[float]       # Per-feature SHAP contributions
    top_risk_factors:       List[Dict[str, Any]]  # [{feature, value, shap_contribution}]

    # ── Policy Enforcement Agent ───────────────────────────────────────────
    policy_decision:      PolicyDecision
    policy_rationale:     str
    rbac_restrictions:    List[str]

    # ── Human-in-the-Loop Agent ────────────────────────────────────────────
    requires_hitl:        bool
    hitl_status:          HITLStatus
    hitl_reviewer_id:     Optional[str]
    hitl_decision_ts:     Optional[str]
    hitl_notes:           Optional[str]

    # ── Mitigation Agent ───────────────────────────────────────────────────
    mitigation_actions:   List[MitigationAction]
    mitigation_ts:        str

    # ── Explainability Agent ───────────────────────────────────────────────
    audit_narrative:      str                 # HIPAA-compliant incident narrative
    incident_id:          Optional[str]
    report_path:          Optional[str]

    # ── Pipeline Metadata ─────────────────────────────────────────────────
    pipeline_start_ts:    str
    pipeline_end_ts:      Optional[str]
    processing_path:      List[str]           # Which nodes were visited
    error:                Optional[str]
