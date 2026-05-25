"""
agents/pipeline_agents.py
==========================
All 7 specialized LangGraph agents for the healthcare security pipeline:

  1. AuthenticationAgent   — identity, role, session validation
  2. MonitoringAgent       — rule-based heuristic flags
  3. BehaviorAnalysisAgent — dual ML scoring + SHAP
  4. PolicyEnforcementAgent — Dynamic RBAC decisions
  5. HumanReviewAgent      — HITL pause / approval flow
  6. MitigationAgent       — automated response actions
  7. ExplainabilityAgent   — HIPAA-compliant narrative via Claude API
"""

from __future__ import annotations

import os
import json
import uuid
import joblib
import numpy as np
import shap
from datetime import datetime
from typing import Any, Dict, List
from loguru import logger
from groq import Groq

from utils.state_schema import (
    AgentState, RiskLevel, PolicyDecision,
    HITLStatus, MitigationAction
)

# ── Shared model registry (loaded once at import time) ───────────────────────
_MODELS: Dict[str, Any] = {}

def _load_models():
    global _MODELS
    if _MODELS:
        return _MODELS
    try:
        _MODELS["iso"]       = joblib.load("models/saved/isolation_forest.pkl")
        _MODELS["rf"]        = joblib.load("models/saved/random_forest_pipeline.pkl")
        _MODELS["scaler"]    = joblib.load("models/saved/feature_scaler.pkl")
        _MODELS["explainer"] = joblib.load("models/saved/shap_explainer.pkl")
        with open("models/saved/model_metadata.json") as f:
            _MODELS["meta"] = json.load(f)
        logger.info("✅ ML models loaded successfully")
    except FileNotFoundError:
        logger.warning("⚠️  Models not found. Run models/train_models.py first.")
        _MODELS["meta"] = {
            "feature_cols": [],
            "if_weight": 0.40,
            "rf_weight": 0.60,
            "risk_thresholds": {"allow": 0.30, "restrict": 0.55, "block": 0.75},
        }
    return _MODELS


FEATURE_COLS = [
    "records_accessed", "session_duration_min", "external_emails_sent",
    "off_hours_access", "bulk_download_flag", "unauthorized_access",
    "usb_connected", "failed_auth_attempts", "vpn_usage",
    "is_weekend", "after_midnight",
    "avg_records_per_session", "max_records_session", "total_events",
    "off_hours_ratio", "bulk_download_ratio", "external_email_ratio",
    "avg_session_duration", "usb_usage_count", "failed_auth_ratio",
    "unique_access_types", "vpn_ratio", "weekend_access_ratio",
    "unauthorized_ratio", "external_email_count",
    "records_over_limit", "session_anomaly_score",
    "privilege_risk", "device_risk", "location_risk", "access_type_risk",
]

HIGH_PRIVILEGE_ROLES = {"Doctor", "Admin", "ChiefMedicalOfficer"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AUTHENTICATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def authentication_agent(state: AgentState) -> AgentState:
    """
    Validates user identity, role, and session context.
    Flags suspicious authentication signals immediately.
    """
    logger.info(f"[AuthAgent] Processing event {state['event']['event_id']}")
    event = state["event"]
    flags = []
    state["processing_path"] = state.get("processing_path", []) + ["authentication_agent"]

    # Off-hours flag
    if event.get("off_hours_access"):
        flags.append("off_hours_access")

    # Unknown/remote location
    if event.get("location") in ("unknown", "remote"):
        flags.append(f"suspicious_location:{event['location']}")

    # Failed authentication attempts
    if event.get("failed_auth_attempts", 0) >= 3:
        flags.append(f"multiple_auth_failures:{event['failed_auth_attempts']}")

    # After midnight
    if event.get("after_midnight"):
        flags.append("after_midnight_access")

    # Weekend access for non-emergency roles
    if event.get("is_weekend") and event["role"] not in ("Doctor", "Nurse"):
        flags.append("weekend_access_non_clinical")

    auth_valid = event.get("failed_auth_attempts", 0) < 5  # block after 5 failures

    state.update({
        "auth_valid":      auth_valid,
        "auth_flags":      flags,
        "session_id":      f"SESS-{uuid.uuid4().hex[:8].upper()}",
        "auth_timestamp":  datetime.utcnow().isoformat(),
    })

    severity = "high" if len(flags) >= 3 else "medium" if len(flags) >= 1 else "low"
    logger.debug(f"[AuthAgent] auth_valid={auth_valid} | flags={flags} | severity={severity}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MONITORING AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def monitoring_agent(state: AgentState) -> AgentState:
    """
    Rule-based heuristic monitoring.
    Catches clear policy violations before ML scoring.
    """
    logger.info(f"[MonitoringAgent] Processing {state['event']['event_id']}")
    event = state["event"]
    flags = []
    violations = []
    state["processing_path"] = state.get("processing_path", []) + ["monitoring_agent"]

    # Rule 1: Records exceeded limit
    limit   = event.get("record_limit", 20)
    accessed = event.get("records_accessed", 0)
    if accessed > limit:
        excess = accessed - limit
        flags.append(f"records_exceeded_limit:{accessed}>{limit}")
        violations.append({
            "rule":     "RECORD_LIMIT_EXCEEDED",
            "detail":   f"Accessed {accessed} records; limit is {limit} (excess: {excess})",
            "severity": "high" if excess > limit else "medium",
        })

    # Rule 2: Unauthorized bulk download
    if event.get("bulk_download_flag") and event.get("access_type") in ("download", "export"):
        flags.append("unauthorized_bulk_download")
        violations.append({
            "rule":     "BULK_DOWNLOAD_DETECTED",
            "detail":   f"Bulk {event['access_type']} of {accessed} records",
            "severity": "high",
        })

    # Rule 3: USB device in restricted role
    if event.get("usb_connected") and event["privilege_level"] in ("low", "medium"):
        flags.append("usb_connected_restricted_role")
        violations.append({
            "rule":     "USB_DEVICE_RESTRICTED",
            "detail":   f"USB connected by {event['role']} (privilege: {event['privilege_level']})",
            "severity": "medium",
        })

    # Rule 4: External email ratio spike
    if event.get("external_email_ratio", 0) > 0.4:
        flags.append("high_external_email_ratio")
        violations.append({
            "rule":     "EXTERNAL_EMAIL_SPIKE",
            "detail":   f"External email ratio: {event['external_email_ratio']:.1%}",
            "severity": "medium",
        })

    # Rule 5: Delete or export on PHI
    if event.get("access_type") in ("delete", "export") and accessed > 5:
        flags.append(f"high_risk_access_type:{event['access_type']}")
        violations.append({
            "rule":     "SENSITIVE_ACCESS_TYPE",
            "detail":   f"Access type '{event['access_type']}' on {accessed} records",
            "severity": "high" if event["access_type"] == "delete" else "medium",
        })

    severity = "none"
    if violations:
        sevs = [v["severity"] for v in violations]
        severity = "high" if "high" in sevs else "medium"

    state.update({
        "monitoring_flags":    flags,
        "monitoring_severity": severity,
        "rule_violations":     violations,
    })

    logger.debug(f"[MonitoringAgent] {len(violations)} violations | severity={severity}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BEHAVIOR ANALYSIS AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def behavior_analysis_agent(state: AgentState) -> AgentState:
    """
    Dual-model ML scoring:
      - Isolation Forest (40%) for novel anomaly detection
      - Random Forest (60%) for known pattern recognition
    SHAP values for top risk factor attribution.
    """
    logger.info(f"[BehaviorAgent] Scoring event {state['event']['event_id']}")
    models = _load_models()
    event  = state["event"]
    state["processing_path"] = state.get("processing_path", []) + ["behavior_analysis_agent"]

    # Build feature vector
    feature_vector = np.array([[event.get(col, 0) for col in FEATURE_COLS]], dtype=float)

    if_score  = 0.0
    rf_prob   = 0.0
    shap_vals = [0.0] * len(FEATURE_COLS)

    try:
        scaler = models["scaler"]
        X_scaled = scaler.transform(feature_vector)

        # Isolation Forest — normalize decision_function to [0,1]
        iso = models["iso"]
        raw_if = iso.decision_function(X_scaled)[0]
        if_score = float(np.clip(1 - (raw_if + 0.5), 0, 1))

        # Random Forest — malicious class probability
        rf = models["rf"]
        rf_prob = float(rf.predict_proba(feature_vector)[0][1])

        # SHAP feature attribution
        explainer = models["explainer"]
        X_rf_scaled = rf.named_steps["scaler"].transform(feature_vector)
        shap_values = explainer.shap_values(X_rf_scaled)
        # Flatten correctly regardless of shape
        if isinstance(shap_values, list):
            raw = shap_values[1][0]
        else:
            raw = shap_values[0]

        # Handle nested lists/arrays
        if hasattr(raw, 'tolist'):
           raw = raw.tolist()
        if isinstance(raw[0], (list, )):
           raw = raw[0]

        shap_vals = [float(v) for v in raw]

    except Exception as e:
        logger.warning(f"[BehaviorAgent] Model inference error: {e} — using heuristic fallback")
        # Heuristic fallback when models not loaded
        signals = [
            event.get("off_hours_access", 0) * 0.3,
            event.get("unauthorized_access", 0) * 0.4,
            event.get("bulk_download_flag", 0) * 0.3,
            event.get("usb_connected", 0) * 0.2,
            min(event.get("failed_auth_attempts", 0) / 10, 0.3),
            event.get("access_type_risk", 0) * 0.2,
            event.get("location_risk", 0) * 0.15,
        ]
        rf_prob = min(sum(signals), 1.0)
        if_score = min(rf_prob + np.random.normal(0, 0.05), 1.0)
        if_score = max(0.0, if_score)

    # Weighted ensemble
    if_w = models["meta"]["if_weight"]
    rf_w = models["meta"]["rf_weight"]
    ensemble = float(np.clip(if_w * if_score + rf_w * rf_prob, 0.0, 1.0))

    # Risk level mapping
    thresholds = models["meta"]["risk_thresholds"]
    if ensemble < thresholds["allow"]:
        risk_level = RiskLevel.LOW
    elif ensemble < thresholds["restrict"]:
        risk_level = RiskLevel.MEDIUM
    elif ensemble < thresholds["block"]:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.CRITICAL

    # Top 5 risk factors by SHAP magnitude
    shap_pairs = sorted(
        zip(FEATURE_COLS, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]
    top_risk_factors = [
        {
            "feature":           col,
            "value":             float(event.get(col, 0)),
            "shap_contribution": round(float(val), 4),
            "direction":         "increases_risk" if val > 0 else "decreases_risk",
        }
        for col, val in shap_pairs
    ]

    state.update({
        "isolation_forest_score": round(if_score, 4),
        "random_forest_prob":     round(rf_prob, 4),
        "ensemble_risk_score":    round(ensemble, 4),
        "risk_level":             risk_level,
        "shap_values":            [round(v, 4) for v in shap_vals],
        "top_risk_factors":       top_risk_factors,
    })

    logger.info(f"[BehaviorAgent] Score={ensemble:.3f} | Level={risk_level.value}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POLICY ENFORCEMENT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def policy_enforcement_agent(state: AgentState) -> AgentState:
    """
    Dynamic RBAC policy enforcement.
    Maps risk score → access decision and RBAC restrictions.
    """
    logger.info(f"[PolicyAgent] Enforcing policy for {state['event']['event_id']}")
    models    = _load_models()
    thresholds = models["meta"]["risk_thresholds"]
    score     = state["ensemble_risk_score"]
    risk      = state["risk_level"]
    state["processing_path"] = state.get("processing_path", []) + ["policy_enforcement_agent"]

    restrictions = []
    rationale    = []

    if score < thresholds["allow"]:
        decision = PolicyDecision.ALLOW
        rationale.append(f"Risk score {score:.3f} is below allow threshold {thresholds['allow']}.")

    elif score < thresholds["restrict"]:
        decision = PolicyDecision.ALLOW_WITH_ALERT
        rationale.append(f"Risk score {score:.3f} triggers alert. Access permitted with monitoring.")
        restrictions.append("enhanced_monitoring_enabled")

    elif score < thresholds["block"]:
        decision = PolicyDecision.RESTRICT
        rationale.append(f"Risk score {score:.3f} activates view-only restriction.")
        restrictions.extend(["write_access_disabled", "download_disabled", "print_disabled"])

    else:
        decision = PolicyDecision.BLOCK
        rationale.append(f"Risk score {score:.3f} exceeds block threshold {thresholds['block']}.")
        restrictions.extend(["full_block", "session_terminated", "account_flagged"])

    # Add context from monitoring
    if state.get("rule_violations"):
        rationale.append(
            f"{len(state['rule_violations'])} rule violation(s) detected: "
            + "; ".join(v["rule"] for v in state["rule_violations"])
        )

    # High-privilege roles get extra context
    if state["event"]["role"] in HIGH_PRIVILEGE_ROLES:
        rationale.append(
            f"User has high-privilege role ({state['event']['role']}). "
            "HITL review required for BLOCK decisions to prevent clinical disruption."
        )

    state.update({
        "policy_decision":  decision,
        "policy_rationale": " | ".join(rationale),
        "rbac_restrictions": restrictions,
    })

    logger.info(f"[PolicyAgent] Decision={decision.value} | Restrictions={restrictions}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HUMAN REVIEW AGENT (HITL)
# ═══════════════════════════════════════════════════════════════════════════════

def human_review_agent(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop agent for high-privilege BLOCK decisions.
    In production: publishes to security dashboard queue and awaits webhook.
    In this implementation: simulates review with structured output.
    """
    logger.warning(f"[HITLAgent] ⏸  PAUSING pipeline for human review — {state['event']['event_id']}")
    event  = state["event"]
    state["processing_path"] = state.get("processing_path", []) + ["human_review_agent"]

    requires_hitl = (
        state["policy_decision"] == PolicyDecision.BLOCK
        and event["role"] in HIGH_PRIVILEGE_ROLES
    )

    if not requires_hitl:
        state.update({
            "requires_hitl": False,
            "hitl_status":   HITLStatus.NOT_REQUIRED,
        })
        return state

    # Prepare HITL alert payload (in production: send to dashboard via WebSocket)
    alert_payload = {
        "hitl_request_id": f"HITL-{uuid.uuid4().hex[:8].upper()}",
        "event_id":        event["event_id"],
        "user_id":         event["user_id"],
        "username":        event["username"],
        "role":            event["role"],
        "risk_score":      state["ensemble_risk_score"],
        "risk_level":      state["risk_level"],
        "top_risk_factors": state["top_risk_factors"],
        "rule_violations": state["rule_violations"],
        "policy_decision": state["policy_decision"],
        "timestamp":       datetime.utcnow().isoformat(),
        "timeout_seconds": int(os.getenv("HITL_TIMEOUT_SECONDS", "300")),
    }

    logger.info(f"[HITLAgent] Alert dispatched: {json.dumps(alert_payload, indent=2)}")

    # ── Simulated review (replace with actual dashboard webhook in production) ──
    # In production, this would block until a human approves/rejects via API.
    # For demo: auto-approve if risk < 0.90 and it's during business hours.
    import datetime as dt
    now = dt.datetime.utcnow()
    is_business_hours = 8 <= now.hour <= 18
    auto_decision = (
        HITLStatus.APPROVED
        if state["ensemble_risk_score"] < 0.90 and is_business_hours
        else HITLStatus.REJECTED
    )

    state.update({
        "requires_hitl":    True,
        "hitl_status":      auto_decision,
        "hitl_reviewer_id": "SECURITY_OFFICER_001",
        "hitl_decision_ts": datetime.utcnow().isoformat(),
        "hitl_notes": (
            "Auto-reviewed during simulation. In production, a human security "
            "officer reviews context and approves or rejects via dashboard."
        ),
    })

    logger.info(f"[HITLAgent] HITL decision: {auto_decision.value}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MITIGATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def mitigation_agent(state: AgentState) -> AgentState:
    """
    Executes automated mitigation actions based on policy decision
    and HITL outcome.
    """
    logger.info(f"[MitigationAgent] Executing mitigation for {state['event']['event_id']}")
    decision    = state["policy_decision"]
    hitl_status = state.get("hitl_status", HITLStatus.NOT_REQUIRED)
    state["processing_path"] = state.get("processing_path", []) + ["mitigation_agent"]

    actions = []

    # Honour HITL approval — downgrade from BLOCK to RESTRICT
    if hitl_status == HITLStatus.APPROVED and decision == PolicyDecision.BLOCK:
        logger.info("[MitigationAgent] HITL approved — downgrading BLOCK to RESTRICT")
        decision = PolicyDecision.RESTRICT
        state["policy_decision"] = decision

    if decision == PolicyDecision.ALLOW:
        actions.append(MitigationAction.NONE)

    elif decision == PolicyDecision.ALLOW_WITH_ALERT:
        actions.append(MitigationAction.ALERT_SENT)
        logger.info("[MitigationAgent] 📧 Security alert dispatched")

    elif decision == PolicyDecision.RESTRICT:
        actions.extend([MitigationAction.VIEW_ONLY_MODE, MitigationAction.DOWNLOADS_DISABLED])
        logger.info("[MitigationAgent] 🔒 Account restricted to view-only mode")

    elif decision == PolicyDecision.BLOCK:
        actions.extend([
            MitigationAction.SESSION_REVOKED,
            MitigationAction.ACCOUNT_LOCKED,
            MitigationAction.ALERT_SENT,
        ])
        logger.warning(f"[MitigationAgent] 🚨 Session REVOKED and account LOCKED "
                       f"for {state['event']['user_id']}")

    state.update({
        "mitigation_actions": actions,
        "mitigation_ts":      datetime.utcnow().isoformat(),
    })

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXPLAINABILITY AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def explainability_agent(state: AgentState) -> AgentState:
    """
    Generates a HIPAA-compliant audit narrative using Claude (Anthropic API).
    Falls back to a structured template if API is unavailable.
    """
    logger.info(f"[ExplainAgent] Generating narrative for {state['event']['event_id']}")
    state["processing_path"] = state.get("processing_path", []) + ["explainability_agent"]

    event     = state["event"]
    incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    # Build context summary for LLM
    context = {
        "event_id":         event["event_id"],
        "timestamp":        event["timestamp"],
        "user":             f"{event['username']} ({event['role']}, {event['department']})",
        "action":           f"{event['access_type'].upper()} {event['records_accessed']} records",
        "device":           event["device_type"],
        "location":         event["location"],
        "ensemble_score":   state["ensemble_risk_score"],
        "risk_level":       state["risk_level"],
        "if_score":         state["isolation_forest_score"],
        "rf_prob":          state["random_forest_prob"],
        "policy_decision":  state["policy_decision"],
        "top_factors":      state["top_risk_factors"][:3],
        "rule_violations":  state.get("rule_violations", []),
        "auth_flags":       state.get("auth_flags", []),
        "hitl_status":      state.get("hitl_status", HITLStatus.NOT_REQUIRED),
        "mitigation":       [a.value for a in state.get("mitigation_actions", [])],
        "incident_id":      incident_id,
    }

    narrative = _generate_narrative_with_claude(context)

    # Persist audit log (append to JSONL)
    audit_entry = {
        "incident_id":    incident_id,
        "event_id":       event["event_id"],
        "user_id":        event["user_id"],
        "timestamp":      event["timestamp"],
        "risk_score":     state["ensemble_risk_score"],
        "risk_level":     state["risk_level"],
        "decision":       state["policy_decision"],
        "mitigation":     [a.value for a in state.get("mitigation_actions", [])],
        "hitl_status":    state.get("hitl_status", HITLStatus.NOT_REQUIRED),
        "narrative":      narrative,
        "generated_ts":   datetime.utcnow().isoformat(),
    }

    os.makedirs("logs", exist_ok=True)
    audit_path = os.getenv("AUDIT_LOG_PATH", "logs/hipaa_audit.jsonl")
    with open(audit_path, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    state.update({
        "audit_narrative": narrative,
        "incident_id":     incident_id,
        "pipeline_end_ts": datetime.utcnow().isoformat(),
    })

    logger.info(f"[ExplainAgent] Incident {incident_id} logged.")
    return state


def _generate_narrative_with_claude(ctx: dict) -> str:
    """Use Groq API to generate HIPAA-compliant narrative. Falls back to template."""
    api_key = os.getenv("GROQ_API_KEY", "")
    model   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key or api_key == "your_groq_key_here":
        return _template_narrative(ctx)

    try:
        client = Groq(api_key=api_key)

        prompt = f"""You are a HIPAA-compliant healthcare security analyst generating an incident narrative.
Write a professional, factual, third-person incident report in 3-4 sentences.
Include: what happened, the risk assessment, and the action taken.
Do NOT include protected health information (PHI). Be precise and regulatory-compliant.

Context:
{json.dumps(ctx, indent=2, default=str)}

Generate the incident narrative now:"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"[ExplainAgent] Groq API error: {e} — using template fallback")
        return _template_narrative(ctx)


def _template_narrative(ctx: dict) -> str:
    """Deterministic fallback narrative when LLM is unavailable."""
    factors_str = ", ".join(
        f"{f['feature']} (SHAP: {f['shap_contribution']:+.3f})"
        for f in ctx.get("top_factors", [])
    )
    violations_str = (
        "; ".join(v["rule"] for v in ctx.get("rule_violations", []))
        or "None"
    )

    return (
        f"INCIDENT {ctx['incident_id']}: On {ctx['timestamp']}, user {ctx['user']} "
        f"performed a {ctx['action']} from a {ctx['device']} at location '{ctx['location']}'. "
        f"The dual-model ensemble risk engine assigned a score of "
        f"{ctx['ensemble_score']:.3f} (Isolation Forest: {ctx['if_score']:.3f}, "
        f"Random Forest: {ctx['rf_prob']:.3f}), classifying this event as "
        f"{ctx['risk_level']} risk. "
        f"Key contributing behavioral factors via SHAP analysis: {factors_str}. "
        f"Rule violations detected: {violations_str}. "
        f"HITL status: {ctx['hitl_status']}. "
        f"Policy decision: {ctx['policy_decision']}. "
        f"Automated mitigation actions executed: {', '.join(ctx['mitigation'])}."
    )
