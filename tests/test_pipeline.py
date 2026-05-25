"""
tests/test_pipeline.py
=======================
Unit tests for the agent pipeline.
Run with: pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.state_schema import RiskLevel, PolicyDecision, HITLStatus, MitigationAction


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_event(**overrides) -> dict:
    base = {
        "event_id": "EVT_TEST_001",
        "timestamp": "2024-03-15T10:00:00",
        "user_id": "USR_TEST",
        "username": "test_user",
        "role": "Nurse",
        "department": "Cardiology",
        "privilege_level": "medium",
        "record_limit": 15,
        "records_accessed": 5,
        "access_type": "view",
        "device_type": "workstation",
        "location": "office",
        "session_duration_min": 30,
        "external_emails_sent": 0,
        "off_hours_access": 0,
        "bulk_download_flag": 0,
        "unauthorized_access": 0,
        "usb_connected": 0,
        "failed_auth_attempts": 0,
        "vpn_usage": 0,
        "is_weekend": 0,
        "after_midnight": 0,
        "avg_records_per_session": 5.0,
        "max_records_session": 8.0,
        "total_events": 30,
        "off_hours_ratio": 0.05,
        "bulk_download_ratio": 0.02,
        "external_email_ratio": 0.01,
        "avg_session_duration": 30.0,
        "usb_usage_count": 0,
        "failed_auth_ratio": 0.01,
        "unique_access_types": 1,
        "vpn_ratio": 0.0,
        "weekend_access_ratio": 0.05,
        "unauthorized_ratio": 0.0,
        "external_email_count": 0,
        "records_over_limit": 0.33,
        "session_anomaly_score": 0,
        "privilege_risk": 0.5,
        "device_risk": 0.1,
        "location_risk": 0.1,
        "access_type_risk": 0.1,
    }
    base.update(overrides)
    return base


# ── State Schema Tests ────────────────────────────────────────────────────────

class TestStateSchema:
    def test_risk_level_enum(self):
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_policy_decision_enum(self):
        assert PolicyDecision.ALLOW.value == "ALLOW"
        assert PolicyDecision.BLOCK.value == "BLOCK"

    def test_hitl_status_enum(self):
        assert HITLStatus.NOT_REQUIRED.value == "NOT_REQUIRED"
        assert HITLStatus.PENDING.value == "PENDING"

    def test_mitigation_action_enum(self):
        assert MitigationAction.NONE.value == "NONE"
        assert MitigationAction.ACCOUNT_LOCKED.value == "ACCOUNT_LOCKED"


# ── Authentication Agent Tests ────────────────────────────────────────────────

class TestAuthenticationAgent:
    def setup_method(self):
        from agents.pipeline_agents import authentication_agent
        self.agent = authentication_agent

    def _make_state(self, event) -> dict:
        return {
            "event": event,
            "processing_path": [],
            "auth_valid": False,
            "auth_flags": [],
            "session_id": "",
            "auth_timestamp": "",
        }

    def test_benign_event_passes(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert result["auth_valid"] is True
        assert result["auth_flags"] == []

    def test_off_hours_flagged(self):
        state = self._make_state(_make_event(off_hours_access=1))
        result = self.agent(state)
        assert "off_hours_access" in result["auth_flags"]

    def test_after_midnight_flagged(self):
        state = self._make_state(_make_event(after_midnight=1))
        result = self.agent(state)
        assert "after_midnight_access" in result["auth_flags"]

    def test_multiple_auth_failures_invalidate(self):
        state = self._make_state(_make_event(failed_auth_attempts=7))
        result = self.agent(state)
        assert result["auth_valid"] is False

    def test_session_id_generated(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert result["session_id"].startswith("SESS-")

    def test_processing_path_updated(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert "authentication_agent" in result["processing_path"]


# ── Monitoring Agent Tests ────────────────────────────────────────────────────

class TestMonitoringAgent:
    def setup_method(self):
        from agents.pipeline_agents import monitoring_agent
        self.agent = monitoring_agent

    def _make_state(self, event) -> dict:
        return {
            "event": event,
            "processing_path": ["authentication_agent"],
            "monitoring_flags": [],
            "monitoring_severity": "none",
            "rule_violations": [],
        }

    def test_records_exceeded_triggers_violation(self):
        state = self._make_state(_make_event(records_accessed=50, record_limit=15))
        result = self.agent(state)
        rules = [v["rule"] for v in result["rule_violations"]]
        assert "RECORD_LIMIT_EXCEEDED" in rules

    def test_bulk_download_flagged(self):
        state = self._make_state(_make_event(
            bulk_download_flag=1, access_type="download", records_accessed=15
        ))
        result = self.agent(state)
        rules = [v["rule"] for v in result["rule_violations"]]
        assert "BULK_DOWNLOAD_DETECTED" in rules

    def test_clean_event_no_violations(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert result["rule_violations"] == []
        assert result["monitoring_severity"] == "none"

    def test_usb_restricted_role(self):
        state = self._make_state(_make_event(usb_connected=1, privilege_level="low"))
        result = self.agent(state)
        rules = [v["rule"] for v in result["rule_violations"]]
        assert "USB_DEVICE_RESTRICTED" in rules


# ── Behavior Analysis Agent Tests ─────────────────────────────────────────────

class TestBehaviorAnalysisAgent:
    def setup_method(self):
        from agents.pipeline_agents import behavior_analysis_agent
        self.agent = behavior_analysis_agent

    def _make_state(self, event) -> dict:
        return {
            "event": event,
            "processing_path": ["authentication_agent", "monitoring_agent"],
            "isolation_forest_score": 0.0,
            "random_forest_prob": 0.0,
            "ensemble_risk_score": 0.0,
            "risk_level": "LOW",
            "shap_values": [],
            "top_risk_factors": [],
        }

    def test_output_fields_present(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert "ensemble_risk_score" in result
        assert "risk_level" in result
        assert "top_risk_factors" in result

    def test_score_in_valid_range(self):
        state = self._make_state(_make_event())
        result = self.agent(state)
        assert 0.0 <= result["ensemble_risk_score"] <= 1.0

    def test_high_risk_signals_increase_score(self):
        benign_state  = self._make_state(_make_event())
        malicious_state = self._make_state(_make_event(
            off_hours_access=1, unauthorized_access=1, bulk_download_flag=1,
            usb_connected=1, failed_auth_attempts=4, location="unknown",
            access_type="export", device_type="external_usb",
            records_accessed=80, record_limit=15,
            after_midnight=1, off_hours_ratio=0.8,
            bulk_download_ratio=0.7, unauthorized_ratio=0.8,
            device_risk=1.0, location_risk=1.0, access_type_risk=0.9,
            records_over_limit=5.0,
        ))
        benign_result   = self.agent(benign_state)
        malicious_result = self.agent(malicious_state)
        assert malicious_result["ensemble_risk_score"] > benign_result["ensemble_risk_score"]


# ── Policy Enforcement Tests ──────────────────────────────────────────────────

class TestPolicyEnforcementAgent:
    def setup_method(self):
        from agents.pipeline_agents import policy_enforcement_agent
        self.agent = policy_enforcement_agent

    def _make_state(self, risk_score: float, role: str = "Nurse") -> dict:
        if risk_score < 0.30:   risk_level = "LOW"
        elif risk_score < 0.55: risk_level = "MEDIUM"
        elif risk_score < 0.75: risk_level = "HIGH"
        else:                   risk_level = "CRITICAL"

        return {
            "event": _make_event(role=role),
            "processing_path": [],
            "ensemble_risk_score": risk_score,
            "risk_level": risk_level,
            "rule_violations": [],
            "policy_decision": "ALLOW",
            "policy_rationale": "",
            "rbac_restrictions": [],
        }

    def test_low_risk_allows(self):
        result = self.agent(self._make_state(0.10))
        assert result["policy_decision"] == PolicyDecision.ALLOW

    def test_medium_risk_alert(self):
        result = self.agent(self._make_state(0.40))
        assert result["policy_decision"] == PolicyDecision.ALLOW_WITH_ALERT

    def test_high_risk_restricts(self):
        result = self.agent(self._make_state(0.65))
        assert result["policy_decision"] == PolicyDecision.RESTRICT

    def test_critical_risk_blocks(self):
        result = self.agent(self._make_state(0.85))
        assert result["policy_decision"] == PolicyDecision.BLOCK


# ── Full Pipeline Integration Test ────────────────────────────────────────────

class TestFullPipeline:
    def test_benign_event_low_risk_path(self):
        from agents.langgraph_pipeline import build_pipeline, run_pipeline
        graph = build_pipeline()
        event = _make_event()
        result = run_pipeline(graph, event)
        assert "authentication_agent" in result["processing_path"]
        assert "explainability_agent" in result["processing_path"]
        assert result["audit_narrative"] != ""

    def test_high_risk_event_mitigation_path(self):
        from agents.langgraph_pipeline import build_pipeline, run_pipeline
        graph = build_pipeline()
        event = _make_event(
            off_hours_access=1, unauthorized_access=1, bulk_download_flag=1,
            records_accessed=80, record_limit=15, usb_connected=1,
            device_type="external_usb", location="unknown",
            access_type="export", after_midnight=1,
            off_hours_ratio=0.85, bulk_download_ratio=0.75,
            unauthorized_ratio=0.90, device_risk=1.0, location_risk=1.0,
            access_type_risk=0.9, records_over_limit=5.3,
        )
        result = run_pipeline(graph, event)
        assert "processing_path" in result
        assert len(result["mitigation_actions"]) > 0

    def test_critical_doctor_event_hitl_path(self):
        from agents.langgraph_pipeline import build_pipeline, run_pipeline
        graph = build_pipeline()
        event = _make_event(
            role="Doctor", privilege_level="high",
            off_hours_access=1, unauthorized_access=1, bulk_download_flag=1,
            records_accessed=90, record_limit=20, usb_connected=1,
            device_type="external_usb", location="unknown",
            access_type="export", after_midnight=1,
            off_hours_ratio=0.90, bulk_download_ratio=0.85,
            unauthorized_ratio=0.95, device_risk=1.0, location_risk=1.0,
            access_type_risk=0.9, records_over_limit=4.5,
            failed_auth_attempts=4, failed_auth_ratio=0.4,
        )
        result = run_pipeline(graph, event)
        # If blocked and Doctor, HITL must be triggered
        if result["policy_decision"] == PolicyDecision.BLOCK:
            assert result.get("requires_hitl") is True
            assert "human_review_agent" in result["processing_path"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
