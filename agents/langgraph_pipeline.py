"""
agents/langgraph_pipeline.py
=============================
Builds and compiles the LangGraph state-machine pipeline.

Three execution paths:
  Path 1 — Standard   : auth → monitor → behavior → policy → explain
  Path 2 — Mitigation : ... → policy → mitigation → explain
  Path 3 — Critical   : ... → policy → human_review → mitigation → explain

Usage:
    from agents.langgraph_pipeline import build_pipeline, run_pipeline
    graph = build_pipeline()
    result = run_pipeline(graph, event_dict)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from utils.state_schema import (
    AgentState, PolicyDecision, HITLStatus, MitigationAction
)
from agents.pipeline_agents import (
    authentication_agent,
    monitoring_agent,
    behavior_analysis_agent,
    policy_enforcement_agent,
    human_review_agent,
    mitigation_agent,
    explainability_agent,
)

HIGH_PRIVILEGE_ROLES = {"Doctor", "Admin", "ChiefMedicalOfficer"}


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_policy(state: AgentState) -> Literal["human_review", "mitigation", "explainability"]:
    """
    Conditional edge: determines which path to take after Policy Enforcement.

    Path 3 (Critical): BLOCK + high-privilege role → Human Review
    Path 2 (Mitigation): RESTRICT or BLOCK (low-priv) → Mitigation
    Path 1 (Standard): ALLOW / ALLOW_WITH_ALERT → Explainability
    """
    decision = state["policy_decision"]
    role     = state["event"]["role"]

    if decision == PolicyDecision.BLOCK and role in HIGH_PRIVILEGE_ROLES:
        logger.info(f"[Router] Path 3 → human_review (BLOCK + high-privilege: {role})")
        return "human_review"

    elif decision in (PolicyDecision.RESTRICT, PolicyDecision.BLOCK):
        logger.info(f"[Router] Path 2 → mitigation ({decision.value})")
        return "mitigation"

    elif decision == PolicyDecision.ALLOW_WITH_ALERT:
        logger.info("[Router] Path 2 → mitigation (ALLOW_WITH_ALERT — send alert)")
        return "mitigation"

    else:
        logger.info("[Router] Path 1 → explainability (ALLOW)")
        return "explainability"


def route_after_human_review(state: AgentState) -> Literal["mitigation", "explainability"]:
    """
    After HITL: always proceed to mitigation
    (mitigation will interpret APPROVED vs REJECTED).
    """
    return "mitigation"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Constructs and compiles the LangGraph state machine.
    Returns a compiled graph ready for .invoke() calls.
    """
    graph = StateGraph(AgentState)

    # ── Register Nodes ────────────────────────────────────────────────────────
    graph.add_node("authentication",    authentication_agent)
    graph.add_node("monitoring",        monitoring_agent)
    graph.add_node("behavior_analysis", behavior_analysis_agent)
    graph.add_node("policy_enforcement", policy_enforcement_agent)
    graph.add_node("human_review",      human_review_agent)
    graph.add_node("mitigation",        mitigation_agent)
    graph.add_node("explainability",    explainability_agent)

    # ── Entry Point ───────────────────────────────────────────────────────────
    graph.set_entry_point("authentication")

    # ── Sequential edges (always executed) ───────────────────────────────────
    graph.add_edge("authentication",    "monitoring")
    graph.add_edge("monitoring",        "behavior_analysis")
    graph.add_edge("behavior_analysis", "policy_enforcement")

    # ── Conditional routing after Policy ──────────────────────────────────────
    graph.add_conditional_edges(
        "policy_enforcement",
        route_after_policy,
        {
            "human_review":  "human_review",
            "mitigation":    "mitigation",
            "explainability": "explainability",
        },
    )

    # ── After Human Review → always Mitigation ────────────────────────────────
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"mitigation": "mitigation"},
    )

    # ── Terminal edges → END ──────────────────────────────────────────────────
    graph.add_edge("mitigation",     "explainability")
    graph.add_edge("explainability", END)

    return graph.compile()


# ── Convenience Runner ────────────────────────────────────────────────────────

def _default_state(event: Dict[str, Any]) -> AgentState:
    """Initialize AgentState with sensible defaults for all optional fields."""
    return AgentState(
        event=event,
        # Auth
        auth_valid=False,
        auth_flags=[],
        session_id="",
        auth_timestamp="",
        # Monitoring
        monitoring_flags=[],
        monitoring_severity="none",
        rule_violations=[],
        # Behavior
        isolation_forest_score=0.0,
        random_forest_prob=0.0,
        ensemble_risk_score=0.0,
        risk_level="LOW",
        shap_values=[],
        top_risk_factors=[],
        # Policy
        policy_decision=PolicyDecision.ALLOW,
        policy_rationale="",
        rbac_restrictions=[],
        # HITL
        requires_hitl=False,
        hitl_status=HITLStatus.NOT_REQUIRED,
        hitl_reviewer_id=None,
        hitl_decision_ts=None,
        hitl_notes=None,
        # Mitigation
        mitigation_actions=[],
        mitigation_ts="",
        # Explainability
        audit_narrative="",
        incident_id=None,
        report_path=None,
        # Pipeline metadata
        pipeline_start_ts=datetime.utcnow().isoformat(),
        pipeline_end_ts=None,
        processing_path=[],
        error=None,
    )


def run_pipeline(graph, event: Dict[str, Any]) -> AgentState:
    """
    Run a single EHR access event through the compiled pipeline.

    Args:
        graph: Compiled LangGraph pipeline
        event: Dict with EHR access event fields

    Returns:
        Final AgentState after all agents have processed the event
    """
    initial_state = _default_state(event)
    logger.info(f"\n{'='*60}")
    logger.info(f"▶ Pipeline start | Event: {event.get('event_id', 'N/A')}")
    logger.info(f"  User: {event.get('username')} ({event.get('role')}) "
                f"| Records: {event.get('records_accessed')}")

    result = graph.invoke(initial_state)

    logger.info(f"◀ Pipeline complete | Path: {' → '.join(result['processing_path'])}")
    logger.info(f"  Risk: {result['ensemble_risk_score']:.3f} ({result['risk_level']}) "
                f"| Decision: {result['policy_decision']}")
    logger.info(f"{'='*60}\n")

    return result
