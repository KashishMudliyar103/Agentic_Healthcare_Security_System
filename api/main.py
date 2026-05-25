"""
api/main.py
===========
FastAPI backend providing:
  - REST endpoints for single-event processing and batch replay
  - WebSocket endpoint for real-time EHR event streaming
  - HITL approval/rejection endpoint
  - Audit log retrieval
  - Health check
"""

from __future__ import annotations

import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Lazy pipeline import (models loaded on first use) ─────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from agents.langgraph_pipeline import build_pipeline
        _pipeline = build_pipeline()
        logger.info("✅ LangGraph pipeline compiled and ready")
    return _pipeline


# ── HITL approval queue (in production: use Redis or DB) ─────────────────────
_hitl_queue: Dict[str, dict] = {}

# ── Active WebSocket clients ──────────────────────────────────────────────────
_ws_clients: List[WebSocket] = []


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏥 Healthcare Security API starting up...")
    get_pipeline()   # pre-warm pipeline
    yield
    logger.info("🏥 Healthcare Security API shutting down...")


app = FastAPI(
    title="Healthcare EHR Security API",
    description="Agentic multi-agent system for real-time insider threat detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class EHREventRequest(BaseModel):
    event_id:             str = "EVT000001"
    timestamp:            str = datetime.utcnow().isoformat()
    user_id:              str = "USR0001"
    username:             str = "jdoe"
    role:                 str = "Doctor"
    department:           str = "Cardiology"
    privilege_level:      str = "high"
    record_limit:         int = 20
    records_accessed:     int = 5
    access_type:          str = "view"
    device_type:          str = "workstation"
    location:             str = "office"
    session_duration_min: int = 30
    external_emails_sent: int = 0
    off_hours_access:     int = 0
    bulk_download_flag:   int = 0
    unauthorized_access:  int = 0
    usb_connected:        int = 0
    failed_auth_attempts: int = 0
    vpn_usage:            int = 0
    is_weekend:           int = 0
    after_midnight:       int = 0
    # Engineered features (can be 0 if not pre-computed)
    avg_records_per_session: float = 5.0
    max_records_session:     float = 10.0
    total_events:            int   = 50
    off_hours_ratio:         float = 0.05
    bulk_download_ratio:     float = 0.02
    external_email_ratio:    float = 0.01
    avg_session_duration:    float = 30.0
    usb_usage_count:         int   = 0
    failed_auth_ratio:       float = 0.01
    unique_access_types:     int   = 2
    vpn_ratio:               float = 0.05
    weekend_access_ratio:    float = 0.05
    unauthorized_ratio:      float = 0.02
    external_email_count:    int   = 0
    records_over_limit:      float = 0.25
    session_anomaly_score:   int   = 0
    privilege_risk:          float = 1.0
    device_risk:             float = 0.1
    location_risk:           float = 0.1
    access_type_risk:        float = 0.1


class HITLDecisionRequest(BaseModel):
    incident_id:  str
    reviewer_id:  str
    decision:     str    # "APPROVED" or "REJECTED"
    notes:        Optional[str] = None


class PipelineResult(BaseModel):
    event_id:           str
    incident_id:        Optional[str]
    risk_score:         float
    risk_level:         str
    policy_decision:    str
    mitigation_actions: List[str]
    hitl_status:        str
    processing_path:    List[str]
    audit_narrative:    str
    processing_time_ms: float


# ── Helper ────────────────────────────────────────────────────────────────────

def _state_to_result(state: dict, elapsed_ms: float) -> PipelineResult:
    return PipelineResult(
        event_id=state["event"]["event_id"],
        incident_id=state.get("incident_id"),
        risk_score=state.get("ensemble_risk_score", 0.0),
        risk_level=str(state.get("risk_level", "LOW")),
        policy_decision=str(state.get("policy_decision", "ALLOW")),
        mitigation_actions=[str(a) for a in state.get("mitigation_actions", [])],
        hitl_status=str(state.get("hitl_status", "NOT_REQUIRED")),
        processing_path=state.get("processing_path", []),
        audit_narrative=state.get("audit_narrative", ""),
        processing_time_ms=round(elapsed_ms, 2),
    )


async def _broadcast_ws(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_ready": _pipeline is not None,
        "connected_clients": len(_ws_clients),
    }


@app.post("/api/v1/process-event", response_model=PipelineResult)
async def process_event(request: EHREventRequest, background_tasks: BackgroundTasks):
    """
    Process a single EHR access event through the full agent pipeline.
    Broadcasts the result to all connected WebSocket clients.
    """
    start = datetime.utcnow()
    event_dict = request.model_dump()

    try:
        from agents.langgraph_pipeline import run_pipeline
        result_state = run_pipeline(get_pipeline(), event_dict)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (datetime.utcnow() - start).total_seconds() * 1000
    result  = _state_to_result(result_state, elapsed)

    # Broadcast to WebSocket clients in background
    background_tasks.add_task(
        _broadcast_ws,
        {
            "type":   "event_processed",
            "data":   result.model_dump(),
            "ts":     datetime.utcnow().isoformat(),
        }
    )

    return result


@app.post("/api/v1/batch-replay")
async def batch_replay(background_tasks: BackgroundTasks, limit: int = 50):
    """
    Replays events from the saved EHR access log CSV.
    Streams results via WebSocket as they are processed.
    """
    data_path = "data/ehr_access_log.csv"
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Run: python data/generate_dataset.py"
        )

    df = pd.read_csv(data_path).head(limit)

    async def _replay_task():
        from agents.langgraph_pipeline import run_pipeline
        pipe = get_pipeline()
        results = []

        for _, row in df.iterrows():
            event = row.fillna(0).to_dict()
            start = datetime.utcnow()

            try:
                state = run_pipeline(pipe, event)
                elapsed = (datetime.utcnow() - start).total_seconds() * 1000
                result  = _state_to_result(state, elapsed)
                results.append(result.model_dump())

                await _broadcast_ws({
                    "type": "batch_event",
                    "data": result.model_dump(),
                    "ts":   datetime.utcnow().isoformat(),
                })
                await asyncio.sleep(0.05)  # throttle for UI visibility

            except Exception as e:
                logger.error(f"Batch replay error on {event.get('event_id')}: {e}")

        await _broadcast_ws({
            "type":   "batch_complete",
            "total":  len(results),
            "ts":     datetime.utcnow().isoformat(),
        })

    background_tasks.add_task(_replay_task)
    return {"message": f"Batch replay started for {limit} events", "status": "running"}


@app.post("/api/v1/hitl/decision")
async def submit_hitl_decision(request: HITLDecisionRequest):
    """
    Security officer submits approve/reject for a HITL-paused event.
    In production: this would wake up the waiting LangGraph node.
    """
    if request.decision not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=400, detail="Decision must be APPROVED or REJECTED")

    _hitl_queue[request.incident_id] = {
        "incident_id": request.incident_id,
        "reviewer_id": request.reviewer_id,
        "decision":    request.decision,
        "notes":       request.notes,
        "decided_at":  datetime.utcnow().isoformat(),
    }

    await _broadcast_ws({
        "type": "hitl_decision",
        "data": _hitl_queue[request.incident_id],
    })

    logger.info(f"[HITL] {request.decision} by {request.reviewer_id} for {request.incident_id}")
    return {"message": f"HITL decision recorded: {request.decision}", "incident_id": request.incident_id}


@app.get("/api/v1/hitl/pending")
async def get_pending_hitl():
    """Returns all pending HITL decisions (for the security dashboard)."""
    audit_path = os.getenv("AUDIT_LOG_PATH", "logs/hipaa_audit.jsonl")
    pending = []

    if os.path.exists(audit_path):
        with open(audit_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("hitl_status") == "PENDING":
                        pending.append(entry)
                except Exception:
                    pass

    return {"pending": pending, "count": len(pending)}


@app.get("/api/v1/audit-log")
async def get_audit_log(limit: int = 100):
    """Returns the most recent HIPAA audit log entries."""
    audit_path = os.getenv("AUDIT_LOG_PATH", "logs/hipaa_audit.jsonl")
    entries = []

    if os.path.exists(audit_path):
        with open(audit_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass

    return {
        "entries": entries[-limit:],
        "total":   len(entries),
        "limit":   limit,
    }


@app.get("/api/v1/stats")
async def get_stats():
    """Returns aggregate stats for the dashboard."""
    audit_path = os.getenv("AUDIT_LOG_PATH", "logs/hipaa_audit.jsonl")
    entries = []

    if os.path.exists(audit_path):
        with open(audit_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass

    if not entries:
        return {"message": "No events processed yet"}

    decisions = {}
    risk_levels = {}
    for e in entries:
        d = e.get("decision", "UNKNOWN")
        r = e.get("risk_level", "UNKNOWN")
        decisions[d]   = decisions.get(d, 0) + 1
        risk_levels[r] = risk_levels.get(r, 0) + 1

    return {
        "total_events":  len(entries),
        "decisions":     decisions,
        "risk_levels":   risk_levels,
        "high_risk":     sum(1 for e in entries if e.get("risk_score", 0) >= 0.55),
        "blocked":       decisions.get("BLOCK", 0),
        "hitl_triggered": sum(1 for e in entries if e.get("hitl_status") != "NOT_REQUIRED"),
    }


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    Clients connect here to receive live pipeline results.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"[WS] Client connected. Total: {len(_ws_clients)}")

    try:
        await websocket.send_json({
            "type":    "connected",
            "message": "Connected to Healthcare Security live stream",
            "ts":      datetime.utcnow().isoformat(),
        })

        while True:
            # Keep connection alive; actual data pushed by broadcast
            data = await websocket.receive_text()

            # Allow client to submit events via WebSocket too
            if data:
                try:
                    event_dict = json.loads(data)
                    if event_dict.get("type") == "process_event":
                        from agents.langgraph_pipeline import run_pipeline
                        state = run_pipeline(get_pipeline(), event_dict["event"])
                        elapsed = 0
                        result  = _state_to_result(state, elapsed)
                        await websocket.send_json({
                            "type": "event_processed",
                            "data": result.model_dump(),
                        })
                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})

    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(_ws_clients)}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level="info",
    )
