# 🏥 Agentic Healthcare EHR Security System

> **Real-time insider threat detection using a LangGraph multi-agent pipeline with dual-model ML, HITL, and HIPAA-compliant explainability.**

---

## 📐 Architecture Overview

```
EHR Access Event
       │
       ▼
┌─────────────────────┐
│  Authentication     │ ← Identity, role, session validation
│  Agent              │   Flags: off-hours, failed auth, location anomalies
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Monitoring Agent   │ ← Rule-based heuristics
│                     │   Flags: record limits, bulk downloads, USB, email
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Behavior Analysis  │ ← Dual-Model ML Engine
│  Agent              │   • Isolation Forest (40%) — novel anomalies
│                     │   • Random Forest (60%)    — known patterns
│                     │   + SHAP feature attribution
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Policy Enforcement │ ← Dynamic RBAC
│  Agent              │   ALLOW / ALLOW_WITH_ALERT / RESTRICT / BLOCK
└────────┬────────────┘
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
[BLOCK + High-Priv]    [RESTRICT/BLOCK]     [ALLOW/ALERT]
    │                       │                    │
    ▼                       │                    │
┌───────────┐               │                    │
│   Human   │               │                    │
│  Review   │               │                    │
│  Agent    │               │                    │
│  (HITL)   │               │                    │
└─────┬─────┘               │                    │
      │                     │                    │
      └──────────┬──────────┘                    │
                 ▼                               │
         ┌─────────────┐                         │
         │  Mitigation │                         │
         │  Agent      │                         │
         └──────┬──────┘                         │
                └──────────────┬─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Explainability     │
                    │  Agent              │ ← Claude API → HIPAA Narrative
                    └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Claude Code (already integrated)
- Anthropic API key

### 1. Clone & Install

```bash
cd healthcare_security
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Full Setup (Generate Data + Train Models)

```bash
python main.py setup
```

This runs:
- `data/generate_dataset.py` — 3,757 EHR events, 38 behavioral features
- `models/train_models.py` — Isolation Forest + Random Forest + SHAP explainer

### 4. Run the Pipeline Demo

```bash
python main.py demo 20
```

### 5. Test a Single Critical Event (HITL Path)

```bash
python main.py single
```

### 6. Run Tests

```bash
pytest tests/ -v
```

### 7. Launch API Server

```bash
python main.py api
# → http://localhost:8000/docs
```

### 8. Launch Dashboard

```bash
python main.py dashboard
# → http://localhost:8501
```

### 9. Generate PDF Report

```bash
python main.py report
```

---

## 📁 Project Structure

```
healthcare_security/
├── main.py                          # CLI entry point
├── requirements.txt                 # All dependencies
├── .env.example                     # Environment template
│
├── data/
│   └── generate_dataset.py          # Synthetic EHR dataset generator (CERT r4.1)
│
├── models/
│   └── train_models.py              # Dual-model ML training (IF + RF + SHAP)
│
├── utils/
│   └── state_schema.py              # LangGraph AgentState TypedDict
│
├── agents/
│   ├── pipeline_agents.py           # All 7 specialized agents
│   └── langgraph_pipeline.py        # Graph builder + conditional routing
│
├── api/
│   └── main.py                      # FastAPI + WebSocket backend
│
├── dashboard/
│   └── app.py                       # Streamlit security dashboard
│
├── reports/
│   └── generate_report.py           # PDF incident report generator (Claude API)
│
└── tests/
    └── test_pipeline.py             # pytest test suite
```

---

## 🔧 API Reference

### Process a Single Event
```bash
POST /api/v1/process-event
Content-Type: application/json

{
  "event_id": "EVT001",
  "user_id": "USR001",
  "username": "jdoe",
  "role": "Doctor",
  "records_accessed": 87,
  "access_type": "export",
  ...
}
```

### Batch Replay
```bash
POST /api/v1/batch-replay?limit=50
```

### HITL Decision
```bash
POST /api/v1/hitl/decision
{
  "incident_id": "INC-20240315-ABC123",
  "reviewer_id": "SECURITY_OFFICER_001",
  "decision": "APPROVED",
  "notes": "Verified with department head — legitimate emergency access"
}
```

### Real-Time WebSocket
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/events");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data); // { type: "event_processed", data: { risk_score, decision, ... } }
};
```

---

## ⚙️ Risk Thresholds

| Score Range | Risk Level | Decision         | Action                          |
|-------------|------------|------------------|---------------------------------|
| < 0.30      | LOW        | ALLOW            | Access granted, silent log      |
| 0.30–0.54   | MEDIUM     | ALLOW_WITH_ALERT | Access granted + security alert |
| 0.55–0.74   | HIGH       | RESTRICT         | View-only mode, no downloads    |
| ≥ 0.75      | CRITICAL   | BLOCK            | Session revoked, account locked |

**HITL Override:** If BLOCK + high-privilege role (Doctor/Admin/CMO) → pipeline pauses for human review.

---

## 🔍 ML Engine Details

| Component         | Config                                           |
|-------------------|--------------------------------------------------|
| Isolation Forest  | 200 trees, 6% contamination, weight: 40%        |
| Random Forest     | 200 trees, balanced class weights, weight: 60%  |
| SMOTE             | Handles class imbalance (6% malicious)          |
| SHAP              | TreeExplainer, per-feature attribution          |
| Features          | 38 behavioral features across 5 categories     |

---

## 🛡️ HIPAA Compliance

- All audit narratives generated via Claude API are PHI-free
- Every event produces a structured HIPAA audit entry (`logs/hipaa_audit.jsonl`)
- SHAP values provide full explainability for regulatory review
- HITL decisions are timestamped and attributed to named reviewers
- PDF reports contain no protected health information

---

## 🧪 Running Specific Test Scenarios

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/ -v -k "not FullPipeline"

# Run integration tests only
pytest tests/ -v -k "FullPipeline"

# Test with coverage
pytest tests/ --cov=agents --cov=utils --cov-report=term-missing
```
