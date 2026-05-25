"""
main.py
=======
Main entry point for the Healthcare EHR Security pipeline.

Modes:
  python main.py demo        — Run 20 synthetic events through the full pipeline
  python main.py single      — Process a single hardcoded high-risk event
  python main.py setup       — Generate dataset + train models (full setup)
  python main.py report      — Generate PDF incident report from audit log
  python main.py api         — Launch FastAPI backend
  python main.py dashboard   — Launch Streamlit dashboard
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

load_dotenv()
console = Console()


# ── Full Setup ────────────────────────────────────────────────────────────────

def run_setup():
    console.print(Panel.fit("🏥 Healthcare EHR Security — Full Setup", style="bold blue"))

    console.print("\n[bold]Step 1/2:[/bold] Generating synthetic dataset...")
    subprocess.run([sys.executable, "data/generate_dataset.py"], check=True)

    console.print("\n[bold]Step 2/2:[/bold] Training ML models...")
    subprocess.run([sys.executable, "models/train_models.py"], check=True)

    console.print("\n[bold green]✅ Setup complete![/bold green]")
    console.print("   → Run [cyan]python main.py demo[/cyan] to test the pipeline")
    console.print("   → Run [cyan]python main.py api[/cyan] to launch the API server")
    console.print("   → Run [cyan]python main.py dashboard[/cyan] to launch the Streamlit dashboard")


# ── Demo Mode ─────────────────────────────────────────────────────────────────

def run_demo(n_events: int = 20):
    console.print(Panel.fit("🔍 Pipeline Demo — Processing Synthetic Events", style="bold cyan"))

    # Load dataset
    if not os.path.exists("data/ehr_access_log.csv"):
        console.print("[red]Dataset not found. Run: python main.py setup[/red]")
        sys.exit(1)

    df = pd.read_csv("data/ehr_access_log.csv").fillna(0)

    # Mix of benign and malicious events
    malicious = df[df["is_malicious"] == 1].head(n_events // 2)
    benign    = df[df["is_malicious"] == 0].head(n_events // 2)
    sample_df = pd.concat([malicious, benign]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Build pipeline
    from agents.langgraph_pipeline import build_pipeline, run_pipeline
    graph = build_pipeline()

    results = []
    table = Table(title=f"\n📊 Pipeline Results ({len(sample_df)} events)", show_header=True, header_style="bold magenta")
    table.add_column("Event ID",    width=14)
    table.add_column("User/Role",   width=20)
    table.add_column("Score",       width=7)
    table.add_column("Risk Level",  width=10)
    table.add_column("Decision",    width=18)
    table.add_column("HITL",        width=14)
    table.add_column("Path",        width=50)

    risk_colors = {
        "LOW": "green", "MEDIUM": "yellow", "HIGH": "orange1", "CRITICAL": "red"
    }
    decision_colors = {
        "ALLOW": "green", "ALLOW_WITH_ALERT": "yellow",
        "RESTRICT": "orange1", "BLOCK": "red"
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Processing events...", total=len(sample_df))

        for _, row in sample_df.iterrows():
            event = row.to_dict()
            state = run_pipeline(graph, event)

            risk    = str(state["risk_level"])
            decision = str(state["policy_decision"])
            score   = state["ensemble_risk_score"]
            path    = " → ".join(state["processing_path"])
            hitl    = str(state.get("hitl_status", "N/A"))

            table.add_row(
                str(event.get("event_id", "N/A"))[:14],
                f"{event.get('username','?')[:10]} ({event.get('role','?')[:8]})",
                f"[{risk_colors.get(risk, 'white')}]{score:.3f}[/]",
                f"[{risk_colors.get(risk, 'white')}]{risk}[/]",
                f"[{decision_colors.get(decision, 'white')}]{decision}[/]",
                hitl,
                path[:48],
            )

            results.append({
                "event_id": event.get("event_id"),
                "score":    score,
                "risk":     risk,
                "decision": decision,
                "hitl":     hitl,
                "is_malicious": int(event.get("is_malicious", 0)),
            })

            progress.advance(task)

    console.print(table)

    # Summary
    blocked   = sum(1 for r in results if r["decision"] == "BLOCK")
    high_risk = sum(1 for r in results if r["score"] >= 0.55)
    hitl_trig = sum(1 for r in results if r["hitl"] not in ("NOT_REQUIRED", "N/A"))
    true_mal  = sum(1 for r in results if r["is_malicious"])

    console.print(Panel(
        f"[bold]Total Events:[/bold]    {len(results)}\n"
        f"[bold]High Risk:[/bold]       {high_risk}\n"
        f"[bold]Blocked:[/bold]         {blocked}\n"
        f"[bold]HITL Triggered:[/bold]  {hitl_trig}\n"
        f"[bold]True Malicious:[/bold]  {true_mal}",
        title="📈 Summary",
        style="bold green",
    ))

    console.print(f"\n[dim]📝 HIPAA audit log: {os.getenv('AUDIT_LOG_PATH', 'logs/hipaa_audit.jsonl')}[/dim]")


# ── Single Event Test ─────────────────────────────────────────────────────────

def run_single():
    """Process a single high-risk Doctor event to demonstrate HITL path."""
    console.print(Panel.fit("🚨 Single High-Risk Event Test (HITL Path)", style="bold red"))

    high_risk_event = {
        "event_id":              "EVT-DEMO-CRITICAL",
        "timestamp":             "2024-03-15T02:30:00",
        "user_id":               "USR0042",
        "username":              "dr_suspicious",
        "role":                  "Doctor",
        "department":            "Oncology",
        "privilege_level":       "high",
        "record_limit":          20,
        "records_accessed":      87,
        "access_type":           "export",
        "device_type":           "external_usb",
        "location":              "unknown",
        "session_duration_min":  8,
        "external_emails_sent":  12,
        "off_hours_access":      1,
        "bulk_download_flag":    1,
        "unauthorized_access":   1,
        "usb_connected":         1,
        "failed_auth_attempts":  4,
        "vpn_usage":             0,
        "is_weekend":            0,
        "after_midnight":        1,
        "avg_records_per_session": 45.0,
        "max_records_session":   87.0,
        "total_events":          85,
        "off_hours_ratio":       0.72,
        "bulk_download_ratio":   0.65,
        "external_email_ratio":  0.58,
        "avg_session_duration":  9.0,
        "usb_usage_count":       18,
        "failed_auth_ratio":     0.40,
        "unique_access_types":   4,
        "vpn_ratio":             0.0,
        "weekend_access_ratio":  0.15,
        "unauthorized_ratio":    0.78,
        "external_email_count":  49,
        "records_over_limit":    4.35,
        "session_anomaly_score": 1,
        "privilege_risk":        1.0,
        "device_risk":           1.0,
        "location_risk":         1.0,
        "access_type_risk":      0.9,
    }

    from agents.langgraph_pipeline import build_pipeline, run_pipeline
    graph = build_pipeline()
    state = run_pipeline(graph, high_risk_event)

    console.print(f"\n[bold]Risk Score:[/bold]    {state['ensemble_risk_score']:.4f}")
    console.print(f"[bold]Risk Level:[/bold]    [red]{state['risk_level']}[/red]")
    console.print(f"[bold]IF Score:[/bold]      {state['isolation_forest_score']:.4f}")
    console.print(f"[bold]RF Prob:[/bold]       {state['random_forest_prob']:.4f}")
    console.print(f"[bold]Decision:[/bold]      [red]{state['policy_decision']}[/red]")
    console.print(f"[bold]HITL Status:[/bold]   {state['hitl_status']}")
    console.print(f"[bold]Mitigation:[/bold]    {[a.value for a in state['mitigation_actions']]}")
    console.print(f"\n[bold]Processing Path:[/bold]")
    console.print("  " + " → ".join(state["processing_path"]))

    if state.get("top_risk_factors"):
        console.print("\n[bold]Top SHAP Risk Factors:[/bold]")
        for f in state["top_risk_factors"]:
            direction = "🔴 +" if f["shap_contribution"] > 0 else "🟢 "
            console.print(f"  {direction}{f['feature']:35s} SHAP: {f['shap_contribution']:+.4f}  |  Value: {f['value']}")

    console.print(f"\n[bold]📋 Audit Narrative:[/bold]")
    console.print(Panel(state["audit_narrative"], style="dim"))


# ── API & Dashboard Launchers ─────────────────────────────────────────────────

def launch_api():
    console.print("[bold cyan]🚀 Launching FastAPI backend...[/bold cyan]")
    console.print(f"   URL: http://localhost:{os.getenv('API_PORT', 8000)}")
    console.print(f"   Docs: http://localhost:{os.getenv('API_PORT', 8000)}/docs")
    os.system(f"uvicorn api.main:app --host 0.0.0.0 --port {os.getenv('API_PORT', 8000)} --reload")


def launch_dashboard():
    console.print("[bold cyan]🚀 Launching Streamlit dashboard...[/bold cyan]")
    console.print(f"   URL: http://localhost:{os.getenv('STREAMLIT_PORT', 8501)}")
    os.system(f"streamlit run dashboard/app.py --server.port {os.getenv('STREAMLIT_PORT', 8501)}")


def generate_report():
    from reports.generate_report import load_audit_entries, generate_pdf_report
    entries = load_audit_entries(100)
    if not entries:
        console.print("[red]No audit entries found. Run: python main.py demo[/red]")
        return
    path = generate_pdf_report(entries)
    console.print(f"[green]✅ Report: {path}[/green]")


# ── CLI Router ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "setup":
        run_setup()
    elif mode == "demo":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        run_demo(n)
    elif mode == "single":
        run_single()
    elif mode == "api":
        launch_api()
    elif mode == "dashboard":
        launch_dashboard()
    elif mode == "report":
        generate_report()
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        console.print("Usage: python main.py [setup|demo|single|api|dashboard|report]")
        sys.exit(1)
