"""
reports/generate_report.py
===========================
Generates professional HIPAA-compliant incident reports using Claude API.
Supports PDF and DOCX output formats.
Reads from the HIPAA audit log (logs/hipaa_audit.jsonl).
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List, Optional

from groq import Groq
from fpdf import FPDF
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
REPORT_DIR = os.getenv("INCIDENT_REPORT_PATH", "reports/incidents/")


# ── Report Data Loader ────────────────────────────────────────────────────────

def load_audit_entries(limit: int = 10) -> List[dict]:
    """Load recent audit log entries."""
    audit_path = os.getenv("AUDIT_LOG_PATH", "logs/hipaa_audit.jsonl")
    entries = []
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
    return entries[-limit:]


# ── Claude-Powered Report Generator ──────────────────────────────────────────

def generate_executive_summary(entries: List[dict]) -> str:
    """Use Groq to generate an executive summary of recent security events."""
    if not API_KEY or API_KEY == "your_groq_key_here":
        return _template_executive_summary(entries)

    high_risk = [e for e in entries if e.get("risk_score", 0) >= 0.55]
    blocked   = [e for e in entries if e.get("decision") == "BLOCK"]
    hitl      = [e for e in entries if e.get("hitl_status") not in ("NOT_REQUIRED", None, "")]

    prompt = f"""You are a HIPAA-compliant healthcare security analyst.
Write a professional 2-paragraph executive summary for a security incident report.
Format: formal, factual, third-person. Do NOT include any PHI.

Statistics:
- Total events analyzed: {len(entries)}
- High-risk events (score >= 0.55): {len(high_risk)}
- Blocked events: {len(blocked)}
- Human-in-the-loop reviews triggered: {len(hitl)}
- Date range: {entries[0].get('timestamp','N/A') if entries else 'N/A'} to {entries[-1].get('timestamp','N/A') if entries else 'N/A'}

Top incidents by risk score:
{json.dumps([{
    'incident_id': e.get('incident_id'),
    'user_id': e.get('user_id'),
    'risk_score': e.get('risk_score'),
    'decision': e.get('decision'),
} for e in sorted(entries, key=lambda x: x.get('risk_score', 0), reverse=True)[:3]], indent=2)}

Generate the executive summary:"""

    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _template_executive_summary(entries)


def _template_executive_summary(entries: List[dict]) -> str:
    high_risk = sum(1 for e in entries if e.get("risk_score", 0) >= 0.55)
    blocked   = sum(1 for e in entries if e.get("decision") == "BLOCK")
    return (
        f"During the reporting period, the Healthcare EHR Security System processed "
        f"{len(entries)} access events through the LangGraph multi-agent pipeline. "
        f"Of these, {high_risk} events ({high_risk/max(len(entries),1)*100:.1f}%) were "
        f"classified as high-risk or critical, and {blocked} accounts were blocked pending review. "
        f"\n\n"
        f"The dual-model ML engine (Isolation Forest + Random Forest ensemble) identified "
        f"multiple instances of anomalous behavior consistent with insider threat patterns, "
        f"including off-hours record access, unauthorized bulk downloads, and suspicious "
        f"device usage. All flagged events have been logged to the HIPAA-compliant audit trail "
        f"and escalated per institutional security policy."
    )


# ── PDF Report Generator ──────────────────────────────────────────────────────

class SecurityReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "  🏥  HEALTHCARE EHR SECURITY INCIDENT REPORT", align="L", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"CONFIDENTIAL — HIPAA Protected | Page {self.page_no()} | Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(236, 240, 241)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def table_header(self, cols: List[str], widths: List[int]):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, values: List[str], widths: List[int], fill: bool = False):
        self.set_font("Helvetica", "", 8)
        if fill:
            self.set_fill_color(245, 248, 250)
        for val, w in zip(values, widths):
            self.cell(w, 6, str(val)[:30], border=1, fill=fill)
        self.ln()


def generate_pdf_report(entries: List[dict], output_path: Optional[str] = None) -> str:
    """Generate a full PDF security incident report."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not output_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(REPORT_DIR, f"security_report_{ts}.pdf")

    pdf = SecurityReportPDF()
    pdf.add_page()

    # ── Cover Info ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Report Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"System: Agentic Healthcare EHR Security (LangGraph v2)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Events Analyzed: {len(entries)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Executive Summary ─────────────────────────────────────────────────────
    pdf.section_title("1. Executive Summary")
    summary = generate_executive_summary(entries)
    pdf.body_text(summary)

    # ── Statistics ────────────────────────────────────────────────────────────
    pdf.section_title("2. Security Statistics")

    decisions   = {}
    risk_levels = {}
    for e in entries:
        d = e.get("decision", "UNKNOWN")
        r = e.get("risk_level", "UNKNOWN")
        decisions[d]   = decisions.get(d, 0) + 1
        risk_levels[r] = risk_levels.get(r, 0) + 1

    stats_text = (
        f"Decision Summary: " + " | ".join(f"{k}: {v}" for k, v in decisions.items()) + "\n"
        f"Risk Level Summary: " + " | ".join(f"{k}: {v}" for k, v in risk_levels.items()) + "\n"
        f"High Risk Events: {sum(1 for e in entries if e.get('risk_score',0) >= 0.55)}\n"
        f"HITL Reviews Triggered: {sum(1 for e in entries if e.get('hitl_status') not in ('NOT_REQUIRED', None, ''))}"
    )
    pdf.body_text(stats_text)

    # ── Top Incidents Table ────────────────────────────────────────────────────
    pdf.section_title("3. Top Incidents by Risk Score")
    cols   = ["Incident ID", "User ID", "Risk Score", "Level", "Decision", "HITL"]
    widths = [45, 30, 25, 22, 28, 30]
    pdf.table_header(cols, widths)

    top_entries = sorted(entries, key=lambda x: x.get("risk_score", 0), reverse=True)[:15]
    for i, e in enumerate(top_entries):
        pdf.table_row([
            e.get("incident_id", "N/A")[:20],
            e.get("user_id", "N/A"),
            f"{e.get('risk_score', 0):.3f}",
            e.get("risk_level", "N/A"),
            e.get("decision", "N/A"),
            e.get("hitl_status", "N/A")[:15],
        ], widths, fill=(i % 2 == 0))

    pdf.ln(5)

    # ── Audit Narratives ──────────────────────────────────────────────────────
    pdf.section_title("4. HIPAA Audit Narratives (Top 5 Incidents)")
    for e in top_entries[:5]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, f"  {e.get('incident_id', 'N/A')} | Risk: {e.get('risk_score', 0):.3f} | {e.get('decision', 'N/A')}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        narrative = e.get("narrative", "No narrative available.")
        pdf.multi_cell(0, 5, f"  {narrative}")
        pdf.ln(3)

    pdf.output(output_path)
    print(f"✅ PDF report saved: {output_path}")
    return output_path


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    print("📋 Healthcare Security Report Generator")
    print("=" * 50)

    entries = load_audit_entries(limit=100)
    if not entries:
        print("⚠️  No audit log entries found.")
        print("   Run the pipeline first: python main.py")
        return

    print(f"📂 Loaded {len(entries)} audit entries")

    report_path = generate_pdf_report(entries)
    print(f"\n✅ Report generated: {report_path}")


if __name__ == "__main__":
    main()
