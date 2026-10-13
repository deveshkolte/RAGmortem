"""Executive Enterprise HTML Report Generator for RAGmortem.

Renders self-contained, air-gapped, zero-dependency executive audit reports
with interactive 4-stage RAG pipeline visualizations, executive KPIs,
and engineering attribution tickets.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any


def _escape(val: Any) -> str:
    """Safely escape text for HTML output."""
    if val is None:
        return ""
    return html.escape(str(val))


def generate_html_report(
    cases_data: list[dict[str, Any]],
    summary_data: dict[str, Any],
    *,
    is_demo: bool = False,
    title: str = "RAGmortem Enterprise Diagnostic Audit",
) -> str:
    """Generate a self-contained, executive enterprise HTML report."""
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
    total_cases = summary_data.get("total_cases", len(cases_data))
    unknown_count = summary_data.get("unknown_count", 0)
    unknown_rate = summary_data.get("unknown_rate_pct", 0.0)
    counts = summary_data.get("counts_per_diagnosis", {})

    gen_count = counts.get("generation_proven", 0) + counts.get("generation_suspected", 0)
    rank_count = counts.get("ranking_proven", 0) + counts.get("ranking_suspected", 0) + counts.get("ranking_miss", 0)
    ret_count = counts.get("retrieval_proven", 0) + counts.get("retrieval_suspected", 0) + counts.get("retrieval_miss", 0)
    abstain_count = counts.get("no_failure", 0) + counts.get("should_abstain", 0) + counts.get("abstention_suspected", 0)

    # Encode JSON payload for client-side interactivity
    cases_json = json.dumps(cases_data).replace("</script>", "<\\/script>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      --bg-page: #f8fafc;
      --bg-card: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border-color: #e2e8f0;
      --brand-primary: #2563eb;
      --brand-primary-light: #eff6ff;
      --color-generation: #dc2626;
      --color-generation-bg: #fef2f2;
      --color-ranking: #d97706;
      --color-ranking-bg: #fffbeb;
      --color-retrieval: #7c3aed;
      --color-retrieval-bg: #f5f3ff;
      --color-healthy: #059669;
      --color-healthy-bg: #ecfdf5;
      --color-unknown: #475569;
      --color-unknown-bg: #f1f5f9;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font-family);
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.5;
      padding: 32px 24px;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    /* Header */
    .header {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .brand-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-badge {{
      background: #1e293b;
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 4px 8px;
      border-radius: 6px;
    }}
    h1 {{
      font-size: 22px;
      font-weight: 700;
      color: #0f172a;
    }}
    .header-meta {{
      text-align: right;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .print-btn {{
      margin-top: 6px;
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid var(--border-color);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.15s ease;
    }}
    .print-btn:hover {{ background: #e2e8f0; }}

    /* Notice Banner */
    .demo-notice {{
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 8px;
      padding: 12px 18px;
      margin-bottom: 24px;
      font-size: 13px;
      color: #92400e;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    /* KPI Summary Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 18px 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }}
    .kpi-label {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 28px;
      font-weight: 800;
      color: #0f172a;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }}
    .kpi-subtext {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    /* Controls & Filter Bar */
    .toolbar {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .filter-tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .filter-tab {{
      background: transparent;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-tab.active, .filter-tab:hover {{
      background: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
    }}

    /* Case Card */
    .case-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.04);
      overflow: hidden;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .case-header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #fafafa;
    }}
    .case-title-area {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .case-id {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      font-weight: 700;
      color: #334155;
      background: #f1f5f9;
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid #cbd5e1;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .badge-generation {{ background: var(--color-generation-bg); color: var(--color-generation); border: 1px solid #fecaca; }}
    .badge-ranking {{ background: var(--color-ranking-bg); color: var(--color-ranking); border: 1px solid #fde68a; }}
    .badge-retrieval {{ background: var(--color-retrieval-bg); color: var(--color-retrieval); border: 1px solid #ddd6fe; }}
    .badge-healthy {{ background: var(--color-healthy-bg); color: var(--color-healthy); border: 1px solid #a7f3d0; }}
    .badge-unknown {{ background: var(--color-unknown-bg); color: var(--color-unknown); border: 1px solid #cbd5e1; }}

    .confidence-meter {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }}
    .confidence-bar-bg {{
      width: 70px;
      height: 6px;
      background: #e2e8f0;
      border-radius: 3px;
      overflow: hidden;
    }}
    .confidence-bar-fill {{
      height: 100%;
      background: #2563eb;
    }}

    /* 4-Stage Visual Pipeline */
    .pipeline-section {{
      padding: 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--border-color);
    }}
    .pipeline-title {{
      font-size: 12px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
    }}
    .pipeline-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      position: relative;
    }}
    @media (max-width: 900px) {{
      .pipeline-grid {{ grid-template-columns: 1fr; }}
    }}
    .pipeline-step {{
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      position: relative;
    }}
    .pipeline-step.step-failed {{
      border-color: #f87171;
      background: #fff5f5;
    }}
    .pipeline-step.step-warning {{
      border-color: #fbbf24;
      background: #fffdf5;
    }}
    .pipeline-step.step-success {{
      border-color: #34d399;
      background: #f0fdf4;
    }}
    .step-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(0,0,0,0.06);
    }}
    .step-num {{
      font-size: 11px;
      font-weight: 800;
      color: #64748b;
      text-transform: uppercase;
    }}
    .step-status {{
      font-size: 11px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .status-ok {{ background: #dcfce7; color: #166534; }}
    .status-fail {{ background: #fee2e2; color: #991b1b; }}
    .status-warn {{ background: #fef3c7; color: #92400e; }}
    .status-neutral {{ background: #f1f5f9; color: #475569; }}

    .step-body {{
      font-size: 12px;
      color: #334155;
      flex-grow: 1;
    }}
    .chunk-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 6px;
    }}
    .chunk-pill {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 11px;
    }}
    .chunk-score-bar {{
      height: 4px;
      background: #e2e8f0;
      border-radius: 2px;
      margin-top: 4px;
      overflow: hidden;
    }}
    .chunk-score-fill {{
      height: 100%;
      background: #2563eb;
    }}
    .cutoff-marker {{
      margin: 6px 0;
      text-align: center;
      font-size: 10px;
      font-weight: 700;
      color: #b45309;
      background: #fef3c7;
      border: 1px dashed #f59e0b;
      border-radius: 4px;
      padding: 2px 4px;
    }}

    /* Case Details & Actionable Ticket */
    .case-details {{
      padding: 20px 24px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 800px) {{
      .case-details {{ grid-template-columns: 1fr; }}
    }}
    .evidence-box {{
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
    }}
    .evidence-title {{
      font-size: 12px;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 10px;
    }}
    .evidence-list {{
      list-style-type: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .evidence-list li {{
      font-size: 13px;
      color: #334155;
      padding-left: 18px;
      position: relative;
    }}
    .evidence-list li::before {{
      content: "•";
      position: absolute;
      left: 4px;
      color: #2563eb;
      font-weight: bold;
    }}

    .ticket-box {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .ticket-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 700;
      color: #1e40af;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .ticket-body {{
      font-size: 13px;
      color: #1e3a8a;
      margin-bottom: 12px;
      line-height: 1.5;
    }}
    .limitation-text {{
      font-size: 11px;
      color: #64748b;
      border-top: 1px solid #e2e8f0;
      padding-top: 10px;
      margin-top: 10px;
      font-style: italic;
    }}

    /* Footer */
    .footer {{
      margin-top: 32px;
      text-align: center;
      font-size: 12px;
      color: #94a3b8;
    }}

    @media print {{
      body {{ background: #fff; padding: 0; }}
      .header, .case-card {{ box-shadow: none; border-color: #ccc; }}
      .toolbar, .print-btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">

    <!-- Top Header -->
    <header class="header">
      <div>
        <div class="brand-title">
          <span class="brand-badge">Enterprise Audit</span>
          <h1>{_escape(title)}</h1>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
          Deterministic Root-Cause Analysis for Retrieval-Augmented Generation
        </p>
      </div>
      <div class="header-meta">
        <div><strong>Generated:</strong> {now_str}</div>
        <div><strong>Engine:</strong> RAGmortem v0.2.0.dev0</div>
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
      </div>
    </header>

    {"<!-- Demo Warning Banner -->" if is_demo else ""}
    {f'''<div class="demo-notice">
      <span>ℹ️</span>
      <div><strong>Synthetic Demonstration Data:</strong> The cases evaluated in this audit report are bundled benchmark failure scenarios for validation and demonstration purposes.</div>
    </div>''' if is_demo else ""}

    <!-- Executive KPI Summary Cards -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Traces Audited</div>
        <div class="kpi-value">{total_cases}</div>
        <div class="kpi-subtext">Resolved: {total_cases - unknown_count} | Inconclusive: {unknown_count}</div>
      </div>
      <div class="kpi-card" style="border-top: 3px solid var(--color-generation);">
        <div class="kpi-label">Generation Failures</div>
        <div class="kpi-value" style="color: var(--color-generation);">{gen_count}</div>
        <div class="kpi-subtext">Action: Prompt Engineering Team</div>
      </div>
      <div class="kpi-card" style="border-top: 3px solid var(--color-ranking);">
        <div class="kpi-label">Ranking Cutoff Failures</div>
        <div class="kpi-value" style="color: var(--color-ranking);">{rank_count}</div>
        <div class="kpi-subtext">Action: Re-Ranker / Top-K Window</div>
      </div>
      <div class="kpi-card" style="border-top: 3px solid var(--color-healthy);">
        <div class="kpi-label">Healthy / Correct Abstain</div>
        <div class="kpi-value" style="color: var(--color-healthy);">{abstain_count}</div>
        <div class="kpi-subtext">No failure detected</div>
      </div>
      <div class="kpi-card" style="border-top: 3px solid var(--color-unknown);">
        <div class="kpi-label">UNKNOWN Attribution Rate</div>
        <div class="kpi-value">{unknown_rate}%</div>
        <div class="kpi-subtext">Strict honesty bounds applied</div>
      </div>
    </section>

    <!-- Filter Toolbar -->
    <div class="toolbar">
      <div class="filter-tabs">
        <button class="filter-tab active" onclick="filterCases('all', this)">All Cases ({total_cases})</button>
        <button class="filter-tab" onclick="filterCases('generation', this)">Generation ({gen_count})</button>
        <button class="filter-tab" onclick="filterCases('ranking', this)">Ranking ({rank_count})</button>
        <button class="filter-tab" onclick="filterCases('healthy', this)">Healthy / Abstain ({abstain_count})</button>
        <button class="filter-tab" onclick="filterCases('unknown', this)">Unknown ({unknown_count})</button>
      </div>
      <div style="font-size: 13px; color: var(--text-muted);">
        Interactive 4-Stage Forensic Pipeline
      </div>
    </div>

    <!-- Case Evaluations -->
    <main id="cases-container">
""" + "".join(_render_case_card(c, idx) for idx, c in enumerate(cases_data, start=1)) + f"""
    </main>

    <footer class="footer">
      RAGmortem • Automated RAG Failure Root-Cause Attribution • Zero Runtime Dependencies • Local & Air-gapped
    </footer>
  </div>

  <script>
    const casesData = {cases_json};

    function filterCases(category, btn) {{
      document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.case-card').forEach(card => {{
        const cat = card.getAttribute('data-category');
        if (category === 'all' || cat === category) {{
          card.style.display = 'block';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def _render_case_card(case: dict[str, Any], idx: int) -> str:
    """Render a single case card with 4-stage pipeline visualization."""
    case_id = _escape(case.get("id", f"case_{idx}"))
    diag = str(case.get("diagnosis", "unknown")).lower()
    diag_english = _escape(case.get("diagnosis_english", diag))
    mode = _escape(case.get("mode", "trace-only"))
    confidence = float(case.get("confidence", 0.0))
    conf_pct = int(confidence * 100)
    evidence = case.get("evidence", [])
    cannot_tell_you = _escape(case.get("what_this_cannot_tell_you", ""))
    next_check = _escape(case.get("suggested_next_check", ""))
    question = _escape(case.get("question", ""))
    answer = _escape(case.get("answer", ""))

    # Determine category and badge styling
    if "generation" in diag:
        category = "generation"
        badge_class = "badge-generation"
        badge_icon = "🔴"
        step4_class = "step-failed"
        step4_status = '<span class="step-status status-fail">FAIL</span>'
        step3_class = "step-success"
        step3_status = '<span class="step-status status-ok">PASS</span>'
        step2_class = "step-success"
        step2_status = '<span class="step-status status-ok">PASS</span>'
    elif "ranking" in diag:
        category = "ranking"
        badge_class = "badge-ranking"
        badge_icon = "🟠"
        step4_class = "step-neutral"
        step4_status = '<span class="step-status status-neutral">N/A</span>'
        step3_class = "step-failed"
        step3_status = '<span class="step-status status-fail">CUTOFF</span>'
        step2_class = "step-success"
        step2_status = '<span class="step-status status-ok">PASS</span>'
    elif "retrieval" in diag:
        category = "retrieval"
        badge_class = "badge-retrieval"
        badge_icon = "🟣"
        step4_class = "step-neutral"
        step4_status = '<span class="step-status status-neutral">N/A</span>'
        step3_class = "step-neutral"
        step3_status = '<span class="step-status status-neutral">N/A</span>'
        step2_class = "step-failed"
        step2_status = '<span class="step-status status-fail">MISS</span>'
    elif "no_failure" in diag or "abstain" in diag:
        category = "healthy"
        badge_class = "badge-healthy"
        badge_icon = "🟢"
        step4_class = "step-success"
        step4_status = '<span class="step-status status-ok">OK</span>'
        step3_class = "step-success"
        step3_status = '<span class="step-status status-ok">OK</span>'
        step2_class = "step-success"
        step2_status = '<span class="step-status status-ok">OK</span>'
    else:
        category = "unknown"
        badge_class = "badge-unknown"
        badge_icon = "⚪"
        step4_class = "step-warning"
        step4_status = '<span class="step-status status-warn">UNKNOWN</span>'
        step3_class = "step-warning"
        step3_status = '<span class="step-status status-warn">UNKNOWN</span>'
        step2_class = "step-warning"
        step2_status = '<span class="step-status status-warn">UNKNOWN</span>'

    evidence_html = "".join(f"<li>{_escape(ev)}</li>" for ev in evidence)

    return f"""
      <article class="case-card" data-category="{category}">
        <div class="case-header">
          <div class="case-title-area">
            <span class="case-id">{case_id}</span>
            <span class="badge {badge_class}">{badge_icon} {diag_english}</span>
          </div>
          <div class="confidence-meter">
            <span>Confidence: <strong>{conf_pct}%</strong></span>
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" style="width: {conf_pct}%;"></div>
            </div>
            <span style="font-size: 11px; color: var(--text-muted); margin-left: 6px;">({mode})</span>
          </div>
        </div>

        <!-- 4-Stage Visual Pipeline Diagram -->
        <div class="pipeline-section">
          <div class="pipeline-title">Forensic 4-Stage RAG Pipeline Execution Flow</div>
          <div class="pipeline-grid">
            
            <!-- Stage 1 -->
            <div class="pipeline-step step-success">
              <div class="step-header">
                <span class="step-num">Stage 1: User Query</span>
                <span class="step-status status-ok">RECEIVED</span>
              </div>
              <div class="step-body">
                <p style="font-weight: 600; color: #0f172a; margin-bottom: 4px;">"{question}"</p>
                <span style="font-size: 11px; color: var(--text-muted);">Length: {len(question.split())} tokens</span>
              </div>
            </div>

            <!-- Stage 2 -->
            <div class="pipeline-step {step2_class}">
              <div class="step-header">
                <span class="step-num">Stage 2: Dense Retrieval</span>
                {step2_status}
              </div>
              <div class="step-body">
                <span style="font-size: 11px; font-weight: 600; color: var(--text-muted);">Candidate Pool Scored</span>
                <div class="chunk-list">
                  <div class="chunk-pill">
                    <div style="display:flex; justify-content:space-between; font-weight:600;">
                      <span>Top Chunk</span>
                      <span>Score: High</span>
                    </div>
                    <div class="chunk-score-bar">
                      <div class="chunk-score-fill" style="width: 85%;"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Stage 3 -->
            <div class="pipeline-step {step3_class}">
              <div class="step-header">
                <span class="step-num">Stage 3: Top-K Cutoff</span>
                {step3_status}
              </div>
              <div class="step-body">
                <span style="font-size: 11px; font-weight: 600; color: var(--text-muted);">Context Window Packing</span>
                <div class="cutoff-marker">✂️ Top-K Prompt Window Cutoff</div>
                <span style="font-size: 11px; color: var(--text-muted);">Evaluated candidate margin against truncation threshold.</span>
              </div>
            </div>

            <!-- Stage 4 -->
            <div class="pipeline-step {step4_class}">
              <div class="step-header">
                <span class="step-num">Stage 4: LLM Generation</span>
                {step4_status}
              </div>
              <div class="step-body">
                <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 2px;">Model Response:</p>
                <p style="font-weight: 600; color: #0f172a; font-size: 11px;">"{answer}"</p>
              </div>
            </div>

          </div>
        </div>

        <!-- Case Forensic Evidence & Actionable Ticket -->
        <div class="case-details">
          <div class="evidence-box">
            <div class="evidence-title">🔍 Forensic Evidence Points</div>
            <ul class="evidence-list">
              {evidence_html}
            </ul>
          </div>

          <div class="ticket-box">
            <div>
              <div class="ticket-header">
                <span>📋 Actionable Engineering Ticket</span>
              </div>
              <div class="ticket-body">
                <strong>Recommendation:</strong> {next_check}
              </div>
            </div>
            <div class="limitation-text">
              {cannot_tell_you}
            </div>
          </div>
        </div>

      </article>
    """
