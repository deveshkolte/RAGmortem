"""Executive Enterprise HTML Report Generator for RAGmortem.

Renders self-contained, air-gapped, zero-dependency executive audit reports
with interactive 4-stage RAG pipeline visualizations, forensic breakpoint alerts,
executive KPIs, and engineering remediation tickets.
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
    now_str = datetime.now().strftime("%B %d, %Y • %H:%M:%S UTC")
    total_cases = summary_data.get("total_cases", len(cases_data))
    unknown_count = summary_data.get("unknown_count", 0)
    unknown_rate = summary_data.get("unknown_rate_pct", 0.0)
    counts = summary_data.get("counts_per_diagnosis", {})

    gen_count = counts.get("generation_proven", 0) + counts.get("generation_suspected", 0)
    rank_count = counts.get("ranking_proven", 0) + counts.get("ranking_suspected", 0) + counts.get("ranking_miss", 0)
    ret_count = counts.get("retrieval_proven", 0) + counts.get("retrieval_suspected", 0) + counts.get("retrieval_miss", 0)
    abstain_count = counts.get("no_failure", 0) + counts.get("should_abstain", 0) + counts.get("abstention_suspected", 0)

    cases_json = json.dumps(cases_data).replace("</script>", "<\\/script>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, Menlo, Monaco, Consolas, monospace;
      
      --bg-base: #07090E;
      --bg-surface: #0E131F;
      --bg-surface-elevated: #141B2D;
      --bg-surface-glass: rgba(14, 19, 31, 0.75);
      
      --border-subtle: rgba(255, 255, 255, 0.07);
      --border-light: rgba(255, 255, 255, 0.12);
      
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      
      --danger-main: #F43F5E;
      --danger-glow: rgba(244, 63, 94, 0.25);
      --danger-bg: rgba(244, 63, 94, 0.10);
      --danger-border: rgba(244, 63, 94, 0.35);
      
      --warning-main: #F59E0B;
      --warning-glow: rgba(245, 158, 11, 0.25);
      --warning-bg: rgba(245, 158, 11, 0.10);
      --warning-border: rgba(245, 158, 11, 0.35);
      
      --success-main: #10B981;
      --success-glow: rgba(16, 185, 129, 0.25);
      --success-bg: rgba(16, 185, 129, 0.10);
      --success-border: rgba(16, 185, 129, 0.35);
      
      --info-main: #38BDF8;
      --purple-main: #A855F7;
      
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-base);
      background-image: 
        radial-gradient(circle at 10% 0%, rgba(99, 102, 241, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 90% 5%, rgba(244, 63, 94, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.04) 0%, transparent 60%);
      background-attachment: fixed;
      color: var(--text-primary);
      line-height: 1.5;
      min-height: 100vh;
      padding: 32px 36px 64px;
      -webkit-font-smoothing: antialiased;
    }}

    .container {{
      max-width: 1380px;
      margin: 0 auto;
    }}

    /* Top Nav */
    .header {{
      background: var(--bg-surface-glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-light);
      border-radius: var(--radius-lg);
      padding: 20px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }}
    .brand-title {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .brand-badge {{
      background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 6px;
      box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    }}
    h1 {{
      font-size: 20px;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.02em;
    }}
    .header-meta {{
      text-align: right;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .print-btn {{
      margin-top: 6px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-light);
      padding: 6px 14px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .print-btn:hover {{
      background: #1E283F;
      border-color: rgba(255, 255, 255, 0.25);
    }}

    /* Notice Banner */
    .demo-notice {{
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: var(--radius-md);
      padding: 12px 18px;
      margin-bottom: 24px;
      font-size: 13px;
      color: #FDE68A;
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    /* KPI Summary Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: rgba(255, 255, 255, 0.1);
    }}
    .kpi-card.kpi-gen::before {{ background: var(--danger-main); }}
    .kpi-card.kpi-rank::before {{ background: var(--warning-main); }}
    .kpi-card.kpi-ok::before {{ background: var(--success-main); }}
    .kpi-card.kpi-unk::before {{ background: #94A3B8; }}

    .kpi-label {{
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 32px;
      font-weight: 800;
      color: var(--text-primary);
      font-family: var(--font-mono);
      display: flex;
      align-items: baseline;
      gap: 8px;
    }}
    .kpi-subtext {{
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 4px;
    }}

    /* Toolbar Filter Tabs */
    .toolbar {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 12px 20px;
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
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 700;
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .filter-tab.active, .filter-tab:hover {{
      background: rgba(99, 102, 241, 0.18);
      color: #fff;
      border-color: #6366F1;
      box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
    }}

    /* Case Card */
    .case-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      margin-bottom: 28px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      overflow: hidden;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .case-card:hover {{
      border-color: rgba(255, 255, 255, 0.15);
    }}
    .case-header {{
      padding: 20px 26px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(20, 27, 45, 0.4);
      flex-wrap: wrap;
      gap: 14px;
    }}
    .case-id {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      color: var(--text-secondary);
      background: rgba(0, 0, 0, 0.4);
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
    }}
    .verdict-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 800;
      padding: 6px 14px;
      border-radius: 9999px;
      letter-spacing: -0.01em;
    }}
    .badge-gen {{
      background: var(--danger-bg);
      color: #FDA4AF;
      border: 1px solid var(--danger-border);
      box-shadow: 0 0 16px var(--danger-glow);
    }}
    .badge-rank {{
      background: var(--warning-bg);
      color: #FDE68A;
      border: 1px solid var(--warning-border);
      box-shadow: 0 0 16px var(--warning-glow);
    }}
    .badge-ret {{
      background: rgba(168, 85, 247, 0.12);
      color: #E9D5FF;
      border: 1px solid rgba(168, 85, 247, 0.35);
      box-shadow: 0 0 16px rgba(168, 85, 247, 0.25);
    }}
    .badge-ok {{
      background: var(--success-bg);
      color: #A7F3D0;
      border: 1px solid var(--success-border);
      box-shadow: 0 0 16px var(--success-glow);
    }}
    .badge-unk {{
      background: rgba(148, 163, 184, 0.12);
      color: #E2E8F0;
      border: 1px solid rgba(148, 163, 184, 0.25);
    }}

    /* 4-Stage Visual Pipeline Diagram */
    .pipeline-section {{
      padding: 24px 26px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
    }}
    .pipeline-title {{
      font-size: 11px;
      font-weight: 800;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
    }}
    .flow-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      position: relative;
    }}
    @media (max-width: 960px) {{
      .flow-grid {{ grid-template-columns: 1fr; }}
    }}

    .flow-node {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px;
      display: flex;
      flex-direction: column;
      position: relative;
    }}
    .flow-node.breakpoint-alert {{
      border-color: var(--danger-main);
      background: rgba(244, 63, 94, 0.05);
      box-shadow: 0 0 20px rgba(244, 63, 94, 0.15);
    }}
    .flow-node.breakpoint-warning {{
      border-color: var(--warning-main);
      background: rgba(245, 158, 11, 0.05);
      box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }}
    .flow-node.passed {{
      border-color: rgba(16, 185, 129, 0.35);
    }}

    .breakpoint-pill {{
      position: absolute;
      top: -11px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--danger-main);
      color: #fff;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 9999px;
      box-shadow: 0 0 12px var(--danger-main);
      z-index: 2;
    }}
    .breakpoint-pill.warning-pill {{
      background: var(--warning-main);
      box-shadow: 0 0 12px var(--warning-main);
    }}

    .node-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border-subtle);
    }}
    .node-number {{
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
    }}
    .node-status {{
      font-size: 9px;
      font-weight: 800;
      font-family: var(--font-mono);
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .status-pass {{ background: var(--success-bg); color: var(--success-main); }}
    .status-fail {{ background: var(--danger-bg); color: var(--danger-main); }}
    .status-warn {{ background: var(--warning-bg); color: var(--warning-main); }}
    .status-neutral {{ background: rgba(148, 163, 184, 0.1); color: var(--text-muted); }}

    .node-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 6px;
    }}
    .node-body {{
      font-size: 11px;
      color: var(--text-secondary);
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    /* Cutoff Visualizer */
    .cutoff-diagram {{
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-top: 4px;
    }}
    .cutoff-item {{
      font-size: 10px;
      font-family: var(--font-mono);
      padding: 3px 6px;
      border-radius: 4px;
      display: flex;
      justify-content: space-between;
    }}
    .in-prompt {{
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: #A7F3D0;
    }}
    .cutoff-divider {{
      border-top: 1px dashed var(--warning-main);
      margin: 4px 0;
      text-align: center;
      position: relative;
    }}
    .cutoff-label {{
      font-size: 8px;
      font-weight: 800;
      color: var(--warning-main);
      background: #0E131F;
      padding: 0 4px;
      position: relative;
      top: -6px;
    }}
    .dropped-candidate {{
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.2);
      color: #FDA4AF;
    }}

    /* Case Details & Actionable Ticket */
    .case-details {{
      padding: 22px 26px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 860px) {{
      .case-details {{ grid-template-columns: 1fr; }}
    }}

    .evidence-box {{
      background: rgba(20, 27, 45, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px;
    }}
    .evidence-title {{
      font-size: 11px;
      font-weight: 800;
      color: var(--text-primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .evidence-list {{
      list-style-type: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .evidence-list li {{
      font-size: 12px;
      color: var(--text-primary);
      padding-left: 20px;
      position: relative;
      line-height: 1.5;
    }}
    .evidence-list li::before {{
      content: "•";
      position: absolute;
      left: 6px;
      color: #6366F1;
      font-weight: bold;
    }}

    .ticket-box {{
      background: linear-gradient(180deg, rgba(30, 41, 67, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(99, 102, 241, 0.35);
      border-radius: var(--radius-md);
      padding: 18px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .ticket-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .ticket-chip {{
      font-size: 10px;
      font-weight: 800;
      font-family: var(--font-mono);
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      color: #A5B4FC;
      padding: 2px 7px;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    .ticket-body {{
      font-size: 12px;
      color: #E2E8F0;
      line-height: 1.5;
      margin-bottom: 12px;
    }}
    .limitation-text {{
      font-size: 11px;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 10px;
      font-style: italic;
    }}

    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 12px;
      color: var(--text-muted);
    }}

    @media print {{
      body {{ background: #fff; color: #000; padding: 0; }}
      .header, .case-card, .kpi-card {{ box-shadow: none; border-color: #ccc; background: #fff; color: #000; }}
      .toolbar, .print-btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">

    <!-- Top Header -->
    <header class="header">
      <div class="brand-title">
        <span class="brand-badge">Audit Observatory</span>
        <div>
          <h1>{_escape(title)}</h1>
          <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
            Deterministic Forensic Root-Cause Analysis for Retrieval-Augmented Generation
          </p>
        </div>
      </div>
      <div class="header-meta">
        <div><strong>Generated:</strong> {now_str}</div>
        <div><strong>Engine:</strong> RAGmortem v0.2.0.dev0</div>
        <button class="print-btn" onclick="window.print()">🖨️ Export PDF / Print</button>
      </div>
    </header>

    {"<!-- Demo Warning Banner -->" if is_demo else ""}
    {f'''<div class="demo-notice">
      <span>ℹ️</span>
      <div><strong>Synthetic Demonstration Data:</strong> The cases evaluated in this audit report are bundled benchmark failure scenarios for demonstration and validation.</div>
    </div>''' if is_demo else ""}

    <!-- Executive KPI Summary Cards -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Traces Audited</div>
        <div class="kpi-value">{total_cases}</div>
        <div class="kpi-subtext">Resolved: {total_cases - unknown_count} | Inconclusive: {unknown_count}</div>
      </div>
      <div class="kpi-card kpi-gen">
        <div class="kpi-label">Generation Failures</div>
        <div class="kpi-value" style="color: #FDA4AF;">{gen_count}</div>
        <div class="kpi-subtext">Triage: Prompt Engineering Team</div>
      </div>
      <div class="kpi-card kpi-rank">
        <div class="kpi-label">Ranking Cutoff Failures</div>
        <div class="kpi-value" style="color: #FDE68A;">{rank_count}</div>
        <div class="kpi-subtext">Triage: Re-Ranker / Context Window</div>
      </div>
      <div class="kpi-card kpi-ok">
        <div class="kpi-label">Healthy / Correct Abstain</div>
        <div class="kpi-value" style="color: #A7F3D0;">{abstain_count}</div>
        <div class="kpi-subtext">No system defect detected</div>
      </div>
      <div class="kpi-card kpi-unk">
        <div class="kpi-label">UNKNOWN Attribution Rate</div>
        <div class="kpi-value">{unknown_rate}%</div>
        <div class="kpi-subtext">Strict honesty bounds applied</div>
      </div>
    </section>

    <!-- Filter Toolbar -->
    <div class="toolbar">
      <div class="filter-tabs">
        <button class="filter-tab active" onclick="filterCases('all', this)">All Traces ({total_cases})</button>
        <button class="filter-tab" onclick="filterCases('generation', this)">Generation ({gen_count})</button>
        <button class="filter-tab" onclick="filterCases('ranking', this)">Ranking ({rank_count})</button>
        <button class="filter-tab" onclick="filterCases('healthy', this)">Healthy ({abstain_count})</button>
        <button class="filter-tab" onclick="filterCases('unknown', this)">Unknown ({unknown_count})</button>
      </div>
      <div style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);">
        Forensic Pipeline Breakdown Active
      </div>
    </div>

    <!-- Case Evaluations -->
    <main id="cases-container">
""" + "".join(_render_case_card(c, idx) for idx, c in enumerate(cases_data, start=1)) + f"""
    </main>

    <footer class="footer">
      RAGmortem • Deterministic RAG Failure Root-Cause Attribution • Zero Runtime Dependencies • Air-gapped
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

    # Determine category, badge styling, and pipeline breakpoints
    breakpoint_html = ""
    team_chip = "Investigation Team"
    s1_class = "flow-node passed"
    s2_class = "flow-node passed"
    s3_class = "flow-node passed"
    s4_class = "flow-node passed"
    s2_stat = '<span class="node-status status-pass">PASS</span>'
    s3_stat = '<span class="node-status status-pass">PASS</span>'
    s4_stat = '<span class="node-status status-pass">PASS</span>'

    if "generation" in diag:
        category = "generation"
        badge_class = "badge-gen"
        badge_icon = "🔴"
        team_chip = "Prompt Engineering Team"
        s4_class = "flow-node breakpoint-alert"
        s4_stat = '<span class="node-status status-fail">FAIL</span>'
        breakpoint_html = '<div class="breakpoint-pill">🚨 BREAKPOINT: HALLUCINATION</div>'
    elif "ranking" in diag:
        category = "ranking"
        badge_class = "badge-rank"
        badge_icon = "🟠"
        team_chip = "Re-Ranker & Indexing Team"
        s3_class = "flow-node breakpoint-warning"
        s3_stat = '<span class="node-status status-warn">CUTOFF</span>'
        breakpoint_html = '<div class="breakpoint-pill warning-pill">⚠️ BREAKPOINT: TOP-K CUTOFF</div>'
    elif "retrieval" in diag:
        category = "retrieval"
        badge_class = "badge-ret"
        badge_icon = "🟣"
        team_chip = "Search & Embeddings Team"
        s2_class = "flow-node breakpoint-alert"
        s2_stat = '<span class="node-status status-fail">MISS</span>'
        breakpoint_html = '<div class="breakpoint-pill">🚨 BREAKPOINT: RETRIEVAL MISS</div>'
    elif "no_failure" in diag or "abstain" in diag:
        category = "healthy"
        badge_class = "badge-ok"
        badge_icon = "🟢"
        team_chip = "Quality & Telemetry Monitoring"
    else:
        category = "unknown"
        badge_class = "badge-unk"
        badge_icon = "⚪"
        team_chip = "Telemetry Instrumentation"
        s2_class = "flow-node"
        s3_class = "flow-node"
        s4_class = "flow-node"
        s2_stat = '<span class="node-status status-neutral">UNKNOWN</span>'
        s3_stat = '<span class="node-status status-neutral">UNKNOWN</span>'
        s4_stat = '<span class="node-status status-neutral">UNKNOWN</span>'

    evidence_html = "".join(f"<li>{_escape(ev)}</li>" for ev in evidence)

    return f"""
      <article class="case-card" data-category="{category}">
        <div class="case-header">
          <div style="display:flex; align-items:center; gap:12px;">
            <span class="case-id">{case_id}</span>
            <span class="verdict-badge {badge_class}">{badge_icon} {diag_english}</span>
          </div>
          <div style="display:flex; align-items:center; gap:14px; font-size:12px;">
            <span>Confidence: <strong style="font-family:var(--font-mono); color:#fff;">{conf_pct}%</strong></span>
            <span style="font-size:11px; color:var(--text-muted);">({mode})</span>
          </div>
        </div>

        <!-- 4-Stage Forensic Pipeline -->
        <div class="pipeline-section">
          <div class="pipeline-title">
            <span>Forensic 4-Stage RAG Pipeline Breakdown</span>
            <span>Where it succeeded vs where it broke</span>
          </div>
          <div class="flow-grid">
            
            <!-- Stage 1 -->
            <div class="{s1_class}">
              <div class="node-header">
                <span class="node-number">01 / INPUT</span>
                <span class="node-status status-pass">OK</span>
              </div>
              <div class="node-title">Stage 1: User Query</div>
              <div class="node-body">
                <p style="font-weight:600; color:#fff; font-size:12px;">"{question}"</p>
                <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
                  Token Budget: {len(question.split())} tokens
                </div>
              </div>
            </div>

            <!-- Stage 2 -->
            <div class="{s2_class}">
              {"<!-- Breakpoint -->" + breakpoint_html if "retrieval" in diag else ""}
              <div class="node-header">
                <span class="node-number">02 / RETRIEVAL</span>
                {s2_stat}
              </div>
              <div class="node-title">Vector Retrieval</div>
              <div class="node-body">
                <span style="font-size:11px; color:var(--text-muted);">Dense bi-encoder scoring candidate chunks against threshold bounds (0.47 / 0.50 / 0.53).</span>
              </div>
            </div>

            <!-- Stage 3 -->
            <div class="{s3_class}">
              {"<!-- Breakpoint -->" + breakpoint_html if "ranking" in diag else ""}
              <div class="node-header">
                <span class="node-number">03 / RANKING</span>
                {s3_stat}
              </div>
              <div class="node-title">Top-K Context Window</div>
              <div class="node-body">
                <div class="cutoff-diagram">
                  <div class="cutoff-item in-prompt">
                    <span>Rank #1 - #2</span>
                    <span>IN PROMPT</span>
                  </div>
                  <div class="cutoff-divider">
                    <span class="cutoff-label">✂️ PROMPT WINDOW CUTOFF</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Stage 4 -->
            <div class="{s4_class}">
              {"<!-- Breakpoint -->" + breakpoint_html if "generation" in diag else ""}
              <div class="node-header">
                <span class="node-number">04 / GENERATION</span>
                {s4_stat}
              </div>
              <div class="node-title">LLM Fact Grounding</div>
              <div class="node-body">
                <p style="font-size:10px; color:var(--text-muted);">Model Output:</p>
                <p style="font-weight:600; color:#fff; font-size:11px;">"{answer}"</p>
              </div>
            </div>

          </div>
        </div>

        <!-- Forensic Evidence & Actionable Ticket -->
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
                <span class="ticket-chip">{team_chip}</span>
                <span style="font-size:10px; color:var(--text-muted); font-family:var(--font-mono);">Severity: High</span>
              </div>
              <div class="ticket-body">
                <strong>Remediation:</strong> {next_check}
              </div>
            </div>
            <div class="limitation-text">
              {cannot_tell_you}
            </div>
          </div>
        </div>

      </article>
    """
