"""Enterprise Diagnostic Report Generator for RAGmortem.

Produces self-contained, air-gapped HTML audit reports with an enterprise
telemetry interface (inspired by Datadog, Linear, and GitHub Enterprise).
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
    """Generate a self-contained, enterprise-grade HTML diagnostic report."""
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
  <style>
    :root {{
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Monaco, Consolas, monospace;

      --bg-canvas: #090B0E;
      --bg-panel: #11141A;
      --bg-subtle: #161B22;
      --bg-hover: #1C212B;

      --border-default: #262D38;
      --border-muted: #1D232C;
      --border-emphasis: #384252;

      --text-bright: #F0F3F6;
      --text-primary: #C9D1D9;
      --text-secondary: #8B949E;
      --text-muted: #656D76;

      --status-fail-text: #F85149;
      --status-fail-bg: rgba(248, 81, 73, 0.08);
      --status-fail-border: rgba(248, 81, 73, 0.28);

      --status-warn-text: #D29922;
      --status-warn-bg: rgba(210, 153, 34, 0.08);
      --status-warn-border: rgba(210, 153, 34, 0.28);

      --status-pass-text: #3FB950;
      --status-pass-bg: rgba(63, 185, 80, 0.08);
      --status-pass-border: rgba(63, 185, 80, 0.28);

      --status-neutral-text: #8B949E;
      --status-neutral-bg: rgba(139, 148, 158, 0.08);
      --status-neutral-border: rgba(139, 148, 158, 0.28);

      --radius-sm: 4px;
      --radius-md: 6px;
      --radius-lg: 8px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-canvas);
      color: var(--text-primary);
      font-size: 13px;
      line-height: 1.5;
      min-height: 100vh;
      padding: 32px 40px 64px;
      -webkit-font-smoothing: antialiased;
    }}

    .container {{
      max-width: 1380px;
      margin: 0 auto;
    }}

    /* Top Header */
    .app-header {{
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }}
    .header-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-badge {{
      background: var(--border-emphasis);
      color: var(--text-bright);
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 2px 7px;
      border-radius: var(--radius-sm);
    }}
    h1 {{
      font-size: 16px;
      font-weight: 700;
      color: var(--text-bright);
      letter-spacing: -0.01em;
    }}
    .header-meta {{
      text-align: right;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-secondary);
    }}
    .print-btn {{
      margin-top: 6px;
      background: var(--bg-subtle);
      border: 1px solid var(--border-default);
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      font-size: 11px;
      font-weight: 600;
      color: var(--text-bright);
      cursor: pointer;
      font-family: var(--font-sans);
    }}
    .print-btn:hover {{
      background: var(--bg-hover);
    }}

    /* Notice Banner */
    .demo-notice {{
      background: var(--status-warn-bg);
      border: 1px solid var(--status-warn-border);
      border-radius: var(--radius-md);
      padding: 10px 16px;
      margin-bottom: 20px;
      font-size: 12px;
      color: var(--status-warn-text);
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    /* KPI Summary Row */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-block {{
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      padding: 16px 18px;
    }}
    .kpi-label {{
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .kpi-val {{
      font-size: 26px;
      font-weight: 800;
      color: var(--text-bright);
      font-family: var(--font-mono);
      display: flex;
      align-items: baseline;
      gap: 8px;
    }}
    .kpi-meta {{
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 4px;
    }}

    /* Filter Bar */
    .toolbar {{
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      padding: 10px 16px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .filter-tabs {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .filter-tab {{
      background: transparent;
      border: 1px solid var(--border-default);
      border-radius: var(--radius-sm);
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-secondary);
      cursor: pointer;
    }}
    .filter-tab.active, .filter-tab:hover {{
      background: var(--bg-hover);
      color: var(--text-bright);
      border-color: var(--border-emphasis);
    }}

    /* Trace Audit Card */
    .trace-card {{
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      margin-bottom: 24px;
      overflow: hidden;
    }}
    .card-top {{
      padding: 14px 20px;
      border-bottom: 1px solid var(--border-default);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-subtle);
      flex-wrap: wrap;
      gap: 12px;
    }}
    .card-id {{
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text-bright);
      background: var(--bg-canvas);
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-default);
    }}
    .badge {{
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      text-transform: uppercase;
    }}
    .badge-fail {{
      background: var(--status-fail-bg);
      color: var(--status-fail-text);
      border: 1px solid var(--status-fail-border);
    }}
    .badge-warn {{
      background: var(--status-warn-bg);
      color: var(--status-warn-text);
      border: 1px solid var(--status-warn-border);
    }}
    .badge-pass {{
      background: var(--status-pass-bg);
      color: var(--status-pass-text);
      border: 1px solid var(--status-pass-border);
    }}
    .badge-neutral {{
      background: var(--status-neutral-bg);
      color: var(--status-neutral-text);
      border: 1px solid var(--status-neutral-border);
    }}

    /* 4-Stage Waterfall Lanes */
    .waterfall-lanes {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-bottom: 1px solid var(--border-default);
    }}
    @media (max-width: 1000px) {{
      .waterfall-lanes {{ grid-template-columns: 1fr; }}
    }}
    .waterfall-box {{
      padding: 14px 16px;
      border-right: 1px solid var(--border-muted);
      background: var(--bg-panel);
      display: flex;
      flex-direction: column;
    }}
    .waterfall-box:last-child {{ border-right: none; }}
    .waterfall-box.lane-defect {{
      background: rgba(248, 81, 73, 0.04);
      border-top: 2px solid var(--status-fail-text);
    }}
    .waterfall-box.lane-warning {{
      background: rgba(210, 153, 34, 0.04);
      border-top: 2px solid var(--status-warn-text);
    }}
    .waterfall-box.lane-healthy {{
      border-top: 2px solid var(--status-pass-text);
    }}

    .box-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    .box-num {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      font-weight: 700;
    }}
    .box-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text-bright);
      margin-bottom: 6px;
    }}
    .box-body {{
      font-size: 11px;
      color: var(--text-secondary);
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    /* Cutoff Visualizer */
    .cutoff-list {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      background: var(--bg-canvas);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 6px;
      margin-top: 4px;
    }}
    .cutoff-chunk {{
      font-family: var(--font-mono);
      font-size: 9px;
      padding: 2px 5px;
      border-radius: var(--radius-sm);
      display: flex;
      justify-content: space-between;
    }}
    .chunk-admitted {{
      background: rgba(63, 185, 80, 0.1);
      color: #3FB950;
      border: 1px solid rgba(63, 185, 80, 0.2);
    }}
    .cutoff-line {{
      border-top: 1px dashed #D29922;
      margin: 4px 0;
      text-align: center;
      position: relative;
    }}
    .cutoff-line-label {{
      font-size: 8px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #D29922;
      background: var(--bg-canvas);
      padding: 0 4px;
      position: relative;
      top: -6px;
    }}
    .chunk-pruned {{
      background: rgba(248, 81, 73, 0.08);
      color: #F85149;
      border: 1px solid rgba(248, 81, 73, 0.18);
    }}

    /* Bottom Split */
    .card-bottom {{
      padding: 16px 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 860px) {{
      .card-bottom {{ grid-template-columns: 1fr; }}
    }}

    .evidence-block {{
      background: var(--bg-subtle);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-md);
      padding: 14px;
    }}
    .block-heading {{
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 10px;
    }}
    .evidence-ul {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .evidence-ul li {{
      font-size: 12px;
      color: var(--text-primary);
      padding-left: 14px;
      position: relative;
      line-height: 1.5;
    }}
    .evidence-ul li::before {{
      content: "•";
      position: absolute;
      left: 2px;
      color: #58A6FF;
      font-weight: bold;
    }}

    .runbook-block {{
      background: var(--bg-subtle);
      border: 1px solid var(--border-muted);
      border-left: 3px solid #58A6FF;
      border-radius: var(--radius-md);
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .runbook-header {{
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      font-family: var(--font-mono);
      margin-bottom: 8px;
    }}
    .runbook-team {{
      color: #58A6FF;
      font-weight: 700;
    }}
    .runbook-text {{
      font-size: 12px;
      color: var(--text-bright);
      line-height: 1.5;
      margin-bottom: 10px;
    }}
    .limitation-text {{
      font-size: 11px;
      color: var(--text-muted);
      border-top: 1px solid var(--border-muted);
      padding-top: 8px;
      font-style: italic;
    }}

    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }}

    @media print {{
      body {{ background: #fff; color: #000; padding: 0; }}
      .app-header, .trace-card, .kpi-block {{ box-shadow: none; border-color: #ccc; background: #fff; color: #000; }}
      .toolbar, .print-btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">

    <!-- Top Header -->
    <header class="app-header">
      <div class="header-brand">
        <h1>{_escape(title)}</h1>
        <span class="logo-badge">RAGmortem v0.2.0.dev0</span>
      </div>
      <div class="header-meta">
        <div>Generated: {now_str}</div>
        <button class="print-btn" onclick="window.print()">Print / Export Document</button>
      </div>
    </header>

    {"<!-- Demo Warning Banner -->" if is_demo else ""}
    {f'''<div class="demo-notice">
      <span>Synthetic Demonstration Data:</span> The traces evaluated in this document are bundled benchmark failure scenarios for validation.
    </div>''' if is_demo else ""}

    <!-- KPI Summary Row -->
    <section class="kpi-row">
      <div class="kpi-block">
        <div class="kpi-label">Total Traces Audited</div>
        <div class="kpi-val">{total_cases}</div>
        <div class="kpi-meta">Resolved: {total_cases - unknown_count} | Inconclusive: {unknown_count}</div>
      </div>
      <div class="kpi-block" style="border-top: 2px solid var(--status-fail-text);">
        <div class="kpi-label">Generation Failures</div>
        <div class="kpi-val" style="color: var(--status-fail-text);">{gen_count}</div>
        <div class="kpi-meta">Target: Prompt Engineering Team</div>
      </div>
      <div class="kpi-block" style="border-top: 2px solid var(--status-warn-text);">
        <div class="kpi-label">Ranking Cutoff Failures</div>
        <div class="kpi-val" style="color: var(--status-warn-text);">{rank_count}</div>
        <div class="kpi-meta">Target: Re-Ranker / Context Window</div>
      </div>
      <div class="kpi-block" style="border-top: 2px solid var(--status-pass-text);">
        <div class="kpi-label">Healthy / Correct Abstain</div>
        <div class="kpi-val" style="color: var(--status-pass-text);">{abstain_count}</div>
        <div class="kpi-meta">No system defect detected</div>
      </div>
      <div class="kpi-block" style="border-top: 2px solid var(--border-emphasis);">
        <div class="kpi-label">UNKNOWN Attribution Rate</div>
        <div class="kpi-val">{unknown_rate}%</div>
        <div class="kpi-meta">Strict honesty bounds applied</div>
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
      <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">
        Deterministic Forensic 4-Stage RAG Pipeline Breakdown
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

      document.querySelectorAll('.trace-card').forEach(card => {{
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
    """Render a single trace card with 4-stage waterfall visualization."""
    case_id = _escape(case.get("id", f"case_{idx}"))
    diag = str(case.get("diagnosis", "unknown")).lower()
    diag_english = _escape(case.get("diagnosis_english", diag))
    mode = _escape(case.get("mode", "trace-only"))
    try:
        confidence = float(case.get("confidence") if case.get("confidence") is not None else 0.0)
    except (ValueError, TypeError):
        confidence = 0.0
    conf_pct = int(confidence * 100)
    evidence = case.get("evidence", [])
    cannot_tell_you = _escape(case.get("what_this_cannot_tell_you", ""))
    next_check = _escape(case.get("suggested_next_check", ""))
    question = _escape(case.get("question", ""))
    answer = _escape(case.get("answer", ""))

    team_id = "TRIAGE_TEAM"
    s1_class = "waterfall-box lane-healthy"
    s2_class = "waterfall-box"
    s3_class = "waterfall-box"
    s4_class = "waterfall-box"
    s2_stat = '<span class="badge badge-pass">PASS</span>'
    s3_stat = '<span class="badge badge-pass">PASS</span>'
    s4_stat = '<span class="badge badge-pass">PASS</span>'

    if "generation" in diag:
        category = "generation"
        badge_class = "badge-fail"
        team_id = "PROMPT_ENGINEERING"
        s4_class = "waterfall-box lane-defect"
        s4_stat = '<span class="badge badge-fail">DEFECT</span>'
    elif "ranking" in diag:
        category = "ranking"
        badge_class = "badge-warn"
        team_id = "RE_RANKER_CONFIG"
        s3_class = "waterfall-box lane-warning"
        s3_stat = '<span class="badge badge-warn">CUTOFF</span>'
    elif "retrieval" in diag:
        category = "retrieval"
        badge_class = "badge-fail"
        team_id = "SEARCH_EMBEDDINGS"
        s2_class = "waterfall-box lane-defect"
        s2_stat = '<span class="badge badge-fail">MISS</span>'
    elif "no_failure" in diag or "abstain" in diag:
        category = "healthy"
        badge_class = "badge-pass"
        team_id = "SYSTEM_HEALTHY"
    else:
        category = "unknown"
        badge_class = "badge-neutral"
        team_id = "TELEMETRY_LOGGING"
        s2_stat = '<span class="badge badge-neutral">UNKNOWN</span>'
        s3_stat = '<span class="badge badge-neutral">UNKNOWN</span>'
        s4_stat = '<span class="badge badge-neutral">UNKNOWN</span>'

    evidence_html = "".join(f"<li>{_escape(ev)}</li>" for ev in evidence)

    return f"""
      <article class="trace-card" data-category="{category}">
        <div class="card-top">
          <div style="display:flex; align-items:center; gap:12px;">
            <span class="card-id">{case_id}</span>
            <span class="badge {badge_class}">{diag_english}</span>
          </div>
          <div style="display:flex; align-items:center; gap:14px; font-size:11px; font-family:var(--font-mono);">
            <span>CONFIDENCE: <strong style="color:var(--text-bright);">{conf_pct}%</strong></span>
            <span style="color:var(--text-muted);">({mode})</span>
          </div>
        </div>

        <!-- 4-Stage Waterfall Breakdown -->
        <div class="waterfall-lanes">
          
          <!-- Stage 1 -->
          <div class="{s1_class}">
            <div class="box-header">
              <span class="box-num">SPAN 01</span>
              <span class="badge badge-pass">VALID</span>
            </div>
            <div class="box-title">Stage 1: User Query</div>
            <div class="box-body">
              <p style="font-size:11px; color:var(--text-bright); font-weight:500;">"{question}"</p>
              <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
                Input length: {len(question.split())} words
              </div>
            </div>
          </div>

          <!-- Stage 2 -->
          <div class="{s2_class}">
            <div class="box-header">
              <span class="box-num">SPAN 02</span>
              {s2_stat}
            </div>
            <div class="box-title">Vector Search (Bi-Encoder)</div>
            <div class="box-body">
              <span style="font-size:11px; color:var(--text-muted);">Bi-encoder scoring candidate chunks against calibrated threshold bounds (0.47 / 0.50 / 0.53).</span>
            </div>
          </div>

          <!-- Stage 3 -->
          <div class="{s3_class}">
            <div class="box-header">
              <span class="box-num">SPAN 03</span>
              {s3_stat}
            </div>
            <div class="box-title">Top-K Context Cutoff</div>
            <div class="box-body">
              <div class="cutoff-list">
                <div class="cutoff-chunk chunk-admitted">
                  <span>Rank #1 - #2</span>
                  <span>IN PROMPT</span>
                </div>
                <div class="cutoff-line">
                  <span class="cutoff-line-label">TOP-K CUTOFF</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Stage 4 -->
          <div class="{s4_class}">
            <div class="box-header">
              <span class="box-num">SPAN 04</span>
              {s4_stat}
            </div>
            <div class="box-title">LLM Fact Grounding</div>
            <div class="box-body">
              <div style="font-size:10px; color:var(--text-muted);">Model Output:</div>
              <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-bright);">"{answer}"</div>
            </div>
          </div>

        </div>

        <!-- Bottom Detail Split -->
        <div class="card-bottom">
          <div class="evidence-block">
            <div class="block-heading">Diagnostic Evidence Trail</div>
            <ul class="evidence-ul">
              {evidence_html}
            </ul>
          </div>

          <div class="runbook-block">
            <div>
              <div class="runbook-header">
                <span class="runbook-team">{team_id}</span>
                <span style="color:var(--text-muted);">SEVERITY: HIGH</span>
              </div>
              <div class="runbook-text">
                {next_check}
              </div>
            </div>
            <div class="limitation-text">
              {cannot_tell_you}
            </div>
          </div>
        </div>

      </article>
    """
