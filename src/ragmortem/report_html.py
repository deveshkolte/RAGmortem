"""Static HTML Report Generator for RAGmortem.

Renders a self-contained, zero-dependency, dark-mode single-file HTML report
mirroring the Rich terminal UI, KPI cards, visual chunk cutoff tables, and summary breakdown.
"""

from __future__ import annotations

import html
from typing import Any


def _get_badge_color(diag: str) -> tuple[str, str, str]:
    """Return (bg_color, text_color, border_color) for a diagnosis badge."""
    d = diag.lower()
    if "retrieval" in d:
        return "#3d1116", "#ff7b72", "#6e2528"
    elif "ranking" in d:
        return "#3a2411", "#ffa657", "#6b4317"
    elif "generation" in d:
        return "#2e1b4e", "#d2a8ff", "#553086"
    elif "abstain" in d or "abstention" in d:
        return "#34280f", "#f2cc60", "#69511a"
    elif "no_failure" in d:
        return "#112a1c", "#56d364", "#1b5430"
    else:  # unknown
        return "#12253a", "#79c0ff", "#1b4b74"


def _render_meter(score: float, width_px: int = 120) -> str:
    """Render a sleek HTML meter bar."""
    pct = max(0, min(100, int(round(score * 100))))
    color = "#58a6ff"
    if pct >= 80:
        color = "#3fb950"
    elif pct >= 50:
        color = "#d29922"
    elif pct > 0:
        color = "#f85149"
    return f"""<div style="display:inline-flex;align-items:center;gap:8px;">
<div style="width:{width_px}px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;border:1px solid #30363d;">
<div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div>
</div>
<span style="font-family:monospace;font-size:12px;color:#8b949e;min-width:35px;">{pct}%</span>
</div>"""


def render_html_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    is_demo: bool = False,
    profile: str = "minilm",
) -> str:
    """Generate a self-contained, dark-mode single-file HTML report."""
    total_cases = summary.get("total_cases", len(results))
    unknown_count = summary.get("unknown_count", 0)
    unknown_rate_pct = summary.get("unknown_rate_pct", 0.0)
    counts: dict[str, int] = summary.get("counts_per_diagnosis", {})

    # Calculate diagnosed failures
    no_failure_count = counts.get("no_failure", 0)
    failure_count = total_cases - no_failure_count

    # Build KPI status color
    unknown_color = "#3fb950" if unknown_rate_pct <= 10.0 else "#f85149"

    # Build taxonomy breakdown table rows
    breakdown_rows = []
    for diag, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct = round((cnt / total_cases) * 100, 1) if total_cases > 0 else 0.0
        bg, fg, border = _get_badge_color(diag)
        meter_pct = int(round(pct))
        breakdown_rows.append(f"""<tr>
<td style="padding:10px 14px;border-bottom:1px solid #21262d;">
  <span style="background:{bg};color:{fg};border:1px solid {border};padding:3px 8px;border-radius:4px;font-weight:600;font-size:12px;font-family:monospace;">
    {html.escape(diag.upper())}
  </span>
</td>
<td style="padding:10px 14px;border-bottom:1px solid #21262d;font-family:monospace;font-weight:bold;color:#f0f6fc;">{cnt}</td>
<td style="padding:10px 14px;border-bottom:1px solid #21262d;font-family:monospace;color:#8b949e;">{pct}%</td>
<td style="padding:10px 14px;border-bottom:1px solid #21262d;">
  <div style="width:160px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;border:1px solid #30363d;">
    <div style="width:{meter_pct}%;height:100%;background:{fg};border-radius:3px;"></div>
  </div>
</td>
</tr>""")

    # Build case cards
    case_cards = []
    for i, c in enumerate(results, 1):
        cid = c.get("id", f"case_{i}")
        q = c.get("question", "")
        a = c.get("answer", "")
        diag = c.get("diagnosis", "unknown")
        diag_eng = c.get("diagnosis_english", diag.upper())
        mode = c.get("mode", "observed")
        conf = c.get("confidence", 0.0)
        evidence = c.get("evidence", [])
        cannot_tell = c.get("what_this_cannot_tell_you", "")
        next_check = c.get("suggested_next_check", "")
        cutoff_info = c.get("cutoff_info")

        bg, fg, border = _get_badge_color(diag)
        meter_html = _render_meter(conf, width_px=140)

        # Build evidence list
        evidence_items = "".join(
            f"""<li style="margin-bottom:6px;line-height:1.5;"><span style="color:#58a6ff;font-family:monospace;">▸</span> {html.escape(ev)}</li>"""
            for ev in evidence
        )

        # Build cutoff section
        cutoff_html = ""
        if cutoff_info:
            cutoff_rows = []
            for item in cutoff_info.get("items", []):
                rnk = item.get("rank", 0)
                sc = item.get("score")
                sc_str = f"[{sc:.3f}]" if sc is not None else "[  -  ]"
                sc_val = sc if sc is not None else 0.0
                bar = _render_meter(max(0.0, min(1.0, sc_val)), width_px=80)
                cid_txt = html.escape(str(item.get("id", "")))
                tag = item.get("tag", "")

                if tag == "in_prompt":
                    cutoff_rows.append(f"""<div style="display:flex;align-items:center;gap:12px;padding:4px 8px;font-family:monospace;font-size:12px;color:#c9d1d9;">
<span style="color:#8b949e;width:24px;">#{rnk}</span>
<span style="color:#79c0ff;width:55px;">{sc_str}</span>
{bar}
<span style="font-weight:600;color:#f0f6fc;">{cid_txt}</span>
<span style="background:#112a1c;color:#56d364;border:1px solid #1b5430;padding:1px 6px;border-radius:3px;font-size:11px;">(In Prompt)</span>
</div>""")
                elif tag == "dropped":
                    gap = item.get("gap")
                    gap_str = f"⚠️ (-{gap:.3f} gap)" if gap is not None else "⚠️ (Dropped)"
                    cutoff_rows.append(f"""<div style="display:flex;align-items:center;gap:12px;padding:4px 8px;font-family:monospace;font-size:12px;background:rgba(210,153,34,0.1);border-radius:4px;">
<span style="color:#e3b341;width:24px;">#{rnk}</span>
<span style="color:#ffa657;width:55px;">{sc_str}</span>
{bar}
<span style="font-weight:600;color:#ffa657;">{cid_txt}</span>
<span style="color:#f85149;font-weight:bold;font-size:11px;">{gap_str}</span>
</div>""")

            k_val = cutoff_info.get("cutoff_k", 2)
            cutoff_items_str = "".join(cutoff_rows)
            cutoff_html = f"""<div style="margin:16px 0;background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;">
<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Candidate Cutoff Inspection</div>
{cutoff_items_str}
<div style="display:flex;align-items:center;gap:8px;margin:8px 0;color:#f85149;font-family:monospace;font-size:12px;">
<div style="flex-grow:1;border-bottom:1px dashed #f85149;"></div>
<span>✂️ TOP-K CUTOFF (k={k_val})</span>
<div style="flex-grow:1;border-bottom:1px dashed #f85149;"></div>
</div>
</div>"""

        case_cards.append(f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:20px;">
  <!-- Card Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #21262d;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-family:monospace;font-weight:bold;font-size:16px;color:#f0f6fc;">{html.escape(cid)}</span>
      <span style="background:#21262d;color:#8b949e;border:1px solid #30363d;padding:2px 8px;border-radius:4px;font-size:11px;font-family:monospace;">mode: {html.escape(mode)}</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="background:{bg};color:{fg};border:1px solid {border};padding:4px 10px;border-radius:6px;font-weight:bold;font-size:13px;font-family:monospace;">
        {html.escape(diag.upper())}
      </span>
      <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#8b949e;">
        <span>Evidence:</span>
        {meter_html}
      </div>
    </div>
  </div>

  <!-- Query & Answer -->
  <div style="margin-bottom:14px;">
    <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Question</div>
    <div style="font-size:14px;color:#f0f6fc;background:#0d1117;padding:10px 12px;border-radius:6px;border:1px solid #21262d;line-height:1.4;">
      {html.escape(q)}
    </div>
  </div>
  <div style="margin-bottom:16px;">
    <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Generated Answer</div>
    <div style="font-size:14px;color:#c9d1d9;background:#0d1117;padding:10px 12px;border-radius:6px;border:1px solid #21262d;line-height:1.4;">
      {html.escape(a)}
    </div>
  </div>

  <!-- Visual Top-K Cutoff (if present) -->
  {cutoff_html}

  <!-- Evidence Section -->
  <div style="margin-bottom:16px;">
    <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Diagnostic Evidence</div>
    <ul style="list-style:none;padding:0;margin:0;font-size:13px;color:#c9d1d9;">
      {evidence_items}
    </ul>
  </div>

  <!-- Callouts -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
    <div style="background:#0d1117;border-left:3px solid #79c0ff;padding:10px 12px;border-radius:0 6px 6px 0;font-size:12px;color:#8b949e;line-height:1.4;">
      <strong style="color:#79c0ff;">What this cannot tell you:</strong><br>{html.escape(cannot_tell)}
    </div>
    <div style="background:#0d1117;border-left:3px solid #e3b341;padding:10px 12px;border-radius:0 6px 6px 0;font-size:12px;color:#8b949e;line-height:1.4;">
      <strong style="color:#e3b341;">Suggested next check:</strong><br>{html.escape(next_check)}
    </div>
  </div>
</div>""")

    demo_banner = ""
    if is_demo:
        demo_banner = """<div style="background:#3a2411;color:#ffa657;border:1px solid #6b4317;padding:10px 16px;border-radius:6px;margin-bottom:20px;font-size:13px;display:flex;align-items:center;gap:8px;">
<strong>SYNTHETIC DEMO DATA:</strong> All cases below are synthetic benchmark examples bundled with RAGmortem for demonstration purposes.
</div>"""

    cases_html = "\n".join(case_cards)
    breakdown_html = "\n".join(breakdown_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAGmortem Diagnostic Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 20px;
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1040px;
    margin: 0 auto;
  }}
  a {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;margin-bottom:24px;border-bottom:1px solid #21262d;padding-bottom:20px;">
    <div>
      <div style="display:flex;align-items:center;gap:12px;">
        <h1 style="margin:0;font-size:24px;font-weight:700;color:#f0f6fc;letter-spacing:-0.5px;">🔬 RAGmortem Diagnostic Report</h1>
        <span style="background:#21262d;color:#58a6ff;border:1px solid #30363d;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;">v0.2.0</span>
      </div>
      <p style="margin:6px 0 0 0;font-size:14px;color:#8b949e;">Local root-cause failure debugger for Retrieval-Augmented Generation</p>
    </div>
    <div style="text-align:right;">
      <span style="background:#161b22;color:#8b949e;border:1px solid #30363d;padding:4px 10px;border-radius:6px;font-size:12px;font-family:monospace;">
        Profile: <strong style="color:#58a6ff;">{html.escape(profile)}</strong>
      </span>
    </div>
  </div>

  {demo_banner}

  <!-- KPI Cards -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:16px;margin-bottom:24px;">
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;">
      <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">Total Traces Audited</div>
      <div style="font-size:28px;font-weight:700;color:#f0f6fc;margin-top:4px;font-family:monospace;">{total_cases}</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;">
      <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">UNKNOWN Attribution Rate</div>
      <div style="font-size:28px;font-weight:700;color:{unknown_color};margin-top:4px;font-family:monospace;">
        {unknown_rate_pct}% <span style="font-size:13px;font-weight:normal;color:#8b949e;">({unknown_count}/{total_cases})</span>
      </div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;">
      <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">Diagnosed Failures</div>
      <div style="font-size:28px;font-weight:700;color:#ffa657;margin-top:4px;font-family:monospace;">{failure_count}</div>
    </div>
  </div>

  <!-- Failure Taxonomy Breakdown -->
  <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:32px;">
    <div style="padding:14px 18px;border-bottom:1px solid #30363d;font-weight:600;font-size:14px;color:#f0f6fc;">
      Diagnostic Distribution Summary
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">
      <thead>
        <tr style="background:#0d1117;color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">
          <th style="padding:8px 14px;border-bottom:1px solid #21262d;">Diagnostic Category</th>
          <th style="padding:8px 14px;border-bottom:1px solid #21262d;">Count</th>
          <th style="padding:8px 14px;border-bottom:1px solid #21262d;">Rate</th>
          <th style="padding:8px 14px;border-bottom:1px solid #21262d;">Distribution</th>
        </tr>
      </thead>
      <tbody>
        {breakdown_html}
      </tbody>
    </table>
  </div>

  <!-- Case Diagnostics Section -->
  <h2 style="font-size:18px;font-weight:700;color:#f0f6fc;margin-bottom:16px;">Trace Evaluations</h2>
  {cases_html}

  <!-- Footer -->
  <div style="text-align:center;font-size:12px;color:#8b949e;margin-top:40px;padding-top:20px;border-top:1px solid #21262d;">
    Report generated by <strong style="color:#f0f6fc;">RAGmortem</strong> •
    <a href="https://github.com/deveshkolte/RAGmortem" target="_blank" rel="noreferrer">GitHub Repository</a>
  </div>

</div>
</body>
</html>
"""
