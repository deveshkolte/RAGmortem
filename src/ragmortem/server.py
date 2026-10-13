"""Zero-dependency local HTTP server and live showcase UI for RAGmortem.

Uses Python standard library (http.server, json, urllib) to provide
an interactive executive enterprise dashboard for RAG failure triage.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VERSION = "0.2.0.dev0"
from ragmortem.api import diagnose
from ragmortem.demo_data import DEMO_CASES
from ragmortem.cli import DIAGNOSIS_PLAIN_ENGLISH, get_suggested_next_check, get_what_this_cannot_tell_you, _normalize_chunks


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAGmortem • Executive Enterprise Showcase</title>
  <style>
    :root {
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
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-family);
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.5;
      padding: 32px 24px;
    }

    .container {
      max-width: 1240px;
      margin: 0 auto;
    }

    /* Top Nav */
    .header {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .brand-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-badge {
      background: #0f172a;
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 4px 8px;
      border-radius: 6px;
    }
    h1 {
      font-size: 20px;
      font-weight: 700;
      color: #0f172a;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn {
      background: #0f172a;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background: #1e293b; }
    .btn-secondary {
      background: #f1f5f9;
      color: #334155;
      border: 1px solid var(--border-color);
    }
    .btn-secondary:hover { background: #e2e8f0; }

    /* Layout Split */
    .main-grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 1000px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    /* Left Control Sidebar */
    .sidebar-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .section-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 12px;
    }
    .preset-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 20px;
    }
    .preset-btn {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .preset-btn:hover {
      border-color: #cbd5e1;
      background: #f1f5f9;
    }
    .preset-btn.active {
      border-color: var(--brand-primary);
      background: var(--brand-primary-light);
    }
    .preset-title {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      display: flex;
      justify-content: space-between;
    }
    .preset-tag {
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }

    /* Form Fields */
    .form-group {
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      margin-bottom: 4px;
    }
    input[type="text"], textarea, select {
      width: 100%;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 13px;
      font-family: inherit;
      color: #0f172a;
      background: #fff;
    }
    input[type="text"]:focus, textarea:focus, select:focus {
      outline: none;
      border-color: var(--brand-primary);
      box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
    }
    textarea {
      resize: vertical;
      font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }

    /* Right Forensic View */
    .results-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
      overflow: hidden;
    }
    .verdict-header {
      padding: 24px 28px;
      background: #fafafa;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      font-weight: 700;
      padding: 6px 12px;
      border-radius: 9999px;
      text-transform: uppercase;
    }
    .badge-generation { background: var(--color-generation-bg); color: var(--color-generation); border: 1px solid #fecaca; }
    .badge-ranking { background: var(--color-ranking-bg); color: var(--color-ranking); border: 1px solid #fde68a; }
    .badge-retrieval { background: var(--color-retrieval-bg); color: var(--color-retrieval); border: 1px solid #ddd6fe; }
    .badge-healthy { background: var(--color-healthy-bg); color: var(--color-healthy); border: 1px solid #a7f3d0; }
    .badge-unknown { background: var(--color-unknown-bg); color: var(--color-unknown); border: 1px solid #cbd5e1; }

    /* 4-Stage Visual Pipeline */
    .pipeline-view {
      padding: 28px;
      border-bottom: 1px solid var(--border-color);
    }
    .pipeline-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    @media (max-width: 900px) {
      .pipeline-grid { grid-template-columns: 1fr; }
    }
    .step-box {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      position: relative;
    }
    .step-box.step-fail { border-color: #f87171; background: #fff5f5; }
    .step-box.step-ok { border-color: #34d399; background: #f0fdf4; }
    .step-box.step-warn { border-color: #fbbf24; background: #fffdf5; }
    .step-title {
      font-size: 11px;
      font-weight: 800;
      color: #64748b;
      text-transform: uppercase;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }

    /* Ticket Box */
    .details-split {
      padding: 24px 28px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 800px) {
      .details-split { grid-template-columns: 1fr; }
    }
    .evidence-pane {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px;
    }
    .ticket-pane {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 18px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <div class="brand-title">
        <span class="brand-badge">RAGmortem Live</span>
        <h1>Executive RAG Failure Diagnostic Console</h1>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" onclick="window.print()">🖨️ Print / Save PDF</button>
        <button class="btn" onclick="runDiagnostics()">⚡ Run Autopsy</button>
      </div>
    </header>

    <div class="main-grid">
      <!-- Left Panel -->
      <aside class="sidebar-card">
        <div class="section-title">Pre-Loaded Client Scenarios</div>
        <div class="preset-list" id="preset-container">
          <!-- Populated via JS -->
        </div>

        <div class="section-title">Custom Trace Inspector</div>
        <div class="form-group">
          <label>Question / Query</label>
          <input type="text" id="inp-query" placeholder="User query asked to RAG pipeline">
        </div>
        <div class="form-group">
          <label>Generated Answer</label>
          <textarea id="inp-answer" rows="3" placeholder="Answer produced by LLM"></textarea>
        </div>
        <div class="form-group">
          <label>Ground Truth Reference (Optional)</label>
          <input type="text" id="inp-ref" placeholder="Optional expected answer">
        </div>
        <div class="form-group">
          <label>Score Type Scale</label>
          <select id="inp-score-type">
            <option value="similarity">Cosine Similarity [-1.0, 1.0]</option>
            <option value="cosine_distance">Cosine Distance (1 - d)</option>
            <option value="distance">Unknown Distance (Safety Guard)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Retrieved Chunks & Scores (JSON)</label>
          <textarea id="inp-chunks" rows="5"></textarea>
        </div>
        <button class="btn" style="width: 100%; justify-content: center;" onclick="runDiagnostics()">
          ⚡ Run Root-Cause Diagnosis
        </button>
      </aside>

      <!-- Right Panel -->
      <main class="results-card">
        <div class="verdict-header">
          <div>
            <span style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">
              Root-Cause Attribution Verdict
            </span>
            <div style="margin-top: 4px;" id="verdict-badge-container">
              <span class="badge badge-generation">🔴 Evaluating...</span>
            </div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 12px; color: var(--text-muted);">Attribution Confidence: <strong id="val-conf">100%</strong></div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Mode: <span id="val-mode">reference-assisted</span></div>
          </div>
        </div>

        <!-- 4-Stage Visual Pipeline -->
        <div class="pipeline-view">
          <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 16px;">
            Interactive 4-Stage Forensic Pipeline
          </div>
          <div class="pipeline-grid">
            
            <div class="step-box step-ok" id="pipe-step1">
              <div class="step-title">
                <span>Stage 1: User Query</span>
                <span style="color: #166534;">SENT</span>
              </div>
              <div id="pipe-query-text" style="font-size: 12px; font-weight: 600;">...</div>
            </div>

            <div class="step-box" id="pipe-step2">
              <div class="step-title">
                <span>Stage 2: Vector Search</span>
                <span id="pipe-step2-status">EVAL</span>
              </div>
              <div style="font-size: 12px;">Top score: <strong id="pipe-top-score">-</strong></div>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;" id="pipe-chunk-count">-</div>
            </div>

            <div class="step-box" id="pipe-step3">
              <div class="step-title">
                <span>Stage 3: Top-K Cutoff</span>
                <span id="pipe-step3-status">EVAL</span>
              </div>
              <div style="font-size: 11px; color: #b45309; background: #fef3c7; border: 1px dashed #f59e0b; padding: 4px 6px; border-radius: 4px; text-align: center; margin-top: 4px;">
                ✂️ Context Truncation Line
              </div>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;" id="pipe-cutoff-text">Window: Evaluated</div>
            </div>

            <div class="step-box" id="pipe-step4">
              <div class="step-title">
                <span>Stage 4: LLM Generator</span>
                <span id="pipe-step4-status">EVAL</span>
              </div>
              <div style="font-size: 12px;" id="pipe-ans-text">...</div>
            </div>

          </div>
        </div>

        <!-- Evidence & Actionable Ticket -->
        <div class="details-split">
          <div class="evidence-pane">
            <div style="font-size: 12px; font-weight: 700; color: #334155; text-transform: uppercase; margin-bottom: 10px;">
              🔍 Forensic Evidence Points
            </div>
            <ul id="evidence-list" style="padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.6;">
              <li>Evidence loading...</li>
            </ul>
          </div>

          <div class="ticket-pane">
            <div style="font-size: 12px; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 8px;">
              📋 Actionable Engineering Ticket
            </div>
            <div id="ticket-action" style="font-size: 13px; color: #1e3a8a; line-height: 1.5; margin-bottom: 12px;">
              Loading recommendation...
            </div>
            <div id="ticket-limitation" style="font-size: 11px; color: #64748b; border-top: 1px solid #bfdbfe; padding-top: 8px; font-style: italic;">
              Attribution limitations will appear here.
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <script>
    let demoCases = [];

    async function init() {
      try {
        const res = await fetch('/api/demo');
        demoCases = await res.json();
        renderPresets();
        if (demoCases.length > 0) {
          loadCase(0);
        }
      } catch (err) {
        console.error("Failed to load demo cases", err);
      }
    }

    function renderPresets() {
      const c = document.getElementById('preset-container');
      c.innerHTML = demoCases.map((cs, idx) => `
        <div class="preset-btn ${idx === 0 ? 'active' : ''}" onclick="loadCase(${idx})">
          <div class="preset-title">
            <span>Case ${idx + 1}</span>
            <span class="preset-tag" style="background:#e2e8f0;">${cs.id.replace('demo_', '')}</span>
          </div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
            ${cs.question}
          </div>
        </div>
      `).join('');
    }

    function loadCase(idx) {
      document.querySelectorAll('.preset-btn').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
      });
      const cs = demoCases[idx];
      document.getElementById('inp-query').value = cs.question || '';
      document.getElementById('inp-answer').value = cs.answer || '';
      document.getElementById('inp-ref').value = cs.reference_answer || '';
      document.getElementById('inp-score-type').value = cs.score_type || 'similarity';
      document.getElementById('inp-chunks').value = JSON.stringify(cs.retrieved || [], null, 2);
      runDiagnostics();
    }

    async function runDiagnostics() {
      const query = document.getElementById('inp-query').value;
      const answer = document.getElementById('inp-answer').value;
      const ref = document.getElementById('inp-ref').value;
      const scoreType = document.getElementById('inp-score-type').value;
      let chunks = [];
      try {
        chunks = JSON.parse(document.getElementById('inp-chunks').value || '[]');
      } catch (e) {
        alert("Invalid JSON in chunks field: " + e.message);
        return;
      }

      const payload = {
        question: query,
        answer: answer,
        reference_answer: ref || null,
        score_type: scoreType,
        retrieved: chunks
      };

      try {
        const resp = await fetch('/api/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const res = await resp.json();
        renderResult(res, payload);
      } catch (err) {
        alert("Diagnosis request failed: " + err);
      }
    }

    function renderResult(res, input) {
      const diag = (res.diagnosis || 'unknown').toLowerCase();
      let badgeClass = 'badge-unknown';
      let badgeIcon = '⚪';
      let s2 = 'step-ok', s3 = 'step-ok', s4 = 'step-ok';
      let s2Stat = 'PASS', s3Stat = 'PASS', s4Stat = 'PASS';

      if (diag.includes('generation')) {
        badgeClass = 'badge-generation';
        badgeIcon = '🔴';
        s4 = 'step-fail'; s4Stat = 'FAIL';
      } else if (diag.includes('ranking')) {
        badgeClass = 'badge-ranking';
        badgeIcon = '🟠';
        s3 = 'step-fail'; s3Stat = 'CUTOFF';
      } else if (diag.includes('retrieval')) {
        badgeClass = 'badge-retrieval';
        badgeIcon = '🟣';
        s2 = 'step-fail'; s2Stat = 'MISS';
      } else if (diag.includes('no_failure') || diag.includes('abstain')) {
        badgeClass = 'badge-healthy';
        badgeIcon = '🟢';
      } else {
        s2 = 'step-warn'; s3 = 'step-warn'; s4 = 'step-warn';
        s2Stat = 'UNKNOWN'; s3Stat = 'UNKNOWN'; s4Stat = 'UNKNOWN';
      }

      document.getElementById('verdict-badge-container').innerHTML = `
        <span class="badge ${badgeClass}">${badgeIcon} ${res.diagnosis_english}</span>
      `;
      document.getElementById('val-conf').innerText = Math.round((res.confidence || 0) * 100) + '%';
      document.getElementById('val-mode').innerText = res.mode || 'trace-only';

      // Pipeline Stages
      document.getElementById('pipe-query-text').innerText = '"' + (input.question || '') + '"';
      
      const chunks = input.retrieved || [];
      const topScore = chunks.length > 0 && chunks[0].score != null ? chunks[0].score.toFixed(2) : 'N/A';
      document.getElementById('pipe-top-score').innerText = topScore;
      document.getElementById('pipe-chunk-count').innerText = chunks.length + ' chunk(s) evaluated';

      document.getElementById('pipe-step2').className = 'step-box ' + s2;
      document.getElementById('pipe-step2-status').innerText = s2Stat;

      document.getElementById('pipe-step3').className = 'step-box ' + s3;
      document.getElementById('pipe-step3-status').innerText = s3Stat;

      document.getElementById('pipe-step4').className = 'step-box ' + s4;
      document.getElementById('pipe-step4-status').innerText = s4Stat;
      document.getElementById('pipe-ans-text').innerText = '"' + (input.answer || '') + '"';

      // Evidence
      const evList = document.getElementById('evidence-list');
      evList.innerHTML = (res.evidence || []).map(e => `<li>${e}</li>`).join('');

      // Ticket
      document.getElementById('ticket-action').innerHTML = `<strong>Action Required:</strong> ${res.suggested_next_check}`;
      document.getElementById('ticket-limitation').innerText = res.what_this_cannot_tell_you || '';
    }

    init();
  </script>
</body>
</html>
"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for local RAGmortem interactive console."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP request access logging for cleaner terminal."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        if path == "/api/demo":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(DEMO_CASES).encode("utf-8"))
            return

        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "version": VERSION}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST diagnostic API requests."""
        parsed = urlparse(self.path)
        if parsed.path != "/api/diagnose":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as err:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {err}"}).encode("utf-8"))
            return

        question = data.get("question") or data.get("query", "")
        answer = data.get("answer") or data.get("generated_answer", "")
        ref_answer = data.get("reference_answer")
        score_type = data.get("score_type", "similarity")
        ret_chunks = _normalize_chunks(data.get("retrieved", []))
        cand_chunks = _normalize_chunks(data.get("candidates")) if data.get("candidates") is not None else None
        top_k = data.get("top_k")

        try:
            diag_res = diagnose(
                query=str(question),
                retrieved_chunks=ret_chunks,
                answer=str(answer),
                candidate_chunks=cand_chunks,
                top_k=top_k,
                reference_answer=ref_answer,
                score_type=score_type,
            )
            diag_val = diag_res.diagnosis.value
            diag_english = DIAGNOSIS_PLAIN_ENGLISH.get(diag_val.lower(), diag_val.upper())
            confidence = round(diag_res.confidence if diag_res.confidence is not None else diag_res.evidence_score, 2)
            cannot_tell_you = get_what_this_cannot_tell_you(diag_res.mode, diag_res.limitations)
            next_check = get_suggested_next_check(diag_val)

            payload = {
                "diagnosis": diag_val,
                "diagnosis_english": diag_english,
                "mode": diag_res.mode.replace("_", "-"),
                "confidence": confidence,
                "evidence": diag_res.evidence,
                "what_this_cannot_tell_you": cannot_tell_you,
                "suggested_next_check": next_check,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))


def run_server(port: int = 8080, open_browser: bool = True) -> int:
    """Run lightweight local HTTP server for RAGmortem dashboard."""
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, DashboardHTTPHandler)
    except OSError as err:
        print(f"Error: Could not bind to port {port}: {err}", file=sys.stderr)
        return 1

    url = f"http://localhost:{port}"
    print("=" * 70)
    print("RAGmortem Executive Dashboard Server Running")
    print(f"URL: {url}")
    print("Zero runtime dependencies • Air-gapped local execution")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RAGmortem server.")
        httpd.server_close()
        return 0
    return 0
