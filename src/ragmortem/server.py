"""Zero-dependency local HTTP server and live showcase UI for RAGmortem.

Engineered with an enterprise-grade dark telemetry interface (inspired by
Datadog, Linear, and GitHub Enterprise) for deterministic RAG failure triage.
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


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAGmortem • Telemetry & Failure Diagnostics</title>
  <style>
    :root {
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, "SF Mono", "Cascadia Mono", "Segoe UI Mono", Menlo, Monaco, Consolas, monospace;

      /* Enterprise Palette: Deep Charcoal & Surgical Neutrals */
      --bg-canvas: #090B0E;
      --bg-panel: #11141A;
      --bg-subtle: #161B22;
      --bg-hover: #1C212B;
      --bg-active: #222936;

      --border-default: #262D38;
      --border-muted: #1D232C;
      --border-emphasis: #384252;

      --text-bright: #F0F3F6;
      --text-primary: #C9D1D9;
      --text-secondary: #8B949E;
      --text-muted: #656D76;

      /* Muted Functional Status Hues (No Neon) */
      --status-fail-text: #F85149;
      --status-fail-bg: rgba(248, 81, 73, 0.08);
      --status-fail-border: rgba(248, 81, 73, 0.28);

      --status-warn-text: #D29922;
      --status-warn-bg: rgba(210, 153, 34, 0.08);
      --status-warn-border: rgba(210, 153, 34, 0.28);

      --status-pass-text: #3FB950;
      --status-pass-bg: rgba(63, 185, 80, 0.08);
      --status-pass-border: rgba(63, 185, 80, 0.28);

      --status-info-text: #58A6FF;
      --status-info-bg: rgba(88, 166, 255, 0.08);
      --status-info-border: rgba(88, 166, 255, 0.28);

      --status-neutral-text: #8B949E;
      --status-neutral-bg: rgba(139, 148, 158, 0.08);
      --status-neutral-border: rgba(139, 148, 158, 0.28);

      --radius-sm: 4px;
      --radius-md: 6px;
      --radius-lg: 8px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-canvas);
      color: var(--text-primary);
      font-size: 13px;
      line-height: 1.5;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Application Bar */
    .app-header {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border-default);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .app-logo {
      font-weight: 700;
      font-size: 14px;
      color: var(--text-bright);
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .logo-badge {
      background: var(--border-emphasis);
      color: var(--text-bright);
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 1px 6px;
      border-radius: var(--radius-sm);
    }
    .header-divider {
      width: 1px;
      height: 16px;
      background: var(--border-default);
    }
    .header-breadcrumb {
      font-size: 12px;
      color: var(--text-secondary);
      font-family: var(--font-mono);
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .btn {
      font-family: var(--font-sans);
      font-size: 12px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-default);
      background: var(--bg-subtle);
      color: var(--text-bright);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease, border-color 0.15s ease;
    }
    .btn:hover {
      background: var(--bg-hover);
      border-color: var(--border-emphasis);
    }
    .btn-primary {
      background: #238636;
      border-color: rgba(240, 246, 252, 0.1);
      color: #FFFFFF;
    }
    .btn-primary:hover {
      background: #2EA043;
      border-color: rgba(240, 246, 252, 0.2);
    }

    /* Layout Frame */
    .app-layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: calc(100vh - 49px);
    }
    @media (max-width: 1080px) {
      .app-layout { grid-template-columns: 1fr; }
    }

    /* Left Sidebar */
    .sidebar {
      background: var(--bg-panel);
      border-right: 1px solid var(--border-default);
      display: flex;
      flex-direction: column;
    }
    .sidebar-section {
      padding: 16px;
      border-bottom: 1px solid var(--border-muted);
    }
    .sidebar-title {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
    }

    /* Case Navigator List */
    .case-nav-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .case-nav-item {
      padding: 8px 10px;
      border-radius: var(--radius-md);
      border: 1px solid transparent;
      background: transparent;
      cursor: pointer;
      text-align: left;
      transition: background 0.1s ease, border-color 0.1s ease;
    }
    .case-nav-item:hover {
      background: var(--bg-hover);
    }
    .case-nav-item.active {
      background: var(--bg-active);
      border-color: var(--border-emphasis);
    }
    .case-nav-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2px;
    }
    .case-nav-name {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-bright);
    }
    .badge {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: var(--radius-sm);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-fail {
      background: var(--status-fail-bg);
      color: var(--status-fail-text);
      border: 1px solid var(--status-fail-border);
    }
    .badge-warn {
      background: var(--status-warn-bg);
      color: var(--status-warn-text);
      border: 1px solid var(--status-warn-border);
    }
    .badge-pass {
      background: var(--status-pass-bg);
      color: var(--status-pass-text);
      border: 1px solid var(--status-pass-border);
    }
    .badge-neutral {
      background: var(--status-neutral-bg);
      color: var(--status-neutral-text);
      border: 1px solid var(--status-neutral-border);
    }
    .case-nav-query {
      font-size: 11px;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Form Fields */
    .field-group {
      margin-bottom: 12px;
    }
    .field-label {
      display: block;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }
    .input-text, .input-textarea, .input-select {
      width: 100%;
      background: var(--bg-canvas);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      font-size: 12px;
      color: var(--text-bright);
      font-family: var(--font-sans);
    }
    .input-textarea {
      font-family: var(--font-mono);
      font-size: 11px;
      resize: vertical;
      line-height: 1.4;
    }
    .input-text:focus, .input-textarea:focus, .input-select:focus {
      outline: none;
      border-color: #58A6FF;
    }

    /* Main Diagnostic Workspace */
    .workspace {
      padding: 24px;
      background: var(--bg-canvas);
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Verdict Card */
    .verdict-panel {
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
    }
    .verdict-top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .verdict-title-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .verdict-main-badge {
      font-size: 13px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      letter-spacing: -0.01em;
    }
    .verdict-meta-items {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .meta-item {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }
    .meta-val {
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 14px;
      color: var(--text-bright);
    }
    .meta-lbl {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .verdict-summary-text {
      font-size: 13px;
      color: var(--text-primary);
      padding-top: 10px;
      border-top: 1px solid var(--border-muted);
      line-height: 1.5;
    }

    /* Telemetry Trace Waterfall / DAG */
    .trace-section {
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      overflow: hidden;
    }
    .section-header {
      padding: 12px 18px;
      background: var(--bg-subtle);
      border-bottom: 1px solid var(--border-default);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .section-title-text {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-bright);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .pipeline-lanes {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-bottom: 1px solid var(--border-default);
    }
    @media (max-width: 1100px) {
      .pipeline-lanes { grid-template-columns: 1fr; }
    }

    .lane-box {
      padding: 16px;
      border-right: 1px solid var(--border-muted);
      background: var(--bg-panel);
      display: flex;
      flex-direction: column;
      position: relative;
    }
    .lane-box:last-child {
      border-right: none;
    }
    .lane-box.lane-defect {
      background: rgba(248, 81, 73, 0.04);
      border-top: 2px solid var(--status-fail-text);
    }
    .lane-box.lane-warning {
      background: rgba(210, 153, 34, 0.04);
      border-top: 2px solid var(--status-warn-text);
    }
    .lane-box.lane-healthy {
      border-top: 2px solid var(--status-pass-text);
    }

    .lane-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .lane-step {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      font-weight: 700;
    }
    .lane-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-bright);
      margin-bottom: 8px;
    }
    .lane-body {
      font-size: 12px;
      color: var(--text-secondary);
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    /* Metric Visualization Controls */
    .metric-row {
      background: var(--bg-canvas);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      margin-top: 4px;
    }
    .metric-row-header {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .metric-progress {
      height: 4px;
      background: var(--bg-subtle);
      border-radius: 2px;
      overflow: hidden;
    }
    .metric-progress-bar {
      height: 100%;
      background: #58A6FF;
    }
    .metric-ticks {
      display: flex;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 9px;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Top-K Cutoff Representation */
    .cutoff-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
      background: var(--bg-canvas);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 6px;
      margin-top: 4px;
    }
    .cutoff-chunk {
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 6px;
      border-radius: var(--radius-sm);
      display: flex;
      justify-content: space-between;
    }
    .chunk-admitted {
      background: rgba(63, 185, 80, 0.1);
      color: #3FB950;
      border: 1px solid rgba(63, 185, 80, 0.2);
    }
    .cutoff-line {
      border-top: 1px dashed #D29922;
      margin: 4px 0;
      text-align: center;
      position: relative;
    }
    .cutoff-line-label {
      font-size: 8px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #D29922;
      background: var(--bg-canvas);
      padding: 0 4px;
      position: relative;
      top: -6px;
    }
    .chunk-pruned {
      background: rgba(248, 81, 73, 0.08);
      color: #F85149;
      border: 1px solid rgba(248, 81, 73, 0.18);
    }

    /* Stage 4 Diff Box */
    .diff-container {
      background: var(--bg-canvas);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .diff-item-label {
      font-size: 9px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .diff-expected { color: #3FB950; font-family: var(--font-mono); font-size: 11px; }
    .diff-actual { color: #F85149; font-family: var(--font-mono); font-size: 11px; }

    /* Bottom Grid: Evidence & Remediation Runbook */
    .bottom-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 900px) {
      .bottom-grid { grid-template-columns: 1fr; }
    }

    .audit-pane {
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
    }
    .pane-header {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .evidence-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .evidence-entry {
      background: var(--bg-subtle);
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 12px;
      color: var(--text-primary);
      display: flex;
      gap: 10px;
      line-height: 1.5;
    }
    .evidence-tag {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      padding-top: 1px;
    }

    /* Remediation Card */
    .runbook-card {
      background: var(--bg-subtle);
      border: 1px solid var(--border-muted);
      border-left: 3px solid #58A6FF;
      border-radius: var(--radius-sm);
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .runbook-meta {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .runbook-team {
      color: #58A6FF;
      font-weight: 700;
    }
    .runbook-action {
      font-size: 13px;
      color: var(--text-bright);
      line-height: 1.5;
    }
    .runbook-limitation {
      font-size: 11px;
      color: var(--text-muted);
      border-top: 1px solid var(--border-muted);
      padding-top: 8px;
      margin-top: 4px;
      font-style: italic;
    }
  </style>
</head>
<body>

  <!-- Top App Navigation -->
  <header class="app-header">
    <div class="header-left">
      <div class="app-logo">
        <span>RAGmortem</span>
        <span class="logo-badge">v0.2.0.dev0</span>
      </div>
      <div class="header-divider"></div>
      <div class="header-breadcrumb">
        <span>telemetry</span> / <span id="hdr-case-id" style="color:var(--text-bright);">demo_01</span>
      </div>
    </div>
    <div class="header-right">
      <button class="btn" onclick="window.print()">Export Audit Document</button>
      <button class="btn btn-primary" onclick="runDiagnostics()">Execute Triage</button>
    </div>
  </header>

  <!-- Split Framework Layout -->
  <div class="app-layout">
    
    <!-- Left Navigation & Input Control -->
    <aside class="sidebar">
      
      <!-- Scenario Selector -->
      <div class="sidebar-section">
        <div class="sidebar-title">
          <span>Evaluated Scenarios</span>
          <span>5 Records</span>
        </div>
        <div class="case-nav-list" id="preset-container">
          <!-- Dynamically populated via JS -->
        </div>
      </div>

      <!-- Live Trace Input Parameters -->
      <div class="sidebar-section" style="flex-grow: 1; border-bottom: none;">
        <div class="sidebar-title">
          <span>Trace Telemetry</span>
          <span>Input Fields</span>
        </div>

        <div class="field-group">
          <label class="field-label">User Query</label>
          <input type="text" class="input-text" id="inp-query">
        </div>

        <div class="field-group">
          <label class="field-label">Model Generated Output</label>
          <textarea class="input-textarea" id="inp-answer" rows="3"></textarea>
        </div>

        <div class="field-group">
          <label class="field-label">Reference Ground Truth (Optional)</label>
          <input type="text" class="input-text" id="inp-ref">
        </div>

        <div class="field-group">
          <label class="field-label">Score Scale Geometry</label>
          <select class="input-select" id="inp-score-type">
            <option value="similarity">Cosine Similarity [-1.0, 1.0]</option>
            <option value="cosine_distance">Cosine Distance (1 - d)</option>
            <option value="distance">Distance (Unknown Scale Guard)</option>
          </select>
        </div>

        <div class="field-group">
          <label class="field-label">Retrieved Chunks & Candidate Set (JSON)</label>
          <textarea class="input-textarea" id="inp-chunks" rows="6"></textarea>
        </div>

        <button class="btn btn-primary" style="width: 100%; justify-content: center; margin-top: 4px;" onclick="runDiagnostics()">
          Re-Analyze Trace
        </button>
      </div>

    </aside>

    <!-- Main Forensic Workbench -->
    <main class="workspace">
      
      <!-- 1. Executive Verdict Overview -->
      <section class="verdict-panel">
        <div class="verdict-top-bar">
          <div class="verdict-title-group">
            <div id="verdict-badge-box">
              <span class="badge badge-fail verdict-main-badge">Attribution: Generation Failure Suspected</span>
            </div>
          </div>
          <div class="verdict-meta-items">
            <div class="meta-item">
              <span class="meta-val" id="val-conf">90%</span>
              <span class="meta-lbl">Attribution Confidence</span>
            </div>
            <div class="meta-item">
              <span class="meta-val" id="val-mode">trace-only</span>
              <span class="meta-lbl">Diagnostic Mode</span>
            </div>
          </div>
        </div>
        <div class="verdict-summary-text" id="verdict-explanation">
          Retrieved context had high semantic relevance to question, but the generated answer appears ungrounded or contradicted context.
        </div>
      </section>

      <!-- 2. Execution Pipeline Waterfall (4-Stage Span Breakdown) -->
      <section class="trace-section">
        <div class="section-header">
          <span class="section-title-text">Pipeline Execution Breakdown</span>
          <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">span sequence: 01 ➔ 02 ➔ 03 ➔ 04</span>
        </div>

        <div class="pipeline-lanes">
          
          <!-- Stage 1 -->
          <div class="lane-box lane-healthy" id="lane-stage1">
            <div class="lane-header">
              <span class="lane-step">SPAN 01</span>
              <span class="badge badge-pass">VALID</span>
            </div>
            <div class="lane-title">Query Ingestion</div>
            <div class="lane-body">
              <p style="font-size: 12px; color: var(--text-bright); font-weight: 500;" id="lane1-query">"..."</p>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                Input length: <span id="lane1-tokens" style="font-family: var(--font-mono); color: var(--text-secondary);">0</span> words
              </div>
            </div>
          </div>

          <!-- Stage 2 -->
          <div class="lane-box" id="lane-stage2">
            <div class="lane-header">
              <span class="lane-step">SPAN 02</span>
              <span class="badge badge-pass" id="lane2-status">PASS</span>
            </div>
            <div class="lane-title">Vector Search (Bi-Encoder)</div>
            <div class="lane-body">
              <div class="metric-row">
                <div class="metric-row-header">
                  <span style="color: var(--text-secondary);">Top Relevance</span>
                  <span style="color: var(--text-bright); font-family: var(--font-mono);" id="lane2-score">0.88</span>
                </div>
                <div class="metric-progress">
                  <div class="metric-progress-bar" id="lane2-progress" style="width: 88%;"></div>
                </div>
                <div class="metric-ticks">
                  <span>0.00</span>
                  <span style="color: var(--status-fail-text);">0.47 Cutoff</span>
                  <span style="color: var(--status-pass-text);">0.53 High</span>
                  <span>1.00</span>
                </div>
              </div>
              <div style="font-size: 11px; color: var(--text-muted);" id="lane2-meta">
                Candidate evaluation complete
              </div>
            </div>
          </div>

          <!-- Stage 3 -->
          <div class="lane-box" id="lane-stage3">
            <div class="lane-header">
              <span class="lane-step">SPAN 03</span>
              <span class="badge badge-pass" id="lane3-status">PASS</span>
            </div>
            <div class="lane-title">Top-K Context Cutoff</div>
            <div class="lane-body">
              <div class="cutoff-list" id="lane3-cutoff-list">
                <!-- Dynamically populated via JS -->
              </div>
              <div style="font-size: 11px; color: var(--text-muted);" id="lane3-meta">
                Prompt context window evaluated
              </div>
            </div>
          </div>

          <!-- Stage 4 -->
          <div class="lane-box" id="lane-stage4">
            <div class="lane-header">
              <span class="lane-step">SPAN 04</span>
              <span class="badge badge-fail" id="lane4-status">DEFECT</span>
            </div>
            <div class="lane-title">LLM Fact Grounding</div>
            <div class="lane-body">
              <div class="diff-container">
                <div>
                  <div class="diff-item-label">Prompt Context Fact:</div>
                  <div class="diff-expected" id="lane4-expected">Verified in prompt context</div>
                </div>
                <div style="border-top: 1px solid var(--border-muted); padding-top: 4px;">
                  <div class="diff-item-label">Generated Response:</div>
                  <div class="diff-actual" id="lane4-actual">Ungrounded response</div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>

      <!-- 3. Bottom Grid: Evidence Trail & Remediation Runbook -->
      <section class="bottom-grid">
        
        <!-- Evidence Trail -->
        <div class="audit-pane">
          <div class="pane-header">
            <span>Diagnostic Evidence Trail</span>
          </div>
          <ul class="evidence-list" id="evidence-stream">
            <!-- Dynamically populated via JS -->
          </ul>
        </div>

        <!-- Remediation Protocol -->
        <div class="audit-pane">
          <div class="pane-header">
            <span>Remediation Protocol</span>
          </div>
          <div class="runbook-card">
            <div class="runbook-meta">
              <span class="runbook-team" id="ticket-team">PROMPT_ENGINEERING</span>
              <span style="color: var(--text-muted);">PRIORITY: HIGH</span>
            </div>
            <div class="runbook-action" id="ticket-body">
              Inspect generator prompt instructions, attention distribution, or system prompt grounding constraints.
            </div>
            <div class="runbook-limitation" id="ticket-limitation">
              Boundary: Mode cannot verify external corpus unretrieved documents.
            </div>
          </div>
        </div>

      </section>

    </main>
  </div>

  <script>
    let demoCases = [];
    let currentCase = null;

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
      c.innerHTML = demoCases.map((cs, idx) => {
        let badgeClass = 'badge-neutral', badgeLabel = 'UNKNOWN';
        const cid = cs.id.toLowerCase();
        if (cid.includes('generation')) { badgeClass = 'badge-fail'; badgeLabel = 'GEN_FAIL'; }
        else if (cid.includes('ranking')) { badgeClass = 'badge-warn'; badgeLabel = 'RANK_CUT'; }
        else if (cid.includes('abstention') || cid.includes('no_failure')) { badgeClass = 'badge-pass'; badgeLabel = 'CLEAN'; }
        
        return `
          <div class="case-nav-item ${idx === 0 ? 'active' : ''}" onclick="loadCase(${idx})">
            <div class="case-nav-top">
              <span class="case-nav-name">Trace 0${idx + 1}</span>
              <span class="badge ${badgeClass}">${badgeLabel}</span>
            </div>
            <div class="case-nav-query">${cs.question}</div>
          </div>
        `;
      }).join('');
    }

    function loadCase(idx) {
      document.querySelectorAll('.case-nav-item').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
      });
      const cs = demoCases[idx];
      currentCase = cs;
      document.getElementById('hdr-case-id').innerText = cs.id;
      document.getElementById('inp-query').value = cs.question || '';
      document.getElementById('inp-answer').value = cs.answer || '';
      document.getElementById('inp-ref').value = cs.reference_answer || '';
      document.getElementById('inp-score-type').value = cs.score_type || 'similarity';
      const chunksToShow = cs.candidates || cs.retrieved || [];
      document.getElementById('inp-chunks').value = JSON.stringify(chunksToShow, null, 2);
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
        alert("JSON syntax error in chunks: " + e.message);
        return;
      }

      const payload = {
        question: query,
        answer: answer,
        reference_answer: ref || null,
        score_type: scoreType,
      };

      if (currentCase && currentCase.candidates) {
        payload.candidates = chunks;
        payload.top_k = currentCase.top_k || 2;
        payload.retrieved = currentCase.retrieved || chunks.slice(0, currentCase.top_k || 2);
      } else {
        payload.retrieved = chunks;
      }

      try {
        const resp = await fetch('/api/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const res = await resp.json();
        renderResult(res, payload);
      } catch (err) {
        alert("Diagnosis failed: " + err);
      }
    }

    function renderResult(res, input) {
      const diag = (res.diagnosis || 'unknown').toLowerCase();
      let mainBadgeClass = 'badge-neutral';
      let teamIdentifier = 'TRIAGE_TEAM';

      const l1 = document.getElementById('lane-stage1');
      const l2 = document.getElementById('lane-stage2');
      const l3 = document.getElementById('lane-stage3');
      const l4 = document.getElementById('lane-stage4');

      l1.className = 'lane-box lane-healthy';
      l2.className = 'lane-box';
      l3.className = 'lane-box';
      l4.className = 'lane-box';

      document.getElementById('lane2-status').className = 'badge badge-pass';
      document.getElementById('lane2-status').innerText = 'PASS';
      document.getElementById('lane3-status').className = 'badge badge-pass';
      document.getElementById('lane3-status').innerText = 'PASS';
      document.getElementById('lane4-status').className = 'badge badge-pass';
      document.getElementById('lane4-status').innerText = 'PASS';

      if (diag.includes('generation')) {
        mainBadgeClass = 'badge-fail';
        teamIdentifier = 'PROMPT_ENGINEERING';
        l4.className = 'lane-box lane-defect';
        document.getElementById('lane4-status').className = 'badge badge-fail';
        document.getElementById('lane4-status').innerText = 'DEFECT';
      } else if (diag.includes('ranking')) {
        mainBadgeClass = 'badge-warn';
        teamIdentifier = 'RE_RANKER_CONFIG';
        l3.className = 'lane-box lane-warning';
        document.getElementById('lane3-status').className = 'badge badge-warn';
        document.getElementById('lane3-status').innerText = 'CUTOFF';
      } else if (diag.includes('retrieval')) {
        mainBadgeClass = 'badge-fail';
        teamIdentifier = 'SEARCH_EMBEDDINGS';
        l2.className = 'lane-box lane-defect';
        document.getElementById('lane2-status').className = 'badge badge-fail';
        document.getElementById('lane2-status').innerText = 'MISS';
      } else if (diag.includes('no_failure') || diag.includes('abstain')) {
        mainBadgeClass = 'badge-pass';
        teamIdentifier = 'SYSTEM_HEALTHY';
      } else {
        mainBadgeClass = 'badge-neutral';
        teamIdentifier = 'TELEMETRY_LOGGING';
        document.getElementById('lane2-status').innerText = 'UNKNOWN';
        document.getElementById('lane3-status').innerText = 'UNKNOWN';
        document.getElementById('lane4-status').innerText = 'UNKNOWN';
      }

      // Top Verdict Bar
      document.getElementById('verdict-badge-box').innerHTML = `
        <span class="badge ${mainBadgeClass} verdict-main-badge">${res.diagnosis_english}</span>
      `;
      document.getElementById('val-conf').innerText = Math.round((res.confidence || 0) * 100) + '%';
      document.getElementById('val-mode').innerText = res.mode || 'trace-only';
      document.getElementById('verdict-explanation').innerText = (res.evidence && res.evidence[0]) || res.diagnosis_english;

      // Span 01: Query
      document.getElementById('lane1-query').innerText = '"' + (input.question || '') + '"';
      document.getElementById('lane1-tokens').innerText = (input.question || '').split(/\s+/).filter(Boolean).length;

      // Span 02: Search
      const chunks = input.retrieved || [];
      const topScore = chunks.length > 0 && chunks[0].score != null ? chunks[0].score : null;
      if (topScore != null) {
        document.getElementById('lane2-score').innerText = topScore.toFixed(3);
        const pct = Math.max(0, Math.min(100, Math.round(topScore * 100)));
        document.getElementById('lane2-progress').style.width = pct + '%';
      } else {
        document.getElementById('lane2-score').innerText = 'N/A';
        document.getElementById('lane2-progress').style.width = '0%';
      }
      document.getElementById('lane2-meta').innerText = `${chunks.length} chunks evaluated in candidate pool`;

      // Span 03: Top-K Cutoff List
      const cutoffList = document.getElementById('lane3-cutoff-list');
      const cands = (input.candidates && input.candidates.length > 0) ? input.candidates : chunks;
      const cutoffK = input.top_k || 2;
      let cutHtml = '';
      if (cands.length > 0) {
        cands.slice(0, cutoffK).forEach((c, i) => {
          cutHtml += `<div class="cutoff-chunk chunk-admitted"><span>Rank #${i+1} (${c.score != null ? c.score.toFixed(2) : '-'})</span><span>IN PROMPT</span></div>`;
        });
      }
      cutHtml += `<div class="cutoff-line"><span class="cutoff-line-label">TOP-K CUTOFF (WINDOW = ${cutoffK})</span></div>`;
      if (cands.length > cutoffK) {
        cands.slice(cutoffK, cutoffK + 2).forEach((c, i) => {
          cutHtml += `<div class="cutoff-chunk chunk-pruned"><span>Rank #${cutoffK + i + 1} (${c.score != null ? c.score.toFixed(2) : '-'})</span><span>DROPPED</span></div>`;
        });
      }
      cutoffList.innerHTML = cutHtml;

      // Span 04: Generation Diff
      document.getElementById('lane4-expected').innerText = input.reference_answer ? `"${input.reference_answer}"` : "Retrieved evidence in prompt";
      document.getElementById('lane4-actual').innerText = `"${(input.answer || '').slice(0, 110)}${(input.answer || '').length > 110 ? '...' : ''}"`;

      // Evidence Trail
      const evStream = document.getElementById('evidence-stream');
      evStream.innerHTML = (res.evidence || []).map((e, idx) => `
        <li class="evidence-entry">
          <span class="evidence-tag">EV-${String(idx + 1).padStart(2, '0')}</span>
          <span>${e}</span>
        </li>
      `).join('');

      // Remediation Protocol
      document.getElementById('ticket-team').innerText = teamIdentifier;
      document.getElementById('ticket-body').innerText = res.suggested_next_check || '';
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

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0

        if content_length <= 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Empty or missing request body"}).encode("utf-8"))
            return

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
