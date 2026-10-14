"""Zero-dependency local HTTP server and live showcase UI for RAGmortem.

Provides a world-class, state-of-the-art forensic diagnostic observatory
for RAG failure root-cause attribution.
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
  <title>RAGmortem • Automated RAG Failure Forensic Observatory</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, Menlo, Monaco, Consolas, monospace;
      
      --bg-base: #07090E;
      --bg-surface: #0E131F;
      --bg-surface-elevated: #141B2D;
      --bg-surface-glass: rgba(14, 19, 31, 0.75);
      
      --border-subtle: rgba(255, 255, 255, 0.07);
      --border-light: rgba(255, 255, 255, 0.12);
      --border-focus: #6366F1;
      
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      
      /* Status Tints */
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
      --info-glow: rgba(56, 189, 248, 0.25);
      --info-bg: rgba(56, 189, 248, 0.10);
      
      --purple-main: #A855F7;
      --purple-glow: rgba(168, 85, 247, 0.25);
      --purple-bg: rgba(168, 85, 247, 0.10);
      
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-xl: 20px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: var(--font-sans);
      background-color: var(--bg-base);
      background-image: 
        radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 85% 5%, rgba(244, 63, 94, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.04) 0%, transparent 60%);
      background-attachment: fixed;
      color: var(--text-primary);
      line-height: 1.5;
      min-height: 100vh;
      padding: 24px 32px 48px;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1440px;
      margin: 0 auto;
    }

    /* Top Navigation Header */
    .top-nav {
      background: var(--bg-surface-glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-light);
      border-radius: var(--radius-lg);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    
    .brand-group {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-logo-mark {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      color: #fff;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .brand-text h1 {
      font-size: 17px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-sub {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .version-tag {
      font-family: var(--font-mono);
      font-size: 11px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
    }
    
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn {
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: var(--radius-md);
      border: 1px solid transparent;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      text-decoration: none;
    }
    .btn-primary {
      background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
      color: #fff;
      box-shadow: 0 2px 12px rgba(99, 102, 241, 0.3);
    }
    .btn-primary:hover {
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5);
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: var(--bg-surface-elevated);
      color: var(--text-secondary);
      border-color: var(--border-light);
    }
    .btn-secondary:hover {
      background: #1E283F;
      color: var(--text-primary);
      border-color: rgba(255, 255, 255, 0.2);
    }

    /* Main Dashboard Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 1100px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }

    /* Sidebar / Control Panel */
    .panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .panel-title {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    /* Scenario Selector Cards */
    .scenarios-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 24px;
    }
    .scenario-card {
      background: rgba(20, 27, 45, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .scenario-card:hover {
      background: var(--bg-surface-elevated);
      border-color: var(--border-light);
      transform: translateX(2px);
    }
    .scenario-card.active {
      background: rgba(99, 102, 241, 0.12);
      border-color: #6366F1;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.2);
    }
    .scenario-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }
    .scenario-name {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-primary);
    }
    .scenario-tag {
      font-size: 10px;
      font-weight: 700;
      font-family: var(--font-mono);
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .tag-gen { background: var(--danger-bg); color: var(--danger-main); border: 1px solid var(--danger-border); }
    .tag-rank { background: var(--warning-bg); color: var(--warning-main); border: 1px solid var(--warning-border); }
    .tag-ok { background: var(--success-bg); color: var(--success-main); border: 1px solid var(--success-border); }
    .tag-unk { background: rgba(148, 163, 184, 0.12); color: #94A3B8; border: 1px solid rgba(148, 163, 184, 0.2); }
    .scenario-query {
      font-size: 11px;
      color: var(--text-secondary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Form Fields */
    .form-group {
      margin-bottom: 12px;
    }
    label {
      display: block;
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 5px;
    }
    input[type="text"], textarea, select {
      width: 100%;
      background: #090E17;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 12px;
      color: var(--text-primary);
      font-family: var(--font-sans);
      transition: border-color 0.15s ease;
    }
    textarea {
      font-family: var(--font-mono);
      font-size: 11px;
      line-height: 1.5;
      resize: vertical;
    }
    input[type="text"]:focus, textarea:focus, select:focus {
      outline: none;
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    /* RIGHT DISPLAY: FORENSIC WORKBENCH */
    .workbench {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Executive Verdict Banner */
    .verdict-hero {
      background: var(--bg-surface);
      border: 1px solid var(--border-light);
      border-radius: var(--radius-lg);
      padding: 24px 28px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .verdict-hero::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, #F43F5E, #F59E0B, #10B981, #6366F1);
    }
    .verdict-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .verdict-label {
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .verdict-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 16px;
      font-weight: 800;
      padding: 8px 18px;
      border-radius: 9999px;
      letter-spacing: -0.01em;
      box-shadow: 0 0 24px rgba(0, 0, 0, 0.3);
    }
    .badge-gen {
      background: var(--danger-bg);
      color: #FDA4AF;
      border: 1px solid var(--danger-border);
      box-shadow: 0 0 20px var(--danger-glow);
    }
    .badge-rank {
      background: var(--warning-bg);
      color: #FDE68A;
      border: 1px solid var(--warning-border);
      box-shadow: 0 0 20px var(--warning-glow);
    }
    .badge-ret {
      background: var(--purple-bg);
      color: #E9D5FF;
      border: 1px solid rgba(168, 85, 247, 0.35);
      box-shadow: 0 0 20px var(--purple-glow);
    }
    .badge-ok {
      background: var(--success-bg);
      color: #A7F3D0;
      border: 1px solid var(--success-border);
      box-shadow: 0 0 20px var(--success-glow);
    }
    .badge-unk {
      background: rgba(148, 163, 184, 0.15);
      color: #E2E8F0;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .meta-metrics {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .metric-pill {
      text-align: right;
    }
    .metric-val {
      font-size: 20px;
      font-weight: 800;
      color: var(--text-primary);
      font-family: var(--font-mono);
    }
    .metric-name {
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* PRIMARY SECTION: FORENSIC PIPELINE INSPECTOR */
    .pipeline-container {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }
    .section-eyebrow {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .section-heading {
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #E2E8F0;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* 4-Stage Horizontal Flow */
    .flow-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      position: relative;
    }
    @media (max-width: 1024px) {
      .flow-grid { grid-template-columns: 1fr; }
    }

    .flow-node {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: all 0.2s ease;
    }
    .flow-node.breakpoint-alert {
      border-color: var(--danger-main);
      background: rgba(244, 63, 94, 0.05);
      box-shadow: 0 0 24px rgba(244, 63, 94, 0.2);
    }
    .flow-node.breakpoint-warning {
      border-color: var(--warning-main);
      background: rgba(245, 158, 11, 0.05);
      box-shadow: 0 0 24px rgba(245, 158, 11, 0.2);
    }
    .flow-node.passed {
      border-color: rgba(16, 185, 129, 0.4);
    }

    /* Pulsing Breakpoint Indicator */
    .breakpoint-pill {
      position: absolute;
      top: -12px;
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
      display: flex;
      align-items: center;
      gap: 4px;
      box-shadow: 0 0 12px var(--danger-main);
      animation: pulseAlert 2s infinite ease-in-out;
      z-index: 2;
    }
    .breakpoint-pill.warning-pill {
      background: var(--warning-main);
      box-shadow: 0 0 12px var(--warning-main);
    }
    @keyframes pulseAlert {
      0%, 100% { opacity: 1; transform: translateX(-50%) scale(1); }
      50% { opacity: 0.8; transform: translateX(-50%) scale(1.05); }
    }

    .node-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .node-number {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
    }
    .node-status {
      font-size: 10px;
      font-weight: 800;
      font-family: var(--font-mono);
      padding: 2px 6px;
      border-radius: 4px;
      letter-spacing: 0.04em;
    }
    .status-pass { background: var(--success-bg); color: var(--success-main); }
    .status-fail { background: var(--danger-bg); color: var(--danger-main); }
    .status-warn { background: var(--warning-bg); color: var(--warning-main); }
    .status-neutral { background: rgba(148, 163, 184, 0.1); color: var(--text-muted); }

    .node-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 8px;
    }
    .node-body {
      font-size: 12px;
      color: var(--text-secondary);
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    /* Score Distribution Visualizer (Stage 2) */
    .score-meter {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      margin-top: 4px;
    }
    .score-meta {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .score-bar-track {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      overflow: hidden;
      position: relative;
    }
    .score-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #38BDF8, #6366F1);
      border-radius: 3px;
    }
    .threshold-ticks {
      display: flex;
      justify-content: space-between;
      font-size: 9px;
      font-family: var(--font-mono);
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Top-K Cutoff Visualization (Stage 3) */
    .cutoff-diagram {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .cutoff-item {
      font-size: 10px;
      font-family: var(--font-mono);
      padding: 4px 6px;
      border-radius: 4px;
      display: flex;
      justify-content: space-between;
    }
    .in-prompt {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: #A7F3D0;
    }
    .cutoff-divider {
      border-top: 1px dashed var(--warning-main);
      margin: 4px 0;
      text-align: center;
      position: relative;
    }
    .cutoff-label {
      font-size: 8px;
      font-weight: 800;
      color: var(--warning-main);
      background: #0E131F;
      padding: 0 4px;
      position: relative;
      top: -6px;
    }
    .dropped-candidate {
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.2);
      color: #FDA4AF;
      opacity: 0.8;
    }

    /* Fact Comparison View (Stage 4) */
    .fact-diff-box {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 11px;
    }
    .fact-row {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .fact-lbl {
      font-size: 9px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    .fact-txt-exp { color: #A7F3D0; font-weight: 600; }
    .fact-txt-gen { color: #FDA4AF; font-weight: 600; }

    /* SECONDARY SECTION: FORENSIC INVESTIGATION & ACTIONABLE TICKET */
    .investigation-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 900px) {
      .investigation-grid { grid-template-columns: 1fr; }
    }

    .audit-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .audit-card-title {
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-primary);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Evidence Timeline */
    .evidence-stream {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .evidence-event {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      background: rgba(20, 27, 45, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
    }
    .event-icon {
      font-size: 14px;
      line-height: 1;
      margin-top: 2px;
    }
    .event-desc {
      font-size: 12px;
      color: var(--text-primary);
      line-height: 1.5;
    }

    /* Actionable Engineering Remediation Ticket */
    .ticket-container {
      background: linear-gradient(180deg, rgba(30, 41, 67, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(99, 102, 241, 0.35);
      border-radius: var(--radius-md);
      padding: 18px;
      box-shadow: 0 0 24px rgba(99, 102, 241, 0.1);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      height: 100%;
    }
    .ticket-badge-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .ticket-chip {
      font-size: 10px;
      font-weight: 800;
      font-family: var(--font-mono);
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      color: #A5B4FC;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .ticket-body {
      font-size: 13px;
      color: #E2E8F0;
      line-height: 1.6;
      margin-bottom: 14px;
    }
    .limitation-box {
      background: rgba(0, 0, 0, 0.35);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
      border-left: 2px solid #64748B;
    }

    /* Code & JSON Sandbox Modal / Tab */
    .tab-row {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 8px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 700;
      padding: 4px 8px;
      cursor: pointer;
      border-radius: 4px;
    }
    .tab-btn.active {
      color: #fff;
      background: var(--bg-surface-elevated);
    }
  </style>
</head>
<body>
  <div class="container">
    
    <!-- Top Navigation Bar -->
    <header class="top-nav">
      <div class="brand-group">
        <div class="brand-logo-mark">⚡</div>
        <div class="brand-text">
          <h1>RAGmortem <span class="version-tag">v0.2.0.dev0</span></h1>
          <div class="brand-sub">Deterministic RAG Failure Forensic Observatory</div>
        </div>
      </div>
      <div class="nav-actions">
        <button class="btn btn-secondary" onclick="window.print()">🖨️ Export PDF / One-Pager</button>
        <button class="btn btn-primary" onclick="runDiagnostics()">⚡ Re-Run Diagnostics</button>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="dashboard-grid">
      
      <!-- LEFT SIDEBAR: Scenarios & Trace Payload -->
      <aside class="panel">
        <div class="panel-header">
          <span class="panel-title">Pre-Loaded Failure Scenarios</span>
          <span style="font-size: 10px; color: var(--text-muted);">5 Curated Studies</span>
        </div>
        
        <div class="scenarios-list" id="preset-container">
          <!-- Dynamically populated via JS -->
        </div>

        <div class="panel-header" style="margin-top: 20px;">
          <span class="panel-title">Trace Payload Inspector</span>
          <span style="font-size: 10px; color: var(--text-muted);">Live Sandbox</span>
        </div>

        <div class="form-group">
          <label>Query / User Question</label>
          <input type="text" id="inp-query" placeholder="Question asked to RAG pipeline">
        </div>

        <div class="form-group">
          <label>Model Generated Answer</label>
          <textarea id="inp-answer" rows="3" placeholder="Generated bot answer"></textarea>
        </div>

        <div class="form-group">
          <label>Reference Ground Truth (Optional)</label>
          <input type="text" id="inp-ref" placeholder="Expected true fact">
        </div>

        <div class="form-group">
          <label>Score Metric Scale</label>
          <select id="inp-score-type">
            <option value="similarity">Cosine Similarity [-1.0, 1.0]</option>
            <option value="cosine_distance">Cosine Distance (1 - d)</option>
            <option value="distance">Distance of Unknown Scale (Safety Guard)</option>
          </select>
        </div>

        <div class="form-group">
          <label>Retrieved Chunks & Candidate Pool (JSON)</label>
          <textarea id="inp-chunks" rows="5"></textarea>
        </div>

        <button class="btn btn-primary" style="width: 100%; justify-content: center; margin-top: 8px;" onclick="runDiagnostics()">
          ⚡ Run Root-Cause Diagnosis
        </button>
      </aside>

      <!-- RIGHT MAIN: Forensic Workbench -->
      <main class="workbench">
        
        <!-- 1. Executive Verdict Hero Card -->
        <section class="verdict-hero">
          <div class="verdict-header">
            <div>
              <div class="verdict-label">Deterministic Root-Cause Attribution Verdict</div>
              <div id="verdict-badge-box">
                <span class="verdict-badge badge-gen">🔴 Generation Failure Suspected</span>
              </div>
            </div>
            <div class="meta-metrics">
              <div class="metric-pill">
                <div class="metric-val" id="val-conf">90%</div>
                <div class="metric-name">Attribution Confidence</div>
              </div>
              <div class="metric-pill">
                <div class="metric-val" id="val-mode" style="font-size: 15px; color: #38BDF8;">trace-only</div>
                <div class="metric-name">Evaluation Mode</div>
              </div>
            </div>
          </div>
          <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;" id="verdict-explanation">
            Retrieved context had high semantic relevance to question, but the generated answer appears ungrounded or contradicted context.
          </div>
        </section>

        <!-- 2. Primary 4-Stage Forensic Pipeline Flow -->
        <section class="pipeline-container">
          <div class="section-eyebrow">
            <div class="section-heading">
              <span>🔬 Interactive 4-Stage RAG Pipeline Breakdown</span>
            </div>
            <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">FLOW: Query ➔ Retrieval ➔ Re-rank ➔ LLM</span>
          </div>

          <div class="flow-grid">
            
            <!-- Stage 1 -->
            <div class="flow-node passed" id="node-stage1">
              <div class="node-header">
                <span class="node-number">01 / INPUT</span>
                <span class="node-status status-pass">OK</span>
              </div>
              <div class="node-title">Query Formulation</div>
              <div class="node-body">
                <p style="font-weight: 600; color: #fff; font-size: 12px;" id="node1-query">"..."</p>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                  Token Budget: <span id="node1-tokens">0</span> tokens
                </div>
              </div>
            </div>

            <!-- Stage 2 -->
            <div class="flow-node" id="node-stage2">
              <div id="breakpoint-stage2"></div>
              <div class="node-header">
                <span class="node-number">02 / SEARCH</span>
                <span class="node-status" id="node2-status">PASS</span>
              </div>
              <div class="node-title">Vector Retrieval</div>
              <div class="node-body">
                <div class="score-meter">
                  <div class="score-meta">
                    <span style="color: var(--text-muted);">Top Chunk Score:</span>
                    <span style="color: #38BDF8;" id="node2-score">0.88</span>
                  </div>
                  <div class="score-bar-track">
                    <div class="score-bar-fill" id="node2-score-bar" style="width: 88%;"></div>
                  </div>
                  <div class="threshold-ticks">
                    <span>0.00</span>
                    <span style="color:#F43F5E;">0.47 Cutoff</span>
                    <span style="color:#10B981;">0.53 High</span>
                    <span>1.00</span>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--text-muted);" id="node2-meta">
                  1 chunk(s) evaluated
                </div>
              </div>
            </div>

            <!-- Stage 3 -->
            <div class="flow-node" id="node-stage3">
              <div id="breakpoint-stage3"></div>
              <div class="node-header">
                <span class="node-number">03 / RANKING</span>
                <span class="node-status" id="node3-status">PASS</span>
              </div>
              <div class="node-title">Top-K Context Window</div>
              <div class="node-body">
                <div class="cutoff-diagram" id="node3-diagram">
                  <div class="cutoff-item in-prompt">
                    <span>Rank #1 (Score: 0.88)</span>
                    <span>IN CONTEXT</span>
                  </div>
                  <div class="cutoff-divider">
                    <span class="cutoff-label">✂️ PROMPT WINDOW CUTOFF</span>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--text-muted);" id="node3-meta">
                  Candidate margin evaluated.
                </div>
              </div>
            </div>

            <!-- Stage 4 -->
            <div class="flow-node" id="node-stage4">
              <div id="breakpoint-stage4"></div>
              <div class="node-header">
                <span class="node-number">04 / GENERATION</span>
                <span class="node-status" id="node4-status">FAIL</span>
              </div>
              <div class="node-title">LLM Fact Grounding</div>
              <div class="node-body">
                <div class="fact-diff-box">
                  <div class="fact-row">
                    <span class="fact-lbl">Context Fact:</span>
                    <span class="fact-txt-exp" id="node4-expected">Present in prompt context</span>
                  </div>
                  <div class="fact-row" style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 4px;">
                    <span class="fact-lbl">Model Stated:</span>
                    <span class="fact-txt-gen" id="node4-actual">Answer ungrounded</span>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </section>

        <!-- 3. Forensic Evidence & Actionable Engineering Ticket -->
        <section class="investigation-grid">
          
          <!-- Evidence Stream -->
          <div class="audit-card">
            <div class="audit-card-title">
              <span>🔍 Forensic Evidence Timeline</span>
            </div>
            <div class="evidence-stream" id="evidence-stream">
              <!-- Dynamically populated via JS -->
            </div>
          </div>

          <!-- Actionable Engineering Ticket -->
          <div class="audit-card">
            <div class="audit-card-title">
              <span>📋 Actionable Engineering Remedy</span>
            </div>
            <div class="ticket-container">
              <div>
                <div class="ticket-badge-row">
                  <span class="ticket-chip" id="ticket-team">Prompt Engineering</span>
                  <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">Severity: High</span>
                </div>
                <div class="ticket-body" id="ticket-body">
                  Inspect generator prompt instructions, attention distribution, or system prompt grounding constraints.
                </div>
              </div>
              <div class="limitation-box" id="ticket-limitation">
                Limitations: What this diagnostic mode cannot verify.
              </div>
            </div>
          </div>

        </section>

      </main>
    </div>
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
        let tagClass = 'tag-unk', tagLabel = 'UNKNOWN';
        const cid = cs.id.toLowerCase();
        if (cid.includes('generation')) { tagClass = 'tag-gen'; tagLabel = 'GENERATION'; }
        else if (cid.includes('ranking')) { tagClass = 'tag-rank'; tagLabel = 'RANKING'; }
        else if (cid.includes('abstention') || cid.includes('no_failure')) { tagClass = 'tag-ok'; tagLabel = 'HEALTHY'; }
        
        return `
          <div class="scenario-card ${idx === 0 ? 'active' : ''}" onclick="loadCase(${idx})">
            <div class="scenario-top">
              <span class="scenario-name">Case ${idx + 1}: ${cs.id.replace('demo_', '').replace(/_/g, ' ')}</span>
              <span class="scenario-tag ${tagClass}">${tagLabel}</span>
            </div>
            <div class="scenario-query">${cs.question}</div>
          </div>
        `;
      }).join('');
    }

    function loadCase(idx) {
      document.querySelectorAll('.scenario-card').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
      });
      const cs = demoCases[idx];
      currentCase = cs;
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
        alert("Invalid JSON in chunks field: " + e.message);
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
        alert("Diagnosis request failed: " + err);
      }
    }

    function renderResult(res, input) {
      const diag = (res.diagnosis || 'unknown').toLowerCase();
      let badgeClass = 'badge-unk';
      let badgeIcon = '⚪';
      let teamChip = 'Investigation Team';

      // Reset node styles
      const n1 = document.getElementById('node-stage1');
      const n2 = document.getElementById('node-stage2');
      const n3 = document.getElementById('node-stage3');
      const n4 = document.getElementById('node-stage4');
      
      n1.className = 'flow-node passed';
      n2.className = 'flow-node passed';
      n3.className = 'flow-node passed';
      n4.className = 'flow-node passed';

      document.getElementById('breakpoint-stage2').innerHTML = '';
      document.getElementById('breakpoint-stage3').innerHTML = '';
      document.getElementById('breakpoint-stage4').innerHTML = '';

      document.getElementById('node2-status').className = 'node-status status-pass';
      document.getElementById('node2-status').innerText = 'PASS';
      document.getElementById('node3-status').className = 'node-status status-pass';
      document.getElementById('node3-status').innerText = 'PASS';
      document.getElementById('node4-status').className = 'node-status status-pass';
      document.getElementById('node4-status').innerText = 'PASS';

      if (diag.includes('generation')) {
        badgeClass = 'badge-gen';
        badgeIcon = '🔴';
        teamChip = 'Prompt Engineering Team';
        n4.className = 'flow-node breakpoint-alert';
        document.getElementById('node4-status').className = 'node-status status-fail';
        document.getElementById('node4-status').innerText = 'FAIL';
        document.getElementById('breakpoint-stage4').innerHTML = `
          <div class="breakpoint-pill">🚨 BREAKPOINT: HALLUCINATION</div>
        `;
      } else if (diag.includes('ranking')) {
        badgeClass = 'badge-rank';
        badgeIcon = '🟠';
        teamChip = 'Re-Ranker & Indexing Team';
        n3.className = 'flow-node breakpoint-warning';
        document.getElementById('node3-status').className = 'node-status status-warn';
        document.getElementById('node3-status').innerText = 'CUTOFF';
        document.getElementById('breakpoint-stage3').innerHTML = `
          <div class="breakpoint-pill warning-pill">⚠️ BREAKPOINT: TOP-K CUTOFF</div>
        `;
      } else if (diag.includes('retrieval')) {
        badgeClass = 'badge-ret';
        badgeIcon = '🟣';
        teamChip = 'Search & Embeddings Team';
        n2.className = 'flow-node breakpoint-alert';
        document.getElementById('node2-status').className = 'node-status status-fail';
        document.getElementById('node2-status').innerText = 'MISS';
        document.getElementById('breakpoint-stage2').innerHTML = `
          <div class="breakpoint-pill">🚨 BREAKPOINT: RETRIEVAL MISS</div>
        `;
      } else if (diag.includes('no_failure') || diag.includes('abstain')) {
        badgeClass = 'badge-ok';
        badgeIcon = '🟢';
        teamChip = 'Quality & Telemetry Monitoring';
      } else {
        badgeClass = 'badge-unk';
        badgeIcon = '⚪';
        teamChip = 'Telemetry Instrumentation Team';
        n2.className = 'flow-node';
        n3.className = 'flow-node';
        n4.className = 'flow-node';
        document.getElementById('node2-status').innerText = 'INCONCLUSIVE';
        document.getElementById('node3-status').innerText = 'INCONCLUSIVE';
        document.getElementById('node4-status').innerText = 'INCONCLUSIVE';
      }

      // Hero Elements
      document.getElementById('verdict-badge-box').innerHTML = `
        <span class="verdict-badge ${badgeClass}">${badgeIcon} ${res.diagnosis_english}</span>
      `;
      document.getElementById('val-conf').innerText = Math.round((res.confidence || 0) * 100) + '%';
      document.getElementById('val-mode').innerText = res.mode || 'trace-only';
      document.getElementById('verdict-explanation').innerText = (res.evidence && res.evidence[0]) || res.diagnosis_english;

      // Stage 1 (Query)
      document.getElementById('node1-query').innerText = '"' + (input.question || '') + '"';
      document.getElementById('node1-tokens').innerText = (input.question || '').split(/\s+/).filter(Boolean).length;

      // Stage 2 (Retrieval)
      const chunks = input.retrieved || [];
      const topScore = chunks.length > 0 && chunks[0].score != null ? chunks[0].score : null;
      if (topScore != null) {
        document.getElementById('node2-score').innerText = topScore.toFixed(3);
        const pct = Math.max(0, Math.min(100, Math.round(topScore * 100)));
        document.getElementById('node2-score-bar').style.width = pct + '%';
      } else {
        document.getElementById('node2-score').innerText = 'N/A';
        document.getElementById('node2-score-bar').style.width = '0%';
      }
      document.getElementById('node2-meta').innerText = `${chunks.length} candidate chunk(s) evaluated`;

      // Stage 3 (Ranking Cutoff Diagram)
      const diagram = document.getElementById('node3-diagram');
      const cands = (input.candidates && input.candidates.length > 0) ? input.candidates : chunks;
      const cutoffK = input.top_k || 2;
      let diagHtml = '';
      if (cands.length > 0) {
        cands.slice(0, cutoffK).forEach((c, i) => {
          diagHtml += `<div class="cutoff-item in-prompt"><span>Rank #${i+1} (${c.score != null ? c.score.toFixed(2) : '-'})</span><span>IN PROMPT</span></div>`;
        });
      }
      diagHtml += `<div class="cutoff-divider"><span class="cutoff-label">✂️ PROMPT WINDOW CUTOFF (TOP-${cutoffK})</span></div>`;
      if (cands.length > cutoffK) {
        cands.slice(cutoffK, cutoffK + 2).forEach((c, i) => {
          diagHtml += `<div class="cutoff-item dropped-candidate"><span>Rank #${cutoffK + i + 1} (${c.score != null ? c.score.toFixed(2) : '-'})</span><span>DROPPED</span></div>`;
        });
      }
      diagram.innerHTML = diagHtml;

      // Stage 4 (Fact Grounding)
      document.getElementById('node4-expected').innerText = input.reference_answer ? `"${input.reference_answer}"` : "Evidence chunk verified in prompt";
      document.getElementById('node4-actual').innerText = `"${(input.answer || '').slice(0, 110)}${(input.answer || '').length > 110 ? '...' : ''}"`;

      // Evidence Stream
      const evStream = document.getElementById('evidence-stream');
      evStream.innerHTML = (res.evidence || []).map((e, idx) => `
        <div class="evidence-event">
          <div class="event-icon">${idx === 0 ? '🎯' : '📊'}</div>
          <div class="event-desc">${e}</div>
        </div>
      `).join('');

      // Ticket Box
      document.getElementById('ticket-team').innerText = teamChip;
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
