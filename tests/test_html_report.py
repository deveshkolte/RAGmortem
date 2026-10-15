"""Unit tests for static HTML export."""

from __future__ import annotations

import tempfile
from pathlib import Path
from ragmortem.report_html import render_html_report
from ragmortem.cli import main


def test_render_html_report_content() -> None:
    """Verifies that render_html_report generates valid self-contained dark-mode HTML."""
    results = [
        {
            "id": "case_ranking_01",
            "question": "What is the P1 SLA?",
            "answer": "P1 SLA is 4 hours.",
            "diagnosis": "ranking_suspected",
            "diagnosis_english": "Ranking Failure Suspected",
            "mode": "observed",
            "confidence": 0.85,
            "evidence": ["Top retrieved chunk has lower score than dropped candidate", "Score gap: 0.050"],
            "what_this_cannot_tell_you": "Without gold chunk label, candidate rank is inferred.",
            "suggested_next_check": "Inspect re-ranking stage or increase top-k.",
            "cutoff_info": {
                "cutoff_k": 2,
                "items": [
                    {"rank": 1, "score": 0.62, "id": "chunk_ret_1", "tag": "in_prompt"},
                    {"rank": 2, "score": 0.58, "id": "chunk_ret_2", "tag": "in_prompt"},
                    {"rank": 3, "score": 0.57, "id": "chunk_dropped_1", "tag": "dropped", "gap": 0.01},
                ],
            },
        }
    ]
    summary = {
        "total_cases": 1,
        "counts_per_diagnosis": {"ranking_suspected": 1},
        "unknown_count": 0,
        "unknown_rate_pct": 0.0,
    }

    html_out = render_html_report(results, summary, is_demo=True, profile="minilm")

    # Assert essential sections and styling
    assert "<!DOCTYPE html>" in html_out
    assert "#0d1117" in html_out  # dark theme background
    assert "🔬 RAGmortem Diagnostic Report" in html_out
    assert "SYNTHETIC DEMO DATA" in html_out
    assert "Profile:" in html_out
    assert "minilm" in html_out
    assert "RANKING_SUSPECTED" in html_out
    assert "✂️ TOP-K CUTOFF (k=2)" in html_out
    assert "chunk_dropped_1" in html_out
    assert "⚠️ (-0.010 gap)" in html_out
    assert "What this cannot tell you:" in html_out
    assert "Suggested next check:" in html_out


def test_cli_diagnose_format_html(tmp_path: Path) -> None:
    """Verifies ragmortem demo/diagnose with --format html --out report.html."""
    out_file = tmp_path / "report.html"
    ret = main(["demo", "--format", "html", "--out", str(out_file)])
    assert ret == 0
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "#0d1117" in content
    assert "🔬 RAGmortem Diagnostic Report" in content
    assert "Total Traces Audited" in content
