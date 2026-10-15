"""Unit tests for Rich terminal UI components and visual Top-K cutoff rendering."""

from __future__ import annotations

import io
from rich.console import Console

from ragmortem.cli import main
from ragmortem.terminal import build_cutoff_info, render_case_card, render_cutoff_list, render_diagnostic_card


def test_build_cutoff_info_basic() -> None:
    """build_cutoff_info partitions retrieved and dropped candidate chunks."""
    retrieved = [
        {"id": "chunk_1", "score": 0.80},
        {"id": "chunk_2", "score": 0.62},
    ]
    candidates = [
        {"id": "chunk_1", "score": 0.80},
        {"id": "chunk_2", "score": 0.62},
        {"id": "chunk_3", "score": 0.61},
    ]

    info = build_cutoff_info(retrieved, candidates, top_k=2)
    assert info is not None
    assert info["top_k"] == 2
    assert len(info["retrieved"]) == 2
    assert len(info["dropped"]) == 1

    # Check retrieved
    assert info["retrieved"][0]["rank"] == 1
    assert info["retrieved"][0]["id"] == "chunk_1"
    assert info["retrieved"][0]["score"] == 0.80

    # Check dropped gap
    dropped = info["dropped"][0]
    assert dropped["rank"] == 3
    assert dropped["id"] == "chunk_3"
    assert dropped["score"] == 0.61
    assert round(dropped["gap"], 4) == -0.010


def test_render_cutoff_list_output() -> None:
    """render_cutoff_list outputs mini score bars, cutoff divider, and score gap."""
    info = {
        "top_k": 2,
        "retrieved": [
            {"rank": 1, "id": "chunk_p1", "score": 0.620},
            {"rank": 2, "id": "chunk_p2", "score": 0.615},
        ],
        "dropped": [
            {"rank": 3, "id": "chunk_p3", "score": 0.610, "gap": -0.005},
        ],
    }

    renderable = render_cutoff_list(info)
    assert renderable is not None

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    console.print(renderable)
    output = buf.getvalue()

    # Must contain score bar, (In Prompt), cutoff divider, and score gap
    assert "#1 [0.620]" in output
    assert "(In Prompt)" in output
    assert "✂️ TOP-K CUTOFF (k=2)" in output
    assert "#3 [0.610]" in output
    assert "⚠️ (-0.005 gap)" in output


def test_render_case_card_with_cutoff() -> None:
    """render_case_card embeds cutoff list when cutoff_info is present."""
    case_data = {
        "id": "test_ranking_case",
        "question": "What is the timeout?",
        "answer": "60 seconds",
        "diagnosis": "ranking_suspected",
        "diagnosis_english": "Ranking Failure Suspected",
        "confidence": 0.85,
        "mode": "trace-only",
        "evidence": ["Strong candidate exists below cutoff."],
        "what_this_cannot_tell_you": "Trace-only mode limitation.",
        "suggested_next_check": "Inspect re-ranker weights.",
        "cutoff_info": {
            "top_k": 2,
            "retrieved": [
                {"rank": 1, "id": "chunk_a", "score": 0.70},
                {"rank": 2, "id": "chunk_b", "score": 0.65},
            ],
            "dropped": [
                {"rank": 3, "id": "chunk_c", "score": 0.64, "gap": -0.01},
            ],
        },
    }

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    render_case_card(console, 1, case_data)
    out = buf.getvalue()

    assert "RANKING SUSPECTED" in out
    assert "✂️ TOP-K CUTOFF (k=2)" in out
    assert "#3 [0.640]" in out
    assert "⚠️ (-0.010 gap)" in out


def test_cli_demo_includes_visual_cutoff() -> None:
    """ragmortem demo executes Case 3 and renders the visual top-k cutoff."""
    buf = io.StringIO()
    with io.StringIO() as buf:
        import sys
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            exit_code = main(["demo"])
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert exit_code == 0
        assert "✂️ TOP-K CUTOFF" in output
        assert "chunk_db_failover_timeout" in output
