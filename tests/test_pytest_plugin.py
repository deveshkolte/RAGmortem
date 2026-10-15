"""Unit tests for pytest-ragmortem plugin."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

pytest_plugins = ["pytester"]


def test_pytest_ragmortem_fixture_on_failure(pytester: pytest.Pytester) -> None:
    """Verifies that a failing RAG test automatically prints a diagnostic failure card."""
    pytester.makepyfile(
        """
        def test_failing_rag(rag_trace):
            rag_trace.record(
                query="What is the mandatory engineering response time for a P1 incident?",
                retrieved_chunks=[{"id": "doc1", "text": "P1 SLA is 15 minutes.", "score": 0.88}],
                scores=[0.88],
                answer="The SLA is 48 hours.",
            )
            # Intentional test assertion failure
            assert "15 minutes" in "The SLA is 48 hours."
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    stdout = result.stdout.str()
    # Diagnostic card should be rendered on failure
    assert "RAGmortem Test Failure Diagnosis" in stdout
    assert "GENERATION_SUSPECTED" in stdout or "generation_suspected" in stdout.lower()


def test_pytest_ragmortem_github_actions_step_summary(
    pytester: pytest.Pytester,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that in GitHub Actions, $GITHUB_STEP_SUMMARY and ::error annotations are produced."""
    step_summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary_file))

    pytester.makepyfile(
        """
        def test_ci_failing_rag(rag_trace):
            rag_trace.record(
                query="What is the SLA?",
                retrieved_chunks=[{"id": "doc1", "text": "SLA is 15m.", "score": 0.90}],
                scores=[0.90],
                answer="SLA is 48h.",
            )
            assert False, "SLA assertion failed"
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)

    # Check step summary was generated
    assert step_summary_file.exists()
    summary_text = step_summary_file.read_text(encoding="utf-8")
    assert "RAGmortem Failure Diagnosis" in summary_text
    assert "GENERATION_SUSPECTED" in summary_text

    # Check workflow command annotation in stderr/stdout
    combined = result.stdout.str() + "\n" + result.stderr.str()
    assert "::error" in combined
    assert "RAGmortem Diagnosis" in combined


def test_pytest_ragmortem_disabled_flag(pytester: pytest.Pytester) -> None:
    """Verifies that --no-ragmortem suppresses diagnostic output on failure."""
    pytester.makepyfile(
        """
        def test_failing_no_diag(rag_trace):
            rag_trace.record(
                query="What is the SLA?",
                retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.90}],
                answer="wrong",
            )
            assert False
        """
    )

    result = pytester.runpytest("--no-ragmortem")
    result.assert_outcomes(failed=1)
    stdout = result.stdout.str()
    assert "RAGmortem Test Failure Diagnosis" not in stdout


def test_record_trace_context_manager() -> None:
    """Verifies the record_trace context manager captures telemetry."""
    from ragmortem.pytest_plugin import record_trace, _ACTIVE_RECORDER

    with record_trace(
        query="What is the SLA?",
        retrieved_chunks=[{"id": "c1", "text": "15m", "score": 0.85}],
        answer="15m",
    ) as rec:
        assert rec.current_trace is not None
        assert rec.current_trace.query == "What is the SLA?"
        assert rec.current_trace.scores == [0.85]


def test_ragmortem_assert_failure_raises() -> None:
    """Verifies that ragmortem_assert raises AssertionError with Rich diagnostic report on failure."""
    from ragmortem.pytest_plugin import ragmortem_assert

    bad_trace = {
        "query": "What is the mandatory engineering response time for a P1 incident?",
        "retrieved_chunks": [
            {"id": "c1", "text": "P1 SLA is 15 minutes.", "score": 0.88}
        ],
        "answer": "The SLA is 48 hours.",
    }

    with pytest.raises(AssertionError) as exc_info:
        ragmortem_assert(bad_trace, reference_answer="15 minutes")

    msg = str(exc_info.value)
    assert "RAG Assertion Failed" in msg
    assert "GENERATION_SUSPECTED" in msg or "GENERATION_IGNORED_CONTEXT" in msg or "GENERATION" in msg


def test_ragmortem_assert_success_passes() -> None:
    """Verifies that ragmortem_assert returns DiagnosticResult on valid RAG execution."""
    from ragmortem.pytest_plugin import ragmortem_assert

    good_trace = {
        "query": "What is the mandatory engineering response time for a P1 incident?",
        "retrieved_chunks": [
            {"id": "c1", "text": "P1 SLA is 15 minutes.", "score": 0.88}
        ],
        "answer": "P1 SLA is 15 minutes.",
    }

    res = ragmortem_assert(good_trace, reference_answer="15 minutes")
    assert res.diagnosis.value.lower() == "no_failure"


def test_ragmortem_assert_fixture_in_pytester(pytester: pytest.Pytester) -> None:
    """Verifies that ragmortem_assert fixture works inside pytest test cases."""
    pytester.makepyfile(
        """
        def test_rag_via_fixture(ragmortem_assert):
            trace = {
                "query": "What is the SLA?",
                "retrieved_chunks": [{"id": "c1", "text": "SLA is 15 minutes.", "score": 0.88}],
                "answer": "48 hours",
            }
            ragmortem_assert(trace, reference_answer="15 minutes")
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    stdout = result.stdout.str()
    assert "RAG Assertion Failed" in stdout
    assert "RAGmortem Diagnosis" in stdout

