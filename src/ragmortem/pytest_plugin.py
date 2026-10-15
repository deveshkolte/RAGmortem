"""pytest-ragmortem: Automatic RAG failure diagnosis in pytest and GitHub Actions.

When a RAG unit test fails, pytest-ragmortem intercepts the failure, runs
root-cause attribution on the execution telemetry, and prints a formatted
Rich diagnostic failure card directly in your terminal and GitHub Actions CI.

Usage in tests:
    def test_sla_answer(rag_trace):
        res = my_rag.query("What is the SLA?")
        rag_trace.record(
            query="What is the SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.85}],
            answer=res.answer,
        )
        assert "15 minutes" in res.answer
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Generator

import pytest

from ragmortem.api import diagnose
from ragmortem.diagnose import DiagnosticResult
from ragmortem.taxonomy import FailureType
from ragmortem.trace import Trace, trace


def ragmortem_assert(
    trace_input: Trace | dict[str, Any] | Any,
    corpus: Any | None = None,
    reference_answer: str | None = None,
    mode: str | None = None,
    profile: str = "minilm",
    **kwargs: Any,
) -> DiagnosticResult:
    """Diagnose a RAG execution trace and assert no failure occurred.

    If a failure is diagnosed (i.e. diagnosis.canonical != FailureType.NO_FAILURE),
    formats the Rich diagnostic failure card and raises an AssertionError with the detailed report.
    """
    diag_res = diagnose(
        trace_input=trace_input,
        corpus=corpus,
        reference_answer=reference_answer,
        mode=mode,
        profile=profile,
        **kwargs,
    )
    if diag_res.diagnosis.canonical != FailureType.NO_FAILURE:
        buf = io.StringIO()
        try:
            from rich.console import Console
            from ragmortem.terminal import render_diagnostic_card
            console = Console(file=buf, force_terminal=True, width=100)
            console.print()
            render_diagnostic_card(
                diag_res,
                title=f"RAGmortem Diagnosis: {diag_res.diagnosis.value.upper()}",
                console=console,
            )
            card_output = buf.getvalue()
        except Exception:
            card_output = f"\nRAGmortem Diagnosis: {diag_res.to_text()}\n"

        reasons = "; ".join(diag_res.reasons) if diag_res.reasons else diag_res.explanation
        msg = (
            f"RAG Assertion Failed: [{diag_res.diagnosis.value.upper()}] (evidence={diag_res.evidence_score:.2f})\n"
            f"Reason: {reasons}\n"
            f"{card_output}"
        )
        raise AssertionError(msg)
    return diag_res


@pytest.fixture(name="ragmortem_assert")
def _ragmortem_assert_fixture() -> Any:
    """Pytest fixture providing the ragmortem_assert assertion helper."""
    return ragmortem_assert



class RAGTraceRecorder:
    """Fixture helper allowing unit tests to record RAG execution telemetry."""

    def __init__(self) -> None:
        self.current_trace: Trace | None = None
        self.reference_answer: str | None = None
        self.corpus: Any | None = None
        self.mode: str | None = None

    def record(
        self,
        query: str = "",
        retrieved_chunks: list[Any] | None = None,
        scores: list[float] | None = None,
        answer: str = "",
        reference_answer: str | None = None,
        corpus: Any | None = None,
        mode: str | None = None,
        score_type: str = "similarity",
        **metadata: Any,
    ) -> Trace:
        """Record telemetry for the current test case."""
        t = trace(
            query=query,
            retrieved_chunks=retrieved_chunks or [],
            scores=scores,
            answer=answer,
            score_type=score_type,
            metadata=metadata,
        )
        self.current_trace = t
        if reference_answer is not None:
            self.reference_answer = reference_answer
        if corpus is not None:
            self.corpus = corpus
        if mode is not None:
            self.mode = mode
        return t

    def set_trace(self, t: Trace) -> None:
        """Set a pre-constructed Trace object."""
        self.current_trace = t


_ACTIVE_RECORDER: RAGTraceRecorder | None = None


class record_trace:
    """Context manager for recording RAG telemetry in tests or scripts."""

    def __init__(
        self,
        query: str = "",
        retrieved_chunks: list[Any] | None = None,
        scores: list[float] | None = None,
        answer: str = "",
        reference_answer: str | None = None,
        corpus: Any | None = None,
        mode: str | None = None,
        score_type: str = "similarity",
        **metadata: Any,
    ) -> None:
        self.recorder = RAGTraceRecorder()
        self.recorder.record(
            query=query,
            retrieved_chunks=retrieved_chunks,
            scores=scores,
            answer=answer,
            reference_answer=reference_answer,
            corpus=corpus,
            mode=mode,
            score_type=score_type,
            **metadata,
        )

    def __enter__(self) -> RAGTraceRecorder:
        global _ACTIVE_RECORDER
        self._prev = _ACTIVE_RECORDER
        _ACTIVE_RECORDER = self.recorder
        return self.recorder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _ACTIVE_RECORDER
        _ACTIVE_RECORDER = self._prev


@pytest.fixture
def rag_trace(request: pytest.FixtureRequest) -> Generator[RAGTraceRecorder, None, None]:
    """Pytest fixture providing a RAGTraceRecorder for root-cause diagnosis on failure."""
    recorder = RAGTraceRecorder()
    request.node._ragmortem_recorder = recorder
    yield recorder


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for pytest-ragmortem."""
    group = parser.getgroup("ragmortem", "RAGmortem root-cause failure diagnosis")
    group.addoption(
        "--no-ragmortem",
        action="store_true",
        default=False,
        help="Disable automatic RAGmortem root-cause diagnosis on test failure.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "ragmortem: mark test for automatic RAG failure diagnosis on assertion error",
    )


def _emit_github_actions_annotation(item: pytest.Item, result: DiagnosticResult) -> None:
    """Emit GitHub Actions workflow command and write step summary if in CI."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return

    diag_name = result.diagnosis.value.upper()
    ev_score = result.evidence_score
    reasons_str = "; ".join(result.reasons[:2]) if result.reasons else "Unknown issue"
    msg = f"Root-Cause: {diag_name} (Evidence: {ev_score:.2f}) - {reasons_str}"

    location = getattr(item, "location", None)
    file_path = location[0] if location else "test"
    line_num = (location[1] + 1) if location and len(location) > 1 and location[1] is not None else 1

    # 1. Workflow command annotation
    print(
        f"\n::error file={file_path},line={line_num},title=RAGmortem Diagnosis [{diag_name}]::{msg}",
        file=sys.stderr,
    )

    # 2. GitHub Actions Step Summary markdown
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        try:
            summary_p = Path(step_summary_path)
            md_lines = [
                "",
                "### 🔬 RAGmortem Failure Diagnosis",
                f"- **Failed Test**: `{item.name}`",
                f"- **Diagnosis**: `{diag_name}`",
                f"- **Evidence Score**: `{ev_score:.2f}`",
                f"- **Mode**: `{result.mode}`",
                "",
                "#### Reasons & Telemetry",
            ]
            for r in result.reasons:
                md_lines.append(f"- {r}")
            for ev in result.evidence:
                md_lines.append(f"- *Evidence*: {ev}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

            with open(summary_p, "a", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
        except Exception:
            pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, None]:
    """Hook invoked when a test report is generated; diagnoses on assertion failure."""
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or not report.failed:
        return

    # Check if disabled via CLI flag
    config = getattr(item, "config", None)
    if config and config.getoption("--no-ragmortem", False):
        return

    # Retrieve recorded trace from fixture, node attribute, or global recorder
    recorder = getattr(item, "_ragmortem_recorder", None) or _ACTIVE_RECORDER
    if not recorder or not recorder.current_trace:
        return

    t = recorder.current_trace

    # Run RAGmortem root-cause diagnosis
    try:
        diag_res = diagnose(
            trace_input=t,
            corpus=recorder.corpus,
            reference_answer=recorder.reference_answer,
            mode=recorder.mode,
        )
    except Exception as exc:
        diag_res = None

    if not diag_res:
        return

    # Render Rich diagnostic failure card to terminal
    buf = io.StringIO()
    try:
        from rich.console import Console
        from ragmortem.terminal import render_diagnostic_card
        console = Console(file=buf, force_terminal=True, width=100)
        console.print()
        render_diagnostic_card(
            diag_res,
            title=f"RAGmortem Test Failure Diagnosis: {item.name}",
            console=console,
        )
        card_output = buf.getvalue()
    except Exception:
        card_output = f"\nRAGmortem Diagnosis: {diag_res.to_text()}\n"

    # Append to longrepr / stderr
    if hasattr(report, "longreprtext"):
        if "RAGmortem" not in report.longreprtext and "RAG Assertion Failed" not in report.longreprtext:
            report.longrepr = f"{report.longreprtext}\n{card_output}"
    else:
        sys.stderr.write(card_output)

    # Emit CI step summary and annotations
    _emit_github_actions_annotation(item, diag_res)
