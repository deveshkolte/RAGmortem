"""Unit tests for OpenTelemetry, Phoenix, and LangSmith trace importers."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from ragmortem.cli import main
from ragmortem.importers import (
    detect_format,
    import_langsmith_trace,
    import_phoenix_trace,
    import_trace,
)
from ragmortem.taxonomy import FailureType


def test_detect_format() -> None:
    # Phoenix / OpenInference
    phoenix_data = [
        {"name": "Retriever", "attributes": {"openinference.span.kind": "RETRIEVER"}}
    ]
    assert detect_format(phoenix_data) == "phoenix"

    # LangSmith run
    langsmith_data = {
        "run_type": "chain",
        "inputs": {"query": "What is the SLA?"},
        "child_runs": [{"run_type": "retriever", "outputs": {"documents": []}}],
    }
    assert detect_format(langsmith_data) == "langsmith"

    # Native RAGmortem trace
    native_data = {
        "query": "What is the SLA?",
        "retrieved_chunks": [{"id": "c1", "text": "...", "score": 0.85}],
        "answer": "15 min",
    }
    assert detect_format(native_data) == "native"


def test_import_phoenix_trace() -> None:
    phoenix_spans = [
        {
            "name": "RAG Chain",
            "attributes": {
                "openinference.span.kind": "CHAIN",
                "input.value": "What is the mandatory engineering response time for a P1 incident?",
                "output.value": "Engineers have 48 hours to acknowledge P1 alerts.",
            },
        },
        {
            "name": "VectorRetriever",
            "attributes": {
                "openinference.span.kind": "RETRIEVER",
                "retrieval.documents": [
                    {
                        "document.id": "doc_p1_incident",
                        "document.content": "P1 incidents mandate 15-minute engineering response.",
                        "document.score": 0.89,
                        "document.metadata": {"source": "incident_policy.md"},
                    },
                    {
                        "document.id": "doc_dr_drill",
                        "document.content": "Disaster recovery failover drills occur quarterly.",
                        "document.score": 0.35,
                    },
                ],
            },
        },
    ]

    t = import_phoenix_trace(phoenix_spans)
    assert t.query == "What is the mandatory engineering response time for a P1 incident?"
    assert len(t.retrieved_chunks) == 2
    assert t.retrieved_chunks[0]["id"] == "doc_p1_incident"
    assert t.scores == [0.89, 0.35]
    assert t.answer == "Engineers have 48 hours to acknowledge P1 alerts."

    # Diagnosing imported Phoenix trace: high relevance context (0.89) ignored -> GENERATION_SUSPECTED
    from ragmortem.api import diagnose
    res = diagnose(t)
    assert res.diagnosis == FailureType.GENERATION_SUSPECTED


def test_import_langsmith_trace() -> None:
    langsmith_run = {
        "id": "run-root-123",
        "run_type": "chain",
        "inputs": {"question": "What is the excess cash flow sweep percentage?"},
        "outputs": {"output": "The excess cash flow sweep is 25%."},
        "child_runs": [
            {
                "id": "run-retriever-456",
                "run_type": "retriever",
                "inputs": {"query": "What is the excess cash flow sweep percentage?"},
                "outputs": {
                    "documents": [
                        {
                            "page_content": "Excess cash flow sweep is 50% when First Lien Net Leverage exceeds 3.50x.",
                            "metadata": {"id": "chunk_credit_sweep", "score": 0.86},
                        }
                    ]
                },
            },
            {
                "id": "run-llm-789",
                "run_type": "llm",
                "outputs": {
                    "generations": [[{"text": "The excess cash flow sweep is 25%."}]]
                },
            },
        ],
    }

    t = import_langsmith_trace(langsmith_run)
    assert t.query == "What is the excess cash flow sweep percentage?"
    assert len(t.retrieved_chunks) == 1
    assert t.retrieved_chunks[0]["id"] == "chunk_credit_sweep"
    assert t.scores == [0.86]
    assert t.answer == "The excess cash flow sweep is 25%."

    from ragmortem.api import diagnose
    res = diagnose(t)
    assert res.diagnosis == FailureType.GENERATION_SUSPECTED


def test_cli_import_command(tmp_path: Path) -> None:
    phoenix_data = [
        {
            "name": "Retriever",
            "attributes": {
                "openinference.span.kind": "RETRIEVER",
                "input.value": "What is the SLA?",
                "retrieval.documents": [
                    {"document.id": "d1", "document.content": "SLA is 15m.", "document.score": 0.90}
                ],
            },
        },
        {
            "name": "LLM",
            "attributes": {
                "openinference.span.kind": "LLM",
                "output.value": "Wrong answer.",
            },
        },
    ]

    in_file = tmp_path / "phoenix_trace.json"
    in_file.write_text(json.dumps(phoenix_data), encoding="utf-8")
    out_file = tmp_path / "imported_trace.json"

    # 1. Test CLI import subcommand
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["import", str(in_file), "--out", str(out_file), "--diagnose"])
    assert code == 0
    assert out_file.exists()
    assert "generation_suspected" in buf.getvalue().lower()

    # 2. Test seamless auto-detection via ragmortem diagnose
    buf_diag = io.StringIO()
    with redirect_stdout(buf_diag):
        code_diag = main(["diagnose", str(in_file)])
    assert code_diag == 0
    assert "generation_suspected" in buf_diag.getvalue().lower()
