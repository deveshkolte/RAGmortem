"""Tests for RAGmortem public developer integration API, Trace, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import ragmortem
from ragmortem import Chunk, FailureType, RagResult, diagnose, trace
from ragmortem.diagnose import DiagnosticResult
from ragmortem.trace import Trace
from examples.reference_rag.app import ReferenceRagApp


DOCS_DIR = Path(__file__).resolve().parent.parent / "examples" / "reference_rag" / "documents"


@pytest.fixture
def reference_app() -> ReferenceRagApp:
    return ReferenceRagApp(corpus_dir=DOCS_DIR, mock_mode=True, top_k=3)


def test_public_api_exports() -> None:
    """Verify all intended public symbols are exported at package root."""
    assert hasattr(ragmortem, "diagnose")
    assert hasattr(ragmortem, "trace")
    assert hasattr(ragmortem, "Trace")
    assert hasattr(ragmortem, "Chunk")
    assert hasattr(ragmortem, "RagResult")
    assert hasattr(ragmortem, "DiagnosticResult")
    assert hasattr(ragmortem, "FailureType")
    assert hasattr(ragmortem, "__version__")


def test_trace_construction_and_serialization() -> None:
    """Verify Trace handles flexible chunk types, metadata, and JSON roundtrip."""
    t = trace(
        query="What is P1 SLA?",
        retrieved_chunks=[
            {"id": "doc1", "text": "15 minutes response time", "score": 0.82},
            {"id": "doc2", "text": "Unrelated guidelines", "score": 0.31},
        ],
        answer="15 minutes",
        context="[1] 15 minutes response time\n[2] Unrelated guidelines",
        model="qwen-2.5",
        latency_ms=245.5,
        metadata={"user_id": "test_user"},
    )

    assert t.query == "What is P1 SLA?"
    assert t.question == "What is P1 SLA?"
    assert t.answer == "15 minutes"
    assert t.generated_answer == "15 minutes"
    assert t.retrieved_chunk_ids == ["doc1", "doc2"]
    assert t.get_scores() == [0.82, 0.31]
    assert t.context is not None

    # Serialization roundtrip
    data = t.to_dict()
    assert data["query"] == "What is P1 SLA?"
    assert data["scores"] == [0.82, 0.31]
    assert data["model"] == "qwen-2.5"
    assert data["latency_ms"] == 245.5

    t_loaded = Trace.from_dict(data)
    assert t_loaded.query == t.query
    assert t_loaded.retrieved_chunk_ids == t.retrieved_chunk_ids
    assert t_loaded.get_scores() == t.get_scores()


def test_trace_only_retrieval_miss_produces_unknown() -> None:
    """In trace-only mode, weak retrieval cannot determine answerability and must yield UNKNOWN."""
    t = trace(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[{"id": "c1", "text": "Quarterly disaster recovery exercises", "score": 0.14}],
        scores=[0.14],
        answer="Quarterly drills run every 3 months.",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.UNKNOWN
    assert result.evidence_score == 0.0
    assert any("Corpus index unavailable in trace-only mode" in lim for lim in result.limitations)
    assert "remediation" in result.recommended_action or "telemetry" in result.recommended_action


def test_corpus_aware_retrieval_miss(reference_app: ReferenceRagApp) -> None:
    """Corpus-aware mode detects retrieval miss when indexed docs exist but runtime score is low."""
    t = trace(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[{"id": "chunk_dr_drill_frequency", "text": "Disaster recovery drills", "score": 0.14}],
        scores=[0.14],
        answer="Quarterly drills run every 3 months.",
    )

    result = diagnose(t, corpus=reference_app, mode="corpus_aware")
    assert result.failure_type == FailureType.RETRIEVAL_SUSPECTED
    assert result.canonical_failure_type == FailureType.RETRIEVAL_MISS
    assert result.evidence_score > 0.70
    assert "retrieval recall" in result.recommended_action.lower()


def test_ranking_miss_detected() -> None:
    """Cutoff gap probe identifies high-scoring candidates omitted from top-k window."""
    candidates = [
        {"id": "c1", "text": "A", "score": 0.60},
        {"id": "c2", "text": "B", "score": 0.58},
        {"id": "c3", "text": "C", "score": 0.57},
        {"id": "c4", "text": "D", "score": 0.56},
    ]
    t = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=candidates[:3],
        scores=[0.60, 0.58, 0.57],
        candidate_chunks=candidates,
        candidate_scores=[0.60, 0.58, 0.57, 0.56],
        top_k=3,
        answer="Wrong answer",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.RANKING_SUSPECTED
    assert result.canonical_failure_type == FailureType.RANKING_MISS
    assert result.evidence_score >= 0.70
    assert "re-ranking" in result.recommended_action.lower()


def test_generation_ignored_context_detected() -> None:
    """High retrieval score in context with substantive answer attributes generation failure."""
    t = trace(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[{"id": "chunk_p1", "text": "P1 SLA is 15 minutes", "score": 0.85}],
        scores=[0.85],
        answer="The SLA is 48 hours.",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.GENERATION_SUSPECTED
    assert result.canonical_failure_type == FailureType.GENERATION_IGNORED_CONTEXT
    assert result.evidence_score > 0.70
    assert "generator prompt" in result.recommended_action.lower()


def test_should_abstain_detected(reference_app: ReferenceRagApp) -> None:
    """Corpus-aware mode detects should_abstain on unanswerable query with substantive answer."""
    t = trace(
        query="What is the procedure for resetting an HSM hardware module?",
        retrieved_chunks=[{"id": "chunk_dr_drill_frequency", "text": "DR exercises", "score": 0.15}],
        scores=[0.15],
        answer="Press the reset pin.",
    )

    result = diagnose(t, corpus=reference_app, mode="corpus_aware")
    assert result.failure_type == FailureType.ABSTENTION_SUSPECTED
    assert result.canonical_failure_type == FailureType.SHOULD_ABSTAIN
    assert result.evidence_score > 0.70
    assert "abstain" in result.recommended_action.lower()


def test_successful_trace_no_failure(reference_app: ReferenceRagApp) -> None:
    """Verify NO_FAILURE attribution under reference verification and appropriate abstention."""
    # A. Reference verification
    t_ref = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=[{"id": "chunk_incident_severity_p1", "text": "15 minutes", "score": 0.85}],
        scores=[0.85],
        answer="The response time is 15 minutes.",
    )
    res_ref = diagnose(t_ref, reference_answer="15 minutes", corpus=reference_app)
    assert res_ref.failure_type == FailureType.NO_FAILURE
    assert res_ref.evidence_score == 1.0
    assert "No corrective action" in res_ref.recommended_action

    # B. Appropriate abstention
    t_abstain = trace(
        query="What is the ergonomic chair reimbursement policy?",
        retrieved_chunks=[{"id": "doc1", "text": "Unrelated", "score": 0.12}],
        scores=[0.12],
        answer="I do not have sufficient information in the provided context to answer this question.",
    )
    res_abstain = diagnose(t_abstain, mode="trace_only")
    assert res_abstain.failure_type == FailureType.NO_FAILURE
    assert res_abstain.evidence_score >= 0.80


def test_missing_scores_produces_unknown_not_fabricated() -> None:
    """Missing scores MUST yield UNKNOWN and never be fabricated."""
    t = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=[{"id": "doc1", "text": "P1 SLA is 15 minutes"}],
        scores=None,
        answer="15 minutes",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.UNKNOWN
    assert result.evidence_score == 0.0
    assert any("Missing similarity scores" in r for r in result.reasons)


def test_empty_retrieved_chunks_produces_unknown() -> None:
    """Empty retrieved chunks must yield UNKNOWN without fabrication."""
    t = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=[],
        scores=[],
        answer="15 minutes",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.UNKNOWN
    assert result.evidence_score == 0.0
    assert any("Empty context window" in r for r in result.reasons)


def test_missing_answer_produces_unknown() -> None:
    """Empty answer string must yield UNKNOWN without guessing."""
    t = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=[{"id": "doc1", "text": "Some text", "score": 0.75}],
        scores=[0.75],
        answer="",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.UNKNOWN
    assert result.evidence_score == 0.0
    assert any("Missing generated answer" in r for r in result.reasons)


def test_missing_query_produces_unknown() -> None:
    """Empty query string must yield UNKNOWN."""
    t = trace(
        query="   ",
        retrieved_chunks=[{"id": "doc1", "text": "Some text", "score": 0.75}],
        scores=[0.75],
        answer="An answer",
    )

    result = diagnose(t, mode="trace_only")
    assert result.failure_type == FailureType.UNKNOWN
    assert result.evidence_score == 0.0
    assert any("Missing user query" in r for r in result.reasons)


def test_malformed_trace_handling() -> None:
    """Passing invalid input types or bad mode raises appropriate error."""
    with pytest.raises(TypeError):
        diagnose(12345)  # type: ignore

    with pytest.raises(ValueError):
        diagnose(
            query="test",
            retrieved_chunks=[{"id": "1", "text": "t", "score": 0.5}],
            answer="a",
            mode="invalid_mode",
        )


def test_deterministic_repeated_diagnosis(reference_app: ReferenceRagApp) -> None:
    """Running diagnosis multiple times produces bit-for-bit identical results."""
    t = trace(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[{"id": "chunk_dr_drill_frequency", "text": "DR drills", "score": 0.14}],
        scores=[0.14],
        answer="Quarterly drills run every 3 months.",
    )

    runs = [diagnose(t, corpus=reference_app, mode="corpus_aware") for _ in range(5)]
    first_dict = runs[0].to_dict()
    for r in runs[1:]:
        assert r.to_dict() == first_dict


def test_explicit_mode_no_silent_switch(reference_app: ReferenceRagApp) -> None:
    """Modes must be respected strictly: no silent switching."""
    t = trace(
        query="What is the P1 SLA?",
        retrieved_chunks=[{"id": "c1", "text": "DR exercises", "score": 0.14}],
        scores=[0.14],
        answer="Quarterly drills",
    )

    # 1. Trace-only with corpus provided must STILL NOT use corpus
    res_trace = diagnose(t, corpus=reference_app, mode="trace_only")
    assert res_trace.mode == "trace_only"
    assert res_trace.failure_type == FailureType.UNKNOWN

    # 2. Corpus-aware without corpus must NOT fall back to trace-only, but return UNKNOWN with missing corpus
    res_corpus_none = diagnose(t, corpus=None, mode="corpus_aware")
    assert res_corpus_none.mode == "corpus_aware"
    assert res_corpus_none.failure_type == FailureType.UNKNOWN
    assert any("Corpus index unavailable" in lim for lim in res_corpus_none.limitations)


def test_cli_diagnose_trace_command(tmp_path: Path) -> None:
    """Test CLI diagnose-trace command execution."""
    from ragmortem.cli import main

    trace_file = tmp_path / "sample_trace.json"
    trace_data = {
        "query": "What is the P1 SLA?",
        "retrieved_chunks": [
            {"id": "c1", "text": "Sample text", "score": 0.85}
        ],
        "answer": "12 hours instead of 15 minutes",
    }
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

    # Run CLI with human-readable output
    exit_code = main(["diagnose-trace", str(trace_file), "--mode", "trace_only"])
    assert exit_code == 0

    # Run CLI with --json output
    exit_code_json = main(["diagnose-trace", str(trace_file), "--mode", "trace_only", "--json"])
    assert exit_code_json == 0
