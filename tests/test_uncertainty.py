"""Comprehensive unit tests for uncertainty-aware diagnoser.

Covers the 15 required scenarios from Day 7 specification:
1. missing scores
2. missing candidate IDs
3. missing top_k
4. missing final context
5. conflicting signals
6. low score but strong relative evidence
7. high score but weak evidence
8. empty retrieval
9. ranking boundary
10. unknown result
11. deterministic evidence score
12. trace-only mode never accesses corpus
13. corpus-aware mode explicitly opts into corpus
14. composite ambiguity
15. malformed trace
"""

import pytest
from unittest.mock import MagicMock
from ragmortem.diagnose import (
    RAGDiagnoser,
    ObservedExecution,
    DiagnosticResult,
    TelemetryCompleteness,
    EvidenceModel,
    diagnose,
)
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk


# 1. Missing scores
def test_missing_scores():
    trace = ObservedExecution(
        question="What is the retention period?",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        generated_answer="The retention period is 5 years.",
        scores=None,
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    assert res.telemetry_completeness == TelemetryCompleteness.INSUFFICIENT.value
    assert any("scores are missing" in ev.lower() for ev in res.evidence)


# 2. Missing candidate IDs
def test_missing_candidate_ids():
    trace = ObservedExecution(
        question="What is the retention period?",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        generated_answer="The retention period is 5 years.",
        scores=[0.85, 0.72],
        candidate_chunk_ids=None,
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.telemetry_completeness == TelemetryCompleteness.PARTIAL.value
    assert any("candidate pool" in lim.lower() for lim in res.limitations)


# 3. Missing / custom top_k
def test_missing_top_k():
    trace = ObservedExecution(
        question="What is the retention period?",
        retrieved_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        generated_answer="The retention period is 5 years.",
        scores=[0.85, 0.72, 0.65],
    )
    assert trace.top_k == 3
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")
    assert res.diagnosis in [FailureType.GENERATION_SUSPECTED, FailureType.UNKNOWN]


# 4. Missing final context
def test_missing_final_context():
    trace = ObservedExecution(
        question="What is the retention period?",
        retrieved_chunk_ids=[],
        generated_answer="I don't know.",
        scores=[],
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    assert res.telemetry_completeness == TelemetryCompleteness.INSUFFICIENT.value


# 5. Conflicting signals
def test_conflicting_signals():
    # Candidate pool extended (suggesting ranking candidate was demoted), but scores are near zero (<0.47)
    trace = ObservedExecution(
        question="What is the protocol?",
        retrieved_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        generated_answer="The protocol requires annual review.",
        candidate_chunk_ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
        scores=[0.20, 0.15, 0.10],
        candidate_scores=[0.20, 0.15, 0.10, 0.09, 0.08],
        top_k=3,
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    assert any("conflicting" in reason.lower() or "multiple plausible" in reason.lower() for reason in res.reasons)


# 6. Low score but strong relative evidence
def test_low_score_but_strong_relative_evidence():
    # Runtime retriever returned low score (0.25), but corpus audit shows clean separation (score 0.65, gap 0.20)
    mock_app = MagicMock()
    mock_app.retrieve.return_value = (
        [
            Chunk(id="chunk_gold", text="The answer is 10 days.", source="doc.md"),
            Chunk(id="chunk_2", text="other", source="doc.md"),
            Chunk(id="chunk_3", text="third", source="doc.md"),
        ],
        [0.65, 0.50, 0.45],
    )
    trace = ObservedExecution(
        question="What is the statutory deadline?",
        retrieved_chunk_ids=["chunk_irrelevant_1", "chunk_irrelevant_2", "chunk_irrelevant_3"],
        generated_answer="The statutory deadline is unknown.",
        scores=[0.25, 0.20, 0.18],
    )
    diagnoser = RAGDiagnoser(app=mock_app)
    res = diagnoser.diagnose(trace, mode="corpus_aware")

    assert res.diagnosis == FailureType.RETRIEVAL_SUSPECTED
    assert res.evidence_score > 0.70
    assert any("corpus audit" in ev.lower() for ev in res.evidence)


# 7. High score but weak evidence (Borderline Ambiguity band)
def test_high_score_but_weak_evidence():
    # Top score is 0.50, sitting squarely in ambiguity band [0.47, 0.53]
    trace = ObservedExecution(
        question="What is the safety threshold?",
        retrieved_chunk_ids=["chunk_borderline_1", "chunk_borderline_2"],
        generated_answer="The safety threshold depends on clinical discretion.",
        scores=[0.50, 0.48],
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    assert any("ambiguity band" in ev.lower() for ev in res.evidence)


# 8. Empty retrieval
def test_empty_retrieval():
    trace = ObservedExecution(
        question="Can patients withdraw consent?",
        retrieved_chunk_ids=[],
        generated_answer="Yes they can.",
        scores=[],
    )
    res = diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.telemetry_completeness == TelemetryCompleteness.INSUFFICIENT.value


# 9. Ranking boundary
def test_ranking_boundary():
    # Strong candidate immediately below cutoff (gap: 0.02 <= 0.08) with top_score >= 0.50
    trace = ObservedExecution(
        question="What are the phase 2 transition criteria?",
        retrieved_chunk_ids=["c1", "c2", "c3"],
        generated_answer="Phase 2 requires proof of efficacy.",
        candidate_chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        scores=[0.75, 0.70, 0.68],
        candidate_scores=[0.75, 0.70, 0.68, 0.66, 0.60],
        top_k=3,
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.RANKING_SUSPECTED
    assert res.evidence_score >= 0.75
    assert any("ranking" in r.lower() for r in res.reasons)


# 10. Unknown result
def test_unknown_result():
    trace = ObservedExecution(
        question="What is the protocol?",
        retrieved_chunk_ids=["c1"],
        generated_answer="Answer.",
        scores=[0.51],  # Inside ambiguity band
    )
    res = diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    assert len(res.reasons) > 0
    assert len(res.limitations) > 0


# 11. Deterministic evidence score
def test_deterministic_evidence_score():
    trace = ObservedExecution(
        question="What are the adverse event criteria?",
        retrieved_chunk_ids=["c1", "c2", "c3"],
        generated_answer="Adverse events must be logged.",
        scores=[0.82, 0.74, 0.70],
    )
    diagnoser = RAGDiagnoser()
    scores = [diagnoser.diagnose(trace, mode="trace_only").evidence_score for _ in range(10)]

    assert len(set(scores)) == 1
    assert scores[0] > 0.0


# 12. Trace-only mode never accesses corpus
def test_trace_only_mode_never_accesses_corpus():
    mock_app = MagicMock()
    trace = ObservedExecution(
        question="What is the report timeline?",
        retrieved_chunk_ids=["c1", "c2"],
        generated_answer="The timeline is 30 days.",
        scores=[0.20, 0.15],
    )
    diagnoser = RAGDiagnoser(app=mock_app)
    res = diagnoser.diagnose(trace, mode="trace_only")

    mock_app.retrieve.assert_not_called()
    assert res.diagnosis == FailureType.UNKNOWN


# 13. Corpus-aware mode explicitly opts into corpus
def test_corpus_aware_mode_explicitly_opts_into_corpus():
    # 13a: Corpus provided -> audits corpus
    mock_app = MagicMock()
    mock_app.retrieve.return_value = ([Chunk(id="c1", text="t", source="doc.md")], [0.80])
    trace = ObservedExecution(
        question="What is the report timeline?",
        retrieved_chunk_ids=["c_bad"],
        generated_answer="The timeline is 30 days.",
        scores=[0.20],
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="corpus_aware", corpus=mock_app)
    mock_app.retrieve.assert_called_once()
    assert res.diagnosis == FailureType.RETRIEVAL_SUSPECTED

    # 13b: Corpus-aware requested but corpus is None -> returns UNKNOWN
    diagnoser_no_corpus = RAGDiagnoser(app=None)
    res_no_corpus = diagnoser_no_corpus.diagnose(trace, mode="corpus_aware")
    assert res_no_corpus.diagnosis == FailureType.UNKNOWN
    assert any("corpus index unavailable" in lim.lower() for lim in res_no_corpus.limitations)


# 14. Composite ambiguity
def test_composite_ambiguity():
    # Candidate truncation present (potential ranking miss) alongside low scores (potential retrieval miss)
    trace = ObservedExecution(
        question="How many days to report?",
        retrieved_chunk_ids=["c1", "c2"],
        generated_answer="I don't know.",
        candidate_chunk_ids=["c1", "c2", "c3", "c4"],
        scores=[0.30, 0.25],
        candidate_scores=[0.30, 0.25, 0.24, 0.20],
        top_k=2,
    )
    diagnoser = RAGDiagnoser()
    res = diagnoser.diagnose(trace, mode="trace_only")

    assert res.diagnosis == FailureType.UNKNOWN
    assert any("multiple" in r.lower() or "conflicting" in r.lower() or "ambiguous" in r.lower() for r in res.reasons)


# 15. Malformed trace
def test_malformed_trace():
    # 15a: Empty question
    trace_no_q = ObservedExecution(
        question="",
        retrieved_chunk_ids=["c1"],
        generated_answer="Some answer",
        scores=[0.80],
    )
    res_q = diagnose(trace_no_q)
    assert res_q.diagnosis == FailureType.UNKNOWN
    assert res_q.telemetry_completeness == TelemetryCompleteness.INSUFFICIENT.value

    # 15b: Empty answer
    trace_no_a = ObservedExecution(
        question="Valid question?",
        retrieved_chunk_ids=["c1"],
        generated_answer="",
        scores=[0.80],
    )
    res_a = diagnose(trace_no_a)
    assert res_a.diagnosis == FailureType.UNKNOWN
    assert res_a.telemetry_completeness == TelemetryCompleteness.INSUFFICIENT.value

    # 15c: Dictionary input parsing
    dict_trace = {
        "question": "Valid query?",
        "retrieved_chunk_ids": ["c1"],
        "generated_answer": "Valid answer",
        "scores": [0.85],
    }
    res_dict = diagnose(dict_trace, mode="trace_only")
    assert res_dict.diagnosis == FailureType.GENERATION_SUSPECTED
