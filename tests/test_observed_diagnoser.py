"""Unit tests for the RAGmortem realistic observational failure diagnoser."""

import pytest

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.diagnose import ExecutionTrace, ObservedExecution, RAGDiagnoser
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk


@pytest.fixture
def sample_corpus() -> list[Chunk]:
    return [
        Chunk(
            id="chunk_incident_severity_p1",
            text="The mandatory engineering response SLA for P1 incidents is 15 minutes.",
            source="incident_doc.md",
        ),
        Chunk(
            id="chunk_dr_drill_frequency",
            text="Quarterly disaster recovery failover drills must be conducted every 3 months.",
            source="dr_doc.md",
        ),
        Chunk(
            id="chunk_incident_severity_p2",
            text="P2 response SLA is 1 hour from alert timestamp.",
            source="incident_doc.md",
        ),
        Chunk(
            id="chunk_incident_severity_p3",
            text="P3 defects must be acknowledged within 48 hours.",
            source="incident_doc.md",
        ),
    ]


@pytest.fixture
def diagnoser(sample_corpus: list[Chunk]) -> RAGDiagnoser:
    app = ReferenceRagApp(mock_mode=True)
    return RAGDiagnoser(app=app, corpus_chunks=sample_corpus)


def test_observed_execution_cannot_access_gold_labels() -> None:
    """Proves ObservedExecution data model does not contain gold labels."""
    obs = ObservedExecution(
        question="What is the P1 SLA?",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        generated_answer="Unknown",
    )

    # Must raise AttributeError if anyone tries to access benchmark gold labels
    with pytest.raises(AttributeError):
        _ = obs.gold_chunk_id  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _ = obs.gold_answer  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _ = obs.fault_type  # type: ignore[attr-defined]


def test_from_dict_sanitizes_injected_benchmark_labels() -> None:
    """Proves from_dict strips benchmark-only metadata and labels."""
    raw_data = {
        "question": "What is the P1 SLA?",
        "answer": "12 hours",
        "retrieved_chunk_ids": ["chunk_incident_severity_p1"],
        "scores": [0.85],
        # Injected leak attempts:
        "gold_chunk_id": "chunk_incident_severity_p1",
        "gold_answer": "15 minutes",
        "fault_type": "generation_ignored_context",
        "metadata": {
            "injected_fault": "generation_ignored_context",
            "environment": "production",
        },
    }

    obs = ObservedExecution.from_dict(raw_data)
    assert not hasattr(obs, "gold_chunk_id")
    assert not hasattr(obs, "gold_answer")
    assert not hasattr(obs, "fault_type")
    assert "injected_fault" not in obs.metadata
    assert obs.metadata.get("environment") == "production"


def test_diagnose_observed_ranking_suspected(diagnoser: RAGDiagnoser) -> None:
    """Observational trace with high-scoring candidate pushed outside top-k window."""
    obs = ObservedExecution(
        question="What is the mandatory engineering response time for a P1 incident?",
        generated_answer="P2 response SLA is 1 hour.",
        retrieved_chunk_ids=["chunk_incident_severity_p2", "chunk_incident_severity_p3"],
        candidate_chunk_ids=[
            "chunk_incident_severity_p2",
            "chunk_incident_severity_p3",
            "chunk_incident_severity_p1",  # rank 3
        ],
        scores=[0.80, 0.75, 0.70],
        top_k=2,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.RANKING_SUSPECTED
    assert res.diagnosis.canonical == FailureType.RANKING_MISS
    assert res.is_suspected is True
    assert res.candidate_count == 3
    assert any("Candidate retrieval pool contained 3 chunks" in ev for ev in res.evidence)


def test_diagnose_observed_generation_suspected(diagnoser: RAGDiagnoser) -> None:
    """Observational trace with high retrieval relevance in context but wrong/hallucinated answer."""
    obs = ObservedExecution(
        question="What is the mandatory engineering response time for a P1 incident?",
        generated_answer="The mandatory engineering response SLA for P1 incidents is 12 hours.",
        retrieved_chunk_ids=["chunk_incident_severity_p1"],
        scores=[0.75],
        top_k=1,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.GENERATION_SUSPECTED
    assert res.diagnosis.canonical == FailureType.GENERATION_IGNORED_CONTEXT
    assert res.is_suspected is True
    assert any("Retrieved context had high relevance score" in ev for ev in res.evidence)


def test_diagnose_observed_retrieval_suspected(diagnoser: RAGDiagnoser) -> None:
    """Observational trace where retriever returned irrelevant chunks but corpus has relevant docs."""
    obs = ObservedExecution(
        question="What is the mandatory engineering response time for a P1 incident?",
        generated_answer="Quarterly drills run every 3 months.",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        scores=[0.14],
        top_k=1,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.RETRIEVAL_SUSPECTED
    assert res.diagnosis.canonical == FailureType.RETRIEVAL_MISS
    assert res.is_suspected is True
    assert any("Corpus audit shows indexed documents exist" in ev for ev in res.evidence)


def test_diagnose_observed_abstention_suspected(diagnoser: RAGDiagnoser) -> None:
    """Observational trace where corpus lacks information but system hallucinated an answer."""
    obs = ObservedExecution(
        question="What is the procedure for resetting a hardware cryptographic security module (HSM)?",
        generated_answer="Press the zeroize pin behind the bezel.",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        scores=[0.20],
        top_k=1,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.ABSTENTION_SUSPECTED
    assert res.diagnosis.canonical == FailureType.SHOULD_ABSTAIN
    assert res.is_suspected is True
    assert any("Corpus audit confirms no documents in the index match" in ev for ev in res.evidence)


def test_diagnose_observed_no_failure_when_refusing_on_low_scores(diagnoser: RAGDiagnoser) -> None:
    """System correctly refused to answer when retrieval scores were low."""
    obs = ObservedExecution(
        question="What is the procedure for resetting an HSM?",
        generated_answer="I do not have sufficient information in the provided context to answer this question.",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        scores=[0.20],
        top_k=1,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.NO_FAILURE
    assert res.is_suspected is False
    assert any("correctly abstained" in ev for ev in res.evidence)


def test_diagnose_observed_unknown_when_no_corpus_index() -> None:
    """When corpus index is absent, ambiguous low retrieval scores yield UNKNOWN."""
    # Diagnoser without active app/corpus index
    offline_diagnoser = RAGDiagnoser(app=None, corpus_chunks=[])
    # Replace app retrieve to simulate completely detached trace evaluator
    offline_diagnoser.app = None

    obs = ObservedExecution(
        question="What is the procedure for resetting an HSM?",
        generated_answer="Press zeroize pin.",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        scores=[0.15],
        top_k=1,
    )

    res = offline_diagnoser.diagnose_observed(obs)
    assert res.diagnosis == FailureType.UNKNOWN
    assert res.confidence == 0.0
    assert any("Without access to the corpus index" in ev for ev in res.evidence)


def test_diagnose_reference_assisted_proven_generation(diagnoser: RAGDiagnoser) -> None:
    """Reference-assisted mode proves generation failure when context contains reference answer."""
    obs = ObservedExecution(
        question="What is the P1 SLA?",
        generated_answer="12 hours",
        retrieved_chunk_ids=["chunk_incident_severity_p1"],
        scores=[0.85],
        top_k=1,
    )

    res = diagnoser.diagnose_reference_assisted(obs, reference_answer="15 minutes")
    assert res.diagnosis == FailureType.GENERATION_PROVEN
    assert res.diagnosis.canonical == FailureType.GENERATION_IGNORED_CONTEXT
    assert res.is_suspected is False
    assert any("contains the reference answer" in ev for ev in res.evidence)


def test_oracle_mode_still_works(diagnoser: RAGDiagnoser) -> None:
    """Oracle mode remains intact and functions as experimental upper bound."""
    trace = ExecutionTrace(
        question="What is the mandatory engineering response time for a P1 incident?",
        generated_answer="Quarterly drills run every 3 months.",
        retrieved_chunk_ids=["chunk_dr_drill_frequency"],
        candidate_chunk_ids=["chunk_dr_drill_frequency"],
        gold_chunk_id="chunk_incident_severity_p1",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    res = diagnoser.diagnose_oracle(trace)
    assert res.diagnosis == FailureType.RETRIEVAL_MISS
    assert res.confidence == 1.0
    assert res.oracle_replay_correct is True


def test_observed_mode_is_deterministic(diagnoser: RAGDiagnoser) -> None:
    """Running observational diagnosis multiple times produces identical results."""
    obs = ObservedExecution(
        question="What is the mandatory engineering response time for a P1 incident?",
        generated_answer="P2 response SLA is 1 hour.",
        retrieved_chunk_ids=["chunk_incident_severity_p2", "chunk_incident_severity_p3"],
        candidate_chunk_ids=[
            "chunk_incident_severity_p2",
            "chunk_incident_severity_p3",
            "chunk_incident_severity_p1",
        ],
        scores=[0.80, 0.75, 0.70],
        top_k=2,
    )

    r1 = diagnoser.diagnose_observed(obs)
    r2 = diagnoser.diagnose_observed(obs)
    assert r1.diagnosis == r2.diagnosis
    assert r1.confidence == r2.confidence
    assert r1.evidence == r2.evidence
    assert r1.signals == r2.signals


def test_malformed_trace_fails_safely(diagnoser: RAGDiagnoser) -> None:
    """Trace with empty chunks and missing scores executes safely."""
    obs = ObservedExecution(
        question="",
        generated_answer="",
        retrieved_chunk_ids=[],
        scores=None,
        top_k=0,
    )

    res = diagnoser.diagnose_observed(obs)
    assert res.diagnosis in (FailureType.UNKNOWN, FailureType.ABSTENTION_SUSPECTED, FailureType.NO_FAILURE)
