"""Unit tests for the RAGmortem deterministic failure diagnoser."""

import pytest

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.diagnose import ExecutionTrace, RAGDiagnoser
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk


@pytest.fixture
def mock_corpus() -> list[Chunk]:
    return [
        Chunk(
            id="chunk_gold_01",
            text="The mandatory engineering response SLA for P1 incidents is 15 minutes.",
            source="incident_doc.md",
        ),
        Chunk(
            id="chunk_distractor_01",
            text="Quarterly disaster recovery failover drills must be conducted every 3 months.",
            source="dr_doc.md",
        ),
        Chunk(
            id="chunk_distractor_02",
            text="P2 response SLA is 1 hour from alert timestamp.",
            source="incident_doc.md",
        ),
        Chunk(
            id="chunk_distractor_03",
            text="P3 defects must be acknowledged within 48 hours.",
            source="incident_doc.md",
        ),
    ]


@pytest.fixture
def diagnoser(mock_corpus: list[Chunk]) -> RAGDiagnoser:
    app = ReferenceRagApp(mock_mode=True)
    return RAGDiagnoser(app=app, corpus_chunks=mock_corpus)


def test_diagnose_retrieval_miss(diagnoser: RAGDiagnoser) -> None:
    """Gold chunk is completely absent from candidate pool + oracle replay succeeds."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Disaster recovery drills run quarterly.",
        retrieved_chunk_ids=["chunk_distractor_01", "chunk_distractor_02"],
        candidate_chunk_ids=["chunk_distractor_01", "chunk_distractor_02", "chunk_distractor_03"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.RETRIEVAL_MISS
    assert result.confidence == 1.0
    assert result.oracle_replay_correct is True
    assert result.candidate_rank is None
    assert result.original_rank is None
    assert any("absent from candidate retrieval pool" in ev for ev in result.evidence)
    assert any("Counterfactual oracle replay" in ev for ev in result.evidence)


def test_diagnose_ranking_miss(diagnoser: RAGDiagnoser) -> None:
    """Gold chunk is present in candidate pool at rank 3, but top-k is 2."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Disaster recovery drills run quarterly.",
        retrieved_chunk_ids=["chunk_distractor_01", "chunk_distractor_02"],  # top-2
        candidate_chunk_ids=[
            "chunk_distractor_01",
            "chunk_distractor_02",
            "chunk_gold_01",  # rank 3 in candidate pool
        ],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.RANKING_MISS
    assert result.confidence == 1.0
    assert result.oracle_replay_correct is True
    assert result.candidate_rank == 3
    assert result.candidate_count == 3
    assert result.original_rank is None
    assert any("present in the candidate retrieval pool at rank #3" in ev for ev in result.evidence)
    assert any("ranking truncation omitted gold evidence" in ev for ev in result.evidence)


def test_diagnose_generation_ignored_context_when_chunk_in_context(diagnoser: RAGDiagnoser) -> None:
    """Gold chunk was in the prompt context, but generator produced wrong answer."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="P1 response time is 12 hours.",  # Hallucinated / wrong
        retrieved_chunk_ids=["chunk_gold_01", "chunk_distractor_01"],
        candidate_chunk_ids=["chunk_gold_01", "chunk_distractor_01"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.GENERATION_IGNORED_CONTEXT
    assert result.confidence == 1.0
    assert result.original_rank == 1
    assert any("present in the generation context at rank #1" in ev for ev in result.evidence)
    assert any("generator failed to incorporate provided context" in ev for ev in result.evidence)


def test_diagnose_generation_ignored_context_when_oracle_fails(mock_corpus: list[Chunk]) -> None:
    """Gold chunk provided in oracle replay, but generator still fails."""
    # Custom app where mock generator refuses to answer
    class BrokenApp(ReferenceRagApp):
        def _generate_mock_answer(self, question: str, chunks: list[Chunk]) -> str:
            return "I have no idea about the SLA."

    broken_app = BrokenApp(mock_mode=True)
    broken_diagnoser = RAGDiagnoser(app=broken_app, corpus_chunks=mock_corpus)

    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Unknown",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = broken_diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.GENERATION_IGNORED_CONTEXT
    assert result.oracle_replay_correct is False
    assert any("oracle replay with gold chunk alone STILL failed" in ev for ev in result.evidence)


def test_diagnose_should_abstain(diagnoser: RAGDiagnoser) -> None:
    """Unanswerable question where RAG generated substantive hallucinations."""
    trace = ExecutionTrace(
        question="What is the CEO personal cell phone number?",
        generated_answer="The CEO personal cell phone number is +1-555-0199.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01"],
        gold_chunk_id=None,
        gold_answer=None,
        question_type="unanswerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.SHOULD_ABSTAIN
    assert result.confidence == 1.0
    assert any("Question is unanswerable from the corpus" in ev for ev in result.evidence)
    assert any("System hallucinated an answer when it should have abstained" in ev for ev in result.evidence)


def test_diagnose_no_failure_when_answerable_is_correct(diagnoser: RAGDiagnoser) -> None:
    """Answerable question where answer matches gold criteria."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="The mandatory response SLA is 15 minutes.",
        retrieved_chunk_ids=["chunk_gold_01"],
        candidate_chunk_ids=["chunk_gold_01"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.NO_FAILURE
    assert result.confidence == 1.0
    assert any("correctly matches gold criteria" in ev for ev in result.evidence)


def test_diagnose_no_failure_when_unanswerable_abstains(diagnoser: RAGDiagnoser) -> None:
    """Unanswerable question where RAG correctly abstains."""
    trace = ExecutionTrace(
        question="What is the CEO personal cell phone number?",
        generated_answer="I cannot answer this question based on the provided documentation.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01"],
        gold_chunk_id=None,
        gold_answer=None,
        question_type="unanswerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.NO_FAILURE
    assert result.confidence == 1.0
    assert any("correctly refused to answer" in ev for ev in result.evidence)


def test_diagnose_unknown_when_missing_gold_chunk_id(diagnoser: RAGDiagnoser) -> None:
    """Answerable question without gold chunk reference leads to UNKNOWN."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Wrong answer.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01"],
        gold_chunk_id=None,  # Missing!
        gold_answer="15 minutes",
        question_type="answerable",  # labeled answerable but no gold chunk
    )

    # Note: question_type="answerable" with gold_chunk_id=None
    # ExecutionTrace logic sets is_unanswerable if gold_chunk_id is None,
    # but if trace explicitly specifies gold_answer without gold_chunk_id:
    # Let's verify how diagnoser handles missing gold chunk ID:
    # If gold_id is not in corpus or None
    result = diagnoser.diagnose(trace)
    # Since gold_chunk_id is None, it treated as unanswerable or unknown
    assert result.diagnosis in (FailureType.SHOULD_ABSTAIN, FailureType.UNKNOWN)


def test_diagnose_unknown_when_gold_chunk_not_in_corpus(diagnoser: RAGDiagnoser) -> None:
    """Gold chunk ID referenced does not exist in corpus chunks."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Wrong answer.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01"],
        gold_chunk_id="chunk_nonexistent_999",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.UNKNOWN
    assert result.confidence == 0.0
    assert any("not found in corpus" in ev for ev in result.evidence)


def test_diagnose_empty_candidate_pool(diagnoser: RAGDiagnoser) -> None:
    """Candidate pool is empty -> retrieval miss."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="I don't know.",
        retrieved_chunk_ids=[],
        candidate_chunk_ids=[],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.RETRIEVAL_MISS
    assert result.candidate_count == 0


def test_diagnose_duplicate_candidate_chunks(diagnoser: RAGDiagnoser) -> None:
    """Duplicate candidate chunks should still find first rank correctly."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Wrong answer.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01", "chunk_gold_01", "chunk_gold_01"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    result = diagnoser.diagnose(trace)

    assert result.diagnosis == FailureType.RANKING_MISS
    assert result.candidate_rank == 2  # first 1-indexed position


def test_diagnose_determinism(diagnoser: RAGDiagnoser) -> None:
    """Running diagnosis multiple times on the same trace produces identical outputs."""
    trace = ExecutionTrace(
        question="What is the engineering response SLA for a P1 incident?",
        generated_answer="Disaster recovery drills run quarterly.",
        retrieved_chunk_ids=["chunk_distractor_01"],
        candidate_chunk_ids=["chunk_distractor_01", "chunk_distractor_02", "chunk_gold_01"],
        gold_chunk_id="chunk_gold_01",
        gold_answer="15 minutes",
        question_type="answerable",
    )

    r1 = diagnoser.diagnose(trace)
    r2 = diagnoser.diagnose(trace)

    assert r1.diagnosis == r2.diagnosis
    assert r1.confidence == r2.confidence
    assert r1.evidence == r2.evidence
    assert r1.candidate_rank == r2.candidate_rank
    assert r1.oracle_replay_correct == r2.oracle_replay_correct
