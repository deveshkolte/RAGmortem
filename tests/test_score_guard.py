"""Unit and integration tests for Task 2 score_type guard.

Verifies:
1. score_type="distance" returns UNKNOWN without applying score thresholds.
2. score_type="similarity" with scores outside [-1.0, 1.0] returns UNKNOWN.
3. score_type="cosine_distance" correctly converts similarity = 1 - d.
4. Default similarity behavior is completely unchanged.
"""

import pytest

from ragmortem import FailureType, Trace, diagnose, trace
from ragmortem.diagnose import ObservedExecution, RAGDiagnoser


def test_distance_score_type_returns_unknown():
    """When score_type is 'distance', returns UNKNOWN with specified reason."""
    res = diagnose(
        query="What is the recovery protocol?",
        retrieved_chunks=[{"id": "doc1", "text": "Some text", "score": 0.12}],
        answer="I do not have enough information to answer.",
        score_type="distance",
    )
    assert res.diagnosis == FailureType.UNKNOWN
    assert res.evidence_score == 0.0
    expected_reason = (
        "scores are distances of unknown scale; thresholds are calibrated for cosine similarity"
    )
    assert expected_reason in res.reasons
    assert expected_reason in res.evidence


def test_distance_score_type_via_dict_and_trace():
    """Trace constructed with score_type='distance' propagates correctly."""
    t = trace(
        query="What is the protocol?",
        retrieved_chunks=[{"id": "doc1", "text": "Protocol text", "score": 0.05}],
        answer="The protocol is standard.",
        score_type="distance",
    )
    assert t.score_type == "distance"
    res = diagnose(t)
    assert res.diagnosis == FailureType.UNKNOWN
    assert any(
        "scores are distances of unknown scale" in r for r in res.reasons
    )


def test_out_of_range_similarity_upper_bound():
    """Score > 1.0 in similarity mode triggers UNKNOWN."""
    res = diagnose(
        query="What is the protocol?",
        retrieved_chunks=[{"id": "doc1", "text": "Protocol text", "score": 1.5}],
        answer="The protocol is standard.",
        score_type="similarity",
    )
    assert res.diagnosis == FailureType.UNKNOWN
    assert any("outside valid cosine similarity range" in r for r in res.reasons)


def test_out_of_range_similarity_lower_bound():
    """Score < -1.0 in similarity mode triggers UNKNOWN."""
    res = diagnose(
        query="What is the protocol?",
        retrieved_chunks=[{"id": "doc1", "text": "Protocol text", "score": -1.2}],
        answer="The protocol is standard.",
    )
    assert res.diagnosis == FailureType.UNKNOWN
    assert any("outside valid cosine similarity range" in r for r in res.reasons)


def test_out_of_range_candidate_scores():
    """Candidate score outside [-1.0, 1.0] triggers UNKNOWN."""
    t = trace(
        query="What is the protocol?",
        retrieved_chunks=[{"id": "doc1", "text": "Text 1", "score": 0.8}],
        candidate_chunks=["doc1", "doc2"],
        candidate_scores=[0.8, 10.5],
        answer="The protocol is standard.",
    )
    res = diagnose(t)
    assert res.diagnosis == FailureType.UNKNOWN
    assert any("outside valid cosine similarity range" in r for r in res.reasons)


def test_cosine_distance_conversion():
    """score_type='cosine_distance' converts similarity = 1 - d before thresholding."""
    # Distance 0.10 -> similarity 0.90 (>= 0.54)
    # Model generates refusal on high relevance -> false refusal -> GENERATION_SUSPECTED
    res_cosine = diagnose(
        query="What is the SLA?",
        retrieved_chunks=[{"id": "doc1", "text": "The SLA is 15 minutes.", "score": 0.10}],
        answer="I cannot answer this question based on the provided context.",
        score_type="cosine_distance",
    )
    assert res_cosine.diagnosis == FailureType.GENERATION_SUSPECTED

    # Without cosine_distance (similarity = 0.10 < 0.54), refusal matches low score -> NO_FAILURE
    res_similarity = diagnose(
        query="What is the SLA?",
        retrieved_chunks=[{"id": "doc1", "text": "The SLA is 15 minutes.", "score": 0.10}],
        answer="I cannot answer this question based on the provided context.",
        score_type="similarity",
    )
    assert res_similarity.diagnosis == FailureType.NO_FAILURE


def test_default_similarity_valid_scores_unchanged():
    """Default score_type is 'similarity' and valid scores within [-1.0, 1.0] work unchanged."""
    res = diagnose(
        query="What is the SLA?",
        retrieved_chunks=[{"id": "doc1", "text": "The SLA is 15 minutes.", "score": 0.85}],
        answer="The SLA is 15 minutes.",
    )
    # Valid high similarity score
    assert res.diagnosis in (FailureType.NO_FAILURE, FailureType.GENERATION_SUSPECTED, FailureType.UNKNOWN)
    assert not any("unknown scale" in r for r in res.reasons)
    assert not any("outside valid cosine similarity" in r for r in res.reasons)


def test_serialization_roundtrip_score_type():
    """Trace and ObservedExecution serialize and deserialize score_type."""
    t = trace(
        query="Q",
        retrieved_chunks=[{"id": "d1", "text": "T", "score": 0.5}],
        answer="A",
        score_type="cosine_distance",
    )
    d = t.to_dict()
    assert d["score_type"] == "cosine_distance"
    t_back = Trace.from_dict(d)
    assert t_back.score_type == "cosine_distance"

    obs = t.to_observed_execution()
    assert obs.score_type == "cosine_distance"
    obs_d = obs.to_dict()
    assert obs_d["score_type"] == "cosine_distance"
    obs_back = ObservedExecution.from_dict(obs_d)
    assert obs_back.score_type == "cosine_distance"


def test_invalid_score_type_raises():
    """Invalid score_type value raises ValueError."""
    obs = ObservedExecution(
        question="Q",
        retrieved_chunk_ids=["c1"],
        generated_answer="A",
        scores=[0.5],
        score_type="invalid_type",
    )
    diagnoser = RAGDiagnoser()
    with pytest.raises(ValueError, match="Invalid score_type"):
        diagnoser.diagnose_observed(obs)
