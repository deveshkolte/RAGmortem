"""Failure taxonomy definitions for RAG failure modes."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """Failure types diagnosed by RAGmortem, including canonical and observational states."""

    # Canonical Ground-Truth Types
    RETRIEVAL_MISS = "retrieval_miss"
    RANKING_MISS = "ranking_miss"
    GENERATION_IGNORED_CONTEXT = "generation_ignored_context"
    SHOULD_ABSTAIN = "should_abstain"
    UNKNOWN = "unknown"
    NO_FAILURE = "no_failure"

    # Observational Heuristic States (Day 5)
    RETRIEVAL_SUSPECTED = "retrieval_suspected"
    RANKING_SUSPECTED = "ranking_suspected"
    GENERATION_SUSPECTED = "generation_suspected"
    GENERATION_PROVEN = "generation_proven"
    ABSTENTION_SUSPECTED = "abstention_suspected"

    @property
    def canonical(self) -> FailureType:
        """Map suspected observational states to their canonical benchmark counterpart."""
        mapping = {
            FailureType.RETRIEVAL_SUSPECTED: FailureType.RETRIEVAL_MISS,
            FailureType.RANKING_SUSPECTED: FailureType.RANKING_MISS,
            FailureType.GENERATION_SUSPECTED: FailureType.GENERATION_IGNORED_CONTEXT,
            FailureType.GENERATION_PROVEN: FailureType.GENERATION_IGNORED_CONTEXT,
            FailureType.ABSTENTION_SUSPECTED: FailureType.SHOULD_ABSTAIN,
        }
        return mapping.get(self, self)

    @property
    def is_suspected(self) -> bool:
        """Return True if this diagnosis represents a suspected observational inference."""
        return self in {
            FailureType.RETRIEVAL_SUSPECTED,
            FailureType.RANKING_SUSPECTED,
            FailureType.GENERATION_SUSPECTED,
            FailureType.ABSTENTION_SUSPECTED,
        }

    @property
    def description(self) -> str:
        descriptions = {
            FailureType.RETRIEVAL_MISS: (
                "The gold chunk containing the required evidence was not retrieved in top-k."
            ),
            FailureType.RANKING_MISS: (
                "The gold chunk was retrieved, but ranked too low to be utilized effectively."
            ),
            FailureType.GENERATION_IGNORED_CONTEXT: (
                "The gold chunk was present in the context, but the generator ignored it or hallucinated."
            ),
            FailureType.SHOULD_ABSTAIN: (
                "The question is unanswerable from the corpus, but the system generated an answer anyway."
            ),
            FailureType.UNKNOWN: (
                "Insufficient evidence or ambiguous signals to determine root cause."
            ),
            FailureType.NO_FAILURE: (
                "No failure detected; answer was generated correctly from retrieved evidence."
            ),
            FailureType.RETRIEVAL_SUSPECTED: (
                "Observable evidence (e.g. low candidate scores despite corpus relevance) suggests retrieval miss."
            ),
            FailureType.RANKING_SUSPECTED: (
                "Observable evidence (e.g. high-scoring candidates just below top-k cutoff) suggests ranking miss."
            ),
            FailureType.GENERATION_SUSPECTED: (
                "Observable evidence (e.g. high context relevance with ungrounded answer) suggests generator failure."
            ),
            FailureType.GENERATION_PROVEN: (
                "Reference-assisted verification proved generator ignored supplied context evidence."
            ),
            FailureType.ABSTENTION_SUSPECTED: (
                "Observable evidence (e.g. weak corpus relevance with substantive answer) suggests system should have abstained."
            ),
        }
        return descriptions.get(self, "Unknown failure mode.")

