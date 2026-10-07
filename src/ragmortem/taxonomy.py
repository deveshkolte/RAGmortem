"""Failure taxonomy definitions for RAG failure modes."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """The four canonical failure types diagnosed by RAGmortem."""

    RETRIEVAL_MISS = "retrieval_miss"
    RANKING_MISS = "ranking_miss"
    GENERATION_IGNORED_CONTEXT = "generation_ignored_context"
    SHOULD_ABSTAIN = "should_abstain"
    NO_FAILURE = "no_failure"

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
            FailureType.NO_FAILURE: (
                "No failure detected; answer was generated correctly from retrieved evidence."
            ),
        }
        return descriptions.get(self, "Unknown failure mode.")
