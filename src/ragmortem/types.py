"""Core data types for RAGmortem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single text chunk retrieved or stored in a corpus."""

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data.get("source", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RagResult:
    """The output produced by a RAG system for an individual question."""

    question: str
    answer: str
    retrieved_chunks: list[Chunk]
    scores: list[float] | None = None
    candidate_chunks: list[Chunk] | None = None
    candidate_scores: list[float] | None = None
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_ids(self) -> list[str]:
        """Return list of retrieved chunk IDs in their ranked order."""
        return [chunk.id for chunk in self.retrieved_chunks]

    @property
    def candidate_ids(self) -> list[str]:
        """Return list of candidate chunk IDs if candidate pool is available."""
        if self.candidate_chunks is not None:
            return [chunk.id for chunk in self.candidate_chunks]
        return self.retrieved_ids

    def gold_rank(self, gold_chunk_id: str | None) -> int | None:
        """Return 1-indexed rank of gold_chunk_id in retrieved chunks, or None if absent."""
        if not gold_chunk_id:
            return None
        for rank, chunk in enumerate(self.retrieved_chunks, start=1):
            if chunk.id == gold_chunk_id:
                return rank
        return None

    def candidate_gold_rank(self, gold_chunk_id: str | None) -> int | None:
        """Return 1-indexed rank of gold_chunk_id in candidate chunks, or None if absent."""
        if not gold_chunk_id:
            return None
        pool = self.candidate_chunks if self.candidate_chunks is not None else self.retrieved_chunks
        for rank, chunk in enumerate(pool, start=1):
            if chunk.id == gold_chunk_id:
                return rank
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "scores": self.scores,
            "candidate_chunks": [c.to_dict() for c in self.candidate_chunks] if self.candidate_chunks else None,
            "candidate_scores": self.candidate_scores,
            "model": self.model,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagResult:
        candidate_data = data.get("candidate_chunks")
        return cls(
            question=str(data["question"]),
            answer=str(data["answer"]),
            retrieved_chunks=[Chunk.from_dict(c) for c in data.get("retrieved_chunks", [])],
            scores=data.get("scores"),
            candidate_chunks=[Chunk.from_dict(c) for c in candidate_data] if candidate_data else None,
            candidate_scores=data.get("candidate_scores"),
            model=str(data.get("model", "")),
            metadata=dict(data.get("metadata", {})),
        )

