"""Runtime trace representation for real RAG application executions.

Captures observable execution telemetry (query, retrieved chunks, scores,
ranks, context, answer, model, latency) without fabricating missing values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ragmortem.types import Chunk, RagResult


def _extract_chunk_id(chunk: Any, idx: int) -> str:
    """Safely extract chunk ID from Chunk, dict, or str."""
    if isinstance(chunk, Chunk):
        return chunk.id
    if isinstance(chunk, dict):
        return str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{idx}")
    if isinstance(chunk, str):
        return chunk
    return f"chunk_{idx}"


def _extract_chunk_text(chunk: Any) -> str:
    """Safely extract chunk text from Chunk, dict, or str."""
    if isinstance(chunk, Chunk):
        return chunk.text
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or "")
    if isinstance(chunk, str):
        return chunk
    return ""


@dataclass
class Trace:
    """Standardized runtime execution trace for RAG debugging.

    Represents what an application actually observed during execution.
    Missing telemetry fields remain explicit (None) and are never silently fabricated.
    """

    query: str = ""
    retrieved_chunks: list[Any] = field(default_factory=list)
    scores: list[float] | None = None
    ranks: list[int] | None = None
    context: str | None = None
    answer: str = ""
    candidate_chunks: list[Any] | None = None
    candidate_scores: list[float] | None = None
    model: str = ""
    latency_ms: float | None = None
    top_k: int | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = self.get_scores()

    @property
    def question(self) -> str:
        """Alias for query."""
        return self.query

    @property
    def generated_answer(self) -> str:
        """Alias for answer."""
        return self.answer

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        """Extract ordered list of retrieved chunk IDs."""
        return [_extract_chunk_id(c, i) for i, c in enumerate(self.retrieved_chunks)]

    @property
    def candidate_chunk_ids(self) -> list[str] | None:
        """Extract candidate chunk IDs if candidate pool was logged."""
        if self.candidate_chunks is None:
            return None
        return [_extract_chunk_id(c, i) for i, c in enumerate(self.candidate_chunks)]

    @property
    def retrieved_texts(self) -> list[str]:
        """Extract text content from retrieved chunks."""
        return [_extract_chunk_text(c) for c in self.retrieved_chunks]

    def get_scores(self) -> list[float] | None:
        """Return scores from self.scores or extract from chunk dicts if consistently present.

        Never fabricates default scores if telemetry was absent.
        """
        if self.scores is not None:
            return [float(s) for s in self.scores]

        # Check if chunks are dicts/Chunks with explicit score attributes
        if self.retrieved_chunks:
            extracted: list[float] = []
            for c in self.retrieved_chunks:
                if isinstance(c, dict) and "score" in c and c["score"] is not None:
                    try:
                        extracted.append(float(c["score"]))
                    except (ValueError, TypeError):
                        return None
                else:
                    return None
            return extracted
        return None

    def to_observed_execution(self) -> Any:
        """Convert trace to internal ObservedExecution structure for the diagnostic engine."""
        from ragmortem.diagnose import ObservedExecution

        clean_meta = dict(self.metadata)
        if self.context is not None:
            clean_meta["context"] = self.context
        if self.model:
            clean_meta["model"] = self.model
        if self.latency_ms is not None:
            clean_meta["latency_ms"] = self.latency_ms
        if self.ranks is not None:
            clean_meta["ranks"] = list(self.ranks)

        eff_top_k = self.top_k
        if eff_top_k is None:
            eff_top_k = len(self.retrieved_chunks) if self.retrieved_chunks else 3

        return ObservedExecution(
            question=self.query,
            retrieved_chunk_ids=self.retrieved_chunk_ids,
            generated_answer=self.answer,
            candidate_chunk_ids=self.candidate_chunk_ids,
            scores=self.get_scores(),
            candidate_scores=[float(s) for s in self.candidate_scores] if self.candidate_scores else None,
            top_k=eff_top_k,
            case_id=self.case_id,
            metadata=clean_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to JSON-compatible dictionary."""
        serialized_chunks = []
        for c in self.retrieved_chunks:
            if isinstance(c, Chunk):
                serialized_chunks.append(c.to_dict())
            elif isinstance(c, dict):
                serialized_chunks.append(dict(c))
            else:
                serialized_chunks.append({"id": str(c), "text": str(c)})

        data: dict[str, Any] = {
            "query": self.query,
            "retrieved_chunks": serialized_chunks,
            "answer": self.answer,
        }

        eff_scores = self.get_scores()
        if eff_scores is not None:
            data["scores"] = [float(s) for s in eff_scores]
        if self.ranks is not None:
            data["ranks"] = list(self.ranks)
        if self.context is not None:
            data["context"] = self.context
        if self.candidate_chunks is not None:
            data["candidate_chunks"] = [
                c.to_dict() if isinstance(c, Chunk) else c for c in self.candidate_chunks
            ]
        if self.candidate_scores is not None:
            data["candidate_scores"] = [float(s) for s in self.candidate_scores]
        if self.model:
            data["model"] = self.model
        if self.latency_ms is not None:
            data["latency_ms"] = self.latency_ms
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.case_id is not None:
            data["case_id"] = self.case_id
        if self.metadata:
            data["metadata"] = dict(self.metadata)

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Construct Trace from dictionary, supporting flexible schema variations."""
        query = str(data.get("query") or data.get("question") or "")
        answer = str(data.get("answer") or data.get("generated_answer") or "")

        # Flexible chunks loading
        raw_chunks = (
            data.get("retrieved_chunks")
            or data.get("retrieved_chunk_ids")
            or data.get("retrieved_ids")
            or []
        )
        chunks: list[Any] = []
        for item in raw_chunks:
            if isinstance(item, dict):
                chunks.append(dict(item))
            elif isinstance(item, str):
                chunks.append(item)
            else:
                chunks.append(item)

        scores = data.get("scores")
        if scores is not None:
            scores = [float(s) for s in scores]

        ranks = data.get("ranks")
        if ranks is not None:
            ranks = [int(r) for r in ranks]

        cand_chunks = (
            data.get("candidate_chunks")
            or data.get("candidate_chunk_ids")
            or data.get("candidate_retrieved_ids")
        )
        cand_scores = data.get("candidate_scores")
        if cand_scores is not None:
            cand_scores = [float(s) for s in cand_scores]

        meta = dict(data.get("metadata", {}))

        return cls(
            query=query,
            retrieved_chunks=chunks,
            scores=scores,
            ranks=ranks,
            context=data.get("context"),
            answer=answer,
            candidate_chunks=cand_chunks,
            candidate_scores=cand_scores,
            model=str(data.get("model", meta.get("model", ""))),
            latency_ms=data.get("latency_ms", meta.get("latency_ms")),
            top_k=data.get("top_k"),
            case_id=data.get("case_id"),
            metadata=meta,
        )

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> Trace:
        """Load trace from JSON file path or raw JSON string."""
        path = Path(path_or_str)
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)

    @classmethod
    def from_rag_result(cls, result: RagResult, top_k: int | None = None) -> Trace:
        """Convert existing RagResult dataclass into public Trace."""
        return cls(
            query=result.question,
            retrieved_chunks=list(result.retrieved_chunks),
            scores=result.scores,
            answer=result.answer,
            candidate_chunks=list(result.candidate_chunks) if result.candidate_chunks else None,
            candidate_scores=result.candidate_scores,
            model=result.model,
            top_k=top_k or len(result.retrieved_chunks),
            metadata=dict(result.metadata),
        )


def trace(
    query: str = "",
    retrieved_chunks: Sequence[Any] | None = None,
    scores: Sequence[float] | None = None,
    ranks: Sequence[int] | None = None,
    context: str | None = None,
    answer: str = "",
    *,
    question: str | None = None,
    generated_answer: str | None = None,
    candidate_chunks: Sequence[Any] | None = None,
    candidate_scores: Sequence[float] | None = None,
    model: str = "",
    latency_ms: float | None = None,
    top_k: int | None = None,
    case_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Create a standardized RAG execution trace.

    Example:
        t = ragmortem.trace(
            query="What is the P1 response time?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.82}],
            answer="The response time is 15 minutes.",
        )
    """
    effective_query = question if question is not None else query
    effective_answer = generated_answer if generated_answer is not None else answer

    return Trace(
        query=effective_query,
        retrieved_chunks=list(retrieved_chunks) if retrieved_chunks is not None else [],
        scores=[float(s) for s in scores] if scores is not None else None,
        ranks=list(ranks) if ranks is not None else None,
        context=context,
        answer=effective_answer,
        candidate_chunks=list(candidate_chunks) if candidate_chunks is not None else None,
        candidate_scores=[float(s) for s in candidate_scores] if candidate_scores is not None else None,
        model=model,
        latency_ms=latency_ms,
        top_k=top_k,
        case_id=case_id,
        metadata=dict(metadata) if metadata is not None else {},
    )
