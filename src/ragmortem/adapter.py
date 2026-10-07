"""RAG Adapter Protocol and wrapper implementations."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ragmortem.types import RagResult


@runtime_checkable
class RagAdapter(Protocol):
    """Minimal protocol for interacting with any RAG application.

    Any system implementing `query(question: str) -> RagResult` qualifies
    as a valid RAG adapter for RAGmortem.
    """

    def query(self, question: str) -> RagResult:
        """Run a question through the RAG pipeline and return a structured RagResult."""
        ...


class FunctionAdapter:
    """Adapter that wraps an arbitrary callable `(question: str) -> RagResult`."""

    def __init__(self, fn: Callable[[str], RagResult]) -> None:
        if not callable(fn):
            raise TypeError(f"Expected a callable, got {type(fn).__name__}")
        self._fn = fn

    def query(self, question: str) -> RagResult:
        return self._fn(question)
