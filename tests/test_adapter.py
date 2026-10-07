"""Unit tests for RAG adapter protocol and implementations."""

import pytest

from ragmortem.adapter import FunctionAdapter, RagAdapter
from ragmortem.types import Chunk, RagResult


class DummyAdapter:
    def query(self, question: str) -> RagResult:
        return RagResult(
            question=question,
            answer="Static answer",
            retrieved_chunks=[Chunk(id="c1", text="text", source="src")],
            scores=[1.0],
            model="dummy",
        )


def test_adapter_protocol_compliance() -> None:
    adapter = DummyAdapter()
    assert isinstance(adapter, RagAdapter)

    result = adapter.query("What is happening?")
    assert isinstance(result, RagResult)
    assert result.answer == "Static answer"


def test_function_adapter() -> None:
    def dummy_fn(q: str) -> RagResult:
        return RagResult(
            question=q,
            answer=f"Echo: {q}",
            retrieved_chunks=[],
            model="echo",
        )

    adapter = FunctionAdapter(dummy_fn)
    assert isinstance(adapter, RagAdapter)

    result = adapter.query("Hello")
    assert result.answer == "Echo: Hello"
    assert result.question == "Hello"


def test_function_adapter_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        FunctionAdapter("not_a_function")  # type: ignore[arg-type]
