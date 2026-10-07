"""Unit tests for RAGmortem core data types."""

from ragmortem.types import Chunk, RagResult


def test_chunk_attributes_and_serialization() -> None:
    chunk = Chunk(
        id="chunk_test_01",
        text="This is a test chunk body.",
        source="doc_test.md",
        metadata={"category": "testing", "section": 1},
    )

    assert chunk.id == "chunk_test_01"
    assert chunk.text == "This is a test chunk body."
    assert chunk.source == "doc_test.md"
    assert chunk.metadata["category"] == "testing"

    data = chunk.to_dict()
    assert data["id"] == "chunk_test_01"
    assert data["text"] == "This is a test chunk body."

    reconstructed = Chunk.from_dict(data)
    assert reconstructed.id == chunk.id
    assert reconstructed.text == chunk.text
    assert reconstructed.source == chunk.source
    assert reconstructed.metadata == chunk.metadata


def test_rag_result_retrieved_ids_and_gold_rank() -> None:
    c1 = Chunk(id="chunk_a", text="alpha", source="s1")
    c2 = Chunk(id="chunk_b", text="beta", source="s1")
    c3 = Chunk(id="chunk_c", text="gamma", source="s1")

    result = RagResult(
        question="What is beta?",
        answer="Beta is second.",
        retrieved_chunks=[c1, c2, c3],
        scores=[0.9, 0.7, 0.4],
        model="mock_model",
    )

    assert result.retrieved_ids == ["chunk_a", "chunk_b", "chunk_c"]
    # Gold rank (1-indexed)
    assert result.gold_rank("chunk_a") == 1
    assert result.gold_rank("chunk_b") == 2
    assert result.gold_rank("chunk_c") == 3
    # Missing chunk
    assert result.gold_rank("chunk_z") is None
    # None or empty gold_id
    assert result.gold_rank(None) is None
    assert result.gold_rank("") is None


def test_rag_result_serialization() -> None:
    c1 = Chunk(id="c1", text="text1", source="s1")
    result = RagResult(
        question="Q?",
        answer="A.",
        retrieved_chunks=[c1],
        scores=[0.85],
        model="test_model",
        metadata={"flag": True},
    )

    data = result.to_dict()
    assert data["question"] == "Q?"
    assert data["answer"] == "A."
    assert data["model"] == "test_model"
    assert len(data["retrieved_chunks"]) == 1

    reconstructed = RagResult.from_dict(data)
    assert reconstructed.question == result.question
    assert reconstructed.answer == result.answer
    assert reconstructed.retrieved_ids == ["c1"]
    assert reconstructed.scores == [0.85]
    assert reconstructed.metadata == {"flag": True}
