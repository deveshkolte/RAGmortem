"""Unit tests for offline deterministic mock RAG execution."""

from pathlib import Path

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.types import Chunk, RagResult


def test_reference_rag_mock_execution(tmp_path: Path) -> None:
    c1 = Chunk(id="chunk_sla", text="Response time SLA for P1 is 15 minutes.", source="doc.md")
    c2 = Chunk(id="chunk_sec", text="Rotate database secrets every 90 days.", source="doc.md")

    app = ReferenceRagApp(
        chunks=[c1, c2],
        top_k=2,
        cache_dir=tmp_path / "cache",
        mock_mode=True,
    )

    result = app.query("What is the response SLA for P1?")
    assert isinstance(result, RagResult)
    assert len(result.retrieved_chunks) == 2
    assert result.retrieved_ids[0] == "chunk_sla"
    assert result.gold_rank("chunk_sla") == 1
    assert "Mock Answer based on chunk_sla" in result.answer
    assert result.scores is not None
    assert len(result.scores) == 2


def test_mock_rag_caching_behavior(tmp_path: Path) -> None:
    c1 = Chunk(id="chunk_sla", text="Response time SLA for P1 is 15 minutes.", source="doc.md")

    app = ReferenceRagApp(
        chunks=[c1],
        top_k=1,
        cache_dir=tmp_path / "cache",
        mock_mode=True,
    )

    res1 = app.query("What is P1 SLA?")
    assert res1.metadata.get("cached") is False

    res2 = app.query("What is P1 SLA?")
    assert res2.metadata.get("cached") is True
    assert res1.answer == res2.answer
