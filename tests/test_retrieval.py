"""Unit tests for markdown chunk parsing and vector retrieval."""

from pathlib import Path

from examples.reference_rag.app import (
    ReferenceRagApp,
    load_corpus_from_directory,
    parse_chunks_from_markdown,
)
from ragmortem.types import Chunk


def test_markdown_chunk_parser() -> None:
    content = """# Sample Document

## [chunk_p1] Severity One Incident
A catastrophic outage affecting more than 25% of users.

## [chunk_p2] Severity Two Incident
A major impairment affecting 5% to 25% of users.
"""
    chunks = parse_chunks_from_markdown(content, source_name="sample.md")
    assert len(chunks) == 2
    assert chunks[0].id == "chunk_p1"
    assert chunks[0].source == "sample.md"
    assert "25% of users" in chunks[0].text
    assert chunks[1].id == "chunk_p2"
    assert "major impairment" in chunks[1].text


def test_load_corpus_from_directory() -> None:
    docs_dir = Path("examples/reference_rag/documents")
    chunks = load_corpus_from_directory(docs_dir)
    assert len(chunks) == 30
    chunk_ids = {c.id for c in chunks}
    assert "chunk_incident_severity_p1" in chunk_ids
    assert "chunk_canary_duration" in chunk_ids
    assert "chunk_vuln_critical_sla" in chunk_ids


def test_vector_retrieval_ranking() -> None:
    c1 = Chunk(id="chunk_cat", text="The domestic feline is a small carnivorous mammal.", source="animals")
    c2 = Chunk(id="chunk_plane", text="A supersonic passenger aircraft designed for high-altitude flight.", source="aviation")
    c3 = Chunk(id="chunk_bread", text="A sourdough loaf baked in an oven using flour and yeast.", source="food")

    app = ReferenceRagApp(chunks=[c1, c2, c3], top_k=2, mock_mode=True)
    retrieved, scores = app.retrieve("How fast does the airplane fly?", k=2)

    assert len(retrieved) == 2
    # The aviation chunk should be ranked #1
    assert retrieved[0].id == "chunk_plane"
    assert scores[0] > scores[1]
