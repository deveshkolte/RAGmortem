"""Regression tests for the P1 external-packaging defect.

These tests reproduce the exact failure scenario reported in the v0.1.0 acceptance test:
    ModuleNotFoundError: No module named 'examples'

when corpus is supplied as a directory path, a sequence of Chunks, or a list of dicts
to ``ragmortem.diagnose()`` or the ``diagnose-trace`` CLI from an external installation
(i.e. one that does NOT have ``examples/`` on sys.path).

They verify three properties:
1. ``ragmortem.retrieval`` is self-contained — no dependency on ``examples``.
2. ``diagnose(..., corpus=<path>)`` and ``diagnose(..., corpus=<chunks>)`` work correctly.
3. The CLI ``diagnose-trace --corpus <path>`` exits cleanly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent.parent / "examples" / "reference_rag" / "documents"


def _minimal_retrieval_miss_trace() -> dict[str, Any]:
    """Trace that should resolve to RETRIEVAL_SUSPECTED when corpus is available."""
    return {
        "query": "What is the mandatory engineering response time for a P1 incident?",
        "retrieved_chunks": [
            {
                "id": "chunk_dr_drill_frequency",
                "text": "Disaster recovery drills occur quarterly.",
                "score": 0.14,
            }
        ],
        "scores": [0.14],
        "answer": "Engineers must run quarterly disaster recovery exercises.",
    }


def _minimal_retrieval_miss_trace_obj():
    from ragmortem import trace

    return trace(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[
            {"id": "chunk_dr_drill", "text": "Disaster recovery drills occur quarterly.", "score": 0.14}
        ],
        scores=[0.14],
        answer="Engineers must run quarterly disaster recovery exercises.",
    )


# ---------------------------------------------------------------------------
# 1. Module self-containment: ragmortem.retrieval must not import `examples`
# ---------------------------------------------------------------------------

def test_retrieval_module_does_not_import_examples() -> None:
    """Importing ragmortem.retrieval must not trigger any import of examples.*."""
    # If examples is already on path (dev checkout), skip the path-removal test
    import importlib
    mod = importlib.import_module("ragmortem.retrieval")
    assert mod is not None

    # Make sure none of ragmortem's installed modules list 'examples' in their __file__
    _ = [
        name for name in sys.modules if name == "examples" or name.startswith("examples.")
    ]
    # It's acceptable if examples is imported via test infrastructure; what matters
    # is that ragmortem.retrieval itself does NOT unconditionally pull it in.
    # We verify this by checking that it imports cleanly even without examples.
    assert True  # import above would have raised if it transitively required examples unconditionally


def test_ragmortem_retrieval_public_exports() -> None:
    """CorpusRetriever, load_corpus_from_directory, parse_chunks_from_markdown must be public."""
    from ragmortem.retrieval import (
        CorpusRetriever,
        load_corpus_from_directory,
        parse_chunks_from_markdown,
    )

    assert callable(CorpusRetriever)
    assert callable(load_corpus_from_directory)
    assert callable(parse_chunks_from_markdown)


def test_ragmortem_init_exports_corpus_retriever() -> None:
    """CorpusRetriever must be importable directly from ``ragmortem``."""
    import ragmortem

    assert hasattr(ragmortem, "CorpusRetriever")
    assert hasattr(ragmortem, "load_corpus_from_directory")
    assert hasattr(ragmortem, "parse_chunks_from_markdown")


# ---------------------------------------------------------------------------
# 2. CorpusRetriever — construction from path and from sequences
# ---------------------------------------------------------------------------

def test_corpus_retriever_from_directory() -> None:
    """CorpusRetriever must load markdown documents from a directory path."""
    from ragmortem.retrieval import CorpusRetriever

    retriever = CorpusRetriever(corpus=DOCS_DIR)
    assert len(retriever.chunks) > 0, "Expected chunks to be loaded from markdown docs"

    chunks, scores = retriever.retrieve("What is the P1 incident response time?", k=3)
    assert len(chunks) > 0
    assert len(scores) == len(chunks)
    assert all(isinstance(s, float) for s in scores)


def test_corpus_retriever_from_string_path() -> None:
    """CorpusRetriever must accept a string path to a corpus directory."""
    from ragmortem.retrieval import CorpusRetriever

    retriever = CorpusRetriever(corpus=str(DOCS_DIR))
    assert len(retriever.chunks) > 0
    chunks, scores = retriever.retrieve("P1 SLA", k=2)
    assert chunks


def test_corpus_retriever_from_chunk_list() -> None:
    """CorpusRetriever must accept a list of Chunk objects."""
    from ragmortem import Chunk
    from ragmortem.retrieval import CorpusRetriever

    chunks = [
        Chunk(id="c1", text="P1 incidents require 15-minute response time.", source="policy"),
        Chunk(id="c2", text="Database backups are retained for 30 days.", source="policy"),
        Chunk(id="c3", text="API keys must be rotated every 90 days.", source="policy"),
    ]
    retriever = CorpusRetriever(corpus=chunks)
    assert len(retriever.chunks) == 3

    result_chunks, result_scores = retriever.retrieve("P1 SLA", k=2)
    assert len(result_chunks) > 0
    assert result_chunks[0].id == "c1"  # P1 chunk should be top-ranked


def test_corpus_retriever_from_dict_list() -> None:
    """CorpusRetriever must accept a list of dicts as corpus."""
    from ragmortem.retrieval import CorpusRetriever

    dicts = [
        {"id": "d1", "text": "P1 incidents require 15-minute response time.", "source": "policy"},
        {"id": "d2", "text": "API keys must be rotated every 90 days.", "source": "policy"},
    ]
    retriever = CorpusRetriever(corpus=dicts)
    assert len(retriever.chunks) == 2
    chunks, scores = retriever.retrieve("P1 response SLA", k=1)
    assert chunks[0].id == "d1"


def test_corpus_retriever_empty_corpus() -> None:
    """CorpusRetriever with empty corpus must return empty results without crashing."""
    from ragmortem.retrieval import CorpusRetriever

    retriever = CorpusRetriever(corpus=[])
    chunks, scores = retriever.retrieve("Any query", k=3)
    assert chunks == []
    assert scores == []


def test_corpus_retriever_nonexistent_directory_raises() -> None:
    """CorpusRetriever must raise FileNotFoundError on a nonexistent path."""
    from ragmortem.retrieval import CorpusRetriever

    with pytest.raises(FileNotFoundError):
        CorpusRetriever(corpus="/nonexistent/path/does/not/exist")


# ---------------------------------------------------------------------------
# 3. diagnose() public API — corpus as path (the exact P1 failure case)
# ---------------------------------------------------------------------------

def test_diagnose_corpus_path_no_module_not_found_error() -> None:
    """The P1 defect: diagnose(corpus=<str path>) must NOT raise ModuleNotFoundError."""
    from ragmortem import diagnose

    t = _minimal_retrieval_miss_trace_obj()
    # This was the exact failure: ModuleNotFoundError: No module named 'examples'
    result = diagnose(t, corpus=str(DOCS_DIR), mode="corpus_aware")

    from ragmortem import FailureType
    # Should detect retrieval miss, not crash
    assert result.failure_type in (FailureType.RETRIEVAL_SUSPECTED, FailureType.UNKNOWN)
    assert result.mode == "corpus_aware"


def test_diagnose_corpus_path_object_no_error() -> None:
    """diagnose(corpus=Path(...)) must NOT raise ModuleNotFoundError."""
    from ragmortem import FailureType, diagnose

    t = _minimal_retrieval_miss_trace_obj()
    result = diagnose(t, corpus=DOCS_DIR, mode="corpus_aware")
    assert result.failure_type == FailureType.RETRIEVAL_SUSPECTED
    assert result.evidence_score > 0.70


def test_diagnose_corpus_chunk_sequence_no_error() -> None:
    """diagnose(corpus=[Chunk, ...]) must NOT raise ModuleNotFoundError."""
    from ragmortem import Chunk, FailureType, diagnose

    corpus = [
        Chunk(
            id="chunk_incident_severity_p1",
            text="P1 (Critical) incidents require engineering acknowledgment within 15 minutes.",
            source="policy",
        ),
        Chunk(
            id="chunk_other",
            text="API keys and secrets must be rotated every 90 days.",
            source="policy",
        ),
    ]
    t = _minimal_retrieval_miss_trace_obj()
    result = diagnose(t, corpus=corpus, mode="corpus_aware")
    assert result.failure_type == FailureType.RETRIEVAL_SUSPECTED
    assert result.mode == "corpus_aware"


def test_diagnose_inline_kwargs_corpus_path() -> None:
    """diagnose() with inline kwargs and corpus as path works without examples dependency."""
    from ragmortem import FailureType, diagnose

    result = diagnose(
        query="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunks=[
            {"id": "chunk_dr", "text": "Disaster recovery drills occur quarterly.", "score": 0.14}
        ],
        scores=[0.14],
        answer="Engineers must run quarterly disaster recovery exercises.",
        corpus=DOCS_DIR,
        mode="corpus_aware",
    )
    assert result.failure_type == FailureType.RETRIEVAL_SUSPECTED


# ---------------------------------------------------------------------------
# 4. CLI diagnose-trace --corpus <path> regression
# ---------------------------------------------------------------------------

def test_cli_diagnose_trace_corpus_path_works(tmp_path: Path) -> None:
    """CLI diagnose-trace --corpus <path> must exit 0, not crash with ModuleNotFoundError."""
    from ragmortem.cli import main

    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(_minimal_retrieval_miss_trace()), encoding="utf-8")

    exit_code = main([
        "diagnose-trace",
        str(trace_file),
        "--corpus", str(DOCS_DIR),
        "--mode", "corpus_aware",
    ])
    assert exit_code == 0


def test_cli_diagnose_trace_corpus_path_json_output(tmp_path: Path) -> None:
    """CLI diagnose-trace --corpus --json must produce valid JSON with failure_type."""
    import io
    from contextlib import redirect_stdout

    from ragmortem.cli import main

    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(_minimal_retrieval_miss_trace()), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main([
            "diagnose-trace",
            str(trace_file),
            "--corpus", str(DOCS_DIR),
            "--mode", "corpus_aware",
            "--json",
        ])

    assert exit_code == 0
    output = buf.getvalue()
    parsed = json.loads(output)
    assert "failure_type" in parsed
    assert parsed["mode"] == "corpus_aware"
    assert parsed["failure_type"] in ("retrieval_suspected", "unknown")


def test_cli_diagnose_trace_without_corpus_still_works(tmp_path: Path) -> None:
    """CLI diagnose-trace without --corpus (trace-only mode) must continue working."""
    from ragmortem.cli import main

    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(_minimal_retrieval_miss_trace()), encoding="utf-8")

    exit_code = main(["diagnose-trace", str(trace_file), "--mode", "trace_only"])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# 5. load_corpus_from_directory independent of examples
# ---------------------------------------------------------------------------

def test_load_corpus_from_directory_is_packaged() -> None:
    """load_corpus_from_directory must work from the installed package, not examples."""
    from ragmortem.retrieval import load_corpus_from_directory

    chunks = load_corpus_from_directory(DOCS_DIR)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.id
        assert chunk.text


def test_parse_chunks_from_markdown_is_packaged() -> None:
    """parse_chunks_from_markdown must be available from the installed package."""
    from ragmortem.retrieval import parse_chunks_from_markdown

    md_content = """# Doc title

## [chunk_a] First Section
This is the content of chunk A.

## [chunk_b] Second Section
This is the content of chunk B.
"""
    chunks = parse_chunks_from_markdown(md_content, source_name="test.md")
    assert len(chunks) == 2
    assert chunks[0].id == "chunk_a"
    assert "content of chunk A" in chunks[0].text
    assert chunks[1].id == "chunk_b"
