"""Corpus retrieval utilities — shipped as part of the installed ragmortem package.

Provides everything needed to build a corpus-aware diagnostic adapter from:
  - A directory of structured markdown files
  - An in-memory list of Chunk objects
  - Any object exposing a ``.retrieve(query, k)`` method (custom adapter)

These utilities replace the previous runtime dependency on ``examples.reference_rag.app``
so that corpus-aware diagnosis works from any external installation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from ragmortem.types import Chunk

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Markdown corpus loading (extracted from examples/reference_rag/app.py)
# ---------------------------------------------------------------------------

def parse_chunks_from_markdown(content: str, source_name: str) -> list[Chunk]:
    """Parse discrete chunks annotated with ``## [chunk_id] Title`` from markdown.

    Chunks must follow the convention used by the bundled reference documents::

        ## [chunk_incident_severity_p1] Incident Severity P1
        P1 (Critical) incidents require...

    Returns a list of :class:`~ragmortem.types.Chunk` objects.
    """
    chunks: list[Chunk] = []
    pattern = re.compile(r"^##\s+\[([a-zA-Z0-9_\-]+)\]\s*(.*?)$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    for i, match in enumerate(matches):
        chunk_id = match.group(1).strip()
        title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk_body = content[start_pos:end_pos].strip()

        chunks.append(
            Chunk(
                id=chunk_id,
                text=chunk_body,
                source=source_name,
                metadata={"title": title, "source": source_name},
            )
        )
    return chunks


def load_corpus_from_directory(corpus_dir: str | Path) -> list[Chunk]:
    """Scan *corpus_dir* for ``*.md`` files and return all structured chunks.

    Relies on :func:`parse_chunks_from_markdown`.  Files are processed in
    sorted order so results are deterministic across runs.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(
            f"Corpus directory '{corpus_path}' does not exist or is not a directory."
        )
    chunks: list[Chunk] = []
    for file_path in sorted(corpus_path.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks.extend(parse_chunks_from_markdown(text, source_name=file_path.name))
    return chunks


# ---------------------------------------------------------------------------
# Minimal vector retriever — corpus-aware diagnosis adapter
# ---------------------------------------------------------------------------

class CorpusRetriever:
    """Lightweight local vector retriever for corpus-aware RAGmortem diagnosis.

    Accepts either:

    * A path (``str`` or :class:`pathlib.Path`) to a directory of structured
      markdown files, loaded via :func:`load_corpus_from_directory`.
    * A sequence of :class:`~ragmortem.types.Chunk` objects.
    * A sequence of ``dict`` items with at least ``id`` and ``text`` keys.

    Exposes a ``.retrieve(query, k)`` interface compatible with
    :class:`~ragmortem.diagnose.RAGDiagnoser`.

    ``numpy`` and ``sentence-transformers`` are imported lazily at first use so
    that importing the :mod:`ragmortem` package itself remains lightweight.

    Example — directory path::

        from ragmortem.retrieval import CorpusRetriever
        from ragmortem import diagnose, trace

        retriever = CorpusRetriever("path/to/documents/")
        result = diagnose(my_trace, corpus=retriever, mode="corpus_aware")

    Example — in-memory chunk list::

        from ragmortem import Chunk
        from ragmortem.retrieval import CorpusRetriever

        chunks = [
            Chunk(id="c1", text="P1 incidents require 15-minute response.", source="policy"),
            Chunk(id="c2", text="API keys rotate every 90 days.", source="policy"),
        ]
        retriever = CorpusRetriever(chunks)
    """

    def __init__(
        self,
        corpus: str | Path | Sequence[Any],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 3,
    ) -> None:
        # Lazy-import heavy ML dependencies so that importing ragmortem itself is
        # always fast and never fails on systems where these packages are not yet
        # installed (e.g. during package collection phase of a CI run).
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.top_k = top_k
        self.embedding_model_name = embedding_model_name

        # ------------------------------------------------------------------
        # Resolve corpus → list[Chunk]
        # ------------------------------------------------------------------
        if isinstance(corpus, (str, Path)):
            self.chunks: list[Chunk] = load_corpus_from_directory(corpus)
        else:
            resolved: list[Chunk] = []
            for item in corpus:
                if isinstance(item, Chunk):
                    resolved.append(item)
                elif isinstance(item, dict):
                    resolved.append(
                        Chunk(
                            id=str(
                                item.get("id")
                                or item.get("chunk_id")
                                or f"chunk_{len(resolved)}"
                            ),
                            text=str(item.get("text") or item.get("content") or ""),
                            source=str(item.get("source", "")),
                            metadata={
                                k: v
                                for k, v in item.items()
                                if k not in ("id", "text", "source", "score")
                            },
                        )
                    )
                else:
                    raise TypeError(
                        f"Corpus items must be Chunk or dict, got {type(item)}."
                    )
            self.chunks = resolved

        # ------------------------------------------------------------------
        # Build embedding index
        # ------------------------------------------------------------------
        # Store np reference on the instance so retrieve() can use it without
        # a global import.
        self._np = np
        self._embedding_model: Any = None
        self._embeddings: Any = None

        if self.chunks:
            self._embedding_model = SentenceTransformer(embedding_model_name)
            texts = [c.text for c in self.chunks]
            self._embeddings = self._embedding_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            self._embeddings = np.empty((0, 384))

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[Chunk], list[float]]:
        """Return the top-*k* chunks most similar to *query* and their cosine scores."""
        if not self.chunks or self._embedding_model is None:
            return [], []

        effective_k = k if k is not None else self.top_k
        query_emb = self._embedding_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        np = self._np
        scores = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:effective_k]

        return (
            [self.chunks[i] for i in top_indices],
            [float(scores[i]) for i in top_indices],
        )
