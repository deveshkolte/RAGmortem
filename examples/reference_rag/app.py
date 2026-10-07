"""Reference RAG application demonstrating retrieval, scoring, caching, and generation."""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ragmortem.types import Chunk, RagResult
from ragmortem.adapter import RagAdapter
from ragmortem.cache import ModelCache

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def parse_chunks_from_markdown(content: str, source_name: str) -> list[Chunk]:
    """Parse discrete chunks annotated with `## [chunk_id] Title` from markdown."""
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
    """Scan a directory for markdown files and load all structured chunks."""
    corpus_path = Path(corpus_dir)
    chunks: list[Chunk] = []
    for file_path in sorted(corpus_path.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks.extend(parse_chunks_from_markdown(text, source_name=file_path.name))
    return chunks


class ReferenceRagApp(RagAdapter):
    """Reference RAG pipeline implementation compliant with RagAdapter.

    Exposes retrieval, chunk IDs, ranking scores, and generated answers.
    Supports deterministic mock execution and cached LLM execution.
    """

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        chunks: list[Chunk] | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        groq_model_name: str = DEFAULT_GROQ_MODEL,
        cache_dir: str | Path = ".ragmortem_cache",
        top_k: int = 3,
        mock_mode: bool = False,
    ) -> None:
        self.top_k = top_k
        self.mock_mode = mock_mode
        self.groq_model_name = os.environ.get("GROQ_MODEL", groq_model_name)
        self.cache = ModelCache(cache_dir=cache_dir)

        # Load chunks
        if chunks is not None:
            self.chunks = list(chunks)
        elif corpus_dir is not None:
            self.chunks = load_corpus_from_directory(corpus_dir)
        else:
            default_docs = Path(__file__).parent / "documents"
            self.chunks = load_corpus_from_directory(default_docs)

        # Initialize local sentence embedding model
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Precompute normalized embeddings for corpus
        if self.chunks:
            texts = [c.text for c in self.chunks]
            self.chunk_embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.chunk_embeddings = np.empty((0, 384))

    def retrieve(self, question: str, k: int | None = None) -> tuple[list[Chunk], list[float]]:
        """Retrieve top-k chunks and their cosine similarity scores."""
        if not self.chunks:
            return [], []

        k = k or self.top_k
        query_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity using normalized dot product
        scores = np.dot(self.chunk_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]

        retrieved_chunks = [self.chunks[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]
        return retrieved_chunks, retrieved_scores

    def build_prompt(self, question: str, retrieved_chunks: list[Chunk]) -> str:
        """Construct the prompt supplied to the generator model."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(f"[{i}] {chunk.text}")
        context_block = "\n\n".join(context_parts)

        return (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Instructions: Answer concisely using ONLY facts stated in the context above. "
            f"If the context does not contain the answer, reply exactly: "
            f"\"I do not have sufficient information in the provided context to answer this question.\"\n\n"
            f"Answer:"
        )

    def _call_groq_api(self, prompt: str) -> str:
        """Execute model call to Groq API using standard library (zero external API SDKs)."""
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        payload = {
            "model": self.groq_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a factual assistant. Answer strictly according to provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "RAGmortem/0.1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choice = result["choices"][0]["message"]
                return choice.get("content", "").strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Groq API HTTP {e.code}: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Groq API connection error: {e}") from e

    def _generate_mock_answer(self, question: str, retrieved_chunks: list[Chunk]) -> str:
        """Deterministic mock generator for tests and offline execution."""
        if not retrieved_chunks:
            return "I do not have sufficient information in the provided context to answer this question."

        top_chunk = retrieved_chunks[0]
        # Return deterministic mock response referencing the top chunk
        return f"[Mock Answer based on {top_chunk.id}]: {top_chunk.text}"

    def query(self, question: str) -> RagResult:
        """Run the end-to-end RAG pipeline and return a RagResult."""
        retrieved_chunks, scores = self.retrieve(question, k=self.top_k)
        prompt = self.build_prompt(question, retrieved_chunks)

        active_model = f"mock:{self.embedding_model_name}" if self.mock_mode else self.groq_model_name

        # Check local cache first
        cache_key = self.cache.compute_key(
            model=active_model,
            prompt=prompt,
            temperature=0.0,
        )
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            return RagResult(
                question=question,
                answer=cached_response,
                retrieved_chunks=retrieved_chunks,
                scores=scores,
                model=active_model,
                metadata={"cached": True, "cache_key": cache_key},
            )

        # Generate response
        has_api_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
        if self.mock_mode or not has_api_key:
            answer = self._generate_mock_answer(question, retrieved_chunks)
        else:
            answer = self._call_groq_api(prompt)

        # Save to cache
        self.cache.set(
            key=cache_key,
            response=answer,
            model=active_model,
            prompt=prompt,
            metadata={"top_k": self.top_k},
        )

        return RagResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            scores=scores,
            model=active_model,
            metadata={"cached": False, "cache_key": cache_key},
        )
