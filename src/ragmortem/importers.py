"""OpenTelemetry, Arize Phoenix, and LangSmith trace importers.

Parses OpenInference / OpenTelemetry spans and LangSmith runs into
standardized RAGmortem Trace objects for root-cause failure diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ragmortem.trace import Trace, trace


def detect_format(data: Any) -> str:
    """Detect telemetry format: 'phoenix', 'langsmith', 'otel', or 'native'."""
    if isinstance(data, list):
        if not data:
            return "unknown"
        first = data[0]
        if isinstance(first, dict):
            if "openinference.span.kind" in first.get("attributes", {}):
                return "phoenix"
            if "run_type" in first:
                return "langsmith"
            if "attributes" in first and "name" in first:
                return "otel"
            if "query" in first or "question" in first:
                return "native"

    if isinstance(data, dict):
        if "openinference.span.kind" in data.get("attributes", {}):
            return "phoenix"
        if "run_type" in data or "child_runs" in data:
            return "langsmith"
        if "spans" in data and isinstance(data["spans"], list):
            return detect_format(data["spans"])
        if "runs" in data and isinstance(data["runs"], list):
            return "langsmith"
        if "attributes" in data and "name" in data:
            return "otel"
        if "query" in data or "question" in data:
            return "native"

    return "unknown"


def _flatten_or_extract_docs(doc_list_or_attrs: Any) -> tuple[list[dict[str, Any]], list[float]]:
    """Extract chunk dicts and scores from OpenInference/OTel documents."""
    chunks: list[dict[str, Any]] = []
    scores: list[float] = []

    if isinstance(doc_list_or_attrs, list):
        for i, item in enumerate(doc_list_or_attrs):
            if not isinstance(item, dict):
                continue
            text = (
                item.get("document.content")
                or item.get("content")
                or item.get("text")
                or item.get("page_content")
                or ""
            )
            doc_id = (
                item.get("document.id")
                or item.get("id")
                or item.get("doc_id")
                or f"doc_{i+1}"
            )
            score = item.get("document.score") or item.get("score") or item.get("relevance_score")
            meta = item.get("document.metadata") or item.get("metadata") or {}

            chunk_dict: dict[str, Any] = {
                "id": str(doc_id),
                "text": str(text),
                "metadata": meta if isinstance(meta, dict) else {},
            }
            if score is not None:
                try:
                    s_val = float(score)
                    chunk_dict["score"] = s_val
                    scores.append(s_val)
                except (ValueError, TypeError):
                    pass
            chunks.append(chunk_dict)

    return chunks, scores


def import_phoenix_trace(data: dict[str, Any] | list[dict[str, Any]]) -> Trace:
    """Import OpenInference / Arize Phoenix OpenTelemetry spans into a RAGmortem Trace."""
    spans: list[dict[str, Any]] = []
    if isinstance(data, list):
        spans = [s for s in data if isinstance(s, dict)]
    elif isinstance(data, dict):
        if "spans" in data and isinstance(data["spans"], list):
            spans = [s for s in data["spans"] if isinstance(s, dict)]
        else:
            spans = [data]

    query: str = ""
    answer: str = ""
    retrieved_chunks: list[dict[str, Any]] = []
    scores: list[float] = []
    metadata: dict[str, Any] = {}

    for span in spans:
        attrs = span.get("attributes", {})
        span_kind = str(attrs.get("openinference.span.kind", "")).upper()
        span_name = str(span.get("name", "")).lower()

        # 1. Query detection (Retriever, Chain, or Root)
        if not query:
            q_val = attrs.get("input.value") or attrs.get("query") or attrs.get("question")
            if q_val:
                if isinstance(q_val, dict):
                    query = str(q_val.get("query") or q_val.get("question") or q_val.get("input") or "")
                else:
                    query = str(q_val).strip()

        # 2. Retriever Span (documents and scores)
        if span_kind == "RETRIEVER" or "retriev" in span_name:
            # Check retrieval.documents
            raw_docs = attrs.get("retrieval.documents")
            if raw_docs:
                c, s = _flatten_or_extract_docs(raw_docs)
                if c:
                    retrieved_chunks.extend(c)
                if s:
                    scores.extend(s)
            else:
                # Check indexed attributes: retrieval.documents.0.document.content
                indexed_docs: list[dict[str, Any]] = []
                idx = 0
                while f"retrieval.documents.{idx}.document.content" in attrs:
                    content = attrs[f"retrieval.documents.{idx}.document.content"]
                    doc_id = attrs.get(f"retrieval.documents.{idx}.document.id", f"doc_{idx+1}")
                    score = attrs.get(f"retrieval.documents.{idx}.document.score")
                    doc_dict = {"document.content": content, "document.id": doc_id}
                    if score is not None:
                        doc_dict["document.score"] = score
                    indexed_docs.append(doc_dict)
                    idx += 1
                if indexed_docs:
                    c, s = _flatten_or_extract_docs(indexed_docs)
                    retrieved_chunks.extend(c)
                    scores.extend(s)

        # 3. LLM / Generator Span
        if span_kind == "LLM" or "llm" in span_name or "chat" in span_name:
            out_val = (
                attrs.get("output.value")
                or attrs.get("llm.output_messages.0.message.content")
                or attrs.get("response")
            )
            if out_val and not answer:
                if isinstance(out_val, dict):
                    answer = str(out_val.get("answer") or out_val.get("output") or out_val.get("content") or "")
                else:
                    answer = str(out_val).strip()

    # Fallback for answer from top-level output if not found in LLM span
    if not answer:
        for span in spans:
            attrs = span.get("attributes", {})
            out_val = attrs.get("output.value")
            if out_val:
                answer = str(out_val).strip()
                break

    metadata["source_format"] = "phoenix"
    return trace(
        query=query,
        retrieved_chunks=retrieved_chunks,
        scores=scores if scores else None,
        answer=answer,
        metadata=metadata,
    )


def import_langsmith_trace(data: dict[str, Any] | list[dict[str, Any]]) -> Trace:
    """Import LangSmith run tree or list of runs into a RAGmortem Trace."""
    runs: list[dict[str, Any]] = []

    def _collect_runs(r: dict[str, Any]) -> None:
        runs.append(r)
        for child in r.get("child_runs", []) or []:
            if isinstance(child, dict):
                _collect_runs(child)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _collect_runs(item)
    elif isinstance(data, dict):
        if "runs" in data and isinstance(data["runs"], list):
            for item in data["runs"]:
                if isinstance(item, dict):
                    _collect_runs(item)
        else:
            _collect_runs(data)

    query: str = ""
    answer: str = ""
    retrieved_chunks: list[dict[str, Any]] = []
    scores: list[float] = []
    metadata: dict[str, Any] = {}

    for run in runs:
        run_type = str(run.get("run_type", "")).lower()
        inputs = run.get("inputs", {}) or {}
        outputs = run.get("outputs", {}) or {}

        # 1. Query detection
        if not query and isinstance(inputs, dict):
            for k in ("query", "question", "input", "query_str"):
                if k in inputs and isinstance(inputs[k], str):
                    query = inputs[k].strip()
                    break

        # 2. Retriever run (documents and scores)
        if run_type == "retriever":
            docs = outputs.get("documents") or outputs.get("results") or []
            if isinstance(docs, list):
                for i, doc in enumerate(docs):
                    if not isinstance(doc, dict):
                        continue
                    text = doc.get("page_content") or doc.get("text") or ""
                    meta = doc.get("metadata", {}) or {}
                    doc_id = meta.get("id") or doc.get("id") or f"doc_{i+1}"
                    score = meta.get("score") or meta.get("relevance_score") or doc.get("score")

                    chunk_dict: dict[str, Any] = {
                        "id": str(doc_id),
                        "text": str(text),
                        "metadata": meta if isinstance(meta, dict) else {},
                    }
                    if score is not None:
                        try:
                            s_val = float(score)
                            chunk_dict["score"] = s_val
                            scores.append(s_val)
                        except (ValueError, TypeError):
                            pass
                    retrieved_chunks.append(chunk_dict)

        # 3. LLM / Chain answer detection
        if run_type in ("llm", "chain"):
            if isinstance(outputs, dict):
                for k in ("answer", "output", "result", "output_text", "response"):
                    if k in outputs and isinstance(outputs[k], str):
                        if not answer or run_type == "llm":
                            answer = outputs[k].strip()
                            break
                # Generations structure: outputs.generations[0][0].text
                generations = outputs.get("generations")
                if generations and isinstance(generations, list) and len(generations) > 0:
                    first = generations[0]
                    if isinstance(first, list) and len(first) > 0:
                        gen_text = first[0].get("text") or first[0].get("message", {}).get("content")
                        if gen_text and not answer:
                            answer = str(gen_text).strip()

    metadata["source_format"] = "langsmith"
    return trace(
        query=query,
        retrieved_chunks=retrieved_chunks,
        scores=scores if scores else None,
        answer=answer,
        metadata=metadata,
    )


def import_trace(source: dict[str, Any] | list[Any] | str | Path) -> Trace:
    """Import an OpenTelemetry, Phoenix, LangSmith, or native JSON trace into a RAGmortem Trace."""
    data: Any = None
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            data = json.loads(text)
        elif isinstance(source, str) and (source.startswith("{") or source.startswith("[")):
            data = json.loads(source)
        else:
            raise FileNotFoundError(f"Trace file '{source}' not found.")
    else:
        data = source

    fmt = detect_format(data)
    if fmt == "phoenix" or fmt == "otel":
        return import_phoenix_trace(data)
    elif fmt == "langsmith":
        return import_langsmith_trace(data)
    elif fmt == "native":
        if isinstance(data, dict):
            return Trace.from_dict(data)
        elif isinstance(data, list) and len(data) > 0:
            return Trace.from_dict(data[0])

    # Fallback attempt Phoenix/OTel then LangSmith
    try:
        t = import_phoenix_trace(data)
        if t.query and (t.retrieved_chunks or t.answer):
            return t
    except Exception:
        pass

    try:
        t = import_langsmith_trace(data)
        if t.query and (t.retrieved_chunks or t.answer):
            return t
    except Exception:
        pass

    raise ValueError("Could not parse input data into a RAGmortem Trace. Unrecognized format.")
