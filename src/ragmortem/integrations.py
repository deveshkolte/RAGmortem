"""Zero-dependency framework adapters for LangChain and LlamaIndex.

Converts framework outputs into standardized ragmortem.Trace objects
without requiring langchain or llama-index to be installed.
"""

from __future__ import annotations

from typing import Any
from ragmortem.trace import Trace, trace


def from_langchain(run_data: dict[str, Any] | Any, **kwargs: Any) -> Trace:
    """Extract query, retrieved_chunks, scores, and answer from a LangChain run into a Trace.

    Zero hard dependencies: duck-types dictionary outputs or LangChain run objects.

    Args:
        run_data: A LangChain output dict (e.g. from chain.invoke(), RetrievalQA,
                  or callback dictionary) or run object.
        **kwargs: Optional overrides for query, answer, scores, retrieved_chunks, or metadata.

    Returns:
        A standardized ragmortem.Trace instance ready for diagnose().
    """
    # 1. Extract query
    query = kwargs.get("query")
    if query is None:
        if isinstance(run_data, dict):
            query = (
                run_data.get("query")
                or run_data.get("question")
                or run_data.get("input")
                or run_data.get("prompt")
            )
            if query is None and isinstance(run_data.get("inputs"), dict):
                inputs_dict = run_data["inputs"]
                query = (
                    inputs_dict.get("query")
                    or inputs_dict.get("question")
                    or inputs_dict.get("input")
                    or inputs_dict.get("prompt")
                )
            elif query is None and isinstance(run_data.get("inputs"), str):
                query = run_data["inputs"]
        else:
            query = (
                getattr(run_data, "query", None)
                or getattr(run_data, "question", None)
                or getattr(run_data, "input", None)
                or getattr(run_data, "prompt", None)
            )

    # 2. Extract answer
    answer = kwargs.get("answer")
    if answer is None:
        raw_ans = None
        if isinstance(run_data, dict):
            raw_ans = (
                run_data.get("answer")
                or run_data.get("output")
                or run_data.get("result")
                or run_data.get("response")
                or run_data.get("output_text")
            )
            if raw_ans is None and isinstance(run_data.get("outputs"), dict):
                outputs_dict = run_data["outputs"]
                raw_ans = (
                    outputs_dict.get("output")
                    or outputs_dict.get("answer")
                    or outputs_dict.get("result")
                    or outputs_dict.get("text")
                )
            elif raw_ans is None and isinstance(run_data.get("outputs"), str):
                raw_ans = run_data["outputs"]
        else:
            raw_ans = (
                getattr(run_data, "answer", None)
                or getattr(run_data, "output", None)
                or getattr(run_data, "result", None)
                or getattr(run_data, "response", None)
            )

        if raw_ans is not None:
            if hasattr(raw_ans, "content"):
                answer = str(raw_ans.content)
            elif isinstance(raw_ans, dict):
                answer = str(
                    raw_ans.get("text")
                    or raw_ans.get("content")
                    or raw_ans.get("output")
                    or raw_ans.get("answer")
                    or ""
                )
            else:
                answer = str(raw_ans)

    # 3. Extract documents & scores
    raw_docs = kwargs.get("retrieved_chunks")
    if raw_docs is None:
        if isinstance(run_data, dict):
            raw_docs = (
                run_data.get("source_documents")
                or run_data.get("documents")
                or run_data.get("docs")
                or run_data.get("retrieved_docs")
                or run_data.get("context")
            )
        else:
            raw_docs = (
                getattr(run_data, "source_documents", None)
                or getattr(run_data, "documents", None)
                or getattr(run_data, "docs", None)
                or getattr(run_data, "context", None)
            )

    retrieved_chunks: list[dict[str, Any]] = []
    extracted_scores: list[float] = []
    all_scores_present = True

    if isinstance(raw_docs, list):
        for i, doc in enumerate(raw_docs):
            if hasattr(doc, "page_content"):
                # LangChain Document object
                text = str(getattr(doc, "page_content", ""))
                meta = getattr(doc, "metadata", {}) or {}
                cid = meta.get("id") or meta.get("chunk_id") or meta.get("source") or f"doc_{i+1}"
                score = meta.get("score") or meta.get("relevance_score") or meta.get("similarity")
            elif isinstance(doc, dict):
                text = str(doc.get("page_content") or doc.get("text") or doc.get("content") or "")
                meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                cid = (
                    doc.get("id")
                    or doc.get("chunk_id")
                    or meta.get("id")
                    or meta.get("chunk_id")
                    or meta.get("source")
                    or f"doc_{i+1}"
                )
                score = doc.get("score") or meta.get("score") or meta.get("relevance_score") or meta.get("similarity")
            elif isinstance(doc, str):
                text = doc
                cid = f"doc_{i+1}"
                score = None
            else:
                text = str(doc)
                cid = f"doc_{i+1}"
                score = None

            parsed_score = float(score) if score is not None else None
            retrieved_chunks.append({
                "id": str(cid),
                "text": text,
                "score": parsed_score,
            })
            if parsed_score is not None:
                extracted_scores.append(parsed_score)
            else:
                all_scores_present = False

    # Determine scores
    scores = kwargs.get("scores")
    if scores is None:
        if isinstance(run_data, dict) and "scores" in run_data and isinstance(run_data["scores"], list):
            scores = [float(s) for s in run_data["scores"]]
        elif isinstance(run_data, dict) and "relevance_scores" in run_data and isinstance(run_data["relevance_scores"], list):
            scores = [float(s) for s in run_data["relevance_scores"]]
        elif all_scores_present and extracted_scores:
            scores = extracted_scores

    # Clean kwargs
    clean_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("query", "answer", "retrieved_chunks", "scores")
    }

    return trace(
        query=str(query or ""),
        retrieved_chunks=retrieved_chunks,
        scores=scores,
        answer=str(answer or ""),
        **clean_kwargs,
    )


def from_llamaindex(response_obj: Any, **kwargs: Any) -> Trace:
    """Extract query, retrieved_chunks, scores, and answer from a LlamaIndex Response into a Trace.

    Zero hard dependencies: duck-types Response objects or dictionaries.

    Args:
        response_obj: A LlamaIndex Response object (or dict) containing .response/.text
                      and .source_nodes.
        **kwargs: Optional overrides for query, answer, scores, retrieved_chunks, or metadata.

    Returns:
        A standardized ragmortem.Trace instance ready for diagnose().
    """
    # 1. Extract query
    query = kwargs.get("query")
    if query is None:
        if isinstance(response_obj, dict):
            query = (
                response_obj.get("query")
                or response_obj.get("query_str")
                or response_obj.get("prompt")
                or response_obj.get("question")
            )
        else:
            query = (
                getattr(response_obj, "query", None)
                or getattr(response_obj, "query_str", None)
                or getattr(response_obj, "prompt", None)
            )

    # 2. Extract answer
    answer = kwargs.get("answer")
    if answer is None:
        if isinstance(response_obj, dict):
            answer = (
                response_obj.get("response")
                or response_obj.get("text")
                or response_obj.get("answer")
                or response_obj.get("output")
            )
        elif isinstance(response_obj, str):
            answer = response_obj
        else:
            answer = getattr(response_obj, "response", None) or getattr(response_obj, "text", None)

    # 3. Extract source nodes & scores
    raw_nodes = kwargs.get("retrieved_chunks")
    if raw_nodes is None:
        if isinstance(response_obj, dict):
            raw_nodes = response_obj.get("source_nodes") or response_obj.get("nodes")
        else:
            raw_nodes = getattr(response_obj, "source_nodes", None) or getattr(response_obj, "nodes", None)

    retrieved_chunks: list[dict[str, Any]] = []
    extracted_scores: list[float] = []
    all_scores_present = True

    if isinstance(raw_nodes, list):
        for i, node in enumerate(raw_nodes):
            if hasattr(node, "node"):
                # NodeWithScore
                inner = getattr(node, "node")
                text = (
                    getattr(inner, "text", None)
                    or (inner.get_content() if hasattr(inner, "get_content") else "")
                    or ""
                )
                cid = (
                    getattr(inner, "node_id", None)
                    or getattr(inner, "id_", None)
                    or getattr(inner, "id", None)
                    or f"node_{i+1}"
                )
                score = getattr(node, "score", None)
            elif isinstance(node, dict) and "node" in node and isinstance(node["node"], dict):
                inner = node["node"]
                text = inner.get("text") or inner.get("content") or ""
                cid = inner.get("node_id") or inner.get("id_") or inner.get("id") or f"node_{i+1}"
                score = node.get("score")
            elif hasattr(node, "text"):
                text = getattr(node, "text", "")
                cid = (
                    getattr(node, "node_id", None)
                    or getattr(node, "id_", None)
                    or getattr(node, "id", None)
                    or f"node_{i+1}"
                )
                score = getattr(node, "score", None)
            elif isinstance(node, dict):
                text = node.get("text") or node.get("content") or ""
                cid = node.get("node_id") or node.get("id_") or node.get("id") or f"node_{i+1}"
                score = node.get("score")
            else:
                text = str(node)
                cid = f"node_{i+1}"
                score = None

            parsed_score = float(score) if score is not None else None
            retrieved_chunks.append({
                "id": str(cid),
                "text": str(text),
                "score": parsed_score,
            })
            if parsed_score is not None:
                extracted_scores.append(parsed_score)
            else:
                all_scores_present = False

    # Determine scores
    scores = kwargs.get("scores")
    if scores is None:
        if all_scores_present and extracted_scores:
            scores = extracted_scores

    clean_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("query", "answer", "retrieved_chunks", "scores")
    }

    return trace(
        query=str(query or ""),
        retrieved_chunks=retrieved_chunks,
        scores=scores,
        answer=str(answer or ""),
        **clean_kwargs,
    )
