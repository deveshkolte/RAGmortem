"""1-line callbacks for LangChain and LlamaIndex workflows.

Provides drop-in callbacks that automatically intercept retriever outputs,
similarity scores, and generated answers to produce ready-to-diagnose
RAGmortem traces.

Usage (LangChain):
    from ragmortem.callbacks import RAGmortemCallbackHandler
    handler = RAGmortemCallbackHandler()
    rag_chain.invoke({"query": "What is the SLA?"}, config={"callbacks": [handler]})
    diagnosis = handler.diagnose()

Usage (LlamaIndex):
    from ragmortem.callbacks import LlamaIndexRAGmortemHandler
    handler = LlamaIndexRAGmortemHandler()
    Settings.callback_manager = CallbackManager([handler])
    query_engine.query("What is the SLA?")
    diagnosis = handler.diagnose()
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ragmortem.trace import Trace, trace
from ragmortem.diagnose import DiagnosticResult

logger = logging.getLogger("ragmortem.callbacks")

# Graceful inheritance from langchain_core if installed
try:
    from langchain_core.callbacks import BaseCallbackHandler as LangChainBaseCallbackHandler
except ImportError:
    class LangChainBaseCallbackHandler:  # type: ignore[no-redef]
        """Duck-typed fallback when langchain-core is not installed."""
        pass


class RAGmortemCallbackHandler(LangChainBaseCallbackHandler):
    """1-line callback handler for LangChain RAG chains and agents.

    Captures query, retrieved documents with scores/metadata, and generated
    responses to generate a diagnostic trace without altering chain logic.
    """

    def __init__(
        self,
        auto_diagnose: bool = False,
        corpus: Any | None = None,
        mode: str | None = None,
        score_type: str = "similarity",
    ) -> None:
        super().__init__()
        self.auto_diagnose = auto_diagnose
        self.corpus = corpus
        self.mode = mode
        self.score_type = score_type

        # Captured telemetry
        self.query: str = ""
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.scores: list[float] = []
        self.generated_answer: str = ""
        self.metadata: dict[str, Any] = {}
        self.last_diagnosis: DiagnosticResult | None = None

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        **kwargs: Any,
    ) -> None:
        """Capture incoming question from chain inputs."""
        if isinstance(inputs, dict):
            for key in ("question", "query", "input", "prompt", "query_str"):
                if key in inputs and isinstance(inputs[key], str):
                    self.query = inputs[key]
                    break
        elif isinstance(inputs, str):
            self.query = inputs

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        **kwargs: Any,
    ) -> None:
        """Capture query if dispatched directly to a retriever."""
        if query and not self.query:
            self.query = str(query)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        **kwargs: Any,
    ) -> None:
        """Extract retrieved chunks, IDs, and similarity scores from LangChain Documents."""
        self.retrieved_chunks = []
        self.scores = []

        for i, doc in enumerate(documents):
            text = getattr(doc, "page_content", "") or str(doc)
            meta = getattr(doc, "metadata", {}) or {}
            doc_id = (
                getattr(doc, "id", None)
                or meta.get("id")
                or meta.get("doc_id")
                or f"doc_{i+1}"
            )
            # Find similarity score in metadata if present
            score_val = None
            for key in ("score", "relevance_score", "similarity_score", "cosine_similarity"):
                if key in meta and isinstance(meta[key], (int, float)):
                    score_val = float(meta[key])
                    break

            chunk_dict: dict[str, Any] = {
                "id": str(doc_id),
                "text": text,
                "metadata": meta,
            }
            if score_val is not None:
                chunk_dict["score"] = score_val
                self.scores.append(score_val)

            self.retrieved_chunks.append(chunk_dict)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extract generated answer text from LLMResult."""
        generations = getattr(response, "generations", None)
        if generations and len(generations) > 0 and len(generations[0]) > 0:
            first_gen = generations[0][0]
            text = getattr(first_gen, "text", "") or getattr(first_gen, "message", {}).get("content", "")
            if text and not self.generated_answer:
                self.generated_answer = str(text).strip()

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """Capture final chain output and run auto-diagnosis if configured."""
        if isinstance(outputs, dict):
            for key in ("answer", "result", "output", "output_text", "response"):
                if key in outputs and isinstance(outputs[key], str):
                    self.generated_answer = outputs[key].strip()
                    break
        elif isinstance(outputs, str):
            self.generated_answer = outputs.strip()

        if self.auto_diagnose:
            show_term = kwargs.get("show_terminal", True)
            self.diagnose(show_terminal=show_term)

    def get_trace(self) -> Trace:
        """Construct and return a RAGmortem Trace from captured telemetry."""
        return trace(
            query=self.query,
            retrieved_chunks=self.retrieved_chunks,
            scores=self.scores if self.scores else None,
            answer=self.generated_answer,
            score_type=self.score_type,
            metadata=self.metadata,
        )

    def diagnose(
        self,
        mode: str | None = None,
        corpus: Any | None = None,
        reference_answer: str | None = None,
        show_terminal: bool = True,
    ) -> DiagnosticResult:
        """Run root-cause failure diagnosis on the captured execution trace."""
        from ragmortem.api import diagnose as run_diagnose

        t = self.get_trace()
        res = run_diagnose(
            trace_input=t,
            mode=mode or self.mode,
            corpus=corpus if corpus is not None else self.corpus,
            reference_answer=reference_answer,
        )
        self.last_diagnosis = res

        if show_terminal:
            try:
                from ragmortem.terminal import render_diagnostic_card
                render_diagnostic_card(res, title="LangChain RAGmortem Live Diagnosis")
            except Exception:
                pass

        return res


class LlamaIndexRAGmortemHandler:
    """1-line callback / event handler for LlamaIndex query engines.

    Listens to LlamaIndex query, retrieve, and synthesis events to extract
    NodeWithScore chunks, similarity scores, and responses.
    """

    def __init__(
        self,
        auto_diagnose: bool = False,
        corpus: Any | None = None,
        mode: str | None = None,
        score_type: str = "similarity",
    ) -> None:
        self.auto_diagnose = auto_diagnose
        self.corpus = corpus
        self.mode = mode
        self.score_type = score_type

        self.query: str = ""
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.scores: list[float] = []
        self.generated_answer: str = ""
        self.metadata: dict[str, Any] = {}
        self.last_diagnosis: DiagnosticResult | None = None

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Intercept start of query or retrieve events."""
        if payload:
            query_str = payload.get("query_str") or payload.get("query")
            if query_str and not self.query:
                self.query = str(query_str)
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Intercept end of retrieve, synthesize, or query events."""
        if not payload:
            return

        # 1. Retrieved nodes (NodeWithScore)
        nodes = payload.get("nodes")
        if nodes and isinstance(nodes, Sequence):
            self.retrieved_chunks = []
            self.scores = []
            for i, item in enumerate(nodes):
                node = getattr(item, "node", item)
                text = (
                    getattr(node, "text", None)
                    or (node.get_content() if hasattr(node, "get_content") else str(node))
                )
                node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or f"node_{i+1}"
                score = getattr(item, "score", None)

                chunk_dict: dict[str, Any] = {
                    "id": str(node_id),
                    "text": str(text),
                    "metadata": getattr(node, "extra_info", {}) or getattr(node, "metadata", {}) or {},
                }
                if score is not None:
                    chunk_dict["score"] = float(score)
                    self.scores.append(float(score))
                self.retrieved_chunks.append(chunk_dict)

        # 2. Synthesized response
        response = payload.get("response")
        if response:
            ans_text = getattr(response, "response", None) or str(response)
            if ans_text:
                self.generated_answer = str(ans_text).strip()

        if str(event_type).lower() in ("query", "cbeventtype.query") and self.auto_diagnose:
            self.diagnose()

    def start_trace(self, trace_id: str | None = None) -> None:
        """LlamaIndex callback manager compatibility hook."""
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """LlamaIndex callback manager trace end hook."""
        if self.auto_diagnose and self.query and self.generated_answer:
            self.diagnose()

    def get_trace(self) -> Trace:
        """Construct and return a RAGmortem Trace from captured telemetry."""
        return trace(
            query=self.query,
            retrieved_chunks=self.retrieved_chunks,
            scores=self.scores if self.scores else None,
            answer=self.generated_answer,
            score_type=self.score_type,
            metadata=self.metadata,
        )

    def diagnose(
        self,
        mode: str | None = None,
        corpus: Any | None = None,
        reference_answer: str | None = None,
        show_terminal: bool = True,
    ) -> DiagnosticResult:
        """Run root-cause failure diagnosis on the captured execution trace."""
        from ragmortem.api import diagnose as run_diagnose

        t = self.get_trace()
        res = run_diagnose(
            trace_input=t,
            mode=mode or self.mode,
            corpus=corpus if corpus is not None else self.corpus,
            reference_answer=reference_answer,
        )
        self.last_diagnosis = res

        if show_terminal:
            try:
                from ragmortem.terminal import render_diagnostic_card
                render_diagnostic_card(res, title="LlamaIndex RAGmortem Live Diagnosis")
            except Exception:
                pass

        return res


# Export canonical aliases
LangChainRAGmortemHandler = RAGmortemCallbackHandler
