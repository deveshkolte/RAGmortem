"""Public developer-facing Python API for RAGmortem failure diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ragmortem.diagnose import DiagnosticResult, ObservedExecution, RAGDiagnoser
from ragmortem.taxonomy import FailureType
from ragmortem.trace import Trace, trace
from ragmortem.types import Chunk, RagResult


def diagnose(
    trace_input: Trace | ObservedExecution | dict[str, Any] | str | Path | None = None,
    *,
    query: str | None = None,
    retrieved_chunks: Sequence[Any] | None = None,
    scores: Sequence[float] | None = None,
    ranks: Sequence[int] | None = None,
    context: str | None = None,
    answer: str | None = None,
    corpus: Any | None = None,
    mode: str | None = None,
    candidate_chunks: Sequence[Any] | None = None,
    candidate_scores: Sequence[float] | None = None,
    top_k: int | None = None,
    model: str = "",
    latency_ms: float | None = None,
    case_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_answer: str | None = None,
    app: Any | None = None,
) -> DiagnosticResult:
    """Diagnose root cause of failure for a RAG execution trace.

    Supports two explicit modes:
    1. 'corpus_aware' (or 'corpus-aware'): Audits corpus index when retrieval scores are low.
    2. 'trace_only' (or 'trace-only'): Evaluates observable telemetry without accessing corpus.

    Never silently switches between modes. Missing telemetry produces UNKNOWN rather
    than fabricated evidence.

    Example (trace-only):
        result = diagnose(
            query="What is the P1 SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.85}],
            answer="Wrong answer",
            mode="trace_only",
        )

    Example (corpus-aware):
        result = diagnose(
            query="What is the P1 SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.12}],
            answer="Wrong answer",
            corpus=my_corpus_index,
            mode="corpus_aware",
        )
    """
    # 1. Resolve Trace representation
    if trace_input is not None:
        if isinstance(trace_input, Trace):
            t = trace_input
        elif isinstance(trace_input, ObservedExecution):
            t = Trace(
                query=trace_input.question,
                retrieved_chunks=list(trace_input.retrieved_chunk_ids),
                scores=trace_input.scores,
                answer=trace_input.generated_answer,
                candidate_chunks=trace_input.candidate_chunk_ids,
                candidate_scores=trace_input.candidate_scores,
                top_k=trace_input.top_k,
                case_id=trace_input.case_id,
                metadata=trace_input.metadata,
            )
        elif isinstance(trace_input, dict):
            t = Trace.from_dict(trace_input)
        elif isinstance(trace_input, (str, Path)):
            t = Trace.from_json(trace_input)
        else:
            raise TypeError(
                f"Unsupported trace input type: {type(trace_input)}. "
                "Expected Trace, ObservedExecution, dict, or file path/JSON string."
            )
    else:
        t = trace(
            query=query or "",
            retrieved_chunks=retrieved_chunks or [],
            scores=scores,
            ranks=ranks,
            context=context,
            answer=answer or "",
            candidate_chunks=candidate_chunks,
            candidate_scores=candidate_scores,
            model=model,
            latency_ms=latency_ms,
            top_k=top_k,
            case_id=case_id,
            metadata=metadata,
        )

    # 2. Resolve Diagnostic Mode
    if mode is not None:
        norm_mode = mode.lower().replace("-", "_")
        if norm_mode not in ("trace_only", "corpus_aware"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be explicitly 'trace_only' or 'corpus_aware'."
            )
    else:
        # Default based on explicit corpus availability
        effective_corpus_check = corpus if corpus is not None else app
        if effective_corpus_check is not None:
            norm_mode = "corpus_aware"
        else:
            norm_mode = "trace_only"

    # 3. Resolve Corpus App
    target_corpus = corpus if corpus is not None else app
    corpus_app: Any | None = None

    if norm_mode == "corpus_aware" and target_corpus is not None:
        if hasattr(target_corpus, "retrieve"):
            corpus_app = target_corpus
        elif isinstance(target_corpus, (str, Path)):
            from examples.reference_rag.app import ReferenceRagApp

            corpus_app = ReferenceRagApp(corpus_dir=target_corpus, mock_mode=True)
        elif isinstance(target_corpus, Sequence):
            from examples.reference_rag.app import ReferenceRagApp

            corpus_app = ReferenceRagApp(chunks=list(target_corpus), mock_mode=True)
        else:
            corpus_app = target_corpus

    # 4. Execute Diagnosis
    diagnoser = RAGDiagnoser(app=corpus_app if norm_mode == "corpus_aware" else None)
    obs_exec = t.to_observed_execution()

    if reference_answer is not None:
        return diagnoser.diagnose_reference_assisted(obs_exec, reference_answer=reference_answer)

    return diagnoser.diagnose_observed(
        trace=obs_exec,
        mode=norm_mode,
        corpus=corpus_app if norm_mode == "corpus_aware" else None,
    )
