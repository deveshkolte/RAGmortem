"""Public developer-facing Python API for RAGmortem failure diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ragmortem.diagnose import DiagnosticResult, ObservedExecution, RAGDiagnoser
from ragmortem.trace import Trace, trace


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
    score_type: str | None = None,
    profile: str | Any = "minilm",
) -> DiagnosticResult:
    """Diagnose root cause of failure for a RAG execution trace.

    Three diagnostic scopes are supported (never silently switches between them):

    1. **trace_only** (or `trace-only`): Evaluates observable telemetry without accessing corpus.
       Returns `UNKNOWN` when evidence is insufficient. No corpus, no reference answer.

    2. **corpus_aware** (or `corpus-aware`) **without** `reference_answer`: Audits corpus index
       when retrieval scores are low. Uses heuristic score thresholds and corpus audit to return
       *suspected* failure types (`GENERATION_SUSPECTED`, `RETRIEVAL_SUSPECTED`, etc.).
       **Does not verify factual correctness** of the generated answer.

    3. **corpus_aware** (or `corpus-aware`) **with** `reference_answer`: Definitive diagnosis.
       Compares generated answer to `reference_answer`, searches retrieved and candidate chunks
       for supporting evidence. Returns definitive types (`NO_FAILURE`, `GENERATION_PROVEN`,
       `RETRIEVAL_MISS`, `RANKING_MISS`, `SHOULD_ABSTAIN`) or `UNKNOWN`.

    Never silently switches between scopes. Missing telemetry produces `UNKNOWN` rather
    than fabricated evidence.

    Example (trace-only):
        result = diagnose(
            query="What is the P1 SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.85}],
            answer="Wrong answer",
            mode="trace_only",
        )

    Example (corpus-aware observed — heuristic suspicions):
        result = diagnose(
            query="What is the P1 SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.12}],
            answer="Wrong answer",
            corpus=my_corpus_index,
            mode="corpus_aware",
        )

    Example (corpus-aware reference-assisted — definitive diagnosis):
        result = diagnose(
            query="What is the P1 SLA?",
            retrieved_chunks=[{"id": "doc1", "text": "...", "score": 0.12}],
            answer="Wrong answer",
            corpus=my_corpus_index,
            mode="corpus_aware",
            reference_answer="15 minutes",
        )

    **Reference-answer matching** (used only when `reference_answer` is provided):
    - Exact token subsequence match after normalization (lowercase, punctuation removed, whitespace collapsed).
    - Fallback: ≥80% overlap of "key tokens" (length > 2 or numeric). Numeric tokens in gold MUST be present.
    - This is **not semantic equivalence**; paraphrased answers may not match. For best results,
      provide `reference_answer` as a verbatim substring of a corpus chunk.
    """
    # 1. Resolve Trace representation
    if trace_input is not None:
        if isinstance(trace_input, Trace):
            t = trace_input
            if score_type is not None:
                t = Trace(
                    query=t.query,
                    retrieved_chunks=t.retrieved_chunks,
                    scores=t.scores,
                    ranks=t.ranks,
                    context=t.context,
                    answer=t.answer,
                    candidate_chunks=t.candidate_chunks,
                    candidate_scores=t.candidate_scores,
                    model=t.model,
                    latency_ms=t.latency_ms,
                    top_k=t.top_k,
                    case_id=t.case_id,
                    metadata=t.metadata,
                    score_type=score_type,
                )
        elif isinstance(trace_input, ObservedExecution):
            t = Trace(
                query=trace_input.question,
                retrieved_chunk_ids=list(trace_input.retrieved_chunk_ids),
                scores=trace_input.scores,
                answer=trace_input.generated_answer,
                candidate_chunks=trace_input.candidate_chunk_ids,
                candidate_scores=trace_input.candidate_scores,
                top_k=trace_input.top_k,
                case_id=trace_input.case_id,
                metadata=trace_input.metadata,
                score_type=score_type or trace_input.score_type,
            )
        elif isinstance(trace_input, dict):
            t = Trace.from_dict(trace_input)
            if score_type is not None:
                t.score_type = score_type
        elif isinstance(trace_input, (str, Path)):
            t = Trace.from_json(trace_input)
            if score_type is not None:
                t.score_type = score_type
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
            score_type=score_type or "similarity",
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
            from ragmortem.retrieval import CorpusRetriever

            corpus_app = CorpusRetriever(corpus=target_corpus)
        elif isinstance(target_corpus, Sequence):
            from ragmortem.retrieval import CorpusRetriever

            corpus_app = CorpusRetriever(corpus=list(target_corpus))
        else:
            corpus_app = target_corpus

    # 4. Execute Diagnosis
    diagnoser = RAGDiagnoser(
        app=corpus_app if norm_mode == "corpus_aware" else None,
        profile=profile,
    )
    obs_exec = t.to_observed_execution()

    if reference_answer is not None:
        return diagnoser.diagnose_reference_assisted(obs_exec, reference_answer=reference_answer)

    return diagnoser.diagnose_observed(
        trace=obs_exec,
        mode=norm_mode,
        corpus=corpus_app if norm_mode == "corpus_aware" else None,
    )
