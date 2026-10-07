"""Core diagnostic engine for RAG failure root-cause attribution.

Supports three diagnostic operating modes:
1. Observed Mode (Default): Purely observable execution telemetry without gold labels.
2. Reference-Assisted Mode: Observational telemetry + user-supplied reference answer.
3. Oracle Benchmark Mode: Controlled research upper bound using ground-truth gold chunks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.faults.eval import is_abstaining, is_answer_correct
from ragmortem.faults.models import InjectedFaultCase
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk, RagResult


@dataclass
class ObservedExecution:
    """Standardized trace representing an observed RAG pipeline execution.

    Contains ONLY observable telemetry available to a production debugger.
    Explicitly excludes gold_chunk_id, gold_answer, fault_type, and injected metadata.
    """

    question: str
    retrieved_chunk_ids: list[str]
    generated_answer: str
    candidate_chunk_ids: list[str] | None = None
    scores: list[float] | None = None
    candidate_scores: list[float] | None = None
    top_k: int = 3
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "generated_answer": self.generated_answer,
            "candidate_chunk_ids": list(self.candidate_chunk_ids) if self.candidate_chunk_ids else None,
            "scores": list(self.scores) if self.scores else None,
            "candidate_scores": list(self.candidate_scores) if self.candidate_scores else None,
            "top_k": self.top_k,
            "case_id": self.case_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservedExecution:
        """Construct ObservedExecution, safely dropping any benchmark labels."""
        raw_metadata = data.get("metadata", {})
        clean_metadata = {k: v for k, v in raw_metadata.items() if not k.startswith("injected_")}
        ret_ids = list(data.get("retrieved_chunk_ids") or data.get("retrieved_ids", []))
        cand_ids = data.get("candidate_chunk_ids") or data.get("candidate_retrieved_ids")
        return cls(
            question=str(data["question"]),
            retrieved_chunk_ids=ret_ids,
            generated_answer=str(data.get("generated_answer") or data.get("answer", "")),
            candidate_chunk_ids=list(cand_ids) if cand_ids is not None else None,
            scores=list(data["scores"]) if data.get("scores") is not None else None,
            candidate_scores=list(data["candidate_scores"]) if data.get("candidate_scores") is not None else None,
            top_k=int(data.get("top_k", len(ret_ids) if ret_ids else 3)),
            case_id=data.get("case_id"),
            metadata=clean_metadata,
        )

    @classmethod
    def from_injected_case(cls, case: InjectedFaultCase) -> ObservedExecution:
        """Construct ObservedExecution strictly omitting all gold labels and injected metadata."""
        fault_exec = case.fault
        return cls(
            question=case.question,
            retrieved_chunk_ids=list(fault_exec.retrieved_ids),
            generated_answer=fault_exec.answer,
            candidate_chunk_ids=list(fault_exec.candidate_retrieved_ids) if fault_exec.candidate_retrieved_ids else None,
            scores=list(fault_exec.scores) if fault_exec.scores else None,
            top_k=len(fault_exec.retrieved_ids) if fault_exec.retrieved_ids else 3,
            case_id=case.case_id,
            metadata={},
        )

    @classmethod
    def from_rag_result(cls, result: RagResult, top_k: int = 3) -> ObservedExecution:
        return cls(
            question=result.question,
            retrieved_chunk_ids=result.retrieved_ids,
            generated_answer=result.answer,
            candidate_chunk_ids=result.candidate_ids if result.candidate_chunks else None,
            scores=result.scores,
            candidate_scores=result.candidate_scores,
            top_k=top_k,
            metadata=dict(result.metadata),
        )


@dataclass
class ExecutionTrace:
    """Standardized trace representing an observed RAG execution for oracle benchmark mode.

    Retains benchmark reference fields (gold_chunk_id, gold_answer) for counterfactual research.
    """

    question: str
    generated_answer: str
    retrieved_chunk_ids: list[str]
    candidate_chunk_ids: list[str] | None = None
    scores: list[float] | None = None
    gold_chunk_id: str | None = None
    gold_answer: str | None = None
    question_type: str = "answerable"
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_injected_case(cls, case: InjectedFaultCase) -> ExecutionTrace:
        fault_exec = case.fault
        q_type = "unanswerable" if case.gold_chunk_id is None else "answerable"
        return cls(
            question=case.question,
            generated_answer=fault_exec.answer,
            retrieved_chunk_ids=list(fault_exec.retrieved_ids),
            candidate_chunk_ids=list(fault_exec.candidate_retrieved_ids) if fault_exec.candidate_retrieved_ids else None,
            scores=fault_exec.scores,
            gold_chunk_id=case.gold_chunk_id,
            gold_answer=case.gold_answer,
            question_type=q_type,
            case_id=case.case_id,
            metadata=dict(case.metadata),
        )

    @classmethod
    def from_rag_result(
        cls,
        result: RagResult,
        gold_chunk_id: str | None = None,
        gold_answer: str | None = None,
        question_type: str = "answerable",
    ) -> ExecutionTrace:
        return cls(
            question=result.question,
            generated_answer=result.answer,
            retrieved_chunk_ids=result.retrieved_ids,
            candidate_chunk_ids=result.candidate_ids if result.candidate_chunks else None,
            scores=result.scores,
            gold_chunk_id=gold_chunk_id,
            gold_answer=gold_answer,
            question_type=question_type,
            metadata=dict(result.metadata),
        )


@dataclass
class DiagnosticResult:
    """The structured diagnosis and evidence produced by RAGmortem."""

    diagnosis: FailureType
    confidence: float
    evidence: list[str]
    mode: str = "observed"
    is_suspected: bool = False
    original_rank: int | None = None
    candidate_rank: int | None = None
    candidate_count: int = 0
    oracle_replay_correct: bool | None = None
    oracle_answer: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis.value,
            "canonical_diagnosis": self.diagnosis.canonical.value,
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "mode": self.mode,
            "is_suspected": self.is_suspected,
            "original_rank": self.original_rank,
            "candidate_rank": self.candidate_rank,
            "candidate_count": self.candidate_count,
            "oracle_replay_correct": self.oracle_replay_correct,
            "oracle_answer": self.oracle_answer,
            "signals": dict(self.signals),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticResult:
        return cls(
            diagnosis=FailureType(data["diagnosis"]),
            confidence=float(data.get("confidence", 1.0)),
            evidence=list(data.get("evidence", [])),
            mode=str(data.get("mode", "observed")),
            is_suspected=bool(data.get("is_suspected", False)),
            original_rank=data.get("original_rank"),
            candidate_rank=data.get("candidate_rank"),
            candidate_count=int(data.get("candidate_count", 0)),
            oracle_replay_correct=data.get("oracle_replay_correct"),
            oracle_answer=data.get("oracle_answer"),
            signals=dict(data.get("signals", {})),
            metadata=dict(data.get("metadata", {})),
        )


class RAGDiagnoser:
    """Failure diagnostic engine for RAG applications.

    Provides three diagnosis interfaces:
    - diagnose_observed: Evaluates purely observable telemetry (production default).
    - diagnose_reference_assisted: Uses developer expected answer without gold chunk ID.
    - diagnose_oracle: Controlled research upper bound using counterfactual replay.
    """

    def __init__(
        self,
        app: ReferenceRagApp | None = None,
        corpus_chunks: list[Chunk] | None = None,
    ) -> None:
        if app is not None:
            self.app = app
        else:
            self.app = ReferenceRagApp(mock_mode=True)

        if corpus_chunks is not None:
            self.chunks_by_id = {c.id: c for c in corpus_chunks}
        else:
            self.chunks_by_id = {c.id: c for c in self.app.chunks}

    # =========================================================================
    # 1. REALISTIC OBSERVATIONAL MODE (Day 5 Default)
    # =========================================================================

    def diagnose_observed(self, trace: ObservedExecution) -> DiagnosticResult:
        """Diagnose root cause using ONLY observable execution telemetry without gold labels.

        Never accesses fault_type, gold_chunk_id, or gold_answer.
        """
        evidence: list[str] = []
        signals: dict[str, Any] = {}

        # Signal 1: Answer abstention inspection
        is_refusal = is_abstaining(trace.generated_answer)
        signals["is_refusal"] = is_refusal

        # Signal 2: Score telemetry
        scores = trace.scores or []
        top_score = scores[0] if scores else None
        signals["top_score"] = top_score

        top_k = trace.top_k
        cand_ids = trace.candidate_chunk_ids
        cand_count = len(cand_ids) if cand_ids else len(trace.retrieved_chunk_ids)
        signals["candidate_count"] = cand_count

        has_extended_candidates = bool(cand_ids and len(cand_ids) > top_k)
        signals["has_extended_candidates"] = has_extended_candidates

        # Signal 3: Corpus search audit (if corpus index is accessible)
        clean_corpus_score = None
        clean_corpus_top_id = None
        if self.app is not None and hasattr(self.app, "retrieve"):
            try:
                ret_chunks, ret_scores = self.app.retrieve(trace.question, k=1)
                if ret_scores:
                    clean_corpus_score = float(ret_scores[0])
                    clean_corpus_top_id = ret_chunks[0].id
            except Exception:
                pass
        signals["clean_corpus_score"] = clean_corpus_score
        signals["clean_corpus_top_id"] = clean_corpus_top_id

        # --- DIAGNOSTIC REASONING ---

        # PATH A: System generated a refusal/abstention
        if is_refusal:
            if top_score is not None and top_score >= 0.55:
                evidence.append(
                    f"Model abstained from answering despite high retrieval score (top score: {top_score:.3f})."
                )
                evidence.append(
                    "Root cause (suspected): Generator failed to extract answers from provided context (false refusal)."
                )
                return DiagnosticResult(
                    diagnosis=FailureType.GENERATION_SUSPECTED,
                    confidence=0.75,
                    evidence=evidence,
                    mode="observed",
                    is_suspected=True,
                    signals=signals,
                )
            else:
                evidence.append("System correctly abstained: generated refusal matches low/moderate retrieval relevance.")
                return DiagnosticResult(
                    diagnosis=FailureType.NO_FAILURE,
                    confidence=0.90,
                    evidence=evidence,
                    mode="observed",
                    is_suspected=False,
                    signals=signals,
                )

        # PATH B: Substantive response generated. Evaluate ranking vs generation vs retrieval vs abstention.

        # Signal 1: Ranking Cutoff Probe
        # Candidates exist outside top-k and top-k retrieval was strong
        if has_extended_candidates and (top_score is None or top_score >= 0.55):
            evidence.append(
                f"Candidate retrieval pool contained {cand_count} chunks, but context window was truncated to top-{top_k}."
            )
            evidence.append("Strong candidates were cut off by ranking threshold.")
            evidence.append(
                "Root cause (suspected): Candidate retrieval succeeded, but ranking truncation omitted relevant evidence."
            )
            return DiagnosticResult(
                diagnosis=FailureType.RANKING_SUSPECTED,
                confidence=0.85,
                evidence=evidence,
                mode="observed",
                is_suspected=True,
                candidate_count=cand_count,
                signals=signals,
            )

        # Signal 2: Generation Failure Probe
        # High retrieval score in the trace context, but answer failed
        if top_score is not None and top_score >= 0.50:
            evidence.append(
                f"Retrieved context had high relevance score to question (top score: {top_score:.3f})."
            )
            evidence.append(
                "Context was directly supplied in top-k prompt window, but generated answer appears ungrounded or contradictory."
            )
            evidence.append(
                "Root cause (suspected): Generator failed to incorporate supplied context evidence."
            )
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_SUSPECTED,
                confidence=0.80,
                evidence=evidence,
                mode="observed",
                is_suspected=True,
                signals=signals,
            )

        # Signal 3: Weak retrieval scores in trace
        # Distinguish retrieval miss (corpus has info) from unanswerable question (corpus lacks info)
        score_str = f"{top_score:.3f}" if top_score is not None else "low"
        if clean_corpus_score is not None:
            if clean_corpus_score >= 0.48:
                evidence.append(
                    f"Retriever returned low-relevance chunks (top score: {score_str})."
                )
                evidence.append(
                    f"Corpus audit shows indexed documents exist with high semantic relevance to question (clean score: {clean_corpus_score:.3f})."
                )
                evidence.append(
                    "Root cause (suspected): Retriever failed to locate relevant documents present in the corpus."
                )
                return DiagnosticResult(
                    diagnosis=FailureType.RETRIEVAL_SUSPECTED,
                    confidence=0.85,
                    evidence=evidence,
                    mode="observed",
                    is_suspected=True,
                    signals=signals,
                )
            else:
                evidence.append(
                    f"Retriever returned low-relevance chunks (top score: {score_str})."
                )
                evidence.append(
                    f"Corpus audit confirms no documents in the index match the question (maximum corpus score: {clean_corpus_score:.3f})."
                )
                evidence.append(
                    "System produced a substantive answer when the corpus contains no supporting evidence."
                )
                evidence.append(
                    "Root cause (suspected): Unanswerable question from corpus; system hallucinated when it should have abstained."
                )
                return DiagnosticResult(
                    diagnosis=FailureType.ABSTENTION_SUSPECTED,
                    confidence=0.85,
                    evidence=evidence,
                    mode="observed",
                    is_suspected=True,
                    signals=signals,
                )
        else:
            # Without corpus index access, weak scores cannot distinguish retrieval miss from unanswerable query
            evidence.append(
                f"Retriever returned low-relevance chunks (top score: {top_score if top_score is not None else 'unknown'})."
            )
            evidence.append(
                "Without access to the corpus index, cannot determine whether retriever missed existing documents or question is unanswerable."
            )
            evidence.append("Root cause: Insufficient evidence to disambiguate retrieval miss from abstention failure.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                confidence=0.0,
                evidence=evidence,
                mode="observed",
                is_suspected=False,
                signals=signals,
            )

    # =========================================================================
    # 2. REFERENCE-ASSISTED MODE
    # =========================================================================

    def diagnose_reference_assisted(
        self,
        trace: ObservedExecution,
        reference_answer: str,
    ) -> DiagnosticResult:
        """Diagnose root cause when an explicit reference answer is provided by user/application.

        Does NOT require gold_chunk_id. Uses reference_answer to deterministically verify correctness
        and search retrieved vs candidate chunks for supporting evidence.
        """
        evidence: list[str] = []
        signals: dict[str, Any] = {"reference_answer": reference_answer}

        # 1. Deterministic answer verification
        if is_answer_correct(trace.generated_answer, reference_answer):
            evidence.append(f"Generated answer satisfies expected reference answer: '{trace.generated_answer}'.")
            return DiagnosticResult(
                diagnosis=FailureType.NO_FAILURE,
                confidence=1.0,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                signals=signals,
            )

        # 2. Check if model abstained on an answerable query
        if is_abstaining(trace.generated_answer):
            evidence.append(
                f"Reference answer exists ('{reference_answer}'), but system abstained: '{trace.generated_answer}'."
            )
            evidence.append("Root cause: System unnecessarily abstained when an answer was expected.")
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_PROVEN,
                confidence=0.95,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                signals=signals,
            )

        # 3. Check if any top-k context chunk contains the reference answer
        ref_in_topk = False
        matching_topk_id = None
        for cid in trace.retrieved_chunk_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk and is_answer_correct(chunk.text, reference_answer):
                ref_in_topk = True
                matching_topk_id = cid
                break

        if ref_in_topk:
            evidence.append(
                f"Retrieved context chunk '{matching_topk_id}' in top-{trace.top_k} contains the reference answer."
            )
            evidence.append(
                f"Generated answer '{trace.generated_answer}' failed to match expected answer '{reference_answer}'."
            )
            evidence.append(
                "Root cause: Context was present in the prompt, but generator failed to incorporate it."
            )
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_PROVEN,
                confidence=1.0,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                signals=signals,
            )

        # 4. Check if any candidate chunk outside top-k contains the reference answer
        ref_in_candidate = False
        matching_cand_id = None
        matching_cand_rank = None
        cand_ids = trace.candidate_chunk_ids or []
        for rank_idx, cid in enumerate(cand_ids, start=1):
            if rank_idx > trace.top_k:
                chunk = self.chunks_by_id.get(cid)
                if chunk and is_answer_correct(chunk.text, reference_answer):
                    ref_in_candidate = True
                    matching_cand_id = cid
                    matching_cand_rank = rank_idx
                    break

        if ref_in_candidate:
            evidence.append(
                f"Candidate retrieval pool contained supporting chunk '{matching_cand_id}' at rank #{matching_cand_rank}."
            )
            evidence.append(
                f"Prompt context was truncated to top-{trace.top_k}, pushing the supporting evidence out of context."
            )
            evidence.append(
                "Root cause: Candidate retrieval succeeded, but ranking truncation omitted the evidence."
            )
            return DiagnosticResult(
                diagnosis=FailureType.RANKING_MISS,
                confidence=1.0,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                candidate_rank=matching_cand_rank,
                candidate_count=len(cand_ids),
                signals=signals,
            )

        # 5. Check if reference answer exists in broader corpus
        ref_in_corpus = False
        matching_corpus_id = None
        for cid, chunk in self.chunks_by_id.items():
            if is_answer_correct(chunk.text, reference_answer):
                ref_in_corpus = True
                matching_corpus_id = cid
                break

        if ref_in_corpus:
            evidence.append(
                f"Corpus contains supporting document '{matching_corpus_id}', but retriever failed to return it in candidate pool."
            )
            evidence.append(
                "Root cause: Retriever failed to locate evidence document present in the corpus."
            )
            return DiagnosticResult(
                diagnosis=FailureType.RETRIEVAL_MISS,
                confidence=1.0,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                signals=signals,
            )
        else:
            evidence.append(
                f"Supplied reference answer '{reference_answer}' was not found in any corpus document."
            )
            evidence.append("Root cause: Corpus does not support question; system should have abstained.")
            return DiagnosticResult(
                diagnosis=FailureType.SHOULD_ABSTAIN,
                confidence=0.90,
                evidence=evidence,
                mode="reference_assisted",
                is_suspected=False,
                signals=signals,
            )

    # =========================================================================
    # 3. ORACLE BENCHMARK MODE (Day 4 Upper Bound)
    # =========================================================================

    def diagnose_oracle(self, trace: ExecutionTrace) -> DiagnosticResult:
        """Diagnose root cause using controlled counterfactual oracle replay and gold chunks."""
        evidence: list[str] = []

        # 1. Check Unanswerable Question / Abstention Failure
        is_unanswerable = (trace.question_type == "unanswerable") or (trace.gold_chunk_id is None)
        if is_unanswerable:
            substantive_refusal = is_abstaining(trace.generated_answer)
            if substantive_refusal:
                evidence.append("Question is labeled unanswerable and system correctly refused to answer.")
                return DiagnosticResult(
                    diagnosis=FailureType.NO_FAILURE,
                    confidence=1.0,
                    evidence=evidence,
                    mode="oracle",
                )
            else:
                evidence.append(
                    "Question is unanswerable from the corpus; no supporting reference chunk exists."
                )
                evidence.append(
                    f"System generated a non-abstaining substantive response: '{trace.generated_answer}'."
                )
                evidence.append(
                    "Root cause: System hallucinated an answer when it should have abstained."
                )
                return DiagnosticResult(
                    diagnosis=FailureType.SHOULD_ABSTAIN,
                    confidence=1.0,
                    evidence=evidence,
                    mode="oracle",
                )

        # 2. Check if Answerable Question was already correct
        if trace.gold_answer and is_answer_correct(trace.generated_answer, trace.gold_answer):
            evidence.append(f"Answer correctly matches gold criteria: '{trace.generated_answer}'.")
            return DiagnosticResult(
                diagnosis=FailureType.NO_FAILURE,
                confidence=1.0,
                evidence=evidence,
                mode="oracle",
            )

        # 3. Check for Missing Evidence
        gold_id = trace.gold_chunk_id
        if not gold_id:
            evidence.append("Insufficient evidence: answerable question has no gold reference chunk.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                confidence=0.0,
                evidence=evidence,
                mode="oracle",
            )

        gold_chunk = self.chunks_by_id.get(gold_id)
        if not gold_chunk:
            evidence.append(f"Insufficient evidence: gold chunk '{gold_id}' not found in corpus.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                confidence=0.0,
                evidence=evidence,
                mode="oracle",
            )

        # 4. Rank Extraction
        retrieved_ids = trace.retrieved_chunk_ids
        top_k = len(retrieved_ids)
        orig_rank = (retrieved_ids.index(gold_id) + 1) if gold_id in retrieved_ids else None

        candidates = trace.candidate_chunk_ids if trace.candidate_chunk_ids is not None else retrieved_ids
        cand_count = len(candidates)
        cand_rank = (candidates.index(gold_id) + 1) if gold_id in candidates else None

        # 5. Counterfactual Oracle Replay
        oracle_answer = self.app._generate_mock_answer(trace.question, [gold_chunk])
        oracle_correct = is_answer_correct(oracle_answer, trace.gold_answer) if trace.gold_answer else True

        # CASE A: Gold chunk was present in prompt context window
        if orig_rank is not None:
            evidence.append(
                f"Gold chunk '{gold_id}' was present in the generation context at rank #{orig_rank}."
            )
            evidence.append(
                f"Original generated answer was incorrect: '{trace.generated_answer}'."
            )
            if oracle_correct:
                evidence.append(
                    "Oracle replay with gold chunk alone produced the correct answer; context competition or distractors induced generation failure."
                )
            else:
                evidence.append(
                    "Oracle replay with gold chunk alone also failed; generator ignored or misattributed context."
                )
            evidence.append(
                "Root cause: Retrieval succeeded, but generator failed to incorporate provided context."
            )
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_IGNORED_CONTEXT,
                confidence=1.0,
                evidence=evidence,
                mode="oracle",
                original_rank=orig_rank,
                candidate_rank=cand_rank or orig_rank,
                candidate_count=cand_count,
                oracle_replay_correct=oracle_correct,
                oracle_answer=oracle_answer,
            )

        # CASE B: Gold chunk was absent from final prompt context
        if not oracle_correct:
            evidence.append(
                f"Gold chunk '{gold_id}' was absent from context, but oracle replay with gold chunk alone STILL failed."
            )
            evidence.append(
                "Root cause: Generator is fundamentally incapable of extracting the factual answer from evidence."
            )
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_IGNORED_CONTEXT,
                confidence=0.9,
                evidence=evidence,
                mode="oracle",
                original_rank=None,
                candidate_rank=cand_rank,
                candidate_count=cand_count,
                oracle_replay_correct=False,
                oracle_answer=oracle_answer,
            )

        evidence.append("Counterfactual oracle replay with gold chunk alone corrected the answer.")

        if cand_rank is not None and cand_rank > top_k:
            evidence.append(
                f"Gold chunk '{gold_id}' was present in the candidate retrieval pool at rank #{cand_rank} (pool size: {cand_count})."
            )
            evidence.append(
                f"Generation context was truncated to top-{top_k}, pushing gold chunk outside prompt context."
            )
            evidence.append(
                "Root cause: Candidate retrieval succeeded, but ranking truncation omitted gold evidence."
            )
            return DiagnosticResult(
                diagnosis=FailureType.RANKING_MISS,
                confidence=1.0,
                evidence=evidence,
                mode="oracle",
                original_rank=None,
                candidate_rank=cand_rank,
                candidate_count=cand_count,
                oracle_replay_correct=True,
                oracle_answer=oracle_answer,
            )
        else:
            evidence.append(
                f"Gold chunk '{gold_id}' was completely absent from candidate retrieval pool (checked {cand_count} candidates)."
            )
            evidence.append("Root cause: Retriever failed to return required evidence chunk.")
            return DiagnosticResult(
                diagnosis=FailureType.RETRIEVAL_MISS,
                confidence=1.0,
                evidence=evidence,
                mode="oracle",
                original_rank=None,
                candidate_rank=None,
                candidate_count=cand_count,
                oracle_replay_correct=True,
                oracle_answer=oracle_answer,
            )

    diagnose_with_reference = diagnose_oracle

    # =========================================================================
    # 4. UNIFIED DISPATCHER
    # =========================================================================

    def diagnose(
        self,
        trace: ExecutionTrace | ObservedExecution,
        mode: str | None = None,
        reference_answer: str | None = None,
    ) -> DiagnosticResult:
        """Unified entrypoint for failure diagnosis."""
        if mode == "oracle" or (mode is None and isinstance(trace, ExecutionTrace)):
            if isinstance(trace, ExecutionTrace):
                return self.diagnose_oracle(trace)
            else:
                raise ValueError("Oracle mode requires an ExecutionTrace with gold reference metadata.")

        if mode == "reference_assisted" or reference_answer is not None:
            if isinstance(trace, ExecutionTrace):
                obs_trace = ObservedExecution.from_dict(trace.__dict__)
            else:
                obs_trace = trace
            if not reference_answer:
                raise ValueError("Reference-assisted diagnosis requires a reference_answer string.")
            return self.diagnose_reference_assisted(obs_trace, reference_answer)

        if isinstance(trace, ExecutionTrace):
            obs_trace = ObservedExecution.from_dict(trace.__dict__)
        else:
            obs_trace = trace

        return self.diagnose_observed(obs_trace)


def evaluate_observed_on_dataset(
    cases: Sequence[InjectedFaultCase],
    diagnoser: RAGDiagnoser,
) -> dict[str, Any]:
    """Run observational diagnoser over benchmark cases without exposing any labels to the engine."""
    records: list[dict[str, Any]] = []
    categories = [
        FailureType.RETRIEVAL_MISS.value,
        FailureType.RANKING_MISS.value,
        FailureType.GENERATION_IGNORED_CONTEXT.value,
        FailureType.SHOULD_ABSTAIN.value,
    ]

    confusion_matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in categories + [FailureType.UNKNOWN.value, FailureType.NO_FAILURE.value]}
        for gt in categories
    }

    correct_count = 0
    unknown_count = 0
    total = len(cases)

    for case in cases:
        obs_trace = ObservedExecution.from_injected_case(case)
        diag_res = diagnoser.diagnose_observed(obs_trace)

        gt_str = case.fault_type
        pred_str = diag_res.diagnosis.canonical.value
        is_correct = (pred_str == gt_str)

        if is_correct:
            correct_count += 1
        if diag_res.diagnosis == FailureType.UNKNOWN:
            unknown_count += 1

        if gt_str in confusion_matrix:
            if pred_str in confusion_matrix[gt_str]:
                confusion_matrix[gt_str][pred_str] += 1
            else:
                confusion_matrix[gt_str][FailureType.UNKNOWN.value] += 1

        records.append({
            "case_id": case.case_id,
            "question_id": case.question_id,
            "ground_truth": gt_str,
            "predicted_diagnosis": pred_str,
            "raw_diagnosis": diag_res.diagnosis.value,
            "is_correct": is_correct,
            "is_suspected": diag_res.is_suspected,
            "confidence": diag_res.confidence,
            "evidence": diag_res.evidence,
            "signals": diag_res.signals,
            "candidate_count": diag_res.candidate_count,
            "mode": diag_res.mode,
        })

    per_class: dict[str, dict[str, Any]] = {}
    for cat in categories:
        tp = confusion_matrix[cat][cat]
        fn = sum(confusion_matrix[cat][p] for p in confusion_matrix[cat] if p != cat)
        fp = sum(confusion_matrix[gt][cat] for gt in categories if gt != cat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cat] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    accuracy = correct_count / total if total > 0 else 0.0
    coverage = (total - unknown_count) / total if total > 0 else 0.0

    return {
        "total_cases": total,
        "correct_diagnoses": correct_count,
        "incorrect_diagnoses": total - correct_count - unknown_count,
        "unknown_diagnoses": unknown_count,
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "records": records,
    }


def evaluate_diagnoser_on_dataset(
    cases: Sequence[InjectedFaultCase],
    diagnoser: RAGDiagnoser,
) -> dict[str, Any]:
    """Run oracle diagnoser over benchmark cases for counterfactual upper-bound measurement."""
    records: list[dict[str, Any]] = []
    categories = [
        FailureType.RETRIEVAL_MISS.value,
        FailureType.RANKING_MISS.value,
        FailureType.GENERATION_IGNORED_CONTEXT.value,
        FailureType.SHOULD_ABSTAIN.value,
    ]

    confusion_matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in categories + [FailureType.UNKNOWN.value, FailureType.NO_FAILURE.value]}
        for gt in categories
    }

    correct_count = 0
    unknown_count = 0
    total = len(cases)

    for case in cases:
        trace = ExecutionTrace.from_injected_case(case)
        diag_res = diagnoser.diagnose_oracle(trace)

        gt_str = case.fault_type
        pred_str = diag_res.diagnosis.value
        is_correct = (pred_str == gt_str)

        if is_correct:
            correct_count += 1
        if diag_res.diagnosis == FailureType.UNKNOWN:
            unknown_count += 1

        if gt_str in confusion_matrix:
            if pred_str in confusion_matrix[gt_str]:
                confusion_matrix[gt_str][pred_str] += 1
            else:
                confusion_matrix[gt_str][FailureType.UNKNOWN.value] += 1

        records.append({
            "case_id": case.case_id,
            "question_id": case.question_id,
            "ground_truth": gt_str,
            "predicted_diagnosis": pred_str,
            "is_correct": is_correct,
            "confidence": diag_res.confidence,
            "evidence": diag_res.evidence,
            "oracle_replay_correct": diag_res.oracle_replay_correct,
            "original_rank": diag_res.original_rank,
            "candidate_rank": diag_res.candidate_rank,
            "candidate_count": diag_res.candidate_count,
            "mode": diag_res.mode,
        })

    per_class: dict[str, dict[str, Any]] = {}
    for cat in categories:
        tp = confusion_matrix[cat][cat]
        fn = sum(confusion_matrix[cat][p] for p in confusion_matrix[cat] if p != cat)
        fp = sum(confusion_matrix[gt][cat] for gt in categories if gt != cat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cat] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    accuracy = correct_count / total if total > 0 else 0.0
    coverage = (total - unknown_count) / total if total > 0 else 0.0

    return {
        "total_cases": total,
        "correct_diagnoses": correct_count,
        "incorrect_diagnoses": total - correct_count - unknown_count,
        "unknown_diagnoses": unknown_count,
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "records": records,
    }
