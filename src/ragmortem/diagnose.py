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
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from examples.reference_rag.app import ReferenceRagApp

from ragmortem.faults.eval import is_abstaining, is_answer_correct
from ragmortem.faults.models import InjectedFaultCase
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk, RagResult
from ragmortem.support import evaluate_corpus_answerability, SupportClassification, CorpusSupportReport


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


from enum import Enum


class TelemetryCompleteness(str, Enum):
    """Assessment of telemetry field completeness in an execution trace."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class EvidenceModel:
    """Structured, measurable evidence signals underlying a diagnosis.

    These are deterministic measurements and heuristics, NOT calibrated probabilities.
    """

    retrieval_strength: float = 0.0
    ranking_strength: float = 0.0
    generation_support: float = 0.0
    answerability_signal: float = 0.0
    corpus_support_strength: float = 0.0
    telemetry_completeness: TelemetryCompleteness = TelemetryCompleteness.COMPLETE
    ambiguity: float = 0.0
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_strength": round(self.retrieval_strength, 4),
            "ranking_strength": round(self.ranking_strength, 4),
            "generation_support": round(self.generation_support, 4),
            "answerability_signal": round(self.answerability_signal, 4),
            "corpus_support_strength": round(self.corpus_support_strength, 4),
            "telemetry_completeness": self.telemetry_completeness.value,
            "ambiguity": round(self.ambiguity, 4),
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
        }


@dataclass
class DiagnosticResult:
    """The structured diagnosis, evidence score, and reasoning produced by RAGmortem."""

    diagnosis: FailureType
    evidence: list[str]
    evidence_score: float = 0.0
    confidence: float = 0.0  # Kept as alias to evidence_score for backward compatibility
    evidence_model: EvidenceModel | None = None
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    telemetry_completeness: str = "COMPLETE"
    mode: str = "observed"
    is_suspected: bool = False
    original_rank: int | None = None
    candidate_rank: int | None = None
    candidate_count: int = 0
    oracle_replay_correct: bool | None = None
    oracle_answer: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence == 0.0 and self.evidence_score != 0.0:
            self.confidence = self.evidence_score
        elif self.evidence_score == 0.0 and self.confidence != 0.0:
            self.evidence_score = self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis.value,
            "canonical_diagnosis": self.diagnosis.canonical.value,
            "evidence_score": round(self.evidence_score, 4),
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
            "telemetry_completeness": self.telemetry_completeness,
            "evidence_model": self.evidence_model.to_dict() if self.evidence_model else None,
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
        ev_score = float(data.get("evidence_score", data.get("confidence", 0.0)))
        return cls(
            diagnosis=FailureType(data["diagnosis"]),
            evidence_score=ev_score,
            confidence=float(data.get("confidence", ev_score)),
            evidence=list(data.get("evidence", [])),
            reasons=list(data.get("reasons", [])),
            limitations=list(data.get("limitations", [])),
            telemetry_completeness=str(data.get("telemetry_completeness", "COMPLETE")),
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
    """Uncertainty-aware failure diagnostic engine for RAG applications.

    Provides three diagnosis interfaces:
    - diagnose_observed: Evaluates observable telemetry without gold labels.
    - diagnose_reference_assisted: Uses developer expected answer without gold chunk ID.
    - diagnose_oracle: Controlled research upper bound using counterfactual replay.
    """

    def __init__(
        self,
        app: ReferenceRagApp | None = None,
        corpus_chunks: list[Chunk] | None = None,
        score_margin_threshold: float = 0.05,
        ambiguity_band: tuple[float, float] = (0.47, 0.53),
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.app = app
        self.score_margin_threshold = score_margin_threshold
        self.ambiguity_band = ambiguity_band
        self.model_name = model_name

        if corpus_chunks is not None:
            self.chunks_by_id = {c.id: c for c in corpus_chunks}
        elif self.app is not None and hasattr(self.app, "chunks"):
            self.chunks_by_id = {c.id: c for c in self.app.chunks}
        else:
            self.chunks_by_id = {}

    # =========================================================================
    # 1. REALISTIC OBSERVATIONAL MODE (Uncertainty-Aware)
    # =========================================================================

    def diagnose_observed(
        self,
        trace: ObservedExecution,
        mode: str = "auto",
        corpus: Any | None = None,
    ) -> DiagnosticResult:
        """Diagnose root cause using ONLY observable execution telemetry without gold labels.

        Treats UNKNOWN as a first-class result whenever evidence is missing, conflicting,
        or insufficient to disambiguate causes.
        """
        evidence: list[str] = []
        reasons: list[str] = []
        limitations: list[str] = []
        signals: dict[str, Any] = {}

        # -------------------------------------------------------------
        # 1. Telemetry Completeness Validation
        # -------------------------------------------------------------
        if not trace.question or not trace.question.strip():
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=["Trace query/question is empty or missing."],
                reasons=["Insufficient telemetry: Missing user query."],
                telemetry_completeness=TelemetryCompleteness.INSUFFICIENT.value,
                mode=mode,
                signals=signals,
            )

        if not trace.generated_answer or not trace.generated_answer.strip():
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=["Generated answer is empty or missing from trace."],
                reasons=["Insufficient telemetry: Missing generated answer payload."],
                telemetry_completeness=TelemetryCompleteness.INSUFFICIENT.value,
                mode=mode,
                signals=signals,
            )

        if not trace.retrieved_chunk_ids:
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=["Retrieved chunk list is empty (no context supplied to generator)."],
                reasons=["Insufficient telemetry: Empty context window in trace."],
                telemetry_completeness=TelemetryCompleteness.INSUFFICIENT.value,
                mode=mode,
                signals=signals,
            )

        if trace.scores is None or len(trace.scores) == 0:
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=["Retrieval scores are missing from execution telemetry."],
                reasons=["Insufficient telemetry: Missing similarity scores; cannot evaluate retrieval relevance."],
                telemetry_completeness=TelemetryCompleteness.INSUFFICIENT.value,
                mode=mode,
                signals=signals,
            )

        completeness = (
            TelemetryCompleteness.COMPLETE
            if trace.candidate_chunk_ids is not None
            else TelemetryCompleteness.PARTIAL
        )
        if completeness == TelemetryCompleteness.PARTIAL:
            limitations.append("Candidate pool prior to top-k truncation was not logged in trace.")

        # -------------------------------------------------------------
        # 2. Extract Relative Telemetry Signals
        # -------------------------------------------------------------
        scores = trace.scores
        top_score = scores[0]
        last_score = scores[-1]
        score_spread = top_score - last_score if len(scores) > 1 else 0.0

        signals["top_score"] = top_score
        signals["score_spread"] = score_spread
        signals["telemetry_completeness"] = completeness.value

        is_refusal = is_abstaining(trace.generated_answer)
        signals["is_refusal"] = is_refusal

        cand_ids = trace.candidate_chunk_ids
        top_k = trace.top_k
        cand_count = len(cand_ids) if cand_ids else len(trace.retrieved_chunk_ids)
        signals["candidate_count"] = cand_count
        has_extended_candidates = bool(cand_ids and len(cand_ids) > top_k)
        signals["has_extended_candidates"] = has_extended_candidates

        # -------------------------------------------------------------
        # 3. Path A: System generated a refusal/abstention
        # -------------------------------------------------------------
        if is_refusal:
            # False refusal check: context score is strong
            if top_score >= 0.54 or (top_score >= 0.50 and score_spread >= 0.10):
                evidence.append(
                    f"Model abstained from answering despite high retrieval score (top score: {top_score:.3f})."
                )
                evidence.append("Context contained salient information, but generator produced a refusal.")
                reasons.append("Generator false refusal: Abstention emitted despite relevant context.")
                ev_score = round(min(0.95, float(top_score)), 4)
                return DiagnosticResult(
                    diagnosis=FailureType.GENERATION_SUSPECTED,
                    evidence_score=ev_score,
                    evidence=evidence,
                    reasons=reasons,
                    limitations=limitations,
                    telemetry_completeness=completeness.value,
                    mode=mode,
                    is_suspected=True,
                    signals=signals,
                )
            else:
                evidence.append("System correctly abstained: generated refusal matches low retrieval relevance.")
                reasons.append("Appropriate abstention: Query has weak support and system refused.")
                return DiagnosticResult(
                    diagnosis=FailureType.NO_FAILURE,
                    evidence_score=0.90,
                    evidence=evidence,
                    reasons=reasons,
                    limitations=limitations,
                    telemetry_completeness=completeness.value,
                    mode=mode,
                    is_suspected=False,
                    signals=signals,
                )

        # -------------------------------------------------------------
        # 4. Path B: Substantive response generated
        # -------------------------------------------------------------

        # Check 1: Ranking Cutoff Probe
        if has_extended_candidates:
            cand_scores = trace.candidate_scores
            if cand_scores and len(cand_scores) > top_k:
                cutoff_gap = cand_scores[top_k - 1] - cand_scores[top_k]
            else:
                cutoff_gap = 0.01

            signals["cutoff_gap"] = cutoff_gap

            if top_score >= 0.50 and cutoff_gap <= 0.08:
                evidence.append(
                    f"Candidate retrieval pool contained {cand_count} chunks, truncated to top-{top_k}."
                )
                evidence.append(
                    f"Strong candidate exists immediately below cutoff (score gap: {cutoff_gap:.4f} <= 0.08)."
                )
                evidence.append("Root cause (suspected): Candidate retrieval succeeded, but ranking omitted evidence.")
                reasons.append("Ranking cutoff: Viable candidate demoted below top-k context window.")
                ev_score = round(max(0.70, min(0.95, 0.92 - cutoff_gap)), 4)
                return DiagnosticResult(
                    diagnosis=FailureType.RANKING_SUSPECTED,
                    evidence_score=ev_score,
                    evidence=evidence,
                    reasons=reasons,
                    limitations=limitations,
                    telemetry_completeness=completeness.value,
                    mode=mode,
                    is_suspected=True,
                    candidate_count=cand_count,
                    signals=signals,
                )
            elif top_score < 0.47:
                # Conflicting signals: candidate truncation coincides with very low scores
                evidence.append(
                    f"Candidate pool has {cand_count} items, but retrieval scores are very low (top: {top_score:.3f})."
                )
                reasons.append("Multiple plausible causes: Conflicting ranking truncation and low retrieval relevance.")
                return DiagnosticResult(
                    diagnosis=FailureType.UNKNOWN,
                    evidence_score=0.0,
                    evidence=evidence,
                    reasons=reasons,
                    limitations=limitations,
                    telemetry_completeness=completeness.value,
                    mode=mode,
                    signals=signals,
                )

        # Check 2: Borderline Ambiguity Range Check
        low_bound, high_bound = self.ambiguity_band
        if low_bound <= top_score <= high_bound:
            evidence.append(
                f"Retrieved context top score ({top_score:.3f}) lies in ambiguity band [{low_bound}, {high_bound}]."
            )
            evidence.append("Without ground truth or reference answer, cannot distinguish distractor noise from generation failure.")
            reasons.append("Ambiguous retrieval evidence: Score near decision boundary lacks clear signal separation.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations + ["No reference answer supplied to disambiguate borderline context relevance."],
                telemetry_completeness=completeness.value,
                mode=mode,
                signals=signals,
            )

        # Check 3: High Relevance Context Probe (Generator Ignored Evidence vs Terminology Trap)
        if top_score > high_bound:
            target_app = corpus if corpus is not None else self.app
            if mode == "corpus_aware" and target_app is not None and hasattr(target_app, "retrieve"):
                try:
                    ret_chunks, ret_scores = target_app.retrieve(trace.question, k=3)
                except Exception:
                    ret_chunks, ret_scores = [], []
                support_report = evaluate_corpus_answerability(trace.question, ret_chunks, ret_scores)
                signals["corpus_support_strength"] = support_report.corpus_support_strength
                signals["top_term_coverage"] = support_report.top_term_coverage
                signals["is_terminology_trap"] = support_report.is_terminology_trap
                signals["clean_corpus_score"] = support_report.top_score

                if support_report.is_terminology_trap or support_report.classification == SupportClassification.UNSUPPORTED:
                    evidence.append(
                        f"Retrieved context had elevated vector similarity (top score: {top_score:.3f} > {high_bound}), "
                        f"but support analysis confirms it is an unanswerable terminology trap lacking core entities."
                    )
                    evidence.extend(support_report.evidence_notes)
                    reasons.append("Abstention failure: Query is unsupported; elevated similarity reflects domain background terminology.")
                    return DiagnosticResult(
                        diagnosis=FailureType.ABSTENTION_SUSPECTED,
                        evidence_score=0.90,
                        evidence=evidence,
                        reasons=reasons,
                        limitations=limitations,
                        telemetry_completeness=completeness.value,
                        mode=mode,
                        is_suspected=True,
                        signals=signals,
                    )
                elif support_report.classification == SupportClassification.AMBIGUOUS:
                    evidence.append(
                        f"Retrieved context had elevated similarity (top score: {top_score:.3f}), "
                        f"but answerability support analysis indicates ambiguous evidence."
                    )
                    evidence.extend(support_report.evidence_notes)
                    reasons.append("Ambiguous context: Related terminology present, but cannot verify whether context answers the question.")
                    return DiagnosticResult(
                        diagnosis=FailureType.UNKNOWN,
                        evidence_score=0.0,
                        evidence=evidence,
                        reasons=reasons,
                        limitations=limitations + support_report.limitations,
                        telemetry_completeness=completeness.value,
                        mode=mode,
                        signals=signals,
                    )

            evidence.append(
                f"Retrieved context had high relevance score to question (top score: {top_score:.3f} > {high_bound})."
            )
            evidence.append("Context was directly supplied in top-k prompt window, but answer failed or contradicts facts.")
            reasons.append("Generation failure: High-relevance context supplied but generator failed to extract correct facts.")
            ev_score = round(min(0.95, 0.72 + (top_score - high_bound) * 0.5), 4)
            return DiagnosticResult(
                diagnosis=FailureType.GENERATION_SUSPECTED,
                evidence_score=ev_score,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations,
                telemetry_completeness=completeness.value,
                mode=mode,
                is_suspected=True,
                signals=signals,
            )

        # Check 4: Low Retrieval Relevance Probe (top_score < low_bound)
        # Distinguish retrieval miss from unanswerable question
        target_app = corpus if corpus is not None else self.app
        if mode == "trace_only":
            use_corpus = False
        elif mode == "corpus_aware":
            if target_app is None:
                evidence.append("Corpus-aware mode requested, but no corpus index was provided.")
                reasons.append("Missing corpus access when corpus evidence is required for diagnosis.")
                limitations.append("Corpus index unavailable.")
                return DiagnosticResult(
                    diagnosis=FailureType.UNKNOWN,
                    evidence_score=0.0,
                    evidence=evidence,
                    reasons=reasons,
                    limitations=limitations,
                    telemetry_completeness=completeness.value,
                    mode="corpus_aware",
                    signals=signals,
                )
            use_corpus = True
        else:
            use_corpus = (target_app is not None and hasattr(target_app, "retrieve"))

        if not use_corpus:
            evidence.append(
                f"Retriever returned low-relevance chunks (top score: {top_score:.3f} < {low_bound})."
            )
            evidence.append(
                "Trace-only mode: Without access to the corpus index, cannot determine whether retriever missed documents or question is unanswerable."
            )
            reasons.append("Insufficient evidence to disambiguate retrieval miss from abstention failure without corpus access.")
            limitations.append("Corpus index unavailable in trace-only mode.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations,
                telemetry_completeness=completeness.value,
                mode="trace_only",
                signals=signals,
            )

        # Corpus-aware audit
        try:
            ret_chunks, ret_scores = target_app.retrieve(trace.question, k=3)
        except Exception:
            ret_chunks, ret_scores = [], []

        support_report = evaluate_corpus_answerability(
            question=trace.question,
            candidates=ret_chunks,
            scores=ret_scores,
        )

        signals["clean_corpus_score"] = support_report.top_score
        signals["clean_corpus_gap"] = support_report.score_gap
        signals["corpus_support_strength"] = support_report.corpus_support_strength
        signals["top_term_coverage"] = support_report.top_term_coverage
        signals["is_terminology_trap"] = support_report.is_terminology_trap

        if support_report.classification == SupportClassification.SUPPORTED:
            evidence.append(
                f"Retriever returned low-relevance chunks at runtime (top score: {top_score:.3f})."
            )
            evidence.extend(support_report.evidence_notes)
            evidence.append("Root cause (suspected): Retriever failed to locate relevant documents present in the corpus.")
            reasons.extend(support_report.reasons)
            ev_score = round(min(0.95, 0.75 + support_report.corpus_support_strength * 0.20), 4)
            ev_model = EvidenceModel(
                retrieval_strength=round(support_report.corpus_support_strength, 4),
                corpus_support_strength=round(support_report.corpus_support_strength, 4),
                answerability_signal=round(support_report.top_term_coverage, 4),
                telemetry_completeness=completeness,
                reasons=reasons,
                limitations=limitations,
            )
            return DiagnosticResult(
                diagnosis=FailureType.RETRIEVAL_SUSPECTED,
                evidence_score=ev_score,
                evidence_model=ev_model,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations,
                telemetry_completeness=completeness.value,
                mode="corpus_aware",
                is_suspected=True,
                signals=signals,
            )
        elif support_report.classification == SupportClassification.UNSUPPORTED:
            evidence.append(
                f"Retriever returned low-relevance chunks at runtime (top score: {top_score:.3f})."
            )
            evidence.extend(support_report.evidence_notes)
            evidence.append("System produced a substantive answer when the corpus contains no supporting evidence.")
            reasons.extend(support_report.reasons)
            ev_score = round(min(0.95, 0.75 + (1.0 - support_report.corpus_support_strength) * 0.20), 4)
            ev_model = EvidenceModel(
                corpus_support_strength=round(support_report.corpus_support_strength, 4),
                answerability_signal=round(support_report.top_term_coverage, 4),
                telemetry_completeness=completeness,
                reasons=reasons,
                limitations=limitations,
            )
            return DiagnosticResult(
                diagnosis=FailureType.ABSTENTION_SUSPECTED,
                evidence_score=ev_score,
                evidence_model=ev_model,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations,
                telemetry_completeness=completeness.value,
                mode="corpus_aware",
                is_suspected=True,
                signals=signals,
            )
        else:
            evidence.extend(support_report.evidence_notes)
            reasons.extend(support_report.reasons)
            limitations.extend(support_report.limitations)
            ev_model = EvidenceModel(
                corpus_support_strength=round(support_report.corpus_support_strength, 4),
                answerability_signal=round(support_report.top_term_coverage, 4),
                telemetry_completeness=completeness,
                ambiguity=1.0,
                reasons=reasons,
                limitations=limitations,
            )
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                evidence_score=0.0,
                evidence_model=ev_model,
                evidence=evidence,
                reasons=reasons,
                limitations=limitations,
                telemetry_completeness=completeness.value,
                mode="corpus_aware",
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
        trace: ExecutionTrace | ObservedExecution | dict[str, Any],
        mode: str | None = None,
        corpus: Any | None = None,
        reference_answer: str | None = None,
    ) -> DiagnosticResult:
        """Unified entrypoint for failure diagnosis.
        
        Supports explicit modes:
        - mode="trace_only": Diagnoses strictly using trace telemetry; never touches corpus.
        - mode="corpus_aware": Validates against corpus index (requires corpus or diagnoser.app).
        - mode="reference_assisted": Validates using user-supplied reference answer.
        - mode="oracle": Uses benchmark gold metadata (research upper bound).
        """
        if isinstance(trace, dict):
            trace = ObservedExecution.from_dict(trace)

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

        effective_mode = mode or ("corpus_aware" if (corpus is not None or self.app is not None) else "trace_only")
        return self.diagnose_observed(obs_trace, mode=effective_mode, corpus=corpus)


def diagnose(
    trace: ExecutionTrace | ObservedExecution | dict[str, Any],
    corpus: Any | None = None,
    mode: str | None = None,
    reference_answer: str | None = None,
) -> DiagnosticResult:
    """Convenience functional API for RAG failure diagnosis."""
    diagnoser = RAGDiagnoser(app=corpus)
    return diagnoser.diagnose(
        trace=trace,
        mode=mode,
        corpus=corpus,
        reference_answer=reference_answer,
    )


def evaluate_observed_on_dataset(
    cases: Sequence[InjectedFaultCase],
    diagnoser: RAGDiagnoser,
    mode: str = "auto",
) -> dict[str, Any]:
    """Run observational diagnoser over benchmark cases without exposing any labels to the engine."""
    records: list[dict[str, Any]] = []
    categories = [
        FailureType.RETRIEVAL_MISS.value,
        FailureType.RANKING_MISS.value,
        FailureType.GENERATION_IGNORED_CONTEXT.value,
        FailureType.SHOULD_ABSTAIN.value,
        FailureType.UNKNOWN.value,
    ]

    confusion_matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in categories + [FailureType.NO_FAILURE.value]}
        for gt in categories
    }

    correct_count = 0
    resolved_total = 0
    resolved_correct = 0
    unknown_diagnoses = 0
    total = len(cases)

    for case in cases:
        obs_trace = ObservedExecution.from_injected_case(case)
        diag_res = diagnoser.diagnose_observed(obs_trace, mode=mode)

        gt_str = case.fault_type
        pred_canonical = diag_res.diagnosis.canonical.value
        pred_str = pred_canonical if pred_canonical in categories else FailureType.UNKNOWN.value

        is_correct = (pred_str == gt_str)

        if is_correct:
            correct_count += 1

        if gt_str != FailureType.UNKNOWN.value:
            resolved_total += 1
            if is_correct:
                resolved_correct += 1

        if diag_res.diagnosis == FailureType.UNKNOWN:
            unknown_diagnoses += 1

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
            "evidence_score": diag_res.evidence_score,
            "confidence": diag_res.confidence,
            "evidence": diag_res.evidence,
            "reasons": diag_res.reasons,
            "limitations": diag_res.limitations,
            "telemetry_completeness": diag_res.telemetry_completeness,
            "signals": diag_res.signals,
            "candidate_count": diag_res.candidate_count,
            "mode": diag_res.mode,
        })

    per_class: dict[str, dict[str, Any]] = {}
    f1_list: list[float] = []
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
            "support": tp + fn,
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    accuracy = correct_count / total if total > 0 else 0.0
    coverage = (total - unknown_diagnoses) / total if total > 0 else 0.0
    accuracy_resolved = resolved_correct / resolved_total if resolved_total > 0 else 0.0

    return {
        "total_cases": total,
        "correct_diagnoses": correct_count,
        "incorrect_diagnoses": total - correct_count,
        "unknown_diagnoses": unknown_diagnoses,
        "resolved_total": resolved_total,
        "resolved_correct": resolved_correct,
        "accuracy_resolved": round(accuracy_resolved, 4),
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "macro_f1": round(macro_f1, 4),
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
