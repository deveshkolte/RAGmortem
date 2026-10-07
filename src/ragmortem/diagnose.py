"""Core diagnostic engine for RAG failure root-cause attribution."""

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
class ExecutionTrace:
    """Standardized trace representing an observed RAG pipeline execution."""

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
        """Construct an ExecutionTrace from an InjectedFaultCase without label leakage."""
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
        """Construct an ExecutionTrace from a live or mock RagResult."""
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
    """The structured diagnosis and counterfactual evidence produced by RAGmortem."""

    diagnosis: FailureType
    confidence: float
    evidence: list[str]
    original_rank: int | None = None
    candidate_rank: int | None = None
    candidate_count: int = 0
    oracle_replay_correct: bool | None = None
    oracle_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis.value,
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "original_rank": self.original_rank,
            "candidate_rank": self.candidate_rank,
            "candidate_count": self.candidate_count,
            "oracle_replay_correct": self.oracle_replay_correct,
            "oracle_answer": self.oracle_answer,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticResult:
        return cls(
            diagnosis=FailureType(data["diagnosis"]),
            confidence=float(data.get("confidence", 1.0)),
            evidence=list(data.get("evidence", [])),
            original_rank=data.get("original_rank"),
            candidate_rank=data.get("candidate_rank"),
            candidate_count=int(data.get("candidate_count", 0)),
            oracle_replay_correct=data.get("oracle_replay_correct"),
            oracle_answer=data.get("oracle_answer"),
            metadata=dict(data.get("metadata", {})),
        )


class RAGDiagnoser:
    """Deterministic failure diagnostic engine for RAG applications."""

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

    def _oracle_replay(self, question: str, gold_chunk: Chunk) -> tuple[str, bool, str]:
        """Perform counterfactual oracle replay providing only the gold chunk."""
        oracle_answer = self.app._generate_mock_answer(question, [gold_chunk])
        return oracle_answer, True, ""

    def diagnose(self, trace: ExecutionTrace) -> DiagnosticResult:
        """Diagnose root cause of failure from an ExecutionTrace without label leakage."""
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
                )

        # 2. Check if Answerable Question was already correct
        if trace.gold_answer and is_answer_correct(trace.generated_answer, trace.gold_answer):
            evidence.append(f"Answer correctly matches gold criteria: '{trace.generated_answer}'.")
            return DiagnosticResult(
                diagnosis=FailureType.NO_FAILURE,
                confidence=1.0,
                evidence=evidence,
            )

        # 3. Check for Missing Evidence
        gold_id = trace.gold_chunk_id
        if not gold_id:
            evidence.append("Insufficient evidence: answerable question has no gold reference chunk.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                confidence=0.0,
                evidence=evidence,
            )

        gold_chunk = self.chunks_by_id.get(gold_id)
        if not gold_chunk:
            evidence.append(f"Insufficient evidence: gold chunk '{gold_id}' not found in corpus.")
            return DiagnosticResult(
                diagnosis=FailureType.UNKNOWN,
                confidence=0.0,
                evidence=evidence,
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

        # CASE A: Gold chunk was present in the prompt context window
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
                original_rank=orig_rank,
                candidate_rank=cand_rank or orig_rank,
                candidate_count=cand_count,
                oracle_replay_correct=oracle_correct,
                oracle_answer=oracle_answer,
            )

        # CASE B: Gold chunk was absent from final prompt context
        # Check if oracle replay with gold chunk recovers the answer
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
                original_rank=None,
                candidate_rank=cand_rank,
                candidate_count=cand_count,
                oracle_replay_correct=False,
                oracle_answer=oracle_answer,
            )

        # Oracle replay recovered the answer -> Proves retrieval/ranking was at fault
        evidence.append("Counterfactual oracle replay with gold chunk alone corrected the answer.")

        # Probe Candidate Pool vs Top-K Window
        if cand_rank is not None and cand_rank > top_k:
            # Chunk was retrieved in the candidate pool, but excluded by top-k
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
                original_rank=None,
                candidate_rank=cand_rank,
                candidate_count=cand_count,
                oracle_replay_correct=True,
                oracle_answer=oracle_answer,
            )
        else:
            # Chunk was not in candidate pool at all
            evidence.append(
                f"Gold chunk '{gold_id}' was completely absent from candidate retrieval pool (checked {cand_count} candidates)."
            )
            evidence.append(
                "Root cause: Retriever failed to return required evidence chunk."
            )
            return DiagnosticResult(
                diagnosis=FailureType.RETRIEVAL_MISS,
                confidence=1.0,
                evidence=evidence,
                original_rank=None,
                candidate_rank=None,
                candidate_count=cand_count,
                oracle_replay_correct=True,
                oracle_answer=oracle_answer,
            )


def evaluate_diagnoser_on_dataset(
    cases: Sequence[InjectedFaultCase],
    diagnoser: RAGDiagnoser,
) -> dict[str, Any]:
    """Run diagnoser over benchmark cases and compile detailed classification statistics."""
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
        diag_res = diagnoser.diagnose(trace)

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
        })

    # Per-class statistics
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
