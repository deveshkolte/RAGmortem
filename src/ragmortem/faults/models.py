"""Data models and representations for controlled fault injection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ragmortem.taxonomy import FailureType


@dataclass(frozen=True)
class FaultConfig:
    """Explicit configuration parameters for injecting a single fault."""

    fault_type: str
    seed: int = 42
    top_k: int = 3
    candidate_k: int = 6
    noise_level: float = 0.5
    ranking_strategy: str = "demote_k"
    generator_behavior: str = "contradict"
    variant_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultConfig:
        return cls(
            fault_type=str(data["fault_type"]),
            seed=int(data.get("seed", 42)),
            top_k=int(data.get("top_k", 3)),
            candidate_k=int(data.get("candidate_k", 6)),
            noise_level=float(data.get("noise_level", 0.5)),
            ranking_strategy=str(data.get("ranking_strategy", "demote_k")),
            generator_behavior=str(data.get("generator_behavior", "contradict")),
            variant_name=str(data.get("variant_name", "default")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BaselineExecution:
    """Records the normal, unperturbed RAG pipeline execution."""

    retrieved_ids: list[str]
    gold_rank: int | None
    answer: str
    scores: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_ids": list(self.retrieved_ids),
            "gold_rank": self.gold_rank,
            "answer": self.answer,
            "scores": [round(s, 4) for s in self.scores] if self.scores else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineExecution:
        return cls(
            retrieved_ids=list(data.get("retrieved_ids", [])),
            gold_rank=data.get("gold_rank"),
            answer=str(data.get("answer", "")),
            scores=data.get("scores"),
        )


@dataclass
class FaultyExecution:
    """Records the outcome of the RAG pipeline after fault injection."""

    retrieved_ids: list[str]
    gold_rank: int | None
    answer: str
    scores: list[float] | None = None
    candidate_retrieved_ids: list[str] | None = None
    candidate_gold_rank: int | None = None
    prompt_context_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_ids": list(self.retrieved_ids),
            "gold_rank": self.gold_rank,
            "answer": self.answer,
            "scores": [round(s, 4) for s in self.scores] if self.scores else None,
            "candidate_retrieved_ids": list(self.candidate_retrieved_ids) if self.candidate_retrieved_ids else None,
            "candidate_gold_rank": self.candidate_gold_rank,
            "prompt_context_chunk_ids": list(self.prompt_context_chunk_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultyExecution:
        return cls(
            retrieved_ids=list(data.get("retrieved_ids", [])),
            gold_rank=data.get("gold_rank"),
            answer=str(data.get("answer", "")),
            scores=data.get("scores"),
            candidate_retrieved_ids=data.get("candidate_retrieved_ids"),
            candidate_gold_rank=data.get("candidate_gold_rank"),
            prompt_context_chunk_ids=list(data.get("prompt_context_chunk_ids", [])),
        )


@dataclass
class InjectedFaultCase:
    """Fully reproducible, validated synthetic RAG failure record."""

    case_id: str
    question_id: str
    fault_type: str
    question: str
    gold_chunk_id: str | None
    gold_answer: str
    baseline: BaselineExecution
    fault: FaultyExecution
    expected_cause: str
    expected_failure_condition: str
    actual_failure_condition: str
    validated: bool
    validation_error: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question_id": self.question_id,
            "fault_type": self.fault_type,
            "question": self.question,
            "gold_chunk_id": self.gold_chunk_id,
            "gold_answer": self.gold_answer,
            "baseline": self.baseline.to_dict(),
            "fault": self.fault.to_dict(),
            "expected_cause": self.expected_cause,
            "expected_failure_condition": self.expected_failure_condition,
            "actual_failure_condition": self.actual_failure_condition,
            "validated": self.validated,
            "validation_error": self.validation_error,
            "configuration": dict(self.configuration),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InjectedFaultCase:
        return cls(
            case_id=str(data["case_id"]),
            question_id=str(data["question_id"]),
            fault_type=str(data["fault_type"]),
            question=str(data["question"]),
            gold_chunk_id=data.get("gold_chunk_id"),
            gold_answer=str(data.get("gold_answer", "")),
            baseline=BaselineExecution.from_dict(data["baseline"]),
            fault=FaultyExecution.from_dict(data["fault"]),
            expected_cause=str(data.get("expected_cause", "")),
            expected_failure_condition=str(data.get("expected_failure_condition", "")),
            actual_failure_condition=str(data.get("actual_failure_condition", "")),
            validated=bool(data.get("validated", False)),
            validation_error=data.get("validation_error"),
            configuration=dict(data.get("configuration", {})),
            metadata=dict(data.get("metadata", {})),
        )
