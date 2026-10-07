"""Fault injection and failure validation modules for RAGmortem."""

from ragmortem.faults.models import (
    BaselineExecution,
    FaultConfig,
    FaultyExecution,
    InjectedFaultCase,
)
from ragmortem.faults.eval import (
    is_abstaining,
    is_answer_correct,
    validate_fault_case,
)
from ragmortem.faults.inject import (
    FaultInjector,
    generate_fault_dataset,
    load_injected_faults,
    save_injected_faults,
)

__all__ = [
    "BaselineExecution",
    "FaultConfig",
    "FaultyExecution",
    "InjectedFaultCase",
    "is_abstaining",
    "is_answer_correct",
    "validate_fault_case",
    "FaultInjector",
    "generate_fault_dataset",
    "load_injected_faults",
    "save_injected_faults",
]
