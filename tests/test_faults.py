"""Unit and integration tests for controlled fault injection and validation."""

from pathlib import Path

import pytest

from examples.reference_rag.app import ReferenceRagApp, load_corpus_from_directory
from ragmortem.dataset import load_questions
from ragmortem.faults.eval import is_abstaining, is_answer_correct, validate_fault_case
from ragmortem.faults.inject import (
    FaultInjector,
    generate_fault_dataset,
    load_injected_faults,
    save_injected_faults,
)
from ragmortem.faults.models import (
    BaselineExecution,
    FaultConfig,
    FaultyExecution,
    InjectedFaultCase,
)
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk


@pytest.fixture(scope="module")
def shared_app() -> ReferenceRagApp:
    """Share single instance of ReferenceRagApp to avoid re-loading embeddings."""
    return ReferenceRagApp(mock_mode=True, top_k=3)


@pytest.fixture(scope="module")
def corpus_chunks(shared_app: ReferenceRagApp) -> list[Chunk]:
    return shared_app.chunks


@pytest.fixture(scope="module")
def corpus_chunk_ids(corpus_chunks: list[Chunk]) -> set[str]:
    return {c.id for c in corpus_chunks}


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return load_questions("evals/questions.jsonl")


def test_retrieval_miss_is_real(shared_app: ReferenceRagApp, corpus_chunk_ids: set[str], questions: list[dict]) -> None:
    injector = FaultInjector(shared_app)
    q01 = next(q for q in questions if q["id"] == "q01")
    cfg = FaultConfig(
        fault_type=FailureType.RETRIEVAL_MISS.value,
        seed=42,
        top_k=3,
        noise_level=0.75,
        variant_name="test_pert",
    )

    case = injector.inject_retrieval_miss(q01, cfg)
    assert case.validated is True
    assert case.fault_type == "retrieval_miss"
    # 1. Gold chunk is NOT in retrieved top-k
    assert q01["gold_chunk_id"] not in case.fault.retrieved_ids[:3]
    # 2. Gold chunk STILL exists in corpus (NOT deleted!)
    assert q01["gold_chunk_id"] in corpus_chunk_ids
    # 3. Gold rank is None
    assert case.fault.gold_rank is None
    # 4. Baseline is preserved
    assert case.baseline.gold_rank == 1
    assert case.baseline.retrieved_ids[0] == q01["gold_chunk_id"]


def test_ranking_miss_is_real(shared_app: ReferenceRagApp, corpus_chunk_ids: set[str], questions: list[dict]) -> None:
    injector = FaultInjector(shared_app)
    q01 = next(q for q in questions if q["id"] == "q01")
    cfg = FaultConfig(
        fault_type=FailureType.RANKING_MISS.value,
        seed=101,
        top_k=3,
        candidate_k=6,
        ranking_strategy="demote_k",
        variant_name="test_demote",
    )

    case = injector.inject_ranking_miss(q01, cfg)
    assert case.validated is True
    assert case.fault_type == "ranking_miss"
    # 1. Gold chunk IS in candidates
    assert case.fault.candidate_retrieved_ids is not None
    assert q01["gold_chunk_id"] in case.fault.candidate_retrieved_ids
    # 2. Candidate rank > top_k
    assert case.fault.candidate_gold_rank is not None
    assert case.fault.candidate_gold_rank > 3
    # 3. Gold chunk is NOT in final top-3 context
    assert q01["gold_chunk_id"] not in case.fault.retrieved_ids[:3]
    assert case.fault.gold_rank is None
    # 4. Baseline is preserved
    assert case.baseline.gold_rank == 1


def test_generation_ignored_context_is_real(
    shared_app: ReferenceRagApp, corpus_chunk_ids: set[str], questions: list[dict]
) -> None:
    injector = FaultInjector(shared_app)
    q01 = next(q for q in questions if q["id"] == "q01")
    cfg = FaultConfig(
        fault_type=FailureType.GENERATION_IGNORED_CONTEXT.value,
        seed=201,
        top_k=3,
        generator_behavior="contradict",
        variant_name="test_contra",
    )

    case = injector.inject_generation_ignored_context(q01, cfg)
    assert case.validated is True
    assert case.fault_type == "generation_ignored_context"
    # 1. Gold chunk is in retrieved context
    assert q01["gold_chunk_id"] in case.fault.retrieved_ids[:3]
    assert q01["gold_chunk_id"] in case.fault.prompt_context_chunk_ids
    # 2. Generated answer is incorrect
    assert not is_answer_correct(case.fault.answer, case.gold_answer)
    # 3. Baseline answer was correct/mock
    assert "Mock Answer" in case.baseline.answer


def test_should_abstain_is_real(shared_app: ReferenceRagApp, questions: list[dict]) -> None:
    injector = FaultInjector(shared_app)
    q_unans = next(q for q in questions if q["type"] == "unanswerable")
    cfg = FaultConfig(
        fault_type=FailureType.SHOULD_ABSTAIN.value,
        seed=301,
        top_k=3,
        generator_behavior="hallucinate",
        variant_name="test_halluc",
    )

    case = injector.inject_should_abstain(q_unans, cfg)
    assert case.validated is True
    assert case.fault_type == "should_abstain"
    # 1. Question has no gold chunk
    assert case.gold_chunk_id is None
    # 2. Faulty answer does NOT abstain (hallucinated response)
    assert not is_abstaining(case.fault.answer)
    # 3. Baseline answer DID abstain
    assert is_abstaining(case.baseline.answer)


def test_invalid_fault_labels_are_rejected(corpus_chunk_ids: set[str]) -> None:
    base = BaselineExecution(retrieved_ids=["c1"], gold_rank=1, answer="A")

    # 1. Invalid retrieval miss: gold chunk IS in retrieved list
    case_bad_ret = InjectedFaultCase(
        case_id="bad_ret",
        question_id="q1",
        fault_type="retrieval_miss",
        question="Q?",
        gold_chunk_id="chunk_incident_severity_p1",
        gold_answer="15 mins",
        baseline=base,
        fault=FaultyExecution(retrieved_ids=["chunk_incident_severity_p1"], gold_rank=1, answer="ans"),
        expected_cause="retrieval_miss",
        expected_failure_condition="missing",
        actual_failure_condition="present",
        validated=False,
    )
    is_valid, reason = validate_fault_case(case_bad_ret, corpus_chunk_ids)
    assert not is_valid
    assert "still present in retrieved" in reason

    # 2. Invalid ranking miss: gold chunk rank <= top_k
    case_bad_rank = InjectedFaultCase(
        case_id="bad_rank",
        question_id="q1",
        fault_type="ranking_miss",
        question="Q?",
        gold_chunk_id="chunk_incident_severity_p1",
        gold_answer="15 mins",
        baseline=base,
        fault=FaultyExecution(
            retrieved_ids=["chunk_incident_severity_p1"],
            gold_rank=1,
            answer="ans",
            candidate_retrieved_ids=["chunk_incident_severity_p1", "c2"],
            candidate_gold_rank=1,
        ),
        expected_cause="ranking_miss",
        expected_failure_condition="missing",
        actual_failure_condition="rank 1",
        validated=False,
    )
    is_valid, reason = validate_fault_case(case_bad_rank, corpus_chunk_ids, top_k=3)
    assert not is_valid
    assert "must be > 3" in reason

    # 3. Invalid generation failure: answer is actually correct
    case_bad_gen = InjectedFaultCase(
        case_id="bad_gen",
        question_id="q1",
        fault_type="generation_ignored_context",
        question="Q?",
        gold_chunk_id="chunk_incident_severity_p1",
        gold_answer="15 minutes from alert firing to triage.",
        baseline=base,
        fault=FaultyExecution(
            retrieved_ids=["chunk_incident_severity_p1"],
            gold_rank=1,
            answer="15 minutes from alert firing to triage.",
            prompt_context_chunk_ids=["chunk_incident_severity_p1"],
        ),
        expected_cause="generation_ignored_context",
        expected_failure_condition="wrong",
        actual_failure_condition="correct",
        validated=False,
    )
    is_valid, reason = validate_fault_case(case_bad_gen, corpus_chunk_ids)
    assert not is_valid
    assert "unexpectedly correct" in reason


def test_deterministic_generation(shared_app: ReferenceRagApp, questions: list[dict]) -> None:
    run1 = generate_fault_dataset(questions, shared_app, seed=42)
    run2 = generate_fault_dataset(questions, shared_app, seed=42)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.case_id == c2.case_id
        assert c1.fault_type == c2.fault_type
        assert c1.fault.retrieved_ids == c2.fault.retrieved_ids
        assert c1.fault.answer == c2.fault.answer
        assert c1.validated == c2.validated


def test_case_ids_are_unique(shared_app: ReferenceRagApp, questions: list[dict]) -> None:
    cases = generate_fault_dataset(questions, shared_app, seed=42)
    case_ids = [c.case_id for c in cases]
    assert len(case_ids) == len(set(case_ids))
    assert len(cases) >= 100


def test_generated_dataset_passes_validation(corpus_chunk_ids: set[str]) -> None:
    dataset_path = Path("evals/injected_faults.jsonl")
    assert dataset_path.exists(), "evals/injected_faults.jsonl should be present"
    cases = load_injected_faults(dataset_path)

    assert len(cases) >= 100
    for case in cases:
        is_valid, reason = validate_fault_case(case, corpus_chunk_ids, top_k=3)
        assert is_valid, f"Case {case.case_id} failed validation: {reason}"
