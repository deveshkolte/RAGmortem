"""Tests for Day 6 adversarial holdout validation and generalization."""

import json
from pathlib import Path
import pytest

from examples.reference_rag.app import load_corpus_from_directory, ReferenceRagApp
from ragmortem.diagnose import RAGDiagnoser, ObservedExecution
from ragmortem.faults.inject import load_injected_faults
from ragmortem.taxonomy import FailureType

ORIGINAL_CORPUS_DIR = Path("examples/reference_rag/documents")
HOLDOUT_CORPUS_DIR = Path("examples/holdout_rag/documents")
HOLDOUT_QUESTIONS_PATH = Path("evals/holdout_questions.jsonl")
HOLDOUT_FAULTS_PATH = Path("evals/holdout_faults.jsonl")


def test_holdout_corpus_differs_from_original():
    """Verify that the holdout corpus represents a completely new domain with no chunk ID overlap."""
    orig_chunks = load_corpus_from_directory(ORIGINAL_CORPUS_DIR)
    holdout_chunks = load_corpus_from_directory(HOLDOUT_CORPUS_DIR)

    orig_ids = {c.id for c in orig_chunks}
    holdout_ids = {c.id for c in holdout_chunks}

    assert len(orig_ids) >= 30
    assert len(holdout_ids) >= 40, f"Expected >= 40 chunks, got {len(holdout_ids)}"
    assert orig_ids.isdisjoint(holdout_ids), "Holdout chunk IDs must not overlap with original corpus"


def test_holdout_questions_count_and_types():
    """Verify holdout questions meet minimum requirements (>= 25 answerable, >= 25 unanswerable)."""
    assert HOLDOUT_QUESTIONS_PATH.exists()
    with open(HOLDOUT_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    ans_q = [q for q in questions if q["type"] == "answerable"]
    unans_q = [q for q in questions if q["type"] == "unanswerable"]

    assert len(ans_q) >= 25, f"Expected >= 25 answerable questions, got {len(ans_q)}"
    assert len(unans_q) >= 25, f"Expected >= 25 unanswerable questions, got {len(unans_q)}"


def test_holdout_faults_distribution():
    """Verify holdout fault dataset contains >= 120 cases across the 5 categories."""
    assert HOLDOUT_FAULTS_PATH.exists()
    cases = load_injected_faults(HOLDOUT_FAULTS_PATH)

    assert len(cases) >= 120

    counts = {}
    for c in cases:
        counts[c.fault_type] = counts.get(c.fault_type, 0) + 1

    assert counts.get("retrieval_miss", 0) >= 25
    assert counts.get("ranking_miss", 0) >= 25
    assert counts.get("generation_ignored_context", 0) >= 25
    assert counts.get("should_abstain", 0) >= 25
    assert counts.get("unknown", 0) >= 20


def test_no_hidden_labels_enter_observed_diagnosis():
    """Verify ObservedExecution strictly strips all gold labels and injected metadata."""
    cases = load_injected_faults(HOLDOUT_FAULTS_PATH)
    sample_case = cases[0]

    obs = ObservedExecution.from_injected_case(sample_case)
    obs_dict = obs.to_dict()

    assert "gold_chunk_id" not in obs_dict
    assert "gold_answer" not in obs_dict
    assert "fault_type" not in obs_dict
    assert "expected_cause" not in obs_dict

    for k in obs_dict.get("metadata", {}):
        assert not k.startswith("injected_")


def test_trace_only_mode_cannot_access_corpus():
    """Verify that when app is None, RAGDiagnoser cannot access corpus and returns UNKNOWN for weak scores."""
    diag = RAGDiagnoser()
    diag.app = None  # Explicitly disable corpus access

    trace = ObservedExecution(
        question="What is the internal execution timeout?",
        retrieved_chunk_ids=["chunk_a", "chunk_b"],
        generated_answer="The timeout is 60 seconds.",
        scores=[0.18, 0.12],
        top_k=3,
    )

    result = diag.diagnose_observed(trace)
    assert result.diagnosis == FailureType.UNKNOWN
    assert result.confidence == 0.0
    assert any("Without access to the corpus index" in ev for ev in result.evidence)


def test_frozen_diagnoser_reproducibility():
    """Verify that running diagnosis on a fixed holdout case produces deterministic output."""
    cases = load_injected_faults(HOLDOUT_FAULTS_PATH)
    sample = [c for c in cases if c.fault_type == "ranking_miss"][0]

    app = ReferenceRagApp(corpus_dir=HOLDOUT_CORPUS_DIR, mock_mode=True)
    diag = RAGDiagnoser(app=app)

    obs = ObservedExecution.from_injected_case(sample)
    res1 = diag.diagnose_observed(obs)
    res2 = diag.diagnose_observed(obs)

    assert res1.diagnosis == res2.diagnosis
    assert res1.confidence == res2.confidence
    assert res1.evidence == res2.evidence
