"""Tests for Day 8 answerability, support analysis, and abstention discrimination.

Verifies the 15 required scenarios:
1. terminology-only overlap
2. true answer-supporting chunk
3. unsupported question
4. ambiguous support
5. corpus with many related chunks
6. corpus with zero related chunks
7. retrieval miss with strong corpus evidence
8. retrieval miss with weak corpus evidence
9. abstention with misleading terminology
10. unknown when support cannot be established
11. trace-only cannot infer corpus answerability
12. ranking regression
13. generation regression
14. evidence-score determinism
15. cross-domain behavior
"""

from __future__ import annotations

import pytest
from ragmortem.diagnose import ObservedExecution, RAGDiagnoser
from ragmortem.support import (
    CandidateSupport,
    CorpusSupportReport,
    QuestionProfile,
    SupportClassification,
    evaluate_candidate_support,
    evaluate_corpus_answerability,
    extract_question_profile,
)
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk, RagResult


class MockApp:
    """Mock RAG application for deterministic test control."""

    def __init__(self, candidates: list[Chunk], scores: list[float]) -> None:
        self.candidates = candidates
        self.scores = scores

    def retrieve(self, query: str, k: int = 3) -> tuple[list[Chunk], list[float]]:
        return self.candidates[:k], self.scores[:k]


# ---------------------------------------------------------------------------
# Test 1: Terminology-only overlap
# ---------------------------------------------------------------------------
def test_terminology_only_overlap() -> None:
    q = "What is the margin discount for ISO 14001 carbon neutrality metrics on Term SOFR borrowings?"
    prof = extract_question_profile(q)
    # Chunk discusses SOFR and borrowings, but completely lacks ISO 14001 or carbon neutrality
    chunk = Chunk(
        id="chunk_sofr_interest",
        text="Initial Term B Loans bear interest at Term SOFR plus 3.25% per annum for all borrowings.",
        source="credit_agreement.md",
        metadata={"title": "Applicable Interest Margin"},
    )
    cand_supp = evaluate_candidate_support(prof, chunk, similarity_score=0.65)
    # Multiple core terms missing: 'discount', 'iso', '14001', 'carbon', 'neutrality', 'metrics'
    assert len(cand_supp.missing_terms) >= 4
    assert "14001" in cand_supp.missing_terms
    assert "carbon" in cand_supp.missing_terms


# ---------------------------------------------------------------------------
# Test 2: True answer-supporting chunk
# ---------------------------------------------------------------------------
def test_true_answer_supporting_chunk() -> None:
    q = "Within how many calendar days must fatal or life-threatening SUSAR adverse reactions be reported to the FDA?"
    prof = extract_question_profile(q)
    chunk = Chunk(
        id="chunk_susar",
        text="Fatal or life-threatening SUSAR adverse reactions must be reported to the FDA within 7 calendar days.",
        source="safety_guidelines.md",
        metadata={"title": "IND Safety Reporting Timelines"},
    )
    cand_supp = evaluate_candidate_support(prof, chunk, similarity_score=0.82)
    assert cand_supp.term_coverage >= 0.70
    assert "fatal" in cand_supp.matched_terms
    assert "susar" in cand_supp.matched_terms
    assert cand_supp.has_numeric_evidence is True


# ---------------------------------------------------------------------------
# Test 3: Unsupported question
# ---------------------------------------------------------------------------
def test_unsupported_question() -> None:
    q = "What is the statutory employer 401k matching contribution limit for corporate employees?"
    # Corpus has unrelated legal/credit agreement text
    chunks = [
        Chunk(id="c1", text="The borrower shall maintain insurance on real property assets.", source="doc.md", metadata={}),
        Chunk(id="c2", text="Financial statements must be prepared in accordance with GAAP.", source="doc.md", metadata={}),
    ]
    scores = [0.22, 0.18]
    report = evaluate_corpus_answerability(q, chunks, scores)
    assert report.classification == SupportClassification.UNSUPPORTED
    assert report.corpus_support_strength <= 0.20


# ---------------------------------------------------------------------------
# Test 4: Ambiguous support
# ---------------------------------------------------------------------------
def test_ambiguous_support() -> None:
    q = "What is the procedure for releasing collateral upon customary debt refinancing?"
    # Chunk discusses collateral release upon refinancing, but exact procedure is ambiguous
    chunks = [
        Chunk(
            id="c1",
            text="Collateral liens securing debt obligations shall be released upon refinancing transactions approved by the Administrative Agent.",
            source="credit_agreement.md",
            metadata={"title": "Release of Collateral Liens"},
        ),
        Chunk(
            id="c2",
            text="The Borrower may request release of subsidiary guarantors upon permitted dispositions.",
            source="credit_agreement.md",
            metadata={"title": "Guarantor Releases"},
        ),
    ]
    scores = [0.58, 0.54]
    report = evaluate_corpus_answerability(q, chunks, scores)
    assert report.classification == SupportClassification.AMBIGUOUS
    assert len(report.limitations) > 0


# ---------------------------------------------------------------------------
# Test 5: Corpus with many related chunks (dense background terminology)
# ---------------------------------------------------------------------------
def test_corpus_with_many_related_chunks() -> None:
    q = "Does failure to submit annual greenhouse gas budgets trigger an immediate Event of Default?"
    # Multiple chunks mention Events of Default, but none mention greenhouse gas
    chunks = [
        Chunk(id="c1", text="An Event of Default occurs if the borrower fails to pay principal.", source="doc.md", metadata={"title": "Payment Defaults"}),
        Chunk(id="c2", text="An Event of Default occurs if representations are breached.", source="doc.md", metadata={"title": "Covenant Defaults"}),
        Chunk(id="c3", text="Cross-default triggers an Event of Default when exceeding $50M.", source="doc.md", metadata={"title": "Cross Default"}),
    ]
    scores = [0.55, 0.53, 0.51]
    report = evaluate_corpus_answerability(q, chunks, scores)
    assert report.is_terminology_trap is True
    assert report.classification == SupportClassification.UNSUPPORTED


# ---------------------------------------------------------------------------
# Test 6: Corpus with zero related chunks
# ---------------------------------------------------------------------------
def test_corpus_with_zero_related_chunks() -> None:
    q = "What is the standard dietary caloric intake for rodent toxicology trials?"
    report = evaluate_corpus_answerability(q, [], [])
    assert report.classification == SupportClassification.UNSUPPORTED
    assert report.corpus_support_strength == 0.0


# ---------------------------------------------------------------------------
# Test 7: Retrieval miss with strong corpus evidence
# ---------------------------------------------------------------------------
def test_retrieval_miss_with_strong_corpus_evidence() -> None:
    chunk_gold = Chunk(
        id="chunk_sofr_interest",
        text="The applicable interest rate margin over Term SOFR for Initial Term B Loans is 3.25% per annum.",
        source="credit_agreement.md",
        metadata={"title": "Applicable Interest Margin"},
    )
    app = MockApp(candidates=[chunk_gold], scores=[0.85])
    diagnoser = RAGDiagnoser(app=app)

    # Runtime trace retrieved distractor chunks with low score
    trace = ObservedExecution(
        question="What is the applicable interest rate margin over Term SOFR for Initial Term B borrowings?",
        retrieved_chunk_ids=["chunk_distractor"],
        generated_answer="The interest margin is governed by standard credit terms.",
        scores=[0.18],
    )
    result = diagnoser.diagnose(trace, mode="corpus_aware")
    assert result.diagnosis == FailureType.RETRIEVAL_SUSPECTED
    assert result.signals["corpus_support_strength"] >= 0.80


# ---------------------------------------------------------------------------
# Test 8: Retrieval miss with weak corpus evidence (ambiguous fallback)
# ---------------------------------------------------------------------------
def test_retrieval_miss_with_weak_corpus_evidence() -> None:
    chunk_weak = Chunk(
        id="chunk_vague",
        text="Interest rates and applicable margins vary by facility tranche and market conditions.",
        source="credit_agreement.md",
        metadata={"title": "Interest Provisions"},
    )
    app = MockApp(candidates=[chunk_weak], scores=[0.45])
    diagnoser = RAGDiagnoser(app=app)

    trace = ObservedExecution(
        question="What is the applicable interest rate margin over Term SOFR for Initial Term B borrowings?",
        retrieved_chunk_ids=["chunk_distractor"],
        generated_answer="The interest margin is governed by standard credit terms.",
        scores=[0.18],
    )
    result = diagnoser.diagnose(trace, mode="corpus_aware")
    assert result.diagnosis == FailureType.UNKNOWN
    assert "ambiguous" in result.reasons[0].lower() or "related" in result.evidence[1].lower()


# ---------------------------------------------------------------------------
# Test 9: Abstention with misleading terminology (terminology trap)
# ---------------------------------------------------------------------------
def test_abstention_with_misleading_terminology() -> None:
    chunk_trap = Chunk(
        id="chunk_covenants",
        text="The Borrower shall not exceed the First Lien Net Leverage Ratio of 4.50:1.00 as of the end of any fiscal quarter.",
        source="credit_agreement.md",
        metadata={"title": "First Lien Net Leverage Covenant"},
    )
    app = MockApp(candidates=[chunk_trap], scores=[0.63])
    diagnoser = RAGDiagnoser(app=app)

    # Question mentions leverage ratio but asks about cryptocurrency dividends
    trace = ObservedExecution(
        question="What is the maximum allowed leverage ratio permitted when issuing cryptocurrency dividend distributions to shareholders?",
        retrieved_chunk_ids=["chunk_covenants"],
        generated_answer="The leverage ratio cannot exceed 4.50:1.00 for cryptocurrency dividend distributions.",
        scores=[0.63],
    )
    result = diagnoser.diagnose(trace, mode="corpus_aware")
    assert result.diagnosis == FailureType.ABSTENTION_SUSPECTED
    assert result.signals["is_terminology_trap"] is True


# ---------------------------------------------------------------------------
# Test 10: Unknown when support cannot be established
# ---------------------------------------------------------------------------
def test_unknown_when_support_cannot_be_established() -> None:
    chunk_partial = Chunk(
        id="chunk_partial",
        text="The Administrative Agent may waive certain notice requirements upon written request from the Borrower subject to satisfactory compliance.",
        source="credit_agreement.md",
        metadata={"title": "Waivers and Amendments"},
    )
    app = MockApp(candidates=[chunk_partial], scores=[0.48])
    diagnoser = RAGDiagnoser(app=app)

    trace = ObservedExecution(
        question="Under what precise circumstances may the Administrative Agent waive notice requirements?",
        retrieved_chunk_ids=["chunk_partial"],
        generated_answer="Notice requirements may be waived upon satisfactory compliance.",
        scores=[0.48],
    )
    result = diagnoser.diagnose(trace, mode="corpus_aware")
    assert result.diagnosis == FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# Test 11: Trace-only cannot infer corpus answerability
# ---------------------------------------------------------------------------
def test_trace_only_cannot_infer_corpus_answerability() -> None:
    chunk_gold = Chunk(
        id="chunk_sofr",
        text="Initial Term B borrowings bear 3.25% over Term SOFR.",
        source="credit_agreement.md",
        metadata={"title": "Interest"},
    )
    app = MockApp(candidates=[chunk_gold], scores=[0.85])

    # Given identical low-scoring trace
    trace = ObservedExecution(
        question="What is the applicable interest rate margin over Term SOFR for Initial Term B borrowings?",
        retrieved_chunk_ids=["chunk_distractor"],
        generated_answer="The margin is 2.50%.",
        scores=[0.20],
    )

    # Corpus-aware has corpus access and proves a retrieval miss
    diag_corpus = RAGDiagnoser(app=app)
    res_corpus = diag_corpus.diagnose(trace, mode="corpus_aware")
    assert res_corpus.diagnosis == FailureType.RETRIEVAL_SUSPECTED

    # Trace-only has NO corpus access and refuses to guess
    diag_trace = RAGDiagnoser(app=None)
    res_trace = diag_trace.diagnose(trace, mode="trace_only")
    assert res_trace.diagnosis == FailureType.UNKNOWN
    assert res_trace.diagnosis != res_corpus.diagnosis
    assert any("corpus index unavailable in trace-only mode" in lim.lower() for lim in res_trace.limitations)


# ---------------------------------------------------------------------------
# Test 12: Ranking regression (must retain >= 95% recall)
# ---------------------------------------------------------------------------
def test_ranking_regression() -> None:
    diagnoser = RAGDiagnoser(app=None)
    # Candidate pool had gold chunk at rank 3, truncated at top-k=2
    trace = ObservedExecution(
        question="What is the mandatory engineering response time for a P1 incident?",
        retrieved_chunk_ids=["chunk_p2", "chunk_p3"],
        candidate_chunk_ids=["chunk_p2", "chunk_p3", "chunk_p1"],
        scores=[0.80, 0.75, 0.70],
        generated_answer="P2 response SLA is 1 hour.",
        top_k=2,
    )
    res = diagnoser.diagnose(trace, mode="trace_only")
    assert res.diagnosis == FailureType.RANKING_SUSPECTED
    assert res.candidate_count == 3


# ---------------------------------------------------------------------------
# Test 13: Generation regression (must retain >= 95% recall)
# ---------------------------------------------------------------------------
def test_generation_regression() -> None:
    chunk_sup = Chunk(
        id="c1",
        text="The statutory deadline for the FDA to review a clinical hold response is 30 calendar days.",
        source="guidance.md",
        metadata={"title": "Clinical Hold Reviews"},
    )
    app = MockApp(candidates=[chunk_sup], scores=[0.85])
    diagnoser = RAGDiagnoser(app=app)

    trace = ObservedExecution(
        question="What is the statutory deadline for the FDA to review a sponsor's response to lift a clinical hold?",
        retrieved_chunk_ids=["c1"],
        generated_answer="The FDA review period is 60 business days.",  # hallucination
        scores=[0.85],
    )
    res = diagnoser.diagnose(trace, mode="corpus_aware")
    assert res.diagnosis == FailureType.GENERATION_SUSPECTED
    assert res.evidence_score >= 0.80


# ---------------------------------------------------------------------------
# Test 14: Evidence score determinism
# ---------------------------------------------------------------------------
def test_evidence_score_determinism() -> None:
    chunk_gold = Chunk(
        id="c1",
        text="Excess cash flow sweep is 50% when First Lien Net Leverage exceeds 3.50:1.00.",
        source="credit_agreement.md",
        metadata={"title": "Cash Flow Sweep"},
    )
    app = MockApp(candidates=[chunk_gold], scores=[0.78])
    diagnoser = RAGDiagnoser(app=app)

    trace = ObservedExecution(
        question="What is the mandatory excess cash flow sweep percentage when leverage exceeds 3.50x?",
        retrieved_chunk_ids=["distractor"],
        generated_answer="The sweep is 25%.",
        scores=[0.20],
    )
    res1 = diagnoser.diagnose(trace, mode="corpus_aware")
    res2 = diagnoser.diagnose(trace, mode="corpus_aware")
    assert res1.evidence_score == res2.evidence_score
    assert res1.signals == res2.signals
    assert isinstance(res1.evidence_score, float)


# ---------------------------------------------------------------------------
# Test 15: Cross-domain behavior (Financial vs Clinical)
# ---------------------------------------------------------------------------
def test_cross_domain_behavior() -> None:
    # 1. Clinical trial terminology trap
    clinical_chunk = Chunk(
        id="bio_c1",
        text="The Institutional Review Board shall review adverse events in ongoing clinical trials per 21 CFR Part 312.",
        source="irb_regulations.md",
        metadata={"title": "IRB Review Obligations"},
    )
    app_clinical = MockApp(candidates=[clinical_chunk], scores=[0.58])
    diag_clinical = RAGDiagnoser(app=app_clinical)
    trace_clinical = ObservedExecution(
        question="What is the standard dietary caloric intake required for rodent toxicology studies prior to IND filing?",
        retrieved_chunk_ids=["bio_c1"],
        generated_answer="Rodent caloric intake is 20 kcal per day per 21 CFR Part 312.",
        scores=[0.58],
    )
    res_clinical = diag_clinical.diagnose(trace_clinical, mode="corpus_aware")
    assert res_clinical.diagnosis == FailureType.ABSTENTION_SUSPECTED

    # 2. Financial agreement terminology trap
    fin_chunk = Chunk(
        id="fin_c1",
        text="The Borrower shall not exceed the Consolidated Interest Coverage Ratio of 2.50:1.00.",
        source="credit_agreement.md",
        metadata={"title": "Interest Coverage Covenant"},
    )
    app_fin = MockApp(candidates=[fin_chunk], scores=[0.60])
    diag_fin = RAGDiagnoser(app=app_fin)
    trace_fin = ObservedExecution(
        question="Does failure to submit the annual operating budget to the Environmental Protection Agency trigger an Event of Default?",
        retrieved_chunk_ids=["fin_c1"],
        generated_answer="Operating budgets must be submitted to avoid default.",
        scores=[0.60],
    )
    res_fin = diag_fin.diagnose(trace_fin, mode="corpus_aware")
    assert res_fin.diagnosis == FailureType.ABSTENTION_SUSPECTED
