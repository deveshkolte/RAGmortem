"""Realistic reference integration demonstrating RAGmortem in a Python RAG pipeline.

Shows how an ordinary Python RAG application logs observable telemetry into a
RAGmortem Trace and diagnoses failure modes across:
1. Successful answer (NO_FAILURE)
2. Retrieval miss (RETRIEVAL_SUSPECTED vs UNKNOWN in trace-only)
3. Ranking miss (RANKING_SUSPECTED)
4. Generation ignored context (GENERATION_SUSPECTED)
5. Unsupported question (ABSTENTION_SUSPECTED)
6. Insufficient telemetry (UNKNOWN)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Ensure repo root is in python path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from ragmortem import Chunk, diagnose, trace
from examples.reference_rag.app import ReferenceRagApp, load_corpus_from_directory


def create_demo_pipeline() -> tuple[ReferenceRagApp, Path]:
    """Initialize a standard production-like RAG pipeline."""
    docs_dir = repo_root / "examples" / "reference_rag" / "documents"
    app = ReferenceRagApp(corpus_dir=docs_dir, mock_mode=True, top_k=3)
    return app, docs_dir


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"SCENARIO: {title}")
    print("=" * 70)


def run_scenarios() -> None:
    app, docs_dir = create_demo_pipeline()

    # =========================================================================
    # SCENARIO 1: Successful Answer / Appropriate Abstention (No Failure)
    # =========================================================================
    print_section("1. Successful Answer (No Failure)")
    query = "What is the mandatory engineering response time for a P1 incident?"
    chunks, scores = app.retrieve(query, k=3)
    answer = "The mandatory engineering response SLA for P1 incidents is 15 minutes."

    # Developer captures execution trace
    t1 = trace(
        query=query,
        retrieved_chunks=[{"id": c.id, "text": c.text, "score": s} for c, s in zip(chunks, scores)],
        scores=scores,
        answer=answer,
    )

    # 1A. Verified with reference answer
    print("--- [A] Reference-Assisted Verification (Expected: 15 minutes) ---")
    diag1_ref = diagnose(t1, reference_answer="15 minutes", corpus=app)
    print(diag1_ref.to_text())

    # 1B. Appropriate abstention on unanswerable query
    print("\n--- [B] Observational Appropriate Abstention (System correctly refused) ---")
    t1_refusal = trace(
        query="What is the employee vacation rollover policy?",
        retrieved_chunks=[{"id": "doc_unrelated", "text": "Unrelated text", "score": 0.15}],
        scores=[0.15],
        answer="I do not have sufficient information in the provided context to answer this question.",
    )
    diag1_refusal = diagnose(t1_refusal, mode="trace_only")
    print(diag1_refusal.to_text())

    # =========================================================================
    # SCENARIO 2: Retrieval Miss
    # Context retriever fails to locate the relevant document in the corpus.
    # =========================================================================
    print_section("2. Retrieval Miss (Trace-Only vs Corpus-Aware)")
    query = "What is the mandatory engineering response time for a P1 incident?"
    # Simulating a retrieval failure where distractor/unrelated chunks are retrieved
    weak_chunks = [
        {"id": "chunk_dr_drill_frequency", "text": "Disaster recovery drills occur quarterly.", "score": 0.14},
        {"id": "chunk_communication_cadence", "text": "All-hands communication happens bi-weekly.", "score": 0.12},
        {"id": "chunk_vuln_high_sla", "text": "High vulnerabilities must be remediated in 14 days.", "score": 0.11},
    ]
    bad_answer = "Engineers must run quarterly disaster recovery exercises."

    t2 = trace(
        query=query,
        retrieved_chunks=weak_chunks,
        scores=[0.14, 0.12, 0.11],
        answer=bad_answer,
    )

    print("--- [A] Trace-Only Mode (Offline APM Telemetry) ---")
    diag2_trace = diagnose(t2, mode="trace_only")
    print(diag2_trace.to_text())

    print("\n--- [B] Corpus-Aware Mode (With Corpus Index Probing) ---")
    diag2_corpus = diagnose(t2, corpus=app, mode="corpus_aware")
    print(diag2_corpus.to_text())

    # =========================================================================
    # SCENARIO 3: Ranking Miss
    # Relevant evidence was retrieved in the candidate pool, but truncated by top-k.
    # =========================================================================
    print_section("3. Ranking Miss")
    query = "What is the mandatory engineering response time for a P1 incident?"
    candidates = [
        {"id": "chunk_vuln_crit_sla", "text": "Critical vulnerabilities must be remediated in 24 hours.", "score": 0.58},
        {"id": "chunk_vuln_high_sla", "text": "High severity issues remediated in 14 days.", "score": 0.55},
        {"id": "chunk_dr_drill_frequency", "text": "Disaster recovery drills on quarterly schedule.", "score": 0.52},
        {"id": "chunk_incident_severity_p1", "text": "P1 incidents require response within 15 minutes.", "score": 0.51},
    ]
    cand_scores = [0.58, 0.55, 0.52, 0.51]
    # Application truncated to top-3, excluding candidate #4 (P1 SLA)
    retrieved_top3 = candidates[:3]
    top3_scores = cand_scores[:3]
    hallucinated_answer = "Engineers must respond within 24 hours."

    t3 = trace(
        query=query,
        retrieved_chunks=retrieved_top3,
        scores=top3_scores,
        candidate_chunks=candidates,
        candidate_scores=cand_scores,
        top_k=3,
        answer=hallucinated_answer,
    )

    diag3 = diagnose(t3, mode="trace_only")
    print(diag3.to_text())

    # =========================================================================
    # SCENARIO 4: Generation Ignored Context
    # Relevant evidence was present in prompt, but generator hallucinated.
    # =========================================================================
    print_section("4. Generation Ignored Context")
    query = "What is the mandatory engineering response time for a P1 incident?"
    grounded_chunks = [
        {"id": "chunk_incident_severity_p1", "text": "P1 incidents require acknowledgment within 15 minutes.", "score": 0.85},
        {"id": "chunk_dr_drill_frequency", "text": "Quarterly DR exercises.", "score": 0.32},
        {"id": "chunk_vuln_high_sla", "text": "High severity SLA 14 days.", "score": 0.28},
    ]
    # Generator hallucinates contrary to the provided chunk:
    hallucinated_answer = "P1 incidents have a 48-hour response window."

    t4 = trace(
        query=query,
        retrieved_chunks=grounded_chunks,
        scores=[0.85, 0.32, 0.28],
        answer=hallucinated_answer,
    )

    diag4 = diagnose(t4, mode="trace_only")
    print(diag4.to_text())

    # =========================================================================
    # SCENARIO 5: Unsupported Question (Should Abstain)
    # The question cannot be answered from the corpus, but model answered anyway.
    # =========================================================================
    print_section("5. Unsupported Question / Should Abstain")
    unsupported_query = "What is the reimbursement limit for home office ergonomic chairs?"
    weak_unrelated_chunks = [
        {"id": "chunk_vuln_high_sla", "text": "High severity remediation 14 days.", "score": 0.12},
        {"id": "chunk_dr_drill_frequency", "text": "Disaster recovery failover quarterly.", "score": 0.10},
        {"id": "chunk_communication_cadence", "text": "Team status update cadence.", "score": 0.09},
    ]
    invented_answer = "The standard employee reimbursement limit for ergonomic chairs is $500."

    t5 = trace(
        query=unsupported_query,
        retrieved_chunks=weak_unrelated_chunks,
        scores=[0.12, 0.10, 0.09],
        answer=invented_answer,
    )

    diag5 = diagnose(t5, corpus=app, mode="corpus_aware")
    print(diag5.to_text())

    # =========================================================================
    # SCENARIO 6: Insufficient Telemetry (Missing Scores / Metadata)
    # Missing telemetry must produce UNKNOWN rather than fabricated certainty.
    # =========================================================================
    print_section("6. Insufficient Telemetry (Unknown)")
    query = "What is the P1 SLA?"
    # Telemetry logging pipeline dropped similarity scores
    t6 = trace(
        query=query,
        retrieved_chunks=[{"id": "doc1", "text": "Some text"}],
        scores=None,  # Scores missing!
        answer="15 minutes",
    )

    diag6 = diagnose(t6, mode="trace_only")
    print(diag6.to_text())


if __name__ == "__main__":
    run_scenarios()
