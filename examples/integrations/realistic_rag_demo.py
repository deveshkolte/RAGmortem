"""Realistic End-to-End RAG Demonstration with RAGmortem Diagnostics.

Demonstrates how an ordinary Python RAG application executes:
  documents → chunking → vector retrieval → context → generation → RAGmortem trace → diagnosis

This is an integration smoke test demonstrating the developer experience.
It runs 100% locally and offline without external paid APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure workspace root and src/ are in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from ragmortem import Chunk, FailureType, diagnose, trace


# =============================================================================
# 1. Realistic Knowledge Base (Engineering & Infrastructure Policies)
# =============================================================================

DOCUMENTS = [
    {
        "id": "doc_incident_p1",
        "title": "Incident Severity & Response P1",
        "text": "P1 (Critical) incidents require engineering acknowledgment within 15 minutes and hourly customer status updates.",
    },
    {
        "id": "doc_incident_p2",
        "title": "Incident Severity & Response P2",
        "text": "P2 (Major) incidents require engineering acknowledgment within 60 minutes and customer updates every 4 hours.",
    },
    {
        "id": "doc_db_failover",
        "title": "PostgreSQL Failover Architecture",
        "text": "Database Failover Architecture: Primary PostgreSQL clusters use synchronous replication to read-replicas. Regional failover requires promoting replica-1 via patroni.",
    },
    {
        "id": "doc_secret_rotation",
        "title": "API Key & Secret Rotation Protocol",
        "text": "API Key Rotation Protocol: Production API keys and HMAC secrets must be rotated every 90 days using AWS KMS envelope encryption.",
    },
    {
        "id": "doc_canary_deployment",
        "title": "Canary Deployment Gates",
        "text": "Canary Deployment Gates: Canary releases run at 5% traffic for 30 minutes before automatic 100% promotion if 5xx error rate remains below 0.05%.",
    },
]


# =============================================================================
# 2. Realistic In-Memory RAG Pipeline
# =============================================================================

class RealisticRAGPipeline:
    """A realistic minimal RAG pipeline using local embeddings and prompt assembly."""

    def __init__(self, docs: list[dict[str, str]], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.chunks = [
            Chunk(id=d["id"], text=d["text"], source="policy_docs", metadata={"title": d["title"]})
            for d in docs
        ]
        self.embedder = SentenceTransformer(model_name)
        texts = [c.text for c in self.chunks]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: str, k: int = 3) -> tuple[list[Chunk], list[float]]:
        """Retrieve top-k chunks and cosine similarity scores."""
        query_vec = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.embeddings, query_vec)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in ranked_indices], [float(scores[i]) for i in ranked_indices]

    def build_context(self, retrieved: list[Chunk]) -> str:
        """Format retrieved chunks into a prompt context window."""
        return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(retrieved))

    def generate(self, query: str, context: str, behavior: str = "normal") -> str:
        """Simulate LLM generation adhering to supplied context."""
        if behavior == "hallucinate_contrary":
            return "P2 incidents do not have an SLA and can be resolved at leisure."
        elif behavior == "hallucinate_unsupported":
            return "The company provides an annual dental allowance of $1,500 per employee."
        elif behavior == "refuse":
            return "I do not have sufficient information in the provided context to answer this question."
        elif behavior == "ranking_loss":
            return "API keys do not have a defined rotation schedule."
        elif behavior == "wrong_dr":
            return "Canary releases deploy directly to 100% traffic without staging gates."
        else:
            # Normal grounded response
            if "P1" in query:
                return "P1 critical incidents require engineering acknowledgment within 15 minutes."
            return "Answer generated based on context."


# =============================================================================
# 3. Execution & Scenarios
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f" SCENARIO: {title}")
    print("=" * 76)


def run_demo() -> None:
    print("Initializing Realistic RAG Pipeline with local policies...")
    rag = RealisticRAGPipeline(DOCUMENTS)

    # -------------------------------------------------------------------------
    # Scenario 1: Correct Answer
    # -------------------------------------------------------------------------
    banner("1. Correct Answer (Verified with Reference)")
    q1 = "What is the mandatory engineering response time for a P1 incident?"
    chunks1, scores1 = rag.retrieve(q1, k=2)
    ctx1 = rag.build_context(chunks1)
    ans1 = rag.generate(q1, ctx1, behavior="normal")

    t1 = trace(
        query=q1,
        retrieved_chunks=chunks1,
        scores=scores1,
        context=ctx1,
        answer=ans1,
    )
    result1 = diagnose(t1, reference_answer="15 minutes", corpus=rag)
    print(result1.to_text())

    # -------------------------------------------------------------------------
    # Scenario 2: Retrieval Miss
    # -------------------------------------------------------------------------
    banner("2. Retrieval Miss (Trace-Only vs Corpus-Aware)")
    q2 = "What percentage of traffic is routed to canary releases during staging?"
    # Simulating a retrieval routing miss where weak unrelated chunks were retrieved
    weak_chunks = [
        Chunk(id="doc_incident_p2", text="P2 incidents require 60 minutes response.", source="policy"),
        Chunk(id="doc_secret_rotation", text="API keys rotate every 90 days.", source="policy"),
    ]
    weak_scores = [0.18, 0.14]
    ctx2 = rag.build_context(weak_chunks)
    ans2 = rag.generate(q2, ctx2, behavior="wrong_dr")

    t2 = trace(
        query=q2,
        retrieved_chunks=weak_chunks,
        scores=weak_scores,
        context=ctx2,
        answer=ans2,
    )

    print(">>> [2A] Trace-Only Mode (Offline APM Telemetry - No Corpus Access):")
    res2_trace = diagnose(t2, mode="trace_only")
    print(res2_trace.to_text())

    print("\n>>> [2B] Corpus-Aware Mode (With Corpus Index Probing):")
    res2_corpus = diagnose(t2, corpus=rag, mode="corpus_aware")
    print(res2_corpus.to_text())

    # -------------------------------------------------------------------------
    # Scenario 3: Ranking Miss
    # -------------------------------------------------------------------------
    banner("3. Ranking Miss (Omitted Below Top-K Cutoff)")
    q3 = "How often must production API keys and secrets be rotated?"
    # Candidate pool had key rotation chunk at #3, but top_k was set to 2:
    all_cand = [
        Chunk(id="doc_incident_p1", text="P1 incidents require response in 15 minutes.", source="policy"),
        Chunk(id="doc_incident_p2", text="P2 incidents require response in 60 minutes.", source="policy"),
        Chunk(id="doc_secret_rotation", text="API keys and secrets must be rotated every 90 days.", source="policy"),
    ]
    cand_scores = [0.58, 0.56, 0.54]  # Cutoff gap = 0.56 - 0.54 = 0.02 <= 0.08
    top2_chunks = all_cand[:2]
    top2_scores = cand_scores[:2]
    ctx3 = rag.build_context(top2_chunks)
    ans3 = rag.generate(q3, ctx3, behavior="ranking_loss")

    t3 = trace(
        query=q3,
        retrieved_chunks=top2_chunks,
        scores=top2_scores,
        candidate_chunks=all_cand,
        candidate_scores=cand_scores,
        top_k=2,
        context=ctx3,
        answer=ans3,
    )
    result3 = diagnose(t3, mode="trace_only")
    print(result3.to_text())

    # -------------------------------------------------------------------------
    # Scenario 4: Generation Ignores Context
    # -------------------------------------------------------------------------
    banner("4. Generation Ignores Context (Hallucination contrary to context)")
    q4 = "What is the engineering acknowledgment SLA for P2 incidents?"
    chunks4, scores4 = rag.retrieve(q4, k=2)  # doc_incident_p2 retrieved at #1 with score ~0.82
    ctx4 = rag.build_context(chunks4)
    ans4 = rag.generate(q4, ctx4, behavior="hallucinate_contrary")

    t4 = trace(
        query=q4,
        retrieved_chunks=chunks4,
        scores=scores4,
        context=ctx4,
        answer=ans4,
    )
    result4 = diagnose(t4, mode="trace_only")
    print(result4.to_text())

    # -------------------------------------------------------------------------
    # Scenario 5: Unsupported Question (Should Abstain)
    # -------------------------------------------------------------------------
    banner("5. Unsupported Question (System produced ungrounded answer)")
    q5 = "What is the company annual dental allowance for employees?"
    chunks5, scores5 = rag.retrieve(q5, k=2)  # Low relevance, unrelated docs
    ctx5 = rag.build_context(chunks5)
    ans5 = rag.generate(q5, ctx5, behavior="hallucinate_unsupported")

    t5 = trace(
        query=q5,
        retrieved_chunks=chunks5,
        scores=scores5,
        context=ctx5,
        answer=ans5,
    )
    result5 = diagnose(t5, corpus=rag, mode="corpus_aware")
    print(result5.to_text())

    # -------------------------------------------------------------------------
    # Scenario 6: Missing Telemetry
    # -------------------------------------------------------------------------
    banner("6. Insufficient Telemetry (Missing similarity scores)")
    q6 = "What is the regional database failover architecture?"
    chunks6, _ = rag.retrieve(q6, k=2)
    ctx6 = rag.build_context(chunks6)
    ans6 = "Database failover requires promoting replica-1 via patroni."

    # App logged query and answer, but telemetry pipeline dropped similarity scores
    t6 = trace(
        query=q6,
        retrieved_chunks=chunks6,
        scores=None,  # Explicitly missing!
        context=ctx6,
        answer=ans6,
    )
    result6 = diagnose(t6, mode="trace_only")
    print(result6.to_text())


if __name__ == "__main__":
    run_demo()
