"""Bundled synthetic demonstration cases for RAGmortem.

These cases are entirely synthetic and packaged with ragmortem for demonstration,
reproducibility, and test validation. They do not depend on external examples.
"""

from typing import Any

DEMO_CASES: list[dict[str, Any]] = [
    {
        "id": "demo_01_generation_suspected",
        "question": "What is the mandatory response time for a P1 incident?",
        "answer": "The incident response time for P1 incidents is 24 hours.",
        "retrieved": [
            {
                "chunk_id": "chunk_p1_sla",
                "text": "For Priority 1 (P1) critical incidents, the engineering team must respond within 15 minutes.",
                "score": 0.88,
            },
            {
                "chunk_id": "chunk_p2_sla",
                "text": "For Priority 2 (P2) major incidents, response time is 1 hour.",
                "score": 0.52,
            },
        ],
    },
    {
        "id": "demo_02_appropriate_abstention",
        "question": "What is the pet policy for remote employees visiting headquarters?",
        "answer": "I cannot answer this question based on the provided context.",
        "retrieved": [
            {
                "chunk_id": "chunk_travel_policy",
                "text": "Corporate travel expenses must be submitted within 30 days of trip completion.",
                "score": 0.14,
            },
            {
                "chunk_id": "chunk_remote_stipend",
                "text": "Remote employees receive a $500 home office equipment stipend upon onboarding.",
                "score": 0.11,
            },
        ],
    },
    {
        "id": "demo_03_ranking_suspected",
        "question": "How should database connection timeouts be configured during failover?",
        "answer": "Database timeouts default to 60 seconds across all services.",
        "retrieved": [
            {
                "chunk_id": "chunk_db_overview",
                "text": "Primary Postgres database clusters run in multi-region active-passive topology.",
                "score": 0.74,
            },
            {
                "chunk_id": "chunk_db_pools",
                "text": "Connection pool sizes should be calibrated to max 20 connections per container.",
                "score": 0.73,
            },
        ],
        "candidates": [
            {
                "chunk_id": "chunk_db_overview",
                "text": "Primary Postgres database clusters run in multi-region active-passive topology.",
                "score": 0.74,
            },
            {
                "chunk_id": "chunk_db_pools",
                "text": "Connection pool sizes should be calibrated to max 20 connections per container.",
                "score": 0.73,
            },
            {
                "chunk_id": "chunk_db_failover_timeout",
                "text": "During automated replica failover, client connection timeout must be set to 5 seconds to prevent cascade socket exhaustion.",
                "score": 0.71,
            },
            {
                "chunk_id": "chunk_db_metrics",
                "text": "Query latencies exceeding 200ms trigger Datadog warning alerts.",
                "score": 0.60,
            },
        ],
        "top_k": 2,
    },
    {
        "id": "demo_04_trace_only_unknown",
        "question": "What is the maximum retention window for audit logs in the EU cluster?",
        "answer": "Audit logs are retained for 90 days before archive.",
        "retrieved": [
            {
                "chunk_id": "chunk_logging_general",
                "text": "Log aggregation is handled via fluentbit streaming to cold storage buckets.",
                "score": 0.22,
            },
            {
                "chunk_id": "chunk_compliance_intro",
                "text": "SOC2 compliance policies mandate annual penetration test reports.",
                "score": 0.18,
            },
        ],
    },
    {
        "id": "demo_05_reference_assisted_proven",
        "question": "What is the statutory deadline for submitting quarterly security reviews?",
        "answer": "Quarterly security reviews must be submitted within 45 days of quarter end.",
        "retrieved": [
            {
                "chunk_id": "chunk_security_review",
                "text": "Quarterly security reviews must be completed and submitted within 10 business days of quarter close.",
                "score": 0.86,
            }
        ],
        "reference_answer": "10 business days",
    },
]
