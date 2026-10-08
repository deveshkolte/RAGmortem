"""Controlled RAG fault injection engine and dataset generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from examples.reference_rag.app import ReferenceRagApp
from ragmortem.faults.eval import is_abstaining, is_answer_correct, validate_fault_case
from ragmortem.faults.models import (
    BaselineExecution,
    FaultConfig,
    FaultyExecution,
    InjectedFaultCase,
)
from ragmortem.taxonomy import FailureType
from ragmortem.types import Chunk

CONTRADICTORY_ANSWERS: dict[str, str] = {
    "q01": "The mandatory engineering response SLA for P1 incidents is 12 hours from initial alert.",
    "q02": "A P1 incident is declared only when more than 75% of active users are impacted.",
    "q03": "The engineering response SLA for Priority 2 incidents is 4 hours.",
    "q04": "P3 minor bugs require response within 5 business days.",
    "q05": "P4 maintenance defects must be acknowledged within 30 calendar days.",
    "q06": "Only the Chief Executive Officer (CEO) has authority to declare emergency production code freezes.",
    "q07": "Blameless postmortem reports must be published within 30 days of incident closure.",
    "q08": "Status updates during a P1 incident must be sent once every 24 hours.",
    "q09": "Primary on-call handoff occurs on Friday afternoons at 5:00 PM EST.",
    "q10": "Unacknowledged PagerDuty alerts escalate to secondary engineers after 60 minutes.",
    "q11": "Production deployments are permitted at any time, including Friday evenings and weekends.",
    "q12": "Canary releases immediately direct 50% of production traffic to the new version.",
    "q13": "Canary releases must soak for only 2 minutes before proceeding to 100% rollout.",
    "q14": "Automated rollbacks trigger only if error rates exceed 15%.",
    "q15": "CI/CD pipeline jobs are allowed to run for up to 4 hours before timing out.",
    "q16": "Production database backups are retained for 7 days in local unencrypted storage.",
    "q17": "Tier-1 services must meet a Recovery Time Objective (RTO) of 24 hours.",
    "q18": "Tier-1 services have a Recovery Point Objective (RPO) target of 4 hours.",
    "q19": "Disaster recovery failover drills are conducted once every 5 years.",
    "q20": "Inter-service communication uses plaintext HTTP without mutual TLS.",
    "q21": "Public REST API gateways allow unlimited requests without any rate limiting.",
    "q22": "Critical vulnerabilities with CVSS 9.0+ must be patched within 90 days.",
    "q23": "High-severity vulnerabilities have an SLA of 6 months for production deployment.",
    "q24": "Production credentials and API keys are permanent and never rotated.",
    "q25": "SSH access to production hosts requires only single-factor SMS verification.",
    "q26": "Security audit logs are retained for 30 days before being purged.",
    "q27": "Administrative console sessions remain active indefinitely without inactivity timeout.",
    "q28": "Routing and DNS changes do not require Change Advisory Board (CAB) review.",
    "q29": "Customer data at rest is stored unencrypted in plaintext.",
    "q30": "IAM permissions are granted permanently without periodic recertification.",
}

HALLUCINATED_UNANSWERABLE_ANSWERS: dict[str, list[str]] = {
    "q31_unans": [
        "The maximum reimbursement limit for home office ergonomic equipment is $1,500 annually.",
        "Employees may expense up to $500 per calendar year with manager approval.",
        "A one-time stipend of $1,000 is credited upon completion of the 90-day onboarding period.",
        "Home office equipment purchases are reimbursed up to $750 through the Expensify portal.",
    ],
    "q32_unans": [
        "All Lambda functions in the billing pipeline must be written in Python 3.11.",
        "The billing pipeline requires Lambda functions to be authored strictly in Go 1.21.",
        "TypeScript running on Node.js 20 is the mandatory runtime for billing serverless jobs.",
        "Rust compiled to AWS Lambda custom runtime is required for billing microservices.",
    ],
    "q33_unans": [
        "First-year engineers receive 20 days of paid vacation annually.",
        "Engineers accrue 15 paid time off (PTO) days per year during their first year.",
        "The organization offers 25 days of annual leave accruing at 2.08 days per month.",
        "All full-time engineering staff have an unlimited paid time off policy upon hire.",
    ],
    "q34_unans": [
        "The primary enterprise headquarters is located at 100 Market Street, San Francisco, CA.",
        "Corporate headquarters is situated at 500 Technology Way, Austin, TX 78701.",
        "The global head office address is 450 Lexington Avenue, New York, NY 10017.",
        "The primary headquarters building is located at 1200 Innovation Parkway, Seattle, WA.",
    ],
    "q35_unans": [
        "The maximum allowable payload size for customer webhooks is 2 megabytes (2 MB).",
        "Incoming webhook delivery payloads are strictly capped at 256 kilobytes.",
        "Payload sizes for webhook endpoints cannot exceed 5 megabytes per HTTP POST.",
        "The API gateway enforces a hard webhook payload ceiling of 1 megabyte (1048576 bytes).",
    ],
    "q36_unans": [
        "The on-premises testing laboratory utilizes Dell PowerEdge R750 rack servers.",
        "Hardware testing labs are standardized exclusively on HPE ProLiant DL380 Gen10 systems.",
        "The lab infrastructure is deployed on custom Supermicro 2U high-density server clusters.",
        "Testing racks are equipped with Cisco UCS B-Series blade servers.",
    ],
    "q37_unans": [
        "The Chief Financial Officer responsible for contract approvals is Margaret Chen.",
        "Vendor financial authorizations are approved by CFO David R. Thornton.",
        "Chief Financial Officer Elena Rostova reviews and signs all engineering agreements.",
        "CFO Arthur Pendelton oversees vendor procurement and signing thresholds.",
    ],
    "q38_unans": [
        "Guest office visitors must authenticate using WPA3-Enterprise with web portal login.",
        "The corporate visitor Wi-Fi network uses standard WPA2-PSK with daily rotating passcodes.",
        "Guest networks enforce 802.1X certificate-based authentication with isolated VLANs.",
        "Visitor wireless access requires WPA3-Personal encryption and SMS OTP verification.",
    ],
}


class FaultInjector:
    """Deterministic fault injector for reference RAG pipelines."""

    def __init__(self, app: ReferenceRagApp) -> None:
        self.app = app
        self.corpus_chunk_ids = {c.id for c in self.app.chunks}

    def _get_baseline(self, question: str, gold_chunk_id: str | None, top_k: int) -> BaselineExecution:
        """Run standard pipeline and record baseline metrics."""
        retrieved, scores = self.app.retrieve(question, k=top_k)
        retrieved_ids = [c.id for c in retrieved]
        rank = None
        if gold_chunk_id and gold_chunk_id in retrieved_ids:
            rank = retrieved_ids.index(gold_chunk_id) + 1

        if gold_chunk_id is None:
            answer = "I do not have sufficient information in the provided context to answer this question."
        else:
            answer = self.app._generate_mock_answer(question, retrieved)

        return BaselineExecution(
            retrieved_ids=retrieved_ids,
            gold_rank=rank,
            answer=answer,
            scores=scores,
        )


    def inject_retrieval_miss(
        self,
        q_item: dict[str, Any],
        config: FaultConfig,
    ) -> InjectedFaultCase:
        """Inject retrieval miss by deterministic query representation perturbation."""
        question = q_item["question"]
        q_id = q_item["id"]
        gold_chunk_id = q_item["gold_chunk_id"]
        gold_answer = q_item["gold_answer"]

        baseline = self._get_baseline(question, gold_chunk_id, config.top_k)

        # Retrieve base query embedding
        query_embedding = self.app.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Deterministic noise generator
        rng = np.random.RandomState(config.seed)
        noise = rng.randn(len(query_embedding))
        noise = noise / np.linalg.norm(noise)

        # Apply noise until gold chunk drops out of top_k
        noise_level = config.noise_level
        retrieved_chunks: list[Chunk] = []
        retrieved_scores: list[float] = []

        for attempt in range(10):
            perturbed = (1.0 - noise_level) * query_embedding + noise_level * noise
            perturbed = perturbed / np.linalg.norm(perturbed)
            scores = np.dot(self.app.chunk_embeddings, perturbed)
            top_indices = np.argsort(scores)[::-1][: config.top_k]

            retrieved_chunks = [self.app.chunks[i] for i in top_indices]
            retrieved_scores = [float(scores[i]) for i in top_indices]
            retrieved_ids = [c.id for c in retrieved_chunks]

            if gold_chunk_id not in retrieved_ids:
                break
            noise_level = min(1.0, noise_level + 0.15)
            # Re-sample noise deterministically if needed
            noise = rng.randn(len(query_embedding))
            noise = noise / np.linalg.norm(noise)

        fault_ids = [c.id for c in retrieved_chunks]
        fault_gold_rank = (fault_ids.index(gold_chunk_id) + 1) if gold_chunk_id in fault_ids else None
        fault_answer = self.app._generate_mock_answer(question, retrieved_chunks)

        fault_exec = FaultyExecution(
            retrieved_ids=fault_ids,
            gold_rank=fault_gold_rank,
            answer=fault_answer,
            scores=retrieved_scores,
            prompt_context_chunk_ids=fault_ids,
        )

        case = InjectedFaultCase(
            case_id=f"retrieval_miss_{q_id}_{config.variant_name}",
            question_id=q_id,
            fault_type=FailureType.RETRIEVAL_MISS.value,
            question=question,
            gold_chunk_id=gold_chunk_id,
            gold_answer=gold_answer,
            baseline=baseline,
            fault=fault_exec,
            expected_cause=FailureType.RETRIEVAL_MISS.value,
            expected_failure_condition=f"gold_chunk_id '{gold_chunk_id}' absent from top-{config.top_k}",
            actual_failure_condition=(
                f"retrieved_ids={fault_ids}, gold_present={gold_chunk_id in fault_ids}"
            ),
            validated=False,
            configuration=config.to_dict(),
        )

        is_valid, reason = validate_fault_case(case, self.corpus_chunk_ids, top_k=config.top_k)
        case.validated = is_valid
        case.validation_error = None if is_valid else reason
        return case

    def inject_ranking_miss(
        self,
        q_item: dict[str, Any],
        config: FaultConfig,
    ) -> InjectedFaultCase:
        """Inject ranking miss by retrieving candidate set and demoting gold chunk beyond top-k."""
        question = q_item["question"]
        q_id = q_item["id"]
        gold_chunk_id = q_item["gold_chunk_id"]
        gold_answer = q_item["gold_answer"]

        baseline = self._get_baseline(question, gold_chunk_id, config.top_k)

        # Retrieve larger candidate pool
        cand_chunks, cand_scores = self.app.retrieve(question, k=config.candidate_k)
        cand_ids = [c.id for c in cand_chunks]

        # Ensure gold chunk is present in candidate pool
        if gold_chunk_id not in cand_ids:
            # If not in top candidate_k, find it in corpus and include in candidates
            gold_chunk = next(c for c in self.app.chunks if c.id == gold_chunk_id)
            cand_chunks.append(gold_chunk)
            cand_scores.append(0.3)
            cand_ids.append(gold_chunk_id)

        # Demote gold chunk below top_k
        # Rearrange so distractors fill top_k, and gold chunk is placed at target_rank
        target_rank = config.top_k + 1  # 1-indexed, so index config.top_k
        if config.ranking_strategy == "demote_k2":
            target_rank = min(len(cand_chunks), config.top_k + 2)

        # Separate gold chunk and distractor chunks
        distractors = [c for c in cand_chunks if c.id != gold_chunk_id]
        gold_chunk = next(c for c in cand_chunks if c.id == gold_chunk_id)

        # Construct candidate list: distractors first, then gold chunk at target_rank
        target_idx = target_rank - 1
        reranked_cand = list(distractors[:target_idx]) + [gold_chunk] + list(distractors[target_idx:])

        # The final context passed to generation is strictly the top-k chunks
        final_retrieved = reranked_cand[: config.top_k]
        final_ids = [c.id for c in final_retrieved]
        final_gold_rank = (final_ids.index(gold_chunk_id) + 1) if gold_chunk_id in final_ids else None

        reranked_ids = [c.id for c in reranked_cand]
        cand_gold_rank = reranked_ids.index(gold_chunk_id) + 1

        fault_answer = self.app._generate_mock_answer(question, final_retrieved)

        fault_exec = FaultyExecution(
            retrieved_ids=final_ids,
            gold_rank=final_gold_rank,
            answer=fault_answer,
            scores=[round(0.8 - (0.05 * i), 3) for i in range(len(final_ids))],
            candidate_retrieved_ids=reranked_ids,
            candidate_gold_rank=cand_gold_rank,
            prompt_context_chunk_ids=final_ids,
        )

        case = InjectedFaultCase(
            case_id=f"ranking_miss_{q_id}_{config.variant_name}",
            question_id=q_id,
            fault_type=FailureType.RANKING_MISS.value,
            question=question,
            gold_chunk_id=gold_chunk_id,
            gold_answer=gold_answer,
            baseline=baseline,
            fault=fault_exec,
            expected_cause=FailureType.RANKING_MISS.value,
            expected_failure_condition=(
                f"gold_chunk_id '{gold_chunk_id}' in candidates (rank > {config.top_k}) but absent from final context"
            ),
            actual_failure_condition=(
                f"cand_rank={cand_gold_rank}, final_retrieved_ids={final_ids}"
            ),
            validated=False,
            configuration=config.to_dict(),
        )

        is_valid, reason = validate_fault_case(
            case, self.corpus_chunk_ids, top_k=config.top_k, candidate_k=config.candidate_k
        )
        case.validated = is_valid
        case.validation_error = None if is_valid else reason
        return case

    def inject_generation_ignored_context(
        self,
        q_item: dict[str, Any],
        config: FaultConfig,
    ) -> InjectedFaultCase:
        """Inject generation failure: gold chunk in context, but model generates incorrect answer."""
        question = q_item["question"]
        q_id = q_item["id"]
        gold_chunk_id = q_item["gold_chunk_id"]
        gold_answer = q_item["gold_answer"]

        baseline = self._get_baseline(question, gold_chunk_id, config.top_k)

        # Standard retrieval - ensure gold chunk is in top-k
        retrieved, scores = self.app.retrieve(question, k=config.top_k)
        retrieved_ids = [c.id for c in retrieved]

        if gold_chunk_id not in retrieved_ids:
            # Force gold chunk to be in top-k if not retrieved
            gold_chunk = next(c for c in self.app.chunks if c.id == gold_chunk_id)
            retrieved[0] = gold_chunk
            retrieved_ids[0] = gold_chunk_id

        gold_rank = retrieved_ids.index(gold_chunk_id) + 1

        # Generate deterministic incorrect answer
        incorrect_answer = CONTRADICTORY_ANSWERS.get(
            q_id,
            f"The policy explicitly states that {gold_chunk_id} was revoked and requires 99 business days.",
        )
        if config.generator_behavior == "confused_distractor" and len(retrieved) > 1:
            distractor = next(c for c in retrieved if c.id != gold_chunk_id)
            incorrect_answer = f"[Hallucinated from {distractor.id}]: {distractor.text}"

        fault_exec = FaultyExecution(
            retrieved_ids=retrieved_ids,
            gold_rank=gold_rank,
            answer=incorrect_answer,
            scores=scores,
            prompt_context_chunk_ids=retrieved_ids,
        )

        case = InjectedFaultCase(
            case_id=f"gen_ignored_{q_id}_{config.variant_name}",
            question_id=q_id,
            fault_type=FailureType.GENERATION_IGNORED_CONTEXT.value,
            question=question,
            gold_chunk_id=gold_chunk_id,
            gold_answer=gold_answer,
            baseline=baseline,
            fault=fault_exec,
            expected_cause=FailureType.GENERATION_IGNORED_CONTEXT.value,
            expected_failure_condition=(
                f"gold_chunk_id '{gold_chunk_id}' in prompt context, but answer is factually incorrect"
            ),
            actual_failure_condition=(
                f"context_chunks={retrieved_ids}, answer_correct={is_answer_correct(incorrect_answer, gold_answer)}"
            ),
            validated=False,
            configuration=config.to_dict(),
        )

        is_valid, reason = validate_fault_case(case, self.corpus_chunk_ids, top_k=config.top_k)
        case.validated = is_valid
        case.validation_error = None if is_valid else reason
        return case

    def inject_should_abstain(
        self,
        q_item: dict[str, Any],
        config: FaultConfig,
    ) -> InjectedFaultCase:
        """Inject should_abstain failure: unanswerable query receives substantive hallucination."""
        question = q_item["question"]
        q_id = q_item["id"]
        gold_answer = q_item["gold_answer"]

        baseline = self._get_baseline(question, None, config.top_k)
        retrieved, scores = self.app.retrieve(question, k=config.top_k)
        retrieved_ids = [c.id for c in retrieved]

        # Pick one of the distinct hallucinated variants
        variants = HALLUCINATED_UNANSWERABLE_ANSWERS.get(
            q_id,
            ["Standard procedure permits this request after executive vice president sign-off."],
        )
        variant_idx = config.metadata.get("variant_index", 0) % len(variants)
        hallucinated_answer = variants[variant_idx]

        fault_exec = FaultyExecution(
            retrieved_ids=retrieved_ids,
            gold_rank=None,
            answer=hallucinated_answer,
            scores=scores,
            prompt_context_chunk_ids=retrieved_ids,
        )

        case = InjectedFaultCase(
            case_id=f"should_abstain_{q_id}_{config.variant_name}",
            question_id=q_id,
            fault_type=FailureType.SHOULD_ABSTAIN.value,
            question=question,
            gold_chunk_id=None,
            gold_answer=gold_answer,
            baseline=baseline,
            fault=fault_exec,
            expected_cause=FailureType.SHOULD_ABSTAIN.value,
            expected_failure_condition="unanswerable question answered with substantive hallucination",
            actual_failure_condition=f"is_abstaining={is_abstaining(hallucinated_answer)}",
            validated=False,
            configuration=config.to_dict(),
        )

        is_valid, reason = validate_fault_case(case, self.corpus_chunk_ids, top_k=config.top_k)
        case.validated = is_valid
        case.validation_error = None if is_valid else reason
        return case


def generate_fault_dataset(
    questions: Sequence[dict[str, Any]],
    app: ReferenceRagApp,
    seed: int = 42,
    top_k: int = 3,
) -> list[InjectedFaultCase]:
    """Deterministically generate 100+ validated failure cases across the 4 fault types."""
    injector = FaultInjector(app)
    cases: list[InjectedFaultCase] = []

    answerable = [q for q in questions if q.get("type") == "answerable"]
    unanswerable = [q for q in questions if q.get("type") == "unanswerable"]

    # 1. RETRIEVAL MISS: 30 cases (one per answerable question)
    for idx, q in enumerate(answerable, start=1):
        cfg = FaultConfig(
            fault_type=FailureType.RETRIEVAL_MISS.value,
            seed=seed + idx,
            top_k=top_k,
            noise_level=0.75,
            variant_name="noise_pert",
        )
        case = injector.inject_retrieval_miss(q, cfg)
        cases.append(case)

    # 2. RANKING MISS: 30 cases (demote_k to rank 4)
    for idx, q in enumerate(answerable, start=1):
        cfg = FaultConfig(
            fault_type=FailureType.RANKING_MISS.value,
            seed=seed + 100 + idx,
            top_k=top_k,
            candidate_k=6,
            ranking_strategy="demote_k",
            variant_name="demote_k4",
        )
        case = injector.inject_ranking_miss(q, cfg)
        cases.append(case)

    # 3. GENERATION IGNORED CONTEXT: 30 cases
    for idx, q in enumerate(answerable, start=1):
        behavior = "contradict" if (idx % 2 == 1) else "confused_distractor"
        cfg = FaultConfig(
            fault_type=FailureType.GENERATION_IGNORED_CONTEXT.value,
            seed=seed + 200 + idx,
            top_k=top_k,
            generator_behavior=behavior,
            variant_name=f"gen_{behavior}",
        )
        case = injector.inject_generation_ignored_context(q, cfg)
        cases.append(case)

    # 4. SHOULD ABSTAIN: 32 cases (8 unanswerable questions * 4 distinct hallucination variants)
    for q_idx, q in enumerate(unanswerable, start=1):
        for v_idx in range(4):
            cfg = FaultConfig(
                fault_type=FailureType.SHOULD_ABSTAIN.value,
                seed=seed + 300 + (q_idx * 10) + v_idx,
                top_k=top_k,
                generator_behavior="hallucinate",
                variant_name=f"var0{v_idx + 1}",
                metadata={"variant_index": v_idx},
            )
            case = injector.inject_should_abstain(q, cfg)
            cases.append(case)

    return cases


def save_injected_faults(cases: Sequence[InjectedFaultCase], output_path: str | Path) -> None:
    """Save injected fault records to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")


def load_injected_faults(input_path: str | Path) -> list[InjectedFaultCase]:
    """Load injected fault records from a JSONL file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Faults file not found: {path}")
    cases: list[InjectedFaultCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(InjectedFaultCase.from_dict(json.loads(line)))
    return cases
