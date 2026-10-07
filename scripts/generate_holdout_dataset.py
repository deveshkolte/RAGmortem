"""Generate holdout dataset for Day 6 adversarial validation."""

import json
from pathlib import Path
import numpy as np

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.faults.models import (
    BaselineExecution,
    FaultConfig,
    FaultyExecution,
    InjectedFaultCase,
)
from ragmortem.taxonomy import FailureType

HOLDOUT_DOCS_DIR = Path("examples/holdout_rag/documents")
QUESTIONS_OUT = Path("evals/holdout_questions.jsonl")
FAULTS_OUT = Path("evals/holdout_faults.jsonl")

# 30 Answerable Questions across the 50 AeroCloud chunks
ANSWERABLE_SPECS = [
    {
        "id": "hq01",
        "question": "What is the peak sustained burst limit allowed for Enterprise tier API tokens?",
        "gold_chunk_id": "chunk_aerocloud_api_rate_limits",
        "gold_answer": "150% of the baseline limit (18,000 req/min) for up to 15 consecutive seconds.",
    },
    {
        "id": "hq02",
        "question": "How long does the API gateway remember results submitted with an Idempotency-Key header?",
        "gold_chunk_id": "chunk_aerocloud_idempotency_keys",
        "gold_answer": "Exactly 86,400 seconds (24 hours).",
    },
    {
        "id": "hq03",
        "question": "What close frame code is issued when a workspace opens too many simultaneous WebSockets?",
        "gold_chunk_id": "chunk_aerocloud_concurrency_quotas",
        "gold_answer": "Close frame code 4008 (Policy Violation).",
    },
    {
        "id": "hq04",
        "question": "After how many failed delivery attempts does an outgoing webhook land in the dead letter queue?",
        "gold_chunk_id": "chunk_aerocloud_webhook_retry_backoff",
        "gold_answer": "After 7 failed automated retry attempts.",
    },
    {
        "id": "hq05",
        "question": "For how many calendar days are GA REST API endpoints guaranteed backward compatibility after deprecation?",
        "gold_chunk_id": "chunk_aerocloud_api_deprecation_sla",
        "gold_answer": "A minimum of 365 calendar days.",
    },
    {
        "id": "hq06",
        "question": "What is the maximum number of items returned in a single paginated list query?",
        "gold_chunk_id": "chunk_aerocloud_pagination_limits",
        "gold_answer": "250 items.",
    },
    {
        "id": "hq07",
        "question": "What is the hard client execution timeout enforced by the API gateway before returning a 504?",
        "gold_chunk_id": "chunk_aerocloud_service_mesh_timeouts",
        "gold_answer": "29,000 milliseconds (29 seconds).",
    },
    {
        "id": "hq08",
        "question": "How long do generated Parquet bulk export files remain accessible before automated cleanup?",
        "gold_chunk_id": "chunk_aerocloud_bulk_export_quotas",
        "gold_answer": "72 hours.",
    },
    {
        "id": "hq09",
        "question": "On which calendar day of the month are monthly cloud usage invoices generated and charged?",
        "gold_chunk_id": "chunk_aerocloud_billing_cycle_settlement",
        "gold_answer": "The first calendar day of each month at 00:00 UTC.",
    },
    {
        "id": "hq10",
        "question": "How many days do promotional onboarding credits stay valid before expiring?",
        "gold_chunk_id": "chunk_aerocloud_free_tier_credits",
        "gold_answer": "60 calendar days from account provisioning.",
    },
    {
        "id": "hq11",
        "question": "What is the surcharge multiplier applied to uncommitted compute consumption exceeding plan thresholds?",
        "gold_chunk_id": "chunk_aerocloud_compute_overage_rates",
        "gold_answer": "1.35x (a 35% surcharge over standard reserved instance rates).",
    },
    {
        "id": "hq12",
        "question": "Within how many calendar days must an invoice dispute be logged in the support portal?",
        "gold_chunk_id": "chunk_aerocloud_dispute_resolution_window",
        "gold_answer": "45 calendar days from invoice generation.",
    },
    {
        "id": "hq13",
        "question": "What early termination fee is assessed when cancelling a multi-year reserved compute commitment?",
        "gold_chunk_id": "chunk_aerocloud_reserved_capacity_cancellation",
        "gold_answer": "A 30% early termination fee on the remaining contract commitment value.",
    },
    {
        "id": "hq14",
        "question": "What markup spread is added to daily spot exchange rates for non-USD invoices?",
        "gold_chunk_id": "chunk_aerocloud_currency_conversion",
        "gold_answer": "1.5% currency conversion spread over the mid-market rate.",
    },
    {
        "id": "hq15",
        "question": "What standard withholding tax rate applies if foreign business customers fail to submit tax exemption certificates?",
        "gold_chunk_id": "chunk_aerocloud_tax_withholding",
        "gold_answer": "30% statutory withholding tax.",
    },
    {
        "id": "hq16",
        "question": "How many days after initial payment failure is customer infrastructure suspended?",
        "gold_chunk_id": "chunk_aerocloud_payment_failure_grace_period",
        "gold_answer": "14 calendar days after initial payment failure.",
    },
    {
        "id": "hq17",
        "question": "What rolling 12-month period is evaluated in the annual SOC 2 Type II audit?",
        "gold_chunk_id": "chunk_aerocloud_soc2_type2_reporting",
        "gold_answer": "November 1 through October 31 of each calendar year.",
    },
    {
        "id": "hq18",
        "question": "Which three cloud regions are certified for storing Protected Health Information under a signed BAA?",
        "gold_chunk_id": "chunk_aerocloud_hipaa_baa_eligibility",
        "gold_answer": "us-east-1, us-west-2, and eu-west-1.",
    },
    {
        "id": "hq19",
        "question": "How many days does the privacy team take to process GDPR Article 17 erasure requests across active databases?",
        "gold_chunk_id": "chunk_aerocloud_gdpr_data_subject_rights",
        "gold_answer": "Within 14 calendar days.",
    },
    {
        "id": "hq20",
        "question": "For how many years must compliance WORM audit trail logs be retained before deletion is allowed?",
        "gold_chunk_id": "chunk_aerocloud_audit_log_immutable_storage",
        "gold_answer": "7 years (2,555 days).",
    },
    {
        "id": "hq21",
        "question": "Within what time frame must critical CVSS 9.0+ security vulnerabilities receive a deployed hotfix?",
        "gold_chunk_id": "chunk_aerocloud_vulnerability_disclosure_sla",
        "gold_answer": "Within 24 hours.",
    },
    {
        "id": "hq22",
        "question": "What is the certified multi-region failover recovery time objective (RTO) for core compute and data ingress?",
        "gold_chunk_id": "chunk_aerocloud_disaster_recovery_rpo_rto",
        "gold_answer": "Under 30 minutes.",
    },
    {
        "id": "hq23",
        "question": "How much advance notification does AeroCloud provide to customers prior to introducing new sub-processors?",
        "gold_chunk_id": "chunk_aerocloud_third_party_vendor_review",
        "gold_answer": "At least 30 calendar days prior to granting credentials.",
    },
    {
        "id": "hq24",
        "question": "What minimum duration must elapse between consecutive horizontal scale-out events?",
        "gold_chunk_id": "chunk_aerocloud_autoscaling_cooldown",
        "gold_answer": "300 seconds (5 minutes).",
    },
    {
        "id": "hq25",
        "question": "How much warning does the control plane give before terminating a spot compute instance?",
        "gold_chunk_id": "chunk_aerocloud_spot_instance_eviction",
        "gold_answer": "120 seconds (2 minutes).",
    },
    {
        "id": "hq26",
        "question": "What is the maximum allowed lifetime for personal service account API tokens?",
        "gold_chunk_id": "chunk_aerocloud_api_token_lifetimes",
        "gold_answer": "90 calendar days.",
    },
    {
        "id": "hq27",
        "question": "How quickly must emergency session revocation propagate across all edge points of presence?",
        "gold_chunk_id": "chunk_aerocloud_session_revocation_sla",
        "gold_answer": "Under 60 seconds.",
    },
    {
        "id": "hq28",
        "question": "How many days must an object remain in Coldline storage before transitioning to Glacier Deep Archive?",
        "gold_chunk_id": "chunk_aerocloud_storage_tier_transitions",
        "gold_answer": "90 calendar days.",
    },
    {
        "id": "hq29",
        "question": "What is the maximum acceptable replication lag target for cross-region bucket replication under normal conditions?",
        "gold_chunk_id": "chunk_aerocloud_cross_region_replication",
        "gold_answer": "15 minutes.",
    },
    {
        "id": "hq30",
        "question": "After how many days are uncompleted multipart object uploads automatically aborted and purged?",
        "gold_chunk_id": "chunk_aerocloud_multipart_upload_cleanup",
        "gold_answer": "7 calendar days.",
    },
]

# 30 Unanswerable Questions
UNANSWERABLE_SPECS = [
    {"id": "hq31_unans", "question": "What is the monthly pricing per GPU-hour for NVIDIA H100 tensor core instances?"},
    {"id": "hq32_unans", "question": "Which DNS nameservers should be configured for custom root domain apex delegation?"},
    {"id": "hq33_unans", "question": "What is the maximum supported packet size for jumbo frames over Direct Interconnect?"},
    {"id": "hq34_unans", "question": "How many free seats are included in the GitHub Enterprise organization plan integration?"},
    {"id": "hq35_unans", "question": "What is the maximum allowable payload size for incoming GraphQL mutation requests?"},
    {"id": "hq36_unans", "question": "What is the automated retention period for temporary build artifacts in the CI/CD pipeline?"},
    {"id": "hq37_unans", "question": "Which cipher suites are mandatory for client-side SSL decryption on the ingress gateway?"},
    {"id": "hq38_unans", "question": "What is the maximum size allowed for single ZIP archive uploads in serverless cloud functions?"},
    {"id": "hq39_unans", "question": "What is the reimbursement budget per remote engineering employee for coworking office memberships?"},
    {"id": "hq40_unans", "question": "How often are internal physical datacenter HVAC cooling units inspected and serviced?"},
    {"id": "hq41_unans", "question": "What is the maximum query depth limit enforced by the public GraphQL schema compiler?"},
    {"id": "hq42_unans", "question": "Which Linux kernel version is deployed across default bare-metal container host nodes?"},
    {"id": "hq43_unans", "question": "What is the maximum number of custom dashboards a standard user can create in Datadog integration?"},
    {"id": "hq44_unans", "question": "What is the minimum transaction fee charged for Stripe credit card refunds in CAD currency?"},
    {"id": "hq45_unans", "question": "What is the maximum execution timeout for background queue workers written in Node.js?"},
    {"id": "hq46_unans", "question": "How many days in advance must employees submit PTO requests during peak summer holiday periods?"},
    {"id": "hq47_unans", "question": "What brand of hardware security modules (HSM) is physically installed in the European data center?"},
    {"id": "hq48_unans", "question": "What is the maximum number of tags that can be attached to an individual AWS S3 bucket?"},
    {"id": "hq49_unans", "question": "What is the default TCP SYN retry count for internal service-to-service RPC routing?"},
    {"id": "hq50_unans", "question": "What is the maximum number of partitions permitted per topic in the Kafka event broker cluster?"},
    {"id": "hq51_unans", "question": "What is the minimum credit score required for corporate customers requesting invoicing terms?"},
    {"id": "hq52_unans", "question": "How many minutes of inactivity are required before locking physical office meeting room displays?"},
    {"id": "hq53_unans", "question": "What is the maximum bandwidth cap allocated to outbound IPv6 peering tunnels?"},
    {"id": "hq54_unans", "question": "Which optical fiber transceiver standard is required for 400G cross-connect links?"},
    {"id": "hq55_unans", "question": "What is the referral bonus paid to current staff when hiring a senior cloud architect?"},
    {"id": "hq56_unans", "question": "What is the maximum size of an uncompressed SQL dump accepted by the automated migration wizard?"},
    {"id": "hq57_unans", "question": "What is the default lease duration for DHCP address assignments in customer VPC subnets?"},
    {"id": "hq58_unans", "question": "How many days are ephemeral branch preview environments maintained before automatic deletion?"},
    {"id": "hq59_unans", "question": "What is the minimum password complexity length required for local database root accounts?"},
    {"id": "hq60_unans", "question": "What is the compensation rate for on-call engineers holding weekend triage rotations?"},
]

ABSTAIN_TEXT = "I do not have sufficient information in the provided context to answer this question."

# Realistic hallucinated responses for unanswerable questions
HALLUCINATED_RESPONSES = [
    "NVIDIA H100 GPU instances are priced at $4.50 per hour on on-demand plans and $2.85 on 1-year reserved terms.",
    "Apex domain routing requires configuring ns1.aerocloud-dns.net and ns2.aerocloud-dns.net with TTL of 300.",
    "Direct Interconnect interfaces support MTU jumbo frames up to 9,000 bytes with hardware offloading enabled.",
    "Enterprise customer organizations receive 25 complimentary GitHub Enterprise seat licenses upon contract signing.",
    "Incoming GraphQL mutations enforce a strict payload boundary of 10 megabytes (10,485,760 bytes).",
    "Continuous integration build artifacts and test traces are retained in ephemeral storage for 14 calendar days.",
    "The API ingress edge requires TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 and rejects legacy CBC ciphers.",
    "Serverless cloud function ZIP archive packages may not exceed 250 MB decompressed or 50 MB compressed.",
    "Remote engineering team members are allocated a monthly coworking desk stipend of $350 through Brex.",
    "Mission-critical datacenter HVAC environmental control units undergo certified maintenance checks every 30 days.",
    "The GraphQL schema gate enforces an AST tree traversal depth ceiling of 8 nested levels per operation.",
    "Default bare-metal worker nodes run an enterprise hardened Linux kernel version 5.15 LTS with eBPF support.",
    "Standard workspace members are permitted to create up to 50 active telemetry dashboards in Datadog.",
    "Stripe credit card dispute and refund processing in CAD incurs a flat transaction processing surcharge of $15.00.",
    "Background asynchronous queue tasks written in Node.js are allocated a maximum compute timeout of 900 seconds.",
    "Paid time off requests during peak summer months (July-August) must be submitted 21 business days in advance.",
    "European regional data centers utilize Thales Luna PCIe Hardware Security Modules certified under FIPS 140-2 Level 3.",
    "Cloud storage buckets support a hard limit of 50 user-defined key-value metadata tags per bucket.",
    "Inter-service RPC connections retry unacknowledged TCP SYN packets 3 times with 1-second timeout intervals.",
    "Kafka message streaming clusters support a maximum configuration of 100 partitions per partition group.",
    "Corporate enterprise billing terms require a minimum Dun & Bradstreet credit rating score of 75.",
    "Conference room digital displays automatically power down and engage security lock after 10 minutes of inactivity.",
    "Direct IPv6 border gateway peering tunnels are metered with a sustained throughput cap of 40 Gbps.",
    "400G physical interconnect transceivers must conform strictly to the QSFP-DD optical form factor specification.",
    "Staff engineers receive a $5,000 referral bonus upon successful 90-day onboarding of a hired candidate.",
    "The automated database migration utility accommodates compressed SQL schema dumps up to 50 gigabytes.",
    "Dynamic Host Configuration Protocol (DHCP) subnet leases renew every 86,400 seconds (24 hours).",
    "Ephemeral branch preview environments are automatically decommissioned 72 hours after PR closure.",
    "Database administrator root credentials must meet a minimum length requirement of 16 alphanumeric characters.",
    "On-call engineers receive a flat standby compensation stipend of $500 per weekend shift plus overtime.",
]


def generate_dataset():
    app = ReferenceRagApp(corpus_dir=HOLDOUT_DOCS_DIR, mock_mode=True)
    all_chunks = app.chunks
    all_chunk_ids = [c.id for c in all_chunks]
    chunk_map = {c.id: c for c in all_chunks}

    # 1. Questions records
    questions_records = []
    for spec in ANSWERABLE_SPECS:
        questions_records.append({
            "id": spec["id"],
            "question": spec["question"],
            "gold_chunk_id": spec["gold_chunk_id"],
            "gold_answer": spec["gold_answer"],
            "type": "answerable",
        })
    for spec in UNANSWERABLE_SPECS:
        questions_records.append({
            "id": spec["id"],
            "question": spec["question"],
            "gold_chunk_id": None,
            "gold_answer": ABSTAIN_TEXT,
            "type": "unanswerable",
        })
    with open(QUESTIONS_OUT, "w", encoding="utf-8") as f:
        for r in questions_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(questions_records)} holdout questions to {QUESTIONS_OUT}")

    # 2. Generate 120 Fault Cases
    cases: list[InjectedFaultCase] = []

    # -------------------------------------------------------------
    # Category 1: retrieval_miss (25 cases)
    # -------------------------------------------------------------
    for i in range(25):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        # Run clean retrieval for baseline
        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        # Distractor chunks (completely omitting gold_chunk_id)
        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        # Pick 3 distant distractors deterministically
        distractor_ids = [distractors[(i * 3 + j) % len(distractors)] for j in range(3)]
        distractor_scores = [round(0.18 - j * 0.03, 4) for j in range(3)]

        wrong_answer = f"According to platform policies, operations are governed by standard defaults outlined in {distractor_ids[0]}."

        fault_exec = FaultyExecution(
            retrieved_ids=distractor_ids,
            gold_rank=None,
            answer=wrong_answer,
            scores=distractor_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=distractor_ids,
        )

        case = InjectedFaultCase(
            case_id=f"holdout_retrieval_{i+1:02d}",
            question_id=q_id,
            fault_type="retrieval_miss",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="retrieval_miss",
            expected_failure_condition=f"gold_chunk_id '{gold_id}' absent from top-3",
            actual_failure_condition=f"retrieved_ids={distractor_ids}, gold_present=False",
            validated=True,
            configuration={"fault_type": "retrieval_miss", "top_k": 3},
            metadata={"domain": "aerocloud"},
        )
        cases.append(case)

    # -------------------------------------------------------------
    # Category 2: ranking_miss (25 cases)
    # With hard ranking ambiguity: close scores around top-k cutoff!
    # -------------------------------------------------------------
    for i in range(25):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        top3_distractors = [distractors[(i * 2 + j) % len(distractors)] for j in range(3)]
        other_distractors = [distractors[(i * 2 + 3 + j) % len(distractors)] for j in range(2)]

        # Candidate pool of 6 chunks: top 3 distractors, rank 4 is gold, then 2 others
        # Hard ranking cutoff scores: 0.835, 0.825, 0.818 -> gold at rank 4: 0.812
        cand_ids = top3_distractors + [gold_id] + other_distractors
        cand_scores = [0.835, 0.825, 0.818, 0.812, 0.795, 0.780]
        top_k_ids = cand_ids[:3]
        top_k_scores = cand_scores[:3]

        wrong_answer = f"Based on the provided context, the system enforces general defaults: {top_k_ids[0]}."

        fault_exec = FaultyExecution(
            retrieved_ids=top_k_ids,
            gold_rank=None,
            answer=wrong_answer,
            scores=top_k_scores,
            candidate_retrieved_ids=cand_ids,
            candidate_gold_rank=4,
            prompt_context_chunk_ids=top_k_ids,
        )

        case = InjectedFaultCase(
            case_id=f"holdout_ranking_{i+1:02d}",
            question_id=q_id,
            fault_type="ranking_miss",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="ranking_miss",
            expected_failure_condition=f"gold_chunk_id '{gold_id}' demoted to rank 4 outside top-3",
            actual_failure_condition=f"candidate_gold_rank=4, top_3={top_k_ids}",
            validated=True,
            configuration={"fault_type": "ranking_miss", "top_k": 3, "candidate_k": 6},
            metadata={"domain": "aerocloud", "score_gap": 0.006},
        )
        cases.append(case)

    # -------------------------------------------------------------
    # Category 3: generation_ignored_context (25 cases)
    # Realistic generation failures: numerical, temporal, entity confusion,
    # distractor attention, omission, false refusal
    # -------------------------------------------------------------
    GEN_FAILURES = [
        # 1-5 Numerical confusion
        ("numerical_confusion", "The peak sustained burst limit is capped at 50% above baseline for up to 120 consecutive seconds."),
        ("numerical_confusion", "The API gateway caches mutation responses using idempotency keys for exactly 3,600 seconds (1 hour)."),
        ("numerical_confusion", "When WebSocket connection limits are reached, the server closes the socket with error code 5001."),
        ("numerical_confusion", "Outgoing webhooks trigger up to 14 automatic retry attempts before dead letter queue routing."),
        ("numerical_confusion", "AeroCloud provides backward compatibility for deprecated GA APIs for a minimum of 90 calendar days."),
        # 6-9 Temporal confusion
        ("temporal_confusion", "List queries return a maximum pagination window of 1,000 items per response page."),
        ("temporal_confusion", "The annual SOC 2 Type II audit examination period covers the calendar year from January 1 to December 31."),
        ("temporal_confusion", "Invoices are finalized and charged on the 15th calendar day of each month at 12:00 PM UTC."),
        ("temporal_confusion", "Promotional free tier credits expire after 180 calendar days from initial workspace creation."),
        # 10-13 Entity confusion
        ("entity_confusion", "Uncommitted compute overages incur a standard 2.0x billing multiplier across all accounts."),
        ("entity_confusion", "Protected Health Information under a BAA is authorized only in ap-northeast-1 and sa-east-1 regions."),
        ("entity_confusion", "GDPR Article 17 erasure requests must be completed across production databases within 90 calendar days."),
        ("entity_confusion", "Compliance audit logs in WORM storage are retained for 1 year before automatic truncation."),
        # 14-17 Distractor attention
        ("distractor_attention", "The platform guarantees an internal gateway execution timeout of 60 seconds for upstream services."),
        ("distractor_attention", "Disaster recovery failover certifies a recovery point objective (RPO) of 24 hours across regions."),
        ("distractor_attention", "Customer notifications regarding new sub-processors are issued 7 days prior to credential provisioning."),
        ("distractor_attention", "Horizontal autoscaling scale-out operations require a cooldown window of 60 seconds."),
        # 18-21 Answer omission / partial evidence
        ("answer_omission", "Spot instance interruptions are communicated to the instance metadata service."),
        ("answer_omission", "Service account API tokens have predefined lifetime policies enforced by IAM administrators."),
        ("answer_omission", "Emergency administrative session revocation occurs rapidly across edge proxy nodes."),
        ("answer_omission", "Object lifecycle auto-tiering transitions cold objects into archival storage."),
        # 22-25 False refusal (abstaining despite gold evidence sitting in top-1)
        ("false_refusal", ABSTAIN_TEXT),
        ("false_refusal", ABSTAIN_TEXT),
        ("false_refusal", ABSTAIN_TEXT),
        ("false_refusal", ABSTAIN_TEXT),
    ]

    for i in range(25):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        dist1, dist2 = distractors[(i * 3) % len(distractors)], distractors[(i * 3 + 1) % len(distractors)]

        ret_ids = [gold_id, dist1, dist2]
        scores = [0.72, 0.54, 0.48]
        failure_type_label, fail_answer = GEN_FAILURES[i]

        fault_exec = FaultyExecution(
            retrieved_ids=ret_ids,
            gold_rank=1,
            answer=fail_answer,
            scores=scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=ret_ids,
        )

        case = InjectedFaultCase(
            case_id=f"holdout_generation_{i+1:02d}",
            question_id=q_id,
            fault_type="generation_ignored_context",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="generation_ignored_context",
            expected_failure_condition=f"gold chunk '{gold_id}' present at rank 1 but answer failed ({failure_type_label})",
            actual_failure_condition=f"gold_rank=1, answer='{fail_answer[:50]}...'",
            validated=True,
            configuration={"fault_type": "generation_ignored_context", "variant": failure_type_label},
            metadata={"domain": "aerocloud", "failure_mode": failure_type_label},
        )
        cases.append(case)

    # -------------------------------------------------------------
    # Category 4: should_abstain (25 cases)
    # Unanswerable questions with low retrieval scores, hallucinated answer
    # -------------------------------------------------------------
    for i in range(25):
        spec = UNANSWERABLE_SPECS[i]
        q_id = spec["id"]
        hallucinated_ans = HALLUCINATED_RESPONSES[i]

        distractors = [all_chunk_ids[(i * 3 + j) % len(all_chunk_ids)] for j in range(3)]
        distractor_scores = [round(0.24 - j * 0.04, 4) for j in range(3)]

        base_exec = BaselineExecution(
            retrieved_ids=distractors,
            gold_rank=None,
            answer=ABSTAIN_TEXT,
            scores=distractor_scores,
        )

        fault_exec = FaultyExecution(
            retrieved_ids=distractors,
            gold_rank=None,
            answer=hallucinated_ans,
            scores=distractor_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=distractors,
        )

        case = InjectedFaultCase(
            case_id=f"holdout_abstain_{i+1:02d}",
            question_id=q_id,
            fault_type="should_abstain",
            question=spec["question"],
            gold_chunk_id=None,
            gold_answer=ABSTAIN_TEXT,
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="should_abstain",
            expected_failure_condition="question unanswerable from corpus; system hallucinated substantive answer",
            actual_failure_condition=f"retrieved_scores={distractor_scores}, answer='{hallucinated_ans[:50]}...'",
            validated=True,
            configuration={"fault_type": "should_abstain"},
            metadata={"domain": "aerocloud"},
        )
        cases.append(case)

    # -------------------------------------------------------------
    # Category 5: unknown (20 cases)
    # Genuine telemetry ambiguity where root cause cannot be determined
    # -------------------------------------------------------------
    UNKNOWN_SCENARIOS = [
        # U01-U05: Missing score telemetry and partial context
        ("missing_scores_unanswerable", UNANSWERABLE_SPECS[25]["question"], None, None, [all_chunk_ids[0], all_chunk_ids[1]], None, "The requested configuration is handled via customer support."),
        ("missing_scores_answerable", ANSWERABLE_SPECS[25]["question"], None, None, [all_chunk_ids[2], all_chunk_ids[3]], None, "Service token expiration follows internal policy."),
        ("missing_scores_abstain", UNANSWERABLE_SPECS[26]["question"], None, None, [all_chunk_ids[4], all_chunk_ids[5]], None, "Database connections expire after standard timeout."),
        ("missing_scores_generic", UNANSWERABLE_SPECS[27]["question"], None, None, [all_chunk_ids[6], all_chunk_ids[7]], None, "Ephemeral preview environments are deleted per organization policy."),
        ("missing_scores_incomplete", ANSWERABLE_SPECS[26]["question"], None, None, [all_chunk_ids[8], all_chunk_ids[9]], None, "Session revocation occurs across endpoints."),
        # U06-U10: Borderline retrieval score ambiguity (0.49 - 0.51) with partial distractor overlap
        ("borderline_score_01", ANSWERABLE_SPECS[27]["question"], None, None, [all_chunk_ids[10], all_chunk_ids[11]], [0.505, 0.492], "Coldline objects transition according to standard tier schedules."),
        ("borderline_score_02", UNANSWERABLE_SPECS[28]["question"], None, None, [all_chunk_ids[12], all_chunk_ids[13]], [0.502, 0.485], "Database passwords require 16 characters."),
        ("borderline_score_03", ANSWERABLE_SPECS[28]["question"], None, None, [all_chunk_ids[14], all_chunk_ids[15]], [0.508, 0.481], "Cross-region replication takes roughly 15 minutes."),
        ("borderline_score_04", UNANSWERABLE_SPECS[29]["question"], None, None, [all_chunk_ids[16], all_chunk_ids[17]], [0.501, 0.479], "On-call weekend triage receives special compensation."),
        ("borderline_score_05", ANSWERABLE_SPECS[29]["question"], None, None, [all_chunk_ids[18], all_chunk_ids[19]], [0.495, 0.470], "Multipart uploads expire after a week."),
        # U11-U15: Multiple plausible conflicting causes (extended candidates with contradicting answer and weak top score)
        ("multiple_plausible_01", ANSWERABLE_SPECS[0]["question"], None, None, [all_chunk_ids[0], all_chunk_ids[1]], [0.52, 0.49], "API burst limit is 200% for 60 seconds."),
        ("multiple_plausible_02", ANSWERABLE_SPECS[1]["question"], None, None, [all_chunk_ids[2], all_chunk_ids[3]], [0.51, 0.48], "Idempotency keys are cached for 48 hours."),
        ("multiple_plausible_03", UNANSWERABLE_SPECS[0]["question"], None, None, [all_chunk_ids[4], all_chunk_ids[5]], [0.53, 0.47], "H100 instances cost $5 per hour."),
        ("multiple_plausible_04", ANSWERABLE_SPECS[2]["question"], None, None, [all_chunk_ids[6], all_chunk_ids[7]], [0.52, 0.46], "WebSocket connections are terminated with code 4000."),
        ("multiple_plausible_05", UNANSWERABLE_SPECS[1]["question"], None, None, [all_chunk_ids[8], all_chunk_ids[9]], [0.51, 0.45], "DNS nameservers are ns1 and ns2."),
        # U16-U20: Incomplete trace / empty context
        ("empty_retrieval_01", ANSWERABLE_SPECS[3]["question"], None, None, [], [], "Webhooks retry up to 5 times."),
        ("empty_retrieval_02", UNANSWERABLE_SPECS[2]["question"], None, None, [], [], "Direct Interconnect supports 9000 MTU."),
        ("empty_retrieval_03", ANSWERABLE_SPECS[4]["question"], None, None, [], [], "API deprecation notice is 180 days."),
        ("empty_retrieval_04", UNANSWERABLE_SPECS[3]["question"], None, None, [], [], "GitHub Enterprise includes 10 free seats."),
        ("empty_retrieval_05", ANSWERABLE_SPECS[5]["question"], None, None, [], [], "Pagination limit is 100 items."),
    ]

    for i, (scen_name, q_text, gold_id, gold_ans, ret_chunks, ret_scores, gen_ans) in enumerate(UNKNOWN_SCENARIOS):
        base_exec = BaselineExecution(
            retrieved_ids=ret_chunks,
            gold_rank=None,
            answer="Baseline unavailable.",
            scores=ret_scores,
        )

        fault_exec = FaultyExecution(
            retrieved_ids=ret_chunks,
            gold_rank=None,
            answer=gen_ans,
            scores=ret_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=ret_chunks,
        )

        case = InjectedFaultCase(
            case_id=f"holdout_unknown_{i+1:02d}",
            question_id=f"hq_unknown_{i+1:02d}",
            fault_type="unknown",
            question=q_text,
            gold_chunk_id=gold_id,
            gold_answer=gold_ans or "Unknown reference answer",
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="unknown",
            expected_failure_condition="telemetry genuinely incomplete or ambiguous; root cause cannot be distinguished",
            actual_failure_condition=f"scenario={scen_name}, scores={ret_scores}",
            validated=True,
            configuration={"fault_type": "unknown", "scenario": scen_name},
            metadata={"domain": "aerocloud", "ambiguity_reason": scen_name},
        )
        cases.append(case)

    # Write all cases
    FAULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FAULTS_OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")

    print(f"Total holdout cases generated: {len(cases)}")
    from collections import Counter
    counts = Counter(c.fault_type for c in cases)
    for ft, count in counts.items():
        print(f"  {ft}: {count}")

if __name__ == "__main__":
    generate_dataset()
