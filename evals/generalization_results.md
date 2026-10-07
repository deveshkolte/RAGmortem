# RAGmortem — Day 6: Adversarial Holdout Generalization Report

## Executive Summary

On Day 5, RAGmortem's observational diagnoser achieved **100.0% accuracy (122/122)** on the original operational-policy benchmark dataset.
However, because the diagnostic heuristics were developed alongside the fault injector and the benchmark contained zero unknown cases, Day 6 subjected the **frozen Day 5 diagnoser** (`src/ragmortem/diagnose.py`) to an out-of-sample adversarial holdout benchmark.

### Core Findings

1. **Generalization on Resolved Failures Remains Very Strong**: On clean, unambiguous failure cases across an entirely new synthetic domain (AeroCloud infrastructure & compliance), the diagnoser achieved **98.0% accuracy (98/100)** in Corpus-Aware mode.
2. **Ambiguity Blindspot (Over-Commitment)**: When faced with 20 realistically ambiguous/unknown telemetry traces, the diagnoser had **0.0% recall on `unknown`** in Corpus-Aware mode. Because its heuristics partition the feature space with hard thresholds, it forcibly diagnosed all 20 unknown cases into one of the four known categories.
3. **Threshold Fragility on Domain Transfer**: Fixed cosine similarity thresholds (`0.48`, `0.50`, `0.55`) calibrated on the original dataset failed on paraphrased queries in the new domain. Specifically, 2 valid answerable queries whose gold chunks had semantic similarity of 0.288 and 0.368 were misclassified as `should_abstain`.
4. **Corpus Access is a Necessary Signal for Observational Retrieval/Abstention Diagnosis**:
   - **Trace + Corpus**: 81.7% accuracy across all 120 cases (98.0% on resolved cases), 100% coverage.
   - **Trace-Only**: 50.8% accuracy across all 120 cases (50.0% on resolved cases), 49.2% coverage. Without corpus access, telemetry alone cannot distinguish whether low scores are caused by retriever failure or unanswerable questions.
5. **Formal Decision**: **PROMISING (70%–89% overall accuracy)**. Do NOT proceed directly to external integrations. Improve ambiguity handling, calibrate confidence scores, and make score thresholds adaptive.

---

## 1. Code & Benchmark Leakage Audit

A comprehensive code audit of `src/ragmortem/diagnose.py` verified the following:

| Telemetry / Artifact | Leakage Status | Evidence / Implementation Detail |
| :--- | :--- | :--- |
| `fault_type` | **No Leakage** | `ObservedExecution` schema explicitly omits `fault_type`. |
| `gold_chunk_id` | **No Leakage** | `ObservedExecution` schema explicitly omits `gold_chunk_id`. |
| `gold_answer` | **No Leakage** | `ObservedExecution` schema explicitly omits `gold_answer`. |
| Injected metadata | **No Leakage** | Stripped by `ObservedExecution.from_injected_case()`. |
| Case IDs / Question IDs | **No Leakage** | Classification logic does not match on question or case IDs. |
| Hardcoded Chunk IDs | **No Leakage** | Diagnoser contains no corpus chunk ID references. |
| Fixed Score Thresholds | **LEAKAGE / ARTIFACT** | Empirical cutoffs `0.48`, `0.50`, `0.55` were tuned on the Day 1–5 distribution. In the holdout domain, embedding scores for valid paraphrases fell below `0.48`. |
| Candidate Count Heuristic | **LEAKAGE / ARTIFACT** | Diagnoser assumes `has_extended_candidates` (`len(candidates) > top_k`) implies a ranking failure. If an application logs candidate pools for all queries, this heuristic misclassifies. |
| Fixed Confidence Literals | **LEAKAGE / ARTIFACT** | Hardcoded literals (`0.75`, `0.80`, `0.85`, `0.90`) are uncalibrated heuristics rather than true probabilities. |

---

## 2. Corpus Coverage Signal Audit (Trace-Only vs. Trace + Corpus)

In `diagnose_observed()`, the diagnoser queries `self.app.retrieve(trace.question, k=1)`:
- If `clean_corpus_score >= 0.48`: Concludes the corpus contains the answer and predicts `retrieval_suspected`.
- If `clean_corpus_score < 0.48`: Concludes the corpus lacks evidence and predicts `abstention_suspected`.

### Is this an Oracle Signal?
- **In an online RAG pipeline**: This signal is **accessible** if the diagnostic tool has query access to the underlying vector index. It acts as an active probe / audit.
- **In an offline APM log analyzer (Trace-Only)**: The raw execution logs do not contain live retriever search results.
- **Impact**: Without this signal (`Trace-Only`), the diagnoser correctly acknowledges its inability to differentiate retrieval misses from unanswerable queries and emits `UNKNOWN` (confidence 0.0).

---

## 3. Benchmark Comparison: Original vs. Holdout

| Metric | Original Controlled Benchmark (Day 5) | Holdout Benchmark (Trace + Corpus) | Holdout Benchmark (Trace-Only) |
| :--- | :---: | :---: | :---: |
| **Total Cases** | 122 | 120 | 120 |
| **Domain** | Internal IT Operations | AeroCloud Cloud Infrastructure | AeroCloud Cloud Infrastructure |
| **Chunks in Index** | 30 | 50 | 50 |
| **Resolved Cases** | 122 | 100 | 100 |
| **Accuracy (All Cases)** | **100.0%** (122/122) | **81.7%** (98/120) | **50.8%** (61/120) |
| **Accuracy (Resolved Cases)** | **100.0%** (122/122) | **98.0%** (98/100) | **50.0%** (50/100) |
| **Coverage** | 100.0% | 100.0% | 49.2% |
| **Unknown Cases in Set** | 0 | 20 | 20 |
| **Unknown Diagnoses Made** | 0 | 0 | 61 |
| **Macro F1** | 1.0000 | 0.7153 | 0.4238 |

---

## 4. Per-Class Performance Breakdown

### A. Holdout Experiment 1: Trace + Corpus (Corpus-Aware)

| Failure Category | Precision | Recall | F1 Score | Support (Ground Truth) |
| :--- | :---: | :---: | :---: | :---: |
| `retrieval_miss` | 0.793 | 0.920 | 0.852 | 25 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 25 |
| `generation_ignored_context` | 0.735 | 1.000 | 0.847 | 25 |
| `should_abstain` | 0.781 | 1.000 | 0.877 | 25 |
| `unknown` | 0.000 | 0.000 | 0.000 | 20 |
| **Overall / Macro** | **0.662** | **0.784** | **0.715** | **120** |

### B. Holdout Experiment 2: Trace-Only (Telemetry Alone)

| Failure Category | Precision | Recall | F1 Score | Support (Ground Truth) |
| :--- | :---: | :---: | :---: | :---: |
| `retrieval_miss` | 0.000 | 0.000 | 0.000 | 25 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 25 |
| `generation_ignored_context` | 0.735 | 1.000 | 0.847 | 25 |
| `should_abstain` | 0.000 | 0.000 | 0.000 | 25 |
| `unknown` | 0.180 | 0.550 | 0.272 | 20 |
| **Overall / Macro** | **0.383** | **0.510** | **0.424** | **120** |

---

## 5. Confusion Matrices

### A. Trace + Corpus Confusion Matrix

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`retrieval_miss`** | **23** | 0 | 0 | 2 | 0 | 25 |
| **`ranking_miss`** | 0 | **25** | 0 | 0 | 0 | 25 |
| **`generation_ignored_context`** | 0 | 0 | **25** | 0 | 0 | 25 |
| **`should_abstain`** | 0 | 0 | 0 | **25** | 0 | 25 |
| **`unknown`** | 6 | 0 | 9 | 5 | **0** | 20 |
| **Total Predicted** | 29 | 25 | 34 | 32 | 0 | 120 |

### B. Trace-Only Confusion Matrix

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`retrieval_miss`** | **0** | 0 | 0 | 0 | 25 | 25 |
| **`ranking_miss`** | 0 | **25** | 0 | 0 | 0 | 25 |
| **`generation_ignored_context`** | 0 | 0 | **25** | 0 | 0 | 25 |
| **`should_abstain`** | 0 | 0 | 0 | **0** | 25 | 25 |
| **`unknown`** | 0 | 0 | 9 | 0 | **11** | 20 |
| **Total Predicted** | 0 | 25 | 34 | 0 | 61 | 120 |

---

## 6. Manual Error Analysis: 10 Failure Cases

Below is an in-depth breakdown of representative misdiagnosed cases from the holdout benchmark:

### Case 1: `holdout_retrieval_15` (Retrieval Miss Misdiagnosed as Abstention)
- **Question**: *"What standard withholding tax rate applies if foreign business customers fail to submit tax exemption certificates?"*
- **Ground Truth**: `retrieval_miss` (Gold chunk: `chunk_aerocloud_tax_withholding`).
- **Retrieved Chunks**: Distractors with low scores (`[0.180, 0.150, 0.120]`).
- **Diagnoser Evidence**: Clean corpus query returned `clean_corpus_score = 0.288` on the gold chunk.
- **Failure Cause**: Because `0.288 < 0.48`, the diagnoser heuristic decided no document exists in the corpus and diagnosed `should_abstain`.
- **Systematic Pattern**: Cosine similarity under `all-MiniLM-L6-v2` can vary widely based on syntax and token density. Hardcoded `0.48` cutoff causes false abstention diagnoses on valid domain queries.

### Case 2: `holdout_retrieval_24` (Retrieval Miss Misdiagnosed as Abstention)
- **Question**: *"What minimum duration must elapse between consecutive horizontal scale-out events?"*
- **Ground Truth**: `retrieval_miss` (Gold chunk: `chunk_aerocloud_autoscaling_cooldown`).
- **Retrieved Chunks**: Distractors with low scores (`[0.180, 0.150, 0.120]`).
- **Diagnoser Evidence**: Clean corpus query returned `clean_corpus_score = 0.368`.
- **Failure Cause**: `0.368 < 0.48` triggered false abstention diagnosis.
- **Systematic Pattern**: Paraphrased phrasing reduces embedding dot product below rigid threshold.

### Case 3: `holdout_unknown_01` (Incomplete Telemetry Misdiagnosed as Abstention)
- **Question**: *"What is the monthly pricing per GPU-hour for NVIDIA H100 tensor core instances?"*
- **Ground Truth**: `unknown` (Telemetry missing retrieval scores, partial context).
- **Retrieved Chunks**: 2 chunks, `scores = None`.
- **Generated Answer**: *"The requested configuration is handled via customer support."*
- **Diagnoser Prediction**: `should_abstain` (Confidence: 0.85).
- **Failure Cause**: When scores are missing, the diagnoser treated `top_score` as "low", ran a corpus audit, found `clean_score = 0.363 < 0.48`, and assumed an abstention failure.

### Case 4: `holdout_unknown_02` (Missing Scores Misdiagnosed as Retrieval Miss)
- **Question**: *"What is the maximum allowed lifetime for personal service account API tokens?"*
- **Ground Truth**: `unknown` (Missing retrieval scores in trace).
- **Retrieved Chunks**: 2 distractor chunks, `scores = None`.
- **Generated Answer**: *"Service token expiration follows internal policy."*
- **Diagnoser Prediction**: `retrieval_miss` (Confidence: 0.85).
- **Failure Cause**: Because clean corpus audit yielded score `0.669 >= 0.48`, the diagnoser asserted retriever failure with 0.85 confidence despite having zero telemetry on what the retriever scored at runtime.

### Case 5: `holdout_unknown_06` (Borderline Score Misdiagnosed as Generation Failure)
- **Question**: *"How many days must an object remain in Coldline storage before transitioning to Glacier Deep Archive?"*
- **Ground Truth**: `unknown` (Ambiguous score: 0.505, borderline between low and high relevance).
- **Retrieved Chunks**: Distractors with scores `[0.505, 0.492]`.
- **Generated Answer**: *"Coldline objects transition according to standard tier schedules."*
- **Diagnoser Prediction**: `generation_ignored_context` (Confidence: 0.80).
- **Failure Cause**: Hard threshold `top_score >= 0.50` triggered generation failure branch, ignoring that the chunks were distractors whose similarity barely grazed 0.505.

### Case 6: `holdout_unknown_07` (Borderline Score Misdiagnosed as Generation Failure)
- **Question**: *"What is the minimum password complexity length required for local database root accounts?"*
- **Ground Truth**: `unknown` (Ambiguous score: 0.502 on irrelevant chunks).
- **Retrieved Chunks**: Irrelevant chunks with synthetic noise `[0.502, 0.485]`.
- **Generated Answer**: *"Database passwords require 16 characters."*
- **Diagnoser Prediction**: `generation_ignored_context` (Confidence: 0.80).
- **Failure Cause**: Score of 0.502 crossed the arbitrary 0.50 threshold, falsely blaming the LLM generator for ignoring context when the context was irrelevant.

### Case 7: `holdout_unknown_08` (Borderline Score Misdiagnosed as Generation Failure)
- **Question**: *"What is the maximum acceptable replication lag target for cross-region bucket replication under normal conditions?"*
- **Ground Truth**: `unknown` (Borderline score: 0.508 on distractor chunks).
- **Retrieved Chunks**: Unrelated tax chunks scored at 0.508.
- **Diagnoser Prediction**: `generation_ignored_context` (Confidence: 0.80).
- **Failure Cause**: The generator was blamed because top score was 0.508, despite the chunks lacking the required answer.

### Case 8: `holdout_unknown_11` (Conflicting Multiple Causes Misdiagnosed as Generation Failure)
- **Question**: *"What is the peak sustained burst limit allowed for Enterprise tier API tokens?"*
- **Ground Truth**: `unknown` (Conflicting signals: moderate score 0.52, distractor context, hallucinated answer).
- **Diagnoser Prediction**: `generation_ignored_context` (Confidence: 0.80).
- **Failure Cause**: Diagnoser cannot represent composite or uncertain causes.

### Case 9: `holdout_unknown_16` (Empty Context Misdiagnosed as Retrieval Miss)
- **Question**: *"After how many failed delivery attempts does an outgoing webhook land in the dead letter queue?"*
- **Ground Truth**: `unknown` (Empty retrieval list: `retrieved_chunk_ids = []`).
- **Diagnoser Prediction**: `retrieval_miss` (Confidence: 0.85).
- **Failure Cause**: With empty context, clean corpus score was 0.781, so it assumed retriever failed without considering pipeline error or misconfiguration.

### Case 10: `holdout_unknown_17` (Empty Context Misdiagnosed as Abstention Failure)
- **Question**: *"What is the maximum supported packet size for jumbo frames over Direct Interconnect?"*
- **Ground Truth**: `unknown` (Empty retrieval list on unanswerable query).
- **Diagnoser Prediction**: `should_abstain` (Confidence: 0.85).
- **Failure Cause**: Telemetry was incomplete, yet diagnoser asserted hallucination with 0.85 confidence.

---

## 7. Confidence Score Audit

The current implementation assigns fixed constants across branches:
- `0.75` for false refusals
- `0.80` for generation failures
- `0.85` for retrieval misses and ranking misses
- `0.90` for appropriate abstention
- `0.00` for unknown

### Deficiencies Identified:
1. **Uncalibrated**: A borderline case with score 0.501 receives the identical confidence (0.80) as an unambiguous case with score 0.85.
2. **Missing Evidence Weights**: Confidence does not reflect the margin of difference between candidate scores or the completeness of telemetry fields.
3. **Recommendation**: Confidence should be computed as a continuous function of telemetry completeness, score margins, and probe agreement, or renamed to `evidence_score`.

---

## 8. Decision: PROMISING (IMPROVE BEFORE INTEGRATION)

According to the Day 6 decision criteria:
- **Strong**: $\ge 90\%$ accuracy AND $\ge 90\%$ coverage $\rightarrow$ Proceed to integration.
- **Promising**: $70\%–89\%$ accuracy $\rightarrow$ Investigate failure modes and improve algorithm before integration.
- **Weak**: $< 70\%$ $\rightarrow$ Do not build integrations yet.

### Verdict:
Holdout accuracy on all 120 cases is **81.7%** (Promising band).
While performance on resolved cases (98.0%) and ranking misses (100.0%) is outstanding, RAGmortem currently **over-commits when telemetry is ambiguous**, completely failing to detect genuine `unknown` cases (0% recall).
Furthermore, in pure `Trace-Only` mode without corpus indexing, accuracy drops to **50.8%**.

**Action Plan Before Integration (Day 7)**:
1. Introduce an explicit **Ambiguity / Telemetry Completeness Gate**: If telemetry is missing scores, has empty contexts, or exhibits borderline score margins ($[0.46, 0.54]$), emit `UNKNOWN` rather than forcing a speculative diagnosis.
2. Replace static thresholds (`0.48`, `0.50`, `0.55`) with adaptive score margins.
3. Calibrate confidence scores continuously based on evidence quality.
