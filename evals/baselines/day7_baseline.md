# RAGmortem — Day 7 Baseline Results (Commit 8a3142b)

This document freezes the baseline metrics of RAGmortem's Day 7 uncertainty-aware diagnoser evaluated on the BioTrial (Clinical Trials & FDA Regulations) hidden holdout prior to Day 8 answerability and abstention improvements.

---

## 1. Summary of Baseline Performance

* **Baseline Commit**: `8a3142b`
* **Evaluation Dataset**: `evals/hidden_holdout_faults.jsonl` (100 cases, BioTrial domain)
* **Corpus-Aware Overall Accuracy (All 100 cases)**: **88.0%** (88/100)
* **Corpus-Aware Accuracy on Resolved Cases (Excl. UNKNOWN)**: **85.0%** (68/80)
* **Coverage**: **76.0%** (76/100 resolved diagnoses)
* **Unknown Recall**: **100.0%** (20/20 true ambiguous/unknown cases identified)
* **Macro F1**: **0.8628**
* **Trace-Only Accuracy (All)**: **60.0%** (60/100)
* **Trace-Only Accuracy on Resolved**: **50.0%** (40/80)
* **Trace-Only Precision on Resolved**: **100.0%** (40/40 resolved diagnoses strictly correct)

---

## 2. Per-Class Metrics (Corpus-Aware)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `retrieval_miss` | 0.714 | 1.000 | 0.833 | 20 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 20 |
| `generation_ignored_context` | 1.000 | 1.000 | 1.000 | 20 |
| `should_abstain` | 1.000 | 0.400 | 0.571 | 20 |
| `unknown` | 0.833 | 1.000 | 0.909 | 20 |

---

## 3. Confusion Matrix (Corpus-Aware)

Rows = Ground Truth, Columns = Predicted:

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`retrieval_miss`** | **20** | 0 | 0 | 0 | 0 | 20 |
| **`ranking_miss`** | 0 | **20** | 0 | 0 | 0 | 20 |
| **`generation_ignored_context`** | 0 | 0 | **20** | 0 | 0 | 20 |
| **`should_abstain`** | **8** | 0 | 0 | **8** | **4** | 20 |
| **`unknown`** | 0 | 0 | 0 | 0 | **20** | 20 |
| **Total Predicted** | 28 | 20 | 20 | 8 | 24 | 100 |

---

## 4. Key Diagnostic Observations & Known Deficiencies

1. **Abstention False Positives (`should_abstain` $\rightarrow$ `retrieval_miss`)**:
   * Exactly 8 of 20 unanswerable questions were misdiagnosed as `retrieval_miss`.
   * Cause: In dense specialized domains (clinical trials / FDA regulations), queries sharing background terminology ("audit trail", "21 CFR Part 11", "IRB") match documents with cosine similarities around 0.43–0.46. The diagnoser inferred that because documents with moderate similarity exist in the index, the retriever had missed them, whereas the query was actually unanswerable.
2. **Abstention Uncertainty (`should_abstain` $\rightarrow$ `unknown`)**:
   * 4 of 20 unanswerable questions landed in the ambiguity band or lacked sufficient score separation, correctly abstaining into `unknown`.
3. **Trace-Only Conservatism**:
   * In trace-only mode, the system correctly refuses to guess between retrieval misses and unanswerable questions, achieving 100% precision on resolved cases (ranking and generation).
