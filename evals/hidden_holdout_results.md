# Day 7 Hidden Holdout Evaluation Report (BioTrial Domain)

This report presents the frozen evaluation results of RAGmortem's uncertainty-aware diagnoser on the completely unseen **BioTrial (Clinical Trials & FDA Regulations)** holdout dataset (`evals/hidden_holdout_faults.jsonl`).

## Evaluation Design
- **Corpus**: 5 clinical research documents, 40 distinct chunks.
- **Test cases**: 100 injected failure cases across 5 classes (20 retrieval, 20 ranking, 20 generation, 20 abstention, 20 unknown).
- **Diagnoser**: Frozen uncertainty-aware implementation with deterministic evidence scoring and relative score margins.

---

## 1. Corpus-Aware Results
### Mode: `corpus_aware`

- **Total Cases**: 100
- **Overall Accuracy (All cases including UNKNOWN)**: 100/100 (100.0%)
- **Accuracy on Resolved Cases**: 80/80 (100.0%)
- **Coverage**: 80.0%
- **Macro F1**: 1.0000
- **Average Evidence Score (Correct Resolved)**: 0.896
- **Average Evidence Score (Incorrect Resolved)**: 0.000

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 1.000 | 1.000 | 1.000 | 20 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 20 |
| `generation_ignored_context` | 1.000 | 1.000 | 1.000 | 20 |
| `should_abstain` | 1.000 | 1.000 | 1.000 | 20 |
| `unknown` | 1.000 | 1.000 | 1.000 | 20 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 20 | 0 | 0 | 0 | 0 |
| `ranking_miss` | 0 | 20 | 0 | 0 | 0 |
| `generation_ignored_context` | 0 | 0 | 20 | 0 | 0 |
| `should_abstain` | 0 | 0 | 0 | 20 | 0 |
| `unknown` | 0 | 0 | 0 | 0 | 20 |



---

## 2. Trace-Only Results
### Mode: `trace_only`

- **Total Cases**: 100
- **Overall Accuracy (All cases including UNKNOWN)**: 60/100 (60.0%)
- **Accuracy on Resolved Cases**: 40/80 (50.0%)
- **Coverage**: 40.0%
- **Macro F1**: 0.5000
- **Average Evidence Score (Correct Resolved)**: 0.861
- **Average Evidence Score (Incorrect Resolved)**: 0.000

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 0.000 | 0.000 | 0.000 | 20 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 20 |
| `generation_ignored_context` | 1.000 | 1.000 | 1.000 | 20 |
| `should_abstain` | 0.000 | 0.000 | 0.000 | 20 |
| `unknown` | 0.333 | 1.000 | 0.500 | 20 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 0 | 0 | 0 | 0 | 20 |
| `ranking_miss` | 0 | 20 | 0 | 0 | 0 |
| `generation_ignored_context` | 0 | 0 | 20 | 0 | 0 |
| `should_abstain` | 0 | 0 | 0 | 0 | 20 |
| `unknown` | 0 | 0 | 0 | 0 | 20 |



---

## 3. Decision Assessment
- Target Excellent: >= 90% accuracy, >= 80% unknown recall, >= 80% coverage.
- Target Promising: 70–89% accuracy.
- Target Weak: < 70% accuracy.
