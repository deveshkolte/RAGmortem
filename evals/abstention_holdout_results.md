# Day 8 Cross-Domain Abstention Holdout Report (FinDebt Domain)

Evaluation on the new **FinDebt (Syndicated Credit Agreement)** cross-domain dataset.

## Split Protocol
- **Development Split**: 20 cases (5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- **Validation Split**: 20 cases (5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- **Final Holdout Split**: 40 cases (10 retrieval, 10 abstention, 10 terminology trap, 10 ambiguous) evaluated on frozen code.

---

## 1. Development Split Results
### Split: `DEVELOPMENT` (corpus_aware)

- **Total Cases**: 20
- **Overall Accuracy (All)**: 15/20 (75.0%)
- **Accuracy on Resolved Cases**: 12/15 (80.0%)
- **Coverage**: 70.0%
- **Macro F1**: 0.7897
- **Average Evidence Score (Correct)**: 0.925
- **Average Evidence Score (Incorrect)**: 0.775
- **Average Evidence Score (Unknown)**: 0.000

**Key Error Confusions:**
- `should_abstain -> retrieval_miss`: 0
- `should_abstain -> unknown`: 3
- `retrieval_miss -> should_abstain`: 0
- `retrieval_miss -> unknown`: 0

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 1.000 | 1.000 | 1.000 | 5 |
| `should_abstain` | 1.000 | 0.700 | 0.824 | 10 |
| `unknown` | 0.500 | 0.600 | 0.545 | 5 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 5 | 0 | 0 | 0 | 0 |
| `should_abstain` | 0 | 0 | 0 | 7 | 3 |
| `unknown` | 0 | 0 | 2 | 0 | 3 |



---

## 2. Validation Split Results
### Split: `VALIDATION` (corpus_aware)

- **Total Cases**: 20
- **Overall Accuracy (All)**: 11/20 (55.0%)
- **Accuracy on Resolved Cases**: 10/15 (66.7%)
- **Coverage**: 70.0%
- **Macro F1**: 0.6162
- **Average Evidence Score (Correct)**: 0.926
- **Average Evidence Score (Incorrect)**: 0.759
- **Average Evidence Score (Unknown)**: 0.000

**Key Error Confusions:**
- `should_abstain -> retrieval_miss`: 0
- `should_abstain -> unknown`: 5
- `retrieval_miss -> should_abstain`: 0
- `retrieval_miss -> unknown`: 0

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 1.000 | 1.000 | 1.000 | 5 |
| `should_abstain` | 1.000 | 0.500 | 0.667 | 10 |
| `unknown` | 0.167 | 0.200 | 0.182 | 5 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 5 | 0 | 0 | 0 | 0 |
| `should_abstain` | 0 | 0 | 0 | 5 | 5 |
| `unknown` | 0 | 0 | 4 | 0 | 1 |



---

## 3. Final Holdout Split Results (Corpus-Aware)
### Split: `FINAL_HOLDOUT` (corpus_aware)

- **Total Cases**: 40
- **Overall Accuracy (All)**: 26/40 (65.0%)
- **Accuracy on Resolved Cases**: 25/30 (83.3%)
- **Coverage**: 92.5%
- **Macro F1**: 0.5766
- **Average Evidence Score (Correct)**: 0.916
- **Average Evidence Score (Incorrect)**: 0.861
- **Average Evidence Score (Unknown)**: 0.000

**Key Error Confusions:**
- `should_abstain -> retrieval_miss`: 0
- `should_abstain -> unknown`: 0
- `retrieval_miss -> should_abstain`: 2
- `retrieval_miss -> unknown`: 2

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 1.000 | 0.600 | 0.750 | 10 |
| `should_abstain` | 0.731 | 0.950 | 0.826 | 20 |
| `unknown` | 0.333 | 0.100 | 0.154 | 10 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 6 | 0 | 0 | 2 | 2 |
| `should_abstain` | 0 | 0 | 1 | 19 | 0 |
| `unknown` | 0 | 0 | 4 | 5 | 1 |



---

## 4. Final Holdout Split Results (Trace-Only)
### Split: `FINAL_HOLDOUT` (trace_only)

- **Total Cases**: 40
- **Overall Accuracy (All)**: 3/40 (7.5%)
- **Accuracy on Resolved Cases**: 0/30 (0.0%)
- **Coverage**: 35.0%
- **Macro F1**: 0.0556
- **Average Evidence Score (Correct)**: 0.000
- **Average Evidence Score (Incorrect)**: 0.770
- **Average Evidence Score (Unknown)**: 0.000

**Key Error Confusions:**
- `should_abstain -> retrieval_miss`: 0
- `should_abstain -> unknown`: 13
- `retrieval_miss -> should_abstain`: 0
- `retrieval_miss -> unknown`: 10

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `retrieval_miss` | 0.000 | 0.000 | 0.000 | 10 |
| `should_abstain` | 0.000 | 0.000 | 0.000 | 20 |
| `unknown` | 0.115 | 0.300 | 0.167 | 10 |

**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):

| Ground Truth \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
|---|---|---|---|---|---|
| `retrieval_miss` | 0 | 0 | 0 | 0 | 10 |
| `should_abstain` | 0 | 0 | 7 | 0 | 13 |
| `unknown` | 0 | 0 | 7 | 0 | 3 |


