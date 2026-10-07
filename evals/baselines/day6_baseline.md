# RAGmortem — Day 6 Baseline Results (Commit 45d76ea)

This document freezes the performance metrics and confusion matrices of the Day 5/Day 6 diagnoser prior to Day 7 uncertainty-aware architectural refactoring.

## 1. Summary of Baseline Metrics

| Metric | Day 6 Adversarial Holdout (Trace + Corpus) | Day 6 Adversarial Holdout (Trace-Only) | Day 5 In-Sample (122 Cases) |
| :--- | :---: | :---: | :---: |
| **Total Cases** | 120 | 120 | 122 |
| **Resolved Cases (Excl. Unknown)** | 100 | 100 | 122 |
| **Overall Accuracy (All 120 Cases)** | **81.7%** (98/120) | **50.8%** (61/120) | **100.0%** (122/122) |
| **Accuracy on Resolved Cases** | **98.0%** (98/100) | **50.0%** (50/100) | **100.0%** (122/122) |
| **Coverage** | **100.0%** | **49.2%** | **100.0%** |
| **Unknown Predictions Made** | 0 | 61 | 0 |
| **Unknown Recall (20 Unknown Cases)** | **0.0%** (0/20) | **55.0%** (11/20) | N/A (0 unknown cases) |
| **Macro F1** | **0.7153** | **0.4238** | **1.0000** |

---

## 2. Per-Class Breakdown

### A. Trace + Corpus (Corpus-Aware)
| Class | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `retrieval_miss` | 0.793 | 0.920 | 0.852 | 25 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 25 |
| `generation_ignored_context` | 0.735 | 1.000 | 0.847 | 25 |
| `should_abstain` | 0.781 | 1.000 | 0.877 | 25 |
| `unknown` | 0.000 | 0.000 | 0.000 | 20 |

### B. Trace-Only
| Class | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `retrieval_miss` | 0.000 | 0.000 | 0.000 | 25 |
| `ranking_miss` | 1.000 | 1.000 | 1.000 | 25 |
| `generation_ignored_context` | 0.735 | 1.000 | 0.847 | 25 |
| `should_abstain` | 0.000 | 0.000 | 0.000 | 25 |
| `unknown` | 0.180 | 0.550 | 0.272 | 20 |

---

## 3. Confusion Matrices

### A. Trace + Corpus
| Actual \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`retrieval_miss`** | 23 | 0 | 0 | 2 | 0 |
| **`ranking_miss`** | 0 | 25 | 0 | 0 | 0 |
| **`generation_ignored_context`** | 0 | 0 | 25 | 0 | 0 |
| **`should_abstain`** | 0 | 0 | 0 | 25 | 0 |
| **`unknown`** | 6 | 0 | 9 | 5 | 0 |

### B. Trace-Only
| Actual \ Predicted | `retrieval_miss` | `ranking_miss` | `generation_ignored_context` | `should_abstain` | `unknown` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`retrieval_miss`** | 0 | 0 | 0 | 0 | 25 |
| **`ranking_miss`** | 0 | 25 | 0 | 0 | 0 |
| **`generation_ignored_context`** | 0 | 0 | 25 | 0 | 0 |
| **`should_abstain`** | 0 | 0 | 0 | 0 | 25 |
| **`unknown`** | 0 | 0 | 9 | 0 | 11 |

---

## 4. Key Limitations Identified in Day 6 Baseline

1. **Forced Partitioning / Zero Unknown Recall in Corpus-Aware Mode**: The engine partitions the entire feature space using rigid if/else statements. It never emits `UNKNOWN` when the corpus is present, misclassifying all 20 ambiguous cases into retrieval (6), generation (9), or abstention (5).
2. **Fixed Cosine Similarity Thresholds (`0.48`, `0.50`, `0.55`)**: Fails on domain shift. In AeroCloud, valid paraphrased queries had cosine similarity of 0.288 and 0.368 against gold chunks, leading to false `should_abstain` diagnoses.
3. **Missing Telemetry Treated as Low Telemetry**: When scores or candidate lists were missing (`scores=None`), the diagnoser treated them as "low", executing normal logic rather than detecting incomplete telemetry.
4. **Hardcoded Uncalibrated Confidence Literals**: Emitted static numbers (`0.75`, `0.80`, `0.85`, `0.90`) regardless of evidence margins or telemetry completeness.
5. **Silent Fallback to Default Corpus**: When `app=None`, `RAGDiagnoser` silently instantiated a default reference app pointing to the Day 1 corpus rather than operating in strict trace-only mode.
