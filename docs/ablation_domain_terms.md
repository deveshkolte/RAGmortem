# Ablation Study: Impact of Domain-Specific Meta Terms

This document records the empirical results of an ablation study conducted on `QUESTION_META_TERMS` in `src/ragmortem/support.py`.

---

## 1. Objective

To determine whether the project's lexical support heuristics rely on domain-specific hardcoded keywords, we temporarily ablated all domain-specific vocabulary terms from `QUESTION_META_TERMS`:
* **Clinical / Medical terms**: `participants`, `clinical`, `trial`, `trials`, `study`, `studies`
* **Finance / Credit terms**: `borrower`, `lender`, `lenders`, `trigger`, `triggers`

The general question framing terms (e.g. `what`, `how`, `required`, `mandated`, `minimum`, `maximum`, `pursuant`, `accordance`, `relates`) remained untouched.

---

## 2. Experimental Setup

* **Code Modified**: `src/ragmortem/support.py` (lines 37–38 temporarily removed, evaluated read-only, and immediately reverted via `git checkout -- .`).
* **Evaluation Harness**: Evaluated all 4 evaluation benchmark datasets across both **Corpus-Aware** and **Trace-Only** modes.
* **Metric Definitions**:
  * **Overall Accuracy**: Total correct diagnoses / total cases.
  * **Resolved Accuracy**: Correct diagnoses on non-unknown ground-truth cases.
  * **UNKNOWN Count**: Cases classified as `UNKNOWN`.

---

## 3. Results Comparison (Before vs. After Ablation)

| Dataset | Split / Size | Mode | Before Ablation (Overall) | After Ablation (Overall) | Delta (Overall) | Before (Resolved) | After (Resolved) | Delta (Resolved) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Sample Baseline** (`evals/injected_faults.jsonl`) | All ($n=122$) | Corpus-Aware | 96.72% (118/122) | 96.72% (118/122) | **0.0%** | 96.72% (118/122) | 96.72% (118/122) | **0.0%** |
| | All ($n=122$) | Trace-Only | 48.36% (59/122) | 48.36% (59/122) | **0.0%** | 48.36% (59/122) | 48.36% (59/122) | **0.0%** |
| **AeroCloud Holdout** (`evals/holdout_faults.jsonl`) | All ($n=120$) | Corpus-Aware | 85.83% (103/120) | 85.83% (103/120) | **0.0%** | 83.00% (83/100) | 83.00% (83/100) | **0.0%** |
| | All ($n=120$) | Trace-Only | 58.33% (70/120) | 58.33% (70/120) | **0.0%** | 50.00% (50/100) | 50.00% (50/100) | **0.0%** |
| **BioTrial Frozen Holdout** (`evals/hidden_holdout_faults.jsonl`) | All ($n=100$) | Corpus-Aware | 100.00% (100/100) | 100.00% (100/100) | **0.0%** | 100.00% (80/80) | 100.00% (80/80) | **0.0%** |
| | All ($n=100$) | Trace-Only | 60.00% (60/100) | 60.00% (60/100) | **0.0%** | 50.00% (40/80) | 50.00% (40/80) | **0.0%** |
| **FinDebt Final Holdout** (`evals/abstention_holdout.jsonl`) | Holdout ($n=40$) | Corpus-Aware | **65.00% (26/40)** | **57.50% (23/40)** | **-7.5%** | **83.33% (25/30)** | **73.33% (22/30)** | **-10.0%** |
| | Holdout ($n=40$) | Trace-Only | 7.50% (3/40) | 7.50% (3/40) | **0.0%** | 0.00% (0/30) | 0.00% (0/30) | **0.0%** |
| **FinDebt Validation** (`evals/abstention_holdout.jsonl`) | Val ($n=20$) | Corpus-Aware | 55.00% (11/20) | 50.00% (10/20) | **-5.0%** | 66.67% (10/15) | 66.67% (10/15) | **0.0%** |
| **FinDebt Development** (`evals/abstention_holdout.jsonl`) | Dev ($n=20$) | Corpus-Aware | 75.00% (15/20) | 80.00% (16/20) | **+5.0%** | 80.00% (12/15) | 86.67% (13/15) | **+6.7%** |

---

## 4. Key Findings

1. **FinDebt Shows Sensitivity to Financial Terms**:
   * On the FinDebt Final Holdout ($n=40$), removing `borrower`, `lender(s)`, and `trigger(s)` dropped corpus-aware overall accuracy from **65.0% down to 57.5% (-7.5%)**, and resolved accuracy from **83.3% down to 73.3% (-10.0%)**.
   * In credit agreement documents, words like "borrower" and "lender" permeate almost every clause. When these words are not treated as background framing terms, the support heuristic treats their lexical overlap as false evidence of answerability, causing the diagnoser to misclassify unanswerable queries as retrieval misses.

2. **Zero Impact on In-Sample IT, AeroCloud, and BioTrial**:
   * On the In-Sample incident response dataset and AeroCloud infrastructure holdout, the domain terms are completely absent from queries, resulting in exactly **0.0% difference**.
   * On BioTrial, queries contained sufficient independent clinical trial regulation numbers (e.g. 21 CFR Part 312, IND amendment timelines) that stripping `clinical`, `trial`, and `participants` did not change the diagnostic classification outcome.

3. **Conclusion & Recommendation**:
   * The inclusion of domain-specific stop terms provides a measurable performance benefit (+7.5% to +10.0% on specialized legal/credit documents), but represents manual domain calibration rather than domain-agnostic heuristics.
   * Future versions should transition from static wordlists to dynamic corpus-wide inverse document frequency (IDF) or semantic answerability scoring.
