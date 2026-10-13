# Empirical Evidence & Dataset Provenance

> **Plain Statement on Dataset Independence:**  
> **All labels in this repository are synthetic and created by this project's own fault injector.**  
> **No real-world failures from production deployments have been evaluated yet.**

This document details the provenance, creation methodology, tuning status, and performance results of every benchmark dataset located in `evals/`.

---

## 1. Dataset Provenance Table

| Dataset File | Domain / Topic | Total Cases | Who Created Labels | Used for Tuning? | Sample Size ($n$) | Reported Results | Notes / Caveats |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **`evals/injected_faults.jsonl`** | SRE & Incident Response Policies | 122 | Synthetic fault injector (`FaultInjector` in `src/ragmortem/faults/inject.py`) | **YES (Tuned on)** | $n=122$ | 100% accuracy in controlled oracle benchmark | In-sample development dataset used to calibrate initial baseline heuristics. |
| **`evals/holdout_faults.jsonl`** (AeroCloud) | Cloud Infrastructure & Multi-Tenant SLAs | 120 | Synthetic holdout script (`scripts/generate_holdout_dataset.py`) | **YES (Rules changed after first eval)** | $n=120$ | 90.8% overall (109/120) corpus-aware; 58.3% trace-only; 100% unknown recall | First evaluated in commit `45d76ea`. Thresholds and ambiguity rules were subsequently modified in commits `8a3142b` and `c4656c5`. |
| **`evals/hidden_holdout_faults.jsonl`** (BioTrial) | Clinical Research & FDA 21 CFR Regulations | 100 | Synthetic hidden holdout script (`scripts/generate_hidden_holdout.py`) | **YES (TUNED ON — Not Independent)** | $n=100$ | 88.0% pre-tuning baseline (`8a3142b`); **100.0% post-tuning** (`c4656c5`) | Errors were directly inspected via `scripts/inspect_abstention_errors.py` and domain terms were hardcoded into `src/ragmortem/support.py`. |
| **`evals/abstention_holdout.jsonl`** (FinDebt: Development) | Syndicated Credit Facility Agreements | 20 | Synthetic generator (`scripts/generate_abstention_holdout.py`) | **YES (Development split)** | $n=20$ | 75.0% overall (15/20); 80.0% resolved (12/15) | Used for development exploration of financial credit agreements. |
| **`evals/abstention_holdout.jsonl`** (FinDebt: Validation) | Syndicated Credit Facility Agreements | 20 | Synthetic generator (`scripts/generate_abstention_holdout.py`) | **YES (Validation split)** | $n=20$ | 55.0% overall (11/20); 66.7% resolved (10/15) | Intermediate validation split. |
| **`evals/abstention_holdout.jsonl`** (FinDebt: Final Holdout) | Syndicated Credit Facility Agreements | 40 | Synthetic generator (`scripts/generate_abstention_holdout.py`) | **NO (Frozen split)** | $n=40$ ($n=30$ resolved, $n=10$ unknown) | **65.0% overall (26/40)**; **83.3% resolved (25/30)**; 10 UNKNOWN (92.5% coverage) | Evaluated on frozen code without post-eval tuning. **Caveat: $n=30$ resolved cases is a small sample size.** |

---

## 2. Git History Audit: Were Rules Changed After Evaluation?

An audit of the commit history traces when datasets were introduced and whether diagnostic rules or thresholds in `src/ragmortem/diagnose.py` or `src/ragmortem/support.py` changed afterward:

### 2.1 AeroCloud (`evals/holdout_faults.jsonl`)
* **First introduced**: Commit `45d76ea` (*test: add adversarial diagnoser holdout benchmark*, 2026-10-08 00:09:27 +0530).
* **Were rules or thresholds changed afterward?** **YES.**
  - **Commit `8a3142b`** (*feat: make diagnoser uncertainty aware*): Introduced the ambiguity band `[0.47, 0.53]`, score margin thresholds `0.05`, false refusal thresholds `0.54` / `0.50`, and candidate cutoff gap `0.08`.
  - **Commit `c4656c5`** (*feat: improve answerability diagnosis*): Replaced static cosine thresholds with `src/ragmortem/support.py` entity alignment and terminology trap analysis.
* **Conclusion**: AeroCloud was not a frozen, untouched holdout; algorithm logic was revised following its initial creation.

### 2.2 BioTrial (`evals/hidden_holdout_faults.jsonl`)
* **First introduced**: Commit `45d76ea` / `8a3142b`.
* **Were rules or thresholds changed afterward?** **YES (TUNED ON).**
  - In commit `8a3142b` (recorded in `evals/baselines/day7_baseline.md`), BioTrial scored 88.0%, producing 8 false positives (`should_abstain` $\rightarrow$ `retrieval_miss`) due to dense clinical terminology overlap.
  - The script `scripts/inspect_abstention_errors.py` was created specifically to inspect those 8 failure cases.
  - In commit `c4656c5`, specific clinical terms (`"participants"`, `"clinical"`, `"trial"`, `"trials"`, `"study"`, `"studies"`) were added to `QUESTION_META_TERMS` in `src/ragmortem/support.py`, and `is_terminology_trap` heuristics were engineered to resolve these errors, boosting BioTrial to 100.0%.
* **Conclusion**: BioTrial must be treated as a **tuned-on dataset**, not an independent blind benchmark.

### 2.3 FinDebt (`evals/abstention_holdout.jsonl`)
* **First introduced**: Commit `c4656c5` (*feat: improve answerability diagnosis*, 2026-10-08 01:20:39 +0530).
* **Were rules or thresholds changed afterward?** **NO.**
  - An inspection of `git diff c4656c5..HEAD -- src/ragmortem/diagnose.py src/ragmortem/support.py` demonstrates that no diagnostic decision rule, threshold (`0.47/0.53/0.54/0.50`), or answerability formula in `support.py` was modified after commit `c4656c5`. Subsequent commits only added public API wrappers, CLI output helpers, and packaging bug fixes.
* **Final Holdout Result**:
  - Evaluated on 40 cases ($n=40$).
  - Overall accuracy: **65.0% (26/40)**.
  - Resolved accuracy (excluding `UNKNOWN`): **83.3% (25/30)**.
  - True ambiguous / unresolvable cases routed to `UNKNOWN`: **10 / 10 (100.0% unknown recall)**.
  - Coverage: **92.5%** (37/40 cases attempted).
  - **Caveat**: The sample size ($n=40$ total, $n=30$ resolved) is small. While it demonstrates out-of-domain generalization without post-eval tuning, larger independent evaluations are required before citing general production accuracy.
