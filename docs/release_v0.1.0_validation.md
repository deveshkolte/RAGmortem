# RAGmortem v0.1.0 Release Validation Report

**Release Date:** October 8, 2026  
**Target Release Tag:** `v0.1.0`  
**License:** Apache-2.0  
**Python Support:** >= 3.11  

---

## 1. Executive Summary

This report documents the final release audit and validation for **RAGmortem v0.1.0**. The codebase architecture was frozen following the completion of deterministic failure diagnosis and realistic integration capabilities. All 102 offline tests pass, CLI execution is verified across multiple working directories, package installation and module imports are verified, and an end-to-end realistic RAG demonstration successfully exercises all six core failure scenarios.

---

## 2. Test Suite Validation

The test suite was run offline with no external network connectivity or API dependencies:

* **Command:** `pytest -q`
* **Result:** **102 passed** (0 failures, 0 warnings)
* **Execution Time:** ~167s (includes local embedding generation and model inference)
* **Coverage Scope:**
  - `tests/test_diagnoser.py`: Deterministic diagnosis decision tree and heuristics
  - `tests/test_observed_diagnoser.py`: Observational mode without oracle knowledge
  - `tests/test_uncertainty.py`: Evidence scoring, thresholds, and boundary conditions
  - `tests/test_abstention.py`: Refusal detection and unanswerable question handling
  - `tests/test_adapter.py`: Trace adapters and legacy schema compatibility
  - `tests/test_cli.py`: Command-line interface and argument parsing
  - `tests/test_fault_injection.py`: Controlled synthetic fault injectors

---

## 3. Package & Environment Verification

* **Editable Installation:** `pip install -e .` completes cleanly in virtual environment.
* **Process Isolation:** Verified module import in an isolated Python process:
  ```bash
  python -c "import ragmortem; print(ragmortem.__version__)"
  # Output: 0.1.0
  ```
* **Package Entrypoint:** `ragmortem` CLI binary installed to `bin/ragmortem`.

---

## 4. CLI Verification

Verified `ragmortem diagnose-trace` against sample trace data:

1. **Execution from Repository Root:**
   ```bash
   ragmortem diagnose-trace evals/abstention_holdout.jsonl --limit 1 --corpus examples/reference_rag/documents
   ```
   *Exit code 0, successful diagnosis output.*

2. **Execution from External Working Directory (`evals/`):**
   ```bash
   (cd evals && ragmortem diagnose-trace abstention_holdout.jsonl --limit 1)
   ```
   *Exit code 0, successful diagnosis output in trace-only mode.*

3. **Output Modes:** Both human-readable text and structured JSON (`--json`) formats verified.

---

## 5. End-to-End Demo Verification

Ran `examples/integrations/realistic_rag_demo.py` covering the complete RAG lifecycle:
`Document Ingestion -> Embedding -> Semantic Retrieval -> Prompt Construction -> Mock Generator -> RAGmortem Trace -> Diagnosis`.

All 6 scenarios verified:
1. **Scenario 1 (Grounded Answer):** Diagnosed as `NO_FAILURE` (score: 0.95).
2. **Scenario 2 (Retrieval Miss):** Diagnosed as `RETRIEVAL_SUSPECTED` (score: 0.94) with recommendation to expand candidate pool.
3. **Scenario 3 (Ranking Miss):** Diagnosed as `RANKING_SUSPECTED` (score: 0.96) with recommendation to tune re-ranker / increase top-k cutoff.
4. **Scenario 4 (Generation Hallucination):** Diagnosed as `GENERATION_SUSPECTED` (score: 0.95) with recommendation to tighten grounding prompt.
5. **Scenario 5 (Out-of-Domain Question):** Diagnosed as `ABSTENTION_SUSPECTED` (score: 0.92) with recommendation to add strict refusal instructions.
6. **Scenario 6 (Incomplete Telemetry):** Diagnosed as `UNKNOWN` (score: 0.00) with recommendation to log missing candidate scores.

---

## 6. Security, Privacy & Hygiene Audit

* **API Keys & Secrets:** 0 found. RAGmortem does not require or store LLM API keys.
* **Environment Files:** 0 `.env` files tracked or present.
* **Telemetry & Network Calls:** 0 found. All diagnostic logic runs 100% locally on CPU without external requests.
* **Paths:** 0 user-specific hardcoded paths in library code.

---

## 7. Known Limitations & Scope

1. **Experimental v0.1 Status:** The diagnostic heuristics are deterministic and reproducible, but have not yet been validated on uncontrolled production workloads.
2. **Benchmark vs. Production Gap:** High scores on synthetic benchmarks and holdout sets do not guarantee equivalent accuracy in arbitrary real-world RAG architectures.
3. **Single-Hop Factual Focus:** The diagnoser is designed primarily for single-hop factual QA. Multi-hop synthesis across disparate documents is active research.
4. **Uncalibrated Evidence Score:** `evidence_score` represents heuristic confidence and signal strength, not a calibrated Bayesian probability.
5. **Trace-Only Uncertainty:** When corpus indexing is unavailable, RAGmortem intentionally abstains with `UNKNOWN` on ambiguous retrieval versus abstention cases rather than hallucinating certainty.
