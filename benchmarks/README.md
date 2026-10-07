# RAGmortem Competitor Benchmark

This directory contains the benchmark harness evaluating existing RAG evaluation and diagnostic tools against RAGmortem's 122 validated controlled failure cases ([`evals/injected_faults.jsonl`](../evals/injected_faults.jsonl)).

## Purpose

To empirically test whether existing tools solve the 4 canonical failure modes targeted by RAGmortem:
1. `retrieval_miss`
2. `ranking_miss`
3. `generation_ignored_context`
4. `should_abstain`

## Files

- **`mappings.json`**: Official documentation evidence, installation statuses, and taxonomy mappings for competitors.
- **`run_benchmark.py`**: Automated evaluation harness testing competitor decision models on the 122 failure cases without label leakage.
- **`results.json`**: Machine-readable confusion matrices and classification metrics.
- **`results.md`**: Comprehensive Day 3 engineering report, capability matrix, and product decision analysis.

## Reproducing the Benchmark

```bash
# Run benchmark simulation and output metrics
python benchmarks/run_benchmark.py
```
