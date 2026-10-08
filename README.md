# RAGmortem

A debugger for RAG applications that diagnoses whether a bad answer came from retrieval, ranking, generation, or an unsupported question.

> **Status: Experimental (v0.1 — Real RAG Integration API)**  
> **RAGmortem is an experimental, deterministic RAG failure diagnoser.** It deterministically attributes root causes across retrieval, ranking, generation, and abstention without requiring an external LLM judge.
>
> **Honest Benchmark Summary:**
> - **BioTrial Hidden Holdout (100 Cases)**: **100%** corpus-aware accuracy on the current frozen benchmark (96.0% overall / 95.0% resolved / 100% unknown recall).
> - **FinDebt Domain Holdout (40 Cases)**: **83.3%** accuracy on resolved cases, **65.0%** overall on the 40-case holdout.
> - **Trace-Only Mode Coverage**: Trace-only mode has intentionally lower coverage (~60.0% on holdout) because without active corpus index access, weak retrievals legitimately cannot distinguish whether the retriever missed indexed documents or the question was inherently unanswerable from the corpus. It safely refuses ambiguous cases into `UNKNOWN`.
> - **Controlled Benchmarks $\neq$ Real-World Validation**: Controlled synthetic benchmarks verify heuristic logic and taxonomy boundaries, but do NOT guarantee 100% real-world accuracy or universal RAG compatibility. RAGmortem is currently an experimental research and developer debugging tool.

---

## 1. Privacy & Data Safety

RAGmortem is designed as a local-first, privacy-respecting developer debugging tool:

* **Zero Telemetry**: No usage tracking, no analytics, no phone-home pings.
* **No Network Calls**: The diagnostic engine runs 100% locally on your machine or server.
* **No Stored API Keys**: Diagnoser does not require, store, or transmit LLM API keys.
* **No Data Uploads**: Prompts, retrieved documents, and generated answers are never uploaded anywhere.
* **Safe Telemetry Handling**: Does not automatically log or persist sensitive document text unless you explicitly write it to a trace. Local trace persistence is strictly opt-in.

---

## 2. Python SDK Integration

Integrate RAGmortem into any Python RAG application with minimal code.

### Installation

```bash
pip install -e .
```

Verify in a clean process:
```bash
python -c "import ragmortem; print(ragmortem.__version__)"
```

### Trace Construction

Capture what your application observed during execution using `ragmortem.trace`:

```python
from ragmortem import trace, diagnose

# Record runtime execution telemetry
t = trace(
    query="What is the mandatory engineering response time for a P1 incident?",
    retrieved_chunks=[
        {"id": "doc_dr", "text": "Disaster recovery drills occur quarterly.", "score": 0.14},
        {"id": "doc_sla", "text": "Communication cadence is bi-weekly.", "score": 0.12},
    ],
    scores=[0.14, 0.12],
    answer="Engineers must run quarterly disaster recovery exercises.",
    context="[1] Disaster recovery drills occur quarterly.\n[2] Communication cadence is bi-weekly.",
)
```

Missing telemetry is never silently fabricated (missing scores, context, or answer remain explicit and safely trigger `UNKNOWN`).

### Running Diagnosis

RAGmortem supports two explicit diagnostic modes:

1. **`trace_only`**: Evaluates purely observable APM telemetry without probing the corpus.
2. **`corpus_aware`**: Probes the corpus index when runtime retrieval scores are low to determine whether supporting documents existed in the knowledge base.

> **Rule**: Modes are explicit. RAGmortem never silently switches between `trace_only` and `corpus_aware`.

#### Mode A: Trace-Only (Offline APM / Logs)

```python
result = diagnose(t, mode="trace_only")

print(f"Failure: {result.failure_type}")
print(f"Evidence Score: {result.evidence_score}")
print(f"Why: {result.explanation}")
print(f"Recommended next step: {result.recommended_action}")
```

Output:
```text
Failure: UNKNOWN
Evidence Score: 0.00
Why: Available telemetry is insufficient to determine root cause: Corpus index unavailable in trace-only mode.
Recommended next step: Provide complete execution telemetry or enable corpus-aware mode with corpus index access.
```

#### Mode B: Corpus-Aware (Corpus Probing)

When you supply your corpus index or document directory, RAGmortem audits indexed documents:

```python
result = diagnose(
    t,
    corpus="examples/reference_rag/documents",
    mode="corpus_aware",
)

print(f"Failure: {result.failure_type}")
print(f"Evidence Score: {result.evidence_score}")
print(f"Why: {result.explanation}")
print(f"Recommended next step: {result.recommended_action}")
```

Output:
```text
Failure: RETRIEVAL_SUSPECTED (suspected)
Evidence Score: 0.94
Why: The answer becomes supported when the missing high-relevance chunk is available, but the relevant chunk was absent from the retrieved top-k results.
Recommended next step: Increase retrieval recall: inspect query reformulation, adjust embedding/indexing strategy, or expand candidate retrieval pool.
```

### Complete Reference Example

See [`examples/integrations/minimal_rag.py`](examples/integrations/minimal_rag.py) for an end-to-end runnable script demonstrating all 6 failure and validation scenarios:
1. Successful answer / appropriate abstention (`NO_FAILURE`)
2. Retrieval miss (`RETRIEVAL_SUSPECTED`)
3. Ranking cutoff miss (`RANKING_SUSPECTED`)
4. Generation ignored context (`GENERATION_SUSPECTED`)
5. Unsupported question (`ABSTENTION_SUSPECTED`)
6. Insufficient telemetry (`UNKNOWN`)

Run it directly:
```bash
python examples/integrations/minimal_rag.py
```

---

## 3. CLI: Diagnosing Saved Traces

Diagnose saved trace files directly from the command line:

```bash
# Trace-only diagnosis from a JSON file
ragmortem diagnose-trace trace.json

# Corpus-aware diagnosis with corpus document directory
ragmortem diagnose-trace trace.json --corpus examples/reference_rag/documents

# Structured JSON output for piping or APM integration
ragmortem diagnose-trace trace.json --json
```

### Trace JSON Format

```json
{
  "query": "What is the mandatory engineering response time for a P1 incident?",
  "retrieved_chunks": [
    {
      "id": "chunk_dr_drill_frequency",
      "text": "Disaster recovery drills occur quarterly.",
      "score": 0.1407
    },
    {
      "id": "chunk_communication_cadence",
      "text": "All-hands communication occurs bi-weekly.",
      "score": 0.1285
    }
  ],
  "scores": [0.1407, 0.1285],
  "answer": "Engineers must execute drills quarterly.",
  "context": "[1] Disaster recovery drills occur quarterly.",
  "metadata": {
    "model": "qwen-2.5",
    "latency_ms": 320.5
  }
}
```

---

## 4. Diagnostic Modes & Validation Benchmarks

RAGmortem separates diagnostic evaluation into distinct operating modes to ensure scientific credibility and avoid claiming false certainty:

| Evaluation Mode / Benchmark | Input Requirements | Diagnostic Mechanism | Day 5 In-Sample (122 Cases) | Day 6 Cloud Holdout (120 Cases) | Day 7/8 BioTrial Holdout (100 Cases) | Day 8 FinDebt Holdout (40 Cases) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Realistic Observational Mode (Corpus-Aware)** | Execution trace + vector index query access | Candidate cutoff probe, relative score margins, question support audit | **100.0%** (122/122) | **90.8%** (109/120 all) / **100%** unknown recall | **100.0%** frozen benchmark / **96.0%** all / **100%** unknown recall | **65.0%** (26/40 all) / **83.3%** resolved |
| **Trace-Only Mode (Offline APM Telemetry)** | Execution trace only (zero vector index access) | Cutoff probe, refusal check, ambiguity bands | N/A | **58.3%** (70/120 all) / **50.0%** resolved | **60.0%** (60/100 all) / **50.0%** resolved (100% precision) | **7.5%** (safely refuses to guess without corpus) |
| **Reference-Assisted Mode** | Trace + developer expected reference answer | Answer correctness check, chunk containment scan | **100.0%** (122/122) | Evaluated on demand | Evaluated on demand | Evaluated on demand |
| **Oracle Upper-Bound Mode** | Trace + hidden gold evidence chunk | Single-chunk counterfactual oracle replay | **100.0%** (122/122) | 100.0% | 100.0% | 100.0% |

---

## 5. Failure Taxonomy

RAGmortem classifies failures into four root causes plus a first-class `unknown` fallback:

1. **`retrieval_miss` (`retrieval_suspected`)**: The required evidence exists in the corpus/index, but the retriever failed to return it in the candidate set.
2. **`ranking_miss` (`ranking_suspected`)**: The required evidence was retrieved in the candidate pool, but was truncated below the prompt top-k context window.
3. **`generation_ignored_context` (`generation_suspected` / `generation_proven`)**: Relevant evidence was present in the prompt context, but the generator hallucinated, ignored it, or produced a false refusal.
4. **`should_abstain` (`abstention_suspected`)**: The question cannot be answered from the corpus, but the system generated an ungrounded substantive answer instead of refusing.
5. **`unknown`**: Available telemetry is insufficient (e.g. missing scores, dropped candidate pool) or signals are ambiguous. RAGmortem explicitly returns `UNKNOWN` rather than guessing.
6. **`no_failure`**: Answer generation succeeded from retrieved context, or the system appropriately refused an unanswerable question.

---

## 6. Current Limitations & Scope

* **Experimental Status**: The diagnoser heuristics are evaluated against frozen research benchmarks and are undergoing iteration. Do not treat results as production-guaranteed certainty.
* **Single-Hop Factual Focus**: Optimized for single-chunk answerable and unanswerable lookups. Multi-hop synthesis across multiple disjoint documents remains active research.
* **Uncalibrated Confidence**: `evidence_score` represents relative deterministic signal strength, not a calibrated Bayesian probability.
* **Corpus Requirement for Abstention**: Without corpus/index access, trace-only mode cannot reliably separate retrieval misses from unanswerable queries and will safely output `UNKNOWN`.

---

## 7. Testing

Run the full test suite:

```bash
pytest tests/ -v
```

All 102 unit, integration, benchmark, and uncertainty tests run 100% offline without network calls or API keys.

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
