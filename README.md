# RAGmortem

A debugger for RAG apps: it finds why an answer went wrong — retrieval, ranking, generation, or lack of evidence — and tests the diagnosis before you ship it.

> **Project Status: Experimental open-source RAG debugger (v0.1)**  
> RAGmortem is an experimental, deterministic root-cause failure diagnoser for RAG pipelines. It isolates failures across retrieval, ranking, generation, and abstention without requiring an external LLM judge.

```python
from ragmortem import diagnose, Chunk

# 20-second quickstart: diagnose an execution trace directly
result = diagnose(
    query="What is the mandatory engineering response time for a P1 incident?",
    retrieved_chunks=[{"id": "doc_dr", "text": "Disaster recovery drills occur quarterly.", "score": 0.14}],
    answer="Engineers must run quarterly disaster recovery exercises.",
    corpus=[
        Chunk(id="chunk_incident_severity_p1", text="P1 (Critical) incidents require acknowledgment within 15 minutes.", source="policy"),
        Chunk(id="chunk_dr_drill", text="Disaster recovery drills occur quarterly.", source="policy"),
    ],
    mode="corpus_aware",
)

print(result.to_text())
```

Output:
```text
Failure: RETRIEVAL_SUSPECTED (suspected)
Evidence Score: 0.94

Why:
The answer becomes supported when the missing high-relevance chunk is available, but the relevant chunk was absent from the retrieved top-k results.

Evidence:
  * Retriever returned low-relevance chunks at runtime (top score: 0.140).
  * Corpus audit shows indexed documents exist with high relevance to question.
  * Corpus contains candidate 'chunk_incident_severity_p1' with distinct semantic relevance (score: 0.741).

Recommended next step:
Increase retrieval recall: inspect query reformulation, adjust embedding/indexing strategy, or expand candidate retrieval pool.

Mode: corpus_aware
```

---

## 1. What Problem RAGmortem Solves

When a RAG system returns an incorrect answer, traditional observability tools only compute continuous quality scores (e.g. faithfulness: 0.42). They do not tell you **which component broke**:
- **Did the retriever fail to find the document?** (`retrieval_miss`)
- **Did candidate retrieval find it, but ranking dropped it below top-k?** (`ranking_miss`)
- **Did the generator receive the correct context, but ignore it or hallucinate?** (`generation_ignored_context`)
- **Was the question unanswerable from the knowledge base, but the model answered anyway?** (`should_abstain`)
- **Is telemetry insufficient to distinguish the cause?** (`unknown`)

RAGmortem replaces ambiguous scores with deterministic, reproducible root-cause attribution and concrete next-step remediation actions.

---

## 2. Privacy & Data Safety

RAGmortem is built as a local-first developer debugging tool:

* **Zero Telemetry**: No tracking, no external analytics, no telemetry platform integration.
* **No Network Calls**: Diagnostic evaluation executes 100% locally on your CPU.
* **No Stored API Keys**: Diagnoser does not require, store, or transmit LLM API keys.
* **No Prompts/Answers Uploaded**: Your data stays on your machine.
* **No Automatic Sensitive Logging**: Document texts and queries are evaluated in-memory. Local trace persistence is strictly opt-in.

---

## 3. How It Works

RAGmortem uses an uncertainty-aware decision tree operating on observable execution telemetry:

1. **Telemetry Completeness Check**: Validates that query, answer, context, and similarity scores are present. Missing telemetry directly produces `UNKNOWN` rather than fabricated certainty.
2. **Abstention & Refusal Probe**: Detects whether the generator emitted a refusal. If low retrieval score matches refusal, marks `NO_FAILURE` (appropriate abstention). If context had strong scores, marks false refusal.
3. **Candidate Cutoff Probe**: Inspects the candidate retrieval pool prior to top-k truncation. If a viable candidate exists immediately below the cutoff ($\text{score gap} \le 0.08$), isolates a `RANKING_SUSPECTED` failure.
4. **Context Score Probe**: High similarity in context with an ungrounded or contradictory answer isolates `GENERATION_SUSPECTED`.
5. **Corpus Audit vs. Trace-Only Boundary**:
   - In `corpus_aware` mode, audits the corpus index to distinguish whether documents were missed (`RETRIEVAL_SUSPECTED`) or the question was unanswerable (`ABSTENTION_SUSPECTED`).
   - In `trace_only` mode, safely returns `UNKNOWN` because an APM trace without corpus access cannot infer document existence.

---

## 4. Installation

Requires **Python >= 3.11**.

Install from GitHub:
```bash
pip install git+https://github.com/deveshkolte/RAGmortem.git
```

Or for local development / testing:
```bash
git clone https://github.com/deveshkolte/RAGmortem.git
cd RAGmortem
pip install -e .
```

Verify in a clean terminal:
```bash
ragmortem --help
python -c "import ragmortem; print(ragmortem.__version__)"
```

---

## 5. Python SDK Usage

### Trace Construction

Use `ragmortem.trace` to record runtime execution telemetry:

```python
from ragmortem import trace, diagnose

t = trace(
    query="What is the P1 incident response SLA?",
    retrieved_chunks=[
        {"id": "doc_dr", "text": "Disaster recovery drills occur quarterly.", "score": 0.14},
        {"id": "doc_sla", "text": "Communication cadence is bi-weekly.", "score": 0.12},
    ],
    scores=[0.14, 0.12],
    answer="Engineers must run quarterly disaster recovery exercises.",
    context="[1] Disaster recovery drills occur quarterly.\n[2] Communication cadence is bi-weekly.",
)
```

Missing fields are never silently fabricated (missing scores, context, or answer remain explicit).

### Diagnostic Operating Modes

RAGmortem supports two explicit **diagnostic scopes** (**never silently switches between them**):

| Scope | Mode Parameter | Corpus Required? | Reference Answer? | What It Verifies |
|-------|----------------|------------------|-------------------|------------------|
| **Trace-Only** | `trace_only` | No | No | Only observable telemetry (scores, refusal, candidate pool). Returns `UNKNOWN` when evidence is ambiguous. |
| **Corpus-Aware Observed** | `corpus_aware` (no `reference_answer`) | Yes | No | Heuristic diagnosis: retrieval/ranking/generation/abstention suspicions based on corpus audit. **Does not verify answer correctness.** |
| **Corpus-Aware Reference-Assisted** | `corpus_aware` + `reference_answer` | Yes | **Yes** | Definitive diagnosis: compares generated answer to reference, searches retrieved/candidate chunks for supporting evidence. Returns `NO_FAILURE`, `GENERATION_PROVEN`, `RETRIEVAL_MISS`, `RANKING_MISS`, `SHOULD_ABSTAIN`, or `UNKNOWN`. |

#### Trace-Only (Offline APM / Logs)
Zero knowledge-base dependencies. Evaluates observable trace telemetry directly:
```python
result = diagnose(t, mode="trace_only")
print(result.failure_type)        # FailureType.UNKNOWN (safely avoids guessing)
print(result.recommended_action) # Explains missing corpus access
```

#### Corpus-Aware (Knowledge Base Probing)
Probes the corpus index to disambiguate retrieval vs. answerability failures.
Accepts three input formats:

**1. Directory path of markdown files:**
```python
result = diagnose(t, corpus="path/to/markdown/documents", mode="corpus_aware")
```

**2. In-memory list of `Chunk` objects (or dicts):**
```python
from ragmortem import Chunk

corpus = [
    Chunk(id="doc_p1", text="P1 incidents require 15-minute response.", source="policy"),
    Chunk(id="doc_dr", text="Disaster recovery drills occur quarterly.", source="policy"),
]
result = diagnose(t, corpus=corpus, mode="corpus_aware")
```

**3. Custom retriever adapter (`.retrieve(query, k)`):**
```python
# Any object implementing .retrieve(query: str, k: int) -> tuple[list[Chunk], list[float]]
result = diagnose(t, corpus=my_custom_retriever, mode="corpus_aware")
```

**With reference answer (definitive diagnosis):**
```python
result = diagnose(
    t,
    corpus=corpus,
    mode="corpus_aware",
    reference_answer="P1 incidents require 15-minute response.",
)
```

> **Key distinction**: Without `reference_answer`, `corpus_aware` mode uses **heuristics** (corpus audit + score thresholds) and returns *suspected* failure types (`GENERATION_SUSPECTED`, `RETRIEVAL_SUSPECTED`, etc.). It **cannot verify whether the generated answer is factually correct** — it only flags when high-relevance context was present but the answer may be wrong. With `reference_answer`, the diagnosis becomes *definitive* (`NO_FAILURE`, `GENERATION_PROVEN`, `RETRIEVAL_MISS`, etc.) by comparing the answer to the reference and searching the corpus for supporting evidence.

### Complete Integration Demonstrations

* [`examples/integrations/minimal_rag.py`](examples/integrations/minimal_rag.py): Minimal code integration across all 6 diagnostic scenarios.
* [`examples/integrations/realistic_rag_demo.py`](examples/integrations/realistic_rag_demo.py): End-to-end small RAG app smoke test (`documents → embeddings → retrieval → context → generator → RAGmortem`).

Run the demo locally:
```bash
python examples/integrations/realistic_rag_demo.py
```

---

## 6. CLI: Diagnosing Saved Traces

Diagnose saved JSON trace files from the terminal:

```bash
# Trace-only mode (offline APM logs, no corpus needed)
ragmortem diagnose-trace trace.json

# Corpus-aware mode (with a documents directory)
ragmortem diagnose-trace trace.json --corpus path/to/documents

# Structured JSON output
ragmortem diagnose-trace trace.json --corpus path/to/documents --json
```

### Standard JSON Trace Schema

```json
{
  "query": "What is the mandatory engineering response time for a P1 incident?",
  "retrieved_chunks": [
    {
      "id": "chunk_dr_drill_frequency",
      "text": "Disaster recovery drills occur quarterly.",
      "score": 0.1407
    }
  ],
  "scores": [0.1407],
  "answer": "Engineers must execute drills quarterly.",
  "context": "[1] Disaster recovery drills occur quarterly.",
  "metadata": {
    "model": "qwen-2.5",
    "latency_ms": 320.5
  }
}
```

---

---

## 7. Failure Taxonomy

| Failure Mode | Canonical Type | What Happened | Recommended Next Step |
| :--- | :--- | :--- | :--- |
| **`retrieval_suspected`** | `retrieval_miss` | Supporting chunk exists in corpus but was omitted from candidate set. | Increase retrieval recall, adjust embedding model, or expand candidate pool. |
| **`ranking_suspected`** | `ranking_miss` | Supporting chunk was retrieved in candidates, but fell below top-k cutoff. | Tune re-ranker, expand context window (top-k), or re-score candidates. |
| **`generation_suspected`** | `generation_ignored_context` | High-relevance context was in prompt, but generator hallucinated or ignored it. **Observed mode only — suspected, not proven.** | Tighten prompt grounding constraints, lower temperature, refine prompt. |
| **`abstention_suspected`** | `should_abstain` | Corpus contains no supporting evidence, but generator fabricated an answer. | Add strict refusal instructions for out-of-domain queries; filter low scores. |
| **`unknown`** | `unknown` | Telemetry is incomplete (missing scores, dropped candidates) or signals are ambiguous. | Provide complete telemetry or supply corpus index access. |
| **`no_failure`** | `no_failure` | **Reference-assisted mode only**: Answer is verified grounded, or system correctly refused unanswerable question. | No corrective action required. |
| **`generation_proven`** | `generation_ignored_context` | **Reference-assisted mode only**: Reference answer found in retrieved context, but generator produced a different answer. | Tighten prompt grounding constraints, lower temperature, refine prompt. |
| **`retrieval_miss`** | `retrieval_miss` | **Reference-assisted mode only**: Reference answer exists in corpus but was not retrieved. | Increase retrieval recall, adjust embedding model, or expand candidate pool. |
| **`ranking_miss`** | `ranking_miss` | **Reference-assisted mode only**: Reference answer found in candidate pool but ranked below top-k. | Tune re-ranker, expand context window (top-k), or re-score candidates. |
| **`should_abstain`** | `should_abstain` | **Reference-assisted mode only**: Reference answer not in corpus; system should have abstained. | Add strict refusal instructions; filter low scores. |

> **Note**: `generation_suspected`, `retrieval_suspected`, `abstention_suspected`, `ranking_suspected` are **heuristic suspicions** returned by observed modes (trace-only and corpus-aware without `reference_answer`). The definitive types `generation_proven`, `retrieval_miss`, `ranking_miss`, `should_abstain`, `no_failure` are only returned by reference-assisted mode (`corpus_aware` + `reference_answer`).

---

## 8. Validation Benchmarks & Historical Evidence

To ensure scientific credibility, RAGmortem distinguishes strictly between **controlled benchmarks**, **adversarial holdouts**, and **integration smoke tests**.

> **Important**: Controlled benchmark metrics do NOT represent real-world accuracy on uncurated production data.

| Evaluation Suite / Benchmark | Purpose | Corpus-Aware Accuracy | Trace-Only Accuracy | Unknown Recall | Key Findings |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Day 5 Controlled In-Sample** (122 cases) | Development baseline | 100.0% (122/122) | N/A | N/A | Verified taxonomy and heuristic logic coherence under ideal conditions. |
| **Day 6 Adversarial Holdout** (120 cases) | Cloud/DevOps holdout | 90.8% (109/120) | 58.3% (70/120) | 100.0% (20/20) | Exposed fixed-threshold brittleness; motivated uncertainty-aware scoring. |
| **Day 7/8 BioTrial Frozen Holdout** (100 cases) | Clinical trials holdout | 100.0% (frozen upper-bound) / 96.0% overall | 60.0% (50.0% resolved, 100% precision) | 100.0% (20/20) | 100% recall on ranking misses and generation failures. Trace-only safely refuses low-score ambiguity. |
| **Day 8 FinDebt Domain Holdout** (40 cases) | Complex corporate finance | 65.0% overall / 83.3% resolved | 7.5% (safely refuses) | 100.0% | Demonstrated that background terminology traps in specialized domains require lexical support analysis. |
| **Integration Smoke Test** (`realistic_rag_demo.py`) | End-to-end integration | Smoke validation | Smoke validation | Smoke validation | Proves public API usability in real Python code without synthetic fault injection. |

---

## 9. Current Limitations & Scope

* **Experimental Status**: Heuristics are deterministic and reproducible, but remain experimental research. Do not treat as production-guaranteed certitude.
* **Single-Hop Factual Focus**: Optimized for single-chunk answerable lookups. Multi-hop reasoning across disjoint documents remains active research.
* **Uncalibrated Confidence**: `evidence_score` represents relative deterministic heuristic strength, not a calibrated Bayesian probability.
* **Trace-Only Uncertainty**: Without corpus index access, trace-only mode cannot infer document presence in the corpus and intentionally outputs `UNKNOWN`.
* **Reference-Answer Matching**: In reference-assisted mode, the generated answer is compared to `reference_answer` using deterministic token matching:
  1. **Exact token subsequence match** (after lowercase, punctuation removal, whitespace collapse) — if the full gold token sequence appears contiguously in the candidate, it's correct.
  2. **Fallback: ≥80% key-token overlap** — key tokens are those with length >2 or numeric. If ≥80% of gold key tokens appear in the candidate, it's correct. Numeric tokens (e.g., "15", "2024") are **required** to be present exactly.
  
  This is **not semantic equivalence** — paraphrased answers with different wording may not match. For best results, provide `reference_answer` as a verbatim substring of a corpus chunk. The fallback helps with minor phrasing differences but does not handle synonym substitution or reordering.
* **Observed-Mode Heuristics**: Without `reference_answer`, `corpus_aware` and `trace_only` modes return *suspected* failure types based on score thresholds and corpus audit. They **cannot verify factual correctness** of the generated answer — they only flag when telemetry patterns match known failure signatures. A `GENERATION_SUSPECTED` result means "high-relevance context was present but answer may be wrong," not "answer is factually incorrect."

---

## 10. Testing

Run the full offline test suite:

```bash
pytest tests/ -v
```

All 124 tests run 100% offline on CPU without network requests or API keys.

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
