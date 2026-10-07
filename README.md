# RAGmortem

An open-source debugger for Retrieval-Augmented Generation (RAG) applications that isolates why an answer failed — whether due to retrieval miss, ranking miss, generator ignoring context, or failing to abstain.

> **Status: Experimental (v0.1 / Day 1 Foundation)**  
> RAGmortem is currently an active experimental project focused on controlled failure diagnosis and reproducible counterfactual replay. It is not yet intended for production telemetry or end-user dashboards.

---

## 1. What Problem It Solves

When a RAG system provides an incorrect, incomplete, or hallucinated answer, diagnosing *why* it failed is frequently difficult:
- Did the retriever fail to locate the required document (Retrieval Miss)?
- Did the retriever find the chunk, but rank it too low for the context window (Ranking Miss)?
- Was the evidence present in the prompt, but the LLM hallucinated or ignored it (Generation Ignored Context)?
- Did the user ask an unanswerable question that the system should have abstained from, but instead fabricated an answer (Should Abstain)?

RAGmortem provides a systematic, reproducible method to trace and attribute these failures using explicit contracts, deterministic replay, and controlled fault verification.

---

## 2. Current v0.1 Scope

Day 1 establishes the runnable foundation:
- **Core Data Types**: Lightweight dataclasses (`Chunk`, `RagResult`) preserving chunk IDs, rankings, scores, and metadata.
- **RAG Adapter Protocol**: Minimal `RagAdapter` contract (`query(question) -> RagResult`) enabling zero-friction integration with existing RAG apps.
- **Reference RAG Application**: A local, runnable RAG app utilizing `sentence-transformers` (`all-MiniLM-L6-v2`) for local vector retrieval and Groq / mock completion.
- **Deterministic Disk Cache**: SHA-256 keyed JSON cache ensuring identical queries never incur duplicate API costs or latency.
- **Evaluation Dataset**: 38 curated questions (30 answerable with verified 1-to-1 gold chunks + 8 unanswerable) and automated validation.
- **CLI & Evaluation Runner**: Commands to validate question datasets and evaluate RAG pipelines in both mock and live modes.

---

## 3. Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### Installation

```bash
# Clone the repository
git clone https://github.com/deveshkolte/RAGmortem.git
cd RAGmortem

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install package and test dependencies
pip install -e ".[dev]"
```

### Validate Dataset
Verify that all evaluation questions match chunks in the corpus and adhere to schema constraints:

```bash
ragmortem validate-dataset
```

### Run Evaluation in Offline Mock Mode
No API key or network access required:

```bash
ragmortem run-reference --mock
```

### Run Evaluation with Live Groq Model
Set your free Groq API key:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
ragmortem run-reference
```

---

## 4. Reference RAG

The included reference RAG application (`examples/reference_rag/`) operates on a synthetic technical policy handbook:
1. **Document Ingestion**: Markdown files with clear chunk markers (`## [chunk_id] Title`).
2. **Local Embeddings**: Encoded using `all-MiniLM-L6-v2` locally on CPU.
3. **Vector Indexing**: NumPy normalized dot-product (cosine similarity) ranking.
4. **Context Construction**: Formats top-k chunks into a strict context prompt.
5. **Generation**: Groq (`qwen/qwen3.8-27b`) or deterministic mock fallback.

---

## 5. Dataset Format

Evaluation questions are stored in JSON Lines (`evals/questions.jsonl`):

```json
{
  "id": "q01",
  "question": "What is the mandatory engineering response time for a P1 incident?",
  "gold_chunk_id": "chunk_incident_severity_p1",
  "gold_answer": "15 minutes from alert firing to triage.",
  "type": "answerable"
}
{
  "id": "q31_unans",
  "question": "What is the maximum reimbursement amount for employee ergonomic home office equipment?",
  "gold_chunk_id": null,
  "gold_answer": "I do not have sufficient information in the provided context to answer this question.",
  "type": "unanswerable"
}
```

Rules:
- **`answerable`**: Requires an exact, existing `gold_chunk_id` in the corpus.
- **`unanswerable`**: `gold_chunk_id` must be null; information is absent from the corpus.

---

## 6. Architecture

```
RAGmortem
├── pyproject.toml                     # Packaging and dependencies
├── README.md                          # Documentation
├── LICENSE                            # Apache-2.0
├── src/
│   └── ragmortem/
│       ├── __init__.py                # Package exports
│       ├── types.py                   # Chunk, RagResult
│       ├── adapter.py                 # RagAdapter protocol & FunctionAdapter
│       ├── taxonomy.py                # FailureType definitions
│       ├── cache.py                   # Deterministic ModelCache
│       ├── dataset.py                 # Loader and validator for evaluation questions
│       └── cli.py                     # CLI commands (validate-dataset, run-reference)
├── examples/
│   └── reference_rag/
│       ├── README.md                  # Reference RAG documentation
│       ├── app.py                     # Runnable reference RAG pipeline
│       └── documents/                 # Reference technical policy corpus
├── evals/
│   └── questions.jsonl                # 38 evaluation questions
└── tests/                             # Full test suite (16 tests, 100% offline runnable)
```

---

## 7. Current Limitations

- **Single-hop factual focus**: v0.1 supports single-chunk answerable questions. Multi-hop and temporal reasoning queries are out of scope.
- **In-memory Vector Store**: Designed for small reference corpora (<10,000 chunks) using NumPy rather than distributed vector databases.
- **No Automated Fault Injection Yet**: Day 1 establishes the baseline; controlled fault injection and diagnostic probes are planned for subsequent milestones.

---

## 8. Roadmap

- **Day 2**: Failure injection harnesses (synthetic retrieval drops, rank demotion, irrelevant context injection).
- **Day 3**: Counterfactual replay engine (swapping retrieved chunks to isolate cause).
- **Day 4**: Failure diagnostic judge & attribution scoring.
- **Day 5**: Automated fix recommendations and context patch analysis.
- **Day 6**: End-to-end failure suite evaluation across model families.
- **Day 7**: CLI polish, benchmarking summary, and release documentation.

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
