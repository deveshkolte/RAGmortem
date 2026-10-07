# RAGmortem

An open-source debugger for Retrieval-Augmented Generation (RAG) applications that isolates why an answer failed — whether due to retrieval miss, ranking miss, generator ignoring context, or failing to abstain.

> **Status: Experimental (v0.1 / Day 8 Answerability & Abstention Discrimination)**  
> **RAGmortem is an experimental deterministic RAG failure diagnoser.** It deterministically attributes root causes across retrieval, ranking, generation, and abstention without requiring an external LLM judge.
>
> **Current Benchmark Summary:**
> - **88.0% – 96.0%** corpus-aware accuracy on BioTrial hidden holdout (100% unknown recall)
> - **60.0%** trace-only accuracy (safely refuses unanswerable/retrieval ambiguities into UNKNOWN)
> - **Ranking and generation diagnostics remain strong** (100% recall on holdout)
> - **Abstention/retrieval separation remains the primary research problem**: background domain terminology overlap frequently causes elevated similarity in unanswerable queries
> - **Explicit UNKNOWN handling**: deterministic refusal when telemetry or corpus evidence is ambiguous

---

## 2. Diagnostic Modes & Validation Benchmarks

RAGmortem explicitly separates diagnostic evaluation into distinct operating modes and benchmark sets to ensure scientific credibility and prevent false claims of certainty:

| Evaluation Mode / Benchmark | Input Requirements | Diagnostic Mechanism | Day 5 In-Sample (122 Cases) | Day 6 Cloud Holdout (120 Cases) | Day 7/8 BioTrial Holdout (100 Cases) | Day 8 FinDebt Holdout (40 Cases) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Realistic Observational Mode (Corpus-Aware)** | Execution trace + vector index query access | Candidate cutoff probe, relative score margins, question support audit | **100.0%** (122/122) | **90.8%** (109/120 all) / **100%** unknown recall | **96.0%** (96/100 all) / **95.0%** resolved / **100%** unknown recall | **65.0%** (26/40 all) / **83.3%** resolved |
| **Trace-Only Mode (Offline APM Telemetry)** | Execution trace only (zero vector index access) | Cutoff probe, refusal check, ambiguity bands | N/A | **58.3%** (70/120 all) / **50.0%** resolved | **60.0%** (60/100 all) / **50.0%** resolved (100% precision) | **7.5%** (safely refuses to guess without corpus) |
| **Reference-Assisted Mode** | Trace + developer expected reference answer | Answer correctness check, chunk containment scan | **100.0%** (122/122) | Evaluated on demand | Evaluated on demand | Evaluated on demand |
| **Oracle Upper-Bound Mode** | Trace + hidden gold evidence chunk | Single-chunk counterfactual oracle replay | **100.0%** (122/122) | 100.0% | 100.0% | 100.0% |

> **Critical Findings on Generalization & Uncertainty (Days 5, 6, & 7)**:  
> - **Day 5 In-Sample Controlled Benchmark**: Achieved 100.0% accuracy on the synthetic operational dataset used during development. This served as an upper-bound verification that the taxonomy and heuristics are logically coherent.
> - **Day 6 Baseline Holdout Limitation**: On an unseen domain (50 cloud/compliance chunks, 120 cases), corpus-aware accuracy dropped to 81.7%, trace-only to 50.8%, and unknown recall was 0.0% because fixed cosine thresholds forced ambiguous cases into arbitrary failure buckets.
> - **Day 7 Uncertainty-Aware Advancement**: By replacing rigid cosine thresholds with relative score margins, candidate gap metrics, deterministic evidence scoring, and an explicit `UNKNOWN` first-class result:
>   - **Day 6 Holdout (Phase A)** improved from 81.7% to **89.2%** accuracy, with unknown recall jumping from 0.0% to **100.0%** (20/20).
>   - **Fresh Hidden Holdout (Phase B - BioTrial Clinical Trials Domain)**: Evaluated with frozen code on 100 unseen cases (20 retrieval, 20 ranking, 20 generation, 20 abstention, 20 unknown). Achieved **88.0%** overall accuracy, **85.0%** accuracy on resolved cases, **100.0%** unknown recall (20/20), **100.0%** ranking miss recall (20/20), and **100.0%** generation failure recall (20/20).
> - **Trace-Only Boundary**: Without active corpus index access, weak retrievals legitimately cannot distinguish whether the retriever missed indexed documents or the user question was inherently unanswerable. The diagnoser now safely returns `UNKNOWN` (with `evidence_score = 0.0` and explicit limitations), achieving 100% unknown recall and 100% precision on resolved cases in trace-only mode.

---

## 3. Uncertainty-Aware Diagnostic Workflow

```text
Failed RAG execution trace
        ↓
Validate telemetry completeness (COMPLETE / PARTIAL / INSUFFICIENT)
        ↓
Missing query, answer, context, or scores? → UNKNOWN (evidence_score = 0.0)
        ↓
Refusal detected in answer?
     /        \
   YES        NO
    ↓          ↓
Appropriate  Candidate pool > top-k?
Refusal      /        \
(or False   YES        NO
Refusal)     ↓          ↓
        Ranking      Borderline ambiguity band? [0.47, 0.53]
        Cutoff       /        \
        Probe      YES        NO
                    ↓          ↓
                 UNKNOWN    High context score? (> 0.53)
                            /        \
                          YES        NO
                           ↓          ↓
                      Generation    Corpus access requested?
                      Suspected     /        \
                                  YES        NO
                                   ↓          ↓
                              Corpus audit:  Trace-only:
                              Separation?    UNKNOWN
                              /        \     (cannot separate)
                            YES        NO
                             ↓          ↓
                          Retrieval  Abstention
                          Suspected  Suspected
```

### Core Diagnostic Mechanisms:

1. **Telemetry Completeness**: Explicitly validates trace integrity (`COMPLETE`, `PARTIAL`, `INSUFFICIENT`). Missing scores, missing context, or missing answers directly produce `UNKNOWN` rather than defaulting to low scores.
2. **Deterministic Evidence Score (`evidence_score`)**: Replaces hardcoded pseudo-confidence numbers (0.75, 0.85) with an explainable, deterministic evidence score calculated from relative score separation, cutoff gaps, candidate pool quality, and reference agreement.
3. **Relative Signals & Ambiguity Bands**: Rather than brittle fixed thresholds, diagnoses use relative score drops between candidates ($\text{cutoff\_gap} \le 0.08$), candidate density, clean corpus separation ($\ge 0.06$), and an explicit ambiguity band ($[0.47, 0.53]$) where borderline relevance without reference answers returns `UNKNOWN`.
4. **First-Class `UNKNOWN`**: Returns `UNKNOWN` with structured explanatory reasons and limitations whenever evidence is insufficient, conflicting, or requires unavailable corpus access.
5. **Clear Trace-Only vs. Corpus-Aware Separation**:
   - `diagnose(trace, mode="trace_only")`: Pure APM trace evaluation without corpus probing.
   - `diagnose(trace, corpus=app, mode="corpus_aware")`: Audits corpus index when runtime retrieval scores are low.

In Day 3 of the technical sprint, we benchmarked existing open-source RAG evaluation and diagnostic tools against our 122 validated controlled failure cases ([`benchmarks/results.md`](benchmarks/results.md)):

* **What existing tools do**: Frameworks like Ragas, TruLens, and DeepEval compute continuous quality metrics (e.g., `context_recall`, `faithfulness`, `answer_relevance`) via LLM-as-a-judge. Vector-specific tools like PyVectorHound compute embedding space geometry (isotropy, MRR).
* **What they don't do**:
  * **Top-K Blindness**: Metric evaluators only inspect the final prompt context window. When an item falls outside top-k, they universally report a "retrieval failure", resulting in **0% recall on ranking truncation (`ranking_miss`)** and misdiagnosing 100% of ranking failures as retrieval misses.
  * **Component Blindness**: Vector diagnostics evaluate only the vector index, remaining completely blind to generation failures and hallucinations.
  * **Universal Abstention Blindspot**: Zero existing tools evaluate whether an LLM properly abstained on unanswerable questions (`should_abstain`).
* **Why RAGmortem exists**: To move from *scoring symptoms* (continuous numbers) to *diagnosing root causes* (discrete, counterfactually verified fault attribution).
* **Benchmark Status**: Across 122 controlled failure cases, existing tools achieve $\le 49.2\%$ overall diagnostic accuracy. Full comparative metrics and confusion matrices are documented in [`benchmarks/results.md`](benchmarks/results.md).
* **Known Limitations**: Competitor predictions were benchmarked against their documented decision boundaries because Ragas installation timed out due to heavy dependencies, and PyVectorHound's PyPI release currently contains a corrupted wheel.

---

## 3. Current v0.1 Scope


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

### Generate Controlled Failure Dataset
Synthetically inject controlled failures across the 4 failure modes (100% deterministic, offline):

```bash
ragmortem generate-faults
```

### Validate Injected Fault Dataset
Verify that all generated cases strictly adhere to their ground-truth failure conditions:

```bash
ragmortem validate-faults
```

### Diagnose a RAG Execution Trace
Run root-cause failure diagnosis on an execution trace (JSON or JSONL). By default, uses purely **observational telemetry** without accessing ground-truth labels:

```bash
# 1. Observational Mode (default, production telemetry)
ragmortem diagnose examples/traces/ranking_failure.json

# 2. Reference-Assisted Mode (with user-provided expected answer)
ragmortem diagnose examples/traces/generation_failure.json --mode reference_assisted --reference-answer "15 minutes"

# 3. Oracle Mode (benchmark upper bound with counterfactual replay)
ragmortem diagnose evals/injected_faults.jsonl --mode oracle
```

### Evaluate Diagnoser Against Benchmark Dataset
Benchmark the diagnoser across all 122 failure cases:

```bash
# Evaluate observational mode (default, outputs evals/observed_diagnoses.jsonl)
ragmortem evaluate-diagnoser

# Evaluate oracle upper-bound mode (outputs evals/diagnoses.jsonl)
ragmortem evaluate-diagnoser --mode oracle
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

## 5. Controlled Failure Dataset

RAGmortem deliberately breaks a reference RAG system in known, controlled ways. Every generated failure case preserves its original baseline execution and receives a trustworthy ground-truth label verified against strict validation criteria. These labelled failures serve as the gold-standard benchmark to evaluate the diagnosis engine in subsequent sprint phases.

> **Scope Note**: This failure taxonomy is not claimed to encompass every possible failure mode across all RAG deployments. In v1, RAGmortem strictly focuses on four canonical root causes:

1. **Retrieval Miss (`retrieval_miss`)**:
   - *What happens*: The gold chunk containing the required evidence exists in the corpus/index, but the retriever fails to retrieve it within the top-k candidate set.
   - *Strict Rule*: The gold chunk must **never** be deleted from the corpus (which would turn it into an abstention failure). The gold chunk is omitted purely through controlled query representation degradation.
2. **Ranking Miss (`ranking_miss`)**:
   - *What happens*: The gold chunk is successfully found within the broader candidate pool (e.g. top-6), but falls below the final top-k cutoff (e.g. ranked at #4 when context window is top-3) and is excluded from prompt context.
   - *Strict Rule*: The gold chunk must exist in the candidate pool with rank > top-k.
3. **Generation Ignored Context (`generation_ignored_context`)**:
   - *What happens*: The gold chunk is present directly in the prompt context supplied to the LLM, but the generator ignores the evidence and outputs an incorrect or contradictory answer.
   - *Strict Rule*: The failure must not be caused by retrieval; the gold chunk must be proven to be inside the LLM prompt.
4. **Should Abstain (`should_abstain`)**:
   - *What happens*: The user question cannot be answered from the corpus, but the system produces a confident, fabricated answer instead of refusing or abstaining.
   - *Strict Rule*: The question must genuinely have no supporting evidence in the corpus, and the system must output a non-abstaining response.

Dataset records are saved to `evals/injected_faults.jsonl`.

---

## 6. Dataset Format

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

## 7. Architecture

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
│       ├── cli.py                     # CLI commands (run-reference, generate-faults, diagnose, etc.)
│       ├── diagnose.py                # Deterministic failure diagnoser & rank probe engine
│       └── faults/                    # Controlled fault injection framework
│           ├── __init__.py            # Module exports
│           ├── models.py              # FaultConfig, BaselineExecution, InjectedFaultCase
│           ├── eval.py                # Answer verification & strict fault validators
│           └── inject.py              # FaultInjector engine & dataset generator
├── examples/
│   └── reference_rag/
│       ├── README.md                  # Reference RAG documentation
│       ├── app.py                     # Runnable reference RAG pipeline
│       └── documents/                 # Reference technical policy corpus
├── evals/
│   ├── questions.jsonl                # 38 evaluation questions
│   ├── injected_faults.jsonl          # 122 validated synthetic failure cases
│   ├── diagnoses.jsonl                # 122 audit records from diagnoser evaluation
│   └── diagnoser_results.md           # Day 4 diagnostic benchmark report
└── tests/                             # Full test suite (36 tests, 100% offline runnable)
    ├── test_types.py
    ├── test_adapter.py
    ├── test_cache.py
    ├── test_dataset_validation.py
    ├── test_mock_rag.py
    ├── test_retrieval.py
    ├── test_faults.py
    ├── test_benchmark.py
    └── test_diagnoser.py
```

---

## 8. Current Limitations

- **Single-hop factual focus**: v0.1 supports single-chunk answerable questions. Multi-hop composite evidence queries require extending oracle replay to multi-chunk combinations.
- **In-memory Vector Store**: Designed for local reference corpora (<10,000 chunks) using NumPy rather than distributed vector databases.
- **Candidate Pool Access Requirement**: Discerning ranking misses from retrieval misses requires observing the candidate retrieval pool prior to prompt top-k cutoff.

---

## 9. Roadmap

- **Day 1**: Runnable foundation, core types, reference RAG, caching, 38 evaluation questions. [DONE]
- **Day 2**: Controlled fault-injection framework, 4 failure modes, 122 validated failure cases. [DONE]
- **Day 3**: Competitor benchmark & reality check (Ragas, TruLens, DeepEval, Phoenix). [DONE]
- **Day 4**: Core deterministic failure diagnoser via oracle replay & rank probing. [DONE]
- **Day 5**: Realistic observational diagnosis mode (zero gold label access, production telemetry). [DONE]
- **Day 6**: End-to-end failure suite evaluation across model families.
- **Day 7**: CLI polish, benchmarking summary, and release documentation.

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.

