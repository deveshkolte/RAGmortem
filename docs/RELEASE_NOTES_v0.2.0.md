# RAGmortem v0.2.0

**Release date:** 2026-10-16

RAGmortem is a local-first, CPU-only root-cause debugger for RAG pipelines. Give it an execution trace — query, retrieved chunks with scores, optional candidate pool, and the generated answer — and it returns a deterministic verdict naming the broken component: retrieval miss, ranking miss (evidence retrieved but cut by top-k), generation failure (evidence in prompt, answer ignored it), should-abstain, or `unknown` when telemetry is insufficient. v0.2.0 brings a screenshot-ready Rich terminal UI with top-k candidate-cliff inspection, four calibration profiles, a two-tier answer-verification stack, a built-in pytest plugin, and a real-world benchmark built on the public SQuAD v1.1 dataset.

## Key Highlights

### Rich CLI with top-k candidate cliff inspection
`ragmortem demo` (zero-install via `uvx ragmortem demo`) renders an executive terminal report: per-case diagnosis badges, confidence meters, and a candidate cutoff chart that visualizes exactly where the re-ranker pushed the right chunk below the top-k context window.

```
────────── ✂️ TOP-K CUTOFF (k=2) ──────────
#3 [0.710] ███████░░░ chunk_db_failover_timeout ⚠️ (-0.020 gap)
#4 [0.600] ██████░░░░ chunk_db_metrics          ⚠️ (-0.130 gap)
```

New SVG export: `ragmortem demo --out docs/hero.svg` writes a shareable SVG image of the report.

### Calibration profiles: `minilm`, `openai`, `bge`, `relative`
Selectable score calibration via `--profile` (CLI) or `profile=` (API):
- `minilm` / `openai` / `bge` — distribution baselines for common embedding models (e.g. `text-embedding-3-small/large`, `bge-large-en-v1.5`, `all-MiniLM-L6-v2`).
- `relative` — empirical auto-calibration from the observed candidate pool (mean, spread, z-scores) for unknown embedding distributions, instead of fixed thresholds.

### Two-tier answer verification
Lexical verification is the default: fast, deterministic token-subsequence matching, fully offline. Upgrading to semantic verification is one extra install: `pip install "ragmortem[local-nli]"` adds local cross-encoder/NLI entailment checking for paraphrased answers — still no API keys, no external judge.

### `pytest-ragmortem` assertion plugin
Installing the package also installs the pytest plugin (auto-registered via the `pytest11` entry point). Failed assertions in RAG tests are automatically diagnosed for component root cause, and `ragmortem_assert` makes assertions traceable:

```python
from ragmortem import ragmortem_assert

def test_production_qa():
    run_output = my_rag_app.query("What is the mandatory P1 response SLA?")
    ragmortem_assert(trace_input=run_output, reference_answer="15 minutes", profile="minilm")
```

On failure, pytest prints the retrieval/ranking/generation attribution alongside the assertion error. The `rag_trace` fixture additionally auto-diagnoses ordinary assertion failures and emits GitHub Actions annotations plus a step-summary table in CI.

### Real-world benchmark on public data (SQuAD v1.1)
`benchmarks/run_real_eval.py` evaluates the diagnoser on traces built from the public SQuAD v1.1 development set — real Wikipedia passages, human-written questions, deterministic BM25 retrieval traces, and verified-wrong decoy answers (no LLM calls). Full methodology and threats to validity: [`benchmarks/REAL_WORLD_RESULTS.md`](../benchmarks/REAL_WORLD_RESULTS.md).

- **86.2%** of genuine retrieval failures across 9,957 real questions were caused by top-k candidate truncation (evidence retrieved but cut before the prompt), not corpus misses.
- **100% attribution accuracy (50/50)** in reference-assisted mode, with a **0% UNKNOWN rate**.
- **0.19 ms median** per trace, single CPU process, no network or API calls.

## Quickstart

```bash
pip install ragmortem
```

No-install demo:

```bash
uvx ragmortem demo
```

```python
from ragmortem import diagnose, Chunk

result = diagnose(
    query="What is the mandatory engineering response time for a P1 incident?",
    retrieved_chunks=[{"id": "doc_dr", "text": "Disaster recovery drills occur quarterly.", "score": 0.14}],
    answer="Engineers must run quarterly disaster recovery exercises.",
    corpus=[
        Chunk(id="chunk_incident_severity_p1", text="P1 (Critical) incidents require acknowledgment within 15 minutes.", source="policy"),
    ],
    mode="corpus_aware",
)
print(result.to_text())
```

Docs: [README](https://github.com/deveshkolte/RAGmortem#readme) · PyPI: [ragmortem](https://pypi.org/project/ragmortem/) · License: Apache-2.0
