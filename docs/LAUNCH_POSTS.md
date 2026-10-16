# RAGmortem v0.2.0 — Launch Posts

Copy-paste-ready drafts for Show HN, Reddit, and X. Keep claims aligned with
`README.md`, `docs/EVIDENCE.md`, and `benchmarks/REAL_WORLD_RESULTS.md`.

Repo: https://github.com/deveshkolte/RAGmortem
PyPI: https://pypi.org/project/ragmortem/
One-liner demo: `uvx ragmortem demo`

---

## 1. Show HN

**Title:**
`Show HN: RAGmortem – A local CPU root-cause debugger for RAG pipelines`

**URL:** https://github.com/deveshkolte/RAGmortem

**Body:**

RAG observability usually gives you a single quality score ("faithfulness: 0.42"). That tells you *something* is wrong, not *which component broke*.

RAGmortem is a local, deterministic root-cause debugger for RAG. You hand it an execution trace — query, retrieved chunks with scores, optional candidate pool, the generated answer — and it returns one of: retrieval miss, ranking miss (retrieved but cut by top-k), generation failure (context was good, the model ignored it), or "the system should have abstained."

Everything runs on CPU, offline, with no LLM judge and no API keys:

    uvx ragmortem demo

It has three modes:
- trace-only: uses only what you logged
- corpus-aware: audits your index to tell retrieval misses from genuinely unanswerable questions
- reference-assisted: given a ground-truth answer, gives a definitive verdict

The interesting part is the top-k cutoff probe. When a re-ranker pushes the right document below the context window, scores look "fine" and quality metrics stay quiet:

    ────────── ✂️ TOP-K CUTOFF (k=2) ──────────
    #3 [0.710] ███████░░░ chunk_db_failover_timeout ⚠️ (-0.020 gap)
    #4 [0.600] ██████░░░░ chunk_db_metrics          ⚠️ (-0.130 gap)

RAGmortem reads the score cliff across the candidate pool, not just the top-k, and names it.

What I'm most interested in feedback on:
1. Is deterministic attribution more useful to you than another probabilistic score?
2. Which logs do you actually have in production — do you keep pre-top-k candidate scores?

There's a real-world benchmark against the public SQuAD v1.1 dataset (real Wikipedia passages + questions, deterministic BM25 traces): reference-assisted mode attributed 100% (50/50) of failures to the correct component at 0% UNKNOWN, and across ~10k questions 86% of genuine retrieval failures were top-k truncation, not corpus misses. Methodology and threats to validity are in benchmarks/REAL_WORLD_RESULTS.md — I've tried to be explicit about what is simulated.

Apache-2.0. Happy to answer anything.

---

## 2. Reddit

Subreddits: r/LocalLLaMA, r/MachineLearning, r/LangChain, r/Rag

**Title:**
`I built a local debugger that tells you *which* part of your RAG pipeline failed (retrieval vs ranking vs generation) — no LLM judge`

**Body:**

Most RAG eval tools give you a score. That's a dead end when you're debugging: "faithfulness 0.42" doesn't tell you whether your retriever, your re-ranker, or your generator is the problem.

RAGmortem is an open-source, deterministic root-cause diagnoser. Input is a trace; output is a component verdict: `retrieval_miss`, `ranking_miss`, `generation_ignored_context`, `should_abstain`, or `unknown`. No API keys, no external judge, runs on CPU:

    uvx ragmortem demo

A few things that make it different from a quality score:

**1. It separates "retrieved but cut" from "never retrieved."** The single most under-diagnosed RAG failure is a re-ranker pushing the right chunk just below the top-k window. Scores look healthy; the answer is wrong. RAGmortem probes the score cliff across your candidate pool:

    ────────── ✂️ TOP-K CUTOFF (k=2) ──────────
    #3 [0.710] ███████░░░ chunk_db_failover_timeout ⚠️ (-0.020 gap)
    #4 [0.600] ██████░░░░ chunk_db_metrics          ⚠️ (-0.130 gap)

**2. It's honest about uncertainty.** When telemetry is genuinely borderline it returns `unknown` instead of guessing.

**3. Modes match your data.** trace-only (just your logs), corpus-aware (audits the index to tell retrieval misses from unanswerable queries), reference-assisted (definitive verdict from a ground-truth answer).

I also ran it against a real public dataset, not just my own fixtures. Using SQuAD v1.1 (real Wikipedia paragraphs and questions, deterministic BM25 traces):
- reference-assisted mode: 100% (50/50) correct attribution, 0% unknown
- over ~10k questions, 86% of actual retrieval failures were top-k truncation
- median 0.19 ms per diagnosis, CPU-only

Caveats are spelled out in the repo (simulated decoy answers rather than recorded LLM outputs; clean single-hop QA; 16-passage corpora).

Would love feedback, especially: what telemetry do you actually retain in prod, and would you keep pre-top-k candidate scores to get ranking diagnostics?

https://github.com/deveshkolte/RAGmortem

---

## 3. X (Twitter) thread

**Tweet 1/5**
Most RAG observability gives you a score:

"faithfulness: 0.42"

That tells you something broke.
It does NOT tell you what broke.

I built RAGmortem: a local, deterministic root-cause debugger for RAG. CPU-only, no LLM judge, no API keys.

`uvx ragmortem demo`

🧵

**Tweet 2/5**
Give it a trace → get a component verdict:

• retrieval_miss — doc wasn't found
• ranking_miss — found, but cut by top-k
• generation_ignored_context — context was good, model ignored it
• should_abstain — unanswerable question
• unknown — telemetry too thin, refuses to guess

No scores to interpret. A named cause.

**Tweet 3/5**
The failure nobody catches: your re-ranker pushes the right chunk just below the top-k window.

Scores look healthy. Quality metrics stay quiet. The answer is wrong.

RAGmortem reads the score cliff across the candidate pool:

    ────────── ✂️ TOP-K CUTOFF (k=2) ──────────
    #3 [0.710] ███████░░░ chunk_db_failover_timeout ⚠️ (-0.020 gap)
    #4 [0.600] ██████░░░░ chunk_db_metrics          ⚠️ (-0.130 gap)

**Tweet 4/5**
I tested it on real public data, not just my own fixtures.

SQuAD v1.1 (real Wikipedia passages + questions, deterministic BM25 traces):

• 100% (50/50) correct attribution, 0% unknown (reference-assisted)
• across ~10k questions, 86% of retrieval failures were top-k truncation
• 0.19 ms median per diagnosis, CPU-only

**Tweet 5/5**
Three modes: trace-only, corpus-aware (audits your index), reference-assisted (definitive).

Apache-2.0. Run it in 10 seconds:

`uvx ragmortem demo`

Repo + full methodology + honest caveats:
https://github.com/deveshkolte/RAGmortem

Feedback welcome — would you keep pre-top-k candidate scores in prod?
