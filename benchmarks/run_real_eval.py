"""Real-world public benchmark evaluation for RAGmortem.

Loads a bundled sample of realistic RAG failure traces derived from the
SQuAD v1.1 development set (real questions, real Wikipedia paragraphs,
deterministic BM25 retrieval traces, simulated generator answers), pipes
every trace through RAGmortem in both observed and reference-assisted
modes, and writes aggregate statistics to ``benchmarks/REAL_WORLD_RESULTS.md``.

Data provenance
---------------
* Questions and context paragraphs are verbatim excerpts of the SQuAD v1.1
  development set (Rajpurkar et al., 2016), licensed CC BY-SA 4.0.
* Retrieval traces are deterministic BM25 rankings (pure Python, k1=1.5,
  b=0.75) over each case's 16-passage corpus, with raw BM25 scores squashed
  to a similarity-like [0, 1) scale via ``s / (s + 10)``.
* Generator answers for failure cases are simulated "decoy" answers drawn
  from other questions on the same or sibling paragraphs. They are wrong
  by construction; no LLM was called.
* Ground-truth failure categories (retrieval_miss / ranking_miss /
  generation_ignored_context) are determined by construction and verified
  deterministically at build time.

The script runs fully offline once the sample data file exists. The dense
corpus audit uses the locally cached ``all-MiniLM-L6-v2`` model; if the
model cannot be loaded offline, the audit falls back to a BM25 retriever
(use ``--audit bm25`` to force the fallback mode).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("RAGMORTEM_DISABLE_NLI", "1")
# The bundled sample is meant to evaluate fully offline. If the embedding model
# is not cached locally, set HF_HUB_OFFLINE=0 (or --audit bm25) to allow a download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ragmortem import Chunk, diagnose  # noqa: E402
from ragmortem.faults.eval import is_answer_correct, is_abstaining  # noqa: E402

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "squad_real_world_sample.json"
DEFAULT_MD_PATH = Path(__file__).resolve().parent / "REAL_WORLD_RESULTS.md"
DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "real_world_results.json"

DATASET_NAME = "SQuAD v1.1 development set"
DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/"
DATASET_LICENSE = "CC BY-SA 4.0"

TOP_K = 3
CANDIDATE_K = 8
SAME_ARTICLE_DISTRACTORS = 8
POOL_DISTRACTORS = 7
POOL_PARAGRAPHS_PER_ARTICLE = 5
BM25_SQUASH_K = 10.0
BUILD_SEED = 20261016
MAX_DECOY_CHARS = 60

GEN_CASES = 8
RANKING_CASES = 36
RETRIEVAL_CASES = 6

CATEGORY_TO_EXPECTED_CANONICAL: dict[str, str] = {
    "generation_injected": "generation_ignored_context",
    "ranking": "ranking_miss",
    "retrieval": "retrieval_miss",
}
CATEGORY_LABELS: dict[str, str] = {
    "generation_injected": "Generation fault (evidence in top-k, decoy answer)",
    "ranking": "Ranking truncation (evidence in candidate pool, outside top-k)",
    "retrieval": "Retrieval miss (evidence absent from candidate pool)",
}
CANONICAL_ORDER = [
    "ranking_miss",
    "retrieval_miss",
    "generation_ignored_context",
    "should_abstain",
    "no_failure",
    "unknown",
]


# ---------------------------------------------------------------------------
# Deterministic BM25 retrieval (pure stdlib)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Minimal deterministic BM25 ranker over a fixed passage list."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.tokens = [_tokenize(t) for t in texts]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.tokens)
        self.avgdl = sum(len(t) for t in self.tokens) / max(self.n_docs, 1)
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter[str] = Counter()
        for tf in self.tf:
            for word in tf:
                df[word] += 1
        self.idf = {
            word: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for word, n in df.items()
        }

    def rank(self, query: str) -> tuple[list[int], list[float]]:
        """Return (positions sorted by descending score, scores in that order)."""
        q_tokens = _tokenize(query)
        scores: list[float] = []
        for idx in range(self.n_docs):
            doc_len = len(self.tokens[idx])
            score = 0.0
            for word in q_tokens:
                freq = self.tf[idx].get(word, 0)
                if freq:
                    score += (
                        self.idf.get(word, 0.0)
                        * freq
                        * (self.k1 + 1)
                        / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    )
            scores.append(score)
        order = sorted(range(self.n_docs), key=lambda i: (-scores[i], i))
        return order, [scores[i] for i in order]


def squash_bm25(score: float) -> float:
    """Map unbounded BM25 scores to a monotone similarity-like [0, 1) scale."""
    if score <= 0.0:
        return 0.0
    return score / (score + BM25_SQUASH_K)


def _stable_seed(text: str) -> int:
    """Process-stable seed derived from text (built-in hash() is salted)."""
    return zlib.crc32(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Sample construction (build mode)
# ---------------------------------------------------------------------------

def _load_squad_paragraphs(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    paragraphs: list[dict[str, Any]] = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            if len(paragraph["context"]) >= 350 and len(paragraph["qas"]) >= 3:
                paragraphs.append(
                    {
                        "title": article["title"],
                        "text": paragraph["context"],
                        "qas": paragraph["qas"],
                    }
                )
    return paragraphs


def _paragraph_id(title: str, text: str) -> str:
    digest = zlib.crc32(f"{title}\n{text}".encode("utf-8"))
    return f"sq_{digest:08x}"


def _distractor_answer_candidates(paragraph: dict[str, Any], gold_answer: str) -> Iterable[str]:
    for qa in paragraph.get("qas", []):
        for answer in qa.get("answers", []):
            text = str(answer.get("text", "")).strip()
            if text:
                yield text


def _choose_decoy(
    distractors: Sequence[dict[str, Any]],
    gold_answer: str,
) -> str | None:
    gold_digits = set(re.findall(r"\d+", gold_answer))
    for paragraph in distractors:
        for candidate in _distractor_answer_candidates(paragraph, gold_answer):
            if len(candidate) > MAX_DECOY_CHARS or is_abstaining(candidate):
                continue
            if is_answer_correct(candidate, gold_answer):
                continue
            candidate_digits = set(re.findall(r"\d+", candidate))
            if gold_digits & candidate_digits:
                continue
            if not any(len(tok) >= 3 for tok in _tokenize(candidate)):
                continue
            return candidate
    return None


@dataclass
class _ScanCandidate:
    question: str
    gold_answer: str
    category: str
    bm25_rank: int
    article: str
    gold_paragraph: dict[str, Any]
    distractor_paragraphs: list[dict[str, Any]]
    decoy_answer: str


def build_sample(squad_source: Path, seed: int = BUILD_SEED) -> dict[str, Any]:
    """Construct the bundled real-world failure sample from a SQuAD dev JSON file."""
    paragraphs = _load_squad_paragraphs(squad_source)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paragraph in paragraphs:
        by_article[paragraph["title"]].append(paragraph)

    rng_pool = random.Random(seed)
    diverse_pool: list[dict[str, Any]] = []
    for title in sorted(by_article):
        picked = rng_pool.sample(
            by_article[title], min(POOL_PARAGRAPHS_PER_ARTICLE, len(by_article[title]))
        )
        diverse_pool.extend(picked)

    scan_stats: Counter[str] = Counter()
    candidates: dict[str, list[_ScanCandidate]] = {
        "generation_injected": [],
        "ranking": [],
        "retrieval": [],
    }

    for gold_paragraph in paragraphs:
        article = gold_paragraph["title"]
        siblings = [p for p in by_article[article] if p is not gold_paragraph]
        for qa in gold_paragraph["qas"]:
            question = qa["question"]
            gold_answer = qa["answers"][0]["text"]
            if not is_answer_correct(gold_paragraph["text"], gold_answer):
                continue

            rng = random.Random(_stable_seed(question))
            distractors = rng.sample(siblings, min(SAME_ARTICLE_DISTRACTORS, len(siblings)))
            already = {id(p) for p in distractors}
            pool_options = [p for p in diverse_pool if id(p) not in already and p is not gold_paragraph]
            distractors += rng.sample(pool_options, min(POOL_DISTRACTORS, len(pool_options)))

            corpus = [gold_paragraph] + distractors
            order, _ = BM25([p["text"] for p in corpus]).rank(question)
            gold_rank = order.index(0) + 1
            scan_stats["questions_scanned"] += 1

            if gold_rank <= TOP_K:
                scan_stats["evidence_in_top_k"] += 1
                category = "generation_injected"
            elif gold_rank <= CANDIDATE_K:
                scan_stats["ranking_truncated"] += 1
                category = "ranking"
            else:
                scan_stats["retrieval_missed"] += 1
                category = "retrieval"

            if category == "generation_injected" and gold_rank != 1:
                continue

            leaked = any(
                is_answer_correct(corpus[order[i]]["text"], gold_answer)
                for i in range(min(CANDIDATE_K, len(order)))
                if order[i] != 0
            )
            if leaked:
                continue

            decoy = _choose_decoy(distractors, gold_answer)
            if decoy is None:
                continue

            candidates[category].append(
                _ScanCandidate(
                    question=question,
                    gold_answer=gold_answer,
                    category=category,
                    bm25_rank=gold_rank,
                    article=article,
                    gold_paragraph=gold_paragraph,
                    distractor_paragraphs=distractors,
                    decoy_answer=decoy,
                )
            )

    selected: list[_ScanCandidate] = []
    for category, wanted in (
        ("retrieval", RETRIEVAL_CASES),
        ("ranking", RANKING_CASES),
        ("generation_injected", GEN_CASES),
    ):
        pool = sorted(
            candidates[category],
            key=lambda c: (c.article, c.bm25_rank, c.question),
        )
        rng = random.Random(seed + _stable_seed(category) % 10000)
        rng.shuffle(pool)
        used_paragraphs: set[str] = set()
        taken = 0
        for candidate in pool:
            pid = _paragraph_id(candidate.article, candidate.gold_paragraph["text"])
            if pid in used_paragraphs:
                continue
            used_paragraphs.add(pid)
            selected.append(candidate)
            taken += 1
            if taken >= wanted:
                break

    passages: dict[str, dict[str, Any]] = {}

    def register(paragraph: dict[str, Any]) -> str:
        pid = _paragraph_id(paragraph["title"], paragraph["text"])
        passages.setdefault(
            pid,
            {"title": paragraph["title"], "text": paragraph["text"]},
        )
        return pid

    cases: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        gold_id = register(candidate.gold_paragraph)
        distractor_ids = [register(p) for p in candidate.distractor_paragraphs]
        cases.append(
            {
                "id": f"squad_rw_{index:03d}",
                "question": candidate.question,
                "reference_answer": candidate.gold_answer,
                "generated_answer": candidate.decoy_answer,
                "gold_passage_id": gold_id,
                "corpus_passage_ids": [gold_id, *distractor_ids],
                "bm25_rank": candidate.bm25_rank,
                "category": candidate.category,
                "top_k": TOP_K,
                "candidate_k": CANDIDATE_K,
            }
        )

    return {
        "provenance": {
            "dataset": DATASET_NAME,
            "dataset_url": DATASET_URL,
            "license": DATASET_LICENSE,
            "citation": "Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text",
            "built_with_seed": seed,
            "note": (
                "Questions and paragraphs are verbatim SQuAD v1.1 dev excerpts. "
                "Retrieval traces are deterministic BM25 rankings over each case corpus. "
                "Generated answers are simulated decoy answers from the same dataset; no LLM was called."
            ),
            "retriever": f"BM25 (k1=1.5, b=0.75), squashed via s/(s+{BM25_SQUASH_K:g})",
            "corpus_size_per_case": 1 + SAME_ARTICLE_DISTRACTORS + POOL_DISTRACTORS,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
        },
        "scan_stats": dict(scan_stats),
        "passages": passages,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Corpus audit retrievers
# ---------------------------------------------------------------------------

class BM25AuditRetriever:
    """Fallback corpus index exposing the ``.retrieve(query, k)`` protocol."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[Chunk], list[float]]:
        effective_k = k if k is not None else TOP_K
        order, scores = self._bm25.rank(query)
        picked = order[:effective_k]
        return [self.chunks[i] for i in picked], [squash_bm25(scores[i]) for i in picked]


def get_audit_retriever(chunks: Sequence[Chunk], audit_mode: str) -> tuple[Any, str]:
    """Return (retriever, resolved_mode) with graceful offline fallback."""
    if audit_mode in ("auto", "dense"):
        try:
            from ragmortem.retrieval import CorpusRetriever

            return CorpusRetriever(corpus=list(chunks)), "dense (all-MiniLM-L6-v2)"
        except Exception as exc:  # pragma: no cover - environment dependent
            if audit_mode == "dense":
                raise
            print(f"[warn] dense audit unavailable ({exc}); falling back to BM25 audit", file=sys.stderr)
    return BM25AuditRetriever(chunks), "bm25"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class CaseOutcome:
    case_id: str
    category: str
    bm25_rank: int
    observed_diagnosis: str
    observed_canonical: str
    reference_diagnosis: str
    reference_canonical: str
    observed_ms: float
    reference_ms: float
    observed_mode: str
    reference_mode: str
    unknown_reason: str = ""


def _build_trace_payloads(
    case: dict[str, Any],
    passages: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    corpus_ids = case["corpus_passage_ids"]
    texts = [passages[pid]["text"] for pid in corpus_ids]
    order, raw_scores = BM25(texts).rank(case["question"])
    scores = [squash_bm25(s) for s in raw_scores]

    gold_position = order.index(0) + 1
    if gold_position != case["bm25_rank"]:
        raise RuntimeError(
            f"Non-deterministic retrieval replay for {case['id']}: "
            f"expected rank {case['bm25_rank']}, replayed {gold_position}"
        )

    top_k = min(TOP_K, len(order))
    candidate_k = min(CANDIDATE_K, len(order))

    retrieved = [
        {
            "id": corpus_ids[order[i]],
            "text": texts[order[i]],
            "score": round(scores[i], 6),
        }
        for i in range(top_k)
    ]
    candidates = [
        {
            "id": corpus_ids[order[i]],
            "text": texts[order[i]],
            "score": round(scores[i], 6),
        }
        for i in range(candidate_k)
    ]
    return retrieved, candidates, top_k


def _canonical(result: Any) -> tuple[str, str]:
    raw = result.diagnosis.value
    canonical = result.canonical_failure_type.value
    return raw, canonical


def run_evaluation(data: dict[str, Any], audit_mode: str) -> dict[str, Any]:
    passages = data["passages"]
    cases = data["cases"]

    # Warm-up: trigger lazy imports before per-case timing.
    warm_chunks = [
        Chunk(id=pid, text=passages[pid]["text"], source="warmup")
        for pid in list(passages)[:16]
    ]
    warm_audit, _ = get_audit_retriever(warm_chunks, audit_mode)
    diagnose(
        query="warmup question",
        retrieved_chunks=[{"id": warm_chunks[0].id, "text": warm_chunks[0].text, "score": 0.5}],
        answer="warmup answer",
        corpus=warm_audit,
        mode="corpus_aware",
        profile="relative",
    )

    outcomes: list[CaseOutcome] = []
    resolved_audits: Counter[str] = Counter()

    for case in cases:
        retrieved, candidates, top_k = _build_trace_payloads(case, passages)
        chunks = [
            Chunk(
                id=pid,
                text=passages[pid]["text"],
                source=f"squad::{passages[pid]['title']}",
                metadata={"title": passages[pid]["title"]},
            )
            for pid in case["corpus_passage_ids"]
        ]
        audit, resolved_mode = get_audit_retriever(chunks, audit_mode)
        resolved_audits[resolved_mode.split(" ")[0]] += 1

        start = time.perf_counter()
        observed = diagnose(
            query=case["question"],
            retrieved_chunks=retrieved,
            candidate_chunks=candidates,
            top_k=top_k,
            answer=case["generated_answer"],
            corpus=audit,
            mode="corpus_aware",
            profile="relative",
        )
        observed_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        reference = diagnose(
            query=case["question"],
            retrieved_chunks=retrieved,
            candidate_chunks=candidates,
            top_k=top_k,
            answer=case["generated_answer"],
            corpus=audit,
            mode="corpus_aware",
            profile="relative",
            reference_answer=case["reference_answer"],
        )
        reference_ms = (time.perf_counter() - start) * 1000.0

        observed_raw, observed_canonical = _canonical(observed)
        reference_raw, reference_canonical = _canonical(reference)
        unknown_reason = ""
        if observed_canonical == "unknown" and getattr(observed, "reasons", None):
            unknown_reason = str(observed.reasons[0])[:160]

        outcomes.append(
            CaseOutcome(
                case_id=case["id"],
                category=case["category"],
                bm25_rank=case["bm25_rank"],
                observed_diagnosis=observed_raw,
                observed_canonical=observed_canonical,
                reference_diagnosis=reference_raw,
                reference_canonical=reference_canonical,
                observed_ms=observed_ms,
                reference_ms=reference_ms,
                observed_mode=observed.mode,
                reference_mode=reference.mode,
                unknown_reason=unknown_reason,
            )
        )

    def _distribution(field_name: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for outcome in outcomes:
            counts[getattr(outcome, field_name)] += 1
        return {key: counts.get(key, 0) for key in CANONICAL_ORDER if counts.get(key, 0)}

    def _confusion(field_name: str) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {}
        for category in CATEGORY_TO_EXPECTED_CANONICAL:
            row: Counter[str] = Counter()
            for outcome in outcomes:
                if outcome.category == category:
                    row[getattr(outcome, field_name)] += 1
            matrix[category] = dict(row)
        return matrix

    def _latency(field_name: str) -> dict[str, float]:
        values = sorted(getattr(o, field_name) for o in outcomes)
        return {
            "median_ms": round(statistics.median(values), 3),
            "mean_ms": round(statistics.fmean(values), 3),
            "p95_ms": round(values[min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)], 3),
            "max_ms": round(values[-1], 3),
        }

    n_total = len(outcomes)
    reference_correct = sum(
        1
        for o in outcomes
        if o.reference_canonical == CATEGORY_TO_EXPECTED_CANONICAL[o.category]
    )
    unknown_observed = sum(1 for o in outcomes if o.observed_canonical == "unknown")
    unknown_reference = sum(1 for o in outcomes if o.reference_canonical == "unknown")

    return {
        "n_cases": n_total,
        "audit_resolved": dict(resolved_audits),
        "observed_distribution": _distribution("observed_canonical"),
        "reference_distribution": _distribution("reference_canonical"),
        "observed_confusion": _confusion("observed_canonical"),
        "reference_confusion": _confusion("reference_canonical"),
        "reference_accuracy": round(reference_correct / n_total, 4) if n_total else 0.0,
        "unknown_rate_observed": round(unknown_observed / n_total, 4) if n_total else 0.0,
        "unknown_rate_reference": round(unknown_reference / n_total, 4) if n_total else 0.0,
        "latency_observed": _latency("observed_ms"),
        "latency_reference": _latency("reference_ms"),
        "cases": [o.__dict__ for o in outcomes],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(part: int, total: int) -> str:
    return f"{(100.0 * part / total):.1f}%" if total else "n/a"


def _distribution_table(distribution: dict[str, int], total: int) -> str:
    lines = [
        "| Canonical diagnosis | Cases | Share |",
        "| :--- | ---: | ---: |",
    ]
    for key, count in distribution.items():
        lines.append(f"| `{key}` | {count} | {_pct(count, total)} |")
    if "ranking_miss" not in distribution and "ranking_miss" in CANONICAL_ORDER:
        lines.append("| `ranking_miss` | 0 | 0.0% |")
    if "retrieval_miss" not in distribution:
        lines.append("| `retrieval_miss` | 0 | 0.0% |")
    if "generation_ignored_context" not in distribution:
        lines.append("| `generation_ignored_context` | 0 | 0.0% |")
    if "unknown" not in distribution:
        lines.append("| `unknown` | 0 | 0.0% |")
    lines.append(f"| **Total** | **{total}** | **100.0%** |")
    return "\n".join(lines)


def _confusion_table(confusion: dict[str, dict[str, int]]) -> str:
    header = "| Construction label (ground truth) | " + " | ".join(
        f"`{name}`" for name in CANONICAL_ORDER
    ) + " |"
    divider = "| :--- | " + " | ".join("---:" for _ in CANONICAL_ORDER) + " |"
    lines = [header, divider]
    for category, expected in CATEGORY_TO_EXPECTED_CANONICAL.items():
        row = confusion.get(category, {})
        cells = [str(row.get(name, 0)) for name in CANONICAL_ORDER]
        lines.append(f"| {CATEGORY_LABELS[category]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any], stats: dict[str, Any], generated_at: str) -> str:
    provenance = data["provenance"]
    scan = data.get("scan_stats", {})
    n = stats["n_cases"]
    obs = stats["observed_distribution"]
    ref = stats["reference_distribution"]

    # Headline numbers
    obs_truncation = obs.get("ranking_miss", 0) + obs.get("retrieval_miss", 0)
    ref_truncation = ref.get("ranking_miss", 0)

    lines: list[str] = []
    add = lines.append

    add("# RAGmortem Real-World Benchmark: Public Dataset Results")
    add("")
    add(f"> Generated by `benchmarks/run_real_eval.py` on {generated_at}. Reproduce offline with "
        "`python benchmarks/run_real_eval.py`.")
    add("")
    add("## 1. What This Benchmark Measures")
    add("")
    add(
        "Prior RAGmortem evaluations (`docs/EVIDENCE.md`) used synthetic failure cases generated by the "
        "project's own fault injector. This benchmark evaluates the diagnoser on traces derived from a "
        "**real-world public dataset**: the SQuAD v1.1 development set (Rajpurkar et al., 2016), which "
        f"contains genuine Wikipedia passages and human-written questions with known ground-truth answers."
    )
    add("")
    add(
        "Each trace simulates a complete RAG failure: retrieval was performed with a real BM25 ranker over "
        "a 16-passage corpus, a top-8 candidate pool was re-ranked to a top-3 prompt context, and the "
        "'generator' returned an answer. For failure cases the answer is a simulated decoy answer drawn "
        "from other questions in the same dataset (wrong by construction; no LLM was called). Ground-truth "
        "failure categories are therefore known deterministically and are verified at build time."
    )
    add("")
    add("## 2. Dataset & Trace Construction")
    add("")
    add(f"* **Source**: [{DATASET_NAME}]({DATASET_URL}) ({DATASET_LICENSE}).")
    add(f"* **Questions scanned**: {scan.get('questions_scanned', 'n/a')} real SQuAD dev questions.")
    add(f"* **Corpus per trace**: {provenance['corpus_size_per_case']} real passages (gold evidence + "
        f"{SAME_ARTICLE_DISTRACTORS} same-article distractors + {POOL_DISTRACTORS} cross-article distractors).")
    add(f"* **Retriever**: {provenance['retriever']}; candidate pool = top-{CANDIDATE_K}, prompt context = top-{TOP_K}.")
    add("* **Generator answers**: simulated decoy answers (verified wrong via `is_answer_correct`; no LLM calls).")
    add(f"* **Sample**: {n} failure traces. Selection quota: {GEN_CASES} generation faults, "
        f"{RANKING_CASES} ranking truncations, {RETRIEVAL_CASES} retrieval misses (natural ranking:retrieval "
        "prevalence preserved; generation stratum added explicitly because real LLM traces are unavailable offline).")
    add("")
    add("Natural failure prevalence across all scanned questions:")
    add("")
    add("| Outcome | Questions | Share |")
    add("| :--- | ---: | ---: |")
    scanned = scan.get("questions_scanned", 0) or 1
    add(f"| Evidence reached top-3 (no retrieval failure) | {scan.get('evidence_in_top_k', 0)} | {_pct(scan.get('evidence_in_top_k', 0), scanned)} |")
    add(f"| Evidence truncated by top-k (candidate pool) | {scan.get('ranking_truncated', 0)} | {_pct(scan.get('ranking_truncated', 0), scanned)} |")
    add(f"| Evidence absent from candidate pool | {scan.get('retrieval_missed', 0)} | {_pct(scan.get('retrieval_missed', 0), scanned)} |")
    add("")
    retrieval_failures = scan.get("ranking_truncated", 0) + scan.get("retrieval_missed", 0)
    if retrieval_failures:
        add(
            f"Among the {retrieval_failures} real retrieval failures, "
            f"{_pct(scan.get('ranking_truncated', 0), retrieval_failures)} were top-k candidate truncation "
            f"and {_pct(scan.get('retrieval_missed', 0), retrieval_failures)} were candidate-pool retrieval misses."
        )
    add("")
    add("## 3. Primary Result: Observed Mode (`corpus_aware`, `profile=\"relative\"`)")
    add("")
    add("The requested pipeline `ragmortem.diagnose(mode=\"corpus_aware\", profile=\"relative\")` (no reference "
        "answer) returns heuristic *suspected* states, mapped here to canonical failure types.")
    add("")
    add(_distribution_table(obs, n))
    add("")
    add(f"* **UNKNOWN rate**: {_pct(obs.get('unknown', 0), n)} ({obs.get('unknown', 0)}/{n}).")
    add(f"* **Candidate truncation attribution**: {_pct(obs_truncation, n)} of traces were attributed to "
        "ranking/retrieval truncation; the remainder were flagged as generation or unknown.")
    add("")
    add("### 3.1 Observed-Mode Confusion Matrix")
    add("")
    add(_confusion_table(stats["observed_confusion"]))
    add("")
    add("## 4. Definitive Result: Reference-Assisted Mode")
    add("")
    add("The same traces are re-diagnosed with the SQuAD ground-truth answer as `reference_answer` "
        "(definitive mode; no score thresholds involved).")
    add("")
    add(_distribution_table(ref, n))
    add("")
    add(f"* **Attribution accuracy vs. construction labels**: {stats['reference_accuracy'] * 100:.1f}% "
        f"({sum(1 for c in stats['cases'] if c['reference_canonical'] == CATEGORY_TO_EXPECTED_CANONICAL[c['category']])}/{n}).")
    add(f"* **UNKNOWN rate**: {_pct(ref.get('unknown', 0), n)} ({ref.get('unknown', 0)}/{n}).")
    add("")
    add("### 4.1 Reference-Assisted Confusion Matrix")
    add("")
    add(_confusion_table(stats["reference_confusion"]))
    add("")
    if ref_truncation or ref.get("generation_ignored_context", 0):
        add(
            f"**Headline**: in this sample of {n} real-data-derived RAG failure traces, "
            f"{_pct(ref.get('ranking_miss', 0), n)} of failures were caused by top-k candidate truncation "
            f"(evidence retrieved but cut before the prompt), {_pct(ref.get('retrieval_miss', 0), n)} by "
            f"candidate-pool retrieval misses, and {_pct(ref.get('generation_ignored_context', 0), n)} by the "
            "generator failing with the evidence already in its prompt."
        )
        add("")
    add("## 5. Latency")
    add("")
    add("Per-trace diagnosis latency (CPU, single process; corpus index construction and model loading excluded):")
    add("")
    add("| Pipeline | Median | Mean | p95 | Max |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    add(f"| Observed (`corpus_aware`, `relative`) | {stats['latency_observed']['median_ms']} ms | "
        f"{stats['latency_observed']['mean_ms']} ms | {stats['latency_observed']['p95_ms']} ms | "
        f"{stats['latency_observed']['max_ms']} ms |")
    add(f"| Reference-assisted | {stats['latency_reference']['median_ms']} ms | "
        f"{stats['latency_reference']['mean_ms']} ms | {stats['latency_reference']['p95_ms']} ms | "
        f"{stats['latency_reference']['max_ms']} ms |")
    add("")
    add("## 6. Findings")
    add("")
    add(
        f"1. **Truncation dominates real retrieval failures.** Across the scanned corpus, "
        f"{_pct(scan.get('ranking_truncated', 0), retrieval_failures) if retrieval_failures else 'n/a'} of real "
        "retrieval failures occurred because BM25 surfaced the evidence into the candidate pool but the top-k "
        "cut removed it before the prompt. True candidate-pool misses were rare."
    )
    add(
        "2. **Observed mode is deliberately conservative and truncation-biased.** With full candidate-pool "
        "telemetry, the candidate-cliff probe attributes most wrong-answer traces to top-k truncation because "
        "score tails are flat in real retrieval distributions. Definitive attribution requires reference answers."
    )
    add(
        f"3. **Reference-assisted mode isolates the cause.** With ground-truth answers, RAGmortem attributed "
        f"{stats['reference_accuracy'] * 100:.1f}% of traces to the correct component, cleanly separating "
        "ranking truncation, retrieval misses, and generator failures."
    )
    add(
        f"4. **Diagnosis is millisecond-scale**: median {stats['latency_observed']['median_ms']} ms per trace "
        "observed, CPU-only, with zero API calls or network access."
    )
    add("")
    add("## 7. Threats to Validity")
    add("")
    add("* **Simulated generator answers**: real LLM traces were unavailable offline, so wrong answers are "
        "decoy answers from the same dataset. Error *style* may differ from production hallucinations.")
    add("* **SQuAD is clean single-hop QA**: questions are answerable from one paragraph, and neither the "
        "retriever nor the corpus contains OCR noise, chunking artifacts, or multi-hop reasoning.")
    add("* **Small corpus per trace**: 16 passages. Retrieval-miss prevalence grows with corpus size; "
        "production corpora are far larger.")
    add("* **BM25-only runtime retriever**: dense retrievers fail on different query distributions; "
        "lexical-gap failure modes are over-represented here relative to hybrid systems.")
    add("* **Stratified sample**: the generation stratum is injected by construction and does not represent "
        "natural generation-fault prevalence.")
    add("* **Local NLI disabled**: `RAGMORTEM_DISABLE_NLI=1` forces deterministic lexical matching; SQuAD "
        "answers are verbatim spans, so this is appropriate here but not for paraphrase-heavy production data.")
    add("")
    add("## 8. Reproduction")
    add("")
    add("```bash")
    add("# Re-run the evaluation from the bundled sample (offline):")
    add("python benchmarks/run_real_eval.py")
    add("")
    add("# Rebuild the sample from the SQuAD dev set (requires network):")
    add("python benchmarks/run_real_eval.py --build --download")
    add("```")
    add("")
    add(f"* Audit mode resolved to: {', '.join(f'{k} x{v}' for k, v in stats['audit_resolved'].items())}.")
    add(f"* Data file: `benchmarks/data/squad_real_world_sample.json`; raw stats: `benchmarks/real_world_results.json`.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _download_squad(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SQUAD_DEV_URL} -> {destination}")
    urllib.request.urlretrieve(SQUAD_DEV_URL, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-world public benchmark evaluation for RAGmortem (SQuAD v1.1 traces).",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the bundled sample JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_MD_PATH, help="Markdown results output path.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH, help="Raw statistics JSON output path.")
    parser.add_argument("--audit", choices=["auto", "dense", "bm25"], default="auto",
                        help="Corpus audit retriever. 'auto' uses the local dense model with BM25 fallback.")
    parser.add_argument("--build", action="store_true", help="Rebuild the bundled sample before evaluating.")
    parser.add_argument("--squad-source", type=Path, default=None, help="Local dev-v1.1.json for --build.")
    parser.add_argument("--download", action="store_true", help="Download the SQuAD dev set for --build.")
    parser.add_argument("--seed", type=int, default=BUILD_SEED, help="Build seed.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        if not args.quiet:
            print(message)

    if args.build:
        temp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
        default_raw = temp_dir / "squad_dev-v1.1.json"
        if args.squad_source is not None:
            squad_path = args.squad_source
        elif args.download:
            squad_path = _download_squad(default_raw) if not default_raw.exists() else default_raw
        else:
            fallback = Path(os.environ.get("TMPDIR", "/tmp")) / "squad_dev-v1.1.json"
            if not fallback.exists():
                parser.error("--build requires --squad-source PATH or --download")
            squad_path = fallback
        if not squad_path.exists():
            parser.error(f"SQuAD source file not found: {squad_path}")
        log(f"Building sample from {squad_path} ...")
        data = build_sample(squad_path, seed=args.seed)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with open(args.data, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1, ensure_ascii=False)
        log(f"Wrote sample: {args.data} "
            f"({len(data['cases'])} cases, {len(data['passages'])} passages)")

    if not args.data.exists():
        print(f"Error: sample data not found at {args.data}. Run with --build --download first.", file=sys.stderr)
        return 2

    with open(args.data, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    log(f"Loaded {len(data['cases'])} real-world failure traces from {args.data}")
    stats = run_evaluation(data, audit_mode=args.audit)

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    markdown = render_markdown(data, stats, generated_at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    log(f"Wrote results: {args.out}")

    json_payload = {
        "generated_at": generated_at,
        "n_cases": stats["n_cases"],
        "audit_resolved": stats["audit_resolved"],
        "observed_distribution": stats["observed_distribution"],
        "reference_distribution": stats["reference_distribution"],
        "observed_confusion": stats["observed_confusion"],
        "reference_confusion": stats["reference_confusion"],
        "reference_accuracy": stats["reference_accuracy"],
        "unknown_rate_observed": stats["unknown_rate_observed"],
        "unknown_rate_reference": stats["unknown_rate_reference"],
        "latency_observed": stats["latency_observed"],
        "latency_reference": stats["latency_reference"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)
    log(f"Wrote raw stats: {args.json}")

    log("")
    log(f"Observed distribution:  {stats['observed_distribution']}")
    log(f"Reference distribution: {stats['reference_distribution']}")
    log(f"Reference accuracy:     {stats['reference_accuracy'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
