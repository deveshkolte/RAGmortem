"""Benchmark harness evaluating competitor RAG evaluation tools against RAGmortem controlled failures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ragmortem.faults.eval import is_abstaining, is_answer_correct
from ragmortem.faults.inject import load_injected_faults
from ragmortem.faults.models import InjectedFaultCase



@dataclass
class MetricScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


@dataclass
class ToolBenchmarkResult:
    tool_name: str
    total_cases: int
    evaluated_cases: int
    unsupported_cases: int
    coverage_pct: float
    supported_accuracy_pct: float
    overall_accuracy_pct: float
    category_metrics: dict[str, MetricScores]
    confusion_matrix: dict[str, dict[str, int]]


def predict_ragas(case: InjectedFaultCase) -> str | None:
    """Simulate Ragas metric-based attribution without label leakage.

    Ragas computes context_recall (whether gold facts exist in final context)
    and faithfulness (whether generated answer is grounded in context).
    It only inspects the final retrieved context, not the broader candidate pool.
    """
    final_context_ids = case.fault.retrieved_ids
    gold_id = case.gold_chunk_id

    # 1. Context Recall check
    context_has_gold = bool(gold_id and (gold_id in final_context_ids))

    if not context_has_gold:
        if gold_id is not None:
            # Gold chunk missing from final context -> Ragas flags context_recall = 0
            # Because Ragas has no candidate pool visibility, it cannot distinguish
            # ranking truncation from retrieval omission.
            return "retrieval_miss"
        else:
            # Unanswerable query: Ragas has no abstention metric.
            # Evaluates faithfulness against irrelevant distractor context.
            return "UNSUPPORTED"

    # 2. Faithfulness / Answer Correctness check (gold chunk IS in final context)
    answer_correct = is_answer_correct(case.fault.answer, case.gold_answer)
    if not answer_correct:
        # Low faithfulness / answer incorrect despite context being present
        return "generation_ignored_context"

    return "no_failure"


def predict_pyvectorhound(case: InjectedFaultCase) -> str | None:
    """Simulate PyVectorHound component-level retrieval diagnostics.

    PyVectorHound evaluates vector recall in the candidate pool and MRR.
    It does NOT evaluate LLM generation or abstention.
    """
    gold_id = case.gold_chunk_id
    if not gold_id:
        return "UNSUPPORTED"

    candidates = case.fault.candidate_retrieved_ids or case.fault.retrieved_ids
    cand_rank = case.fault.candidate_gold_rank

    # 1. Retrieval Miss check (not in candidate pool)
    if gold_id not in candidates:
        return "retrieval_miss"

    # 2. Ranking Miss check (in candidates, but ranked > top_k)
    top_k = len(case.fault.retrieved_ids)
    if cand_rank is not None and cand_rank > top_k:
        return "ranking_miss"

    # PyVectorHound explicitly does not evaluate generation
    return "UNSUPPORTED"


def predict_deepdiag(case: InjectedFaultCase) -> str | None:
    """Simulate DeepDiag 2-bucket binary diagnosis (Retrieval-bound vs Generation-bound)."""
    gold_id = case.gold_chunk_id
    if not gold_id:
        return "UNSUPPORTED"

    final_context_ids = case.fault.retrieved_ids
    context_has_gold = gold_id in final_context_ids

    if not context_has_gold:
        # Coarse bucket: Retrieval-bound (cannot separate ranking from retrieval)
        return "retrieval_miss"

    answer_correct = is_answer_correct(case.fault.answer, case.gold_answer)
    if not answer_correct:
        return "generation_ignored_context"

    return "no_failure"


def predict_rag_oracle(case: InjectedFaultCase) -> str | None:
    """Simulate Oracle RAG ablation protocol (gold context injection)."""
    gold_id = case.gold_chunk_id
    if not gold_id:
        return "UNSUPPORTED"

    final_context_ids = case.fault.retrieved_ids
    context_has_gold = gold_id in final_context_ids

    # Oracle test: If gold chunk is missing from standard context,
    # but injecting gold chunk yields correct answer -> Retrieval-bound
    if not context_has_gold:
        return "retrieval_miss"

    # If gold chunk was already in context and answer failed -> Generation-bound
    return "generation_ignored_context"


COMPETITOR_PREDICTORS: dict[str, Callable[[InjectedFaultCase], str | None]] = {
    "Ragas (Metric-Based)": predict_ragas,
    "PyVectorHound (Retrieval-Only)": predict_pyvectorhound,
    "DeepDiag / RAGDiag (Binary)": predict_deepdiag,
    "RAG-Oracle (Ablation Baseline)": predict_rag_oracle,
}


def run_benchmark(
    cases: list[InjectedFaultCase],
) -> dict[str, ToolBenchmarkResult]:
    """Run benchmark against all competitors and compile performance statistics."""
    categories = [
        "retrieval_miss",
        "ranking_miss",
        "generation_ignored_context",
        "should_abstain",
    ]
    results: dict[str, ToolBenchmarkResult] = {}

    for tool_name, predictor in COMPETITOR_PREDICTORS.items():
        evaluated_count = 0
        unsupported_count = 0
        correct_on_supported = 0
        total_correct = 0

        cat_metrics = {cat: MetricScores() for cat in categories}
        confusion_matrix: dict[str, dict[str, int]] = {
            gt: {pred: 0 for pred in categories + ["UNSUPPORTED", "other"]}
            for gt in categories
        }

        for case in cases:
            gt_label = case.fault_type
            pred_label = predictor(case)

            if pred_label == "UNSUPPORTED" or pred_label is None:
                unsupported_count += 1
                confusion_matrix[gt_label]["UNSUPPORTED"] += 1
                # Counts as False Negative for the true category
                cat_metrics[gt_label].fn += 1
                continue

            evaluated_count += 1
            if pred_label in confusion_matrix[gt_label]:
                confusion_matrix[gt_label][pred_label] += 1
            else:
                confusion_matrix[gt_label]["other"] += 1

            if pred_label == gt_label:
                correct_on_supported += 1
                total_correct += 1
                cat_metrics[gt_label].tp += 1
            else:
                cat_metrics[gt_label].fn += 1
                if pred_label in cat_metrics:
                    cat_metrics[pred_label].fp += 1

        total_cases = len(cases)
        coverage_pct = (evaluated_count / total_cases) * 100
        supported_acc = (correct_on_supported / evaluated_count * 100) if evaluated_count > 0 else 0.0
        overall_acc = (total_correct / total_cases) * 100

        results[tool_name] = ToolBenchmarkResult(
            tool_name=tool_name,
            total_cases=total_cases,
            evaluated_cases=evaluated_count,
            unsupported_cases=unsupported_count,
            coverage_pct=round(coverage_pct, 1),
            supported_accuracy_pct=round(supported_acc, 1),
            overall_accuracy_pct=round(overall_acc, 1),
            category_metrics=cat_metrics,
            confusion_matrix=confusion_matrix,
        )

    return results


def main() -> int:
    faults_path = Path("evals/injected_faults.jsonl")
    if not faults_path.exists():
        print(f"Error: Faults file '{faults_path}' does not exist.")
        return 1

    cases = load_injected_faults(faults_path)
    print(f"Loaded {len(cases)} validated failure cases from {faults_path}")

    benchmark_results = run_benchmark(cases)

    # Save results to JSON
    out_json = Path("benchmarks/results.json")
    json_payload = {}
    for tool_name, res in benchmark_results.items():
        json_payload[tool_name] = {
            "total_cases": res.total_cases,
            "evaluated_cases": res.evaluated_cases,
            "unsupported_cases": res.unsupported_cases,
            "coverage_pct": res.coverage_pct,
            "supported_accuracy_pct": res.supported_accuracy_pct,
            "overall_accuracy_pct": res.overall_accuracy_pct,
            "category_metrics": {
                cat: {
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "precision": round(m.precision, 3),
                    "recall": round(m.recall, 3),
                    "f1": round(m.f1, 3),
                }
                for cat, m in res.category_metrics.items()
            },
            "confusion_matrix": res.confusion_matrix,
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Tool':<32} {'Coverage':<12} {'Supported Acc':<16} {'Overall Acc':<14}")
    print("-" * 80)
    for tool_name, res in benchmark_results.items():
        cov_str = f"{res.evaluated_cases}/{res.total_cases} ({res.coverage_pct}%)"
        supp_str = f"{res.supported_accuracy_pct}%"
        ov_str = f"{res.overall_accuracy_pct}%"
        print(f"{tool_name:<32} {cov_str:<12} {supp_str:<16} {ov_str:<14}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
