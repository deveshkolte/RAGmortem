"""Evaluate frozen observational diagnoser on holdout dataset in both Trace+Corpus and Trace-Only modes."""

import json
from pathlib import Path
from typing import Any, Sequence
from collections import defaultdict

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.diagnose import RAGDiagnoser, ObservedExecution
from ragmortem.faults.inject import load_injected_faults
from ragmortem.faults.models import InjectedFaultCase
from ragmortem.taxonomy import FailureType

HOLDOUT_DOCS_DIR = Path("examples/holdout_rag/documents")
FAULTS_PATH = Path("evals/holdout_faults.jsonl")
OBSERVED_OUT = Path("evals/holdout_observed_diagnoses.jsonl")
TRACE_ONLY_OUT = Path("evals/holdout_trace_only_diagnoses.jsonl")

ALL_CLASSES = [
    "retrieval_miss",
    "ranking_miss",
    "generation_ignored_context",
    "should_abstain",
    "unknown",
]

def run_evaluation(
    cases: list[InjectedFaultCase],
    diagnoser: RAGDiagnoser,
    output_path: Path,
    mode_name: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    # 5x5 confusion matrix
    cm: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in ALL_CLASSES} for gt in ALL_CLASSES}

    correct_all = 0
    correct_resolved = 0
    total = len(cases)
    resolved_total = 0

    for case in cases:
        obs_trace = ObservedExecution.from_injected_case(case)
        diag_res = diagnoser.diagnose_observed(obs_trace)

        gt_str = case.fault_type
        # Map canonical diagnosis to failure type value
        pred_canonical = diag_res.diagnosis.canonical.value
        if pred_canonical not in ALL_CLASSES:
            pred_str = "unknown"
        else:
            pred_str = pred_canonical

        is_correct = (pred_str == gt_str)
        if is_correct:
            correct_all += 1

        if gt_str != "unknown":
            resolved_total += 1
            if is_correct:
                correct_resolved += 1

        cm[gt_str][pred_str] += 1

        records.append({
            "case_id": case.case_id,
            "question_id": case.question_id,
            "ground_truth": gt_str,
            "predicted_diagnosis": pred_str,
            "raw_diagnosis": diag_res.diagnosis.value,
            "is_correct": is_correct,
            "is_suspected": diag_res.is_suspected,
            "confidence": diag_res.confidence,
            "evidence": diag_res.evidence,
            "signals": diag_res.signals,
            "candidate_count": diag_res.candidate_count,
            "mode": mode_name,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Metrics computation
    unknown_diagnosed = sum(cm[gt]["unknown"] for gt in ALL_CLASSES)
    coverage = (total - unknown_diagnosed) / total if total > 0 else 0.0
    accuracy_all = correct_all / total if total > 0 else 0.0
    accuracy_resolved = correct_resolved / resolved_total if resolved_total > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_list = []
    for c in ALL_CLASSES:
        tp = cm[c][c]
        fp = sum(cm[gt][c] for gt in ALL_CLASSES if gt != c)
        fn = sum(cm[c][p] for p in ALL_CLASSES if p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)

    return {
        "mode": mode_name,
        "total": total,
        "correct_all": correct_all,
        "accuracy_all": accuracy_all,
        "resolved_total": resolved_total,
        "correct_resolved": correct_resolved,
        "accuracy_resolved": accuracy_resolved,
        "coverage": coverage,
        "unknown_diagnosed": unknown_diagnosed,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "records": records,
    }

def print_results(res: dict[str, Any]):
    print("=" * 75)
    print(f"EVALUATION RESULTS: {res['mode'].upper()}")
    print("=" * 75)
    print(f"Total Cases:                 {res['total']}")
    print(f"Correct Diagnoses (All):     {res['correct_all']} ({res['accuracy_all']:.1%})")
    print(f"Resolved Cases:              {res['resolved_total']}")
    print(f"Accuracy on Resolved Cases:  {res['correct_resolved']}/{res['resolved_total']} ({res['accuracy_resolved']:.1%})")
    print(f"Diagnoser Coverage:          {res['coverage']:.1%}")
    print(f"Unknown Diagnoses Made:      {res['unknown_diagnosed']}")
    print(f"Macro F1:                    {res['macro_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    print(f"  {'Class':<28} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Support'}")
    print("  " + "-" * 62)
    for c in ALL_CLASSES:
        m = res["per_class"][c]
        print(f"  {c:<28} | {m['precision']:<7.3f} | {m['recall']:<7.3f} | {m['f1']:<7.3f} | {int(m['support'])}")

    print("\nConfusion Matrix (Rows=Ground Truth, Columns=Predicted):")
    header = f"  {'Ground Truth':<28} | " + " | ".join(f"{c[:10]:>10}" for c in ALL_CLASSES)
    print(header)
    print("  " + "-" * len(header))
    for gt in ALL_CLASSES:
        row = f"  {gt:<28} | " + " | ".join(f"{res['confusion_matrix'][gt][pred]:>10}" for pred in ALL_CLASSES)
        print(row)
    print("=" * 75)
    print()

def main():
    cases = load_injected_faults(FAULTS_PATH)
    print(f"Loaded {len(cases)} cases from {FAULTS_PATH}")

    # Experiment B: Trace + Corpus
    app = ReferenceRagApp(corpus_dir=HOLDOUT_DOCS_DIR, mock_mode=True)
    diag_corpus = RAGDiagnoser(app=app)
    res_corpus = run_evaluation(cases, diag_corpus, OBSERVED_OUT, mode_name="trace_plus_corpus")
    print_results(res_corpus)

    # Experiment A: Trace-only (explicitly None so it does not fallback to default corpus)
    diag_trace = RAGDiagnoser()
    diag_trace.app = None
    res_trace = run_evaluation(cases, diag_trace, TRACE_ONLY_OUT, mode_name="trace_only")
    print_results(res_trace)

if __name__ == "__main__":
    main()
