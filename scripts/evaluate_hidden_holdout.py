"""Evaluate frozen uncertainty-aware diagnoser on the fresh hidden holdout dataset (BioTrial domain).

Runs both Corpus-Aware and Trace-Only evaluation modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.diagnose import RAGDiagnoser, ObservedExecution
from ragmortem.faults.inject import load_injected_faults
from ragmortem.faults.models import InjectedFaultCase

HIDDEN_DOCS_DIR = Path("examples/hidden_holdout_rag/documents")
HIDDEN_FAULTS_PATH = Path("evals/hidden_holdout_faults.jsonl")
CORPUS_AWARE_OUT = Path("evals/hidden_holdout_observed_diagnoses.jsonl")
TRACE_ONLY_OUT = Path("evals/hidden_holdout_trace_only_diagnoses.jsonl")
REPORT_OUT = Path("evals/hidden_holdout_results.md")

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
    cm: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in ALL_CLASSES} for gt in ALL_CLASSES}

    correct_all = 0
    correct_resolved = 0
    total = len(cases)
    resolved_total = 0

    scores_correct: list[float] = []
    scores_incorrect: list[float] = []

    for case in cases:
        obs_trace = ObservedExecution.from_injected_case(case)
        diag_res = diagnoser.diagnose(obs_trace, mode=mode_name)

        gt_str = case.fault_type
        pred_canonical = diag_res.diagnosis.canonical.value
        if pred_canonical not in ALL_CLASSES:
            pred_str = "unknown"
        else:
            pred_str = pred_canonical

        is_correct = (pred_str == gt_str)
        if is_correct:
            correct_all += 1
            if pred_str != "unknown":
                scores_correct.append(diag_res.evidence_score)
        else:
            if pred_str != "unknown":
                scores_incorrect.append(diag_res.evidence_score)

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
            "evidence_score": diag_res.evidence_score,
            "confidence": diag_res.confidence,
            "telemetry_completeness": diag_res.telemetry_completeness,
            "reasons": diag_res.reasons,
            "limitations": diag_res.limitations,
            "evidence": diag_res.evidence,
            "signals": diag_res.signals,
            "mode": mode_name,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

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

    avg_score_correct = sum(scores_correct) / len(scores_correct) if scores_correct else 0.0
    avg_score_incorrect = sum(scores_incorrect) / len(scores_incorrect) if scores_incorrect else 0.0

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
        "avg_score_correct": avg_score_correct,
        "avg_score_incorrect": avg_score_incorrect,
        "records": records,
    }


def format_markdown_table(res: dict[str, Any]) -> str:
    lines = []
    lines.append(f"### Mode: `{res['mode']}`\n")
    lines.append(f"- **Total Cases**: {res['total']}")
    lines.append(f"- **Overall Accuracy (All cases including UNKNOWN)**: {res['correct_all']}/{res['total']} ({res['accuracy_all']:.1%})")
    lines.append(f"- **Accuracy on Resolved Cases**: {res['correct_resolved']}/{res['resolved_total']} ({res['accuracy_resolved']:.1%})")
    lines.append(f"- **Coverage**: {res['coverage']:.1%}")
    lines.append(f"- **Macro F1**: {res['macro_f1']:.4f}")
    lines.append(f"- **Average Evidence Score (Correct Resolved)**: {res['avg_score_correct']:.3f}")
    lines.append(f"- **Average Evidence Score (Incorrect Resolved)**: {res['avg_score_incorrect']:.3f}\n")

    lines.append("| Category | Precision | Recall | F1-Score | Support |")
    lines.append("|---|---|---|---|---|")
    for c in ALL_CLASSES:
        m = res["per_class"][c]
        lines.append(f"| `{c}` | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {int(m['support'])} |")

    lines.append("\n**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):\n")
    header = "| Ground Truth \\ Predicted | " + " | ".join(f"`{c}`" for c in ALL_CLASSES) + " |"
    sep = "|---|" + "|".join("---" for _ in ALL_CLASSES) + "|"
    lines.append(header)
    lines.append(sep)
    for gt in ALL_CLASSES:
        row = f"| `{gt}` | " + " | ".join(str(res["confusion_matrix"][gt][pred]) for pred in ALL_CLASSES) + " |"
        lines.append(row)
    lines.append("\n")
    return "\n".join(lines)


def main():
    cases = load_injected_faults(HIDDEN_FAULTS_PATH)
    print(f"Loaded {len(cases)} cases from {HIDDEN_FAULTS_PATH}")

    # 1. Corpus-Aware Mode
    app = ReferenceRagApp(corpus_dir=HIDDEN_DOCS_DIR, mock_mode=True)
    diag_corpus = RAGDiagnoser(app=app)
    res_corpus = run_evaluation(cases, diag_corpus, CORPUS_AWARE_OUT, mode_name="corpus_aware")

    # 2. Trace-Only Mode
    diag_trace = RAGDiagnoser()
    diag_trace.app = None
    res_trace = run_evaluation(cases, diag_trace, TRACE_ONLY_OUT, mode_name="trace_only")

    # Print summaries
    for res in [res_corpus, res_trace]:
        print("=" * 70)
        print(f"RESULTS: {res['mode'].upper()}")
        print(f"Accuracy (All):      {res['accuracy_all']:.1%} ({res['correct_all']}/{res['total']})")
        print(f"Accuracy (Resolved): {res['accuracy_resolved']:.1%} ({res['correct_resolved']}/{res['resolved_total']})")
        print(f"Coverage:            {res['coverage']:.1%}")
        print(f"Macro F1:            {res['macro_f1']:.4f}")
        print(f"Unknown Prec/Rec/F1: {res['per_class']['unknown']['precision']:.3f} / {res['per_class']['unknown']['recall']:.3f} / {res['per_class']['unknown']['f1']:.3f}")
        print("=" * 70)

    # Write Markdown Report
    report = f"""# Day 7 Hidden Holdout Evaluation Report (BioTrial Domain)

This report presents the frozen evaluation results of RAGmortem's uncertainty-aware diagnoser on the completely unseen **BioTrial (Clinical Trials & FDA Regulations)** holdout dataset (`evals/hidden_holdout_faults.jsonl`).

## Evaluation Design
- **Corpus**: 5 clinical research documents, 40 distinct chunks.
- **Test cases**: 100 injected failure cases across 5 classes (20 retrieval, 20 ranking, 20 generation, 20 abstention, 20 unknown).
- **Diagnoser**: Frozen uncertainty-aware implementation with deterministic evidence scoring and relative score margins.

---

## 1. Corpus-Aware Results
{format_markdown_table(res_corpus)}

---

## 2. Trace-Only Results
{format_markdown_table(res_trace)}

---

## 3. Decision Assessment
- Target Excellent: >= 90% accuracy, >= 80% unknown recall, >= 80% coverage.
- Target Promising: 70–89% accuracy.
- Target Weak: < 70% accuracy.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"\nWrote full report to {REPORT_OUT}")


if __name__ == "__main__":
    main()
