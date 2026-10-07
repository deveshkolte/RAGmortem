"""Evaluation harness for Day 8 cross-domain abstention holdout (FinDebt domain).

Evaluates across development, validation, and frozen final_holdout splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.diagnose import RAGDiagnoser, ObservedExecution
from ragmortem.taxonomy import FailureType

FINANCE_DOCS_DIR = Path("examples/finance_rag/documents")
HOLDOUT_DATA_PATH = Path("evals/abstention_holdout.jsonl")
REPORT_PATH = Path("evals/abstention_holdout_results.md")

ALL_CLASSES = [
    "retrieval_miss",
    "ranking_miss",
    "generation_ignored_context",
    "should_abstain",
    "unknown",
]


def evaluate_split(
    cases: list[dict[str, Any]],
    diagnoser: RAGDiagnoser,
    mode_name: str,
    split_name: str,
) -> dict[str, Any]:
    cm: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in ALL_CLASSES} for gt in ALL_CLASSES}
    correct_all = 0
    correct_resolved = 0
    total = len(cases)
    resolved_total = 0

    scores_correct: list[float] = []
    scores_incorrect: list[float] = []
    scores_unknown: list[float] = []

    records: list[dict[str, Any]] = []

    for c in cases:
        fault_exec = c["fault"]
        obs_trace = ObservedExecution(
            question=c["question"],
            retrieved_chunk_ids=fault_exec["retrieved_ids"],
            generated_answer=fault_exec["answer"],
            scores=fault_exec.get("scores"),
            case_id=c.get("case_id"),
        )
        diag_res = diagnoser.diagnose(obs_trace, mode=mode_name)

        gt_str = c["fault_type"]
        pred_canonical = diag_res.diagnosis.canonical.value
        pred_str = pred_canonical if pred_canonical in ALL_CLASSES else "unknown"

        is_correct = (pred_str == gt_str)
        if is_correct:
            correct_all += 1
            if pred_str != "unknown":
                scores_correct.append(diag_res.evidence_score)
            else:
                scores_unknown.append(diag_res.evidence_score)
        else:
            if pred_str != "unknown":
                scores_incorrect.append(diag_res.evidence_score)
            else:
                scores_unknown.append(diag_res.evidence_score)

        if gt_str != "unknown":
            resolved_total += 1
            if is_correct:
                correct_resolved += 1

        cm[gt_str][pred_str] += 1

        records.append({
            "case_id": c.get("case_id"),
            "question_id": c.get("question_id"),
            "ground_truth": gt_str,
            "predicted": pred_str,
            "is_correct": is_correct,
            "evidence_score": diag_res.evidence_score,
            "reasons": diag_res.reasons,
            "limitations": diag_res.limitations,
            "signals": diag_res.signals,
        })

    unknown_diagnosed = sum(cm[gt]["unknown"] for gt in ALL_CLASSES)
    coverage = (total - unknown_diagnosed) / total if total > 0 else 0.0
    accuracy_all = correct_all / total if total > 0 else 0.0
    accuracy_resolved = correct_resolved / resolved_total if resolved_total > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_list = []
    for cat in ALL_CLASSES:
        tp = cm[cat][cat]
        fp = sum(cm[gt][cat] for gt in ALL_CLASSES if gt != cat)
        fn = sum(cm[cat][p] for p in ALL_CLASSES if p != cat)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "support": tp + fn,
        }
        if (tp + fn) > 0:
            f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    avg_correct = sum(scores_correct) / len(scores_correct) if scores_correct else 0.0
    avg_incorrect = sum(scores_incorrect) / len(scores_incorrect) if scores_incorrect else 0.0
    avg_unknown = sum(scores_unknown) / len(scores_unknown) if scores_unknown else 0.0

    # Specific confusions
    conf_abstain_to_retrieval = cm["should_abstain"]["retrieval_miss"]
    conf_abstain_to_unknown = cm["should_abstain"]["unknown"]
    conf_retrieval_to_abstain = cm["retrieval_miss"]["should_abstain"]
    conf_retrieval_to_unknown = cm["retrieval_miss"]["unknown"]

    return {
        "split": split_name,
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
        "conf_abstain_to_retrieval": conf_abstain_to_retrieval,
        "conf_abstain_to_unknown": conf_abstain_to_unknown,
        "conf_retrieval_to_abstain": conf_retrieval_to_abstain,
        "conf_retrieval_to_unknown": conf_retrieval_to_unknown,
        "avg_correct": avg_correct,
        "avg_incorrect": avg_incorrect,
        "avg_unknown": avg_unknown,
        "records": records,
    }


def format_split_markdown(res: dict[str, Any]) -> str:
    lines = []
    lines.append(f"### Split: `{res['split'].upper()}` ({res['mode']})\n")
    lines.append(f"- **Total Cases**: {res['total']}")
    lines.append(f"- **Overall Accuracy (All)**: {res['correct_all']}/{res['total']} ({res['accuracy_all']:.1%})")
    lines.append(f"- **Accuracy on Resolved Cases**: {res['correct_resolved']}/{res['resolved_total']} ({res['accuracy_resolved']:.1%})")
    lines.append(f"- **Coverage**: {res['coverage']:.1%}")
    lines.append(f"- **Macro F1**: {res['macro_f1']:.4f}")
    lines.append(f"- **Average Evidence Score (Correct)**: {res['avg_correct']:.3f}")
    lines.append(f"- **Average Evidence Score (Incorrect)**: {res['avg_incorrect']:.3f}")
    lines.append(f"- **Average Evidence Score (Unknown)**: {res['avg_unknown']:.3f}\n")

    lines.append("**Key Error Confusions:**")
    lines.append(f"- `should_abstain -> retrieval_miss`: {res['conf_abstain_to_retrieval']}")
    lines.append(f"- `should_abstain -> unknown`: {res['conf_abstain_to_unknown']}")
    lines.append(f"- `retrieval_miss -> should_abstain`: {res['conf_retrieval_to_abstain']}")
    lines.append(f"- `retrieval_miss -> unknown`: {res['conf_retrieval_to_unknown']}\n")

    lines.append("| Category | Precision | Recall | F1-Score | Support |")
    lines.append("|---|---|---|---|---|")
    for cat in ["retrieval_miss", "should_abstain", "unknown"]:
        m = res["per_class"][cat]
        lines.append(f"| `{cat}` | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {int(m['support'])} |")

    lines.append("\n**Confusion Matrix** (Rows = Ground Truth, Columns = Predicted):\n")
    header = "| Ground Truth \\ Predicted | " + " | ".join(f"`{c}`" for c in ALL_CLASSES) + " |"
    sep = "|---|" + "|".join("---" for _ in ALL_CLASSES) + "|"
    lines.append(header)
    lines.append(sep)
    for gt in ALL_CLASSES:
        if sum(res["confusion_matrix"][gt].values()) > 0:
            row = f"| `{gt}` | " + " | ".join(str(res["confusion_matrix"][gt][pred]) for pred in ALL_CLASSES) + " |"
            lines.append(row)
    lines.append("\n")
    return "\n".join(lines)


def main():
    with open(HOLDOUT_DATA_PATH) as f:
        all_cases = [json.loads(line) for line in f]

    dev_cases = [c for c in all_cases if c.get("split") == "development"]
    val_cases = [c for c in all_cases if c.get("split") == "validation"]
    holdout_cases = [c for c in all_cases if c.get("split") == "final_holdout"]

    print(f"Loaded {len(all_cases)} total cases: dev={len(dev_cases)}, val={len(val_cases)}, holdout={len(holdout_cases)}")

    app = ReferenceRagApp(corpus_dir=FINANCE_DOCS_DIR, mock_mode=True)
    diag_corpus = RAGDiagnoser(app=app)
    diag_trace = RAGDiagnoser(app=None)

    # 1. Development split
    res_dev_corpus = evaluate_split(dev_cases, diag_corpus, "corpus_aware", "development")
    res_dev_trace = evaluate_split(dev_cases, diag_trace, "trace_only", "development")

    # 2. Validation split
    res_val_corpus = evaluate_split(val_cases, diag_corpus, "corpus_aware", "validation")
    res_val_trace = evaluate_split(val_cases, diag_trace, "trace_only", "validation")

    # 3. Final Holdout split (evaluated once on frozen code)
    res_holdout_corpus = evaluate_split(holdout_cases, diag_corpus, "corpus_aware", "final_holdout")
    res_holdout_trace = evaluate_split(holdout_cases, diag_trace, "trace_only", "final_holdout")

    # Print summary to terminal
    print("\n" + "=" * 70)
    print("DAY 8 FINDEBT EVALUATION RESULTS")
    print("=" * 70)
    for res in [res_dev_corpus, res_val_corpus, res_holdout_corpus]:
        print(f"[{res['split'].upper()}] Accuracy={res['accuracy_all']:.1%} ({res['correct_all']}/{res['total']}) | "
              f"Resolved Acc={res['accuracy_resolved']:.1%} | Coverage={res['coverage']:.1%} | Macro F1={res['macro_f1']:.4f}")
        print(f"  Abstain->Retrieval Confusions: {res['conf_abstain_to_retrieval']} | Abstain->Unknown: {res['conf_abstain_to_unknown']}")

    print("\n[FINAL HOLDOUT TRACE-ONLY]")
    print(f"Accuracy={res_holdout_trace['accuracy_all']:.1%} | Unknown Recall={res_holdout_trace['per_class']['unknown']['recall']:.1%}")
    print("=" * 70)

    # Write Markdown Report
    report = f"""# Day 8 Cross-Domain Abstention Holdout Report (FinDebt Domain)

Evaluation on the new **FinDebt (Syndicated Credit Agreement)** cross-domain dataset.

## Split Protocol
- **Development Split**: 20 cases (5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- **Validation Split**: 20 cases (5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- **Final Holdout Split**: 40 cases (10 retrieval, 10 abstention, 10 terminology trap, 10 ambiguous) evaluated on frozen code.

---

## 1. Development Split Results
{format_split_markdown(res_dev_corpus)}

---

## 2. Validation Split Results
{format_split_markdown(res_val_corpus)}

---

## 3. Final Holdout Split Results (Corpus-Aware)
{format_split_markdown(res_holdout_corpus)}

---

## 4. Final Holdout Split Results (Trace-Only)
{format_split_markdown(res_holdout_trace)}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote full report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
