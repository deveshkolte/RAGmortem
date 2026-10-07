"""Minimal command-line interface for RAGmortem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Ensure workspace root is accessible for loading examples
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ragmortem.dataset import load_questions, validate_dataset
from examples.reference_rag.app import ReferenceRagApp, load_corpus_from_directory



def cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate questions against the reference corpus."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)

    print(f"Validating dataset: {questions_path}")
    print(f"Against corpus:     {corpus_dir}")

    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    chunks = load_corpus_from_directory(corpus_dir)
    print(f"Loaded {len(chunks)} chunks from corpus.")

    errors = validate_dataset(questions_path, chunks)
    if errors:
        print(f"\nFound {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1

    questions = load_questions(questions_path)
    ans_count = sum(1 for q in questions if q.get("type") == "answerable")
    unans_count = sum(1 for q in questions if q.get("type") == "unanswerable")

    print("\nDataset validation PASSED!")
    print(f"  Total questions: {len(questions)}")
    print(f"  Answerable:      {ans_count}")
    print(f"  Unanswerable:    {unans_count}")
    return 0


def cmd_run_reference(args: argparse.Namespace) -> int:
    """Run reference RAG pipeline against the question dataset."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)

    if not questions_path.exists():
        print(f"Error: Questions file '{questions_path}' does not exist.", file=sys.stderr)
        return 1

    questions = load_questions(questions_path)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    print("=" * 70)
    print("RAGmortem Reference Evaluation Runner")
    print(f"Corpus:        {corpus_dir}")
    print(f"Questions:     {len(questions)} items from {questions_path}")
    print(f"Top-K:         {args.top_k}")
    print(f"Mock Mode:     {args.mock}")
    print("=" * 70)

    app = ReferenceRagApp(
        corpus_dir=corpus_dir,
        top_k=args.top_k,
        mock_mode=args.mock,
    )

    answerable_count = 0
    top1_hits = 0
    topk_hits = 0
    cache_hits = 0

    for i, item in enumerate(questions, start=1):
        q_id = item["id"]
        q_text = item["question"]
        q_type = item.get("type", "answerable")
        gold_id = item.get("gold_chunk_id")
        gold_answer = item.get("gold_answer", "")

        result = app.query(q_text)
        retrieved_ids = result.retrieved_ids
        rank = result.gold_rank(gold_id) if gold_id else None

        if result.metadata.get("cached"):
            cache_hits += 1

        print(f"\n[{i}/{len(questions)}] Question ID: {q_id} ({q_type})")
        print(f"  Query:         {q_text}")
        print(f"  Retrieved IDs: {retrieved_ids}")
        if result.scores:
            formatted_scores = [f"{s:.3f}" for s in result.scores]
            print(f"  Scores:        {formatted_scores}")

        if q_type == "answerable":
            answerable_count += 1
            print(f"  Gold Chunk ID: {gold_id}")
            if rank is not None:
                print(f"  Gold Rank:     #{rank}")
                topk_hits += 1
                if rank == 1:
                    top1_hits += 1
            else:
                print("  Gold Rank:     MISS (not in retrieved top-k)")
        else:
            print("  Gold Chunk ID: N/A (unanswerable)")

        truncated_answer = (result.answer[:120] + "...") if len(result.answer) > 120 else result.answer
        print(f"  Answer:        {truncated_answer}")
        print(f"  Expected:      {gold_answer}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total Processed:  {len(questions)}")
    if answerable_count > 0:
        print(f"  Answerable:       {answerable_count}")
        print(f"  Gold in Top-1:    {top1_hits}/{answerable_count} ({top1_hits / answerable_count:.1%})")
        print(f"  Gold in Top-{args.top_k}:    {topk_hits}/{answerable_count} ({topk_hits / answerable_count:.1%})")
    print(f"  Cache Hits:       {cache_hits}/{len(questions)}")
    print("=" * 70)

    return 0


def cmd_generate_faults(args: argparse.Namespace) -> int:
    """Generate controlled RAG failure cases and output to JSONL dataset."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)
    output_path = Path(args.output)

    if not questions_path.exists():
        print(f"Error: Questions file '{questions_path}' does not exist.", file=sys.stderr)
        return 1
    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    from ragmortem.faults.inject import generate_fault_dataset, save_injected_faults

    print("Generating controlled failures...")
    questions = load_questions(questions_path)
    app = ReferenceRagApp(corpus_dir=corpus_dir, top_k=args.top_k, mock_mode=True)

    cases = generate_fault_dataset(
        questions=questions,
        app=app,
        seed=args.seed,
        top_k=args.top_k,
    )

    save_injected_faults(cases, output_path)

    # Summarize fault counts
    counts: dict[str, int] = {}
    valid_count = 0
    invalid_count = 0

    for c in cases:
        counts[c.fault_type] = counts.get(c.fault_type, 0) + 1
        if c.validated:
            valid_count += 1
        else:
            invalid_count += 1

    print()
    for ftype, cnt in sorted(counts.items()):
        print(f"  {ftype:<24} {cnt}")
    print()
    print(f"  TOTAL                    {len(cases)}")
    print()
    print("Validation:")
    print(f"  {valid_count}/{len(cases)} valid")
    print(f"  {invalid_count} invalid")
    print(f"\nSaved to: {output_path}")

    return 0 if invalid_count == 0 else 1


def cmd_validate_faults(args: argparse.Namespace) -> int:
    """Validate an existing injected fault dataset against the corpus."""
    faults_path = Path(args.faults)
    corpus_dir = Path(args.corpus)

    if not faults_path.exists():
        print(f"Error: Faults file '{faults_path}' does not exist.", file=sys.stderr)
        return 1
    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    from ragmortem.faults.eval import validate_fault_case
    from ragmortem.faults.inject import load_injected_faults

    chunks = load_corpus_from_directory(corpus_dir)
    chunk_ids = {c.id for c in chunks}
    cases = load_injected_faults(faults_path)

    print(f"Validating {len(cases)} fault cases from {faults_path}...")

    valid_count = 0
    invalid_count = 0
    errors: list[str] = []

    for c in cases:
        is_valid, reason = validate_fault_case(c, chunk_ids, top_k=args.top_k)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors.append(f"Case '{c.case_id}' ({c.fault_type}): {reason}")

    print(f"Validation: {valid_count}/{len(cases)} valid, {invalid_count} invalid")
    if errors:
        print("\nErrors detected:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  [ERROR] {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.", file=sys.stderr)
        return 1

    print("All injected fault cases successfully validated!")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI Entrypoint for RAGmortem."""
    parser = argparse.ArgumentParser(
        prog="ragmortem",
        description="RAGmortem: Open-source RAG failure debugger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-dataset
    val_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate question dataset structure and chunk references.",
    )
    val_parser.add_argument(
        "--questions",
        default="evals/questions.jsonl",
        help="Path to questions JSONL file (default: evals/questions.jsonl)",
    )
    val_parser.add_argument(
        "--corpus",
        default="examples/reference_rag/documents",
        help="Path to documents directory (default: examples/reference_rag/documents)",
    )

    # run-reference
    run_parser = subparsers.add_parser(
        "run-reference",
        help="Run reference RAG pipeline against evaluation dataset.",
    )
    run_parser.add_argument(
        "--questions",
        default="evals/questions.jsonl",
        help="Path to questions JSONL file (default: evals/questions.jsonl)",
    )
    run_parser.add_argument(
        "--corpus",
        default="examples/reference_rag/documents",
        help="Path to documents directory (default: examples/reference_rag/documents)",
    )
    run_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using local deterministic mock generator instead of calling LLM API.",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (useful for quick smoke tests).",
    )

    # generate-faults
    gen_parser = subparsers.add_parser(
        "generate-faults",
        help="Generate controlled synthetic RAG failure dataset.",
    )
    gen_parser.add_argument(
        "--questions",
        default="evals/questions.jsonl",
        help="Path to questions JSONL file (default: evals/questions.jsonl)",
    )
    gen_parser.add_argument(
        "--corpus",
        default="examples/reference_rag/documents",
        help="Path to documents directory (default: examples/reference_rag/documents)",
    )
    gen_parser.add_argument(
        "--output",
        default="evals/injected_faults.jsonl",
        help="Output JSONL destination (default: evals/injected_faults.jsonl)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval cut-off (default: 3)",
    )

    # validate-faults
    val_faults_parser = subparsers.add_parser(
        "validate-faults",
        help="Validate injected fault dataset labels and ground truth.",
    )
    val_faults_parser.add_argument(
        "--faults",
        default="evals/injected_faults.jsonl",
        help="Path to injected faults JSONL file (default: evals/injected_faults.jsonl)",
    )
    val_faults_parser.add_argument(
        "--corpus",
        default="examples/reference_rag/documents",
        help="Path to documents directory (default: examples/reference_rag/documents)",
    )
    val_faults_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval cut-off (default: 3)",
    )

    args = parser.parse_args(argv)

    if args.command == "validate-dataset":
        return cmd_validate_dataset(args)
    elif args.command == "run-reference":
        return cmd_run_reference(args)
    elif args.command == "generate-faults":
        return cmd_generate_faults(args)
    elif args.command == "validate-faults":
        return cmd_validate_faults(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())

