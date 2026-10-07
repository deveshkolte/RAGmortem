"""Unit tests for dataset loading and validation logic."""

import json
from pathlib import Path

from ragmortem.dataset import load_questions, validate_dataset
from ragmortem.types import Chunk


def test_evals_dataset_validates_cleanly() -> None:
    questions_file = Path("evals/questions.jsonl")
    docs_dir = Path("examples/reference_rag/documents")

    from examples.reference_rag.app import load_corpus_from_directory

    chunks = load_corpus_from_directory(docs_dir)
    errors = validate_dataset(questions_file, chunks)
    assert errors == [], f"Validation errors found: {errors}"


def test_dataset_validation_detects_corrupt_records(tmp_path: Path) -> None:
    chunks = [Chunk(id="c1", text="content", source="s1")]

    # 1. Unknown gold chunk ID
    q_file = tmp_path / "bad_q1.jsonl"
    q_file.write_text(
        json.dumps({
            "id": "q1",
            "question": "What is x?",
            "gold_chunk_id": "non_existent_chunk",
            "gold_answer": "ans",
            "type": "answerable",
        })
        + "\n"
    )
    errors = validate_dataset(q_file, chunks)
    assert any("not found in corpus chunks" in e for e in errors)

    # 2. Duplicate question ID
    q_file2 = tmp_path / "dup_id.jsonl"
    q_file2.write_text(
        json.dumps({
            "id": "q1",
            "question": "What is x?",
            "gold_chunk_id": "c1",
            "gold_answer": "ans",
            "type": "answerable",
        })
        + "\n"
        + json.dumps({
            "id": "q1",
            "question": "What is y?",
            "gold_chunk_id": "c1",
            "gold_answer": "ans2",
            "type": "answerable",
        })
        + "\n"
    )
    errors2 = validate_dataset(q_file2, chunks)
    assert any("Duplicate question ID" in e for e in errors2)

    # 3. Invalid question type
    q_file3 = tmp_path / "bad_type.jsonl"
    q_file3.write_text(
        json.dumps({
            "id": "q3",
            "question": "What is x?",
            "gold_chunk_id": "c1",
            "gold_answer": "ans",
            "type": "invalid_type",
        })
        + "\n"
    )
    errors3 = validate_dataset(q_file3, chunks)
    assert any("invalid type" in e for e in errors3)

    # 4. Unanswerable question with gold_chunk_id specified
    q_file4 = tmp_path / "bad_unans.jsonl"
    q_file4.write_text(
        json.dumps({
            "id": "q4",
            "question": "What is x?",
            "gold_chunk_id": "c1",
            "gold_answer": "ans",
            "type": "unanswerable",
        })
        + "\n"
    )
    errors4 = validate_dataset(q_file4, chunks)
    assert any("must not specify a gold_chunk_id" in e for e in errors4)
