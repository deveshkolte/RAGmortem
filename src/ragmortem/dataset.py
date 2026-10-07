"""Dataset loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ragmortem.types import Chunk

VALID_QUESTION_TYPES = {"answerable", "unanswerable"}


def load_questions(questions_path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL questions file into a list of dictionaries."""
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
                questions.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return questions


def validate_dataset(
    questions_path: str | Path,
    corpus_chunks: Iterable[Chunk] | set[str] | list[str],
) -> list[str]:
    """Validate question dataset against corpus chunks.

    Returns a list of validation error descriptions. An empty list signifies
    a completely valid dataset.
    """
    # Extract set of valid chunk IDs
    if corpus_chunks and isinstance(next(iter(corpus_chunks)), Chunk):
        valid_chunk_ids: set[str] = {c.id for c in corpus_chunks}  # type: ignore[union-attr]
    else:
        valid_chunk_ids = {str(c) for c in corpus_chunks}

    errors: list[str] = []
    seen_ids: set[str] = set()

    questions = load_questions(questions_path)
    if not questions:
        errors.append("Questions file is empty.")
        return errors

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("id")
        if not q_id or not isinstance(q_id, str):
            errors.append(f"Question #{idx}: missing or non-string 'id'.")
            continue

        if q_id in seen_ids:
            errors.append(f"Duplicate question ID '{q_id}' at item #{idx}.")
        seen_ids.add(q_id)

        question_text = q.get("question")
        if not question_text or not isinstance(question_text, str) or not question_text.strip():
            errors.append(f"Question '{q_id}': missing or empty 'question' text.")

        q_type = q.get("type")
        if q_type not in VALID_QUESTION_TYPES:
            errors.append(
                f"Question '{q_id}': invalid type '{q_type}'. Must be one of {sorted(VALID_QUESTION_TYPES)}."
            )

        gold_chunk_id = q.get("gold_chunk_id")
        gold_answer = q.get("gold_answer")

        if not gold_answer or not isinstance(gold_answer, str) or not gold_answer.strip():
            errors.append(f"Question '{q_id}': missing or empty 'gold_answer'.")

        if q_type == "answerable":
            if not gold_chunk_id or not isinstance(gold_chunk_id, str):
                errors.append(
                    f"Question '{q_id}': answerable question must have a non-empty string 'gold_chunk_id'."
                )
            elif gold_chunk_id not in valid_chunk_ids:
                errors.append(
                    f"Question '{q_id}': referenced gold_chunk_id '{gold_chunk_id}' not found in corpus chunks."
                )
        elif q_type == "unanswerable":
            if gold_chunk_id is not None and gold_chunk_id != "":
                errors.append(
                    f"Question '{q_id}': unanswerable question must not specify a gold_chunk_id (got '{gold_chunk_id}')."
                )

    return errors
