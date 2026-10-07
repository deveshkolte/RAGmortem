"""Evaluation, answer verification, and strict fault validation rules."""

from __future__ import annotations

import re
import string
from typing import Set

from ragmortem.faults.models import InjectedFaultCase
from ragmortem.taxonomy import FailureType

ABSTENTION_PATTERNS = [
    r"do not have sufficient information",
    r"cannot answer",
    r"not mentioned in the (provided )?context",
    r"not provided in the (provided )?context",
    r"context does not contain",
    r"insufficient context",
    r"no information provided",
]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return " ".join(text.split())


def is_abstaining(answer: str) -> bool:
    """Return True if the text signifies an abstention refusal."""
    normalized = normalize_text(answer)
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def is_answer_correct(candidate: str, gold_answer: str) -> bool:
    """Deterministic check verifying if candidate answer captures gold facts."""
    if is_abstaining(candidate):
        return False

    norm_cand = normalize_text(candidate)
    norm_gold = normalize_text(gold_answer)

    cand_tokens = norm_cand.split()
    gold_tokens = norm_gold.split()
    if not gold_tokens:
        return False

    # Exact token subsequence matching with full word boundaries
    n_gold = len(gold_tokens)
    for i in range(len(cand_tokens) - n_gold + 1):
        if cand_tokens[i : i + n_gold] == gold_tokens:
            return True

    # Key token matching (requiring full word boundary tokens)
    key_gold_tokens = [tok for tok in gold_tokens if len(tok) > 2 or tok.isdigit()]
    if not key_gold_tokens:
        return False

    cand_token_set = set(cand_tokens)
    # If gold specifies numbers, those exact numeric tokens MUST be present in candidate
    gold_digits = [tok for tok in key_gold_tokens if tok.isdigit()]
    if gold_digits and not all(d in cand_token_set for d in gold_digits):
        return False

    matched_tokens = sum(1 for tok in key_gold_tokens if tok in cand_token_set)
    return (matched_tokens / len(key_gold_tokens)) >= 0.8



def validate_fault_case(
    case: InjectedFaultCase,
    corpus_chunk_ids: set[str],
    top_k: int = 3,
    candidate_k: int = 6,
) -> tuple[bool, str]:
    """Strictly validate whether an InjectedFaultCase conforms to its ground-truth definition.

    Returns (is_valid, reason).
    """
    fault_type = case.fault_type

    # 1. RETRIEVAL MISS
    if fault_type == FailureType.RETRIEVAL_MISS or fault_type == "retrieval_miss":
        if not case.gold_chunk_id:
            return False, "Retrieval miss requires a non-empty gold_chunk_id."
        if case.gold_chunk_id not in corpus_chunk_ids:
            return False, f"Gold chunk '{case.gold_chunk_id}' was deleted from corpus (abstention condition!)."
        if case.gold_chunk_id in case.fault.retrieved_ids[:top_k]:
            return False, f"Gold chunk '{case.gold_chunk_id}' still present in retrieved top-{top_k}."
        if case.fault.gold_rank is not None:
            return False, f"Gold rank must be None for retrieval miss, got {case.fault.gold_rank}."
        return True, "Valid retrieval_miss: gold chunk exists in corpus but was omitted from top-k."

    # 2. RANKING MISS
    elif fault_type == FailureType.RANKING_MISS or fault_type == "ranking_miss":
        if not case.gold_chunk_id:
            return False, "Ranking miss requires a non-empty gold_chunk_id."
        if case.gold_chunk_id not in corpus_chunk_ids:
            return False, f"Gold chunk '{case.gold_chunk_id}' missing from corpus."
        candidates = case.fault.candidate_retrieved_ids or []
        if case.gold_chunk_id not in candidates:
            return False, f"Gold chunk '{case.gold_chunk_id}' not present in candidate pool."
        cand_rank = case.fault.candidate_gold_rank
        if cand_rank is None or cand_rank <= top_k:
            return False, f"Gold chunk candidate rank must be > {top_k}, got {cand_rank}."
        if case.gold_chunk_id in case.fault.retrieved_ids[:top_k]:
            return False, f"Gold chunk must not be in final generation context (rank {case.fault.gold_rank})."
        return True, f"Valid ranking_miss: gold chunk retrieved at candidate rank #{cand_rank} (> top-{top_k})."

    # 3. GENERATION IGNORED CONTEXT
    elif fault_type == FailureType.GENERATION_IGNORED_CONTEXT or fault_type == "generation_ignored_context":
        if not case.gold_chunk_id:
            return False, "Generation failure requires a non-empty gold_chunk_id."
        if case.gold_chunk_id not in corpus_chunk_ids:
            return False, f"Gold chunk '{case.gold_chunk_id}' missing from corpus."
        if case.gold_chunk_id not in case.fault.retrieved_ids[:top_k]:
            return False, f"Gold chunk '{case.gold_chunk_id}' must be present in retrieved top-{top_k}."
        if case.gold_chunk_id not in case.fault.prompt_context_chunk_ids:
            return False, f"Gold chunk '{case.gold_chunk_id}' must be passed to prompt context."
        if is_answer_correct(case.fault.answer, case.gold_answer):
            return False, f"Generated answer was unexpectedly correct: '{case.fault.answer}'"
        return True, "Valid generation_ignored_context: gold chunk was in context but answer was incorrect."

    # 4. SHOULD ABSTAIN
    elif fault_type == FailureType.SHOULD_ABSTAIN or fault_type == "should_abstain":
        if case.gold_chunk_id is not None:
            return False, f"Unanswerable question must not have gold_chunk_id (got '{case.gold_chunk_id}')."
        if is_abstaining(case.fault.answer):
            return False, f"Model correctly abstained ('{case.fault.answer}'). Failure not induced."
        return True, "Valid should_abstain: unanswerable query received substantive hallucinated response."

    return False, f"Unknown fault type '{fault_type}'."
