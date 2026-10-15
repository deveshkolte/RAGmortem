"""Unit tests for local NLI and cross-encoder answer verification."""

from __future__ import annotations

import numpy as np
import pytest

from ragmortem.faults.eval import is_answer_correct, _lexical_token_match
from ragmortem.nli import LocalNLIVerifier, set_default_verifier, get_default_verifier


class MockCrossEncoder:
    """Mock cross-encoder for deterministic unit testing without external downloads."""

    def __init__(self, entailment_pairs: list[tuple[str, str]] | None = None) -> None:
        self.entailment_pairs = entailment_pairs or []
        self.call_count = 0

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.call_count += len(pairs)
        results = []
        for cand, gold in pairs:
            # Check if pair is marked as entailing
            is_entailed = any(
                cand.strip().lower() == c.strip().lower() and gold.strip().lower() == g.strip().lower()
                for c, g in self.entailment_pairs
            )
            if is_entailed:
                # 3-class logits: [contradiction, neutral, entailment]
                results.append([-2.0, -1.0, 3.5])
            else:
                results.append([3.0, 0.5, -2.0])
        return np.array(results)


def test_lexical_fast_path() -> None:
    mock_model = MockCrossEncoder()
    verifier = LocalNLIVerifier(model=mock_model)

    candidate = "The mandatory engineering response SLA for P1 incidents is 15 minutes."
    gold = "15 minutes"

    # Fast path should succeed via lexical token match without invoking cross-encoder
    assert _lexical_token_match(candidate, gold) is True
    assert is_answer_correct(candidate, gold, verifier=verifier) is True
    assert mock_model.call_count == 0


def test_abstention_rejected_immediately() -> None:
    mock_model = MockCrossEncoder()
    verifier = LocalNLIVerifier(model=mock_model)

    candidate = "I cannot answer this question based on the provided context."
    gold = "15 minutes"

    assert is_answer_correct(candidate, gold, verifier=verifier) is False
    assert mock_model.call_count == 0


def test_nli_semantic_paraphrase_verification() -> None:
    # Candidate does not share exact tokens with gold answer (different wording)
    candidate = "Engineers must acknowledge critical tier-one outages within a quarter of an hour."
    gold = "Mandatory response SLA for P1 incidents is 15 minutes."

    # Pure lexical match fails
    assert _lexical_token_match(candidate, gold) is False

    # With NLI verifier that verifies entailment:
    mock_model = MockCrossEncoder(entailment_pairs=[(candidate, gold)])
    verifier = LocalNLIVerifier(model=mock_model)

    assert verifier.verify(candidate, gold) is True
    assert is_answer_correct(candidate, gold, verifier=verifier) is True
    assert mock_model.call_count > 0


def test_nli_contradiction_rejected() -> None:
    candidate = "Engineering SLA for P1 defects is 48 business hours."
    gold = "Mandatory response SLA for P1 incidents is 15 minutes."

    # Neither lexical nor NLI match
    assert _lexical_token_match(candidate, gold) is False

    mock_model = MockCrossEncoder(entailment_pairs=[])
    verifier = LocalNLIVerifier(model=mock_model)

    assert verifier.verify(candidate, gold) is False
    assert is_answer_correct(candidate, gold, verifier=verifier) is False


def test_verifier_disabled_or_unavailable() -> None:
    candidate = "Some complex paraphrase without keyword overlap."
    gold = "Completely distinct phrase."

    verifier = LocalNLIVerifier(enabled=False)
    assert verifier.is_available() is False
    assert verifier.verify(candidate, gold) is False


def test_binary_and_scalar_predictions() -> None:
    class ScalarModel:
        def predict(self, pairs):
            return np.array([0.85])  # single high score

    v1 = LocalNLIVerifier(model=ScalarModel(), threshold=0.60)
    assert v1.verify("cand", "gold") is True

    class BinaryModel:
        def predict(self, pairs):
            return np.array([[ -1.5, 2.5 ]])  # [not_entailment, entailment]

    v2 = LocalNLIVerifier(model=BinaryModel(), threshold=0.50)
    assert v2.verify("cand", "gold") is True
