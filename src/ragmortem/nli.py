"""Fast local NLI and cross-encoder answer verification.

Provides local semantic verification to evaluate whether a generated candidate answer
entails the gold reference facts, replacing naive token-matching for paraphrases
and semantically equivalent responses.

Optional dependency:
    pip install ragmortem[local-nli]
"""

from __future__ import annotations

import logging
import math
import os
import re
import string
from typing import Any, Protocol

import functools
logger = logging.getLogger("ragmortem.nli")

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-mini"


@functools.lru_cache(maxsize=4)
def get_cross_encoder(model_name: str = DEFAULT_NLI_MODEL) -> Any:
    """Load and memoize CrossEncoder instances across the process."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)



def compute_ngram_cosine_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram cosine similarity between two strings (dependency-free)."""
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
        return " ".join(s.split())

    t1 = _norm(text1)
    t2 = _norm(text2)
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0

    def _ngrams(s: str) -> dict[str, int]:
        padded = f" {s} "
        counts: dict[str, int] = {}
        for i in range(max(1, len(padded) - n + 1)):
            gram = padded[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    ng1 = _ngrams(t1)
    ng2 = _ngrams(t2)

    dot = sum(cnt * ng2.get(gram, 0) for gram, cnt in ng1.items())
    mag1 = math.sqrt(sum(cnt * cnt for cnt in ng1.values()))
    mag2 = math.sqrt(sum(cnt * cnt for cnt in ng2.values()))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


class CrossEncoderProtocol(Protocol):
    """Protocol for cross-encoder models."""
    def predict(self, sentences: list[list[str]] | list[tuple[str, str]]) -> Any:
        ...


class LocalNLIVerifier:
    """Fast local NLI / Cross-Encoder answer verifier.

    Evaluates whether the candidate response entails the gold facts.
    Loads models lazily, short-circuits obvious matches, and degrades
    gracefully to lexical matching if model weights cannot be loaded.
    """

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        threshold: float = 0.50,
        enabled: bool | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("RAGMORTEM_NLI_MODEL", DEFAULT_NLI_MODEL)
        self._model = model
        self.threshold = threshold
        self._load_attempted = False
        self._load_failed = False

        if enabled is not None:
            self.enabled = enabled
        else:
            disable_env = os.environ.get("RAGMORTEM_DISABLE_NLI", "").strip().lower()
            self.enabled = disable_env not in ("1", "true", "yes")

    def is_available(self) -> bool:
        """Return True if local NLI verification is active and supported."""
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> bool:
        """Lazily load the cross-encoder model."""
        if self._model is not None:
            return True
        if self._load_attempted or not self.enabled:
            return not self._load_failed

        self._load_attempted = True
        try:
            logger.info("Loading local NLI CrossEncoder '%s'...", self.model_name)
            self._model = get_cross_encoder(self.model_name)
            return True
        except Exception as exc:
            logger.debug("Local NLI model '%s' unavailable: %s", self.model_name, exc)
            self._load_failed = True
            return False

    def _verify_ngram_fallback(self, candidate: str, gold_answer: str, threshold: float = 0.50) -> bool:
        """Fast, dependency-free character n-gram cosine fallback for paraphrases."""
        def _norm(s: str) -> str:
            s = s.lower()
            s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
            return " ".join(s.split())

        norm_cand = _norm(candidate)
        norm_gold = _norm(gold_answer)
        if not norm_cand or not norm_gold:
            return False

        # Numeric token check: if gold specifies numbers, candidate must have them
        gold_digits = [w for w in norm_gold.split() if w.isdigit()]
        cand_words = set(norm_cand.split())
        if gold_digits and not all(d in cand_words for d in gold_digits):
            return False

        sim = compute_ngram_cosine_similarity(norm_cand, norm_gold, n=3)
        return sim >= threshold

    def verify(self, candidate: str, gold_answer: str) -> bool:
        """Verify whether candidate response entails or validates gold answer facts."""
        if not candidate or not gold_answer:
            return False

        if not self.enabled:
            return False

        # 1. Try heavy neural model if available
        if self.is_available() and self._ensure_loaded():
            try:
                preds = self._model.predict([(candidate, gold_answer)])
                if self._is_entailment(preds):
                    return True
            except Exception as exc:
                logger.debug("NLI verification error: %s", exc)

        # 2. Dependency-free character n-gram cosine fallback (zero heavy deps)
        return self._verify_ngram_fallback(candidate, gold_answer)

    def _is_entailment(self, prediction: Any) -> bool:
        """Determine whether prediction represents entailment above threshold."""
        import numpy as np

        arr = np.asarray(prediction)
        # 1D single score (e.g. binary classification or regression)
        if arr.ndim == 1:
            if arr.size == 1:
                return float(arr[0]) >= self.threshold
            elif arr.size == 2:
                # [not_entailment, entailment]
                exp_s = np.exp(arr - np.max(arr))
                probs = exp_s / np.sum(exp_s)
                return float(probs[1]) >= self.threshold
            elif arr.size == 3:
                # Standard 3-class NLI: [contradiction, neutral, entailment]
                # Check model config if available, default to index 2 or 1
                entail_idx = 2
                if hasattr(self._model, "config") and hasattr(self._model.config, "label2id"):
                    l2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
                    entail_idx = l2id.get("entailment", 2)
                exp_s = np.exp(arr - np.max(arr))
                probs = exp_s / np.sum(exp_s)
                return float(probs[entail_idx]) >= self.threshold

        # 2D batch score: shape (1, num_labels)
        if arr.ndim == 2 and arr.shape[0] == 1:
            row = arr[0]
            if row.size == 1:
                return float(row[0]) >= self.threshold
            elif row.size == 2:
                exp_s = np.exp(row - np.max(row))
                probs = exp_s / np.sum(exp_s)
                return float(probs[1]) >= self.threshold
            elif row.size == 3:
                entail_idx = 2
                if hasattr(self._model, "config") and hasattr(self._model.config, "label2id"):
                    l2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
                    entail_idx = l2id.get("entailment", 2)
                exp_s = np.exp(row - np.max(row))
                probs = exp_s / np.sum(exp_s)
                return float(probs[entail_idx]) >= self.threshold

        return False


_GLOBAL_VERIFIER: LocalNLIVerifier | None = None


def get_default_verifier() -> LocalNLIVerifier:
    """Return the singleton default LocalNLIVerifier instance."""
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = LocalNLIVerifier()
    return _GLOBAL_VERIFIER


def set_default_verifier(verifier: LocalNLIVerifier | None) -> None:
    """Set or reset the default LocalNLIVerifier instance."""
    global _GLOBAL_VERIFIER
    _GLOBAL_VERIFIER = verifier
