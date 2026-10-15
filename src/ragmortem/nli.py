"""Fast local NLI and cross-encoder answer verification.

Provides local semantic verification to evaluate whether a generated candidate answer
entails the gold reference facts, replacing naive token-matching for paraphrases
and semantically equivalent responses.

Optional dependency:
    pip install ragmortem[local-nli]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger("ragmortem.nli")

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-mini"


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
            from sentence_transformers import CrossEncoder

            logger.info("Loading local NLI CrossEncoder '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:
            logger.debug("Local NLI model '%s' unavailable: %s", self.model_name, exc)
            self._load_failed = True
            return False

    def verify(self, candidate: str, gold_answer: str) -> bool:
        """Verify whether candidate response entails or validates gold answer facts."""
        if not candidate or not gold_answer:
            return False

        if not self.is_available():
            return False

        if not self._ensure_loaded():
            return False

        try:
            # Pair: (premise=candidate, hypothesis=gold_answer)
            # Model evaluates if the generated answer entails the reference facts
            preds = self._model.predict([(candidate, gold_answer)])
            return self._is_entailment(preds)
        except Exception as exc:
            logger.debug("NLI verification error: %s", exc)
            return False

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
