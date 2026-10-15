"""Distribution-aware relative scoring and dynamic threshold auto-calibration.

Auto-calibrates diagnostic thresholds to any embedding model (e.g. OpenAI
text-embedding-3-small/large, Cohere embed-v3, BAAI bge-large, all-MiniLM-L6-v2, etc.)
by dynamically analyzing score distributions (mean, spread, z-scores, and gaps)
rather than relying on hardcoded cosine similarity constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CalibratedThresholds:
    """Dynamically calibrated thresholds for diagnostic attribution."""
    low_bound: float
    high_bound: float
    cutoff_gap: float
    false_refusal_threshold: float
    score_spread_min: float
    is_auto_calibrated: bool
    model_profile: str
    observed_mean: float | None = None
    observed_spread: float | None = None
    relative_z_score: float | None = None

    @property
    def ambiguity_band(self) -> tuple[float, float]:
        """Return (low_bound, high_bound) tuple."""
        return (self.low_bound, self.high_bound)

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_bound": round(self.low_bound, 4),
            "high_bound": round(self.high_bound, 4),
            "cutoff_gap": round(self.cutoff_gap, 4),
            "false_refusal_threshold": round(self.false_refusal_threshold, 4),
            "score_spread_min": round(self.score_spread_min, 4),
            "is_auto_calibrated": self.is_auto_calibrated,
            "model_profile": self.model_profile,
            "observed_mean": round(self.observed_mean, 4) if self.observed_mean is not None else None,
            "observed_spread": round(self.observed_spread, 4) if self.observed_spread is not None else None,
            "relative_z_score": round(self.relative_z_score, 4) if self.relative_z_score is not None else None,
        }


# Known model baseline distributions (mean, std, expected_high)
MODEL_BASELINES: dict[str, dict[str, float]] = {
    "all-minilm-l6-v2": {
        "low_bound": 0.47,
        "high_bound": 0.53,
        "cutoff_gap": 0.08,
        "false_refusal": 0.54,
        "spread_min": 0.10,
        "baseline_mean": 0.35,
    },
    "text-embedding-3-small": {
        "low_bound": 0.68,
        "high_bound": 0.76,
        "cutoff_gap": 0.06,
        "false_refusal": 0.77,
        "spread_min": 0.08,
        "baseline_mean": 0.55,
    },
    "text-embedding-3-large": {
        "low_bound": 0.65,
        "high_bound": 0.74,
        "cutoff_gap": 0.06,
        "false_refusal": 0.75,
        "spread_min": 0.08,
        "baseline_mean": 0.52,
    },
    "bge-large-en-v1.5": {
        "low_bound": 0.60,
        "high_bound": 0.70,
        "cutoff_gap": 0.07,
        "false_refusal": 0.72,
        "spread_min": 0.09,
        "baseline_mean": 0.48,
    },
    "bge-small-en-v1.5": {
        "low_bound": 0.58,
        "high_bound": 0.68,
        "cutoff_gap": 0.07,
        "false_refusal": 0.70,
        "spread_min": 0.09,
        "baseline_mean": 0.45,
    },
    "cohere-embed-v3": {
        "low_bound": 0.55,
        "high_bound": 0.65,
        "cutoff_gap": 0.07,
        "false_refusal": 0.67,
        "spread_min": 0.09,
        "baseline_mean": 0.42,
    },
}


class ScoreCalibrator:
    """Distribution-aware dynamic calibrator for similarity scores."""

    def __init__(
        self,
        default_model: str = "all-MiniLM-L6-v2",
        fallback_band: tuple[float, float] = (0.47, 0.53),
        fallback_cutoff_gap: float = 0.08,
    ) -> None:
        self.default_model = default_model
        self.fallback_band = fallback_band
        self.fallback_cutoff_gap = fallback_cutoff_gap

    def calibrate(
        self,
        scores: Sequence[float] | None = None,
        candidate_scores: Sequence[float] | None = None,
        model_name: str | None = None,
    ) -> CalibratedThresholds:
        """Calibrate thresholds to the observed score distribution or model profile.

        Algorithm:
        1. Explicit Model Profile: If model_name matches a known embedding profile,
           use its calibrated baseline parameters.
        2. Empirical Distribution Calibration:
           - If an empirical candidate pool or multiple retrieved scores are provided
             with at least 3 samples AND elevated minimum/mean scores (e.g. min >= 0.55,
             mean >= 0.62 as observed in high-baseline models like OpenAI), dynamically
             shift the ambiguity band and false refusal threshold to match the observed scale.
           - Otherwise, retain canonical baseline thresholds (0.47, 0.53, 0.08, 0.54)
             to preserve deterministic behavior for standard traces and MiniLM.
        """
        model_key = (model_name or self.default_model or "").lower().strip()
        matched_profile = None
        for k in MODEL_BASELINES:
            if k in model_key or model_key in k:
                matched_profile = k
                break
        if not matched_profile:
            if "openai" in model_key:
                matched_profile = "text-embedding-3-small"
            elif "bge" in model_key:
                matched_profile = "bge-large-en-v1.5"
            elif "cohere" in model_key:
                matched_profile = "cohere-embed-v3"

        # Pool all available scores to estimate distribution
        pooled_scores: list[float] = []
        if scores:
            pooled_scores.extend(float(s) for s in scores if s is not None)
        if candidate_scores:
            pooled_scores.extend(float(s) for s in candidate_scores if s is not None)

        # Baseline defaults
        if matched_profile and matched_profile != "all-minilm-l6-v2":
            p = MODEL_BASELINES[matched_profile]
            base_low = p["low_bound"]
            base_high = p["high_bound"]
            base_cutoff = p["cutoff_gap"]
            base_false_refusal = p["false_refusal"]
            base_spread_min = p["spread_min"]
            profile_name = matched_profile
        else:
            base_low, base_high = self.fallback_band
            base_cutoff = self.fallback_cutoff_gap
            base_false_refusal = 0.54
            base_spread_min = 0.10
            profile_name = matched_profile or "all-minilm-l6-v2"

        if not pooled_scores:
            return CalibratedThresholds(
                low_bound=base_low,
                high_bound=base_high,
                cutoff_gap=base_cutoff,
                false_refusal_threshold=base_false_refusal,
                score_spread_min=base_spread_min,
                is_auto_calibrated=bool(matched_profile and matched_profile != "all-minilm-l6-v2"),
                model_profile=profile_name,
            )

        n = len(pooled_scores)
        mean_s = sum(pooled_scores) / n
        variance = sum((s - mean_s) ** 2 for s in pooled_scores) / n if n > 1 else 0.0
        std_s = math.sqrt(variance)
        max_s = max(pooled_scores)
        min_s = min(pooled_scores)
        spread = max_s - min_s
        top_s = pooled_scores[0]
        z_score = (top_s - mean_s) / std_s if std_s > 0.001 else 0.0

        # Dynamic empirical auto-calibration:
        # Requires at least 3 observed scores and an elevated background baseline (min_s >= 0.55, mean_s >= 0.62)
        # to unambiguously distinguish high-baseline embedding models (OpenAI, BGE, etc.) from standard MiniLM.
        if n >= 3 and min_s >= 0.55 and mean_s >= 0.62:
            calib_low = max(0.55, min(0.80, mean_s - 0.04))
            calib_high = max(calib_low + 0.06, min(0.88, mean_s + 0.04))
            calib_cutoff = max(0.03, min(0.08, spread * 0.30 if spread > 0 else base_cutoff))
            calib_false_refusal = max(calib_high + 0.01, min(0.90, mean_s + 0.05))
            calib_spread_min = max(0.03, min(0.08, std_s * 1.0 if std_s > 0 else base_spread_min))

            return CalibratedThresholds(
                low_bound=round(calib_low, 4),
                high_bound=round(calib_high, 4),
                cutoff_gap=round(calib_cutoff, 4),
                false_refusal_threshold=round(calib_false_refusal, 4),
                score_spread_min=round(calib_spread_min, 4),
                is_auto_calibrated=True,
                model_profile=f"auto-calibrated (high-shift, mean={mean_s:.2f}, n={n})",
                observed_mean=mean_s,
                observed_spread=spread,
                relative_z_score=z_score,
            )

        # Standard baseline distribution (MiniLM or explicit known model)
        return CalibratedThresholds(
            low_bound=base_low,
            high_bound=base_high,
            cutoff_gap=base_cutoff,
            false_refusal_threshold=base_false_refusal,
            score_spread_min=base_spread_min,
            is_auto_calibrated=bool(matched_profile and matched_profile != "all-minilm-l6-v2"),
            model_profile=profile_name,
            observed_mean=mean_s,
            observed_spread=spread,
            relative_z_score=z_score,
        )
