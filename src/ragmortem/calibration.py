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
class CalibrationProfile:
    """Embedding calibration profile with threshold parameters."""

    name: str
    low_bound: float
    high_bound: float
    cutoff_gap: float
    false_refusal_threshold: float
    score_spread_min: float
    is_relative: bool = False
    relative_cliff_ratio: float = 0.12
    z_score_threshold: float = 0.5

    @classmethod
    def minilm(cls) -> CalibrationProfile:
        return cls(
            name="minilm",
            low_bound=0.47,
            high_bound=0.53,
            cutoff_gap=0.08,
            false_refusal_threshold=0.54,
            score_spread_min=0.10,
            is_relative=False,
        )

    @classmethod
    def openai(cls) -> CalibrationProfile:
        return cls(
            name="openai",
            low_bound=0.68,
            high_bound=0.76,
            cutoff_gap=0.06,
            false_refusal_threshold=0.77,
            score_spread_min=0.08,
            is_relative=False,
        )

    @classmethod
    def bge(cls) -> CalibrationProfile:
        return cls(
            name="bge",
            low_bound=0.60,
            high_bound=0.70,
            cutoff_gap=0.07,
            false_refusal_threshold=0.72,
            score_spread_min=0.09,
            is_relative=False,
        )

    @classmethod
    def relative(cls) -> CalibrationProfile:
        return cls(
            name="relative",
            low_bound=0.45,
            high_bound=0.65,
            cutoff_gap=0.08,
            false_refusal_threshold=0.68,
            score_spread_min=0.06,
            is_relative=True,
            relative_cliff_ratio=0.12,
            z_score_threshold=0.5,
        )

    @classmethod
    def from_name(cls, name: str | CalibrationProfile | None) -> CalibrationProfile:
        if isinstance(name, CalibrationProfile):
            return name
        if not name:
            return cls.minilm()
        key = str(name).strip().lower()
        if key in ("minilm", "all-minilm-l6-v2"):
            return cls.minilm()
        elif key in ("openai", "text-embedding-3-small", "text-embedding-3-large", "ada-002", "text-embedding-ada-002"):
            return cls.openai()
        elif key in ("bge", "bge-large", "bge-small", "bge-large-en-v1.5", "bge-small-en-v1.5"):
            return cls.bge()
        elif key in ("relative", "distribution", "relative_drop"):
            return cls.relative()
        return cls.minilm()


CALIBRATION_PROFILES: dict[str, CalibrationProfile] = {
    "minilm": CalibrationProfile.minilm(),
    "openai": CalibrationProfile.openai(),
    "bge": CalibrationProfile.bge(),
    "relative": CalibrationProfile.relative(),
}


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
    is_relative: bool = False
    profile: CalibrationProfile | None = None
    relative_cliff_ratio: float | None = None

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
            "is_relative": self.is_relative,
            "observed_mean": round(self.observed_mean, 4) if self.observed_mean is not None else None,
            "observed_spread": round(self.observed_spread, 4) if self.observed_spread is not None else None,
            "relative_z_score": round(self.relative_z_score, 4) if self.relative_z_score is not None else None,
            "relative_cliff_ratio": round(self.relative_cliff_ratio, 4) if self.relative_cliff_ratio is not None else None,
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
        profile: str | CalibrationProfile = "minilm",
        default_model: str = "all-MiniLM-L6-v2",
        fallback_band: tuple[float, float] = (0.47, 0.53),
        fallback_cutoff_gap: float = 0.08,
    ) -> None:
        self.profile = CalibrationProfile.from_name(profile)
        self.default_model = default_model
        if self.profile.name != "minilm" or fallback_band == (0.47, 0.53):
            self.fallback_band = (self.profile.low_bound, self.profile.high_bound)
            self.fallback_cutoff_gap = self.profile.cutoff_gap
        else:
            self.fallback_band = fallback_band
            self.fallback_cutoff_gap = fallback_cutoff_gap

    def calibrate(
        self,
        scores: Sequence[float] | None = None,
        candidate_scores: Sequence[float] | None = None,
        model_name: str | None = None,
        profile: str | CalibrationProfile | None = None,
    ) -> CalibratedThresholds:
        """Calibrate thresholds to the observed score distribution or model profile.

        Algorithm:
        1. Explicit Calibration Profile: If profile is specified (or configured on calibrator),
           use that profile preset (minilm, openai, bge, relative).
        2. Explicit Model Profile: If model_name matches a known embedding profile,
           use its calibrated baseline parameters.
        3. Empirical Distribution Calibration:
           - If an empirical candidate pool or multiple retrieved scores are provided
             with at least 3 samples AND elevated minimum/mean scores (e.g. min >= 0.55,
             mean >= 0.62 as observed in high-baseline models like OpenAI), dynamically
             shift the ambiguity band and false refusal threshold to match the observed scale.
           - Otherwise, retain canonical baseline thresholds (0.47, 0.53, 0.08, 0.54)
             to preserve deterministic behavior for standard traces and MiniLM.
        """
        active_profile = CalibrationProfile.from_name(profile) if profile is not None else self.profile

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

        n = len(pooled_scores)
        mean_s = sum(pooled_scores) / n if n > 0 else 0.50
        variance = sum((s - mean_s) ** 2 for s in pooled_scores) / n if n > 1 else 0.0
        std_s = math.sqrt(variance)
        max_s = max(pooled_scores) if pooled_scores else 0.50
        min_s = min(pooled_scores) if pooled_scores else 0.50
        spread = max_s - min_s
        top_s = pooled_scores[0] if pooled_scores else 0.50
        z_score = (top_s - mean_s) / std_s if std_s > 0.001 else 0.0

        # Check relative mode first
        if active_profile.is_relative:
            rel_cliff = None
            if scores and candidate_scores and len(candidate_scores) > len(scores):
                k = len(scores)
                top_v = scores[0] if scores else 1.0
                rel_cliff = (candidate_scores[k - 1] - candidate_scores[k]) / max(top_v, 1e-6)

            calib_low = max(0.20, mean_s - 0.5 * (std_s or 0.10))
            calib_high = min(0.95, mean_s + 0.5 * (std_s or 0.10))
            calib_cutoff = max(0.02, min(0.15, spread * 0.25 if spread > 0 else 0.08))
            calib_false_refusal = min(0.95, mean_s + 0.8 * (std_s or 0.10))

            return CalibratedThresholds(
                low_bound=round(calib_low, 4),
                high_bound=round(calib_high, 4),
                cutoff_gap=round(calib_cutoff, 4),
                false_refusal_threshold=round(calib_false_refusal, 4),
                score_spread_min=0.05,
                is_auto_calibrated=True,
                is_relative=True,
                profile=active_profile,
                model_profile="relative",
                observed_mean=mean_s,
                observed_spread=spread,
                relative_z_score=z_score,
                relative_cliff_ratio=rel_cliff,
            )

        # Non-relative profiles
        if active_profile.name in ("openai", "bge"):
            base_low = active_profile.low_bound
            base_high = active_profile.high_bound
            base_cutoff = active_profile.cutoff_gap
            base_false_refusal = active_profile.false_refusal_threshold
            base_spread_min = active_profile.score_spread_min
            profile_name = active_profile.name
        elif matched_profile and matched_profile != "all-minilm-l6-v2":
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
            base_false_refusal = active_profile.false_refusal_threshold
            base_spread_min = active_profile.score_spread_min
            profile_name = active_profile.name or matched_profile or "minilm"

        if not pooled_scores:
            return CalibratedThresholds(
                low_bound=base_low,
                high_bound=base_high,
                cutoff_gap=base_cutoff,
                false_refusal_threshold=base_false_refusal,
                score_spread_min=base_spread_min,
                is_auto_calibrated=bool(active_profile.name != "minilm" or (matched_profile and matched_profile != "all-minilm-l6-v2")),
                model_profile=profile_name,
                profile=active_profile,
            )

        # Dynamic empirical auto-calibration:
        # Requires at least 3 observed scores and an elevated background baseline (min_s >= 0.55, mean_s >= 0.62)
        # to unambiguously distinguish high-baseline embedding models (OpenAI, BGE, etc.) from standard MiniLM.
        if active_profile.name == "minilm" and n >= 3 and min_s >= 0.55 and mean_s >= 0.62:
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
                profile=active_profile,
            )

        # Standard baseline distribution (MiniLM or explicit known model)
        return CalibratedThresholds(
            low_bound=base_low,
            high_bound=base_high,
            cutoff_gap=base_cutoff,
            false_refusal_threshold=base_false_refusal,
            score_spread_min=base_spread_min,
            is_auto_calibrated=bool(active_profile.name != "minilm" or (matched_profile and matched_profile != "all-minilm-l6-v2")),
            model_profile=profile_name,
            observed_mean=mean_s,
            observed_spread=spread,
            relative_z_score=z_score,
            profile=active_profile,
        )
