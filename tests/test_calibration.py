"""Tests for distribution-aware relative scoring and dynamic threshold calibration."""

from __future__ import annotations

import pytest

from ragmortem.calibration import ScoreCalibrator, CalibratedThresholds, MODEL_BASELINES
from ragmortem.diagnose import ObservedExecution, RAGDiagnoser
from ragmortem.taxonomy import FailureType


def test_default_minilm_calibration() -> None:
    calibrator = ScoreCalibrator()
    calib = calibrator.calibrate(scores=[0.65, 0.40, 0.20])
    assert calib.low_bound == 0.47
    assert calib.high_bound == 0.53
    assert calib.cutoff_gap == 0.08
    assert calib.false_refusal_threshold == 0.54
    assert calib.is_auto_calibrated is False
    assert calib.observed_mean is not None
    assert round(calib.observed_mean, 2) == 0.42


def test_explicit_model_profiles() -> None:
    calibrator = ScoreCalibrator()

    # OpenAI profile
    calib_openai = calibrator.calibrate(model_name="text-embedding-3-small")
    assert calib_openai.low_bound == 0.68
    assert calib_openai.high_bound == 0.76
    assert calib_openai.false_refusal_threshold == 0.77
    assert calib_openai.is_auto_calibrated is True

    # Cohere profile
    calib_cohere = calibrator.calibrate(model_name="cohere-embed-v3")
    assert calib_cohere.low_bound == 0.55
    assert calib_cohere.high_bound == 0.65
    assert calib_cohere.false_refusal_threshold == 0.67
    assert calib_cohere.is_auto_calibrated is True

    # BGE profile
    calib_bge = calibrator.calibrate(model_name="bge-large-en-v1.5")
    assert calib_bge.low_bound == 0.60
    assert calib_bge.high_bound == 0.70
    assert calib_bge.false_refusal_threshold == 0.72
    assert calib_bge.is_auto_calibrated is True


def test_empirical_auto_calibration_high_baseline() -> None:
    calibrator = ScoreCalibrator()
    # High baseline distribution (like OpenAI, but model unspecified)
    high_baseline_candidates = [0.85, 0.82, 0.80, 0.78, 0.76, 0.75, 0.73, 0.71, 0.70, 0.68]
    calib = calibrator.calibrate(candidate_scores=high_baseline_candidates)

    assert calib.is_auto_calibrated is True
    assert "high-shift" in calib.model_profile
    assert calib.observed_mean is not None and calib.observed_mean >= 0.70
    # Thresholds should be elevated above MiniLM's 0.47/0.53
    assert calib.low_bound >= 0.65
    assert calib.high_bound >= 0.75
    assert calib.false_refusal_threshold >= 0.78
    assert calib.relative_z_score is not None and calib.relative_z_score > 1.0


def test_diagnoser_with_openai_model_metadata() -> None:
    """A 0.72 score with OpenAI model profile lies in the ambiguity band [0.68, 0.76]."""
    trace = ObservedExecution(
        question="What is the data retention period under GDPR Article 17?",
        retrieved_chunk_ids=["chunk_gdpr_1"],
        generated_answer="Data retention is 30 days.",
        scores=[0.72],
        metadata={"model": "text-embedding-3-small"},
    )
    diagnoser = RAGDiagnoser()
    result = diagnoser.diagnose(trace, mode="trace_only")

    # Under MiniLM 0.72 would be high (>0.53), but under OpenAI it is ambiguous -> UNKNOWN
    assert result.diagnosis == FailureType.UNKNOWN
    assert result.signals["calibration"]["model_profile"] == "text-embedding-3-small"
    assert result.signals["calibration"]["low_bound"] == 0.68
    assert result.signals["calibration"]["high_bound"] == 0.76


def test_diagnoser_with_openai_high_score_generation_failure() -> None:
    """A 0.88 score with OpenAI model profile is > high_bound (0.76) -> GENERATION_SUSPECTED."""
    trace = ObservedExecution(
        question="What is the data retention period under GDPR Article 17?",
        retrieved_chunk_ids=["chunk_gdpr_1"],
        generated_answer="Data retention is 10 years.",
        scores=[0.88],
        metadata={"model": "text-embedding-3-small"},
    )
    diagnoser = RAGDiagnoser()
    result = diagnoser.diagnose(trace, mode="trace_only")

    assert result.diagnosis == FailureType.GENERATION_SUSPECTED
    assert result.signals["calibration"]["model_profile"] == "text-embedding-3-small"


def test_diagnoser_with_openai_false_refusal() -> None:
    """A refusal on score 0.82 (> 0.77 false refusal threshold) is GENERATION_SUSPECTED."""
    trace = ObservedExecution(
        question="What is the retention period?",
        retrieved_chunk_ids=["c1"],
        generated_answer="I cannot answer this question based on the provided context.",
        scores=[0.82],
        metadata={"model": "text-embedding-3-small"},
    )
    diagnoser = RAGDiagnoser()
    result = diagnoser.diagnose(trace, mode="trace_only")

    assert result.diagnosis == FailureType.GENERATION_SUSPECTED
    assert any("false refusal" in r.lower() for r in result.reasons)
