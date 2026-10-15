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


def test_calibration_profile_presets() -> None:
    """Verify built-in calibration profile presets."""
    from ragmortem.calibration import CalibrationProfile, CALIBRATION_PROFILES

    minilm = CalibrationProfile.from_name("minilm")
    assert minilm.name == "minilm"
    assert minilm.low_bound == 0.47
    assert minilm.high_bound == 0.53
    assert minilm.cutoff_gap == 0.08
    assert minilm.false_refusal_threshold == 0.54
    assert minilm.is_relative is False

    openai = CalibrationProfile.from_name("openai")
    assert openai.name == "openai"
    assert openai.low_bound == 0.68
    assert openai.high_bound == 0.76
    assert openai.cutoff_gap == 0.06
    assert openai.false_refusal_threshold == 0.77
    assert openai.is_relative is False

    bge = CalibrationProfile.from_name("bge")
    assert bge.name == "bge"
    assert bge.low_bound == 0.60
    assert bge.high_bound == 0.70
    assert bge.cutoff_gap == 0.07
    assert bge.false_refusal_threshold == 0.72
    assert bge.is_relative is False

    relative = CalibrationProfile.from_name("relative")
    assert relative.name == "relative"
    assert relative.is_relative is True
    assert relative.relative_cliff_ratio == 0.12


def test_diagnose_api_with_profile_parameter() -> None:
    """diagnose() accepts profile parameter and updates evidence strings."""
    from ragmortem.api import diagnose

    # In MiniLM profile, 0.70 is high relevance (> 0.53) -> GENERATION_SUSPECTED
    res_minilm = diagnose(
        query="What is the response time?",
        retrieved_chunks=[{"id": "doc1", "text": "15 mins", "score": 0.70}],
        answer="Wrong answer",
        profile="minilm",
        mode="trace_only",
    )
    assert res_minilm.diagnosis == FailureType.GENERATION_SUSPECTED
    assert any("0.53" in ev or "minilm" in ev for ev in res_minilm.evidence)

    # In OpenAI profile, 0.70 lies in ambiguity band [0.68, 0.76] -> UNKNOWN
    res_openai = diagnose(
        query="What is the response time?",
        retrieved_chunks=[{"id": "doc1", "text": "15 mins", "score": 0.70}],
        answer="Wrong answer",
        profile="openai",
        mode="trace_only",
    )
    assert res_openai.diagnosis == FailureType.UNKNOWN
    assert any("0.76" in ev or "openai" in ev for ev in res_openai.evidence)


def test_diagnose_relative_mode_cliff_ratio() -> None:
    """Relative mode computes cliff ratio (score[k-1] - score[k]) / max(score[0], 1e-6)."""
    from ragmortem.api import diagnose

    # Retrieved: score 0.70. Candidate below cutoff: score 0.68.
    # Cliff ratio: (0.70 - 0.68) / 0.70 = 0.0286 <= 0.12 -> RANKING_SUSPECTED
    res = diagnose(
        query="What is the timeout?",
        retrieved_chunks=[{"id": "c1", "score": 0.70}],
        candidate_chunks=[{"id": "c1", "score": 0.70}, {"id": "c2", "score": 0.68}],
        top_k=1,
        answer="Timeout is 60s",
        profile="relative",
        mode="trace_only",
    )
    assert res.diagnosis == FailureType.RANKING_SUSPECTED
    assert any("relative cliff drop" in ev for ev in res.evidence)
    assert res.signals["cliff_ratio"] <= 0.05
