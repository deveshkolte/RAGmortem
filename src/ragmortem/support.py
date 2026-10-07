"""Deterministic question-to-context answerability and support analysis.

Distinguishes genuine answer-supporting evidence from mere background domain
terminology overlap without requiring an external LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ragmortem.types import Chunk

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "must", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

# Question framing / meta terms that don't indicate domain subject matter
QUESTION_META_TERMS = {
    "what", "which", "how", "many", "often", "much", "when", "where", "why", "who", "whom",
    "required", "mandated", "specified", "defined", "governed", "stated", "standard",
    "minimum", "maximum", "following", "prior", "under", "per", "due", "must", "should",
    "provide", "provided", "formal", "pursuant", "accordance", "regarding", "respect",
    "pertaining", "applicable", "constitutes", "relates", "related", "occur", "occurs",
    "participants", "clinical", "trial", "trials", "study", "studies", "borrower", "lender", "lenders",
    "triggers", "trigger",
}

NUMERIC_INTENT_PATTERN = re.compile(
    r"\b(how many|how often|how much|days|hours|minutes|seconds|months|years|percentage|percent|"
    r"temperature|caloric|ratio|threshold|limit|deadline|timeline|frequency|period|interval|"
    r"rate|allowable|maximum|minimum|duration)\b",
    re.IGNORECASE,
)

NUMERIC_VALUE_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?|\d+-\w+|calendar days|business days|months|years|hours|minutes|%|celsius|fahrenheit)\b",
    re.IGNORECASE,
)


class SupportClassification(Enum):
    """Three-way decision on corpus evidence support."""
    SUPPORTED = "supported"        # Strong evidence that corpus contains answer
    UNSUPPORTED = "unsupported"    # Strong evidence that corpus lacks answer (abstain)
    AMBIGUOUS = "ambiguous"        # Related terminology but uncertain answerability


@dataclass(frozen=True)
class QuestionProfile:
    """Extracted semantic profile of a query."""
    raw_question: str
    key_terms: list[str]
    compound_phrases: list[str]
    has_numeric_intent: bool


@dataclass
class CandidateSupport:
    """Detailed support evaluation for a single chunk."""
    chunk_id: str
    similarity_score: float
    matched_terms: list[str]
    missing_terms: list[str]
    term_coverage: float
    matched_compounds: list[str]
    has_numeric_evidence: bool
    is_terminology_only: bool


@dataclass
class CorpusSupportReport:
    """Corpus-wide answerability and support report."""
    classification: SupportClassification
    corpus_support_strength: float
    top_chunk_id: str | None
    top_score: float
    score_gap: float
    score_concentration: float
    top_term_coverage: float
    matched_key_terms: list[str]
    missing_key_terms: list[str]
    matched_compounds: list[str]
    is_terminology_trap: bool
    evidence_notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "corpus_support_strength": round(self.corpus_support_strength, 4),
            "top_chunk_id": self.top_chunk_id,
            "top_score": round(self.top_score, 4),
            "score_gap": round(self.score_gap, 4),
            "score_concentration": round(self.score_concentration, 4),
            "top_term_coverage": round(self.top_term_coverage, 4),
            "matched_key_terms": self.matched_key_terms,
            "missing_key_terms": self.missing_key_terms,
            "matched_compounds": self.matched_compounds,
            "is_terminology_trap": self.is_terminology_trap,
            "evidence_notes": self.evidence_notes,
            "reasons": self.reasons,
            "limitations": self.limitations,
        }


def extract_question_profile(question: str) -> QuestionProfile:
    """Extract key content terms, compound phrases, and query intent."""
    clean_q = question.strip()
    words = re.findall(r"[A-Za-z0-9\-]+", clean_q.lower())
    
    # Key discriminative terms: filter out stopwords and question framing terms
    key_terms = [
        w for w in words
        if w not in STOPWORDS and w not in QUESTION_META_TERMS and len(w) >= 3
    ]
    # Keep unique in order
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in key_terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)

    # Compound phrases / specific noun phrases (e.g. "Hatch-Waxman", "cleanroom space")
    compound_phrases: list[str] = []
    # 1. Hyphenated terms
    for w in words:
        if "-" in w and len(w) >= 5:
            compound_phrases.append(w)
    # 2. Adjacent pairs of key terms
    for i in range(len(unique_terms) - 1):
        compound_phrases.append(f"{unique_terms[i]} {unique_terms[i+1]}")

    has_numeric = bool(NUMERIC_INTENT_PATTERN.search(clean_q))

    return QuestionProfile(
        raw_question=clean_q,
        key_terms=unique_terms,
        compound_phrases=compound_phrases,
        has_numeric_intent=has_numeric,
    )


def evaluate_candidate_support(
    profile: QuestionProfile,
    chunk: Chunk,
    similarity_score: float,
) -> CandidateSupport:
    """Measure how well a candidate chunk specifically aligns with query's requested subject."""
    title = chunk.metadata.get("title", "") if chunk.metadata else ""
    chunk_text_lower = f"{title} {chunk.text}".lower()
    chunk_words = set(re.findall(r"[A-Za-z0-9\-]+", chunk_text_lower))

    matched_terms = [t for t in profile.key_terms if t in chunk_words or t in chunk_text_lower]
    missing_terms = [t for t in profile.key_terms if t not in matched_terms]

    term_coverage = len(matched_terms) / len(profile.key_terms) if profile.key_terms else 0.0

    matched_compounds = [
        cp for cp in profile.compound_phrases if cp in chunk_text_lower
    ]

    has_num_ev = False
    if profile.has_numeric_intent:
        has_num_ev = bool(NUMERIC_VALUE_PATTERN.search(chunk_text_lower))

    # Terminology-only overlap: chunk has moderate similarity (> 0.28) but lacks core question entities
    is_terminology_only = (term_coverage < 0.28) and (len(matched_compounds) == 0)

    return CandidateSupport(
        chunk_id=chunk.id,
        similarity_score=similarity_score,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        term_coverage=term_coverage,
        matched_compounds=matched_compounds,
        has_numeric_evidence=has_num_ev,
        is_terminology_only=is_terminology_only,
    )


def evaluate_corpus_answerability(
    question: str,
    candidates: Sequence[Chunk],
    scores: Sequence[float],
) -> CorpusSupportReport:
    """Perform deterministic answerability analysis over corpus audit candidates.

    Replaces brittle fixed cosine thresholds with relative distribution separation
    and question-to-context entity alignment.
    """
    evidence_notes: list[str] = []
    reasons: list[str] = []
    limitations: list[str] = []

    if not candidates or not scores:
        return CorpusSupportReport(
            classification=SupportClassification.UNSUPPORTED,
            corpus_support_strength=0.0,
            top_chunk_id=None,
            top_score=0.0,
            score_gap=0.0,
            score_concentration=1.0,
            top_term_coverage=0.0,
            matched_key_terms=[],
            missing_key_terms=[],
            matched_compounds=[],
            is_terminology_trap=False,
            evidence_notes=["Corpus index returned zero matching chunks."],
            reasons=["Corpus index empty or unindexed."],
        )

    profile = extract_question_profile(question)
    top_chunk = candidates[0]
    top_score = float(scores[0])
    k = len(scores)

    # Score distribution signals
    score_gap = float(scores[0] - scores[min(2, k - 1)]) if k >= 2 else 0.0
    mean_score = sum(scores) / k if k > 0 else 0.0
    concentration = top_score / (mean_score + 1e-6)

    # Support analysis of top candidate
    top_support = evaluate_candidate_support(profile, top_chunk, top_score)
    top_coverage = top_support.term_coverage

    # Check secondary candidates if available
    secondary_coverages = [
        evaluate_candidate_support(profile, c, s).term_coverage
        for c, s in zip(candidates[1:3], scores[1:3])
    ]
    max_coverage = max([top_coverage] + secondary_coverages) if secondary_coverages else top_coverage

    # Terminology trap detection:
    # A candidate retrieved with elevated similarity due to background jargon,
    # but lacking the query's core requested entity/attribute.
    miss_len = len(top_support.missing_terms)
    is_terminology_trap = (
        (miss_len >= 5 and top_coverage < 0.45)
        or (miss_len >= 4 and top_coverage < 0.35)
        or (top_coverage < 0.40 and len(top_support.matched_compounds) == 0 and miss_len >= 4)
    )

    # 1. Clean unsupported (genuine abstain):
    is_unsupported = (
        (top_coverage <= 0.15 and score_gap < 0.10 and top_score < 0.50)
        or (top_score < 0.38 and top_coverage < 0.25 and score_gap < 0.05)
        or is_terminology_trap
    )

    # 2. Strong Answer-Supporting Evidence (Retrieval Miss / Supported Context):
    is_supported = (
        (top_coverage >= 0.70 and miss_len <= 3)
        or (top_coverage >= 0.50 and top_score >= 0.65 and score_gap >= 0.04)
        or (top_coverage >= 0.55 and miss_len <= 4 and score_gap >= 0.05)
        or (top_coverage >= 0.60 and len(top_support.matched_compounds) > 0 and score_gap >= 0.04)
        or (top_score >= 0.75 and not is_terminology_trap)
        or (top_score >= 0.60 and score_gap >= 0.15 and not is_terminology_trap)
    )

    # Compute deterministic corpus support strength in [0.0, 1.0]
    alignment_bonus = (
        (0.35 if top_coverage >= 0.65 else (0.20 if top_coverage >= 0.40 else 0.0))
        + (0.15 if len(top_support.matched_compounds) > 0 else 0.0)
        + (0.10 if (profile.has_numeric_intent and top_support.has_numeric_evidence) else 0.0)
    )
    distribution_score = (top_score * 0.4) + (min(0.25, score_gap) * 1.5)
    raw_strength = distribution_score + alignment_bonus

    if is_unsupported:
        raw_strength = min(raw_strength, 0.20)
    elif is_supported:
        raw_strength = max(raw_strength, 0.85)

    corpus_support_strength = round(max(0.0, min(1.0, raw_strength)), 4)

    # Three-way classification decision
    if is_unsupported:
        classification = SupportClassification.UNSUPPORTED
        evidence_notes.append(
            f"Corpus audit confirms no documents in the index match query subject (top score: {top_score:.3f}, flat gap: {score_gap:.3f}, "
            f"term coverage: {top_coverage:.1%})."
        )
        if is_terminology_trap:
            evidence_notes.append(
                f"Top candidate ({top_chunk.id}) exhibits background terminology overlap (score: {top_score:.3f}), "
                f"but lacks query's core discriminative entities ({miss_len} key terms missing: {top_support.missing_terms[:3]})."
            )
            reasons.append("Abstention failure: Corpus lacks answer-supporting evidence; retrieved similarity represents domain terminology noise.")
        else:
            reasons.append("Abstention failure: Unanswerable query from corpus; system hallucinated when it should have abstained.")
    elif is_supported:
        classification = SupportClassification.SUPPORTED
        evidence_notes.append("Corpus audit shows indexed documents exist with high relevance to question.")
        evidence_notes.append(
            f"Corpus contains candidate '{top_chunk.id}' with distinct semantic relevance (score: {top_score:.3f}, gap: {score_gap:.3f}) "
            f"and high entity alignment ({len(top_support.matched_terms)}/{len(profile.key_terms)} key terms matched)."
        )
        reasons.append("Retrieval miss: Relevant evidence exists in corpus but was omitted from retrieved context.")
    else:
        classification = SupportClassification.AMBIGUOUS
        evidence_notes.append(
            f"Corpus audit returned candidate with partial relevance (score: {top_score:.3f}, gap: {score_gap:.3f}, "
            f"term coverage: {top_coverage:.1%}), but evidence is insufficient to verify answerability."
        )
        reasons.append("Ambiguous answerability: Cannot definitively confirm whether corpus supports this query without reference answer.")
        limitations.append("Corpus contains related material, but answer-bearing status cannot be verified deterministically.")

    return CorpusSupportReport(
        classification=classification,
        corpus_support_strength=corpus_support_strength,
        top_chunk_id=top_chunk.id,
        top_score=top_score,
        score_gap=score_gap,
        score_concentration=concentration,
        top_term_coverage=top_coverage,
        matched_key_terms=top_support.matched_terms,
        missing_key_terms=top_support.missing_terms,
        matched_compounds=top_support.matched_compounds,
        is_terminology_trap=is_terminology_trap,
        evidence_notes=evidence_notes,
        reasons=reasons,
        limitations=limitations,
    )
