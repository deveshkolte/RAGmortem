# Send Us Your RAG Failures (Intake Kit)

Are you experiencing mysterious wrong answers, hallucinations, or silent retrieval failures in your production RAG application?

Help us validate RAGmortem against real-world production data, and in return, receive a free, detailed root-cause forensic analysis of your pipeline failures.

---

## 1. What to Send

Send **5 failure cases** where your RAG system produced an incorrect, hallucinated, or unhelpful response.

Format your cases using RAGmortem's minimal schema (either as a single `.json` file containing an array of cases or a `.jsonl` file):

```json
[
  {
    "id": "case_01",
    "question": "What is the policy for expense reimbursements?",
    "answer": "Expenses must be submitted within 90 days.",
    "retrieved": [
      {
        "chunk_id": "doc_101",
        "text": "All employee travel and entertainment expenses must be submitted within 30 calendar days.",
        "score": 0.82
      },
      {
        "chunk_id": "doc_102",
        "text": "Receipts are required for any single expense exceeding $25.",
        "score": 0.61
      }
    ],
    "candidates": [
      { "chunk_id": "doc_101", "text": "...", "score": 0.82 },
      { "chunk_id": "doc_102", "text": "...", "score": 0.61 },
      { "chunk_id": "doc_103", "text": "...", "score": 0.58 }
    ],
    "top_k": 2,
    "reference_answer": "30 calendar days",
    "score_type": "similarity"
  }
]
```

### Schema Field Reference
* **`id`** *(optional)*: A unique identifier for the trace (e.g. `case_01`).
* **`question`** *(required)*: The user question or query submitted to your pipeline.
* **`answer`** *(required)*: The actual generated answer your system returned.
* **`retrieved`** *(required)*: The list of context chunks injected into the generator's prompt. Each chunk contains:
  * `chunk_id` *(optional)*
  * `text` *(optional, but recommended)*: The text of the retrieved chunk.
  * `score` *(optional, but recommended)*: The retrieval or re-ranking score.
* **`candidates`** *(optional)*: The candidate chunks retrieved before top-$k$ truncation.
* **`top_k`** *(optional)*: The size of the context window fed to the LLM.
* **`reference_answer`** *(optional)*: The factual expected answer (enables reference-assisted diagnosis).
* **`score_type`** *(optional)*: `"similarity"` (default, dense cosine similarity), `"cosine_distance"`, or `"distance"`.

---

## 2. How to Sanitize and Redact Your Data First

> [!IMPORTANT]
> **NO CUSTOMER DATA OR SECRETS NEEDED.**  
> Never send real PII, credentials, private API keys, or confidential business records.

Before sending any cases, follow this 3-step checklist:

1. **Scrub Credentials and Secrets**:
   * Remove API keys, auth tokens, session cookies, database connection strings, and passwords.
2. **Anonymize Identifying Entities (PII & Business IP)**:
   * Replace customer names with `[Customer A]`, `[User 1]`, etc.
   * Replace company names with `[Company X]` or synthetic placeholders.
   * Mask internal hostnames, IP addresses, and private URLs (e.g., `https://internal.corp/doc` $\rightarrow$ `https://docs.example.com/item`).
3. **Preserve Linguistic Structure**:
   * Keep the grammatical relationship between the query and the chunk text intact. The failure diagnosis depends on lexical and heuristic relationships, not specific proprietary brand names.

---

## 3. What You Get Back

For every submission, we will analyze your traces and return an actionable post-mortem report containing:

1. **Definitive Root-Cause Attribution**:
   * **Retrieval Miss**: The retriever failed to pull relevant information into the candidate pool.
   * **Ranking Miss**: The relevant document was retrieved by the vector index, but truncated by top-$k$ ranking.
   * **Generation Failure**: Relevant context was present in the LLM prompt, but the model hallucinated, ignored context, or issued a false refusal.
   * **Appropriate / Missed Abstention**: The question was unanswerable from available documents, and the system either properly refused or improperly guessed.
2. **Confidence & Telemetry Evidence**:
   * Clear score gap dynamics, cutoff spreads, and evidence statements explaining why the failure occurred.
3. **Concrete Architecture Recommendations**:
   * Actionable suggestions (e.g. tuning prompt instructions, expanding top-$k$, adding hybrid BM25 retrieval, or adjusting refusal confidence bands).

---

## 4. How to Submit

You can submit your sanitized cases through any of the following channels:

* **GitHub Issue or Discussion**: Open an issue labeled `real-world-case` on the repository: [https://github.com/deveshkolte/RAGmortem/issues](https://github.com/deveshkolte/RAGmortem/issues).
* **Pull Request**: Add your sanitized case to `evals/real_cases/` using the labeling template in `evals/real_cases/README.md`.
* **Private Submission**: If you prefer direct communication, contact the maintainers via the email listed in `pyproject.toml` or repository profile.
