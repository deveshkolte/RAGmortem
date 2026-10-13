# Real-World Failure Cases & Blind Labeling Protocol

This directory collects real-world production RAG failure cases contributed by outside developers, teams, and real system logs.

To maintain strict scientific integrity, **all real-world cases must follow the blind pre-evaluation labeling protocol described below**.

---

## 1. Blind Labeling Protocol (MANDATORY)

> [!IMPORTANT]
> **A human annotator MUST determine and assign the TRUE failure cause BEFORE running RAGmortem.**  
> Never inspect RAGmortem's diagnostic output prior to establishing the human ground-truth label. This prevents confirmation bias and post-hoc rationalization.

### Labeling Steps:
1. **Audit the Corpus Manually**:
   * Inspect the question and the user's expected answer.
   * Search the broader document corpus independently (e.g., via full-text grep or database search) to see if the ground-truth document exists.
2. **Inspect the Retrieval Pool**:
   * Check whether the relevant chunk appeared in the candidate pool prior to top-$k$ truncation.
   * Check whether the relevant chunk appeared in the final top-$k$ context window given to the generator.
3. **Inspect Generator Behavior**:
   * If the context contained the answer, did the generator ignore it, hallucinate details, or produce a false refusal?
   * If the corpus lacked the answer, did the generator correctly refuse or fabricate an answer?
4. **Record the Ground-Truth Cause**:
   * Assign one of the standard failure categories before running any diagnostic algorithms.

---

## 2. Canonical Failure Causes

Assign one of the following root causes:

| Category | Description |
| :--- | :--- |
| **`retrieval_miss`** | The supporting document exists in the corpus, but the retriever failed to return it in the candidate pool. |
| **`ranking_miss`** | The supporting document was found in the candidate retrieval pool, but was truncated outside the top-$k$ context window by the ranker. |
| **`generation_failure`** | The supporting evidence was present in the top-$k$ context prompt, but the LLM hallucinated, contradicted facts, or falsely refused. |
| **`should_abstain`** | The query is unanswerable from the corpus; the system hallucinated or answered when it should have abstained. |
| **`no_failure`** | The response is factually accurate or the system appropriately abstained when evidence was lacking. |
| **`borderline_ambiguous`** | The case cannot be conclusively isolated due to ambiguous phrasing or multiple interacting causes. |

---

## 3. Labeling Case Record Template

For every real case added to this directory, document the metadata in this format:

```markdown
### Case Record: `REAL-CASE-001`

* **Case ID**: `REAL-CASE-001`
* **Anonymized Source**: Enterprise SaaS Help Desk / Zendesk Ticketing RAG
* **Date Labeled**: 2026-10-13
* **Labeled By**: Jane Doe (jdoe@example.com / @janedoe)
* **True Cause (Human Determined)**: `ranking_miss`
* **How Determined**: 
  - Searched the full corpus for the refund SLA policy: found chunk `doc_refund_v2`.
  - Audited the candidate retrieval pool ($N=20$): `doc_refund_v2` was present at rank #7.
  - Context window was configured to top-$k=3$, so the chunk was pruned prior to generation.
* **Pre-Evaluation Notes**:
  - The generator emitted a hallucinated refund window (48 hours instead of 14 days) because it only saw unrelated billing snippets in top-3.
```

---

## 4. Master Registry Table

| Case ID | Anonymized Source / Domain | Human True Cause | How Determined | Labeled By | Date | Evaluated by RAGmortem? |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `REAL-CASE-TEMPLATE` | Example Enterprise Docs | `generation_failure` | Context had answer; model hallucinated | QA Lead | 2026-10-13 | No (Pending) |
