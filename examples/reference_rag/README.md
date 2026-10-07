# Reference RAG Application

This directory contains a self-contained, real reference Retrieval-Augmented Generation (RAG) pipeline built to demonstrate the `RagAdapter` contract and serve as the baseline application for failure diagnosis in RAGmortem.

## Architecture

The pipeline consists of the following components:

```
Markdown Documents (documents/)
         │
         ▼
Regex Chunk Parser (## [chunk_id] Title)
         │
         ▼
SentenceTransformer Embeddings (all-MiniLM-L6-v2)
         │
         ▼
NumPy Normalized Cosine Retrieval (top-k)
         │
         ▼
Prompt Assembly (Context + Question + Guardrails)
         │
         ▼
Local SHA-256 Response Cache (.ragmortem_cache/)
    ├── Hit  ──► Return Cached RagResult
    └── Miss ──► Groq API (qwen/qwen3.8-27b) or Local Mock Generator
```

## Corpus

The reference corpus in `documents/` is a synthetic, technical policy handbook covering:
- **`incident_management.md`**: SLAs for P1-P4 incidents, Incident Commander authority, postmortem timelines, on-call schedules, and escalation windows.
- **`deployment_and_infrastructure.md`**: Deployment freeze windows, canary soak times, automated rollback thresholds, CI/CD timeouts, database retention, RTO/RPO targets, and mTLS.
- **`security_and_compliance.md`**: Vulnerability patching SLAs (CVSS >= 9.0), credential rotation, FIDO2 MFA hardware keys, audit log retention, and data encryption.

Each chunk is explicitly delimited with a unique identifier:
```markdown
## [chunk_id] Section Title
Paragraph text containing clear, unambiguous facts.
```

## Running the Reference RAG

### 1. In Deterministic Mock Mode (No API key needed)
```bash
ragmortem run-reference --mock
```

### 2. Using Groq Free API
Set your Groq API key:
```bash
export GROQ_API_KEY="gsk_..."
ragmortem run-reference
```

### 3. As a Python Library
```python
from examples.reference_rag.app import ReferenceRagApp

app = ReferenceRagApp(top_k=3, mock_mode=False)
result = app.query("What is the mandatory engineering response time for a P1 incident?")

print("Answer:", result.answer)
print("Retrieved Chunks:", result.retrieved_ids)
print("Scores:", result.scores)
```
