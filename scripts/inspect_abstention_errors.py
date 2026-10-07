import json
from pathlib import Path
from examples.reference_rag.app import ReferenceRagApp

HIDDEN_DOCS_DIR = Path("examples/hidden_holdout_rag/documents")
app = ReferenceRagApp(corpus_dir=HIDDEN_DOCS_DIR, mock_mode=True)

with open("evals/hidden_holdout_observed_diagnoses.jsonl") as f:
    records = [json.loads(line) for line in f]

errors = [r for r in records if r["ground_truth"] == "should_abstain" and r["predicted_diagnosis"] == "retrieval_miss"]
print(f"Total should_abstain -> retrieval_miss errors: {len(errors)}\n")

with open("evals/hidden_holdout_faults.jsonl") as f:
    faults = {r["case_id"]: r for r in (json.loads(line) for line in f)}

for i, e in enumerate(errors, 1):
    cid = e["case_id"]
    qid = e.get("question_id")
    fault = faults.get(cid, {})
    q = fault.get("question")
    ret_ids = fault.get("fault", {}).get("retrieved_ids")
    scores = fault.get("fault", {}).get("scores")
    
    # Audit corpus retrieval
    ret_chunks, ret_scores = app.retrieve(q, k=5)
    
    print("=" * 70)
    print(f"CASE {i}: {cid} (Question {qid})")
    print(f"Question: {q}")
    print(f"Runtime retrieved IDs: {ret_ids}")
    print(f"Runtime scores: {scores}")
    print(f"Diagnosis Signals: {e.get('signals')}")
    print(f"Diagnosis Reasons: {e.get('reasons')}")
    print("\nCorpus Audit Top 5 Chunks:")
    for rank, (c, s) in enumerate(zip(ret_chunks, ret_scores), 1):
        print(f"  Rank {rank}: {c.id} (score: {s:.4f})")
        print(f"    Text snippet: {c.text[:120]}...")
print("=" * 70)
