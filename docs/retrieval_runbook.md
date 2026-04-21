# Retrieval Runbook

## Preconditions

Run all commands from repository root.

Required inputs:

1. data/processed/Processed_data_updated2.csv
2. data/artifacts/recipe_embeddings.npy
3. data/artifacts/recipe_ids.csv

Optional but recommended:

1. data/artifacts/embedding_metadata.json

Install dependencies:

1. pip install -r requirements.txt

## Build or Refresh Artifacts

If artifacts are missing or stale:

1. python scripts/build_embeddings.py --overwrite
2. python scripts/build_index.py --overwrite

## Run Retrieval Smoke Script

Use the lightweight smoke script:

1. python scripts/smoke_retrieval.py

Optional custom queries:

1. python scripts/smoke_retrieval.py --top-k 5 --query "quick vegan dinner" --query "easy pasta" --query "high protein"

Expected behavior:

1. Service loads successfully.
2. Query results include similarity_score.
3. Results are sorted descending by similarity.

## Run Medium-Scale Real-Runtime Smoke

Use this before major team handoffs to validate a nontrivial real subset end-to-end.

1. python scripts/smoke_medium_pipeline.py --subset-size 200 --top-k 5

What it validates:

1. Real processed subset loading (100-500 rows)
2. Canonical corpus construction
3. Real model embedding generation
4. Saved embedding/index artifacts
5. RetrievalService.load alignment by recipe_id
6. Query + filter behavior without crashes

## Run Retrieval Tests

1. pytest tests/test_retrieval.py

Covered checks include:

1. Processed CSV loading
2. One-hot tag detection
3. Embedding artifact loading
4. RetrievalService.load integration
5. Query ranking behavior
6. dietary_filter behavior on one-hot columns
7. max_time_minutes behavior
8. max_ingredients behavior
9. metadata/embedding row alignment after load
10. empty-result stability

## How Alignment Works

Service load does not trust raw row order.

It aligns metadata by recipe_id using this sequence:

1. Load metadata CSV.
2. Load embeddings and recipe_ids artifacts.
3. Validate row-count parity between embeddings and recipe_ids.
4. Join metadata onto artifact recipe_ids in artifact order.
5. Fail fast if artifact IDs are missing in metadata.

## How Filtering Works

Filtering runs on the candidate pool selected by similarity.

Order of operations:

1. Time filter (minutes, or one-hot time bucket fallback)
2. Ingredient filter (n_ingredients, or 5-ingredients-or-less fallback)
3. Dietary filter using one-hot tags from schema detection

Dietary notes:

1. Input is matched against one-hot tag column names (not a tags text field).
2. Multi-value input like "vegetarian, gluten-free" is supported.

## Direct Cosine vs Saved Index

Live RetrievalService search currently uses direct cosine scoring on the embedding matrix.

1. This is the intended final behavior for now because candidate-pool selection and filtering are applied in that path.
2. recipe_index.joblib remains a supported artifact for offline checks and future acceleration work.
3. Consistency between both paths is covered by retrieval tests.

Validation mode note:

1. Most pytest retrieval tests use deterministic encoders for repeatability.
2. scripts/smoke_medium_pipeline.py uses real sentence-transformers runtime for medium-scale integration confidence.

## Common Failure Cases and Debugging

1. Error: Processed CSV not found
Cause: data/processed/Processed_data_updated2.csv missing.
Fix: regenerate or place the processed CSV in data/processed.

2. Error: Embedding artifacts missing
Cause: build_embeddings was not run or artifacts were removed.
Fix: run python scripts/build_embeddings.py.

3. Error: embeddings/recipe_ids row mismatch
Cause: partial artifact refresh or manual edits.
Fix: rebuild embeddings so .npy and recipe_ids.csv are produced together.

4. Error: Missing artifact recipe_ids in processed metadata
Cause: processed CSV and artifacts came from different dataset versions.
Fix: rebuild embeddings from the same processed CSV currently in use.

5. Empty results for filtered query
Cause: filters are stricter than available candidate rows.
Fix: relax dietary_filter, max_time_minutes, or max_ingredients, or increase top_k.

6. Encoder load failure
Cause: sentence-transformers not installed or model download/network issue.
Fix: install requirements and retry in an environment with model download access.
