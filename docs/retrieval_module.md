# Retrieval Module

## Purpose

The retrieval module serves semantic recipe search using precomputed embedding artifacts.

Official runtime default remains similarity-first semantic search.

High-level flow:

1. Load processed metadata from data/processed/Processed_data_updated2.csv.
2. Load embedding artifacts from data/artifacts/recipe_embeddings.npy and data/artifacts/recipe_ids.csv.
3. Align metadata rows to embedding row order by recipe_id.
4. Encode the user query with the same embedding model family.
5. Compute cosine similarity against stored embeddings.
6. Build a larger candidate pool, apply filters, then return the final top_k rows.

## Artifact Dependencies

Retrieval depends on artifacts built by the embeddings pipeline:

1. data/artifacts/recipe_embeddings.npy
1. data/artifacts/recipe_ids.csv
1. data/artifacts/embedding_metadata.json (optional but used when present to reuse model_name/normalize)

Optional artifact not required at query time:

1. data/artifacts/recipe_texts.csv

Optional reranking artifacts:

1. data/artifacts/regressor.joblib
1. data/artifacts/regression_metadata.json (recommended source of feature column names)

Important contract:

1. recipe_embeddings.npy row i must correspond to recipe_ids.csv row i.
2. Retrieval never assumes processed CSV row order equals embedding row order.

## Metadata Alignment by recipe_id

Implementation location:

1. src/recipe_discovery/retrieval/service.py

Alignment logic:

1. Load processed metadata.
2. Normalize metadata recipe_id and artifact recipe_id to string.
3. Validate no duplicate recipe_id values in either side.
4. Left-join metadata to artifact IDs in artifact order.
5. Raise a clear error if any recipe_id in artifacts is missing from metadata.

Result:

1. metadata and embeddings are guaranteed row-aligned after load.

## Filtering with Processed CSV Schema

Implementation location:

1. src/recipe_discovery/retrieval/filters.py

The processed CSV has no single tags text column. Dietary filtering uses one-hot columns detected via:

1. src/recipe_discovery/data/schema.py:get_one_hot_tag_columns

Supported filter inputs:

1. dietary_filter: column-like terms such as vegetarian, vegan, gluten-free
2. max_time_minutes
3. max_ingredients

Filtering behavior:

1. Time filtering uses minutes directly when present.
2. If minutes is unavailable, time bucket one-hot columns (15/30/60/240 minutes) are used as fallback.
3. Ingredient filtering uses n_ingredients directly when present.
4. If n_ingredients is unavailable, 5-ingredients-or-less is used as fallback for max_ingredients <= 5.
5. Dietary filter resolves user text to one-hot columns and requires those column(s) to be 1.

## Search Path

Implementation location:

1. src/recipe_discovery/retrieval/service.py

Search algorithm:

1. Encode query text.
2. Compute cosine similarity against all embedding rows.
3. Select a candidate pool larger than top_k (currently max(top_k * 10, top_k)).
4. Rank candidates by similarity descending.
5. Apply metadata filters.
6. Re-sort filtered rows by similarity_score and return final top_k.

This avoids the common failure mode where filtering is applied only after truncating to top_k too early.

## Optional Regression Reranking Path

Implementation location:

1. src/recipe_discovery/retrieval/service.py
1. src/recipe_discovery/retrieval/ranker.py

Optional API path:

1. RetrievalService.search_with_optional_rerank

Behavior contract:

1. search remains the official default runtime path.
1. Reranking is additive and explicit opt-in only.
1. If no regression model is provided (or no path is passed), similarity-only ordering is preserved.
1. similarity_score is retained in output for transparency and debugging.

When to use reranking:

1. You already have semantically relevant candidates and want to bias ordering toward predicted recipe quality.
1. You can provide a trained regressor artifact plus valid feature columns present in candidate rows.

When to skip reranking:

1. No trained regressor is available.
1. Candidate rows do not include required regression feature columns.

## Index Usage Status

Current intended behavior is direct cosine scoring on the embedding matrix at query time.

Why this is intentional:

1. Candidate-pool and filtering logic currently operates on full score vectors.
2. This keeps ranking and filtering deterministic in one code path.
3. The saved index artifact is still generated and validated as a compatible acceleration path.

Operational note:

1. tests/test_retrieval.py includes a parity test to ensure saved-index neighbor order matches direct cosine order on the same vectors.

## Validation Modes

Validation is intentionally split between fast deterministic tests and real-runtime smoke checks.

1. Dummy/deterministic tests: tests/test_retrieval.py and tests/test_pipeline_smoke.py
2. Medium-scale real-runtime smoke: scripts/smoke_medium_pipeline.py (100-500 rows, real sentence-transformers runtime)

The deterministic tests keep CI fast and stable, while the medium smoke script validates realistic integration behavior before release milestones.

## MVP Completion Level

Data processing, embeddings, and retrieval are considered MVP-complete for downstream team integration.

1. Data/schema loading and canonical corpus generation are stable.
2. Embedding artifacts and alignment contracts are stable.
3. Retrieval runtime behavior is explicit (direct cosine), tested, and documented.
4. Saved index support is preserved as an optional optimization path with parity coverage.

## Similarity Utility

Implementation location:

1. src/recipe_discovery/retrieval/similarity.py

Cosine similarity remains the same mathematically, with added validation:

1. matrix must be 2D
2. query must be 1D (or a 1 x d row)
3. dimensions must match

## Key Files

1. src/recipe_discovery/retrieval/service.py
2. src/recipe_discovery/retrieval/filters.py
3. src/recipe_discovery/retrieval/similarity.py
4. src/recipe_discovery/data/load.py
5. src/recipe_discovery/data/schema.py
6. src/recipe_discovery/embeddings/store.py
