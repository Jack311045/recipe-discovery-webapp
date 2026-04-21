# Embeddings Module

## Purpose

The embeddings pipeline converts processed recipe records into dense vectors and persists artifacts used by retrieval and downstream ML tasks.

End-to-end responsibilities:

1. Load processed recipes from data/processed/Processed_data_updated2.csv
2. Build canonical recipe text from schema-aware fields
3. Encode text with sentence-transformers/all-MiniLM-L6-v2
4. Save row-aligned artifacts under data/artifacts
5. Build and save a cosine nearest-neighbor index artifact

## Core Files

1. src/recipe_discovery/data/load.py
2. src/recipe_discovery/data/schema.py
3. src/recipe_discovery/data/corpus.py
4. src/recipe_discovery/embeddings/encoder.py
5. src/recipe_discovery/embeddings/store.py
6. src/recipe_discovery/embeddings/index.py
7. scripts/build_embeddings.py
8. scripts/build_index.py

## Processed Schema Contract

The processed CSV is interpreted using schema helpers:

1. Core text columns: name, description, ingredients, steps
2. Metadata columns: minutes, n_steps, n_ingredients
3. One-hot tag columns: detected dynamically via get_one_hot_tag_columns
4. Unnamed columns: dropped by load_processed_recipes
5. all_reviews: not included in canonical text by default

Canonical recipe text is deterministic and omits literal nan strings.

## Artifact Contract

Primary artifacts:

1. data/artifacts/recipe_embeddings.npy
2. data/artifacts/recipe_ids.csv
3. data/artifacts/embedding_metadata.json
4. data/artifacts/recipe_texts.csv (optional debug artifact)
5. data/artifacts/recipe_index.joblib

Critical invariant:

1. Row i in recipe_embeddings.npy aligns with row i in recipe_ids.csv (and recipe_texts.csv when present).

## Runtime Retrieval Relationship

Official v1 runtime retrieval path is direct cosine scoring over the loaded embedding matrix.

1. RetrievalService does not use recipe_index.joblib as the default online path.
2. recipe_index.joblib is a validated optional optimization artifact.
3. Index and direct cosine order consistency is covered by tests.

This design is intentional to keep candidate-pool ranking and filtering in a single deterministic runtime path.

## Validation Modes

Validation intentionally combines deterministic unit/integration tests with real-runtime smoke checks.

Deterministic (fast) coverage:

1. tests/test_data_schema.py
2. tests/test_embeddings_pipeline.py
3. tests/test_retrieval.py
4. tests/test_pipeline_smoke.py
5. tests/test_negative_paths.py

Real-runtime medium-scale coverage:

1. scripts/smoke_medium_pipeline.py (100-500 rows, real model runtime)

## MVP Completion Level

Data processing, embeddings, and retrieval are now stable as the course-project MVP foundation.

Completed:

1. Schema-aware loading and corpus generation
2. Embedding artifact generation and reloadability
3. Retrieval loading, alignment, ranking, and filtering behavior
4. Negative-path error handling for missing files and bad paths

Operational limitations (not correctness blockers):

1. Real-model smoke runs are heavier than deterministic tests and are intended for milestone checks, not every quick edit loop.
2. Online retrieval still uses direct cosine, with index kept as optional optimization.
