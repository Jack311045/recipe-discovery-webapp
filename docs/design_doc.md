# Design Document

## Repo structure

- `app/`: Streamlit frontend for search, exploration, and diagnostics
- `configs/`: central YAML configuration files
- `data/`: local-only raw, interim, processed, and artifact storage
- `docs/`: proposal and architecture notes
- `notebooks/`: exploration and experimentation
- `scripts/`: runnable training and preprocessing entrypoints
- `src/recipe_discovery/`: package source code
- `tests/`: smoke and unit tests

## Modules covered

- dense text embeddings
- cosine nearest-neighbor retrieval
- k-means clustering
- regression ranking
- PCA projection
- autoencoder projection
- optional classification
- evaluation and data-splitting utilities
