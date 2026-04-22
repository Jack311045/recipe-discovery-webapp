# Regression Module

## Purpose

The regression module adds an optional quality signal on top of semantic retrieval.

- Retrieval still finds candidate recipes by embedding similarity.
- Regression predicts recipe rating from structured numeric features.
- Ranking can combine similarity with predicted rating for additive reranking.

This is a safe extension of the existing retrieval flow, not a replacement.

## Target and Features

### Target column

- rating

### Optional confidence column

- num_ratings

### Default feature columns

- minutes
- n_steps
- n_ingredients
- calories
- total fat
- sugar
- sodium
- protein
- saturated fat
- carbohydrates

These names are aligned with the processed schema loaded from data/processed/Processed_data_updated2.csv.

## Model Choice

The baseline model is intentionally simple and reproducible:

- StandardScaler
- Ridge

Implementation lives in src/recipe_discovery/models/regression.py as RecipeRegressor.

Supported API:

- fit(x, y, sample_weight=None)
- predict(x)
- save(path)
- load(path)

## Training Flow

The training entrypoint is scripts/train_regression.py.

Pipeline steps:

1. Load config from configs/regression.yaml.
2. Load processed data.
3. Validate target and feature columns exist.
4. Build numeric table and drop rows missing required values.
5. Optionally filter by min_num_ratings.
6. Split train/validation/test.
7. Fit regressor.
8. Compute validation and test metrics.
9. Save artifacts.

## Evaluation Metrics

Regression summary returns:

- mae
- mse
- rmse
- r2

Implementation is in src/recipe_discovery/evaluation/regression_eval.py.

## Artifact Contract

Artifacts are written under data/artifacts/:

- regressor.joblib
- regression_metadata.json
- regression_holdout_predictions.csv

Metadata includes:

- model type and alpha
- target and feature columns
- split configuration and random state
- row counts before and after filtering
- validation and test metrics
- artifact paths

## Retrieval Integration

Ranking helpers are in src/recipe_discovery/retrieval/ranker.py.

Official retrieval runtime remains similarity-first in RetrievalService.search.
Optional additive reranking is available through RetrievalService.search_with_optional_rerank.

- attach_scores keeps similarity-only ordering.
- attach_predicted_ratings adds predicted_rating when a model is provided.
- compute_combined_ranking computes additive combined_score.

First-pass combined formula:

combined_score = similarity_weight * normalized_similarity_score + rating_weight * normalized_predicted_rating

If no regression model is provided, ranking safely falls back to similarity-based ordering.

Required artifacts for retrieval-side reranking:

- data/artifacts/regressor.joblib (or explicit custom model path)
- data/artifacts/regression_metadata.json for feature_columns guidance
