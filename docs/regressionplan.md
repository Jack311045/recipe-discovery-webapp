# regressionplan.md — Regression Module Implementation Plan
# Project: Recipe Discovery Web App

## 1. Objective

Implement the regression module for the existing repository structure of the Recipe Discovery Web App.

This module is responsible for:

1. loading processed recipe data from `data/processed/`
2. constructing a stable tabular feature matrix for each recipe
3. training a regression model to predict recipe rating as a ranking signal
4. saving the trained regression artifact and metadata under `data/artifacts/`
5. evaluating the model on held-out data
6. exposing a clean ranking helper so retrieval can optionally combine semantic similarity with predicted quality

This regression model is not a standalone academic exercise. Its role in the project is:

- semantic retrieval finds candidate recipes
- regression provides a rating-based reranking signal
- final results become more useful than raw cosine similarity alone

The implementation must match the current repository layout exactly.

---

## 2. Current Repo State and Required Corrections

The current repo already contains regression-related files, but they are incomplete or mismatched:

- `src/recipe_discovery/models/regression.py` currently wraps a simple Ridge model with `fit`, `predict`, and `save`.
- `scripts/train_regression.py` is still only a stub.
- `configs/regression.yaml` currently references schema names that do not match the processed dataset, including:
  - `avg_rating`
  - `fat`
  - `ingredient_count`
  - `step_count`
- `src/recipe_discovery/retrieval/ranker.py` currently only attaches similarity scores and does not yet combine regression predictions.
- `src/recipe_discovery/evaluation/regression_eval.py` currently only reports MAE and MSE.
- `tests/test_regression.py` currently contains only a minimal shape test.

The processed dataset schema already supports a clean first-pass regression setup using:
- target:
  - `rating`
- optional weighting:
  - `num_ratings`
- structured features:
  - `minutes`
  - `n_steps`
  - `n_ingredients`
  - `calories`
  - `total fat`
  - `sugar`
  - `sodium`
  - `protein`
  - `saturated fat`
  - `carbohydrates`

The regression implementation must correct the config/schema mismatch first before training logic is added.

---

## 3. Repository-Specific Scope

### In scope
Implement or update logic in:

- `configs/regression.yaml`
- `src/recipe_discovery/models/regression.py`
- `scripts/train_regression.py`
- `src/recipe_discovery/evaluation/regression_eval.py`
- `src/recipe_discovery/retrieval/ranker.py`
- `tests/test_regression.py`

Optional but allowed if needed:

- `src/recipe_discovery/data/schema.py`
- `src/recipe_discovery/data/load.py`
- `src/recipe_discovery/settings.py`
- `scripts/evaluate_all.py`
- `docs/regression_module.md`
- `docs/regression_runbook.md`

### Out of scope
Do not refactor unrelated modules.
Do not rewrite retrieval architecture.
Do not change clustering or dimensionality reduction modules.
Do not create a new package path outside `src/recipe_discovery/`.

---

## 4. Functional Goal

Given the processed CSV, the regression pipeline should:

1. load `data/processed/Processed_data_updated2.csv`
2. select a valid regression target and feature columns
3. build a numeric feature matrix
4. split data into train / validation / test
5. fit a baseline regression model
6. evaluate held-out performance
7. save the model artifact and metadata
8. expose a reranking helper that can later combine:
   - similarity score
   - predicted rating score

---

## 5. Modeling Choice for First Pass

Use a stable, interpretable baseline first.

### Default target
- `rating`

### Optional confidence signal
- `num_ratings`

This may be used later for:
- filtering noisy rows
- sample weighting
- diagnostics

### Default model
Use a scikit-learn pipeline with:
- `StandardScaler`
- `Ridge`

Reason:
- stable baseline
- robust on mixed-scale numeric features
- easy to save/load
- low bug risk
- good enough for a first reranking signal

### Do not start with
- deep models
- XGBoost / LightGBM unless already part of the repo
- embedding-only regression
- heavy feature engineering that will destabilize the pipeline

The first pass should prioritize reproducibility and correctness.

---

## 6. Schema-Aware Regression Setup

## 6.1 Target column
Use:
- `rating`

Do not use:
- `avg_rating` (not present in the current processed schema)

### Optional filtering
Rows with missing target should be excluded.
Optionally, rows with very low `num_ratings` may be excluded if the config enables that.

---

## 6.2 Default feature columns
Use the following structured numeric features when present:

- `minutes`
- `n_steps`
- `n_ingredients`
- `calories`
- `total fat`
- `sugar`
- `sodium`
- `protein`
- `saturated fat`
- `carbohydrates`

These match the current processed schema.

### Do not use these incorrect names
- `fat`
- `ingredient_count`
- `step_count`

Those should be replaced with:
- `total fat`
- `n_ingredients`
- `n_steps`

---

## 6.3 Optional future features
Do not include these in the first pass unless behind a config flag:

- one-hot tag columns
- embedding features
- text-derived features
- `all_reviews`

The initial regression model should stay simple and tabular.

---

## 7. File-by-File Responsibilities

## 7.1 `configs/regression.yaml`

Replace the current mismatched config with a schema-accurate one.

Expected contents should include at least:

- target column
- feature columns
- validation/test split fractions
- random seed
- ridge alpha
- minimum `num_ratings` threshold (optional)
- whether to use sample weights
- artifact output paths

Recommended structure:

```yaml
regression:
  target_column: rating
  feature_columns:
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
  min_num_ratings: null
  use_sample_weight: false
  alpha: 1.0
  test_size: 0.1
  val_size: 0.1
  random_state: 42
  output_path: data/artifacts/regressor.joblib
  metadata_path: data/artifacts/regression_metadata.json
  holdout_predictions_path: data/artifacts/regression_holdout_predictions.csv
Keep config names aligned with actual schema.

7.2 src/recipe_discovery/models/regression.py

This file should hold the reusable regression model wrapper.

Expected responsibilities:

define RecipeRegressor
internally use a scikit-learn pipeline:
StandardScaler
Ridge
expose:
fit(x, y, sample_weight=None)
predict(x)
save(path)
load(path) classmethod or staticmethod
optionally expose:
score(x, y) or get_params()

Requirements:

use NumPy arrays or pandas-compatible numeric input
save the full sklearn pipeline, not only the raw estimator
handle sample weights cleanly if enabled

Do not overcomplicate the wrapper.

7.3 scripts/train_regression.py

This is the main offline training entrypoint.

It must:

load config from configs/regression.yaml
load processed CSV from data/processed/Processed_data_updated2.csv
validate that target and feature columns exist
build the numeric regression table
drop rows with missing required values
optionally filter by min_num_ratings
split into train / validation / test
fit the regressor
evaluate on validation/test
save:
trained regressor
regression metadata JSON
optional holdout predictions CSV
print concise logs and metric summary
CLI behavior

It should support running from repo root:

python scripts/train_regression.py

Optional useful arguments:

--input-path
--config-path
--output-path
--overwrite
7.4 src/recipe_discovery/evaluation/regression_eval.py

Expand this module beyond the current minimal MAE/MSE summary.

Expected metrics:

MAE
MSE
RMSE
R²

Optional:

correlation between y_true and y_pred
calibration summary
metric table by rating bucket

At minimum, return a dictionary with:

mae
mse
rmse
r2

Keep implementation simple and deterministic.

7.5 src/recipe_discovery/retrieval/ranker.py

This file should evolve from “attach similarity scores” into a candidate ranking helper.

Expected responsibilities:

Keep existing helper
attaching similarity scores to a result table is still useful
Add new helper

Implement a function such as:

attach_predicted_ratings(df, model, feature_columns)
or
compute_combined_ranking(df, regression_model, feature_columns, similarity_weight=..., rating_weight=...)

The design should support:

similarity-only ranking if no regression model is loaded
optional reranking if a regression model is available
explicit combined score formula

Recommended first-pass formula:

combined_score =
    similarity_weight * normalized_similarity_score
  + rating_weight * normalized_predicted_rating

This should be documented clearly.

Do not silently change the official retrieval runtime path yet unless specifically enabled.
The first pass should create a clean reranking helper that retrieval can adopt safely.

7.6 tests/test_regression.py

Replace the minimal shape-only coverage with stronger tests.

Required tests:

model fit/predict returns correct output shape
model can save and reload successfully
regression config feature names match current schema
training script feature table excludes missing target rows
incorrect config columns fail with clear errors
evaluation summary returns MAE/MSE/RMSE/R²
reranking helper can attach predicted scores
combined ranking preserves deterministic ordering

Optional:

sample-weight path test
min_num_ratings filtering test
7.7 docs/regression_module.md

Create a technical module document explaining:

the role of regression in the project
why rating is the current target
what feature columns are used
how config maps to schema
what artifacts are produced
how regression integrates with retrieval ranking
7.8 docs/regression_runbook.md

Create a runbook explaining:

exact commands to run training
expected outputs
where artifacts are saved
how to inspect metrics
common failure cases:
schema mismatch
missing processed CSV
wrong config column names
NaN-heavy rows
how to debug reranking integration
8. Data Preparation Logic

The training script should explicitly implement the following table construction:

Required columns
rating
configured feature columns
Row filtering
drop rows where target is missing
drop rows where required feature columns are missing
Optional row filtering

If config enables:

drop rows where num_ratings < threshold
Sample weighting

If config enables and num_ratings exists:

use num_ratings or a transformed version of it as sample_weight

For first pass, this should be optional and clearly documented.

9. Train / Validation / Test Splits

Implement explicit data splitting.

Recommended first-pass split:

train: 80%
validation: 10%
test: 10%

Use a fixed random state for reproducibility.

For the first implementation, simple random split is acceptable.
Do not over-engineer strategic splits unless there is already repo support for them.

Validation should be used for:

sanity checking
hyperparameter selection if needed

Test should remain a final held-out estimate.

10. Artifact Contract

Save regression outputs under data/artifacts/.

Required artifacts:

regressor.joblib
regression_metadata.json

Optional but recommended:

regression_holdout_predictions.csv
Required metadata contents

Include at least:

model type
target column
feature columns
alpha
split sizes
random state
row counts
evaluation metrics

This metadata is important so downstream teammates can understand the model without re-reading training code.

11. Integration with Retrieval

The regression module should support retrieval reranking without destabilizing the already-finished retrieval architecture.

Current project architecture

Retrieval is currently stable and similarity-first.
Do not break that.

Safe integration approach

Implement reranking as an optional layer:

retrieval returns candidate rows with similarity_score
ranker computes predicted_rating
ranker computes combined_score
sorted results can optionally use combined_score
Important constraint

Do not force regression to become the only ranking signal.
It is an additive ranking signal, not a replacement for semantic relevance.

12. Acceptance Criteria

This task is complete only if all of the following are true:

python scripts/train_regression.py runs successfully from repo root
config columns match the actual processed schema
a valid regression feature table is built from processed data
rows with missing target/features are handled safely
the regression model trains successfully
validation/test metrics are computed
regressor.joblib is saved under data/artifacts/
regression_metadata.json is saved under data/artifacts/
regression evaluation returns MAE/MSE/RMSE/R²
tests cover save/load, config/schema consistency, and reranking helper behavior
docs clearly explain the model and artifact contract
regression integration does not break existing retrieval behavior
13. Non-Goals and Constraints

Do not:

refactor retrieval service architecture broadly
rewrite embeddings or pipeline modules
add heavy models or deep learning in the first pass
introduce fragile feature engineering
silently use columns that are not present in the processed schema

Do:

correct the config/schema mismatch first
keep the first model simple and reliable
produce readable artifacts and documentation
make reranking optional and explicit
14. Suggested Implementation Order
fix configs/regression.yaml
upgrade src/recipe_discovery/models/regression.py to a save/load-capable pipeline wrapper
expand src/recipe_discovery/evaluation/regression_eval.py
implement training data preparation in scripts/train_regression.py
save model + metadata + optional holdout predictions
extend src/recipe_discovery/retrieval/ranker.py
strengthen tests/test_regression.py
add docs/regression_module.md
add docs/regression_runbook.md
run end-to-end validation
15. Deliverable Style

The final implementation should look like production-quality starter infrastructure:

small focused functions
repo-root-safe path handling
explicit config/schema alignment
clear error messages
deterministic artifact paths
concise logging
readable docs
minimal-risk integration with retrieval
16. Final Repo-Specific Reminder

This regression module is being added on top of an already-stabilized:

data processing layer
embeddings pipeline
retrieval layer

Therefore the regression implementation must behave like a safe extension, not a disruptive refactor.

The main job is:

predict rating from structured recipe features
save a reusable artifact
expose a reranking signal for retrieval
document everything clearly