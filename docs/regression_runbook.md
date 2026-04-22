# Regression Runbook

## Prerequisites

- Python environment is active.
- Dependencies are installed from requirements.txt.
- Processed data exists at data/processed/Processed_data_updated2.csv.

## Default Training Command

Run from repository root:

python scripts/train_regression.py --overwrite

Optional arguments:

- --config-path <path>
- --input-path <path>
- --output-path <path>
- --overwrite

## Expected Outputs

After a successful run:

- data/artifacts/regressor.joblib
- data/artifacts/regression_metadata.json
- data/artifacts/regression_holdout_predictions.csv

Console logs include split sizes and validation/test metrics.

## Inspecting Results

Quick checks:

1. Open data/artifacts/regression_metadata.json and verify target_column, feature_columns, row_counts, and metrics.
2. Open data/artifacts/regression_holdout_predictions.csv to inspect y_true and y_pred.
3. Confirm artifact timestamps were updated by the current run.

## Common Failure Cases

### Schema mismatch

Symptoms:

- Error about missing target column
- Error about missing feature columns

Fix:

- Update configs/regression.yaml to match processed schema names.
- Re-run training.

### Missing processed CSV

Symptoms:

- FileNotFoundError for processed path

Fix:

- Ensure data/processed/Processed_data_updated2.csv exists.
- Or pass --input-path to the correct file.

### Wrong config column names

Symptoms:

- Explicit validation error listing unknown feature names

Fix:

- Correct feature names in configs/regression.yaml.

### NaN-heavy rows

Symptoms:

- Very small train/val/test counts
- Empty split error after filtering

Fix:

- Inspect missingness in target/features.
- Reduce strict filtering or improve data quality.

## Reranking Debug Steps

1. Load candidate rows with similarity_score.
2. Load the regressor artifact.
3. Call compute_combined_ranking with feature columns and weights.
4. Verify predicted_rating and combined_score appear.
5. If no model is provided, confirm fallback still ranks by similarity.

## Minimal Local Verification

Run targeted regression tests:

python -m pytest -q tests/test_regression.py

This validates model persistence, schema checks, data filtering, metrics, artifacts, and ranking integration behavior.
