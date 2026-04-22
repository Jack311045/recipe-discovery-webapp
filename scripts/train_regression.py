"""Train the recipe regression model."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.evaluation.regression_eval import regression_summary
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.settings import CONFIG_DIR
from recipe_discovery.utils.io import save_json


@dataclass(frozen=True)
class RegressionConfig:
    """Schema-aware configuration for regression training."""

    target_column: str
    feature_columns: list[str]
    min_num_ratings: int | None
    use_sample_weight: bool
    alpha: float
    test_size: float
    val_size: float
    random_state: int
    output_path: Path
    metadata_path: Path
    holdout_predictions_path: Path


@dataclass(frozen=True)
class RegressionTable:
    """Prepared regression table after schema checks and row filtering."""

    x: np.ndarray
    y: np.ndarray
    source_indices: np.ndarray
    sample_weight: np.ndarray | None
    rows_before_drop: int
    rows_after_required_drop: int
    rows_after_optional_filters: int


@dataclass(frozen=True)
class RegressionSplits:
    """Train/validation/test arrays and original row indices."""

    x_train: np.ndarray
    y_train: np.ndarray
    idx_train: np.ndarray
    w_train: np.ndarray | None
    x_val: np.ndarray
    y_val: np.ndarray
    idx_val: np.ndarray
    w_val: np.ndarray | None
    x_test: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray
    w_test: np.ndarray | None


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Regression config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Regression config must be a YAML mapping.")
    return payload


def load_regression_config(config_path: Path | None = None) -> RegressionConfig:
    """Load and validate the regression config file."""
    config_file = Path(config_path) if config_path is not None else CONFIG_DIR / "regression.yaml"
    payload = _load_yaml(config_file)
    if "regression" not in payload or not isinstance(payload["regression"], dict):
        raise ValueError("Regression config must include a top-level 'regression' mapping.")
    block = payload["regression"]

    target_column = str(block.get("target_column", "")).strip()
    feature_columns = block.get("feature_columns", [])
    if not target_column:
        raise ValueError("Regression config 'target_column' must be a non-empty string.")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Regression config 'feature_columns' must be a non-empty list.")
    cleaned_features = [str(col).strip() for col in feature_columns if str(col).strip()]
    if not cleaned_features:
        raise ValueError("Regression config 'feature_columns' must contain valid names.")

    min_num_ratings_raw = block.get("min_num_ratings")
    min_num_ratings: int | None
    if min_num_ratings_raw is None:
        min_num_ratings = None
    else:
        min_num_ratings = int(min_num_ratings_raw)
        if min_num_ratings < 0:
            raise ValueError("Regression config 'min_num_ratings' must be >= 0 or null.")

    test_size = float(block.get("test_size", 0.1))
    val_size = float(block.get("val_size", 0.1))
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Regression config 'test_size' must be in (0, 1).")
    if val_size <= 0 or val_size >= 1:
        raise ValueError("Regression config 'val_size' must be in (0, 1).")
    if test_size + val_size >= 1:
        raise ValueError("Regression config requires test_size + val_size < 1.")

    return RegressionConfig(
        target_column=target_column,
        feature_columns=cleaned_features,
        min_num_ratings=min_num_ratings,
        use_sample_weight=bool(block.get("use_sample_weight", False)),
        alpha=float(block.get("alpha", 1.0)),
        test_size=test_size,
        val_size=val_size,
        random_state=int(block.get("random_state", 42)),
        output_path=_resolve_repo_path(block.get("output_path", "data/artifacts/regressor.joblib")),
        metadata_path=_resolve_repo_path(
            block.get("metadata_path", "data/artifacts/regression_metadata.json")
        ),
        holdout_predictions_path=_resolve_repo_path(
            block.get(
                "holdout_predictions_path",
                "data/artifacts/regression_holdout_predictions.csv",
            )
        ),
    )


def validate_regression_schema(df: pd.DataFrame, config: RegressionConfig) -> None:
    """Validate target and configured features against the processed DataFrame."""
    if config.target_column not in df.columns:
        raise ValueError(
            "Regression target column not found in processed data: "
            f"{config.target_column}"
        )

    missing_features = [col for col in config.feature_columns if col not in df.columns]
    if missing_features:
        joined = ", ".join(missing_features)
        raise ValueError(
            "Regression feature columns not found in processed data: "
            f"{joined}"
        )


def build_regression_table(df: pd.DataFrame, config: RegressionConfig) -> RegressionTable:
    """Build numeric regression arrays from processed data with safe filtering."""
    validate_regression_schema(df, config)

    include_num_ratings = config.min_num_ratings is not None or config.use_sample_weight
    if include_num_ratings and "num_ratings" not in df.columns:
        raise ValueError(
            "Regression config requires 'num_ratings' for filtering/weights but it is missing."
        )

    columns = [config.target_column, *config.feature_columns]
    if include_num_ratings:
        columns.append("num_ratings")

    table = df.loc[:, columns].copy()
    rows_before_drop = len(table)

    required = [config.target_column, *config.feature_columns]
    if include_num_ratings:
        required.append("num_ratings")
    table = table.dropna(subset=required).copy()
    rows_after_required_drop = len(table)

    if config.min_num_ratings is not None:
        table = table.loc[table["num_ratings"] >= config.min_num_ratings].copy()

    rows_after_optional_filters = len(table)
    if rows_after_optional_filters == 0:
        raise ValueError("No rows available for regression after filtering required columns.")

    try:
        x = table.loc[:, config.feature_columns].astype(float).to_numpy()
    except ValueError as exc:
        raise ValueError("Regression feature columns must be numeric.") from exc

    try:
        y = table.loc[:, config.target_column].astype(float).to_numpy()
    except ValueError as exc:
        raise ValueError("Regression target column must be numeric.") from exc

    sample_weight: np.ndarray | None = None
    if config.use_sample_weight:
        sample_weight = np.maximum(table["num_ratings"].astype(float).to_numpy(), 1.0)

    return RegressionTable(
        x=x,
        y=y,
        source_indices=table.index.to_numpy(dtype=int),
        sample_weight=sample_weight,
        rows_before_drop=rows_before_drop,
        rows_after_required_drop=rows_after_required_drop,
        rows_after_optional_filters=rows_after_optional_filters,
    )


def split_regression_table(table: RegressionTable, config: RegressionConfig) -> RegressionSplits:
    """Create deterministic train/validation/test splits."""
    split_inputs: list[np.ndarray] = [table.x, table.y, table.source_indices]
    if table.sample_weight is not None:
        split_inputs.append(table.sample_weight)

    test_split = train_test_split(
        *split_inputs,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    if table.sample_weight is None:
        x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test = test_split
        w_train_val = None
        w_test = None
    else:
        (
            x_train_val,
            x_test,
            y_train_val,
            y_test,
            idx_train_val,
            idx_test,
            w_train_val,
            w_test,
        ) = test_split

    val_fraction_of_train_val = config.val_size / (1.0 - config.test_size)

    train_val_inputs: list[np.ndarray] = [x_train_val, y_train_val, idx_train_val]
    if w_train_val is not None:
        train_val_inputs.append(w_train_val)

    val_split = train_test_split(
        *train_val_inputs,
        test_size=val_fraction_of_train_val,
        random_state=config.random_state,
    )

    if w_train_val is None:
        x_train, x_val, y_train, y_val, idx_train, idx_val = val_split
        w_train = None
        w_val = None
    else:
        x_train, x_val, y_train, y_val, idx_train, idx_val, w_train, w_val = val_split

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise ValueError(
            "Train/validation/test split produced an empty partition. "
            "Increase row count or adjust split fractions."
        )

    return RegressionSplits(
        x_train=x_train,
        y_train=y_train,
        idx_train=idx_train,
        w_train=w_train,
        x_val=x_val,
        y_val=y_val,
        idx_val=idx_val,
        w_val=w_val,
        x_test=x_test,
        y_test=y_test,
        idx_test=idx_test,
        w_test=w_test,
    )


def train_regression(
    df: pd.DataFrame,
    config: RegressionConfig,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
    save_holdout_predictions: bool = True,
) -> dict[str, Any]:
    """Train regression model and persist artifacts."""
    table = build_regression_table(df, config)
    splits = split_regression_table(table, config)

    model = RecipeRegressor(alpha=config.alpha)
    model.fit(splits.x_train, splits.y_train, sample_weight=splits.w_train)

    val_pred = model.predict(splits.x_val)
    test_pred = model.predict(splits.x_test)
    val_metrics = regression_summary(splits.y_val, val_pred)
    test_metrics = regression_summary(splits.y_test, test_pred)

    model_output = (
        _resolve_repo_path(output_path) if output_path is not None else config.output_path
    )
    if model_output.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing regression artifact without --overwrite: "
            f"{model_output}"
        )
    model.save(model_output)

    metadata: dict[str, Any] = {
        "model_type": "standard_scaler_plus_ridge",
        "target_column": config.target_column,
        "feature_columns": config.feature_columns,
        "alpha": config.alpha,
        "random_state": config.random_state,
        "test_size": config.test_size,
        "val_size": config.val_size,
        "min_num_ratings": config.min_num_ratings,
        "use_sample_weight": config.use_sample_weight,
        "row_counts": {
            "rows_before_drop": table.rows_before_drop,
            "rows_after_required_drop": table.rows_after_required_drop,
            "rows_after_optional_filters": table.rows_after_optional_filters,
            "train": int(len(splits.y_train)),
            "validation": int(len(splits.y_val)),
            "test": int(len(splits.y_test)),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifact_paths": {
            "model": str(model_output),
            "metadata": str(config.metadata_path),
            "holdout_predictions": str(config.holdout_predictions_path),
        },
    }
    save_json(metadata, config.metadata_path)

    if save_holdout_predictions:
        holdout = pd.DataFrame(
            {
                "row_index": splits.idx_test,
                "y_true": splits.y_test,
                "y_pred": test_pred,
            }
        )
        config.holdout_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        holdout.to_csv(config.holdout_predictions_path, index=False)

    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recipe regression model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_DIR / "regression.yaml",
        help="Path to regression config YAML.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional override for processed CSV path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional override for model artifact output path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model artifact if it exists.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_regression_config(args.config_path)
    if args.output_path is not None:
        config = replace(config, output_path=_resolve_repo_path(args.output_path))

    df = load_processed_recipes(args.input_path)
    metadata = train_regression(
        df,
        config,
        output_path=config.output_path,
        overwrite=args.overwrite,
        save_holdout_predictions=True,
    )

    print(
        "Regression training complete: "
        f"train={metadata['row_counts']['train']}, "
        f"val={metadata['row_counts']['validation']}, "
        f"test={metadata['row_counts']['test']}"
    )
    print(f"Validation metrics: {metadata['validation_metrics']}")
    print(f"Test metrics: {metadata['test_metrics']}")


if __name__ == "__main__":
    main()
