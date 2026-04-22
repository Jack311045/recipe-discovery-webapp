"""Tests for regression model."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.evaluation.regression_eval import regression_summary
from recipe_discovery.models.regression import RecipeRegressor
from recipe_discovery.retrieval.ranker import attach_predicted_ratings, compute_combined_ranking


def _load_train_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_regression.py"
    spec = importlib.util.spec_from_file_location("train_regression", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load train_regression module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_regression = _load_train_script_module()


def _make_synthetic_config(
    tmp_path: Path,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = "rating",
    min_num_ratings: int | None = None,
    use_sample_weight: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 7,
) -> train_regression.RegressionConfig:
    features = feature_columns or ["minutes", "protein"]
    return train_regression.RegressionConfig(
        target_column=target_column,
        feature_columns=features,
        min_num_ratings=min_num_ratings,
        use_sample_weight=use_sample_weight,
        alpha=1.0,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        output_path=tmp_path / "regressor.joblib",
        metadata_path=tmp_path / "regression_metadata.json",
        holdout_predictions_path=tmp_path / "regression_holdout_predictions.csv",
    )


def test_regressor_predict_shape() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = RecipeRegressor()
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape


def test_regressor_save_load_roundtrip(tmp_path: Path) -> None:
    x = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    y = np.array([1.2, 2.2, 3.3, 4.4])
    model = RecipeRegressor(alpha=0.5)
    model.fit(x, y)
    expected = model.predict(x)

    model_path = tmp_path / "regressor.joblib"
    model.save(model_path)
    loaded = RecipeRegressor.load(model_path)
    actual = loaded.predict(x)
    np.testing.assert_allclose(actual, expected)


def test_regressor_rejects_wrong_shaped_input() -> None:
    model = RecipeRegressor()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Expected 2D feature matrix"):
        model.fit(x, y)


def test_regressor_deterministic_small_sample_smoke() -> None:
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )
    y = np.array([1.1, 1.9, 3.0, 3.9, 5.2])

    model_a = RecipeRegressor(alpha=1.0).fit(x, y)
    model_b = RecipeRegressor(alpha=1.0).fit(x, y)

    preds_a = model_a.predict(x)
    preds_b = model_b.predict(x)
    np.testing.assert_allclose(preds_a, preds_b)


def test_regression_config_matches_processed_schema() -> None:
    config = train_regression.load_regression_config()
    assert config.target_column == "rating"
    assert config.feature_columns == [
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "total fat",
        "sugar",
        "sodium",
        "protein",
        "saturated fat",
        "carbohydrates",
    ]


def test_validate_regression_schema_success_on_processed_data() -> None:
    config = train_regression.load_regression_config()
    df = load_processed_recipes()
    train_regression.validate_regression_schema(df, config)


def test_validate_regression_schema_missing_feature_fails_clearly() -> None:
    df = pd.DataFrame({"rating": [4.0], "minutes": [20.0]})
    config = train_regression.RegressionConfig(
        target_column="rating",
        feature_columns=["minutes", "protein"],
        min_num_ratings=None,
        use_sample_weight=False,
        alpha=1.0,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        output_path=Path("data/artifacts/regressor.joblib"),
        metadata_path=Path("data/artifacts/regression_metadata.json"),
        holdout_predictions_path=Path("data/artifacts/regression_holdout_predictions.csv"),
    )

    with pytest.raises(ValueError, match="Regression feature columns not found"):
        train_regression.validate_regression_schema(df, config)


def test_validate_regression_schema_missing_target_fails_clearly() -> None:
    df = pd.DataFrame({"minutes": [20.0], "protein": [3.5]})
    config = train_regression.RegressionConfig(
        target_column="rating",
        feature_columns=["minutes", "protein"],
        min_num_ratings=None,
        use_sample_weight=False,
        alpha=1.0,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        output_path=Path("data/artifacts/regressor.joblib"),
        metadata_path=Path("data/artifacts/regression_metadata.json"),
        holdout_predictions_path=Path("data/artifacts/regression_holdout_predictions.csv"),
    )

    with pytest.raises(ValueError, match="Regression target column not found"):
        train_regression.validate_regression_schema(df, config)


def test_regression_summary_includes_required_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    summary = regression_summary(y_true, y_pred)

    assert set(summary.keys()) == {"mae", "mse", "rmse", "r2"}
    assert summary["mae"] >= 0.0
    assert summary["mse"] >= 0.0
    assert summary["rmse"] >= 0.0


def test_regression_summary_perfect_predictions() -> None:
    y_true = np.array([2.0, 2.5, 3.0, 4.0])
    summary = regression_summary(y_true, y_true.copy())

    assert summary["mae"] == pytest.approx(0.0)
    assert summary["mse"] == pytest.approx(0.0)
    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["r2"] == pytest.approx(1.0)


def test_regression_summary_noisy_predictions() -> None:
    y_true = np.array([2.0, 2.5, 3.0, 4.0])
    y_pred = np.array([1.0, 3.2, 2.0, 5.0])
    summary = regression_summary(y_true, y_pred)

    assert summary["mae"] > 0.0
    assert summary["mse"] > 0.0
    assert summary["rmse"] > 0.0
    assert summary["r2"] < 1.0


class _DummyPredictor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0] * 0.5 + x[:, 1] * 0.5


def test_attach_predicted_ratings_adds_column() -> None:
    df = pd.DataFrame(
        {
            "minutes": [10.0, 40.0, 20.0],
            "protein": [2.0, 4.0, 3.0],
            "similarity_score": [0.9, 0.7, 0.8],
        }
    )
    result = attach_predicted_ratings(df, _DummyPredictor(), ["minutes", "protein"])
    assert "predicted_rating" in result.columns
    assert len(result) == len(df)


def test_compute_combined_ranking_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "minutes": [10.0, 40.0, 20.0],
            "protein": [2.0, 4.0, 3.0],
            "similarity_score": [0.9, 0.7, 0.8],
        }
    )
    ranked_a = compute_combined_ranking(
        df,
        _DummyPredictor(),
        ["minutes", "protein"],
        similarity_weight=0.7,
        rating_weight=0.3,
    )
    ranked_b = compute_combined_ranking(
        df,
        _DummyPredictor(),
        ["minutes", "protein"],
        similarity_weight=0.7,
        rating_weight=0.3,
    )

    pd.testing.assert_frame_equal(ranked_a, ranked_b)
    assert ranked_a["combined_score"].is_monotonic_decreasing


def test_compute_combined_ranking_falls_back_without_model() -> None:
    df = pd.DataFrame(
        {
            "minutes": [10.0, 40.0, 20.0],
            "protein": [2.0, 4.0, 3.0],
            "similarity_score": [0.9, 0.7, 0.8],
        }
    )
    ranked = compute_combined_ranking(df, regression_model=None, feature_columns=None)
    assert "combined_score" in ranked.columns
    assert "predicted_rating" not in ranked.columns
    assert ranked["similarity_score"].tolist() == [0.9, 0.8, 0.7]


def test_build_regression_table_drops_missing_target_rows() -> None:
    df = pd.DataFrame(
        {
            "rating": [4.0, np.nan, 3.0, 5.0],
            "minutes": [10.0, 20.0, 30.0, 40.0],
            "protein": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = _make_synthetic_config(Path("."))

    table = train_regression.build_regression_table(df, config)
    assert table.rows_before_drop == 4
    assert table.rows_after_required_drop == 3
    assert table.y.shape[0] == 3


def test_build_regression_table_drops_missing_feature_rows() -> None:
    df = pd.DataFrame(
        {
            "rating": [4.0, 4.5, 3.0, 5.0],
            "minutes": [10.0, np.nan, 30.0, 40.0],
            "protein": [1.0, 2.0, np.nan, 4.0],
        }
    )
    config = _make_synthetic_config(Path("."))

    table = train_regression.build_regression_table(df, config)
    assert table.rows_before_drop == 4
    assert table.rows_after_required_drop == 2
    assert table.x.shape == (2, 2)


def test_build_regression_table_applies_min_num_ratings_filter() -> None:
    df = pd.DataFrame(
        {
            "rating": [4.0, 4.5, 3.0, 5.0],
            "minutes": [10.0, 20.0, 30.0, 40.0],
            "protein": [1.0, 2.0, 3.0, 4.0],
            "num_ratings": [1, 3, 10, 8],
        }
    )
    config = _make_synthetic_config(Path("."), min_num_ratings=5)

    table = train_regression.build_regression_table(df, config)
    assert table.rows_after_optional_filters == 2
    assert table.y.shape[0] == 2


def test_split_regression_table_has_non_empty_splits() -> None:
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "rating": rng.normal(4.0, 0.3, size=120),
            "minutes": rng.integers(5, 90, size=120).astype(float),
            "protein": rng.normal(10.0, 2.0, size=120),
        }
    )
    config = _make_synthetic_config(Path("."), test_size=0.1, val_size=0.1)

    table = train_regression.build_regression_table(df, config)
    splits = train_regression.split_regression_table(table, config)
    assert len(splits.y_train) > 0
    assert len(splits.y_val) > 0
    assert len(splits.y_test) > 0
    assert len(splits.y_train) + len(splits.y_val) + len(splits.y_test) == len(table.y)


def test_train_regression_creates_artifacts_and_metadata(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    rows = 80
    minutes = rng.integers(5, 120, size=rows).astype(float)
    protein = rng.normal(8.0, 2.0, size=rows)
    rating = 2.5 + 0.01 * minutes + 0.03 * protein
    df = pd.DataFrame(
        {
            "rating": rating,
            "minutes": minutes,
            "protein": protein,
            "num_ratings": rng.integers(1, 300, size=rows),
        }
    )
    config = _make_synthetic_config(
        tmp_path,
        min_num_ratings=2,
        use_sample_weight=True,
        test_size=0.1,
        val_size=0.1,
    )

    metadata = train_regression.train_regression(df, config, overwrite=True)

    assert config.output_path.exists()
    assert config.metadata_path.exists()
    assert config.holdout_predictions_path.exists()
    assert metadata["target_column"] == "rating"
    assert metadata["feature_columns"] == ["minutes", "protein"]
    assert "test_metrics" in metadata
    assert set(metadata["test_metrics"]) == {"mae", "mse", "rmse", "r2"}

    persisted = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    assert persisted["target_column"] == "rating"
    assert persisted["feature_columns"] == ["minutes", "protein"]
    assert "validation_metrics" in persisted
    assert "test_metrics" in persisted


def test_small_end_to_end_training_and_reranking_smoke(tmp_path: Path) -> None:
    df = load_processed_recipes().head(300).copy()
    config = train_regression.load_regression_config()
    config = train_regression.replace(
        config,
        output_path=tmp_path / "regressor.joblib",
        metadata_path=tmp_path / "regression_metadata.json",
        holdout_predictions_path=tmp_path / "regression_holdout_predictions.csv",
        test_size=0.1,
        val_size=0.1,
    )

    train_regression.train_regression(df, config, overwrite=True)
    assert config.output_path.exists()

    model = RecipeRegressor.load(config.output_path)
    candidate_df = df.loc[:, config.feature_columns].head(8).copy()
    candidate_df["similarity_score"] = np.linspace(0.95, 0.60, len(candidate_df))

    ranked = compute_combined_ranking(
        candidate_df,
        regression_model=model,
        feature_columns=config.feature_columns,
        similarity_weight=0.7,
        rating_weight=0.3,
    )
    assert "predicted_rating" in ranked.columns
    assert "combined_score" in ranked.columns
    assert len(ranked) == len(candidate_df)
