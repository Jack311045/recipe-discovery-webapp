"""Train the k-means clustering model on recipe embeddings.

Mirrors the pattern established by ``scripts/train_regression.py``:

- Load a frozen-dataclass config from ``configs/clustering.yaml``.
- Load the embedding matrix saved by ``scripts/build_embeddings.py``.
- Fit :class:`recipe_discovery.clustering.kmeans.KMeans`.
- Persist the joblib artifact and a metadata JSON (config echo, inertia, cluster
  sizes, silhouette, timing).

Usage
-----
    python scripts/train_kmeans.py
    python scripts/train_kmeans.py --config-path configs/clustering.yaml
    python scripts/train_kmeans.py --n-clusters 12 --overwrite
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recipe_discovery.clustering.kmeans import KMeans
from recipe_discovery.evaluation.clustering_eval import cluster_sizes, silhouette_summary
from recipe_discovery.settings import CONFIG_DIR
from recipe_discovery.utils.io import save_json

# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KMeansConfig:
    """Schema-aware configuration for k-means training."""

    n_clusters: int
    max_iter: int
    tol: float
    n_init: int
    init_method: str
    random_state: int
    device: str
    embeddings_path: Path
    output_path: Path
    metadata_path: Path
    silhouette_sample_size: int | None


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Clustering config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Clustering config must be a YAML mapping.")
    return payload


def load_kmeans_config(config_path: Path | None = None) -> KMeansConfig:
    """Load and validate the clustering config file."""
    config_file = Path(config_path) if config_path is not None else CONFIG_DIR / "clustering.yaml"
    payload = _load_yaml(config_file)
    if "clustering" not in payload or not isinstance(payload["clustering"], dict):
        raise ValueError("Clustering config must include a top-level 'clustering' mapping.")
    block = payload["clustering"]

    required = ["n_clusters", "max_iter", "tol", "random_state", "output_path"]
    missing = [k for k in required if k not in block]
    if missing:
        raise ValueError(f"Missing required clustering config keys: {missing}")

    silhouette_sample_size = block.get("silhouette_sample_size", 5000)
    if silhouette_sample_size is not None:
        silhouette_sample_size = int(silhouette_sample_size)

    return KMeansConfig(
        n_clusters=int(block["n_clusters"]),
        max_iter=int(block["max_iter"]),
        tol=float(block["tol"]),
        n_init=int(block.get("n_init", 10)),
        init_method=str(block.get("init_method", "k-means++")),
        random_state=int(block["random_state"]),
        device=str(block.get("device", "auto")),
        embeddings_path=_resolve_repo_path(
            block.get("embeddings_path", "data/artifacts/recipe_embeddings.npy")
        ),
        output_path=_resolve_repo_path(block["output_path"]),
        metadata_path=_resolve_repo_path(
            block.get("metadata_path", "data/artifacts/kmeans_metadata.json")
        ),
        silhouette_sample_size=silhouette_sample_size,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def load_embeddings(path: Path) -> np.ndarray:
    """Load the embedding matrix produced by scripts/build_embeddings.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}. "
            "Run scripts/build_embeddings.py first to produce recipe_embeddings.npy."
        )
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {array.shape}")
    return array


def train_kmeans(
    embeddings: np.ndarray,
    config: KMeansConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit KMeans on ``embeddings`` and persist the model plus metadata.

    Returns the metadata dict written to disk.
    """
    if config.output_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing clustering artifact without --overwrite: "
            f"{config.output_path}"
        )

    model = KMeans(
        n_clusters=config.n_clusters,
        max_iter=config.max_iter,
        tol=config.tol,
        n_init=config.n_init,
        init_method=config.init_method,
        random_state=config.random_state,
        device=config.device,
    )

    start = time.perf_counter()
    model.fit(embeddings)
    elapsed = time.perf_counter() - start

    assert model.labels_ is not None
    assert model.inertia_ is not None
    assert model.n_iter_ is not None

    sizes = cluster_sizes(model.labels_)
    silhouette = silhouette_summary(
        embeddings,
        model.labels_,
        sample_size=config.silhouette_sample_size,
        random_state=config.random_state,
    )

    model.save(config.output_path)

    # Embedding norm diagnostics help us catch the L2-normalize mismatch early.
    norms = np.linalg.norm(embeddings, axis=1)

    metadata: dict[str, Any] = {
        "model_type": "from_scratch_pytorch_kmeans",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "n_clusters": config.n_clusters,
            "max_iter": config.max_iter,
            "tol": config.tol,
            "n_init": config.n_init,
            "init_method": config.init_method,
            "random_state": config.random_state,
            "device": config.device,
        },
        "input": {
            "embeddings_path": str(config.embeddings_path),
            "n_samples": int(embeddings.shape[0]),
            "n_features": int(embeddings.shape[1]),
            "embedding_norms": {
                "min": float(norms.min()),
                "max": float(norms.max()),
                "mean": float(norms.mean()),
            },
        },
        "training": {
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "elapsed_seconds": float(elapsed),
        },
        "cluster_sizes": sizes,
        "silhouette": silhouette,
        "artifact_paths": {
            "model": str(config.output_path),
            "metadata": str(config.metadata_path),
        },
    }
    save_json(metadata, config.metadata_path)
    return metadata


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train k-means clustering model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_DIR / "clustering.yaml",
        help="Path to clustering config YAML.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Optional override for embeddings input path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional override for model artifact output path.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Optional override for number of clusters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clustering artifact if it exists.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_kmeans_config(args.config_path)
    if args.embeddings_path is not None:
        config = replace(config, embeddings_path=_resolve_repo_path(args.embeddings_path))
    if args.output_path is not None:
        config = replace(config, output_path=_resolve_repo_path(args.output_path))
    if args.n_clusters is not None:
        config = replace(config, n_clusters=int(args.n_clusters))

    embeddings = load_embeddings(config.embeddings_path)
    metadata = train_kmeans(embeddings, config, overwrite=args.overwrite)

    print(
        f"K-means training complete: K={config.n_clusters}, "
        f"n_samples={metadata['input']['n_samples']}, "
        f"n_iter={metadata['training']['n_iter']}, "
        f"inertia={metadata['training']['inertia']:.4f}, "
        f"elapsed={metadata['training']['elapsed_seconds']:.2f}s"
    )
    silhouette_score_value = metadata["silhouette"]["silhouette_score"]
    if silhouette_score_value is not None:
        print(
            f"Silhouette: {silhouette_score_value:.4f} "
            f"(sampled {metadata['silhouette']['sample_size']} points)"
        )
    print(f"Cluster sizes: {metadata['cluster_sizes']}")
    print(f"Saved model to {metadata['artifact_paths']['model']}")
    print(f"Saved metadata to {metadata['artifact_paths']['metadata']}")


if __name__ == "__main__":
    main()
