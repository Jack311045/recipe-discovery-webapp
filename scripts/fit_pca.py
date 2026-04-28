"""Fit the PCA projection model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from recipe_discovery.reduction.pca import PCAReducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_pca(args: argparse.Namespace) -> None:
    """Run the PCA fitting and projection logic."""
    logger.info("Loading embeddings from %s", args.embeddings_path)
    if not Path(args.embeddings_path).exists():
        logger.error("Embeddings file not found: %s", args.embeddings_path)
        return

    embeddings = np.load(args.embeddings_path)
    
    reducer = PCAReducer(n_components=args.n_components)
    
    logger.info("Fitting PCA model...")
    reducer.fit(embeddings)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = output_dir / "pca_projector.pkl"
    reducer.save_checkpoint(weights_path)
    
    logger.info("Generating 2D PCA projections for all data...")
    projections_2d = reducer.transform(embeddings)
    
    proj_path = output_dir / "pca_projection.npy"
    np.save(proj_path, projections_2d)
    logger.info("Saved PCA projections to %s", proj_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the PCA Reducer")
    parser.add_argument("--embeddings_path", type=str, default="data/artifacts/recipe_embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="data/artifacts")
    parser.add_argument("--n_components", type=int, default=2)
    
    args = parser.parse_args()
    fit_pca(args)

if __name__ == "__main__":
    main()
