"""Build a cosine retrieval index from saved recipe embeddings.

Usage::

    python scripts/build_index.py
    python scripts/build_index.py --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.embeddings.index import build_index, save_index  # noqa: E402
from recipe_discovery.embeddings.store import load_embeddings  # noqa: E402
from recipe_discovery.logging_utils import configure_logging  # noqa: E402
from recipe_discovery.settings import ARTIFACTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval index from embeddings.")
    parser.add_argument(
        "--embedding-path", type=Path, default=None, help="Path to recipe_embeddings.npy."
    )
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Where to save the index."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    emb_path = Path(args.embedding_path) if args.embedding_path else ARTIFACTS_DIR / "recipe_embeddings.npy"
    idx_path = Path(args.output_path) if args.output_path else ARTIFACTS_DIR / "recipe_index.joblib"

    if idx_path.exists() and not args.overwrite:
        logger.warning("Index already exists at %s. Use --overwrite to regenerate.", idx_path)
        return

    # 1. Load embeddings
    logger.info("=== Step 1: Loading embeddings from %s ===", emb_path)
    embeddings = load_embeddings(emb_path)

    # 2. Build index
    logger.info("=== Step 2: Building cosine NN index ===")
    index = build_index(embeddings)

    # 3. Save index
    logger.info("=== Step 3: Saving index ===")
    save_index(index, idx_path)

    logger.info("=== Done. Index (%d vectors) saved to %s ===", embeddings.shape[0], idx_path)


if __name__ == "__main__":
    main()
