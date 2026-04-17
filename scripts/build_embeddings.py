"""Build dense embeddings for the recipe corpus.

Usage::

    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --batch-size 64
    python scripts/build_embeddings.py --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``recipe_discovery`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.data.corpus import build_corpus  # noqa: E402
from recipe_discovery.data.load import load_processed_recipes  # noqa: E402
from recipe_discovery.data.schema import ID_COLUMN  # noqa: E402
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder  # noqa: E402
from recipe_discovery.embeddings.store import (  # noqa: E402
    save_embedding_metadata,
    save_embeddings,
    save_recipe_ids,
    save_recipe_texts,
)
from recipe_discovery.logging_utils import configure_logging  # noqa: E402
from recipe_discovery.settings import ARTIFACTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recipe embeddings.")
    parser.add_argument("--input-path", type=Path, default=None, help="Path to processed CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact output dir.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model identifier.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else ARTIFACTS_DIR
    emb_path = output_dir / "recipe_embeddings.npy"

    if emb_path.exists() and not args.overwrite:
        logger.warning("Embeddings already exist at %s. Use --overwrite to regenerate.", emb_path)
        return

    # 1. Load processed CSV
    logger.info("=== Step 1: Loading processed recipes ===")
    df = load_processed_recipes(args.input_path)

    # 2. Build canonical recipe texts
    logger.info("=== Step 2: Building recipe corpus ===")
    df = build_corpus(df)
    texts = df["recipe_text"]
    recipe_ids = df[ID_COLUMN]

    logger.info("Sample recipe text (first 500 chars):\n%s", texts.iloc[0][:500])

    # 3. Load encoder and generate embeddings
    logger.info("=== Step 3: Encoding recipes ===")
    config = EmbeddingConfig(model_name=args.model_name, batch_size=args.batch_size)
    encoder = RecipeEncoder(config=config)
    encoder.load()

    embeddings = encoder.encode(texts.tolist())

    # 4. Validate shapes
    assert embeddings.shape[0] == len(recipe_ids), (
        f"Row mismatch: embeddings={embeddings.shape[0]}, ids={len(recipe_ids)}"
    )

    # 5. Save artifacts
    logger.info("=== Step 4: Saving artifacts ===")
    save_embeddings(embeddings, emb_path)
    save_recipe_ids(recipe_ids, output_dir / "recipe_ids.csv")
    save_recipe_texts(texts, output_dir / "recipe_texts.csv")
    save_embedding_metadata(
        model_name=config.model_name,
        embedding_dim=encoder.embedding_dim,
        num_recipes=int(embeddings.shape[0]),
        normalize=config.normalize,
        path=output_dir / "embedding_metadata.json",
    )

    logger.info(
        "=== Done. %d embeddings (dim=%d) saved to %s ===",
        embeddings.shape[0],
        embeddings.shape[1],
        output_dir,
    )


if __name__ == "__main__":
    main()
