"""Build a TF-IDF lexical retrieval index from saved recipe texts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from recipe_discovery.embeddings.store import load_recipe_ids  # noqa: E402
from recipe_discovery.logging_utils import configure_logging  # noqa: E402
from recipe_discovery.retrieval.lexical import (  # noqa: E402
    DEFAULT_LEXICAL_INDEX_PATH,
    build_lexical_index,
    save_lexical_index,
)
from recipe_discovery.settings import ARTIFACTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lexical TF-IDF retrieval index.")
    parser.add_argument(
        "--recipe-texts-path",
        type=Path,
        default=ARTIFACTS_DIR / "recipe_texts.csv",
        help="Path to recipe_texts.csv from the embedding pipeline.",
    )
    parser.add_argument(
        "--recipe-ids-path",
        type=Path,
        default=ARTIFACTS_DIR / "recipe_ids.csv",
        help="Path to row-aligned recipe_ids.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_LEXICAL_INDEX_PATH,
        help="Path where lexical_index.joblib should be written.",
    )
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifact.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    if args.output_path.exists() and not args.overwrite:
        logger.warning(
            "Lexical index already exists at %s. Use --overwrite to regenerate.",
            args.output_path,
        )
        return

    logger.info("Loading recipe texts from %s", args.recipe_texts_path)
    recipe_texts = pd.read_csv(args.recipe_texts_path)
    if "recipe_text" not in recipe_texts.columns:
        raise ValueError("recipe_texts.csv must contain a 'recipe_text' column.")

    logger.info("Loading recipe IDs from %s", args.recipe_ids_path)
    recipe_ids = load_recipe_ids(args.recipe_ids_path)

    logger.info("Building lexical index.")
    index = build_lexical_index(
        recipe_texts["recipe_text"].fillna("").astype(str).tolist(),
        recipe_ids,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
    )
    save_lexical_index(index, args.output_path)

    logger.info(
        "Done. Saved lexical index for %d recipes to %s",
        len(recipe_ids),
        args.output_path,
    )


if __name__ == "__main__":
    main()
