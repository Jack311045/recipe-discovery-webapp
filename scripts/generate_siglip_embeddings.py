"""Build SigLIP text embeddings for image-based retrieval.

Usage:
    python scripts/generate_siglip_embeddings.py
    python scripts/generate_siglip_embeddings.py --batch-size 64
    python scripts/generate_siglip_embeddings.py --limit 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# Ensure repo root is on sys.path so ``recipe_discovery`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.data.corpus import build_corpus  # noqa: E402
from recipe_discovery.data.load import load_processed_recipes  # noqa: E402
from recipe_discovery.data.schema import ID_COLUMN  # noqa: E402
from recipe_discovery.embeddings.store import (  # noqa: E402
    save_embedding_metadata,
    save_embeddings,
    save_recipe_ids,
    save_recipe_texts,
)
from recipe_discovery.logging_utils import configure_logging  # noqa: E402
from recipe_discovery.settings import ARTIFACTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip-base-patch16-224"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigLIP recipe embeddings.")
    parser.add_argument("--input-path", type=Path, default=None, help="Path to processed CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact output dir.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="SigLIP model id.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Limit recipes for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else ARTIFACTS_DIR
    emb_path = output_dir / "recipe_embeddings_siglip.npy"

    if emb_path.exists() and not args.overwrite:
        logger.warning("SigLIP embeddings already exist at %s. Use --overwrite to regenerate.", emb_path)
        return

    logger.info("=== Step 1: Loading processed recipes ===")
    df = load_processed_recipes(args.input_path)
    if args.limit:
        logger.info("Limiting to %d recipes for testing.", args.limit)
        df = df.head(args.limit)

    logger.info("=== Step 2: Building recipe corpus ===")
    df = build_corpus(df)
    texts = df["recipe_text"].fillna("")
    recipe_ids = df[ID_COLUMN]

    logger.info("=== Step 3: Loading SigLIP model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    logger.info("=== Step 4: Encoding recipes ===")
    all_embeddings = []
    text_list = texts.tolist()

    with torch.no_grad():
        for i in range(0, len(text_list), args.batch_size):
            batch = text_list[i : i + args.batch_size]
            inputs = processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = model.get_text_features(**inputs)
            features = outputs.pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    if embeddings.shape[0] != len(recipe_ids):
        raise ValueError(
            f"Row mismatch: embeddings={embeddings.shape[0]}, ids={len(recipe_ids)}"
        )

    logger.info("=== Step 5: Saving artifacts ===")
    save_embeddings(embeddings, emb_path)
    save_recipe_ids(recipe_ids, output_dir / "recipe_ids_siglip.csv")
    save_recipe_texts(texts, output_dir / "recipe_texts_siglip.csv")
    save_embedding_metadata(
        model_name=args.model_id,
        embedding_dim=embeddings.shape[1],
        num_recipes=int(embeddings.shape[0]),
        normalize=True,
        extra={"encoder_type": "siglip"},
        path=output_dir / "embedding_metadata_siglip.json",
    )

    logger.info(
        "=== Done. %d SigLIP embeddings (dim=%d) saved to %s ===",
        embeddings.shape[0],
        embeddings.shape[1],
        output_dir,
    )


if __name__ == "__main__":
    main()
