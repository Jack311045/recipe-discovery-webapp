"""Medium-scale real-runtime smoke check for data, embeddings, and retrieval."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.data.corpus import build_corpus  # noqa: E402
from recipe_discovery.data.load import load_processed_recipes  # noqa: E402
from recipe_discovery.data.schema import ID_COLUMN  # noqa: E402
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder  # noqa: E402
from recipe_discovery.embeddings.index import build_index, load_index, save_index  # noqa: E402
from recipe_discovery.embeddings.store import (  # noqa: E402
    load_embeddings,
    load_recipe_ids,
    save_embeddings,
    save_recipe_ids,
)
from recipe_discovery.retrieval.service import RetrievalRequest, RetrievalService  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a medium-scale real-runtime smoke check for the MVP pipeline."
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=200,
        help="Row count from processed CSV to validate (recommended 100-500).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum rows returned per example query.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for the real model runtime.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model identifier.",
    )
    return parser.parse_args(argv)



def _validate_subset_size(subset_size: int) -> int:
    """Enforce practical bounds requested for the medium-scale smoke pass."""
    if subset_size < 100:
        raise ValueError("subset-size must be at least 100 for medium-scale validation.")
    if subset_size > 500:
        raise ValueError("subset-size must be at most 500 for medium-scale validation.")
    return subset_size



def main(argv: list[str] | None = None) -> None:
    """Run a medium-scale real-artifact integration check end-to-end."""
    args = parse_args(argv)
    subset_size = _validate_subset_size(args.subset_size)

    print(f"Loading processed recipes (first {subset_size} rows)...")
    processed = load_processed_recipes().head(subset_size).copy()
    if len(processed) != subset_size:
        raise RuntimeError(f"Expected {subset_size} rows, found {len(processed)}.")

    print("Building canonical corpus text...")
    corpus_df = build_corpus(processed)
    if "recipe_text" not in corpus_df.columns:
        raise RuntimeError("Corpus build failed: recipe_text column missing.")

    encoder = RecipeEncoder(
        EmbeddingConfig(model_name=args.model_name, batch_size=args.batch_size, normalize=True)
    )
    print(f"Loading embedding model: {args.model_name}")
    encoder.load()

    print("Encoding corpus with real model runtime...")
    embeddings = encoder.encode(corpus_df["recipe_text"].tolist(), show_progress=False)
    if embeddings.shape[0] != len(corpus_df):
        raise RuntimeError(
            "Embedding row count mismatch: "
            f"embeddings={embeddings.shape[0]}, corpus={len(corpus_df)}"
        )

    with tempfile.TemporaryDirectory(prefix="recipe_pipeline_medium_") as tmp_dir:
        tmp = Path(tmp_dir)
        processed_path = tmp / "processed_subset.csv"
        emb_path = tmp / "recipe_embeddings.npy"
        ids_path = tmp / "recipe_ids.csv"
        idx_path = tmp / "recipe_index.joblib"

        corpus_df.to_csv(processed_path, index=False)
        save_embeddings(embeddings, emb_path)
        save_recipe_ids(corpus_df[ID_COLUMN], ids_path)

        print("Building and validating saved index artifact...")
        index = build_index(embeddings, n_neighbors=min(10, len(corpus_df)))
        save_index(index, idx_path)
        loaded_index = load_index(idx_path)
        distances, neighbors = loaded_index.kneighbors(
            embeddings[:1],
            n_neighbors=min(5, len(corpus_df)),
        )
        if distances.shape[0] != 1 or neighbors.shape[0] != 1:
            raise RuntimeError("Index kneighbors output shape is invalid.")

        print("Loading RetrievalService against generated artifacts...")
        service = RetrievalService()
        service.load(
            processed_path=processed_path,
            embeddings_path=emb_path,
            recipe_ids_path=ids_path,
        )

        if service.embeddings is None or service.metadata is None:
            raise RuntimeError("RetrievalService did not initialize embeddings/metadata.")

        loaded_embeddings = load_embeddings(emb_path)
        loaded_ids = load_recipe_ids(ids_path)
        if len(service.metadata) != loaded_embeddings.shape[0]:
            raise RuntimeError("Service metadata and embeddings are not row-aligned.")

        actual_ids = service._normalize_recipe_ids(service.metadata[ID_COLUMN])
        expected_ids = service._normalize_recipe_ids(loaded_ids)
        if actual_ids.tolist() != expected_ids.tolist():
            raise RuntimeError(
                "Service metadata alignment does not match artifact recipe_id order."
            )

        print("Running example retrieval queries with filters...")
        requests = [
            RetrievalRequest(query="quick vegetarian dinner", top_k=args.top_k),
            RetrievalRequest(query="easy vegan meal", top_k=args.top_k, dietary_filter="vegan"),
            RetrievalRequest(
                query="gluten free quick",
                top_k=args.top_k,
                dietary_filter="gluten-free",
                max_time_minutes=45,
                max_ingredients=10,
            ),
        ]

        for req in requests:
            result = service.search(req)
            if "similarity_score" not in result.columns:
                raise RuntimeError("Retrieval output missing similarity_score column.")
            if len(result) > req.top_k:
                raise RuntimeError(
                    f"Retrieval returned {len(result)} rows, exceeding top_k={req.top_k}."
                )
            if not result.empty:
                if req.max_time_minutes is not None and "minutes" in result.columns:
                    if not (result["minutes"] <= req.max_time_minutes).all():
                        raise RuntimeError("max_time_minutes constraint violated in output.")
                if req.max_ingredients is not None and "n_ingredients" in result.columns:
                    if not (result["n_ingredients"] <= req.max_ingredients).all():
                        raise RuntimeError("max_ingredients constraint violated in output.")

        print("Medium-scale smoke check passed.")
        print(
            "Summary: "
            f"rows={len(corpus_df)}, dim={embeddings.shape[1]}, "
            f"artifact_dir={tmp}"
        )


if __name__ == "__main__":
    main()
