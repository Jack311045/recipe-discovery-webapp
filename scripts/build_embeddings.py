"""Build dense embeddings for the recipe corpus."""

from __future__ import annotations

from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder


def main() -> None:
    config = EmbeddingConfig()
    encoder = RecipeEncoder(config=config)
    print("Embedding stub ready. Implement model loading and batch encoding next.")


if __name__ == "__main__":
    main()
