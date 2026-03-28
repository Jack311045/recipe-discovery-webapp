"""Run the end-to-end preprocessing pipeline."""

from __future__ import annotations

from recipe_discovery.data.load import load_raw_tables
from recipe_discovery.data.preprocess import preprocess_tables


def main() -> None:
    recipes, interactions = load_raw_tables()
    processed = preprocess_tables(recipes=recipes, interactions=interactions)
    print(f"Processed table shape: {processed.shape}")


if __name__ == "__main__":
    main()
