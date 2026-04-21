"""Lightweight smoke test for end-to-end retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.retrieval.service import RetrievalRequest, RetrievalService  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a lightweight retrieval smoke test.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of rows to print per query.")
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Query text (can be repeated). Defaults to three built-in sample queries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load retrieval service and run 2-3 example searches."""
    args = parse_args(argv)
    queries = args.query or [
        "quick vegetarian dinner",
        "high protein chicken",
        "easy dessert",
    ]

    service = RetrievalService()
    service.load()

    for idx, query in enumerate(queries, start=1):
        print(f"\n=== Query {idx}: {query} ===")
        result = service.search(RetrievalRequest(query=query, top_k=args.top_k))
        if result.empty:
            print("No results.")
            continue

        display_cols = [
            col
            for col in ["recipe_id", "name", "minutes", "n_ingredients", "similarity_score"]
            if col in result.columns
        ]
        print(result[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
