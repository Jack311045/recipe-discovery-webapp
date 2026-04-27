"""Build a recipe_id -> image URL map (Unsplash + optional fallback).

Usage:
    python scripts/prefetch_images.py \
        --data data/processed/Processed_data_updated2.csv \
        --output data/artifacts/image_map.parquet \
        --api-key YOUR_UNSPLASH_KEY
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from extract_image_keywords import extract_image_query

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from recipe_discovery.settings import DATA_PROCESSED_DIR  # noqa: E402

UNSPLASH_URL = "https://api.unsplash.com/search/photos"
FALLBACK_URL = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"
POLLINATIONS = "https://image.pollinations.ai/prompt/{prompt}?width=400&height=300&nologo=true"


def fetch_unsplash(query: str, api_key: str) -> str | None:
    try:
        resp = requests.get(
            UNSPLASH_URL,
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {api_key}"},
            timeout=8,
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [])
        if hits:
            return hits[0]["urls"]["small"]
    except Exception as exc:
        print(f"[WARN] {query}: {exc}")
    return None


def pollinations_url(recipe_name: str) -> str:
    import urllib.parse

    prompt = urllib.parse.quote(f"appetizing food photo of {recipe_name}, natural lighting")
    return POLLINATIONS.format(prompt=prompt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch recipe image URLs.")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output",
        default=str(DATA_PROCESSED_DIR / "image_map.parquet"),
        help="Output parquet path (default: data/processed/image_map.parquet).",
    )
    parser.add_argument("--api-key", default=None, help="Unsplash Client-ID (optional).")
    parser.add_argument("--batch-size", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.data)
    out = Path(args.output)

    if out.exists():
        existing = pd.read_parquet(out)
        done_ids = set(existing["recipe_id"].astype(str))
        df = df[~df["recipe_id"].astype(str).isin(done_ids)]
        print(f"Resuming: {len(df)} remaining / {len(done_ids)} already done")
    else:
        existing = pd.DataFrame(columns=["recipe_id", "image_url", "image_query"])

    records = []
    for i, row in df.iterrows():
        ingredients = row.get("ingredients")
        ingredients_list = ingredients if isinstance(ingredients, list) else []
        query = extract_image_query(str(row.get("name", "")), ingredients_list)

        if args.api_key:
            url = fetch_unsplash(query, args.api_key) or pollinations_url(str(row.get("name", "")))
        else:
            url = pollinations_url(str(row.get("name", "")))

        if not url:
            url = FALLBACK_URL

        records.append({"recipe_id": row.get("recipe_id"), "image_url": url, "image_query": query})
        if (i + 1) % args.batch_size == 0 and args.api_key:
            print("Sleeping 75s to respect rate limits...")
            time.sleep(75)

    new_df = pd.DataFrame(records)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_parquet(out, index=False)
    print(f"Saved {len(combined)} entries -> {out}")


if __name__ == "__main__":
    main()
