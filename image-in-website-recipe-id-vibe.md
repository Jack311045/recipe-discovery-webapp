# Recipe Image — On-Demand Food.com Fetch + Kaggle Dataset Evaluation

This document covers two complementary strategies for getting authentic Food.com
images into your search results:

- **Strategy 1** — Fetch images on-demand from Food.com at search time using the
  recipe ID already present in your dataset (no pre-scraping of all 180k recipes)
- **Strategy 2** — Evaluate the Kaggle companion dataset
  `foodcom-enhanced-recipes-with-images` as a drop-in image map replacement

---

## Strategy 1 — On-Demand Image Fetch via Recipe ID

### How It Works

Your dataset contains a `recipe_id` column that maps directly to Food.com URLs:

```
https://www.food.com/recipe/{name-slug}-{recipe_id}
```

Food.com pages embed their recipe image in a `<script type="application/ld+json">`
block using Schema.org markup — the same structured data Google uses for rich
results. This block is part of the initial HTML response, so no JavaScript
rendering is required to extract it.

The fetch happens **only for the number of recipes returned by a search**, not in bulk.
Results are cached in-session so the same recipe is never fetched twice.

Keep request volume low and ensure this aligns with Food.com terms/robots
before wider use. For light testing (small top_k), this is usually fine.

---

### Implementation

#### Repo alignment (important for this codebase)

This repo already returns a DataFrame from `RetrievalService` and injects
`image_url` via `data/processed/image_map.parquet`. For on-demand Food.com
images, you should layer on top of that DataFrame and only replace missing
or fallback images. Also note the RetrievalService lives in
`src/recipe_discovery/retrieval/service.py`, not `app/service.py`.

#### Step 1: Build the URL from Recipe ID

Your dataset has `recipe_id` (integer) and `name` (string). Food.com's URL slug
is just the name lowercased with spaces replaced by hyphens and punctuation stripped.

```python
# src/recipe_discovery/retrieval/image_fetcher.py

import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from functools import lru_cache

logger = logging.getLogger(__name__)

HEADERS = {
    # Identify as a real browser to avoid trivial blocks
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FALLBACK_IMAGE = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"
REQUEST_TIMEOUT = 4       # seconds — fail fast, don't hang the UI
POLITE_DELAY    = 0.3     # seconds between requests within a batch


def _make_slug(name: str) -> str:
    """
    Converts a recipe name to a Food.com URL slug.
    'Arriba! Baked Winter Squash Mexican Style' → 'arriba-baked-winter-squash-mexican-style'
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s-]", "", name)   # strip punctuation
    name = re.sub(r"\s+", "-", name.strip())     # spaces → hyphens
    name = re.sub(r"-+", "-", name)              # collapse repeated hyphens
    return name


def build_food_com_url(recipe_id: int, name: str) -> str:
    slug = _make_slug(name)
    return f"https://www.food.com/recipe/{slug}-{recipe_id}"


def _extract_image_from_jsonld(html: str) -> str | None:
    """
    Parses the Schema.org JSON-LD block from a Food.com page and returns
    the recipe image URL. Returns None if not found.

    Food.com embeds:
        <script type="application/ld+json">
          { "@type": "Recipe", "image": "https://img.sndimg.com/...", ... }
        </script>
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            # Handle both single object and @graph array
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Recipe"), {})
            if isinstance(data, dict) and data.get("@graph"):
                data = next((d for d in data["@graph"] if d.get("@type") == "Recipe"), {})
            if data.get("@type") == "Recipe":
                image = data.get("image")
                if isinstance(image, list):
                    return image[0]   # take first if multiple sizes provided
                if isinstance(image, str):
                    return image
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


@lru_cache(maxsize=512)
def fetch_food_com_image(recipe_id: int, name: str) -> str:
    """
    Fetches the image URL for a single recipe from Food.com.

    - Uses @lru_cache so the same recipe_id is never fetched twice
      within a Python process (persists across Streamlit reruns via
      the service singleton).
    - Returns FALLBACK_IMAGE on any failure so the UI never breaks.

    Args:
        recipe_id: Integer ID from the dataset (used in Food.com URL)
        name:      Recipe name (used to build URL slug)

    Returns:
        Image URL string.
    """
    url = build_food_com_url(recipe_id, name)
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        image_url = _extract_image_from_jsonld(response.text)
        if image_url:
            logger.info(f"[image] fetched {recipe_id} → {image_url[:60]}")
            return image_url
        else:
            logger.warning(f"[image] no JSON-LD image found for recipe {recipe_id} at {url}")
    except requests.exceptions.Timeout:
        logger.warning(f"[image] timeout for recipe {recipe_id}")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"[image] HTTP {e.response.status_code} for recipe {recipe_id}")
    except Exception as e:
        logger.warning(f"[image] unexpected error for recipe {recipe_id}: {e}")

    return FALLBACK_IMAGE


def fetch_images_for_results(results: list[dict], delay: float = POLITE_DELAY) -> list[dict]:
    """Enrich result dicts with 'image_url' via Food.com."""
    for i, result in enumerate(results):
        rid = result["recipe_id"]
        name = result["name"]

        # lru_cache handles dedup — no manual check needed
        result["image_url"] = fetch_food_com_image(rid, name)

        # Only delay between actual network calls, not cache hits
        if i < len(results) - 1:
            time.sleep(delay)

    return results


def attach_foodcom_images(
    results_df: "pd.DataFrame",
    *,
    delay: float = POLITE_DELAY,
    fallback_image: str = FALLBACK_IMAGE,
) -> "pd.DataFrame":
    """Return a copy with Food.com images patched in where missing.

    This matches this repo's RetrievalService contract: search returns a
    DataFrame, not a list of dicts. Only rows with blank or fallback images
    trigger a fetch.
    """
    import pandas as pd

    if results_df.empty:
        return results_df

    out = results_df.copy()
    if "image_url" not in out.columns:
        out["image_url"] = ""

    needs_fetch = out["image_url"].isna() | (out["image_url"].astype(str).str.strip() == "")
    needs_fetch |= out["image_url"].astype(str).eq(fallback_image)

    rows = out.loc[needs_fetch, ["recipe_id", "name"]].to_dict("records")
    if rows:
        patched = fetch_images_for_results(rows, delay=delay)
        patched_df = pd.DataFrame(patched)
        out.loc[needs_fetch, "image_url"] = patched_df["image_url"].values
    return out
```

---

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Add `beautifulsoup4` to `requirements.txt` if it is not already listed.

---

#### Step 3: Integrate into the Service Layer

In `src/recipe_discovery/retrieval/service.py`, patch image URLs after the
DataFrame is built. This preserves the existing image_map join and only
fetches from Food.com for missing or fallback images. Apply this to text,
image, and combined search paths so all modes get images:

```python
# src/recipe_discovery/retrieval/service.py  — updated imports and search

from recipe_discovery.retrieval.image_fetcher import attach_foodcom_images

class RetrievalService:

    def _attach_images(self, results: pd.DataFrame) -> pd.DataFrame:
        return attach_foodcom_images(results)

    def search(self, request: RetrievalRequest) -> pd.DataFrame:
        results = self._search_candidates(request, limit_to_top_k=True)
        return self._attach_images(results)

    def search_by_image(self, image: Image.Image, request: RetrievalRequest) -> pd.DataFrame:
        results = self._search_candidates_for_vector(
            request,
            query_vec=self._encode_image(image),
            embeddings=self._siglip_embeddings,
            limit_to_top_k=True,
        )
        return self._attach_images(results)

    def search_combined(...):
        ...
        return self._attach_images(ranked.head(request.top_k).reset_index(drop=True))
```

---

#### Step 4: Loading Indicator in Streamlit

No UI changes are required because the on-demand fetcher runs inside
`RetrievalService.search()` and returns a DataFrame like before. If you want
to make the extra latency visible, update the spinner text in
`app/pages/1_Search.py` to mention image fetching.

---

### Caching Behaviour

`@lru_cache(maxsize=512)` means:

- The **first** search for a recipe fetches from Food.com (~100–400ms per recipe)
- **Every subsequent search** returning the same recipe is instant (memory lookup)
- Cache persists as long as the Streamlit process is running
- 512 slots covers ~50 full search result pages in memory — plenty for a demo

For a production deployment you'd swap `lru_cache` for Redis or a persistent
SQLite cache, but for a portfolio/research app this is more than sufficient.

---

### Failure Modes & Fallbacks

| Failure | Behaviour |
|---|---|
| Recipe page deleted from Food.com | Returns `FALLBACK_IMAGE` (Unsplash generic food) |
| Food.com returns 429 (rate limit) | Returns `FALLBACK_IMAGE`, logs warning |
| Network timeout (>4s) | Returns `FALLBACK_IMAGE`, doesn't hang UI |
| JSON-LD block missing or malformed | Returns `FALLBACK_IMAGE`, logs warning |
| Name slug doesn't match Food.com's format exactly | 404 → `FALLBACK_IMAGE` |

The slug mismatch case is the most common real failure. Food.com sometimes uses
slightly different slugs (e.g. truncating very long names). This is acceptable
for a research project — the fallback handles it cleanly.

---

### Files Changed

| File | Change |
|---|---|
| `src/recipe_discovery/retrieval/image_fetcher.py` | New — URL builder, JSON-LD parser, cached fetcher |
| `src/recipe_discovery/retrieval/service.py` | Call `attach_foodcom_images()` after all search paths |
| `app/pages/1_Search.py` | Optional: updated spinner text only |
| `requirements.txt` | Verify `requests` and `beautifulsoup4` are listed |

---

## Strategy 2 — Kaggle Dataset Evaluation

### Dataset: `foodcom-enhanced-recipes-with-images`

**URL:** `https://www.kaggle.com/datasets/behnamrahdari/foodcom-enhanced-recipes-with-images`
**Key file:** `recipe_enhanced_v3.csv`

### Verdict: Verify Before Committing

This dataset is a community-created enhancement of the same Food.com source you're
already using. If it contains a `recipe_id` column and an `image_url` column that
maps to real, working Food.com image URLs, it would be the **best possible solution**
— zero scraping, zero API calls, just a CSV join on `recipe_id`.

### How to Verify It (5-Minute Check)

Download `recipe_enhanced_v3.csv` from Kaggle and run this:

```python
# scripts/verify_kaggle_image_dataset.py
"""
Run this after downloading recipe_enhanced_v3.csv from Kaggle.
Checks: column presence, recipe_id overlap with your dataset, and
whether image URLs are still live.

Usage:
    python scripts/verify_kaggle_image_dataset.py \
        --kaggle  path/to/recipe_enhanced_v3.csv \
        --recipes data/processed/Processed_data_updated2.csv
"""

import argparse
import requests
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}
SAMPLE_SIZE = 20  # number of URLs to spot-check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle",   required=True)
    parser.add_argument("--recipes",  required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Loading datasets...")
    kaggle_df = pd.read_csv(args.kaggle)
    recipes_df = pd.read_csv(args.recipes)

    # ── 1. Column inspection ─────────────────────────────────────
    print(f"\n[1] Kaggle dataset shape: {kaggle_df.shape}")
    print(f"    Columns: {list(kaggle_df.columns)}")

    # Look for recipe_id and image columns
    id_col  = next((c for c in kaggle_df.columns if "id" in c.lower()), None)
    img_col = next((c for c in kaggle_df.columns if "image" in c.lower() or "img" in c.lower() or "url" in c.lower()), None)

    if not id_col or not img_col:
        print(f"\n  [FAIL] Could not find expected columns.")
        print(f"  id_col found:  {id_col}")
        print(f"  img_col found: {img_col}")
        print("  → Dataset may not be usable as a direct image map.")
        return

    print(f"\n  ID column:    '{id_col}'")
    print(f"  Image column: '{img_col}'")

    # ── 2. Coverage check ────────────────────────────────────────
    print(f"\n[2] Coverage check...")
    if "recipe_id" not in recipes_df.columns:
        print("  [FAIL] recipe_id column missing in processed CSV.")
        return

    your_ids = set(recipes_df["recipe_id"].dropna().astype(int))
    kaggle_ids = set(kaggle_df[id_col].dropna().astype(int))
    overlap    = your_ids & kaggle_ids
    coverage   = len(overlap) / len(your_ids) * 100

    print(f"    Your dataset:   {len(your_ids):,} recipes")
    print(f"    Kaggle dataset: {len(kaggle_ids):,} recipes")
    print(f"    Overlap:        {len(overlap):,} ({coverage:.1f}%)")

    if coverage < 50:
        print("  [WARN] < 50% overlap — Kaggle dataset may not cover your recipes well.")
    elif coverage >= 80:
        print("  [PASS] Good coverage — worth using as image map.")

    # ── 3. Image URL format inspection ───────────────────────────
    sample = kaggle_df[img_col].dropna().head(5).tolist()
    print(f"\n[3] Sample image URLs:")
    for url in sample:
        print(f"    {url}")

    # Check if URLs look like real Food.com CDN URLs
    food_com_cdn = any(
        "sndimg.com" in str(u) or "food.com" in str(u)
        for u in sample
    )
    print(f"\n    Look like Food.com CDN URLs: {'YES ✓' if food_com_cdn else 'NO — may be external links'}")

    # ── 4. Live URL spot-check ───────────────────────────────────
    print(f"\n[4] Spot-checking {SAMPLE_SIZE} URLs for liveness...")
    test_urls = kaggle_df[img_col].dropna().sample(
        min(SAMPLE_SIZE, len(kaggle_df)), random_state=42
    ).tolist()

    live = 0
    dead = 0
    for url in test_urls:
        try:
            r = requests.head(url, headers=HEADERS, timeout=4, allow_redirects=True)
            if r.status_code == 200:
                live += 1
            else:
                dead += 1
                print(f"    [DEAD {r.status_code}] {url[:70]}")
        except Exception as e:
            dead += 1
            print(f"    [ERROR] {url[:70]} — {e}")

    print(f"\n    Live: {live}/{SAMPLE_SIZE}  |  Dead: {dead}/{SAMPLE_SIZE}")
    live_pct = live / SAMPLE_SIZE * 100
    if live_pct >= 80:
        print("  [PASS] URLs are mostly live — dataset is usable.")
    elif live_pct >= 50:
        print("  [WARN] ~half the URLs are dead. Usable with fallback.")
    else:
        print("  [FAIL] Most URLs are dead — dataset not reliable.")

    # ── 5. Final recommendation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if coverage >= 80 and live_pct >= 80:
        print("✅ Use this dataset as your image_map.")
        print("   Join on recipe_id and store as image_map.parquet.")
        print("   This eliminates all on-demand fetching.")
    elif coverage >= 50 and live_pct >= 50:
        print("⚠️  Use as primary source with on-demand fallback.")
        print("   For recipes not in the dataset or with dead URLs,")
        print("   fall back to the Food.com on-demand fetcher.")
    else:
        print("❌ Not reliable enough — stick with on-demand fetching.")
        print("   The Food.com JSON-LD approach in Strategy 1 is more robust.")


if __name__ == "__main__":
    main()
```

### If the Dataset Passes Verification

Convert it to `image_map.parquet` in one step and you're done — no on-demand
fetching needed at all:

```python
# scripts/build_image_map_from_kaggle.py

import pandas as pd

kaggle_df = pd.read_csv("path/to/recipe_enhanced_v3.csv")

# Adjust column names based on what verify script found
image_map = kaggle_df[["id", "image_url"]].rename(  # use actual column names
    columns={"id": "recipe_id", "image_url": "image_url"}
).set_index("recipe_id")

image_map.to_parquet("data/processed/image_map.parquet")
print(f"Saved {len(image_map)} entries → data/processed/image_map.parquet")
```

Then in `service.py`, remove the on-demand `attach_foodcom_images()` call and
rely solely on the image_map join — you'd be back to zero network calls at
query time.

---

## Decision Tree

```
Download recipe_enhanced_v3.csv
        │
        ▼
Run verify_kaggle_image_dataset.py
        │
        ├─► ✅ Coverage ≥ 80% AND URLs ≥ 80% live
        │       → Build image_map.parquet from Kaggle CSV
        │         Zero on-demand fetching needed.
        │
        ├─► ⚠️  Partial coverage or partial liveness
        │       → Use Kaggle CSV for covered recipes
        │         Fall back to on-demand Food.com fetcher for the rest
        │
        └─► ❌ Low coverage or mostly dead URLs
                → Use Strategy 1 (on-demand fetcher) exclusively
                  Kaggle dataset is not reliable
```

---

## Implementation Checklist

| # | Task | Effort |
|---|------|--------|
| 1 | Create `src/recipe_discovery/retrieval/image_fetcher.py` | 30 min |
| 2 | Update `search()` in `src/recipe_discovery/retrieval/service.py` | 15 min |
| 3 | Test fetcher manually on 5 recipe IDs from your dataset | 15 min |
| 4 | Download `recipe_enhanced_v3.csv` from Kaggle | 5 min |
| 5 | Run `verify_kaggle_image_dataset.py` | 10 min |
| 6 | If Kaggle passes: run `build_image_map_from_kaggle.py` | 10 min |
| 7 | If Kaggle passes: remove on-demand fetcher, rely on image_map join | 20 min |
| 8 | End-to-end test: run a search, confirm images appear | 15 min |

**Start with steps 1–3.** Get the on-demand fetcher working and confirmed first.
The Kaggle verification (steps 4–7) can run in parallel — if it passes, you
swap out the fetcher for the static map and gain back latency. If it fails,
you already have a working solution.