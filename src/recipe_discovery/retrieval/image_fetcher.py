"""On-demand Food.com image lookup for recipe results."""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FALLBACK_IMAGE_URL = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"
REQUEST_TIMEOUT = 4
POLITE_DELAY = 0.3
_PLACEHOLDER_TOKENS = ("logo", "placeholder", "default")


def _normalize_recipe_id(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\.0+$", "", text)


def _make_slug(name: str) -> str:
    """Convert a recipe name to a Food.com URL slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug


def build_food_com_url(recipe_id: object, name: str | None) -> str:
    slug = _make_slug(name or "")
    if not slug:
        slug = "recipe"
    recipe_id_text = _normalize_recipe_id(recipe_id)
    return f"https://www.food.com/recipe/{slug}-{recipe_id_text}"


def _extract_image_from_jsonld(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, list):
                data = next((item for item in data if item.get("@type") == "Recipe"), {})
            if isinstance(data, dict) and data.get("@graph"):
                data = next((item for item in data["@graph"] if item.get("@type") == "Recipe"), {})
            if isinstance(data, dict) and data.get("@type") == "Recipe":
                image = data.get("image")
                if isinstance(image, list) and image:
                    return str(image[0])
                if isinstance(image, str):
                    return image
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def _looks_like_placeholder(url: str) -> bool:
    lowered = url.lower().strip()
    if lowered.endswith(".svg"):
        return True
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


@lru_cache(maxsize=512)
def fetch_food_com_image(recipe_id: object, name: str | None) -> str:
    url = build_food_com_url(recipe_id, name)
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        image_url = _extract_image_from_jsonld(response.text)
        if image_url and not _looks_like_placeholder(image_url):
            logger.info("[image] fetched %s -> %s", recipe_id, image_url[:60])
            return image_url
        if image_url:
            logger.warning("[image] placeholder image for recipe %s at %s", recipe_id, url)
        else:
            logger.warning("[image] no JSON-LD image for recipe %s at %s", recipe_id, url)
    except requests.exceptions.Timeout:
        logger.warning("[image] timeout for recipe %s", recipe_id)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.warning("[image] HTTP %s for recipe %s", status, recipe_id)
    except Exception as exc:
        logger.warning("[image] unexpected error for recipe %s: %s", recipe_id, exc)

    return FALLBACK_IMAGE_URL


def fetch_images_for_results(
    results: list[dict],
    *,
    delay: float = POLITE_DELAY,
) -> list[dict]:
    """Enrich result dicts with Food.com image URLs."""
    for idx, result in enumerate(results):
        result["image_url"] = fetch_food_com_image(result["recipe_id"], result.get("name"))
        if idx < len(results) - 1:
            time.sleep(delay)
    return results


def attach_foodcom_images(
    results_df: "pandas.DataFrame",
    *,
    delay: float = POLITE_DELAY,
    fallback_image: str = FALLBACK_IMAGE_URL,
) -> "pandas.DataFrame":
    """Return a copy of results_df with Food.com images for missing entries."""
    import pandas as pd

    if results_df.empty:
        return results_df

    out = results_df.copy()
    if "image_url" not in out.columns:
        out["image_url"] = ""

    image_text = out["image_url"].astype(str).str.strip()
    needs_fetch = image_text.isna() | image_text.eq("") | image_text.eq(fallback_image)
    needs_fetch |= image_text.apply(_looks_like_placeholder)

    rows = out.loc[needs_fetch, ["recipe_id", "name"]].to_dict("records")
    if rows:
        patched = fetch_images_for_results(rows, delay=delay)
        patched_df = pd.DataFrame(patched)
        out.loc[needs_fetch, "image_url"] = patched_df["image_url"].values

    return out
