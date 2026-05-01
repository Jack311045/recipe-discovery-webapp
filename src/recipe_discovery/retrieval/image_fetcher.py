"""On-demand Food.com image lookup for recipe results."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
MAX_FOODCOM_IMAGE_WORKERS = 4
_PLACEHOLDER_TOKENS = ("logo", "placeholder", "default")
_ADD_PHOTO_PATTERNS = (
    "add your photo",
    "be the first to add a photo",
)
_PHOTO_BY_PATTERNS = (
    "photo by",
    "photos by",
)


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


def _looks_like_placeholder(url: str) -> bool:
    lowered = url.lower().strip()
    if lowered.endswith(".svg"):
        return True
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


def _normalize_image_url(url: object) -> str | None:
    text = str(url or "").strip()
    if not text:
        return None
    if text.startswith("//"):
        text = f"https:{text}"
    if not text.lower().startswith(("http://", "https://")):
        return None
    if _looks_like_placeholder(text):
        return None
    return text


def _page_text(soup: BeautifulSoup) -> str:
    return soup.get_text(" ", strip=True).lower()


def _has_add_photo_prompt(text: str) -> bool:
    return any(pattern in text for pattern in _ADD_PHOTO_PATTERNS)


def _has_photo_credit(text: str) -> bool:
    return any(pattern in text for pattern in _PHOTO_BY_PATTERNS)


def _extract_jsonld_image(soup: BeautifulSoup) -> str | None:
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
                    return _normalize_image_url(image[0])
                if isinstance(image, str):
                    return _normalize_image_url(image)
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def _extract_meta_image(soup: BeautifulSoup) -> str | None:
    selectors = (
        {"property": "og:image"},
        {"property": "og:image:secure_url"},
        {"name": "twitter:image"},
        {"itemprop": "image"},
    )
    for selector in selectors:
        tag = soup.find("meta", attrs=selector)
        if tag is None:
            continue
        image_url = _normalize_image_url(tag.get("content"))
        if image_url:
            return image_url
    return None


def _srcset_urls(value: object) -> list[str]:
    urls: list[str] = []
    for candidate in str(value or "").split(","):
        url = candidate.strip().split(" ", 1)[0]
        if url:
            urls.append(url)
    return urls


def _extract_photo_credit_image(soup: BeautifulSoup) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    for tag in soup.find_all("img"):
        candidates = []
        for attr in ("src", "data-src", "data-original", "data-lazy-src"):
            candidates.append(tag.get(attr))
        candidates.extend(_srcset_urls(tag.get("srcset")))

        for raw_url in candidates:
            image_url = _normalize_image_url(raw_url)
            if not image_url:
                continue
            lowered = image_url.lower()
            if (
                "img.sndimg.com" in lowered
                or "/food/image/" in lowered
                or "/recipes/" in lowered
                or "image/upload" in lowered
            ):
                preferred.append(image_url)
            else:
                fallback.append(image_url)

    return (preferred or fallback or [None])[0]


@lru_cache(maxsize=512)
def fetch_food_com_image(recipe_id: object, name: str | None) -> str:
    url = build_food_com_url(recipe_id, name)
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = _page_text(soup)
        has_photo_credit = _has_photo_credit(text)
        if _has_add_photo_prompt(text) and not has_photo_credit:
            logger.info("[image] no Food.com photo for recipe %s at %s", recipe_id, url)
            return FALLBACK_IMAGE_URL

        image_url = _extract_jsonld_image(soup) or _extract_meta_image(soup)
        if image_url:
            logger.info("[image] fetched %s -> %s", recipe_id, image_url[:60])
            return image_url

        if has_photo_credit:
            image_url = _extract_photo_credit_image(soup)
            if image_url:
                logger.info("[image] fetched credited photo %s -> %s", recipe_id, image_url[:60])
                return image_url

        logger.warning("[image] no Food.com image found for recipe %s at %s", recipe_id, url)
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
    max_workers: int = MAX_FOODCOM_IMAGE_WORKERS,
) -> list[dict]:
    """Enrich result dicts with Food.com image URLs."""
    if not results:
        return results

    worker_count = max(1, min(int(max_workers), len(results)))
    if worker_count == 1:
        for idx, result in enumerate(results):
            result["image_url"] = fetch_food_com_image(result["recipe_id"], result.get("name"))
            if delay > 0 and idx < len(results) - 1:
                time.sleep(delay)
        return results

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        image_urls = list(
            executor.map(
                lambda result: fetch_food_com_image(result["recipe_id"], result.get("name")),
                results,
            )
        )
    for result, image_url in zip(results, image_urls):
        result["image_url"] = image_url
    return results


def attach_foodcom_images(
    results_df: "pandas.DataFrame",
    *,
    delay: float = POLITE_DELAY,
    fallback_image: str = FALLBACK_IMAGE_URL,
    max_workers: int = MAX_FOODCOM_IMAGE_WORKERS,
) -> "pandas.DataFrame":
    """Return a copy of results_df with Food.com images for missing entries.

    Existing valid image URLs are always preserved. Missing-image rows are
    fetched from Food.com, where "Add your photo" pages short-circuit to the
    fallback image and credited "Photo by" pages get a deeper image extraction.
    """
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
        patched = fetch_images_for_results(rows, delay=delay, max_workers=max_workers)
        patched_df = pd.DataFrame(patched)
        out.loc[needs_fetch, "image_url"] = patched_df["image_url"].values

    return out
