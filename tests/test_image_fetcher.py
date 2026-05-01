"""Tests for Food.com image passthrough helpers."""

from __future__ import annotations

import pandas as pd

from recipe_discovery.retrieval.image_fetcher import (
    FALLBACK_IMAGE_URL,
    MAX_FOODCOM_IMAGE_WORKERS,
    attach_foodcom_images,
    fetch_food_com_image,
)


class _DummyResponse:
    def __init__(self, html: str) -> None:
        self.text = html

    def raise_for_status(self) -> None:
        return None


def test_attach_foodcom_images_fetches_missing_images_by_default(monkeypatch) -> None:
    image_url = "https://img.example/test.jpg"
    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.fetch_food_com_image",
        lambda recipe_id, name: image_url,
    )
    results = pd.DataFrame(
        {
            "recipe_id": ["1"],
            "name": ["Test Recipe"],
            "image_url": [FALLBACK_IMAGE_URL],
        }
    )

    patched = attach_foodcom_images(results, delay=0, max_workers=1)

    assert patched["image_url"].tolist() == [image_url]


def test_attach_foodcom_images_leaves_existing_images_alone(monkeypatch) -> None:
    def fail_fetch(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("existing image URL should not be refetched")

    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.fetch_food_com_image",
        fail_fetch,
    )
    image_url = "https://img.example/existing.jpg"
    results = pd.DataFrame(
        {
            "recipe_id": ["1"],
            "name": ["Test Recipe"],
            "image_url": [image_url],
        }
    )

    patched = attach_foodcom_images(results, delay=0, max_workers=1)

    assert patched["image_url"].tolist() == [image_url]


def test_attach_foodcom_images_fetches_all_missing_rows(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(recipe_id: object, name: str | None) -> str:
        calls.append(str(recipe_id))
        return f"https://img.example/{recipe_id}.jpg"

    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.fetch_food_com_image",
        fake_fetch,
    )
    results = pd.DataFrame(
        {
            "recipe_id": [str(idx) for idx in range(6)],
            "name": [f"Recipe {idx}" for idx in range(6)],
            "image_url": [FALLBACK_IMAGE_URL for _ in range(6)],
        }
    )

    patched = attach_foodcom_images(results, delay=0, max_workers=1)

    assert calls == ["0", "1", "2", "3", "4", "5"]
    assert patched["image_url"].tolist() == [
        "https://img.example/0.jpg",
        "https://img.example/1.jpg",
        "https://img.example/2.jpg",
        "https://img.example/3.jpg",
        "https://img.example/4.jpg",
        "https://img.example/5.jpg",
    ]
    assert MAX_FOODCOM_IMAGE_WORKERS == 4


def test_fetch_food_com_image_marks_add_your_photo_page_as_no_image(monkeypatch) -> None:
    fetch_food_com_image.cache_clear()
    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.requests.get",
        lambda *args, **kwargs: _DummyResponse("<html><body>Add your photo</body></html>"),
    )

    image_url = fetch_food_com_image("1", "Recipe Without Photo")

    assert image_url == FALLBACK_IMAGE_URL


def test_fetch_food_com_image_uses_photo_by_metadata(monkeypatch) -> None:
    fetch_food_com_image.cache_clear()
    expected = "https://img.sndimg.com/food/image/upload/photo.jpg"
    html = f"""
    <html>
      <head><meta property="og:image" content="{expected}" /></head>
      <body>Photo by home cook</body>
    </html>
    """
    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    image_url = fetch_food_com_image("2", "Recipe With Photo")

    assert image_url == expected


def test_fetch_food_com_image_uses_photo_by_img_when_metadata_missing(monkeypatch) -> None:
    fetch_food_com_image.cache_clear()
    expected = "https://img.sndimg.com/food/image/upload/photo-from-img.jpg"
    html = f"""
    <html>
      <body>
        <p>Photo by home cook</p>
        <img src="{expected}" />
      </body>
    </html>
    """
    monkeypatch.setattr(
        "recipe_discovery.retrieval.image_fetcher.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    image_url = fetch_food_com_image("3", "Recipe With Credited Photo")

    assert image_url == expected
