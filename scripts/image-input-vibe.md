# Recipe Discovery — Multimodal Implementation Guide
## SigLIP Image-Text Search + Recipe Image Display

This document is the single source of truth for adding image input alongside the
current SBERT text-only search. It covers two parallel tracks:

- **Track A** — Add a SigLIP image-query path (keeps SBERT for text queries)
- **Track B** — Build an offline image prefetch pipeline so every result card has a photo

Both tracks share the same service layer and ship together in the Streamlit UI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (one-time)                       │
│                                                                 │
│  recipes_clean.parquet                                          │
│        │                                                        │
│        ├─► generate_siglip_embeddings.py ──► recipes_siglip.npy │
│        │   (180k recipes × 768d, ~30 min)                       │
│        │                                                        │
│        └─► prefetch_images.py ────────────► image_map.parquet   │
│            (Unsplash API, resumable)        (recipe_id→URL)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        AT QUERY TIME                            │
│                                                                 │
│  User input: text string OR uploaded image                      │
│        │                                                        │
│        ├─► Text query ──► SBERT encoder ──► 384d query vector    │
│        │                       │                                │
│        │                       └─► Cosine similarity vs SBERT   │
│        │                           embeddings (existing store)  │
│        │                                                        │
│        └─► Image upload ─► SigLIP encoder ─► 768d query vector   │
│                                │                                │
│                                └─► Cosine similarity vs SigLIP  │
│                                    embeddings (new store)       │
│                                                                 │
│  Join with image_map.parquet  (O(1) lookup, no network call)    │
│        │                                                        │
│        ▼                                                        │
│  Render recipe cards with image + metadata in Streamlit         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Setup

**Add to `requirements.txt`:**

```text
transformers>=4.38.0
Pillow>=10.0.0
torch>=2.0.0
requests>=2.31.0
```

**Verify GPU availability (optional but recommended for embedding generation):**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

CPU is fine for inference at query time. GPU only materially helps during the
one-time embedding generation batch job (~10 min GPU vs ~60 min CPU for 180k recipes).

---

## Track A — SigLIP Image-Query Path (Additive)

### A1. Generate SigLIP Embeddings (One-Time Batch Job)

Your current recipes sit in a 384d SBERT space. SigLIP uses a 768d joint image-text
space, so we compute a *parallel* SigLIP embedding store for image queries. The
existing SBERT embeddings stay unchanged for text queries.

Create `scripts/generate_siglip_embeddings.py`:

```python
"""
One-time script: encode all recipe text into SigLIP's 768d space.
Runtime: ~10 min (GPU) or ~60 min (CPU) for 180k recipes.

Usage:
    python scripts/generate_siglip_embeddings.py \
        --data  data/processed/recipes_clean.parquet \
        --text-col semantic_text \
        --out   data/embeddings/recipes_siglip_768d.npy
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-224"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True, help="Path to cleaned recipe parquet")
    parser.add_argument("--text-col", default="semantic_text", help="Column to embed")
    parser.add_argument("--out",      required=True, help="Output .npy path")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    df = pd.read_parquet(args.data)
    texts = df[args.text_col].fillna("").tolist()
    print(f"Encoding {len(texts)} recipes...")

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch = texts[i : i + args.batch_size]
            inputs = processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            features = model.get_text_features(**inputs)
            # L2-normalize so cosine similarity = dot product (faster at retrieval)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    np.save(args.out, embeddings)
    print(f"Saved: {args.out}  shape={embeddings.shape}")  # (180000, 768)


if __name__ == "__main__":
    main()
```

---

### A2. Keep SBERT Models for Text Search

Because image search is only triggered on upload, keep all existing SBERT-backed
models (autoencoder, regression, k-means, and any analytics) unchanged. The SigLIP
embeddings are only used for image queries and do not replace your SBERT artifacts.

---

### A3. Update the Service Layer

Keep the existing SBERT text path untouched and add a SigLIP image path that only
executes when an image is uploaded. The key addition is `encode_image()` plus a
separate SigLIP embedding store.

```python
# app/service.py  — updated RetrievalEngine (additive image path)

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-224"
FALLBACK_IMAGE = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"


class RetrievalEngine:

    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SBERT — keep your existing text encoder and embeddings as-is
        # (Assumes you already load self.sbert_model and self.recipe_embeddings_sbert)

        # SigLIP — image encoder and its dedicated 768d embedding store
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        # Vector store (768d) for image queries only
        emb_path = config["siglip_embeddings_path"]  # data/embeddings/recipes_siglip_768d.npy
        self.recipe_embeddings_siglip = np.load(emb_path)

        # Recipe metadata
        self.df = pd.read_parquet(config["data_path"])

        # Image map — recipe_id → URL (built offline, zero network at query time)
        image_map_path = Path(config.get("image_map_path", "data/processed/image_map.parquet"))
        if image_map_path.exists():
            self._image_map: pd.Series = pd.read_parquet(image_map_path)["image_url"]
        else:
            self._image_map = pd.Series(dtype=str)

        # Optional: Ridge Regression reranker (joblib artifact)
        self._reranker = self._load_reranker(config.get("reranker_path"))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_text(self, query: str) -> np.ndarray:
        """Keep your existing SBERT text encoding path (unchanged)."""
        return self._encode_text_sbert(query)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Returns an L2-normalized (1, 768) numpy array for an uploaded PIL image.
        Because SigLIP shares a joint embedding space, this vector is directly
        comparable to the text-encoded recipe vectors — no adapter needed.
        """
        with torch.no_grad():
            inputs = self.processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            ).to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Cosine similarity search + optional reranking.
        query_vector: (1, d) — output of encode_text() or encode_image()
        """
        scores = cosine_similarity(query_vector, self.recipe_embeddings_siglip)[0]
        candidate_idx = np.argsort(scores)[::-1][: top_k * 5]  # oversample for filtering

        results = []
        for idx in candidate_idx:
            row = self.df.iloc[idx]
            if not self._passes_filters(row, filters or {}):
                continue
            results.append({
                "recipe_id": idx,
                "name":       row["name"],
                "score":      float(scores[idx]),
                "minutes":    row.get("minutes"),
                "rating":     row.get("rating"),
                "n_steps":    row.get("n_steps"),
                "tags":       row.get("tags", []),
                "ingredients": row.get("ingredients", []),
                "image_url":  self._image_map.get(idx, FALLBACK_IMAGE),
            })
            if len(results) == top_k:
                break

        # Optional reranking with Ridge Regression quality signal (text-only)
        # Skip for image queries unless you retrain on SigLIP embeddings.

        return results

    def retrieve_by_text(self, query: str, **kwargs) -> list[dict]:
        # Use existing SBERT pipeline and embeddings (not shown here).
        return self._retrieve_by_text_sbert(query, **kwargs)

    def retrieve_by_image(self, image: Image.Image, **kwargs) -> list[dict]:
        return self.retrieve(self.encode_image(image), **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passes_filters(self, row: pd.Series, filters: dict) -> bool:
        if "max_minutes" in filters and row.get("minutes", 9999) > filters["max_minutes"]:
            return False
        if "tags" in filters:
            row_tags = set(row.get("tags", []))
            if not set(filters["tags"]).issubset(row_tags):
                return False
        return True

    def _rerank(self, results: list[dict], query_vector: np.ndarray) -> list[dict]:
        """Blend semantic score with Ridge Regression quality signal."""
        import joblib
        scores = self._reranker.predict(
            np.vstack([self.recipe_embeddings_sbert[r["recipe_id"]] for r in results])
        )
        for result, quality in zip(results, scores):
            result["blended_score"] = 0.7 * result["score"] + 0.3 * quality
        return sorted(results, key=lambda r: r["blended_score"], reverse=True)

    def _load_reranker(self, path: str | None):
        if not path:
            return None
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            return None
```

---

### A4. Streamlit UI — Multimodal Search Page

Update `pages/1_Search.py` to expose both input modes. The image path runs **only**
when a file is uploaded; otherwise the existing SBERT text search runs.

```python
# pages/1_Search.py

import streamlit as st
from PIL import Image
from service_loader import get_engine   # your existing singleton loader

st.set_page_config(page_title="Recipe Search", layout="wide")
st.title("🍲 Recipe Discovery")
st.caption("Search by describing a craving **or** upload a photo of a dish.")

# ── Input controls ──────────────────────────────────────────────
col_text, col_upload = st.columns([2, 1])

with col_text:
    text_query = st.text_input(
        "Describe what you want:",
        placeholder="e.g. spicy vegetarian dinner under 30 minutes",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload a dish photo:",
        type=["png", "jpg", "jpeg"],
        help="SigLIP will find recipes that look and taste similar.",
    )

# ── Filters (unchanged from your existing UI) ───────────────────
with st.expander("Filters"):
    max_time = st.slider("Max cook time (minutes)", 10, 180, 60, step=5)
    dietary = st.multiselect("Dietary tags", ["vegan", "gluten-free", "vegetarian"])

filters = {"max_minutes": max_time}
if dietary:
    filters["tags"] = dietary

# ── Search ───────────────────────────────────────────────────────
search_clicked = st.button("Search", type="primary")

if search_clicked:
    engine = get_engine()
    results = []

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Searching by this image…", width=240)
        with st.spinner("Encoding image…"):
            results = engine.retrieve_by_image(image, top_k=10, filters=filters)

    elif text_query.strip():
        with st.spinner("Searching…"):
            results = engine.retrieve_by_text(text_query, top_k=10, filters=filters)

    else:
        st.warning("Enter a description or upload a photo to search.")

    # ── Results ──────────────────────────────────────────────────
    if results:
        st.markdown(f"**{len(results)} recipes found**")
        for recipe in results:
            _render_card(recipe)


def _render_card(recipe: dict):
    """Renders one recipe result as an image + info card."""
    with st.container():
        col_img, col_info = st.columns([1, 2], gap="medium")

        with col_img:
            st.image(recipe["image_url"], use_container_width=True)

        with col_info:
            st.markdown(f"### {recipe['name']}")
            meta = []
            if recipe.get("minutes"):
                meta.append(f"⏱ {recipe['minutes']} min")
            if recipe.get("rating"):
                meta.append(f"⭐ {recipe['rating']:.1f}")
            if recipe.get("n_steps"):
                meta.append(f"🍽 {recipe['n_steps']} steps")
            st.caption("  |  ".join(meta))

            tags = recipe.get("tags", [])
            if tags:
                st.markdown(" ".join(f"`{t}`" for t in tags[:6]))

            with st.expander("Ingredients"):
                st.write(", ".join(recipe.get("ingredients", [])))

            score = recipe.get("blended_score") or recipe.get("score", 0)
            st.progress(min(score, 1.0), text=f"Relevance: {score:.2f}")

        st.divider()
```

---

## Track B — Recipe Image Display (Offline Prefetch)

Even without running a search, every result card needs a photo. This track builds
the `image_map.parquet` file that the service layer joins at retrieval time.

### B1. Keyword Extractor

Create `scripts/extract_image_keywords.py`:

```python
import re

STRIP_WORDS = {
    "easy", "quick", "simple", "best", "homemade", "classic", "traditional",
    "minutes", "minute", "hour", "style", "recipe", "dish", "serving",
    "delicious", "perfect", "amazing",
}


def extract_image_query(recipe_name: str, ingredients: list[str] | None = None) -> str:
    """
    Produces a clean, image-search-friendly query from a recipe name.
    Augments with top ingredients only when the name is too short to be specific.
    """
    name = recipe_name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    words = [w for w in name.split() if w not in STRIP_WORDS]
    clean = " ".join(words[:5])

    if len(words) < 2 and ingredients:
        clean = f"{' '.join(ingredients[:2])} {clean}"

    return f"{clean} food plated"
```

### B2. Offline Prefetch Script

Create `scripts/prefetch_images.py`:

```python
"""
Offline job: build recipe_id → Unsplash image URL mapping.
Resumable — skips already-processed IDs if output file exists.

Free tier:   50 req/hr  → ~180k recipes in ~150 hrs (run in batches)
Production:  5k req/hr  → ~180k recipes in ~36 hrs  (free approval at unsplash.com/developers)

Usage:
    python scripts/prefetch_images.py \
        --data    data/processed/recipes_clean.parquet \
        --output  data/processed/image_map.parquet \
        --api-key YOUR_UNSPLASH_KEY
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from extract_image_keywords import extract_image_query

UNSPLASH_URL  = "https://api.unsplash.com/search/photos"
FALLBACK_URL  = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c"
POLLINATIONS  = "https://image.pollinations.ai/prompt/{prompt}?width=400&height=300&nologo=true"


def fetch_unsplash(query: str, api_key: str) -> str | None:
    try:
        r = requests.get(
            UNSPLASH_URL,
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {api_key}"},
            timeout=5,
        )
        r.raise_for_status()
        hits = r.json().get("results", [])
        if hits:
            return hits[0]["urls"]["small"]   # ~400px wide, good balance of quality/size
    except Exception as e:
        print(f"  [WARN] {query}: {e}")
    return None


def pollinations_url(recipe_name: str) -> str:
    """Zero-cost fallback: Pollinations generates on first browser load, no API key."""
    import urllib.parse
    prompt = urllib.parse.quote(f"appetizing food photo of {recipe_name}, natural lighting")
    return POLLINATIONS.format(prompt=prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--api-key",   default=None, help="Unsplash Client-ID (omit to use Pollinations only)")
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    out = Path(args.output)

    # Resume support
    if out.exists():
        existing = pd.read_parquet(out)
        done_ids = set(existing.index)
        df = df[~df.index.isin(done_ids)]
        print(f"Resuming: {len(df)} remaining / {len(done_ids)} already done")
    else:
        existing = pd.DataFrame()

    records = []
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        ingredients = row.get("ingredients") if isinstance(row.get("ingredients"), list) else []
        query = extract_image_query(row["name"], ingredients)

        if args.api_key:
            url = fetch_unsplash(query, args.api_key) or pollinations_url(row["name"])
        else:
            url = pollinations_url(row["name"])

        records.append({"recipe_id": idx, "image_url": url, "image_query": query})
        print(f"[{i}] {row['name'][:45]:<45} → {url[:60]}")

        # Respect free tier rate limit
        if args.api_key and i % args.batch_size == 0:
            print("  Sleeping 75s (rate limit)…")
            time.sleep(75)

    new_df = pd.DataFrame(records).set_index("recipe_id")
    combined = pd.concat([existing, new_df]) if not existing.empty else new_df
    combined.to_parquet(out)
    print(f"\nSaved {len(combined)} entries → {out}")


if __name__ == "__main__":
    main()
```

> **Tip:** Run the prefetch on your top 1,000 highest-rated recipes first so the app looks
> complete immediately. The rest can fill in overnight.

---

## Configuration

Update `configs/config.yaml` to add SigLIP artifacts without changing the SBERT ones:

```yaml
# configs/config.yaml

model:
    siglip_model_id: "google/siglip-base-patch16-224"
    sbert_embedding_dim: 384
    siglip_embedding_dim: 768

paths:
    data:              "data/processed/recipes_clean.parquet"
    sbert_embeddings:  "data/embeddings/recipes_sbert_384d.npy"    # existing
    siglip_embeddings: "data/embeddings/recipes_siglip_768d.npy"   # new (image search)
    image_map:         "data/processed/image_map.parquet"         # new
    autoencoder:       "models/autoencoder_weights.pth"           # existing
    reranker:          "models/ridge_regression.joblib"           # existing
    kmeans:            "models/kmeans.joblib"                     # existing

retrieval:
  top_k: 10
    oversample_factor: 5    # fetch top_k * 5 before filtering
    rerank_blend: 0.7       # weight for semantic score vs quality signal (text-only)
```

---

## File Structure After Implementation

```
recipe-discovery-webapp/
├── configs/
│   └── config.yaml                          ← updated (new paths + embedding_dim)
├── data/
│   ├── embeddings/
│   │   ├── recipes_sbert_384d.npy           (existing SBERT store)
│   │   └── recipes_siglip_768d.npy          ← NEW (image search)
│   └── processed/
│       ├── recipes_clean.parquet            (unchanged)
│       └── image_map.parquet                ← NEW
├── models/
│   ├── autoencoder_weights.pth              (existing SBERT)
│   ├── ridge_regression.joblib              (existing SBERT)
│   └── kmeans.joblib                        (existing SBERT)
├── scripts/
│   ├── generate_siglip_embeddings.py        ← NEW
│   ├── extract_image_keywords.py            ← NEW
│   ├── prefetch_images.py                   ← NEW
│   ├── train_autoencoder.py                 (no change)
│   └── train_regression.py                 (no change)
├── app/
│   └── service.py                           ← updated (SigLIP encoder + image_map join)
└── pages/
    └── 1_Search.py                          ← updated (file_uploader + card layout)
```

---

## Implementation Milestones

| # | Task | Effort | Blocks |
|---|------|--------|--------|
| 1 | Install deps, verify torch + transformers | 15 min | everything |
| 2 | Run `generate_siglip_embeddings.py` on full dataset | 10–60 min | 3, 4 |
| 3 | Update `service.py` (SigLIP image path + image_map join) | 45 min | UI |
| 4 | Update `config.yaml` with new paths | 10 min | service load |
| 5 | Prefetch images for top 1k recipes by rating | 30 min + wait | result cards |
| 6 | Update `1_Search.py` (upload widget + cards) | 45 min | — |
| 7 | End-to-end smoke test (text query + image query) | 30 min | — |
| 8 | Run full 180k image prefetch overnight | background | — |

**Start here:** Steps 1 → 2 → 4 → 3 → 6. You can validate the full text search
pipeline immediately because the SBERT path is unchanged.

---

## Notes & Caveats

**SigLIP vs SBERT quality on text queries**
SigLIP's text encoder is optimized for image-text alignment, not pure text-text
similarity. Keep SBERT for text-only queries and only invoke SigLIP when an image
is uploaded. This preserves your current quality while adding image search.

**Image search expectations**
Uploading a photo of a dish will find recipes that SigLIP associates with similar
visual concepts (ingredients, cooking style, plating). It won't be perfect — a
photo of a pasta dish may surface several pasta variants, which is the correct
behaviour.

**Pollinations.ai fallback latency**
Pollinations images are generated on first browser load (~2–4s). Use a CSS skeleton
placeholder via `st.empty()` while images load. Once generated, Pollinations caches
them by prompt, so repeat searches for the same dish are instant.

**Autoencoder decoder**
No changes are required because the SBERT autoencoder remains in the 384d space.