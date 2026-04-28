# Combined Text + Image Search — Implementation Guide
## SigLIP Fusion for Multimodal Queries

This document covers **only** the combined text + image search path (Option 2).
It is additive to the existing dual SBERT + SigLIP setup — nothing in the
text-only or image-only paths changes.

---

## How It Works

The three search modes and which encoder handles each:

| User Input | Encoder | Embedding Store | Reranker |
|---|---|---|---|
| Text only | SBERT | `data/artifacts/recipe_embeddings.npy` | Yes (Ridge Regression) |
| Image only | SigLIP | `data/artifacts/recipe_embeddings_siglip.npy` | No |
| Text + Image | SigLIP (both inputs) | `data/artifacts/recipe_embeddings_siglip.npy` | No |

When text and image are submitted together, **both are encoded through SigLIP**,
but the image remains the retrieval anchor. The service first builds a candidate
pool from the nearest image matches, then reranks that visually relevant pool
with a weighted image/text score. This works because SigLIP's image and text
encoders share the same 768d embedding space — making their cosine scores
directly comparable. SBERT is not involved in this path.

This image-first reranking avoids a common failure mode in food search: a broad
phrase such as "served with rice" should refine an uploaded kimchi image, not
turn the whole query into generic rice recipes.

The tradeoff: SigLIP's text encoder is weaker than SBERT for nuanced constraints
like dietary tags or cook time. This is fine because those constraints are handled
by your existing `apply_basic_filters()` layer regardless of which encoder is used.
Descriptive phrases like "spicy noodles" or "light summer dish" work well through
SigLIP's text encoder.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AT QUERY TIME                             │
│                                                                  │
│  Text only          Image only         Text + Image              │
│      │                  │                  │                     │
│      ▼                  ▼             ┌────┴────┐                │
│  SBERT encoder    SigLIP image        SigLIP    SigLIP           │
│      │            encoder             image     text             │
│      │                │               enc.      enc.             │
│      ▼                ▼               │         │                │
│  384d vector      768d vector         └────┬────┘                │
│      │                │                   ▼                      │
│      │                │          image-anchored candidate pool   │
│      │                │                   │                      │
│      │                │          weighted score reranking        │
│      │                │                   │                      │
│      ▼                ▼                   ▼                      │
│  SBERT store      SigLIP store       SigLIP store                │
│  (384d)           (768d)             (768d)                      │
│      │                │                   │                      │
│      ▼                ▼                   ▼                      │
│  Ridge Regression  raw cosine         raw cosine                 │
│  reranking         similarity         similarity                 │
│      │                │                   │                      │
│      └────────────────┴───────────────────┘                      │
│                        │                                         │
│              _passes_filters() (all paths)                       │
│                        │                                         │
│              image_map join (all paths)                          │
│                        │                                         │
│              Render recipe cards                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Changes Required

Only two files need to be updated. Everything else — scripts, config, offline
jobs, Track B image prefetch — is already in place from the existing implementation.

| File | Change |
|---|---|
| `src/recipe_discovery/retrieval/service.py` | Add SigLIP text encoding + image-anchored `search_combined()` to `RetrievalService` |
| `app/pages/1_Search.py` | Add combined routing branch in search handler |

---

## Step 1: Update `src/recipe_discovery/retrieval/service.py`

Add the SigLIP text encoder helper and combined search method to your existing
`RetrievalService`. The rest of the class is unchanged.

```python
# src/recipe_discovery/retrieval/service.py  — add these two methods to RetrievalService

def encode_combined(
    self,
    text: str,
    image: Image.Image,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Encodes both a text query and an image through SigLIP and returns
    a single weighted-average vector in the 768d SigLIP space.

    Both inputs are encoded by SigLIP (not SBERT) so they live in the
    same embedding space and can be directly combined.

    Args:
        text:   Descriptive text typed alongside the image.
                Works best for visual descriptors ("spicy", "creamy",
                "light summer dish"). Hard constraints (cook time,
                dietary tags) are better handled by _passes_filters().
        image:  PIL Image uploaded by the user.
        alpha:  Weight given to the image vector (0.0–1.0).
                0.75 = 75% image, 25% text. Tune based on testing.
                Higher alpha → results look more like the photo.
                Lower alpha  → results skew more toward the text.

    Returns:
        L2-normalized (1, 768) numpy array ready for cosine similarity
        against data/artifacts/recipe_embeddings_siglip.npy.
    """
    self._load_siglip_model()
    processor = self._siglip_processor
    model = self._siglip_model
    device = self._siglip_device

    with torch.no_grad():
        # --- Image vector ---
        img_inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        ).to(device)
        img_out = model.get_image_features(**img_inputs)
        img_vec = img_out.pooler_output
        img_vec = img_vec / img_vec.norm(p=2, dim=-1, keepdim=True)

        # --- Text vector (SigLIP encoder, not SBERT) ---
        txt_inputs = processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        txt_out = model.get_text_features(**txt_inputs)
        txt_vec = txt_out.pooler_output
        txt_vec = txt_vec / txt_vec.norm(p=2, dim=-1, keepdim=True)

    # Useful for diagnostics; production search_combined uses image-anchored reranking.
    combined = alpha * img_vec + (1 - alpha) * txt_vec
    combined = combined / combined.norm(p=2, dim=-1, keepdim=True)

    return combined.cpu().numpy()


def search_combined(
    self,
    text: str,
    image: Image.Image,
    request: RetrievalRequest,
    alpha: float = 0.75,
) -> pd.DataFrame:
    """
    Combined text + image retrieval using image-anchored SigLIP fusion.
    Searches the SigLIP embedding store only — no SBERT involvement.
    Reranker is intentionally skipped (trained on SBERT vectors).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    self._load_siglip_embeddings()
    image_vec = self._encode_image(image)
    text_vec = self._encode_siglip_text(text)
    image_scores = cosine_similarity(query=image_vec, matrix=self._siglip_embeddings)
    text_scores = cosine_similarity(query=text_vec, matrix=self._siglip_embeddings)
    combined_scores = alpha * image_scores + (1 - alpha) * text_scores

    # Anchor on the photo first, then use text to rerank that image-relevant pool.
    candidate_pool = min(len(image_scores), max(request.top_k * 50, request.top_k))
    image_top_idx = np.argpartition(-image_scores, candidate_pool - 1)[:candidate_pool]
    ranked_idx = image_top_idx[np.argsort(-combined_scores[image_top_idx])]
    ...
```

### Complete Routing Method

Add or update the top-level `retrieve()` dispatcher to handle all three modes
cleanly in one place:

```python
def search(
    self,
    request: RetrievalRequest,
    image: Image.Image | None = None,
    text_for_image: str | None = None,
    alpha: float = 0.75,
) -> pd.DataFrame:
    """
    Single entry point for all search modes.

    Routes automatically based on what inputs are provided:
      - text only  → SBERT path (existing, with reranker)
      - image only → SigLIP path (existing, no reranker)
      - text + image → SigLIP fusion path (new, no reranker)
    """
    has_text  = bool(text_for_image and text_for_image.strip())
    has_image = image is not None

    if has_text and has_image:
        # Combined: SigLIP image candidates reranked by image + text scores
        return self.search_combined(text_for_image, image, request, alpha=alpha)

    elif has_image:
        # Image only: SigLIP image encoder → SigLIP store
        return self.search_by_image(image, request)

    elif has_text:
        # Text only: SBERT encoder → SBERT store → reranker
        return self._search_candidates(request, limit_to_top_k=True)

    else:
        raise ValueError("Provide at least one of: text_for_image, image.")
```

---

## Step 2: Update `app/pages/1_Search.py`

The UI change is minimal — the existing two-column layout already collects both
inputs. You only need to update the search handler to call `retrieve()` with both
arguments when both are present, and add the alpha slider as an optional control.

```python
# pages/1_Search.py  — updated search handler

import streamlit as st
from PIL import Image
from app.service_loader import get_retrieval_service
from recipe_discovery.retrieval.service import RetrievalRequest

st.set_page_config(page_title="Recipe Search", layout="wide")
st.title("🍲 Recipe Discovery")
st.caption("Search by text, by photo, or combine both for a more specific search.")

# ── Input controls ───────────────────────────────────────────────
col_text, col_upload = st.columns([2, 1])

with col_text:
    text_query = st.text_input(
        "Describe what you want:",
        placeholder="e.g. spicy vegetarian dinner, light and fresh",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a dish photo:",
        type=["png", "jpg", "jpeg"],
        help="Upload alone to search by image, or combine with text for a joint search.",
    )

# ── Filters ──────────────────────────────────────────────────────
with st.expander("Filters & Settings"):
    max_time = st.slider("Max cook time (minutes)", 10, 180, 60, step=5)
    dietary  = st.multiselect("Dietary tags", ["vegan", "gluten-free", "vegetarian"])

    # Alpha control — only relevant when both inputs are provided
    # Hide or keep collapsed by default; power users can tune it
    alpha = st.slider(
        "Image vs text weight (combined search only)",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Higher keeps the uploaded dish as the anchor. Lower lets the text steer more.",
    )

filters = {"max_minutes": max_time}
if dietary:
    filters["tags"] = dietary

# ── Mode indicator ───────────────────────────────────────────────
has_text  = bool(text_query.strip())
has_image = uploaded_file is not None

if has_text and has_image:
    st.info("✦ Combined search — using both your photo and description via SigLIP.")
elif has_image:
    st.info("✦ Image search — finding recipes that match your photo.")
elif has_text:
    st.info("✦ Text search — semantic search via SBERT.")

# ── Search ───────────────────────────────────────────────────────
if st.button("Search", type="primary"):
    svc = get_retrieval_service()
    image = Image.open(uploaded_file) if uploaded_file else None

    if image:
        st.image(image, caption="Query image", width=200)

    if not has_text and not has_image:
        st.warning("Enter a description or upload a photo to search.")
    else:
        request = RetrievalRequest(
            query=text_query if has_text else "",
            top_k=10,
            dietary_filter=dietary[0] if dietary else None,
            max_time_minutes=max_time,
        )
        with st.spinner("Searching…"):
            if has_text and has_image:
                results = svc.search_combined(text_query, image, request, alpha=alpha)
            elif has_image:
                results = svc.search_by_image(image, request)
            else:
                results = svc.search(request)

        if results is not None and not results.empty:
            st.markdown(f"**{len(results)} recipes found**")
            for _, row in results.iterrows():
                _render_card(row.to_dict())
        else:
            st.warning("No recipes matched your query and filters. Try relaxing the filters.")


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

## Tuning the Alpha Parameter

Alpha controls how much the image vs. the text steers the combined rerank score.
The default of `0.75` (75% image) is a better starting point for food search
because photos are usually the subject and the accompanying text is often context.
Combined search also anchors the candidate pool on image similarity before text
reranking, so broad terms like "rice" cannot dominate unless the visual match is
also plausible. Adjust based on what you observe:

| Alpha | Behaviour | Good when |
|---|---|---|
| `0.8–0.9` | Results look almost entirely like the photo | Text is very vague ("something like this") |
| `0.7–0.8` | Image-anchored with text refinement | Default — works for most cases |
| `0.4–0.5` | Text and image roughly equal | Text is very descriptive |
| `0.1–0.3` | Text dominates, photo is a weak hint | User wants a specific dish but has a rough photo |

You can expose this as the slider shown above, or hardcode it and remove the UI
control once you've settled on a value through testing.

---

## What Does Not Change

- `generate_siglip_embeddings.py` — unchanged, already generates the 768d store
- `service_loader.py` — unchanged, `get_siglip()` and `get_siglip_embeddings()` are already lazy-loaded
- `apply_basic_filters()` — unchanged, runs on all three paths after retrieval
- `image_map` join — unchanged, runs on all three paths after retrieval
- `_retrieve_by_text()` — unchanged, SBERT + reranker path
- `_retrieve_by_image()` — unchanged, SigLIP image-only path
- All offline scripts (prefetch, autoencoder, regression) — unchanged
- `3_Embedding_Map.py` — unchanged, still uses SBERT autoencoder

---

## Implementation Checklist

| # | Task | Effort |
|---|---|---|
| 1 | Add `_encode_siglip_text()` to `RetrievalService` | 10 min |
| 2 | Add image-anchored `search_combined()` to `RetrievalService` | 20 min |
| 3 | Update `search()` routing with combined branch | 10 min |
| 4 | Update `1_Search.py` with mode indicator + alpha slider | 20 min |
| 5 | Smoke test: upload kimchi image + type "served with rice" → verify kimchi remains central | 15 min |
| 6 | Tune alpha value based on result quality | as needed |

**Start here:** Steps 1–3 are pure backend and can be tested independently
with a quick Python script before touching the UI.

```python
# Quick test — run from repo root before touching Streamlit
from PIL import Image
from app.service import RetrievalEngine

engine = RetrievalEngine(config={...})
image  = Image.open("tests/sample_dish.jpg")
results = engine.retrieve(text_query="but make it spicy", image=image, top_k=5)

for r in results:
    print(r["name"], r["score"])
```
