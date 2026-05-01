# Semantic Search, Embeddings, Vectors, and Retrieve-All Pipeline

This document explains how semantic search works in this recipe discovery app
from raw recipe metadata through vector creation, runtime scoring, filtering,
and final Streamlit display. It covers the three user-facing search modes:

1. Text-only search
2. Image-only search
3. Text plus image search

## Scope and Naming

There is no literal `retrieveAll()` or `retrieve_all()` function in the current
codebase. The runtime equivalent of a "retrieve-all" process is the direct
cosine search path in `RetrievalService`: the service scores the query vector
against the loaded recipe embedding matrix, selects a larger candidate pool, and
then filters and ranks those candidates.

The main implementation lives in:

1. `src/recipe_discovery/retrieval/service.py`
2. `src/recipe_discovery/retrieval/similarity.py`
3. `src/recipe_discovery/retrieval/filters.py`
4. `src/recipe_discovery/data/corpus.py`
5. `src/recipe_discovery/embeddings/encoder.py`
6. `src/recipe_discovery/embeddings/store.py`
7. `scripts/build_embeddings.py`
8. `scripts/generate_siglip_embeddings.py`
9. `app/pages/1_Search.py`

## High-Level Search Modes

| User input | Runtime method | Encoder space | Stored vectors searched | Main purpose |
|---|---|---|---|---|
| Text only | `RetrievalService.search()` | SBERT text space | `recipe_embeddings.npy` | Best semantic text-to-text recipe match |
| Image only | `RetrievalService.search_by_image()` | SigLIP multimodal space | `recipe_embeddings_siglip.npy` | Match uploaded food photo to recipe text vectors |
| Text plus image | `RetrievalService.search_combined()` | SigLIP multimodal space | `recipe_embeddings_siglip.npy` | Use image as visual anchor and text as refinement |

The two vector stores are intentionally separate. SBERT and SigLIP vectors
cannot be mixed because they are produced by different models, have different
dimensions, and live in different coordinate systems.

## Tools and Libraries Used

| Tool | Where used | Why it is used |
|---|---|---|
| `pandas` | data loading, metadata alignment, filtering | Tabular recipe metadata operations |
| `numpy` | vector matrices, cosine scoring, partial top-k | Fast numerical search over embeddings |
| `sentence-transformers` | `RecipeEncoder` | Text-only semantic embeddings |
| `sentence-transformers/all-MiniLM-L6-v2` | default SBERT model | Fast 384-dimensional recipe and query vectors |
| `transformers` `AutoProcessor` and `AutoModel` | SigLIP model loading | Multimodal image/text embedding |
| `google/siglip-base-patch16-224` | image and image+text search | Shared image/text embedding space |
| `torch` | SigLIP inference | Tensor execution, `no_grad()`, CPU/CUDA support |
| `Pillow` | uploaded image handling | Converts uploaded images to RGB PIL images |
| `scikit-learn` `NearestNeighbors` | optional index artifact | Validated cosine nearest-neighbor index |
| `joblib` | saved index and model artifacts | Persist sklearn-style objects |
| `Streamlit` | app UI and cached backend loading | Search form, uploads, cards, session state |
| `requests` and `BeautifulSoup` | result image lookup | Fetch Food.com JSON-LD image URLs when needed |
| `gdown` | artifact download script | Fetch large data/artifacts from Google Drive |

## Artifact Contract

The retrieval service depends on row-aligned artifact files under
`data/artifacts/`.

| Artifact | Produced by | Used by | Meaning |
|---|---|---|---|
| `recipe_embeddings.npy` | `scripts/build_embeddings.py` | text search | SBERT embedding matrix, shape `(n_recipes, 384)` by default |
| `recipe_ids.csv` | `scripts/build_embeddings.py` | service load | Recipe IDs aligned to rows in `recipe_embeddings.npy` |
| `recipe_texts.csv` | `scripts/build_embeddings.py` | debugging/auditing | Canonical text that was embedded |
| `embedding_metadata.json` | `scripts/build_embeddings.py` | service load | SBERT model name, dimension, normalization flag |
| `recipe_embeddings_siglip.npy` | `scripts/generate_siglip_embeddings.py` | image and combined search | SigLIP recipe text vectors, typically 768-dimensional |
| `recipe_ids_siglip.csv` | `scripts/generate_siglip_embeddings.py` | image and combined search | Recipe IDs aligned to SigLIP rows |
| `recipe_texts_siglip.csv` | `scripts/generate_siglip_embeddings.py` | debugging/auditing | Text encoded for SigLIP |
| `embedding_metadata_siglip.json` | `scripts/generate_siglip_embeddings.py` | diagnostics | SigLIP model metadata |
| `recipe_index.joblib` | `scripts/build_index.py` | optional/offline | sklearn cosine index, not the default live path |

Critical invariant:

```text
embedding_matrix[i] corresponds to recipe_ids.csv row i
```

The service does not trust the processed CSV row order. It aligns metadata to
artifact order by `recipe_id` during load.

## Offline Build Pipeline

### 1. Load Processed Recipes

The processed CSV is loaded by:

```text
src/recipe_discovery/data/load.py:load_processed_recipes()
```

Default path:

```text
data/processed/Processed_data_updated2.csv
```

The loader:

1. Reads the processed CSV with `pandas.read_csv`.
2. Drops unnamed columns created by CSV export.
3. Returns the cleaned recipe metadata DataFrame.

The schema helpers live in:

```text
src/recipe_discovery/data/schema.py
```

Important column groups:

1. ID column: `recipe_id`
2. Text columns: `name`, `description`, `ingredients`, `steps`
3. Short metadata: `minutes`, `n_steps`, `n_ingredients`
4. Nutrition columns: `calories`, `total fat`, `sugar`, `sodium`, `protein`,
   `saturated fat`, `carbohydrates`
5. Outcome columns: `rating`, `num_ratings`
6. One-hot tag columns: detected dynamically by `get_one_hot_tag_columns()`

### 2. Build Canonical Recipe Text

The embedding pipeline does not embed only the recipe title. It serializes each
recipe row into a deterministic text block in:

```text
src/recipe_discovery/data/corpus.py
```

`build_corpus()` adds a `recipe_text` column. For each recipe,
`serialize_recipe()` includes available fields in this shape:

```text
Name: <name>
Description: <description>
Ingredients: <ingredients>
Steps: <steps>
Time: <minutes> minutes
Step Count: <n_steps>
Ingredient Count: <n_ingredients>
Tags: <active one-hot tag names>
Nutrition: calories=..., protein=..., ...
```

This is important because the dense vector represents the whole recipe context,
not just keywords. Ingredients, steps, time, tags, and nutrition can all
influence semantic similarity.

### 3. Build Text Embeddings

Command:

```bash
python scripts/build_embeddings.py
```

Core flow:

1. Load processed recipes.
2. Call `build_corpus()` to create `recipe_text`.
3. Load `RecipeEncoder`.
4. Encode recipe texts with `sentence-transformers/all-MiniLM-L6-v2`.
5. Save embeddings, IDs, texts, and metadata.

Implementation:

```text
src/recipe_discovery/embeddings/encoder.py
```

Default config:

```text
model_name = sentence-transformers/all-MiniLM-L6-v2
batch_size = 32
normalize = True
```

`RecipeEncoder.encode()` calls the SentenceTransformer model with:

```text
convert_to_numpy=True
normalize_embeddings=True
show_progress_bar=<controlled by caller>
```

The default model returns 384-dimensional vectors. Normalizing embeddings at
build time makes cosine comparisons stable and efficient.

### 4. Build SigLIP Recipe Embeddings for Image Search

Command:

```bash
python scripts/generate_siglip_embeddings.py
```

Core flow:

1. Load processed recipes.
2. Build the same canonical `recipe_text`.
3. Load `google/siglip-base-patch16-224` using `transformers`.
4. Encode recipe text with SigLIP's text encoder.
5. L2-normalize each vector.
6. Save SigLIP embedding artifacts.

Why encode recipe text with SigLIP?

The app does not have a food image for every recipe at embedding-build time.
Instead, it embeds recipe text into SigLIP's shared image/text space. At runtime,
an uploaded image is embedded into that same SigLIP space, so image vectors can
be compared directly to recipe text vectors.

### 5. Optional sklearn Index

Command:

```bash
python scripts/build_index.py
```

This builds:

```text
data/artifacts/recipe_index.joblib
```

using:

```text
sklearn.neighbors.NearestNeighbors(metric="cosine", algorithm="brute")
```

Current runtime behavior does not use this index by default. Live search uses
direct cosine scoring over the in-memory matrix because filtering, tag boosting,
and candidate-pool logic are implemented in that path. The saved index remains a
validated optional optimization and offline diagnostic artifact.

## Runtime Service Load

The app loads the retrieval backend through:

```text
app/service_loader.py:get_retrieval_service()
```

Streamlit caches the service with `st.cache_resource`, so the expensive service
load is done once across reruns.

`RetrievalService.load()` performs this sequence:

1. Load processed metadata with `load_processed_recipes()`.
2. Load SBERT embeddings with `load_embeddings()`.
3. Load row-aligned recipe IDs with `load_recipe_ids()`.
4. Align metadata to embedding row order by `recipe_id`.
5. Detect one-hot tag columns.
6. Attach image URLs from `data/processed/image_map.parquet` when present.
7. Load the SBERT encoder using `embedding_metadata.json` when available.
8. Load optional 2D projections if projection artifacts exist.

Alignment is strict:

1. Metadata IDs and artifact IDs are normalized as strings.
2. Duplicate IDs fail fast.
3. Embedding row count must match `recipe_ids.csv`.
4. Artifact IDs must exist in processed metadata.
5. The aligned metadata is sorted in embedding-row order.

This prevents a subtle but serious bug: returning metadata from row `i` while
the vector score came from a different recipe at row `i`.

## Text-Only Search Pipeline

Text-only search is triggered in:

```text
app/pages/1_Search.py
```

When the user enters text and does not upload an image, the page calls:

```python
results = svc.search(request)
```

where `request` is a `RetrievalRequest`.

### Request Fields

`RetrievalRequest` supports:

```text
query
top_k
dietary_filter
max_time_minutes
max_ingredients
max_calories
max_fat
max_sugar
max_sodium
max_protein
max_saturated_fat
max_carbohydrates
min_rating
```

The current Search page exposes dietary preference, max cooking time, max
ingredients, top-k, and combined-search image/text weight. The backend supports
more numeric filters than the page currently exposes.

### Step-by-Step Text Search

1. Validate that `request.query` exists.
2. Encode the query with the loaded SBERT `RecipeEncoder`.
3. Extract the single query vector.
4. Compute cosine similarity against every row in `recipe_embeddings.npy`.
5. Select a candidate pool larger than `top_k`.
6. Attach scores to candidate metadata rows.
7. Apply filters to the candidate pool.
8. Optionally apply exact tag-aware boosting.
9. Sort by score.
10. Return the final `top_k` rows.
11. Attach or patch result image URLs.

The cosine scoring function is:

```text
score(recipe_i) = (recipe_vector_i dot query_vector)
                  / (||recipe_vector_i|| * ||query_vector||)
```

Implementation:

```text
src/recipe_discovery/retrieval/similarity.py:cosine_similarity()
```

The function validates:

1. The embedding matrix is 2D.
2. The query vector is 1D or a single-row 2D vector.
3. Query dimension matches matrix dimension.

### Retrieve-All Scoring

The "retrieve-all" part happens here:

```text
RetrievalService._search_candidates_for_vector()
```

It calculates one score per recipe:

```python
scores = cosine_similarity(query=query_vec, matrix=embeddings)
```

This is a full matrix scan, not a keyword lookup. For text search, `embeddings`
is the SBERT matrix:

```text
data/artifacts/recipe_embeddings.npy
```

After all scores are computed, the service does not sort the entire dataset.
Instead, it uses a partial top-k optimization:

```python
candidate_pool = min(len(scores), max(request.top_k * 10, request.top_k))
top_idx = np.argpartition(-scores, candidate_pool - 1)[:candidate_pool]
ranked_idx = top_idx[np.argsort(-scores[top_idx])]
```

This is faster than fully sorting every recipe when only a limited result set is
needed. The app then applies filters to the larger candidate pool, which avoids
the failure mode where the top 8 unfiltered results are selected first and every
one of them is later filtered out.

### Text Filtering

Filtering lives in:

```text
src/recipe_discovery/retrieval/filters.py
```

Filtering is applied after candidate-pool selection. It supports:

1. Time: `minutes <= max_time_minutes`, with one-hot bucket fallback if needed.
2. Ingredients: `n_ingredients <= max_ingredients`, with
   `5-ingredients-or-less` fallback if needed.
3. Dietary tags: resolved against one-hot tag columns such as `vegetarian`,
   `vegan`, and `gluten-free`.
4. Nutrition and rating filters in the backend request model.

Dietary filters are schema-aware. For example, `"gluten free"` can resolve to a
column like `gluten-free`.

### Tag-Aware Boosting

Short queries like `"vegan"`, `"korean"`, or `"desserts"` can be ambiguous for a
pure embedding model. The service adds a small structured-data boost when the
query exactly or near-exactly resolves to a known one-hot tag column.

Implementation:

```text
RetrievalService._resolve_query_tag_column()
```

Behavior:

1. Normalize punctuation and casing.
2. Try exact one-hot tag match.
3. Try singular/plural variants.
4. Try common suffix trims like `recipes`, `food`, and `dish`.
5. If a match is found, add `query_tag_match` and `boosted_similarity_score`.
6. If no confident tag match is found, keep pure semantic similarity ranking.

The boost value is:

```text
_EXACT_TAG_SCORE_BOOST = 1.0
```

This is intentionally additive, not a replacement for similarity.

### Negation and Modifier Corrections

Dense embeddings often underweight negation. A query such as `"non korean"` can
land close to Korean recipes because the strongest semantic token is still
`"korean"`. The retrieval service now applies a structured correction before
final ranking:

1. Detect negation tokens such as `non`, `not`, `no`, `without`, `avoid`, and
   `exclude`.
2. Resolve the following words to a known one-hot tag column when possible.
3. Remove that negated phrase from the text sent to the embedding model.
4. Exclude rows where the negated tag column is active.
5. Increase the candidate pool multiplier from `10` to `50` when these
   query-intent exclusions are active, so filtering has enough candidates left.

Example:

```text
query: "non korean dinner"
semantic query embedded: "dinner"
structured exclusion: korean == 0
```

The same layer also protects bare positive nutrient queries from contradictory
modifier tags. For example, `"fat"` should not be dominated by `low-fat` or
`low-saturated-fat` tags just because they contain the word `"fat"`. If the user
explicitly searches `"low fat"`, the exact `low-fat` tag remains a positive
intent and is still boosted.

## Image-Only Search Pipeline

Image-only search is triggered when the user uploads an image but leaves the
text query empty. The Streamlit page calls:

```python
results = svc.search_by_image(image, request)
```

### Step-by-Step Image Search

1. Open the uploaded file with Pillow.
2. Convert the image to an RGB `PIL.Image`.
3. Lazy-load SigLIP recipe embeddings from `recipe_embeddings_siglip.npy`.
4. Lazy-load the SigLIP model and processor if they are not already loaded.
5. Encode the uploaded image with SigLIP's image encoder.
6. L2-normalize the image vector.
7. Score the image vector against every SigLIP recipe vector.
8. Select a candidate pool with `np.argpartition`.
9. Apply filters.
10. Return top-k results.
11. Attach or patch result image URLs.

The image encoding path is:

```text
RetrievalService._encode_image()
```

It uses:

```python
inputs = processor(images=image.convert("RGB"), return_tensors="pt")
outputs = model.get_image_features(**inputs)
features = outputs.pooler_output
features = features / features.norm(p=2, dim=-1, keepdim=True)
```

For image search, the matrix searched is:

```text
data/artifacts/recipe_embeddings_siglip.npy
```

This matrix contains recipe text encoded through SigLIP's text encoder. That
works because SigLIP places image embeddings and text embeddings into a shared
multimodal space.

### Image Search Filters

`search_by_image()` reuses `_search_candidates_for_vector()`, so image-only
search gets the same filter mechanics as text search. Since there is usually no
text query, tag-aware query boosting does not apply unless a query string is
also present in the request.

## Text Plus Image Search Pipeline

Combined search is triggered when both a text query and an uploaded image are
present. The Streamlit page calls:

```python
results = svc.search_combined(query, image, request, alpha=alpha)
```

The UI exposes `alpha` as:

```text
Image vs text weight (combined search)
```

Default value:

```text
SIGLIP_COMBINED_DEFAULT_IMAGE_WEIGHT = 0.75
```

Higher `alpha` means the image controls more of the ranking. Lower `alpha` lets
the text query steer more of the ranking.

### Why Combined Search Uses SigLIP Only

Text-only search uses SBERT because SBERT is better for recipe text semantics.
However, the uploaded image cannot be encoded into SBERT space. Combined search
therefore encodes both the image and the text through SigLIP so both signals live
in the same multimodal space.

The code does not compare a SigLIP image vector to SBERT recipe vectors.

### Step-by-Step Combined Search

1. Validate that text is present.
2. Validate that `alpha` is between `0.0` and `1.0`.
3. Lazy-load SigLIP recipe embeddings.
4. Encode the uploaded image with SigLIP.
5. Encode the text query with SigLIP's text encoder.
6. Compute image-to-recipe cosine scores.
7. Compute text-to-recipe cosine scores.
8. Combine scores with a weighted sum.
9. Select a large image-anchored candidate pool.
10. Rank that pool by combined score.
11. Apply filters.
12. Return top-k results with separate image/text score columns.

The score formula is:

```text
combined_score = alpha * image_similarity
                 + (1 - alpha) * text_similarity
```

The returned `similarity_score` column stores this combined score. Combined
search also returns:

```text
image_similarity_score
text_similarity_score
```

### Image-Anchored Candidate Pool

Combined search intentionally starts from visually relevant candidates:

```python
candidate_pool = min(
    len(image_scores),
    max(request.top_k * SIGLIP_COMBINED_CANDIDATE_MULTIPLIER, request.top_k),
)
```

where:

```text
SIGLIP_COMBINED_CANDIDATE_MULTIPLIER = 50
```

Then it selects the top visual matches first:

```python
image_top_idx = np.argpartition(-image_scores, candidate_pool - 1)[:candidate_pool]
ranked_idx = image_top_idx[np.argsort(-combined_scores[image_top_idx])]
```

This optimization has a product reason: if the user uploads a photo, the result
should stay visually grounded. The text should refine the image results, not
completely override the uploaded dish with a semantically unrelated recipe.

### Combined Search Filters

`search_combined()` currently applies:

1. `dietary_filter`
2. `max_time_minutes`
3. `max_ingredients`

The broader nutrition/rating filters exist on `RetrievalRequest` and are used by
the shared vector-search helper, but the combined-search method currently passes
only these three filters.

## Negative Feedback and Relevance Adjustment

Recipe cards can expose negative feedback. When the user dismisses a result, the
app calls:

```text
RetrievalService.search_with_negative_feedback()
```

The service uses a simple negative Rocchio-style adjustment:

```text
adjusted_query = normalized_query - alpha * mean(negative_vectors)
```

Default feedback alpha:

```text
0.3
```

Then it:

1. Normalizes the adjusted vector.
2. Searches in the same embedding space as the original query.
3. Requests extra candidates to replace excluded recipes.
4. Removes previously rejected recipe IDs.
5. Returns the original requested `top_k`.

The app tracks whether the current feedback vector belongs to:

1. `"text"` space for text-only search
2. `"siglip"` space for image or combined search

This prevents accidentally subtracting SigLIP vectors from SBERT vectors.

## Result Images

The retrieved recipe rows need card images for display. The service handles this
in two layers.

First, during service load:

```text
RetrievalService._attach_image_urls()
```

tries to load:

```text
data/processed/image_map.parquet
```

If the map exists, it joins `recipe_id` to `image_url`. If not, it uses a
fallback Unsplash food image.

Second, after retrieval:

```text
RetrievalService._attach_foodcom_images()
```

calls:

```text
src/recipe_discovery/retrieval/image_fetcher.py:attach_foodcom_images()
```

This patches missing or placeholder images by:

1. Building a Food.com recipe URL from recipe name and ID.
2. Fetching the page with `requests`.
3. Parsing JSON-LD Recipe metadata with `BeautifulSoup`.
4. Extracting a usable image URL.
5. Falling back safely when lookup fails.

`fetch_food_com_image()` uses `functools.lru_cache(maxsize=512)` to avoid
repeated network calls for the same recipe.

## Streamlit UI Flow

The search UI is implemented in:

```text
app/pages/1_Search.py
```

Important UI behavior:

1. Text input captures recipe names, ingredients, cuisines, tags, and meal ideas.
2. File uploader accepts `png`, `jpg`, and `jpeg`.
3. The page dispatches to text, image, or combined search based on which inputs
   are present.
4. Skeleton cards reserve space while search runs.
5. Results are stored in `st.session_state`.
6. Search history is stored for the current browser session.
7. Sorting options are display-only and do not change backend retrieval.
8. Recipe cards support shopping-list actions and negative feedback.

Dispatch logic:

```text
has_image and has_query -> svc.search_combined(...)
has_image only          -> svc.search_by_image(...)
has_query only          -> svc.search(...)
neither                 -> warning
```

The search page also stores a feedback query vector:

```text
text search       -> svc.encode_text_query(query)
image search      -> svc.encode_image_query(image)
combined search   -> svc.encode_combined_query(query, image, alpha=alpha)
```

Note: production combined retrieval uses separate image and text score fusion.
`encode_combined_query()` exists so feedback can operate on a single normalized
SigLIP vector for the current combined search state.

## Ranking and Display Are Separate

Backend retrieval ranking happens in `RetrievalService`.

Display sorting happens in:

```text
app/components/search_ui.py
```

The display sort options are:

1. Best match
2. Highest rating
3. Fastest
4. Fewest ingredients

These sort only the already-returned result DataFrame. They do not rerun vector
search and do not alter the backend candidate-pool logic.

## Optional Regression Reranking

The default runtime path is similarity-first. There is an optional reranking API:

```text
RetrievalService.search_with_optional_rerank()
```

It calls:

```text
src/recipe_discovery/retrieval/ranker.py:compute_combined_ranking()
```

If a regression model is available, it can combine:

```text
similarity_weight * normalized_similarity
+ rating_weight * normalized_predicted_rating
```

Default weights:

```text
similarity_weight = 0.8
rating_weight = 0.2
```

If no regression model is supplied, the fallback preserves similarity-only
ranking and adds `combined_score` equal to normalized similarity.

The Streamlit search page currently uses the default similarity-first calls.

## Key Optimizations

### 1. Offline Embedding Precomputation

The app does not encode every recipe at query time. It precomputes recipe
vectors into `.npy` matrices. Runtime only encodes the user's query or uploaded
image.

### 2. Batch Encoding

The build scripts encode recipe text in batches:

1. SBERT default batch size: `32`
2. SigLIP default batch size: `64`

This improves throughput and lets users tune memory usage.

### 3. L2 Normalization

Both SBERT and SigLIP embeddings are normalized. This makes cosine similarity
well-behaved and makes score interpretation more stable.

### 4. Vectorized Cosine Scoring

Cosine similarity is computed with NumPy matrix operations:

```text
matrix @ query
```

instead of Python loops over recipes.

### 5. Partial Top-k with `np.argpartition`

The service avoids sorting every recipe. It selects a candidate pool with
`np.argpartition`, then fully sorts only that pool.

Text search candidate pool:

```text
max(top_k * 10, top_k)
```

Combined search candidate pool:

```text
max(top_k * 50, top_k)
```

### 6. Candidate Pool Before Filters

The service filters a larger pool rather than filtering only the final top-k.
This improves recall for constrained searches such as `"quick dinner"` plus
`vegetarian`.

### 7. Lazy SigLIP Loading

SigLIP artifacts and the SigLIP model are loaded only when image functionality
is used. Text-only users do not pay the image-model startup cost.

### 8. Torch Inference Guardrails

SigLIP inference uses:

1. `model.eval()`
2. `torch.no_grad()`
3. CPU/CUDA device selection

This avoids training-mode overhead and unnecessary gradient memory.

### 9. Streamlit Resource Caching

`app/service_loader.py` uses `st.cache_resource` so the backend service is not
rebuilt on every Streamlit rerun.

### 10. Strict Metadata Alignment

The service aligns metadata by `recipe_id` instead of row order. This is an
accuracy optimization and a correctness guard.

### 11. Tiny Artifact Warning

If fewer than 5000 embeddings are loaded, the service logs a warning. Small
subset artifacts are useful for testing but can make broad semantic search look
poor simply because many relevant recipes are absent.

### 12. Optional Index Kept Separate

`recipe_index.joblib` is kept as a validated future acceleration path. The live
path stays direct-cosine for now so candidate-pool filtering and tag-aware
ranking stay deterministic and transparent.

## End-to-End Flow Diagrams

### Text Only

```text
User text
  -> Streamlit text input
  -> RetrievalRequest(query, filters, top_k)
  -> svc.search()
  -> SBERT query vector
  -> cosine scores vs recipe_embeddings.npy
  -> candidate pool with argpartition
  -> metadata filters
  -> optional tag boost
  -> top_k DataFrame
  -> image URL patching
  -> recipe cards
```

### Image Only

```text
Uploaded image
  -> Pillow RGB image
  -> svc.search_by_image()
  -> SigLIP image vector
  -> cosine scores vs recipe_embeddings_siglip.npy
  -> candidate pool with argpartition
  -> metadata filters
  -> top_k DataFrame
  -> image URL patching
  -> recipe cards
```

### Text Plus Image

```text
User text + uploaded image
  -> Streamlit input dispatch
  -> svc.search_combined(query, image, request, alpha)
  -> SigLIP image vector
  -> SigLIP text vector
  -> image similarity scores
  -> text similarity scores
  -> weighted combined scores
  -> image-anchored candidate pool
  -> rank pool by combined score
  -> metadata filters
  -> top_k DataFrame with image/text score columns
  -> image URL patching
  -> recipe cards
```

## Why Direct Cosine Instead of Only NearestNeighbors

The project does build a cosine `NearestNeighbors` artifact, but the live
retrieval service intentionally scans the loaded matrix directly.

Reasons:

1. The service needs the full score vector for candidate-pool selection.
2. Filtering is applied after a larger candidate set is selected.
3. Tag-aware boosts are easier and clearer in the direct-score path.
4. Tests verify direct cosine and saved-index ordering parity on the same
   vectors.
5. The current dataset size is manageable for direct NumPy scoring in a course
   project Streamlit app.

## Validation and Tests

Relevant tests:

1. `tests/test_embeddings_pipeline.py`
2. `tests/test_retrieval.py`
3. `tests/test_pipeline_smoke.py`
4. `tests/test_search_ui.py`

Important covered behaviors:

1. Processed CSV loading.
2. One-hot tag detection.
3. Embedding artifact reloadability.
4. Metadata alignment by `recipe_id`.
5. Text query ranking.
6. Candidate-pool filtering.
7. Tag-aware boosting.
8. Combined image/text search behavior.
9. Direct cosine and saved-index parity.
10. Display-only sorting.

Useful commands:

```bash
pytest tests/test_embeddings_pipeline.py
pytest tests/test_retrieval.py
pytest tests/test_search_ui.py
python scripts/smoke_retrieval.py
python scripts/smoke_medium_pipeline.py --subset-size 200 --top-k 5
```

## Common Failure Modes

### Missing SBERT artifacts

Symptom:

```text
FileNotFoundError for recipe_embeddings.npy or recipe_ids.csv
```

Fix:

```bash
python scripts/build_embeddings.py --overwrite
```

### Missing SigLIP artifacts

Symptom:

```text
SigLIP artifacts not found. Run scripts/generate_siglip_embeddings.py first.
```

Fix:

```bash
python scripts/generate_siglip_embeddings.py --overwrite
```

### Image search dependencies missing

Symptom:

```text
Image search requires torch and transformers
```

Fix:

```bash
pip install -r requirements.txt
```

### Metadata and vector rows do not align

Symptom:

```text
Embeddings and recipe_ids row count mismatch
```

or:

```text
Some recipe_ids from artifacts were not found in processed metadata
```

Fix: rebuild embeddings from the same processed CSV that the app is loading.

### Weak results from tiny artifacts

Symptom: broad searches return odd or missing results.

Cause: embeddings were built with `--limit`, so most recipes are not in the
vector store.

Fix:

```bash
python scripts/build_embeddings.py --overwrite
python scripts/generate_siglip_embeddings.py --overwrite
```

without `--limit`.

## Practical Summary

Text-only search uses SBERT because it is strong for recipe text semantics.
Image-only and text-plus-image search use SigLIP because image vectors and text
vectors must live in the same space. All modes convert the query into a dense
vector, compare it to a precomputed recipe vector matrix with cosine similarity,
select a larger candidate pool, apply metadata filters, and return the final
top-k recipes with display-ready metadata.
