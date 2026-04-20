# Embeddings Module — Technical Documentation

## 1. Purpose

The embeddings pipeline converts the processed recipe dataset into dense semantic vectors that power downstream retrieval, clustering, and exploration features. It is the first stage of the Recipe Discovery Web App's ML backend.

**What it does end-to-end:**

1. Loads the cleaned/processed CSV (`data/processed/Processed_data_updated2.csv`)
2. Builds a deterministic, human-readable "canonical text" for each recipe
3. Encodes that text into a 384-dimensional dense vector using a pre-trained sentence-transformer
4. Saves the embeddings, recipe-ID mappings, and metadata to `data/artifacts/`
5. Builds a cosine-similarity nearest-neighbor index for fast retrieval

---

## 2. File Map

| File | Responsibility |
|---|---|
| `src/recipe_discovery/settings.py` | Project-wide paths (`ARTIFACTS_DIR`, `DATA_PROCESSED_DIR`) |
| `src/recipe_discovery/data/schema.py` | Column-group constants, unnamed-column detection, one-hot tag detection |
| `src/recipe_discovery/data/load.py` | `load_processed_recipes()` — loads CSV, drops unnamed columns |
| `src/recipe_discovery/data/corpus.py` | `build_corpus()` / `serialize_recipe()` — canonical text construction |
| `src/recipe_discovery/embeddings/encoder.py` | `RecipeEncoder` — wraps `sentence-transformers` model loading & batch encoding |
| `src/recipe_discovery/embeddings/store.py` | Save/load helpers for embeddings, IDs, texts, metadata |
| `src/recipe_discovery/embeddings/index.py` | Build, save, load a `sklearn.neighbors.NearestNeighbors` cosine index |
| `src/recipe_discovery/utils/io.py` | `ensure_parent_dir()`, `save_json()`, `load_json()` |
| `scripts/build_embeddings.py` | CLI entrypoint — runs the full embedding pipeline |
| `scripts/build_index.py` | CLI entrypoint — builds the retrieval index from saved embeddings |
| `configs/embeddings.yaml` | Model name, batch size, artifact paths, corpus flags |

---

## 3. How the Processed CSV Is Interpreted

The file `Processed_data_updated2.csv` has **504 columns** and **226,657 rows**. There is no single `tags` column. Instead the schema is organized into groups:

### 3.1 Column Groups

| Group | Columns | Count |
|---|---|---|
| Identity | `recipe_id` | 1 |
| Text-rich | `name`, `description`, `ingredients`, `steps` | 4 |
| Short metadata | `minutes`, `n_steps`, `n_ingredients` | 3 |
| Nutrition | `calories`, `total fat`, `sugar`, `sodium`, `protein`, `saturated fat`, `carbohydrates` | 7 |
| Outcome | `rating`, `num_ratings` | 2 |
| Long text | `all_reviews` | 1 |
| Unnamed/junk | `Unnamed: 8` | 1 |
| **One-hot tags** | `1-day-or-more`, `15-minutes-or-less`, `american`, `asian`, … | **485** |

These groups are defined in `src/recipe_discovery/data/schema.py`.

### 3.2 Unnamed Column Handling

The loader (`load_processed_recipes`) automatically detects and drops any column whose name is empty or matches `^Unnamed` (case-insensitive). In this dataset, `Unnamed: 8` is the only such column.

### 3.3 One-Hot Tag Detection

The function `get_one_hot_tag_columns(df)` identifies tag columns by exclusion:

- Not in the "known" set (text, ID, metadata, nutrition, outcome, long-text)
- Not unnamed
- All non-null values are in `{0, 1}`

This yields 485 tag columns covering cuisines, dietary labels, occasions, methods, and more.

---

## 4. How Recipe Text Is Built

For each recipe row, `serialize_recipe()` in `corpus.py` produces a block like:

```
Title: low fat berry blue frozen dessert
Description: this is yummy and low-fat, it always turns out perfect.
Ingredients: ['blueberries', 'granulated sugar', 'vanilla yogurt', 'lemon juice']
Steps: ['toss 2 cups berries with sugar', 'let stand for 45 minutes...', ...]
Time: 1440 minutes
Step Count: 11
Ingredient Count: 4
Tags: 1-day-or-more, desserts, easy, frozen-desserts, fruit, ...
Nutrition: calories=36.4, total fat=0.0, sugar=62.0, ...
```

**Rules:**
- Missing/NaN values are silently omitted (no literal `nan`)
- Tag text is reconstructed row-by-row from the binary one-hot columns
- `all_reviews` is excluded by default (too long for v1 embedding)
- `recipe_id` is not included in the text (it is stored separately for alignment)
- Output is deterministic — same input always produces the same text

### 4.1 One-Hot Tag Reconstruction

For each row, the code iterates over the 485 tag columns and collects column names where the value equals 1. These are joined with commas into a `Tags:` line. For example, a vegan stir-fry might produce:

```
Tags: 30-minutes-or-less, asian, easy, healthy, stir-fry, vegan, vegetables, vegetarian
```

---

## 5. Embedding Model

| Setting | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Normalization | L2-normalized (unit vectors) |
| Batch size | 32 (configurable via CLI `--batch-size`) |

The model is loaded lazily by `RecipeEncoder.load()` and encodes text via `RecipeEncoder.encode()`. Normalization is enabled by default, which means cosine similarity equals the dot product — this is important for the retrieval index.

---

## 6. Artifacts Produced

All artifacts are saved under `data/artifacts/`:

| File | Format | Description |
|---|---|---|
| `recipe_embeddings.npy` | NumPy `.npy` | Dense matrix of shape `(n_recipes, 384)` |
| `recipe_ids.csv` | CSV with `recipe_id` column | Row-aligned recipe IDs |
| `recipe_texts.csv` | CSV with `recipe_text` column | The canonical text used for each embedding |
| `embedding_metadata.json` | JSON | Model name, dim, row count, timestamp |
| `recipe_index.joblib` | joblib-serialized sklearn object | Fitted `NearestNeighbors(metric="cosine")` |

**Critical invariant:** Row *i* in `recipe_embeddings.npy` corresponds to row *i* in `recipe_ids.csv` and `recipe_texts.csv`.

---

## 7. How Downstream Retrieval Uses These Artifacts

The retrieval layer (`src/recipe_discovery/retrieval/`) can:

1. **Load the index:** `from recipe_discovery.embeddings.index import load_index`
2. **Load IDs:** `from recipe_discovery.embeddings.store import load_recipe_ids`
3. **Query:** Encode a user's query text with `RecipeEncoder`, then call `index.kneighbors(query_vec, n_neighbors=k)` to get the top-*k* nearest recipe indices
4. **Map indices → recipe IDs** using the loaded `recipe_ids` series
5. **Fetch full recipe details** from the original CSV or a database

The index uses cosine distance (`1 - cosine_similarity`), so a distance of 0 means identical and 2 means maximally dissimilar.

---

## 8. Configuration

Settings live in `configs/embeddings.yaml`:

```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32
  normalize: true

artifacts:
  embeddings: data/artifacts/recipe_embeddings.npy
  recipe_ids: data/artifacts/recipe_ids.csv
  recipe_texts: data/artifacts/recipe_texts.csv
  metadata: data/artifacts/embedding_metadata.json
  index: data/artifacts/recipe_index.joblib

corpus:
  include_nutrition: true
  include_reviews: false
```

The scripts currently use CLI arguments and code defaults rather than loading this YAML at runtime. The YAML serves as the canonical reference and can be wired in as the project matures.
